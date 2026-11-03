// GOD PROMPT "CKKS RUST + CUDA PHASE 4" round. Path A (existing CUDA C++
// kernels, host orchestration entirely in C++) of the fair comparison
// against rust/crates/ckks-cuda::EvalMultContext (Path B). Same native
// (OpenFHE-free) fixture recipe as bench_keyswitch_cpp_vs_rust.cpp
// (n=1024, sizeQ=8, sizeP=2, numPart=4 -- GOD PROMPT "CKKS PHASE 10": was
// numPart=2, which violates the Hybrid key-switch's required
// P>=Q_part_max condition and is now rejected at ks_context_create
// construction time; see research/EXPERIMENT_LEDGER.md entry #42/#43),
// extended with the SAME
// tensor+relin+combine+rescale(x2) composition
// bench/bench_evalmult_resident.cpp already demonstrates at a different
// (OpenFHE-sourced, N=16-batched, CUDA-graph) scale -- this file is NOT a
// replacement for that benchmark, it is a SEPARATE, simpler single-
// operation latency comparison at the SAME small native configuration
// Rust's own test/benchmark uses, so the two languages are measured doing
// the SAME setup and SAME per-call work.
//
// UNLIKE the Rust side, this file does NOT need a second, separate
// RescaleContext-equivalent: gpufhe::rescale_resident_raw already accepts
// the SAME DeviceKSContext the key-switch fixture builds (it only reads
// plain moduli + cached root tables from it, never the eval key -- see
// research/EXPERIMENT_LEDGER.md entry #35), so this fixture reuses `C`
// directly for both the relin step and the two rescale steps, exactly
// like bench_evalmult_resident.cpp's own `ch[c]` struct does.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <random>

namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void native_rescale_consts(std::vector<uint64_t>&, std::vector<uint64_t>&,
                            const std::vector<uint64_t>&, uint32_t);
}
extern "C" void LaunchTensor(const uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*,
    uint64_t*, uint64_t*, uint64_t*, const uint64_t*, uint32_t, uint32_t, cudaStream_t);
extern "C" void LaunchCombine(uint64_t*, uint64_t*, const uint64_t*, const uint64_t*,
    const uint64_t*, const uint64_t*, const uint64_t*, uint32_t, uint32_t, cudaStream_t);

static uint64_t mulmod_b(uint64_t a, uint64_t b, uint64_t q) { return (uint64_t)(((unsigned __int128)a * b) % q); }

// Mirrors rust/crates/ckks-cuda::evalmult::EvalMultContext field-for-field
// (modulo language idiom): K/C/W (the key-switch fixture, reused for
// relin AND rescale), rs1/rs2 (fused-rescale constants over the SAME
// chain), d_mods (for Tensor/Combine), and the 11 per-call buffers.
struct EvalMultFixtureCpp {
    gpufhe::KeySwitchConstants K;
    gpufhe::DeviceKSContext C;
    gpufhe::DeviceKSWork W;
    std::vector<uint64_t> rs1, rs2;
    uint64_t* d_mods = nullptr;
    uint64_t *c0a, *c1a, *c0b, *c1b, *t0, *t1, *t2, *ba0, *ba1, *r0, *r1;
    uint64_t *scr, *drp;
    uint32_t n, sizeQ;

    EvalMultFixtureCpp(uint32_t n_, uint32_t sizeQ_, uint32_t sizeP, uint32_t numPart, uint64_t seed)
        : n(n_), sizeQ(sizeQ_)
    {
        std::vector<uint64_t> mod, modP;
        gpufhe::native_primes(mod, 1, 60, n, {});
        { std::vector<uint64_t> mids; gpufhe::native_primes(mids, sizeQ - 1, 50, n, mod);
          for (auto m : mids) mod.push_back(m); }
        gpufhe::native_primes(modP, sizeP, 60, n, mod);

        std::vector<uint64_t> modQP = mod; for (auto p : modP) modQP.push_back(p);
        std::vector<uint64_t> root(sizeQ), rootQP;
        for (uint32_t i = 0; i < sizeQ; ++i) root[i] = gpufhe::native_root(n, mod[i]);
        rootQP = root; for (auto p : modP) rootQP.push_back(gpufhe::native_root(n, p));

        auto KPqp = gpufhe::keygen_host(n, modQP, rootQP, /*ns=*/1, /*sigma=*/3.19, seed);

        std::vector<uint64_t> PModq_QP(sizeQ + sizeP);
        for (uint32_t t = 0; t < sizeQ + sizeP; ++t) {
            uint64_t q = modQP[t], P = 1 % q;
            for (uint32_t j = 0; j < sizeP; ++j) P = mulmod_b(P, modQP[sizeQ + j] % q, q);
            PModq_QP[t] = P;
        }

        K.n = n;
        gpufhe::compute_keyswitch_constants(K, mod, modP, numPart);
        for (uint32_t i = 0; i < sizeQ; ++i) { K.rootModList.push_back(mod[i]); K.rootValList.push_back(root[i]); }
        for (uint32_t j = 0; j < sizeP; ++j) { K.rootModList.push_back(modQP[sizeQ + j]); K.rootValList.push_back(rootQP[sizeQ + j]); }

        gpufhe::evalkeygen_host(K, KPqp.s, KPqp.pkA, KPqp.pkB, PModq_QP, modQP, rootQP,
                                 /*ns=*/1, /*sigma=*/3.19, seed + 10007);

        C = gpufhe::ks_context_create(K);
        W = gpufhe::ks_work_create(C);

        gpufhe::native_rescale_consts(rs1, rs2, mod, sizeQ - 1);

        const size_t T = (size_t)sizeQ * n;
        cudaMalloc(&d_mods, sizeQ * 8);
        cudaMemcpy(d_mods, mod.data(), sizeQ * 8, cudaMemcpyHostToDevice);
        for (uint64_t** p : {&c0a, &c1a, &c0b, &c1b, &t0, &t1, &t2, &ba0, &ba1, &r0, &r1})
            cudaMalloc(p, T * 8);
        cudaMalloc(&scr, (size_t)n * 8);
        cudaMalloc(&drp, (size_t)n * 8);
    }
    ~EvalMultFixtureCpp() {
        gpufhe::ks_work_destroy(W);
        gpufhe::ks_context_destroy(C);
        cudaFree(d_mods);
        for (uint64_t* p : {c0a, c1a, c0b, c1b, t0, t1, t2, ba0, ba1, r0, r1}) cudaFree(p);
        cudaFree(scr); cudaFree(drp);
    }

    // Steady-state per-call path: upload the 4 operands, run the SAME
    // tensor->relin->combine->rescale(x2) sequence
    // bench_evalmult_resident.cpp's own per-chain graph captures, output
    // to host.
    void multiply_relin_rescale(
        const std::vector<uint64_t>& h_c0a, const std::vector<uint64_t>& h_c1a,
        const std::vector<uint64_t>& h_c0b, const std::vector<uint64_t>& h_c1b,
        std::vector<uint64_t>& out0, std::vector<uint64_t>& out1)
    {
        const size_t T = (size_t)sizeQ * n;
        cudaMemcpy(c0a, h_c0a.data(), T * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(c1a, h_c1a.data(), T * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(c0b, h_c0b.data(), T * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(c1b, h_c1b.data(), T * 8, cudaMemcpyHostToDevice);

        LaunchTensor(c0a, c1a, c0b, c1b, t0, t1, t2, d_mods, sizeQ, n, 0);
        gpufhe::keyswitch_resident(t2, ba0, ba1, C, W, 0);
        LaunchCombine(r0, r1, t0, t1, ba0, ba1, d_mods, sizeQ, n, 0);
        gpufhe::rescale_resident_raw(r0, sizeQ, C, rs1, rs2, scr, drp, 0);
        gpufhe::rescale_resident_raw(r1, sizeQ, C, rs1, rs2, scr, drp, 0);
        cudaDeviceSynchronize();

        const size_t outT = (size_t)(sizeQ - 1) * n;
        out0.resize(outT); out1.resize(outT);
        cudaMemcpy(out0.data(), r0, outT * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(out1.data(), r1, outT * 8, cudaMemcpyDeviceToHost);
    }
};

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n/2] : 0.5 * (v[n/2-1] + v[n/2]);
}
static void report(const char* label, std::vector<double>& per_trial) {
    double mn = *std::min_element(per_trial.begin(), per_trial.end());
    double mx = *std::max_element(per_trial.begin(), per_trial.end());
    std::vector<double> copy = per_trial;
    double m = median(copy);
    std::cout << label << ": median=" << m << "us min=" << mn << "us max=" << mx << "us (n=" << per_trial.size() << " trials)\n";
}

int main(int argc, char** argv) {
    const uint32_t n = 1024, sizeQ = 8, sizeP = 2, numPart = 4; // see entry #42/#43: numPart=2 violates P>=Q_part_max
    size_t reps = argc > 1 ? (size_t)std::atoi(argv[1]) : 20;
    size_t trials = argc > 2 ? (size_t)std::atoi(argv[2]) : 7;

    // --- Tier 1: fixture creation (full setup, GPU-touching throughout
    // -- see bench_keyswitch_cpp_vs_rust.cpp's header comment for why
    // this isn't split into a host-only "cold-start" sub-tier). ---
    std::vector<double> init_us;
    for (size_t trial = 0; trial < trials; ++trial) {
        auto t0 = std::chrono::steady_clock::now();
        EvalMultFixtureCpp fx(n, sizeQ, sizeP, numPart, 1000 + trial);
        auto t1 = std::chrono::steady_clock::now();
        init_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    report("fixture creation (EvalMultFixtureCpp constructor)", init_us);

    // --- Correctness gate, BEFORE any timing is trusted: compare against
    // keyswitch_core_resident's SAME-K host reference for the relin step,
    // composed with from-scratch tensor/combine/rescale-identity math,
    // the SAME methodology as the Rust side's own differential test. ---
    {
        EvalMultFixtureCpp checkFx(n, sizeQ, sizeP, numPart, /*seed=*/777);
        std::mt19937_64 rng(555);
        std::vector<uint64_t> c0a(sizeQ * (size_t)n), c1a(sizeQ * (size_t)n), c0b(sizeQ * (size_t)n), c1b(sizeQ * (size_t)n);
        for (uint32_t t = 0; t < sizeQ; ++t) {
            std::uniform_int_distribution<uint64_t> d(0, checkFx.K.qMod[t] - 1);
            for (uint32_t k = 0; k < n; ++k) {
                size_t x = (size_t)t * n + k;
                c0a[x] = d(rng); c1a[x] = d(rng); c0b[x] = d(rng); c1b[x] = d(rng);
            }
        }
        std::vector<uint64_t> got0, got1;
        checkFx.multiply_relin_rescale(c0a, c1a, c0b, c1b, got0, got1);

        // Independent reference: schoolbook tensor+combine (eval domain,
        // matching kernels/tensor.cu's formulas exactly), relin via
        // keyswitch_core_resident (the established host oracle), rescale
        // via rescale_resident_raw itself run a SECOND time on a tensor+
        // combine result computed independently of the GPU's own t0/t1/t2
        // -- i.e. this checks the WHOLE chain's wiring (right buffers in
        // the right order), not just that each kernel individually works
        // (already proven by entries #32/#35/#36).
        std::vector<uint64_t> refT0(sizeQ * (size_t)n), refT1(sizeQ * (size_t)n), refT2(sizeQ * (size_t)n);
        for (uint32_t t = 0; t < sizeQ; ++t) {
            uint64_t q = checkFx.K.qMod[t];
            for (uint32_t k = 0; k < n; ++k) {
                size_t x = (size_t)t * n + k;
                uint64_t a0 = c0a[x], a1 = c1a[x], b0 = c0b[x], b1 = c1b[x];
                refT0[x] = mulmod_b(a0, b0, q);
                uint64_t s = mulmod_b(a0, b1, q) + mulmod_b(a1, b0, q);
                refT1[x] = (s >= q) ? s - q : s;
                refT2[x] = mulmod_b(a1, b1, q);
            }
        }
        auto R = gpufhe::keyswitch_core_resident(refT2, checkFx.K);
        std::vector<uint64_t> refR0(sizeQ * (size_t)n), refR1(sizeQ * (size_t)n);
        for (uint32_t t = 0; t < sizeQ; ++t) {
            uint64_t q = checkFx.K.qMod[t];
            for (uint32_t k = 0; k < n; ++k) {
                size_t x = (size_t)t * n + k;
                uint64_t a = refT0[x] + R.ba0[x]; refR0[x] = (a >= q) ? a - q : a;
                uint64_t b = refT1[x] + R.ba1[x]; refR1[x] = (b >= q) ? b - q : b;
            }
        }
        // Rescale the independently-built refR0/refR1 via the SAME
        // rescale_resident_raw the fixture itself uses, on a scratch
        // device buffer -- this checks tensor+combine+relin wiring
        // end-to-end against an independent computation, while still
        // delegating the already-proven rescale kernel itself (entry
        // #35) rather than re-deriving its own identity a third time.
        uint64_t *d_ref0, *d_ref1, *d_scr2, *d_drp2;
        cudaMalloc(&d_ref0, sizeQ * (size_t)n * 8); cudaMalloc(&d_ref1, sizeQ * (size_t)n * 8);
        cudaMalloc(&d_scr2, (size_t)n * 8); cudaMalloc(&d_drp2, (size_t)n * 8);
        cudaMemcpy(d_ref0, refR0.data(), sizeQ * (size_t)n * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ref1, refR1.data(), sizeQ * (size_t)n * 8, cudaMemcpyHostToDevice);
        gpufhe::rescale_resident_raw(d_ref0, sizeQ, checkFx.C, checkFx.rs1, checkFx.rs2, d_scr2, d_drp2, 0);
        gpufhe::rescale_resident_raw(d_ref1, sizeQ, checkFx.C, checkFx.rs1, checkFx.rs2, d_scr2, d_drp2, 0);
        cudaDeviceSynchronize();
        std::vector<uint64_t> exp0((sizeQ - 1) * (size_t)n), exp1((sizeQ - 1) * (size_t)n);
        cudaMemcpy(exp0.data(), d_ref0, exp0.size() * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(exp1.data(), d_ref1, exp1.size() * 8, cudaMemcpyDeviceToHost);
        cudaFree(d_ref0); cudaFree(d_ref1); cudaFree(d_scr2); cudaFree(d_drp2);

        size_t mismatches = 0;
        for (size_t x = 0; x < exp0.size(); ++x) {
            if (got0[x] != exp0[x]) ++mismatches;
            if (got1[x] != exp1[x]) ++mismatches;
        }
        if (mismatches != 0) {
            std::cerr << "CORRECTNESS CHECK FAILED: " << mismatches << " mismatches -- refusing to report timing\n";
            return 1;
        }
        std::cout << "correctness check: 0/" << (2 * exp0.size()) << " mismatches vs independent tensor+relin+combine+rescale reference\n";
    }

    // --- Tier 2: steady-state, ONE fixture reused across all reps/trials. ---
    EvalMultFixtureCpp ctxFx(n, sizeQ, sizeP, numPart, /*seed=*/42);
    std::mt19937_64 rng(2024);
    std::vector<uint64_t> c0a(sizeQ * (size_t)n), c1a(sizeQ * (size_t)n), c0b(sizeQ * (size_t)n), c1b(sizeQ * (size_t)n);
    for (uint32_t t = 0; t < sizeQ; ++t) {
        std::uniform_int_distribution<uint64_t> d(0, ctxFx.K.qMod[t] - 1);
        for (uint32_t k = 0; k < n; ++k) {
            size_t x = (size_t)t * n + k;
            c0a[x] = d(rng); c1a[x] = d(rng); c0b[x] = d(rng); c1b[x] = d(rng);
        }
    }
    std::vector<uint64_t> out0, out1;
    ctxFx.multiply_relin_rescale(c0a, c1a, c0b, c1b, out0, out1); // warm-up

    std::vector<double> per_trial_median;
    for (size_t trial = 0; trial < trials; ++trial) {
        std::vector<double> per_call;
        for (size_t r = 0; r < reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            ctxFx.multiply_relin_rescale(c0a, c1a, c0b, c1b, out0, out1);
            auto t1 = std::chrono::steady_clock::now();
            per_call.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        per_trial_median.push_back(median(per_call));
    }
    report("steady-state multiply_relin_rescale (reused fixture)", per_trial_median);

    // --- Tier 3: end-to-end, fresh fixture + first call, combined. ---
    std::vector<double> e2e_us;
    for (size_t trial = 0; trial < trials; ++trial) {
        auto t0 = std::chrono::steady_clock::now();
        EvalMultFixtureCpp fresh(n, sizeQ, sizeP, numPart, 5000 + trial);
        std::vector<uint64_t> fo0, fo1;
        fresh.multiply_relin_rescale(c0a, c1a, c0b, c1b, fo0, fo1);
        e2e_us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    report("end-to-end (fresh fixture + first multiply_relin_rescale)", e2e_us);

    return 0;
}
