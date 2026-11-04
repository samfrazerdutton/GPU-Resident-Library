// GOD PROMPT "CKKS PHASE 11": real-world performance baseline for the
// complete encode -> encrypt -> EvalMult -> decrypt -> decode pipeline,
// broken down into every individual stage, with CUDA-event GPU timing,
// host-clock wall timing, GPU/host memory reporting, and full statistics
// (min/median/p95/max/mean over many reps). This is a NEW, additive
// benchmark -- no production file is modified. Reuses the EXACT same
// fixture-construction recipe as bench_evalmult_cpp_vs_rust.cpp
// (native_primes/keygen_host/compute_keyswitch_constants/evalkeygen_host/
// ks_context_create/ks_work_create/native_rescale_consts), extended to
// also retain the Q-restricted secret/public key (same pattern
// cpp_bridge/keyswitch_bridge.cpp's KsFixture already establishes) so a
// REAL encode/encrypt/decrypt/decode round trip can run end to end.
//
// Mandatory, per the governing directive: every configuration is checked
// via gpufhe::check_hybrid_keyswitch_params (the Phase 10 centralized,
// exact validator) BEFORE any fixture is built. A configuration that
// fails this check is reported and skipped -- never bypassed.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include "ks_params_validate.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <random>
#include <complex>
#include <sys/resource.h>

namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void native_rescale_consts(std::vector<uint64_t>&, std::vector<uint64_t>&,
                            const std::vector<uint64_t>&, uint32_t);
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void decode_host(std::vector<std::complex<double>>&, const std::vector<int64_t>&, uint32_t, double);
// GOD PROMPT "CKKS PHASE 12": the new O(N log N) FFT-based encode/decode
// (src/encode_fft.cpp) -- now the actual production path (cpp_bridge/
// encdec_bridge.cpp calls these, not encode_host/decode_host above).
// This benchmark can measure EITHER path (see run_config's useFft
// parameter) for a direct old-vs-new full-pipeline comparison.
void encode_host_fft(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void decode_host_fft(std::vector<std::complex<double>>&, const std::vector<int64_t>&, uint32_t, double);
void encrypt_host(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
// GOD PROMPT "CKKS PHASE 13": the new Shoup-modmul + pooled-buffer
// encrypt_host_shoup_pooled (src/encrypt_host_shoup_pooled.cpp) -- now
// the actual production path (cpp_bridge/encdec_bridge.cpp calls this,
// not encrypt_host above). This benchmark can measure EITHER path (see
// run_config's useFastEncrypt parameter) for a direct old-vs-new
// full-pipeline comparison.
void encrypt_host_shoup_pooled(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                                const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                                const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
// GOD PROMPT "CKKS PHASE 14": encrypt_host_shoup_pooled_barrett (src/
// encrypt_host_shoup_pooled_barrett.cpp) -- now the actual production
// path (cpp_bridge/encdec_bridge.cpp calls this, not
// encrypt_host_shoup_pooled above). This benchmark can measure any of
// the three variants (see run_config's encryptVariant parameter: 0=OLD
// encrypt_host, 1=Phase 13 shoup_pooled, 2=Phase 14 shoup_pooled_barrett).
void encrypt_host_shoup_pooled_barrett(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                                        const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                                        const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
// GOD PROMPT "CKKS PHASE 15": encrypt_host_shoup_pooled_barrett_batched
// (src/encrypt_host_shoup_pooled_barrett_batched.cpp) -- now the
// actual production path (cpp_bridge/encdec_bridge.cpp calls this, not
// encrypt_host_shoup_pooled_barrett above). This benchmark can measure
// any of the four variants (see run_config's encryptVariant parameter:
// 0=OLD encrypt_host, 1=Phase 13, 2=Phase 14, 3=Phase 15 batched).
void encrypt_host_shoup_pooled_barrett_batched(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                                                const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                                                const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
// GOD PROMPT "CKKS PHASE 15" Experiment 2: encrypt_host_shoup_pooled_
// barrett_batched_pinned (src/encrypt_host_shoup_pooled_barrett_
// batched_pinned.cpp) -- built, bit-exact verified, and measured a
// real steady-state win, but NOT promoted to production (see
// research/TRANSFER_PROFILING.md: cudaMallocHost's one-time cost per
// (n,sizeQ) key can exceed a single encrypt call's entire latency).
// cpp_bridge/encdec_bridge.cpp does NOT call this. encryptVariant:
// 0=OLD, 1=Phase13, 2=Phase14, 3=Phase15 Experiment1 (batched,
// pageable -- PRODUCTION), 4=Phase15 Experiment2 (batched, pinned --
// measured only, not shipped).
void encrypt_host_shoup_pooled_barrett_batched_pinned(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                                                        const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                                                        const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
// GOD PROMPT "CKKS PHASE 20": encrypt_host_shoup_pooled_barrett_batched_registered
// (src/encrypt_host_shoup_pooled_barrett_batched_registered.cpp) --
// CUDA-REGISTERED (cudaHostRegister on pre-touched page-aligned memory,
// NOT cudaMallocHost-pinned) host staging buffer, built on Phase 19's
// decisive finding that registration is 1.5x-4x cheaper than
// cudaMallocHost/cudaHostAlloc for the full allocation lifecycle.
// encryptVariant 5 measures this real encrypted path end to end.
void encrypt_host_shoup_pooled_barrett_batched_registered(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                                                             const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                                                             const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void decrypt_host(std::vector<int64_t>&, const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                   const std::vector<uint64_t>&, uint32_t, const std::vector<uint64_t>&, const std::vector<uint64_t>&);
}
extern "C" void LaunchTensor(const uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*,
    uint64_t*, uint64_t*, uint64_t*, const uint64_t*, uint32_t, uint32_t, cudaStream_t);
extern "C" void LaunchCombine(uint64_t*, uint64_t*, const uint64_t*, const uint64_t*,
    const uint64_t*, const uint64_t*, const uint64_t*, uint32_t, uint32_t, cudaStream_t);

static uint64_t mulmod_b(uint64_t a, uint64_t b, uint64_t q) { return (uint64_t)(((unsigned __int128)a * b) % q); }

static double gpu_used_mb() {
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    return (double)(t - f) / 1024.0 / 1024.0;
}
static double host_rss_mb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (double)ru.ru_maxrss / 1024.0; // ru_maxrss is KB on Linux
}

struct Stats {
    double min_, p50, p95, max_, mean;
    size_t n;
};
static Stats stats_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    Stats s;
    s.n = v.size();
    s.min_ = v.front();
    s.max_ = v.back();
    s.p50 = v[v.size() / 2];
    s.p95 = v[(size_t)(0.95 * (v.size() - 1))];
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / v.size();
    return s;
}
static void report(const char* label, const Stats& s) {
    std::cout << "  " << label << ": median=" << s.p50 << "us p95=" << s.p95
              << "us min=" << s.min_ << "us max=" << s.max_ << "us mean=" << s.mean
              << "us (n=" << s.n << ")\n";
}

// GPU-event-timed kernel stage: records before/after, synchronizes,
// returns elapsed milliseconds*1000 (us). Correct synchronization --
// cudaEventSynchronize blocks until the GPU work between the two events
// has actually completed, so this never mistakes an async launch for a
// finished operation.
struct EventTimer {
    cudaEvent_t a, b;
    EventTimer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    ~EventTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start(cudaStream_t s = 0) { cudaEventRecord(a, s); }
    double stop_us(cudaStream_t s = 0) {
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        return (double)ms * 1000.0;
    }
};

struct EvalMultFixtureCpp {
    gpufhe::KeySwitchConstants K;
    gpufhe::DeviceKSContext C;
    gpufhe::DeviceKSWork W;
    std::vector<uint64_t> rs1, rs2;
    std::vector<uint64_t> secretKey, pubKeyA, pubKeyB; // Q-restricted, for real encrypt/decrypt
    std::vector<uint64_t> modQ, rootQ;
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
        modQ = mod;

        std::vector<uint64_t> modQP = mod; for (auto p : modP) modQP.push_back(p);
        std::vector<uint64_t> root(sizeQ), rootQP;
        for (uint32_t i = 0; i < sizeQ; ++i) root[i] = gpufhe::native_root(n, mod[i]);
        rootQ = root;
        rootQP = root; for (auto p : modP) rootQP.push_back(gpufhe::native_root(n, p));

        auto KPqp = gpufhe::keygen_host(n, modQP, rootQP, /*ns=*/1, /*sigma=*/3.19, seed);
        secretKey.assign(KPqp.s.begin(), KPqp.s.begin() + (size_t)sizeQ * n);
        pubKeyA.assign(KPqp.pkA.begin(), KPqp.pkA.begin() + (size_t)sizeQ * n);
        pubKeyB.assign(KPqp.pkB.begin(), KPqp.pkB.begin() + (size_t)sizeQ * n);

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

    double scale() const { return (double)modQ[sizeQ - 1]; }

    // Per-stage breakdown of the device-resident hot path (upload, tensor,
    // keyswitch, combine, rescale x2, download), each independently timed
    // via a dedicated EventTimer -- NOT one aggregate call like the
    // existing bench_evalmult_cpp_vs_rust.cpp. Returns microseconds for
    // each stage, appended to the caller's accumulator vectors.
    void multiply_relin_rescale_breakdown(
        const std::vector<uint64_t>& h_c0a, const std::vector<uint64_t>& h_c1a,
        const std::vector<uint64_t>& h_c0b, const std::vector<uint64_t>& h_c1b,
        std::vector<uint64_t>& out0, std::vector<uint64_t>& out1,
        double& upload_us, double& tensor_us, double& keyswitch_us,
        double& combine_us, double& rescale_us, double& download_us)
    {
        const size_t T = (size_t)sizeQ * n;
        EventTimer ev;

        ev.start();
        cudaMemcpy(c0a, h_c0a.data(), T * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(c1a, h_c1a.data(), T * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(c0b, h_c0b.data(), T * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(c1b, h_c1b.data(), T * 8, cudaMemcpyHostToDevice);
        upload_us = ev.stop_us();

        ev.start();
        LaunchTensor(c0a, c1a, c0b, c1b, t0, t1, t2, d_mods, sizeQ, n, 0);
        tensor_us = ev.stop_us();

        ev.start();
        gpufhe::keyswitch_resident(t2, ba0, ba1, C, W, 0);
        keyswitch_us = ev.stop_us();

        ev.start();
        LaunchCombine(r0, r1, t0, t1, ba0, ba1, d_mods, sizeQ, n, 0);
        combine_us = ev.stop_us();

        ev.start();
        gpufhe::rescale_resident_raw(r0, sizeQ, C, rs1, rs2, scr, drp, 0);
        gpufhe::rescale_resident_raw(r1, sizeQ, C, rs1, rs2, scr, drp, 0);
        rescale_us = ev.stop_us();

        const size_t outT = (size_t)(sizeQ - 1) * n;
        out0.resize(outT); out1.resize(outT);
        ev.start();
        cudaMemcpy(out0.data(), r0, outT * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(out1.data(), r1, outT * 8, cudaMemcpyDeviceToHost);
        download_us = ev.stop_us();
    }
};

// Returns false on any correctness failure (and true if a config was
// validly SKIPPED for violating P>=Q_part_max -- that is a correct,
// intentional outcome, not a test failure). The caller (main) tracks
// this so a correctness regression actually fails the exit code, not
// just prints a warning that a CI/ctest run would silently ignore.
// encryptVariant: 0 = OLD encrypt_host, 1 = Phase 13 encrypt_host_
// shoup_pooled, 2 = Phase 14 encrypt_host_shoup_pooled_barrett,
// 3 = Phase 15 encrypt_host_shoup_pooled_barrett_batched.
static bool run_config(uint32_t n, uint32_t sizeQ, uint32_t sizeP, uint32_t numPart,
                        size_t reps, size_t trials, bool useFft, int encryptVariant)
{
    const char* encLabel = encryptVariant == 0 ? "batched-NTT (OLD, Phase 11)"
                          : encryptVariant == 1 ? "Shoup+pooled (Phase 13)"
                          : encryptVariant == 2 ? "Shoup+pooled+Barrett-embed (Phase 14)"
                          : encryptVariant == 3 ? "Shoup+pooled+Barrett+batched-pageable (Phase 15 Exp1, PRODUCTION)"
                          : encryptVariant == 4 ? "Shoup+pooled+Barrett+batched-pinned (Phase 15 Exp2, measured, NOT promoted -- see research/TRANSFER_PROFILING.md)"
                          : "Shoup+pooled+Barrett+batched-REGISTERED (Phase 20, cudaHostRegister prototype, see research/TRANSFER_PROFILING.md)";
    std::cout << "\n=== Configuration: n=" << n << " sizeQ=" << sizeQ
              << " sizeP=" << sizeP << " numPart=" << numPart
              << " encode/decode=" << (useFft ? "FFT (NEW, O(N log N))" : "parallel O(N^2) (OLD)")
              << " encrypt=" << encLabel << " ===\n";

    // --- Mandatory validation BEFORE building anything. Never bypassed. ---
    std::vector<uint64_t> mod, modP;
    gpufhe::native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids, sizeQ - 1, 50, n, mod);
      for (auto m : mids) mod.push_back(m); }
    gpufhe::native_primes(modP, sizeP, 60, n, mod);
    gpufhe::HybridKsValidation pv = gpufhe::check_hybrid_keyswitch_params(mod, modP, numPart);
    std::cout << "  P=" << pv.p_decimal << "\n  Q_part_max=" << pv.q_part_max_decimal
              << " (part " << pv.worst_part << ")\n  valid (P>=Q_part_max): " << (pv.ok ? "YES" : "NO") << "\n";
    if (!pv.ok) {
        std::cout << "  [SKIPPED] configuration violates the Hybrid key-switch's required "
                     "P>=Q_part_max condition -- refusing to run (see research/EXPERIMENT_LEDGER.md entry #42/#43)\n";
        return true; // a correct, intentional skip -- not a test failure
    }

    double gpu_before = gpu_used_mb();
    double host_before = host_rss_mb();

    // --- Cold start: fresh fixture creation, small sample (expensive). ---
    std::vector<double> cold_us;
    size_t cold_trials = std::min<size_t>(trials, 5);
    for (size_t trial = 0; trial < cold_trials; ++trial) {
        auto t0 = std::chrono::steady_clock::now();
        EvalMultFixtureCpp fx(n, sizeQ, sizeP, numPart, 1000 + trial);
        auto t1 = std::chrono::steady_clock::now();
        cold_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    report("1. Context + eval-key creation (cold)", stats_of(cold_us));

    // --- Build ONE fixture, reused for everything below. ---
    EvalMultFixtureCpp fx(n, sizeQ, sizeP, numPart, 42);
    double gpu_after_ctx = gpu_used_mb();

    // --- Correctness gate before trusting any timing. ---
    {
        std::mt19937_64 rng(555);
        std::vector<uint64_t> c0a(sizeQ * (size_t)n), c1a(sizeQ * (size_t)n), c0b(sizeQ * (size_t)n), c1b(sizeQ * (size_t)n);
        for (uint32_t t = 0; t < sizeQ; ++t) {
            std::uniform_int_distribution<uint64_t> d(0, fx.K.qMod[t] - 1);
            for (uint32_t k = 0; k < n; ++k) {
                size_t x = (size_t)t * n + k;
                c0a[x] = d(rng); c1a[x] = d(rng); c0b[x] = d(rng); c1b[x] = d(rng);
            }
        }
        std::vector<uint64_t> got0, got1;
        double u, te, ks, co, rs, dl;
        fx.multiply_relin_rescale_breakdown(c0a, c1a, c0b, c1b, got0, got1, u, te, ks, co, rs, dl);

        std::vector<uint64_t> refT0(sizeQ * (size_t)n), refT1(sizeQ * (size_t)n), refT2(sizeQ * (size_t)n);
        for (uint32_t t = 0; t < sizeQ; ++t) {
            uint64_t q = fx.K.qMod[t];
            for (uint32_t k = 0; k < n; ++k) {
                size_t x = (size_t)t * n + k;
                uint64_t a0 = c0a[x], a1 = c1a[x], b0 = c0b[x], b1 = c1b[x];
                refT0[x] = mulmod_b(a0, b0, q);
                uint64_t s = mulmod_b(a0, b1, q) + mulmod_b(a1, b0, q);
                refT1[x] = (s >= q) ? s - q : s;
                refT2[x] = mulmod_b(a1, b1, q);
            }
        }
        auto R = gpufhe::keyswitch_core_resident(refT2, fx.K);
        std::vector<uint64_t> refR0(sizeQ * (size_t)n), refR1(sizeQ * (size_t)n);
        for (uint32_t t = 0; t < sizeQ; ++t) {
            uint64_t q = fx.K.qMod[t];
            for (uint32_t k = 0; k < n; ++k) {
                size_t x = (size_t)t * n + k;
                uint64_t a = refT0[x] + R.ba0[x]; refR0[x] = (a >= q) ? a - q : a;
                uint64_t b = refT1[x] + R.ba1[x]; refR1[x] = (b >= q) ? b - q : b;
            }
        }
        uint64_t *d_ref0, *d_ref1, *d_scr2, *d_drp2;
        cudaMalloc(&d_ref0, sizeQ * (size_t)n * 8); cudaMalloc(&d_ref1, sizeQ * (size_t)n * 8);
        cudaMalloc(&d_scr2, (size_t)n * 8); cudaMalloc(&d_drp2, (size_t)n * 8);
        cudaMemcpy(d_ref0, refR0.data(), sizeQ * (size_t)n * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ref1, refR1.data(), sizeQ * (size_t)n * 8, cudaMemcpyHostToDevice);
        gpufhe::rescale_resident_raw(d_ref0, sizeQ, fx.C, fx.rs1, fx.rs2, d_scr2, d_drp2, 0);
        gpufhe::rescale_resident_raw(d_ref1, sizeQ, fx.C, fx.rs1, fx.rs2, d_scr2, d_drp2, 0);
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
            std::cerr << "  CORRECTNESS CHECK FAILED: " << mismatches << " mismatches -- refusing to report timing\n";
            return false;
        }
        std::cout << "  correctness check (RNS-level, tensor+relin+combine+rescale vs independent reference): 0/"
                  << (2 * exp0.size()) << " mismatches\n";
    }

    // --- REAL encode -> encrypt -> EvalMult -> decrypt -> decode round trip. ---
    {
        double delta = fx.scale();
        std::vector<double> slots_a = {1.5, -2.25, 3.0, 0.5};
        std::vector<double> slots_b = {2.0, -1.0, 0.5, 4.0};
        // encode_host requires a FULL n/2-length slot vector (zero-padded
        // for unused slots) -- the Rust FFI bridge (gpufhe_encode in
        // cpp_bridge/encdec_bridge.cpp) does this padding before calling
        // encode_host; calling encode_host directly with a short vector
        // is undefined behavior (out-of-bounds read inside its own S=n/2
        // loop) -- caught via a diagnostic before trusting this benchmark.
        std::vector<std::complex<double>> za(n / 2, {0.0, 0.0}), zb(n / 2, {0.0, 0.0});
        for (size_t i = 0; i < slots_a.size(); ++i) za[i] = {slots_a[i], 0.0};
        for (size_t i = 0; i < slots_b.size(); ++i) zb[i] = {slots_b[i], 0.0};

        std::vector<int64_t> ma, mb;
        auto t0 = std::chrono::steady_clock::now();
        if (useFft) {
            gpufhe::encode_host_fft(ma, za, n, delta);
            gpufhe::encode_host_fft(mb, zb, n, delta);
        } else {
            gpufhe::encode_host(ma, za, n, delta);
            gpufhe::encode_host(mb, zb, n, delta);
        }
        auto t1 = std::chrono::steady_clock::now();
        double encode_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        std::vector<uint64_t> c0a, c1a, c0b, c1b;
        t0 = std::chrono::steady_clock::now();
        if (encryptVariant == 5) {
            gpufhe::encrypt_host_shoup_pooled_barrett_batched_registered(c0a, c1a, ma, fx.pubKeyA, fx.pubKeyB, n, fx.modQ, fx.rootQ, 1, 3.19, 9001);
            gpufhe::encrypt_host_shoup_pooled_barrett_batched_registered(c0b, c1b, mb, fx.pubKeyA, fx.pubKeyB, n, fx.modQ, fx.rootQ, 1, 3.19, 9002);
        } else if (encryptVariant == 4) {
            gpufhe::encrypt_host_shoup_pooled_barrett_batched_pinned(c0a, c1a, ma, fx.pubKeyA, fx.pubKeyB, n, fx.modQ, fx.rootQ, 1, 3.19, 9001);
            gpufhe::encrypt_host_shoup_pooled_barrett_batched_pinned(c0b, c1b, mb, fx.pubKeyA, fx.pubKeyB, n, fx.modQ, fx.rootQ, 1, 3.19, 9002);
        } else if (encryptVariant == 3) {
            gpufhe::encrypt_host_shoup_pooled_barrett_batched(c0a, c1a, ma, fx.pubKeyA, fx.pubKeyB, n, fx.modQ, fx.rootQ, 1, 3.19, 9001);
            gpufhe::encrypt_host_shoup_pooled_barrett_batched(c0b, c1b, mb, fx.pubKeyA, fx.pubKeyB, n, fx.modQ, fx.rootQ, 1, 3.19, 9002);
        } else if (encryptVariant == 2) {
            gpufhe::encrypt_host_shoup_pooled_barrett(c0a, c1a, ma, fx.pubKeyA, fx.pubKeyB, n, fx.modQ, fx.rootQ, 1, 3.19, 9001);
            gpufhe::encrypt_host_shoup_pooled_barrett(c0b, c1b, mb, fx.pubKeyA, fx.pubKeyB, n, fx.modQ, fx.rootQ, 1, 3.19, 9002);
        } else if (encryptVariant == 1) {
            gpufhe::encrypt_host_shoup_pooled(c0a, c1a, ma, fx.pubKeyA, fx.pubKeyB, n, fx.modQ, fx.rootQ, 1, 3.19, 9001);
            gpufhe::encrypt_host_shoup_pooled(c0b, c1b, mb, fx.pubKeyA, fx.pubKeyB, n, fx.modQ, fx.rootQ, 1, 3.19, 9002);
        } else {
            gpufhe::encrypt_host(c0a, c1a, ma, fx.pubKeyA, fx.pubKeyB, n, fx.modQ, fx.rootQ, 1, 3.19, 9001);
            gpufhe::encrypt_host(c0b, c1b, mb, fx.pubKeyA, fx.pubKeyB, n, fx.modQ, fx.rootQ, 1, 3.19, 9002);
        }
        t1 = std::chrono::steady_clock::now();
        double encrypt_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        std::vector<uint64_t> out0, out1;
        double u, te, ks, co, rs, dl;
        fx.multiply_relin_rescale_breakdown(c0a, c1a, c0b, c1b, out0, out1, u, te, ks, co, rs, dl);

        std::vector<uint64_t> mods_resc(fx.modQ.begin(), fx.modQ.end() - 1);
        std::vector<uint64_t> roots_resc(fx.rootQ.begin(), fx.rootQ.end() - 1);
        std::vector<uint64_t> s_resc(fx.secretKey.begin(), fx.secretKey.begin() + (size_t)(sizeQ - 1) * n);

        std::vector<int64_t> mdec;
        t0 = std::chrono::steady_clock::now();
        gpufhe::decrypt_host(mdec, out0, out1, s_resc, n, mods_resc, roots_resc);
        t1 = std::chrono::steady_clock::now();
        double decrypt_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        double delta_post = delta * delta / (double)fx.modQ[sizeQ - 1];
        std::vector<std::complex<double>> zdec;
        t0 = std::chrono::steady_clock::now();
        if (useFft) gpufhe::decode_host_fft(zdec, mdec, n, delta_post);
        else gpufhe::decode_host(zdec, mdec, n, delta_post);
        t1 = std::chrono::steady_clock::now();
        double decode_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        double max_err = 0;
        for (size_t i = 0; i < slots_a.size(); ++i) {
            double expected = slots_a[i] * slots_b[i];
            double err = std::abs(zdec[i].real() - expected);
            if (err > max_err) max_err = err;
        }
        std::cout << "  real encrypted round trip: max abs error = " << max_err
                  << " (tolerance 1e-6) -> " << (max_err < 1e-6 ? "[PASS]" : "[FAIL]") << "\n";
        if (max_err >= 1e-6) { std::cerr << "  REAL ROUND TRIP FAILED -- refusing to report timing\n"; return false; }

        std::cout << "  -- single-call stage latencies (this real round trip) --\n";
        std::cout << "  2. Encode (host, x2 calls): " << encode_us << "us\n";
        std::cout << "  3. Encrypt (host+GPU, x2 calls): " << encrypt_us << "us\n";
        std::cout << "  4. Tensor multiplication (GPU): " << te << "us\n";
        std::cout << "  5. Relin/key-switch (GPU): " << ks << "us\n";
        std::cout << "     (combine (GPU): " << co << "us)\n";
        std::cout << "  6. Rescale x2 (GPU): " << rs << "us\n";
        std::cout << "  7. Decrypt (host+GPU): " << decrypt_us << "us\n";
        std::cout << "  8. Decode (host): " << decode_us << "us\n";
        std::cout << "  (upload: " << u << "us, download: " << dl << "us)\n";
        std::cout << "  9. Complete end-to-end (sum of above): "
                  << (encode_us + encrypt_us + u + te + ks + co + rs + dl + decrypt_us + decode_us) << "us\n";
    }

    // --- Steady-state statistics over many reps/trials (RNS-level, no
    // real encode/decode -- isolates the GPU-resident hot path's own
    // variance from host-side encode/decode cost). ---
    std::mt19937_64 rng(2024);
    std::vector<uint64_t> c0a(sizeQ * (size_t)n), c1a(sizeQ * (size_t)n), c0b(sizeQ * (size_t)n), c1b(sizeQ * (size_t)n);
    for (uint32_t t = 0; t < sizeQ; ++t) {
        std::uniform_int_distribution<uint64_t> d(0, fx.K.qMod[t] - 1);
        for (uint32_t k = 0; k < n; ++k) {
            size_t x = (size_t)t * n + k;
            c0a[x] = d(rng); c1a[x] = d(rng); c0b[x] = d(rng); c1b[x] = d(rng);
        }
    }
    std::vector<uint64_t> out0, out1;
    double u, te, ks, co, rs, dl;
    fx.multiply_relin_rescale_breakdown(c0a, c1a, c0b, c1b, out0, out1, u, te, ks, co, rs, dl); // warm-up

    std::vector<double> up_s, te_s, ks_s, co_s, rs_s, dl_s, total_s;
    for (size_t trial = 0; trial < trials; ++trial) {
        std::vector<double> up_r, te_r, ks_r, co_r, rs_r, dl_r, tot_r;
        for (size_t r = 0; r < reps; ++r) {
            auto h0 = std::chrono::steady_clock::now();
            fx.multiply_relin_rescale_breakdown(c0a, c1a, c0b, c1b, out0, out1, u, te, ks, co, rs, dl);
            auto h1 = std::chrono::steady_clock::now();
            up_r.push_back(u); te_r.push_back(te); ks_r.push_back(ks);
            co_r.push_back(co); rs_r.push_back(rs); dl_r.push_back(dl);
            tot_r.push_back(std::chrono::duration<double, std::micro>(h1 - h0).count());
        }
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        up_s.push_back(med(up_r)); te_s.push_back(med(te_r)); ks_s.push_back(med(ks_r));
        co_s.push_back(med(co_r)); rs_s.push_back(med(rs_r)); dl_s.push_back(med(dl_r));
        total_s.push_back(med(tot_r));
    }
    std::cout << "  -- steady-state (RNS-level, reused context, " << trials << " trials x " << reps << " reps, per-trial medians) --\n";
    report("upload (H2D, GPU event)", stats_of(up_s));
    report("tensor (GPU event)", stats_of(te_s));
    report("keyswitch (GPU event)", stats_of(ks_s));
    report("combine (GPU event)", stats_of(co_s));
    report("rescale x2 (GPU event)", stats_of(rs_s));
    report("download (D2H, GPU event)", stats_of(dl_s));
    report("full multiply_relin_rescale wall time (host clock)", stats_of(total_s));

    double gpu_after_run = gpu_used_mb();
    double host_after = host_rss_mb();
    std::cout << "  -- memory --\n";
    std::cout << "  GPU used before context: " << gpu_before << "MB\n";
    std::cout << "  GPU used after context+key creation: " << gpu_after_ctx << "MB (delta "
              << (gpu_after_ctx - gpu_before) << "MB)\n";
    std::cout << "  GPU used after steady-state run: " << gpu_after_run << "MB (delta "
              << (gpu_after_run - gpu_after_ctx) << "MB)\n";
    std::cout << "  Host RSS before: " << host_before << "MB, after: " << host_after << "MB\n";
    return true;
}

int main(int argc, char** argv) {
    size_t reps = argc > 1 ? (size_t)std::atoi(argv[1]) : 20;
    size_t trials = argc > 2 ? (size_t)std::atoi(argv[2]) : 7;
    bool allPass = true;

    // Fixed sizeQ=8/sizeP=2/numPart=4 (the Phase 9/10-corrected, validated
    // configuration), swept across n. Per the directive: start small,
    // increase only as memory allows; every config is validated before
    // running (see run_config's own mandatory check).
    if (argc > 3) {
        uint32_t n = (uint32_t)std::atoi(argv[3]);
        uint32_t sizeQ = argc > 4 ? (uint32_t)std::atoi(argv[4]) : 8;
        uint32_t sizeP = argc > 5 ? (uint32_t)std::atoi(argv[5]) : 2;
        uint32_t numPart = argc > 6 ? (uint32_t)std::atoi(argv[6]) : 4;
        bool useFft = argc > 7 ? (std::atoi(argv[7]) != 0) : true; // default: NEW (production) path
        int encryptVariant = argc > 8 ? std::atoi(argv[8]) : 3; // default: NEW (production) path -- see run_config's own comment: variant 4 (pinned) was measured and NOT promoted to production, see research/TRANSFER_PROFILING.md
        allPass = run_config(n, sizeQ, sizeP, numPart, reps, trials, useFft, encryptVariant);
    } else {
        // Full sweep, all six encrypt variants per n, for a direct
        // old-vs-Phase13-vs-Phase14-vs-Phase15(Exp1)-vs-Phase15(Exp2)-
        // vs-Phase20 full-pipeline comparison (GOD PROMPT "CKKS PHASE
        // 12" for encode/decode, "CKKS PHASE 13"/"14"/"15"/"20" for
        // encrypt).
        for (uint32_t n : {1024u, 2048u, 4096u, 8192u}) {
            if (!run_config(n, 8, 2, 4, reps, trials, /*useFft=*/true, /*encryptVariant=*/0)) allPass = false;
            if (!run_config(n, 8, 2, 4, reps, trials, /*useFft=*/true, /*encryptVariant=*/1)) allPass = false;
            if (!run_config(n, 8, 2, 4, reps, trials, /*useFft=*/true, /*encryptVariant=*/2)) allPass = false;
            if (!run_config(n, 8, 2, 4, reps, trials, /*useFft=*/true, /*encryptVariant=*/3)) allPass = false;
            if (!run_config(n, 8, 2, 4, reps, trials, /*useFft=*/true, /*encryptVariant=*/4)) allPass = false;
            if (!run_config(n, 8, 2, 4, reps, trials, /*useFft=*/true, /*encryptVariant=*/5)) allPass = false;
        }
    }
    if (!allPass) { std::cerr << "\n[OVERALL FAIL] at least one configuration's correctness check failed\n"; return 1; }
    std::cout << "\n[OVERALL PASS]\n";
    return 0;
}
