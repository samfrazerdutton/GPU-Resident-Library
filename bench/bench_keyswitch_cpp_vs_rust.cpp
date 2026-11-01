// GOD PROMPT "CKKS RUST + CUDA PHASE 3" round. Path A (existing CUDA C++
// kernels, host orchestration entirely in C++) of the fair comparison
// against rust/crates/ckks-cuda::KeySwitchContext (Path B). Deliberately
// calls the EXACT SAME sequence of EXISTING, unmodified gpufhe:: functions
// this round's cpp_bridge/keyswitch_bridge.cpp wraps for Rust --
// native_primes/native_root (src/params.cpp), keygen_host (src/keygen.cpp),
// compute_keyswitch_constants (src/ks_constants.cpp), evalkeygen_host
// (src/evalkeygen.cpp), ks_context_create/ks_work_create/keyswitch_resident
// (src/keyswitch_resident.cpp) -- so both languages are measured doing the
// SAME setup work, same kernels, same configuration (n=1024, sizeQ=8,
// sizeP=2, numPart=2, matching tests/test_keyswitch_core_native_diff.cpp
// and this round's Rust test module).
//
// UNLIKE rescale's benchmark, there is no "narrower setup" story available
// here: key switching genuinely needs the full eval key + DeviceKSContext +
// DeviceKSWork, so both languages build the SAME full fixture -- this file
// is not a deliberately-narrowed reimplementation, it is the same recipe
// invoked directly instead of through Rust FFI.
//
// Also unlike ntt/rescale's benchmarks, there is no meaningful host-only
// "cold-start" tier to measure separately: native_primes/native_root are
// host-only, but keygen_host and evalkeygen_host BOTH launch GPU kernels
// internally (LaunchNTT_CT / LaunchNTT_CT_Batched) as part of building the
// key -- there is no point in this pipeline before GPU allocation and
// kernel launches are already involved. So this benchmark reports three
// tiers, not four: fixture creation (the full setup, GPU-touching
// throughout), steady-state keyswitch (reused context), and end-to-end
// (fresh fixture + first keyswitch). This is an honest methodological
// difference from the ntt/rescale benchmarks, not an oversight.
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
}

static uint64_t mulmod_b(uint64_t a, uint64_t b, uint64_t q) { return (uint64_t)(((unsigned __int128)a * b) % q); }

// Mirrors rust/crates/ckks-cuda::keyswitch::KeySwitchContext field-for-field
// (modulo language idiom): owns K (host, incl. eval key), C (device
// context), W (device scratch). Built ONCE; keyswitch_resident() is the
// hot path.
struct KeySwitchFixtureCpp {
    gpufhe::KeySwitchConstants K;
    gpufhe::DeviceKSContext C;
    gpufhe::DeviceKSWork W;

    KeySwitchFixtureCpp(uint32_t n, uint32_t sizeQ, uint32_t sizeP, uint32_t numPart, uint64_t seed) {
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
    }
    ~KeySwitchFixtureCpp() {
        gpufhe::ks_work_destroy(W);
        gpufhe::ks_context_destroy(C);
    }

    void keyswitch(const std::vector<uint64_t>& a, std::vector<uint64_t>& ba0, std::vector<uint64_t>& ba1) {
        const size_t bytes = (size_t)K.n * K.sizeQl * 8;
        uint64_t *d_a, *d_ba0, *d_ba1;
        cudaMalloc(&d_a, bytes); cudaMalloc(&d_ba0, bytes); cudaMalloc(&d_ba1, bytes);
        cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
        gpufhe::keyswitch_resident(d_a, d_ba0, d_ba1, C, W, 0);
        cudaDeviceSynchronize();
        ba0.resize((size_t)K.n * K.sizeQl); ba1.resize((size_t)K.n * K.sizeQl);
        cudaMemcpy(ba0.data(), d_ba0, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(ba1.data(), d_ba1, bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_a); cudaFree(d_ba0); cudaFree(d_ba1);
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
    const uint32_t n = 1024, sizeQ = 8, sizeP = 2, numPart = 2;
    size_t reps = argc > 1 ? (size_t)std::atoi(argv[1]) : 20;
    size_t trials = argc > 2 ? (size_t)std::atoi(argv[2]) : 7;

    // --- Tier 1: fixture creation (full setup, GPU-touching -- see header
    // comment for why this isn't split into a separate host-only
    // "cold-start" sub-tier the way ntt/rescale's benchmarks are). ---
    std::vector<double> init_us;
    for (size_t trial = 0; trial < trials; ++trial) {
        auto t0 = std::chrono::steady_clock::now();
        KeySwitchFixtureCpp fx(n, sizeQ, sizeP, numPart, 1000 + trial);
        auto t1 = std::chrono::steady_clock::now();
        init_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    report("fixture creation (KeySwitchFixtureCpp constructor: modulus chain + keygen + evalkeygen + device context/work)", init_us);

    // --- Correctness gate, BEFORE any timing is trusted: compare
    // keyswitch_resident (device-resident, what we're benchmarking) against
    // keyswitch_core_resident (host-orchestrated reference) on the SAME K,
    // the established methodology for this operation (tests/
    // test_ks_resident.cpp does the same comparison at a different,
    // OpenFHE-sourced configuration). ---
    {
        KeySwitchFixtureCpp checkFx(n, sizeQ, sizeP, numPart, /*seed=*/777);
        std::mt19937_64 rng(555);
        std::vector<uint64_t> a((size_t)sizeQ * n);
        for (uint32_t t = 0; t < sizeQ; ++t) {
            std::uniform_int_distribution<uint64_t> d(0, checkFx.K.qMod[t] - 1);
            for (uint32_t k = 0; k < n; ++k) a[(size_t)t * n + k] = d(rng);
        }
        std::vector<uint64_t> ba0, ba1;
        checkFx.keyswitch(a, ba0, ba1);
        auto R = gpufhe::keyswitch_core_resident(a, checkFx.K);
        size_t mismatches = 0;
        for (size_t x = 0; x < ba0.size(); ++x) {
            if (ba0[x] != R.ba0[x]) ++mismatches;
            if (ba1[x] != R.ba1[x]) ++mismatches;
        }
        if (mismatches != 0) {
            std::cerr << "CORRECTNESS CHECK FAILED: " << mismatches << " mismatches -- refusing to report timing\n";
            return 1;
        }
        std::cout << "correctness check: 0/" << (2 * ba0.size()) << " mismatches vs keyswitch_core_resident (host reference)\n";
    }

    // --- Tier 2: steady-state, ONE fixture reused across all reps/trials. ---
    KeySwitchFixtureCpp ctxFx(n, sizeQ, sizeP, numPart, /*seed=*/42);
    std::mt19937_64 rng(2024);
    std::vector<uint64_t> input((size_t)sizeQ * n);
    for (uint32_t t = 0; t < sizeQ; ++t) {
        std::uniform_int_distribution<uint64_t> d(0, ctxFx.K.qMod[t] - 1);
        for (uint32_t k = 0; k < n; ++k) input[(size_t)t * n + k] = d(rng);
    }
    std::vector<uint64_t> ba0, ba1;
    ctxFx.keyswitch(input, ba0, ba1); // warm-up

    std::vector<double> per_trial_median;
    for (size_t trial = 0; trial < trials; ++trial) {
        std::vector<double> per_call;
        for (size_t r = 0; r < reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            ctxFx.keyswitch(input, ba0, ba1);
            auto t1 = std::chrono::steady_clock::now();
            per_call.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        per_trial_median.push_back(median(per_call));
    }
    report("steady-state keyswitch (reused fixture)", per_trial_median);

    // --- Tier 3: end-to-end, fresh fixture + first keyswitch, combined. ---
    std::vector<double> e2e_us;
    for (size_t trial = 0; trial < trials; ++trial) {
        auto t0 = std::chrono::steady_clock::now();
        KeySwitchFixtureCpp fresh(n, sizeQ, sizeP, numPart, 5000 + trial);
        std::vector<uint64_t> out0, out1;
        fresh.keyswitch(input, out0, out1);
        e2e_us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    report("end-to-end (fresh fixture + first keyswitch)", e2e_us);

    return 0;
}
