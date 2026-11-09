// GOD PROMPT "CKKS PHASE 26", Mission 2: standalone differential/correctness
// test for the fused plaintext-multiply-accumulate kernel (LaunchPtMulAcc,
// kernels/stage_acc.cu), extending tests/test_stage_acc.cpp's single-shot
// n=8192/tw=30 gate to the directive's explicit coverage bar.
//
// BACKGROUND (see research/PHASE26_RECONCILIATION.md for the full story):
// this kernel's origin is the Windows-clone commit "Fused pt-mul-accumulate
// kernel (acc += rot_ct * diag): bit-exact vs host ct_mul_pt + ct_add_ct",
// which turned out to ALREADY be merged into main 82 commits ago. It exists,
// has its own bit-exact test, but is NOT wired into tests/test_boot.cpp's
// actual per-diagonal hot loop (which still calls ct_mul_pt_host +
// ct_add_ct_host on the CPU). This file does not change that wiring -- it is
// purely a correctness investigation of the kernel itself, to the bar Mission
// 2 requires, so that a future wiring decision has real evidence behind it.
//
// SEMANTICS UNDER TEST (src/ptops.cpp is the ground truth being fused):
//   host: ct_mul_pt_host(t0,t1,d)   : t0 = t0*d mod q, t1 = t1*d mod q  (eval domain)
//         ct_add_ct_host(acc,t)     : acc0 += t0 mod q, acc1 += t1 mod q
//   device (LaunchPtMulAcc), ONE call replacing both of the above:
//         acc0 = (acc0 + b0*d) mod q, acc1 = (acc1 + b1*d) mod q
// Both sides use a plain 128-bit-intermediate modulo (no Barrett/Shoup
// approximation on either side), so equality is EXACT, not noise-tolerant --
// any mismatch, however small, is a real defect.
//
// WHAT THIS FILE ADDS over test_stage_acc.cpp's existing single-shot gate:
//   - multiple (n, towers) pairs, including the bootstrap's own n=8192/tw=30
//   - BOUNDARY coefficient values (0, q-1, (q-1)/2, 1, q-2) deliberately
//     placed, not just uniform random -- these are exactly the values a
//     modular-add/modular-mul off-by-one would mishandle
//   - MULTI-DIAGONAL accumulation: a single accumulator updated by 16
//     sequential fused calls on one stream (mirroring the ~16-diagonal BSGS
//     structure test_boot.cpp's own stage loop uses -- see
//     research/BOOTSTRAP_GPU_RESIDENCY_AUDIT.md section 2), verified against
//     the SAME 16-step host sequence, not just a single accumulate
//   - REPEATED executions: the full sweep runs multiple times with distinct
//     RNG seeds, not once
//   - CONCURRENT execution: K independent accumulation chains, each on its
//     own stream and own disjoint device buffers, launched to overlap, then
//     verified independently -- this is the test for exactly the class of bug
//     Phase 25 found in xf_multi's shared `scratch()` buffer. LaunchPtMulAcc
//     takes every buffer as an explicit caller-owned pointer and holds no
//     static/global device state at all (confirmed by reading
//     kernels/stage_acc.cu: zero `static` declarations), so concurrent calls
//     on disjoint memory should be safe by construction -- this test is the
//     evidence for that claim, not a substitute for it.
//   - buffer-lifetime / stream-ordering check: the kernel is launched async
//     with no internal synchronization (by design, so a caller can chain many
//     diagonals on one stream without a sync between each); this test
//     confirms that relying on stream ordering alone (one
//     cudaStreamSynchronize after N chained launches) is sufficient and that
//     no diagonal's result is lost or duplicated.
//
// WHAT THIS FILE DOES NOT DO: it does not change which code path
// test_boot.cpp's hot loop uses, and it is not a bootstrap-level measurement.
// Any performance numbers below describe the KERNEL AND HOST-OP PAIR IN
// ISOLATION, not an end-to-end bootstrap effect.
#include "stage_acc.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <random>
#include <chrono>
#include <algorithm>

namespace gpufhe {
void ct_mul_pt_host(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<uint64_t>&,
                    uint32_t, uint32_t, const std::vector<uint64_t>&);
void ct_add_ct_host(std::vector<uint64_t>&, std::vector<uint64_t>&,
                    const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                    uint32_t, uint32_t, const std::vector<uint64_t>&);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
}

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  [PASS] %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  [FAIL] %s\n", what.c_str()); }
}

static std::vector<uint64_t> buildModChain(uint32_t n, uint32_t tw) {
    std::vector<uint64_t> mod;
    gpufhe::native_primes(mod, 1, 60, n, {});
    if (tw > 1) {
        std::vector<uint64_t> mid;
        gpufhe::native_primes(mid, tw - 1, 55, n, mod);
        for (auto m : mid) mod.push_back(m);
    }
    return mod;
}

// ===========================================================================
// PART 1 -- boundary coefficient values, deliberately placed, not random.
// ===========================================================================
static void part1_boundaryValues() {
    std::printf("\n=== PART 1: boundary coefficient values ===\n");
    const uint32_t n = 2048, tw = 4;
    auto mod = buildModChain(n, tw);
    const size_t T = (size_t)tw * n;

    std::vector<uint64_t> a0(T), a1(T), b0(T), b1(T), d(T);
    std::mt19937_64 rng(31337);
    for (uint32_t t = 0; t < tw; ++t) {
        uint64_t q = mod[t];
        // Boundary set cycled across coefficients, mixed with a few random
        // values so the boundary cases sit next to ordinary ones (catching a
        // bug that only appears at a boundary/non-boundary transition).
        const uint64_t boundarySet[7] = {0, 1, q - 1, q - 2, (q - 1) / 2, (q + 1) / 2, q / 2};
        for (uint32_t k = 0; k < n; ++k) {
            size_t x = (size_t)t * n + k;
            a0[x] = boundarySet[k % 7];
            a1[x] = boundarySet[(k + 3) % 7];
            b0[x] = boundarySet[(k + 1) % 7];
            b1[x] = boundarySet[(k + 5) % 7];
            d[x]  = (k % 2 == 0) ? boundarySet[(k + 2) % 7] : (rng() % q);
        }
    }

    std::vector<uint64_t> h0 = a0, h1 = a1, t0 = b0, t1 = b1;
    gpufhe::ct_mul_pt_host(t0, t1, d, tw, n, mod);
    gpufhe::ct_add_ct_host(h0, h1, t0, t1, tw, n, mod);

    uint64_t *dA0, *dA1, *dB0, *dB1, *dD;
    cudaMalloc(&dA0, T * 8); cudaMalloc(&dA1, T * 8);
    cudaMalloc(&dB0, T * 8); cudaMalloc(&dB1, T * 8); cudaMalloc(&dD, T * 8);
    cudaMemcpy(dA0, a0.data(), T * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dA1, a1.data(), T * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dB0, b0.data(), T * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dB1, b1.data(), T * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dD,  d.data(),  T * 8, cudaMemcpyHostToDevice);
    for (uint32_t t = 0; t < tw; ++t)
        LaunchPtMulAcc(dA0 + (size_t)t * n, dA1 + (size_t)t * n,
                       dB0 + (size_t)t * n, dB1 + (size_t)t * n,
                       dD + (size_t)t * n, mod[t], n, 0);
    cudaDeviceSynchronize();
    std::vector<uint64_t> g0(T), g1(T);
    cudaMemcpy(g0.data(), dA0, T * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(g1.data(), dA1, T * 8, cudaMemcpyDeviceToHost);

    size_t bad = 0;
    for (size_t i = 0; i < T; ++i) { if (g0[i] != h0[i]) ++bad; if (g1[i] != h1[i]) ++bad; }
    check(bad == 0, "boundary values (0, 1, q-1, q-2, (q-1)/2, (q+1)/2, q/2) bit-exact, " +
                    std::to_string(bad) + "/" + std::to_string(2 * T) + " mismatches");

    cudaFree(dA0); cudaFree(dA1); cudaFree(dB0); cudaFree(dB1); cudaFree(dD);
}

// ===========================================================================
// PART 2 -- multiple (n, towers) pairs, including the bootstrap's own config.
// ===========================================================================
static void part2_multipleSizes(uint64_t seed) {
    std::printf("\n=== PART 2: multiple (n, towers) pairs (seed=%llu) ===\n", (unsigned long long)seed);
    struct Spec { uint32_t n, tw; };
    const std::vector<Spec> specs = {
        {1024, 1}, {1024, 4}, {2048, 4}, {4096, 8}, {8192, 30},  // 8192/30 = the real bootstrap config
    };
    for (auto& sp : specs) {
        auto mod = buildModChain(sp.n, sp.tw);
        const size_t T = (size_t)sp.tw * sp.n;
        std::mt19937_64 rng(seed + sp.n + sp.tw);
        std::vector<uint64_t> a0(T), a1(T), b0(T), b1(T), d(T);
        for (uint32_t t = 0; t < sp.tw; ++t) {
            uint64_t q = mod[t];
            for (uint32_t k = 0; k < sp.n; ++k) {
                size_t x = (size_t)t * sp.n + k;
                a0[x] = rng() % q; a1[x] = rng() % q;
                b0[x] = rng() % q; b1[x] = rng() % q; d[x] = rng() % q;
            }
        }
        std::vector<uint64_t> h0 = a0, h1 = a1, t0 = b0, t1 = b1;
        gpufhe::ct_mul_pt_host(t0, t1, d, sp.tw, sp.n, mod);
        gpufhe::ct_add_ct_host(h0, h1, t0, t1, sp.tw, sp.n, mod);

        uint64_t *dA0, *dA1, *dB0, *dB1, *dD;
        cudaMalloc(&dA0, T * 8); cudaMalloc(&dA1, T * 8);
        cudaMalloc(&dB0, T * 8); cudaMalloc(&dB1, T * 8); cudaMalloc(&dD, T * 8);
        cudaMemcpy(dA0, a0.data(), T * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dA1, a1.data(), T * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dB0, b0.data(), T * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dB1, b1.data(), T * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dD,  d.data(),  T * 8, cudaMemcpyHostToDevice);
        for (uint32_t t = 0; t < sp.tw; ++t)
            LaunchPtMulAcc(dA0 + (size_t)t * sp.n, dA1 + (size_t)t * sp.n,
                           dB0 + (size_t)t * sp.n, dB1 + (size_t)t * sp.n,
                           dD + (size_t)t * sp.n, mod[t], sp.n, 0);
        cudaDeviceSynchronize();
        std::vector<uint64_t> g0(T), g1(T);
        cudaMemcpy(g0.data(), dA0, T * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(g1.data(), dA1, T * 8, cudaMemcpyDeviceToHost);
        size_t bad = 0;
        for (size_t i = 0; i < T; ++i) { if (g0[i] != h0[i]) ++bad; if (g1[i] != h1[i]) ++bad; }
        check(bad == 0, "n=" + std::to_string(sp.n) + " tw=" + std::to_string(sp.tw) +
                        " bit-exact, " + std::to_string(bad) + "/" + std::to_string(2 * T) + " mismatches");
        cudaFree(dA0); cudaFree(dA1); cudaFree(dB0); cudaFree(dB1); cudaFree(dD);
    }
}

// ===========================================================================
// PART 3 -- multi-diagonal accumulation: ONE accumulator, 16 sequential fused
// calls on ONE stream (mirroring the real BSGS stage structure), verified
// after all 16 against the same 16-step host sequence -- not just a single
// accumulate. This is the realistic usage pattern, not the isolated op.
// ===========================================================================
static void part3_multiDiagonal(uint64_t seed) {
    std::printf("\n=== PART 3: multi-diagonal accumulation (16 sequential diagonals, seed=%llu) ===\n",
                (unsigned long long)seed);
    const uint32_t n = 4096, tw = 8, NDIAG = 16;  // 16 matches 2^RST at RST=4, test_boot.cpp's own LST/RST split
    auto mod = buildModChain(n, tw);
    const size_t T = (size_t)tw * n;
    std::mt19937_64 rng(seed + 777);

    std::vector<uint64_t> h0(T, 0), h1(T, 0);   // host accumulator, starts at zero like a fresh stage
    std::vector<std::vector<uint64_t>> allB0(NDIAG), allB1(NDIAG), allD(NDIAG);
    for (uint32_t g = 0; g < NDIAG; ++g) {
        allB0[g].resize(T); allB1[g].resize(T); allD[g].resize(T);
        for (uint32_t t = 0; t < tw; ++t) {
            uint64_t q = mod[t];
            for (uint32_t k = 0; k < n; ++k) {
                size_t x = (size_t)t * n + k;
                allB0[g][x] = rng() % q; allB1[g][x] = rng() % q; allD[g][x] = rng() % q;
            }
        }
        std::vector<uint64_t> t0 = allB0[g], t1 = allB1[g];
        gpufhe::ct_mul_pt_host(t0, t1, allD[g], tw, n, mod);
        gpufhe::ct_add_ct_host(h0, h1, t0, t1, tw, n, mod);
    }

    // Device: same 16-step sequence, ALL on ONE stream, ONE
    // cudaStreamSynchronize at the very end -- proving stream ordering alone
    // (no per-diagonal sync) correctly serializes the accumulation.
    cudaStream_t s; cudaStreamCreate(&s);
    uint64_t *dAcc0, *dAcc1, *dB0, *dB1, *dD;
    cudaMalloc(&dAcc0, T * 8); cudaMalloc(&dAcc1, T * 8);
    cudaMalloc(&dB0, T * 8); cudaMalloc(&dB1, T * 8); cudaMalloc(&dD, T * 8);
    cudaMemsetAsync(dAcc0, 0, T * 8, s);
    cudaMemsetAsync(dAcc1, 0, T * 8, s);
    for (uint32_t g = 0; g < NDIAG; ++g) {
        cudaMemcpyAsync(dB0, allB0[g].data(), T * 8, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(dB1, allB1[g].data(), T * 8, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(dD,  allD[g].data(),  T * 8, cudaMemcpyHostToDevice, s);
        for (uint32_t t = 0; t < tw; ++t)
            LaunchPtMulAcc(dAcc0 + (size_t)t * n, dAcc1 + (size_t)t * n,
                           dB0 + (size_t)t * n, dB1 + (size_t)t * n,
                           dD + (size_t)t * n, mod[t], n, s);
    }
    cudaStreamSynchronize(s);
    std::vector<uint64_t> g0(T), g1(T);
    cudaMemcpy(g0.data(), dAcc0, T * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(g1.data(), dAcc1, T * 8, cudaMemcpyDeviceToHost);

    size_t bad = 0;
    for (size_t i = 0; i < T; ++i) { if (g0[i] != h0[i]) ++bad; if (g1[i] != h1[i]) ++bad; }
    check(bad == 0, "16-diagonal accumulation (1 stream, 1 sync at the end) bit-exact, " +
                    std::to_string(bad) + "/" + std::to_string(2 * T) + " mismatches");

    cudaFree(dAcc0); cudaFree(dAcc1); cudaFree(dB0); cudaFree(dB1); cudaFree(dD);
    cudaStreamDestroy(s);
}

// ===========================================================================
// PART 4 -- concurrent execution: K independent accumulation chains, each on
// its own stream and own DISJOINT device buffers, launched to overlap in
// time, then verified independently. LaunchPtMulAcc holds no static/global
// device state (unlike xf_multi's scratch() buffer, the actual root cause of
// Phase 25's concurrency bug) -- this test is the evidence for that claim.
// ===========================================================================
static void part4_concurrentChains(uint64_t seed) {
    std::printf("\n=== PART 4: concurrent independent accumulation chains (seed=%llu) ===\n",
                (unsigned long long)seed);
    const uint32_t n = 2048, tw = 4, NDIAG = 8;
    const size_t K = 6;  // 6 independent concurrent chains
    auto mod = buildModChain(n, tw);
    const size_t T = (size_t)tw * n;

    struct Chain {
        std::vector<uint64_t> hAcc0, hAcc1;
        std::vector<std::vector<uint64_t>> b0, b1, d;
        cudaStream_t stream;
        uint64_t *dAcc0, *dAcc1, *dB0, *dB1, *dD;
        std::vector<uint64_t> gAcc0, gAcc1;
    };
    std::vector<Chain> chains(K);

    // Build each chain's own host reference FIRST (serially), exactly as
    // Part 3 does, so each chain has an independently-correct expectation.
    for (size_t c = 0; c < K; ++c) {
        Chain& C = chains[c];
        std::mt19937_64 rng(seed + 1000 * c + 55);
        C.hAcc0.assign(T, 0); C.hAcc1.assign(T, 0);
        C.b0.resize(NDIAG); C.b1.resize(NDIAG); C.d.resize(NDIAG);
        for (uint32_t g = 0; g < NDIAG; ++g) {
            C.b0[g].resize(T); C.b1[g].resize(T); C.d[g].resize(T);
            for (uint32_t t = 0; t < tw; ++t) {
                uint64_t q = mod[t];
                for (uint32_t k = 0; k < n; ++k) {
                    size_t x = (size_t)t * n + k;
                    C.b0[g][x] = rng() % q; C.b1[g][x] = rng() % q; C.d[g][x] = rng() % q;
                }
            }
            std::vector<uint64_t> t0 = C.b0[g], t1 = C.b1[g];
            gpufhe::ct_mul_pt_host(t0, t1, C.d[g], tw, n, mod);
            gpufhe::ct_add_ct_host(C.hAcc0, C.hAcc1, t0, t1, tw, n, mod);
        }
    }

    // Allocate disjoint device buffers and streams for every chain BEFORE
    // launching anything, so the launches below genuinely overlap rather than
    // being serialized by allocation.
    for (size_t c = 0; c < K; ++c) {
        Chain& C = chains[c];
        cudaStreamCreate(&C.stream);
        cudaMalloc(&C.dAcc0, T * 8); cudaMalloc(&C.dAcc1, T * 8);
        cudaMalloc(&C.dB0, T * 8); cudaMalloc(&C.dB1, T * 8); cudaMalloc(&C.dD, T * 8);
        cudaMemsetAsync(C.dAcc0, 0, T * 8, C.stream);
        cudaMemsetAsync(C.dAcc1, 0, T * 8, C.stream);
    }

    // Launch all K chains' full diagonal sequences, interleaved round-robin
    // across diagonals (not chain-by-chain) to maximise actual overlap on the
    // GPU's queues before any synchronization happens.
    for (uint32_t g = 0; g < NDIAG; ++g) {
        for (size_t c = 0; c < K; ++c) {
            Chain& C = chains[c];
            cudaMemcpyAsync(C.dB0, C.b0[g].data(), T * 8, cudaMemcpyHostToDevice, C.stream);
            cudaMemcpyAsync(C.dB1, C.b1[g].data(), T * 8, cudaMemcpyHostToDevice, C.stream);
            cudaMemcpyAsync(C.dD,  C.d[g].data(),  T * 8, cudaMemcpyHostToDevice, C.stream);
            for (uint32_t t = 0; t < tw; ++t)
                LaunchPtMulAcc(C.dAcc0 + (size_t)t * n, C.dAcc1 + (size_t)t * n,
                               C.dB0 + (size_t)t * n, C.dB1 + (size_t)t * n,
                               C.dD + (size_t)t * n, mod[t], n, C.stream);
        }
    }
    for (size_t c = 0; c < K; ++c) cudaStreamSynchronize(chains[c].stream);

    size_t totalBad = 0;
    for (size_t c = 0; c < K; ++c) {
        Chain& C = chains[c];
        C.gAcc0.resize(T); C.gAcc1.resize(T);
        cudaMemcpy(C.gAcc0.data(), C.dAcc0, T * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(C.gAcc1.data(), C.dAcc1, T * 8, cudaMemcpyDeviceToHost);
        size_t bad = 0;
        for (size_t i = 0; i < T; ++i) { if (C.gAcc0[i] != C.hAcc0[i]) ++bad; if (C.gAcc1[i] != C.hAcc1[i]) ++bad; }
        totalBad += bad;
        check(bad == 0, "chain " + std::to_string(c) + "/" + std::to_string(K) +
                        " (interleaved concurrent launch) bit-exact vs its OWN host reference, " +
                        std::to_string(bad) + " mismatches");
        cudaFree(C.dAcc0); cudaFree(C.dAcc1); cudaFree(C.dB0); cudaFree(C.dB1); cudaFree(C.dD);
        cudaStreamDestroy(C.stream);
    }
    check(totalBad == 0, std::to_string(K) + " concurrent chains, zero cross-contamination (no shared state)");
}

// ===========================================================================
// PART 5 -- benchmark, NOT a correctness check: kernel time vs transfer time
// vs total op time, measured separately, for the fused device op against the
// host pair it replaces. Warm-up, alternating A/B order, multiple trials --
// same methodology as bench/bench_hoist_ab.cpp. This describes the KERNEL AND
// HOST-OP PAIR IN ISOLATION, explicitly NOT a bootstrap or end-to-end figure.
// ===========================================================================
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

static void part5_benchmark() {
    std::printf("\n=== PART 5: benchmark (isolated op, NOT a bootstrap figure) ===\n");
    const uint32_t n = 8192, tw = 30;  // the real bootstrap config
    auto mod = buildModChain(n, tw);
    const size_t T = (size_t)tw * n;
    std::mt19937_64 rng(999);
    std::vector<uint64_t> a0(T), a1(T), b0(T), b1(T), d(T);
    for (uint32_t t = 0; t < tw; ++t) { uint64_t q = mod[t];
        for (uint32_t k = 0; k < n; ++k) { size_t x = (size_t)t * n + k;
            a0[x] = rng() % q; a1[x] = rng() % q; b0[x] = rng() % q; b1[x] = rng() % q; d[x] = rng() % q; } }

    uint64_t *dA0, *dA1, *dB0, *dB1, *dD;
    cudaMalloc(&dA0, T * 8); cudaMalloc(&dA1, T * 8);
    cudaMalloc(&dB0, T * 8); cudaMalloc(&dB1, T * 8); cudaMalloc(&dD, T * 8);
    cudaMemcpy(dB0, b0.data(), T * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dB1, b1.data(), T * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dD,  d.data(),  T * 8, cudaMemcpyHostToDevice);

    auto nowS = [](){ return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count(); };

    // Arm FUSED: data already resident; measure KERNEL-ONLY time (all towers).
    auto armFusedKernelOnly = [&]() {
        cudaMemcpy(dA0, a0.data(), T * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dA1, a1.data(), T * 8, cudaMemcpyHostToDevice);
        double t0 = nowS();
        for (uint32_t t = 0; t < tw; ++t)
            LaunchPtMulAcc(dA0 + (size_t)t * n, dA1 + (size_t)t * n,
                           dB0 + (size_t)t * n, dB1 + (size_t)t * n,
                           dD + (size_t)t * n, mod[t], n, 0);
        cudaDeviceSynchronize();
        return nowS() - t0;
    };
    // Arm FUSED total: includes the H2D of acc (the only transfer this
    // isolated op needs beyond what's already resident) + kernel + D2H.
    auto armFusedTotal = [&]() {
        double t0 = nowS();
        cudaMemcpy(dA0, a0.data(), T * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dA1, a1.data(), T * 8, cudaMemcpyHostToDevice);
        for (uint32_t t = 0; t < tw; ++t)
            LaunchPtMulAcc(dA0 + (size_t)t * n, dA1 + (size_t)t * n,
                           dB0 + (size_t)t * n, dB1 + (size_t)t * n,
                           dD + (size_t)t * n, mod[t], n, 0);
        std::vector<uint64_t> o0(T), o1(T);
        cudaMemcpy(o0.data(), dA0, T * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(o1.data(), dA1, T * 8, cudaMemcpyDeviceToHost);
        return nowS() - t0;
    };
    // Arm HOST pair: ct_mul_pt_host + ct_add_ct_host, pure CPU, no transfer at
    // all (data already in host vectors) -- the two are NOT directly
    // comparable without matching what crosses PCIe; both numbers are
    // reported, scope stated explicitly in the output.
    auto armHostPair = [&]() {
        std::vector<uint64_t> acc0 = a0, acc1 = a1, t0v = b0, t1v = b1;
        double t0 = nowS();
        gpufhe::ct_mul_pt_host(t0v, t1v, d, tw, n, mod);
        gpufhe::ct_add_ct_host(acc0, acc1, t0v, t1v, tw, n, mod);
        return nowS() - t0;
    };

    // warm-up
    for (int i = 0; i < 3; ++i) { armFusedKernelOnly(); armFusedTotal(); armHostPair(); }

    const int TRIALS = 11;
    std::vector<double> vKernel, vTotal, vHost;
    for (int i = 0; i < TRIALS; ++i) {
        if (i % 2 == 0) { vKernel.push_back(armFusedKernelOnly()); vHost.push_back(armHostPair()); }
        else            { vHost.push_back(armHostPair()); vKernel.push_back(armFusedKernelOnly()); }
        vTotal.push_back(armFusedTotal());
    }
    size_t vramFree = 0, vramTotal = 0;
    cudaMemGetInfo(&vramFree, &vramTotal);

    std::printf("  n=%u tw=%u, %d trials, median [min..max]:\n", n, tw, TRIALS);
    std::printf("    fused kernel-only (data resident)      : %.6f [%.6f..%.6f] s\n",
                median(vKernel), *std::min_element(vKernel.begin(), vKernel.end()),
                *std::max_element(vKernel.begin(), vKernel.end()));
    std::printf("    fused total (+ acc H2D/D2H this call)  : %.6f [%.6f..%.6f] s\n",
                median(vTotal), *std::min_element(vTotal.begin(), vTotal.end()),
                *std::max_element(vTotal.begin(), vTotal.end()));
    std::printf("    host pair (CPU only, no transfer)      : %.6f [%.6f..%.6f] s\n",
                median(vHost), *std::min_element(vHost.begin(), vHost.end()),
                *std::max_element(vHost.begin(), vHost.end()));
    std::printf("  VRAM free %.0f/%.0f MiB\n", vramFree / 1048576.0, vramTotal / 1048576.0);
    std::printf("  SCOPE: isolated op only (one accumulate step, all towers). NOT a bootstrap or\n");
    std::printf("  end-to-end figure. The host pair pays zero transfer in this harness because its\n");
    std::printf("  inputs are already host vectors; the fused arm's real-world advantage is that in\n");
    std::printf("  test_boot.cpp's ACTUAL hot loop the ciphertext is already device-resident from the\n");
    std::printf("  preceding rotation, so wiring this kernel in would remove a D2H+H2D round trip per\n");
    std::printf("  diagonal that this micro-benchmark does not model (see research/\n");
    std::printf("  BOOTSTRAP_GPU_RESIDENCY_AUDIT.md for that larger-context measurement).\n");

    cudaFree(dA0); cudaFree(dA1); cudaFree(dB0); cudaFree(dB1); cudaFree(dD);
}

int main() {
    std::printf("=== Mission 2: fused pt-mul-accumulate (LaunchPtMulAcc) differential test ===\n");
    part1_boundaryValues();
    // Repeated executions: run the size-sweep and multi-diagonal sweep THREE
    // times with distinct seeds, not once.
    for (uint64_t seed : {1001ull, 2002ull, 3003ull}) {
        part2_multipleSizes(seed);
        part3_multiDiagonal(seed);
    }
    part4_concurrentChains(42ull);
    part5_benchmark();

    std::printf("\n=== CHECKS: %d passed, %d failed ===\n", g_pass, g_fail);
    if (g_fail == 0) {
        std::printf("[PASS] LaunchPtMulAcc is bit-exact across sizes, boundary values, "
                    "multi-diagonal accumulation, repeated runs, and concurrent chains\n");
    } else {
        std::printf("[FAIL] %d check(s) failed\n", g_fail);
    }
    return g_fail == 0 ? 0 : 1;
}
