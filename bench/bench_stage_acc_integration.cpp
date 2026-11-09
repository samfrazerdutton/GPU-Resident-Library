// GOD PROMPT "CKKS PHASE 27", Mission 8: end-to-end STAGE-level A/B for
// wiring the fused pt-mul-accumulate kernel (LaunchPtMulAcc) into the
// bootstrap's per-diagonal accumulate loop, including transfers and
// synchronization -- not just the isolated kernel (which Phase 26 already
// proved bit-exact, 26/26 checks, ~7.6x in isolation via
// tests/test_stage_acc_differential.cpp).
//
// WHAT THIS FILE ADDS: a faithful simulation of ONE transform stage's real
// diagonal loop (NDIAG=16, matching the LST=3/RST=4 structure
// research/BOOTSTRAP_GPU_RESIDENCY_AUDIT.md documents), comparing:
//
//   ARM A (current production pattern, tests/test_boot.cpp's own
//          stageRotDiag/ct_mul_pt_host/ct_add_ct_host sequence, UNCHANGED):
//     per diagonal: encode_host (CPU) -> pt_to_eval_host (ONE H2D+NTT+D2H
//     round trip, unchanged -- NOT the target of this experiment) ->
//     ct_mul_pt_host (CPU) -> ct_add_ct_host (CPU).
//
//   ARM B (GPU-resident accumulate, using ONLY already-exported, already-
//          tested primitives -- no production internals touched):
//     per diagonal: encode_host (CPU, same) -> pt_to_eval_host (SAME round
//     trip, deliberately UNCHANGED -- a separate, larger optimization
//     target this experiment does not attempt) -> upload the diagonal's
//     eval-domain plaintext and the (simulated) rotated ciphertext ONCE ->
//     LaunchPtMulAcc directly into a DEVICE-RESIDENT accumulator (no
//     per-diagonal download). Only at the very END of the stage is the
//     accumulator downloaded ONCE.
//
// WHY pt_to_eval_host IS LEFT UNCHANGED IN BOTH ARMS: it already does its
// own complete H2D/NTT/D2H round trip inside an anonymous-namespace function
// (xf_multi) that is not part of this project's exported API. Making IT
// device-resident too would be a separate, larger change (the residency
// audit's O3) requiring its own seam and its own standalone test -- out of
// this experiment's scope, which is specifically "is the ACCUMULATE step
// worth moving to the fused kernel," per Mission 8's own framing.
//
// WHAT THIS ISOLATES: whether removing (a) the CPU multiply+add and (b) the
// per-diagonal accumulator download -- replacing them with one fused kernel
// launch and one final download -- improves END-TO-END STAGE LATENCY, not
// just the isolated accumulate operation. The ROTATION step itself is
// SIMULATED (a fixed host-side perturbation standing in for what a real
// rotate_ct_resident/rotate_ct_host call would produce) rather than actually
// computed, since rotation correctness is already covered by
// tests/test_rot_resident.cpp and is not what this experiment measures --
// both arms use the IDENTICAL simulated rotation output, so this choice
// cannot bias the comparison between them.
#include "keyswitch.h"
#include "stage_acc.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <random>
#include <sys/resource.h>

namespace gpufhe {
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void pt_to_eval_host(std::vector<uint64_t>&, const std::vector<int64_t>&, uint32_t, uint32_t,
                     const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void ct_mul_pt_host(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<uint64_t>&,
                    uint32_t, uint32_t, const std::vector<uint64_t>&);
void ct_add_ct_host(std::vector<uint64_t>&, std::vector<uint64_t>&,
                    const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                    uint32_t, uint32_t, const std::vector<uint64_t>&);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
}

using cd = std::complex<double>;
static double rssMB() { struct rusage ru; getrusage(RUSAGE_SELF, &ru); return ru.ru_maxrss / 1024.0; }
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end()); size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

int main(int argc, char** argv) {
    const uint32_t n = (argc > 1) ? (uint32_t)std::stoul(argv[1]) : 8192;
    const uint32_t tw = (argc > 2) ? (uint32_t)std::stoul(argv[2]) : 30;     // the bootstrap's own first-stage tw
    const uint32_t NDIAG = (argc > 3) ? (uint32_t)std::stoul(argv[3]) : 16;  // 2^RST at RST=4, matching real LST=3/RST=4
    const uint32_t TRIALS = (argc > 4) ? (uint32_t)std::stoul(argv[4]) : 9;

    std::vector<uint64_t> mod;
    gpufhe::native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> md; gpufhe::native_primes(md, tw - 1, 55, n, mod); for (auto m : md) mod.push_back(m); }
    std::vector<uint64_t> root(tw);
    for (uint32_t i = 0; i < tw; ++i) root[i] = gpufhe::native_root(n, mod[i]);
    const size_t T = (size_t)tw * n, B = T * 8;
    const double Delta = std::pow(2.0, 50);

    std::printf("=== Mission 8: stage-level accumulate A/B (CPU pair vs fused kernel, resident acc) ===\n");
    std::printf("params: n=%u tw=%u, NDIAG=%u diagonals/stage, %u trials\n", n, tw, NDIAG, TRIALS);

    // ---- deterministic diagonals and a SIMULATED rotated ciphertext t0/t1
    // per diagonal (identical in both arms -- see header comment for why
    // simulating rotation here cannot bias the comparison). ----
    std::mt19937_64 rng(12345);
    std::vector<std::vector<cd>> dg(NDIAG, std::vector<cd>(n / 2));
    std::vector<std::vector<uint64_t>> simT0(NDIAG), simT1(NDIAG);
    for (uint32_t x = 0; x < NDIAG; ++x) {
        for (uint32_t i = 0; i < n / 2; ++i) dg[x][i] = { 0.1 * std::sin(0.01 * i + x), 0.1 * std::cos(0.02 * i + x) };
        simT0[x].resize(T); simT1[x].resize(T);
        for (uint32_t t = 0; t < tw; ++t) { uint64_t q = mod[t];
            for (uint32_t k = 0; k < n; ++k) { size_t idx = (size_t)t * n + k;
                simT0[x][idx] = rng() % q; simT1[x][idx] = rng() % q; } }
    }

    auto nowS = []() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };

    // ---- ARM A: current production pattern ----
    auto armCurrentCPU = [&](std::vector<uint64_t>* outAcc0 = nullptr, std::vector<uint64_t>* outAcc1 = nullptr) {
        double t0 = nowS();
        std::vector<uint64_t> acc0(T, 0), acc1(T, 0);
        bool first = true;
        for (uint32_t x = 0; x < NDIAG; ++x) {
            std::vector<int64_t> md; gpufhe::encode_host(md, dg[x], n, Delta);
            std::vector<uint64_t> dE; gpufhe::pt_to_eval_host(dE, md, tw, n, mod, root);  // full round trip, UNCHANGED in both arms
            std::vector<uint64_t> a0 = simT0[x], a1 = simT1[x];
            gpufhe::ct_mul_pt_host(a0, a1, dE, tw, n, mod);
            if (first) { acc0 = a0; acc1 = a1; first = false; }
            else gpufhe::ct_add_ct_host(acc0, acc1, a0, a1, tw, n, mod);
        }
        if (outAcc0) *outAcc0 = acc0;
        if (outAcc1) *outAcc1 = acc1;
        return nowS() - t0;
    };

    // ---- ARM B: GPU-resident accumulate via LaunchPtMulAcc, one download
    // at the END of the stage instead of per-diagonal. pt_to_eval_host's
    // own round trip is UNCHANGED (see header comment). ----
    auto armResidentFused = [&](std::vector<uint64_t>* outAcc0 = nullptr, std::vector<uint64_t>* outAcc1 = nullptr) {
        double t0 = nowS();
        uint64_t *d_acc0, *d_acc1, *d_b0, *d_b1, *d_d;
        cudaMalloc(&d_acc0, B); cudaMalloc(&d_acc1, B);
        cudaMalloc(&d_b0, B); cudaMalloc(&d_b1, B); cudaMalloc(&d_d, B);
        cudaMemsetAsync(d_acc0, 0, B, 0); cudaMemsetAsync(d_acc1, 0, B, 0);
        for (uint32_t x = 0; x < NDIAG; ++x) {
            std::vector<int64_t> md; gpufhe::encode_host(md, dg[x], n, Delta);
            std::vector<uint64_t> dE; gpufhe::pt_to_eval_host(dE, md, tw, n, mod, root);  // SAME round trip as Arm A
            cudaMemcpyAsync(d_b0, simT0[x].data(), B, cudaMemcpyHostToDevice, 0);
            cudaMemcpyAsync(d_b1, simT1[x].data(), B, cudaMemcpyHostToDevice, 0);
            cudaMemcpyAsync(d_d, dE.data(), B, cudaMemcpyHostToDevice, 0);
            for (uint32_t t = 0; t < tw; ++t)
                LaunchPtMulAcc(d_acc0 + (size_t)t * n, d_acc1 + (size_t)t * n,
                               d_b0 + (size_t)t * n, d_b1 + (size_t)t * n,
                               d_d + (size_t)t * n, mod[t], n, 0);
            // NO per-diagonal download -- the accumulator stays resident.
        }
        std::vector<uint64_t> acc0(T), acc1(T);
        cudaMemcpy(acc0.data(), d_acc0, B, cudaMemcpyDeviceToHost);  // ONE download for the whole stage
        cudaMemcpy(acc1.data(), d_acc1, B, cudaMemcpyDeviceToHost);
        if (outAcc0) *outAcc0 = acc0;
        if (outAcc1) *outAcc1 = acc1;
        cudaFree(d_acc0); cudaFree(d_acc1); cudaFree(d_b0); cudaFree(d_b1); cudaFree(d_d);
        return nowS() - t0;
    };

    // ---- correctness FIRST ----
    std::printf("\n[correctness] Arm B's final accumulator must be BIT-EXACT vs Arm A's\n");
    std::vector<uint64_t> accA0, accA1, accB0, accB1;
    armCurrentCPU(&accA0, &accA1);
    armResidentFused(&accB0, &accB1);
    size_t bad = 0;
    for (size_t i = 0; i < T; ++i) { if (accA0[i] != accB0[i]) ++bad; if (accA1[i] != accB1[i]) ++bad; }
    std::printf("  %zu / %zu limbs differ -> %s\n", bad, 2 * T, bad == 0 ? "[PASS bit-exact]" : "[FAIL]");
    if (bad) { std::printf("[FAIL] refusing to report timings\n"); return 1; }

    // ---- warm-up, then alternating-order timed trials ----
    for (int w = 0; w < 2; ++w) { armCurrentCPU(); armResidentFused(); }
    std::vector<double> vA, vB;
    for (uint32_t t = 0; t < TRIALS; ++t) {
        if (t % 2 == 0) { vA.push_back(armCurrentCPU()); vB.push_back(armResidentFused()); }
        else            { vB.push_back(armResidentFused()); vA.push_back(armCurrentCPU()); }
    }
    const double mA = median(vA), mB = median(vB);
    std::printf("\n[timing] median of %u trials, alternating order, %u diagonals/stage:\n", TRIALS, NDIAG);
    std::printf("  current (CPU multiply+add, per-diagonal nothing extra) : %.6f [%.6f..%.6f] s\n",
                mA, *std::min_element(vA.begin(), vA.end()), *std::max_element(vA.begin(), vA.end()));
    std::printf("  resident (fused kernel, ONE final download)            : %.6f [%.6f..%.6f] s\n",
                mB, *std::min_element(vB.begin(), vB.end()), *std::max_element(vB.begin(), vB.end()));
    std::printf("  END-TO-END STAGE speedup (includes pt_to_eval_host's unchanged round trip,\n");
    std::printf("  the CPU multiply/add vs fused-kernel difference, AND the download-count\n");
    std::printf("  difference -- a REAL stage-level figure, not an isolated-kernel one): %.4fx\n", mA / mB);

    size_t vramFree = 0, vramTotal = 0;
    cudaMemGetInfo(&vramFree, &vramTotal);
    std::printf("\n[resources] peak host RSS %.1f MB; VRAM free %.0f/%.0f MiB\n",
                rssMB(), vramFree / 1048576.0, vramTotal / 1048576.0);

    std::printf("\n[SCOPE] pt_to_eval_host's own H2D/NTT/D2H round trip is IDENTICAL and UNCHANGED\n");
    std::printf("in both arms -- this experiment isolates ONLY the accumulate step's cost\n");
    std::printf("(CPU multiply+add+per-diagonal-download vs fused-kernel+one-final-download).\n");
    std::printf("Rotation itself is SIMULATED (identical fixed data in both arms), not computed --\n");
    std::printf("see this file's header for why that cannot bias the A vs B comparison. This is\n");
    std::printf("still NOT a full bootstrap figure: one stage among several, and does not include\n");
    std::printf("whatever rotation/keyswitch itself costs in a real run.\n");

    return 0;
}
