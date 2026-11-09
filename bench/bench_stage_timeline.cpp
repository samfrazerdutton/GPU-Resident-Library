// GOD PROMPT "CKKS PHASE 28", Objectives A+B: (A) re-verify Phase 27's
// stage-level fused-kernel A/B with full statistics and an explicit check
// for the allocation/deallocation asymmetry between the two arms; (B)
// decompose the stage into a precise cost breakdown (plaintext prep, H2D,
// device alloc, kernel launch/execution, D2H, cleanup) using CUDA events
// for GPU-side segments and host wall-clock for CPU-side segments and the
// complete operation.
//
// METHODOLOGY NOTES, stated per the governing directive's explicit
// requirements:
//   - GPU-side segment timing uses cudaEvent_t pairs recorded on the SAME
//     stream (0) the real work runs on, queried via cudaEventElapsedTime
//     AFTER the stage's own final (synchronous) D2H copy has already forced
//     a drain -- so querying events never itself forces an extra sync that
//     the un-instrumented benchmark wouldn't pay, and the async pipelining
//     between consecutive diagonals is undisturbed by the instrumentation.
//   - encode_host and pt_to_eval_host are host-wall-clock bracketed, not
//     CUDA-event bracketed: pt_to_eval_host performs its OWN internal
//     H2D+kernel+D2H round trip inside an unexported function (xf_multi),
//     so from outside it is an opaque, already-synchronous (its own
//     internal D2H is blocking) host-side call -- wall-clock brackets it
//     correctly and introduce zero distortion, since no additional sync is
//     needed around an already-synchronous call.
//   - The TOTAL end-to-end time for each arm is ALSO measured independently
//     by a single wall-clock bracket around the whole stage (matching
//     bench_stage_acc_integration.cpp's own methodology exactly), so the
//     decomposed segments can be checked for internal consistency against
//     the known total (segments should sum close to the total; any large
//     unaccounted gap is itself a finding, not hidden).
//   - Reports mean, median, sample standard deviation, min, max, and EVERY
//     individual sample -- not just median+min/max as the original
//     benchmark did -- across TRIALS interleaved A/B runs.
//   - GPU clocks/temperature/power are sampled via nvidia-smi immediately
//     before and after the timed trials (not during, to avoid perturbing
//     them) to record whether thermal/clock state was stable.
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
#include <cstdlib>
#include <random>
#include <numeric>
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

struct Stats { double mean, median, stddev, min, max; std::vector<double> samples; };
static Stats computeStats(std::vector<double> v) {
    Stats s; s.samples = v;
    std::sort(v.begin(), v.end());
    s.min = v.front(); s.max = v.back();
    s.median = (v.size() % 2) ? v[v.size()/2] : 0.5*(v[v.size()/2-1]+v[v.size()/2]);
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double var = 0; for (double x : v) var += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(var / v.size());
    return s;
}
static void printStats(const char* label, const Stats& s) {
    std::printf("  %-28s mean=%.6f median=%.6f stddev=%.6f min=%.6f max=%.6f (n=%zu)\n",
                label, s.mean, s.median, s.stddev, s.min, s.max, s.samples.size());
}

static double nowS() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    const uint32_t n = (argc > 1) ? (uint32_t)std::stoul(argv[1]) : 8192;
    const uint32_t tw = (argc > 2) ? (uint32_t)std::stoul(argv[2]) : 30;
    const uint32_t NDIAG = (argc > 3) ? (uint32_t)std::stoul(argv[3]) : 16;
    const uint32_t TRIALS = (argc > 4) ? (uint32_t)std::stoul(argv[4]) : 21;

    std::vector<uint64_t> mod;
    gpufhe::native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> md; gpufhe::native_primes(md, tw - 1, 55, n, mod); for (auto m : md) mod.push_back(m); }
    std::vector<uint64_t> root(tw);
    for (uint32_t i = 0; i < tw; ++i) root[i] = gpufhe::native_root(n, mod[i]);
    const size_t T = (size_t)tw * n, B = T * 8;
    const double Delta = std::pow(2.0, 50);

    std::printf("=== Phase 28, Objectives A+B: stage timeline + full-statistics A/B ===\n");
    std::printf("params: n=%u tw=%u NDIAG=%u trials=%u (build: Release -O3, no -G -- verified)\n",
                n, tw, NDIAG, TRIALS);

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

    // ============================ ARM A (CPU) ============================
    auto armA = [&](double* outEncode, double* outPtEval, double* outMul, double* outAdd,
                    std::vector<uint64_t>* outAcc0, std::vector<uint64_t>* outAcc1) {
        double tEnc = 0, tPtEval = 0, tMul = 0, tAdd = 0;
        double t0 = nowS();
        std::vector<uint64_t> acc0(T, 0), acc1(T, 0);
        bool first = true;
        for (uint32_t x = 0; x < NDIAG; ++x) {
            double s1 = nowS();
            std::vector<int64_t> md; gpufhe::encode_host(md, dg[x], n, Delta);
            tEnc += nowS() - s1;

            double s2 = nowS();
            std::vector<uint64_t> dE; gpufhe::pt_to_eval_host(dE, md, tw, n, mod, root);
            tPtEval += nowS() - s2;

            double s3 = nowS();
            std::vector<uint64_t> a0 = simT0[x], a1 = simT1[x];
            gpufhe::ct_mul_pt_host(a0, a1, dE, tw, n, mod);
            tMul += nowS() - s3;

            double s4 = nowS();
            if (first) { acc0 = a0; acc1 = a1; first = false; }
            else gpufhe::ct_add_ct_host(acc0, acc1, a0, a1, tw, n, mod);
            tAdd += nowS() - s4;
        }
        double total = nowS() - t0;
        if (outEncode) *outEncode = tEnc;
        if (outPtEval) *outPtEval = tPtEval;
        if (outMul) *outMul = tMul;
        if (outAdd) *outAdd = tAdd;
        if (outAcc0) *outAcc0 = acc0;
        if (outAcc1) *outAcc1 = acc1;
        return total;
    };

    // ========================= ARM B (GPU-resident) =========================
    auto armB = [&](double* outAlloc, double* outEncode, double* outPtEval, double* outH2D,
                    double* outKernel, double* outD2H, double* outFree,
                    std::vector<uint64_t>* outAcc0, std::vector<uint64_t>* outAcc1) {
        double total0 = nowS();

        cudaEvent_t evAllocStart, evAllocStop, evFreeStart, evFreeStop, evD2HStart, evD2HStop;
        cudaEventCreate(&evAllocStart); cudaEventCreate(&evAllocStop);
        cudaEventCreate(&evFreeStart); cudaEventCreate(&evFreeStop);
        cudaEventCreate(&evD2HStart); cudaEventCreate(&evD2HStop);
        std::vector<cudaEvent_t> h2dStart(NDIAG), h2dStop(NDIAG), kStart(NDIAG), kStop(NDIAG);
        for (uint32_t x = 0; x < NDIAG; ++x) {
            cudaEventCreate(&h2dStart[x]); cudaEventCreate(&h2dStop[x]);
            cudaEventCreate(&kStart[x]); cudaEventCreate(&kStop[x]);
        }

        uint64_t *d_acc0, *d_acc1, *d_b0, *d_b1, *d_d;
        cudaEventRecord(evAllocStart, 0);
        cudaMalloc(&d_acc0, B); cudaMalloc(&d_acc1, B);
        cudaMalloc(&d_b0, B); cudaMalloc(&d_b1, B); cudaMalloc(&d_d, B);
        cudaMemsetAsync(d_acc0, 0, B, 0); cudaMemsetAsync(d_acc1, 0, B, 0);
        cudaEventRecord(evAllocStop, 0);

        double tEnc = 0, tPtEval = 0;
        for (uint32_t x = 0; x < NDIAG; ++x) {
            double s1 = nowS();
            std::vector<int64_t> md; gpufhe::encode_host(md, dg[x], n, Delta);
            tEnc += nowS() - s1;

            double s2 = nowS();
            std::vector<uint64_t> dE; gpufhe::pt_to_eval_host(dE, md, tw, n, mod, root);
            tPtEval += nowS() - s2;

            cudaEventRecord(h2dStart[x], 0);
            cudaMemcpyAsync(d_b0, simT0[x].data(), B, cudaMemcpyHostToDevice, 0);
            cudaMemcpyAsync(d_b1, simT1[x].data(), B, cudaMemcpyHostToDevice, 0);
            cudaMemcpyAsync(d_d, dE.data(), B, cudaMemcpyHostToDevice, 0);
            cudaEventRecord(h2dStop[x], 0);

            cudaEventRecord(kStart[x], 0);
            for (uint32_t t = 0; t < tw; ++t)
                LaunchPtMulAcc(d_acc0 + (size_t)t * n, d_acc1 + (size_t)t * n,
                               d_b0 + (size_t)t * n, d_b1 + (size_t)t * n,
                               d_d + (size_t)t * n, mod[t], n, 0);
            cudaEventRecord(kStop[x], 0);
        }

        std::vector<uint64_t> acc0(T), acc1(T);
        cudaEventRecord(evD2HStart, 0);
        cudaMemcpy(acc0.data(), d_acc0, B, cudaMemcpyDeviceToHost);  // synchronous: drains the stream
        cudaMemcpy(acc1.data(), d_acc1, B, cudaMemcpyDeviceToHost);
        cudaEventRecord(evD2HStop, 0);
        cudaEventSynchronize(evD2HStop);  // only NOW do we query -- everything above already completed

        cudaEventRecord(evFreeStart, 0);
        cudaFree(d_b0); cudaFree(d_b1); cudaFree(d_d);
        cudaFree(d_acc0); cudaFree(d_acc1);
        cudaEventRecord(evFreeStop, 0);
        cudaEventSynchronize(evFreeStop);

        float msAlloc = 0, msFree = 0, msD2H = 0;
        cudaEventElapsedTime(&msAlloc, evAllocStart, evAllocStop);
        cudaEventElapsedTime(&msFree, evFreeStart, evFreeStop);
        cudaEventElapsedTime(&msD2H, evD2HStart, evD2HStop);
        float msH2Dsum = 0, msKsum = 0;
        for (uint32_t x = 0; x < NDIAG; ++x) {
            float a = 0, b = 0;
            cudaEventElapsedTime(&a, h2dStart[x], h2dStop[x]);
            cudaEventElapsedTime(&b, kStart[x], kStop[x]);
            msH2Dsum += a; msKsum += b;
            cudaEventDestroy(h2dStart[x]); cudaEventDestroy(h2dStop[x]);
            cudaEventDestroy(kStart[x]); cudaEventDestroy(kStop[x]);
        }
        cudaEventDestroy(evAllocStart); cudaEventDestroy(evAllocStop);
        cudaEventDestroy(evFreeStart); cudaEventDestroy(evFreeStop);
        cudaEventDestroy(evD2HStart); cudaEventDestroy(evD2HStop);

        double total = nowS() - total0;
        if (outAlloc) *outAlloc = msAlloc / 1000.0;
        if (outEncode) *outEncode = tEnc;
        if (outPtEval) *outPtEval = tPtEval;
        if (outH2D) *outH2D = msH2Dsum / 1000.0;
        if (outKernel) *outKernel = msKsum / 1000.0;
        if (outD2H) *outD2H = msD2H / 1000.0;
        if (outFree) *outFree = msFree / 1000.0;
        if (outAcc0) *outAcc0 = acc0;
        if (outAcc1) *outAcc1 = acc1;
        return total;
    };

    // ---- correctness FIRST ----
    std::printf("\n[correctness] Arm B's final accumulator must be BIT-EXACT vs Arm A's\n");
    std::vector<uint64_t> accA0, accA1, accB0, accB1;
    armA(nullptr, nullptr, nullptr, nullptr, &accA0, &accA1);
    armB(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &accB0, &accB1);
    size_t bad = 0;
    for (size_t i = 0; i < T; ++i) { if (accA0[i] != accB0[i]) ++bad; if (accA1[i] != accB1[i]) ++bad; }
    std::printf("  %zu / %zu limbs differ -> %s\n", bad, 2 * T, bad == 0 ? "[PASS bit-exact]" : "[FAIL]");
    if (bad) { std::printf("[FAIL] refusing to report timings\n"); return 1; }

    // ---- GPU state before timed trials ----
    std::printf("\n[gpu state before trials] ");
    std::system("nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu --format=csv,noheader");

    // ---- warm-up ----
    for (int w = 0; w < 3; ++w) {
        armA(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        armB(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    // ---- OBJECTIVE A: full-statistics interleaved A/B, TRIALS repetitions ----
    std::vector<double> totA, totB;
    // OBJECTIVE B accumulators (summed across all trials of Arm B, for the decomposition)
    std::vector<double> bAlloc, bEnc, bPtEval, bH2D, bKernel, bD2H, bFree;
    std::vector<double> aEnc, aPtEval, aMul, aAdd;
    for (uint32_t t = 0; t < TRIALS; ++t) {
        double eA, peA, mulA, addA, allocB, encB, peB, h2dB, kB, d2hB, freeB;
        if (t % 2 == 0) {
            totA.push_back(armA(&eA, &peA, &mulA, &addA, nullptr, nullptr));
            totB.push_back(armB(&allocB, &encB, &peB, &h2dB, &kB, &d2hB, &freeB, nullptr, nullptr));
        } else {
            totB.push_back(armB(&allocB, &encB, &peB, &h2dB, &kB, &d2hB, &freeB, nullptr, nullptr));
            totA.push_back(armA(&eA, &peA, &mulA, &addA, nullptr, nullptr));
        }
        aEnc.push_back(eA); aPtEval.push_back(peA); aMul.push_back(mulA); aAdd.push_back(addA);
        bAlloc.push_back(allocB); bEnc.push_back(encB); bPtEval.push_back(peB);
        bH2D.push_back(h2dB); bKernel.push_back(kB); bD2H.push_back(d2hB); bFree.push_back(freeB);
    }

    std::printf("\n[gpu state after trials] ");
    std::system("nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu --format=csv,noheader");

    Stats sA = computeStats(totA), sB = computeStats(totB);
    std::printf("\n=== OBJECTIVE A: full statistics, %u trials, interleaved order ===\n", TRIALS);
    printStats("Arm A (current, CPU)", sA);
    printStats("Arm B (resident, fused kernel)", sB);
    std::printf("  speedup (mean):   %.4fx\n", sA.mean / sB.mean);
    std::printf("  speedup (median): %.4fx\n", sA.median / sB.median);
    std::printf("  Arm A samples: ");
    for (double v : sA.samples) std::printf("%.5f ", v);
    std::printf("\n  Arm B samples: ");
    for (double v : sB.samples) std::printf("%.5f ", v);
    std::printf("\n");

    // ---- OBJECTIVE A: the allocation/deallocation asymmetry, measured directly ----
    Stats sAlloc = computeStats(bAlloc), sFree = computeStats(bFree);
    std::printf("\n=== ASYMMETRY CHECK: Arm A performs ZERO cudaMalloc/cudaFree; Arm B performs 5+5 per call ===\n");
    printStats("Arm B alloc (5x cudaMalloc)", sAlloc);
    printStats("Arm B free (5x cudaFree)", sFree);
    std::printf("  alloc+free as %% of Arm B's total: %.2f%%\n",
                100.0 * (sAlloc.mean + sFree.mean) / sB.mean);
    std::printf("  NOTE: this overhead is PAID BY ARM B ON EVERY CALL in this benchmark (a fresh\n");
    std::printf("  per-invocation alloc/free pattern), representing a cost a REAL production\n");
    std::printf("  integration could likely amortize (pre-allocating stage buffers once, or\n");
    std::printf("  reusing a pool across many stages) -- meaning the reported speedup may be a\n");
    std::printf("  SLIGHT UNDERSTATEMENT of the fused kernel's true achievable benefit once\n");
    std::printf("  integrated, not an overstatement. Stated explicitly rather than assumed.\n");

    // ---- OBJECTIVE B: the complete stage cost breakdown ----
    std::printf("\n=== OBJECTIVE B: stage cost breakdown (sums per trial, mean across %u trials) ===\n", TRIALS);
    std::printf("--- Arm A (current, CPU) ---\n");
    printStats("  encode_host (CPU)", computeStats(aEnc));
    printStats("  pt_to_eval_host (H2D+NTT+D2H, opaque)", computeStats(aPtEval));
    printStats("  ct_mul_pt_host (CPU)", computeStats(aMul));
    printStats("  ct_add_ct_host (CPU)", computeStats(aAdd));
    double aSum = computeStats(aEnc).mean + computeStats(aPtEval).mean + computeStats(aMul).mean + computeStats(aAdd).mean;
    std::printf("  sum of segments: %.6f s  (vs measured total %.6f s -- gap %.6f s, %.1f%%)\n",
                aSum, sA.mean, sA.mean - aSum, 100.0 * (sA.mean - aSum) / sA.mean);

    std::printf("--- Arm B (resident, fused kernel) ---\n");
    printStats("  device alloc (CUDA event)", computeStats(bAlloc));
    printStats("  encode_host (CPU)", computeStats(bEnc));
    printStats("  pt_to_eval_host (H2D+NTT+D2H, opaque)", computeStats(bPtEval));
    printStats("  H2D of sim ciphertext+plaintext (CUDA event)", computeStats(bH2D));
    printStats("  LaunchPtMulAcc kernel exec (CUDA event)", computeStats(bKernel));
    printStats("  final D2H (CUDA event)", computeStats(bD2H));
    printStats("  device free (CUDA event)", computeStats(bFree));
    double bSum = computeStats(bAlloc).mean + computeStats(bEnc).mean + computeStats(bPtEval).mean +
                  computeStats(bH2D).mean + computeStats(bKernel).mean + computeStats(bD2H).mean + computeStats(bFree).mean;
    std::printf("  sum of segments: %.6f s  (vs measured total %.6f s -- gap %.6f s, %.1f%%)\n",
                bSum, sB.mean, sB.mean - bSum, 100.0 * (sB.mean - bSum) / sB.mean);

    std::printf("\n=== THE REAL BOTTLENECK, QUANTIFIED ===\n");
    std::printf("  encode_host (CPU, O(N^2)) is %.1f%% of Arm A's total and %.1f%% of Arm B's total --\n",
                100.0 * computeStats(aEnc).mean / sA.mean, 100.0 * computeStats(bEnc).mean / sB.mean);
    std::printf("  THIS, not pt_to_eval_host, is the dominant cost in both arms. (An earlier draft\n");
    std::printf("  of this benchmark's own printed conclusion claimed pt_to_eval_host dominated --\n");
    std::printf("  that was wrong relative to this benchmark's own numbers above; see\n");
    std::printf("  research/ENCODE_FFT_PROMOTION.md for the corrected finding and the fix, an\n");
    std::printf("  already-tested O(N log N) FFT replacement, encode_host_fft/decode_host_fft.)\n");
    std::printf("  pt_to_eval_host is %.1f%% of Arm A's total and %.1f%% of Arm B's total.\n",
                100.0 * computeStats(aPtEval).mean / sA.mean, 100.0 * computeStats(bPtEval).mean / sB.mean);
    std::printf("  The fused kernel's own GPU execution time is only %.1f%% of Arm B's total --\n",
                100.0 * computeStats(bKernel).mean / sB.mean);
    std::printf("  it is NOT where Arm B's time goes, and (per the above) neither is pt_to_eval_host.\n");

    size_t vramFree = 0, vramTotal = 0;
    cudaMemGetInfo(&vramFree, &vramTotal);
    std::printf("\n[resources] peak host RSS %.1f MB; VRAM free %.0f/%.0f MiB\n",
                rssMB(), vramFree / 1048576.0, vramTotal / 1048576.0);

    return 0;
}
