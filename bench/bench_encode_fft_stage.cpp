// GOD PROMPT "CKKS PHASE 28", Objective B (extended): isolated stage-level
// A/B for swapping encode_host (src/encode.cpp, parallelized O(N^2)) for
// encode_host_fft (src/encode_fft.cpp, single-threaded O(N log N)) at
// tests/test_boot.cpp's own two per-diagonal hot-loop call sites (lines
// ~568 and ~889). This is a SEPARATE experiment from the fused
// pt-mul-accumulate kernel (bench/bench_stage_acc_integration.cpp,
// bench/bench_stage_timeline.cpp) -- per the governing directive's explicit
// "do not stack several unverified optimizations" instruction, this changes
// ONLY the encode call and nothing else: pt_to_eval_host, ct_mul_pt_host,
// and ct_add_ct_host are IDENTICAL, unchanged, in both arms.
//
// WHY THIS MATTERS: bench/bench_stage_timeline.cpp's own cost breakdown
// (this round) found encode_host consumes 59-71% of a 16-diagonal stage's
// total time -- more than TWICE pt_to_eval_host's 18-19% share, and far
// more than the fused kernel's own 5.6% execution time. encode_host_fft
// already exists, is already independently derived and tested against
// numpy's FFT (research/FFT_ENCODING_DERIVATION.md,
// tests/test_encode_fft_matches_reference.cpp, 13 cases x 6 ring
// dimensions, [OVERALL PASS] reconfirmed this round) and already measured
// (bench/bench_encode_fft_vs_parallel.cpp) at 45-69x faster for encode and
// 38-71x faster for decode depending on N -- but, exactly like the fused
// kernel before this round's stage-level test, THAT number is an ISOLATED
// microbenchmark result, not an end-to-end one. This file is the stage-level
// check this specific swap has never received.
//
// ARM A (current production call, UNCHANGED): per diagonal --
//   encode_host (parallelized O(N^2)) -> pt_to_eval_host (unchanged) ->
//   ct_mul_pt_host (unchanged) -> ct_add_ct_host (unchanged).
// ARM B (the ONLY change): per diagonal --
//   encode_host_fft (single-threaded O(N log N)) -> pt_to_eval_host
//   (IDENTICAL call, same code) -> ct_mul_pt_host (IDENTICAL) ->
//   ct_add_ct_host (IDENTICAL).
//
// Both arms are pure CPU plus pt_to_eval_host's own internal GPU round
// trip -- there is no device-resident accumulator here (that is the
// SEPARATE fused-kernel experiment); this isolates the encode swap alone.
#include <cstdint>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <random>
#include <numeric>
#include <limits>
#include <sys/resource.h>
#include <cuda_runtime.h>

namespace gpufhe {
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void encode_host_fft(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
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
struct Stats { double mean, median, stddev, min, max; size_t n; };
static Stats computeStats(std::vector<double> v) {
    Stats s; s.n = v.size();
    std::sort(v.begin(), v.end());
    s.min = v.front(); s.max = v.back();
    s.median = (v.size() % 2) ? v[v.size()/2] : 0.5*(v[v.size()/2-1]+v[v.size()/2]);
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double var = 0; for (double x : v) var += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(var / v.size());
    return s;
}
static void printStats(const char* label, const Stats& s) {
    std::printf("  %-50s mean=%.6f median=%.6f stddev=%.6f min=%.6f max=%.6f (n=%zu)\n",
                label, s.mean, s.median, s.stddev, s.min, s.max, s.n);
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
    const size_t T = (size_t)tw * n;
    const double Delta = std::pow(2.0, 50);

    std::printf("=== Phase 28: ISOLATED encode_host -> encode_host_fft stage-level swap ===\n");
    std::printf("params: n=%u tw=%u NDIAG=%u trials=%u\n", n, tw, NDIAG, TRIALS);
    std::printf("ONLY the encode call differs between arms. pt_to_eval_host, ct_mul_pt_host,\n");
    std::printf("ct_add_ct_host are IDENTICAL in both arms -- this is a SEPARATE experiment\n");
    std::printf("from the fused pt-mul-accumulate kernel (bench_stage_acc_integration.cpp).\n");

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

    auto runStage = [&](bool useFFT, double* outEncode, std::vector<uint64_t>* outAcc0, std::vector<uint64_t>* outAcc1,
                        std::vector<std::vector<int64_t>>* outCoeffs = nullptr) {
        double tEnc = 0;
        double t0 = nowS();
        std::vector<uint64_t> acc0(T, 0), acc1(T, 0);
        bool first = true;
        for (uint32_t x = 0; x < NDIAG; ++x) {
            double s1 = nowS();
            std::vector<int64_t> md;
            if (useFFT) gpufhe::encode_host_fft(md, dg[x], n, Delta);
            else        gpufhe::encode_host(md, dg[x], n, Delta);
            tEnc += nowS() - s1;
            if (outCoeffs) outCoeffs->push_back(md);  // captured BEFORE the NTT below

            std::vector<uint64_t> dE; gpufhe::pt_to_eval_host(dE, md, tw, n, mod, root);  // IDENTICAL in both arms
            std::vector<uint64_t> a0 = simT0[x], a1 = simT1[x];
            gpufhe::ct_mul_pt_host(a0, a1, dE, tw, n, mod);                                // IDENTICAL in both arms
            if (first) { acc0 = a0; acc1 = a1; first = false; }
            else gpufhe::ct_add_ct_host(acc0, acc1, a0, a1, tw, n, mod);                   // IDENTICAL in both arms
        }
        if (outEncode) *outEncode = tEnc;
        if (outAcc0) *outAcc0 = acc0;
        if (outAcc1) *outAcc1 = acc1;
        return nowS() - t0;
    };

    // ---- correctness FIRST, done PROPERLY: an earlier version of this
    // benchmark compared the stage's POST-NTT eval-domain accumulator
    // bit-for-bit and found "491520/491520 limbs differ" with a max
    // difference around 10^18 (close to q itself) -- which LOOKED alarming
    // but was a METHODOLOGY MISTAKE, not a real problem, caught and fixed
    // while writing this file: NTT is a LINEAR MIXING transform where
    // every output position is a sum over ALL n input positions. A tiny
    // per-coefficient encode difference (tests/test_encode_fft_matches_reference.cpp
    // already proves this stays within a derived bound, maxAbs=53 at
    // n=8192 in its own worst case) gets multiplied by up to n=8192 terms
    // and reduced mod q, so it can land ANYWHERE in [0,q) after wraparound
    // -- a bit-for-bit comparison AFTER the NTT is therefore the WRONG
    // oracle for "did the encode swap stay within tolerance": two
    // genuinely-close encodings can differ by nearly the full modulus in
    // eval-domain representation, by design, with zero bearing on whether
    // the DECODED message would differ meaningfully (CKKS is approximate
    // precisely so this class of small encode-domain noise is tolerated).
    //
    // The CORRECT, apples-to-apples oracle is the one
    // test_encode_fft_matches_reference.cpp already uses: compare the
    // COEFFICIENT-DOMAIN output of encode_host vs encode_host_fft, BEFORE
    // any NTT, against that same test's own derived tolerance
    // ((N+1000)*eps*Delta). This benchmark captures exactly that via
    // runStage's outCoeffs parameter. ----
    std::printf("\n[correctness] coefficient-domain encode_host vs encode_host_fft (the correct oracle --\n");
    std::printf("see this block's own comment for why comparing POST-NTT output would be the wrong one)\n");
    std::vector<std::vector<int64_t>> coeffsA, coeffsB;
    runStage(false, nullptr, nullptr, nullptr, &coeffsA);
    runStage(true, nullptr, nullptr, nullptr, &coeffsB);
    const double encTol = ((double)n + 1000.0) * std::numeric_limits<double>::epsilon() * Delta;  // same formula as the existing correctness test
    double maxAbsDiff = 0;
    for (uint32_t x = 0; x < NDIAG; ++x) {
        for (uint32_t j = 0; j < n; ++j) {
            double d = std::abs((double)coeffsA[x][j] - (double)coeffsB[x][j]);
            if (d > maxAbsDiff) maxAbsDiff = d;
        }
    }
    bool encOk = maxAbsDiff <= encTol;
    std::printf("  max coefficient-domain |diff| across %u diagonals: %.3f  (tolerance: %.3f)  -> %s\n",
                NDIAG, maxAbsDiff, encTol, encOk ? "[PASS, within established tolerance]" : "[FAIL]");
    if (!encOk) { std::printf("[FAIL] refusing to report timings for an encode swap that is not within tolerance\n"); return 1; }

    // ---- warm-up ----
    for (int w = 0; w < 2; ++w) { runStage(false, nullptr, nullptr, nullptr); runStage(true, nullptr, nullptr, nullptr); }

    // ---- full-statistics interleaved A/B ----
    std::vector<double> totA, totB, encA, encB;
    for (uint32_t t = 0; t < TRIALS; ++t) {
        double ea, eb;
        if (t % 2 == 0) { totA.push_back(runStage(false, &ea, nullptr, nullptr)); totB.push_back(runStage(true, &eb, nullptr, nullptr)); }
        else            { totB.push_back(runStage(true, &eb, nullptr, nullptr)); totA.push_back(runStage(false, &ea, nullptr, nullptr)); }
        encA.push_back(ea); encB.push_back(eb);
    }
    Stats sA = computeStats(totA), sB = computeStats(totB);
    Stats seA = computeStats(encA), seB = computeStats(encB);
    std::printf("\n=== STAGE-LEVEL result, %u trials, interleaved order ===\n", TRIALS);
    printStats("Arm A total (current encode_host)", sA);
    printStats("Arm B total (encode_host_fft swap)", sB);
    printStats("  -- Arm A's encode share alone", seA);
    printStats("  -- Arm B's encode share alone", seB);
    std::printf("  STAGE-LEVEL speedup (mean):   %.4fx\n", sA.mean / sB.mean);
    std::printf("  STAGE-LEVEL speedup (median): %.4fx\n", sA.median / sB.median);
    std::printf("  (isolated encode-only speedup, for comparison: %.4fx mean, %.4fx median --\n",
                seA.mean / seB.mean, seA.median / seB.median);
    std::printf("   the gap between this and the stage-level number above is exactly\n");
    std::printf("   pt_to_eval_host+CPU-mul+CPU-add's now-dominant unchanged share.)\n");

    size_t vramFree = 0, vramTotal = 0;
    cudaMemGetInfo(&vramFree, &vramTotal);
    std::printf("\n[resources] peak host RSS %.1f MB; VRAM free %.0f/%.0f MiB\n",
                rssMB(), vramFree / 1048576.0, vramTotal / 1048576.0);

    return 0;
}
