// GOD PROMPT "CKKS PHASE 29", Objective D: fair CPU baseline for the
// encrypted-aggregation workload.
//
// IMPORTANT LIMITATION, stated per the governing directive's explicit
// instruction to document rather than invent: this codebase (gpufhe) has
// NO pure-CPU-only encrypt/decrypt path. Every encrypt_host* variant in
// src/encrypt_host_shoup*.cpp and src/encdec.cpp calls cudaMalloc/
// cudaMemcpy/an NTT kernel internally (confirmed by grep -- all include
// cuda_runtime.h and call cudaDeviceSynchronize). encode_host_fft/
// decode_host_fft and ct_add_ct_host, by contrast, are ALREADY pure CPU
// (no CUDA at all) -- so there is no "GPU vs CPU" question to ask for
// those specific stages; the only one that exists in this workload is
// encrypt/decrypt's NTT step.
//
// Since gpufhe itself cannot supply a CPU-only encrypt, this benchmark
// uses OpenFHE -- already present in this repository as the project's own
// CPU ground-truth oracle for other tests (e.g. bench/bench_evalmult.cpp)
// -- as an INDEPENDENT, different CKKS implementation performing
// mathematically equivalent work (same n, same approximate security
// budget), run with OPENFHE_GPU=0 (Phase 25/27's own finding: this
// specific OpenFHE build is secretly GPU-shadow-accelerated by an external
// dependency unless disabled) to guarantee it is genuinely CPU-only. This
// is NOT "this library's CPU mode" -- it is a different library, with its
// own (different) secret distribution, noise sampling, and internal
// optimizations, run for comparison only. Differences are stated below,
// not hidden.
#include "openfhe.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
using namespace lbcrypto;
using clk = std::chrono::steady_clock;

struct Stats { double mean, median, stddev, mn, mx; size_t n; };
static Stats computeStats(std::vector<double> v) {
    Stats s{}; s.n = v.size(); if (v.empty()) return s;
    std::sort(v.begin(), v.end()); s.mn = v.front(); s.mx = v.back();
    double sum = 0; for (double x : v) sum += x; s.mean = sum / v.size();
    s.median = v.size() % 2 ? v[v.size()/2] : 0.5*(v[v.size()/2-1]+v[v.size()/2]);
    double sq = 0; for (double x : v) sq += (x-s.mean)*(x-s.mean);
    s.stddev = std::sqrt(sq / v.size());
    return s;
}
static void printStats(const char* label, const Stats& s) {
    std::printf("    %-30s mean=%.4f median=%.4f stddev=%.4f min=%.4f max=%.4f ms (n=%zu)\n",
                label, s.mean, s.median, s.stddev, s.mn, s.mx, s.n);
}

int main() {
    setenv("OPENFHE_GPU", "0", 1);   // force genuinely CPU-only (Phase 25/27 finding)

    const uint32_t K = 32, R = 10;
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(0);     // pure sum consumes no depth, matching the gpufhe workload
    params.SetScalingModSize(45);         // matches gpufhe's Delta=2^45
    params.SetRingDim(8192);              // matches gpufhe's n=8192
    params.SetBatchSize(4096);            // matches gpufhe's S=n/2 slots

    std::printf("=== Phase 29, Objective D: OpenFHE CPU-only baseline for the SAME aggregation workload ===\n");
    std::printf("DIFFERENT implementation from gpufhe -- OpenFHE's own secret distribution, noise\n");
    std::printf("sampling, and internal optimizations, NOT this project's h=64/sigma=3.19 parameters.\n");
    std::printf("Run only to establish a CPU reference point, since gpufhe has no CPU-only encrypt path.\n\n");

    auto tk0 = clk::now();
    auto cc = GenCryptoContext(params);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    auto kp = cc->KeyGen();
    double keygenMs = std::chrono::duration<double, std::milli>(clk::now() - tk0).count();
    std::printf("keygen: %.2f ms (once, amortized)\n\n", keygenMs);

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> d(-50.0, 50.0);
    std::vector<double> encodeEncryptMs, aggregateMs, decryptDecodeMs, e2eMs, errs;

    for (uint32_t rep = 0; rep < R; ++rep) {
        std::vector<double> vals(K);
        for (uint32_t k = 0; k < K; ++k) vals[k] = d(rng);
        double plainSum = 0; for (double v : vals) plainSum += v;

        auto tE0 = clk::now();
        double eeAccum = 0;
        Ciphertext<DCRTPoly> acc;
        for (uint32_t k = 0; k < K; ++k) {
            auto t0 = clk::now();
            std::vector<double> slots(4096, vals[k]);
            auto pt = cc->MakeCKKSPackedPlaintext(slots);
            auto ct = cc->Encrypt(kp.publicKey, pt);
            eeAccum += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            if (k == 0) acc = ct; else acc = cc->EvalAdd(acc, ct);
        }
        encodeEncryptMs.push_back(eeAccum);

        auto t1 = clk::now();
        Plaintext ptOut;
        cc->Decrypt(kp.secretKey, acc, &ptOut);
        ptOut->SetLength(4096);
        auto decoded = ptOut->GetCKKSPackedValue();
        decryptDecodeMs.push_back(std::chrono::duration<double, std::milli>(clk::now() - t1).count());

        double e2e = std::chrono::duration<double, std::milli>(clk::now() - tE0).count();
        e2eMs.push_back(e2e);
        errs.push_back(std::abs(decoded[0].real() - plainSum));
    }

    printStats("encode+encrypt (K calls)", computeStats(encodeEncryptMs));
    printStats("decrypt+decode (1 call)", computeStats(decryptDecodeMs));
    printStats("END-TO-END", computeStats(e2eMs));
    double worstErr = *std::max_element(errs.begin(), errs.end());
    std::printf("    worst abs error across %u reps: %.6e\n\n", R, worstErr);

    std::printf("=== Comparison note ===\n");
    std::printf("See bench_aggregation_decomposed.cpp's PRODUCTION-arm numbers for the GPU side.\n");
    std::printf("Do not treat this as a same-library CPU mode -- see file header for why.\n");
    return 0;
}
