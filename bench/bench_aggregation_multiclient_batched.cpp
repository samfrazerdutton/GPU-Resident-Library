// GOD PROMPT "CKKS PHASE 38": multi-client throughput comparison --
// sequential (K independent encrypt_host_shoup_pooled_barrett_batched
// calls) vs the new genuine batched path
// (encrypt_host_shoup_pooled_barrett_batched_multiclient, ONE call for all
// K clients). Uses the REAL financial aggregation workload shape (same
// parameters, same keygen, same aggregation-via-ct_add_ct_host,
// same decrypt+decode correctness check as bench_private_aggregation_
// workload.cpp / bench_aggregation_client_scaling.cpp), not a separate
// artificial microbenchmark.
//
// Interleaves SEQ/BATCH every repetition (not block-sequential) so both
// arms experience the same GPU clock/thermal drift within one run --
// the same discipline bench_aggregation_decomposed.cpp already
// established, now extended to sweep client count K as the controlled
// variable.
#include "keygen.h"
#include <cuda_runtime.h>
#include <sys/resource.h>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <algorithm>
#include <random>
#include <vector>

namespace gpufhe {
void encode_host_fft(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void decode_host_fft(std::vector<std::complex<double>>&, const std::vector<int64_t>&, uint32_t, double);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void encrypt_host_shoup_pooled_barrett_batched(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void encrypt_host_shoup_pooled_barrett_batched_multiclient(
    std::vector<std::vector<uint64_t>>&, std::vector<std::vector<uint64_t>>&,
    const std::vector<std::vector<int64_t>>&,
    const std::vector<uint64_t>&, const std::vector<uint64_t>&,
    uint32_t, const std::vector<uint64_t>&, const std::vector<uint64_t>&,
    uint64_t, double, const std::vector<uint64_t>&);
void decrypt_host(std::vector<int64_t>&, const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                   const std::vector<uint64_t>&, uint32_t, const std::vector<uint64_t>&,
                   const std::vector<uint64_t>&);
void ct_add_ct_host(std::vector<uint64_t>&, std::vector<uint64_t>&,
                    const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                    uint32_t, uint32_t, const std::vector<uint64_t>&);
}
using cd = std::complex<double>;
using clk = std::chrono::steady_clock;
static double rssMB() { struct rusage ru; getrusage(RUSAGE_SELF, &ru); return ru.ru_maxrss / 1024.0; }

struct Stats { double median, mean, min, max, stddev; };
static Stats stats_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double mean = 0; for (double x : v) mean += x; mean /= v.size();
    double var = 0; for (double x : v) var += (x - mean) * (x - mean); var /= v.size();
    return { v[v.size() / 2], mean, v.front(), v.back(), std::sqrt(var) };
}

int main() {
    gpufhe::set_secret_hamming_weight(64);
    const uint32_t n = 8192, S = n / 2, sizeQ = 4;
    const uint64_t ns = 1;
    const double sigma = 3.19;
    const double Delta = std::pow(2.0, 45);
    const uint32_t R = 16; // repetitions per (K, arm)

    std::vector<uint64_t> mod;
    gpufhe::native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids, sizeQ - 1, 55, n, mod); for (auto m : mids) mod.push_back(m); }
    std::vector<uint64_t> root(sizeQ);
    for (uint32_t i = 0; i < sizeQ; ++i) root[i] = gpufhe::native_root(n, mod[i]);
    auto KP = gpufhe::keygen_host(n, mod, root, ns, sigma, 101);

    std::printf("=== Phase 38: multi-client throughput, sequential vs batched encryption ===\n");
    std::printf("n=%u sizeQ=%u h=64, production parameters\n\n", n, sizeQ);

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> d(-50.0, 50.0);
    std::vector<uint32_t> Ks = {1, 2, 4, 8, 16, 32, 64};

    // First-request (cold) latency: the very first call of each arm, before
    // any pool exists at all, reported separately from the warmed-up
    // steady-state reps below.
    bool firstSeqDone = false, firstBatchDone = false;
    double firstSeqMs = 0, firstBatchMs = 0;

    std::printf("%4s %10s %14s %14s %14s %14s %10s %10s %14s\n",
                "K", "arm", "median(ms)", "mean(ms)", "stddev(ms)", "thrpt(enc/s)", "VRAM_d(MiB)", "maxErr", "firstCall(ms)");

    for (uint32_t K : Ks) {
        std::vector<double> seqTimes, batchTimes;
        double seqMaxErr = 0, batchMaxErr = 0;
        size_t vFreeBeforeSeq = 0, vFreeAfterSeq = 0, vFreeBeforeBatch = 0, vFreeAfterBatch = 0, vTotal = 0;

        for (uint32_t rep = 0; rep < R; ++rep) {
            std::vector<double> vals(K);
            for (uint32_t k = 0; k < K; ++k) vals[k] = d(rng);
            double plainSum = 0; for (double v : vals) plainSum += v;

            // --- SEQ arm (existing path: K independent calls) ---
            cudaMemGetInfo(&vFreeBeforeSeq, &vTotal);
            auto t0 = clk::now();
            std::vector<uint64_t> acc0, acc1; bool first = true;
            for (uint32_t k = 0; k < K; ++k) {
                std::vector<cd> z(S, cd(vals[k], 0.0));
                std::vector<int64_t> m; gpufhe::encode_host_fft(m, z, n, Delta);
                std::vector<uint64_t> c0, c1;
                gpufhe::encrypt_host_shoup_pooled_barrett_batched(c0, c1, m, KP.pkA, KP.pkB, n, mod, root, ns, sigma, 5000 + rep * 1000 + k);
                if (first) { acc0 = c0; acc1 = c1; first = false; } else gpufhe::ct_add_ct_host(acc0, acc1, c0, c1, sizeQ, n, mod);
            }
            std::vector<int64_t> mDec; gpufhe::decrypt_host(mDec, acc0, acc1, KP.s, n, mod, root);
            std::vector<cd> zOut; gpufhe::decode_host_fft(zOut, mDec, n, Delta);
            double seqMs = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            cudaMemGetInfo(&vFreeAfterSeq, &vTotal);
            seqMaxErr = std::max(seqMaxErr, std::abs(zOut[0].real() - plainSum));
            if (!firstSeqDone) { firstSeqMs = seqMs; firstSeqDone = true; } else seqTimes.push_back(seqMs);

            // --- BATCH arm (new path: ONE multi-client call) ---
            std::vector<std::vector<int64_t>> m_list(K);
            std::vector<uint64_t> seeds(K);
            for (uint32_t k = 0; k < K; ++k) {
                std::vector<cd> z(S, cd(vals[k], 0.0));
                gpufhe::encode_host_fft(m_list[k], z, n, Delta);
                seeds[k] = 6000 + rep * 1000 + k;
            }
            cudaMemGetInfo(&vFreeBeforeBatch, &vTotal);
            auto t1 = clk::now();
            std::vector<std::vector<uint64_t>> c0b, c1b;
            gpufhe::encrypt_host_shoup_pooled_barrett_batched_multiclient(c0b, c1b, m_list, KP.pkA, KP.pkB, n, mod, root, ns, sigma, seeds);
            std::vector<uint64_t> accb0 = c0b[0], accb1 = c1b[0];
            for (uint32_t k = 1; k < K; ++k) gpufhe::ct_add_ct_host(accb0, accb1, c0b[k], c1b[k], sizeQ, n, mod);
            std::vector<int64_t> mDecB; gpufhe::decrypt_host(mDecB, accb0, accb1, KP.s, n, mod, root);
            std::vector<cd> zOutB; gpufhe::decode_host_fft(zOutB, mDecB, n, Delta);
            double batchMs = std::chrono::duration<double, std::milli>(clk::now() - t1).count();
            cudaMemGetInfo(&vFreeAfterBatch, &vTotal);
            batchMaxErr = std::max(batchMaxErr, std::abs(zOutB[0].real() - plainSum));
            if (!firstBatchDone) { firstBatchMs = batchMs; firstBatchDone = true; } else batchTimes.push_back(batchMs);
        }

        Stats sSeq = stats_of(seqTimes);
        Stats sBatch = stats_of(batchTimes);
        std::printf("%4u %10s %14.3f %14.3f %14.3f %14.1f %10zd %10.2e %14.3f\n",
                    K, "SEQ", sSeq.median, sSeq.mean, sSeq.stddev, (K * 1000.0) / sSeq.median,
                    (ptrdiff_t)(vFreeBeforeSeq - vFreeAfterSeq) / 1048576, seqMaxErr, (K == Ks.front() ? firstSeqMs : -1.0));
        std::printf("%4u %10s %14.3f %14.3f %14.3f %14.1f %10zd %10.2e %14.3f\n",
                    K, "BATCH", sBatch.median, sBatch.mean, sBatch.stddev, (K * 1000.0) / sBatch.median,
                    (ptrdiff_t)(vFreeBeforeBatch - vFreeAfterBatch) / 1048576, batchMaxErr, (K == Ks.front() ? firstBatchMs : -1.0));
        std::printf("%4u %10s %14.3fx\n", K, "SPEEDUP", sSeq.median / sBatch.median);
    }

    size_t vFree, vTot;
    cudaMemGetInfo(&vFree, &vTot);
    std::printf("\npeak host RSS: %.1f MB; VRAM free at end: %zu/%zu MiB\n", rssMB(), vFree / 1048576, vTot / 1048576);
    std::printf("first-ever SEQ call (fully cold): %.3f ms; first-ever BATCH call (fully cold): %.3f ms\n", firstSeqMs, firstBatchMs);
    return 0;
}
