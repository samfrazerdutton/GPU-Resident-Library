// GOD PROMPT "CKKS PHASE 29", Objectives C+D+E+F. Decomposed, full-statistics
// benchmark for the Phase 28 private-aggregation workload
// (bench/bench_private_aggregation_workload.cpp), separating every stage
// (keygen, per-client encode, per-client encrypt, aggregate, decrypt,
// decode) via wall-clock timing, comparing the OLD plain `encrypt_host`
// (what Phase 28's benchmark actually called) against the real PRODUCTION
// encrypt path this project's own Rust bridge uses,
// `encrypt_host_shoup_pooled_barrett_batched` (rust/crates/ckks-cuda/
// cpp_bridge/encdec_bridge.cpp confirms this is what `gpufhe_encrypt`
// calls) -- already bit-exact verified and already measured 1.57x-2.78x
// faster than plain encrypt_host in isolation (research/
// ENCRYPT_PROFILING.md, research/EMBED_AND_POOL_HARDENING.md). This
// benchmark checks whether that isolated gain survives inside the actual
// K-client aggregation application, not just a single-call microbenchmark.
//
// decrypt_host has no optimized production variant in this codebase (only
// the Phase-11 reference exists) -- used identically in both arms; this is
// a genuine, stated limitation, not an oversight.
//
// encode_host_fft/decode_host_fft and ct_add_ct_host are PURE CPU (no CUDA
// at all -- confirmed by grep of src/encode_fft.cpp and src/ptops.cpp) --
// there is no "GPU-accelerated encode" question to ask for this library;
// the ONLY GPU-vs-CPU question in this whole workload is encrypt/decrypt's
// NTT step, which both arms below exercise identically via the GPU (there
// is no pure-CPU-only NTT/encrypt path anywhere in this codebase to compare
// against -- see this file's own Objective D section for the CPU-baseline
// discussion, and bench/bench_aggregation_openfhe_cpu_baseline.cpp for the
// actual CPU comparison this limitation forces).
#include "keygen.h"
#include <cuda_runtime.h>
#include <sys/resource.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

namespace gpufhe {
void encode_host_fft(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void decode_host_fft(std::vector<std::complex<double>>&, const std::vector<int64_t>&, uint32_t, double);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void encrypt_host(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void encrypt_host_shoup_pooled_barrett_batched(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
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

struct Stats { double mean, median, stddev, mn, mx; size_t n; };
static Stats computeStats(std::vector<double> v) {
    Stats s{}; s.n = v.size();
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.mn = v.front(); s.mx = v.back();
    double sum = 0; for (double x : v) sum += x;
    s.mean = sum / v.size();
    s.median = v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    double sq = 0; for (double x : v) sq += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(sq / v.size());
    return s;
}
static void printStats(const char* label, const Stats& s) {
    std::printf("    %-38s mean=%.4f median=%.4f stddev=%.4f min=%.4f max=%.4f ms (n=%zu)\n",
                label, s.mean, s.median, s.stddev, s.mn, s.mx, s.n);
}

using EncryptFn = void(*)(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                           const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                           const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);

struct StageTimes { double encode=0, encrypt=0, aggregate=0, decrypt=0, decode=0, endToEnd=0; double err=0; };

static StageTimes runOneAggregation(uint32_t K, uint32_t n, uint32_t S, double Delta,
                                     const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root,
                                     uint64_t ns, double sigma, const gpufhe::KeyPairHost& KP,
                                     EncryptFn encryptFn, std::mt19937_64& msgRng, uint32_t repSeedBase) {
    StageTimes T;
    std::uniform_real_distribution<double> d(-50.0, 50.0);
    std::vector<double> clientValues(K);
    for (uint32_t k = 0; k < K; ++k) clientValues[k] = d(msgRng);
    double plainSum = 0; for (double v : clientValues) plainSum += v;

    auto tE0 = clk::now();
    std::vector<uint64_t> acc0, acc1;
    for (uint32_t k = 0; k < K; ++k) {
        std::vector<cd> z(S, cd(clientValues[k], 0.0));
        auto t0 = clk::now();
        std::vector<int64_t> m;
        gpufhe::encode_host_fft(m, z, n, Delta);
        T.encode += std::chrono::duration<double, std::milli>(clk::now() - t0).count();

        auto t1 = clk::now();
        std::vector<uint64_t> c0, c1;
        encryptFn(c0, c1, m, KP.pkA, KP.pkB, n, mod, root, ns, sigma, repSeedBase + k);
        T.encrypt += std::chrono::duration<double, std::milli>(clk::now() - t1).count();

        auto t2 = clk::now();
        if (k == 0) { acc0 = c0; acc1 = c1; }
        else gpufhe::ct_add_ct_host(acc0, acc1, c0, c1, (uint32_t)mod.size(), n, mod);
        T.aggregate += std::chrono::duration<double, std::milli>(clk::now() - t2).count();
    }
    auto t3 = clk::now();
    std::vector<int64_t> mDec;
    gpufhe::decrypt_host(mDec, acc0, acc1, KP.s, n, mod, root);
    T.decrypt = std::chrono::duration<double, std::milli>(clk::now() - t3).count();

    auto t4 = clk::now();
    std::vector<cd> zOut;
    gpufhe::decode_host_fft(zOut, mDec, n, Delta);
    T.decode = std::chrono::duration<double, std::milli>(clk::now() - t4).count();

    T.endToEnd = std::chrono::duration<double, std::milli>(clk::now() - tE0).count();
    T.err = std::abs(zOut[0].real() - plainSum);
    return T;
}

int main() {
    const uint32_t HW = 64;
    gpufhe::set_secret_hamming_weight(HW);
    const uint32_t n = 8192, S = n / 2, sizeQ = 4;
    const uint64_t ns = 1;
    const double sigma = 3.19;
    const double Delta = std::pow(2.0, 45);
    const uint32_t K = 32;
    const uint32_t R = 10;

    std::vector<uint64_t> mod;
    gpufhe::native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids, sizeQ - 1, 55, n, mod); for (auto m : mids) mod.push_back(m); }
    std::vector<uint64_t> root(sizeQ);
    for (uint32_t i = 0; i < sizeQ; ++i) root[i] = gpufhe::native_root(n, mod[i]);

    auto tk0 = clk::now();
    auto KP = gpufhe::keygen_host(n, mod, root, ns, sigma, 101);
    double keygenMs = std::chrono::duration<double, std::milli>(clk::now() - tk0).count();

    std::printf("=== Phase 29: decomposed aggregation benchmark ===\n");
    std::printf("hardware: RTX 2060 Max-Q, 6144 MiB, cc7.5, driver 595.97, CUDA 13.2, Release -O3\n");
    std::printf("params: n=%u S=%u sizeQ=%u h=%u sigma=%.2f Delta=2^45 K=%u clients R=%u reps\n\n",
                n, S, sizeQ, HW, sigma, K, R);
    std::printf("keygen (once, amortized, NOT included in per-aggregation stats below): %.2f ms\n\n", keygenMs);

    struct Arm { const char* name; EncryptFn fn; };
    Arm arms[2] = {
        {"OLD plain encrypt_host (Phase 28's actual call)", gpufhe::encrypt_host},
        {"PRODUCTION encrypt_host_shoup_pooled_barrett_batched (real Rust-bridge path)",
         gpufhe::encrypt_host_shoup_pooled_barrett_batched},
    };

    // Interleaved A/B across repetitions, per the governing directive, to
    // minimize thermal/clock bias between the two arms.
    std::vector<double> encode[2], encrypt[2], aggregate[2], decrypt[2], decode[2], e2e[2], errs[2];
    std::mt19937_64 msgRng(42);
    size_t vramBeforeFree, vramBeforeTotal, vramAfterFree, vramAfterTotal;
    cudaMemGetInfo(&vramBeforeFree, &vramBeforeTotal);

    for (uint32_t rep = 0; rep < R; ++rep) {
        for (int a = 0; a < 2; ++a) {
            StageTimes T = runOneAggregation(K, n, S, Delta, mod, root, ns, sigma, KP, arms[a].fn,
                                              msgRng, 5000 + rep * 100);
            encode[a].push_back(T.encode); encrypt[a].push_back(T.encrypt);
            aggregate[a].push_back(T.aggregate); decrypt[a].push_back(T.decrypt);
            decode[a].push_back(T.decode); e2e[a].push_back(T.endToEnd); errs[a].push_back(T.err);
        }
    }
    cudaMemGetInfo(&vramAfterFree, &vramAfterTotal);

    for (int a = 0; a < 2; ++a) {
        std::printf("--- Arm %c: %s ---\n", 'A' + a, arms[a].name);
        printStats("encode (K calls, CPU, FFT)", computeStats(encode[a]));
        printStats("encrypt (K calls, GPU NTT)", computeStats(encrypt[a]));
        printStats("aggregate (K-1 calls, CPU add)", computeStats(aggregate[a]));
        printStats("decrypt (1 call, GPU NTT)", computeStats(decrypt[a]));
        printStats("decode (1 call, CPU FFT)", computeStats(decode[a]));
        printStats("END-TO-END (full client-to-result)", computeStats(e2e[a]));
        double worstErr = *std::max_element(errs[a].begin(), errs[a].end());
        std::printf("    worst abs error across %u reps: %.6e\n\n", R, worstErr);
    }

    Stats e2eA = computeStats(e2e[0]), e2eB = computeStats(e2e[1]);
    Stats encA = computeStats(encrypt[0]), encB = computeStats(encrypt[1]);
    std::printf("=== Comparison ===\n");
    std::printf("  END-TO-END speedup (median): %.3fx  (OLD %.3f ms -> PRODUCTION %.3f ms)\n",
                e2eA.median / e2eB.median, e2eA.median, e2eB.median);
    std::printf("  encrypt-stage-only speedup (median): %.3fx  (OLD %.3f ms -> PRODUCTION %.3f ms)\n",
                encA.median / encB.median, encA.median, encB.median);
    std::printf("  encrypt stage as %% of end-to-end: OLD=%.1f%%  PRODUCTION=%.1f%%\n",
                100.0 * encA.median / e2eA.median, 100.0 * encB.median / e2eB.median);

    std::printf("\n=== Resources ===\n");
    std::printf("  peak host RSS: %.1f MB\n", rssMB());
    std::printf("  VRAM free before: %zu MiB, after: %zu MiB (delta: %zd MiB), total: %zu MiB\n",
                vramBeforeFree / 1048576, vramAfterFree / 1048576,
                (ptrdiff_t)(vramBeforeFree - vramAfterFree) / 1048576, vramBeforeTotal / 1048576);
    return 0;
}
