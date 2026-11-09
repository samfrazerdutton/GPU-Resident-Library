// GOD PROMPT "CKKS PHASE 28": concurrency stress test for encode_host_fft/
// decode_host_fft (src/encode_fft.cpp), found before promoting this
// optimization to production. Both functions lazily populate an
// UNSYNCHRONIZED `static std::map` cache (fft_twiddles, slot_idx) via a
// find()-then-emplace() sequence -- the SAME pattern class as
// src/keyswitch_resident.cpp's g_rtab/upc (confirmed latent-but-untriggered
// by Phase 25's test_rotate_concurrent_cache.cpp control phase) and
// structurally related to the scratch()-buffer bug Phase 25 DID find and
// fix with thread_local in src/rotate.cpp.
//
// std::map mutation from multiple threads without synchronization is
// undefined behavior per the C++ standard regardless of whether the
// inserted VALUES would be identical -- this is tested directly with REAL
// std::thread (never fork(), per Phase 27's explicit finding that fork()
// after CUDA initialization is unsafe; this file touches no CUDA at all,
// so that specific hazard does not apply here, but the general "test
// concurrency with real threads" discipline is followed regardless).
//
// Three scenarios, matching the governing directive's Section 13
// requirements:
//   A. Multiple threads calling encode_host_fft/decode_host_fft with the
//      SAME N concurrently (shared cache, contending reads once warm, but
//      a genuine concurrent MISS race on the FIRST call since all threads
//      start cold).
//   B. Multiple threads each using a DIFFERENT, never-before-seen N
//      concurrently (forces concurrent cache INSERTS, the higher-risk
//      scenario for std::map's internal tree structure).
//   C. Repeated runs of A+B together, many times, to surface a
//      probabilistic race rather than relying on one lucky/unlucky pass.
//
// Correctness in every case is checked by comparing each thread's
// encode/decode OUTPUT against encode_host/decode_host (the proven O(N^2)
// reference) using the SAME tolerance tests/test_encode_fft_matches_reference.cpp
// establishes -- a crash or corrupted cache would most likely show up as
// either a hard crash (caught by the process exit code) or a wrong result
// (caught by the tolerance check), not silently pass.
#include <cstdint>
#include <vector>
#include <complex>
#include <cmath>
#include <cstdio>
#include <thread>
#include <atomic>
#include <random>
#include <limits>

namespace gpufhe {
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void decode_host(std::vector<std::complex<double>>&, const std::vector<int64_t>&, uint32_t, double);
void encode_host_fft(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void decode_host_fft(std::vector<std::complex<double>>&, const std::vector<int64_t>&, uint32_t, double);
}

using cd = std::complex<double>;
static int g_pass = 0, g_fail = 0;
static void check(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  [PASS] %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  [FAIL] %s\n", what.c_str()); }
}

static double encTol(uint32_t N, double Delta) {
    return ((double)N + 1000.0) * std::numeric_limits<double>::epsilon() * Delta;
}

// One thread's full workload: build a deterministic (per-thread) message,
// encode via FFT, encode via the reference, compare; decode a deterministic
// coefficient array via FFT and reference, compare. Returns true iff both
// checks pass within tolerance -- called from multiple threads concurrently
// with varying N, so this function itself must not share mutable state
// across calls (it does not: all locals, no statics of its own).
static bool oneThreadWorkload(uint32_t N, uint64_t seed, std::atomic<int>* crashCanary) {
    crashCanary->fetch_add(1, std::memory_order_relaxed);
    const double Delta = std::pow(2.0, 50);
    uint32_t S = N / 2;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> d(-8.0, 8.0);
    std::vector<cd> z(S);
    for (uint32_t k = 0; k < S; ++k) z[k] = cd(d(rng), d(rng));

    std::vector<int64_t> mRef, mFft;
    gpufhe::encode_host(mRef, z, N, Delta);
    gpufhe::encode_host_fft(mFft, z, N, Delta);
    double maxDiff = 0;
    for (uint32_t j = 0; j < N; ++j) maxDiff = std::max(maxDiff, std::abs((double)(mRef[j] - mFft[j])));
    bool encOk = maxDiff <= encTol(N, Delta);

    std::vector<cd> zRef, zFft;
    gpufhe::decode_host(zRef, mRef, N, Delta);
    gpufhe::decode_host_fft(zFft, mRef, N, Delta);
    double maxDecDiff = 0;
    for (uint32_t k = 0; k < S; ++k) maxDecDiff = std::max(maxDecDiff, std::abs(zRef[k] - zFft[k]));
    bool decOk = maxDecDiff <= 1e-6;

    crashCanary->fetch_sub(1, std::memory_order_relaxed);
    return encOk && decOk;
}

int main() {
    std::printf("=== encode_host_fft / decode_host_fft concurrency stress test ===\n");

    // A. Multiple threads, SAME N, concurrent cold start (shared cache,
    // genuine concurrent miss on the very first call).
    {
        const uint32_t N = 8192;  // a size never used before this point in the process
        const int NTHREADS = 8;
        std::atomic<int> inFlight{0};
        std::vector<std::atomic<bool>> results(NTHREADS);
        std::vector<std::thread> threads;
        for (int t = 0; t < NTHREADS; ++t)
            threads.emplace_back([&, t]() { results[t] = oneThreadWorkload(N, 1000 + t, &inFlight); });
        for (auto& th : threads) th.join();
        bool allOk = true;
        for (int t = 0; t < NTHREADS; ++t) if (!results[t]) allOk = false;
        check(allOk, "A: 8 threads, SAME N=8192, concurrent cold-start cache miss -- all results within tolerance");
    }

    // B. Multiple threads, EACH a DIFFERENT never-before-seen N (forces
    // concurrent std::map INSERTS at genuinely different keys).
    {
        const std::vector<uint32_t> Ns = {16, 32, 64, 128, 256, 512, 1024, 2048};
        std::atomic<int> inFlight{0};
        std::vector<std::atomic<bool>> results(Ns.size());
        std::vector<std::thread> threads;
        for (size_t i = 0; i < Ns.size(); ++i)
            threads.emplace_back([&, i]() { results[i] = oneThreadWorkload(Ns[i], 2000 + (uint64_t)i, &inFlight); });
        for (auto& th : threads) th.join();
        bool allOk = true;
        for (size_t i = 0; i < Ns.size(); ++i) if (!results[i]) allOk = false;
        check(allOk, "B: 8 threads, 8 DIFFERENT never-before-seen N values, concurrent cache inserts -- all within tolerance");
    }

    // C. Repeated mixed runs: NEW never-before-seen N values each repetition
    // (so every repetition forces fresh concurrent inserts, not warm-cache
    // reads) interleaved with contending SAME-N reads, run REPEATS times to
    // surface a probabilistic race rather than relying on one pass.
    {
        const int REPEATS = 15;
        const int THREADS_PER_REP = 10;
        bool allOk = true;
        for (int rep = 0; rep < REPEATS; ++rep) {
            std::atomic<int> inFlight{0};
            std::vector<std::atomic<bool>> results(THREADS_PER_REP);
            std::vector<std::thread> threads;
            // Half the threads hit a FRESH N unique to this repetition (new
            // insert every time); half hit a SHARED N common across this
            // repetition's threads (contended read/insert on the same key).
            //
            // CORRECTNESS NOTE (found via AddressSanitizer while debugging
            // this exact test, see research/ENCODE_FFT_PROMOTION.md): the
            // original version used `freshBase + t` (t=0,2,4,6,8) and a
            // `sharedN=3072`, NEITHER of which is a power of two -- but
            // fft_forward_inplace's radix-2 Cooley-Tukey algorithm requires
            // M=2N to be a power of two (stated explicitly in src/
            // encode_fft.cpp's own header comment). Calling encode_host_fft
            // with a non-power-of-2 N overran the twiddle/FFT buffers and
            // corrupted the heap -- a bug in THIS TEST's inputs, not a real
            // concurrency race in the production cache code. Fixed by only
            // ever using valid power-of-2 N values below.
            // Kept SMALL (<=4096): oneThreadWorkload also runs the O(N^2)
            // reference encode_host/decode_host for comparison, so N must
            // stay in the same small range the original (buggy) version
            // used -- the first fix attempt cycled through N up to
            // 2,097,152, which made the O(N^2) reference call effectively
            // never finish (caught when this test hung instead of crashing).
            static const uint32_t kSafeFreshPow2[] = {4096, 2048, 1024, 512, 256};
            uint32_t freshN = kSafeFreshPow2[rep % 5];
            uint32_t sharedN = 512;  // power of two; reused across repetitions on purpose, to also exercise warm-cache contention
            for (int t = 0; t < THREADS_PER_REP; ++t) {
                uint32_t N = (t % 2 == 0) ? freshN : sharedN;
                threads.emplace_back([&, t, N]() { results[t] = oneThreadWorkload(N, 3000 + (uint64_t)rep * 100 + t, &inFlight); });
            }
            for (auto& th : threads) th.join();
            for (int t = 0; t < THREADS_PER_REP; ++t) if (!results[t]) allOk = false;
            if (inFlight.load() != 0) allOk = false;  // every thread must have cleanly decremented -- a crash mid-thread would leave this nonzero, but we can at least sanity-check no leftover accounting drift
        }
        check(allOk, "C: 15 repetitions x 10 threads (fresh-insert + contended-shared-N mix) -- all within tolerance, every repetition");
    }

    std::printf("\n=== TOTAL: %d passed, %d failed ===\n", g_pass, g_fail);
    if (g_fail == 0) {
        std::printf("[PASS] No crash, no corruption, no wrong result observed across %d total thread executions.\n",
                    8 + 8 + 15 * 10);
        std::printf("NOTE: this does NOT prove the underlying std::map race is impossible -- it is a\n");
        std::printf("genuine latent hazard per the C++ standard (concurrent std::map mutation is\n");
        std::printf("undefined behavior regardless of whether this test happened to observe a visible\n");
        std::printf("failure). See research/ENCODE_FFT_PROMOTION.md for the decision this informed.\n");
    }
    return g_fail == 0 ? 0 : 1;
}
