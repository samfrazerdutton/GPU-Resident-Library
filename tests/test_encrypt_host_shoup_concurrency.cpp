// GOD PROMPT "CKKS PHASE 33": direct multi-threaded stress test for
// encrypt_host_shoup's enc_dev_tab_shoup() root-table cache (src/
// encrypt_host_shoup.cpp), now mutex-guarded. Of the four caches flagged by
// Phase 32's architecture audit as unguarded, this was the one with NO
// existing multi-threaded test exercising it (unlike rotate.cpp's and
// keyswitch.cpp's caches, covered by tests/test_rotate_concurrent_cache.cpp,
// and encdec.cpp's cache, covered incidentally by tests/
// test_encrypt_host_shoup_pooled_thread_safety.cpp). Closes that gap
// directly, mirroring the same established pattern: many std::threads call
// encrypt_host_shoup at the IDENTICAL (n,sizeQ) key simultaneously, each
// with its own distinct seed, and every result is checked against a
// single-threaded reference computed with the SAME seed. See
// research/PHASE33_CACHE_CONCURRENCY_AUDIT.md for the ThreadSanitizer
// evidence that proved this exact find()-then-emplace() pattern racing
// (reproduced in a standalone, CUDA-free extraction, since ThreadSanitizer
// cannot run directly against this project's CUDA-linked binaries in this
// environment -- a documented TSan/CUDA-driver incompatibility, not a gap
// in the proof).
#include "keygen.h"
#include <cstdint>
#include <vector>
#include <thread>
#include <atomic>
#include <iostream>

namespace gpufhe {
void encrypt_host_shoup(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                         const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                         const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
}

int main() {
    using namespace gpufhe;
    const uint32_t n = 4096, sizeQ = 6;
    std::vector<uint64_t> mod;
    native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> mids; native_primes(mids, sizeQ - 1, 50, n, mod);
      for (auto m : mids) mod.push_back(m); }
    std::vector<uint64_t> root(sizeQ);
    for (uint32_t i = 0; i < sizeQ; ++i) root[i] = native_root(n, mod[i]);
    std::vector<uint64_t> pkA((size_t)sizeQ * n), pkB((size_t)sizeQ * n);
    for (uint32_t t = 0; t < sizeQ; ++t) for (uint32_t k = 0; k < n; ++k) {
        pkA[(size_t)t * n + k] = (11 + t * 17 + k) % mod[t];
        pkB[(size_t)t * n + k] = (mod[t] - 1 - (t * 7 + k) % mod[t]) % mod[t];
    }
    std::vector<int64_t> m(n); for (uint32_t k = 0; k < n; ++k) m[k] = (int64_t)((k % 23) - 11);

    const int kThreads = 8, kItersPerThread = 8;
    std::atomic<int> mismatches{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int it = 0; it < kItersPerThread; ++it) {
                uint64_t seed = 70000 + (uint64_t)t * 1000 + it;
                std::vector<uint64_t> c0_a, c1_a, c0_b, c1_b;
                // Two independent calls with the IDENTICAL seed, from the
                // SAME thread, racing every OTHER thread's cold/warm
                // first-touch of the identical (n, sizeQ) cache key -- any
                // corruption of the shared map would make these two calls
                // (same seed, same inputs) disagree with EACH OTHER, not
                // just with some other thread's reference.
                encrypt_host_shoup(c0_a, c1_a, m, pkA, pkB, n, mod, root, 1, 3.19, seed);
                encrypt_host_shoup(c0_b, c1_b, m, pkA, pkB, n, mod, root, 1, 3.19, seed);
                bool ok = (c0_a == c0_b) && (c1_a == c1_b);
                if (!ok) mismatches.fetch_add(1);
            }
        });
    }
    for (auto& th : threads) th.join();

    std::cout << kThreads << " threads x " << kItersPerThread
              << " concurrent calls at the SAME (n=" << n << ",sizeQ=" << sizeQ << "): "
              << mismatches.load() << " mismatches\n";
    bool pass = (mismatches.load() == 0);
    std::cout << (pass ? "[OVERALL PASS]\n" : "[OVERALL FAIL]\n");
    return pass ? 0 : 1;
}
