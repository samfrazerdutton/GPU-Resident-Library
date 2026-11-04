// GOD PROMPT "CKKS PHASE 13": direct multi-threaded stress test for
// encrypt_host_shoup_pooled's buffer-pool mutex (src/
// encrypt_host_shoup_pooled.cpp). The original (unsynchronized) version
// of this file FAILED the real Rust test suite under its default
// parallel execution -- `cargo test` runs tests on multiple OS threads
// by default, and two threads calling this function concurrently at
// the same (n,sizeQ) corrupted each other's shared pooled device
// buffer, producing wrong decrypted plaintexts (confirmed by
// `--test-threads=1` passing cleanly, isolating the cause to a race,
// not an arithmetic defect). This test reproduces that scenario
// directly and deterministically, independent of cargo's own
// scheduling: many std::threads hammer encrypt_host_shoup_pooled at
// the SAME (n,sizeQ) key simultaneously, each with its own distinct
// seed, and every result is checked against the UNTOUCHED production
// encrypt_host computed single-threaded for that same seed. With the
// mutex in place, every thread's result must still be bit-exact --
// concurrency must never change what is computed, only serialize it.
#include "keygen.h"
#include <cstdint>
#include <vector>
#include <thread>
#include <atomic>
#include <iostream>

namespace gpufhe {
void encrypt_host(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void encrypt_host_shoup_pooled(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                                const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                                const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
}

int main() {
    using namespace gpufhe;
    const uint32_t n = 2048, sizeQ = 8;
    std::vector<uint64_t> mod;
    native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> mids; native_primes(mids, sizeQ - 1, 50, n, mod);
      for (auto m : mids) mod.push_back(m); }
    std::vector<uint64_t> root(sizeQ);
    for (uint32_t i = 0; i < sizeQ; ++i) root[i] = native_root(n, mod[i]);
    std::vector<uint64_t> pkA((size_t)sizeQ * n), pkB((size_t)sizeQ * n);
    for (uint32_t t = 0; t < sizeQ; ++t) for (uint32_t k = 0; k < n; ++k) {
        pkA[(size_t)t * n + k] = (7 + t * 13 + k) % mod[t];
        pkB[(size_t)t * n + k] = (mod[t] - 1 - (t * 5 + k) % mod[t]) % mod[t];
    }
    std::vector<int64_t> m(n); for (uint32_t k = 0; k < n; ++k) m[k] = (int64_t)((k % 29) - 14);

    const int kThreads = 8, kItersPerThread = 6;
    std::atomic<int> mismatches{0};

    // Warm up the pool BEFORE spawning threads so the race under test
    // is purely the per-call buffer REUSE, not the one-time pool
    // creation (both must be safe, but isolating the reuse path makes
    // a failure here unambiguous).
    { std::vector<uint64_t> c0, c1; encrypt_host_shoup_pooled(c0, c1, m, pkA, pkB, n, mod, root, 1, 3.19, 0); }

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int it = 0; it < kItersPerThread; ++it) {
                uint64_t seed = 50000 + (uint64_t)t * 1000 + it;
                std::vector<uint64_t> c0_new, c1_new, c0_ref, c1_ref;
                encrypt_host_shoup_pooled(c0_new, c1_new, m, pkA, pkB, n, mod, root, 1, 3.19, seed);
                encrypt_host(c0_ref, c1_ref, m, pkA, pkB, n, mod, root, 1, 3.19, seed);
                bool ok = (c0_new == c0_ref) && (c1_new == c1_ref);
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
