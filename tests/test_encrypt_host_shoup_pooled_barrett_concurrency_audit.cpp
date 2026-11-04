// GOD PROMPT "CKKS PHASE 14": buffer-pool concurrency audit for
// encrypt_host_shoup_pooled_barrett (src/
// encrypt_host_shoup_pooled_barrett.cpp), which shares the SAME mutex-
// serialized pooling design Phase 13 already hardened in
// encrypt_host_shoup_pooled.cpp. This file exercises every concurrency
// scenario research/EMBED_AND_POOL_HARDENING.md's audit traced through
// by hand, directly and deterministically (not relying on cargo's own
// test scheduling):
//
//  1. Many threads, SAME (n,sizeQ) key, distinct seeds (buffer-content
//     race the original Phase 13 bug exercised).
//  2. Many threads, DIFFERENT (n,sizeQ) keys each -- exercises
//     concurrent first-use pool-entry CREATION (map mutation) under
//     contention, not just reuse of an already-built entry.
//  3. A larger-scale repeated-reuse stress at one key (many more
//     iterations than Phase 13's original 10-call check) to rule out
//     any slow leak/drift across a long-running process.
//  4. Mixed: threads alternating between a small set of shared keys,
//     maximizing the chance of two threads racing on pool-entry
//     creation for the SAME new key simultaneously.
//
// Every call's result is checked against the untouched production
// encrypt_host computed single-threaded for the same seed -- any
// corruption (from a race, not an arithmetic defect, since the
// arithmetic is independently verified elsewhere) shows up as an exact
// mismatch, never a near-miss.
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
void encrypt_host_shoup_pooled_barrett(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                                        const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                                        const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
}

struct Fixture {
    uint32_t n, sizeQ;
    std::vector<uint64_t> mod, root, pkA, pkB;
    std::vector<int64_t> m;
};
static Fixture make_fixture(uint32_t n, uint32_t sizeQ) {
    using namespace gpufhe;
    Fixture F; F.n = n; F.sizeQ = sizeQ;
    native_primes(F.mod, 1, 60, n, {});
    { std::vector<uint64_t> mids; native_primes(mids, sizeQ - 1, 50, n, F.mod);
      for (auto mm_ : mids) F.mod.push_back(mm_); }
    for (uint32_t t = 0; t < sizeQ; ++t) F.root.push_back(native_root(n, F.mod[t]));
    F.pkA.resize((size_t)sizeQ * n); F.pkB.resize((size_t)sizeQ * n);
    for (uint32_t t = 0; t < sizeQ; ++t) for (uint32_t k = 0; k < n; ++k) {
        F.pkA[(size_t)t * n + k] = (7 + t * 13 + k) % F.mod[t];
        F.pkB[(size_t)t * n + k] = (F.mod[t] - 1 - (t * 5 + k) % F.mod[t]) % F.mod[t];
    }
    F.m.resize(n); for (uint32_t k = 0; k < n; ++k) F.m[k] = (int64_t)((k % 31) - 15);
    return F;
}
static void checkOne(const Fixture& F, uint64_t seed, std::atomic<int>& mismatches) {
    using namespace gpufhe;
    std::vector<uint64_t> c0_new, c1_new, c0_ref, c1_ref;
    encrypt_host_shoup_pooled_barrett(c0_new, c1_new, F.m, F.pkA, F.pkB, F.n, F.mod, F.root, 1, 3.19, seed);
    encrypt_host(c0_ref, c1_ref, F.m, F.pkA, F.pkB, F.n, F.mod, F.root, 1, 3.19, seed);
    if (!(c0_new == c0_ref && c1_new == c1_ref)) mismatches.fetch_add(1);
}

int main() {
    std::atomic<int> totalMismatches{0};
    int totalChecks = 0;

    // --- Scenario 1: many threads, SAME key, distinct seeds. ---
    {
        Fixture F = make_fixture(2048, 8);
        { std::vector<uint64_t> c0, c1; gpufhe::encrypt_host_shoup_pooled_barrett(c0, c1, F.m, F.pkA, F.pkB, F.n, F.mod, F.root, 1, 3.19, 0); } // warm the pool
        const int kThreads = 8, kItersPerThread = 8;
        std::vector<std::thread> threads;
        for (int t = 0; t < kThreads; ++t)
            threads.emplace_back([&, t]() { for (int it = 0; it < kItersPerThread; ++it) checkOne(F, 100000 + (uint64_t)t * 1000 + it, totalMismatches); });
        for (auto& th : threads) th.join();
        totalChecks += kThreads * kItersPerThread;
        std::cout << "Scenario 1 (same key, " << kThreads << "x" << kItersPerThread << " concurrent): "
                  << "cumulative mismatches=" << totalMismatches.load() << "\n";
    }

    // --- Scenario 2: many threads, DIFFERENT (n,sizeQ) keys each,
    // all hit for the FIRST TIME simultaneously (fresh pool entries,
    // never warmed up ahead of time) -- stresses concurrent map
    // mutation during pool-entry creation. ---
    {
        // n MUST be a power of 2 -- this codebase's NTT (bit-reversal
        // permutation in build_enc_tab_pooled_barrett/native_primes/
        // native_root) assumes it, same as every n used elsewhere in
        // this project (1024/2048/4096/8192). A first version of this
        // test used non-power-of-2 values here and segfaulted --
        // caught and fixed before trusting any result.
        struct Cfg { uint32_t n, sizeQ; };
        std::vector<Cfg> cfgs = { {512,1}, {256,2}, {1024,4}, {2048,1}, {4096,2}, {8192,4}, {16384,1}, {512,8} };
        std::vector<Fixture> fixtures;
        for (auto& c : cfgs) fixtures.push_back(make_fixture(c.n, c.sizeQ));
        std::vector<std::thread> threads;
        for (size_t i = 0; i < fixtures.size(); ++i)
            threads.emplace_back([&, i]() { for (int it = 0; it < 4; ++it) checkOne(fixtures[i], 200000 + (uint64_t)i * 1000 + it, totalMismatches); });
        for (auto& th : threads) th.join();
        totalChecks += (int)(fixtures.size() * 4);
        std::cout << "Scenario 2 (concurrent pool growth, " << fixtures.size() << " distinct fresh keys): "
                  << "cumulative mismatches=" << totalMismatches.load() << "\n";
    }

    // --- Scenario 3: large-scale repeated reuse at one key, single
    // thread (rules out slow drift/leak across many iterations, not a
    // concurrency check by itself but a longevity check). ---
    {
        Fixture F = make_fixture(1024, 4);
        for (uint64_t seed = 1; seed <= 200; ++seed) checkOne(F, seed * 31, totalMismatches);
        totalChecks += 200;
        std::cout << "Scenario 3 (200 sequential reuses, same key): "
                  << "cumulative mismatches=" << totalMismatches.load() << "\n";
    }

    // --- Scenario 4: many threads cycling through a SMALL shared set
    // of keys (maximizes the chance two threads race to CREATE the
    // same new pool entry at the same instant). ---
    {
        // n must be a power of 2 -- see Scenario 2's comment.
        std::vector<Fixture> fixtures = { make_fixture(512, 2), make_fixture(1024, 3) };
        const int kThreads = 10, kItersPerThread = 10;
        std::vector<std::thread> threads;
        for (int t = 0; t < kThreads; ++t)
            threads.emplace_back([&, t]() {
                for (int it = 0; it < kItersPerThread; ++it)
                    checkOne(fixtures[(t + it) % fixtures.size()], 300000 + (uint64_t)t * 1000 + it, totalMismatches);
            });
        for (auto& th : threads) th.join();
        totalChecks += kThreads * kItersPerThread;
        std::cout << "Scenario 4 (mixed shared-key race, " << kThreads << "x" << kItersPerThread << "): "
                  << "cumulative mismatches=" << totalMismatches.load() << "\n";
    }

    std::cout << "Total checks: " << totalChecks << ", total mismatches: " << totalMismatches.load() << "\n";
    bool pass = (totalMismatches.load() == 0);
    std::cout << (pass ? "[OVERALL PASS]\n" : "[OVERALL FAIL]\n");
    return pass ? 0 : 1;
}
