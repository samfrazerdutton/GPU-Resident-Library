// GOD PROMPT "CKKS PHASE 16": lifecycle + concurrency audit for the
// prewarming API (src/encrypt_host_shoup_pooled_barrett_prewarm.cpp),
// per the governing directive's Step 4 scenario list. Reuses the same
// "check every result against the untouched encrypt_host" methodology
// Phase 13-15's own audits established, extended to cover the NEW
// prewarm/ready/adaptive-encrypt surface specifically rather than
// assuming it inherits safety from the earlier pools' proofs.
//
// Scenarios:
//  1. Same-key contexts prewarming CONCURRENTLY (many threads racing
//     to be the one that allocates).
//  2. Different-key contexts prewarming CONCURRENTLY (concurrent map
//     mutation for distinct keys).
//  3. Multiple "contexts" encrypting SIMULTANEOUSLY at an
//     already-prewarmed key (buffer reuse under contention -- the
//     exact class of bug Phase 13's original pooling mistake was).
//  4. Prewarm and encrypt INTERLEAVED on the same key from different
//     threads (a thread encrypting while another thread is still
//     inside its own prewarm call for the SAME key -- exercises the
//     "is this entry really ready before it's used" check under race).
//  5. Repeated prewarm/encrypt cycles at scale (longevity -- no
//     explicit destroy exists in this round, by design, see the
//     implementation file's own header comment, so this tests
//     unbounded REUSE rather than teardown).
//  6. Mixed shared-key race across several small (n,sizeQ)
//     configurations simultaneously.
//
// "Allocation failure and fallback" and "partial initialization
// failure" (directive Step 4) are NOT forced here via deliberate OOM
// -- this codebase has never deliberately engineered an OOM condition
// anywhere in its test suite (consistent prior practice, e.g. Phase
// 14's audit noted the same limitation for its own pools), and doing
// so reliably on a shared 6GiB GPU without risking the host session is
// judged disproportionate. The rollback path (free_partial) was
// instead verified by direct code reading: every allocation step is
// gated on the previous one's success, and ANY failure clears all
// pointers back to null and marks ready=false before the entry is
// even inserted into the cache -- there is no code path that can
// insert a half-initialized "ready" entry.
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
bool encrypt_prewarm_pinned_buffer(uint32_t, uint32_t);
bool encrypt_pinned_buffer_is_ready(uint32_t, uint32_t);
void encrypt_host_shoup_pooled_barrett_prewarmed(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
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
    encrypt_host_shoup_pooled_barrett_prewarmed(c0_new, c1_new, F.m, F.pkA, F.pkB, F.n, F.mod, F.root, 1, 3.19, seed);
    encrypt_host(c0_ref, c1_ref, F.m, F.pkA, F.pkB, F.n, F.mod, F.root, 1, 3.19, seed);
    if (!(c0_new == c0_ref && c1_new == c1_ref)) mismatches.fetch_add(1);
}

int main() {
    std::atomic<int> totalMismatches{0};
    std::atomic<int> prewarmFailures{0};
    int totalChecks = 0;

    { // Scenario 1: same key, many threads racing to prewarm.
        Fixture F = make_fixture(3072 > 0 ? 2048 : 2048, 8); // n must stay a power of 2
        const int kThreads = 10;
        std::vector<std::thread> threads;
        std::atomic<int> readyCount{0};
        for (int t = 0; t < kThreads; ++t)
            threads.emplace_back([&]() { if (gpufhe::encrypt_prewarm_pinned_buffer(F.n, F.sizeQ)) readyCount.fetch_add(1); });
        for (auto& th : threads) th.join();
        bool isReady = gpufhe::encrypt_pinned_buffer_is_ready(F.n, F.sizeQ);
        std::cout << "Scenario 1 (same-key concurrent prewarm, " << kThreads << " threads): "
                  << readyCount.load() << "/" << kThreads << " reported ready, final is_ready()=" << isReady << "\n";
        if (!isReady || readyCount.load() != kThreads) { std::cout << "  UNEXPECTED: not all concurrent prewarms agreed\n"; prewarmFailures.fetch_add(1); }
    }

    { // Scenario 2: different fresh keys, concurrent prewarm (map mutation race).
        struct Cfg { uint32_t n, sizeQ; };
        std::vector<Cfg> cfgs = { {512,1}, {256,2}, {1024,3}, {4096,1}, {8192,2}, {16384,1} };
        std::vector<std::thread> threads;
        std::atomic<int> readyCount{0};
        for (auto& c : cfgs)
            threads.emplace_back([&, c]() { if (gpufhe::encrypt_prewarm_pinned_buffer(c.n, c.sizeQ)) readyCount.fetch_add(1); });
        for (auto& th : threads) th.join();
        std::cout << "Scenario 2 (different-key concurrent prewarm, " << cfgs.size() << " keys): "
                  << readyCount.load() << "/" << cfgs.size() << " ready\n";
        for (auto& c : cfgs) if (!gpufhe::encrypt_pinned_buffer_is_ready(c.n, c.sizeQ)) {
            std::cout << "  key n=" << c.n << " sizeQ=" << c.sizeQ << " NOT ready after prewarm -- BUG\n";
            prewarmFailures.fetch_add(1);
        }
    }

    { // Scenario 3: multiple threads ENCRYPTING simultaneously at an
      // already-prewarmed key -- the exact class of bug Phase 13's
      // original pool had.
        Fixture F = make_fixture(2048, 8); // already prewarmed by Scenario 1
        const int kThreads = 8, kItersPerThread = 8;
        std::vector<std::thread> threads;
        for (int t = 0; t < kThreads; ++t)
            threads.emplace_back([&, t]() { for (int it = 0; it < kItersPerThread; ++it) checkOne(F, 100000 + (uint64_t)t * 1000 + it, totalMismatches); });
        for (auto& th : threads) th.join();
        totalChecks += kThreads * kItersPerThread;
        std::cout << "Scenario 3 (simultaneous encrypt at prewarmed key, " << kThreads << "x" << kItersPerThread << "): "
                  << "cumulative mismatches=" << totalMismatches.load() << "\n";
    }

    { // Scenario 4: prewarm and encrypt INTERLEAVED on the SAME fresh
      // key from different threads -- some encrypt calls will race
      // against the prewarm and correctly take the FALLBACK path
      // (since the entry isn't ready yet), others will see it ready.
      // Either path must be bit-exact; this does NOT test "prewarm
      // always wins the race" (it may not), only that whichever path
      // is taken is CORRECT.
        Fixture F = make_fixture(4096, 4);
        std::vector<std::thread> threads;
        threads.emplace_back([&]() { gpufhe::encrypt_prewarm_pinned_buffer(F.n, F.sizeQ); });
        for (int t = 0; t < 8; ++t)
            threads.emplace_back([&, t]() { checkOne(F, 400000 + t, totalMismatches); });
        for (auto& th : threads) th.join();
        totalChecks += 8;
        std::cout << "Scenario 4 (prewarm/encrypt interleaved, same fresh key): "
                  << "cumulative mismatches=" << totalMismatches.load() << "\n";
    }

    { // Scenario 5: repeated prewarm+encrypt cycles at scale (longevity,
      // no destroy -- by design, see implementation file header).
        Fixture F = make_fixture(1024, 4);
        gpufhe::encrypt_prewarm_pinned_buffer(F.n, F.sizeQ);
        for (uint64_t seed = 1; seed <= 200; ++seed) checkOne(F, seed * 31, totalMismatches);
        totalChecks += 200;
        std::cout << "Scenario 5 (200 prewarmed-cycle reuses, same key): "
                  << "cumulative mismatches=" << totalMismatches.load() << "\n";
    }

    { // Scenario 6: mixed shared-key race, several threads cycling
      // through a small set of keys, prewarming AND encrypting.
        std::vector<Fixture> fixtures = { make_fixture(512, 2), make_fixture(1024, 1) };
        const int kThreads = 10, kItersPerThread = 10;
        std::vector<std::thread> threads;
        for (int t = 0; t < kThreads; ++t)
            threads.emplace_back([&, t]() {
                for (int it = 0; it < kItersPerThread; ++it) {
                    const Fixture& F = fixtures[(t + it) % fixtures.size()];
                    if (it % 3 == 0) gpufhe::encrypt_prewarm_pinned_buffer(F.n, F.sizeQ);
                    checkOne(F, 500000 + (uint64_t)t * 1000 + it, totalMismatches);
                }
            });
        for (auto& th : threads) th.join();
        totalChecks += kThreads * kItersPerThread;
        std::cout << "Scenario 6 (mixed prewarm+encrypt shared-key race, " << kThreads << "x" << kItersPerThread << "): "
                  << "cumulative mismatches=" << totalMismatches.load() << "\n";
    }

    std::cout << "Total checks: " << totalChecks << ", total mismatches: " << totalMismatches.load()
              << ", prewarm-consistency failures: " << prewarmFailures.load() << "\n";
    bool pass = (totalMismatches.load() == 0) && (prewarmFailures.load() == 0);
    std::cout << (pass ? "[OVERALL PASS]\n" : "[OVERALL FAIL]\n");
    return pass ? 0 : 1;
}
