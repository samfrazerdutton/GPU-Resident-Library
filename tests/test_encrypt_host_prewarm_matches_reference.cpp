// GOD PROMPT "CKKS PHASE 16": correctness test for
// encrypt_host_shoup_pooled_barrett_prewarmed (src/
// encrypt_host_shoup_pooled_barrett_prewarm.cpp) against the UNTOUCHED
// production encrypt_host (src/encdec.cpp). Covers BOTH code paths
// this new function can take:
//   (a) the FALLBACK path -- no prewarm ever requested for this
//       (n,sizeQ) -- must behave identically to today's production
//       pageable packed implementation.
//   (b) the PREWARMED path -- encrypt_prewarm_pinned_buffer() called
//       first and reports true -- must still be bit-exact.
// Same exact-equality rationale as every prior phase's test:
// switching allocation/caching strategy can never change what is
// computed.
#include "keygen.h"
#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>

namespace gpufhe {
void encrypt_host(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void encrypt_host_shoup_pooled_barrett_batched(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
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

static Fixture make_fixture(uint32_t n, uint32_t sizeQ, const std::vector<int64_t>& msg) {
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
    F.m = msg; F.m.resize(n, 0);
    return F;
}

static bool check(const Fixture& F, uint64_t seed, const char* label) {
    using namespace gpufhe;
    std::vector<uint64_t> c0_ref, c1_ref, c0_new, c1_new;
    encrypt_host(c0_ref, c1_ref, F.m, F.pkA, F.pkB, F.n, F.mod, F.root, 1, 3.19, seed);
    encrypt_host_shoup_pooled_barrett_prewarmed(c0_new, c1_new, F.m, F.pkA, F.pkB, F.n, F.mod, F.root, 1, 3.19, seed);
    bool match = (c0_ref == c0_new) && (c1_ref == c1_new);
    std::cout << "n=" << F.n << " sizeQ=" << F.sizeQ << " [" << label << "] -> "
              << (match ? "[EXACT MATCH]" : "[MISMATCH]") << "\n";
    return match;
}

int main() {
    using namespace gpufhe;
    bool allPass = true;

    // --- Part A: FALLBACK path (no prewarm requested for these keys). ---
    std::cout << "--- fallback path (never prewarmed) ---\n";
    for (uint32_t n : {16u, 1024u, 4096u, 8192u}) {
        for (uint32_t sizeQ : {1u, 4u, 8u}) {
            if (encrypt_pinned_buffer_is_ready(n, sizeQ)) {
                std::cout << "n=" << n << " sizeQ=" << sizeQ << " UNEXPECTEDLY already ready -- test setup bug\n";
                allPass = false; continue;
            }
            { std::vector<int64_t> m(n, 0);
              Fixture F = make_fixture(n, sizeQ, m);
              if (!check(F, 111, "fallback-zero")) allPass = false; }
            { std::vector<int64_t> m(n, 0); m[0] = LLONG_MAX; if (n > 3) m[3] = LLONG_MIN;
              Fixture F = make_fixture(n, sizeQ, m);
              if (!check(F, 222, "fallback-long-min-max")) allPass = false; }
        }
    }

    // --- Part B: PREWARMED path. ---
    std::cout << "--- prewarmed path ---\n";
    for (uint32_t n : {16u, 1024u, 4096u, 8192u}) {
        for (uint32_t sizeQ : {1u, 4u, 8u}) {
            // Use DIFFERENT (n,sizeQ) combos than Part A touched, if
            // any overlap exists Part A's keys are now ready too --
            // that's fine (idempotent), but to cleanly test "prewarm
            // actually changes behavior-path, not correctness" we
            // explicitly request it here for every key under test.
            bool ready = encrypt_prewarm_pinned_buffer(n, sizeQ);
            if (!ready) {
                std::cout << "n=" << n << " sizeQ=" << sizeQ << " prewarm FAILED (allocation failure) -- skipping, not a correctness bug\n";
                continue;
            }
            if (!encrypt_pinned_buffer_is_ready(n, sizeQ)) {
                std::cout << "n=" << n << " sizeQ=" << sizeQ << " prewarm reported true but is_ready() reports false -- BUG\n";
                allPass = false; continue;
            }
            { std::vector<int64_t> m(n, 0);
              Fixture F = make_fixture(n, sizeQ, m);
              if (!check(F, 333, "prewarmed-zero")) allPass = false; }
            { std::vector<int64_t> m(n, 0); m[0] = LLONG_MAX; if (n > 3) m[3] = LLONG_MIN;
              Fixture F = make_fixture(n, sizeQ, m);
              if (!check(F, 444, "prewarmed-long-min-max")) allPass = false; }
            { std::vector<int64_t> m(n); for (uint32_t k = 0; k < n; ++k)
                m[k] = (int64_t)((k % 97) - 48) * (int64_t)((k % 3 == 0) ? 1 : ((k % 3 == 1) ? 1000000000000LL : -1000000000000000LL));
              Fixture F = make_fixture(n, sizeQ, m);
              if (!check(F, 555, "prewarmed-dense-mixed")) allPass = false; }
            // Idempotency: prewarming again must report ready (already
            // allocated), not fail or reallocate.
            if (!encrypt_prewarm_pinned_buffer(n, sizeQ)) {
                std::cout << "n=" << n << " sizeQ=" << sizeQ << " second prewarm call unexpectedly failed -- BUG\n";
                allPass = false;
            }
        }
    }

    // --- Part C: repeated calls at the SAME prewarmed key -- pool
    // reuse stress, mirroring every prior phase's pool test. ---
    {
        std::vector<int64_t> m(4096, 0); for (uint32_t k = 0; k < 4096; ++k) m[k] = (int64_t)(k % 23) - 11;
        Fixture F = make_fixture(4096, 8, m);
        encrypt_prewarm_pinned_buffer(4096, 8);
        for (uint64_t seed = 1; seed <= 10; ++seed)
            if (!check(F, seed * 777, "repeated-prewarmed")) allPass = false;
    }

    std::cout << (allPass ? "[OVERALL PASS]\n" : "[OVERALL FAIL]\n");
    return allPass ? 0 : 1;
}
