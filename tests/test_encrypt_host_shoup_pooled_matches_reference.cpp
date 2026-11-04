// GOD PROMPT "CKKS PHASE 13": correctness test for
// encrypt_host_shoup_pooled (src/encrypt_host_shoup_pooled.cpp) against
// the UNTOUCHED production encrypt_host (src/encdec.cpp). Same exact-
// equality rationale as test_encrypt_host_shoup_matches_reference.cpp
// (identical RNG/embedding/NTT, only the combine arithmetic AND the
// device-buffer allocation lifecycle differ -- pooling never changes
// what is computed, only when cudaMalloc/cudaFree happen).
//
// This test ALSO specifically exercises REPEATED calls, including
// INTERLEAVED calls across DIFFERENT (n,sizeQ) configurations, to
// catch any pooling bug where a buffer cached for one size is
// incorrectly reused for another, or where stale content from a
// previous call leaks into a new one.
#include "keygen.h"
#include <cstdint>
#include <vector>
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

struct Fixture {
    uint32_t n, sizeQ;
    std::vector<uint64_t> mod, root, pkA, pkB;
    std::vector<int64_t> m;
};

static Fixture make_fixture(uint32_t n, uint32_t sizeQ, int64_t tag) {
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
    F.m.resize(n); for (uint32_t k = 0; k < n; ++k) F.m[k] = (int64_t)((k % 23) - 11) + tag;
    return F;
}

static bool check(const Fixture& F, uint64_t seed, const char* label) {
    using namespace gpufhe;
    std::vector<uint64_t> c0_ref, c1_ref, c0_new, c1_new;
    encrypt_host(c0_ref, c1_ref, F.m, F.pkA, F.pkB, F.n, F.mod, F.root, 1, 3.19, seed);
    encrypt_host_shoup_pooled(c0_new, c1_new, F.m, F.pkA, F.pkB, F.n, F.mod, F.root, 1, 3.19, seed);
    bool match = (c0_ref == c0_new) && (c1_ref == c1_new);
    std::cout << "n=" << F.n << " sizeQ=" << F.sizeQ << " [" << label << "] -> "
              << (match ? "[EXACT MATCH]" : "[MISMATCH]") << "\n";
    return match;
}

int main() {
    bool allPass = true;

    // Basic sweep, same shape as the non-pooled Shoup test.
    for (uint32_t n : {16u, 1024u, 4096u, 8192u}) {
        for (uint32_t sizeQ : {1u, 4u, 8u}) {
            Fixture F = make_fixture(n, sizeQ, 0);
            if (!check(F, 111, "basic")) allPass = false;
        }
    }

    // Repeated calls at the SAME (n,sizeQ) -- exercises the pool's
    // reuse path directly (buffer allocated once, reused many times).
    {
        Fixture F = make_fixture(4096, 8, 0);
        for (uint64_t seed = 1; seed <= 10; ++seed)
            if (!check(F, seed * 777, "repeated-same-size")) allPass = false;
    }

    // Interleaved calls across DIFFERENT (n,sizeQ) configurations --
    // exercises the pool's per-key separation (no cross-contamination
    // between differently-sized cached buffers).
    {
        Fixture F1 = make_fixture(1024, 4, 1);
        Fixture F2 = make_fixture(8192, 8, 2);
        Fixture F3 = make_fixture(2048, 1, 3);
        for (int round = 0; round < 3; ++round) {
            if (!check(F1, 1000 + round, "interleaved-A")) allPass = false;
            if (!check(F2, 2000 + round, "interleaved-B")) allPass = false;
            if (!check(F3, 3000 + round, "interleaved-C")) allPass = false;
        }
    }

    std::cout << (allPass ? "[OVERALL PASS]\n" : "[OVERALL FAIL]\n");
    return allPass ? 0 : 1;
}
