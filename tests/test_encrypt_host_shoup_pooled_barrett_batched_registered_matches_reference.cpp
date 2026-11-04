// GOD PROMPT "CKKS PHASE 20": correctness test for
// encrypt_host_shoup_pooled_barrett_batched_registered (src/
// encrypt_host_shoup_pooled_barrett_batched_registered.cpp) against the
// UNTOUCHED production encrypt_host (src/encdec.cpp), Phase 15's
// encrypt_host_shoup_pooled_barrett_batched, AND Phase 15's
// encrypt_host_shoup_pooled_barrett_batched_pinned. Same exact-equality
// rationale as every prior phase's test: switching the packed buffer's
// host-side allocation/registration MECHANISM can never change what is
// computed. Mirrors tests/test_encrypt_host_shoup_pooled_barrett_
// batched_pinned_matches_reference.cpp exactly, extended with one more
// reference implementation.
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
void encrypt_host_shoup_pooled_barrett_batched_pinned(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                                                        const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                                                        const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void encrypt_host_shoup_pooled_barrett_batched_registered(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
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
    std::vector<uint64_t> c0_ref, c1_ref, c0_batched, c1_batched, c0_pinned, c1_pinned, c0_new, c1_new;
    encrypt_host(c0_ref, c1_ref, F.m, F.pkA, F.pkB, F.n, F.mod, F.root, 1, 3.19, seed);
    encrypt_host_shoup_pooled_barrett_batched(c0_batched, c1_batched, F.m, F.pkA, F.pkB, F.n, F.mod, F.root, 1, 3.19, seed);
    encrypt_host_shoup_pooled_barrett_batched_pinned(c0_pinned, c1_pinned, F.m, F.pkA, F.pkB, F.n, F.mod, F.root, 1, 3.19, seed);
    encrypt_host_shoup_pooled_barrett_batched_registered(c0_new, c1_new, F.m, F.pkA, F.pkB, F.n, F.mod, F.root, 1, 3.19, seed);
    bool matchRef = (c0_ref == c0_new) && (c1_ref == c1_new);
    bool matchBatched = (c0_batched == c0_new) && (c1_batched == c1_new);
    bool matchPinned = (c0_pinned == c0_new) && (c1_pinned == c1_new);
    std::cout << "n=" << F.n << " sizeQ=" << F.sizeQ << " [" << label << "] -> "
              << "vs-encrypt_host:" << (matchRef ? "MATCH" : "MISMATCH")
              << " vs-batched:" << (matchBatched ? "MATCH" : "MISMATCH")
              << " vs-pinned:" << (matchPinned ? "MATCH" : "MISMATCH") << "\n";
    return matchRef && matchBatched && matchPinned;
}

int main() {
    bool allPass = true;

    for (uint32_t n : {16u, 1024u, 4096u, 8192u}) {
        for (uint32_t sizeQ : {1u, 4u, 8u}) {
            { std::vector<int64_t> m(n, 0);
              Fixture F = make_fixture(n, sizeQ, m);
              if (!check(F, 111, "zero")) allPass = false; }
            { std::vector<int64_t> m(n, 0); m[0] = LLONG_MAX; if (n > 3) m[3] = LLONG_MIN;
              Fixture F = make_fixture(n, sizeQ, m);
              if (!check(F, 222, "long-min-max-adjacent")) allPass = false; }
            { std::vector<int64_t> m(n); for (uint32_t k = 0; k < n; ++k)
                m[k] = (int64_t)((k % 97) - 48) * (int64_t)((k % 3 == 0) ? 1 : ((k % 3 == 1) ? 1000000000000LL : -1000000000000000LL));
              Fixture F = make_fixture(n, sizeQ, m);
              if (!check(F, 333, "dense-mixed-sign-magnitude")) allPass = false; }
        }
    }

    // Packed-buffer boundary: n=8192,sizeQ=8 is exactly the largest
    // size this library's packed layout actually exercises (matches
    // every prior phase's own boundary case).
    {
        std::vector<int64_t> m(8192); for (uint32_t k = 0; k < 8192; ++k) m[k] = (int64_t)(k % 199) - 99;
        Fixture F = make_fixture(8192, 8, m);
        if (!check(F, 444, "packed-buffer-boundary")) allPass = false;
    }

    // Repeated calls at the SAME (n,sizeQ) -- pool reuse stress.
    {
        std::vector<int64_t> m(4096, 0); for (uint32_t k = 0; k < 4096; ++k) m[k] = (int64_t)(k % 23) - 11;
        Fixture F = make_fixture(4096, 8, m);
        for (uint64_t seed = 1; seed <= 10; ++seed)
            if (!check(F, seed * 777, "repeated-same-size")) allPass = false;
    }

    // Interleaved calls across DIFFERENT (n,sizeQ) configurations.
    {
        std::vector<int64_t> m1(1024, 1), m2(8192, 2), m3(2048, 3);
        Fixture F1 = make_fixture(1024, 4, m1);
        Fixture F2 = make_fixture(8192, 8, m2);
        Fixture F3 = make_fixture(2048, 1, m3);
        for (int round = 0; round < 3; ++round) {
            if (!check(F1, 1000 + round, "interleaved-A")) allPass = false;
            if (!check(F2, 2000 + round, "interleaved-B")) allPass = false;
            if (!check(F3, 3000 + round, "interleaved-C")) allPass = false;
        }
    }

    std::cout << (allPass ? "[OVERALL PASS]\n" : "[OVERALL FAIL]\n");
    return allPass ? 0 : 1;
}
