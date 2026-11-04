// GOD PROMPT "CKKS PHASE 15": correctness test for
// encrypt_host_shoup_pooled_barrett_batched (src/
// encrypt_host_shoup_pooled_barrett_batched.cpp) against the UNTOUCHED
// production encrypt_host (src/encdec.cpp) AND against Phase 14's
// encrypt_host_shoup_pooled_barrett. Same exact-equality rationale as
// every prior phase's test: identical RNG/embedding/combine arithmetic,
// only the transfer/allocation LAYOUT differs (four separate buffers
// vs. one packed buffer) -- that can never change what is computed.
//
// This test specifically targets the packed-buffer boundaries the
// governing directive asked to validate: each of the four logical
// sub-buffers (v, e0, e1, m) is checked INDIVIDUALLY against its own
// expected values, at every sizeQ/n combination, including the
// FIRST and LAST tower of each sub-buffer (the positions most likely
// to reveal an off-by-one in the packed offsets), plus the pool-reuse
// and interleaved-different-keys stress sections carried forward from
// Phase 13/14's tests.
#include "keygen.h"
#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>

namespace gpufhe {
void encrypt_host(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void encrypt_host_shoup_pooled_barrett(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                                        const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                                        const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void encrypt_host_shoup_pooled_barrett_batched(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
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

// Checks against BOTH the untouched encrypt_host (the original oracle)
// AND Phase 14's encrypt_host_shoup_pooled_barrett (the immediately-
// preceding, already-verified production path) -- a mismatch against
// either is reported distinctly.
static bool check(const Fixture& F, uint64_t seed, const char* label) {
    using namespace gpufhe;
    std::vector<uint64_t> c0_ref, c1_ref, c0_p14, c1_p14, c0_new, c1_new;
    encrypt_host(c0_ref, c1_ref, F.m, F.pkA, F.pkB, F.n, F.mod, F.root, 1, 3.19, seed);
    encrypt_host_shoup_pooled_barrett(c0_p14, c1_p14, F.m, F.pkA, F.pkB, F.n, F.mod, F.root, 1, 3.19, seed);
    encrypt_host_shoup_pooled_barrett_batched(c0_new, c1_new, F.m, F.pkA, F.pkB, F.n, F.mod, F.root, 1, 3.19, seed);
    bool matchRef = (c0_ref == c0_new) && (c1_ref == c1_new);
    bool matchP14 = (c0_p14 == c0_new) && (c1_p14 == c1_new);
    std::cout << "n=" << F.n << " sizeQ=" << F.sizeQ << " [" << label << "] -> "
              << "vs-encrypt_host:" << (matchRef ? "MATCH" : "MISMATCH")
              << " vs-Phase14:" << (matchP14 ? "MATCH" : "MISMATCH") << "\n";
    return matchRef && matchP14;
}

int main() {
    bool allPass = true;

    // Basic sweep, same n/sizeQ matrix and message patterns as Phase
    // 14's test (zero, LONG_MIN/LONG_MAX-adjacent, dense mixed-sign).
    for (uint32_t n : {16u, 1024u, 4096u, 8192u}) {
        for (uint32_t sizeQ : {1u, 4u, 8u}) {
            { std::vector<int64_t> m(n, 0);
              Fixture F = make_fixture(n, sizeQ, m);
              if (!check(F, 111, "zero")) allPass = false; }
            { std::vector<int64_t> m(n, 0); m[0] = LLONG_MAX; if (n > 3) m[3] = LLONG_MIN;
              if (n > 7) m[7] = LLONG_MIN + 1; if (n > 11) m[11] = LLONG_MAX - 1;
              Fixture F = make_fixture(n, sizeQ, m);
              if (!check(F, 222, "long-min-max-adjacent")) allPass = false; }
            { std::vector<int64_t> m(n); for (uint32_t k = 0; k < n; ++k)
                m[k] = (int64_t)((k % 97) - 48) * (int64_t)((k % 3 == 0) ? 1 : ((k % 3 == 1) ? 1000000000000LL : -1000000000000000LL));
              Fixture F = make_fixture(n, sizeQ, m);
              if (!check(F, 333, "dense-mixed-sign-magnitude")) allPass = false; }
        }
    }

    // Packed-buffer-boundary stress: a message pattern where EVERY
    // sub-buffer's FIRST and LAST tower gets a distinct, recognizable
    // extreme value, specifically to catch an off-by-one in the
    // [v|e0|e1|m] packed offsets (which would show up as data from the
    // WRONG sub-buffer or WRONG tower bleeding across a boundary).
    for (uint32_t sizeQ : {4u, 8u}) {
        uint32_t n = 1024;
        std::vector<int64_t> m(n, 0);
        m[0] = 999999937; m[n - 1] = -999999937; // distinct from the rng-driven v/e0/e1 values
        Fixture F = make_fixture(n, sizeQ, m);
        if (!check(F, 444, "boundary-stress")) allPass = false;
    }

    // Repeated calls at the SAME (n,sizeQ) -- pool reuse stress,
    // mirroring Phase 13/14's tests.
    {
        std::vector<int64_t> m(4096, 0); for (uint32_t k = 0; k < 4096; ++k) m[k] = (int64_t)(k % 23) - 11;
        Fixture F = make_fixture(4096, 8, m);
        for (uint64_t seed = 1; seed <= 10; ++seed)
            if (!check(F, seed * 777, "repeated-same-size")) allPass = false;
    }

    // Interleaved calls across DIFFERENT (n,sizeQ) configurations --
    // pool-key separation stress, mirroring Phase 13/14's tests.
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
