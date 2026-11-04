// GOD PROMPT "CKKS PHASE 13": correctness test for encrypt_host_shoup
// (src/encrypt_host_shoup.cpp) against the UNTOUCHED production
// encrypt_host (src/encdec.cpp). Both functions sample randomness with
// the IDENTICAL seeded RNG sequence (same seed -> same std::mt19937_64/
// std::normal_distribution draws) and use the IDENTICAL embedding
// formula and NTT -- only the final combine loop's modular
// multiplication algorithm differs (generic (a*b)%q vs Shoup's
// precomputed-constant modmul). Shoup reduction is an EXACT modular
// multiplication technique, not an approximation, so the expected
// result is EXACT bit-identical equality -- no tolerance anywhere in
// this test.
#include "keygen.h"
#include <cstdint>
#include <vector>
#include <iostream>

namespace gpufhe {
void encrypt_host(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void encrypt_host_shoup(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                         const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                         const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
}

static bool runCase(uint32_t n, uint32_t sizeQ, const std::vector<int64_t>& m, uint64_t seed, const char* label) {
    using namespace gpufhe;
    std::vector<uint64_t> mod, root;
    native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> mids; native_primes(mids, sizeQ - 1, 50, n, mod);
      for (auto mm_ : mids) mod.push_back(mm_); }
    for (uint32_t t = 0; t < sizeQ; ++t) root.push_back(native_root(n, mod[t]));

    // Deterministic, non-trivial "public key" (doesn't need to be a
    // real keypair -- only the embedding/NTT/combine control flow and
    // arithmetic are under test, same convention as test_encrypt_host_
    // fast_matches_sequential.cpp).
    std::vector<uint64_t> pkA((size_t)sizeQ * n), pkB((size_t)sizeQ * n);
    for (uint32_t t = 0; t < sizeQ; ++t) {
        for (uint32_t k = 0; k < n; ++k) {
            pkA[(size_t)t * n + k] = (7 + t * 13 + k) % mod[t];
            pkB[(size_t)t * n + k] = (mod[t] - 1 - (t * 5 + k) % mod[t]) % mod[t];
        }
    }

    std::vector<uint64_t> c0_ref, c1_ref, c0_new, c1_new;
    encrypt_host(c0_ref, c1_ref, m, pkA, pkB, n, mod, root, 1, 3.19, seed);
    encrypt_host_shoup(c0_new, c1_new, m, pkA, pkB, n, mod, root, 1, 3.19, seed);

    bool match = (c0_ref.size() == c0_new.size()) && (c1_ref.size() == c1_new.size());
    size_t mismatches = 0;
    for (size_t i = 0; i < c0_ref.size() && match; ++i) if (c0_ref[i] != c0_new[i]) { match = false; ++mismatches; }
    for (size_t i = 0; i < c1_ref.size() && match; ++i) if (c1_ref[i] != c1_new[i]) { match = false; ++mismatches; }
    std::cout << "n=" << n << " sizeQ=" << sizeQ << " [" << label << "] -> "
              << (match ? "[EXACT MATCH]" : "[MISMATCH]") << "\n";
    return match;
}

int main() {
    bool allPass = true;

    for (uint32_t n : {16u, 1024u, 2048u, 4096u, 8192u}) {
        for (uint32_t sizeQ : {1u, 4u, 8u}) {
            // Zero message.
            { std::vector<int64_t> m(n, 0);
              if (!runCase(n, sizeQ, m, 111, "zero")) allPass = false; }
            // Single nonzero, large-magnitude coefficient (exercises
            // emb()'s handling of a big |x| relative to q, and a
            // non-trivial vT[k]/pkA/pkB combination through Shoup).
            { std::vector<int64_t> m(n, 0); m[0] = 1234567; if (n > 3) m[3] = -987654;
              if (!runCase(n, sizeQ, m, 222, "single-large")) allPass = false; }
            // Dense, varied-sign, varied-magnitude message.
            { std::vector<int64_t> m(n); for (uint32_t k = 0; k < n; ++k) m[k] = (int64_t)((k % 97) - 48) * ((k % 2) ? 1 : 1000);
              if (!runCase(n, sizeQ, m, 333, "dense-mixed-magnitude")) allPass = false; }
            // Different seed -> different RNG draws (v/e0/e1), checks
            // the Shoup path isn't accidentally correct only for one
            // specific set of sampled residues.
            { std::vector<int64_t> m(n); for (uint32_t k = 0; k < n; ++k) m[k] = (int64_t)(k % 5) - 2;
              if (!runCase(n, sizeQ, m, 999999, "alt-seed")) allPass = false; }
        }
    }

    std::cout << (allPass ? "[OVERALL PASS]\n" : "[OVERALL FAIL]\n");
    return allPass ? 0 : 1;
}
