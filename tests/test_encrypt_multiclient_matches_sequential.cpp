// Phase 38: correctness test for encrypt_host_shoup_pooled_barrett_batched_
// multiclient (src/encrypt_host_shoup_pooled_barrett_batched_multiclient.cpp)
// against M independent calls to the existing, already-verified
// encrypt_host_shoup_pooled_barrett_batched (single-client). Same
// exact-equality rationale as every prior phase's batching/merge test: the
// batched function is designed to be an EXACT reorganization of the same
// per-client work (see that file's own header comment for the kernel-level
// argument), so client i's output must be BIT-IDENTICAL to what a standalone
// call with that client's own (m, seed) would produce.
//
// Also covers: batch-size boundary (M=1, the smallest possible batch, and a
// large M), independent randomness (two clients in the same batch with
// IDENTICAL plaintexts but DIFFERENT seeds must produce DIFFERENT
// ciphertexts), and resource-lifetime (repeated calls at the same (n,sizeQ,M)
// -- pool reuse stress -- and interleaved calls across DIFFERENT (n,sizeQ,M)
// -- pool-key separation stress).
#include "keygen.h"
#include <cstdint>
#include <climits>
#include <vector>
#include <string>
#include <iostream>

namespace gpufhe {
void encrypt_host_shoup_pooled_barrett_batched(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                                                const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                                                const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void encrypt_host_shoup_pooled_barrett_batched_multiclient(
    std::vector<std::vector<uint64_t>>&, std::vector<std::vector<uint64_t>>&,
    const std::vector<std::vector<int64_t>>&,
    const std::vector<uint64_t>&, const std::vector<uint64_t>&,
    uint32_t, const std::vector<uint64_t>&, const std::vector<uint64_t>&,
    uint64_t, double, const std::vector<uint64_t>&);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
}

struct Fixture {
    uint32_t n, sizeQ;
    std::vector<uint64_t> mod, root, pkA, pkB;
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
    return F;
}

static int g_pass = 0, g_fail = 0;
static void check(bool cond, const std::string& label) {
    if (cond) { ++g_pass; std::cout << "  [PASS] " << label << "\n"; }
    else { ++g_fail; std::cout << "  [FAIL] " << label << "\n"; }
}

// Builds M distinct plaintext patterns (client i's own message, derived
// deterministically from i so results are reproducible across runs) and M
// distinct seeds (derived deterministically from a base seed + client index,
// matching how the real workload already derives per-client seeds, e.g.
// bench_private_aggregation_workload.cpp's `5000 + rep*100 + k`).
static void make_batch(uint32_t n, uint32_t M, uint64_t baseSeed,
                        std::vector<std::vector<int64_t>>& m_list, std::vector<uint64_t>& seeds) {
    m_list.resize(M); seeds.resize(M);
    for (uint32_t i = 0; i < M; ++i) {
        m_list[i].assign(n, 0);
        m_list[i][0] = (int64_t)(1000003 * (i + 1));
        if (n > 3) m_list[i][3] = -(int64_t)(777 * (i + 1));
        if (n > 7) m_list[i][7] = LLONG_MAX - (int64_t)i;
        seeds[i] = baseSeed + i * 97;
    }
}

static bool check_batch_matches_sequential(const Fixture& F, uint32_t M, uint64_t baseSeed, const char* label) {
    using namespace gpufhe;
    std::vector<std::vector<int64_t>> m_list;
    std::vector<uint64_t> seeds;
    make_batch(F.n, M, baseSeed, m_list, seeds);

    std::vector<std::vector<uint64_t>> c0_batch, c1_batch;
    encrypt_host_shoup_pooled_barrett_batched_multiclient(c0_batch, c1_batch, m_list, F.pkA, F.pkB,
                                                           F.n, F.mod, F.root, 1, 3.19, seeds);
    bool allMatch = (c0_batch.size() == M) && (c1_batch.size() == M);
    for (uint32_t i = 0; i < M && allMatch; ++i) {
        std::vector<uint64_t> c0_seq, c1_seq;
        encrypt_host_shoup_pooled_barrett_batched(c0_seq, c1_seq, m_list[i], F.pkA, F.pkB,
                                                   F.n, F.mod, F.root, 1, 3.19, seeds[i]);
        if (c0_seq != c0_batch[i] || c1_seq != c1_batch[i]) allMatch = false;
    }
    std::cout << "n=" << F.n << " sizeQ=" << F.sizeQ << " M=" << M << " [" << label << "] -> "
              << (allMatch ? "MATCH (all clients bit-exact vs sequential)" : "MISMATCH") << "\n";
    return allMatch;
}

int main() {
    bool allPass = true;

    // Basic sweep: n/sizeQ matrix x batch sizes including the boundary
    // M=1 (smallest possible batch) and a larger M.
    for (uint32_t n : {1024u, 4096u, 8192u}) {
        for (uint32_t sizeQ : {1u, 4u}) {
            Fixture F = make_fixture(n, sizeQ);
            for (uint32_t M : {1u, 2u, 8u, 17u}) {  // 17: deliberately non-power-of-2
                if (!check_batch_matches_sequential(F, M, 1000 + M, "basic-sweep")) allPass = false;
            }
        }
    }

    // Independent-randomness check: two clients with the IDENTICAL
    // plaintext but DIFFERENT seeds must produce DIFFERENT ciphertexts
    // (confirms per-client randomness is not accidentally shared/reused --
    // without exposing or comparing any secret material, only public
    // ciphertext outputs).
    {
        Fixture F = make_fixture(8192, 4);
        std::vector<std::vector<int64_t>> m_list(2, std::vector<int64_t>(F.n, 0));
        m_list[0][0] = 42; m_list[1][0] = 42; // IDENTICAL plaintext
        std::vector<uint64_t> seeds = {111, 222}; // DIFFERENT seeds
        std::vector<std::vector<uint64_t>> c0, c1;
        gpufhe::encrypt_host_shoup_pooled_barrett_batched_multiclient(c0, c1, m_list, F.pkA, F.pkB,
                                                                       F.n, F.mod, F.root, 1, 3.19, seeds);
        bool differ = (c0[0] != c0[1]) || (c1[0] != c1[1]);
        check(differ, "two clients, identical plaintext, different seeds -> different ciphertexts (independent randomness)");
    }

    // Resource-lifetime: repeated calls at the SAME (n,sizeQ,M) -- pool
    // reuse stress.
    {
        Fixture F = make_fixture(4096, 4);
        for (uint64_t base = 1; base <= 5; ++base)
            if (!check_batch_matches_sequential(F, 8, base * 777, "repeated-same-size")) allPass = false;
    }

    // Resource-lifetime: interleaved calls across DIFFERENT (n,sizeQ,M)
    // configurations -- pool-key separation stress (each (n,sizeQ,M)
    // combination gets its own pool entry; this must not cross-contaminate).
    {
        Fixture F1 = make_fixture(1024, 4);
        Fixture F2 = make_fixture(8192, 1);
        Fixture F3 = make_fixture(2048, 4);
        for (int round = 0; round < 3; ++round) {
            if (!check_batch_matches_sequential(F1, 4, 2000 + round, "interleaved-A")) allPass = false;
            if (!check_batch_matches_sequential(F2, 16, 3000 + round, "interleaved-B")) allPass = false;
            if (!check_batch_matches_sequential(F3, 2, 4000 + round, "interleaved-C")) allPass = false;
        }
    }

    std::cout << (allPass && g_fail == 0 ? "[OVERALL PASS]\n" : "[OVERALL FAIL]\n");
    return (allPass && g_fail == 0) ? 0 : 1;
}
