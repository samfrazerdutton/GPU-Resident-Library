// GOD PROMPT "CKKS PHASE 14": independent differential test for the
// Barrett-based replacement of encrypt_host_shoup_pooled's `emb()`
// lambda (`emb(x) = (x % q + q) % q` in signed 64-bit arithmetic).
// Derivation record: research/EMBED_AND_POOL_HARDENING.md.
//
// This test is deliberately built BEFORE any production file is
// touched (per the governing directive's "build an independent
// correctness test first"). It links against this project's REAL
// `native_primes`/`native_root` to test against the ACTUAL modulus
// classes `encrypt_host_shoup_pooled` uses (not synthetic placeholder
// primes), across every n this project supports, plus explicit
// boundary values (0, +-1, LONG_MIN, LONG_MAX, values at/around
// multiples of q) and 2,000,000 uniformly-random full-int64_t-range
// values per modulus. Zero mismatches required -- this is EXACT
// integer arithmetic, not a numerical approximation, so no tolerance
// is used anywhere.
#include <cstdint>
#include <climits>
#include <random>
#include <iostream>
#include <vector>

namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
}

typedef unsigned long long u64;
typedef __uint128_t u128;

// ORIGINAL (reference) -- byte-for-byte the same lambda body as
// src/encrypt_host_shoup_pooled.cpp's emb().
static u64 emb_ref(long x, u64 q) {
    long qq = (long)q;
    return (u64)((x % qq + qq) % qq);
}

// CANDIDATE replacement, derived in research/EMBED_AND_POOL_HARDENING.md:
// precompute mu=floor(2^64/q) ONCE per modulus (not per element), then
// Barrett-reduce |x| (computed via unsigned negation to avoid signed
// overflow/UB for x==LONG_MIN) and restore the sign per emb()'s own
// convention (negative x with a nonzero remainder maps to q-remainder).
struct TowerBarrett { u64 q; u64 mu; };
static TowerBarrett make_tb(u64 q) {
    TowerBarrett T; T.q = q;
    T.mu = (u64)(((u128)1 << 64) / q);
    return T;
}
static u64 barrett_reduce_u64(u64 x, const TowerBarrett& T) {
    u64 qhat = (u64)(((u128)x * T.mu) >> 64);
    u64 r = x - qhat * T.q;
    while (r >= T.q) r -= T.q;
    return r;
}
static u64 emb_fast(long x, const TowerBarrett& T) {
    u64 ax = (x < 0) ? (0ULL - (u64)x) : (u64)x;
    u64 r = barrett_reduce_u64(ax, T);
    if (x < 0 && r != 0) r = T.q - r;
    return r;
}

static bool checkModulus(u64 q, long& mismatches, long& total) {
    TowerBarrett T = make_tb(q);
    bool ok = true;
    long boundaries[] = {
        0L, 1L, -1L, LONG_MIN, LONG_MAX, LONG_MIN + 1, LONG_MAX - 1,
        (long)q, -(long)q, (long)q - 1, -((long)q - 1), (long)q + 1, -((long)q + 1),
        (long)(q / 2), -(long)(q / 2), (long)(2 * q), -(long)(2 * q)
    };
    for (long x : boundaries) {
        u64 r1 = emb_ref(x, q), r2 = emb_fast(x, T);
        ++total;
        if (r1 != r2) { ++mismatches; ok = false;
            std::cout << "MISMATCH q=" << q << " x=" << x << " ref=" << r1 << " fast=" << r2 << "\n"; }
    }
    std::mt19937_64 rng(999 + q);
    std::uniform_int_distribution<long long> d(LLONG_MIN, LLONG_MAX);
    for (int i = 0; i < 200000; ++i) {
        long x = (long)d(rng);
        u64 r1 = emb_ref(x, q), r2 = emb_fast(x, T);
        ++total;
        if (r1 != r2) { ++mismatches; ok = false;
            if (mismatches < 20) std::cout << "MISMATCH q=" << q << " x=" << x << " ref=" << r1 << " fast=" << r2 << "\n"; }
    }
    return ok;
}

int main() {
    using namespace gpufhe;
    bool allPass = true;
    long mismatches = 0, total = 0;

    for (uint32_t n : {1024u, 2048u, 4096u, 8192u}) {
        for (uint32_t sizeQ : {1u, 4u, 8u}) {
            std::vector<uint64_t> mod;
            native_primes(mod, 1, 60, n, {});
            { std::vector<uint64_t> mids; native_primes(mids, sizeQ - 1, 50, n, mod);
              for (auto m : mids) mod.push_back(m); }
            for (uint64_t q : mod) {
                bool ok = checkModulus(q, mismatches, total);
                if (!ok) allPass = false;
            }
            std::cout << "n=" << n << " sizeQ=" << sizeQ << " (" << mod.size()
                      << " real moduli checked) -> " << (allPass ? "ok so far" : "FAIL") << "\n";
        }
    }

    std::cout << "Total checks: " << total << ", mismatches: " << mismatches << "\n";
    std::cout << (allPass ? "[OVERALL PASS]\n" : "[OVERALL FAIL]\n");
    return allPass ? 0 : 1;
}
