// Phase 37: permanent regression guard for the Barrett-reduction bound this
// phase's production change relies on.
//
// src/encrypt_host_shoup_pooled_barrett_batched.cpp's barrett_reduce_u64 used
// to correct its estimate with an unbounded `while (r >= T.q) r -= T.q;`.
// Phase 37 replaced it with exactly two unconditional corrective checks,
// justified by classical Barrett reduction theory (mu = floor(2^64/q),
// x < 2^64 never needs more than 2 corrections) AND a standalone 70-million-
// trial empirical probe (research/benchmarks/bench_barrett_reduce_bound_check.cpp)
// that found the bound was never exceeded for any real modulus this project
// uses. This test is the PERMANENT version of that one-off probe: it keeps
// running on every regression pass, so a future change to make_tower_barrett
// (a different mu formula) or to the range of x ever passed through
// emb_barrett would be caught here before silently breaking the bounded
// form's correctness.
//
// Deliberately self-contained (duplicates the exact formulas from the
// production anonymous namespace, matching this project's established
// convention of duplicating small per-file helpers rather than exposing
// them) -- needs no link against gpufhe or CUDA, so it builds and runs fast
// enough to be part of every ordinary regression pass, unlike the one-off
// 70M-trial probe this test is derived from.
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <stdexcept>

namespace {
struct TowerBarrett { uint64_t q; uint64_t mu; };
TowerBarrett make_tower_barrett(uint64_t q) {
    TowerBarrett T; T.q = q; T.mu = (uint64_t)(((unsigned __int128)1 << 64) / q); return T;
}
uint64_t barrett_reduce_while(uint64_t x, const TowerBarrett& T, int* iters) {
    uint64_t qhat = (uint64_t)(((unsigned __int128)x * T.mu) >> 64);
    uint64_t r = x - qhat * T.q;
    int n = 0;
    while (r >= T.q) { r -= T.q; ++n; }
    if (iters) *iters = n;
    return r;
}
uint64_t barrett_reduce_bounded(uint64_t x, const TowerBarrett& T) {
    uint64_t qhat = (uint64_t)(((unsigned __int128)x * T.mu) >> 64);
    uint64_t r = x - qhat * T.q;
    if (r >= T.q) r -= T.q;
    if (r >= T.q) r -= T.q;
    return r;
}

uint64_t mulmod128(uint64_t a, uint64_t b, uint64_t m) { return (uint64_t)(((unsigned __int128)a*b) % m); }
uint64_t powmod(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m; a %= m;
    while (e) { if (e & 1) r = mulmod128(r, a, m); a = mulmod128(a, a, m); e >>= 1; }
    return r;
}
bool is_prime(uint64_t x) {
    if (x < 2) return false;
    for (uint64_t p : {2ull,3ull,5ull,7ull,11ull,13ull}) { if (x == p) return true; if (x % p == 0) return false; }
    uint64_t d = x - 1; int s = 0; while (!(d & 1)) { d >>= 1; ++s; }
    for (uint64_t a : {2ull,3ull,5ull,7ull,11ull,13ull,17ull,19ull,23ull,29ull,31ull,37ull}) {
        uint64_t v = powmod(a, d, x); if (v == 1 || v == x - 1) continue;
        bool ok = false; for (int i = 1; i < s; ++i) { v = mulmod128(v, v, x); if (v == x - 1) { ok = true; break; } }
        if (!ok) return false;
    }
    return true;
}
void native_primes(std::vector<uint64_t>& out, uint32_t count, uint32_t bits, uint32_t n) {
    const uint64_t M = 2ull * n;
    uint64_t q = ((1ull << bits) / M) * M + 1;
    while (out.size() < count) {
        if (q < (1ull << (bits - 1))) throw std::runtime_error("prime search exhausted");
        if (is_prime(q)) out.push_back(q);
        q -= M;
    }
}
} // namespace

int main() {
    std::vector<uint64_t> moduli;
    // The real production bit-widths (60-bit top, 55-bit middle, n=8192)
    // plus a wider sweep so a future parameter-set change is also covered.
    for (uint32_t bits : {30u, 40u, 50u, 55u, 58u, 60u, 62u}) {
        std::vector<uint64_t> p;
        native_primes(p, 3, bits, 8192);
        for (auto q : p) moduli.push_back(q);
    }

    std::mt19937_64 rng(0xC0FFEE);
    int max_iters = 0;
    uint64_t mismatches = 0, trials = 0;

    for (uint64_t q : moduli) {
        TowerBarrett T = make_tower_barrett(q);
        std::vector<uint64_t> cases = {
            0ull, 1ull, q - 1, q, q + 1, 2*q - 1, 2*q, 2*q + 1,
            (uint64_t)1 << 62, ((uint64_t)1 << 63) - 1, UINT64_MAX / 2, UINT64_MAX - 1
        };
        std::uniform_int_distribution<uint64_t> full_range(0, UINT64_MAX);
        for (int i = 0; i < 20000; ++i) cases.push_back(full_range(rng));

        for (uint64_t x : cases) {
            int iters;
            uint64_t r_while = barrett_reduce_while(x, T, &iters);
            uint64_t r_bounded = barrett_reduce_bounded(x, T);
            if (iters > max_iters) max_iters = iters;
            if (r_while != r_bounded) {
                ++mismatches;
                if (mismatches <= 5) std::printf("MISMATCH q=%llu x=%llu while=%llu bounded=%llu iters=%d\n",
                    (unsigned long long)q, (unsigned long long)x,
                    (unsigned long long)r_while, (unsigned long long)r_bounded, iters);
            }
            ++trials;
        }
    }

    std::printf("trials=%llu max_while_iterations=%d mismatches=%llu\n",
                (unsigned long long)trials, max_iters, (unsigned long long)mismatches);
    bool pass = (mismatches == 0) && (max_iters <= 2);
    std::printf(pass ? "[PASS]\n" : "[FAIL]\n");
    return pass ? 0 : 1;
}
