// Phase 37 standalone reproducer: does barrett_reduce_u64's "while (r >= T.q)
// r -= T.q;" loop (src/encrypt_host_shoup_pooled_barrett_batched.cpp) ever
// need more than 2 corrective subtractions for the ACTUAL (q, x) pairs this
// project's production encrypt path uses? Classical Barrett reduction theory
// (mu = floor(2^64/q), x < 2^64) bounds the correction to at most 2 steps for
// this construction and input domain -- this probe verifies that bound
// EMPIRICALLY against real moduli (via native_primes, same bit-widths this
// project actually uses) and a large, varied set of x values (random +
// explicit boundary cases), rather than trusting the derivation alone.
//
// NOT registered with ctest. NOT linked into any production target. Exists
// only to justify (or refute) a proposed production change before it is
// ever written into src/. Zero production code touched by this file.
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <stdexcept>

// ---- exact copies of the two functions under test, from the production
// file's own anonymous namespace (kept byte-identical to what production
// has, so this probe tests the REAL formulas, not an approximation). ----
struct TowerBarrett { uint64_t q; uint64_t mu; };
static inline TowerBarrett make_tower_barrett(uint64_t q) {
    TowerBarrett T; T.q = q;
    T.mu = (uint64_t)(((unsigned __int128)1 << 64) / q);
    return T;
}
// Reference: the EXISTING production implementation (while loop), plus an
// iteration counter so we can see exactly how many corrections it ever needs.
static inline uint64_t barrett_reduce_u64_ref(uint64_t x, const TowerBarrett& T, int* iters) {
    uint64_t qhat = (uint64_t)(((unsigned __int128)x * T.mu) >> 64);
    uint64_t r = x - qhat * T.q;
    int n = 0;
    while (r >= T.q) { r -= T.q; ++n; }
    if (iters) *iters = n;
    return r;
}
// Candidate: bounded to exactly 2 unconditional corrective checks (branch-free
// under -O3, typically compiled to CMOV), no loop construct at all.
static inline uint64_t barrett_reduce_u64_fast(uint64_t x, const TowerBarrett& T) {
    uint64_t qhat = (uint64_t)(((unsigned __int128)x * T.mu) >> 64);
    uint64_t r = x - qhat * T.q;
    if (r >= T.q) r -= T.q;
    if (r >= T.q) r -= T.q;
    return r;
}

// ---- minimal native_primes/is_prime port (same algorithm as src/params.cpp,
// just so this standalone file needs no link dependency on the production
// library) ----
namespace {
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
    // Real project usage: 60-bit "top" primes and 55-bit "middle" primes,
    // at the real production n=8192, PLUS a broader sweep across every
    // bit-width actually used elsewhere in this project's test fixtures
    // (30-62 bits) so the bound is checked well beyond just today's
    // exact parameter set.
    for (uint32_t bits : {30u,40u,50u,55u,58u,60u,62u}) {
        std::vector<uint64_t> p;
        native_primes(p, 5, bits, 8192);
        for (auto q : p) moduli.push_back(q);
    }
    std::printf("Testing %zu real moduli (bit-widths 30,40,50,55,58,60,62; 5 each via native_primes)\n", moduli.size());

    std::mt19937_64 rng(0xC0FFEE);
    int global_max_iters = 0;
    uint64_t total_trials = 0;
    uint64_t mismatches = 0;

    for (uint64_t q : moduli) {
        TowerBarrett T = make_tower_barrett(q);
        // Explicit boundary cases, per modulus.
        std::vector<uint64_t> boundaries = {
            0ull, 1ull, q - 1, q, q + 1, 2*q - 1, 2*q, 2*q + 1,
            (uint64_t)1 << 62, ((uint64_t)1 << 63) - 1, UINT64_MAX / 2, UINT64_MAX - 1
        };
        for (uint64_t x : boundaries) {
            int iters_ref;
            uint64_t r_ref = barrett_reduce_u64_ref(x, T, &iters_ref);
            uint64_t r_fast = barrett_reduce_u64_fast(x, T);
            if (iters_ref > global_max_iters) global_max_iters = iters_ref;
            if (r_ref != r_fast) { ++mismatches; std::printf("MISMATCH (boundary) q=%llu x=%llu ref=%llu fast=%llu iters=%d\n",
                (unsigned long long)q,(unsigned long long)x,(unsigned long long)r_ref,(unsigned long long)r_fast,iters_ref); }
            ++total_trials;
        }
        // Realistic production domain: x is |signed coefficient|, bounded by
        // what emb_barrett ever actually passes (ternary -1/0/1, Gaussian
        // noise up to ~6*sigma, or a Delta-scaled plaintext coefficient up
        // to the full int64_t range) -- so sweep the FULL uint64_t-as-abs
        // range, not just values near q.
        std::uniform_int_distribution<uint64_t> full_range(0, UINT64_MAX);
        std::uniform_int_distribution<uint64_t> near_q(0, 4*q); // the realistic "small multiple of q" region
        for (int i = 0; i < 2000000; ++i) {
            uint64_t x = (i % 2 == 0) ? full_range(rng) : near_q(rng);
            int iters_ref;
            uint64_t r_ref = barrett_reduce_u64_ref(x, T, &iters_ref);
            uint64_t r_fast = barrett_reduce_u64_fast(x, T);
            if (iters_ref > global_max_iters) global_max_iters = iters_ref;
            if (r_ref != r_fast) {
                ++mismatches;
                if (mismatches <= 10) std::printf("MISMATCH (random) q=%llu x=%llu ref=%llu fast=%llu iters=%d\n",
                    (unsigned long long)q,(unsigned long long)x,(unsigned long long)r_ref,(unsigned long long)r_fast,iters_ref);
            }
            ++total_trials;
        }
    }

    std::printf("Total trials: %llu\n", (unsigned long long)total_trials);
    std::printf("Max while-loop iterations ever observed (reference impl): %d\n", global_max_iters);
    std::printf("Mismatches between reference (while) and candidate (2x bounded if): %llu\n", (unsigned long long)mismatches);
    std::printf(mismatches == 0 ? "RESULT: PASS -- bounded-if candidate is bit-exact with the while-loop reference across all trials.\n"
                                 : "RESULT: FAIL -- candidate diverges from reference; do NOT port to production.\n");
    return mismatches == 0 ? 0 : 1;
}
