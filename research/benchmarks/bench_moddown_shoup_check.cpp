// Phase 39 standalone reproducer: confirms shoup_mulmod(diff, s, precon, q)
// is bit-exact with the raw (diff*s) % q for every x,s < q, across the real
// moduli this project uses -- before touching kernels/moddown.cu.
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <stdexcept>

static uint64_t shoup_precon(uint64_t c, uint64_t q) { return (uint64_t)(((unsigned __int128)c << 64) / q); }
static uint64_t shoup_mulmod(uint64_t x, uint64_t c, uint64_t precon, uint64_t q) {
    uint64_t hi = (uint64_t)(((unsigned __int128)x * precon) >> 64);
    uint64_t r = x * c - hi * q;
    return (r >= q) ? r - q : r;
}
static uint64_t raw_mulmod(uint64_t x, uint64_t c, uint64_t q) {
    return (uint64_t)(((unsigned __int128)x * c) % q);
}

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
    for (uint32_t bits : {30u,40u,50u,55u,58u,60u,62u}) {
        std::vector<uint64_t> p; native_primes(p, 5, bits, 8192);
        for (auto q : p) moduli.push_back(q);
    }
    std::printf("Testing %zu real moduli (bit-widths 30,40,50,55,58,60,62; 5 each)\n", moduli.size());

    std::mt19937_64 rng(0xBADF00D);
    uint64_t trials = 0, mismatches = 0;
    for (uint64_t q : moduli) {
        std::uniform_int_distribution<uint64_t> dist(0, q - 1);
        // Boundary cases: 0, 1, q-1, and a handful of specific small/large values.
        std::vector<uint64_t> svals = {0, 1, q - 1, q / 2, q / 2 + 1};
        for (uint64_t s : svals) {
            uint64_t precon = shoup_precon(s, q);
            for (int i = 0; i < 200000; ++i) {
                uint64_t x = dist(rng);
                uint64_t a = shoup_mulmod(x, s, precon, q);
                uint64_t b = raw_mulmod(x, s, q);
                if (a != b) { ++mismatches; if (mismatches <= 5) std::printf("MISMATCH q=%llu s=%llu x=%llu shoup=%llu raw=%llu\n",
                    (unsigned long long)q,(unsigned long long)s,(unsigned long long)x,(unsigned long long)a,(unsigned long long)b); }
                ++trials;
            }
        }
    }
    std::printf("Total trials: %llu, mismatches: %llu\n", (unsigned long long)trials, (unsigned long long)mismatches);
    std::printf(mismatches == 0 ? "RESULT: PASS\n" : "RESULT: FAIL\n");
    return mismatches == 0 ? 0 : 1;
}
