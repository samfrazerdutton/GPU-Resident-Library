// GOD PROMPT "CKKS PHASE 14": standalone, isolated A/B benchmark for
// JUST the embedding loop -- OLD (signed-%-based `emb()`, two hardware
// `idiv` instructions per call, confirmed via assembly inspection) vs
// NEW (per-tower-precomputed Barrett reduction, src/
// encrypt_host_shoup_pooled_barrett.cpp's `emb_barrett()`). No GPU, no
// encryption, no randomness beyond generating representative input
// data once -- isolates exactly the arithmetic change this phase
// targets, interleaved per-trial (old,new,old,new...) to cancel
// thermal/scheduling drift. Correctness of this exact derivation is
// established separately and exhaustively in tests/
// test_embed_barrett_matches_reference.cpp -- this file ONLY measures
// timing.
#include <cstdint>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <climits>
#include <random>

namespace gpufhe { void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&); }

typedef unsigned long long u64;
typedef __uint128_t u128;

static u64 emb_old(long x, u64 q) {
    long qq = (long)q;
    return (u64)((x % qq + qq) % qq);
}
struct TowerBarrett { u64 q, mu; };
static TowerBarrett make_tb(u64 q) { TowerBarrett T; T.q = q; T.mu = (u64)(((u128)1 << 64) / q); return T; }
static u64 emb_barrett(long x, const TowerBarrett& T) {
    u64 ax = (x < 0) ? (0ULL - (u64)x) : (u64)x;
    u64 qhat = (u64)(((u128)ax * T.mu) >> 64);
    u64 r = ax - qhat * T.q;
    while (r >= T.q) r -= T.q;
    if (x < 0 && r != 0) r = T.q - r;
    return r;
}

namespace {
struct Stats { double min_, p50, p95, max_, mean; size_t n; };
Stats stats_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    Stats s; s.n = v.size(); s.min_ = v.front(); s.max_ = v.back();
    s.p50 = v[v.size() / 2]; s.p95 = v[(size_t)(0.95 * (v.size() - 1))];
    double sum = 0; for (double x : v) sum += x; s.mean = sum / v.size();
    return s;
}
void report(const char* label, const Stats& s) {
    std::cout << "    " << label << ": median=" << s.p50 << "us p95=" << s.p95
              << "us min=" << s.min_ << "us max=" << s.max_ << "us mean=" << s.mean
              << "us (n=" << s.n << ")\n";
}
} // namespace

int main(int argc, char** argv) {
    size_t reps = argc > 1 ? (size_t)std::atoi(argv[1]) : 20;
    size_t trials = argc > 2 ? (size_t)std::atoi(argv[2]) : 9;
    uint32_t sizeQ = 8;

    for (uint32_t n : {1024u, 2048u, 4096u, 8192u}) {
        std::cout << "\n=== n=" << n << " sizeQ=" << sizeQ << " ===\n";
        std::vector<uint64_t> mod;
        gpufhe::native_primes(mod, 1, 60, n, {});
        { std::vector<uint64_t> mids; gpufhe::native_primes(mids, sizeQ - 1, 50, n, mod);
          for (auto m : mids) mod.push_back(m); }

        // Representative operand set: ternary v (tiny), Gaussian-sized
        // e0/e1 (small), and message coefficients spanning the FULL
        // int64_t range (not just small values) -- same magnitude
        // classes the real embed loop actually sees.
        std::mt19937_64 rng(42);
        std::vector<long> vc(n), e0c(n), e1c(n);
        std::vector<long long> mc(n);
        std::normal_distribution<double> g(0.0, 3.19);
        std::uniform_int_distribution<long long> big(LLONG_MIN / 4, LLONG_MAX / 4);
        for (uint32_t k = 0; k < n; ++k) {
            vc[k] = (long)(rng() % 3) - 1;
            e0c[k] = (long)std::lround(g(rng)); e1c[k] = (long)std::lround(g(rng));
            mc[k] = big(rng);
        }

        std::vector<double> old_s, new_s;
        for (size_t trial = 0; trial < trials; ++trial) {
            std::vector<double> old_r, new_r;
            for (size_t r = 0; r < reps; ++r) {
                volatile uint64_t sink = 0;

                auto t0 = std::chrono::steady_clock::now();
                for (uint32_t t = 0; t < sizeQ; ++t) { uint64_t q = mod[t];
                    for (uint32_t k = 0; k < n; ++k) {
                        sink ^= emb_old(vc[k], q); sink ^= emb_old(e0c[k], q);
                        sink ^= emb_old(e1c[k], q); sink ^= emb_old((long)mc[k], q);
                    }
                }
                auto t1 = std::chrono::steady_clock::now();

                for (uint32_t t = 0; t < sizeQ; ++t) { uint64_t q = mod[t];
                    TowerBarrett TB = make_tb(q);
                    for (uint32_t k = 0; k < n; ++k) {
                        sink ^= emb_barrett(vc[k], TB); sink ^= emb_barrett(e0c[k], TB);
                        sink ^= emb_barrett(e1c[k], TB); sink ^= emb_barrett((long)mc[k], TB);
                    }
                }
                auto t2 = std::chrono::steady_clock::now();

                old_r.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                new_r.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
                (void)sink;
            }
            auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
            old_s.push_back(med(old_r)); new_s.push_back(med(new_r));
        }
        Stats so = stats_of(old_s), sn = stats_of(new_s);
        report("OLD embed (signed %, 2 idiv/call)", so);
        report("NEW embed (per-tower Barrett)", sn);
        std::cout << "    speedup (median): " << (so.p50 / sn.p50) << "x\n";
    }
    return 0;
}
