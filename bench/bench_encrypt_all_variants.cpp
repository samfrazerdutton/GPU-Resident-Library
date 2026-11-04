// GOD PROMPT "CKKS PHASE 13": unified three-way interleaved A/B/C
// benchmark -- OLD (encrypt_host), Shoup-only (encrypt_host_shoup),
// Shoup+pooled (encrypt_host_shoup_pooled) -- all three measured within
// ONE run, interleaved per-rep (old,shoup,pooled,old,shoup,pooled,...)
// so every reported ratio comes from the SAME thermal/scheduling
// conditions (avoids the cross-binary-invocation drift that made
// bench_encrypt_shoup_vs_reference.cpp's and
// bench_encrypt_shoup_pooled_vs_reference.cpp's absolute numbers not
// directly comparable to each other). Correctness is established
// separately (tests/test_encrypt_host_shoup_matches_reference.cpp,
// tests/test_encrypt_host_shoup_pooled_matches_reference.cpp) -- this
// file ONLY measures timing.
#include "keygen.h"
#include <cstdint>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <cstdlib>

namespace gpufhe {
void encrypt_host(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void encrypt_host_shoup(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                         const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                         const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void encrypt_host_shoup_pooled(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                                const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                                const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
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
    size_t reps = argc > 1 ? (size_t)std::atoi(argv[1]) : 15;
    size_t trials = argc > 2 ? (size_t)std::atoi(argv[2]) : 7;
    uint32_t sizeQ = 8;

    for (uint32_t n : {1024u, 2048u, 4096u, 8192u}) {
        std::cout << "\n=== n=" << n << " sizeQ=" << sizeQ << " ===\n";
        std::vector<uint64_t> mod;
        gpufhe::native_primes(mod, 1, 60, n, {});
        { std::vector<uint64_t> mids; gpufhe::native_primes(mids, sizeQ - 1, 50, n, mod);
          for (auto m : mids) mod.push_back(m); }
        std::vector<uint64_t> root(sizeQ);
        for (uint32_t i = 0; i < sizeQ; ++i) root[i] = gpufhe::native_root(n, mod[i]);
        auto KP = gpufhe::keygen_host(n, mod, root, 1, 3.19, 777);

        std::vector<int64_t> msg(n, 0);
        msg[0] = 12345; if (n > 3) msg[3] = -777;

        std::vector<uint64_t> c0, c1;
        gpufhe::encrypt_host(c0, c1, msg, KP.pkA, KP.pkB, n, mod, root, 1, 3.19, 1);
        gpufhe::encrypt_host_shoup(c0, c1, msg, KP.pkA, KP.pkB, n, mod, root, 1, 3.19, 1);
        gpufhe::encrypt_host_shoup_pooled(c0, c1, msg, KP.pkA, KP.pkB, n, mod, root, 1, 3.19, 1);

        std::vector<double> old_s, shoup_s, pooled_s;
        for (size_t trial = 0; trial < trials; ++trial) {
            std::vector<double> old_r, shoup_r, pooled_r;
            for (size_t r = 0; r < reps; ++r) {
                uint64_t seed = 10000 + trial * reps + r;
                auto t0 = std::chrono::steady_clock::now();
                gpufhe::encrypt_host(c0, c1, msg, KP.pkA, KP.pkB, n, mod, root, 1, 3.19, seed);
                auto t1 = std::chrono::steady_clock::now();
                gpufhe::encrypt_host_shoup(c0, c1, msg, KP.pkA, KP.pkB, n, mod, root, 1, 3.19, seed);
                auto t2 = std::chrono::steady_clock::now();
                gpufhe::encrypt_host_shoup_pooled(c0, c1, msg, KP.pkA, KP.pkB, n, mod, root, 1, 3.19, seed);
                auto t3 = std::chrono::steady_clock::now();
                old_r.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                shoup_r.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
                pooled_r.push_back(std::chrono::duration<double, std::micro>(t3 - t2).count());
            }
            auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
            old_s.push_back(med(old_r)); shoup_s.push_back(med(shoup_r)); pooled_s.push_back(med(pooled_r));
        }
        Stats so = stats_of(old_s), ss = stats_of(shoup_s), sp = stats_of(pooled_s);
        report("OLD encrypt_host", so);
        report("Shoup-only", ss);
        report("Shoup+pooled", sp);
        std::cout << "    Shoup-only vs OLD:    " << (so.p50 / ss.p50) << "x\n";
        std::cout << "    Shoup+pooled vs OLD:  " << (so.p50 / sp.p50) << "x\n";
        std::cout << "    pooled incremental vs Shoup-only: " << (ss.p50 / sp.p50) << "x\n";
    }
    return 0;
}
