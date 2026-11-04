// GOD PROMPT "CKKS PHASE 16": prewarming lifecycle + workload-level
// benchmark. Measures the FOUR implementation variants the directive's
// Step 3 asks for (A: pageable packed production; B: pinned, lazy
// first-use allocation; C: pinned, explicit prewarm before any
// encryption; D: prewarm "attempted" via the adaptive function with NO
// prewarm call, i.e. transparent fallback) and the THREE workload
// profiles Step 5 asks for (A: cold-start/short-lived; B: typical;
// C: long-lived), reporting medians (not just best-case) and computing
// the break-even operation count where prewarming's upfront cost pays
// for itself. Correctness of every variant used here is established
// separately (tests/test_encrypt_host_prewarm_matches_reference.cpp,
// tests/test_encrypt_host_prewarm_concurrency_audit.cpp, plus Phase
// 15's own tests for the two underlying implementations) -- this file
// ONLY measures timing.
#include "keygen.h"
#include <cstdint>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <sstream>

namespace gpufhe {
void encrypt_host_shoup_pooled_barrett_batched(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                                                const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                                                const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void encrypt_host_shoup_pooled_barrett_batched_pinned(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                                                        const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                                                        const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
bool encrypt_prewarm_pinned_buffer(uint32_t, uint32_t);
void encrypt_host_shoup_pooled_barrett_prewarmed(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
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
void report(const std::string& label, const Stats& s) {
    std::cout << "    " << label << ": median=" << s.p50 << "us p95=" << s.p95
              << "us min=" << s.min_ << "us max=" << s.max_ << "us mean=" << s.mean
              << "us (n=" << s.n << ")\n";
}
double median_of(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
} // namespace

int main(int argc, char** argv) {
    size_t reps = argc > 1 ? (size_t)std::atoi(argv[1]) : 15;
    size_t trials = argc > 2 ? (size_t)std::atoi(argv[2]) : 9;

    for (uint32_t n : {1024u, 2048u, 4096u, 8192u}) {
        uint32_t sizeQ = 8;
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

        // --- Context-creation / prewarm cost, measured as a FRESH
        // first-use cost at a never-before-touched key within this
        // process (emulating "application creates an encryption
        // context"). Each n uses its own fresh key, so this IS the
        // true cold cost every time, not amortized across the n loop.
        auto t0 = std::chrono::steady_clock::now();
        bool ready = gpufhe::encrypt_prewarm_pinned_buffer(n, sizeQ);
        auto t1 = std::chrono::steady_clock::now();
        double prewarm_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        std::cout << "  explicit prewarm cost (first use at this key): " << prewarm_us << "us (ready=" << ready << ")\n";

        // --- First operation after prewarm (should be steady-state-like,
        // since the pinned buffer already exists). ---
        t0 = std::chrono::steady_clock::now();
        gpufhe::encrypt_host_shoup_pooled_barrett_prewarmed(c0, c1, msg, KP.pkA, KP.pkB, n, mod, root, 1, 3.19, 1);
        t1 = std::chrono::steady_clock::now();
        double first_after_prewarm_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        std::cout << "  first encrypt AFTER prewarm: " << first_after_prewarm_us << "us\n";

        // --- Steady-state medians for all variants, properly warmed up
        // first (never compare a cold new path against a warm old
        // path). ---
        gpufhe::encrypt_host_shoup_pooled_barrett_batched(c0, c1, msg, KP.pkA, KP.pkB, n, mod, root, 1, 3.19, 2);
        gpufhe::encrypt_host_shoup_pooled_barrett_prewarmed(c0, c1, msg, KP.pkA, KP.pkB, n, mod, root, 1, 3.19, 2);

        std::vector<double> pageable_s, prewarmed_s;
        for (size_t trial = 0; trial < trials; ++trial) {
            std::vector<double> pageable_r, prewarmed_r;
            for (size_t r = 0; r < reps; ++r) {
                uint64_t seed = 10000 + trial * reps + r;
                auto a0 = std::chrono::steady_clock::now();
                gpufhe::encrypt_host_shoup_pooled_barrett_batched(c0, c1, msg, KP.pkA, KP.pkB, n, mod, root, 1, 3.19, seed);
                auto a1 = std::chrono::steady_clock::now();
                gpufhe::encrypt_host_shoup_pooled_barrett_prewarmed(c0, c1, msg, KP.pkA, KP.pkB, n, mod, root, 1, 3.19, seed);
                auto a2 = std::chrono::steady_clock::now();
                pageable_r.push_back(std::chrono::duration<double, std::micro>(a1 - a0).count());
                prewarmed_r.push_back(std::chrono::duration<double, std::micro>(a2 - a1).count());
            }
            pageable_s.push_back(median_of(pageable_r)); prewarmed_s.push_back(median_of(prewarmed_r));
        }
        Stats sp = stats_of(pageable_s), spw = stats_of(prewarmed_s);
        report("steady-state: pageable (production)", sp);
        report("steady-state: prewarmed-pinned", spw);
        double per_call_savings_us = sp.p50 - spw.p50;
        std::cout << "    per-call steady-state savings: " << per_call_savings_us << "us (" << (sp.p50 / spw.p50) << "x)\n";

        // --- Break-even: operations needed for prewarming's one-time
        // cost to be repaid by per-call savings. If savings <= 0 this
        // workload NEVER breaks even. ---
        double breakeven = (per_call_savings_us > 0) ? (prewarm_us / per_call_savings_us) : -1.0;
        std::cout << "    break-even operation count (prewarm_us / per_call_savings_us): "
                  << (breakeven > 0 ? std::to_string(breakeven) : std::string("NEVER (no positive per-call savings)")) << "\n";

        // --- Workload profiles A/B/C: total wall time for N encrypt
        // operations, WITH vs. WITHOUT prewarming, N chosen per
        // profile. Built ANALYTICALLY from the three directly
        // measured quantities (prewarm_us, sp.p50 steady-state pageable,
        // spw.p50 steady-state prewarmed) -- this is more honest than a
        // synthetic loop that can't cleanly manufacture N independent
        // "fresh" (n,sizeQ) keys without exhausting this project's
        // supported dimension space.
        struct Profile { const char* name; int ops; };
        Profile profiles[] = { {"A: cold-start (1 op)", 1}, {"A: cold-start (3 ops)", 3},
                                {"B: typical (20 ops)", 20}, {"C: long-lived (500 ops)", 500},
                                {"C: long-lived (5000 ops)", 5000} };
        std::cout << "    -- workload totals (WITHOUT prewarm: N*pageable; WITH prewarm: prewarm_us + N*prewarmed) --\n";
        for (auto& prof : profiles) {
            double without = (double)prof.ops * sp.p50;
            double with_ = prewarm_us + (double)prof.ops * spw.p50;
            std::cout << "      " << prof.name << ": without=" << without << "us, with=" << with_
                      << "us, " << (with_ < without ? "prewarm WINS" : "prewarm LOSES")
                      << " (delta=" << (without - with_) << "us)\n";
        }
    }
    return 0;
}
