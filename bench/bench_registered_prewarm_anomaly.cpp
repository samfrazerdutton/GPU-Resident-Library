// GOD PROMPT "CKKS PHASE 22": controlled investigation of Phase 20's
// explicit-prewarming anomaly (explicitly prewarmed buffers measured
// consistently slightly SLOWER than lazily-created ones in steady
// state, every one of 3 runs). This file removes a confound Phase 20's
// own benchmark had: Phase 20's "D" (lazy) and "E" (prewarmed)
// configurations used DIFFERENT (n,sizeQ) keys (sizeQ vs sizeQ+1, "one
// extra tower, ~12.5% more arithmetic... a small, explicitly-
// acknowledged confound" per that phase's own comment) because the
// SAME process cannot observe "lazy" and "prewarmed" behavior for the
// IDENTICAL key (the pool is keyed by (n,sizeQ); whichever condition
// runs first in a process determines how that key's entry gets
// created, and the pool's own code (src/encrypt_host_shoup_pooled_
// barrett_batched_registered.cpp's encrypt_buf_pool_registered_locked)
// cannot distinguish "reached ready via explicit prewarm" from
// "reached ready via lazy first-use" after the fact -- both paths
// produce the IDENTICAL EncryptBufPoolRegistered entry, with no flag
// recording which happened). This file instead runs "lazy" and
// "prewarm" as SEPARATE FRESH PROCESSES, both using the EXACT SAME
// (n,sizeQ) key, eliminating the sizeQ+1 confound entirely.
//
// Modes (argv[1]):
//   lazy         -- first encrypt call does registration+encrypt
//                    combined (timed as one "first_call" number);
//                    then STEADY_STATE_REPS more calls, individually
//                    timed.
//   prewarm       -- explicit prewarm (timed separately), then
//                    immediately the first encrypt call (timed
//                    separately), then STEADY_STATE_REPS more calls.
//   prewarm_delay -- explicit prewarm, a real 200ms sleep, THEN the
//                    first encrypt call and steady-state reps --
//                    tests whether a delay after prewarm (letting any
//                    transient driver/cache state settle) changes
//                    anything.
// Every stage's raw sample is printed; run this binary 3x per mode (a
// fresh process each time) for independent-repetition reproducibility,
// exactly as Phase 19/20's own diagnostics did.
#include "keygen.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cmath>

namespace gpufhe {
void encrypt_host_shoup_pooled_barrett_batched_registered(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                                                             const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                                                             const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
bool encrypt_prewarm_registered_buffer(uint32_t n, uint32_t sizeQ);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
}

using Clock = std::chrono::steady_clock;
static double us_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

struct Fixture {
    uint32_t n, sizeQ;
    std::vector<uint64_t> mod, root, pkA, pkB;
    std::vector<int64_t> m;
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
    F.m.resize(n); for (uint32_t k = 0; k < n; ++k) F.m[k] = (int64_t)((k % 31) - 15);
    return F;
}
static double time_encrypt(const Fixture& F, uint64_t seed) {
    std::vector<uint64_t> c0, c1;
    auto t0 = Clock::now();
    gpufhe::encrypt_host_shoup_pooled_barrett_batched_registered(c0, c1, F.m, F.pkA, F.pkB, F.n, F.mod, F.root, 1, 3.19, seed);
    auto t1 = Clock::now();
    return us_between(t0, t1);
}

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "lazy";
    uint32_t n = argc > 2 ? (uint32_t)atoi(argv[2]) : 8192;
    uint32_t sizeQ = argc > 3 ? (uint32_t)atoi(argv[3]) : 8;
    const int STEADY_STATE_REPS = 20;

    Fixture F = make_fixture(n, sizeQ);
    printf("mode=%s n=%u sizeQ=%u\n", mode.c_str(), n, sizeQ);

    if (mode == "prewarm" || mode == "prewarm_delay") {
        auto t0 = Clock::now();
        bool ready = gpufhe::encrypt_prewarm_registered_buffer(n, sizeQ);
        auto t1 = Clock::now();
        printf("STAGE prewarm_cost_us=%.2f ready=%d\n", us_between(t0, t1), (int)ready);
        if (mode == "prewarm_delay") {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            printf("STAGE delay_ms=200\n");
        }
    }

    double firstCall = time_encrypt(F, 1);
    printf("STAGE first_call_us=%.2f\n", firstCall);

    std::vector<double> steady;
    for (int i = 0; i < STEADY_STATE_REPS; ++i) {
        double us = time_encrypt(F, 1000 + i);
        steady.push_back(us);
        printf("SAMPLE steady_state[%d]_us=%.2f\n", i, us);
    }
    std::sort(steady.begin(), steady.end());
    double median = steady[steady.size() / 2];
    double sum = 0; for (double v : steady) sum += v;
    double mean = sum / steady.size();
    double var = 0; for (double v : steady) var += (v - mean) * (v - mean);
    double stddev = std::sqrt(var / steady.size());
    printf("STAGE steady_state_median_us=%.2f mean_us=%.2f stddev_us=%.2f min_us=%.2f max_us=%.2f n=%zu\n",
           median, mean, stddev, steady.front(), steady.back(), steady.size());
    return 0;
}
