// GOD PROMPT "CKKS PHASE 19": isolated, controlled comparison of every
// host-memory allocation/pinning path relevant to the variance Phases
// 16-18 investigated. Unlike those phases' diagnostics -- which each
// tested ONE path in isolation, and never cleanly separated CUDA
// context/runtime initialization from steady-state allocation cost --
// this file:
//   (a) times CUDA context initialization as its OWN, separate,
//       one-time stage (an explicit `cudaFree(0)` before anything
//       else is measured), so no other path's "first-use" number is
//       contaminated by context creation;
//   (b) compares FIVE allocation paths at identical sizes, with
//       identical touch/timing boundaries wherever the operations are
//       comparable:
//         A. malloc/free (pageable, page-aligned via posix_memalign)
//         B. malloc+touch+mlock/munlock (pure Linux, NO CUDA)
//         C. cudaMallocHost/cudaFreeHost
//         D. cudaHostAlloc(cudaHostAllocDefault)/cudaFreeHost -- flags
//            value recorded explicitly; CUDA's own documentation
//            states cudaMallocHost is a wrapper around cudaHostAlloc
//            with this exact flag, so C and D are EXPECTED to be
//            statistically indistinguishable if that documentation is
//            accurate -- this file tests that claim empirically
//            rather than assuming it.
//         D2. cudaHostAlloc(cudaHostAllocWriteCombined)/cudaFreeHost --
//            a documented alternate flag, to see whether flag choice
//            itself changes pinning cost.
//         E. malloc(page-aligned)+touch+cudaHostRegister(cudaHostRegisterDefault)
//            /cudaHostUnregister/free -- tests whether registering an
//            ALREADY-allocated, ALREADY-touched buffer avoids the
//            expensive part of cudaHostAlloc/cudaMallocHost, measuring
//            the FULL lifecycle (alloc+touch+register+unregister+free),
//            not just the registration call in isolation;
//   (c) after each CUDA-relevant path's allocation, calls
//       cudaPointerGetAttributes to record whether CUDA ACTUALLY
//       considers the resulting pointer pinned (and, for paths A/B,
//       to confirm CUDA does NOT recognize plain malloc/mlock memory
//       as pinned at all -- demonstrating paths A/B are not a valid
//       stand-in for C/D/E's semantics, a methodology gap in Phases
//       17/18's plain-mlock-vs-cudaMallocHost comparison);
//   (d) runs a warm-up phase, a separately-recorded first-use pass,
//       then multiple RANDOMIZED, INTERLEAVED steady-state rounds
//       (fixed-seed shuffle of every (path,size) pair, repeated),
//       plus a separate short-lived-vs-retained comparison for the
//       pinning paths (B/C/D/D2/E), keeping cumulative live locked
//       memory for any one path's retained test to the 4 packed sizes
//       together (~3.75MB), freed before the next path's retained
//       test begins -- always far under the 64MB RLIMIT_MEMLOCK cap;
//   (e) records every individual sample (not just aggregates) to a
//       CSV file (path given as argv[1], default
//       /tmp/bench_alloc_path_comparison.csv) with a /proc/meminfo
//       snapshot and wall-clock timestamp per sample, and prints
//       full min/median/mean/p90/p95/p99/max/stddev/CV stats per
//       (path,size) at the end.
// Does NOT call into any gpufhe:: code, any production cryptographic
// path, or any cache/pool. Every allocation stays far below the 64MB
// RLIMIT_MEMLOCK limit; no allocation is ever registered twice; no
// attempt is made to exceed any memory limit. Pure diagnostic.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <cuda_runtime.h>

using Clock = std::chrono::steady_clock;
static double us_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}
static long long wallclock_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
static long long meminfo_field(const char* name) {
    std::ifstream f("/proc/meminfo");
    std::string line;
    size_t nlen = strlen(name);
    while (std::getline(f, line)) {
        if (line.compare(0, nlen, name) == 0) {
            long long v = 0;
            sscanf(line.c_str() + nlen, "%lld", &v);
            return v;
        }
    }
    return -1;
}

enum class Path { A_Pageable, B_MlockPure, C_MallocHost, D_HostAllocDefault, D2_HostAllocWC, E_HostRegister };
static const char* path_name(Path p) {
    switch (p) {
        case Path::A_Pageable: return "A_pageable_malloc";
        case Path::B_MlockPure: return "B_mlock_pure_linux";
        case Path::C_MallocHost: return "C_cudaMallocHost";
        case Path::D_HostAllocDefault: return "D_cudaHostAlloc_default";
        case Path::D2_HostAllocWC: return "D2_cudaHostAlloc_writecombined";
        case Path::E_HostRegister: return "E_cudaHostRegister";
    }
    return "?";
}

struct Sample {
    long long wallclock_ms;
    std::string phase;   // warmup | firstuse | steadystate | retained
    Path path;
    size_t bytes;
    int sampleIndex;
    double alloc_us, touch_us, pin_us, unpin_us, free_us, total_us;
    long long memFree_kB, memAvail_kB, swapFree_kB;
    int cudaPtrType;      // cudaMemoryType if queried, -1 if not queried, -2 if query errored
    int errorCode;        // 0 = all CUDA calls in this sample succeeded
};

static std::vector<Sample> g_samples;

// Queries cudaPointerGetAttributes on `p` and returns the memoryType,
// or -2 if the query itself returns an error (expected for plain
// malloc/mlock memory CUDA has never registered). Always clears any
// resulting CUDA error so it cannot leak into the next timed call.
static int query_pointer_type(void* p) {
    cudaPointerAttributes attr;
    cudaError_t e = cudaPointerGetAttributes(&attr, p);
    cudaGetLastError(); // clear sticky error state unconditionally
    if (e != cudaSuccess) return -2;
    return (int)attr.type;
}

// One full lifecycle sample for one (path,bytes) pair: allocate,
// touch, pin/register (if applicable), [optionally stay retained --
// caller handles retention separately], unpin/unregister, free.
// `keepAlive`: if non-null, the allocated/pinned pointer and its
// cudaHostRegister-registration state are returned via out-params
// instead of being released, so the caller can retain it; the caller
// is then responsible for releasing it later via release_retained().
struct Retained { Path path; size_t bytes; void* ptr; bool cudaFreeHostNotPlainFree; bool wasRegistered; };

static void run_one(Path path, size_t bytes, const char* phase, int sampleIndex, Retained* keepAlive) {
    Sample s{};
    s.wallclock_ms = wallclock_ms();
    s.phase = phase;
    s.path = path;
    s.bytes = bytes;
    s.sampleIndex = sampleIndex;
    s.memFree_kB = meminfo_field("MemFree:");
    s.memAvail_kB = meminfo_field("MemAvailable:");
    s.swapFree_kB = meminfo_field("SwapFree:");
    s.cudaPtrType = -1;
    s.errorCode = 0;
    s.alloc_us = s.touch_us = s.pin_us = s.unpin_us = s.free_us = 0.0;

    auto t0 = Clock::now(), t1 = Clock::now();
    long pageSize = sysconf(_SC_PAGESIZE);

    switch (path) {
        case Path::A_Pageable: {
            void* p = nullptr;
            t0 = Clock::now();
            int rc = posix_memalign(&p, pageSize, bytes);
            t1 = Clock::now();
            s.alloc_us = us_between(t0, t1);
            if (rc != 0) { s.errorCode = rc; break; }
            t0 = Clock::now(); memset(p, 0xAB, bytes); t1 = Clock::now();
            s.touch_us = us_between(t0, t1);
            s.cudaPtrType = query_pointer_type(p);
            if (keepAlive) { keepAlive->ptr = p; keepAlive->cudaFreeHostNotPlainFree = false; keepAlive->wasRegistered = false; }
            else { t0 = Clock::now(); free(p); t1 = Clock::now(); s.free_us = us_between(t0, t1); }
            break;
        }
        case Path::B_MlockPure: {
            void* p = nullptr;
            t0 = Clock::now();
            int rc = posix_memalign(&p, pageSize, bytes);
            t1 = Clock::now();
            s.alloc_us = us_between(t0, t1);
            if (rc != 0) { s.errorCode = rc; break; }
            t0 = Clock::now(); memset(p, 0xAB, bytes); t1 = Clock::now();
            s.touch_us = us_between(t0, t1);
            t0 = Clock::now(); int mrc = mlock(p, bytes); t1 = Clock::now();
            s.pin_us = us_between(t0, t1);
            if (mrc != 0) { s.errorCode = errno; }
            s.cudaPtrType = query_pointer_type(p);
            if (keepAlive) { keepAlive->ptr = p; keepAlive->cudaFreeHostNotPlainFree = false; keepAlive->wasRegistered = false; }
            else {
                t0 = Clock::now(); munlock(p, bytes); free(p); t1 = Clock::now();
                s.unpin_us = us_between(t0, t1);
            }
            break;
        }
        case Path::C_MallocHost: {
            void* p = nullptr;
            t0 = Clock::now();
            cudaError_t e = cudaMallocHost(&p, bytes);
            t1 = Clock::now();
            s.alloc_us = us_between(t0, t1); // combined alloc+pin, matching the API's own atomicity
            if (e != cudaSuccess) { s.errorCode = (int)e; cudaGetLastError(); break; }
            t0 = Clock::now(); memset(p, 0xAB, bytes); t1 = Clock::now();
            s.touch_us = us_between(t0, t1);
            s.cudaPtrType = query_pointer_type(p);
            if (keepAlive) { keepAlive->ptr = p; keepAlive->cudaFreeHostNotPlainFree = true; keepAlive->wasRegistered = false; }
            else { t0 = Clock::now(); cudaFreeHost(p); t1 = Clock::now(); s.free_us = us_between(t0, t1); }
            break;
        }
        case Path::D_HostAllocDefault:
        case Path::D2_HostAllocWC: {
            unsigned flags = (path == Path::D_HostAllocDefault) ? cudaHostAllocDefault : cudaHostAllocWriteCombined;
            void* p = nullptr;
            t0 = Clock::now();
            cudaError_t e = cudaHostAlloc(&p, bytes, flags);
            t1 = Clock::now();
            s.alloc_us = us_between(t0, t1);
            if (e != cudaSuccess) { s.errorCode = (int)e; cudaGetLastError(); break; }
            t0 = Clock::now(); memset(p, 0xAB, bytes); t1 = Clock::now();
            s.touch_us = us_between(t0, t1);
            s.cudaPtrType = query_pointer_type(p);
            if (keepAlive) { keepAlive->ptr = p; keepAlive->cudaFreeHostNotPlainFree = true; keepAlive->wasRegistered = false; }
            else { t0 = Clock::now(); cudaFreeHost(p); t1 = Clock::now(); s.free_us = us_between(t0, t1); }
            break;
        }
        case Path::E_HostRegister: {
            void* p = nullptr;
            t0 = Clock::now();
            int rc = posix_memalign(&p, pageSize, bytes);
            t1 = Clock::now();
            s.alloc_us = us_between(t0, t1);
            if (rc != 0) { s.errorCode = rc; break; }
            t0 = Clock::now(); memset(p, 0xAB, bytes); t1 = Clock::now();
            s.touch_us = us_between(t0, t1);
            t0 = Clock::now();
            cudaError_t e = cudaHostRegister(p, bytes, cudaHostRegisterDefault);
            t1 = Clock::now();
            s.pin_us = us_between(t0, t1);
            if (e != cudaSuccess) { s.errorCode = (int)e; cudaGetLastError(); free(p); break; }
            s.cudaPtrType = query_pointer_type(p);
            if (keepAlive) { keepAlive->ptr = p; keepAlive->cudaFreeHostNotPlainFree = false; keepAlive->wasRegistered = true; }
            else {
                t0 = Clock::now(); cudaHostUnregister(p); free(p); t1 = Clock::now();
                s.unpin_us = us_between(t0, t1);
            }
            break;
        }
    }

    s.total_us = s.alloc_us + s.touch_us + s.pin_us + s.unpin_us + s.free_us;
    g_samples.push_back(s);
}

static void release_retained(const Retained& r) {
    if (r.wasRegistered) { cudaHostUnregister(r.ptr); free(r.ptr); }
    else if (r.cudaFreeHostNotPlainFree) { cudaFreeHost(r.ptr); }
    else { free(r.ptr); }
}

struct Stats { double min_, p50, p90, p95, p99, max_, mean, stddev, cv; size_t n; };
static Stats stats_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    Stats s{}; s.n = v.size();
    if (v.empty()) return s;
    s.min_ = v.front(); s.max_ = v.back();
    s.p50 = v[v.size() * 50 / 100];
    s.p90 = v[std::min(v.size() - 1, v.size() * 90 / 100)];
    s.p95 = v[std::min(v.size() - 1, v.size() * 95 / 100)];
    s.p99 = v.size() >= 20 ? v[std::min(v.size() - 1, v.size() * 99 / 100)] : s.max_; // p99 only meaningful at n>=20
    double sum = 0; for (double x : v) sum += x; s.mean = sum / v.size();
    double var = 0; for (double x : v) var += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(var / v.size());
    s.cv = s.mean > 0 ? s.stddev / s.mean : 0.0;
    return s;
}
static void report(const char* label, const Stats& s) {
    printf("    %-20s n=%4zu median=%9.2fus p90=%9.2fus p95=%9.2fus p99=%9.2fus min=%9.2fus max=%9.2fus mean=%9.2fus stddev=%9.2fus CV=%.3f\n",
           label, s.n, s.p50, s.p90, s.p95, s.p99, s.min_, s.max_, s.mean, s.stddev, s.cv);
}

int main(int argc, char** argv) {
    std::string csvPath = argc > 1 ? argv[1] : "/tmp/bench_alloc_path_comparison.csv";

    std::vector<size_t> sizes = {262144, 524288, 1048576, 2097152}; // n=1024,2048,4096,8192 at sizeQ=8
    std::vector<Path> pinningPaths = {Path::B_MlockPure, Path::C_MallocHost, Path::D_HostAllocDefault, Path::D2_HostAllocWC, Path::E_HostRegister};
    std::vector<Path> allPaths = {Path::A_Pageable, Path::B_MlockPure, Path::C_MallocHost, Path::D_HostAllocDefault, Path::D2_HostAllocWC, Path::E_HostRegister};

    // --- Stage 0: CUDA context initialization, timed and recorded
    // as its OWN sample so it never leaks into any path's "first-use"
    // number. cudaFree(0) is the standard idiom to force lazy context
    // creation without allocating any real memory.
    {
        auto t0 = Clock::now();
        cudaError_t e = cudaFree(0);
        auto t1 = Clock::now();
        double us = us_between(t0, t1);
        printf("=== Stage 0: CUDA context initialization (cudaFree(0)) ===\n");
        printf("  cudaFree(0): %.2fus (%s)\n", us, e == cudaSuccess ? "ok" : cudaGetErrorString(e));
        cudaGetLastError();
        Sample s{}; s.wallclock_ms = wallclock_ms(); s.phase = "warmup"; s.path = Path::C_MallocHost;
        s.bytes = 0; s.sampleIndex = 0; s.alloc_us = us; s.total_us = us; s.cudaPtrType = -1; s.errorCode = (int)e;
        s.memFree_kB = meminfo_field("MemFree:"); s.memAvail_kB = meminfo_field("MemAvailable:"); s.swapFree_kB = meminfo_field("SwapFree:");
        g_samples.push_back(s);
    }

    // --- Stage 1: first-use pass. For every path, at every size,
    // exactly ONE sample, recorded with phase="firstuse", run in a
    // FIXED (not randomized) order so results are reproducible and
    // comparable across repeated process runs.
    printf("\n=== Stage 1: first-use pass (one sample per path per size, fixed order) ===\n");
    for (Path p : allPaths)
        for (size_t sz : sizes)
            run_one(p, sz, "firstuse", 1, nullptr);
    printf("  done (%zu samples)\n", allPaths.size() * sizes.size());

    // --- Stage 2: randomized, interleaved steady-state rounds.
    // Build the full (path,size) cross product, replicate it ROUNDS
    // times, shuffle with a FIXED seed (reproducible across process
    // runs), then execute in that shuffled order -- this is what
    // actually eliminates ordering/position confounds, unlike a
    // sequential per-path block.
    const int ROUNDS = 10;
    std::vector<std::pair<Path, size_t>> plan;
    for (int r = 0; r < ROUNDS; ++r)
        for (Path p : allPaths)
            for (size_t sz : sizes)
                plan.push_back({p, sz});
    std::mt19937_64 rng(1234567);
    std::shuffle(plan.begin(), plan.end(), rng);

    printf("\n=== Stage 2: randomized interleaved steady-state (%d rounds x %zu paths x %zu sizes = %zu samples, fixed seed 1234567) ===\n",
           ROUNDS, allPaths.size(), sizes.size(), plan.size());
    std::vector<int> perPathSizeCounter(allPaths.size() * sizes.size(), 0);
    for (auto& ps : plan) {
        run_one(ps.first, ps.second, "steadystate", (int)g_samples.size(), nullptr);
    }
    printf("  done (%zu samples)\n", plan.size());

    // --- Stage 3: repeated alloc/free cycles per path, tight loop,
    // process stays alive (cycle-to-cycle consistency within one
    // path/size, no interleaving) -- 20 repeats at the largest size
    // (2MiB), where Phase 17/18's own variance was most pronounced.
    printf("\n=== Stage 3: repeated alloc/free cycles, 2MiB, 20 repeats per path, same process (no interleaving) ===\n");
    for (Path p : allPaths)
        for (int i = 0; i < 20; ++i)
            run_one(p, 2097152, "steadystate_cycle", i, nullptr);
    printf("  done\n");

    // --- Stage 4: short-lived vs. deliberately RETAINED comparison
    // for the pinning paths only. For each pinning path, allocate all
    // 4 sizes and keep them alive SIMULTANEOUSLY (~3.75MB total,
    // never close to the 64MB RLIMIT_MEMLOCK cap), then release all
    // 4 before moving to the next path -- so cumulative live locked
    // memory from this stage never exceeds ~3.75MB at any instant.
    printf("\n=== Stage 4: retained (never-freed-until-end) vs short-lived comparison, pinning paths only ===\n");
    long long memlockLimitCheckKB = 65536; // RLIMIT_MEMLOCK soft limit, informational only
    printf("  (RLIMIT_MEMLOCK soft limit on this system: %lldKB; this stage's peak per-path retained total: %zuKB, run path-by-path, never summed across paths)\n",
           memlockLimitCheckKB, (262144+524288+1048576+2097152)/1024);
    for (Path p : pinningPaths) {
        std::vector<Retained> retained;
        for (size_t sz : sizes) {
            Retained r{};
            run_one(p, sz, "retained", (int)retained.size(), &r);
            retained.push_back(r);
        }
        // Also run the short-lived equivalent immediately after, same
        // path/sizes, for direct A/B comparison under the SAME
        // process state (retained buffers still live during this).
        for (size_t sz : sizes)
            run_one(p, sz, "shortlived_while_others_retained", 0, nullptr);
        for (auto& r : retained) release_retained(r);
    }
    printf("  done\n");

    // --- Write raw CSV ---
    {
        std::ofstream csv(csvPath);
        csv << "wallclock_ms,phase,path,bytes,sample_index,alloc_us,touch_us,pin_us,unpin_us,free_us,total_us,memfree_kb,memavail_kb,swapfree_kb,cuda_ptr_type,error_code\n";
        for (auto& s : g_samples) {
            csv << s.wallclock_ms << "," << s.phase << "," << path_name(s.path) << "," << s.bytes << ","
                << s.sampleIndex << "," << s.alloc_us << "," << s.touch_us << "," << s.pin_us << ","
                << s.unpin_us << "," << s.free_us << "," << s.total_us << "," << s.memFree_kB << ","
                << s.memAvail_kB << "," << s.swapFree_kB << "," << s.cudaPtrType << "," << s.errorCode << "\n";
        }
        printf("\nRaw CSV (%zu samples) written to: %s\n", g_samples.size(), csvPath.c_str());
    }

    // --- Aggregate stats: steadystate-phase samples only, by (path,size) ---
    printf("\n=== Aggregated steady-state stats by (path,size), total_us ===\n");
    for (Path p : allPaths) {
        for (size_t sz : sizes) {
            std::vector<double> totals;
            int ptrType = -99; int firstErr = 0;
            for (auto& s : g_samples) {
                if (s.path == p && s.bytes == sz && (s.phase == "steadystate" || s.phase == "steadystate_cycle")) {
                    totals.push_back(s.total_us);
                    if (ptrType == -99) ptrType = s.cudaPtrType;
                    if (s.errorCode != 0 && firstErr == 0) firstErr = s.errorCode;
                }
            }
            if (totals.empty()) continue;
            printf("  %s bytes=%zu cudaPointerGetAttributes_type=%d first_error=%d\n", path_name(p), sz, ptrType, firstErr);
            report("total", stats_of(totals));
        }
    }

    // --- First-use vs steady-state median comparison ---
    printf("\n=== First-use vs steady-state median total_us (same path/size) ===\n");
    for (Path p : allPaths) {
        for (size_t sz : sizes) {
            double firstUse = -1;
            std::vector<double> steady;
            for (auto& s : g_samples) {
                if (s.path == p && s.bytes == sz) {
                    if (s.phase == "firstuse") firstUse = s.total_us;
                    if (s.phase == "steadystate") steady.push_back(s.total_us);
                }
            }
            if (steady.empty()) continue;
            std::sort(steady.begin(), steady.end());
            double steadyMedian = steady[steady.size() / 2];
            printf("  %s bytes=%zu: first-use=%.2fus steady-state-median=%.2fus ratio(firstuse/steady)=%.3f\n",
                   path_name(p), sz, firstUse, steadyMedian, firstUse / steadyMedian);
        }
    }

    // --- Retained vs short-lived comparison ---
    printf("\n=== Retained vs short-lived total_us, by path (2MiB shown; see CSV for all sizes) ===\n");
    for (Path p : pinningPaths) {
        double retainedUs = -1, shortLivedUs = -1;
        for (auto& s : g_samples) {
            if (s.path == p && s.bytes == 2097152) {
                if (s.phase == "retained") retainedUs = s.total_us;
                if (s.phase == "shortlived_while_others_retained") shortLivedUs = s.total_us;
            }
        }
        printf("  %s: retained=%.2fus short-lived(with-others-retained)=%.2fus\n", path_name(p), retainedUs, shortLivedUs);
    }

    return 0;
}
