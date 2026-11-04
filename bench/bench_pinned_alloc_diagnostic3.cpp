// GOD PROMPT "CKKS PHASE 18": enhanced per-sample diagnostic for the
// pinned-memory allocation-cost variance investigation. This file
// does NOT toggle any system/THP setting -- Phase 18 found runtime
// THP control requires root (sudo with a password), which was not
// available non-interactively, and per the governing directive's
// explicit safety instruction ("if the runtime setting cannot be
// changed and restored safely, stop without changing it and document
// why"), no attempt was made to work around that. This diagnostic
// instead investigates every OTHER variable Step 5 asks about
// (allocation size, order, cumulative live memory, repeated alloc/
// release, memory pressure, first-touch, pool reuse) under the
// CURRENT, UNCHANGED THP=madvise setting, with full per-sample system
// state recorded (directive Step 4): timestamp, allocation size,
// order, live-allocation count, /proc/meminfo snapshot, and separately
// timed alloc/first-touch/release latencies. Does NOT touch `ulimit
// -l` (kept as a separate, untouched confounding variable per Step 5's
// explicit instruction). No CUDA, no production code -- pure
// diagnostic, read-only with respect to system configuration.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
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

using Clock = std::chrono::steady_clock;
static double us_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}
static long long wallclock_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// Reads one numeric field from /proc/meminfo (in kB, as reported).
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

struct Sample {
    long long wallclock_ms;
    uint32_t allocIndex;     // position in this run's sequence
    size_t bytes;
    size_t liveBefore;       // how many OTHER allocations already live
    long long memFree_kB, memAvailable_kB, anonHuge_kB, swapFree_kB;
    double mlock_us, firsttouch_us, munlock_us;
};

static void print_sample(const Sample& s) {
    printf("t=%lld idx=%u bytes=%zu liveBefore=%zu MemFree=%lldkB MemAvail=%lldkB AnonHuge=%lldkB SwapFree=%lldkB "
           "mlock=%.2fus firsttouch=%.2fus munlock=%.2fus\n",
           s.wallclock_ms, s.allocIndex, s.bytes, s.liveBefore,
           s.memFree_kB, s.memAvailable_kB, s.anonHuge_kB, s.swapFree_kB,
           s.mlock_us, s.firsttouch_us, s.munlock_us);
}

struct Stats { double min_, p50, p90, p95, max_, mean, stddev; size_t n; };
static Stats stats_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    Stats s; s.n = v.size();
    s.min_ = v.front(); s.max_ = v.back();
    s.p50 = v[v.size() * 50 / 100];
    s.p90 = v[std::min(v.size() - 1, v.size() * 90 / 100)];
    s.p95 = v[std::min(v.size() - 1, v.size() * 95 / 100)];
    double sum = 0; for (double x : v) sum += x; s.mean = sum / v.size();
    double var = 0; for (double x : v) var += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(var / v.size());
    return s;
}
static void report(const char* label, const Stats& s) {
    printf("    %-24s median=%9.2fus p90=%9.2fus p95=%9.2fus min=%9.2fus max=%9.2fus mean=%9.2fus stddev=%9.2fus (n=%zu)\n",
           label, s.p50, s.p90, s.p95, s.min_, s.max_, s.mean, s.stddev, s.n);
}

int main(int argc, char** argv) {
    std::string order = argc > 1 ? argv[1] : "asc"; // asc | desc | rand
    bool neverFree = argc > 2 && std::string(argv[2]) == "neverfree";
    int repeatsPerSize = argc > 3 ? std::atoi(argv[3]) : 3;

    std::vector<size_t> packedSizes = {262144, 524288, 1048576, 2097152}; // n=1024,2048,4096,8192 at sizeQ=8
    std::vector<size_t> sizes;
    for (int r = 0; r < repeatsPerSize; ++r)
        for (size_t sz : packedSizes) sizes.push_back(sz);
    if (order == "desc") std::reverse(sizes.begin(), sizes.end());
    else if (order == "rand") {
        std::mt19937_64 rng(12345);
        std::shuffle(sizes.begin(), sizes.end(), rng);
    }

    printf("THP enabled (read-only, not modified): ");
    { std::ifstream f("/sys/kernel/mm/transparent_hugepage/enabled"); std::string l; std::getline(f, l); printf("%s\n", l.c_str()); }
    printf("order=%s neverFree=%d repeatsPerSize=%d total_samples=%zu\n", order.c_str(), (int)neverFree, repeatsPerSize, sizes.size());

    std::vector<void*> live;
    std::vector<size_t> liveSizes;
    std::vector<Sample> samples;

    uint32_t idx = 0;
    for (size_t bytes : sizes) {
        ++idx;
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { printf("mmap FAILED at idx=%u\n", idx); continue; }

        Sample s{};
        s.wallclock_ms = wallclock_ms();
        s.allocIndex = idx;
        s.bytes = bytes;
        s.liveBefore = live.size();
        s.memFree_kB = meminfo_field("MemFree:");
        s.memAvailable_kB = meminfo_field("MemAvailable:");
        s.anonHuge_kB = meminfo_field("AnonHugePages:");
        s.swapFree_kB = meminfo_field("SwapFree:");

        auto t0 = Clock::now();
        int rc = mlock(p, bytes);
        auto t1 = Clock::now();
        s.mlock_us = us_between(t0, t1);
        if (rc != 0) { printf("mlock FAILED at idx=%u (errno-based rc=%d)\n", idx, rc); munmap(p, bytes); continue; }

        t0 = Clock::now();
        memset(p, 0xAB, bytes);
        t1 = Clock::now();
        s.firsttouch_us = us_between(t0, t1);

        if (neverFree) {
            s.munlock_us = -1.0; // not released this call -- released at end
            live.push_back(p); liveSizes.push_back(bytes);
        } else {
            t0 = Clock::now();
            munlock(p, bytes);
            munmap(p, bytes);
            t1 = Clock::now();
            s.munlock_us = us_between(t0, t1);
        }

        print_sample(s);
        samples.push_back(s);
    }

    // release anything kept alive
    for (size_t i = 0; i < live.size(); ++i) { munlock(live[i], liveSizes[i]); munmap(live[i], liveSizes[i]); }

    // Aggregate by size.
    printf("\n-- aggregated by size (all orders/positions pooled) --\n");
    for (size_t sz : packedSizes) {
        std::vector<double> mlock_v, touch_v;
        for (auto& s : samples) if (s.bytes == sz) { mlock_v.push_back(s.mlock_us); touch_v.push_back(s.firsttouch_us); }
        if (mlock_v.empty()) continue;
        printf("  bytes=%zu (n=%zu samples)\n", sz, mlock_v.size());
        report("mlock", stats_of(mlock_v));
        report("first-touch", stats_of(touch_v));
    }

    // Aggregate by liveBefore bucket (0, 1-3, 4-7, 8+) to see if
    // cumulative live memory correlates with latency independent of size.
    printf("\n-- aggregated by cumulative live-allocation count BEFORE this call --\n");
    struct Bucket { const char* label; size_t lo, hi; };
    Bucket buckets[] = { {"0 live", 0, 0}, {"1-3 live", 1, 3}, {"4-7 live", 4, 7}, {"8+ live", 8, 1000000} };
    for (auto& b : buckets) {
        std::vector<double> mlock_v;
        for (auto& s : samples) if (s.liveBefore >= b.lo && s.liveBefore <= b.hi) mlock_v.push_back(s.mlock_us);
        if (mlock_v.empty()) continue;
        printf("  %s (n=%zu samples)\n", b.label, mlock_v.size());
        report("mlock", stats_of(mlock_v));
    }

    return 0;
}
