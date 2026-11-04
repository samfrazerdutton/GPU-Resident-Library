// GOD PROMPT "CKKS PHASE 17": pure-Linux (NO CUDA) diagnostic isolating
// whether Transparent Huge Page (THP) promotion specifically explains
// the pattern bench_pinned_alloc_diagnostic2.cpp found (n=8192's exact
// 2MiB packed-buffer size consistently slow, independent of call order
// or position, while 1MiB/256KB/512KB sizes were not). This file never
// touches CUDA -- it uses raw mmap + madvise(MADV_HUGEPAGE) + mlock,
// mirroring what a pinned-memory allocator plausibly does under the
// hood, to separate "is this a Linux/THP-level effect" (testable
// without any NVIDIA driver involvement) from "is this specific to
// CUDA's own allocator" (Hypothesis C vs Hypothesis D in research/
// TRANSFER_PROFILING.md's Phase 17 section).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <string>

using Clock = std::chrono::steady_clock;
static double us_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

// For each size: mmap (anonymous, private), then EITHER madvise(MADV_HUGEPAGE)
// or madvise(MADV_NOHUGEPAGE) (controlled by `hugeHint`), then mlock, then
// first-touch (memset), each timed separately.
static void test_size(size_t bytes, bool hugeHint, const char* label) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { printf("    mmap FAILED\n"); return; }

    auto t0 = Clock::now();
    int madv_rc = madvise(p, bytes, hugeHint ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    auto t1 = Clock::now();
    double madvise_us = us_between(t0, t1);

    t0 = Clock::now();
    int mlock_rc = mlock(p, bytes);
    t1 = Clock::now();
    double mlock_us = us_between(t0, t1);

    t0 = Clock::now();
    memset(p, 0xAB, bytes); // first-touch -- this is where page faults / THP promotion actually happen
    t1 = Clock::now();
    double firsttouch_us = us_between(t0, t1);

    printf("  %s bytes=%zu hugeHint=%d: madvise=%.2fus(rc=%d) mlock=%.2fus(rc=%d) first-touch=%.2fus\n",
           label, bytes, (int)hugeHint, madvise_us, madv_rc, mlock_us, mlock_rc, firsttouch_us);

    munlock(p, bytes);
    munmap(p, bytes);
}

int main() {
    // Mirror the exact packed-buffer sizes from Phase 16/17 (4*sizeQ*n*8, sizeQ=8).
    struct Sz { uint32_t n; size_t bytes; };
    std::vector<Sz> sizes = { {1024, 262144}, {2048, 524288}, {4096, 1048576}, {8192, 2097152} };

    printf("=== Pass 1: madvise(MADV_HUGEPAGE) requested for every size ===\n");
    for (auto& s : sizes) test_size(s.bytes, true, ("n=" + std::to_string(s.n)).c_str());

    printf("\n=== Pass 2: madvise(MADV_NOHUGEPAGE) requested for every size (THP explicitly disabled) ===\n");
    for (auto& s : sizes) test_size(s.bytes, false, ("n=" + std::to_string(s.n)).c_str());

    printf("\n=== Pass 3: repeat pass 1 (madvise HUGEPAGE), to check reproducibility within this process ===\n");
    for (auto& s : sizes) test_size(s.bytes, true, ("n=" + std::to_string(s.n)).c_str());

    printf("\n=== Pass 4: exactly 2MiB at SEVERAL different alignments/order, with and without the hint ===\n");
    test_size(2097152, true, "2MiB-again-A");
    test_size(1048576, true, "1MiB-again-A");
    test_size(2097152, false, "2MiB-nohuge-B");
    test_size(2097152, true, "2MiB-again-C");
    return 0;
}
