// GOD PROMPT "CKKS PHASE 18": per-call madvise(MADV_HUGEPAGE/
// NOHUGEPAGE) hint comparison across ALL four packed-buffer sizes,
// interleaved, multiple repeats. This is the ONE THP-related variable
// that can be controlled WITHOUT root (madvise on one's own mapped
// memory requires no special privilege) -- the system-wide
// /sys/kernel/mm/transparent_hugepage/enabled toggle itself could not
// be changed (permission denied, no passwordless sudo; see research/
// TRANSFER_PROFILING.md's Phase 18 section for the documented
// blocker). Pure Linux mmap+madvise+mlock, no CUDA.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>
#include <chrono>
#include <vector>
#include <algorithm>

using Clock = std::chrono::steady_clock;
static double us(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

static double one_call(size_t bytes, bool huge) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return -1;
    madvise(p, bytes, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    auto t0 = Clock::now();
    mlock(p, bytes);
    auto t1 = Clock::now();
    memset(p, 0, bytes);
    munlock(p, bytes);
    munmap(p, bytes);
    return us(t0, t1);
}

int main() {
    size_t sizes[] = {262144, 524288, 1048576, 2097152};
    const char* names[] = {"n=1024", "n=2048", "n=4096", "n=8192"};
    const int repeats = 8;

    for (int sidx = 0; sidx < 4; ++sidx) {
        std::vector<double> hugeV, nohugeV;
        // Interleaved: huge, nohuge, huge, nohuge, ...
        for (int r = 0; r < repeats; ++r) {
            hugeV.push_back(one_call(sizes[sidx], true));
            nohugeV.push_back(one_call(sizes[sidx], false));
        }
        auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        double hugeM = median(hugeV), nohugeM = median(nohugeV);
        printf("%s bytes=%zu: HUGEPAGE median=%.2fus  NOHUGEPAGE median=%.2fus  ratio(nohuge/huge)=%.3f\n",
               names[sidx], sizes[sidx], hugeM, nohugeM, nohugeM / hugeM);
        printf("  huge samples:   "); for (double v : hugeV) printf("%.1f ", v); printf("\n");
        printf("  nohuge samples: "); for (double v : nohugeV) printf("%.1f ", v); printf("\n");
    }
    return 0;
}
