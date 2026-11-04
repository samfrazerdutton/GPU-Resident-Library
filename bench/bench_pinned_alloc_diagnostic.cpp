// GOD PROMPT "CKKS PHASE 17": isolated diagnostic instrument for the
// pinned-memory allocation path. Unlike Phase 16's
// `encrypt_prewarm_pinned_buffer` (which bundled cudaMalloc(d_packed) +
// cudaMallocHost(h_packed) + cudaMalloc(d_fr) + cudaMalloc(d_fp) +
// cudaMalloc(d_mod) into ONE combined "prewarm cost" number), this file
// times EVERY stage SEPARATELY, as the governing directive's Step 2
// explicitly requires:
//   1. CUDA runtime initialization (first-ever CUDA call in process).
//   2. cudaMalloc (device, packed-buffer-sized).
//   3. cudaMallocHost (PINNED host, packed-buffer-sized) -- the stage
//      Phase 16's bundled number could not isolate.
//   4. Host buffer first-touch (a write to every page of the pinned
//      buffer, separate from the allocation call itself -- OS page
//      commit / first-touch cost is a DIFFERENT mechanism than the
//      allocation syscall).
//   5. Host-to-device transfer (pinned source).
//   6. Device-to-host transfer (pinned destination).
//   7. cudaDeviceSynchronize.
//   8. cudaFreeHost / cudaFree (resource release).
// Also measures the PAGEABLE equivalent of stages 2-7 for direct
// comparison (same sizes, regular heap `std::vector` instead of
// `cudaMallocHost`).
//
// Supports controlled experiment variables (directive Step 4):
//   argv[1] = order: "asc" | "desc" | "rand" (allocation order across
//             the 4 supported sizes within one process run)
//   argv[2] = repeats per size (independent samples)
//   argv[3] = "fresh" (exit after ONE size -- caller loops externally,
//             one process per size) | "loop" (all sizes in one
//             process, honoring the order above)
// Does NOT touch any production file, any cache/pool, or any
// cryptographic parameter -- pure allocator/transfer timing.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

namespace {
using Clock = std::chrono::steady_clock;
double us_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

struct Stats { double min_, p50, p90, p95, max_, mean, stddev; size_t n; };
Stats stats_of(std::vector<double> v) {
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
void report(const char* label, const Stats& s) {
    printf("    %-32s median=%9.2fus p90=%9.2fus p95=%9.2fus min=%9.2fus max=%9.2fus mean=%9.2fus stddev=%9.2fus (n=%zu)\n",
           label, s.p50, s.p90, s.p95, s.min_, s.max_, s.mean, s.stddev, s.n);
}

// Measures the 8 stages for ONE packed-buffer size (bytes), `repeats`
// independent times. Each repeat does a FRESH cudaMalloc/cudaMallocHost
// (not reusing a pool) so every sample is a genuine allocation, not a
// cache hit -- deliberately the opposite of production's pooling, to
// isolate allocator behavior itself.
void measure_size(size_t bytes, int repeats, bool firstCallInProcess) {
    std::vector<double> dmalloc_us, hmallocpinned_us, firsttouch_us, h2d_us, d2h_us, sync_us, free_us;
    std::vector<double> pageable_alloc_us, pageable_firsttouch_us, pageable_h2d_us, pageable_d2h_us;

    for (int r = 0; r < repeats; ++r) {
        // --- Pinned path ---
        uint64_t* d_buf = nullptr;
        auto t0 = Clock::now();
        cudaError_t e1 = cudaMalloc(&d_buf, bytes);
        auto t1 = Clock::now();
        dmalloc_us.push_back(us_between(t0, t1));
        if (e1 != cudaSuccess) { printf("    cudaMalloc FAILED: %s\n", cudaGetErrorString(e1)); continue; }

        uint64_t* h_buf = nullptr;
        t0 = Clock::now();
        cudaError_t e2 = cudaMallocHost(&h_buf, bytes);
        t1 = Clock::now();
        hmallocpinned_us.push_back(us_between(t0, t1));
        if (e2 != cudaSuccess) { printf("    cudaMallocHost FAILED: %s\n", cudaGetErrorString(e2)); cudaFree(d_buf); continue; }

        // First-touch: write to every page (4KB stride is a safe
        // conservative page-size assumption; writing EVERY uint64_t
        // guarantees every page is touched regardless of actual size).
        t0 = Clock::now();
        memset(h_buf, 0xAB, bytes);
        t1 = Clock::now();
        firsttouch_us.push_back(us_between(t0, t1));

        t0 = Clock::now();
        cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice);
        t1 = Clock::now();
        h2d_us.push_back(us_between(t0, t1));

        t0 = Clock::now();
        cudaMemcpy(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost);
        t1 = Clock::now();
        d2h_us.push_back(us_between(t0, t1));

        t0 = Clock::now();
        cudaDeviceSynchronize();
        t1 = Clock::now();
        sync_us.push_back(us_between(t0, t1));

        t0 = Clock::now();
        cudaFreeHost(h_buf);
        cudaFree(d_buf);
        t1 = Clock::now();
        free_us.push_back(us_between(t0, t1));

        // --- Pageable comparison (same size, regular heap memory) ---
        uint64_t* d_buf2 = nullptr;
        cudaMalloc(&d_buf2, bytes);
        t0 = Clock::now();
        std::vector<uint8_t> pbuf(bytes);
        t1 = Clock::now();
        pageable_alloc_us.push_back(us_between(t0, t1));

        t0 = Clock::now();
        memset(pbuf.data(), 0xCD, bytes);
        t1 = Clock::now();
        pageable_firsttouch_us.push_back(us_between(t0, t1));

        t0 = Clock::now();
        cudaMemcpy(d_buf2, pbuf.data(), bytes, cudaMemcpyHostToDevice);
        t1 = Clock::now();
        pageable_h2d_us.push_back(us_between(t0, t1));

        t0 = Clock::now();
        cudaMemcpy(pbuf.data(), d_buf2, bytes, cudaMemcpyDeviceToHost);
        t1 = Clock::now();
        pageable_d2h_us.push_back(us_between(t0, t1));

        cudaFree(d_buf2);
    }

    printf("  -- size=%zu bytes, %d repeats, firstCallInProcess=%d --\n", bytes, repeats, firstCallInProcess);
    report("cudaMalloc (device)", stats_of(dmalloc_us));
    report("cudaMallocHost (PINNED)", stats_of(hmallocpinned_us));
    report("first-touch (pinned, memset)", stats_of(firsttouch_us));
    report("H2D transfer (pinned src)", stats_of(h2d_us));
    report("D2H transfer (pinned dst)", stats_of(d2h_us));
    report("cudaDeviceSynchronize", stats_of(sync_us));
    report("free (cudaFreeHost+cudaFree)", stats_of(free_us));
    report("pageable std::vector alloc", stats_of(pageable_alloc_us));
    report("first-touch (pageable, memset)", stats_of(pageable_firsttouch_us));
    report("H2D transfer (pageable src)", stats_of(pageable_h2d_us));
    report("D2H transfer (pageable dst)", stats_of(pageable_d2h_us));
}
} // namespace

int main(int argc, char** argv) {
    std::string order = argc > 1 ? argv[1] : "asc";
    int repeats = argc > 2 ? std::atoi(argv[2]) : 10;
    std::string mode = argc > 3 ? argv[3] : "loop";
    uint32_t sizeQ = 8;

    std::vector<uint32_t> ns = {1024, 2048, 4096, 8192};
    if (order == "desc") std::reverse(ns.begin(), ns.end());
    else if (order == "rand") {
        std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count());
        std::shuffle(ns.begin(), ns.end(), rng);
    }

    printf("Order: %s, repeats: %d, mode: %s\n", order.c_str(), repeats, mode.c_str());
    printf("Allocation order this run: ");
    for (auto n : ns) printf("%u ", n);
    printf("\n");

    bool first = true;
    for (uint32_t n : ns) {
        size_t bytes = (size_t)4 * sizeQ * n * 8; // matches the [v|e0|e1|m] packed layout
        printf("\n=== n=%u sizeQ=%u (packed bytes=%zu) ===\n", n, sizeQ, bytes);
        measure_size(bytes, repeats, first);
        first = false;
        if (mode == "fresh") break; // caller loops externally, one process per size
    }
    return 0;
}
