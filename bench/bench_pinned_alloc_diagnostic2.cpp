// GOD PROMPT "CKKS PHASE 17": follow-up diagnostic testing a specific
// hypothesis the first diagnostic's results raised: cudaMallocHost's
// cost may depend on whether at least one OTHER pinned allocation is
// ALREADY LIVE in the process (matching production's actual behavior
// -- pinned buffers are never freed, process-lifetime pool), rather
// than on call order or size alone. This file NEVER frees pinned
// buffers between allocations (mirroring production), and reports
// each cudaMallocHost call's own isolated time plus how many OTHER
// pinned allocations were already live at that moment.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    uint32_t sizeQ = 8;
    std::vector<uint32_t> ns = {1024, 2048, 4096, 8192};
    if (argc > 1 && std::string(argv[1]) == "desc") std::reverse(ns.begin(), ns.end());

    std::vector<uint64_t*> livePinned; // never freed -- mirrors production
    int callIndex = 0;
    for (uint32_t n : ns) {
        size_t bytes = (size_t)4 * sizeQ * n * 8;
        ++callIndex;
        uint64_t* h_buf = nullptr;
        auto t0 = std::chrono::steady_clock::now();
        cudaError_t e = cudaMallocHost(&h_buf, bytes);
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        printf("call #%d: n=%u bytes=%zu live-pinned-allocs-BEFORE-this-call=%zu cudaMallocHost=%.2fus (%s)\n",
               callIndex, n, bytes, livePinned.size(), us, e == cudaSuccess ? "ok" : cudaGetErrorString(e));
        if (e == cudaSuccess) livePinned.push_back(h_buf); // KEEP ALIVE -- never free
    }

    printf("\n--- second pass: re-request the SAME sizes again (all 4 keys already have a live pinned alloc from pass 1, but this is a NEW allocation -- different pointer, same size, to see if sizes themselves are now 'cheap') ---\n");
    for (uint32_t n : ns) {
        size_t bytes = (size_t)4 * sizeQ * n * 8;
        ++callIndex;
        uint64_t* h_buf = nullptr;
        auto t0 = std::chrono::steady_clock::now();
        cudaError_t e = cudaMallocHost(&h_buf, bytes);
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        printf("call #%d: n=%u bytes=%zu live-pinned-allocs-BEFORE-this-call=%zu cudaMallocHost=%.2fus (%s)\n",
               callIndex, n, bytes, livePinned.size(), us, e == cudaSuccess ? "ok" : cudaGetErrorString(e));
        if (e == cudaSuccess) livePinned.push_back(h_buf);
    }

    for (auto p : livePinned) cudaFreeHost(p);
    return 0;
}
