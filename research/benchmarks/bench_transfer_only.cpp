// Phase 34: isolates pure PCIe H2D+D2H transfer cost for a buffer EXACTLY
// the size of encrypt_host_shoup_pooled_barrett_batched's packed buffer
// (4*sizeQ*n*8 bytes, pageable host memory, matching production exactly --
// it does not use pinned memory) -- zero production code touched, pure
// standalone cudaMalloc/cudaMemcpy measurement using the same cudaEvent
// bracketing technique as bench_encrypt_cpu_gpu_split.cpp, to decompose
// that probe's 98.8%-GPU-side time into transfer vs. kernel+sync.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>

int main() {
    const size_t n = 8192, sizeQ = 4;
    const size_t T = sizeQ * n;
    const size_t bytes = 4 * T * 8; // matches packed buffer exactly
    std::vector<uint64_t> h_up(4 * T, 7), h_down(4 * T, 0);
    uint64_t* d_packed; cudaMalloc(&d_packed, bytes);

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart); cudaEventCreate(&evStop);
    const int R = 20;
    std::vector<double> h2dMs(R), d2hMs(R), roundtripMs(R);
    for (int rep = 0; rep < R; ++rep) {
        cudaEventRecord(evStart, 0);
        cudaMemcpy(d_packed, h_up.data(), bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(evStop, 0);
        cudaEventSynchronize(evStop);
        float ms; cudaEventElapsedTime(&ms, evStart, evStop);
        h2dMs[rep] = ms;

        cudaEventRecord(evStart, 0);
        cudaMemcpy(h_down.data(), d_packed, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(evStop, 0);
        cudaEventSynchronize(evStop);
        cudaEventElapsedTime(&ms, evStart, evStop);
        d2hMs[rep] = ms;
        roundtripMs[rep] = h2dMs[rep] + d2hMs[rep];
    }
    cudaFree(d_packed);
    cudaEventDestroy(evStart); cudaEventDestroy(evStop);

    auto median = [](std::vector<double> v){ std::sort(v.begin(), v.end()); size_t k=v.size()/2;
        return v.size()%2 ? v[k] : 0.5*(v[k-1]+v[k]); };
    std::printf("=== Phase 34: pure PCIe transfer cost, %zu bytes (%.2f MB), pageable host memory ===\n",
                bytes, bytes / 1048576.0);
    std::printf("  H2D median: %.4f ms\n", median(h2dMs));
    std::printf("  D2H median: %.4f ms\n", median(d2hMs));
    std::printf("  H2D+D2H round-trip median: %.4f ms\n", median(roundtripMs));
    return 0;
}
