// Phase 34: non-invasive CPU-vs-GPU time split for the CURRENT production
// encrypt path (encrypt_host_shoup_pooled_barrett_batched), measured as a
// black box -- zero production source files modified. Brackets the call
// with cudaEvent_t on the default stream (the same stream the production
// function's kernels and cudaMemcpy calls use internally) to capture
// GPU-side elapsed time (from the first H2D copy through the last D2H
// copy, since cudaMemcpy on the default stream is itself ordered against
// prior work on that stream and cudaEventRecord(stream=0) records after
// everything already queued on stream 0 completes up to that point -- this
// works because the production function performs cudaDeviceSynchronize()
// internally before returning, so by the time control returns here, the
// "stop" event has already been reached). Wall-clock time for the same
// call, measured with std::chrono, minus the GPU-side elapsed time, gives
// the CPU-only portion (RNG + Barrett-embed + Shoup-combine, all of which
// run before/after the GPU section and are NOT concurrent with it, since
// the whole function is serialized under one mutex with no stream overlap
// opportunity -- confirmed by reading the production source directly).
#include "keygen.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <complex>
#include <algorithm>

namespace gpufhe {
void encode_host_fft(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void encrypt_host_shoup_pooled_barrett_batched(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
}
using clk = std::chrono::steady_clock;
using cd = std::complex<double>;

int main() {
    const uint32_t n = 8192, sizeQ = 4;
    const uint64_t ns = 1; const double sigma = 3.19;
    std::vector<uint64_t> mod;
    gpufhe::native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids, sizeQ - 1, 55, n, mod); for (auto x : mids) mod.push_back(x); }
    std::vector<uint64_t> root(sizeQ);
    for (uint32_t i = 0; i < sizeQ; ++i) root[i] = gpufhe::native_root(n, mod[i]);
    auto KP = gpufhe::keygen_host(n, mod, root, ns, sigma, 101);

    std::vector<cd> z(n/2, cd(1.23, 0.0));
    std::vector<int64_t> m;
    gpufhe::encode_host_fft(m, z, n, std::pow(2.0, 45));

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart); cudaEventCreate(&evStop);

    const int R = 20;
    std::vector<double> wallMs(R), gpuMs(R);
    for (int rep = 0; rep < R; ++rep) {
        std::vector<uint64_t> c0, c1;
        auto t0 = clk::now();
        cudaEventRecord(evStart, 0);
        gpufhe::encrypt_host_shoup_pooled_barrett_batched(c0, c1, m, KP.pkA, KP.pkB, n, mod, root, ns, sigma, 1000 + rep);
        cudaEventRecord(evStop, 0);
        cudaEventSynchronize(evStop);
        auto t1 = clk::now();
        float gpuElapsed = 0; cudaEventElapsedTime(&gpuElapsed, evStart, evStop);
        wallMs[rep] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        gpuMs[rep] = (double)gpuElapsed;
    }
    cudaEventDestroy(evStart); cudaEventDestroy(evStop);

    auto median = [](std::vector<double> v){ std::sort(v.begin(), v.end()); size_t k=v.size()/2;
        return v.size()%2 ? v[k] : 0.5*(v[k-1]+v[k]); };
    double wallMed = median(wallMs), gpuMed = median(gpuMs);
    std::printf("=== Phase 34: encrypt_host_shoup_pooled_barrett_batched CPU/GPU split (n=%u, sizeQ=%u, R=%d) ===\n", n, sizeQ, R);
    std::printf("  wall-clock median: %.4f ms\n", wallMed);
    std::printf("  GPU-bracketed (H2D+kernel+D2H via cudaEvent on stream 0) median: %.4f ms (%.1f%% of wall)\n",
                gpuMed, 100.0 * gpuMed / wallMed);
    std::printf("  CPU-only (RNG+Barrett-embed+Shoup-combine, by subtraction) median: %.4f ms (%.1f%% of wall)\n",
                wallMed - gpuMed, 100.0 * (wallMed - gpuMed) / wallMed);
    return 0;
}
