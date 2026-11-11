#include "ntt.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>

namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
}
static uint64_t mm(uint64_t a, uint64_t b, uint64_t q) { return (uint64_t)(((unsigned __int128)a * b) % q); }
static uint32_t br(uint32_t v, uint32_t b) { uint32_t r = 0; for (uint32_t k = 0; k < b; k++) { r = (r << 1) | (v & 1); v >>= 1; } return r; }

int main() {
    const uint32_t n = 8192, sizeQ = 4;
    std::vector<uint64_t> mod;
    gpufhe::native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids, sizeQ - 1, 55, n, mod); for (auto x : mids) mod.push_back(x); }
    std::vector<uint64_t> root(sizeQ);
    for (uint32_t i = 0; i < sizeQ; ++i) root[i] = gpufhe::native_root(n, mod[i]);
    uint32_t logn = 0; while ((1u << logn) < n) ++logn;
    std::vector<uint64_t*> d_fr_real(sizeQ), d_fp_real(sizeQ);
    for (uint32_t t = 0; t < sizeQ; ++t) {
        uint64_t q = mod[t];
        std::vector<uint64_t> pw(n); pw[0] = 1;
        for (uint32_t i = 1; i < n; ++i) pw[i] = mm(pw[i - 1], root[t], q);
        std::vector<uint64_t> fr(n), fp(n);
        for (uint32_t i = 0; i < n; ++i) { uint64_t r = pw[br(i, logn)]; fr[i] = r; fp[i] = (uint64_t)(((__uint128_t)r << 64) / q); }
        cudaMalloc(&d_fr_real[t], n * 8); cudaMemcpy(d_fr_real[t], fr.data(), n * 8, cudaMemcpyHostToDevice);
        cudaMalloc(&d_fp_real[t], n * 8); cudaMemcpy(d_fp_real[t], fp.data(), n * 8, cudaMemcpyHostToDevice);
    }
    const size_t T = (size_t)sizeQ * n;
    uint64_t *d_packed, *d_fr4_ptr, *d_fp4_ptr, *d_mod4, *d_fr16_ptr, *d_fp16_ptr, *d_mod16;
    cudaMalloc(&d_packed, 4 * T * 8);
    cudaMalloc((void**)&d_fr4_ptr, sizeQ * sizeof(uint64_t*));
    cudaMalloc((void**)&d_fp4_ptr, sizeQ * sizeof(uint64_t*));
    cudaMalloc(&d_mod4, sizeQ * 8);
    cudaMalloc((void**)&d_fr16_ptr, 16 * sizeof(uint64_t*));
    cudaMalloc((void**)&d_fp16_ptr, 16 * sizeof(uint64_t*));
    cudaMalloc(&d_mod16, 16 * 8);
    std::vector<uint64_t> packed(4 * T, 42);
    std::vector<uint64_t*> fr16v, fp16v; for (int r = 0; r < 4; ++r) for (auto p : d_fr_real) fr16v.push_back(p);
    for (int r = 0; r < 4; ++r) for (auto p : d_fp_real) fp16v.push_back(p);
    std::vector<uint64_t> mod16; for (int r = 0; r < 4; ++r) for (auto m : mod) mod16.push_back(m);

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart); cudaEventCreate(&evStop);
    const int R = 30;
    std::vector<double> sepMs(R), mergedMs(R);

    for (int rep = 0; rep < R; ++rep) {
        // interleave A/B per-rep to control for clock-state drift
        cudaEventRecord(evStart, 0);
        cudaMemcpy(d_fr4_ptr, d_fr_real.data(), sizeQ * sizeof(uint64_t*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fp4_ptr, d_fp_real.data(), sizeQ * sizeof(uint64_t*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mod4, mod.data(), sizeQ * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(d_packed, packed.data(), 4 * T * 8, cudaMemcpyHostToDevice);
        for (int b = 0; b < 4; ++b)
            LaunchNTT_CT_Batched(d_packed + b * T, (const uint64_t* const*)d_fr4_ptr, (const uint64_t* const*)d_fp4_ptr, d_mod4, sizeQ, n, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(packed.data(), d_packed, 4 * T * 8, cudaMemcpyDeviceToHost);
        cudaEventRecord(evStop, 0); cudaEventSynchronize(evStop);
        float ms; cudaEventElapsedTime(&ms, evStart, evStop);
        sepMs[rep] = ms;

        cudaEventRecord(evStart, 0);
        cudaMemcpy(d_fr16_ptr, fr16v.data(), 16 * sizeof(uint64_t*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fp16_ptr, fp16v.data(), 16 * sizeof(uint64_t*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mod16, mod16.data(), 16 * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(d_packed, packed.data(), 4 * T * 8, cudaMemcpyHostToDevice);
        LaunchNTT_CT_Batched(d_packed, (const uint64_t* const*)d_fr16_ptr, (const uint64_t* const*)d_fp16_ptr, d_mod16, 16, n, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(packed.data(), d_packed, 4 * T * 8, cudaMemcpyDeviceToHost);
        cudaEventRecord(evStop, 0); cudaEventSynchronize(evStop);
        cudaEventElapsedTime(&ms, evStart, evStop);
        mergedMs[rep] = ms;
    }
    auto median = [](std::vector<double> v){ std::sort(v.begin(), v.end()); size_t k=v.size()/2;
        return v.size()%2 ? v[k] : 0.5*(v[k-1]+v[k]); };
    std::printf("=== Phase 34: full-sequence end-to-end impact of the kernel merge (interleaved A/B, R=%d) ===\n", R);
    std::printf("  SEPARATE (4 calls, matches current production): median=%.4f ms\n", median(sepMs));
    std::printf("  MERGED   (1 call, proposed change):             median=%.4f ms\n", median(mergedMs));
    std::printf("  full-sequence speedup: %.3fx\n", median(sepMs) / median(mergedMs));
    return 0;
}
