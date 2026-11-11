// Phase 34 experiment: tests whether merging encrypt_host_shoup_pooled_
// barrett_batched's 4 separate LaunchNTT_CT_Batched calls (one each for
// v/e0/e1/m, towers=sizeQ) into ONE call (towers=4*sizeQ, treating the
// already-contiguous packed buffer as 4*sizeQ "virtual towers") reduces
// per-call latency. Verified safe by reading kernels/ntt_ct.cu's
// ct_stage_b/ct_shared_tail_b directly: tower index is blockIdx.y, each
// tower operates on x+tw*n using roots[tw]/precon[tw]/mods[tw] -- an exact
// reorganization of identical work (repeat the 4 real per-tower table
// pointers 4 times), not an approximation. Zero production code touched;
// this is a pure standalone A/B against the real kernel entry point.
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
using clk = std::chrono::steady_clock;

static uint64_t mm(uint64_t a, uint64_t b, uint64_t q) { return (uint64_t)(((unsigned __int128)a * b) % q); }
static uint64_t invmod(uint64_t a, uint64_t q) {
    unsigned __int128 r = 1, bb = a % q; uint64_t e = q - 2;
    while (e) { if (e & 1) r = (r * bb) % q; bb = (bb * bb) % q; e >>= 1; }
    return (uint64_t)r;
}
static uint32_t br(uint32_t v, uint32_t b) { uint32_t r = 0; for (uint32_t k = 0; k < b; k++) { r = (r << 1) | (v & 1); v >>= 1; } return r; }

int main() {
    const uint32_t n = 8192, sizeQ = 4;
    std::vector<uint64_t> mod;
    gpufhe::native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids, sizeQ - 1, 55, n, mod); for (auto x : mids) mod.push_back(x); }
    std::vector<uint64_t> root(sizeQ);
    for (uint32_t i = 0; i < sizeQ; ++i) root[i] = gpufhe::native_root(n, mod[i]);

    uint32_t logn = 0; while ((1u << logn) < n) ++logn;
    std::vector<std::vector<uint64_t>> frH(sizeQ), fpH(sizeQ);
    std::vector<uint64_t*> d_fr_real(sizeQ), d_fp_real(sizeQ);
    for (uint32_t t = 0; t < sizeQ; ++t) {
        uint64_t q = mod[t];
        std::vector<uint64_t> pw(n); pw[0] = 1;
        for (uint32_t i = 1; i < n; ++i) pw[i] = mm(pw[i - 1], root[t], q);
        frH[t].resize(n); fpH[t].resize(n);
        for (uint32_t i = 0; i < n; ++i) { uint64_t r = pw[br(i, logn)]; frH[t][i] = r; fpH[t][i] = (uint64_t)(((__uint128_t)r << 64) / q); }
        cudaMalloc(&d_fr_real[t], n * 8); cudaMemcpy(d_fr_real[t], frH[t].data(), n * 8, cudaMemcpyHostToDevice);
        cudaMalloc(&d_fp_real[t], n * 8); cudaMemcpy(d_fp_real[t], fpH[t].data(), n * 8, cudaMemcpyHostToDevice);
    }
    uint64_t *d_mod4, *d_mod16;
    cudaMalloc(&d_mod4, sizeQ * 8); cudaMemcpy(d_mod4, mod.data(), sizeQ * 8, cudaMemcpyHostToDevice);
    std::vector<uint64_t> mod16; for (int r = 0; r < 4; ++r) for (auto m : mod) mod16.push_back(m);
    cudaMalloc(&d_mod16, 16 * 8); cudaMemcpy(d_mod16, mod16.data(), 16 * 8, cudaMemcpyHostToDevice);

    uint64_t **d_fr4, **d_fp4, **d_fr16, **d_fp16;
    cudaMalloc(&d_fr4, sizeQ * sizeof(uint64_t*)); cudaMemcpy(d_fr4, d_fr_real.data(), sizeQ * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    cudaMalloc(&d_fp4, sizeQ * sizeof(uint64_t*)); cudaMemcpy(d_fp4, d_fp_real.data(), sizeQ * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    std::vector<uint64_t*> fr16v, fp16v;
    for (int r = 0; r < 4; ++r) for (auto p : d_fr_real) fr16v.push_back(p);
    for (int r = 0; r < 4; ++r) for (auto p : d_fp_real) fp16v.push_back(p);
    cudaMalloc(&d_fr16, 16 * sizeof(uint64_t*)); cudaMemcpy(d_fr16, fr16v.data(), 16 * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    cudaMalloc(&d_fp16, 16 * sizeof(uint64_t*)); cudaMemcpy(d_fp16, fp16v.data(), 16 * sizeof(uint64_t*), cudaMemcpyHostToDevice);

    const size_t T = (size_t)sizeQ * n;
    uint64_t *d_packedA, *d_packedB;
    cudaMalloc(&d_packedA, 4 * T * 8);
    cudaMalloc(&d_packedB, 4 * T * 8);
    std::vector<uint64_t> h_init(4 * T);
    for (size_t i = 0; i < h_init.size(); ++i) h_init[i] = (i * 7 + 13) % mod[i / T < sizeQ ? (i / n) % sizeQ : 0];

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart); cudaEventCreate(&evStop);
    const int R = 30;
    std::vector<double> fourCallMs(R), oneCallMs(R);

    for (int rep = 0; rep < R; ++rep) {
        cudaMemcpy(d_packedA, h_init.data(), 4 * T * 8, cudaMemcpyHostToDevice);
        cudaEventRecord(evStart, 0);
        for (int b = 0; b < 4; ++b)
            LaunchNTT_CT_Batched(d_packedA + b * T, (const uint64_t* const*)d_fr4, (const uint64_t* const*)d_fp4, d_mod4, sizeQ, n, 0);
        cudaEventRecord(evStop, 0); cudaEventSynchronize(evStop);
        float ms; cudaEventElapsedTime(&ms, evStart, evStop);
        fourCallMs[rep] = ms;

        cudaMemcpy(d_packedB, h_init.data(), 4 * T * 8, cudaMemcpyHostToDevice);
        cudaEventRecord(evStart, 0);
        LaunchNTT_CT_Batched(d_packedB, (const uint64_t* const*)d_fr16, (const uint64_t* const*)d_fp16, d_mod16, 16, n, 0);
        cudaEventRecord(evStop, 0); cudaEventSynchronize(evStop);
        cudaEventElapsedTime(&ms, evStart, evStop);
        oneCallMs[rep] = ms;
    }

    std::vector<uint64_t> outA(4 * T), outB(4 * T);
    cudaMemcpy(outA.data(), d_packedA, 4 * T * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(outB.data(), d_packedB, 4 * T * 8, cudaMemcpyDeviceToHost);
    size_t diffs = 0; for (size_t i = 0; i < outA.size(); ++i) if (outA[i] != outB[i]) ++diffs;

    auto median = [](std::vector<double> v){ std::sort(v.begin(), v.end()); size_t k=v.size()/2;
        return v.size()%2 ? v[k] : 0.5*(v[k-1]+v[k]); };
    std::printf("=== Phase 34: NTT kernel-launch-merge experiment (n=%u, sizeQ=%u, R=%d) ===\n", n, sizeQ, R);
    std::printf("  correctness: %zu/%zu limbs differ between 4-separate-calls and 1-merged-call -- %s\n",
                diffs, outA.size(), diffs == 0 ? "[PASS] bit-exact" : "[FAIL]");
    std::printf("  4 separate LaunchNTT_CT_Batched(towers=4) calls, median: %.4f ms\n", median(fourCallMs));
    std::printf("  1 merged LaunchNTT_CT_Batched(towers=16) call,  median: %.4f ms\n", median(oneCallMs));
    std::printf("  speedup: %.3fx\n", median(fourCallMs) / median(oneCallMs));
    return 0;
}
