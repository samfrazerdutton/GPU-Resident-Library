#include <cuda_runtime.h>
#include <cstdint>

// Fused plaintext-multiply + accumulate for the staged transforms:
//   acc0[k] = (acc0[k] + b0[k]*d[k]) mod q
//   acc1[k] = (acc1[k] + b1[k]*d[k]) mod q
// One tower per launch, one thread per coefficient, everything in EVAL form.
// This is the inner op of a transform stage: each diagonal contributes a
// rotated ciphertext times its plaintext diagonal into a running sum. Doing it
// on device is what lets a whole stage stay resident -- previously each
// diagonal round-tripped to host for ct_mul_pt_host + ct_add_ct_host.
__global__ void ptmulacc_kernel(uint64_t* __restrict__ acc0,
                                uint64_t* __restrict__ acc1,
                                const uint64_t* __restrict__ b0,
                                const uint64_t* __restrict__ b1,
                                const uint64_t* __restrict__ d,
                                uint64_t q, uint32_t n)
{
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    uint64_t dk = d[k];
    uint64_t p0 = (uint64_t)(((unsigned __int128)b0[k] * dk) % q);
    uint64_t p1 = (uint64_t)(((unsigned __int128)b1[k] * dk) % q);
    uint64_t s0 = acc0[k] + p0; if (s0 >= q) s0 -= q;
    uint64_t s1 = acc1[k] + p1; if (s1 >= q) s1 -= q;
    acc0[k] = s0;
    acc1[k] = s1;
}

extern "C" void LaunchPtMulAcc(uint64_t* d_acc0, uint64_t* d_acc1,
                               const uint64_t* d_b0, const uint64_t* d_b1,
                               const uint64_t* d_d, uint64_t q, uint32_t n,
                               cudaStream_t s)
{
    uint32_t blocks = (n + 255) / 256;
    ptmulacc_kernel<<<blocks, 256, 0, s>>>(d_acc0, d_acc1, d_b0, d_b1, d_d, q, n);
}
