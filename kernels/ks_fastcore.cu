#include <cuda_runtime.h>
#include <cstdint>

// Fast-keyswitch inner product (EvalFastKeySwitchCoreExt accumulate loop).
// For one QP output tower (modulus q): out0 += digit * b, out1 += digit * a,
// accumulated over all `limit` digits, mod q. Plain modular multiply-add per
// coefficient (Montgomery-domain not needed -- match OpenFHE's plain modmul).
// The caller handles the eval-key idx=(i>=sizeQl)?i+delta:i offset by passing
// the already-offset a/b tower pointers.

__device__ __forceinline__
uint64_t mulmod64(uint64_t x, uint64_t y, uint64_t q) {
    return (uint64_t)(((unsigned __int128)x * y) % q);
}

// Accumulate one digit's contribution into out0/out1 for one tower, in place.
// out0[k] = (out0[k] + digit[k]*b[k]) mod q ; out1 likewise with a.
__global__ void ks_macc_kernel(
    uint64_t* __restrict__ out0, uint64_t* __restrict__ out1,
    const uint64_t* __restrict__ digit,
    const uint64_t* __restrict__ b, const uint64_t* __restrict__ a,
    uint64_t q, uint32_t n)
{
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    uint64_t d = digit[k];
    uint64_t t0 = out0[k] + mulmod64(d, b[k], q); if (t0 >= q) t0 -= q;
    uint64_t t1 = out1[k] + mulmod64(d, a[k], q); if (t1 >= q) t1 -= q;
    out0[k] = t0;
    out1[k] = t1;
}

// One tower's multiply-accumulate of a single digit. Accumulators must be
// zero-initialized before the first digit; call once per digit per tower.
extern "C" void LaunchKSMultAcc(
    uint64_t* d_out0, uint64_t* d_out1,
    const uint64_t* d_digit, const uint64_t* d_b, const uint64_t* d_a,
    uint64_t q, uint32_t n, cudaStream_t s)
{
    uint32_t blocks = (n + 255) / 256;
    ks_macc_kernel<<<blocks, 256, 0, s>>>(d_out0, d_out1, d_digit, d_b, d_a, q, n);
}
