#include <cuda_runtime.h>
#include <cstdint>

// Galois automorphism permutation, coefficient domain:
//   out[(j*k) mod 2n] = in[j], negated when the target lands in [n,2n) (x^n=-1).
// One thread per coefficient. This was a HOST loop per tower (~19 ms per
// rotation at n=8192, run twice), which forced a device->host round trip around
// every rotation and blocked keeping a whole transform stage resident.
__global__ void automorph_permute_kernel(uint64_t* __restrict__ out,
                                         const uint64_t* __restrict__ in,
                                         uint32_t n, uint64_t k, uint64_t q)
{
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    uint64_t M   = 2ull * n;
    uint64_t idx = ((uint64_t)j * k) % M;
    uint64_t v   = in[j];
    if (idx < n) out[idx] = v;
    else         out[idx - n] = v ? (q - v) : 0ull;
}

extern "C" void LaunchAutomorphPermute(uint64_t* d_out, const uint64_t* d_in,
                                       uint32_t n, uint64_t k, uint64_t q,
                                       cudaStream_t s)
{
    uint32_t blocks = (n + 255) / 256;
    automorph_permute_kernel<<<blocks, 256, 0, s>>>(d_out, d_in, n, k, q);
}
