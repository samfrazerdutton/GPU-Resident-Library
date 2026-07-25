#include <cuda_runtime.h>
#include <cstdint>

// Coefficient-wise multiply mod q, plain residues. Matches OpenFHE's
// DCRTPoly operator* in EVALUATION form exactly (a*b mod q per coefficient,
// plain residue domain in and out) -- so results are bit-exact vs OpenFHE
// with no Montgomery domain conversion. Montgomery is a later optimization
// gated on the benchmark showing the 128-bit modulo is the bottleneck.
__global__ void rns_mult_exact_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ r,
    uint64_t q, uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        r[idx] = (uint64_t)(((unsigned __int128)a[idx] * b[idx]) % q);
}

// Multiply one RNS tower (length n) of a by the matching tower of b into r.
// Operates in place on resident device buffers; no host round-trip.
extern "C" void LaunchRNSMultTower(
    const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_r,
    uint64_t q, uint32_t n, cudaStream_t stream)
{
    uint32_t blocks = (n + 255) / 256;
    rns_mult_exact_kernel<<<blocks, 256, 0, stream>>>(d_a, d_b, d_r, q, n);
}
