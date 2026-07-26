#include <cuda_runtime.h>
#include <cstdint>

// Coefficient-wise multiply mod q, plain residues in and out, bit-exact vs the
// plain 128-bit modulo (and thus OpenFHE). Both operands vary per coefficient,
// so Shoup doesn't apply. Montgomery reduction replaces the 128-bit divide with
// two mont_reduce steps (each a multiply + high-word add), no division. The
// second reduction (with R2 = 2^128 mod q) converts the result back to the
// plain domain, so this is a drop-in: plain in, plain out, no domain leakage
// into the NTT/rescale kernels. Ported from the validated cuda_math.cu.

// Montgomery reduction. R = 2^64. Given T < q*R, returns T * R^{-1} mod q.
__device__ __forceinline__
uint64_t mont_reduce(unsigned __int128 T, uint64_t q, uint64_t q_inv) {
    uint64_t m = (uint64_t)T * q_inv;
    unsigned __int128 mq = (unsigned __int128)m * q;
    uint64_t t = (uint64_t)((T + mq) >> 64);
    return (t >= q) ? t - q : t;
}

__global__ void rns_mult_mont_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ r,
    uint64_t q, uint64_t q_inv, uint64_t R2, uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned __int128 T = (unsigned __int128)a[idx] * b[idx];
    uint64_t res1 = mont_reduce(T, q, q_inv);              // a*b*R^{-1} mod q
    unsigned __int128 T2 = (unsigned __int128)res1 * R2;   // * R^2
    r[idx] = mont_reduce(T2, q, q_inv);                    // = a*b mod q (plain)
}

// q_inv = -q^{-1} mod 2^64 (Montgomery); R2 = 2^128 mod q. Both host-computed.
static uint64_t mont_qinv(uint64_t q) {
    // Newton iteration for q^{-1} mod 2^64, then negate.
    uint64_t x = q;                 // x = q^{-1} mod 2^3 is q for odd q (approx); iterate
    for (int i = 0; i < 6; ++i) x *= 2 - q * x;   // 2^6 > 64 bits of convergence
    return (uint64_t)(0) - x;       // -q^{-1} mod 2^64
}
static uint64_t mont_r2(uint64_t q) {
    // R = 2^64 mod q; R2 = R^2 mod q.
    unsigned __int128 R = ((unsigned __int128)1 << 64) % q;
    return (uint64_t)(((unsigned __int128)R * R) % q);
}

extern "C" void LaunchRNSMultTower(
    const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_r,
    uint64_t q, uint32_t n, cudaStream_t stream)
{
    uint64_t q_inv = mont_qinv(q);
    uint64_t R2    = mont_r2(q);
    uint32_t blocks = (n + 255) / 256;
    rns_mult_mont_kernel<<<blocks, 256, 0, stream>>>(d_a, d_b, d_r, q, q_inv, R2, n);
}
