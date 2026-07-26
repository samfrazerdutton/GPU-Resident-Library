#include <cuda_runtime.h>
#include <cstdint>

// ApproxSwitchCRTBasis: RNS basis conversion at the heart of Hybrid keyswitch.
// Reconstructs a polynomial from source basis Q (sizeQ towers) into target
// basis P (sizeP towers). Per coefficient ri:
//   for each source tower i: x_i = a[i][ri] * QHatInvModq[i] mod q_i  (Shoup)
//   for each target tower j: sum[j] += (uint128) x_i * QHatModp[i][j]
//   ans[j][ri] = BarrettUint128ModUint64(sum[j], p_j, mu_j)
// Matches OpenFHE's HAVE_INT128 branch. Borrowed constants (QHatInvModq +
// precon, QHatModp 2D, modpBarrettMu) so the arithmetic is validated before
// native param generation -- the rescale playbook.

__device__ __forceinline__
uint64_t shoup_mulmod(uint64_t x, uint64_t c, uint64_t precon, uint64_t q) {
    uint64_t hi = __umul64hi(x, precon);
    uint64_t r  = x * c - hi * q;
    return (r >= q) ? r - q : r;
}

// Barrett reduce 128-bit a mod 64-bit modulus, mu = floor(2^128/modulus) (128b).
// Computes floor(a*mu / 2^128) via 64x64 limb products (matching OpenFHE's
// BarrettUint128ModUint64 exactly), then result = a_lo - qhat*modulus, corrected.
__device__ __forceinline__
uint64_t barrett_u128_mod_u64(unsigned __int128 a, uint64_t modulus,
                              unsigned __int128 mu) {
    uint64_t a_lo = (uint64_t)a,  a_hi = (uint64_t)(a >> 64);
    uint64_t mu_lo = (uint64_t)mu, mu_hi = (uint64_t)(mu >> 64);

    // floor(a*mu / 2^128): accumulate the cross terms, keep only what OpenFHE keeps.
    uint64_t left_hi = __umul64hi(a_lo, mu_lo);            // hi of a_lo*mu_lo

    unsigned __int128 mid1 = (unsigned __int128)a_lo * mu_hi;
    uint64_t mid1_lo = (uint64_t)mid1, mid1_hi = (uint64_t)(mid1 >> 64);

    // tmp1 = mid1_lo + left_hi (with carry into tmp2)
    unsigned __int128 s1 = (unsigned __int128)mid1_lo + left_hi;
    uint64_t tmp1  = (uint64_t)s1;
    uint64_t carry = (uint64_t)(s1 >> 64);
    uint64_t tmp2  = mid1_hi + carry;

    unsigned __int128 mid2 = (unsigned __int128)a_hi * mu_lo;
    uint64_t mid2_lo = (uint64_t)mid2, mid2_hi = (uint64_t)(mid2 >> 64);

    // carry = overflow of (mid2_lo + tmp1)
    unsigned __int128 s2 = (unsigned __int128)mid2_lo + tmp1;
    carry = (uint64_t)(s2 >> 64);
    uint64_t left_hi2 = mid2_hi + carry;

    uint64_t qhat = a_hi * mu_hi + tmp2 + left_hi2;   // low word of floor(a*mu/2^128)

    uint64_t result = a_lo - qhat * modulus;          // mod 2^64 implicit
    while (result >= modulus) result -= modulus;
    return result;
}

// One thread per coefficient. Source towers laid out [i*n + ri], output [j*n + ri].
// consts: QHatInvModq[i], QHatInvModqPrecon[i], q[i]  (per source tower)
//         QHatModp[i*sizeP + j]                        (source x target, flattened)
//         p[j], mu[j] (128-bit, as two uint64 lo/hi)   (per target tower)
__global__ void approx_switch_kernel(
    const uint64_t* __restrict__ src,       // sizeQ towers x n
    uint64_t* __restrict__ dst,             // sizeP towers x n
    const uint64_t* __restrict__ QHatInvModq,
    const uint64_t* __restrict__ QHatInvModqPrecon,
    const uint64_t* __restrict__ q,
    const uint64_t* __restrict__ QHatModp,  // sizeQ x sizeP
    const uint64_t* __restrict__ p,
    const uint64_t* __restrict__ mu_lo,
    const uint64_t* __restrict__ mu_hi,
    uint32_t sizeQ, uint32_t sizeP, uint32_t n)
{
    uint32_t ri = blockIdx.x * blockDim.x + threadIdx.x;
    if (ri >= n) return;

    // sizeP accumulators. sizeP is small (P basis ~1-4 towers); cap for locals.
    unsigned __int128 sum[8];
    for (uint32_t j = 0; j < sizeP; ++j) sum[j] = 0;

    for (uint32_t i = 0; i < sizeQ; ++i) {
        uint64_t xi = shoup_mulmod(src[(size_t)i*n + ri],
                                   QHatInvModq[i], QHatInvModqPrecon[i], q[i]);
        const uint64_t* row = QHatModp + (size_t)i*sizeP;
        for (uint32_t j = 0; j < sizeP; ++j)
            sum[j] += (unsigned __int128)xi * row[j];
    }

    for (uint32_t j = 0; j < sizeP; ++j) {
        unsigned __int128 mu = ((unsigned __int128)mu_hi[j] << 64) | mu_lo[j];
        dst[(size_t)j*n + ri] = barrett_u128_mod_u64(sum[j], p[j], mu);
    }
}

extern "C" void LaunchApproxSwitchCRTBasis(
    const uint64_t* d_src, uint64_t* d_dst,
    const uint64_t* d_QHatInvModq, const uint64_t* d_QHatInvModqPrecon, const uint64_t* d_q,
    const uint64_t* d_QHatModp,
    const uint64_t* d_p, const uint64_t* d_mu_lo, const uint64_t* d_mu_hi,
    uint32_t sizeQ, uint32_t sizeP, uint32_t n, cudaStream_t s)
{
    uint32_t blocks = (n + 255) / 256;
    approx_switch_kernel<<<blocks, 256, 0, s>>>(
        d_src, d_dst, d_QHatInvModq, d_QHatInvModqPrecon, d_q,
        d_QHatModp, d_p, d_mu_lo, d_mu_hi, sizeQ, sizeP, n);
}
