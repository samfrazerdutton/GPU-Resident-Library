#include <cuda_runtime.h>
#include <cstdint>

// Inverse NTT, Gentleman-Sande, bit-exact vs OpenFHE's
// InverseTransformFromBitReverseInPlace. Bit-reversed order in, standard order
// out. GS butterfly: lo' = lo+hi (mod q); hi' = (lo-hi)*omega (mod q, Shoup).
// OpenFHE fuses the final n^{-1} scaling into the last stage for speed; we run
// plain GS stages then a separate n^{-1} pass -- identical result mod q, and
// the INTT(NTT(x))==x round-trip proves it.

__device__ __forceinline__
uint64_t shoup_mulmod(uint64_t x, uint64_t omega, uint64_t precon, uint64_t q) {
    uint64_t hi = __umul64hi(x, precon);
    uint64_t r  = x * omega - hi * q;
    return (r >= q) ? r - q : r;
}

// One GS stage over the n/2 butterflies. m = blocks this stage, t = half-block
// width, logt = log2(t). omega = invroots[i+m] per block i.
__global__ void gs_stage(uint64_t* __restrict__ x,
                         const uint64_t* __restrict__ invroots,
                         const uint64_t* __restrict__ invprecon,
                         uint32_t n, uint32_t m, uint32_t t, uint32_t logt,
                         uint64_t q) {
    const uint32_t nbf = n >> 1;
    for (uint32_t bf = blockIdx.x * blockDim.x + threadIdx.x;
         bf < nbf; bf += gridDim.x * blockDim.x) {
        const uint32_t i   = bf >> logt;
        const uint32_t off = bf & (t - 1);
        const uint32_t j1  = (i << (logt + 1)) + off;
        const uint32_t j2  = j1 + t;

        const uint64_t omega  = invroots[i + m];
        const uint64_t pomega = invprecon[i + m];

        const uint64_t lo = x[j1];
        const uint64_t hi = x[j2];

        uint64_t sum = lo + hi;
        if (sum >= q) sum -= q;

        uint64_t dif = lo;
        if (dif < hi) dif += q;
        dif -= hi;
        dif = shoup_mulmod(dif, omega, pomega, q);

        x[j1] = sum;
        x[j2] = dif;
    }
}

__global__ void scale_ninv(uint64_t* __restrict__ x, uint64_t n_inv,
                           uint64_t n_inv_precon, uint32_t n, uint64_t q) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = shoup_mulmod(x[idx], n_inv, n_inv_precon, q);
}

static uint32_t ilog2_u(uint32_t v) { uint32_t k = 0; while ((1u << k) < v) k++; return k; }

// Full inverse transform in place. invroots/invprecon are OpenFHE's inverse
// bit-reversed root + Shoup precon tables; n_inv = n^{-1} mod q with its precon.
extern "C" void LaunchINTT_GS(uint64_t* d_x,
                             const uint64_t* d_invroots, const uint64_t* d_invprecon,
                             uint32_t n, uint64_t q,
                             uint64_t n_inv, uint64_t n_inv_precon, cudaStream_t s) {
    const int th  = 256;
    const int nbf = (int)(n >> 1);
    const int bl  = (nbf + th - 1) / th;
    // Reverse stage order vs forward: m from n/2 down to 1, t from 1 up.
    for (uint32_t m = n >> 1, t = 1; m >= 1; m >>= 1, t <<= 1) {
        uint32_t logt = ilog2_u(t);
        gs_stage<<<bl, th, 0, s>>>(d_x, d_invroots, d_invprecon, n, m, t, logt, q);
        if (m == 1) break;
    }
    const int bf = (int)((n + th - 1) / th);
    scale_ninv<<<bf, th, 0, s>>>(d_x, n_inv, n_inv_precon, n, q);
}
