#include <cuda_runtime.h>
#include <cstdint>
// Inverse NTT, Gentleman-Sande, bit-exact vs OpenFHE. Bit-reversed in, standard
// out. GS butterfly: lo'=lo+hi; hi'=(lo-hi)*omega (Shoup). Then a separate
// n^{-1} scale pass (OpenFHE fuses it into the last stage; identical mod q,
// proven by the INTT(NTT(x))==x round-trip).
//
// GS runs stages in REVERSE vs forward CT (m from n/2 down, t from 1 up), so
// the fuseable close-stride stages are the EARLY ones -- mirror of the forward
// NTT, where the late stages fused. Shared kernel runs FIRST (early small-t
// stages), then per-stage global for the large-t tail, then n^{-1}.

__device__ __forceinline__
uint64_t shoup_mulmod(uint64_t x, uint64_t omega, uint64_t precon, uint64_t q) {
    uint64_t hi = __umul64hi(x, precon);
    uint64_t r  = x * omega - hi * q;
    return (r >= q) ? r - q : r;
}

// One GS stage in global memory (late stages, large stride t).
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
        uint64_t sum = lo + hi; if (sum >= q) sum -= q;
        uint64_t dif = lo; if (dif < hi) dif += q; dif -= hi;
        dif = shoup_mulmod(dif, omega, pomega, q);
        x[j1] = sum; x[j2] = dif;
    }
}

// Fused EARLY GS stages in shared memory. Each block owns a contiguous span of
// SPAN = 2*blockDim elements, runs stages from t=1 up to t=t_end (= SPAN/2),
// i.e. m from m_start (=n/2) DOWN to m_end. One global read + one global write.
__global__ void gs_shared_head(uint64_t* __restrict__ x,
                              const uint64_t* __restrict__ invroots,
                              const uint64_t* __restrict__ invprecon,
                              uint32_t n, uint32_t m_start, uint32_t t_end,
                              uint64_t q) {
    extern __shared__ uint64_t sh[];
    const uint32_t span = t_end << 1;
    const uint32_t base = blockIdx.x * span;

    for (uint32_t k = threadIdx.x; k < span; k += blockDim.x)
        sh[k] = x[base + k];
    __syncthreads();

    // Early GS stages: t from 1 up to t_end, m from m_start down.
    uint32_t m = m_start;
    for (uint32_t t = 1; t <= t_end; t <<= 1, m >>= 1) {
        for (uint32_t bf = threadIdx.x; bf < (span >> 1); bf += blockDim.x) {
            const uint32_t blocks_in_span = span / (t << 1);
            const uint32_t sub = bf / t;
            const uint32_t off = bf & (t - 1);
            const uint32_t j1  = sub * (t << 1) + off;
            const uint32_t j2  = j1 + t;
            const uint32_t gi  = m + blockIdx.x * blocks_in_span + sub;
            const uint64_t omega  = invroots[gi];
            const uint64_t pomega = invprecon[gi];
            const uint64_t lo = sh[j1];
            const uint64_t hi = sh[j2];
            uint64_t sum = lo + hi; if (sum >= q) sum -= q;
            uint64_t dif = lo; if (dif < hi) dif += q; dif -= hi;
            dif = shoup_mulmod(dif, omega, pomega, q);
            sh[j1] = sum; sh[j2] = dif;
        }
        __syncthreads();
    }

    for (uint32_t k = threadIdx.x; k < span; k += blockDim.x)
        x[base + k] = sh[k];
}

__global__ void scale_ninv(uint64_t* __restrict__ x, uint64_t n_inv,
                           uint64_t n_inv_precon, uint32_t n, uint64_t q) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = shoup_mulmod(x[idx], n_inv, n_inv_precon, q);
}

static uint32_t ilog2_u(uint32_t v) { uint32_t k = 0; while ((1u << k) < v) k++; return k; }

extern "C" void LaunchINTT_GS(uint64_t* d_x,
                             const uint64_t* d_invroots, const uint64_t* d_invprecon,
                             uint32_t n, uint64_t q,
                             uint64_t n_inv, uint64_t n_inv_precon, cudaStream_t s) {
    const int th  = 256;
    const uint32_t SPAN = (uint32_t)th * 2;      // 512 elements/block
    const uint32_t t_end = SPAN >> 1;            // 256: largest t fusing in shared
    const int nbf = (int)(n >> 1);
    const int bl  = (nbf + th - 1) / th;

    // Early stages (t=1..t_end) fused in shared memory: m from n/2 down.
    const int spans = (int)(n / SPAN);
    const size_t shbytes = SPAN * sizeof(uint64_t);
    gs_shared_head<<<spans, th, shbytes, s>>>(d_x, d_invroots, d_invprecon,
                                              n, n >> 1, t_end, q);

    // Late stages (t > t_end) per-stage global. After the shared head, the next
    // stage has t = SPAN (=2*t_end) and m = (n/2) >> log2(SPAN).
    uint32_t m = (n >> 1) >> ilog2_u(SPAN);      // m after the shared stages
    for (uint32_t t = SPAN; m >= 1; m >>= 1, t <<= 1) {
        uint32_t logt = ilog2_u(t);
        gs_stage<<<bl, th, 0, s>>>(d_x, d_invroots, d_invprecon, n, m, t, logt, q);
        if (m == 1) break;
    }

    const int bf = (int)((n + th - 1) / th);
    scale_ninv<<<bf, th, 0, s>>>(d_x, n_inv, n_inv_precon, n, q);
}
