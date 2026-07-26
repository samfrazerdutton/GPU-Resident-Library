// Cooley-Tukey NTT matching OpenFHE's Shoup butterfly bit-for-bit. Standard
// order in, bit-reversed out, [0,q). Early stages (butterfly stride > block
// span) run one kernel per stage in global memory; the LATE stages (stride
// small enough to fit a block) fuse into one shared-memory kernel -- one global
// read + one global write instead of a round trip per stage. This attacks the
// dominant cost the batch benchmark isolated: NTT global-memory traffic.

#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__
uint64_t shoup_mulmod(uint64_t x, uint64_t omega, uint64_t precon, uint64_t q) {
    uint64_t hi = __umul64hi(x, precon);
    uint64_t r  = x * omega - hi * q;
    return (r >= q) ? r - q : r;
}

// One CT stage in global memory (early stages, large stride).
__global__ void ct_stage(uint64_t* __restrict__ x,
                         const uint64_t* __restrict__ roots,
                         const uint64_t* __restrict__ precon,
                         uint32_t n, uint32_t m, uint32_t t, uint32_t logt,
                         uint64_t q) {
    const uint32_t nbf = n >> 1;
    for (uint32_t bf = blockIdx.x * blockDim.x + threadIdx.x;
         bf < nbf; bf += gridDim.x * blockDim.x) {
        const uint32_t i   = bf >> logt;
        const uint32_t off = bf & (t - 1);
        const uint32_t j1  = (i << (logt + 1)) + off;
        const uint32_t j2  = j1 + t;
        const uint64_t omega  = roots[i + m];
        const uint64_t pomega = precon[i + m];
        const uint64_t of = shoup_mulmod(x[j2], omega, pomega, q);
        const uint64_t lo = x[j1];
        uint64_t hi = lo + of; if (hi >= q) hi -= q;
        uint64_t lv = lo; if (lv < of) lv += q; lv -= of;
        x[j1] = hi; x[j2] = lv;
    }
}

// Fused late stages in shared memory. Each block owns a contiguous span of
// SPAN = 2*blockDim elements, loads it once, runs every stage from m_start
// (stride t_start = SPAN/2) down to t=1, writes back once.
// The block's span index is blockIdx.x; within the CT structure at stage m,
// the omega for a butterfly is roots[m + global_i], and global_i for this
// block's sub-butterflies is derived from blockIdx.x and the in-block position.
__global__ void ct_shared_tail(uint64_t* __restrict__ x,
                              const uint64_t* __restrict__ roots,
                              const uint64_t* __restrict__ precon,
                              uint32_t n, uint32_t m_start, uint32_t t_start,
                              uint64_t q) {
    extern __shared__ uint64_t sh[];
    const uint32_t span = t_start << 1;                 // elements per block
    const uint32_t base = blockIdx.x * span;            // global start of span

    // Cooperative load of the span into shared memory.
    for (uint32_t k = threadIdx.x; k < span; k += blockDim.x)
        sh[k] = x[base + k];
    __syncthreads();

    // Run stages: t from t_start down to 1, m from m_start up.
    uint32_t m = m_start;
    for (uint32_t t = t_start; t >= 1; t >>= 1, m <<= 1) {
        // butterflies within this block for this stage: span/2 of them
        for (uint32_t bf = threadIdx.x; bf < (span >> 1); bf += blockDim.x) {
            const uint32_t blocks_in_span = span / (t << 1); // sub-blocks of width 2t
            const uint32_t sub = bf / t;                     // which sub-block
            const uint32_t off = bf & (t - 1);               // offset within
            const uint32_t j1  = sub * (t << 1) + off;       // shared index
            const uint32_t j2  = j1 + t;
            // global omega index: at stage m, block's first omega is m + base/(2t);
            // sub-block sub adds sub. base/(2t) = blockIdx.x * (span/(2t)).
            const uint32_t gi = m + blockIdx.x * blocks_in_span + sub;
            const uint64_t omega  = roots[gi];
            const uint64_t pomega = precon[gi];
            const uint64_t of = shoup_mulmod(sh[j2], omega, pomega, q);
            const uint64_t lo = sh[j1];
            uint64_t hi = lo + of; if (hi >= q) hi -= q;
            uint64_t lv = lo; if (lv < of) lv += q; lv -= of;
            sh[j1] = hi; sh[j2] = lv;
        }
        __syncthreads();
        if (t == 1) break;  // unsigned guard
    }

    for (uint32_t k = threadIdx.x; k < span; k += blockDim.x)
        x[base + k] = sh[k];
}

static uint32_t ilog2_u(uint32_t v) { uint32_t k = 0; while ((1u << k) < v) k++; return k; }

extern "C" void LaunchNTT_CT(uint64_t* d_x,
                            const uint64_t* d_roots, const uint64_t* d_precon,
                            uint32_t n, uint64_t q, cudaStream_t s) {
    const int th  = 256;
    const uint32_t SPAN = (uint32_t)th * 2;          // 512 elements per block
    const int nbf = (int)(n >> 1);
    const int bl  = (nbf + th - 1) / th;

    // Early stages in global memory: while stride t > SPAN/2, run per-stage.
    uint32_t m = 1, t = n >> 1;
    for (; m < n && t > (SPAN >> 1); m <<= 1, t >>= 1) {
        uint32_t logt = ilog2_u(t);
        ct_stage<<<bl, th, 0, s>>>(d_x, d_roots, d_precon, n, m, t, logt, q);
    }
    // Late stages fused in shared memory: one block per span.
    if (m < n) {
        const int spans = (int)(n / SPAN);
        const size_t shbytes = SPAN * sizeof(uint64_t);
        ct_shared_tail<<<spans, th, shbytes, s>>>(d_x, d_roots, d_precon, n, m, t, q);
    }
}
