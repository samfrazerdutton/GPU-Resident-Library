#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

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

__device__ __forceinline__
uint64_t mont_mul(uint64_t a, uint64_t b, uint64_t q, uint64_t q_inv, uint64_t R2) {
    unsigned __int128 T = (unsigned __int128)a * b;
    uint64_t res1 = mont_reduce(T, q, q_inv);
    unsigned __int128 T2 = (unsigned __int128)res1 * R2;
    return mont_reduce(T2, q, q_inv);
}

// Ciphertext*ciphertext RNS tensor product, batched across all tw towers in
// one launch (grid.y = tower), replacing a CPU loop of 4 128-bit mmu() calls
// per element with one Montgomery-domain pass per thread. t0=a0*b0,
// t1=a0*b1+a1*b0, t2=a1*b1 -- bit-exact vs the plain 128-bit path (same
// two-step Montgomery reduction as rns_mult_mont_kernel).
__global__ void ct_tensor_mont_kernel(
    const uint64_t* __restrict__ a0, const uint64_t* __restrict__ a1,
    const uint64_t* __restrict__ b0, const uint64_t* __restrict__ b1,
    uint64_t* __restrict__ t0, uint64_t* __restrict__ t1, uint64_t* __restrict__ t2,
    const uint64_t* __restrict__ qArr, const uint64_t* __restrict__ qInvArr,
    const uint64_t* __restrict__ r2Arr, uint32_t n)
{
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    uint32_t tower = blockIdx.y;
    size_t x = (size_t)tower * n + k;
    uint64_t q = qArr[tower], qinv = qInvArr[tower], R2 = r2Arr[tower];
    uint64_t A0 = a0[x], A1 = a1[x], B0 = b0[x], B1 = b1[x];
    uint64_t r0 = mont_mul(A0, B0, q, qinv, R2);
    uint64_t p1 = mont_mul(A0, B1, q, qinv, R2);
    uint64_t p2 = mont_mul(A1, B0, q, qinv, R2);
    uint64_t sm = p1 + p2;
    uint64_t r1 = (sm >= q) ? sm - q : sm;
    uint64_t r2v = mont_mul(A1, B1, q, qinv, R2);
    t0[x] = r0; t1[x] = r1; t2[x] = r2v;
}

// mods is a HOST array of the tw tower moduli (same convention as every other
// qModHost-style array in this codebase). q_inv/R2 are derived on the host
// per call -- cheap (O(tw) Newton iterations), negligible next to the
// tw*n-sized transfers this replaces.
extern "C" void LaunchCtTensorBatched(
    const uint64_t* d_a0, const uint64_t* d_a1,
    const uint64_t* d_b0, const uint64_t* d_b1,
    uint64_t* d_t0, uint64_t* d_t1, uint64_t* d_t2,
    const uint64_t* mods, uint32_t tw, uint32_t n, cudaStream_t stream)
{
    std::vector<uint64_t> qinv(tw), r2(tw);
    for (uint32_t t = 0; t < tw; ++t) { qinv[t] = mont_qinv(mods[t]); r2[t] = mont_r2(mods[t]); }
    uint64_t *d_q, *d_qinv, *d_r2;
    cudaMalloc(&d_q, (size_t)tw * 8);
    cudaMalloc(&d_qinv, (size_t)tw * 8);
    cudaMalloc(&d_r2, (size_t)tw * 8);
    cudaMemcpyAsync(d_q, mods, (size_t)tw * 8, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_qinv, qinv.data(), (size_t)tw * 8, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_r2, r2.data(), (size_t)tw * 8, cudaMemcpyHostToDevice, stream);
    dim3 grid((n + 255) / 256, tw);
    ct_tensor_mont_kernel<<<grid, 256, 0, stream>>>(
        d_a0, d_a1, d_b0, d_b1, d_t0, d_t1, d_t2, d_q, d_qinv, d_r2, n);
    // d_q/d_qinv/d_r2 are call-local -- must outlive the kernel that reads
    // them, so synchronize this stream before freeing.
    cudaStreamSynchronize(stream);
    cudaFree(d_q); cudaFree(d_qinv); cudaFree(d_r2);
}
