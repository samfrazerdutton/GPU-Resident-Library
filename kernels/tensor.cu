#include <cuda_runtime.h>
#include <cstdint>
// Fused CKKS tensor + relin-combine kernels, ALL towers in one launch
// (grid.y = tower). Cuts per-op launches from sizeQ to 1.

__device__ __forceinline__
uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t q){
    return (uint64_t)(((unsigned __int128)a*b)%q); }

// t0=c0a*c0b, t1=c0a*c1b+c1a*c0b, t2=c1a*c1b  (pointwise eval, per tower)
__global__ void tensor_kernel(
    const uint64_t* __restrict__ c0a, const uint64_t* __restrict__ c1a,
    const uint64_t* __restrict__ c0b, const uint64_t* __restrict__ c1b,
    uint64_t* __restrict__ t0, uint64_t* __restrict__ t1, uint64_t* __restrict__ t2,
    const uint64_t* __restrict__ mods, uint32_t n)
{
    uint32_t k = blockIdx.x*blockDim.x + threadIdx.x;
    uint32_t t = blockIdx.y;
    if (k >= n) return;
    uint64_t q = mods[t];
    size_t x = (size_t)t*n + k;
    uint64_t a0=c0a[x], a1=c1a[x], b0=c0b[x], b1=c1b[x];
    t0[x] = mulmod64(a0,b0,q);
    uint64_t s = mulmod64(a0,b1,q) + mulmod64(a1,b0,q);
    t1[x] = (s>=q)? s-q : s;
    t2[x] = mulmod64(a1,b1,q);
}
extern "C" void LaunchTensor(
    const uint64_t* c0a, const uint64_t* c1a,
    const uint64_t* c0b, const uint64_t* c1b,
    uint64_t* t0, uint64_t* t1, uint64_t* t2,
    const uint64_t* d_mods, uint32_t sizeQ, uint32_t n, cudaStream_t s)
{
    dim3 g((n+255)/256, sizeQ);
    tensor_kernel<<<g,256,0,s>>>(c0a,c1a,c0b,c1b,t0,t1,t2,d_mods,n);
}

// r0=t0+ba0, r1=t1+ba1 (per tower)
__global__ void combine_kernel(
    uint64_t* __restrict__ r0, uint64_t* __restrict__ r1,
    const uint64_t* __restrict__ t0, const uint64_t* __restrict__ t1,
    const uint64_t* __restrict__ ba0, const uint64_t* __restrict__ ba1,
    const uint64_t* __restrict__ mods, uint32_t n)
{
    uint32_t k = blockIdx.x*blockDim.x + threadIdx.x;
    uint32_t t = blockIdx.y;
    if (k >= n) return;
    uint64_t q = mods[t];
    size_t x = (size_t)t*n + k;
    uint64_t a = t0[x]+ba0[x]; r0[x] = (a>=q)? a-q : a;
    uint64_t b = t1[x]+ba1[x]; r1[x] = (b>=q)? b-q : b;
}
extern "C" void LaunchCombine(
    uint64_t* r0, uint64_t* r1,
    const uint64_t* t0, const uint64_t* t1,
    const uint64_t* ba0, const uint64_t* ba1,
    const uint64_t* d_mods, uint32_t sizeQ, uint32_t n, cudaStream_t s)
{
    dim3 g((n+255)/256, sizeQ);
    combine_kernel<<<g,256,0,s>>>(r0,r1,t0,t1,ba0,ba1,d_mods,n);
}
