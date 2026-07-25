#include <cuda_runtime.h>
#include <cstdint>

// Fused correction for rescale, per surviving tower i:
//   result[j] = ( a[j]*qlInvModq[i] + b[j]*QlQlInvModqlDivqlModq[i] ) mod q_i
// where a = the surviving tower's coefficients (eval form), b = the dropped
// tower switched into q_i and NTT'd back to eval form. Both scalars are the
// per-tower constants OpenFHE precomputes (GetqlInvModq / GetQlQlInvModqlDivqlModq).
// Plain 128-bit mulmod to match OpenFHE bit-for-bit; Shoup is a later opt.
__global__ void rescale_fuse_kernel(
    uint64_t* __restrict__ a,          // surviving tower, overwritten with result
    const uint64_t* __restrict__ b,    // switched+NTT'd dropped tower
    uint64_t s1, uint64_t s2, uint64_t q, uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint64_t t1 = (uint64_t)(((unsigned __int128)a[idx] * s1) % q);
    uint64_t t2 = (uint64_t)(((unsigned __int128)b[idx]  * s2) % q);
    uint64_t s  = t1 + t2;
    a[idx] = (s >= q) ? s - q : s;
}

// One tower's fused correction, in place on a (result), reading b.
extern "C" void LaunchRescaleFuse(
    uint64_t* d_a, const uint64_t* d_b,
    uint64_t s1, uint64_t s2, uint64_t q, uint32_t n, cudaStream_t s)
{
    uint32_t blocks = (n + 255) / 256;
    rescale_fuse_kernel<<<blocks, 256, 0, s>>>(d_a, d_b, s1, s2, q, n);
}
