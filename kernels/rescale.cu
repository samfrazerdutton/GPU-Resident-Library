#include <cuda_runtime.h>
#include <cstdint>

// Fused correction for rescale, per surviving tower i:
//   result[j] = ( a[j]*qlInvModq[i] + b[j]*QlQlInvModqlDivqlModq[i] ) mod q_i
// where a = the surviving tower's coefficients (eval form), b = the dropped
// tower switched into q_i and NTT'd back to eval form. Both scalars are the
// per-tower constants OpenFHE precomputes (GetqlInvModq / GetQlQlInvModqlDivqlModq).
//
// GOD PROMPT "CKKS PHASE 39": this file's own prior comment flagged "Shoup
// is a later opt" -- this is that optimization. The textbook-to-source
// mapping (Ronny Ko, "The Beginner's Textbook for Fully Homomorphic
// Encryption", section D-5.5) identifies this as the same `ModSwitchRNS`
// combine shape as kernels/moddown.cu's own ApproxModDown step, just with
// TWO fixed per-tower scalars (s1, s2) instead of one -- both are known
// before this kernel ever launches and used identically for every one of
// the n coefficients in the tower, the exact shape Shoup's precomputed-
// multiplier technique already targets everywhere else in this codebase.
// Bit-exact: shoup_mulmod(x,s,precon,q) computes the identical (x*s) mod q
// as the original `% q` division for any x,s < q (verified by the same
// 35-million-trial reproducer that justified the ModDown change,
// research/benchmarks/bench_moddown_shoup_check.cpp -- the underlying
// primitive is identical; only the caller differs), and the final
// `t1+t2` modular addition is unchanged, elementary modular addition
// already used identically throughout this codebase.
__device__ __forceinline__
uint64_t shoup_mulmod(uint64_t x, uint64_t c, uint64_t precon, uint64_t q) {
    uint64_t hi = __umul64hi(x, precon);
    uint64_t r  = x * c - hi * q;
    return (r >= q) ? r - q : r;
}

__global__ void rescale_fuse_kernel(
    uint64_t* __restrict__ a,          // surviving tower, overwritten with result
    const uint64_t* __restrict__ b,    // switched+NTT'd dropped tower
    uint64_t s1, uint64_t s1Precon, uint64_t s2, uint64_t s2Precon, uint64_t q, uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint64_t t1 = shoup_mulmod(a[idx], s1, s1Precon, q);
    uint64_t t2 = shoup_mulmod(b[idx], s2, s2Precon, q);
    uint64_t s  = t1 + t2;
    a[idx] = (s >= q) ? s - q : s;
}

// One tower's fused correction, in place on a (result), reading b.
// s1Precon/s2Precon = shoup_precon(s1,q)/shoup_precon(s2,q), precomputed
// ONCE per tower by the caller (s1,s2 are fixed rescale-context constants).
extern "C" void LaunchRescaleFuse(
    uint64_t* d_a, const uint64_t* d_b,
    uint64_t s1, uint64_t s1Precon, uint64_t s2, uint64_t s2Precon,
    uint64_t q, uint32_t n, cudaStream_t s)
{
    uint32_t blocks = (n + 255) / 256;
    rescale_fuse_kernel<<<blocks, 256, 0, s>>>(d_a, d_b, s1, s1Precon, s2, s2Precon, q, n);
}
