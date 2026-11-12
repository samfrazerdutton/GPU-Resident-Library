#include <cuda_runtime.h>
#include <cstdint>

// ApproxModDown combine step: per Q tower, ans[k] = (a[k] - b[k]) * s mod q,
// where a = the Q-part input coefficient, b = the P-part switched back to Q,
// s = PInvModq[i] (scalar per tower). Matches OpenFHE's ApproxModDown final
// loop (CKKS path, t=0 so the BGV t-multiplies are skipped).
//
// GOD PROMPT "CKKS PHASE 39": the textbook-to-source mapping (Ronny Ko,
// "The Beginner's Textbook for Fully Homomorphic Encryption", section D-5)
// identifies this as exactly the `ModSwitchRNS` combine step's final
// `|b^{-1}|_q * (chi - chi_hat) mod q` multiply by a FIXED per-tower scalar
// -- `s` (PInvModq[i]) is a modulus-switch-context constant, known before
// this kernel ever launches, and used identically for all n coefficients in
// the tower. This is precisely the shape Shoup's precomputed-multiplier
// technique is already used for everywhere else in this codebase (e.g.
// src/encrypt_host_shoup_pooled_barrett_batched.cpp's own shoup_mulmod):
// instead of one full 128-bit DIVISION per coefficient (the original `% q`
// below), the caller precomputes `precon = shoup_precon(s, q)` ONCE per
// tower (one division, not n of them) and this kernel does a multiply-
// and-shift per coefficient instead. Bit-exact: shoup_mulmod(x, s, precon,
// q) computes the identical (x*s) mod q as the original `(diff*s) % q` for
// any x,s < q, which holds here (diff is always a mod-q difference, s is
// always a modular inverse mod q) -- verified by a standalone 35-million-
// trial reproducer (research/benchmarks/bench_moddown_shoup_check.cpp)
// across every modulus bit-width this project uses, before this change was
// made, plus this project's own already-verified shoup_mulmod (identical
// formula, already used on other hot paths).
__device__ __forceinline__
uint64_t shoup_mulmod(uint64_t x, uint64_t c, uint64_t precon, uint64_t q) {
    uint64_t hi = __umul64hi(x, precon);
    uint64_t r  = x * c - hi * q;
    return (r >= q) ? r - q : r;
}

__global__ void moddown_combine_kernel(
    uint64_t* __restrict__ ans,        // output, one tower
    const uint64_t* __restrict__ a,    // Q-part input tower (eval)
    const uint64_t* __restrict__ b,    // P-switched-to-Q tower (eval)
    uint64_t s, uint64_t sPrecon, uint64_t q, uint32_t n)
{
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    uint64_t diff = a[k];
    if (diff < b[k]) diff += q;        // (a - b) mod q
    diff -= b[k];
    ans[k] = shoup_mulmod(diff, s, sPrecon, q);
}

extern "C" void LaunchModDownCombine(
    uint64_t* d_ans, const uint64_t* d_a, const uint64_t* d_b,
    uint64_t s, uint64_t sPrecon, uint64_t q, uint32_t n, cudaStream_t st)
{
    uint32_t blocks = (n + 255) / 256;
    moddown_combine_kernel<<<blocks, 256, 0, st>>>(d_ans, d_a, d_b, s, sPrecon, q, n);
}
