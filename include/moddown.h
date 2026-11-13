#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// ApproxModDown combine (CKKS, t=0): ans = (a - b)*s mod q per Q tower.
// a = Q-part input (eval), b = P-part switched back to Q (eval), s = PInvModq[i].
// GOD PROMPT "CKKS PHASE 39": sPrecon = shoup_precon(s, q), precomputed ONCE
// per tower by the caller (s is a fixed modulus-switch-context constant) --
// replaces a 128-bit division per coefficient with Shoup's multiply-and-
// shift, bit-exact (see kernels/moddown.cu's own comment for the proof).
extern "C" void LaunchModDownCombine(
    uint64_t* d_ans, const uint64_t* d_a, const uint64_t* d_b,
    uint64_t s, uint64_t sPrecon, uint64_t q, uint32_t n, cudaStream_t st);

// GOD PROMPT "CKKS PHASE 40" Track E: batched variant, one launch across
// `towers` towers instead of `towers` separate LaunchModDownCombine calls.
// d_a/d_b/d_ans: contiguous [towers][n] (same layout the serial loop used).
// d_s/d_sPrecon/d_q: device arrays of length `towers` (per-tower PInvModq[i],
// shoup_precon(PInvModq[i],q_i), q_i), uploaded once by the caller.
extern "C" void LaunchModDownCombineBatched(
    uint64_t* d_ans, const uint64_t* d_a, const uint64_t* d_b,
    const uint64_t* d_s, const uint64_t* d_sPrecon, const uint64_t* d_q,
    uint32_t towers, uint32_t n, cudaStream_t st);
