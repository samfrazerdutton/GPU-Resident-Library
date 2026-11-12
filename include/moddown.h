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
