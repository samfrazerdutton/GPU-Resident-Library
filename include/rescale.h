#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Fused rescale correction, one tower in place:
//   a[j] = (a[j]*s1 + b[j]*s2) mod q
// s1 = qlInvModq[i], s2 = QlQlInvModqlDivqlModq[i] (OpenFHE's precomputed
// per-tower constants); b = the dropped tower switched into q and NTT'd to eval.
// GOD PROMPT "CKKS PHASE 39": s1Precon/s2Precon = shoup_precon(s1,q)/
// shoup_precon(s2,q), precomputed ONCE per tower by the caller -- replaces
// two 128-bit divisions per coefficient with Shoup's multiply-and-shift,
// bit-exact (see kernels/rescale.cu's own comment for the proof).
extern "C" void LaunchRescaleFuse(
    uint64_t* d_a, const uint64_t* d_b,
    uint64_t s1, uint64_t s1Precon, uint64_t s2, uint64_t s2Precon,
    uint64_t q, uint32_t n, cudaStream_t s);
