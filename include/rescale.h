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

// GOD PROMPT "CKKS PHASE 42": batched variant, one launch across `towers`
// towers instead of `towers` separate LaunchRescaleFuse calls. d_a/d_b use
// the same [towers][n] contiguous layout as moddown.cu's
// LaunchModDownCombineBatched. d_s1/d_s1Precon/d_s2/d_s2Precon/d_q: device
// arrays of length `towers`, the same per-tower rescale constants each call
// already computes, just uploaded once instead of used one tower at a time.
extern "C" void LaunchRescaleFuseBatched(
    uint64_t* d_a, const uint64_t* d_b,
    const uint64_t* d_s1, const uint64_t* d_s1Precon,
    const uint64_t* d_s2, const uint64_t* d_s2Precon, const uint64_t* d_q,
    uint32_t towers, uint32_t n, cudaStream_t s);
