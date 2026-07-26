#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Resident ApproxSwitchCRTBasis (RNS basis conversion, Hybrid keyswitch core).
// Reconstructs sizeQ source towers into sizeP target towers, bit-exact vs
// OpenFHE. Constants borrowed from OpenFHE (QHatInvModq + precon, QHatModp 2D
// flattened sizeQ x sizeP, per-target p and 128-bit Barrett mu split lo/hi).
extern "C" void LaunchApproxSwitchCRTBasis(
    const uint64_t* d_src, uint64_t* d_dst,
    const uint64_t* d_QHatInvModq, const uint64_t* d_QHatInvModqPrecon, const uint64_t* d_q,
    const uint64_t* d_QHatModp,
    const uint64_t* d_p, const uint64_t* d_mu_lo, const uint64_t* d_mu_hi,
    uint32_t sizeQ, uint32_t sizeP, uint32_t n, cudaStream_t s);
