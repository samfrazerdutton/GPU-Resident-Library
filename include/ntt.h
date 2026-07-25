#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Forward CT-NTT, bit-exact vs OpenFHE's ForwardTransformToBitReverseInPlace.
// Carried from openfheNVDIA-GPU (validated on 12 real CKKS tower moduli,
// 20-60 bit). Standard-order in, bit-reversed out, result in [0,q).
extern "C" void LaunchNTT_CT(uint64_t* d_x,
                             const uint64_t* d_roots, const uint64_t* d_precon,
                             uint32_t n, uint64_t q, cudaStream_t s);
