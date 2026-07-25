#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Coefficient-wise multiply mod q of one RNS tower, bit-exact vs OpenFHE's
// eval-form DCRTPoly multiply. Operates in place on resident device buffers.
extern "C" void LaunchRNSMultTower(
    const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_r,
    uint64_t q, uint32_t n, cudaStream_t stream);
