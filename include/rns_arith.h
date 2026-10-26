#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Coefficient-wise multiply mod q of one RNS tower, bit-exact vs OpenFHE's
// eval-form DCRTPoly multiply. Operates in place on resident device buffers.
extern "C" void LaunchRNSMultTower(
    const uint64_t* d_a, const uint64_t* d_b, uint64_t* d_r,
    uint64_t q, uint32_t n, cudaStream_t stream);

// Ciphertext*ciphertext RNS tensor product, batched across all tw towers:
// t0=a0*b0, t1=a0*b1+a1*b0, t2=a1*b1. mods is a HOST array of the tw tower
// moduli; all other pointers are resident device buffers, tw*n each.
extern "C" void LaunchCtTensorBatched(
    const uint64_t* d_a0, const uint64_t* d_a1,
    const uint64_t* d_b0, const uint64_t* d_b1,
    uint64_t* d_t0, uint64_t* d_t1, uint64_t* d_t2,
    const uint64_t* mods, uint32_t tw, uint32_t n, cudaStream_t stream);
