#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Inverse NTT (Gentleman-Sande), bit-exact vs OpenFHE's
// InverseTransformFromBitReverseInPlace. Bit-reversed in, standard out.
extern "C" void LaunchINTT_GS(uint64_t* d_x,
                              const uint64_t* d_invroots, const uint64_t* d_invprecon,
                              uint32_t n, uint64_t q,
                              uint64_t n_inv, uint64_t n_inv_precon, cudaStream_t s);
