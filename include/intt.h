#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Inverse NTT (Gentleman-Sande), bit-exact vs OpenFHE's
// InverseTransformFromBitReverseInPlace. Bit-reversed in, standard out.
extern "C" void LaunchINTT_GS(uint64_t* d_x,
                              const uint64_t* d_invroots, const uint64_t* d_invprecon,
                              uint32_t n, uint64_t q,
                              uint64_t n_inv, uint64_t n_inv_precon, cudaStream_t s);

// Tower-batched INTT: caller contract -- d_invroots, d_invprecon, d_mods, d_n_inv,
// and d_n_inv_precon must all share the same tower ordering and have length == towers.
extern "C" void LaunchINTT_GS_Batched(uint64_t* d_x,
                                      const uint64_t* const* d_invroots,
                                      const uint64_t* const* d_invprecon,
                                      const uint64_t* d_mods,
                                      const uint64_t* d_n_inv,
                                      const uint64_t* d_n_inv_precon,
                                      uint32_t towers, uint32_t n, cudaStream_t s);
