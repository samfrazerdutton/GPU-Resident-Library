#pragma once
#include <cuda_runtime.h>
#include <cstdint>
extern "C" void LaunchAutomorphPermute(uint64_t* d_out, const uint64_t* d_in,
                                       uint32_t n, uint64_t k, uint64_t q,
                                       cudaStream_t s);

// Eval-domain automorphism: pure gather via a precomputed index map (see
// kernels/automorph_eval.cu for the derivation and validation). Replaces
// INTT->permute->NTT with one kernel, for data already in eval domain.
extern "C" void LaunchAutoEvalPermute(uint64_t* d_out, const uint64_t* d_in,
                                      const uint32_t* d_map, uint32_t n,
                                      cudaStream_t s);
extern "C" void LaunchAutoEvalPermuteBatched(uint64_t* d_out, const uint64_t* d_in,
                                             const uint32_t* d_map,
                                             uint32_t towers, uint32_t n,
                                             cudaStream_t s);
