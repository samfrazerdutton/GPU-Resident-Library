#pragma once
#include <cuda_runtime.h>
#include <cstdint>
extern "C" void LaunchAutomorphPermute(uint64_t* d_out, const uint64_t* d_in,
                                       uint32_t n, uint64_t k, uint64_t q,
                                       cudaStream_t s);
