#pragma once
#include <cuda_runtime.h>
#include <cstdint>
extern "C" void LaunchPtMulAcc(uint64_t* d_acc0, uint64_t* d_acc1,
                               const uint64_t* d_b0, const uint64_t* d_b1,
                               const uint64_t* d_d, uint64_t q, uint32_t n,
                               cudaStream_t s);
extern "C" void LaunchAddInto(uint64_t* d_a, const uint64_t* d_b,
                              uint64_t q, uint32_t n, cudaStream_t s);
