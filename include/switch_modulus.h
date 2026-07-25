#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Base conversion of one tower in place, q_old -> q_new, matching OpenFHE's
// NativeVectorT::SwitchModulus bit-exactly.
extern "C" void LaunchSwitchModulus(
    uint64_t* d_v, uint64_t q_old, uint64_t q_new, uint32_t n, cudaStream_t s);
