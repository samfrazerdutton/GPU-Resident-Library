#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Fast-keyswitch multiply-accumulate for one QP tower, one digit, in place:
//   out0 += digit*b mod q ; out1 += digit*a mod q
// Accumulators zero-init before the first digit. Caller passes already
// idx-offset a/b tower pointers (eval key stored over full QP).
extern "C" void LaunchKSMultAcc(
    uint64_t* d_out0, uint64_t* d_out1,
    const uint64_t* d_digit, const uint64_t* d_b, const uint64_t* d_a,
    uint64_t q, uint32_t n, cudaStream_t s);
