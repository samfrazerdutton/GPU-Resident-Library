#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Base conversion of one tower in place, q_old -> q_new, matching OpenFHE's
// NativeVectorT::SwitchModulus bit-exactly.
extern "C" void LaunchSwitchModulus(
    uint64_t* d_v, uint64_t q_old, uint64_t q_new, uint32_t n, cudaStream_t s);

// GOD PROMPT "CKKS PHASE 42": batched variant, one launch across `towers`
// towers instead of `towers` separate LaunchSwitchModulus calls. Reads the
// SAME single-tower source `d_in` for every tower (q_old is identical for
// all of them); writes each tower's result into its own [towers][n] slot of
// `d_out`. d_qNew/d_diff/d_newLarger: device arrays of length `towers`
// (per-tower q_new, |q_old-q_new| (or that mod q_new when q_new<q_old, per
// LaunchSwitchModulus's own rule), and whether q_new>q_old).
extern "C" void LaunchSwitchModulusBatched(
    uint64_t* d_out, const uint64_t* d_in, uint64_t q_old,
    const uint64_t* d_qNew, const uint64_t* d_diff, const int* d_newLarger,
    uint32_t towers, uint32_t n, cudaStream_t s);
