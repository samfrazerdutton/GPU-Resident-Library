#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace gpufhe {

// Shared CUDA error-check helper, promoted from device_ciphertext.cpp's
// local `ck()` (the only file in this codebase with a consistent
// checked-every-call convention before this round -- see
// research/GPU_RESIDENT_ARCHITECTURE.md section 9.2). Phase 31's whole
// root cause was a capture/launch sequence that checked none of its CUDA
// return codes, so a real failure read as a silent, non-deterministic
// data race instead of a loud, named error. Use this (or an explicit
// cudaError_t capture + check, for calls that must not throw mid-sequence)
// at every new CUDA call site.
inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

} // namespace gpufhe
