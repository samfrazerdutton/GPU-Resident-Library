#pragma once
#include "cuda_check.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

namespace gpufhe {

// Minimal, generic, single-shape, move-only RAII device buffer. Generalizes
// DeviceCiphertext's already-proven ownership pattern (allocate once in the
// constructor, mutate in place via kernels, upload/download explicitly, free
// in the destructor) to an arbitrary element type and count, instead of
// DeviceCiphertext's hardcoded two-ciphertext-tower shape.
//
// Deliberately NOT a pool, NOT a cache, NOT an execution context: per
// research/GPU_RESIDENT_ARCHITECTURE.md section 9, no second real caller has
// asked for those yet. This is the smallest primitive that lets new code
// stop hand-rolling cudaMalloc/cudaFree pairs. Upload/download default to
// synchronous (matching DeviceCiphertext's current behavior, so adopting
// this never silently changes existing ordering assumptions); the `_async`
// variants require the caller to pass a stream and take on responsibility
// for ordering against subsequent kernel launches or reads -- this class
// does not guess.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count_ > 0) {
            cuda_check(cudaMalloc(&d_ptr_, count_ * sizeof(T)), "DeviceBuffer: cudaMalloc");
        }
    }
    ~DeviceBuffer() {
        if (d_ptr_) cudaFree(d_ptr_); // destructor: best-effort, matches project convention elsewhere
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : d_ptr_(o.d_ptr_), count_(o.count_) {
        o.d_ptr_ = nullptr; o.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (d_ptr_) cudaFree(d_ptr_);
            d_ptr_ = o.d_ptr_; count_ = o.count_;
            o.d_ptr_ = nullptr; o.count_ = 0;
        }
        return *this;
    }

    // Synchronous: blocks until the copy completes, safe to reuse host_src
    // immediately on return. Matches DeviceCiphertext::upload's behavior.
    void upload(const T* host_src, size_t n, size_t offset = 0) {
        cuda_check(cudaMemcpy(d_ptr_ + offset, host_src, n * sizeof(T), cudaMemcpyHostToDevice),
                   "DeviceBuffer::upload");
    }
    void download(T* host_dst, size_t n, size_t offset = 0) const {
        cuda_check(cudaMemcpy(host_dst, d_ptr_ + offset, n * sizeof(T), cudaMemcpyDeviceToHost),
                   "DeviceBuffer::download");
    }

    // Asynchronous: returns before the copy necessarily completes. Caller
    // must keep host_src/host_dst alive and ordered (via the same stream,
    // an event, or a later sync) until the transfer actually finishes --
    // this is exactly the lifetime contract Phase 30 found violated with a
    // function-local std::vector, so it is stated here explicitly rather
    // than left implicit.
    void upload_async(const T* host_src, size_t n, cudaStream_t s, size_t offset = 0) {
        cuda_check(cudaMemcpyAsync(d_ptr_ + offset, host_src, n * sizeof(T), cudaMemcpyHostToDevice, s),
                   "DeviceBuffer::upload_async");
    }
    void download_async(T* host_dst, size_t n, cudaStream_t s, size_t offset = 0) const {
        cuda_check(cudaMemcpyAsync(host_dst, d_ptr_ + offset, n * sizeof(T), cudaMemcpyDeviceToHost, s),
                   "DeviceBuffer::download_async");
    }

    T* data() { return d_ptr_; }
    const T* data() const { return d_ptr_; }
    size_t size() const { return count_; }

private:
    T* d_ptr_ = nullptr;
    size_t count_ = 0;
};

} // namespace gpufhe
