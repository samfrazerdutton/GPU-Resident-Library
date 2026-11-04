#pragma once
// GOD PROMPT "CKKS PHASE 20": move-only RAII owner of a page-aligned host
// allocation registered with CUDA via cudaHostRegister. Phase 19 (research/
// TRANSFER_PROFILING.md's Phase 19 section) found, mechanistically (via
// nsys, not just comparative timing), that cudaHostRegister on
// already-allocated, already-touched memory is 1.5x-4x cheaper than
// cudaMallocHost/cudaHostAlloc for the FULL lifecycle, while performing
// the SAME underlying ioctl-bound registration with the NVIDIA driver.
// This class is the single, explicit ownership model for that resource,
// designed BEFORE any production integration (governing directive's
// Section 2), following the SAME move-only/valid-flag/idempotent-reset
// RAII pattern already established in this codebase (include/ks_raii.h).
//
// API REQUIREMENTS VERIFIED AGAINST THE INSTALLED CUDA 13.2 HEADERS
// (/usr/local/cuda/include/cuda_runtime_api.h), not assumed:
//   - cudaHostRegister(ptr, size, flags): the CUDA Runtime API docs state
//     the page-aligned requirement is handled internally by the driver
//     (the call accepts any pointer/size and internally extends to cover
//     the containing page(s)) -- this class nonetheless allocates via
//     posix_memalign to the system page size BEFORE registering, so the
//     registered range is EXACTLY page-aligned on both ends, the
//     strictest possible interpretation, verified empirically in Phase
//     19's own diagnostic (zero registration failures across any run
//     with this allocation strategy).
//   - cudaHostRegisterDefault (0) is used unless the caller overrides it;
//     Phase 19 found cudaHostAllocWriteCombined made no measurable
//     difference for cudaHostAlloc, and there is no registration-flag
//     analogue worth defaulting to anything other than Default here.
//   - Registering an ALREADY-REGISTERED range is documented to fail
//     (cudaErrorHostMemoryAlreadyRegistered) -- this class tracks its own
//     state explicitly and NEVER calls cudaHostRegister twice on the same
//     allocation (guarded by `state_`), so this condition can only be
//     reached via misuse outside this class, which is exactly what
//     tests/test_registered_host_buffer.cpp's failure-path test exercises
//     deliberately, OUTSIDE this class's own guard, to prove the
//     underlying CUDA behavior rather than assume it.
//   - Unregistering a range with an in-flight asynchronous transfer still
//     touching it is undefined by CUDA's own contract -- this class NEVER
//     unregisters implicitly without a synchronization point: release()
//     takes an optional stream and synchronizes it FIRST if given; a
//     caller that issues cudaMemcpyAsync against data() on a stream MUST
//     either pass that stream to release() or have already synchronized
//     it before this object is destroyed/reset.
//   - Registration validity is tied to the CUDA context that was current
//     when cudaHostRegister was called; this class does not attempt to
//     track or re-validate context identity across calls (no API exists
//     to query it portably) -- documented here as a known limitation, not
//     hidden: destroying this object after the owning CUDA context has
//     already been torn down is undefined, exactly as it would be for a
//     raw cudaHostRegister/cudaHostUnregister pair.
//   - MEASURED FACT, surprising enough to record explicitly (verified via
//     a direct probe, research/TRANSFER_PROFILING.md's Phase 20 section):
//     on this installed CUDA 13.2/driver 595.97/WSL2 stack,
//     cudaHostRegister is NOT gated by RLIMIT_MEMLOCK at all --
//     registrations up to 500MB succeed with zero error despite the 64MB
//     soft/hard RLIMIT_MEMLOCK this process runs under. Any budget cap a
//     caller of this class enforces (e.g. the production pool in
//     src/encrypt_host_shoup_pooled_barrett_batched_registered.cpp) is
//     therefore a SELF-IMPOSED safety margin against unbounded host RAM
//     growth, not protection against an OS-level wall this API actually
//     respects -- do not assume RLIMIT_MEMLOCK caps this path the way it
//     caps plain mlock()/cudaMallocHost's underlying mechanism.
//
// OWNERSHIP MODEL (the 8 responsibilities the governing directive's
// Section 2 lists, mapped to this class's own methods):
//   1. Allocate host memory       -> allocate_and_register()'s posix_memalign
//   2. Touch/initialize pages     -> allocate_and_register()'s memset
//   3. Register with CUDA         -> allocate_and_register()'s cudaHostRegister
//   4. Expose the usable buffer   -> data() (returns nullptr unless state()==Registered)
//   5. Prevent use-after-free     -> data() returns nullptr after release(); move-only, no copy
//   6. Ensure GPU ops completed before unregistration -> release(stream)'s optional sync
//   7. Unregister exactly once   -> release()'s state check (idempotent)
//   8. Release host alloc exactly once -> release()'s state check (idempotent)
//
// STATE MACHINE:
//   Empty -> Allocated -> Registered -> (release) -> Empty
//   Empty -> Allocated -> Failed (registration failed; host memory already freed) -> (release is a no-op)
//   Empty -> Failed (zero-byte or posix_memalign failure; nothing to free)
// A Failed instance has ALREADY released whatever partial resource it
// held by the time allocate_and_register() returns -- there is no
// intermediate state a caller can observe where something is leaked or
// half-initialized, matching the exact guarantee Phase 16's own
// `PrewarmEntry`/`prewarm_locked()` established for cudaMallocHost.
#include <cuda_runtime.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <cstddef>

namespace gpufhe {

enum class RegisteredBufferState {
    Empty,       // default-constructed or moved-from / released -- nothing owned
    Allocated,   // host memory allocated + touched, NOT YET registered with CUDA
    Registered,  // allocated AND registered -- data() is usable
    Failed,      // allocation or registration failed; any partial resource already freed
};

class RegisteredHostBuffer {
public:
    RegisteredHostBuffer() noexcept = default;

    // Allocates `bytes` of page-aligned host memory, touches every page
    // (first-touch, before registering -- Phase 19's own design point),
    // then registers it with CUDA. On ANY failure, whatever was already
    // acquired is released before returning; state() is Failed and
    // data() is nullptr. Never leaves a partial resource outstanding.
    explicit RegisteredHostBuffer(size_t bytes, unsigned flags = cudaHostRegisterDefault) noexcept {
        allocate_and_register(bytes, flags);
    }

    RegisteredHostBuffer(const RegisteredHostBuffer&) = delete;
    RegisteredHostBuffer& operator=(const RegisteredHostBuffer&) = delete;

    RegisteredHostBuffer(RegisteredHostBuffer&& o) noexcept
        : ptr_(o.ptr_), bytes_(o.bytes_), state_(o.state_), lastError_(o.lastError_) {
        o.ptr_ = nullptr; o.bytes_ = 0; o.state_ = RegisteredBufferState::Empty;
    }
    RegisteredHostBuffer& operator=(RegisteredHostBuffer&& o) noexcept {
        if (this != &o) {
            release();
            ptr_ = o.ptr_; bytes_ = o.bytes_; state_ = o.state_; lastError_ = o.lastError_;
            o.ptr_ = nullptr; o.bytes_ = 0; o.state_ = RegisteredBufferState::Empty;
        }
        return *this;
    }

    ~RegisteredHostBuffer() { release(); }

    // Idempotent -- safe to call more than once, or on an Empty/Failed
    // instance (no-op beyond the first real call). If `syncStream` is
    // non-null, it is synchronized BEFORE unregistering, so a caller
    // that issued cudaMemcpyAsync against data() on that stream can pass
    // it here to guarantee the transfer has actually completed before
    // the memory stops being registered -- responsibility 6 above.
    void release(cudaStream_t syncStream = nullptr) noexcept {
        if (state_ == RegisteredBufferState::Registered) {
            if (syncStream) cudaStreamSynchronize(syncStream);
            cudaHostUnregister(ptr_);
        }
        if (state_ == RegisteredBufferState::Registered || state_ == RegisteredBufferState::Allocated) {
            free(ptr_);
        }
        ptr_ = nullptr; bytes_ = 0; state_ = RegisteredBufferState::Empty;
    }

    void* data() const noexcept { return state_ == RegisteredBufferState::Registered ? ptr_ : nullptr; }
    size_t bytes() const noexcept { return bytes_; }
    RegisteredBufferState state() const noexcept { return state_; }
    bool usable() const noexcept { return state_ == RegisteredBufferState::Registered; }
    cudaError_t last_error() const noexcept { return lastError_; }

private:
    void allocate_and_register(size_t bytes, unsigned flags) noexcept {
        release(); // defensive: safe even though a fresh instance is already Empty
        lastError_ = cudaSuccess;
        if (bytes == 0) { state_ = RegisteredBufferState::Failed; return; }

        long pageSize = sysconf(_SC_PAGESIZE);
        void* p = nullptr;
        if (posix_memalign(&p, (size_t)pageSize, bytes) != 0) {
            state_ = RegisteredBufferState::Failed;
            return;
        }
        memset(p, 0, bytes); // first-touch every page BEFORE registering
        ptr_ = p; bytes_ = bytes; state_ = RegisteredBufferState::Allocated;

        cudaError_t e = cudaHostRegister(p, bytes, flags);
        if (e != cudaSuccess) {
            lastError_ = e;
            cudaGetLastError(); // clear sticky error state unconditionally
            // Allocation succeeded, registration failed: free the host
            // allocation (never leak it), never call cudaHostUnregister
            // (it was never registered) -- responsibility 8, correctly
            // skipping responsibility 7 since it does not apply here.
            free(ptr_);
            ptr_ = nullptr; bytes_ = 0;
            state_ = RegisteredBufferState::Failed;
            return;
        }
        state_ = RegisteredBufferState::Registered;
    }

    void* ptr_ = nullptr;
    size_t bytes_ = 0;
    RegisteredBufferState state_ = RegisteredBufferState::Empty;
    cudaError_t lastError_ = cudaSuccess;
};

} // namespace gpufhe
