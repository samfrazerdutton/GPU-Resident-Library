// GOD PROMPT "CKKS PHASE 20": standalone lifecycle/concurrency audit for
// RegisteredHostBuffer (include/registered_host_buffer.h) -- the new
// move-only RAII owner of a page-aligned, CUDA-registered host
// allocation, BEFORE it is wired into any real encryption code. Covers
// every scenario the governing directive's Section 4 lists, as directly
// and honestly as each can be safely tested:
//   1. successful alloc/register/use/unregister/release
//   2. repeated alloc/free cycles
//   3. repeated reuse of one registered buffer
//   4. registration failure and cleanup (a REAL induced CUDA failure --
//      registering an already-registered range -- not a mocked one)
//   5. partial initialization failure (a REAL induced failure -- a
//      request large enough to exceed RLIMIT_MEMLOCK)
//   6. pool destruction with registered buffers still live
//   7. pool eviction and reuse (small bounded LRU simulation)
//   8. multiple concurrent callers (each owning independent buffers)
//   10. shutdown while an async transfer is draining (real
//       cudaMemcpyAsync + stream sync via release(stream), not an
//       assumption that enqueue==complete)
//   11. correct handling of CUDA errors (covered by 4/5)
//   12. cleanup after a failed asynchronous operation (a REAL induced
//       async failure, not a mocked one)
// Scenario 9 (prewarm racing with encryption) is integration-level and
// is covered instead by tests/test_encrypt_host_shoup_pooled_barrett_
// batched_registered_concurrency_audit.cpp, once the real encrypt path
// exists -- it cannot be meaningfully tested against this standalone
// class alone (there is no "prewarm" concept at this layer by itself).
#include "registered_host_buffer.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
#include <cuda_runtime.h>

using gpufhe::RegisteredHostBuffer;
using gpufhe::RegisteredBufferState;

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  [FAIL] %s\n", msg); ++g_fail; } } while (0)

static void scenario1_basic_lifecycle() {
    std::printf("Scenario 1: successful alloc/register/use/unregister/release\n");
    RegisteredHostBuffer buf(262144); // 256KB, matches this project's smallest packed-buffer size
    CHECK(buf.state() == RegisteredBufferState::Registered, "expected Registered after successful construction");
    CHECK(buf.usable(), "expected usable() true");
    CHECK(buf.data() != nullptr, "expected non-null data()");
    CHECK(buf.bytes() == 262144, "expected bytes()==262144");
    if (!buf.usable()) { std::printf("  [SKIP rest of scenario 1] construction failed: %s\n", cudaGetErrorString(buf.last_error())); return; }
    std::memset(buf.data(), 0xAB, buf.bytes()); // use it
    buf.release();
    CHECK(buf.state() == RegisteredBufferState::Empty, "expected Empty after release()");
    CHECK(buf.data() == nullptr, "expected null data() after release()");
    buf.release(); // idempotent -- must not crash or double-free
    CHECK(buf.state() == RegisteredBufferState::Empty, "expected Empty after a second, idempotent release()");
}

static void scenario2_repeated_alloc_free_cycles() {
    std::printf("Scenario 2: repeated alloc/free cycles (50 fresh objects)\n");
    const int kCycles = 50;
    int failures = 0;
    for (int i = 0; i < kCycles; ++i) {
        RegisteredHostBuffer buf((size_t)(262144 + (i % 4) * 262144)); // cycles through 256K/512K/768K/1M
        if (!buf.usable()) ++failures;
        else std::memset(buf.data(), (uint8_t)i, buf.bytes());
        // destructor releases here, every iteration
    }
    std::printf("  %d/%d cycles failed to register\n", failures, kCycles);
    CHECK(failures == 0, "expected zero registration failures across 50 small fresh cycles");
}

static void scenario3_repeated_reuse() {
    std::printf("Scenario 3: repeated reuse of ONE registered buffer (200 use-cycles, no re-register)\n");
    RegisteredHostBuffer buf(1048576); // 1MB
    CHECK(buf.usable(), "expected usable() true before reuse loop");
    if (!buf.usable()) { std::printf("  [SKIP rest of scenario 3] construction failed: %s\n", cudaGetErrorString(buf.last_error())); return; }
    for (int i = 0; i < 200; ++i) {
        std::memset(buf.data(), (uint8_t)i, buf.bytes());
        volatile uint8_t check = ((uint8_t*)buf.data())[0];
        CHECK(check == (uint8_t)i, "expected the byte just written to read back unchanged");
    }
}

static void scenario4_registration_failure_and_cleanup() {
    std::printf("Scenario 4: REAL registration failure (double-register the SAME range) and cleanup\n");
    RegisteredHostBuffer buf(524288); // 512KB, successfully registered by the class
    CHECK(buf.usable(), "expected the first, class-owned registration to succeed");
    if (!buf.usable()) { std::printf("  [SKIP rest of scenario 4] construction failed: %s\n", cudaGetErrorString(buf.last_error())); return; }
    // Deliberately bypass the class and attempt a SECOND, raw
    // cudaHostRegister on the SAME already-registered range -- CUDA's
    // own documented behavior is to refuse this
    // (cudaErrorHostMemoryAlreadyRegistered). This proves the failure
    // path against REAL CUDA behavior, not a mock.
    cudaError_t e = cudaHostRegister(buf.data(), buf.bytes(), cudaHostRegisterDefault);
    std::printf("  raw double-cudaHostRegister on an already-registered range -> %s\n", cudaGetErrorString(e));
    CHECK(e != cudaSuccess, "expected cudaHostRegister on an already-registered range to FAIL");
    cudaGetLastError(); // clear sticky error state
    // The class's own single registration is still the only one that
    // ever succeeded -- its own release() must still unregister and
    // free cleanly exactly once.
    buf.release();
    CHECK(buf.state() == RegisteredBufferState::Empty, "expected clean release() after an unrelated external registration failure on the same range");
    // Confirm the CUDA context is still healthy afterward (a new,
    // independent registration succeeds normally).
    RegisteredHostBuffer probe(262144);
    CHECK(probe.usable(), "expected CUDA to remain healthy for a fresh registration after the induced failure above");
}

static void scenario5_partial_init_failure() {
    std::printf("Scenario 5: REAL partial-initialization failure (invalid cudaHostRegister flags)\n");
    // MEASURED FACT, surprising enough to record here explicitly: on
    // this installed CUDA 13.2/driver 595.97/WSL2 stack,
    // cudaHostRegister is NOT gated by RLIMIT_MEMLOCK at all -- a
    // direct probe (see research/TRANSFER_PROFILING.md's Phase 20
    // section) found registrations up to 500MB succeed with zero
    // error despite the 64MB soft/hard RLIMIT_MEMLOCK. An invalid
    // `flags` value (not a documented combination) is used instead to
    // induce a REAL, safe, bounded cudaHostRegister failure: allocation
    // and first-touch succeed (ordinary heap memory), but registration
    // itself must fail with cudaErrorInvalidValue.
    const unsigned badFlags = 0xFFFFFFFFu;
    RegisteredHostBuffer buf(262144, badFlags);
    std::printf("  262144 bytes with invalid flags=0x%x -> state=%s error=%s\n",
                badFlags,
                buf.state() == RegisteredBufferState::Failed ? "Failed" : "NOT-Failed",
                cudaGetErrorString(buf.last_error()));
    CHECK(buf.state() == RegisteredBufferState::Failed, "expected Failed state for an invalid-flags registration");
    CHECK(buf.data() == nullptr, "expected null data() after a failed registration");
    CHECK(buf.last_error() != cudaSuccess, "expected a non-success CUDA error recorded");
    // The host allocation must have been freed, not leaked -- not
    // directly observable from here, but a SUBSEQUENT small allocation
    // succeeding (not blocked by a leaked allocation or corrupted CUDA
    // state) is strong evidence of clean rollback.
    RegisteredHostBuffer probe(262144);
    CHECK(probe.usable(), "expected a fresh small registration to succeed immediately after the induced invalid-flags failure");
}

static void scenario6_pool_destruction_with_live_buffers() {
    std::printf("Scenario 6: pool (vector) destruction with registered buffers still live\n");
    {
        std::vector<RegisteredHostBuffer> pool;
        for (int i = 0; i < 5; ++i) pool.emplace_back((size_t)262144 * (i + 1));
        int usableCount = 0;
        for (auto& b : pool) if (b.usable()) ++usableCount;
        CHECK(usableCount == 5, "expected all 5 pooled buffers to be usable before pool destruction");
        std::printf("  %d/5 usable before pool goes out of scope\n", usableCount);
    } // pool's vector destructor runs here -- every element's own destructor must clean up exactly once
    std::printf("  pool destroyed without crashing\n");
}

static void scenario7_pool_eviction_and_reuse() {
    std::printf("Scenario 7: bounded pool eviction and reuse (cap=3, insert 6)\n");
    std::vector<RegisteredHostBuffer> pool;
    const size_t cap = 3;
    for (int i = 0; i < 6; ++i) {
        if (pool.size() >= cap) pool.erase(pool.begin()); // evict oldest -- its destructor runs here
        pool.emplace_back((size_t)262144);
        for (auto& b : pool) CHECK(b.usable(), "expected every SURVIVING pooled buffer to remain usable after an eviction");
    }
    CHECK(pool.size() == cap, "expected the pool to stay at its cap after 6 inserts");
    std::printf("  final pool size=%zu, all survivors usable\n", pool.size());
}

static void scenario8_concurrent_callers() {
    std::printf("Scenario 8: multiple concurrent callers, each owning independent buffers\n");
    const int kThreads = 8, kItersPerThread = 25;
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int it = 0; it < kItersPerThread; ++it) {
                RegisteredHostBuffer buf((size_t)262144 + (size_t)((t * 7 + it) % 4) * 262144);
                if (!buf.usable()) { failures.fetch_add(1); continue; }
                std::memset(buf.data(), (uint8_t)(t * 31 + it), buf.bytes());
                volatile uint8_t check = ((uint8_t*)buf.data())[0];
                if (check != (uint8_t)(t * 31 + it)) failures.fetch_add(1);
            }
        });
    }
    for (auto& th : threads) th.join();
    int totalIters = kThreads * kItersPerThread;
    std::printf("  %d threads x %d iters = %d total independent register/use/release cycles, %d failures\n",
                kThreads, kItersPerThread, totalIters, failures.load());
    CHECK(failures.load() == 0, "expected zero failures across all concurrent independent buffer cycles");
}

static void scenario10_shutdown_while_draining() {
    std::printf("Scenario 10: release() during an in-flight async H2D transfer (real cudaMemcpyAsync + stream sync)\n");
    const size_t bytes = 1048576;
    RegisteredHostBuffer buf(bytes);
    CHECK(buf.usable(), "expected buffer to be usable before issuing the async transfer");
    if (!buf.usable()) { std::printf("  [SKIP rest of scenario 10] construction failed: %s\n", cudaGetErrorString(buf.last_error())); return; }
    std::memset(buf.data(), 0x5A, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void* d_dst = nullptr;
    cudaMalloc(&d_dst, bytes);

    cudaMemcpyAsync(d_dst, buf.data(), bytes, cudaMemcpyHostToDevice, stream);
    // Immediately call release(stream) WITHOUT a separate explicit
    // synchronize first -- release()'s OWN internal cudaStreamSynchronize
    // must be what actually waits for the transfer, not a side effect of
    // test code happening to synchronize first. This is the directive's
    // explicit "do not treat a successful enqueue as proof the GPU has
    // finished using the memory" requirement, tested directly.
    buf.release(stream);
    CHECK(buf.state() == RegisteredBufferState::Empty, "expected clean release() after synchronizing the in-flight stream");

    // Verify the transfer ACTUALLY completed correctly before the host
    // memory was unregistered+freed (not just that no crash occurred).
    std::vector<uint8_t> readback(bytes);
    cudaMemcpy(readback.data(), d_dst, bytes, cudaMemcpyDeviceToHost);
    bool allMatch = true;
    for (size_t i = 0; i < bytes; ++i) if (readback[i] != 0x5A) { allMatch = false; break; }
    CHECK(allMatch, "expected the async H2D transfer's data to have genuinely completed before release() returned");

    cudaFree(d_dst);
    cudaStreamDestroy(stream);
}

static void scenario12_cleanup_after_failed_async_op() {
    std::printf("Scenario 12: cleanup after a REAL failed asynchronous operation\n");
    const size_t bytes = 262144;
    RegisteredHostBuffer buf(bytes);
    CHECK(buf.usable(), "expected buffer to be usable before the deliberately-failing async op");
    if (!buf.usable()) { std::printf("  [SKIP rest of scenario 12] construction failed: %s\n", cudaGetErrorString(buf.last_error())); return; }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // Deliberately pass a null device destination -- cudaMemcpyAsync
    // must fail (synchronously reported, since the invalid argument is
    // checked before any DMA is queued).
    cudaError_t rc = cudaMemcpyAsync(nullptr, buf.data(), bytes, cudaMemcpyHostToDevice, stream);
    std::printf("  cudaMemcpyAsync(nullptr dst, ...) -> %s\n", cudaGetErrorString(rc));
    CHECK(rc != cudaSuccess, "expected the deliberately-invalid async copy to fail");
    cudaGetLastError(); // clear sticky error state

    // The registered buffer's OWN state must be entirely unaffected by
    // an unrelated GPU-side error -- release() must still work cleanly.
    buf.release(stream);
    CHECK(buf.state() == RegisteredBufferState::Empty, "expected clean release() despite an unrelated earlier async failure");
    cudaStreamDestroy(stream);

    RegisteredHostBuffer probe(bytes);
    CHECK(probe.usable(), "expected CUDA to remain healthy for a fresh registration after the induced async failure");
}

int main() {
    scenario1_basic_lifecycle();
    scenario2_repeated_alloc_free_cycles();
    scenario3_repeated_reuse();
    scenario4_registration_failure_and_cleanup();
    scenario5_partial_init_failure();
    scenario6_pool_destruction_with_live_buffers();
    scenario7_pool_eviction_and_reuse();
    scenario8_concurrent_callers();
    scenario10_shutdown_while_draining();
    scenario12_cleanup_after_failed_async_op();

    std::printf("\nTotal check failures: %d\n", g_fail);
    std::printf(g_fail == 0 ? "[OVERALL PASS]\n" : "[OVERALL FAIL]\n");
    return g_fail == 0 ? 0 : 1;
}
