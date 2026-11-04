// GOD PROMPT "CKKS PHASE 20": encrypt_host_shoup_pooled_barrett_batched_registered
// -- experimental encrypt_host variant using CUDA-REGISTERED (not
// cudaMallocHost-PINNED) host staging memory, built on Phase 19's
// decisive, mechanistically-proven finding (research/TRANSFER_PROFILING.md's
// Phase 19 section): cudaHostRegister on pre-touched, page-aligned host
// memory is 1.5x-4x cheaper than cudaMallocHost/cudaHostAlloc for the
// FULL allocation lifecycle, via the SAME underlying ioctl-bound
// mechanism, just a lighter-weight driver call.
//
// This file mirrors encrypt_host_shoup_pooled_barrett_batched_pinned.cpp
// (src/encrypt_host_shoup_pooled_barrett_batched_pinned.cpp) EXACTLY in
// every respect EXCEPT the packed host staging buffer's own allocation
// mechanism: `include/registered_host_buffer.h`'s RegisteredHostBuffer
// (posix_memalign + first-touch + cudaHostRegister) instead of
// cudaMallocHost. Same packed [v|e0|e1|m] layout, same device buffer
// pool, same kernel launches, same single-mutex serialization, same
// embed/combine arithmetic, same RNG/noise generation -- NOTHING about
// what is computed changes, only the host staging memory's own
// allocation path.
//
// OWNERSHIP MODEL (governing directive Section 2): the registered host
// buffer is owned by a SHARED, PROCESS-LEVEL pool keyed by (n,sizeQ) --
// the SAME architecture every pool in this codebase already uses
// (enc_dev_tab/the device buffer pools/Phase 16's PrewarmEntry), now
// reusing RegisteredHostBuffer's own move-only RAII for THAT ONE
// resource instead of a raw cudaMallocHost pointer. Pool entries are
// never evicted (process-lifetime, identical to every other pool here);
// RegisteredHostBuffer's own destructor would run correctly at process
// exit regardless (its state machine guarantees exactly-once
// unregister+free even then), but in practice process exit reclaims
// everything before any destructor runs, exactly as every other
// never-freed pool in this codebase already accepts.
//
// SAFE FALLBACK (governing directive Section 8's explicit requirement):
// registration is attempted ONCE per (n,sizeQ) key, the first time that
// key is used. If it fails (RegisteredHostBuffer::usable()==false --
// either because registration itself failed, or because this pool's
// own strict cumulative-bytes budget, see kRegisteredBudgetBytes below,
// would be exceeded by adding this key), the pool entry is marked
// not-ready and EVERY subsequent call at that key transparently falls
// back to the existing, unmodified, already-production pageable path
// (encrypt_host_shoup_pooled_barrett_batched) -- never retrying
// registration, never leaving a partially-initialized buffer reachable,
// never crashing. This is the SAME fallback shape Phase 16's prewarm
// file already established for its own failure path.
//
// MEASURED FACT worth recording here (not merely in research/
// TRANSFER_PROFILING.md): a direct probe found cudaHostRegister on this
// installed CUDA 13.2/driver 595.97/WSL2 stack is NOT gated by
// RLIMIT_MEMLOCK (64MB on this system) -- registrations up to 500MB
// succeeded with zero error in that probe. kRegisteredBudgetBytes below
// is therefore a SELF-IMPOSED safety margin against unbounded host RAM
// growth across many distinct (n,sizeQ) keys, not protection against an
// OS-level wall this API was found to actually respect.
#include "keygen.h"
#include "ntt.h"
#include "registered_host_buffer.h"
#include <cstdint>
#include <vector>
#include <mutex>
#include <map>
#include <utility>
#include <random>
#include <cmath>
#include <cuda_runtime.h>

namespace gpufhe {
// The existing, unmodified, production pageable implementation --
// called directly as the fallback path, never reimplemented here.
void encrypt_host_shoup_pooled_barrett_batched(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                                                const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                                                const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);

namespace {
uint64_t am(uint64_t a, uint64_t b, uint64_t q) { uint64_t s = a + b; return s >= q ? s - q : s; }
uint64_t* up(const std::vector<uint64_t>& h) {
    uint64_t* d; cudaMalloc(&d, h.size() * 8); cudaMemcpy(d, h.data(), h.size() * 8, cudaMemcpyHostToDevice); return d;
}
uint64_t mulmod_cold(uint64_t a, uint64_t b, uint64_t q) { return (uint64_t)(((unsigned __int128)a * b) % q); }
inline uint64_t shoup_precon(uint64_t c, uint64_t q) { return (uint64_t)(((unsigned __int128)c << 64) / q); }
inline uint64_t shoup_mulmod(uint64_t x, uint64_t c, uint64_t precon, uint64_t q) {
    uint64_t hi = (uint64_t)(((unsigned __int128)x * precon) >> 64);
    uint64_t r = x * c - hi * q;
    return (r >= q) ? r - q : r;
}

struct TowerBarrett { uint64_t q; uint64_t mu; };
inline TowerBarrett make_tower_barrett(uint64_t q) {
    TowerBarrett T; T.q = q;
    T.mu = (uint64_t)(((unsigned __int128)1 << 64) / q);
    return T;
}
inline uint64_t barrett_reduce_u64(uint64_t x, const TowerBarrett& T) {
    uint64_t qhat = (uint64_t)(((unsigned __int128)x * T.mu) >> 64);
    uint64_t r = x - qhat * T.q;
    while (r >= T.q) r -= T.q;
    return r;
}
inline uint64_t emb_barrett(long x, const TowerBarrett& T) {
    uint64_t ax = (x < 0) ? (0ULL - (uint64_t)x) : (uint64_t)x;
    uint64_t r = barrett_reduce_u64(ax, T);
    if (x < 0 && r != 0) r = T.q - r;
    return r;
}

struct EncDevTab { uint64_t* fr = nullptr; uint64_t* fp = nullptr; };
EncDevTab build_enc_tab_registered(uint32_t n, uint64_t q, uint64_t root) {
    auto br = [](uint32_t v, uint32_t b) { uint32_t r = 0; for (uint32_t k = 0; k < b; k++) { r = (r << 1) | (v & 1); v >>= 1; } return r; };
    uint32_t logn = 0; while ((1u << logn) < n) ++logn;
    std::vector<uint64_t> fr(n), fp(n), pw(n); pw[0] = 1;
    for (uint32_t i = 1; i < n; ++i) pw[i] = mulmod_cold(pw[i - 1], root, q);
    for (uint32_t i = 0; i < n; ++i) { uint64_t r = pw[br(i, logn)]; fr[i] = r; fp[i] = (uint64_t)(((__uint128_t)r << 64) / q); }
    EncDevTab T; T.fr = up(fr); T.fp = up(fp);
    return T;
}
EncDevTab& enc_dev_tab_registered(uint32_t n, uint64_t q, uint64_t root) {
    static std::map<std::pair<uint32_t, uint64_t>, EncDevTab> cache;
    auto key = std::make_pair(n, q);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    return cache.emplace(key, build_enc_tab_registered(n, q, root)).first->second;
}

// Self-imposed, measurable, strict cap on cumulative registered host
// bytes across EVERY (n,sizeQ) key this pool has ever created -- see
// this file's header comment for why this is a safety margin, not
// enforcement of an OS limit. 16MiB comfortably covers every size this
// library actually exercises (the largest packed buffer at n=8192,
// sizeQ=8 is 2MiB) many times over, while leaving most of the host
// process's own 64MB RLIMIT_MEMLOCK headroom free for anything else
// concurrently pinning memory in the same process (e.g. Phase 15/16's
// own pinned/prewarmed pools, if exercised in the same process).
constexpr size_t kRegisteredBudgetBytes = 16ull * 1024 * 1024;
size_t g_registeredBytesUsed = 0; // protected by registered_mutex()

struct EncryptBufPoolRegistered {
    uint64_t* d_packed = nullptr; // device, [v | e0 | e1 | m]
    RegisteredHostBuffer h_packed; // host, registered, [v | e0 | e1 | m]
    uint64_t **d_fr = nullptr, **d_fp = nullptr;
    uint64_t* d_mod = nullptr;
    uint32_t sizeQ = 0;
    bool ready = false; // true iff h_packed registered successfully AND stayed within budget
};
using RegisteredKey = std::pair<uint32_t, uint32_t>; // (n, sizeQ)

std::mutex& registered_mutex() { static std::mutex m; return m; }
std::map<RegisteredKey, EncryptBufPoolRegistered>& registered_cache() {
    static std::map<RegisteredKey, EncryptBufPoolRegistered> cache;
    return cache;
}

// Must be called with registered_mutex() held. Creates (or returns the
// existing) pool entry for (n,sizeQ). A failed registration, OR a
// request that would exceed kRegisteredBudgetBytes, leaves `ready`
// false and h_packed empty (RegisteredHostBuffer's own constructor
// already guarantees no leak on its own failure path) -- the device
// buffers it also would have needed are simply never allocated in that
// case, since they are useless without a usable host staging buffer.
EncryptBufPoolRegistered& encrypt_buf_pool_registered_locked(uint32_t n, uint32_t sizeQ) {
    auto& cache = registered_cache();
    RegisteredKey key(n, sizeQ);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    EncryptBufPoolRegistered P; P.sizeQ = sizeQ;
    const size_t T8 = (size_t)sizeQ * n * 8;
    const size_t packedBytes = 4 * T8;

    if (g_registeredBytesUsed + packedBytes <= kRegisteredBudgetBytes) {
        P.h_packed = RegisteredHostBuffer(packedBytes);
        if (P.h_packed.usable()) {
            g_registeredBytesUsed += packedBytes;
            cudaMalloc(&P.d_packed, packedBytes);
            cudaMalloc(&P.d_fr, sizeQ * sizeof(uint64_t*));
            cudaMalloc(&P.d_fp, sizeQ * sizeof(uint64_t*));
            cudaMalloc(&P.d_mod, sizeQ * 8);
            P.ready = true;
        }
        // If registration itself failed, P.h_packed is already Empty
        // (its own constructor rolled back cleanly) -- P.ready stays
        // false, no device buffers are allocated, nothing to free.
    }
    // If the budget would be exceeded, registration is never even
    // attempted -- P.ready stays false, fallback applies.

    return cache.emplace(key, std::move(P)).first->second;
}
} // anon

// Explicit, opt-in, idempotent prewarm entry point -- mirrors Phase
// 16's encrypt_prewarm_pinned_buffer exactly, for the registered pool.
bool encrypt_prewarm_registered_buffer(uint32_t n, uint32_t sizeQ) {
    std::lock_guard<std::mutex> lock(registered_mutex());
    return encrypt_buf_pool_registered_locked(n, sizeQ).ready;
}
bool encrypt_registered_buffer_is_ready(uint32_t n, uint32_t sizeQ) {
    std::lock_guard<std::mutex> lock(registered_mutex());
    auto& cache = registered_cache();
    auto it = cache.find(RegisteredKey(n, sizeQ));
    return it != cache.end() && it->second.ready;
}

void encrypt_host_shoup_pooled_barrett_batched_registered(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1,
                                                            const std::vector<int64_t>& m,
                                                            const std::vector<uint64_t>& pkA, const std::vector<uint64_t>& pkB,
                                                            uint32_t n, const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root,
                                                            uint64_t ns, double sigma, uint64_t seed)
{
    uint32_t sizeQ = (uint32_t)mod.size();

    std::unique_lock<std::mutex> lock(registered_mutex());
    EncryptBufPoolRegistered& P = encrypt_buf_pool_registered_locked(n, sizeQ);
    if (!P.ready) {
        // Release our lock before deferring -- the fallback has its
        // own, entirely separate mutex and owes nothing to this pool.
        lock.unlock();
        encrypt_host_shoup_pooled_barrett_batched(c0, c1, m, pkA, pkB, n, mod, root, ns, sigma, seed);
        return;
    }

    std::mt19937_64 rng(seed); std::normal_distribution<double> g(0.0, sigma);
    std::vector<int> vc(n), e0c(n), e1c(n);
    for (uint32_t k = 0; k < n; ++k) { vc[k] = (int)(rng() % 3) - 1; e0c[k] = (int)std::lround(g(rng)); e1c[k] = (int)std::lround(g(rng)); }
    c0.resize((size_t)sizeQ * n); c1.resize((size_t)sizeQ * n);

    const size_t T = (size_t)sizeQ * n;
    uint64_t* h_packed = (uint64_t*)P.h_packed.data();
    uint64_t* vBuf = h_packed;
    uint64_t* e0Buf = h_packed + T;
    uint64_t* e1Buf = h_packed + 2 * T;
    uint64_t* mBuf = h_packed + 3 * T;
    for (uint32_t t = 0; t < sizeQ; ++t) { uint64_t q = mod[t];
        TowerBarrett TB = make_tower_barrett(q);
        uint64_t* vT = vBuf + (size_t)t * n; uint64_t* e0T = e0Buf + (size_t)t * n;
        uint64_t* e1T = e1Buf + (size_t)t * n; uint64_t* mT = mBuf + (size_t)t * n;
        for (uint32_t k = 0; k < n; ++k) {
            vT[k] = emb_barrett(vc[k], TB); e0T[k] = emb_barrett(e0c[k], TB);
            e1T[k] = emb_barrett(e1c[k], TB); mT[k] = emb_barrett((long)m[k], TB);
        }
    }

    std::vector<uint64_t*> h_fr(sizeQ), h_fp(sizeQ);
    for (uint32_t t = 0; t < sizeQ; ++t) { EncDevTab& Tt = enc_dev_tab_registered(n, mod[t], root[t]); h_fr[t] = Tt.fr; h_fp[t] = Tt.fp; }

    cudaMemcpy(P.d_fr, h_fr.data(), sizeQ * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(P.d_fp, h_fp.data(), sizeQ * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(P.d_mod, mod.data(), sizeQ * 8, cudaMemcpyHostToDevice);

    const size_t T8 = T * 8;
    cudaMemcpy(P.d_packed, h_packed, 4 * T8, cudaMemcpyHostToDevice);

    uint64_t* d_v = P.d_packed;
    uint64_t* d_e0 = P.d_packed + T;
    uint64_t* d_e1 = P.d_packed + 2 * T;
    uint64_t* d_m = P.d_packed + 3 * T;
    LaunchNTT_CT_Batched(d_v, (const uint64_t* const*)P.d_fr, (const uint64_t* const*)P.d_fp, P.d_mod, sizeQ, n, 0);
    LaunchNTT_CT_Batched(d_e0, (const uint64_t* const*)P.d_fr, (const uint64_t* const*)P.d_fp, P.d_mod, sizeQ, n, 0);
    LaunchNTT_CT_Batched(d_e1, (const uint64_t* const*)P.d_fr, (const uint64_t* const*)P.d_fp, P.d_mod, sizeQ, n, 0);
    LaunchNTT_CT_Batched(d_m, (const uint64_t* const*)P.d_fr, (const uint64_t* const*)P.d_fp, P.d_mod, sizeQ, n, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(h_packed, P.d_packed, 4 * T8, cudaMemcpyDeviceToHost);

    for (uint32_t t = 0; t < sizeQ; ++t) { uint64_t q = mod[t];
        uint64_t nsq = ns % q;
        uint64_t nsPrecon = shoup_precon(nsq, q);
        const uint64_t* vT = vBuf + (size_t)t * n; const uint64_t* e0T = e0Buf + (size_t)t * n;
        const uint64_t* e1T = e1Buf + (size_t)t * n; const uint64_t* mT = mBuf + (size_t)t * n;
        for (uint32_t k = 0; k < n; ++k) {
            uint64_t vPrecon = shoup_precon(vT[k], q);
            uint64_t b = am(shoup_mulmod(pkB[(size_t)t * n + k], vT[k], vPrecon, q),
                             shoup_mulmod(e0T[k], nsq, nsPrecon, q), q);
            uint64_t a = am(shoup_mulmod(pkA[(size_t)t * n + k], vT[k], vPrecon, q),
                             shoup_mulmod(e1T[k], nsq, nsPrecon, q), q);
            c0[(size_t)t * n + k] = am(b, mT[k], q);
            c1[(size_t)t * n + k] = a;
        }
    }
}
} // namespace gpufhe
