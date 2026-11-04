// GOD PROMPT "CKKS PHASE 16": explicit pinned-memory PREWARMING for
// encrypt_host, decoupling the ~3-8ms per-(n,sizeQ) `cudaMallocHost`
// cost (measured in Phase 15, research/TRANSFER_PROFILING.md) from the
// first real encryption call. Builds on, and does not modify, Phase
// 15's two implementations: `encrypt_host_shoup_pooled_barrett_batched`
// (pageable, production) and `encrypt_host_shoup_pooled_barrett_
// batched_pinned` (pinned, lazy-allocated, NOT production).
//
// LIFECYCLE (designed before implementation, full rationale in
// research/TRANSFER_PROFILING.md):
//   1. `encrypt_prewarm_pinned_buffer(n, sizeQ)` -- explicit, opt-in,
//      idempotent. Allocates (or confirms already-allocated) the
//      pinned host + device packed buffers for this (n,sizeQ). Returns
//      true iff a valid, fully-initialized buffer is now available.
//      A partial allocation failure is cleaned up immediately and
//      NEVER left half-initialized or marked ready -- the directive's
//      explicit "a failed prewarm must never leave a partially
//      initialized or unsafe buffer available to encryption"
//      requirement. This is the ONE place in this file (and in this
//      round) where CUDA return codes ARE checked, a deliberate,
//      narrowly-scoped exception to this codebase's established
//      "don't check CUDA errors" convention, justified because this
//      function's entire contract is about failure-safety.
//   2. `encrypt_pinned_buffer_is_ready(n, sizeQ)` -- query-only,
//      read-only, never allocates.
//   3. `encrypt_host_shoup_pooled_barrett_prewarmed(...)` -- the
//      actual encrypt call. Checks readiness FIRST; if and only if a
//      prewarmed buffer exists, uses the pinned path (identical
//      embed/combine arithmetic to encrypt_host_shoup_pooled_barrett_
//      batched_pinned). Otherwise it TRANSPARENTLY calls the existing,
//      unmodified, already-production `encrypt_host_shoup_pooled_
//      barrett_batched` (pageable) -- never attempting a lazy pinned
//      allocation itself, so an application that never calls
//      `encrypt_prewarm_pinned_buffer` pays NOTHING extra and behaves
//      EXACTLY like today's production path.
//
// OWNERSHIP MODEL (directive Step 2's explicit question): the pinned
// buffer is owned by a SHARED, PROCESS-LEVEL pool keyed by (n,sizeQ) --
// the SAME model every other cache in this codebase already uses
// (enc_dev_tab/ek_dev_tab/KROT/KSC, and Phase 13-15's own device-
// buffer pools). This C++ layer has no "context" object of its own
// (Rust's EvalMultContext/KeySwitchContext are a DIFFERENT crate layer
// that never touches encrypt_host) -- introducing a new per-context
// ownership model here would be a bigger, riskier, and less consistent
// change than reusing the architecture that already exists, which is
// exactly what the directive's "do not introduce a new global
// ownership model merely for convenience" steers away from in the
// OTHER direction (toward reusing what exists, not inventing a parallel
// one). Multiple "contexts" (callers) at the same (n,sizeQ) safely
// SHARE one prewarmed buffer, identically to how Phase 13-15's device
// pools are already shared and already proven safe under concurrency.
//
// No explicit "destroy a prewarmed buffer" API exists in this round --
// prewarmed buffers, like every other pool in this codebase, live for
// the process lifetime. This is a deliberate, documented scope
// decision (not an oversight): adding reference-counted teardown while
// other callers might be concurrently using the SAME buffer is a
// materially larger design question than this phase's mission, and
// every other resource in this codebase already makes the same
// "never freed, bounded by distinct keys actually used" tradeoff.
#include "keygen.h"
#include "ntt.h"
#include <cstdint>
#include <vector>
#include <mutex>
#include <map>
#include <utility>
#include <random>
#include <cmath>
#include <cuda_runtime.h>

namespace gpufhe {
// The existing, unmodified, production pageable implementation
// (src/encrypt_host_shoup_pooled_barrett_batched.cpp) -- called
// directly as the fallback path, never reimplemented here.
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
EncDevTab build_enc_tab_prewarm(uint32_t n, uint64_t q, uint64_t root) {
    auto br = [](uint32_t v, uint32_t b) { uint32_t r = 0; for (uint32_t k = 0; k < b; k++) { r = (r << 1) | (v & 1); v >>= 1; } return r; };
    uint32_t logn = 0; while ((1u << logn) < n) ++logn;
    std::vector<uint64_t> fr(n), fp(n), pw(n); pw[0] = 1;
    for (uint32_t i = 1; i < n; ++i) pw[i] = mulmod_cold(pw[i - 1], root, q);
    for (uint32_t i = 0; i < n; ++i) { uint64_t r = pw[br(i, logn)]; fr[i] = r; fp[i] = (uint64_t)(((__uint128_t)r << 64) / q); }
    EncDevTab T; T.fr = up(fr); T.fp = up(fp);
    return T;
}
EncDevTab& enc_dev_tab_prewarm(uint32_t n, uint64_t q, uint64_t root) {
    static std::map<std::pair<uint32_t, uint64_t>, EncDevTab> cache;
    auto key = std::make_pair(n, q);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    return cache.emplace(key, build_enc_tab_prewarm(n, q, root)).first->second;
}

// Pool entry. `ready` is the explicit success flag -- an entry is only
// ever inserted into the cache AFTER allocation either fully succeeded
// (ready=true) or was fully rolled back on failure (ready=false, all
// pointers null). There is no intermediate state a caller can observe.
struct PrewarmEntry {
    uint64_t* d_packed = nullptr;
    uint64_t* h_packed = nullptr; // pinned
    uint64_t **d_fr = nullptr, **d_fp = nullptr;
    uint64_t* d_mod = nullptr;
    uint32_t sizeQ = 0;
    bool ready = false;
};
using PrewarmKey = std::pair<uint32_t, uint32_t>; // (n, sizeQ)

// ONE mutex, shared by prewarm/query/encrypt, serializing all access to
// the pool map AND (while actually using a ready buffer) its contents --
// identical rationale to every prior phase's single-mutex design
// (research/EMBED_AND_POOL_HARDENING.md's audit, re-applied here rather
// than assumed).
std::mutex& prewarm_mutex() { static std::mutex m; return m; }
std::map<PrewarmKey, PrewarmEntry>& prewarm_cache() {
    static std::map<PrewarmKey, PrewarmEntry> cache;
    return cache;
}

// Rolls back whatever subset of allocations succeeded. Never called on
// an entry that will be marked ready.
void free_partial(PrewarmEntry& P) {
    if (P.d_packed) { cudaFree(P.d_packed); P.d_packed = nullptr; }
    if (P.h_packed) { cudaFreeHost(P.h_packed); P.h_packed = nullptr; }
    if (P.d_fr) { cudaFree(P.d_fr); P.d_fr = nullptr; }
    if (P.d_fp) { cudaFree(P.d_fp); P.d_fp = nullptr; }
    if (P.d_mod) { cudaFree(P.d_mod); P.d_mod = nullptr; }
}
} // anon

// Explicit, opt-in, idempotent prewarm entry point. Must be called with
// prewarm_mutex() held by the caller (both public functions below take
// the lock themselves; this one is the shared implementation).
static bool prewarm_locked(uint32_t n, uint32_t sizeQ) {
    auto& cache = prewarm_cache();
    PrewarmKey key(n, sizeQ);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second.ready; // idempotent: report existing status, don't re-allocate

    PrewarmEntry P; P.sizeQ = sizeQ;
    const size_t T8 = (size_t)sizeQ * n * 8;
    bool ok = true;
    if (ok && cudaMalloc(&P.d_packed, 4 * T8) != cudaSuccess) ok = false;
    if (ok && cudaMallocHost(&P.h_packed, 4 * T8) != cudaSuccess) ok = false;
    if (ok && cudaMalloc(&P.d_fr, sizeQ * sizeof(uint64_t*)) != cudaSuccess) ok = false;
    if (ok && cudaMalloc(&P.d_fp, sizeQ * sizeof(uint64_t*)) != cudaSuccess) ok = false;
    if (ok && cudaMalloc(&P.d_mod, sizeQ * 8) != cudaSuccess) ok = false;

    if (!ok) { free_partial(P); P.ready = false; }
    else P.ready = true;

    cache.emplace(key, P);
    return P.ready;
}

void encrypt_prewarm_pinned_buffer_impl(uint32_t n, uint32_t sizeQ, bool* out_ready) {
    std::lock_guard<std::mutex> lock(prewarm_mutex());
    *out_ready = prewarm_locked(n, sizeQ);
}
bool encrypt_prewarm_pinned_buffer(uint32_t n, uint32_t sizeQ) {
    bool ready = false;
    encrypt_prewarm_pinned_buffer_impl(n, sizeQ, &ready);
    return ready;
}

bool encrypt_pinned_buffer_is_ready(uint32_t n, uint32_t sizeQ) {
    std::lock_guard<std::mutex> lock(prewarm_mutex());
    auto& cache = prewarm_cache();
    auto it = cache.find(PrewarmKey(n, sizeQ));
    return it != cache.end() && it->second.ready;
}

// Adaptive encrypt: uses the prewarmed pinned buffer ONLY if one is
// ready for this (n,sizeQ); otherwise transparently defers to the
// unmodified production pageable path. Never allocates pinned memory
// itself -- that only ever happens via an explicit
// encrypt_prewarm_pinned_buffer() call made by the caller ahead of time.
void encrypt_host_shoup_pooled_barrett_prewarmed(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1,
                                                  const std::vector<int64_t>& m,
                                                  const std::vector<uint64_t>& pkA, const std::vector<uint64_t>& pkB,
                                                  uint32_t n, const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root,
                                                  uint64_t ns, double sigma, uint64_t seed)
{
    uint32_t sizeQ = (uint32_t)mod.size();

    std::unique_lock<std::mutex> lock(prewarm_mutex());
    auto& cache = prewarm_cache();
    auto it = cache.find(PrewarmKey(n, sizeQ));
    bool usePinned = (it != cache.end()) && it->second.ready;

    if (!usePinned) {
        // Release our lock before deferring -- the fallback has its own,
        // entirely separate mutex and owes nothing to this pool.
        lock.unlock();
        encrypt_host_shoup_pooled_barrett_batched(c0, c1, m, pkA, pkB, n, mod, root, ns, sigma, seed);
        return;
    }

    // Keep the lock held for the remainder of this call: a concurrent
    // thread must not be able to observe this entry mid-use (the SAME
    // guarantee Phase 13-15's single-mutex pools already provide).
    PrewarmEntry& P = it->second;

    std::mt19937_64 rng(seed); std::normal_distribution<double> g(0.0, sigma);
    std::vector<int> vc(n), e0c(n), e1c(n);
    for (uint32_t k = 0; k < n; ++k) { vc[k] = (int)(rng() % 3) - 1; e0c[k] = (int)std::lround(g(rng)); e1c[k] = (int)std::lround(g(rng)); }
    c0.resize((size_t)sizeQ * n); c1.resize((size_t)sizeQ * n);

    const size_t T = (size_t)sizeQ * n;
    uint64_t* vBuf = P.h_packed;
    uint64_t* e0Buf = P.h_packed + T;
    uint64_t* e1Buf = P.h_packed + 2 * T;
    uint64_t* mBuf = P.h_packed + 3 * T;
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
    for (uint32_t t = 0; t < sizeQ; ++t) { EncDevTab& Tt = enc_dev_tab_prewarm(n, mod[t], root[t]); h_fr[t] = Tt.fr; h_fp[t] = Tt.fp; }

    cudaMemcpy(P.d_fr, h_fr.data(), sizeQ * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(P.d_fp, h_fp.data(), sizeQ * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(P.d_mod, mod.data(), sizeQ * 8, cudaMemcpyHostToDevice);

    const size_t T8 = T * 8;
    cudaMemcpy(P.d_packed, P.h_packed, 4 * T8, cudaMemcpyHostToDevice);

    uint64_t* d_v = P.d_packed;
    uint64_t* d_e0 = P.d_packed + T;
    uint64_t* d_e1 = P.d_packed + 2 * T;
    uint64_t* d_m = P.d_packed + 3 * T;
    LaunchNTT_CT_Batched(d_v, (const uint64_t* const*)P.d_fr, (const uint64_t* const*)P.d_fp, P.d_mod, sizeQ, n, 0);
    LaunchNTT_CT_Batched(d_e0, (const uint64_t* const*)P.d_fr, (const uint64_t* const*)P.d_fp, P.d_mod, sizeQ, n, 0);
    LaunchNTT_CT_Batched(d_e1, (const uint64_t* const*)P.d_fr, (const uint64_t* const*)P.d_fp, P.d_mod, sizeQ, n, 0);
    LaunchNTT_CT_Batched(d_m, (const uint64_t* const*)P.d_fr, (const uint64_t* const*)P.d_fp, P.d_mod, sizeQ, n, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(P.h_packed, P.d_packed, 4 * T8, cudaMemcpyDeviceToHost);

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
