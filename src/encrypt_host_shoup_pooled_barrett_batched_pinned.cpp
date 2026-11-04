// GOD PROMPT "CKKS PHASE 15": encrypt_host_shoup_pooled_barrett_batched_pinned
// -- Experiment 2, built on top of the ALREADY-verified Experiment 1
// (encrypt_host_shoup_pooled_barrett_batched.cpp, which merged the
// four separate H2D/D2H transfers into one each). This file adds
// PINNED (page-locked) host staging memory for that one combined
// transfer, replacing the pageable std::vector `packed` buffer.
//
// OUTCOME (see research/TRANSFER_PROFILING.md for the full story):
// this function IS bit-exact correct and DOES deliver a real,
// reproducible 1.06x-1.17x steady-state speedup over Experiment 1 in
// a properly warmed-up benchmark -- but it is NOT wired into
// cpp_bridge/encdec_bridge.cpp and is NOT the production path.
// `cudaMallocHost`'s one-time cost (~3-8ms per distinct (n,sizeQ) key,
// measured directly) exceeds a single encrypt_host call's entire
// typical latency, making this a net LOSS for any usage pattern that
// doesn't repeat enough calls at one key to amortize it -- a
// cold-start risk Experiment 1 does not have. Kept here, fully
// tested, for a future round that wants to revisit it with an
// explicit pre-warming scheme.
//
// JUSTIFICATION FOR BUILDING IT (measured BEFORE implementing, not
// assumed): a standalone microbenchmark comparing pageable vs. pinned
// H2D+D2H round-trip time at the EXACT packed-buffer sizes this
// codebase uses (256KB/2MB/4MB, matching 4*sizeQ*n*8 at
// n=1024..8192,sizeQ=8) showed a real 1.24x-1.42x speedup from pinned
// memory -- large enough to justify the added complexity of a second
// pooled resource.
//
// DESIGN: the packed host buffer is now ALSO pooled (process-lifetime,
// keyed by (n,sizeQ), allocated via cudaMallocHost), alongside the
// existing pooled device buffer -- same lifetime/ownership model,
// same single mutex serializing the whole function (the pinned buffer
// is just as much "shared, mutable state fully overwritten by its
// owning call" as the device buffer already was, so the EXISTING
// concurrency argument from research/EMBED_AND_POOL_HARDENING.md
// carries over unchanged, re-verified here with a dedicated
// concurrency audit test rather than assumed). The embed loop writes
// DIRECTLY into the pinned buffer (no intermediate std::vector); after
// download, the combine loop reads directly from the SAME pinned
// buffer -- zero extra host-side copying versus Experiment 1, only the
// MEMORY TYPE of the existing packed buffer changes.
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
EncDevTab build_enc_tab_pinned(uint32_t n, uint64_t q, uint64_t root) {
    auto br = [](uint32_t v, uint32_t b) { uint32_t r = 0; for (uint32_t k = 0; k < b; k++) { r = (r << 1) | (v & 1); v >>= 1; } return r; };
    uint32_t logn = 0; while ((1u << logn) < n) ++logn;
    std::vector<uint64_t> fr(n), fp(n), pw(n); pw[0] = 1;
    for (uint32_t i = 1; i < n; ++i) pw[i] = mulmod_cold(pw[i - 1], root, q);
    for (uint32_t i = 0; i < n; ++i) { uint64_t r = pw[br(i, logn)]; fr[i] = r; fp[i] = (uint64_t)(((__uint128_t)r << 64) / q); }
    EncDevTab T; T.fr = up(fr); T.fp = up(fp);
    return T;
}
EncDevTab& enc_dev_tab_pinned(uint32_t n, uint64_t q, uint64_t root) {
    static std::map<std::pair<uint32_t, uint64_t>, EncDevTab> cache;
    auto key = std::make_pair(n, q);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    return cache.emplace(key, build_enc_tab_pinned(n, q, root)).first->second;
}

// Pooled state keyed by (n,sizeQ). `h_packed` is PINNED host memory
// (cudaMallocHost) -- same process-lifetime/never-freed pattern as
// every other pool in this codebase, same single mutex protecting
// access (see encrypt_host_shoup_pooled_barrett_batched_pinned's own
// lock below).
struct EncryptBufPoolPinned {
    uint64_t* d_packed = nullptr; // device, [v | e0 | e1 | m]
    uint64_t* h_packed = nullptr; // PINNED host, [v | e0 | e1 | m]
    uint64_t **d_fr = nullptr, **d_fp = nullptr;
    uint64_t* d_mod = nullptr;
    uint32_t sizeQ = 0;
};
EncryptBufPoolPinned& encrypt_buf_pool_pinned(uint32_t n, uint32_t sizeQ) {
    static std::map<std::pair<uint32_t, uint32_t>, EncryptBufPoolPinned> cache;
    auto key = std::make_pair(n, sizeQ);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    EncryptBufPoolPinned P; P.sizeQ = sizeQ;
    const size_t T8 = (size_t)sizeQ * n * 8;
    cudaMalloc(&P.d_packed, 4 * T8);
    cudaMallocHost(&P.h_packed, 4 * T8);
    cudaMalloc(&P.d_fr, sizeQ * sizeof(uint64_t*));
    cudaMalloc(&P.d_fp, sizeQ * sizeof(uint64_t*));
    cudaMalloc(&P.d_mod, sizeQ * 8);
    return cache.emplace(key, P).first->second;
}
} // anon

void encrypt_host_shoup_pooled_barrett_batched_pinned(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1,
                                                        const std::vector<int64_t>& m,
                                                        const std::vector<uint64_t>& pkA, const std::vector<uint64_t>& pkB,
                                                        uint32_t n, const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root,
                                                        uint64_t ns, double sigma, uint64_t seed)
{
    // Serializes the whole call -- identical rationale to Phase
    // 13/14/15's mutex (research/EMBED_AND_POOL_HARDENING.md,
    // research/TRANSFER_PROFILING.md): both the pooled device buffer
    // AND the pooled pinned host buffer are shared, mutable state,
    // always fully overwritten by their owning call before being read.
    static std::mutex pool_mu;
    std::lock_guard<std::mutex> lock(pool_mu);

    uint32_t sizeQ = (uint32_t)mod.size();
    std::mt19937_64 rng(seed); std::normal_distribution<double> g(0.0, sigma);
    std::vector<int> vc(n), e0c(n), e1c(n);
    for (uint32_t k = 0; k < n; ++k) { vc[k] = (int)(rng() % 3) - 1; e0c[k] = (int)std::lround(g(rng)); e1c[k] = (int)std::lround(g(rng)); }
    c0.resize((size_t)sizeQ * n); c1.resize((size_t)sizeQ * n);

    const size_t T = (size_t)sizeQ * n;
    EncryptBufPoolPinned& P = encrypt_buf_pool_pinned(n, sizeQ);
    // Embed writes DIRECTLY into the pinned host buffer -- no
    // intermediate std::vector, no extra copy versus Experiment 1.
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
    for (uint32_t t = 0; t < sizeQ; ++t) { EncDevTab& Tt = enc_dev_tab_pinned(n, mod[t], root[t]); h_fr[t] = Tt.fr; h_fp[t] = Tt.fp; }

    cudaMemcpy(P.d_fr, h_fr.data(), sizeQ * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(P.d_fp, h_fp.data(), sizeQ * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(P.d_mod, mod.data(), sizeQ * 8, cudaMemcpyHostToDevice);

    const size_t T8 = T * 8;
    // ONE upload FROM pinned memory (Experiment 1 already made this
    // one call instead of four; this round changes ONLY the source
    // memory's type from pageable to pinned).
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

    // ONE download INTO pinned memory.
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
