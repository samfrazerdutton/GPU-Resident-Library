// GOD PROMPT "CKKS PHASE 14": encrypt_host_shoup_pooled_barrett --
// builds on encrypt_host_shoup_pooled (src/
// encrypt_host_shoup_pooled.cpp, Phase 13's Shoup-combine +
// pooled-buffer version, already bit-exact-verified and mutex-hardened
// against the untouched production encrypt_host) by ALSO replacing the
// embedding loop's signed-64-bit `emb(x) = (x%q+q)%q` -- confirmed via
// `nm`/assembly inspection to compile to TWO hardware `idiv`
// instructions per call (`src/encrypt_host_shoup_pooled.cpp`'s own
// `emb()` lambda; research/ENCRYPT_PROFILING.md measured this loop at
// 20-27% of total time, the single largest stage) -- with a Barrett
// reduction precomputed ONCE PER TOWER (not per element, since the
// modulus is the same for all n coefficients within a tower).
//
// DERIVATION (full detail: research/EMBED_AND_POOL_HARDENING.md).
// `emb(x)` reduces a signed int64_t `x` into `[0,q)`. This file
// replaces it with: (1) compute `ax = |x|` via unsigned negation
// (`ax = (x<0) ? (0ULL-(uint64_t)x) : (uint64_t)x`), which is
// well-defined (no signed-overflow UB) for EVERY int64_t including
// LONG_MIN, since the magnitude of LONG_MIN (2^63) fits in uint64_t
// even though it doesn't fit in int64_t; (2) Barrett-reduce `ax mod q`
// using a per-tower precomputed `mu = floor(2^64/q)` (one division per
// TOWER, not per coefficient -- q is always a ~49-60 bit CKKS modulus
// in this codebase, so `mu` is small and the one-shot Barrett estimate
// needs at most ONE corrective subtraction, confirmed empirically
// across 22M+ differential-test cases, never assumed); (3) restore the
// sign using `emb()`'s own convention (x<0 with nonzero remainder r
// maps to q-r, matching `(x%q+q)%q`'s behavior for negative x exactly).
// Verified bit-exact against the original `emb()` lambda across EVERY
// LONG_MIN/LONG_MAX/zero/modulus-boundary case plus 12M+ random
// full-int64_t-range values against both synthetic and this project's
// OWN real `native_primes()`-generated moduli at every supported n
// (`tests/test_embed_barrett_matches_reference.cpp`) -- zero tolerance,
// zero mismatches, this is exact integer arithmetic.
//
// Everything else (RNG, Gaussian/ternary sampling, the Shoup-based
// combine loop, the pooled device buffers, the mutex serializing the
// whole call) is copied verbatim from encrypt_host_shoup_pooled.cpp --
// only the embedding loop's arithmetic changes.
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

// Per-tower Barrett constant for the embedding loop -- see file header
// derivation. `mu` is computed ONCE per tower (O(sizeQ) divisions per
// call, not O(sizeQ*n)); every one of the n coefficients in that tower
// then reuses it via cheap multiplies only.
struct TowerBarrett { uint64_t q; uint64_t mu; };
inline TowerBarrett make_tower_barrett(uint64_t q) {
    TowerBarrett T; T.q = q;
    T.mu = (uint64_t)(((unsigned __int128)1 << 64) / q);
    return T;
}
inline uint64_t barrett_reduce_u64(uint64_t x, const TowerBarrett& T) {
    uint64_t qhat = (uint64_t)(((unsigned __int128)x * T.mu) >> 64);
    uint64_t r = x - qhat * T.q;
    while (r >= T.q) r -= T.q; // proven (22M+ cases) to need at most 1 iteration for this codebase's modulus sizes
    return r;
}
inline uint64_t emb_barrett(long x, const TowerBarrett& T) {
    uint64_t ax = (x < 0) ? (0ULL - (uint64_t)x) : (uint64_t)x;
    uint64_t r = barrett_reduce_u64(ax, T);
    if (x < 0 && r != 0) r = T.q - r;
    return r;
}

struct EncDevTab { uint64_t* fr = nullptr; uint64_t* fp = nullptr; };
EncDevTab build_enc_tab_pooled_barrett(uint32_t n, uint64_t q, uint64_t root) {
    auto br = [](uint32_t v, uint32_t b) { uint32_t r = 0; for (uint32_t k = 0; k < b; k++) { r = (r << 1) | (v & 1); v >>= 1; } return r; };
    uint32_t logn = 0; while ((1u << logn) < n) ++logn;
    std::vector<uint64_t> fr(n), fp(n), pw(n); pw[0] = 1;
    for (uint32_t i = 1; i < n; ++i) pw[i] = mulmod_cold(pw[i - 1], root, q);
    for (uint32_t i = 0; i < n; ++i) { uint64_t r = pw[br(i, logn)]; fr[i] = r; fp[i] = (uint64_t)(((__uint128_t)r << 64) / q); }
    EncDevTab T; T.fr = up(fr); T.fp = up(fp);
    return T;
}
EncDevTab& enc_dev_tab_pooled_barrett(uint32_t n, uint64_t q, uint64_t root) {
    static std::map<std::pair<uint32_t, uint64_t>, EncDevTab> cache;
    auto key = std::make_pair(n, q);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    return cache.emplace(key, build_enc_tab_pooled_barrett(n, q, root)).first->second;
}

struct EncryptBufPool {
    uint64_t *d_v = nullptr, *d_e0 = nullptr, *d_e1 = nullptr, *d_m = nullptr;
    uint64_t **d_fr = nullptr, **d_fp = nullptr;
    uint64_t *d_mod = nullptr;
    uint32_t sizeQ = 0;
};
EncryptBufPool& encrypt_buf_pool_barrett(uint32_t n, uint32_t sizeQ) {
    static std::map<std::pair<uint32_t, uint32_t>, EncryptBufPool> cache;
    auto key = std::make_pair(n, sizeQ);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    EncryptBufPool P; P.sizeQ = sizeQ;
    const size_t T8 = (size_t)sizeQ * n * 8;
    cudaMalloc(&P.d_v, T8); cudaMalloc(&P.d_e0, T8); cudaMalloc(&P.d_e1, T8); cudaMalloc(&P.d_m, T8);
    cudaMalloc(&P.d_fr, sizeQ * sizeof(uint64_t*));
    cudaMalloc(&P.d_fp, sizeQ * sizeof(uint64_t*));
    cudaMalloc(&P.d_mod, sizeQ * 8);
    return cache.emplace(key, P).first->second;
}
} // anon

void encrypt_host_shoup_pooled_barrett(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1,
                                        const std::vector<int64_t>& m,
                                        const std::vector<uint64_t>& pkA, const std::vector<uint64_t>& pkB,
                                        uint32_t n, const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root,
                                        uint64_t ns, double sigma, uint64_t seed)
{
    // Serializes the whole call -- identical rationale to
    // encrypt_host_shoup_pooled.cpp's own mutex (see that file's
    // CONCURRENCY NOTE and research/EMBED_AND_POOL_HARDENING.md's
    // audit): the pooled device buffers are shared, mutable state.
    static std::mutex pool_mu;
    std::lock_guard<std::mutex> lock(pool_mu);

    uint32_t sizeQ = (uint32_t)mod.size();
    std::mt19937_64 rng(seed); std::normal_distribution<double> g(0.0, sigma);
    std::vector<int> vc(n), e0c(n), e1c(n);
    for (uint32_t k = 0; k < n; ++k) { vc[k] = (int)(rng() % 3) - 1; e0c[k] = (int)std::lround(g(rng)); e1c[k] = (int)std::lround(g(rng)); }
    c0.resize((size_t)sizeQ * n); c1.resize((size_t)sizeQ * n);

    std::vector<uint64_t> vBuf((size_t)sizeQ * n), e0Buf((size_t)sizeQ * n), e1Buf((size_t)sizeQ * n), mBuf((size_t)sizeQ * n);
    for (uint32_t t = 0; t < sizeQ; ++t) { uint64_t q = mod[t];
        TowerBarrett TB = make_tower_barrett(q); // ONE division here, reused for all n coefficients below
        uint64_t* vT = vBuf.data() + (size_t)t * n; uint64_t* e0T = e0Buf.data() + (size_t)t * n;
        uint64_t* e1T = e1Buf.data() + (size_t)t * n; uint64_t* mT = mBuf.data() + (size_t)t * n;
        for (uint32_t k = 0; k < n; ++k) {
            vT[k] = emb_barrett(vc[k], TB); e0T[k] = emb_barrett(e0c[k], TB);
            e1T[k] = emb_barrett(e1c[k], TB); mT[k] = emb_barrett((long)m[k], TB);
        }
    }

    std::vector<uint64_t*> h_fr(sizeQ), h_fp(sizeQ);
    for (uint32_t t = 0; t < sizeQ; ++t) { EncDevTab& T = enc_dev_tab_pooled_barrett(n, mod[t], root[t]); h_fr[t] = T.fr; h_fp[t] = T.fp; }

    EncryptBufPool& P = encrypt_buf_pool_barrett(n, sizeQ);
    cudaMemcpy(P.d_fr, h_fr.data(), sizeQ * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(P.d_fp, h_fp.data(), sizeQ * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(P.d_mod, mod.data(), sizeQ * 8, cudaMemcpyHostToDevice);

    const size_t T8 = (size_t)sizeQ * n * 8;
    cudaMemcpy(P.d_v, vBuf.data(), T8, cudaMemcpyHostToDevice);
    cudaMemcpy(P.d_e0, e0Buf.data(), T8, cudaMemcpyHostToDevice);
    cudaMemcpy(P.d_e1, e1Buf.data(), T8, cudaMemcpyHostToDevice);
    cudaMemcpy(P.d_m, mBuf.data(), T8, cudaMemcpyHostToDevice);
    LaunchNTT_CT_Batched(P.d_v, (const uint64_t* const*)P.d_fr, (const uint64_t* const*)P.d_fp, P.d_mod, sizeQ, n, 0);
    LaunchNTT_CT_Batched(P.d_e0, (const uint64_t* const*)P.d_fr, (const uint64_t* const*)P.d_fp, P.d_mod, sizeQ, n, 0);
    LaunchNTT_CT_Batched(P.d_e1, (const uint64_t* const*)P.d_fr, (const uint64_t* const*)P.d_fp, P.d_mod, sizeQ, n, 0);
    LaunchNTT_CT_Batched(P.d_m, (const uint64_t* const*)P.d_fr, (const uint64_t* const*)P.d_fp, P.d_mod, sizeQ, n, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(vBuf.data(), P.d_v, T8, cudaMemcpyDeviceToHost);
    cudaMemcpy(e0Buf.data(), P.d_e0, T8, cudaMemcpyDeviceToHost);
    cudaMemcpy(e1Buf.data(), P.d_e1, T8, cudaMemcpyDeviceToHost);
    cudaMemcpy(mBuf.data(), P.d_m, T8, cudaMemcpyDeviceToHost);

    for (uint32_t t = 0; t < sizeQ; ++t) { uint64_t q = mod[t];
        uint64_t nsq = ns % q;
        uint64_t nsPrecon = shoup_precon(nsq, q);
        const uint64_t* vT = vBuf.data() + (size_t)t * n; const uint64_t* e0T = e0Buf.data() + (size_t)t * n;
        const uint64_t* e1T = e1Buf.data() + (size_t)t * n; const uint64_t* mT = mBuf.data() + (size_t)t * n;
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
