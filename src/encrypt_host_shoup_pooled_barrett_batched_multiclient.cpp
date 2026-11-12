// GOD PROMPT "CKKS PHASE 38": encrypt_host_shoup_pooled_barrett_batched_multiclient
// -- a genuine multi-client BATCHED extension of Phase 15/34's
// encrypt_host_shoup_pooled_barrett_batched (src/
// encrypt_host_shoup_pooled_barrett_batched.cpp), for the real recurring
// shape this project's own aggregation workloads already have: K
// INDEPENDENT clients, each encrypting under the SAME shared public key
// (pkA/pkB), modulus chain (mod/root), and n/sigma/ns, but each drawing
// its OWN independent encryption randomness (v/e0/e1, seeded
// independently per client) and encrypting its OWN plaintext.
//
// WHY THIS IS SAFE TO BATCH (traced from source before writing any code,
// matching this project's own established practice for every prior
// kernel-merge phase):
//   - kernels/ntt_ct.cu's ct_stage_b/ct_shared_tail_b index tower state
//     purely via `blockIdx.y` (`tw`): each tower independently reads/
//     writes its own n-length segment at `x + tw*n`, using
//     `roots[tw]`/`precon[tw]`/`mods[tw]` for THAT tower's own data.
//     Phase 34 already proved (and shipped) that repeating the SAME
//     sizeQ per-tower tables 4 times (for v/e0/e1/m) and widening
//     `towers` from `sizeQ` to `4*sizeQ` produces IDENTICAL per-element
//     results to 4 separate calls -- an exact reorganization, not an
//     approximation. Repeating those SAME 4*sizeQ tables M MORE times
//     (once per client, since every client uses the identical modulus
//     chain and root tables) and widening `towers` again, to
//     `4*sizeQ*M`, is the SAME mathematical argument applied one more
//     time: each client's block of 4*sizeQ towers is fully independent
//     of every other client's block (disjoint `n`-length segments,
//     looked-up tables identical across clients because mod/root are
//     shared, never client-specific).
//   - Verified, not just argued: tests/test_encrypt_multiclient_
//     matches_sequential.cpp compares this function's actual output,
//     client by client, against M independent calls to the existing,
//     already-verified encrypt_host_shoup_pooled_barrett_batched, bit-
//     exact, across multiple (n,sizeQ,M) combinations.
//
// WHAT IS SHARED ACROSS CLIENTS (all PUBLIC CKKS parameters, zero
// confidentiality impact): pkA, pkB, mod, root, n, sigma, ns, and the
// per-tower NTT tables (fr/fp, cached by enc_dev_tab_batched exactly as
// before). WHAT IS NEVER SHARED: each client's own v/e0/e1 ephemeral
// randomness (drawn from an INDEPENDENTLY SEEDED std::mt19937_64 per
// client -- caller supplies one seed per client, exactly matching the
// existing single-client function's own per-call `seed` parameter
// semantics) and each client's own plaintext `m`. No client's
// randomness or plaintext is read, mixed, or reused by any other
// client's computation at any point in this function.
//
// Packed host/device buffer layout: client i's block occupies
// `[i*4*sizeQ*n, (i+1)*4*sizeQ*n)`, laid out internally exactly like the
// single-client function's own `[v|e0|e1|m]` (each sizeQ*n elements) --
// client blocks are simply concatenated, not interleaved, so each
// client's data lives in one contiguous, disjoint region.
//
// Pooled device state is keyed by (n,sizeQ,M) -- a NEW pool, entirely
// separate from the single-client function's own (n,sizeQ)-keyed pool
// (different struct, different static cache, different mutex): the two
// functions share no state and cannot race against each other.
#include "keygen.h"
#include "ntt.h"
#include <cstdint>
#include <vector>
#include <mutex>
#include <map>
#include <tuple>
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
// Same bounded form as Phase 37's production change, duplicated here per
// this project's established convention of duplicating small per-file
// helpers rather than centralizing them -- see src/encrypt_host_shoup_
// pooled_barrett_batched.cpp's own comment and research/benchmarks/
// bench_barrett_reduce_bound_check.cpp for the 70M-trial justification.
inline uint64_t barrett_reduce_u64(uint64_t x, const TowerBarrett& T) {
    uint64_t qhat = (uint64_t)(((unsigned __int128)x * T.mu) >> 64);
    uint64_t r = x - qhat * T.q;
    if (r >= T.q) r -= T.q;
    if (r >= T.q) r -= T.q;
    return r;
}
inline uint64_t emb_barrett(long x, const TowerBarrett& T) {
    uint64_t ax = (x < 0) ? (0ULL - (uint64_t)x) : (uint64_t)x;
    uint64_t r = barrett_reduce_u64(ax, T);
    if (x < 0 && r != 0) r = T.q - r;
    return r;
}

struct EncDevTab { uint64_t* fr = nullptr; uint64_t* fp = nullptr; };
EncDevTab build_enc_tab_multi(uint32_t n, uint64_t q, uint64_t root) {
    auto br = [](uint32_t v, uint32_t b) { uint32_t r = 0; for (uint32_t k = 0; k < b; k++) { r = (r << 1) | (v & 1); v >>= 1; } return r; };
    uint32_t logn = 0; while ((1u << logn) < n) ++logn;
    std::vector<uint64_t> fr(n), fp(n), pw(n); pw[0] = 1;
    for (uint32_t i = 1; i < n; ++i) pw[i] = mulmod_cold(pw[i - 1], root, q);
    for (uint32_t i = 0; i < n; ++i) { uint64_t r = pw[br(i, logn)]; fr[i] = r; fp[i] = (uint64_t)(((__uint128_t)r << 64) / q); }
    EncDevTab T; T.fr = up(fr); T.fp = up(fp);
    return T;
}
// Deliberately SHARES the cache key space (n,q) with the single-client
// function's own enc_dev_tab_batched would use, but is a SEPARATE static
// map/cache instance (per this project's convention of duplicating
// rather than cross-linking small per-file caches) -- both end up
// caching the identical (fr,fp) device arrays for the same (n,q), just
// independently; this costs a few KB of duplicate device memory per
// distinct (n,q) at worst, never a correctness concern (both are
// read-only, process-lifetime, immutable once built).
EncDevTab& enc_dev_tab_multi(uint32_t n, uint64_t q, uint64_t root) {
    static std::map<std::pair<uint32_t, uint64_t>, EncDevTab> cache;
    auto key = std::make_pair(n, q);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    return cache.emplace(key, build_enc_tab_multi(n, q, root)).first->second;
}

// Pooled device state, keyed by (n,sizeQ,M) -- M clients' worth of the
// single-client packed buffer, concatenated. d_fr/d_fp/d_mod are
// 4*sizeQ*M-length: the SAME 4*sizeQ per-tower tables (v/e0/e1/m roles),
// repeated once per client block, built ONCE per (n,sizeQ,M) pool entry,
// not per call.
struct EncryptBufPoolMulti {
    uint64_t* d_packed = nullptr; // M client blocks, each [v|e0|e1|m], sizeQ*n elements per role
    uint64_t **d_fr = nullptr, **d_fp = nullptr; // 4*sizeQ*M-length
    uint64_t* d_mod = nullptr;                   // 4*sizeQ*M-length
    uint32_t sizeQ = 0, M = 0;
};
EncryptBufPoolMulti& encrypt_buf_pool_multi(uint32_t n, uint32_t sizeQ, uint32_t M) {
    static std::map<std::tuple<uint32_t, uint32_t, uint32_t>, EncryptBufPoolMulti> cache;
    auto key = std::make_tuple(n, sizeQ, M);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    EncryptBufPoolMulti P; P.sizeQ = sizeQ; P.M = M;
    const size_t T8 = (size_t)sizeQ * n * 8;
    cudaMalloc(&P.d_packed, (size_t)M * 4 * T8);
    cudaMalloc(&P.d_fr, (size_t)M * 4 * sizeQ * sizeof(uint64_t*));
    cudaMalloc(&P.d_fp, (size_t)M * 4 * sizeQ * sizeof(uint64_t*));
    cudaMalloc(&P.d_mod, (size_t)M * 4 * sizeQ * 8);
    return cache.emplace(key, P).first->second;
}
} // anon

// `m_list`/`seeds` must each have exactly M entries (M = number of
// independent clients in this batch). `c0_out`/`c1_out` are resized to M
// entries, one independent ciphertext per client, in the same order as
// `m_list`. `pkA`/`pkB`/`mod`/`root`/`n`/`sigma`/`ns` are SHARED across
// every client in the batch (the same public key and parameters every
// client in a real aggregation round already uses).
void encrypt_host_shoup_pooled_barrett_batched_multiclient(
    std::vector<std::vector<uint64_t>>& c0_out, std::vector<std::vector<uint64_t>>& c1_out,
    const std::vector<std::vector<int64_t>>& m_list,
    const std::vector<uint64_t>& pkA, const std::vector<uint64_t>& pkB,
    uint32_t n, const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root,
    uint64_t ns, double sigma, const std::vector<uint64_t>& seeds)
{
    const uint32_t M = (uint32_t)m_list.size();
    // Serializes the whole call against other callers of THIS pool --
    // identical rationale to the single-client function's own mutex
    // (research/EMBED_AND_POOL_HARDENING.md): the pooled device buffer
    // is shared, mutable state, always fully overwritten by its owning
    // call before being read. A SEPARATE mutex from the single-client
    // function's own (different pool, no shared state -- see file
    // header comment).
    static std::mutex pool_mu_multi;
    std::lock_guard<std::mutex> lock(pool_mu_multi);

    uint32_t sizeQ = (uint32_t)mod.size();
    const size_t T = (size_t)sizeQ * n;     // elements per role, per client
    const size_t blockElems = 4 * T;         // elements per client block ([v|e0|e1|m])

    c0_out.assign(M, {}); c1_out.assign(M, {});
    for (uint32_t i = 0; i < M; ++i) { c0_out[i].resize(T); c1_out[i].resize(T); }

    // ONE local (call-scoped, not pooled, not shared) packed host
    // buffer covering all M clients -- same "fully overwritten before
    // read" safety argument as the single-client function's own local
    // `packed` vector.
    std::vector<uint64_t> packed((size_t)M * blockElems);

    // Independent per-client RNG + embed. Each client's v/e0/e1 are
    // drawn from an INDEPENDENTLY SEEDED generator (seeds[i]) -- no RNG
    // state, random value, or plaintext is ever shared or reused across
    // clients. This loop is the per-client analog of the single-client
    // function's own RNG+embed section, just writing into client i's
    // disjoint block of the shared packed buffer instead of its own
    // local buffer.
    std::vector<TowerBarrett> towerTabs(sizeQ);
    for (uint32_t t = 0; t < sizeQ; ++t) towerTabs[t] = make_tower_barrett(mod[t]);
    for (uint32_t i = 0; i < M; ++i) {
        std::mt19937_64 rng(seeds[i]);
        std::normal_distribution<double> g(0.0, sigma);
        std::vector<int> vc(n), e0c(n), e1c(n);
        for (uint32_t k = 0; k < n; ++k) {
            vc[k] = (int)(rng() % 3) - 1;
            e0c[k] = (int)std::lround(g(rng));
            e1c[k] = (int)std::lround(g(rng));
        }
        uint64_t* block = packed.data() + (size_t)i * blockElems;
        uint64_t* vBuf = block; uint64_t* e0Buf = block + T; uint64_t* e1Buf = block + 2 * T; uint64_t* mBuf = block + 3 * T;
        const std::vector<int64_t>& mi = m_list[i];
        for (uint32_t t = 0; t < sizeQ; ++t) {
            const TowerBarrett& TB = towerTabs[t];
            uint64_t* vT = vBuf + (size_t)t * n; uint64_t* e0T = e0Buf + (size_t)t * n;
            uint64_t* e1T = e1Buf + (size_t)t * n; uint64_t* mT = mBuf + (size_t)t * n;
            for (uint32_t k = 0; k < n; ++k) {
                vT[k] = emb_barrett(vc[k], TB); e0T[k] = emb_barrett(e0c[k], TB);
                e1T[k] = emb_barrett(e1c[k], TB); mT[k] = emb_barrett((long)mi[k], TB);
            }
        }
    }

    // SHARED per-tower tables (fr/fp/mod), repeated 4*M times -- same
    // tables for every client's every role, since mod/root are shared
    // across the whole batch. This is the direct generalization of the
    // single-client function's own Phase 34 comment: there, the sizeQ
    // tables were repeated 4 times (v/e0/e1/m); here they are repeated
    // 4*M times (M clients, 4 roles each).
    const uint32_t totalTowers = 4 * sizeQ * M;
    std::vector<uint64_t*> h_fr(totalTowers), h_fp(totalTowers);
    std::vector<uint64_t> h_mod(totalTowers);
    for (uint32_t t = 0; t < sizeQ; ++t) {
        EncDevTab& Tt = enc_dev_tab_multi(n, mod[t], root[t]);
        for (uint32_t i = 0; i < M; ++i) {
            for (uint32_t r = 0; r < 4; ++r) {
                uint32_t idx = (i * 4 + r) * sizeQ + t;
                h_fr[idx] = Tt.fr; h_fp[idx] = Tt.fp; h_mod[idx] = mod[t];
            }
        }
    }

    EncryptBufPoolMulti& P = encrypt_buf_pool_multi(n, sizeQ, M);
    cudaMemcpy(P.d_fr, h_fr.data(), (size_t)totalTowers * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(P.d_fp, h_fp.data(), (size_t)totalTowers * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(P.d_mod, h_mod.data(), (size_t)totalTowers * 8, cudaMemcpyHostToDevice);

    const size_t totalBytes = (size_t)M * blockElems * 8;
    // ONE upload for the WHOLE batch (was M separate uploads, one per
    // client, in the sequential path).
    cudaMemcpy(P.d_packed, packed.data(), totalBytes, cudaMemcpyHostToDevice);

    // ONE kernel-launch sequence for the WHOLE batch (towers=4*sizeQ*M)
    // -- was M separate launch sequences (each towers=4*sizeQ) in the
    // sequential path. See the file header comment for why this is an
    // exact reorganization, not an approximation.
    LaunchNTT_CT_Batched(P.d_packed, (const uint64_t* const*)P.d_fr, (const uint64_t* const*)P.d_fp, P.d_mod, totalTowers, n, 0);
    cudaDeviceSynchronize();

    // ONE download for the WHOLE batch (was M separate downloads).
    cudaMemcpy(packed.data(), P.d_packed, totalBytes, cudaMemcpyDeviceToHost);

    // Independent per-client combine. pkA/pkB are SHARED (same array for
    // every client, exactly as in the single-client function); each
    // client's own v/e0/e1/m block produces that client's own,
    // independent (c0,c1).
    for (uint32_t i = 0; i < M; ++i) {
        const uint64_t* block = packed.data() + (size_t)i * blockElems;
        const uint64_t* vBuf = block; const uint64_t* e0Buf = block + T; const uint64_t* e1Buf = block + 2 * T; const uint64_t* mBuf = block + 3 * T;
        std::vector<uint64_t>& c0 = c0_out[i]; std::vector<uint64_t>& c1 = c1_out[i];
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
}
} // namespace gpufhe
