// GOD PROMPT "CKKS PHASE 15": encrypt_host_shoup_pooled_barrett_batched
// -- builds on encrypt_host_shoup_pooled_barrett (src/
// encrypt_host_shoup_pooled_barrett.cpp, Phase 14's Barrett-embed +
// Shoup-combine + pooled-buffer version, already bit-exact-verified and
// mutex-hardened) by ALSO merging the four separate per-buffer H2D/D2H
// cudaMemcpy calls (for d_v,d_e0,d_e1,d_m) into ONE H2D and ONE D2H
// call each, over a single contiguous packed region.
//
// DATA FLOW TRACED BEFORE CHANGING ANYTHING (full detail: research/
// TRANSFER_PROFILING.md):
//   - vBuf/e0Buf/e1Buf/mBuf (host, PAGEABLE std::vector<uint64_t>,
//     sizeQ*n elements each, 8-byte aligned by the default allocator)
//     are each uploaded via their OWN blocking cudaMemcpy into their
//     OWN pooled device buffer (d_v/d_e0/d_e1/d_m, each a SEPARATE
//     cudaMalloc of sizeQ*n*8 bytes, 256-byte aligned per CUDA's own
//     guarantee), then each run through its OWN LaunchNTT_CT_Batched
//     call (same sizeQ towers, same d_fr/d_fp/d_mod tables), then each
//     downloaded via its OWN blocking cudaMemcpy back to its OWN host
//     buffer.
//   - All four buffers are the SAME size, use the SAME per-tower
//     modulus/root tables, and are processed by IDENTICAL kernel calls
//     differing only in which buffer pointer is passed -- this is
//     exactly the "same per-element size, same lifetime, same transfer
//     pattern" condition the governing directive asks to confirm
//     before packing buffers together, and it holds here.
//
// NEW DESIGN (directive Section 5, option B/E: one contiguous staging
// buffer + fewer copy calls, WITHOUT changing the kernel-launch
// pattern): the four pooled device buffers are replaced by ONE pooled
// device buffer `d_packed` of size 4*sizeQ*n*8 bytes, laid out as
// [v | e0 | e1 | m] (each sizeQ*n elements, contiguous, no padding --
// uint64_t's natural alignment means every sub-region starts already
// 8-byte aligned, and CUDA's own 256-byte device-allocation alignment
// covers the whole buffer). The host side builds ONE local packed
// vector the SAME way, then uploads it with ONE cudaMemcpy and
// downloads it with ONE cudaMemcpy. The FOUR LaunchNTT_CT_Batched
// calls are UNCHANGED (same signature, same d_fr/d_fp/d_mod tables,
// same sizeQ towers) -- they simply now receive pointer OFFSETS into
// the one packed device buffer (d_packed, d_packed+sizeQ*n,
// d_packed+2*sizeQ*n, d_packed+3*sizeQ*n) instead of four separate
// buffers. This keeps kernel-launch-level risk at zero (no change to
// what the GPU computes or how) while cutting transfer CALL COUNT from
// 4+4=8 down to 1+1=2, with the SAME total bytes moved.
//
// The host-side packed buffer is a LOCAL (call-scoped) std::vector --
// NOT pooled, NOT shared across calls -- so it introduces no new
// concurrency surface at all. The DEVICE-side packed buffer IS pooled
// (same (n,sizeQ)-keyed, process-lifetime pattern as Phase 13/14), and
// is protected by the SAME single mutex that already serializes this
// whole function -- no new synchronization primitive, no change to the
// existing, already-hardened concurrency model (research/
// EMBED_AND_POOL_HARDENING.md's audit conclusions carry over
// unchanged, since the mutex still covers the entire function body and
// the packed buffer's content is, like before, fully overwritten by
// every call before being read).
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
EncDevTab build_enc_tab_batched(uint32_t n, uint64_t q, uint64_t root) {
    auto br = [](uint32_t v, uint32_t b) { uint32_t r = 0; for (uint32_t k = 0; k < b; k++) { r = (r << 1) | (v & 1); v >>= 1; } return r; };
    uint32_t logn = 0; while ((1u << logn) < n) ++logn;
    std::vector<uint64_t> fr(n), fp(n), pw(n); pw[0] = 1;
    for (uint32_t i = 1; i < n; ++i) pw[i] = mulmod_cold(pw[i - 1], root, q);
    for (uint32_t i = 0; i < n; ++i) { uint64_t r = pw[br(i, logn)]; fr[i] = r; fp[i] = (uint64_t)(((__uint128_t)r << 64) / q); }
    EncDevTab T; T.fr = up(fr); T.fp = up(fp);
    return T;
}
EncDevTab& enc_dev_tab_batched(uint32_t n, uint64_t q, uint64_t root) {
    static std::map<std::pair<uint32_t, uint64_t>, EncDevTab> cache;
    auto key = std::make_pair(n, q);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    return cache.emplace(key, build_enc_tab_batched(n, q, root)).first->second;
}

// Pooled device state, keyed by (n,sizeQ) -- same lifetime/ownership
// model as Phase 13/14's EncryptBufPool, EXCEPT the four separate
// per-call data buffers are now ONE packed allocation of 4x the size,
// laid out [v|e0|e1|m].
//
// GOD PROMPT "CKKS PHASE 34": d_fr/d_fp/d_mod are now 4*sizeQ-length,
// not sizeQ-length -- holding the SAME sizeQ per-tower tables repeated
// 4 times (v's towers, then e0's, then e1's, then m's), so the four
// separate LaunchNTT_CT_Batched calls below (one per buffer, towers=
// sizeQ each -- 20 total kernel launches: 4 calls x ~5 internal stage
// launches each) can be replaced by ONE call over the whole packed
// buffer with towers=4*sizeQ (5 total kernel launches). This is an
// EXACT reorganization of identical work, not an approximation --
// verified bit-exact (0 differing limbs) against the original
// four-separate-calls pattern in a standalone, CUDA-only probe before
// this change (research/benchmarks/bench_ntt_merge_experiment.cpp),
// by direct reading of kernels/ntt_ct.cu's ct_stage_b/ct_shared_tail_b
// (tower index is blockIdx.y; each tower independently indexes
// x+tw*n, roots[tw], precon[tw], mods[tw] -- so repeating the real
// per-tower tables 4 times and widening the grid's y-dimension from
// sizeQ to 4*sizeQ produces identical per-element results). Measured
// ~1.2x on the transfer+kernel portion via interleaved A/B (same
// methodology as every prior encrypt-path phase); full detail,
// evidence, and the end-to-end aggregation-workload re-measurement:
// research/PHASE34_PERFORMANCE_INVESTIGATION.md. The tables are built
// ONCE per (n,sizeQ) pool entry below, not per call -- this adds no
// new per-call host-side work.
struct EncryptBufPoolBatched {
    uint64_t* d_packed = nullptr; // [v | e0 | e1 | m], each sizeQ*n elements
    uint64_t **d_fr = nullptr, **d_fp = nullptr; // 4*sizeQ-length (Phase 34)
    uint64_t* d_mod = nullptr;                   // 4*sizeQ-length (Phase 34)
    uint32_t sizeQ = 0;
};
EncryptBufPoolBatched& encrypt_buf_pool_batched(uint32_t n, uint32_t sizeQ) {
    static std::map<std::pair<uint32_t, uint32_t>, EncryptBufPoolBatched> cache;
    auto key = std::make_pair(n, sizeQ);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    EncryptBufPoolBatched P; P.sizeQ = sizeQ;
    const size_t T8 = (size_t)sizeQ * n * 8;
    cudaMalloc(&P.d_packed, 4 * T8);
    cudaMalloc(&P.d_fr, 4 * sizeQ * sizeof(uint64_t*));
    cudaMalloc(&P.d_fp, 4 * sizeQ * sizeof(uint64_t*));
    cudaMalloc(&P.d_mod, 4 * sizeQ * 8);
    return cache.emplace(key, P).first->second;
}
} // anon

void encrypt_host_shoup_pooled_barrett_batched(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1,
                                                const std::vector<int64_t>& m,
                                                const std::vector<uint64_t>& pkA, const std::vector<uint64_t>& pkB,
                                                uint32_t n, const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root,
                                                uint64_t ns, double sigma, uint64_t seed)
{
    // Serializes the whole call -- identical rationale to Phase 13/14's
    // mutex (research/EMBED_AND_POOL_HARDENING.md's audit): the pooled
    // device buffer is shared, mutable state, always fully overwritten
    // by its owning call before being read.
    static std::mutex pool_mu;
    std::lock_guard<std::mutex> lock(pool_mu);

    uint32_t sizeQ = (uint32_t)mod.size();
    std::mt19937_64 rng(seed); std::normal_distribution<double> g(0.0, sigma);
    std::vector<int> vc(n), e0c(n), e1c(n);
    for (uint32_t k = 0; k < n; ++k) { vc[k] = (int)(rng() % 3) - 1; e0c[k] = (int)std::lround(g(rng)); e1c[k] = (int)std::lround(g(rng)); }
    c0.resize((size_t)sizeQ * n); c1.resize((size_t)sizeQ * n);

    // ONE local (call-scoped, not pooled, not shared) packed host
    // buffer: [v | e0 | e1 | m], each sizeQ*n elements, contiguous.
    const size_t T = (size_t)sizeQ * n;
    std::vector<uint64_t> packed(4 * T);
    uint64_t* vBuf = packed.data();
    uint64_t* e0Buf = packed.data() + T;
    uint64_t* e1Buf = packed.data() + 2 * T;
    uint64_t* mBuf = packed.data() + 3 * T;
    for (uint32_t t = 0; t < sizeQ; ++t) { uint64_t q = mod[t];
        TowerBarrett TB = make_tower_barrett(q);
        uint64_t* vT = vBuf + (size_t)t * n; uint64_t* e0T = e0Buf + (size_t)t * n;
        uint64_t* e1T = e1Buf + (size_t)t * n; uint64_t* mT = mBuf + (size_t)t * n;
        for (uint32_t k = 0; k < n; ++k) {
            vT[k] = emb_barrett(vc[k], TB); e0T[k] = emb_barrett(e0c[k], TB);
            e1T[k] = emb_barrett(e1c[k], TB); mT[k] = emb_barrett((long)m[k], TB);
        }
    }

    // GOD PROMPT "CKKS PHASE 34": 4*sizeQ-length, repeating the sizeQ
    // real per-tower tables 4 times (v,e0,e1,m each get the SAME
    // per-tower modulus/root tables) -- see the pool struct's own
    // comment above for why this enables merging the 4 kernel calls
    // below into 1.
    std::vector<uint64_t*> h_fr(4 * sizeQ), h_fp(4 * sizeQ);
    std::vector<uint64_t> h_mod4(4 * sizeQ);
    for (uint32_t t = 0; t < sizeQ; ++t) {
        EncDevTab& Tt = enc_dev_tab_batched(n, mod[t], root[t]);
        for (uint32_t r = 0; r < 4; ++r) { h_fr[r * sizeQ + t] = Tt.fr; h_fp[r * sizeQ + t] = Tt.fp; h_mod4[r * sizeQ + t] = mod[t]; }
    }

    EncryptBufPoolBatched& P = encrypt_buf_pool_batched(n, sizeQ);
    cudaMemcpy(P.d_fr, h_fr.data(), 4 * sizeQ * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(P.d_fp, h_fp.data(), 4 * sizeQ * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(P.d_mod, h_mod4.data(), 4 * sizeQ * 8, cudaMemcpyHostToDevice);

    const size_t T8 = T * 8;
    // ONE upload of the whole packed region (was 4 separate calls).
    cudaMemcpy(P.d_packed, packed.data(), 4 * T8, cudaMemcpyHostToDevice);

    // GOD PROMPT "CKKS PHASE 34": ONE call over the whole packed buffer
    // (towers=4*sizeQ) replaces what was four separate per-buffer calls
    // (Phase 15's v/e0/e1/m offsets, each towers=sizeQ) -- was 20 total
    // kernel launches (4 calls x ~5 internal stage launches each), now
    // 5. See the pool struct's own comment above for why this is exact,
    // not an approximation.
    LaunchNTT_CT_Batched(P.d_packed, (const uint64_t* const*)P.d_fr, (const uint64_t* const*)P.d_fp, P.d_mod, 4 * sizeQ, n, 0);
    cudaDeviceSynchronize();

    // ONE download of the whole packed region (was 4 separate calls).
    cudaMemcpy(packed.data(), P.d_packed, 4 * T8, cudaMemcpyDeviceToHost);

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
