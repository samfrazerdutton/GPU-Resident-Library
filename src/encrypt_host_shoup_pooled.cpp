// GOD PROMPT "CKKS PHASE 13": encrypt_host_shoup_pooled -- builds on
// encrypt_host_shoup (src/encrypt_host_shoup.cpp, already verified
// bit-exact vs. production encrypt_host) by ALSO eliminating the
// repeated cudaMalloc/cudaFree of the four per-call device data
// buffers (d_v,d_e0,d_e1,d_m) and the small pointer-array/modulus
// buffers (d_fr,d_fp,d_mod). research/ENCRYPT_PROFILING.md's stage
// breakdown measured device allocation + free together at ~10-27% of
// encrypt_host's total time depending on n -- a SEPARATE, independent
// opportunity from the combine-loop arithmetic encrypt_host_shoup
// already addresses, tested here in isolation so each optimization's
// contribution can be attributed separately (per the governing
// directive's "do not stack multiple unverified changes").
//
// Buffers are cached in a process-lifetime pool keyed by (n,sizeQ),
// the SAME pattern already used throughout this codebase for
// immutable tables (enc_dev_tab/ek_dev_tab) -- the difference here is
// the cached buffers hold per-call DATA, not immutable constants, so
// every call fully overwrites their contents via its own fresh
// cudaMemcpy before using them.
//
// CONCURRENCY NOTE (found and fixed this round, NOT theoretical): a
// pooled buffer is SHARED, MUTABLE state across calls -- unlike the
// read-only root tables this pattern was borrowed from. The first
// version of this file had no synchronization and FAILED the real
// Rust test suite under its default parallel test execution (`cargo
// test` runs tests on multiple OS threads by default): two threads
// calling this function concurrently at the same (n,sizeQ) corrupted
// each other's buffer content mid-flight, producing wrong decrypted
// plaintexts (confirmed by re-running the identical suite with
// `--test-threads=1`, which passed 38/38 -- isolating the cause to a
// race, not an arithmetic defect; the arithmetic was already
// independently verified bit-exact in isolation). Fixed by serializing
// the entire pooled-buffer-touching body behind one mutex -- simple and
// unconditionally correct; negligible overhead (a few tens of ns) for
// the actual production calling pattern, which is sequential, not
// concurrent.
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

struct EncDevTab { uint64_t* fr = nullptr; uint64_t* fp = nullptr; };
EncDevTab build_enc_tab_pooled(uint32_t n, uint64_t q, uint64_t root) {
    auto br = [](uint32_t v, uint32_t b) { uint32_t r = 0; for (uint32_t k = 0; k < b; k++) { r = (r << 1) | (v & 1); v >>= 1; } return r; };
    uint32_t logn = 0; while ((1u << logn) < n) ++logn;
    std::vector<uint64_t> fr(n), fp(n), pw(n); pw[0] = 1;
    for (uint32_t i = 1; i < n; ++i) pw[i] = mulmod_cold(pw[i - 1], root, q);
    for (uint32_t i = 0; i < n; ++i) { uint64_t r = pw[br(i, logn)]; fr[i] = r; fp[i] = (uint64_t)(((__uint128_t)r << 64) / q); }
    EncDevTab T; T.fr = up(fr); T.fp = up(fp);
    return T;
}
EncDevTab& enc_dev_tab_pooled(uint32_t n, uint64_t q, uint64_t root) {
    static std::map<std::pair<uint32_t, uint64_t>, EncDevTab> cache;
    auto key = std::make_pair(n, q);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    return cache.emplace(key, build_enc_tab_pooled(n, q, root)).first->second;
}

// Process-lifetime pool of the per-call DATA buffers, keyed by
// (n,sizeQ). Every field is fully overwritten by its owning call
// before being read -- see file header.
struct EncryptBufPool {
    uint64_t *d_v = nullptr, *d_e0 = nullptr, *d_e1 = nullptr, *d_m = nullptr;
    uint64_t **d_fr = nullptr, **d_fp = nullptr;
    uint64_t *d_mod = nullptr;
    uint32_t sizeQ = 0;
};
EncryptBufPool& encrypt_buf_pool(uint32_t n, uint32_t sizeQ) {
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

void encrypt_host_shoup_pooled(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1,
                                const std::vector<int64_t>& m,
                                const std::vector<uint64_t>& pkA, const std::vector<uint64_t>& pkB,
                                uint32_t n, const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root,
                                uint64_t ns, double sigma, uint64_t seed)
{
    // Serializes the whole call -- see the CONCURRENCY NOTE in this
    // file's header comment. The pooled device buffers are shared,
    // mutable state; every real caller in this codebase invokes
    // encrypt_host_shoup_pooled sequentially (one ciphertext at a
    // time), so this lock is never contended in the measured
    // performance numbers -- it only matters, and is only needed, for
    // callers that invoke this function from multiple threads
    // concurrently (as this project's own Rust test suite does by
    // default).
    static std::mutex pool_mu;
    std::lock_guard<std::mutex> lock(pool_mu);

    uint32_t sizeQ = (uint32_t)mod.size();
    std::mt19937_64 rng(seed); std::normal_distribution<double> g(0.0, sigma);
    std::vector<int> vc(n), e0c(n), e1c(n);
    for (uint32_t k = 0; k < n; ++k) { vc[k] = (int)(rng() % 3) - 1; e0c[k] = (int)std::lround(g(rng)); e1c[k] = (int)std::lround(g(rng)); }
    c0.resize((size_t)sizeQ * n); c1.resize((size_t)sizeQ * n);

    std::vector<uint64_t> vBuf((size_t)sizeQ * n), e0Buf((size_t)sizeQ * n), e1Buf((size_t)sizeQ * n), mBuf((size_t)sizeQ * n);
    for (uint32_t t = 0; t < sizeQ; ++t) { uint64_t q = mod[t];
        uint64_t* vT = vBuf.data() + (size_t)t * n; uint64_t* e0T = e0Buf.data() + (size_t)t * n;
        uint64_t* e1T = e1Buf.data() + (size_t)t * n; uint64_t* mT = mBuf.data() + (size_t)t * n;
        for (uint32_t k = 0; k < n; ++k) { auto emb = [&](long x) { return (uint64_t)((x % (long)q + (long)q) % (long)q); };
            vT[k] = emb(vc[k]); e0T[k] = emb(e0c[k]); e1T[k] = emb(e1c[k]); mT[k] = emb((long)m[k]); }
    }

    std::vector<uint64_t*> h_fr(sizeQ), h_fp(sizeQ);
    for (uint32_t t = 0; t < sizeQ; ++t) { EncDevTab& T = enc_dev_tab_pooled(n, mod[t], root[t]); h_fr[t] = T.fr; h_fp[t] = T.fp; }

    EncryptBufPool& P = encrypt_buf_pool(n, sizeQ);
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
    // NOTE: no cudaFree here -- buffers are returned to the pool
    // (i.e. simply left allocated) for the next call at this (n,sizeQ).

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
