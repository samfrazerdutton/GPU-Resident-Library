// GOD PROMPT "CKKS PHASE 13": encrypt_host_shoup -- an EXACT,
// bit-identical-output alternative to encrypt_host (src/encdec.cpp,
// UNCHANGED, kept as the reference/fallback) that replaces the final
// combine loop's four generic 128-bit `(a*b)%q` modular multiplications
// with Shoup's precomputed-constant modmul -- the SAME technique
// already used, and already validated, by this codebase's own NTT and
// basis-conversion CUDA kernels (kernels/intt_ct.cu's shoup_mulmod,
// kernels/basis_convert.cu's shoup_mulmod). Direct assembly inspection
// (see research/ENCRYPT_PROFILING.md) confirmed every host-side
// `(unsigned __int128)a*b % q` in this codebase, including encrypt_
// host's own combine loop, compiles to a call into libgcc's __umodti3
// -- a generic, branchy SOFTWARE 128-bit division routine, dramatically
// more expensive than a hardware multiply. Shoup reduction is an EXACT
// modular-multiplication algorithm, not an approximation: for
// correctly-reduced inputs (<q) it always returns the true (a*b) mod q
// -- verified bit-exact against encrypt_host across every supported n
// and multiple seeds/messages in tests/
// test_encrypt_host_shoup_matches_reference.cpp.
//
// Zero change to randomness or security: the RNG (std::mt19937_64), the
// Gaussian error sampling, the ternary v sampling, and the embedding
// formula are all byte-for-byte IDENTICAL to encrypt_host, copied
// verbatim -- only the arithmetic used to COMBINE the already-sampled,
// already-NTT'd values into (c0,c1) is changed.
#include "keygen.h"
#include "ntt.h"
#include <cstdint>
#include <vector>
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
// Only used to build the (infrequent, cached-by-(n,q)) root tables --
// not on the per-ciphertext hot path, so left as the ordinary generic
// reduction (no need to optimize a call that happens once per distinct
// modulus for the lifetime of the process).
uint64_t mulmod_cold(uint64_t a, uint64_t b, uint64_t q) { return (uint64_t)(((unsigned __int128)a * b) % q); }

// Shoup's precomputed-constant modmul: precon(c,q) = floor(c*2^64/q),
// then shoup_mulmod(x,c,precon,q) = x*c - floor(x*precon/2^64)*q,
// corrected by at most one subtraction. IDENTICAL algorithm to
// kernels/intt_ct.cu's / kernels/basis_convert.cu's device shoup_mulmod
// -- this is the host-side twin, using __int128 to extract the high
// 64 bits of x*precon (a fast hardware multiply, not a division)
// instead of CUDA's __umul64hi intrinsic.
inline uint64_t shoup_precon(uint64_t c, uint64_t q) { return (uint64_t)(((unsigned __int128)c << 64) / q); }
inline uint64_t shoup_mulmod(uint64_t x, uint64_t c, uint64_t precon, uint64_t q) {
    uint64_t hi = (uint64_t)(((unsigned __int128)x * precon) >> 64);
    uint64_t r = x * c - hi * q;
    return (r >= q) ? r - q : r;
}

struct EncDevTab { uint64_t* fr = nullptr; uint64_t* fp = nullptr; };
EncDevTab build_enc_tab_shoup(uint32_t n, uint64_t q, uint64_t root) {
    auto br = [](uint32_t v, uint32_t b) { uint32_t r = 0; for (uint32_t k = 0; k < b; k++) { r = (r << 1) | (v & 1); v >>= 1; } return r; };
    uint32_t logn = 0; while ((1u << logn) < n) ++logn;
    std::vector<uint64_t> fr(n), fp(n), pw(n); pw[0] = 1;
    for (uint32_t i = 1; i < n; ++i) pw[i] = mulmod_cold(pw[i - 1], root, q);
    for (uint32_t i = 0; i < n; ++i) { uint64_t r = pw[br(i, logn)]; fr[i] = r; fp[i] = (uint64_t)(((__uint128_t)r << 64) / q); }
    EncDevTab T; T.fr = up(fr); T.fp = up(fp);
    return T;
}
EncDevTab& enc_dev_tab_shoup(uint32_t n, uint64_t q, uint64_t root) {
    static std::map<std::pair<uint32_t, uint64_t>, EncDevTab> cache;
    auto key = std::make_pair(n, q);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    return cache.emplace(key, build_enc_tab_shoup(n, q, root)).first->second;
}
} // anon

// Same signature/contract as encrypt_host (src/encdec.cpp). RNG
// consumption order, Gaussian/ternary sampling, and the RNS embedding
// formula are copied verbatim -- only the final combine loop differs.
void encrypt_host_shoup(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1,
                         const std::vector<int64_t>& m,
                         const std::vector<uint64_t>& pkA, const std::vector<uint64_t>& pkB,
                         uint32_t n, const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root,
                         uint64_t ns, double sigma, uint64_t seed)
{
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
    for (uint32_t t = 0; t < sizeQ; ++t) { EncDevTab& T = enc_dev_tab_shoup(n, mod[t], root[t]); h_fr[t] = T.fr; h_fp[t] = T.fp; }
    uint64_t **d_fr = nullptr, **d_fp = nullptr;
    cudaMalloc(&d_fr, sizeQ * sizeof(uint64_t*));
    cudaMalloc(&d_fp, sizeQ * sizeof(uint64_t*));
    cudaMemcpy(d_fr, h_fr.data(), sizeQ * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fp, h_fp.data(), sizeQ * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    uint64_t* d_mod = up(mod);

    uint64_t *d_v, *d_e0, *d_e1, *d_m;
    const size_t T8 = (size_t)sizeQ * n * 8;
    cudaMalloc(&d_v, T8); cudaMalloc(&d_e0, T8); cudaMalloc(&d_e1, T8); cudaMalloc(&d_m, T8);
    cudaMemcpy(d_v, vBuf.data(), T8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_e0, e0Buf.data(), T8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_e1, e1Buf.data(), T8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, mBuf.data(), T8, cudaMemcpyHostToDevice);
    LaunchNTT_CT_Batched(d_v, (const uint64_t* const*)d_fr, (const uint64_t* const*)d_fp, d_mod, sizeQ, n, 0);
    LaunchNTT_CT_Batched(d_e0, (const uint64_t* const*)d_fr, (const uint64_t* const*)d_fp, d_mod, sizeQ, n, 0);
    LaunchNTT_CT_Batched(d_e1, (const uint64_t* const*)d_fr, (const uint64_t* const*)d_fp, d_mod, sizeQ, n, 0);
    LaunchNTT_CT_Batched(d_m, (const uint64_t* const*)d_fr, (const uint64_t* const*)d_fp, d_mod, sizeQ, n, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(vBuf.data(), d_v, T8, cudaMemcpyDeviceToHost);
    cudaMemcpy(e0Buf.data(), d_e0, T8, cudaMemcpyDeviceToHost);
    cudaMemcpy(e1Buf.data(), d_e1, T8, cudaMemcpyDeviceToHost);
    cudaMemcpy(mBuf.data(), d_m, T8, cudaMemcpyDeviceToHost);
    cudaFree(d_v); cudaFree(d_e0); cudaFree(d_e1); cudaFree(d_m); cudaFree(d_fr); cudaFree(d_fp); cudaFree(d_mod);

    // GOD PROMPT "CKKS PHASE 13": combine loop rewritten with Shoup's
    // precomputed-constant modmul (see file header). `ns%q` is the same
    // scalar for every k within a tower -- its precon is computed ONCE
    // per tower, not per k. `vT[k]` is reused as the fixed multiplicand
    // for BOTH the pkB*v and pkA*v products, so computing its precon
    // once (one division) replaces what would otherwise be two separate
    // slow __umodti3 calls -- net: 4 slow divisions per (t,k) in the
    // original down to 1 (amortized) per (t,k) here, the rest replaced
    // by cheap multiply-based shoup_mulmod calls.
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
