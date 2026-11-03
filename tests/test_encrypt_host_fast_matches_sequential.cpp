// GOD PROMPT "CKKS PHASE 11": correctness test for the batched
// encrypt_host (src/encdec.cpp). The production code's own embedding
// formula, NTT, and combine arithmetic are UNCHANGED by batching (same
// operations, same values -- only the control flow around how many GPU
// round trips happen differs), so this test's oracle is an INDEPENDENT,
// hand-written reimplementation of the exact pre-Phase-11 per-tower
// algorithm (including its own from-scratch, unbatched `xform` helper),
// written fresh here -- NOT a call into src/encdec.cpp, and NOT merely
// diffing against git history. The production implementation is never
// used as its own oracle.
//
// Since encrypt_host samples its OWN randomness internally (seeded by
// the caller's `seed`), and both the oracle and the production code use
// the IDENTICAL seeded RNG sequence (same seed -> same std::mt19937_64 /
// std::normal_distribution draws) and the IDENTICAL modular arithmetic
// per coefficient, the expected result is EXACT bit-identical equality.
#include "keygen.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

namespace gpufhe {
void encrypt_host(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
}

namespace {
uint64_t mm(uint64_t a, uint64_t b, uint64_t q) { return (uint64_t)(((unsigned __int128)a * b) % q); }
uint64_t am(uint64_t a, uint64_t b, uint64_t q) { uint64_t s = a + b; return s >= q ? s - q : s; }
uint64_t* up(const std::vector<uint64_t>& h) { uint64_t* d; size_t B = h.size() * 8;
    cudaMalloc(&d, B); cudaMemcpy(d, h.data(), B, cudaMemcpyHostToDevice); return d; }
uint64_t powmod(uint64_t b, uint64_t e, uint64_t q) { unsigned __int128 r = 1, bb = b % q;
    while (e) { if (e & 1) r = (r * bb) % q; bb = (bb * bb) % q; e >>= 1; } return (uint64_t)r; }
uint64_t invmod(uint64_t a, uint64_t q) { return powmod(a % q, q - 2, q); }
} // namespace

extern "C" void LaunchNTT_CT(uint64_t* d_x, const uint64_t* d_roots, const uint64_t* d_precon,
                              uint32_t n, uint64_t q, cudaStream_t s);

namespace {
void oracle_encrypt_host(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1,
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
    for (uint32_t t = 0; t < sizeQ; ++t) {
        uint64_t q = mod[t], rt = root[t];
        std::vector<uint64_t> vT(n), e0T(n), e1T(n), mT(n);
        for (uint32_t k = 0; k < n; ++k) {
            auto emb = [&](long x) { return (uint64_t)((x % (long)q + (long)q) % (long)q); };
            vT[k] = emb(vc[k]); e0T[k] = emb(e0c[k]); e1T[k] = emb(e1c[k]); mT[k] = emb((long)m[k]);
        }
        // Unbatched: one NTT call per buffer per tower, own fresh root table each time.
        for (std::vector<uint64_t>* buf : {&vT, &e0T, &e1T, &mT}) {
            auto br = [](uint32_t v, uint32_t b) { uint32_t r = 0; for (uint32_t k = 0; k < b; k++) { r = (r << 1) | (v & 1); v >>= 1; } return r; };
            uint32_t logn = 0; while ((1u << logn) < n) ++logn;
            std::vector<uint64_t> tr(n), tp(n), pw(n); pw[0] = 1;
            for (uint32_t i = 1; i < n; ++i) pw[i] = mm(pw[i - 1], rt, q);
            for (uint32_t i = 0; i < n; ++i) { uint64_t r = pw[br(i, logn)]; tr[i] = r; tp[i] = (uint64_t)(((unsigned __int128)r << 64) / q); }
            uint64_t *dx = up(*buf), *dr = up(tr), *dp = up(tp);
            LaunchNTT_CT(dx, dr, dp, n, q, 0);
            cudaDeviceSynchronize();
            cudaMemcpy(buf->data(), dx, (size_t)n * 8, cudaMemcpyDeviceToHost);
            cudaFree(dx); cudaFree(dr); cudaFree(dp);
        }
        for (uint32_t k = 0; k < n; ++k) {
            uint64_t b = am(mm(pkB[(size_t)t * n + k], vT[k], q), mm(ns % q, e0T[k], q), q);
            uint64_t a = am(mm(pkA[(size_t)t * n + k], vT[k], q), mm(ns % q, e1T[k], q), q);
            c0[(size_t)t * n + k] = am(b, mT[k], q);
            c1[(size_t)t * n + k] = a;
        }
    }
}
} // namespace

int main() {
    using namespace gpufhe;
    bool allPass = true;

    for (uint32_t n : {1024u, 2048u, 4096u}) {
        for (uint32_t sizeQ : {1u, 4u, 8u}) {
            std::vector<uint64_t> mod, root;
            native_primes(mod, 1, 60, n, {});
            { std::vector<uint64_t> mids; native_primes(mids, sizeQ - 1, 50, n, mod);
              for (auto m : mids) mod.push_back(m); }
            for (uint32_t t = 0; t < sizeQ; ++t) root.push_back(native_root(n, mod[t]));

            // Minimal, deterministic "public key" (doesn't need to be a
            // real keypair for this test -- only encrypt_host's own
            // embedding/NTT/combine control flow is under test).
            std::vector<uint64_t> pkA((size_t)sizeQ * n), pkB((size_t)sizeQ * n);
            for (uint32_t t = 0; t < sizeQ; ++t) {
                for (uint32_t k = 0; k < n; ++k) {
                    pkA[(size_t)t * n + k] = (7 + t * 13 + k) % mod[t];
                    pkB[(size_t)t * n + k] = (mod[t] - 1 - (t * 5 + k) % mod[t]) % mod[t];
                }
            }
            std::vector<int64_t> m(n);
            for (uint32_t k = 0; k < n; ++k) m[k] = (int64_t)((k % 17) - 8);

            std::vector<uint64_t> c0_fast, c1_fast, c0_oracle, c1_oracle;
            encrypt_host(c0_fast, c1_fast, m, pkA, pkB, n, mod, root, 1, 3.19, 424242);
            oracle_encrypt_host(c0_oracle, c1_oracle, m, pkA, pkB, n, mod, root, 1, 3.19, 424242);

            bool match = (c0_fast.size() == c0_oracle.size()) && (c1_fast.size() == c1_oracle.size());
            for (size_t i = 0; match && i < c0_fast.size(); ++i) if (c0_fast[i] != c0_oracle[i]) match = false;
            for (size_t i = 0; match && i < c1_fast.size(); ++i) if (c1_fast[i] != c1_oracle[i]) match = false;

            std::cout << "n=" << n << " sizeQ=" << sizeQ << " -> " << (match ? "[EXACT MATCH]" : "[MISMATCH]") << "\n";
            if (!match) allPass = false;
        }
    }

    std::cout << (allPass ? "[OVERALL PASS]\n" : "[OVERALL FAIL]\n");
    return allPass ? 0 : 1;
}
