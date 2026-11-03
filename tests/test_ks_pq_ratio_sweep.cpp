// GOD PROMPT "CKKS PHASE 9": decisive discriminator between "sizeP=2 is a
// cursed code defect" (Phase 8's entry #41 framing) and "the auxiliary
// modulus P must be at least as large as the largest part's own modulus
// product Q_part for the Hybrid key-switch approximation to stay small"
// (a parameter-choice/mathematical requirement, not a bug) -- the leading
// hypothesis after tests/test_keyswitch_stage_trace.cpp found numPart=1
// fails EVEN at sizeP=3 (which Phase 8 never checked), and numPart=2 passes
// at sizeP=3 but fails at sizeP=1,2, exactly tracking each config's
// Q_part_max/P bit-length ratio rather than sizeP's raw count.
//
// Holds sizeP=2 FIXED (the value Phase 8 called "cursed") and sweeps ONLY
// numPart, which changes alpha=ceil(sizeQl/numPart) and therefore each
// part's own Q_part size, WITHOUT touching sizeP, WITHOUT touching any
// production code. If the P-vs-Q_part theory is correct, increasing
// numPart (shrinking Q_part below P's ~120-bit budget at sizeP=2) should
// make the SAME relin-identity check pass cleanly -- proving conclusively
// that sizeP=2 itself is not the defect.
#include "keyswitch.h"
#include "keygen.h"
#include "ntt.h"
#include "intt.h"
#include "basis_convert.h"
#include "ks_fastcore.h"
#include "moddown.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>

namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void compute_keyswitch_constants(KeySwitchConstants&, const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t);
void evalkeygen_host(KeySwitchConstants&, const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                      const std::vector<uint64_t>&, const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                      const std::vector<uint64_t>&, uint64_t, double, uint64_t, cudaStream_t);
KeyPairHost keygen_host(uint32_t, const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
KeySwitchResult keyswitch_core_resident(const std::vector<uint64_t>&, const KeySwitchConstants&);
}

namespace {
uint64_t mulmod(uint64_t a, uint64_t b, uint64_t q) { return (uint64_t)(((unsigned __int128)a * b) % q); }
uint64_t addmod(uint64_t a, uint64_t b, uint64_t q) { uint64_t s = a + b; return s >= q ? s - q : s; }
uint64_t submod(uint64_t a, uint64_t b, uint64_t q) { return a >= b ? a - b : a + q - b; }
uint64_t powmod(uint64_t b, uint64_t e, uint64_t q) { unsigned __int128 r = 1, bb = b % q;
    while (e) { if (e & 1) r = (r * bb) % q; bb = (bb * bb) % q; e >>= 1; } return (uint64_t)r; }
uint64_t invmod(uint64_t a, uint64_t q) { return powmod(a % q, q - 2, q); }
uint64_t* up(const std::vector<uint64_t>& h) { uint64_t* d; size_t B = h.size() * 8;
    cudaMalloc(&d, B); cudaMemcpy(d, h.data(), B, cudaMemcpyHostToDevice); return d; }
struct Tab { std::vector<uint64_t> ir, ip; uint64_t ninv, ninv_p; };
Tab build_inv_tab(uint32_t n, uint64_t q, uint64_t root) {
    uint64_t rootInv = invmod(root, q);
    auto br = [](uint32_t v, uint32_t b) { uint32_t r = 0; for (uint32_t k = 0; k < b; k++) { r = (r << 1) | (v & 1); v >>= 1; } return r; };
    uint32_t logn = 0; while ((1u << logn) < n) ++logn;
    Tab T; T.ir.resize(n); T.ip.resize(n);
    std::vector<uint64_t> pw(n); pw[0] = 1;
    for (uint32_t i = 1; i < n; ++i) pw[i] = (uint64_t)(((unsigned __int128)pw[i - 1] * rootInv) % q);
    for (uint32_t i = 0; i < n; ++i) { uint64_t ri = pw[br(i, logn)]; T.ir[i] = ri; T.ip[i] = (uint64_t)(((unsigned __int128)ri << 64) / q); }
    T.ninv = invmod(n, q); T.ninv_p = (uint64_t)(((unsigned __int128)T.ninv << 64) / q);
    return T;
}
std::vector<uint64_t> inverse_ntt(const std::vector<uint64_t>& eval, uint32_t n, uint64_t q, uint64_t root) {
    Tab T = build_inv_tab(n, q, root);
    uint64_t *d_x = up(eval), *d_ir = up(T.ir), *d_ip = up(T.ip);
    LaunchINTT_GS(d_x, d_ir, d_ip, n, q, T.ninv, T.ninv_p, 0);
    cudaDeviceSynchronize();
    std::vector<uint64_t> out(n);
    cudaMemcpy(out.data(), d_x, (size_t)n * 8, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_ir); cudaFree(d_ip);
    return out;
}

double bitlen(uint64_t q) { double b = 0; while (q) { q >>= 1; b++; } return b; }
} // namespace

int main() {
    using namespace gpufhe;
    const uint32_t n = 1024;
    const uint32_t sizeQl = 8, sizeP = 2; // the "cursed" value, held fixed throughout.
    bool overallPass = true;

    for (uint32_t numPart : {1u, 2u, 3u, 4u, 8u}) {
        std::vector<uint64_t> modQ, modP;
        native_primes(modQ, 1, 60, n, {});
        { std::vector<uint64_t> mids; native_primes(mids, sizeQl - 1, 50, n, modQ);
          for (auto m : mids) modQ.push_back(m); }
        native_primes(modP, sizeP, 60, n, modQ);
        std::vector<uint64_t> modQP = modQ; for (auto p : modP) modQP.push_back(p);
        uint32_t sizeQlP = sizeQl + sizeP;
        std::vector<uint64_t> rootQP(sizeQlP);
        for (uint32_t t = 0; t < sizeQlP; ++t) rootQP[t] = native_root(n, modQP[t]);

        KeySwitchConstants K; K.n = n;
        compute_keyswitch_constants(K, modQ, modP, numPart);
        for (uint32_t i = 0; i < sizeQl; ++i) { K.rootModList.push_back(modQ[i]); K.rootValList.push_back(rootQP[i]); }
        for (uint32_t j = 0; j < sizeP; ++j) { K.rootModList.push_back(modQP[sizeQl + j]); K.rootValList.push_back(rootQP[sizeQl + j]); }

        auto KP = keygen_host(n, modQP, rootQP, 1, 3.19, 4242);
        std::vector<uint64_t> PModq(sizeQlP);
        for (uint32_t t = 0; t < sizeQlP; ++t) {
            uint64_t q = modQP[t]; uint64_t P = 1 % q;
            for (uint32_t j = 0; j < sizeP; ++j) P = mulmod(P, modQP[sizeQl + j] % q, q);
            PModq[t] = P;
        }
        evalkeygen_host(K, KP.s, KP.pkA, KP.pkB, PModq, modQP, rootQP, 1, 3.19, 8888, 0);

        // Compute Q_part_max's bit length and P's bit length, for reporting.
        double pBits = 0; for (auto p : modP) pBits += bitlen(p);
        double qPartMaxBits = 0;
        for (uint32_t part = 0; part < numPart; ++part) {
            double b = 0;
            for (uint32_t i = K.startIdx[part]; i < K.startIdx[part] + K.sizePart[part]; ++i) b += bitlen(modQ[i]);
            if (b > qPartMaxBits) qPartMaxBits = b;
        }

        const uint64_t V = 5;
        std::vector<uint64_t> aTowers((size_t)sizeQl * n, V);
        KeySwitchResult R = keyswitch_core_resident(aTowers, K);

        double worstFrac = 0.0; uint32_t worstTower = 0;
        for (uint32_t i = 0; i < sizeQl; ++i) {
            uint64_t q = modQ[i];
            std::vector<uint64_t> sRow(KP.s.begin() + (size_t)i * n, KP.s.begin() + (size_t)(i + 1) * n);
            std::vector<uint64_t> ba0Row(R.ba0.begin() + (size_t)i * n, R.ba0.begin() + (size_t)(i + 1) * n);
            std::vector<uint64_t> ba1Row(R.ba1.begin() + (size_t)i * n, R.ba1.begin() + (size_t)(i + 1) * n);
            std::vector<uint64_t> aRow(aTowers.begin() + (size_t)i * n, aTowers.begin() + (size_t)(i + 1) * n);
            std::vector<uint64_t> lhsEval(n), rhsEval(n);
            for (uint32_t k = 0; k < n; ++k) {
                lhsEval[k] = addmod(ba0Row[k], mulmod(ba1Row[k], sRow[k], q), q);
                uint64_t s2 = mulmod(sRow[k], sRow[k], q);
                rhsEval[k] = mulmod(aRow[k], s2, q);
            }
            std::vector<uint64_t> lhsCoeff = inverse_ntt(lhsEval, n, q, rootQP[i]);
            std::vector<uint64_t> rhsCoeff = inverse_ntt(rhsEval, n, q, rootQP[i]);
            uint64_t worst = 0;
            for (uint32_t k = 0; k < n; ++k) {
                uint64_t diff = submod(lhsCoeff[k], rhsCoeff[k], q);
                uint64_t mag = (diff > q - diff) ? q - diff : diff;
                if (mag > worst) worst = mag;
            }
            double frac = (double)worst / (double)q;
            if (frac > worstFrac) { worstFrac = frac; worstTower = i; }
        }
        bool pass = worstFrac < 0.001;
        if (!pass) overallPass = false;
        std::cout << "numPart=" << numPart << " alpha=" << K.alpha
                  << " Q_part_max=" << qPartMaxBits << "bits P=" << pBits << "bits "
                  << "(P-Qpart=" << (pBits - qPartMaxBits) << " bits) "
                  << "worst=" << (worstFrac*100.0) << "% of q at tower " << worstTower
                  << " -> " << (pass ? "[PASS]" : "[FAIL]") << "\n";
    }

    std::cout << (overallPass ? "[OVERALL PASS]\n" : "[OVERALL FAIL -- but see per-row P-vs-Qpart correlation above]\n");
    return 0; // informational sweep, not a pass/fail gate
}
