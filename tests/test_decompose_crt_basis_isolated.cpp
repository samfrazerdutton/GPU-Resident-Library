// GOD PROMPT "CKKS PHASE 9": Stage C isolation of the DECOMPOSE direction's
// own ApproxSwitchCRTBasis invocation (src/keyswitch.cpp STAGE 2: part's own
// Q towers -> complement basis), using compute_keyswitch_constants's
// partQHatInv/partQHatInvPrec/partQHatModp/partComplMod/partBMuLo/partBMuHi
// constant family -- a COMPLETELY DIFFERENT set of constants from Stage B/D's
// ModDown family (pHatInv/pHatModq/mdBMu), computed by a different loop in
// src/ks_constants.cpp, and NEVER independently tested in isolation until
// this round (the earlier Phase 8 hand-reconstruction check, and this
// round's Stage B/D tests, all exercised the P->Q ModDown direction only).
//
// Uses numPart=1 specifically: with sizePart==sizeQl (the whole Q chain is
// one part), the complement basis is PURE P (sizeCompl==sizeP exactly,
// startIdx=0, endIdx=sizeQl) -- the cleanest possible isolation of "source=Q
// (sizeQl towers), target=P (sizeP towers)", matching the exact decompose
// call keyswitch_core_resident makes for numPart=1 (one of the two numPart
// values CHECK4 found failing at sizeP=2 in Phase 8/tests/
// test_evalkeygen_isolated.cpp).
//
// === Derivation ===
// Represent a known integer value A (0 <= A < Q_total) as residues A mod q_i
// at every SOURCE Q tower. Converting Q -> P (ModUp direction) should
// reconstruct (A + k*Q_total) mod p_j for some SMALL consistent k < sizeQl,
// at EVERY target P tower -- the SAME "approximate CRT, bounded k" theory
// Stage B already confirmed for the reverse (P->Q) direction. This is an
// EXACT modular identity once k is found, checked with zero tolerance.
#include "keyswitch.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>

namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
void compute_keyswitch_constants(KeySwitchConstants&, const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t);
}

extern "C" void LaunchApproxSwitchCRTBasis(
    const uint64_t* d_src, uint64_t* d_dst,
    const uint64_t* d_QHatInvModq, const uint64_t* d_QHatInvModqPrecon, const uint64_t* d_q,
    const uint64_t* d_QHatModp,
    const uint64_t* d_p, const uint64_t* d_mu_lo, const uint64_t* d_mu_hi,
    uint32_t sizeQ, uint32_t sizeP, uint32_t n, cudaStream_t s);

namespace {
uint64_t mulmod(uint64_t a, uint64_t b, uint64_t q) { return (uint64_t)(((unsigned __int128)a * b) % q); }
uint64_t addmod(uint64_t a, uint64_t b, uint64_t q) { uint64_t s = a + b; return s >= q ? s - q : s; }
uint64_t submod(uint64_t a, uint64_t b, uint64_t q) { return a >= b ? a - b : a + q - b; }

uint64_t* up(const std::vector<uint64_t>& h) {
    uint64_t* d; size_t B = h.size() * 8;
    cudaMalloc(&d, B); cudaMemcpy(d, h.data(), B, cudaMemcpyHostToDevice);
    return d;
}

std::vector<uint64_t> run_conversion(
    const std::vector<uint64_t>& srcResidues,
    const gpufhe::KeySwitchConstants& K, uint32_t part, uint32_t sizeCompl)
{
    const uint32_t n = 1;
    uint64_t *d_src = up(srcResidues);
    uint64_t *d_dst; cudaMalloc(&d_dst, (size_t)sizeCompl * 8);
    uint64_t *d_qhi = up(K.partQHatInv[part]), *d_qhip = up(K.partQHatInvPrec[part]),
             *d_q = up(K.partSrcMod[part]), *d_qmp = up(K.partQHatModp[part]),
             *d_p = up(K.partComplMod[part]), *d_mlo = up(K.partBMuLo[part]), *d_mhi = up(K.partBMuHi[part]);
    LaunchApproxSwitchCRTBasis(d_src, d_dst, d_qhi, d_qhip, d_q, d_qmp, d_p, d_mlo, d_mhi,
                                (uint32_t)srcResidues.size(), sizeCompl, n, 0);
    cudaDeviceSynchronize();
    std::vector<uint64_t> out(sizeCompl);
    cudaMemcpy(out.data(), d_dst, (size_t)sizeCompl * 8, cudaMemcpyDeviceToHost);
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_qhi); cudaFree(d_qhip); cudaFree(d_q);
    cudaFree(d_qmp); cudaFree(d_p); cudaFree(d_mlo); cudaFree(d_mhi);
    return out;
}

uint64_t Q_mod(const std::vector<uint64_t>& moduliQ, uint64_t p) {
    uint64_t r = 1 % p;
    for (auto q : moduliQ) r = mulmod(r, q % p, p);
    return r;
}
} // namespace

int main() {
    using namespace gpufhe;
    const uint32_t n = 1024;
    bool allPass = true;
    std::vector<uint64_t> testValues = {0, 1, 123456789ULL, 999999999999ULL};

    for (uint32_t sizeQl : {5u, 8u}) {
        for (uint32_t sizeP : {1u, 2u, 3u}) {
            std::vector<uint64_t> modQ, modP;
            native_primes(modQ, 1, 60, n, {});
            { std::vector<uint64_t> mids; native_primes(mids, sizeQl - 1, 50, n, modQ);
              for (auto m : mids) modQ.push_back(m); }
            native_primes(modP, sizeP, 60, n, modQ);

            KeySwitchConstants K; K.n = n;
            compute_keyswitch_constants(K, modQ, modP, /*numPart=*/1);
            // numPart=1 -> part 0 owns all sizeQl Q towers; complement is pure P.
            uint32_t part = 0;
            uint32_t sizeCompl = K.sizeCompl[part];

            std::cout << "=== sizeQl=" << sizeQl << " sizeP=" << sizeP
                      << " (decompose Q->complement, sizeCompl=" << sizeCompl << ") ===\n";

            bool configPass = true;
            for (uint64_t V : testValues) {
                std::vector<uint64_t> srcResidues(sizeQl);
                for (uint32_t i = 0; i < sizeQl; ++i) srcResidues[i] = V % modQ[i];

                std::vector<uint64_t> out = run_conversion(srcResidues, K, part, sizeCompl);

                std::vector<uint64_t> Qmodpj(sizeCompl);
                for (uint32_t j = 0; j < sizeCompl; ++j) Qmodpj[j] = Q_mod(modQ, K.partComplMod[part][j]);

                bool found = false; uint32_t foundK = 0;
                for (uint32_t k = 0; k <= 4 * sizeQl + 2 && !found; ++k) {
                    bool allMatch = true;
                    for (uint32_t j = 0; j < sizeCompl; ++j) {
                        uint64_t pj = K.partComplMod[part][j];
                        uint64_t expect = addmod(V % pj, mulmod(k % pj, Qmodpj[j], pj), pj);
                        if (expect != out[j]) { allMatch = false; break; }
                    }
                    if (allMatch) { found = true; foundK = k; }
                }

                if (!found) {
                    configPass = false; allPass = false;
                    std::cout << "  [FAIL] V=" << V << " -- no consistent k in [0," << (4*sizeQl+2)
                              << "] reconciles all " << sizeCompl << " target towers.\n";
                    for (uint32_t j = 0; j < sizeCompl; ++j) {
                        uint64_t pj = K.partComplMod[part][j];
                        uint64_t expectK0 = V % pj;
                        uint64_t diff = submod(out[j], expectK0, pj);
                        double frac = (double)diff / (double)pj;
                        std::cout << "    target " << j << " mod=" << pj << " actual=" << out[j]
                                  << " expect(k=0)=" << expectK0 << " residual_vs_k0="
                                  << diff << " (" << (frac*100.0) << "% of modulus)\n";
                    }
                } else {
                    std::cout << "  [PASS] V=" << V << " k=" << foundK << " (consistent across all "
                              << sizeCompl << " targets)\n";
                }
            }
            if (configPass) std::cout << "  [PASS] sizeQl=" << sizeQl << " sizeP=" << sizeP << "\n";
        }
    }

    std::cout << (allPass ? "[OVERALL PASS]\n" : "[OVERALL FAIL]\n");
    return allPass ? 0 : 1;
}
