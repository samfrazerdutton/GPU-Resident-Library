// GOD PROMPT "CKKS PHASE 9": Stage B isolation of LaunchApproxSwitchCRTBasis
// in the EXACT configuration keyswitch_core_resident's ModDown step uses it
// (source = P basis, sizeP towers; target = Q basis, sizeQl towers), using an
// INDEPENDENT CPU reference derived from the algorithm's own mathematical
// definition -- NOT the production kernel, NOT keyswitch_core_resident, and
// NOT the earlier Phase 8 test (which only exercised the DECOMPOSE direction,
// Q/part -> complement, never ModDown's P -> Q direction in isolation).
//
// === The algorithm being tested (per kernels/basis_convert.cu's own doc
// comment, confirmed by reading the kernel source) ===
// Given a polynomial represented by its residues a_j mod p_j (j=0..sizeP-1,
// i.e. the auxiliary P basis), for each target tower q_i:
//   x_j = (a_j * PHatInv[j]) mod p_j               (per source tower)
//   sum = sum_j (uint128) x_j * PHatModq[j][i]
//   out[i] = sum mod q_i                            (Barrett reduce)
// This is the STANDARD approximate CRT basis extension: writing P = prod(p_j)
// and X = the true integer value (0 <= X < P) represented by the a_j's, the
// sum before any final reduction equals X + k*P for some SMALL nonnegative
// integer k (k < sizeP, a well-known bound for this non-exact variant, since
// it never subtracts an estimated quotient the way the EXACT CRT reconstruction
// would). So out[i] should equal (X + k*P) mod q_i for the SAME k at every
// target tower i, for a GIVEN coefficient -- this is an EXACT modular identity
// once k is accounted for, not merely "small noise". If no single k in a
// bounded search range reconciles ALL target towers simultaneously, the
// conversion itself is broken for that configuration.
//
// This does NOT run through keyswitch_core_resident, compute_keyswitch_
// constants's higher-level consumers, or decompose/accumulate/evalkeygen at
// all -- it calls compute_keyswitch_constants (for pHatInv/pHatInvPrec/
// pHatModq/qMod/mdBMuLo/mdBMuHi) and LaunchApproxSwitchCRTBasis directly, with
// a hand-picked, deterministic coefficient-0 test value, varying ONLY sizeP
// (1, 2, 3) with sizeQl held at both 5 and 8, per the GOD PROMPT's required
// discriminating matrix.
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
uint64_t* up_scalar(uint64_t v) { uint64_t* d; cudaMalloc(&d, 8); cudaMemcpy(d, &v, 8, cudaMemcpyHostToDevice); return d; }

// Run LaunchApproxSwitchCRTBasis for ONE coefficient position (n=1 trick:
// treat each source/target tower as a length-1 array containing a single
// residue -- the kernel is purely per-coefficient/per-tower independent, so
// this is a faithful, minimal invocation of exactly the same code path).
std::vector<uint64_t> run_conversion(
    const std::vector<uint64_t>& srcResidues,   // [sizeP_src]
    const gpufhe::KeySwitchConstants& K,
    uint32_t sizeQl)
{
    const uint32_t n = 1;
    uint64_t *d_src = up(srcResidues);
    uint64_t *d_dst; cudaMalloc(&d_dst, (size_t)sizeQl * 8);
    uint64_t *d_qhi = up(K.pHatInv), *d_qhip = up(K.pHatInvPrec), *d_q = up(K.pMod);
    uint64_t *d_qmp = up(K.pHatModq), *d_p = up(K.qMod), *d_mlo = up(K.mdBMuLo), *d_mhi = up(K.mdBMuHi);
    LaunchApproxSwitchCRTBasis(d_src, d_dst, d_qhi, d_qhip, d_q, d_qmp, d_p, d_mlo, d_mhi,
                                (uint32_t)srcResidues.size(), sizeQl, n, 0);
    cudaDeviceSynchronize();
    std::vector<uint64_t> out(sizeQl);
    cudaMemcpy(out.data(), d_dst, (size_t)sizeQl * 8, cudaMemcpyDeviceToHost);
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_qhi); cudaFree(d_qhip); cudaFree(d_q);
    cudaFree(d_qmp); cudaFree(d_p); cudaFree(d_mlo); cudaFree(d_mhi);
    return out;
}

// P mod q_i, computed independently (plain modular product, no Shoup/Barrett).
uint64_t P_mod(const std::vector<uint64_t>& moduliP, uint64_t q) {
    uint64_t r = 1 % q;
    for (auto p : moduliP) r = mulmod(r, p % q, q);
    return r;
}
} // namespace

int main() {
    using namespace gpufhe;
    const uint32_t n = 1024;
    bool allPass = true;

    // Test values: 0 (boundary), 1, a mid-range value, and P-1-ish (filled in
    // per-case since P depends on sizeP). Values are reduced mod each p_j
    // independently when building source residues -- exactly how a real
    // P-basis-resident polynomial's coefficient would be represented.
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

            std::cout << "=== sizeQl=" << sizeQl << " sizeP=" << sizeP << " ===\n";

            bool configPass = true;
            for (uint64_t V : testValues) {
                std::vector<uint64_t> srcResidues(sizeP);
                for (uint32_t j = 0; j < sizeP; ++j) srcResidues[j] = V % modP[j];

                std::vector<uint64_t> out = run_conversion(srcResidues, K, sizeQl);

                // Precompute P mod q_i for every target tower, independently.
                std::vector<uint64_t> Pmodqi(sizeQl);
                for (uint32_t i = 0; i < sizeQl; ++i) Pmodqi[i] = P_mod(modP, modQ[i]);

                // Search for a single consistent k in [0, 4*sizeP] (generous
                // margin over the well-known k < sizeP bound) that reconciles
                // EVERY target tower simultaneously for this coefficient.
                bool found = false;
                uint32_t foundK = 0;
                for (uint32_t k = 0; k <= 4 * sizeP + 2 && !found; ++k) {
                    bool allMatch = true;
                    for (uint32_t i = 0; i < sizeQl; ++i) {
                        uint64_t qi = modQ[i];
                        uint64_t expect = addmod(V % qi, mulmod(k % qi, Pmodqi[i], qi), qi);
                        if (expect != out[i]) { allMatch = false; break; }
                    }
                    if (allMatch) { found = true; foundK = k; }
                }

                if (!found) {
                    // Report the actual per-tower discrepancy for the FIRST
                    // divergence, in modular-fraction-of-q terms, exactly as
                    // the governing directive requires (not just pass/fail).
                    configPass = false; allPass = false;
                    std::cout << "  [FAIL] V=" << V << " -- no consistent k in [0," << (4*sizeP+2)
                              << "] reconciles all " << sizeQl << " target towers.\n";
                    for (uint32_t i = 0; i < sizeQl; ++i) {
                        uint64_t qi = modQ[i];
                        uint64_t expectK0 = V % qi;
                        uint64_t diff = submod(out[i], expectK0, qi);
                        double frac = (double)diff / (double)qi;
                        std::cout << "    tower " << i << " q=" << qi << " actual=" << out[i]
                                  << " expect(k=0)=" << expectK0 << " residual_vs_k0="
                                  << diff << " (" << (frac*100.0) << "% of q)\n";
                    }
                } else {
                    std::cout << "  [PASS] V=" << V << " k=" << foundK << " (consistent across all "
                              << sizeQl << " towers)\n";
                }
            }
            if (configPass) std::cout << "  [PASS] sizeQl=" << sizeQl << " sizeP=" << sizeP << "\n";
        }
    }

    std::cout << (allPass ? "[OVERALL PASS]\n" : "[OVERALL FAIL]\n");
    return allPass ? 0 : 1;
}
