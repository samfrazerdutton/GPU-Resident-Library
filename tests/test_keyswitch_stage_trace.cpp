// GOD PROMPT "CKKS PHASE 9": Stage C/D combined trace. Stage B (ModDown's
// own P->Q ApproxSwitchCRTBasis + full INTT/NTT/Combine chain) and a
// standalone decompose-direction Q->complement test (tests/
// test_decompose_crt_basis_isolated.cpp) BOTH pass cleanly at sizeP=2 in
// isolation -- ruling out the basis-conversion kernel, both constant
// families (ModDown's pHatInv/pHatModq and decompose's partQHatInv/
// partQHatModp), and the ModDownCombine kernel as the defect, even at full
// n=1024 scale. This test re-runs the REAL end-to-end pipeline (real
// keygen_host secret/keypair, real evalkeygen_host eval key, real decompose
// +accumulate+moddown, at the exact failing configuration: n=1024,
// sizeQl=8, sizeP=2, numPart=2, alpha=4) but INSTRUMENTS each stage
// boundary with an INDEPENDENTLY-DERIVED, mostly ZERO-TOLERANCE check, by
// copying keyswitch_core_resident's own stages inline (NOT modifying
// src/keyswitch.cpp) so the exact first-diverging stage can be identified.
//
// Three checks, in pipeline order:
//   CHECK A (decompose, STAGE 2) -- EXACT, zero tolerance. Input is the
//     constant integer V=5 (same trick as Phase 8's CHECK4: an eval-domain
//     constant vector INTTs to the coefficient-domain constant polynomial
//     V). Each part's own-range digit residues are checked to equal V
//     directly (no conversion there); each part's complement-range digit
//     residues are checked against V + k*Q_part (mod that tower), searching
//     for a small consistent k, EXACTLY as test_decompose_crt_basis_
///     isolated.cpp already validated the underlying kernel/constants in
//     isolation -- here confirming the SAME property holds when driven by
//     the real multi-part compute_keyswitch_constants() output.
//   CHECK B (accumulate, STAGE 3a) -- EXACT, zero tolerance. Independently
//     recomputes res0[i]/res1[i] as a direct CPU sum over parts of
//     digit_part[i]*bv_part[i] / digit_part[i]*av_part[i] mod q_i (the
//     literal definition of what LaunchKSMultAcc is supposed to compute),
//     using the SAME digit arrays CHECK A already validated, and compares
//     against the actual GPU-computed res0/res1. This isolates the
//     accumulate KERNEL's own correctness, independent of whether the
//     digits or eval key are "correct" in any cryptographic sense.
//   CHECK C (moddown, STAGE 3b) -- approximate, small-tolerance (the
//     established relin identity: ba0+ba1*s ~= a*s^2 mod q, within ordinary
//     key-switch noise). This is the ONLY check with a non-zero tolerance,
//     consistent with entries #39/#41's finding that this relation is
//     approximate, not exact.
//
// Methodology notes (per the governing directive): INTT is applied before
// every coefficient-domain comparison; NTT/eval-domain pointwise ops are
// used only where both operands are already in eval form; no secret/eval
// key material is logged in bulk (only scalar residuals and a few
// coefficients' worth of diagnostic output).
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
#include <map>

namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void compute_keyswitch_constants(KeySwitchConstants&, const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t);
void evalkeygen_host(KeySwitchConstants&, const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                      const std::vector<uint64_t>&, const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                      const std::vector<uint64_t>&, uint64_t, double, uint64_t, cudaStream_t);
KeyPairHost keygen_host(uint32_t, const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
}

namespace {
uint64_t mulmod(uint64_t a, uint64_t b, uint64_t q) { return (uint64_t)(((unsigned __int128)a * b) % q); }
uint64_t addmod(uint64_t a, uint64_t b, uint64_t q) { uint64_t s = a + b; return s >= q ? s - q : s; }
uint64_t submod(uint64_t a, uint64_t b, uint64_t q) { return a >= b ? a - b : a + q - b; }
uint64_t powmod(uint64_t b, uint64_t e, uint64_t q) { unsigned __int128 r = 1, bb = b % q;
    while (e) { if (e & 1) r = (r * bb) % q; bb = (bb * bb) % q; e >>= 1; } return (uint64_t)r; }
uint64_t invmod(uint64_t a, uint64_t q) { return powmod(a % q, q - 2, q); }

uint64_t* up(const std::vector<uint64_t>& h) {
    uint64_t* d; size_t B = h.size() * 8;
    cudaMalloc(&d, B); cudaMemcpy(d, h.data(), B, cudaMemcpyHostToDevice);
    return d;
}

struct Tab { std::vector<uint64_t> fr, fp, ir, ip; uint64_t ninv, ninv_p; };
Tab build_tab(uint32_t n, uint64_t q, uint64_t root) {
    uint64_t rootInv = invmod(root, q);
    auto br = [](uint32_t v, uint32_t b) { uint32_t r = 0; for (uint32_t k = 0; k < b; k++) { r = (r << 1) | (v & 1); v >>= 1; } return r; };
    uint32_t logn = 0; while ((1u << logn) < n) ++logn;
    Tab T; T.fr.resize(n); T.fp.resize(n); T.ir.resize(n); T.ip.resize(n);
    auto fill = [&](uint64_t base, std::vector<uint64_t>& r, std::vector<uint64_t>& p) {
        std::vector<uint64_t> pw(n); pw[0] = 1;
        for (uint32_t i = 1; i < n; ++i) pw[i] = (uint64_t)(((unsigned __int128)pw[i - 1] * base) % q);
        for (uint32_t i = 0; i < n; ++i) { uint64_t ri = pw[br(i, logn)]; r[i] = ri; p[i] = (uint64_t)(((unsigned __int128)ri << 64) / q); }
    };
    fill(root, T.fr, T.fp); fill(rootInv, T.ir, T.ip);
    T.ninv = invmod(n, q); T.ninv_p = (uint64_t)(((unsigned __int128)T.ninv << 64) / q);
    return T;
}

std::vector<uint64_t> inverse_ntt(const std::vector<uint64_t>& eval, uint32_t n, uint64_t q, uint64_t root) {
    Tab T = build_tab(n, q, root);
    uint64_t *d_x = up(eval), *d_ir = up(T.ir), *d_ip = up(T.ip);
    LaunchINTT_GS(d_x, d_ir, d_ip, n, q, T.ninv, T.ninv_p, 0);
    cudaDeviceSynchronize();
    std::vector<uint64_t> out(n);
    cudaMemcpy(out.data(), d_x, (size_t)n * 8, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_ir); cudaFree(d_ip);
    return out;
}

uint64_t Q_mod(const std::vector<uint64_t>& moduli, uint64_t p) {
    uint64_t r = 1 % p;
    for (auto q : moduli) r = mulmod(r, q % p, p);
    return r;
}
} // namespace

int main() {
    using namespace gpufhe;
    const uint32_t n = 1024;
    bool overallPass = true;

    // The exact real-fixture configuration this round pinpointed as failing.
    for (uint32_t sizeQl : {5u, 8u}) {
        for (uint32_t sizeP : {1u, 2u, 3u}) {
            for (uint32_t numPart : {1u, 2u}) {
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

                auto KP = keygen_host(n, modQP, rootQP, /*ns=*/1, /*sigma=*/3.19, /*seed=*/4242);
                std::vector<uint64_t> PModq(sizeQlP);
                for (uint32_t t = 0; t < sizeQlP; ++t) {
                    uint64_t q = modQP[t]; uint64_t P = 1 % q;
                    for (uint32_t j = 0; j < sizeP; ++j) P = mulmod(P, modQP[sizeQl + j] % q, q);
                    PModq[t] = P;
                }
                evalkeygen_host(K, KP.s, KP.pkA, KP.pkB, PModq, modQP, rootQP, /*ns=*/1, /*sigma=*/3.19, /*seed=*/8888, 0);

                const uint64_t V = 5;
                std::vector<uint64_t> aTowers((size_t)sizeQl * n, V); // eval-domain constant -> coeff-domain constant V

                std::cout << "=== sizeQl=" << sizeQl << " sizeP=" << sizeP << " numPart=" << numPart << " ===\n";

                // ---- STAGE 2: decompose (copied inline from keyswitch_core_resident) ----
                std::vector<std::vector<uint64_t>> digits(numPart);
                bool checkA_pass = true;
                for (uint32_t part = 0; part < numPart; ++part) {
                    uint32_t sizePart = K.sizePart[part], sizeCompl = K.sizeCompl[part];
                    uint32_t startIdx = K.startIdx[part], endIdx = startIdx + sizePart;

                    uint64_t *dsrc = up(std::vector<uint64_t>(aTowers.begin() + (size_t)startIdx * n, aTowers.begin() + (size_t)endIdx * n));
                    std::vector<uint64_t> partCoeff((size_t)sizePart * n);
                    {
                        // INTT each of this part's own towers.
                        for (uint32_t i = 0; i < sizePart; ++i) {
                            uint64_t q = K.partSrcMod[part][i];
                            uint64_t root = native_root(n, q);
                            Tab T = build_tab(n, q, root);
                            uint64_t *dx = dsrc + (size_t)i * n;
                            uint64_t *d_ir = up(T.ir), *d_ip = up(T.ip);
                            LaunchINTT_GS(dx, d_ir, d_ip, n, q, T.ninv, T.ninv_p, 0);
                            cudaFree(d_ir); cudaFree(d_ip);
                        }
                    }
                    uint64_t *dqhi = up(K.partQHatInv[part]), *dqhip = up(K.partQHatInvPrec[part]),
                             *dq = up(K.partSrcMod[part]), *dqmp = up(K.partQHatModp[part]),
                             *dp = up(K.partComplMod[part]), *dmlo = up(K.partBMuLo[part]), *dmhi = up(K.partBMuHi[part]);
                    uint64_t* ddst; cudaMalloc(&ddst, (size_t)sizeCompl * n * 8);
                    LaunchApproxSwitchCRTBasis(dsrc, ddst, dqhi, dqhip, dq, dqmp, dp, dmlo, dmhi, sizePart, sizeCompl, n, 0);
                    // NTT complement back to eval.
                    for (uint32_t j = 0; j < sizeCompl; ++j) {
                        uint64_t q = K.partComplMod[part][j];
                        uint64_t root = native_root(n, q);
                        Tab T = build_tab(n, q, root);
                        uint64_t *dx = ddst + (size_t)j * n;
                        uint64_t *d_fr = up(T.fr), *d_fp = up(T.fp);
                        LaunchNTT_CT(dx, d_fr, d_fp, n, q, 0);
                        cudaFree(d_fr); cudaFree(d_fp);
                    }
                    cudaDeviceSynchronize();
                    std::vector<uint64_t> complEval((size_t)sizeCompl * n);
                    cudaMemcpy(complEval.data(), ddst, (size_t)sizeCompl * n * 8, cudaMemcpyDeviceToHost);
                    cudaFree(dsrc); cudaFree(dqhi); cudaFree(dqhip); cudaFree(dq); cudaFree(dqmp);
                    cudaFree(dp); cudaFree(dmlo); cudaFree(dmhi); cudaFree(ddst);

                    std::vector<uint64_t>& d = digits[part]; d.resize((size_t)sizeQlP * n);
                    for (uint32_t i = 0; i < sizeQlP; ++i) {
                        if (i < startIdx)
                            std::copy(complEval.begin() + (size_t)i * n, complEval.begin() + (size_t)(i + 1) * n, d.begin() + (size_t)i * n);
                        else if (i < endIdx)
                            std::copy(aTowers.begin() + (size_t)i * n, aTowers.begin() + (size_t)(i + 1) * n, d.begin() + (size_t)i * n);
                        else
                            std::copy(complEval.begin() + (size_t)(i - sizePart) * n, complEval.begin() + (size_t)(i - sizePart + 1) * n, d.begin() + (size_t)i * n);
                    }

                    // CHECK A: exact identity, coefficient-0 only (constant input -> all
                    // coefficients identical, so coefficient 0 suffices and avoids
                    // dumping bulk data).
                    for (uint32_t i = 0; i < sizeQlP; ++i) {
                        uint64_t qlpMod = (i < sizeQl) ? modQ[i] : modP[i - sizeQl];
                        std::vector<uint64_t> row(d.begin() + (size_t)i * n, d.begin() + (size_t)(i + 1) * n);
                        std::vector<uint64_t> coeff = inverse_ntt(row, n, qlpMod, rootQP[i]);
                        uint64_t actual0 = coeff[0];
                        if (i >= startIdx && i < endIdx) {
                            if (actual0 != V % qlpMod) {
                                checkA_pass = false;
                                std::cout << "  [CHECK A FAIL] part=" << part << " tower=" << i
                                          << " (own range) expected=" << (V % qlpMod) << " actual=" << actual0 << "\n";
                            }
                        } else {
                            uint64_t Qpart_modqlp = Q_mod(K.partSrcMod[part], qlpMod);
                            bool found = false;
                            for (uint32_t k = 0; k <= 4 * sizePart + 2 && !found; ++k) {
                                uint64_t expect = addmod(V % qlpMod, mulmod(k % qlpMod, Qpart_modqlp, qlpMod), qlpMod);
                                if (expect == actual0) found = true;
                            }
                            if (!found) {
                                checkA_pass = false;
                                uint64_t diff = submod(actual0, V % qlpMod, qlpMod);
                                double frac = (double)diff / (double)qlpMod;
                                std::cout << "  [CHECK A FAIL] part=" << part << " tower=" << i
                                          << " (complement) no consistent k found; residual_vs_k0="
                                          << diff << " (" << (frac*100.0) << "% of q)\n";
                            }
                        }
                    }
                }
                std::cout << "  CHECK A (decompose, exact): " << (checkA_pass ? "[PASS]" : "[FAIL]") << "\n";

                // ---- STAGE 3a: accumulate (copied inline) ----
                auto qlp_mod = [&](uint32_t i) -> uint64_t { return (i < sizeQl) ? K.qMod[i] : K.pMod[i - sizeQl]; };
                std::vector<uint64_t> res0((size_t)sizeQlP * n), res1((size_t)sizeQlP * n);
                {
                    const size_t DP = (size_t)sizeQlP * n, KPs = (size_t)K.evalKeyTowers * n, OT = (size_t)sizeQlP * n;
                    uint64_t *d_dig = nullptr, *d_ka = nullptr, *d_kb = nullptr, *d_o0 = nullptr, *d_o1 = nullptr;
                    cudaMalloc(&d_dig, (size_t)numPart * DP * 8);
                    cudaMalloc(&d_ka, (size_t)numPart * KPs * 8);
                    cudaMalloc(&d_kb, (size_t)numPart * KPs * 8);
                    cudaMalloc(&d_o0, OT * 8); cudaMalloc(&d_o1, OT * 8);
                    for (uint32_t p = 0; p < numPart; ++p) {
                        cudaMemcpy(d_dig + (size_t)p * DP, digits[p].data(), DP * 8, cudaMemcpyHostToDevice);
                        cudaMemcpy(d_ka + (size_t)p * KPs, K.av[p].data(), KPs * 8, cudaMemcpyHostToDevice);
                        cudaMemcpy(d_kb + (size_t)p * KPs, K.bv[p].data(), KPs * 8, cudaMemcpyHostToDevice);
                    }
                    cudaMemset(d_o0, 0, OT * 8); cudaMemset(d_o1, 0, OT * 8);
                    for (uint32_t i = 0; i < sizeQlP; ++i) {
                        uint64_t q = qlp_mod(i);
                        uint32_t idx = i; // delta=0 since K.fullQ==sizeQl here
                        for (uint32_t part = 0; part < numPart; ++part) {
                            LaunchKSMultAcc(d_o0 + (size_t)i * n, d_o1 + (size_t)i * n,
                                            d_dig + (size_t)part * DP + (size_t)i * n,
                                            d_kb + (size_t)part * KPs + (size_t)idx * n,
                                            d_ka + (size_t)part * KPs + (size_t)idx * n, q, n, 0);
                        }
                    }
                    cudaDeviceSynchronize();
                    cudaMemcpy(res0.data(), d_o0, OT * 8, cudaMemcpyDeviceToHost);
                    cudaMemcpy(res1.data(), d_o1, OT * 8, cudaMemcpyDeviceToHost);
                    cudaFree(d_dig); cudaFree(d_ka); cudaFree(d_kb); cudaFree(d_o0); cudaFree(d_o1);
                }

                // CHECK B: exact recomputation of the accumulate step, coefficient 0 only.
                bool checkB_pass = true;
                for (uint32_t i = 0; i < sizeQlP; ++i) {
                    uint64_t q = qlp_mod(i);
                    uint64_t expect0 = 0, expect1 = 0;
                    for (uint32_t part = 0; part < numPart; ++part) {
                        uint64_t d0 = digits[part][(size_t)i * n + 0];
                        expect0 = addmod(expect0, mulmod(d0, K.bv[part][(size_t)i * n + 0], q), q);
                        expect1 = addmod(expect1, mulmod(d0, K.av[part][(size_t)i * n + 0], q), q);
                    }
                    if (expect0 != res0[(size_t)i * n + 0] || expect1 != res1[(size_t)i * n + 0]) {
                        checkB_pass = false;
                        std::cout << "  [CHECK B FAIL] tower=" << i << " expect0=" << expect0
                                  << " actual0=" << res0[(size_t)i*n+0] << " expect1=" << expect1
                                  << " actual1=" << res1[(size_t)i*n+0] << "\n";
                    }
                }
                std::cout << "  CHECK B (accumulate, exact): " << (checkB_pass ? "[PASS]" : "[FAIL]") << "\n";

                // ---- STAGE 3b: moddown (copied inline) ----
                auto moddown = [&](const std::vector<uint64_t>& res) -> std::vector<uint64_t> {
                    std::vector<uint64_t> pCoeff((size_t)sizeP * n);
                    for (uint32_t j = 0; j < sizeP; ++j) {
                        uint64_t q = K.pMod[j];
                        uint64_t root = native_root(n, q);
                        Tab T = build_tab(n, q, root);
                        std::vector<uint64_t> h(res.begin() + (size_t)(sizeQl + j) * n, res.begin() + (size_t)(sizeQl + j + 1) * n);
                        uint64_t *dx = up(h), *d_ir = up(T.ir), *d_ip = up(T.ip);
                        LaunchINTT_GS(dx, d_ir, d_ip, n, q, T.ninv, T.ninv_p, 0);
                        cudaMemcpy(pCoeff.data() + (size_t)j * n, dx, (size_t)n * 8, cudaMemcpyDeviceToHost);
                        cudaFree(dx); cudaFree(d_ir); cudaFree(d_ip);
                    }
                    uint64_t *dsrc = up(pCoeff), *dqhi = up(K.pHatInv), *dqhip = up(K.pHatInvPrec),
                             *dq = up(K.pMod), *dqmp = up(K.pHatModq), *dp = up(K.qMod),
                             *dmlo = up(K.mdBMuLo), *dmhi = up(K.mdBMuHi);
                    uint64_t* ddst; cudaMalloc(&ddst, (size_t)sizeQl * n * 8);
                    LaunchApproxSwitchCRTBasis(dsrc, ddst, dqhi, dqhip, dq, dqmp, dp, dmlo, dmhi, sizeP, sizeQl, n, 0);
                    cudaDeviceSynchronize();
                    std::vector<uint64_t> qSw((size_t)sizeQl * n);
                    cudaMemcpy(qSw.data(), ddst, (size_t)sizeQl * n * 8, cudaMemcpyDeviceToHost);
                    cudaFree(dsrc); cudaFree(dqhi); cudaFree(dqhip); cudaFree(dq); cudaFree(dqmp);
                    cudaFree(dp); cudaFree(dmlo); cudaFree(dmhi); cudaFree(ddst);
                    std::vector<uint64_t> out((size_t)sizeQl * n);
                    for (uint32_t i = 0; i < sizeQl; ++i) {
                        uint64_t q = K.qMod[i];
                        uint64_t root = native_root(n, q);
                        Tab T = build_tab(n, q, root);
                        std::vector<uint64_t> h(qSw.begin() + (size_t)i * n, qSw.begin() + (size_t)(i + 1) * n);
                        uint64_t *dx = up(h), *d_fr = up(T.fr), *d_fp = up(T.fp);
                        LaunchNTT_CT(dx, d_fr, d_fp, n, q, 0);
                        std::vector<uint64_t> ha(res.begin() + (size_t)i * n, res.begin() + (size_t)(i + 1) * n);
                        uint64_t *da = up(ha), *dans; cudaMalloc(&dans, (size_t)n * 8);
                        LaunchModDownCombine(dans, da, dx, K.pInvModq[i], q, n, 0);
                        cudaMemcpy(out.data() + (size_t)i * n, dans, (size_t)n * 8, cudaMemcpyDeviceToHost);
                        cudaFree(dx); cudaFree(d_fr); cudaFree(d_fp); cudaFree(da); cudaFree(dans);
                    }
                    return out;
                };
                std::vector<uint64_t> ba0 = moddown(res0);
                std::vector<uint64_t> ba1 = moddown(res1);

                // CHECK C: relin identity ba0+ba1*s ~= a*s^2 mod q, INTT'd, small tolerance.
                bool checkC_pass = true;
                double worstFrac = 0.0; uint32_t worstTower = 0;
                for (uint32_t i = 0; i < sizeQl; ++i) {
                    uint64_t q = modQ[i];
                    std::vector<uint64_t> sRow(KP.s.begin() + (size_t)i * n, KP.s.begin() + (size_t)(i + 1) * n);
                    std::vector<uint64_t> ba0Row(ba0.begin() + (size_t)i * n, ba0.begin() + (size_t)(i + 1) * n);
                    std::vector<uint64_t> ba1Row(ba1.begin() + (size_t)i * n, ba1.begin() + (size_t)(i + 1) * n);
                    std::vector<uint64_t> aRow(aTowers.begin() + (size_t)i * n, aTowers.begin() + (size_t)(i + 1) * n);
                    std::vector<uint64_t> lhsEval(n), rhsEval(n);
                    for (uint32_t k = 0; k < n; ++k) {
                        lhsEval[k] = addmod(ba0Row[k], mulmod(ba1Row[k], sRow[k], q), q);
                        uint64_t s2 = mulmod(sRow[k], sRow[k], q);
                        rhsEval[k] = mulmod(aRow[k], s2, q);
                    }
                    std::vector<uint64_t> lhsCoeff = inverse_ntt(lhsEval, n, q, modQ[i] == modQP[i] ? rootQP[i] : native_root(n, q));
                    std::vector<uint64_t> rhsCoeff = inverse_ntt(rhsEval, n, q, native_root(n, q));
                    uint64_t worst = 0;
                    for (uint32_t k = 0; k < n; ++k) {
                        uint64_t diff = submod(lhsCoeff[k], rhsCoeff[k], q);
                        uint64_t mag = (diff > q - diff) ? q - diff : diff;
                        if (mag > worst) worst = mag;
                    }
                    double frac = (double)worst / (double)q;
                    if (frac > worstFrac) { worstFrac = frac; worstTower = i; }
                }
                checkC_pass = worstFrac < 0.001; // 0.1% of q -- generous vs the ~1e-9% seen when correct
                std::cout << "  CHECK C (relin identity, noise-tolerant): worst=" << (worstFrac*100.0)
                          << "% of q at tower " << worstTower << " -> " << (checkC_pass ? "[PASS]" : "[FAIL]") << "\n";

                if (!checkA_pass || !checkB_pass || !checkC_pass) overallPass = false;
            }
        }
    }

    std::cout << (overallPass ? "[OVERALL PASS]\n" : "[OVERALL FAIL]\n");
    return overallPass ? 0 : 1;
}
