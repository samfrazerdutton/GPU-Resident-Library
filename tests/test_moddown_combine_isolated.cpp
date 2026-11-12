// GOD PROMPT "CKKS PHASE 9": Stage D isolation of keyswitch_core_resident's
// own moddown() lambda (src/keyswitch.cpp STAGE 3b) at REAL polynomial scale
// (n=1024, not the n=1 coefficient trick Stage B used), replicating its exact
// sequence of operations (INTT P-towers -> coeff; ApproxSwitchCRTBasis P->Q;
// NTT back; LaunchModDownCombine) line-for-line from the production lambda,
// but driven with a HAND-CONSTRUCTED input so the expected output is an EXACT
// modular identity, not an approximate one -- isolating STAGE 3B alone, with
// no decompose/accumulate/eval-key/digit-decomposition involved at all.
//
// === Derivation of the expected result ===
// Construct a fake "res" vector (what stage 3a would normally produce) by
// hand: set res[Q-tower i] = A (same constant, EVERY coefficient, EVERY Q
// tower -- a constant-eval-domain vector, which INTTs to the integer
// constant polynomial A, exactly as established in Phase 8's CHECK4 test).
// ALSO set res[P-tower j] = A (the SAME constant A, reduced mod each P
// modulus) -- i.e. construct the P-part to represent the SAME true integer A
// that the Q-part already holds directly.
//
// Per Stage B's already-confirmed finding, ApproxSwitchCRTBasis(P->Q) on a
// polynomial representing integer A reconstructs (A + k*P) mod q_i for some
// small consistent k (0 <= k <= sizeP, found by Stage B's search). So:
//   da[i]  = A mod q_i                         (direct Q-part residue)
//   dx[i]  = (A + k*P) mod q_i                 (P-part switched to Q)
//   diff   = (da - dx) mod q_i = (-k*P) mod q_i
//   ans[i] = diff * PInvModq[i] mod q_i = (-k) mod q_i   (EXACT, since
//            P*PInvModq[i] == 1 mod q_i by definition)
// This is an EXACT, zero-tolerance prediction -- a tiny, specific residue
// (q_i - k, or 0 if k=0) at EVERY coefficient and EVERY Q tower -- not an
// approximate "small noise" claim. If keyswitch_core_resident's moddown
// logic (the ACTUAL combine step, not just the constants) is correct, this
// must hold EXACTLY. A ~50%-of-q discrepancy here would mean the defect is
// in the INTT/NTT/Combine sequencing or in how `res` values are plumbed
// through moddown -- NOT in the basis-conversion constants (already cleared
// by Stage B).
#include "keyswitch.h"
#include "ntt.h"
#include "intt.h"
#include "basis_convert.h"
#include "moddown.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>

namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void compute_keyswitch_constants(KeySwitchConstants&, const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t);
}

namespace {
uint64_t mulmod(uint64_t a, uint64_t b, uint64_t q) { return (uint64_t)(((unsigned __int128)a * b) % q); }
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

    for (uint32_t sizeQl : {5u, 8u}) {
        for (uint32_t sizeP : {1u, 2u, 3u}) {
            std::vector<uint64_t> modQ, modP;
            native_primes(modQ, 1, 60, n, {});
            { std::vector<uint64_t> mids; native_primes(mids, sizeQl - 1, 50, n, modQ);
              for (auto m : mids) modQ.push_back(m); }
            native_primes(modP, sizeP, 60, n, modQ);

            KeySwitchConstants K; K.n = n;
            compute_keyswitch_constants(K, modQ, modP, /*numPart=*/1);

            const uint64_t A = 999999999999ULL;

            // Build "res": Q-towers eval-domain constant A, P-towers eval-domain constant A.
            // (constant-eval-domain == constant-coefficient-domain, per Phase 8's own
            // established fact, reused here deliberately since it is already verified)
            std::vector<uint64_t> res((size_t)(sizeQl + sizeP) * n);
            for (uint32_t i = 0; i < sizeQl; ++i) {
                uint64_t a_mod = A % modQ[i];
                for (uint32_t k = 0; k < n; ++k) res[(size_t)i * n + k] = a_mod;
            }
            for (uint32_t j = 0; j < sizeP; ++j) {
                uint64_t a_mod = A % modP[j];
                for (uint32_t k = 0; k < n; ++k) res[(size_t)(sizeQl + j) * n + k] = a_mod;
            }

            // === Replicate keyswitch.cpp's moddown() lambda EXACTLY ===
            // Step 1: P towers (eval) -> coeff via INTT.
            std::vector<uint64_t> pCoeff((size_t)sizeP * n);
            for (uint32_t j = 0; j < sizeP; ++j) {
                uint64_t q = modP[j];
                uint64_t root = native_root(n, q);
                Tab T = build_tab(n, q, root);
                std::vector<uint64_t> h(res.begin() + (size_t)(sizeQl + j) * n, res.begin() + (size_t)(sizeQl + j + 1) * n);
                uint64_t *dx = up(h), *d_ir = up(T.ir), *d_ip = up(T.ip);
                LaunchINTT_GS(dx, d_ir, d_ip, n, q, T.ninv, T.ninv_p, 0);
                cudaMemcpy(pCoeff.data() + (size_t)j * n, dx, (size_t)n * 8, cudaMemcpyDeviceToHost);
                cudaFree(dx); cudaFree(d_ir); cudaFree(d_ip);
            }

            // Step 2: ApproxSwitchCRTBasis P -> Q (coeff domain).
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

            // Step 3: NTT qSw back to eval, then combine with original Q-range "da".
            std::vector<uint64_t> out((size_t)sizeQl * n);
            for (uint32_t i = 0; i < sizeQl; ++i) {
                uint64_t q = modQ[i];
                uint64_t root = native_root(n, q);
                Tab T = build_tab(n, q, root);
                std::vector<uint64_t> h(qSw.begin() + (size_t)i * n, qSw.begin() + (size_t)(i + 1) * n);
                uint64_t *dx = up(h), *d_fr = up(T.fr), *d_fp = up(T.fp);
                LaunchNTT_CT(dx, d_fr, d_fp, n, q, 0);
                std::vector<uint64_t> ha(res.begin() + (size_t)i * n, res.begin() + (size_t)(i + 1) * n);
                uint64_t *da = up(ha), *dans; cudaMalloc(&dans, (size_t)n * 8);
                LaunchModDownCombine(dans, da, dx, K.pInvModq[i], (uint64_t)(((unsigned __int128)K.pInvModq[i] << 64) / q), q, n, 0);
                cudaMemcpy(out.data() + (size_t)i * n, dans, (size_t)n * 8, cudaMemcpyDeviceToHost);
                cudaFree(dx); cudaFree(d_fr); cudaFree(d_fp); cudaFree(da); cudaFree(dans);
            }

            // === Independent expected value ===
            // Find k via tower 0 (brute force small range), then verify ALL
            // towers AND all n coefficients match (out[i][*] == (-k) mod q_i)
            // EXACTLY -- zero tolerance, since this is a derived exact identity.
            std::vector<uint64_t> Pmodqi(sizeQl);
            for (uint32_t i = 0; i < sizeQl; ++i) Pmodqi[i] = P_mod(modP, modQ[i]);

            int foundK = -1;
            for (uint32_t k = 0; k <= 4 * sizeP + 2 && foundK < 0; ++k) {
                uint64_t q0 = modQ[0];
                uint64_t expect0 = (q0 - (k % q0)) % q0;
                if (out[0] == expect0) foundK = (int)k;
            }

            bool configPass = (foundK >= 0);
            uint64_t worstResidual = 0; uint32_t worstTower = 0;
            if (configPass) {
                for (uint32_t i = 0; i < sizeQl && configPass; ++i) {
                    uint64_t q = modQ[i];
                    uint64_t expect = (q - ((uint64_t)foundK % q)) % q;
                    for (uint32_t k = 0; k < n; ++k) {
                        uint64_t actual = out[(size_t)i * n + k];
                        if (actual != expect) {
                            uint64_t diff = actual > expect ? actual - expect : expect - actual;
                            if (diff > worstResidual) { worstResidual = diff; worstTower = i; }
                            configPass = false;
                        }
                    }
                }
            }

            std::cout << "=== sizeQl=" << sizeQl << " sizeP=" << sizeP << " A=" << A
                      << " foundK=" << foundK << " -> "
                      << (configPass ? "[PASS]" : "[FAIL]") << "\n";
            if (!configPass) {
                allPass = false;
                if (foundK < 0) {
                    std::cout << "  no k in [0," << (4*sizeP+2) << "] matched tower 0's output="
                              << out[0] << " (q0=" << modQ[0] << ")\n";
                } else {
                    uint64_t q = modQ[worstTower];
                    double frac = (double)worstResidual / (double)q;
                    std::cout << "  first divergence at tower " << worstTower << " q=" << q
                              << " worst_residual=" << worstResidual << " (" << (frac*100.0) << "% of q)\n";
                }
            }
        }
    }

    std::cout << (allPass ? "[OVERALL PASS]\n" : "[OVERALL FAIL]\n");
    return allPass ? 0 : 1;
}
