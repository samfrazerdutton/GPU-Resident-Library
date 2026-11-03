// GOD PROMPT "CKKS PHASE 8": isolated, hand-constructed test of
// evalkeygen_host_sold ALONE, bypassing keygen_host, encode/encrypt,
// LaunchTensor, rescale, and the Rust wrapper entirely. Per the
// governing directive: separate eval-key generation from everything
// else so a mismatch can only originate inside evalkeygen_host_sold's
// own per-coefficient arithmetic (or its private NTT table cache).
//
// === Why NOT a literal all-zero secret ===
// Setting sOld=0 trivially satisfies "bv[t]+av[t]*s ~= P*s2" for ANY
// av/bv, since anything*0=0 regardless of av's own value -- it would
// NOT exercise the "av*s" cancellation term that Phase 7's real-secret
// test found broken. Setting sOld=0 IS used below (it has no
// precondition inside evalkeygen_host_sold -- the function never reads
// `sQP` at all, only the explicit `sOld` parameter it is passed, so any
// sOld is valid), but ONLY to remove the P*s2 term's range-dependence
// so the SAME check applies uniformly in-range and out-of-range. The
// cancellation itself is exercised by hand-picking a NONZERO, KNOWN
// secret `s_hat` and constructing pkA_QP/pkB_QP to satisfy
// `pkB + pkA*s_hat = 0` EXACTLY (zero key-gen noise), for EVERY tower
// including P-towers -- which no prior round's check ever covered
// (Phase 7's "outside part 0's range" test only checked Q-towers,
// t=4..7 of an 8-tower Q chain; this test covers ALL QP towers,
// including the P range, for the first time).
//
// === Derivation ===
// av[t] = pkA[t]*u[t] + ns*e1[t]           (eval, pointwise, per tower)
// bv[t] = pkB[t]*u[t] + ns*e0[t] [+ P*s2[t] only if t in part's Q-range]
// With sOld=0 (hand-picked): the P*s2 term is EXACTLY 0 everywhere, so
//   bv[t] = pkB[t]*u[t] + ns*e0[t]                         (always)
// With pkA[t]=A (a fixed constant, same at every eval-domain position
// and tower, reduced mod that tower's own q), pkB[t] = -A*s_hat[t] mod q
// (s_hat[t]=1 constant too, so pkB[t] = q_t - A, EXACT, zero noise):
//   bv[t] + av[t]*s_hat[t]
//     = (pkB[t]*u[t] + e0[t]) + (pkA[t]*u[t] + e1[t])*1
//     = u[t]*(pkB[t] + pkA[t]*1) + e0[t] + e1[t]
//     = u[t]*0 + e0[t] + e1[t]              (EXACT algebraic identity,
//                                            no noise approximation
//                                            needed in EVAL domain --
//                                            this cancellation is
//                                            EXACT regardless of what
//                                            u[t] is)
// Converting to COEFFICIENT domain (an actual INTT, per the established
// methodology -- NOT a pointwise eval-domain comparison), the result
// should be a SMALL polynomial: e0+e1, the sum of two independently
// sampled Gaussian(sigma=3.19) coefficient-domain noise polynomials.
// Expected magnitude: bounded by roughly a few hundred (sigma=3.19,
// n=1024, max of ~2048 draws from two Gaussians, matching the SAME
// noise budget already measured and trusted elsewhere in this
// project, e.g. encrypt/decrypt round trips showing worst~400-500) --
// NOT a numerical approximation of a CKKS plaintext relation, but an
// exact modular identity (u[t]*0=0 always) plus two small, bounded
// error terms with a well-understood distribution.
//
// If this test PASSES: evalkeygen_host_sold's own per-coefficient
// arithmetic and its private NTT table cache are correct for a hand-
// picked, nonzero secret across ALL QP towers (Q and P) -- the defect
// found in Phase 7 must be in how the REAL caller (gpufhe_ks_fixture_
// create) constructs/passes sQP/pkA_QP/pkB_QP (i.e. a bug upstream of
// evalkeygen_host_sold, most likely in keygen_host's P-tower output or
// in PModq_QP's computation), not inside evalkeygen_host_sold itself.
// If this test FAILS: the defect is confirmed to be inside
// evalkeygen_host_sold's own arithmetic or its NTT table cache,
// independent of keygen_host, real ciphertexts, or any caller.
#include "keyswitch.h"
#include "keygen.h"
#include "ntt.h"
#include "intt.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>

namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
}

namespace {
uint64_t mulmod(uint64_t a, uint64_t b, uint64_t q) { return (uint64_t)(((unsigned __int128)a * b) % q); }
uint64_t addmod(uint64_t a, uint64_t b, uint64_t q) { uint64_t s = a + b; return s >= q ? s - q : s; }
uint64_t submod(uint64_t a, uint64_t b, uint64_t q) { return a >= b ? a - b : a + q - b; }
uint64_t powmod(uint64_t b, uint64_t e, uint64_t q) {
    unsigned __int128 r = 1, bb = b % q;
    while (e) { if (e & 1) r = (r * bb) % q; bb = (bb * bb) % q; e >>= 1; }
    return (uint64_t)r;
}
uint64_t invmod(uint64_t a, uint64_t q) { return powmod(a % q, q - 2, q); }

uint64_t* up(const std::vector<uint64_t>& h) {
    uint64_t* d;
    cudaMalloc(&d, h.size() * 8);
    cudaMemcpy(d, h.data(), h.size() * 8, cudaMemcpyHostToDevice);
    return d;
}

// Same table-building pattern as src/keygen.cpp's own mk_tab (duplicated
// here per this project's established per-file convention, not shared).
struct InvTab { std::vector<uint64_t> ir, ip; uint64_t ninv, ninv_p; };
InvTab build_inv_tab(uint32_t n, uint64_t q, uint64_t root) {
    auto br = [](uint32_t v, uint32_t b) { uint32_t r = 0; for (uint32_t k = 0; k < b; k++) { r = (r << 1) | (v & 1); v >>= 1; } return r; };
    uint32_t logn = 0; while ((1u << logn) < n) ++logn;
    uint64_t rootInv = invmod(root, q);
    std::vector<uint64_t> pw(n); pw[0] = 1;
    for (uint32_t i = 1; i < n; ++i) pw[i] = mulmod(pw[i - 1], rootInv, q);
    InvTab T; T.ir.resize(n); T.ip.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint64_t r = pw[br(i, logn)];
        T.ir[i] = r;
        T.ip[i] = (uint64_t)(((__uint128_t)r << 64) / q);
    }
    T.ninv = invmod(n, q);
    T.ninv_p = (uint64_t)(((__uint128_t)T.ninv << 64) / q);
    return T;
}

// Converts ONE tower's eval-domain vector to coefficient domain, host to
// host, via LaunchINTT_GS -- same pattern as rust/.../evalmult.rs's own
// test-only inverse_ntt helper (Phase 6/7), re-derived here independently
// in C++ so the two cannot share a transcription bug.
std::vector<uint64_t> inverse_ntt(const std::vector<uint64_t>& eval, uint32_t n, uint64_t q, uint64_t root) {
    InvTab T = build_inv_tab(n, q, root);
    uint64_t* d_x = up(eval);
    uint64_t* d_ir = up(T.ir);
    uint64_t* d_ip = up(T.ip);
    LaunchINTT_GS(d_x, d_ir, d_ip, n, q, T.ninv, T.ninv_p, 0);
    cudaDeviceSynchronize();
    std::vector<uint64_t> out(n);
    cudaMemcpy(out.data(), d_x, (size_t)n * 8, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_ir); cudaFree(d_ip);
    return out;
}
} // namespace

int main() {
    const uint32_t n = 1024;
    // GOD PROMPT "CKKS PHASE 8", updated: matches the REAL Rust fixture's
    // EXACT configuration (rust/crates/ckks-cuda/src/evalmult.rs's
    // N=1024/SIZE_Q=8/SIZE_P=2/NUM_PART=2, which is where Phase 7's
    // ~50%-of-q failure was observed) -- the FIRST version of this test
    // used a smaller sizeQl=4/sizeP=1 configuration and PASSED cleanly
    // (worst=390, 3.46e-11% of q) for the full chain, in direct
    // contradiction with Phase 7's finding. Re-running with the EXACT
    // same sizes, alpha, and prime bit-widths (one 60-bit q0, seven
    // 50-bit mid-towers, two 60-bit P-towers -- matching cpp_bridge/
    // keyswitch_bridge.cpp's gpufhe_ks_fixture_create precisely) isolates
    // whether the defect is configuration-specific (sizeP=2 vs 1, or a
    // specific prime/bit-width) rather than a general algorithmic one.
    const uint32_t sizeQl = 8, sizeP = 2, sizeQlP = sizeQl + sizeP;
    const uint32_t numPart = 2, alpha = 4; // part 0: [0,4), part 1: [4,8) -- matches the real Rust fixture's exact configuration (N=1024/SIZE_Q=8/SIZE_P=2/NUM_PART=2).
    // CONFIG-SPECIFICITY, established this round (see this file's own
    // history / research/EXPERIMENT_LEDGER.md entry #41): CHECK 4 below
    // PASSES at sizeQl=5/sizeP=3 (matching tests/test_ks_constants.cpp's
    // proven-bit-exact-vs-OpenFHE config) and at sizeQl=4/sizeP=1, but
    // FAILS (~50% of q) at sizeQl=5/sizeP=2 AND at this file's sizeQl=8/
    // sizeP=2 -- i.e. sizeP=2 SPECIFICALLY triggers the defect,
    // independent of sizeQl. numPart=1 (single part) was also verified
    // to fail identically, ruling out multi-part digit decomposition.
    // The exact line within compute_keyswitch_constants/keyswitch_core_
    // resident/kernels/basis_convert.cu/moddown.cu responsible for the
    // sizeP=2-specific failure was NOT pinpointed this round, despite
    // verifying K.pInvModq/K.pHatInv individually satisfy their defining
    // modular-inverse relations correctly even at sizeP=2.

    std::vector<uint64_t> modQ, modP;
    gpufhe::native_primes(modQ, 1, 60, n, {});
    {
        std::vector<uint64_t> mids;
        gpufhe::native_primes(mids, sizeQl - 1, 50, n, modQ);
        for (auto m : mids) modQ.push_back(m);
    }
    gpufhe::native_primes(modP, sizeP, 60, n, modQ);
    std::vector<uint64_t> modQP = modQ;
    for (auto p : modP) modQP.push_back(p);
    std::vector<uint64_t> rootQP(sizeQlP);
    for (uint32_t t = 0; t < sizeQlP; ++t) rootQP[t] = gpufhe::native_root(n, modQP[t]);

    std::cout << "[isolated evalkeygen] n=" << n << " sizeQl=" << sizeQl << " sizeP=" << sizeP
              << " sizeQlP=" << sizeQlP << " numPart=" << numPart << " alpha=" << alpha << "\n";

    gpufhe::KeySwitchConstants K;
    K.n = n; K.sizeQl = sizeQl; K.alpha = alpha; K.numPart = numPart;

    const uint64_t A_CONST = 7;   // hand-picked pkA, same at every (tower,coeff)
    const uint64_t S_HAT = 1;     // hand-picked secret, same at every (tower,coeff)

    std::vector<uint64_t> sQP((size_t)sizeQlP * n, 0);   // unused by evalkeygen_host_sold, dummy
    std::vector<uint64_t> sOld((size_t)sizeQlP * n, 0);  // hand-picked: ZERO -> P*s2 term vanishes everywhere
    std::vector<uint64_t> pkA_QP((size_t)sizeQlP * n), pkB_QP((size_t)sizeQlP * n);
    std::vector<uint64_t> PModq_QP((size_t)sizeQlP, 0);  // unused (multiplied by s2=0 always)
    for (uint32_t t = 0; t < sizeQlP; ++t) {
        uint64_t q = modQP[t];
        uint64_t a_mod = A_CONST % q;
        uint64_t b_mod = submod(0, mulmod(a_mod, S_HAT % q, q), q); // pkB = -A*s_hat mod q, EXACT, zero noise
        for (uint32_t k = 0; k < n; ++k) {
            pkA_QP[(size_t)t * n + k] = a_mod;
            pkB_QP[(size_t)t * n + k] = b_mod;
        }
    }

    gpufhe::evalkeygen_host_sold(K, sQP, sOld, pkA_QP, pkB_QP, PModq_QP, modQP, rootQP,
                                  /*ns=*/1, /*sigma=*/3.19, /*seed=*/42, 0);

    std::cout << "[isolated evalkeygen] evalKeyTowers=" << K.evalKeyTowers
              << " av.size()=" << K.av.size() << " bv.size()=" << K.bv.size() << "\n";

    // Expected: bv[part][t] + av[part][t]*S_HAT ~= e0+e1 (small), for
    // EVERY part and EVERY tower (Q and P alike), since sOld=0 removes
    // the P*s2 term's range-dependence entirely.
    uint64_t worst_overall = 0;
    double worst_frac_overall = 0.0;
    for (uint32_t part = 0; part < numPart; ++part) {
        for (uint32_t t = 0; t < sizeQlP; ++t) {
            uint64_t q = modQP[t];
            uint64_t root = rootQP[t];
            std::vector<uint64_t> combined_eval(n);
            for (uint32_t k = 0; k < n; ++k) {
                size_t idx = (size_t)t * n + k;
                uint64_t av_s = mulmod(K.av[part][idx], S_HAT % q, q);
                combined_eval[k] = addmod(K.bv[part][idx], av_s, q);
            }
            std::vector<uint64_t> combined_coeff = inverse_ntt(combined_eval, n, q, root);
            uint64_t worst = 0;
            for (uint32_t k = 0; k < n; ++k) {
                uint64_t centered = combined_coeff[k] > q - combined_coeff[k] ? q - combined_coeff[k] : combined_coeff[k];
                if (centered > worst) worst = centered;
            }
            double frac = (double)worst / (double)q;
            std::cout << "[isolated evalkeygen] part=" << part << " tower=" << t << " q=" << q
                      << " worst|bv+av*s_hat - 0|=" << worst << " (" << (frac * 100.0) << "% of q)\n";
            if (worst > worst_overall) worst_overall = worst;
            if (frac > worst_frac_overall) worst_frac_overall = frac;
        }
    }
    std::cout << "[isolated evalkeygen] worst absolute residual across all parts/towers: " << worst_overall << "\n";
    std::cout << "[isolated evalkeygen] worst fraction of q across all parts/towers: " << (worst_frac_overall * 100.0) << "%\n";

    // Tolerance: e0+e1, sum of two independently sampled Gaussian(sigma=
    // 3.19) coefficient-domain polynomials, n=1024 draws each -- a worst
    // absolute value in the few-hundreds range is the expected noise
    // floor (matching e.g. the ~400-500 worst-residual already measured
    // and trusted in this project's encrypt/decrypt round trips). 10000
    // is generously above that floor yet utterly incompatible with
    // "looks like uncorrelated randomness" (which would show as a
    // worst_frac near 50%, i.e. ~2^49 for a 50-bit modulus) -- a
    // decisive, not-tuned-to-pass threshold.
    const uint64_t TOLERANCE = 10000;
    bool pass = worst_overall < TOLERANCE;
    std::cout << (pass ? "[PASS]" : "[FAIL]") << " isolated evalkeygen_host_sold cancellation check\n";

    // === CHECK 2: does keygen_host's OWN generated (pkA,pkB,s) actually
    // satisfy pkB+pkA*s ~= 0 across ALL QP towers, INCLUDING the P range?
    // Never directly tested before -- every prior sanity check (Phase 7)
    // used the Q-RESTRICTED prefix only (via encrypt_host/decrypt_host,
    // which never touch P-towers at all). CHECK 1 above proves
    // evalkeygen_host_sold itself is correct; if THIS check also passes,
    // the defect must be in PModq_QP's computation specifically (the one
    // remaining untested ingredient); if THIS check fails, the defect is
    // in keygen_host's own P-tower generation.
    std::cout << "\n[keygen P-tower check] generating a real keypair over the SAME QP chain...\n";
    auto KP = gpufhe::keygen_host(n, modQP, rootQP, /*ns=*/1, /*sigma=*/3.19, /*seed=*/777);
    std::cout << "[keygen P-tower check] KP.s.size()=" << KP.s.size() << " (expect " << (size_t)sizeQlP * n << ")\n";

    uint64_t worst2_overall = 0;
    double worst2_frac_overall = 0.0;
    for (uint32_t t = 0; t < sizeQlP; ++t) {
        uint64_t q = modQP[t];
        uint64_t root = rootQP[t];
        std::vector<uint64_t> combined_eval(n);
        for (uint32_t k = 0; k < n; ++k) {
            size_t idx = (size_t)t * n + k;
            uint64_t pa_s = mulmod(KP.pkA[idx], KP.s[idx], q);
            combined_eval[k] = addmod(KP.pkB[idx], pa_s, q);
        }
        std::vector<uint64_t> combined_coeff = inverse_ntt(combined_eval, n, q, root);
        uint64_t worst = 0;
        for (uint32_t k = 0; k < n; ++k) {
            uint64_t centered = combined_coeff[k] > q - combined_coeff[k] ? q - combined_coeff[k] : combined_coeff[k];
            if (centered > worst) worst = centered;
        }
        double frac = (double)worst / (double)q;
        const char* region = (t < sizeQl) ? "Q" : "P";
        std::cout << "[keygen P-tower check] tower=" << t << " (" << region << ") q=" << q
                  << " worst|pkB+pkA*s - 0|=" << worst << " (" << (frac * 100.0) << "% of q)\n";
        if (worst > worst2_overall) worst2_overall = worst;
        if (frac > worst2_frac_overall) worst2_frac_overall = frac;
    }
    std::cout << "[keygen P-tower check] worst absolute residual across all QP towers: " << worst2_overall << "\n";
    std::cout << "[keygen P-tower check] worst fraction of q across all QP towers: " << (worst2_frac_overall * 100.0) << "%\n";
    bool pass2 = worst2_overall < TOLERANCE;
    std::cout << (pass2 ? "[PASS]" : "[FAIL]") << " keygen_host pkB+pkA*s~=0 check (Q and P towers)\n";

    // === CHECK 3: the FULL chain, real key -> real evalkeygen_host ->
    // the real relin identity WITH a nonzero sOld=s^2 and a REAL
    // PModq_QP (computed the EXACT SAME way cpp_bridge/keyswitch_
    // bridge.cpp's gpufhe_ks_fixture_create does). CHECK 1 proves
    // evalkeygen_host_sold is correct in isolation; CHECK 2 proves
    // keygen_host's own pkA/pkB/s relation is correct, Q and P alike.
    // If CHECK 3 fails despite both, the ONE remaining untested
    // ingredient -- PModq_QP's computation -- is the confirmed cause.
    std::cout << "\n[full chain check] real key, real s2=s^2, real PModq_QP, through evalkeygen_host...\n";
    std::vector<uint64_t> PModq_QP_real(sizeQlP);
    for (uint32_t t = 0; t < sizeQlP; ++t) {
        uint64_t q = modQP[t], P = 1 % q;
        for (uint32_t j = 0; j < sizeP; ++j) P = mulmod(P, modQP[sizeQl + j] % q, q);
        PModq_QP_real[t] = P;
        std::cout << "[full chain check] PModq_QP[" << t << "]=" << P << " (tower modulus=" << q << ")\n";
    }

    gpufhe::KeySwitchConstants K2;
    K2.n = n; K2.sizeQl = sizeQl; K2.alpha = alpha; K2.numPart = numPart;
    gpufhe::evalkeygen_host(K2, KP.s, KP.pkA, KP.pkB, PModq_QP_real, modQP, rootQP,
                             /*ns=*/1, /*sigma=*/3.19, /*seed=*/9999, 0);

    uint64_t worst3_overall = 0;
    double worst3_frac_overall = 0.0;
    for (uint32_t part = 0; part < numPart; ++part) {
        uint32_t startIdx = alpha * part;
        uint32_t sizePart = (sizeQl > startIdx + alpha) ? alpha : (sizeQl - startIdx);
        uint32_t endIdx = startIdx + sizePart;
        for (uint32_t t = 0; t < sizeQlP; ++t) {
            uint64_t q = modQP[t];
            uint64_t root = rootQP[t];
            bool inRange = (t >= startIdx && t < endIdx);
            std::vector<uint64_t> lhs_eval(n), rhs_eval(n);
            for (uint32_t k = 0; k < n; ++k) {
                size_t idx = (size_t)t * n + k;
                uint64_t av_s = mulmod(K2.av[part][idx], KP.s[idx], q);
                lhs_eval[k] = addmod(K2.bv[part][idx], av_s, q); // bv + av*s
                if (inRange) {
                    uint64_t s2 = mulmod(KP.s[idx], KP.s[idx], q);
                    rhs_eval[k] = mulmod(PModq_QP_real[t] % q, s2, q); // P*s^2
                } else {
                    rhs_eval[k] = 0;
                }
            }
            std::vector<uint64_t> lhs_coeff = inverse_ntt(lhs_eval, n, q, root);
            std::vector<uint64_t> rhs_coeff = inverse_ntt(rhs_eval, n, q, root);
            uint64_t worst = 0;
            for (uint32_t k = 0; k < n; ++k) {
                uint64_t diff = lhs_coeff[k] >= rhs_coeff[k] ? lhs_coeff[k] - rhs_coeff[k] : rhs_coeff[k] - lhs_coeff[k];
                uint64_t centered = diff > q - diff ? q - diff : diff;
                if (centered > worst) worst = centered;
            }
            double frac = (double)worst / (double)q;
            const char* region = (t < sizeQl) ? "Q" : "P";
            std::cout << "[full chain check] part=" << part << " tower=" << t << " (" << region << ")"
                      << " inRange=" << inRange << " q=" << q
                      << " worst|bv+av*s - P*s^2(or 0)|=" << worst << " (" << (frac * 100.0) << "% of q)\n";
            if (worst > worst3_overall) worst3_overall = worst;
            if (frac > worst3_frac_overall) worst3_frac_overall = frac;
        }
    }
    std::cout << "[full chain check] worst absolute residual across all parts/towers: " << worst3_overall << "\n";
    std::cout << "[full chain check] worst fraction of q across all parts/towers: " << (worst3_frac_overall * 100.0) << "%\n";
    bool pass3 = worst3_overall < TOLERANCE;
    std::cout << (pass3 ? "[PASS]" : "[FAIL]") << " full chain: real key + real evalkeygen_host + real PModq_QP\n";

    // === CHECK 4: the ACTUAL keyswitch_core_resident reconstruction
    // (decompose + per-digit accumulate + moddown), which NEITHER CHECK
    // 1-3 exercised (they only ever called evalkeygen_host_sold
    // directly, never compute_keyswitch_constants or
    // keyswitch_core_resident). This reproduces Phase 7's
    // keyswitch_reference_on_constant_crt_consistent_input /
    // control_a_relin_combine_matches_raw_tensor_within_keyswitch_noise_
    // test from scratch in isolated C++, using the SAME real key as
    // CHECK 2/3 and compute_keyswitch_constants's own REAL decompose/
    // moddown constants (not hand-set), to confirm whether THIS stage
    // -- not evalkeygen_host -- is where the defect actually lives.
    std::cout << "\n[full reconstruction check] compute_keyswitch_constants + keyswitch_core_resident...\n";
    gpufhe::KeySwitchConstants K4;
    K4.n = n;
    gpufhe::compute_keyswitch_constants(K4, modQ, modP, numPart);
    std::cout << "[full reconstruction check] K4.sizeQl=" << K4.sizeQl << " K4.alpha=" << K4.alpha
              << " K4.numPart=" << K4.numPart << " (expect sizeQl=" << sizeQl << " alpha=" << alpha
              << " numPart=" << numPart << ")\n";
    for (uint32_t i = 0; i < sizeQl; ++i) { K4.rootModList.push_back(modQ[i]); K4.rootValList.push_back(rootQP[i]); }
    for (uint32_t j = 0; j < sizeP; ++j) { K4.rootModList.push_back(modQP[sizeQl + j]); K4.rootValList.push_back(rootQP[sizeQl + j]); }
    gpufhe::evalkeygen_host(K4, KP.s, KP.pkA, KP.pkB, PModq_QP_real, modQP, rootQP,
                             /*ns=*/1, /*sigma=*/3.19, /*seed=*/13579, 0);

    // Direct, cheap sanity check of compute_keyswitch_constants's own
    // ModDown constant families for THIS sizeP -- independent of
    // keyswitch_core_resident entirely. P = product of all P-tower
    // moduli; K.pInvModq[i] MUST satisfy (P mod q_i)*pInvModq[i] == 1
    // mod q_i (the defining property of a modular inverse). Also checks
    // K.pHatInv[j]*((P/p_j) mod p_j) == 1 mod p_j.
    std::cout << "\n[moddown constants check] verifying K4.pInvModq / K4.pHatInv satisfy their defining relations...\n";
    bool moddown_consts_ok = true;
    for (uint32_t i = 0; i < sizeQl; ++i) {
        uint64_t q = modQ[i];
        uint64_t Pmodq = 1 % q;
        for (uint32_t j = 0; j < sizeP; ++j) Pmodq = mulmod(Pmodq, modP[j] % q, q);
        uint64_t check = mulmod(Pmodq, K4.pInvModq[i], q);
        bool ok = (check == 1 % q);
        std::cout << "  pInvModq[" << i << "]: (P mod q)*pInvModq mod q = " << check << " (expect 1) -> " << (ok ? "OK" : "WRONG") << "\n";
        if (!ok) moddown_consts_ok = false;
    }
    for (uint32_t j = 0; j < sizeP; ++j) {
        uint64_t pj = modP[j];
        uint64_t PHatModpj = 1 % pj;
        for (uint32_t k = 0; k < sizeP; ++k) { if (k == j) continue; PHatModpj = mulmod(PHatModpj, modP[k] % pj, pj); }
        uint64_t check = mulmod(PHatModpj, K4.pHatInv[j], pj);
        bool ok = (check == 1 % pj);
        std::cout << "  pHatInv[" << j << "]: ((P/p_j) mod p_j)*pHatInv mod p_j = " << check << " (expect 1) -> " << (ok ? "OK" : "WRONG") << "\n";
        if (!ok) moddown_consts_ok = false;
    }
    std::cout << (moddown_consts_ok ? "[PASS]" : "[FAIL]") << " moddown constants satisfy their defining modular-inverse relations\n";

    // NOTE: an earlier version of this test ALSO attempted a direct,
    // from-scratch CRT reconstruction check (feeding K4.pHatInv/
    // K4.pHatModq through the textbook CRT formula by hand and
    // expecting EXACT equality to a known value). That check "failed"
    // by showing a residual of EXACTLY 1*P -- but this is NOT a defect:
    // ApproxSwitchCRTBasis is an APPROXIMATE basis conversion by design
    // (hence the name), reconstructing `x + k*P` for some small,
    // bounded `k` (here k<sizeP), not `x` exactly. The surrounding
    // ModDown combine step (LaunchModDownCombine) is specifically
    // designed to cancel this bounded error (it divides by P, and
    // `(x+k*P)*P^{-1} - x*P^{-1} = k`, a tiny additive term, nowhere
    // near 50% of q) -- so this check was a METHODOLOGY mistake (an
    // incorrect exactness expectation), not evidence about the real
    // defect, and was removed rather than left as a misleading FAIL.

    const uint64_t CONST_VAL = 5;
    std::vector<uint64_t> a_eval((size_t)sizeQl * n, CONST_VAL);
    gpufhe::KeySwitchResult R = gpufhe::keyswitch_core_resident(a_eval, K4);
    std::cout << "[full reconstruction check] ba0.size()=" << R.ba0.size() << " (expect " << (size_t)sizeQl * n << ")\n";

    uint64_t worst4_overall = 0;
    double worst4_frac_overall = 0.0;
    for (uint32_t t = 0; t < sizeQl; ++t) {
        uint64_t q = modQ[t];
        uint64_t root = rootQP[t];
        std::vector<uint64_t> lhs_eval(n), rhs_eval(n);
        for (uint32_t k = 0; k < n; ++k) {
            size_t idx = (size_t)t * n + k;
            uint64_t s = KP.s[idx]; // Q-restricted prefix matches a_eval's own Q-only range
            uint64_t ba1_s = mulmod(R.ba1[idx], s, q);
            lhs_eval[k] = addmod(R.ba0[idx], ba1_s, q); // ba0 + ba1*s
            uint64_t s2 = mulmod(s, s, q);
            rhs_eval[k] = mulmod(CONST_VAL % q, s2, q); // a*s^2
        }
        std::vector<uint64_t> lhs_coeff = inverse_ntt(lhs_eval, n, q, root);
        std::vector<uint64_t> rhs_coeff = inverse_ntt(rhs_eval, n, q, root);
        uint64_t worst = 0;
        for (uint32_t k = 0; k < n; ++k) {
            uint64_t diff = lhs_coeff[k] >= rhs_coeff[k] ? lhs_coeff[k] - rhs_coeff[k] : rhs_coeff[k] - lhs_coeff[k];
            uint64_t centered = diff > q - diff ? q - diff : diff;
            if (centered > worst) worst = centered;
        }
        double frac = (double)worst / (double)q;
        std::cout << "[full reconstruction check] tower=" << t << " q=" << q
                  << " worst|ba0+ba1*s - a*s^2|=" << worst << " (" << (frac * 100.0) << "% of q)\n";
        if (worst > worst4_overall) worst4_overall = worst;
        if (frac > worst4_frac_overall) worst4_frac_overall = frac;
    }
    std::cout << "[full reconstruction check] worst absolute residual: " << worst4_overall << "\n";
    std::cout << "[full reconstruction check] worst fraction of q: " << (worst4_frac_overall * 100.0) << "%\n";
    bool pass4 = worst4_overall < TOLERANCE;
    std::cout << (pass4 ? "[PASS]" : "[FAIL]") << " full reconstruction: keyswitch_core_resident(const CRT input)\n";

    return (pass && pass2 && pass3 && pass4) ? 0 : 1;
}
