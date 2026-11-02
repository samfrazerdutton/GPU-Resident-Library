// GOD PROMPT "CKKS PHASE 6": root-causing the relinearization mathematics
// with OpenFHE as the operational reference. Two independent checks:
//
// 1. A GENUINE cc->EvalMult() + cc->Decrypt() round trip, using OpenFHE's
//    actual public API end to end (not the algebraic shortcut) -- proves
//    OpenFHE's own relin decrypts correctly via normal library usage.
// 2. The SAME algebraic identity entry #30/#38 found invalid
//    (ba0+ba1*s =?= t2*s^2), re-checked here using OpenFHE's OWN DCRTPoly
//    arithmetic operators directly (*, +=) instead of hand-extracting raw
//    towers into uint64 arrays and doing manual mulmod/addmod -- entry
//    #30's own check extracted the secret key via
//    `sElem.SetFormat(Format::EVALUATION)` and then read raw per-tower
//    uint64 values by hand; if THAT extraction (or the manual modular
//    arithmetic mirroring it) had a subtle bug, letting OpenFHE's own,
//    already-hardened polynomial multiplication operator do the work
//    instead would avoid it entirely and isolate whether the invalid
//    result was a TEST bug or a genuine mathematical/convention gap.
//
// OpenFHE version used: 1.5.1 (installed at /usr/local, source tree at
// ~/openfhe-development, tag v1.5.1-1-ged361af2 -- confirmed via
// `git describe` before writing this file, not assumed).
//
// Source locations inspected (OpenFHE v1.5.1-1-ged361af2):
//   - src/pke/include/cryptocontext.h:1871 -- cc->EvalMult(ct1,ct2) entry
//     point, calls GetScheme()->EvalMult(ct1,ct2,evalKeyVec[0]).
//   - src/pke/include/schemebase/base-scheme.h:491,517 -- the 2-arg (raw
//     tensor, no relin) and 3-arg (with relin) scheme-level EvalMult
//     overloads, both exposed as public virtual methods -- confirmed
//     usable directly from a test via cc->GetScheme()->EvalMult(...).
//   - src/pke/lib/schemerns/rns-leveledshe.cpp:182 -- LeveledSHERNS::
//     EvalMult(ct1,ct2): calls AdjustForMultInPlace then EvalMultCore.
//   - src/pke/lib/schemebase/base-leveledshe.cpp:608 -- EvalMultCore:
//     confirmed IDENTICAL tensor-product formula to this project's own
//     kernels/tensor.cu (cvr[0]=c0a*c0b, cvr[1]=c0a*c1b+c1a*c0b,
//     cvr[2]=c1a*c1b) -- no discrepancy here.
//   - src/pke/lib/schemebase/base-leveledshe.cpp:202 -- the 3-arg EvalMult
//     WITH relin: `auto ab = ...KeySwitchCore(cv[2], evalKey); cv[0] +=
//     ab[0]; cv[1] += ab[1];` -- CONFIRMED IDENTICAL to this project's own
//     kernels/tensor.cu LaunchCombine formula (r0=t0+ba0, r1=t1+ba1). This
//     means LaunchCombine's addition formula is NOT the source of
//     disagreement -- OpenFHE uses the exact same combine step.
#include "openfhe.h"
#include <iostream>
#include <vector>

using namespace lbcrypto;

int main() {
    // Auto-flush every insertion -- stdout is fully buffered when piped
    // (as in this environment), and an uncaught C++ exception calls
    // std::terminate/abort WITHOUT flushing, which would otherwise lose
    // every diagnostic print before the crash point.
    std::cout << std::unitbuf;

    // Same defense-in-depth as this project's other OpenFHE-sourced tests:
    // disable the external GPU-shadowing fork's intercepts so this is
    // genuinely vanilla OpenFHE CPU code, not a silently-patched path.
    setenv("OPENFHE_GPU", "0", 1);
    setenv("OPENFHE_GPU_KS", "0", 1);

    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(2);
    params.SetScalingModSize(50);
    params.SetRingDim(32768);
    params.SetBatchSize(8);
    params.SetKeySwitchTechnique(HYBRID);
    // Deliberately default ScalingTechnique (do not force FIXEDMANUAL) so
    // cc->EvalMult's own automatic rescale-adjustment runs exactly as a
    // normal application would use it -- the most "real world" check.
    auto cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);

    std::vector<double> va = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> vb = {5.0, 6.0, 7.0, 8.0};
    auto pta = cc->MakeCKKSPackedPlaintext(va);
    auto ptb = cc->MakeCKKSPackedPlaintext(vb);
    auto cta = cc->Encrypt(kp.publicKey, pta);
    auto ctb = cc->Encrypt(kp.publicKey, ptb);

    // ===== Check 1: REAL cc->EvalMult + cc->Decrypt round trip =====
    auto product = cc->EvalMult(cta, ctb);
    Plaintext ptResult;
    cc->Decrypt(kp.secretKey, product, &ptResult);
    ptResult->SetLength(4);
    auto decoded = ptResult->GetRealPackedValue();

    std::vector<double> expected = {5.0, 12.0, 21.0, 32.0};
    double maxErr = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        double err = std::abs(decoded[i] - expected[i]);
        if (err > maxErr) maxErr = err;
        std::cout << "  slot " << i << ": decoded=" << decoded[i] << " expected=" << expected[i]
                  << " abs_err=" << err << "\n";
    }
    // Tolerance: standard CKKS approximation + RLWE encryption noise at
    // ScalingModSize=50 -- expect error many orders of magnitude below 1.
    bool check1_pass = maxErr < 1e-4;
    std::cout << "CHECK 1 (real cc->EvalMult + cc->Decrypt): max_err=" << maxErr
              << " -> " << (check1_pass ? "PASS" : "FAIL") << "\n";

    // ===== Check 2: the algebraic relin identity, using OpenFHE's OWN
    // DCRTPoly arithmetic operators directly (no hand-extraction). =====
    const auto& evalKeyVec = CryptoContextImpl<DCRTPoly>::GetEvalMultKeyVector(kp.secretKey->GetKeyTag());
    auto evalKey = evalKeyVec[0];

    // Raw tensor product, UNRELINEARIZED (the 2-arg scheme-level EvalMult
    // -- confirmed above to be EvalMultCore, matching kernels/tensor.cu).
    auto tensorResult = cc->GetScheme()->EvalMult(cta, ctb);
    auto& cv = tensorResult->GetElements();
    if (cv.size() != 3) {
        std::cout << "CHECK 2: unexpected component count " << cv.size() << " (expected 3) -- aborting\n";
        return 1;
    }
    DCRTPoly t0 = cv[0], t1 = cv[1], t2 = cv[2];

    auto ab = cc->GetScheme()->KeySwitchCore(t2, evalKey);
    DCRTPoly ba0 = (*ab)[0], ba1 = (*ab)[1];

    // Discriminating check (CHECK 1b): does the REAL pipeline's own
    // combined ciphertext (`product` from Check 1, whose cv[0]/cv[1] are
    // literally t0+=ab[0], t1+=ab[1] per base-leveledshe.cpp:202-214) match
    // t0+ba0 / t1+ba1 computed INDEPENDENTLY here via a fresh call to the
    // 2-arg EvalMult + KeySwitchCore? If yes, extraction of t0/t1/ba0/ba1
    // is correct and the bug must be in how Check 2 forms the t2*s^2 side
    // (or in s itself); if no, re-deriving these components via a second,
    // separate call sequence diverges from the real pipeline's internal
    // state (e.g. non-determinism, hidden mutation, or RNG-driven re-sampled
    // noise on the second EvalMult call).
    auto& pcv = product->GetElements();
    DCRTPoly recombined0 = t0 + ba0;
    DCRTPoly recombined1 = t1 + ba1;
    bool combine_matches = (recombined0 == pcv[0]) && (recombined1 == pcv[1]);
    std::cout << "  CHECK 1b: real pipeline's combined c0/c1 == independently recomputed t0+ba0/t1+ba1? "
              << (combine_matches ? "YES (extraction correct)" : "NO (extraction/sequencing bug)") << "\n";

    DCRTPoly sElem = kp.secretKey->GetPrivateElement();
    std::cout << "  secret key format: " << (sElem.GetFormat() == Format::EVALUATION ? "EVALUATION" : "COEFFICIENT")
              << "\n";
    std::cout << "  t2 format: " << (t2.GetFormat() == Format::EVALUATION ? "EVALUATION" : "COEFFICIENT") << "\n";
    std::cout << "  ba0 format: " << (ba0.GetFormat() == Format::EVALUATION ? "EVALUATION" : "COEFFICIENT") << "\n";
    std::cout << "  sElem towers=" << sElem.GetAllElements().size() << " t2 towers=" << t2.GetAllElements().size()
              << " ba0 towers=" << ba0.GetAllElements().size() << "\n";

    // Diagnostic added after Check 2 first failed at ~50% of q despite
    // Check 1 (real decrypt) passing -- that combination is mathematically
    // implausible if the per-tower moduli genuinely line up (decrypting
    // correctly requires (t0+ba0)+(t1+ba1)*s ~= t0+t1*s+t2*s^2, which forces
    // ba0+ba1*s ~= t2*s^2 up to small noise, NOT 50% of q). Print each
    // tower's modulus for t2/ba0 vs. sElem directly, rather than assuming
    // "same index == same modulus" -- the likely bug is a tower-ORDER or
    // tower-SET mismatch between the ciphertext's basis and the secret
    // key's stored basis.
    for (size_t i = 0; i < t2.GetAllElements().size(); ++i) {
        auto ti = t2.GetElementAtIndex(i);
        std::cout << "  t2/ba0 tower " << i << " modulus=" << ti.GetModulus()
                  << " root=" << ti.GetRootOfUnity() << " ringdim=" << ti.GetLength() << "\n";
    }
    for (size_t i = 0; i < sElem.GetAllElements().size(); ++i) {
        auto si = sElem.GetElementAtIndex(i);
        std::cout << "  sElem tower " << i << " modulus=" << si.GetModulus()
                  << " root=" << si.GetRootOfUnity() << " ringdim=" << si.GetLength() << "\n";
    }

    // sElem has MORE towers than ba0/t2 (the Hybrid key's secret is
    // carried over the EXTENDED Q+P basis internally, confirmed
    // empirically: sElem has 4 towers vs. t2/ba0's 3 for this Q-only
    // ciphertext -- a genuinely useful finding in itself, not assumed).
    // DCRTPoly's own `*`/`+` operators hard-reject mismatched tower
    // counts (confirmed: an earlier version of this exact line crashed
    // with "Times(): tower size mismatch"), so compare PER-TOWER via
    // NativePoly directly instead -- this sidesteps that check entirely
    // while still using OpenFHE's own per-tower NativePoly arithmetic
    // (`*`, `+=`), not hand-rolled uint64 modular arithmetic.
    size_t nTowersCompare = std::min({ba0.GetAllElements().size(), t2.GetAllElements().size(),
                                       sElem.GetAllElements().size()});
    std::cout << "  comparing first " << nTowersCompare << " towers (of sElem's "
              << sElem.GetAllElements().size() << ", assuming Q-towers-first ordering, same as this"
              << " project's own test_keyswitch.cpp convention)\n";

    // CHECK 1c: Check 1/1b prove the RELINEARIZED path (t0+ba0, t1+ba1)
    // decrypts correctly. The derivation that ba0+ba1*s ~= t2*s^2 MUST
    // follow from that ALSO assumes the raw, un-relinearized tensor
    // identity t0+t1*s+t2*s^2 ~= Delta*m holds for this same s -- never
    // actually confirmed directly until now. Build that raw combination
    // per tower (NativePoly ops, same pattern as Check 2) and decrypt it
    // via a (raw, 0) ciphertext trick: Decrypt computes c0+c1*s, so with
    // c1=0 this directly decrypts `raw` as if it were the full message.
    {
        DCRTPoly rawCombined = t2;
        DCRTPoly zeroPoly = t2;
        for (size_t i = 0; i < nTowersCompare; ++i) {
            NativePoly t0i = t0.GetElementAtIndex(i);
            NativePoly t1i = t1.GetElementAtIndex(i);
            NativePoly t2i = t2.GetElementAtIndex(i);
            NativePoly si  = sElem.GetElementAtIndex(i);
            NativePoly raw_i = t0i + (t1i * si) + ((t2i * si) * si);
            rawCombined.SetElementAtIndex(i, raw_i);
            zeroPoly.SetElementAtIndex(i, t2i * 0);
        }
        auto rawCt = product->Clone();
        rawCt->SetElements({rawCombined, zeroPoly});
        Plaintext rawPt;
        cc->Decrypt(kp.secretKey, rawCt, &rawPt);
        rawPt->SetLength(4);
        auto rawDecoded = rawPt->GetRealPackedValue();
        double rawMaxErr = 0.0;
        for (size_t i = 0; i < expected.size(); ++i) {
            double err = std::abs(rawDecoded[i] - expected[i]);
            if (err > rawMaxErr) rawMaxErr = err;
        }
        std::cout << "  CHECK 1c: raw tensor t0+t1*s+t2*s^2 decrypted directly (c1=0 trick): decoded=["
                  << rawDecoded[0] << "," << rawDecoded[1] << "," << rawDecoded[2] << "," << rawDecoded[3]
                  << "] expected=[5,12,21,32] max_err=" << rawMaxErr
                  << " -> " << (rawMaxErr < 1e-2 ? "PASS (raw tensor identity holds)" : "FAIL (raw tensor identity itself is broken)")
                  << "\n";
    }

    // CHECK 1c validated GetElementAtIndex for t0/t1/t2/sElem (via the raw
    // tensor identity), but NEVER exercised it on ba0/ba1 (the KEY-SWITCH
    // OUTPUT) at all -- that path is untested. Directly compare
    // ba0.GetElementAtIndex(i) against ba0.GetAllElements()[i] (two
    // different DCRTPoly accessors that must return the identical tower)
    // to isolate whether GetElementAtIndex itself misbehaves specifically
    // for a key-switched polynomial.
    for (size_t i = 0; i < nTowersCompare; ++i) {
        NativePoly viaIndex = ba0.GetElementAtIndex(i);
        const NativePoly& viaAll = ba0.GetAllElements()[i];
        bool same = (viaIndex == viaAll);
        std::cout << "  CHECK 1d: ba0 tower " << i << " GetElementAtIndex == GetAllElements()[i]? "
                  << (same ? "YES" : "NO (GetElementAtIndex bug on key-switch output)") << "\n";
    }

    double worstFracOfQ = 0.0;
    for (size_t i = 0; i < nTowersCompare; ++i) {
        // Per-tower NativePoly views -- OpenFHE's own multiply/add
        // operators, just scoped to ONE tower so the overall tower-count
        // mismatch never enters into it.
        NativePoly ba0i = ba0.GetElementAtIndex(i);
        NativePoly ba1i = ba1.GetElementAtIndex(i);
        NativePoly t2i  = t2.GetElementAtIndex(i);
        NativePoly si   = sElem.GetElementAtIndex(i);

        NativePoly lhs = ba0i + (ba1i * si);
        NativePoly rhs = (t2i * si) * si;

        // CRITICAL METHODOLOGY FIX: lhs/rhs are in Format::EVALUATION
        // (NTT/point-value domain). NTT is a BIJECTIVE but highly MIXING
        // linear transform -- a tiny COEFFICIENT-domain difference (the
        // genuine key-switch noise, small relative to q) spreads, via the
        // NTT, into something that looks like a LARGE, pseudo-random
        // difference at EVERY INDIVIDUAL point-value index (each NTT
        // point is a sum over ALL n=32768 coefficients weighted by powers
        // of the root of unity). Comparing lhs[k] vs rhs[k] pointwise in
        // EVALUATION domain is therefore NOT a valid "are these nearly
        // the same polynomial" test -- it tests "are these the same at
        // this one evaluation point," which a small perturbation already
        // fails almost everywhere. CHECK 1/1b/1c all avoided this because
        // cc->Decrypt always converts to Format::COEFFICIENT internally
        // before doing anything error-sensitive. Do the same here before
        // comparing, by converting COPIES (SetFormat is in-place) to
        // coefficient domain.
        NativePoly lhsCoeff = lhs; lhsCoeff.SetFormat(Format::COEFFICIENT);
        NativePoly rhsCoeff = rhs; rhsCoeff.SetFormat(Format::COEFFICIENT);
        lhs = lhsCoeff;
        rhs = rhsCoeff;

        // CHECK 1b proved t0/t1/ba0/ba1 extraction byte-identical to the
        // real pipeline, and real decrypt of (t0+ba0)+(t1+ba1)*s is
        // correct -- which algebraically FORCES ba0+ba1*s ~= t2*s^2 (up to
        // small noise) given the exact tensor identity
        // t0+t1*s+t2*s^2 = (c0a+c1a*s)(c0b+c1b*s). A 50%-of-q mismatch
        // under those constraints means "s" itself must differ between
        // this check and whatever decrypt uses internally -- so, newly
        // justified by CHECK 1b's isolation (not a blind rerun of the
        // earlier, already-negative sign sweep), also test lhs+rhs here:
        // if decrypt's internal secret convention is actually -s relative
        // to GetPrivateElement()'s sign, the TRUE identity would be
        // ba0+ba1*s ~= -(t2*s^2), i.e. lhs+rhs ~= 0 instead of lhs-rhs ~= 0.
        uint64_t q = lhs.GetModulus().ConvertToInt();
        uint64_t worstDiff = 0, worstSum = 0;
        for (uint32_t k = 0; k < lhs.GetLength(); ++k) {
            uint64_t l = lhs[k].ConvertToInt();
            uint64_t r = rhs[k].ConvertToInt();
            uint64_t diff = (l >= r) ? (l - r) : (r - l);
            uint64_t centeredDiff = std::min(diff, q - diff);
            if (centeredDiff > worstDiff) worstDiff = centeredDiff;
            uint64_t sum = (l + r) % q;
            uint64_t centeredSum = std::min(sum, q - sum);
            if (centeredSum > worstSum) worstSum = centeredSum;
        }
        double frac = (double)worstDiff / (double)q;
        double fracSum = (double)worstSum / (double)q;
        std::cout << "  tower=" << i << " q=" << q << " worst|ba0+ba1*s - t2*s^2|=" << worstDiff << " (" << (frac * 100.0)
                  << "% of q)" << "  worst|ba0+ba1*s + t2*s^2|=" << worstSum << " (" << (fracSum * 100.0) << "% of q)\n";
        if (frac > worstFracOfQ) worstFracOfQ = frac;
    }
    bool check2_pass = worstFracOfQ < 0.01;  // >1% of q is not plausible noise
    std::cout << "CHECK 2 (ba0+ba1*s =?= t2*s^2, via OpenFHE's own DCRTPoly ops): worst_frac_of_q="
              << (worstFracOfQ * 100.0) << "% -> " << (check2_pass ? "PASS" : "FAIL") << "\n";

    std::cout << (check1_pass ? "[PASS]" : "[FAIL]") << " check1 (real decrypt round trip)\n";
    std::cout << (check2_pass ? "[PASS]" : "[FAIL]") << " check2 (algebraic relin identity via DCRTPoly ops)\n";

    return (check1_pass && check2_pass) ? 0 : 1;
}
