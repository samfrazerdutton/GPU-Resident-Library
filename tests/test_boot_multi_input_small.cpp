// GOD PROMPT "CKKS PHASE 25": the runnable sibling of
// tests/test_boot_multi_input.cpp.
//
// WHAT THIS FILE IS, AND WHAT IT IS HONESTLY NOT.
//
// tests/test_boot_multi_input.cpp drives the FULL n=8192 bootstrap over
// several independent inputs. It cannot be executed on this machine: the
// configuration's documented peak host RSS is 8.12-8.32GB and Windows free
// physical RAM has measured 1.8-3.6GB across Phase 24 and Phase 25 (see
// research/PHASE25_RECOVERY.md section 5). Its own recovered header admitted
// as much and pointed at THIS file as the thing that would validate the
// driver's machinery at a scale this project's resources can actually reach.
//
// So, precisely:
//   - This file DOES execute, on this machine, today.
//   - This file DOES validate the shared harness in tests/boot_driver_harness.h
//     -- the case loop, the deterministic-message generator, the strict
//     separation of test seeds from cryptographic randomness, the per-stage
//     bookkeeping, the failure attribution, and the JSONL emitter -- and it
//     validates the SAME harness code the n=8192 driver includes, not a copy
//     of it. That is the entire reason the harness was extracted into a header.
//   - This file DOES exercise real CKKS encode/encrypt/decrypt/decode at
//     n=1024 with the project's real keygen, real sigma=3.19 Gaussian noise,
//     real h=64 sparse secret, and fresh per-case encryption randomness.
//   - This file DOES NOT run a bootstrap. There is no ModRaise, no C2S, no
//     EvalMod, no S2C here. It therefore proves NOTHING about bootstrap
//     correctness, bootstrap precision, or bootstrap repeatability. Any such
//     claim must come from executing tests/test_boot_multi_input.cpp itself,
//     once the memory gate in research/benchmarks/bootstrap_preflight.ps1
//     returns GO. Nothing in this file should ever be cited as bootstrap
//     evidence.
//
// Three independent things are checked, and all three must pass:
//   PART 1  Harness unit tests -- pure, no crypto, no GPU. Includes deliberate
//           fault injection, because a failure-attribution mechanism that has
//           never seen a failure is not evidence of anything.
//   PART 2  Crypto/test-seed separation -- executable proof that identical
//           deterministic message seeds reproduce identical plaintexts while
//           distinct encryption seeds produce genuinely different ciphertexts,
//           and that re-encrypting under a fresh seed never reproduces a prior
//           ciphertext.
//   PART 3  Multi-input, multi-magnitude, multi-execution round trips at
//           n=1024, recorded stage by stage and emitted as JSONL.
#include "keygen.h"
#include "boot_driver_harness.h"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cstdio>
#include <string>
#include <limits>

namespace gpufhe {
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void decode_host(std::vector<std::complex<double>>&, const std::vector<int64_t>&, uint32_t, double);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
}

using cd = std::complex<double>;
using namespace bootdrv;

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  [PASS] %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  [FAIL] %s\n", what.c_str()); }
}

// ===========================================================================
// PART 1 -- harness unit tests (pure; no crypto, no CUDA, no allocation of
// consequence). These exist so that the per-stage failure attribution the
// n=8192 driver depends on is demonstrated to work BEFORE it is ever relied
// on in an 8GB run that cannot currently be repeated on demand.
// ===========================================================================
static void part1_harnessUnitTests() {
    std::printf("\n=== PART 1: harness unit tests (pure, no crypto) ===\n");

    // 1a. An all-good case must pass and must name no failed stage.
    {
        CaseRecord C; C.ran = true;
        recordStage(C, "modraise", 1e-9, 1e-6, 0.01);
        recordStage(C, "c2s",      2e-9, 1e-6, 0.01);
        recordStage(C, "evalmod",  3e-9, 1e-6, 0.01);
        C.eFin = 5e-9; C.tolFin = 1e-6;
        finalizeCase(C);
        check(C.pass, "all stages within tolerance -> case passes");
        check(C.firstFailedStage.empty(), "passing case names no failed stage");
    }

    // 1b. FAULT INJECTION: break exactly one middle stage and require the
    // harness to finger that exact stage -- not the first, not the last, and
    // not merely "something went wrong".
    {
        CaseRecord C; C.ran = true;
        recordStage(C, "modraise", 1e-9, 1e-6, 0.01);
        recordStage(C, "c2s",      5e-2, 1e-6, 0.01);   // <-- injected fault
        recordStage(C, "evalmod",  3e-9, 1e-6, 0.01);
        C.eFin = 5e-9; C.tolFin = 1e-6;
        finalizeCase(C);
        check(!C.pass, "one broken stage -> case fails");
        check(C.firstFailedStage == "c2s",
              "failure attributed to the EXACT broken stage (got '" + C.firstFailedStage + "', want 'c2s')");
    }

    // 1c. Fault injection in the FIRST and LAST stage respectively, to prove
    // the attribution is not an artifact of position.
    {
        CaseRecord C; C.ran = true;
        recordStage(C, "modraise", 9e-1, 1e-6, 0.01);   // <-- injected fault
        recordStage(C, "c2s",      1e-9, 1e-6, 0.01);
        C.eFin = 1e-9; C.tolFin = 1e-6;
        finalizeCase(C);
        check(C.firstFailedStage == "modraise", "first-stage fault attributed to 'modraise'");
    }
    {
        CaseRecord C; C.ran = true;
        recordStage(C, "c2s",     1e-9, 1e-6, 0.01);
        recordStage(C, "s2c",     7e-1, 1e-6, 0.01);    // <-- injected fault
        C.eFin = 1e-9; C.tolFin = 1e-6;
        finalizeCase(C);
        check(C.firstFailedStage == "s2c", "last-stage fault attributed to 's2c'");
    }

    // 1d. When MULTIPLE stages fail, the attribution must name the EARLIEST --
    // a late failure is usually a consequence of an earlier one, so reporting
    // the last would actively mislead a debugging session.
    {
        CaseRecord C; C.ran = true;
        recordStage(C, "modraise", 1e-9, 1e-6, 0.01);
        recordStage(C, "c2s",      4e-1, 1e-6, 0.01);   // earliest failure
        recordStage(C, "evalmod",  8e-1, 1e-6, 0.01);   // downstream consequence
        C.eFin = 9e-1; C.tolFin = 1e-6;
        finalizeCase(C);
        check(C.firstFailedStage == "c2s", "earliest failing stage reported, not the last");
    }

    // 1e. An UNMEASURED error must never pass. This is the fail-closed
    // property that matters most: a comparison that silently produced no data
    // must be a failure, never a zero-error pass.
    {
        CaseRecord C; C.ran = true;
        const bool ok = recordStage(C, "c2s", -1.0, 1e-6, 0.01);  // -1 = not measured
        check(!ok, "unmeasured stage error reports not-ok");
        C.eFin = 1e-9; C.tolFin = 1e-6;
        finalizeCase(C);
        check(!C.pass, "unmeasured stage -> case fails (never an implicit pass)");
        check(C.stages.back().note.find("not measured") != std::string::npos,
              "unmeasured stage records why in its note");
    }

    // 1f. A case that never ran cannot pass, even with empty stage lists and a
    // final error inside tolerance.
    {
        CaseRecord C; C.ran = false;
        C.eFin = 1e-12; C.tolFin = 1e-6;
        finalizeCase(C);
        check(!C.pass, "case with ran=false cannot pass");
    }

    // 1g. An abort reason must veto a pass even if every stage looks fine.
    {
        CaseRecord C; C.ran = true;
        recordStage(C, "modraise", 1e-9, 1e-6, 0.01);
        C.eFin = 1e-9; C.tolFin = 1e-6;
        C.abortReason = "simulated out-of-memory";
        finalizeCase(C);
        check(!C.pass, "non-empty abort reason vetoes a pass");
    }

    // 1h. A final error outside tolerance fails even when every stage passed,
    // and is attributed to final_decode rather than left unattributed.
    {
        CaseRecord C; C.ran = true;
        recordStage(C, "modraise", 1e-9, 1e-6, 0.01);
        C.eFin = 5e-2; C.tolFin = 1e-6;
        finalizeCase(C);
        check(!C.pass, "final error out of tolerance -> case fails");
        check(C.firstFailedStage == "final_decode", "unattributed final failure labelled 'final_decode'");
    }

    // 1i. Boundary: error exactly AT tolerance must pass (<=, not <).
    {
        CaseRecord C; C.ran = true;
        recordStage(C, "c2s", 1e-6, 1e-6, 0.01);
        C.eFin = 1e-6; C.tolFin = 1e-6;
        finalizeCase(C);
        check(C.pass, "error exactly at tolerance passes (boundary is inclusive)");
    }

    // 1j. JSON emission must survive hostile strings and non-finite doubles --
    // a NaN written as the token NaN would make the whole JSONL file
    // unparseable, silently destroying the comparison history.
    {
        CaseRecord C; C.ran = true; C.caseIdx = 7;
        C.profile = "quote\"back\\slash\nnewline\ttab";
        C.abortReason = std::string("ctrl") + char(0x01) + "char";
        recordStage(C, "broken_stage", std::nan(""), 1e-6, 0.01, "note with \"quotes\"");
        C.eFin = std::numeric_limits<double>::infinity();
        C.tolFin = 1e-6;
        finalizeCase(C);
        const std::string line = caseJsonLine(C);
        check(line.find("\\\"") != std::string::npos, "JSON escapes embedded double quotes");
        check(line.find("\\\\") != std::string::npos, "JSON escapes embedded backslashes");
        check(line.find("\\n") != std::string::npos, "JSON escapes embedded newlines");
        check(line.find("\\u0001") != std::string::npos, "JSON escapes control characters");
        // Assert on the EMITTED VALUE, not on a substring of the whole line.
        // An earlier revision of this check searched the line for "nan" and
        // tripped over its own stage NAME ("nan_stage") -- a false positive
        // that the harness was right to survive. Keeping the corrected form
        // documented here because it is the exact mistake a reader of a
        // JSON-escaping test is most likely to re-introduce.
        check(line.find("\"err\":null") != std::string::npos,
              "a NaN stage error is emitted as JSON null");
        check(line.find("\"e_final\":null") != std::string::npos,
              "an infinite final error is emitted as JSON null");
        check(line.find(":NaN") == std::string::npos && line.find(":nan") == std::string::npos &&
              line.find(":inf") == std::string::npos && line.find(":Infinity") == std::string::npos,
              "no non-finite token is ever emitted as a JSON VALUE (would make the file unparseable)");
        check(line.find("\"record\":\"case\"") != std::string::npos, "case line carries its record type");
    }

    // 1k. Deterministic message generation: same seed+profile must reproduce
    // bit-identical plaintext; a different seed must not.
    {
        const uint32_t S = 64;
        auto a = makeDriverMessage(S, 12345, MessageProfile::Reference);
        auto b = makeDriverMessage(S, 12345, MessageProfile::Reference);
        auto c = makeDriverMessage(S, 12346, MessageProfile::Reference);
        bool identical = true;
        for (uint32_t i = 0; i < S; ++i)
            if (a[i] != b[i]) { identical = false; break; }
        check(identical, "same message seed reproduces a bit-identical plaintext");
        check(maxAbsDiff(a, c, S) > 1e-12, "a different message seed produces a different plaintext");
    }

    // 1l. The profiles must actually differ in the way they claim to, otherwise
    // "multiple message magnitudes" is a label with nothing behind it.
    {
        const uint32_t S = 64;
        auto ref   = makeDriverMessage(S, 777, MessageProfile::Reference);
        auto small = makeDriverMessage(S, 777, MessageProfile::SmallMagnitude);
        auto large = makeDriverMessage(S, 777, MessageProfile::LargeMagnitude);
        auto alt   = makeDriverMessage(S, 777, MessageProfile::AlternatingSign);
        auto konst = makeDriverMessage(S, 777, MessageProfile::Constant);
        auto zero  = makeDriverMessage(S, 777, MessageProfile::Zero);

        auto peak = [&](const std::vector<cd>& v) {
            double m = 0; for (auto& x : v) m = std::max(m, std::abs(x)); return m; };

        check(peak(small) < peak(ref) && peak(ref) < peak(large),
              "magnitude profiles are strictly ordered small < reference < large");
        check(peak(zero) == 0.0, "zero-boundary profile is EXACTLY zero (not merely small)");

        bool allSame = true;
        for (uint32_t i = 1; i < S; ++i) if (konst[i] != konst[0]) { allSame = false; break; }
        check(allSame, "constant profile really is slot-invariant");

        bool alternates = true;
        for (uint32_t i = 0; i + 1 < S; ++i)
            if (alt[i].real() * alt[i + 1].real() >= 0.0) { alternates = false; break; }
        check(alternates, "alternating-sign profile really alternates sign between adjacent slots");
    }
}

// ===========================================================================
// PART 2 -- executable proof of the crypto/test-seed separation.
//
// The directive requires fresh cryptographic encryption randomness, kept
// strictly apart from the deterministic test seeds. Phase 24 could only assert
// that property by reading the source. These checks DEMONSTRATE it: they
// encrypt the same plaintext twice and inspect the resulting ciphertexts.
// ===========================================================================
static void part2_cryptoSeparation(uint32_t n, const std::vector<uint64_t>& mod,
                                   const std::vector<uint64_t>& root, uint64_t ns,
                                   const gpufhe::KeyPairHost& KP, double Delta) {
    std::printf("\n=== PART 2: cryptographic-randomness separation (executed) ===\n");
    const uint32_t S = n / 2;

    // One fixed plaintext, generated by the DETERMINISTIC test-vector path.
    auto z = makeDriverMessage(S, 424242, MessageProfile::Reference);
    std::vector<int64_t> m;
    gpufhe::encode_host(m, z, n, Delta);

    // The same plaintext encrypted under two DIFFERENT cryptographic seeds.
    std::vector<uint64_t> a0, a1, b0, b1;
    gpufhe::encrypt_host(a0, a1, m, KP.pkA, KP.pkB, n, mod, root, ns, 3.19, 5001);
    gpufhe::encrypt_host(b0, b1, m, KP.pkA, KP.pkB, n, mod, root, ns, 3.19, 5002);

    bool c0Differs = (a0 != b0), c1Differs = (a1 != b1);
    check(c0Differs, "distinct encryption seeds produce a DIFFERENT c0 (fresh randomness is real)");
    check(c1Differs, "distinct encryption seeds produce a DIFFERENT c1 (fresh randomness is real)");

    // Count how many coefficients actually differ. A single differing limb
    // would technically satisfy "!=" while indicating something badly wrong.
    size_t diff0 = 0;
    for (size_t i = 0; i < a0.size() && i < b0.size(); ++i) if (a0[i] != b0[i]) ++diff0;
    const double fracDiff = a0.empty() ? 0.0 : (double)diff0 / (double)a0.size();
    std::printf("    c0 coefficients differing between the two encryptions: %zu/%zu (%.4f)\n",
                diff0, a0.size(), fracDiff);
    check(fracDiff > 0.90,
          "the two ciphertexts differ in essentially every coefficient, not just a few");

    // Both must nevertheless DECRYPT to the same plaintext -- fresh randomness
    // must change the ciphertext without changing the message.
    std::vector<int64_t> ma, mb;
    gpufhe::decrypt_host(ma, a0, a1, KP.s, n, mod, root);
    gpufhe::decrypt_host(mb, b0, b1, KP.s, n, mod, root);
    std::vector<cd> za, zb;
    gpufhe::decode_host(za, ma, n, Delta);
    gpufhe::decode_host(zb, mb, n, Delta);
    const double eA = maxAbsDiff(za, z, S), eB = maxAbsDiff(zb, z, S);
    std::printf("    decoded error: seed 5001 -> %.6g, seed 5002 -> %.6g\n", eA, eB);
    check(eA >= 0.0 && eA < 1e-6, "ciphertext from seed 5001 decrypts to the original message");
    check(eB >= 0.0 && eB < 1e-6, "ciphertext from seed 5002 decrypts to the original message");

    // And the two independent encryptions must not agree bit-for-bit on their
    // noise either -- i.e. the decoded errors should not be identical.
    check(std::abs(eA - eB) > 0.0,
          "independent encryptions carry independent noise (decoded errors are not identical)");

    // Re-encrypting with a seed already used must NOT be how the driver works:
    // confirm that re-using a seed is reproducible (so the ONLY way the driver
    // gets fresh randomness is by advancing the seed, which it does per case).
    std::vector<uint64_t> r0, r1;
    gpufhe::encrypt_host(r0, r1, m, KP.pkA, KP.pkB, n, mod, root, ns, 3.19, 5001);
    check(r0 == a0 && r1 == a1,
          "an explicitly reused encryption seed is reproducible -- so per-case seed advance is what supplies freshness");

    // The deterministic message path must be entirely unaffected by any of the
    // encryption activity above.
    auto zAgain = makeDriverMessage(S, 424242, MessageProfile::Reference);
    bool msgStable = true;
    for (uint32_t i = 0; i < S; ++i) if (zAgain[i] != z[i]) { msgStable = false; break; }
    check(msgStable,
          "the deterministic message generator is unperturbed by intervening crypto calls (no shared RNG state)");
}

// ===========================================================================
// PART 3 -- the real multi-input case loop at n=1024, stage by stage.
// ===========================================================================
int main() {
    const uint32_t HW = 64;      // h=64 -- the project's secret Hamming weight, UNCHANGED
    gpufhe::set_secret_hamming_weight(HW);

    const uint32_t n = 1024, S = n / 2, sizeQ = 4;
    const uint64_t ns = 1;
    const double Delta = std::pow(2.0, 45);

    std::vector<uint64_t> mod;
    gpufhe::native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> mids;
      gpufhe::native_primes(mids, sizeQ - 1, 55, n, mod);
      for (auto m : mids) mod.push_back(m); }
    std::vector<uint64_t> root(sizeQ);
    for (uint32_t i = 0; i < sizeQ; ++i) root[i] = gpufhe::native_root(n, mod[i]);

    // ONE key pair, shared across every case -- the realistic deployment model
    // (one key, many ciphertexts), and the same choice tests/
    // test_boot_multi_input.cpp makes.
    auto KP = gpufhe::keygen_host(n, mod, root, ns, 3.19, 101);

    part1_harnessUnitTests();
    part2_cryptoSeparation(n, mod, root, ns, KP, Delta);

    std::printf("\n=== PART 3: multi-input round trips at n=%u (executed) ===\n", n);

    const MessageProfile profiles[] = {
        MessageProfile::Reference,
        MessageProfile::SmallMagnitude,
        MessageProfile::LargeMagnitude,
        MessageProfile::AlternatingSign,
        MessageProfile::Constant,
        MessageProfile::Zero,
    };
    const uint32_t NUM_PROFILES = sizeof(profiles) / sizeof(profiles[0]);
    const uint32_t REPEATS = 2;          // repeated executions, per the directive
    const uint64_t DRIVER_MESSAGE_SEED_BASE = 900001;   // feeds makeDriverMessage ONLY
    const uint64_t DRIVER_ENCRYPT_SEED_BASE = 900101;   // feeds encrypt_host ONLY

    // Tolerances. At Delta=2^45 with sigma=3.19 the encryption noise lands
    // ~1e-11 after decoding, so 1e-6 is a loose-but-meaningful absolute bound:
    // comfortably above the real noise floor, far below any actual defect.
    const double TOL_ENCODE = 1e-9;
    const double TOL_COEFF  = 1e-6;
    const double TOL_FINAL  = 1e-6;

    std::vector<CaseRecord> results;
    std::vector<std::vector<uint64_t>> seenC0;   // for cross-case independence

    uint32_t caseIdx = 0;
    for (uint32_t rep = 0; rep < REPEATS; ++rep) {
        for (uint32_t p = 0; p < NUM_PROFILES; ++p, ++caseIdx) {
            CaseRecord C;
            C.caseIdx = caseIdx;
            C.profile = profileName(profiles[p]);
            C.nominalAmplitude = profileNominalAmplitude(profiles[p]);
            // Deliberately distinct bases: the message seed and the encryption
            // seed can never collide or be mistaken for one another.
            C.messageSeed = DRIVER_MESSAGE_SEED_BASE + caseIdx;
            C.encryptSeed = DRIVER_ENCRYPT_SEED_BASE + caseIdx;
            C.twIn = sizeQ; C.twF = sizeQ;
            C.tolFin = TOL_FINAL;
            C.ran = true;
            const double tCase = nowSecs();

            // -- stage: encode/decode round trip (no crypto involved at all) --
            std::vector<cd> z = makeDriverMessage(S, C.messageSeed, profiles[p]);
            double t0 = nowSecs();
            std::vector<int64_t> m;
            gpufhe::encode_host(m, z, n, Delta);
            std::vector<cd> zEnc;
            gpufhe::decode_host(zEnc, m, n, Delta);
            recordStage(C, "encode_decode", maxAbsDiff(zEnc, z, S), TOL_ENCODE, nowSecs() - t0,
                        "plaintext-only round trip; isolates encoding error from all crypto");

            // -- stage: encrypt then decrypt, compared in the COEFFICIENT domain --
            t0 = nowSecs();
            std::vector<uint64_t> c0, c1;
            gpufhe::encrypt_host(c0, c1, m, KP.pkA, KP.pkB, n, mod, root, ns, 3.19, (uint32_t)C.encryptSeed);
            std::vector<int64_t> mDec;
            gpufhe::decrypt_host(mDec, c0, c1, KP.s, n, mod, root);
            double coeffErr = -1.0;
            if (mDec.size() >= n && m.size() >= n) {
                coeffErr = 0.0;
                for (uint32_t i = 0; i < n; ++i)
                    coeffErr = std::max(coeffErr, std::abs((double)(mDec[i] - m[i])) / Delta);
            }
            recordStage(C, "encrypt_decrypt_coeff", coeffErr, TOL_COEFF, nowSecs() - t0,
                        "coefficient-domain check: isolates encryption noise from decoding error");

            // -- stage: decode the decrypted coefficients, independently of the
            //    encode check above --
            t0 = nowSecs();
            std::vector<cd> zOut;
            gpufhe::decode_host(zOut, mDec, n, Delta);
            const double finalErr = maxAbsDiff(zOut, z, S);
            recordStage(C, "decode_after_decrypt", finalErr, TOL_FINAL, nowSecs() - t0,
                        "full round trip verified against the ORIGINAL message, not an intermediate");

            // -- cross-case ciphertext independence: this case's c0 must not
            //    equal any earlier case's c0. Catches a seed-reuse regression
            //    that per-case checks alone would miss.
            bool unique = true;
            for (const auto& prev : seenC0) if (prev == c0) { unique = false; break; }
            recordStage(C, "ciphertext_independence", unique ? 0.0 : 1.0, 0.0, 0.0,
                        unique ? "c0 differs from every prior case"
                               : "DUPLICATE c0 -- encryption randomness was reused across cases");
            seenC0.push_back(c0);

            C.eFin = finalErr;
            C.z0_expected = z.empty() ? cd{} : z[0];
            C.z0_got = zOut.empty() ? cd{} : zOut[0];
            C.secs = nowSecs() - tCase;
            C.peakRssMB = rssMB();
            finalizeCase(C);
            results.push_back(C);

            std::printf("  case %u profile=%-16s msgSeed=%llu encSeed=%llu eFin=%.6g [%s]\n",
                        C.caseIdx, C.profile.c_str(),
                        (unsigned long long)C.messageSeed, (unsigned long long)C.encryptSeed,
                        C.eFin, C.pass ? "PASS" : "FAIL");
            printStages(C);
        }
    }

    // Repeatability across the two passes: the SAME profile at a DIFFERENT seed
    // must land in the same error regime. A large swing would mean the pipeline
    // is input-sensitive in a way a single-shot test could never reveal -- which
    // is the entire reason this driver exists.
    std::printf("\n  -- repeatability across repeated executions --\n");
    for (uint32_t p = 0; p < NUM_PROFILES; ++p) {
        const CaseRecord& a = results[p];
        const CaseRecord& b = results[p + NUM_PROFILES];
        const double worse = std::max(a.eFin, b.eFin), better = std::min(a.eFin, b.eFin);
        // Both must pass, and neither may be wildly worse than the other. The
        // ratio guard uses an absolute floor so that two legitimately tiny
        // errors (1e-13 vs 3e-13) are not flagged as a 3x "regression".
        const bool regimeOk = (worse <= TOL_FINAL) &&
                              (worse <= 1e-9 || worse <= 50.0 * std::max(better, 1e-15));
        std::printf("    %-16s rep0=%.6g rep1=%.6g -> %s\n",
                    a.profile.c_str(), a.eFin, b.eFin, regimeOk ? "consistent" : "INCONSISTENT");
        check(regimeOk, std::string("profile '") + a.profile +
                        "' gives a consistent error regime across independent executions");
    }

    for (const auto& c : results) check(c.pass, std::string("case ") + std::to_string(c.caseIdx) +
                                                " (" + c.profile + ") passed");

    // ---- machine-readable output ----
    ParamRecord P;
    P.driver = "test_boot_multi_input_small";
    P.label = "n=1024,sizeQ=4 encode/encrypt/decrypt/decode round trip -- HARNESS VALIDATION ONLY, NOT A BOOTSTRAP";
    P.n = n; P.slots = S; P.sizeQ = sizeQ; P.sizeP = 0;
    P.hammingWeight = HW; P.sigma = 3.19; P.noiseScale = ns;
    P.Delta_in = Delta; P.Delta_pt = Delta;
    P.numCases = (uint32_t)results.size();
    P.messageSeedBase = DRIVER_MESSAGE_SEED_BASE;
    P.encryptSeedBase = DRIVER_ENCRYPT_SEED_BASE;
    P.modQ = mod;
    P.nKeygen = 1;
    P.notes = "No bootstrap stages are executed here (no ModRaise/C2S/EvalMod/S2C). "
              "This run validates tests/boot_driver_harness.h -- the same harness "
              "tests/test_boot_multi_input.cpp includes -- plus real CKKS "
              "encode/encrypt/decrypt/decode at n=1024. It is NOT bootstrap evidence.";

    const std::string out = jsonlPath("research/benchmarks/boot_multi_input_small.jsonl");
    const bool wrote = writeJsonl(out, P, results);
    check(wrote, "machine-readable JSONL written to " + out);

    printHumanSummary(P, results);
    std::printf("\n=== CHECKS: %d passed, %d failed ===\n", g_pass, g_fail);
    if (g_fail == 0)
        std::printf("[PASS] harness, crypto-seed separation, and n=1024 multi-input round trips all verified\n");
    else
        std::printf("[FAIL] %d check(s) failed\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
