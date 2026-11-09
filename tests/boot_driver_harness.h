#pragma once
// GOD PROMPT "CKKS PHASE 25": shared harness for the multi-input bootstrap
// validation drivers.
//
// WHY THIS FILE EXISTS. Phase 25 recovered a 646-line
// tests/test_boot_multi_input.cpp from the destroyed session whose own header
// promised a sibling file, tests/test_boot_multi_input_small.cpp, that would
// validate "the LOOPING AND COMPARISON LOGIC itself independently of whether
// the n=8192 bootstrap passes". That sibling did not exist. The obvious way to
// write it -- copy the loop and the comparison code down to a small ring
// dimension -- would have validated a COPY of the harness, not the harness the
// n=8192 driver actually runs, which is close to worthless as evidence. So the
// harness lives here instead, header-only, and BOTH drivers include it. When
// the small driver proves the case loop, the seed separation, the per-stage
// bookkeeping, the failure attribution and the JSONL emitter all behave, it is
// proving it about the very same code the n=8192 driver will execute the day
// host memory allows that driver to run.
//
// CRYPTOGRAPHIC BOUNDARY -- the single most important property of this file.
// Nothing here generates, consumes, or transforms cryptographic randomness.
// makeDriverMessage() below owns a std::mt19937_64 that exists ONLY to
// synthesise reproducible TEST VECTORS (plaintext signals). It is never passed
// to any gpufhe:: function, never used to derive a key, a noise sample, or an
// encryption seed, and is deliberately confined to this one function so it
// cannot leak into a crypto call site by accident. Encryption randomness stays
// where it already was: the drivers pass a distinct per-case seed to the
// existing, approved gpufhe::encrypt_host, and this header never sees it except
// as an integer it records for provenance. No new randomness mechanism is
// introduced anywhere in Phase 25, and no security parameter (h=64, sigma=3.19,
// the modulus chain, P >= Q_part_max) is read for modification by this file.
#include <vector>
#include <complex>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <sys/resource.h>

namespace bootdrv {

using cd = std::complex<double>;

// ---------------------------------------------------------------------------
// Peak host RSS. getrusage's ru_maxrss is in kilobytes on Linux, so /1024 is
// MiB. This is the measurement the whole Phase 24/25 memory gate rests on, so
// it is recorded per case rather than only at the end.
// ---------------------------------------------------------------------------
inline double rssMB() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1024.0;
}

inline double nowSecs() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Deterministic test-vector profiles.
//
// The recovered driver generated exactly one shape of message (amplitude
// jittered in [0.15, 0.40]), which cannot exercise the directive's required
// "multiple message magnitudes" and "supported input boundaries". These
// profiles do, and each is documented with WHY that magnitude is interesting
// rather than just being a different number.
//
// The magnitudes are chosen against a real constraint, not arbitrarily.
// test_boot.cpp's EvalMod stage approximates mod-reduction over an input range
// the file itself measures at K = 9.49 (in units of q0/Delta), with a
// double-angle depth rDbl=7 at h=64 that Phase 24 confirmed has quantified
// margin over that range. A bootstrap input whose slots are too large pushes
// EvalMod outside the range its Chebyshev approximant was built for, so
// "larger" is not uniformly "harder but still fine" -- it is the direction in
// which the approximation genuinely breaks. These profiles therefore stay
// inside the envelope test_boot.cpp itself operates in and probe its edges,
// rather than inventing magnitudes the pipeline was never designed to accept.
// ---------------------------------------------------------------------------
enum class MessageProfile {
    Reference,       // the recovered driver's own band -- the continuity case
    SmallMagnitude,  // ~100x smaller: probes precision floor / relative error
    LargeMagnitude,  // near the top of the documented comfortable band
    AlternatingSign, // adjacent slots of opposite sign: worst case for the
                     // slot-mixing linear transforms (C2S/S2C) -- a bug that
                     // averages neighbours is invisible on a smooth signal
    Constant,        // DC-only: every slot identical. A rotation/permutation
                     // bug is invisible here, which is exactly why pairing it
                     // with AlternatingSign is informative
    Zero             // exact boundary: the all-zero plaintext. Must survive the
                     // whole pipeline and come back zero; catches spurious
                     // additive terms that a nonzero signal masks
};

inline const char* profileName(MessageProfile p) {
    switch (p) {
        case MessageProfile::Reference:       return "reference";
        case MessageProfile::SmallMagnitude:  return "small_magnitude";
        case MessageProfile::LargeMagnitude:  return "large_magnitude";
        case MessageProfile::AlternatingSign: return "alternating_sign";
        case MessageProfile::Constant:        return "constant";
        case MessageProfile::Zero:            return "zero_boundary";
    }
    return "unknown";
}

// Nominal peak |Re|,|Im| per profile. Recorded in the JSONL so a later
// comparison can correlate error against magnitude without re-deriving it.
inline double profileNominalAmplitude(MessageProfile p) {
    switch (p) {
        case MessageProfile::Reference:       return 0.275; // midpoint of [0.15,0.40]
        case MessageProfile::SmallMagnitude:  return 0.0030;
        case MessageProfile::LargeMagnitude:  return 0.480;
        case MessageProfile::AlternatingSign: return 0.250;
        case MessageProfile::Constant:        return 0.200;
        case MessageProfile::Zero:            return 0.0;
    }
    return 0.0;
}

// TEST-VECTOR GENERATOR. Deterministic given (S, messageSeed, profile).
// NOT cryptographic randomness -- see this file's header comment.
inline std::vector<cd> makeDriverMessage(uint32_t S, uint64_t messageSeed,
                                         MessageProfile prof = MessageProfile::Reference) {
    std::mt19937_64 msgRng(messageSeed);   // local, never escapes this function
    std::vector<cd> z(S);

    if (prof == MessageProfile::Zero) {
        // No randomness consumed at all: the exact-zero boundary is a single
        // point, not a distribution. Returning a jittered near-zero here would
        // quietly convert an exact boundary case into a small-magnitude one.
        std::fill(z.begin(), z.end(), cd{0.0, 0.0});
        return z;
    }

    std::uniform_real_distribution<double> phaseDist(0.0, 2 * M_PI);

    if (prof == MessageProfile::Constant) {
        const double amp = profileNominalAmplitude(prof);
        const double ph = phaseDist(msgRng);
        const cd v{amp * std::cos(ph), amp * std::sin(ph)};
        std::fill(z.begin(), z.end(), v);
        return z;
    }

    if (prof == MessageProfile::AlternatingSign) {
        const double amp = profileNominalAmplitude(prof);
        const double ph = phaseDist(msgRng);
        for (uint32_t i = 0; i < S; ++i) {
            const double s = (i % 2 == 0) ? 1.0 : -1.0;
            z[i] = {s * amp * std::cos(ph), s * amp * std::sin(ph)};
        }
        return z;
    }

    // Reference / SmallMagnitude / LargeMagnitude: the recovered driver's own
    // smooth bounded signal, with the amplitude band shifted per profile. The
    // SHAPE is kept identical to test_boot.cpp's hardcoded message so these
    // cases stay comparable to the historical baseline.
    double lo, hi;
    switch (prof) {
        case MessageProfile::SmallMagnitude: lo = 0.0015; hi = 0.0045; break;
        case MessageProfile::LargeMagnitude: lo = 0.4400; hi = 0.5200; break;
        default:                             lo = 0.1500; hi = 0.4000; break;
    }
    std::uniform_real_distribution<double> ampDist(lo, hi);
    std::uniform_real_distribution<double> freqDist(0.01, 0.03);
    const double ampR = ampDist(msgRng), ampI = ampDist(msgRng);
    const double freqR = freqDist(msgRng), freqI = freqDist(msgRng);
    const double phR = phaseDist(msgRng), phI = phaseDist(msgRng);
    for (uint32_t i = 0; i < S; ++i)
        z[i] = {ampR * std::sin(freqR * i + phR), ampI * std::cos(freqI * i + phI)};
    return z;
}

// Max |a[i] - b[i]| over the compared prefix. Returns a negative sentinel for
// an empty comparison so "no data" can never masquerade as "zero error".
inline double maxAbsDiff(const std::vector<cd>& a, const std::vector<cd>& b, uint32_t count) {
    if (count == 0 || a.size() < count || b.size() < count) return -1.0;
    double e = 0.0;
    for (uint32_t i = 0; i < count; ++i) e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

// ---------------------------------------------------------------------------
// Per-stage bookkeeping. The recovered driver recorded only a single final
// error, so a failure told you "the bootstrap is wrong" and nothing more. These
// records are what let the driver answer the directive's actual requirement:
// identify exactly WHICH stage failed.
// ---------------------------------------------------------------------------
struct StageRecord {
    std::string name;
    bool  ran  = false;
    bool  ok   = false;
    double err = -1.0;   // negative => not measured
    double tol = -1.0;   // negative => no numeric tolerance for this stage
    double secs = 0.0;
    std::string note;
};

struct CaseRecord {
    uint32_t caseIdx = 0;
    std::string profile;
    double nominalAmplitude = 0.0;
    uint64_t messageSeed = 0;    // feeds makeDriverMessage ONLY
    uint64_t encryptSeed = 0;    // feeds gpufhe::encrypt_host ONLY
    std::vector<StageRecord> stages;
    bool   ran = false;
    bool   pass = false;
    double eFin = -1.0;
    double tolFin = -1.0;
    double secs = 0.0;
    double peakRssMB = 0.0;
    uint32_t twIn = 0, twF = 0;
    cd z0_expected{}, z0_got{};
    std::string firstFailedStage;   // empty when nothing failed
    std::string abortReason;        // set when the case could not complete
};

inline StageRecord& beginStage(CaseRecord& C, const char* name) {
    StageRecord s;
    s.name = name;
    s.ran = true;
    C.stages.push_back(s);
    return C.stages.back();
}

inline void endStage(StageRecord& s, bool ok, double err, double tol,
                     double secs, const std::string& note = std::string()) {
    s.ok = ok; s.err = err; s.tol = tol; s.secs = secs; s.note = note;
}

// Convenience: run a numeric check and record it in one call.
inline bool recordStage(CaseRecord& C, const char* name, double err, double tol,
                        double secs, const std::string& note = std::string()) {
    // A negative err means "not measured" -- that is a harness failure, not a
    // pass, and is recorded as such rather than silently tolerated.
    const bool measured = (err >= 0.0);
    const bool ok = measured && (err <= tol);
    StageRecord& s = beginStage(C, name);
    endStage(s, ok, err, tol, secs,
             measured ? note : (note.empty() ? std::string("error not measured (empty comparison)")
                                             : note + " [error not measured]"));
    return ok;
}

// Derives pass/fail and the failure attribution from the stage records. A case
// passes only if it ran to completion AND every recorded stage passed AND the
// final error is within tolerance -- there is no path by which an unrun or
// unmeasured stage yields a pass.
inline void finalizeCase(CaseRecord& C) {
    C.firstFailedStage.clear();
    bool allOk = true;
    for (const auto& s : C.stages) {
        if (!s.ran || !s.ok) {
            allOk = false;
            if (C.firstFailedStage.empty()) C.firstFailedStage = s.name;
        }
    }
    const bool finalOk = (C.eFin >= 0.0) && (C.tolFin >= 0.0) && (C.eFin <= C.tolFin);
    if (!finalOk && C.firstFailedStage.empty()) C.firstFailedStage = "final_decode";
    C.pass = C.ran && allOk && finalOk && C.abortReason.empty();
}

// ---------------------------------------------------------------------------
// Run-level parameter record. The directive requires every CKKS parameter be
// recorded; this is the struct that carries them into the machine-readable
// output so a future comparison never has to trust a commit message.
// ---------------------------------------------------------------------------
struct ParamRecord {
    std::string label;
    std::string driver;
    uint32_t n = 0, slots = 0, sizeQ = 0, sizeP = 0;
    uint32_t hammingWeight = 0;
    double   sigma = 0.0;
    uint64_t noiseScale = 0;
    double   Delta_in = 0.0, Delta_pt = 0.0;
    uint32_t numCases = 0;
    uint64_t messageSeedBase = 0, encryptSeedBase = 0;
    uint32_t krotCap = 0, kscCap = 0;
    uint32_t nKeygen = 0;
    double   tKeygenTotal = 0.0;
    std::vector<uint64_t> modQ, modP;
    std::string notes;
};

// ---------------------------------------------------------------------------
// JSONL emission. Hand-rolled on purpose: the project has no JSON dependency
// and adding one to a test harness would be a poor trade. Correctness details
// that matter here -- JSON has no NaN/Infinity, so non-finite doubles are
// emitted as null rather than as tokens that would make the file unparseable;
// and doubles use %.17g so a value survives a write/read round trip exactly.
// ---------------------------------------------------------------------------
inline std::string jsonEscape(const std::string& in) {
    std::string o;
    o.reserve(in.size() + 8);
    for (unsigned char c : in) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += (char)c;
        }
    }
    return o;
}

inline std::string jnum(double v) {
    if (!std::isfinite(v)) return "null";
    char b[40];
    std::snprintf(b, sizeof b, "%.17g", v);
    return std::string(b);
}

inline std::string jbool(bool b) { return b ? "true" : "false"; }

inline std::string jstr(const std::string& s) { return "\"" + jsonEscape(s) + "\""; }

inline std::string jU64Array(const std::vector<uint64_t>& v) {
    std::string o = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) o += ",";
        o += std::to_string(v[i]);
    }
    return o + "]";
}

inline std::string paramsJsonLine(const ParamRecord& P) {
    std::string o = "{";
    o += "\"record\":\"run_params\"";
    o += ",\"driver\":" + jstr(P.driver);
    o += ",\"label\":" + jstr(P.label);
    o += ",\"n\":" + std::to_string(P.n);
    o += ",\"slots\":" + std::to_string(P.slots);
    o += ",\"size_q\":" + std::to_string(P.sizeQ);
    o += ",\"size_p\":" + std::to_string(P.sizeP);
    o += ",\"hamming_weight\":" + std::to_string(P.hammingWeight);
    o += ",\"sigma\":" + jnum(P.sigma);
    o += ",\"noise_scale\":" + std::to_string(P.noiseScale);
    o += ",\"delta_in\":" + jnum(P.Delta_in);
    o += ",\"delta_pt\":" + jnum(P.Delta_pt);
    o += ",\"num_cases\":" + std::to_string(P.numCases);
    o += ",\"message_seed_base\":" + std::to_string(P.messageSeedBase);
    o += ",\"encrypt_seed_base\":" + std::to_string(P.encryptSeedBase);
    o += ",\"krot_cap\":" + std::to_string(P.krotCap);
    o += ",\"ksc_cap\":" + std::to_string(P.kscCap);
    o += ",\"n_keygen\":" + std::to_string(P.nKeygen);
    o += ",\"t_keygen_total_s\":" + jnum(P.tKeygenTotal);
    o += ",\"mod_q\":" + jU64Array(P.modQ);
    o += ",\"mod_p\":" + jU64Array(P.modP);
    o += ",\"notes\":" + jstr(P.notes);
    o += "}";
    return o;
}

inline std::string caseJsonLine(const CaseRecord& C) {
    std::string o = "{";
    o += "\"record\":\"case\"";
    o += ",\"case\":" + std::to_string(C.caseIdx);
    o += ",\"profile\":" + jstr(C.profile);
    o += ",\"nominal_amplitude\":" + jnum(C.nominalAmplitude);
    o += ",\"message_seed\":" + std::to_string(C.messageSeed);
    o += ",\"encrypt_seed\":" + std::to_string(C.encryptSeed);
    o += ",\"ran\":" + jbool(C.ran);
    o += ",\"pass\":" + jbool(C.pass);
    o += ",\"e_final\":" + jnum(C.eFin);
    o += ",\"tol_final\":" + jnum(C.tolFin);
    o += ",\"secs\":" + jnum(C.secs);
    o += ",\"peak_rss_mb\":" + jnum(C.peakRssMB);
    o += ",\"tw_in\":" + std::to_string(C.twIn);
    o += ",\"tw_final\":" + std::to_string(C.twF);
    o += ",\"z0_expected_re\":" + jnum(C.z0_expected.real());
    o += ",\"z0_expected_im\":" + jnum(C.z0_expected.imag());
    o += ",\"z0_got_re\":" + jnum(C.z0_got.real());
    o += ",\"z0_got_im\":" + jnum(C.z0_got.imag());
    o += ",\"first_failed_stage\":" + jstr(C.firstFailedStage);
    o += ",\"abort_reason\":" + jstr(C.abortReason);
    o += ",\"stages\":[";
    for (size_t i = 0; i < C.stages.size(); ++i) {
        const StageRecord& s = C.stages[i];
        if (i) o += ",";
        o += "{\"name\":" + jstr(s.name);
        o += ",\"ran\":" + jbool(s.ran);
        o += ",\"ok\":" + jbool(s.ok);
        o += ",\"err\":" + jnum(s.err);
        o += ",\"tol\":" + jnum(s.tol);
        o += ",\"secs\":" + jnum(s.secs);
        o += ",\"note\":" + jstr(s.note);
        o += "}";
    }
    o += "]}";
    return o;
}

inline std::string summaryJsonLine(const ParamRecord& P, const std::vector<CaseRecord>& R) {
    uint32_t nPass = 0, nRan = 0;
    double eMax = -1.0, sMax = 0.0, rssMax = 0.0;
    for (const auto& c : R) {
        if (c.ran) ++nRan;
        if (c.pass) ++nPass;
        if (c.eFin > eMax) eMax = c.eFin;
        if (c.secs > sMax) sMax = c.secs;
        if (c.peakRssMB > rssMax) rssMax = c.peakRssMB;
    }
    std::string o = "{";
    o += "\"record\":\"summary\"";
    o += ",\"driver\":" + jstr(P.driver);
    o += ",\"cases_total\":" + std::to_string((uint32_t)R.size());
    o += ",\"cases_ran\":" + std::to_string(nRan);
    o += ",\"cases_passed\":" + std::to_string(nPass);
    o += ",\"all_passed\":" + jbool(nPass == R.size() && !R.empty());
    o += ",\"e_final_max\":" + jnum(eMax);
    o += ",\"secs_max\":" + jnum(sMax);
    o += ",\"peak_rss_mb_max\":" + jnum(rssMax);
    o += "}";
    return o;
}

// Writes params + one line per case + a summary line. Append mode, so repeated
// runs accumulate into a comparable history rather than overwriting it.
// Returns false if the file could not be opened -- the caller reports that as a
// harness failure rather than ignoring it.
inline bool writeJsonl(const std::string& path, const ParamRecord& P,
                       const std::vector<CaseRecord>& R) {
    if (path.empty()) return false;
    FILE* f = std::fopen(path.c_str(), "a");
    if (!f) return false;
    std::fprintf(f, "%s\n", paramsJsonLine(P).c_str());
    for (const auto& c : R) std::fprintf(f, "%s\n", caseJsonLine(c).c_str());
    std::fprintf(f, "%s\n", summaryJsonLine(P, R).c_str());
    std::fclose(f);
    return true;
}

// Default output path, overridable by the BOOT_DRIVER_JSONL environment
// variable so a benchmark script can redirect it without a rebuild.
inline std::string jsonlPath(const char* fallback) {
    const char* e = std::getenv("BOOT_DRIVER_JSONL");
    if (e && *e) return std::string(e);
    return std::string(fallback);
}

inline void printStages(const CaseRecord& C) {
    for (const auto& s : C.stages) {
        std::printf("    [%s] %-22s err=%-12.6g tol=%-10.6g %.3fs%s%s\n",
                    s.ok ? "ok  " : "FAIL", s.name.c_str(), s.err, s.tol, s.secs,
                    s.note.empty() ? "" : "  // ", s.note.c_str());
    }
}

inline void printHumanSummary(const ParamRecord& P, const std::vector<CaseRecord>& R) {
    uint32_t nPass = 0;
    for (const auto& c : R) if (c.pass) ++nPass;
    std::printf("\n=== %s SUMMARY ===\n", P.driver.c_str());
    std::printf("params: n=%u slots=%u sizeQ=%u sizeP=%u h=%u sigma=%.3f\n",
                P.n, P.slots, P.sizeQ, P.sizeP, P.hammingWeight, P.sigma);
    std::printf("%u/%u cases passed\n", nPass, (uint32_t)R.size());
    if (nPass != R.size()) {
        std::printf("failed cases (with the exact stage that failed):\n");
        for (const auto& c : R) {
            if (c.pass) continue;
            std::printf("  case %u (%s): stage=%s eFin=%.6g%s%s\n",
                        c.caseIdx, c.profile.c_str(),
                        c.firstFailedStage.empty() ? "(none recorded)" : c.firstFailedStage.c_str(),
                        c.eFin,
                        c.abortReason.empty() ? "" : " abort=",
                        c.abortReason.c_str());
        }
    }
}

}  // namespace bootdrv
