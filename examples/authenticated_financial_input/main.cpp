// Phase 35, Mission 5: authenticated financial input envelope prototype.
//
// *** MOCK AUTHENTICATION. NOT TLShare. NOT GtP. NOT TLSNotary. NOT a real
// zero-knowledge proof system. *** See authenticated_financial_input.h's
// header comment and research/PHASE35_TLSHARE_CKKS_INTEGRATION.md for the
// explicit boundary between what is real (the CKKS encryption path, the
// aggregation path, the correctness check) and what is mocked (the
// "attestor," the "attestation" tag, and the lack of any real TLS session
// or cryptographic provenance proof).
//
// This test harness demonstrates:
//   1. K clients each submit a structured financial record, bound to an
//      explicit source identity and session.
//   2. A mock attestor represents an authenticated-source attestation.
//   3. The envelope is bound to the intended application/aggregation
//      session (application_id).
//   4. The REAL, unmodified production CKKS path
//      (encode_host_fft + encrypt_host_shoup_pooled_barrett_batched)
//      receives the validated numeric values.
//   5. The aggregation path (ct_add_ct_host) receives ciphertexts only --
//      never plaintext records.
//   6. Tampered records, changed session identifiers, altered source
//      metadata, duplicate submissions, and invalid attestations are all
//      rejected by the verification boundary, demonstrated explicitly.
#include "authenticated_financial_input.h"
#include "keygen.h"
#include <cstdio>
#include <cstring>
#include <complex>
#include <random>
#include <cmath>

namespace gpufhe {
void encode_host_fft(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void decode_host_fft(std::vector<std::complex<double>>&, const std::vector<int64_t>&, uint32_t, double);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void encrypt_host_shoup_pooled_barrett_batched(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void decrypt_host(std::vector<int64_t>&, const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                   const std::vector<uint64_t>&, uint32_t, const std::vector<uint64_t>&,
                   const std::vector<uint64_t>&);
void ct_add_ct_host(std::vector<uint64_t>&, std::vector<uint64_t>&,
                    const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                    uint32_t, uint32_t, const std::vector<uint64_t>&);
}

using namespace gpufhe::auth_input;
using cd = std::complex<double>;

static int g_pass = 0, g_fail = 0;
static void check(bool cond, const std::string& label) {
    if (cond) { ++g_pass; std::printf("  [PASS] %s\n", label.c_str()); }
    else { ++g_fail; std::printf("  [FAIL] %s\n", label.c_str()); }
}

// Includes the session nonce deliberately: two fetches of the SAME
// underlying value in DIFFERENT sessions must produce DIFFERENT
// commitments (matching TLShare's own session-bound commitment design,
// Section 3.3's ctx domain separation) -- otherwise two legitimately
// distinct submissions that happen to carry equal numeric values would
// collide in the duplicate-detection set for the wrong reason.
static Tag32 digest_of_value(double v, const SourceIdentity& src, const SessionId& sess) {
    std::vector<uint8_t> bytes(sizeof(double) + src.issuer_label.size() + sizeof(uint64_t));
    std::memcpy(bytes.data(), &v, sizeof(double));
    std::memcpy(bytes.data() + sizeof(double), src.issuer_label.data(), src.issuer_label.size());
    uint64_t nonce = sess.session_nonce;
    std::memcpy(bytes.data() + sizeof(double) + src.issuer_label.size(), &nonce, sizeof(uint64_t));
    // Reuses the toy tag function as a toy "digest" too -- explicitly NOT a
    // real hash function; see header comment.
    return toy_keyed_tag({}, bytes);
}

struct Fixture {
    uint32_t n, S, sizeQ;
    uint64_t ns; double sigma, delta;
    std::vector<uint64_t> mod, root;
    gpufhe::KeyPairHost KP;
};

static FinancialInputEnvelope build_envelope(const Fixture& F, const MockAttestor& attestor,
                                              double numeric_value, const SourceIdentity& source,
                                              const SessionId& session, uint64_t encSeed,
                                              double deltaOverride = 0.0) {
    // deltaOverride lets a test build an envelope that is internally
    // self-consistent (correctly attested for its OWN enc_meta) but
    // encoded with a DIFFERENT delta than the fixture's session default --
    // used by the RejectedEncodingMismatch adversarial case below.
    const double delta = (deltaOverride != 0.0) ? deltaOverride : F.delta;
    QueryResponseBinding binding{"GET /mock-statement", 256};
    PayloadCommitment commitment{digest_of_value(numeric_value, source, session)};
    CkksEncodingMetadata encMeta{F.n, delta};
    MockAttestation attestation = attestor.attest(source, session, binding, commitment, encMeta);

    std::vector<cd> z(F.S, cd(numeric_value, 0.0));
    std::vector<int64_t> m;
    gpufhe::encode_host_fft(m, z, F.n, delta);
    std::vector<uint64_t> c0, c1;
    gpufhe::encrypt_host_shoup_pooled_barrett_batched(c0, c1, m, F.KP.pkA, F.KP.pkB, F.n, F.mod, F.root,
                                                       F.ns, F.sigma, encSeed);

    FinancialInputEnvelope env;
    env.source = source; env.session = session; env.binding = binding;
    env.commitment = commitment; env.attestation = attestation;
    env.enc_meta = encMeta;
    env.c0 = c0; env.c1 = c1;
    return env;
}

int main() {
    gpufhe::set_secret_hamming_weight(64); // h=64, UNCHANGED
    Fixture F;
    F.n = 8192; F.S = F.n / 2; F.sizeQ = 4; F.ns = 1; F.sigma = 3.19; F.delta = std::pow(2.0, 45);
    gpufhe::native_primes(F.mod, 1, 60, F.n, {});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids, F.sizeQ - 1, 55, F.n, F.mod); for (auto m : mids) F.mod.push_back(m); }
    F.root.resize(F.sizeQ);
    for (uint32_t i = 0; i < F.sizeQ; ++i) F.root[i] = gpufhe::native_root(F.n, F.mod[i]);
    F.KP = gpufhe::keygen_host(F.n, F.mod, F.root, F.ns, F.sigma, 101);

    std::printf("=== Phase 35: authenticated financial input envelope prototype (MOCK ATTESTATION) ===\n");

    const std::vector<uint8_t> attestorKey = {1,2,3,4,5,6,7,8};
    const std::vector<uint8_t> wrongKey = {9,9,9,9,9,9,9,9};
    MockAttestor attestor(attestorKey);
    MockAttestor wrongAttestor(wrongKey);
    const std::string applicationId = "phase35-financial-aggregation-round-1";

    // ---- Golden path: K clients, each with its own source and session ----
    const uint32_t K = 8;
    std::mt19937_64 msgRng(2024);
    std::uniform_real_distribution<double> d(-50.0, 50.0);
    std::vector<double> values(K);
    for (uint32_t k = 0; k < K; ++k) values[k] = d(msgRng);
    double plainSum = 0; for (double v : values) plainSum += v;

    AggregationSessionVerifier verifier(attestorKey, applicationId, CkksEncodingMetadata{F.n, F.delta}, F.sizeQ);
    std::vector<uint64_t> acc0, acc1; bool first = true;
    uint32_t acceptedGolden = 0;
    for (uint32_t k = 0; k < K; ++k) {
        SourceIdentity source{"mock-bank-" + std::to_string(k) + ".example", "account_balance_statement"};
        SessionId session{applicationId, 1000 + k};
        auto env = build_envelope(F, attestor, values[k], source, session, 5000 + k);
        auto result = verifier.verify(env);
        if (result == VerificationResult::Accepted) {
            ++acceptedGolden;
            if (first) { acc0 = env.c0; acc1 = env.c1; first = false; }
            else gpufhe::ct_add_ct_host(acc0, acc1, env.c0, env.c1, F.sizeQ, F.n, F.mod);
        }
    }
    check(acceptedGolden == K, "all " + std::to_string(K) + " golden-path envelopes accepted");

    std::vector<int64_t> mDec; gpufhe::decrypt_host(mDec, acc0, acc1, F.KP.s, F.n, F.mod, F.root);
    std::vector<cd> zOut; gpufhe::decode_host_fft(zOut, mDec, F.n, F.delta);
    double decryptedSum = zOut[0].real();
    double err = std::abs(decryptedSum - plainSum);
    std::printf("  plaintext sum=%.4f decrypted sum=%.4f abs err=%.6e\n", plainSum, decryptedSum, err);
    check(err < 1e-3, "aggregated ciphertext decrypts to the correct sum (production CKKS path, untouched)");

    // ---- Adversarial cases: each must be rejected ----
    std::printf("\n-- Adversarial cases --\n");
    {
        SourceIdentity source{"mock-bank-0.example", "account_balance_statement"};
        SessionId session{applicationId, 9001};
        auto env = build_envelope(F, attestor, 42.0, source, session, 7001);
        auto baseline = verifier.verify(env);
        check(baseline == VerificationResult::Accepted, "baseline envelope for tamper tests is itself valid");
    }
    {
        SourceIdentity source{"mock-bank-0.example", "account_balance_statement"};
        SessionId session{applicationId, 9002};
        auto env = build_envelope(F, attestor, 42.0, source, session, 7002);
        env.commitment.digest[0] ^= 0xFF; // tamper the record's commitment AFTER attestation
        check(verifier.verify(env) == VerificationResult::RejectedBadAttestation,
              "tampered record (altered commitment post-attestation) rejected");
    }
    {
        SourceIdentity source{"mock-bank-0.example", "account_balance_statement"};
        SessionId session{"a-different-application-round", 9003};
        auto env = build_envelope(F, attestor, 42.0, source, session, 7003);
        check(verifier.verify(env) == VerificationResult::RejectedSessionMismatch,
              "envelope for a different application_id rejected (session mismatch)");
    }
    {
        SourceIdentity source{"mock-bank-0.example", "account_balance_statement"};
        SessionId session{applicationId, 9004};
        auto env = build_envelope(F, attestor, 42.0, source, session, 7004);
        env.session.session_nonce = 777777; // changed AFTER attestation, same application_id
        check(verifier.verify(env) == VerificationResult::RejectedBadAttestation,
              "changed session_nonce after attestation rejected");
    }
    {
        SourceIdentity source{"mock-bank-0.example", "account_balance_statement"};
        SessionId session{applicationId, 9005};
        auto env = build_envelope(F, attestor, 42.0, source, session, 7005);
        env.source.issuer_label = "attacker-controlled.example"; // altered AFTER attestation
        check(verifier.verify(env) == VerificationResult::RejectedBadAttestation,
              "altered source metadata after attestation rejected");
    }
    {
        SourceIdentity source{"mock-bank-0.example", "account_balance_statement"};
        SessionId session{applicationId, 9006};
        auto env = build_envelope(F, attestor, 42.0, source, session, 7006);
        auto first_result = verifier.verify(env);
        auto second_result = verifier.verify(env); // identical envelope submitted twice
        check(first_result == VerificationResult::Accepted, "first submission of a fresh envelope accepted");
        check(second_result == VerificationResult::RejectedDuplicateSubmission,
              "exact duplicate resubmission rejected");
    }
    {
        SourceIdentity source{"mock-bank-0.example", "account_balance_statement"};
        SessionId session{applicationId, 9007};
        auto env = build_envelope(F, wrongAttestor, 42.0, source, session, 7007); // wrong attestor key
        check(verifier.verify(env) == VerificationResult::RejectedBadAttestation,
              "envelope attested with the wrong key rejected");
    }
    {
        SourceIdentity source{"mock-bank-0.example", "account_balance_statement"};
        SessionId session{applicationId, 9008};
        auto env = build_envelope(F, attestor, 42.0, source, session, 7008);
        env.c0.clear(); env.c1.clear(); // simulate a malformed/empty ciphertext
        check(verifier.verify(env) == VerificationResult::RejectedCiphertextMissing,
              "envelope with no ciphertext payload rejected");
    }
    {
        // GOD PROMPT "CKKS PHASE 36": enc_meta tampered AFTER attestation --
        // must now be caught by the attestation check, since enc_meta is
        // bound into canonical_attestation_bytes as of this phase (the OLD
        // verifier didn't even look at enc_meta, so this tamper would have
        // sailed through entirely before this change).
        SourceIdentity source{"mock-bank-0.example", "account_balance_statement"};
        SessionId session{applicationId, 9009};
        auto env = build_envelope(F, attestor, 42.0, source, session, 7009);
        env.enc_meta.delta = std::pow(2.0, 40); // changed AFTER attestation
        check(verifier.verify(env) == VerificationResult::RejectedBadAttestation,
              "altered encoding metadata (delta) after attestation rejected");
    }
    {
        // Internally self-consistent envelope (correctly attested for its
        // OWN enc_meta) but built with a delta the SESSION does not expect --
        // the attestation alone can never catch this (it only proves
        // internal consistency), so this exercises the NEW session-level
        // RejectedEncodingMismatch check specifically.
        SourceIdentity source{"mock-bank-0.example", "account_balance_statement"};
        SessionId session{applicationId, 9010};
        auto env = build_envelope(F, attestor, 42.0, source, session, 7010, std::pow(2.0, 40));
        check(verifier.verify(env) == VerificationResult::RejectedEncodingMismatch,
              "envelope correctly attested for the wrong session delta rejected");
    }
    {
        // Non-empty but wrong-length ciphertext (e.g. a truncated or
        // corrupted transmission) -- must be rejected rather than handed to
        // ct_add_ct_host, which the OLD emptiness-only check would have let
        // through.
        SourceIdentity source{"mock-bank-0.example", "account_balance_statement"};
        SessionId session{applicationId, 9011};
        auto env = build_envelope(F, attestor, 42.0, source, session, 7011);
        env.c0.resize(env.c0.size() / 2); // truncate, but leave non-empty
        check(verifier.verify(env) == VerificationResult::RejectedCiphertextShapeMismatch,
              "truncated (non-empty but wrong-length) ciphertext rejected");
    }

    std::printf("\n=== TOTAL: %d passed, %d failed ===\n", g_pass, g_fail);
    std::printf("=== Accepted-envelope count on verifier: %zu ===\n", verifier.accepted_count());
    return g_fail == 0 ? 0 : 1;
}
