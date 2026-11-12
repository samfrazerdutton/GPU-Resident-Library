#pragma once
// Phase 35: authenticated financial input envelope + verification boundary.
//
// *** THIS IS A MOCK / RESEARCH PROTOTYPE, NOT A SECURITY IMPLEMENTATION. ***
//
// Inspired by TLShare (Santos, Mouris, Xie, de Vega, Lapets -- IACR CiC
// Vol 3, No 3, 2026): "Private Authenticated MPC and FHE Inputs Over TLS."
// Nothing in this file implements TLS, GtP, Pedersen commitments, vector
// commitments, QuickSilver, CGGI, or any zero-knowledge proof. The
// "attestation" here is a toy, non-cryptographic keyed tag used ONLY to
// exercise the envelope's plumbing (session binding, tamper rejection,
// duplicate rejection) in a test harness. See
// research/PHASE35_TLSHARE_CKKS_INTEGRATION.md section 7 for exactly what
// is real, what is mocked, and why a genuine CKKS-compatible provenance
// proof is explicitly NOT attempted this phase (TLShare's own three
// protocols are built for TFHE/CGGI or prime-field LWE, not ring-based
// CKKS -- see that document's section 2B).
//
// This header touches NOTHING in src/ or include/*.h belonging to the core
// CKKS implementation. It only consumes the EXISTING, unmodified production
// encrypt path (encrypt_host_shoup_pooled_barrett_batched) as a black box.
#include <array>
#include <cstdint>
#include <cstddef>
#include <set>
#include <string>
#include <vector>

namespace gpufhe {
namespace auth_input {

// Non-cryptographic 256-bit tag type, used for both the "commitment" (a
// toy digest of the raw payload) and the "attestation" (a toy keyed tag).
// Real deployments would use SHA-256 and HMAC/ECDSA respectively; neither
// is implemented here, and this type does not claim to be either.
using Tag32 = std::array<uint8_t, 32>;

// Toy, non-cryptographic keyed tag over an arbitrary byte sequence.
// NOT HMAC. NOT collision-resistant in any proven sense. Exists only so
// the mock attestor and verifier can agree/disagree deterministically in
// tests. Explicitly not exported as a general-purpose hash -- local to
// this file's mock boundary.
Tag32 toy_keyed_tag(const std::vector<uint8_t>& key, const std::vector<uint8_t>& data);

// Identifies the claimed external source of a financial record (e.g. "a
// bank's statement endpoint"). In a real TLShare-style deployment this
// would be backed by the TLS server's certificate chain, verified during
// the actual TLS handshake. Here it is an uninterpreted label -- there is
// no certificate validation, no TLS session, no network I/O at all.
struct SourceIdentity {
    std::string issuer_label;   // e.g. "mock-bank.example" -- NOT a real cert
    std::string record_type;    // e.g. "account_balance_statement"
};

// Binds an envelope to one aggregation round. `application_id` matches
// TLShare's `aid` (Section 3.3's sequential multi-session composition);
// `session_nonce` matches a fresh `sid`. Mixing envelopes across
// different (application_id) values, or replaying the same
// (application_id, session_nonce, commitment) tuple, must be rejected.
struct SessionId {
    std::string application_id;
    uint64_t session_nonce = 0;
};

// TLS is not length-hiding (TLShare Section 3.1's own stated assumption):
// the query shape and response length are public, never claimed private.
struct QueryResponseBinding {
    std::string query_template;   // e.g. "GET /statement" -- a label, not a real request
    size_t response_length = 0;
};

// A commitment to the raw (pre-parse) payload bytes -- here a toy tag,
// not a real hiding/binding commitment scheme. Exists so the verifier can
// check that the SAME claimed payload is what was attested and what was
// later parsed into a numeric value, without needing to re-transmit the
// raw bytes.
struct PayloadCommitment {
    Tag32 digest{};
};

// The result of parsing/validating the raw payload into the one numeric
// field this application needs. This is where "proving the parse was
// honest" would matter in a real system; this prototype does not prove
// it -- it only checks that the SAME record content maps to the SAME
// commitment deterministically (section below), not that the parse
// itself is correct relative to some external schema.
struct ValidatedApplicationRecord {
    double numeric_value = 0.0;
    SourceIdentity source;
    SessionId session;
};

// Metadata describing how the numeric value was encoded into a CKKS
// plaintext -- NOT a cryptographic object, just bookkeeping so the
// verifier/aggregator know the encoding convention used (matches the
// project's own encode_host_fft convention: one real value broadcast
// into every slot, as bench_private_aggregation_workload.cpp already
// does).
struct CkksEncodingMetadata {
    uint32_t n = 0;
    double delta = 0.0;
};

// *** MOCK ATTESTATION -- NOT A CRYPTOGRAPHIC SIGNATURE. ***
// A toy keyed tag over (source, session, binding, commitment). In
// TLShare, the real attestor signs an authenticated commitment produced
// during an actual garbled-2PC TLS handshake (Figure 3 / Figure 5's step
// 2) using a real signature scheme (e.g. ECDSA) with EUF-CMA security.
// Here, "attestation" is deterministic and reproducible by anyone who
// knows the shared toy key -- it demonstrates the PLUMBING (the verifier
// rejects a mismatched tag) without claiming any unforgeability property.
struct MockAttestation {
    Tag32 tag{};
};

enum class VerificationResult {
    Accepted,
    RejectedTamperedRecord,
    RejectedBadAttestation,
    RejectedSessionMismatch,
    RejectedDuplicateSubmission,
    RejectedCiphertextMissing,
};

const char* to_string(VerificationResult r);

// The envelope submitted to the aggregation boundary. Carries a CKKS
// ciphertext produced by the EXISTING, unmodified production encrypt
// path (gpufhe::encrypt_host_shoup_pooled_barrett_batched) -- this file
// never reimplements or wraps that function's cryptography, it only
// calls it from the test harness and stores the resulting c0/c1.
//
// NOT present, by design: secret keys, TLS session secrets, credentials,
// or the plaintext numeric_value (once the envelope is built, only the
// ciphertext, commitment digest, and public metadata travel to the
// aggregation boundary).
struct FinancialInputEnvelope {
    SourceIdentity source;
    SessionId session;
    QueryResponseBinding binding;
    PayloadCommitment commitment;
    MockAttestation attestation;
    CkksEncodingMetadata enc_meta;
    std::vector<uint64_t> c0, c1; // the real CKKS ciphertext, from the real production path
};

// Builds the deterministic, order-fixed byte sequence the mock attestor
// and verifier both tag over. Exposed so both sides stay in lockstep
// without duplicating the encoding logic.
std::vector<uint8_t> canonical_attestation_bytes(const SourceIdentity&, const SessionId&,
                                                   const QueryResponseBinding&, const PayloadCommitment&);

// *** MOCK ATTESTOR. *** Issues MockAttestation tags given a shared toy
// key. Stands in for TLShare's real attestor (which never sees the
// plaintext payload beyond its length, and signs with a real signature
// scheme). This mock DOES receive the full ValidatedApplicationRecord in
// the test harness for simplicity -- a real attestor in this project's
// architecture would NOT see the plaintext value at all, only commitments
// (see research/PHASE35_TLSHARE_CKKS_INTEGRATION.md section 6 for the
// explicit trust-boundary diagram and why this mock takes a shortcut here).
class MockAttestor {
public:
    explicit MockAttestor(std::vector<uint8_t> shared_key);
    MockAttestation attest(const SourceIdentity&, const SessionId&,
                            const QueryResponseBinding&, const PayloadCommitment&) const;
private:
    std::vector<uint8_t> key_;
};

// The aggregation-side verification boundary. Checks (in order):
//  1. The envelope's session matches the expected application_id for
//     this aggregation round (RejectedSessionMismatch otherwise).
//  2. The attestation tag is correct for the envelope's own
//     (source, session, binding, commitment) under the shared mock key
//     (RejectedBadAttestation otherwise) -- this is what catches a
//     tampered source label, altered session id, or altered binding,
//     since any of those changes the canonical bytes the tag was
//     computed over.
//  3. The commitment has not been seen before in this verifier's
//     lifetime (RejectedDuplicateSubmission otherwise) -- replay/duplicate
//     protection.
//  4. The ciphertext (c0/c1) is non-empty (RejectedCiphertextMissing
//     otherwise) -- the aggregation boundary must never accept an
//     envelope with no encrypted payload.
// On Accepted, the envelope's c0/c1 are the only thing the caller should
// hand to the real aggregation path (ct_add_ct_host) -- this class never
// exposes numeric_value or any plaintext.
class AggregationSessionVerifier {
public:
    AggregationSessionVerifier(std::vector<uint8_t> shared_key, std::string expected_application_id);
    VerificationResult verify(const FinancialInputEnvelope& env);
    size_t accepted_count() const { return accepted_count_; }
private:
    std::vector<uint8_t> key_;
    std::string expected_application_id_;
    std::set<std::string> seen_commitments_; // hex-encoded digest, for dedup
    size_t accepted_count_ = 0;
};

} // namespace auth_input
} // namespace gpufhe
