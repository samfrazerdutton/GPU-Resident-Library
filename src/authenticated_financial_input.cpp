// Phase 35: implementation of the mock authenticated-input envelope.
// See the header for the explicit "this is not a security implementation"
// disclosure -- repeated here briefly: toy_keyed_tag is NOT HMAC, NOT
// collision-resistant, NOT constant-time, and must never be used outside
// this prototype's own test harness.
#include "authenticated_financial_input.h"
#include <cstring>
#include <sstream>
#include <iomanip>

namespace gpufhe {
namespace auth_input {

const char* to_string(VerificationResult r) {
    switch (r) {
        case VerificationResult::Accepted: return "Accepted";
        case VerificationResult::RejectedTamperedRecord: return "RejectedTamperedRecord";
        case VerificationResult::RejectedBadAttestation: return "RejectedBadAttestation";
        case VerificationResult::RejectedSessionMismatch: return "RejectedSessionMismatch";
        case VerificationResult::RejectedDuplicateSubmission: return "RejectedDuplicateSubmission";
        case VerificationResult::RejectedCiphertextMissing: return "RejectedCiphertextMissing";
    }
    return "Unknown";
}

// Toy keyed tag: a simple, explicitly non-cryptographic mixing function
// (FNV-1a-style rolling hash seeded by the key, expanded to 32 bytes by
// running four independent seeds derived from the key). This is NOT a
// MAC with any proven security property -- it exists purely to make the
// mock attestor/verifier agree deterministically and to make tampering
// with any input byte change the output with high probability, which is
// sufficient for exercising this prototype's rejection paths in tests.
static uint64_t fnv1a_mix(uint64_t seed, const std::vector<uint8_t>& data) {
    uint64_t h = seed ^ 0xcbf29ce484222325ULL;
    for (uint8_t b : data) { h ^= b; h *= 0x100000001b3ULL; }
    return h;
}

Tag32 toy_keyed_tag(const std::vector<uint8_t>& key, const std::vector<uint8_t>& data) {
    std::vector<uint8_t> keyed;
    keyed.reserve(key.size() + data.size());
    keyed.insert(keyed.end(), key.begin(), key.end());
    keyed.insert(keyed.end(), data.begin(), data.end());
    Tag32 out{};
    for (int lane = 0; lane < 4; ++lane) {
        uint64_t seed = 0x9E3779B97F4A7C15ULL * (uint64_t)(lane + 1);
        uint64_t h = fnv1a_mix(seed, keyed);
        std::memcpy(out.data() + lane * 8, &h, 8);
    }
    return out;
}

static void append_str(std::vector<uint8_t>& v, const std::string& s) {
    uint32_t len = (uint32_t)s.size();
    const uint8_t* lp = reinterpret_cast<const uint8_t*>(&len);
    v.insert(v.end(), lp, lp + 4);
    v.insert(v.end(), s.begin(), s.end());
}
static void append_u64(std::vector<uint8_t>& v, uint64_t x) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&x);
    v.insert(v.end(), p, p + 8);
}

std::vector<uint8_t> canonical_attestation_bytes(const SourceIdentity& src, const SessionId& sess,
                                                   const QueryResponseBinding& bind, const PayloadCommitment& com) {
    std::vector<uint8_t> b;
    append_str(b, "gpufhe-auth-input-v1"); // domain-separation tag, mirrors TLShare's ctx domain separation
    append_str(b, src.issuer_label);
    append_str(b, src.record_type);
    append_str(b, sess.application_id);
    append_u64(b, sess.session_nonce);
    append_str(b, bind.query_template);
    append_u64(b, (uint64_t)bind.response_length);
    b.insert(b.end(), com.digest.begin(), com.digest.end());
    return b;
}

MockAttestor::MockAttestor(std::vector<uint8_t> shared_key) : key_(std::move(shared_key)) {}

MockAttestation MockAttestor::attest(const SourceIdentity& src, const SessionId& sess,
                                      const QueryResponseBinding& bind, const PayloadCommitment& com) const {
    auto bytes = canonical_attestation_bytes(src, sess, bind, com);
    MockAttestation a;
    a.tag = toy_keyed_tag(key_, bytes);
    return a;
}

AggregationSessionVerifier::AggregationSessionVerifier(std::vector<uint8_t> shared_key, std::string expected_application_id)
    : key_(std::move(shared_key)), expected_application_id_(std::move(expected_application_id)) {}

static std::string hex_digest(const Tag32& t) {
    std::ostringstream os;
    for (uint8_t b : t) os << std::hex << std::setw(2) << std::setfill('0') << (int)b;
    return os.str();
}

VerificationResult AggregationSessionVerifier::verify(const FinancialInputEnvelope& env) {
    // 1. Session binding -- must match this aggregation round's application_id.
    if (env.session.application_id != expected_application_id_) {
        return VerificationResult::RejectedSessionMismatch;
    }
    // 2. Attestation check -- recompute the expected tag from the envelope's
    //    OWN fields and compare. Any tamper to source/session/binding/commitment
    //    changes the canonical bytes and therefore the recomputed tag.
    auto bytes = canonical_attestation_bytes(env.source, env.session, env.binding, env.commitment);
    Tag32 expected = toy_keyed_tag(key_, bytes);
    if (expected != env.attestation.tag) {
        return VerificationResult::RejectedBadAttestation;
    }
    // 3. Duplicate/replay protection, keyed on the commitment digest.
    std::string key = hex_digest(env.commitment.digest);
    if (seen_commitments_.count(key)) {
        return VerificationResult::RejectedDuplicateSubmission;
    }
    // 4. The ciphertext must actually be present -- the aggregation boundary
    //    must never accept an envelope carrying no encrypted payload.
    if (env.c0.empty() || env.c1.empty()) {
        return VerificationResult::RejectedCiphertextMissing;
    }
    seen_commitments_.insert(key);
    ++accepted_count_;
    return VerificationResult::Accepted;
}

} // namespace auth_input
} // namespace gpufhe
