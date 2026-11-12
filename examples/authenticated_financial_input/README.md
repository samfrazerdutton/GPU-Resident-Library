# authenticated_financial_input

Phase 35: an authenticated financial input envelope + verification
boundary, inspired by (but **not implementing**) TLShare (Santos, Mouris,
Xie, de Vega, Lapets — "TLShare: Private Authenticated MPC and FHE
Inputs Over TLS," IACR CiC Vol. 3, No. 3, 2026). Hardened in Phase 36:
the verifier now also checks the envelope's CKKS encoding metadata
(`n`, `delta`) and the ciphertext's shape — see "What changed in Phase
36" below.

**MOCK ATTESTATION. NOT TLS. NOT GtP. NOT a zero-knowledge proof.** See
`include/authenticated_financial_input.h`'s header comment and
`research/PHASE35_TLSHARE_CKKS_INTEGRATION.md` for the full, explicit
boundary between what is real and what is mocked.

## What it demonstrates

- 8 clients each submit a structured financial record, bound to an
  explicit source identity and session.
- A mock attestor stands in for an authenticated-source attestation
  (a non-cryptographic keyed tag — NOT a digital signature).
- Each envelope is bound to one aggregation round
  (`application_id` + `session_nonce`).
- The **real, unmodified** production CKKS path
  (`encode_host_fft` + `encrypt_host_shoup_pooled_barrett_batched`)
  encrypts each client's validated numeric value.
- The aggregation path (`ct_add_ct_host`) only ever receives
  ciphertexts — never plaintext records.
- Tampered records, changed session identifiers, altered source
  metadata, altered encoding metadata, duplicate submissions, invalid
  attestations, encoding-parameter mismatches, and malformed/wrong-shape
  ciphertexts are all rejected by `AggregationSessionVerifier`.
- The aggregated ciphertext decrypts (via the real, unmodified
  `decrypt_host`/`decode_host_fft`) to the correct sum.

## Why this exists

Phase 35's architecture/compatibility analysis
(`research/PHASE35_TLSHARE_CKKS_INTEGRATION.md`) found that none of
TLShare's three published protocols are directly compatible with
ring-based CKKS (two use CGGI/TFHE or prime-field LWE; the third never
touches FHE at all). Rather than force an incompatible construction
into this project, this prototype implements only the well-defined,
protocol-agnostic PLUMBING — session binding, source identity, replay/
duplicate rejection, and a clean trust boundary between "authenticated
input" and "the real CKKS pipeline" — and explicitly documents the
missing cryptographic component (a real provenance proof tying a CKKS
ciphertext to an authenticated external record) as future work, not
something this phase claims to have solved.

## Build and run

```bash
cd ~/GPU-Resident-Library
export PATH=/usr/local/cuda-13.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cmake -S . -B build2
cmake --build build2 --target authenticated_financial_input -j4
./build2/authenticated_financial_input
```

Also registered with `ctest` (`ctest -R authenticated_financial_input`)
— its own 14 correctness/rejection checks are the pass/fail condition
(11 from Phase 35, 3 added in Phase 36 -- see below).

## What changed in Phase 36

A review of the 11 Phase-35 checks against the Track C checklist
(envelope validation, commitment construction, CKKS-pipeline
integration) found two real gaps, both now closed:

1. **`CkksEncodingMetadata` (`n`, `delta`) was carried in every envelope
   but never checked against anything.** An internally-consistent
   envelope encoded with the WRONG `delta` (or `n`) for this aggregation
   session would have been accepted and silently corrupted the
   aggregate (CKKS ciphertexts encoded at different scales do not sum
   meaningfully). `canonical_attestation_bytes` now binds `enc_meta`
   into the same attestation tag as the other fields (so a
   post-attestation tamper is `RejectedBadAttestation`, same as
   tampering source/session/binding/commitment always was), and
   `AggregationSessionVerifier` separately checks the envelope's
   `enc_meta` against the session's own expected `(n, delta)`
   (`RejectedEncodingMismatch`) — this catches an envelope that is
   honestly, self-consistently attested for the WRONG parameters, which
   no attestation check alone can ever catch.
2. **Only ciphertext emptiness was checked, not shape.** A truncated or
   otherwise wrong-length (but non-empty) `c0`/`c1` would have passed
   straight through to the real `ct_add_ct_host` aggregation call. The
   verifier now also checks `c0.size() == c1.size() == sizeQ * n` for
   the session's own parameters (`RejectedCiphertextShapeMismatch`).

Both are small, additive checks at the envelope boundary — no change to
`libgpufhe`, no change to the mock attestor's non-cryptographic nature,
and no new claim about what this prototype proves. Three new tests
(`main.cpp`) exercise: enc_meta tampered after attestation, an envelope
correctly attested for the wrong session delta, and a truncated
ciphertext — 14/14 passing.

## Files

- `examples/authenticated_financial_input/main.cpp` — the test harness
  (golden path + 12 adversarial checks across 11 cases).
- `include/authenticated_financial_input.h`,
  `src/authenticated_financial_input.cpp` — the envelope types, the mock
  attestor, and the verification boundary. Compiled into this example's
  own executable only, **not** into `libgpufhe` — this research module
  cannot affect any core CKKS build target.

## Security status

**Not cryptographically secure end to end.** The "attestation" can be
forged by anyone who knows the shared mock key. There is no real TLS
session, no real external data source, and no proof linking the CKKS
ciphertext's plaintext to the committed value. Do not use this module,
or any output it produces, to make real claims about data provenance,
authenticity, or zero-knowledge properties.
