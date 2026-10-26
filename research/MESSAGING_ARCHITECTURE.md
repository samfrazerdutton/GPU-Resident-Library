# Post-quantum instant messaging — architecture proposal

Produced by a delegated research agent (general-purpose, with web search)
under an explicit instruction NOT to force CKKS into this use case. Full
agent output preserved below with light formatting only; sources are the
agent's own, fetched live.

**Verdict up front: CKKS is the wrong primitive for messaging and should
not be used for it.** CKKS is an approximate, leveled homomorphic
encryption scheme for encrypted NUMERICAL COMPUTATION. It is not an AEAD,
it does not encrypt exact bytes, and its ~115-second bootstrap (this
project's own measured number) is ~6-9 orders of magnitude slower than the
standardized primitives below.

---

## 0. The distinction that governs everything

**Key establishment** produces a short shared secret. **Payload
encryption** turns plaintext bytes into authenticated ciphertext. ML-KEM
(FIPS 203) is a KEM: encapsulation yields `(ciphertext, shared_secret)`
where the ciphertext carries no user data. **ML-KEM cannot encrypt a
message.** The shared secret must be expanded via a KDF into AEAD keys,
and an AEAD encrypts the actual content. Conflating these two roles is the
single most common error in this space — and it is the error that makes a
homomorphic *encryption* scheme look superficially appropriate when it is
not.

## 1. Key establishment: hybrid X25519 + ML-KEM-768

Recommend **ML-KEM-768** (NIST level 3; pk 1184B, ciphertext 1088B) over
512 (smaller margin against future cryptanalysis, negligible size saving)
or 1024 (defensible only for decades-long harvest-now-decrypt-later
threat models). Combine with X25519 as `ss = KDF(ss_ecdh || ss_mlkem ||
transcript)` — hybrid, not PQ-only: ECDH has ~45 years of scrutiny vs.
ML-KEM's ~15, real implementation bugs have already surfaced
(KyberSlash timing leaks, 2024), and this is exactly what Signal's PQXDH
and TLS 1.3 hybrid key exchange both do in production. Bind both public
keys and all identities into the KDF transcript.

## 2. Payload encryption: ChaCha20-Poly1305 or AES-256-GCM via HKDF

```
root_ss  = HKDF-Extract(salt=0^32, IKM = ss_ecdh || ss_mlkem)
msg_key  = HKDF-Expand(chain_key, info="msg-v1" || transcript_hash, 64)
```

AES-256-GCM where hardware AES-NI exists; ChaCha20-Poly1305 (constant-time
by construction) elsewhere. Derive a fresh key per message from the
ratchet so nonces never repeat. Authenticate headers (sender, counter,
chain length, version) as AEAD associated data.

## 3. Session design: Double Ratchet + KEM ratchet

Base on Signal's Double Ratchet (forward secrecy via a symmetric KDF
chain, post-compromise security via re-keying). PQ-safing it needs two
changes: (1) initial handshake becomes PQXDH-style hybrid X25519+ML-KEM
over published prekey bundles including one-time ML-KEM encapsulation
keys; (2) the continuous DH ratchet step — NOT PQ-safe on its own — rides
alongside a KEM ratchet (the mechanism behind Signal's Triple
Ratchet/SPQR and PQ work in MLS): each side piggybacks a fresh ML-KEM key,
amortized over an epoch (not every message, since keys/ciphertexts are
~1.1KB) to bound bandwidth while accepting a bounded post-compromise
healing window.

## 4. Authentication

KEM encapsulation gives only implicit authentication (proves only that the
holder of the matching decapsulation key can read the message) — it does
not bind identity. Sign prekey bundles under the long-term identity key
with **ML-DSA-65** (FIPS 204; ~0.3-0.9ms sign, ~0.15-0.3ms verify), hybrid
with Ed25519 during the transition period, plus out-of-band identity
verification (safety numbers) and key transparency (an append-only
verifiable log) so a malicious server cannot silently swap keys.

## 5. Operational concerns

Replay protection via monotonic per-chain counters in authenticated AD
plus one-time-prekey consumption tracking. RNG must be the OS CSPRNG only
— ML-KEM is catastrophically sensitive to encapsulation-randomness reuse.
Multi-device means per-device identity and independent key establishment,
never shared private keys. Device loss is, by forward-secrecy design,
unrecoverable for lost ratchet state — offer explicit re-pairing and
optional clearly-labeled encrypted backups.

## 6. Latency: the quantitative case against CKKS

| Operation | Latency |
|---|---|
| ML-KEM-768 keygen/encaps/decaps | ~20-40 µs each |
| X25519 scalar mult | ~30-50 µs |
| AES-256-GCM / ChaCha20-Poly1305, 1KB | ~0.2-0.5 µs |
| Full hybrid handshake | ~0.15 ms |
| Per-message encrypt/decrypt | ~5 µs |
| **This project's CKKS bootstrap (measured)** | **~115 s** |

One CKKS bootstrap costs as much as **~3.8 million ML-KEM encapsulations**
or **~575 million AEAD message encryptions**. A messaging app targets
sub-100ms end-to-end; one bootstrap exceeds that by ~1150x, on a GPU,
before any network hop. CKKS ciphertexts are also megabytes for a handful
of slots (10^4-10^6x expansion on a 100-byte chat line vs. AEAD's 16-byte
overhead), CKKS is approximate (message bytes must decrypt exactly; CKKS
returns noisy fixed-point reals), and CKKS provides no AEAD-style
integrity — it is malleable by design, which is fatal for a messenger.
IND-CPA-D attacks on approximate FHE further show that releasing CKKS
decryptions to an adversary can leak the secret key.

CKKS remains the right tool for its actual job: encrypted numerical
analytics, private aggregate statistics, encrypted ML inference — outside
the message path, not in it.

## 7. Required before production

Independent cryptographic review of the hybrid KDF combiner and KEM-ratchet
integration; formal protocol analysis (ProVerif/Tamarin/CryptoVerif);
FIPS 203/204 ACVP conformance testing; side-channel audit of ML-KEM
decapsulation and ML-DSA signing (KyberSlash-class timing bugs are a known
real risk); use reviewed library implementations (libsodium/BoringSSL/
AWS-LC, liboqs or verified ML-KEM) rather than new lattice arithmetic;
protocol-level penetration testing (replay, prekey exhaustion, device
revocation races); a published spec and threat model stating explicitly
what is NOT protected (metadata, traffic analysis, endpoint compromise).

**Sources** (agent-fetched): Signal PQXDH spec (signal.org/docs/specifications/pqxdh/),
Signal Double Ratchet spec (signal.org/docs/specifications/doubleratchet/),
NIST FIPS 203 (csrc.nist.gov/pubs/fips/203/final), FIPS 204
(csrc.nist.gov/pubs/fips/204/final), RFC 5869 (HKDF), RFC 8439
(ChaCha20-Poly1305), NIST SP 800-38D (GCM), arXiv 2404.13544 (ML-KEM TLS
1.3 performance), arXiv 2306.01989 (Dilithium vectorized implementation).
