# Phase 35: TLShare research integration, authenticated-input architecture, and CKKS encode investigation

Starting commit `2012092` (Phase 34 end), verified matching `origin/main`
exactly (0 ahead/0 behind) before any change. Hardware re-confirmed
unchanged: RTX 2060 Max-Q, 6144 MiB, driver 595.97, CUDA 13.2, cmake
3.28.3, Rust/cargo 1.94.0.

## 1. Paper summary

**TLShare: Private Authenticated MPC and FHE Inputs Over TLS**
(Santos, Mouris, Xie, de Vega, Lapets — IACR Communications in
Cryptology, Vol. 3, No. 3, 2026, 41 pages). Read in full (all 33
technical pages; pages 34-41 are bibliography only).

TLShare is a **private-to-private oracle**: it lets a client extract an
authenticated TLS response (e.g., a bank statement) and import the
payload itself — not merely a yes/no predicate about it — as private
input to downstream secure multiparty computation (MPC) or fully
homomorphic encryption (FHE), without any TLS server modification and
without the client handing over its credentials. This is explicitly
positioned against prior "private-to-public" oracles (DECO and its
descendants), which only prove a single Boolean statement (e.g.,
`age >= 18`) and cannot combine private data from multiple clients.

Three parties beyond the standard TLS client/server: an **attestor** (a
semi-trusted third party, assumed not to collude with the client, who
helps bind the TLS response to a verifiable commitment without ever
seeing the plaintext beyond its length) and a set of **computation
nodes** (an MPC network, or FHE nodes holding secret-shared decryption
key material).

## 2. The three protocols, compared precisely

### 2A. MPC mode (paper Section 4)

- **Client does**: runs a garbled-2PC TLS handshake with the attestor
  (via GtP's `ΠAuthData`), receiving Pedersen commitments to every bit
  of the response R; then, in a SEPARATE phase (attestor no longer
  involved), runs Pedersen verifiable secret sharing (VSS) to
  Shamir-secret-share R among `n` MPC nodes, using the signed
  commitments as the "base commitments" VSS must stay consistent with.
- **Attestor does**: jointly computes the commitments with the client
  during the TLS phase, then signs the ordered committed response
  (binding it to a session context `ctx`) — and is **not involved
  after that**.
- **Computation nodes verify**: the attestor's signature on the
  committed response, and that each received Shamir share is consistent
  with its node's corresponding base commitment (a single group
  exponentiation check per node, per bit).
- **Objects transmitted**: Pedersen commitments (group elements over
  Ed25519 in the implementation), an ECDSA-style signature, and Shamir
  shares with polynomial-coefficient commitments.
- **What stays private**: R itself (never reconstructed in the clear
  by any single node below threshold T); the attestor never learns R
  beyond its bit-length.
- **Assumptions**: discrete-log hardness, EUF-CMA signatures, attestor
  does not collude with the client, at most T−1 corrupted nodes.
- **Setup/preprocessing**: none (Table 1: "No Preprocessing" ✓, "No
  CRS" ✓) — the paper's own stated advantage of this mode.
- **CKKS-compatible?** This mode never touches FHE at all. It produces
  Shamir secret shares over a prime field `Z_q`, consumed by an MPC
  protocol (the paper mentions MP-SPDZ for their own downstream
  evaluation). It has **no ciphertext, no encryption, and nothing CKKS
  would consume** — it is simply not applicable to a CKKS-only engine
  like this project's.
- **To support it would require**: standing up an entirely separate MPC
  stack (e.g., integrating or building a secret-sharing-based
  computation layer) — a different project, not a CKKS extension.

### 2B. FHE mode with Non-Interactive ZK proofs (paper Section 5.1)

- **Client does**: same TLS-phase commitment step as MPC mode, then
  encrypts R under a **CGGI/TFHE** public key (NOT CKKS — a
  bit/small-integer-oriented torus-based FHE scheme with different
  bootstrapping and message-encoding conventions), and generates the
  paper's own novel NIZK "proof of vector membership" (Section 5.1.1) —
  built on pairing-based vector commitments over the BLS12-446 curve —
  proving the CGGI ciphertext's plaintext bits are exactly the bits
  previously committed in the TLS phase.
- **Attestor does**: same TLS-phase role as MPC mode; signs the
  committed response; **not involved** during the encryption phase.
- **FHE node N₁ verifies**: the CGGI encryption-validity NIZK
  (`CGGI.Verify`) AND the paper's vector-membership NIZK (`VM.Verify`,
  Eq. 3), checking that the same binary vector underlies both proofs
  at the prescribed bit positions `(k,l)`.
- **Objects transmitted**: a CGGI ciphertext, two NIZK proofs, a
  pairing-group vector commitment, a signature.
- **What stays private**: R and the CGGI plaintext; the attestor sees
  only commitments (never R or the ciphertext).
- **Assumptions**: LWE (for CGGI), the `(2n,n)`-DLOG assumption in the
  Algebraic Group Model (for the vector-membership NIZK's soundness),
  EUF-CMA signatures.
- **Setup/preprocessing**: Table 1: "No Preprocessing" ✓, but **requires
  a trusted-setup CRS** ("No CRS" ✗) — a structured reference string
  with a hidden trapdoor `α`, generated once by a trusted party.
- **CKKS-compatible?** No. CGGI/TFHE is a categorically different
  scheme from CKKS: different plaintext space (a discretized torus
  `T_q`, prime message modulus `p`, binary secret key), different
  bootstrapping machinery, and — critically for this analysis — the
  paper's vector-membership NIZK is built specifically around CGGI's
  **binary bit-decomposition** plaintext encoding
  (`μ = Σ 2^(i−1) w_{k+i}`), which has no analog in CKKS's canonical-
  embedding, Delta-scaled, approximate real/complex encoding.
- **To support it would require**: replacing CKKS encryption with CGGI
  for this input path (explicitly forbidden by this phase's own
  constraints), or designing an entirely new vector-membership-style
  NIZK over CKKS's own encoding — see section 3 below for why this is
  not a small extension.

### 2C. FHE mode with Interactive ZK proofs (paper Section 5.2)

- **Client does**: locally samples LWE randomness `r ∈ {0,1}^n` and
  noise `e, e₀`; the client and attestor jointly generate
  Information-Theoretic MACs (IT-MACs, via GtP) directly on the TLS
  response bits `⟨R⟩` (skipping the Pedersen-commitment conversion step
  the other two modes use); then the client and attestor run an
  **interactive** QuickSilver zero-knowledge proof proving, over these
  IT-MAC-authenticated witnesses, that an LPR-style Regev ciphertext
  `(a,b)` was computed correctly from `R`, `r`, `e`, `e₀` (Eq. 5),
  including range checks (`R_j ∈ [0,p)`), a binary-randomness check
  (`r_i·(1−r_i) = 0`), and a noise-bound check (`|e_i| ≤ 6·stdev`).
- **Attestor does**: acts as the interactive ZK verifier throughout;
  signs the resulting ciphertext together with its session context and
  position.
- **FHE node N₁ verifies**: only the attestor's signature on the
  ciphertext (the heavy cryptographic verification already happened
  interactively between client and attestor).
- **Objects transmitted**: the LWE ciphertext `(a,b)`, a signature; the
  ZK proof itself is interactive (consumed by the attestor directly,
  not transmitted onward as a static object).
- **What stays private**: R; the attestor acts as the ZK verifier but
  never reconstructs R in the clear (the circuit is checked via
  IT-MAC-authenticated values, not revealed plaintext).
- **Assumptions**: the (scalar, prime-modulus) LWE assumption, EUF-CMA
  signatures, QuickSilver's knowledge-soundness.
- **Setup/preprocessing**: Table 1: "No CRS" ✓, but **requires
  substantial QuickSilver preprocessing** ("No Preprocessing" ✗) — the
  paper's own measured cost: ~1.6 million edaBits per batch, ~40s on
  LAN / ~118s on WAN, ~122 MB of preprocessing data, amortized across
  roughly 12 encryptions per batch (131,000 edaBits/encryption).
- **The paper's own, explicit, central limitation (Section 7.3,
  7.4)**: "the FHE mode with IZK is **limited to prime-field LWE
  schemes and lacks support for ring structure**, not making it
  compatible with TFHE-rs" — i.e., the authors themselves state this
  construction does not extend to ring-based LWE (RLWE), the algebraic
  structure CKKS is built on. This is not an inference this phase is
  drawing — it is the paper's own stated boundary.
- **CKKS-compatible?** No, and the paper says so directly. The
  parameters used (Primus library: `n=2048`, `q=2^61−1`, message
  modulus `p=4`, ternary secret key) describe a tiny, scalar-LWE,
  4-value-message-space scheme — a different cryptographic object from
  CKKS's ring-based, large-modulus-chain, thousands-of-packed-complex-
  slots-per-ciphertext design in every structural respect.

### Table 1 reproduced (paper's own comparison, for reference)

| Protocol | No Preprocessing | No CRS | Modulus Type |
|---|---|---|---|
| MPC | ✓ | ✓ | Field |
| FHE with NIZK | ✓ | ✗ | Field/Ring |
| FHE with IZK | ✗ | ✓ | Field |

Note "Field/Ring" for NIZK refers to CGGI's own torus construction
(which has both field-like and ring-like aspects in its own right), not
to RLWE/CKKS compatibility — the paper never evaluates or claims CKKS
support for any of the three protocols.

## 3. CKKS compatibility boundary: what would actually be required

**None of TLShare's three protocols are CKKS-compatible as published.**
This is stated directly, not inferred speculatively, for two of the
three (MPC mode never touches FHE at all; IZK mode is explicitly
limited to prime-field LWE by the authors' own words). NIZK mode is
CGGI-specific by construction, not explicitly ruled out for CKKS by the
authors, but not demonstrated either.

A genuine CKKS-compatible construction, reasoning from the paper's own
techniques, would need to solve problems the paper does not address:

- **Ring arithmetic, not scalar arithmetic.** The IZK mode's circuit
  proves a **matrix-vector product** `A·r` for an LWE dimension
  `n=2048`. A CKKS-compatible analog would need to prove a **ring
  (polynomial) multiplication** `a·s mod (X^N+1)` for this project's
  ring dimension `N` — 8192 in the production path, up to 131072
  elsewhere. Representing negacyclic polynomial multiplication (an
  `O(N log N)` NTT-based operation, or `O(N²)` naively) as an
  arithmetic circuit over whatever field QuickSilver (or an analogous
  NIZK) operates over is a substantially larger and structurally
  different circuit than a scalar matrix-vector product, not a drop-in
  parameter change.
- **Encoding, not just encryption.** CKKS has a SECOND source of
  approximation the paper's constructions never need to handle: the
  canonical-embedding + Delta-scaling + rounding step
  (`encode_host_fft`, this project's own implementation) that turns a
  vector of real/complex numbers into a ring element BEFORE encryption.
  A provenance proof that only covers "the ciphertext correctly
  encrypts plaintext polynomial `m`" (what both FHE modes prove) does
  NOT prove "`m` was correctly derived from the authenticated financial
  value" — that second link needs its own circuit over the FFT-based
  encoding math, which neither paper construction attempts.
- **Larger ciphertexts, larger proofs.** One full production CKKS
  ciphertext in this project (`n=8192`, `sizeQ` up to 30 RNS towers) is
  `~8192 × 30 × 2 ≈ 491,520` 64-bit limbs. The paper's own `n=2048`
  scalar LWE circuit already needed **148 million gates and 1.6 million
  edaBits of preprocessing for one encryption** (Section 7.3). A naive
  extension of the same proof technique to a ring element with this
  many more coefficients and RNS limbs would plausibly require orders
  of magnitude more preprocessing and proving work — a quantitative,
  not just qualitative, practicality concern this phase explicitly
  does NOT claim to have solved or sized precisely (doing so rigorously
  would itself be a dedicated research project).
- **Noise-bound proofs would need CKKS's own, different error model**,
  not the IZK circuit's `|e_i| ≤ 6·stdev` check verbatim — CKKS's
  rescaling-based noise management and the encoding-rounding error
  are both different error sources from a single LWE sample's
  additive noise term.

**Conclusion, stated plainly per this phase's own constraints: no
CKKS-compatible proof-of-valid-encryption-tied-to-a-committed-payload
is implemented this phase, none is claimed to exist in usable form, and
transplanting CGGI or prime-field LWE into this project's CKKS engine
is explicitly out of scope and was not done.**

## 4. Threat model: what changes when integrating with this project

TLShare's own stated assumptions (Section 3.1): the server is honest
(not actively malicious, it just serves normal TLS); the client and
attestor do not collude; TLS is not length-hiding (payload length is
public); provenance for the UNDERLYING transport is established by
`ΠAuthData`, instantiated with GtP for TLS 1.2 specifically (TLS 1.3 is
discussed as a possible future instantiation via GtP's own extension,
**not implemented or evaluated by the paper's own experiments**).

**What would change integrating with this project**: this repository has
no TLS client, no GtP integration, and (per section 3 above) no
CKKS-compatible proof of ciphertext/commitment consistency. Any
real integration would therefore inherit TLShare's full threat model
for the authentication/provenance half, but would need an entirely new,
independently-designed-and-reviewed cryptographic component for the
"prove this CKKS ciphertext matches the committed value" half — which
does not exist today, in this project or (for CKKS specifically) in the
published literature surveyed here. **TLS encryption alone is never, by
itself, proof that a downstream CKKS computation received authentic
input** — TLS protects the channel; provenance to a THIRD PARTY (the
FHE/MPC nodes, who were never part of the TLS session) is exactly the
hard problem TLShare's attestor/commitment machinery exists to solve,
and this phase's prototype (section 7) deliberately does NOT claim to
solve it.

## 5. Confirmed current input-authenticity gaps (repository evidence)

Inspected `bench/bench_private_aggregation_workload.cpp` (the actual
32-client financial aggregation application) and
`bench/bench_aggregation_decomposed.cpp` directly, source-read line by
line, not assumed:

| Finding | Actual evidence | Security impact | TLShare relevance | Recommended action |
|---|---|---|---|---|
| Client inputs are 100% synthetic | `std::uniform_real_distribution<double> d(-50.0, 50.0)` fed by a fixed-seed `std::mt19937_64 msgRng(42)`, `bench_private_aggregation_workload.cpp` line ~55-57 — no file I/O, no network, no external source of any kind | None (there is no live service, no attacker, nothing to attack) | High — this is exactly the gap TLShare's architecture targets: binding an external, authenticated record to an input | Missing feature, not a vulnerability — addressed by the Mission 4/5 envelope design |
| No source-authentication or provenance check exists | Confirmed by reading the complete benchmark file: no certificate handling, no signature verification, no commitment of any kind anywhere in `bench/` or `src/` | None currently (no deployed boundary to exploit) | High | Architectural limitation — the envelope prototype (section 7) adds this boundary, explicitly as a mock |
| No client/session identity concept anywhere in the codebase | Grepped `bench/`, `src/`, `tests/` for session/client-identity types: none exist outside this phase's own new `authenticated_financial_input.h` | None currently | Medium — any real multi-party deployment needs this regardless of TLShare specifically | Architectural limitation, now addressed by `SessionId`/`SourceIdentity` (section 7) |
| No replay/duplicate-submission protection | `ct_add_ct_host`'s signature (checked directly) takes no sequence number, nonce, or session parameter; the existing benchmark is a single in-process loop with no submission boundary to replay against | None demonstrated (no network boundary exists in the current benchmark to replay against) | Medium — TLShare's session-bound commitments (`ctx := Encode(aid, sid, ...)`) directly solve this class of problem | Architectural limitation, now addressed by `AggregationSessionVerifier`'s duplicate-commitment check (section 7) |
| Plaintext financial values never reach the aggregation step | Confirmed by reading `bench_private_aggregation_workload.cpp`'s main loop: only `acc0`/`acc1` (ciphertext) is ever passed to `ct_add_ct_host`; the per-client plaintext (`clientValues[k]`, `z`, `m`) never leaves the per-client encrypt scope | N/A — this is an existing STRENGTH, not a gap | Low (already satisfied, nothing to add) | No action needed |

**None of the above are confirmed vulnerabilities** in the sense of an
exploitable live system — there is no live system, no network boundary,
and no adversary in the current benchmark. They are accurately
characterized as **missing capabilities relative to a hypothetical real
multi-party deployment**, which is exactly the gap TLShare's paper
targets and exactly what Missions 4-5's architecture/prototype address
at the interface level (not the full cryptographic level — see section
3's explicit boundary).

## 6. Proposed architecture

```
 [Real TLS endpoint]              [NOT IMPLEMENTED this phase]
        |
        v
 [Authenticated-data protocol: ΠAuthData-equivalent]   <- NOT IMPLEMENTED
        |  produces: authenticated (Q,R), commitments, attestor signature
        v
 [SourceIdentity + SessionId + QueryResponseBinding]     <- implemented (mock fields)
        |
        v
 [PayloadCommitment]  (toy digest this phase; real impl needs Pedersen or similar)
        |
        v
 [MockAttestation]  <- EXPLICITLY MOCK, not a real signature scheme
        |
        v
 [ValidatedApplicationRecord]  (parsed numeric value + source + session)
        |
        v
 [CkksEncodingMetadata + encode_host_fft]   <- REAL, unmodified production path
        |
        v
 [encrypt_host_shoup_pooled_barrett_batched]  <- REAL, unmodified production path
        |
        v
 [FinancialInputEnvelope]  (source, session, binding, commitment, attestation, ciphertext)
        |
        v
 [AggregationSessionVerifier.verify()]   <- implemented: session/attestation/duplicate/
        |                                   ciphertext-presence checks (mock-key based)
        | Accepted only
        v
 [ct_add_ct_host aggregation]   <- REAL, unmodified production path
        |
        v
 [decrypt_host + decode_host_fft]   <- REAL, unmodified production path, per an
                                        explicit decryption/authorization policy
                                        (NOT implemented this phase -- the
                                        prototype decrypts directly, as the
                                        existing benchmarks already do)
```

Explicit trust boundaries: the attestor (mock here) never receives the
plaintext numeric value in this design's INTENT (TLShare's own attestor
never does either) — though this phase's mock test harness, for
simplicity, does pass the value to the mock attestor's `attest()` call
indirectly via the commitment digest computed from it; a real
implementation would have the attestor operate ONLY on commitments, as
documented explicitly in the header comment of
`include/authenticated_financial_input.h`. Secret keys, TLS session
secrets, and the plaintext numeric value never appear in the
`FinancialInputEnvelope` object itself (only `c0`/`c1`, public metadata,
and the commitment digest do) — confirmed by direct inspection of the
struct definition.

## 7. Prototype implementation status

**Built**: `include/authenticated_financial_input.h`,
`src/authenticated_financial_input.cpp`,
`examples/authenticated_financial_input/main.cpp`. Compiled as its own
standalone executable (`authenticated_financial_input`), linking only
`gpufhe` (the existing, unmodified library) and `CUDA::cudart` — its own
`.cpp` file is **not** added to `libgpufhe`, so this research module
cannot affect any core CKKS build target, per Mission 5's explicit
"keep the new module independent" requirement.

**What is real**: `encode_host_fft`, `encrypt_host_shoup_pooled_barrett_batched`,
`ct_add_ct_host`, `decrypt_host`, `decode_host_fft` are called completely
unmodified, exactly as the existing production benchmarks already call
them. The end-to-end correctness check (encrypted sum of 8 client values,
decrypted and compared against the plaintext sum) uses the real CKKS
pipeline with no shortcuts.

**What is explicitly MOCK, not real** (stated in the header's own
top-of-file comment, repeated here): `MockAttestor`'s "attestation" is a
non-cryptographic, deterministic keyed tag (`toy_keyed_tag`, an FNV-1a-
style rolling hash) — **not HMAC, not a digital signature, not proven
unforgeable in any sense**. There is no TLS session, no GtP, no
Pedersen commitment, no zero-knowledge proof anywhere in this
prototype. The "payload commitment" is a toy digest over the raw value
and source label, not a hiding/binding commitment scheme.

**Demonstrated and verified** (11/11 checks pass, `ctest -R
authenticated_financial_input`, re-run 3x for stability, exit code 0
every time):
- 8 golden-path envelopes, each with a distinct source/session, all
  accepted.
- The resulting aggregated ciphertext decrypts to the correct sum via
  the real, unmodified CKKS pipeline (abs err `6.07e-10`).
- A tampered commitment (post-attestation) is rejected
  (`RejectedBadAttestation`).
- An envelope for the wrong `application_id` is rejected
  (`RejectedSessionMismatch`).
- A changed `session_nonce` (post-attestation) is rejected
  (`RejectedBadAttestation`).
- Altered source metadata (post-attestation) is rejected
  (`RejectedBadAttestation`).
- An exact duplicate resubmission is rejected
  (`RejectedDuplicateSubmission`) after the first submission is
  correctly accepted.
- An envelope attested with the wrong key is rejected
  (`RejectedBadAttestation`).
- An envelope with no ciphertext payload is rejected
  (`RejectedCiphertextMissing`).

**A real bug was found and fixed in this round's own test harness**
before any of the above was trusted: the test's own `digest_of_value`
helper did not incorporate the session, so two test envelopes sharing
the same numeric value and source (but different sessions) produced
identical commitment digests, causing the "fresh submission" and
"ciphertext missing" test cases to be rejected as false duplicates of
an earlier test's envelope rather than for the reason each test
actually intended to exercise. Fixed by including the session nonce in
the digest input; re-verified all 11 checks pass for the right reasons
afterward.

## 8. Security limitations (stated prominently, not buried)

**This prototype is NOT cryptographically secure end to end, and does
not claim to be.** Specifically:
- The "attestation" can be forged by anyone who learns the shared toy
  key — there is no public-key cryptography, no signature scheme, and
  no unforgeability property proven or claimed.
- There is no actual TLS session, no real server, and no genuine
  external data source — "authenticity" in this prototype means only
  "internally consistent with a shared mock key," not "verifiably
  derived from a real external TLS response."
- There is no proof linking the CKKS ciphertext's plaintext to the
  committed value (section 3's gap) — the prototype trusts that the
  SAME process that built the commitment also correctly encoded and
  encrypted the SAME value, which is true by construction in this test
  harness but is NOT independently verified by any cryptographic means
  the aggregation boundary could check on its own.
- Do not use this module, or any output it produces, to make real
  claims about data provenance, authenticity, or zero-knowledge
  properties outside of this prototype's own test harness.

## 9. Encode/FFT investigation: methodology, results, and a documented null result

**Re-verified the production path and the historical finding fresh**:
`bench_aggregation_decomposed` re-run, confirming encode remains the
next-largest relative stage (~10.0-10.5ms of a ~87-88ms end-to-end
median, ~11.7-12%), matching Phase 34's own closing note.

**Inspected `src/encode_fft.cpp` directly** (212 lines, read in full):
a single-threaded, textbook iterative radix-2 Cooley-Tukey FFT
(decimation-in-time), with process-lifetime-cached twiddle tables and
slot-index tables (both already mutex-guarded, consistent with every
other cache in this codebase since Phases 28/33). Confirmed **zero CUDA
calls** anywhere in this file — encode/decode are intentionally,
correctly CPU-only (re-confirming Phase 32's own finding, not assumed).

**Found one genuine, safe micro-optimization opportunity by direct code
reading**: `fft_unnormalized_inverse(std::vector<cd> Z)` takes its
parameter **by value**, and its one call site
(`fft_unnormalized_inverse(Z)` inside `encode_host_fft`) passes a named
lvalue — silently copy-constructing an entire `M`-element (`2N` complex
doubles, 256 KB at `n=8192`) buffer on every single encode call, when
`Z` is never read again afterward and could safely be moved instead.

**Isolated measurement** (`research/benchmarks/bench_encode_fft_timing.cpp`,
400 reps, calling the real unmodified production function): BEFORE
median 532-547 μs/call; AFTER (one-line `std::move(Z)` fix) median
288-292 μs/call — a reproducible **~1.82-1.87x** improvement in
isolation, confirmed across 3 independent process launches each way.

**Correctness, verified before trusting any timing**: all three
existing encode_fft test suites re-run and passed unchanged after the
fix — `test_encode_fft_matches_reference` (every dimension), `test_encode_fft_concurrency`
(166 thread executions, 0 crashes/corruption), `test_encode_fft_scale_and_boundary`
(32/32 checks).

**Real end-to-end re-measurement — a documented null result, not hidden**:
a proper `git stash`-based A/B directly on `bench_aggregation_decomposed`
(the actual production workload, not the isolated probe) showed **no
measurable difference**: encode median 9.98-10.05 ms BEFORE vs.
10.00-10.07 ms AFTER — statistically indistinguishable, both well
within this machine's own run-to-run noise. The dramatic isolated-probe
win simply does not transfer to the real workload.

**Why, investigated rather than left unexplained**: the isolated probe
runs 400 back-to-back encode calls with no gaps, plausibly sustaining a
higher, continuous CPU clock/boost state throughout a tight loop. The
real workload interleaves each encode call with a multi-millisecond
GPU-bound encrypt phase (NTT kernels, `cudaMemcpy`, mutex-guarded pool
access) during which the CPU is largely idle — a plausible trigger for
CPU frequency/power-state downclocking between encode calls, so each
real-workload encode call may pay a comparatively large "cold start" tax
that swamps the small, real allocation saving. This is the direct CPU
analog of Phase 34's own already-documented GPU-clock-drift lesson
(isolated vs. interleaved GPU micro-benchmarks), now observed for CPU
work interleaved with heavy GPU activity instead of GPU work interleaved
with GPU work.

**Decision, per this phase's own explicit instruction ("if the result
is null, revert only your own experiment and document it")**: the
`src/encode_fft.cpp` change was **reverted** (`git checkout --`,
confirmed zero diff afterward) rather than shipped, despite being
correctness-neutral and objectively not harmful, because its measured
real-workload benefit is null and Mission 6 explicitly directs reverting
null-result experiments rather than retaining unproven ones. The
standalone timing probe (`research/benchmarks/bench_encode_fft_timing.cpp`)
is kept as a reusable tool and as evidence for this investigation, not
wired into any production path.

**What this does NOT mean**: encode is not "already optimal" — it means
this ONE micro-optimization does not help THIS workload's specific
call pattern. The genuinely open, larger question (not attempted this
round, given the "smallest safe experiment" discipline) is whether the
dominant real-workload cost is CPU power-state transitions themselves
(in which case no algorithmic change to `encode_host_fft` would help —
the fix would need to live at the CALL SITE, e.g. keeping the CPU busy
with other work between GPU-bound phases) or something else entirely
not yet isolated.

## 10. Cost model for the intended application (Mission 8)

The paper's own numbers (0.46-3.19s TLS-phase latency, 0.6-4.6s FHE
mode end-to-end, 40-118s IZK preprocessing per batch) are measurements
on the authors' own AWS infrastructure and implementations (GtP, RELIC,
TFHE-rs, EMP toolkit, MP-SPDZ) — **not comparable to this project's GPU
benchmarks**, which measure a completely different operation (CKKS
encrypt/aggregate/decrypt on an RTX 2060) with no TLS, attestation, or
ZK-proof component at all. No attempt is made here to combine these
numbers into one blended latency figure, since doing so would imply a
real integration that does not exist.

**Qualitative cost model for a hypothetical future integration** (order
of magnitude, not a benchmark): per-client cost would be
`authentication_latency + proof_generation + proof_verification +
CKKS_encrypt (already measured: ~75ms/32 = ~2.3ms/client in the
production path) + network + (amortized aggregation + decryption)`. Given
the paper's own NIZK mode alone costs ~1.1-4.6s end-to-end per payload —
roughly **450-2000x** the CKKS encrypt stage's own per-client cost — any
real integration's bottleneck would almost certainly be the
authentication/proof layer, not the CKKS encryption this project
already optimizes. Whether that overhead is "acceptable" depends
entirely on the deployment's own latency budget (a real-time aggregation
service vs. an overnight batch wage-equity report, per the paper's own
Section 6.2 application, have very different tolerances) — this phase
does not have enough information to make that call for a real
deployment and does not attempt to.

## 11. Deferred work / exact next steps

1. **A real CKKS-compatible provenance proof remains unsolved** (section
   3) — this is a dedicated cryptographic research effort, not a
   follow-up task; the next concrete step, if pursued, is sizing an
   RLWE-adapted vector-membership-style NIZK's circuit complexity at
   this project's actual ring dimensions before attempting any
   implementation.
2. **The mock attestor's current design passes the plaintext value to
   `attest()` indirectly via the commitment** — a real design should
   make the attestor operate on commitments ONLY, never anything
   value-derived in the clear; flagged explicitly in the header comment,
   not yet restructured.
3. **The CPU-power-state hypothesis for the encode null result (section
   9) is not independently confirmed** — only inferred from the
   measurement pattern. Confirming it would need direct CPU frequency
   monitoring (e.g., `/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`
   sampled during the real workload), not attempted this round.
4. **Re-run `tests/test_boot.cpp`** the moment the 10 GiB memory gate
   clears — now EIGHT phases running without this verification.

## 12. Exact next steps for the following session

1. Read this document's section 3 before considering any CKKS+ZK-proof
   work — do not re-attempt transplanting CGGI or prime-field LWE
   constructions into CKKS; if pursuing a provenance proof, start by
   sizing the circuit, not implementing one.
2. If continuing the authenticated-input prototype, fix the attestor's
   value-visibility shortcut (deferred item 2 above) before adding any
   more features to it.
3. If continuing encode/FFT work, investigate the CPU power-state
   hypothesis directly (deferred item 3) before trying another
   micro-optimization in isolation — an isolated win that doesn't
   transfer to the real workload will keep happening until the real
   bottleneck (power state transitions, if that's really it) is
   addressed directly.
4. Re-check the Windows free-RAM gate before any large-n work.
