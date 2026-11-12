# Phase 36: end-to-end bottleneck investigation, authenticated-input hardening, and two documented null/negative performance results

## 1. Starting commit and environment

Starting commit: `a07ecf8` (Phase 35's final commit), confirmed matching
`origin/main` exactly (0 ahead / 0 behind) at Phase 36 start.

Hardware/software, confirmed at session start: RTX 2060 Max-Q, 6144 MiB
VRAM, compute capability 7.5, driver 595.97, CUDA 13.2, Release `-O3`.
cmake 3.28.3. Rust/cargo 1.94.0. 16 logical CPU cores (`nproc`). Windows
host RAM: 16,168,544 KB total (~15.4 GiB); free RAM was volatile
throughout this session — observed as low as ~1.98 GiB (briefly, mid-way
through `test_n131072_cache_churn`) and as high as ~5.9 GiB, checked via
`Get-CimInstance Win32_OperatingSystem`'s `FreePhysicalMemory` before
every memory-sensitive step (never WSL's own `free -h`, which reports
the WSL VM's fixed ~10 GiB ceiling, not real host availability). An
earlier attempt to run the full C++ `ctest` suite in one pass was killed
by the host for running the system low on memory; the suite was re-run
afterward in smaller, memory-aware batches (section 7).

## 2. Phase 35 baseline (re-stated, not re-measured from scratch)

- C++ `ctest`: 117/127 passing, the same 10 pre-existing documented
  failures as every prior phase, zero regressions.
- Rust: 62 passed / 0 failed / 1 ignored, both debug-serial and
  release-parallel.
- Both n=131072 tests (`test_n131072_keysize`, `test_n131072_cache_churn`)
  passing.
- `authenticated_financial_input`: 11/11 checks passing.
- Encode/FFT: a documented, reverted null result (1.82x isolated speedup,
  0x real-workload benefit — CPU power-state hypothesis, not independently
  confirmed).

## 3. Stage-level measurements and methodology (Track A)

Re-ran `bench/bench_aggregation_decomposed.cpp` fresh at session start
(n=8192, S=4096, sizeQ=4, K=32 clients, R=10 reps, interleaved A/B
methodology already built into this benchmark) to re-confirm the current
bottleneck before assuming Phase 35's numbers still held:

```
Arm B (PRODUCTION encrypt_host_shoup_pooled_barrett_batched):
  encode    median=10.47ms
  encrypt   median=75.04ms   <- 85.6% of end-to-end
  aggregate median=1.27ms
  decrypt   median=0.73ms
  decode    median=0.28ms
  END-TO-END median=87.68ms
```

Confirmed: encrypt remains completely dominant (85.6% of end-to-end),
unchanged in kind from Phase 34/35's findings.

### 3.1 A correction to Phase 34's own prior methodology

Phase 34's `bench_encrypt_cpu_gpu_split.cpp` concluded CPU-only work was
~1-2% of one encrypt call by bracketing the ENTIRE function call
externally with `cudaEventRecord` calls before/after, then computing
`wall_time - gpu_bracketed_time` as "CPU-only." This phase built direct
internal instrumentation (timers placed at the true start/end of the
GPU-touching section, inside the function itself — fully applied, read,
and then reverted, never shipped) and found the real split is
approximately **40-48% CPU-side (RNG + embed + combine) vs. 52-60%
GPU-side (transfer + kernel + sync)** per call. The external-bracketing
approach was methodologically flawed: the closing `cudaEventRecord` can
only be enqueued once the CPU thread reaches that line — i.e. after ALL
preceding host-only work (RNG, embed, combine) has already run — so the
"GPU-bracketed" window ends up re-measuring almost the full wall-clock
span rather than isolating true GPU-only time, making the subtraction
meaningless. This correction is recorded here explicitly, per this
project's practice of documenting corrected findings rather than
silently revising them.

### 3.2 Ranked candidates (Track A deliverable)

1. **Parallelize the embed and combine loops across CPU threads**
   (highest *a priori* rank, given the 40-48% CPU-side finding above;
   low *a priori* correctness risk — pure reordering of independent
   per-coefficient work, no arithmetic change). **Implemented and
   measured (Track B); result: confirmed regression — section 4.1.**
2. **Pool the per-call host-side scratch buffers** (`packed`, `vc`,
   `e0c`, `e1c` — a fresh ~1 MiB allocation per call despite the whole
   function already being globally mutex-serialized; low risk, reuses
   the device-side pool's already-audited pattern). **Implemented and
   measured (Track B); result: statistically null — section 4.2.**
3. **Revisit pinned host memory for the H2D/D2H transfer.** Already
   explored in Phases 15-16 (`research/TRANSFER_PROFILING.md` section 3
   and its Phase 16 prewarming follow-up): measured only 1.06-1.17x
   isolated improvement with real adoption friction (lazy-allocation
   cost on first use), deliberately kept as an opt-in experimental
   feature rather than promoted to production. Not re-attempted this
   phase — restudying an already-quantified, modest, previously-rejected
   win was judged lower expected value than testing new candidates
   within this session's scope.
4. **Replace the combine loop's per-element `shoup_precon` (a 128-bit
   division per coefficient) with a Barrett-style reduction**, reusing
   the `TowerBarrett` machinery already in this file for the embed step
   — plausibly the largest remaining single-call CPU win, since
   division is expensive and Barrett reduction needs no per-element
   division once the per-tower `mu` is precomputed. **Not attempted**:
   this touches the core modular multiplication used by every encrypt
   call, which this project's rules require a standalone reproducer and
   focused test for before modifying, and carries real subtle-correctness
   risk (Barrett bound errors are a classic hard-to-detect bug class).
   Documented as the best-supported remaining direction for a future
   phase with dedicated scope, not something to implement speculatively
   within this session's remaining budget.
5. **Change the RNG algorithm or distribution.** Not recommended at any
   rank — touches security-relevant randomness directly, which this
   project's rules forbid weakening or casually altering.

## 4. Track B: selected optimization, implementation, and why both attempts were reverted

Per the mission's explicit allowance ("if the best candidate can't be
optimized safely, document evidence and move to the next-ranked
candidate ... a null result is acceptable, a fabricated win is not"),
the two lowest-risk candidates were both implemented, validated for
bit-exact correctness, measured rigorously against the real workload,
and then reverted when neither produced a real benefit. **No performance
change ships from Track B this phase** — this is a deliberate, evidenced
null/negative result, not an incomplete investigation.

### 4.1 Candidate 1 — thread-parallelized embed/combine loops: REVERTED, confirmed regression

**Implementation**: duplicated this project's own established
`parallel_for` utility (`kParallelThreshold=256`,
`hardware_concurrency()`-capped worker count, chunked `std::thread`
dispatch — verbatim pattern from `src/encode.cpp`'s anonymous namespace)
into `encrypt_host_shoup_pooled_barrett_batched.cpp`'s own anonymous
namespace, matching this project's established convention of
duplicating small per-file helpers rather than centralizing them (e.g.
`am()`, `shoup_precon()`, `TowerBarrett` are already independently
duplicated across the `encrypt_host_shoup*` variant files). Flattened
both the embed loop and the combine loop from per-tower/per-coefficient
nested loops into a single `parallel_for` over the full `sizeQ*n` index
space — necessary because `sizeQ` alone (4 at the real production
parameters) is far below `kParallelThreshold`, while `sizeQ*n` (32768)
comfortably exceeds it. Per-tower constants (`TowerBarrett` for embed;
`(q, nsq, nsPrecon)` for combine) were hoisted out of the loop once, not
recomputed per element.

**Focused correctness evidence**: all 6 dedicated tests
(`test_encrypt_host_shoup_pooled_barrett_batched_{matches_reference,
concurrency_audit}` and their `_pinned_`/`_registered_` variants) passed
— bit-exact, including the concurrency audits. The reordering changes
no arithmetic; only which thread performs which independent,
disjoint-output computation.

**Before/after performance — isolated vs. end-to-end, both measured**:
on `bench_aggregation_decomposed` (same params as section 3), the
encrypt-stage median went from **75.04ms to ~155-157ms — roughly 2.1x
SLOWER**, reproduced across two independent runs (154.65ms and 157.30ms
median). This is a real, reproducible regression, not noise (stddev per
run was only 2.4-5.0ms). The end-to-end effect tracked the isolated
effect directly (no divergence between isolated and real-workload
measurement here, unlike Phase 34/35's prior lessons) — because the
regression's cause (thread-spawn overhead) is intrinsic to the function
itself, not an interleaving/environmental artifact.

**Root cause**: `std::thread` creation/join overhead per call. At 16
logical cores, each `parallel_for` call spawns up to 16 worker threads;
this function calls `parallel_for` TWICE per encrypt call (embed,
combine), and the aggregation workload calls encrypt 32 times (K=32
clients) — up to ~1024 thread create/join pairs per aggregation round.
The actual CPU work being parallelized (Barrett/Shoup arithmetic over
32768 elements) is itself only a fraction of a millisecond; OS-level
thread spawn/join cost on this platform (WSL2/Linux) is evidently large
enough, relative to that sub-millisecond workload, to dominate and
invert the expected win.

This extends this project's own repeated lesson that an isolated or
theoretical win does not reliably transfer to the real production
workload (Phase 34's GPU-clock-drift finding; Phase 35's
CPU-frequency-scaling finding) with a new variant: here the mismatch is
not environmental drift but a fundamental **granularity mismatch**
between a parallelization primitive's fixed per-call cost and the
actual per-call workload's size.

**Disposition**: reverted (`git checkout --`), confirmed zero diff
afterward, confirmed the ~75ms baseline median is restored after
rebuilding.

### 4.2 Candidate 2 — pooled host-side scratch buffers: REVERTED, confirmed null

**Implementation**: added a `HostScratchPoolBatched` struct (holding
`vc`/`e0c`/`e1c`/`packed`), cached process-lifetime and keyed by
`(n, sizeQ)` using the exact pattern already established and audited for
this file's device-side pools (`EncDevTab`, `EncryptBufPoolBatched`) —
relying on the SAME external mutex (`pool_mu`, held for the entire
function body) for safety, adding no new synchronization primitive.
Every element is fully overwritten by its owning call before being read,
identical to the already-audited invariant for the device-side packed
buffer.

**Focused correctness evidence**: all 6 dedicated tests passed,
bit-exact, including the concurrency audits.

**Before/after performance**: a proper `git stash`-based interleaved A/B
directly on `bench_aggregation_decomposed` (4 alternating trials,
OLD/NEW/OLD/NEW), per this project's established, repeatedly-necessary
methodology:

| Trial | OLD (unpooled) median | NEW (pooled) median |
|---|---|---|
| A | 74.14ms | 74.38ms |
| B | 74.06ms | 75.27ms |

The difference (~0.2-1.2ms out of ~74ms, well under 2%) is within this
benchmark's own observed run-to-run noise (per-run stddev 1.6-3.4ms).
**No measurable isolated or end-to-end benefit.** Plausible explanation:
glibc's allocator evidently reuses freed same-sized blocks cheaply
enough at this buffer size (~1 MiB) that there was little allocator
churn cost to eliminate — the hypothesis motivating this candidate was
reasonable but not borne out empirically.

**Disposition**: reverted (`git checkout --`), confirmed zero diff
afterward.

### 4.3 Track B conclusion

No production performance change ships from this phase. Both attempted
candidates were implemented correctly, validated for bit-exact
correctness, and measured honestly against the real workload — one was
a confirmed regression, the other a confirmed null result. The
best-supported remaining candidate (Barrett-based combine reduction,
section 3.2 item 4) was deliberately not attempted given its correctness
risk and the project rule requiring a standalone reproducer before
touching this class of core numeric code.

## 5. Track C: authenticated-input review and hardening

Reviewed `examples/authenticated_financial_input/`'s 11 existing checks
against the mission's checklist (envelope validation, commitment
construction, CKKS-pipeline integration, payload parsing, failure
handling). Found two real, meaningful gaps, both now closed:

1. **`CkksEncodingMetadata` (`n`, `delta`) was carried in every envelope
   but never checked against anything.** An internally self-consistent
   envelope encoded with the WRONG `delta` (or `n`) for a given
   aggregation session would have been accepted and could silently
   corrupt the aggregate (ciphertexts encoded at different scales do not
   sum meaningfully). Fixed two ways, matching the existing design's own
   symmetry:
   - `canonical_attestation_bytes` (and `MockAttestor::attest`) now also
     take the envelope's `CkksEncodingMetadata` and bind it into the same
     attestation tag as source/session/binding/commitment — a
     post-attestation tamper of `enc_meta` is now caught the same way
     tampering any other field already was (`RejectedBadAttestation`).
   - `AggregationSessionVerifier` separately checks the envelope's
     `enc_meta` against the session's own expected `(n, delta)`
     (new `RejectedEncodingMismatch`) — this catches an envelope that is
     honestly, self-consistently attested for the WRONG parameters, a
     failure mode no attestation check alone can ever catch (the
     attestation only proves internal consistency, not session
     correctness).
2. **Only ciphertext emptiness was checked, not shape.** A truncated or
   otherwise wrong-length (but non-empty) `c0`/`c1` would have passed
   straight through to the real `ct_add_ct_host` aggregation call,
   risking corruption or a crash. The verifier now also checks
   `c0.size() == c1.size() == sizeQ * n` for the session's own
   parameters (new `RejectedCiphertextShapeMismatch`).

The verifier's check order was deliberately designed so the two new
checks are independently exercisable: attestation validity is checked
BEFORE the session-level encoding-parameter check, so a post-attestation
tamper of `enc_meta` is attributed to `RejectedBadAttestation`, while an
envelope that was honestly attested for the wrong session parameters in
the first place is attributed to `RejectedEncodingMismatch` — two
genuinely distinct failure modes, not one check subsuming the other.

Three new adversarial tests were added to `main.cpp`: `enc_meta`
tampered after attestation; an envelope correctly attested for the wrong
session delta; a truncated ciphertext. All pass, alongside the original
11, for **14/14 total**. No core CKKS code (`libgpufhe`) was touched;
the mock attestor remains unmistakably labeled non-cryptographic; no new
claim is made about TLS/TLShare/GtP/ZK provenance. The example's
`README.md` was updated to describe both changes and the new check
count.

## 6. Track D: CKKS-compatible provenance research conclusions

Phase 35 (`research/PHASE35_TLSHARE_CKKS_INTEGRATION.md` section 3)
already established, from direct reading of the TLShare paper, that none
of its three published protocols are CKKS-compatible as published, and
reasoned through four specific obstacles (ring arithmetic vs. scalar
arithmetic; CKKS's second approximation source at the encoding step;
proof-size scaling; a different noise model). This phase does not
repeat that analysis. It adds an explicit separation of the problem into
six distinct sub-claims, and a survey of which modern proof-system
*families* are at least structurally plausible for each — stated as
candidate research directions, not verified solutions, and not
implemented.

### 6.1 The six concepts, kept separate

A genuine "this CKKS ciphertext was produced from an authenticated
external value" proof is not one claim — collapsing it into one is
exactly the mistake this project's own rules forbid (never treat a
hash/mock attestation/ordinary signature as proof of TLS provenance). It
is six separate claims:

1. **TLS provenance** — the payload came from the claimed server over a
   genuine TLS session.
2. **Payload commitment** — a binding, hiding commitment to the raw
   bytes received.
3. **Correct parsing/canonicalization** — the one numeric field needed
   was extracted from the committed payload correctly.
4. **Correct CKKS encoding** — that extracted value was correctly turned
   into a ring element via canonical-embedding + Delta-scaling +
   rounding (`encode_host_fft`).
5. **Correct encryption** — the ring element was correctly encrypted
   into the ciphertext under the claimed public key.
6. **Ciphertext-matches-committed-value** — the conjunction: the
   specific ciphertext the aggregator holds encrypts a plaintext
   consistent with the committed value, through the exact parsing (3)
   and encoding (4) actually used, under the exact encryption (5)
   actually performed. Proving 1-5 individually does not by itself
   prove 6 unless each sub-proof's witnesses are provably the same
   values carried into the next (a standard commit-and-prove composition
   requirement).

### 6.2 Candidate research directions (not implemented; not verified fits)

- **Concepts 1-2**: unchanged from Phase 35 — GtP-style attestation
  (1) and a standard commitment scheme (2) are independent of the
  downstream FHE scheme; not re-analyzed here.
- **Concept 3**: ordinary arithmetic-circuit territory, well within
  reach of general-purpose zk/IZK systems including TLShare's own
  QuickSilver-style machinery; the least novel of the six.
- **Concept 5**: this is exactly the kind of statement — knowledge of a
  short secret/error vector satisfying a public (module-)LWE/RLWE
  relation — that a newer generation of **lattice-native proof systems**
  (e.g. the LaBRADOR/Greyhound/LatticeFold family of constructions
  published 2023-2024, built to argue directly over
  `R_q = Z_q[X]/(X^N+1)` rather than compiling ring operations into a
  generic arithmetic circuit over an unrelated field) is specifically
  designed for — the most promising-*looking* direction of the six,
  precisely because the proof system's native algebra already matches
  the statement's native algebra, unlike QuickSilver's field-based
  circuits (Phase 35's own flagged dominant cost driver). **This is not
  a claim that any such system has been shown to handle this project's
  actual parameters (N up to 131072, sizeQ up to 30) efficiently** —
  that would require dedicated sizing work this phase does not attempt.
- **Concept 4 — the open gap specific to CKKS**: this is the one
  concept with no identified existing tool. BGV/BFV (CKKS's "exact" FHE
  siblings) encode via exact modular arithmetic — proving correct
  BGV/BFV encoding is ordinary integer-circuit work. CKKS instead
  encodes via a **floating-point** canonical-embedding FFT followed by
  **rounding** — proving a floating-point computation's rounding
  behavior inside any proof system (lattice-native or circuit-based
  alike) is a known-hard, unsolved-in-general problem, the same obstacle
  zero-knowledge-ML systems hit proving floating-point inference
  (usually worked around by proving a fixed-point *approximation*
  instead, which reopens the "is the proxy claim equivalent to the real
  claim" question this project's rules already warn against glossing
  over). **No candidate solution is identified for concept 4; it is
  recorded as the specific, CKKS-native open research question**,
  distinct from and harder than the ring-arithmetic-scaling concern
  Phase 35 already identified for concept 5.
- **Concept 6**: composing proofs 2-5 is a general commit-and-prove
  problem, tractable in principle once 1-5 individually exist — but
  concept 4's absence means concept 6 cannot be attempted for CKKS
  today regardless of how mature concepts 1, 2, 3, and 5 become
  independently.

### 6.3 Conclusion

No CKKS-compatible provenance proof is implemented, sized, or claimed to
exist in usable form this phase, matching Phase 35's own conclusion. The
incremental contribution here is a clean six-way separation of the
problem and the identification that **CKKS's approximate,
floating-point-based encoding step — something BGV/BFV simply do not
have — is the specific, so-far-unsolved bottleneck concept for this
scheme**, separate from and in addition to the ring-arithmetic-scaling
cost Phase 35 already identified for the encryption-correctness half.
Transplanting CGGI or prime-field LWE into this project's CKKS engine
remains explicitly out of scope and was not done.

## 7. Regression (Track E)

An initial attempt to run the full C++ `ctest` suite in one `-j1` pass
was **killed by the host** ("system is running low on memory") before
producing any output — confirmed this was host-wide memory pressure, not
a leftover/runaway process (WSL's own footprint was back down to ~517 MB
immediately after the kill; `ps aux` inside WSL showed no orphaned
processes). Re-ran in memory-aware batches instead:

- **Batch 1** (127 tests, excluding `test_n131072_keysize`,
  `test_n131072_cache_churn`, `test_boot_n131072`,
  `test_n131072_gpu_lifecycle`): **117/127 passing**, the exact same 10
  pre-existing failures as every prior phase —
  `test_evalkeygen_isolated`, `test_keyswitch_stage_trace`,
  `test_polyeval`, `test_stage2`, `test_boot_multi_input_small`, and the
  five `test_rescale_*_n1024*` family. Zero new regressions. Includes
  the new 14-check `authenticated_financial_input` (passing).
- **`test_n131072_keysize`** (own measured requirement: ~1.3 GB RSS,
  per `research/MEMORY_FEASIBILITY.md`): free RAM checked at ~5.8 GiB
  beforehand — **Passed**, 8.56s.
- **`test_n131072_cache_churn`** (own measured requirement: ~3.2 GiB RSS
  plateau): free RAM checked at ~5.77 GiB beforehand — **Passed**,
  186.40s. Free RAM dipped to ~1.98 GiB mid-run (consistent with the
  documented plateau plus other host processes) and recovered to ~5.6
  GiB after completion; this was watched but not interrupted.
- **`test_boot_n131072`** and **`test_n131072_gpu_lifecycle`**:
  **deferred**, not run. Both belong to the established high-memory-gate
  group (6-10 GiB class, same grouping used in essentially every prior
  phase's regression notes); free RAM never cleared that gate this
  session (observed range ~1.98-5.9 GiB), and `test_boot_n131072`
  specifically is independently documented in
  `research/MEMORY_FEASIBILITY.md` as never having completed a single
  bootstrap stage even in past sessions with more generous memory. Per
  this phase's explicit instruction ("do not run a full bootstrap if the
  documented memory gate isn't met"), neither was attempted.
- **Rust**: `cargo test --release`, both `RUST_TEST_THREADS=1`
  (debug-serial-equivalent) and default parallel threading —
  **62 passed / 0 failed / 1 ignored**, both modes, unchanged from
  Phase 35.

**Net C++ regression count this phase: 119/129 attempted (117 base +
both light n=131072 tests), 2 deliberately deferred** — matching the
Phase 35 baseline exactly on every test that was run, zero new
regressions, plus 3 net new passing checks in `authenticated_financial_input`
(11 -> 14).

## 8. Resource limitations

- Windows host free RAM was unusually volatile this session (observed
  range ~1.98-5.9 GiB), causing one full-suite run to be killed outright
  — all subsequent memory-sensitive work was done in smaller batches
  with an explicit pre-check, per section 7.
- `test_boot_n131072` and `test_n131072_gpu_lifecycle` remain unverified
  this phase for the reasons in section 7; this is consistent with their
  status in essentially every prior phase of this project, not a new
  limitation this phase introduced.

## 9. Known failures and unresolved risks

- The 10 pre-existing C++ failures and the 1 ignored Rust test are
  unchanged from every prior phase; none are newly introduced or newly
  investigated this phase (consistent with the project's own
  documentation of their root causes elsewhere).
- No production performance improvement was found or shipped this phase
  for the dominant (85%+) encrypt stage — the two lowest-risk candidates
  were both ruled out empirically (section 4). The most promising
  remaining direction (Barrett-based combine reduction) requires
  dedicated, carefully-reviewed work on core modular-arithmetic code that
  this phase did not attempt.
- The CPU-thread-parallelization regression (section 4.1) is a new,
  useful negative data point for this codebase specifically: naive
  per-call `std::thread`-based parallelization of small (sub-millisecond),
  high-frequency CPU work is actively harmful on this platform, not
  merely unhelpful. Any future attempt to parallelize similarly-small
  per-call CPU work here should budget for this before trying.
- Track D's conclusion (CKKS's floating-point encoding step has no
  identified proof technique) is a research gap, not a solved problem —
  it should not be read as implying any near-term implementation path
  exists.

## 10. Exact commits and push status

Two commits, on top of Phase 35's `a07ecf8`:
- `2d32b60` -- Track C: harden authenticated_financial_input envelope validation.
- `4161262` -- Track A/B/D/E: end-to-end bottleneck investigation and research synthesis (this document, the ledger entry, and the NEXT_SESSION.md handoff).

Both verified fast-forward-pushable and pushed to `origin/main` (see the
end of this session's final report for push confirmation).
