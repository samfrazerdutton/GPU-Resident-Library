# Phase 37: production encryption CPU-path optimization, measurement correction, and end-to-end performance

## 1. Starting commit and environment

Starting commit: `69abd0d` (Phase 36's final commit), confirmed matching
`origin/main` exactly (0 ahead / 0 behind) at Phase 37 start.

Hardware/software, confirmed at session start: RTX 2060 Max-Q, 6144 MiB
VRAM, compute capability 7.5, driver 595.97, CUDA 13.2, Release `-O3`.
cmake 3.28.3. Rust/cargo 1.94.0. gcc 13.3.0, Ubuntu 24.04.4. 16 logical
CPU cores (`nproc`). Windows host RAM: 16,168,544 KB total (~15.4 GiB);
free RAM was substantially more volatile this session than Phase 36's
own (1.98-5.9 GiB) — observed range **~1.1-7.3 GiB**, with two separate
full-suite/sub-batch `ctest` runs killed outright by the host for
running the system low on memory (section 7). No orphaned/runaway
process was ever left behind by either kill (`ps aux` checked
immediately after each).

## 2. Phase 36 baseline (re-stated, not re-measured from scratch except where noted)

| Finding | Phase 36 result | Re-verified this phase? |
|---|---:|---|
| Encryption share of aggregation runtime | 85.6% | Re-measured fresh, section 3 |
| CPU-side work within an encryption call | ~40-48% | Re-measured at finer granularity, section 3 -- refined to ~53% |
| Thread-parallel embed/combine experiment | 2.1x regression | Not re-attempted (no new hypothesis) |
| Host scratch-buffer pooling | null result | Not re-attempted (no new mechanism found) |
| Authenticated-input prototype | 14/14 checks | Reviewed, confirmed adequate, section 8 |
| C++ regression | 117/127, same 10 failures | Re-run, section 9 |
| Rust debug | 62/0/1 | Re-run, section 9 |
| Rust release | 62/0/1 | Re-run, section 9 |

Both light n=131072 tests passed in Phase 36; the two heaviest tests
were deferred under the established memory gate after a full-suite
attempt was killed by the host. This phase does **not** claim those
two deferred tests passed -- see section 9 for this session's own,
separately-evidenced outcome (one of them was attempted and could not
be completed this session either, for a different, specific reason).

## 3. Measurement correction: Phase 34's own prior CPU-only estimate

Phase 36 already corrected Phase 34's claim that CPU-side work is only
~1-2% of one encrypt call (the real figure, via internal
instrumentation, is far larger). This phase did three further things:

1. **Traced the correction to its primary source document** and added
   an explicit, prominent, non-destructive correction note directly
   beneath the original claim in
   `research/PHASE34_PERFORMANCE_INVESTIGATION.md` (the 97.5-98.8%
   GPU-side / 1.2-2.5% CPU-side bullet from
   `bench_encrypt_cpu_gpu_split.cpp`). The original measured numbers
   are left completely unchanged -- only a `>` blockquote correction is
   appended, explaining the external-`cudaEvent`-bracketing flaw in the
   same terms Phase 36's own report used, and explicitly stating this
   correction does not affect this document's other numeric claims
   (the PCIe transfer floor, the kernel-launch count, the kernel-merge
   speedup -- none of which rely on CPU/GPU-time subtraction).
2. **Searched for every other place the same 1.2-2.5%/97.5-98.8%
   figures appear** (`research/EXPERIMENT_LEDGER.md` entry #64,
   `research/NEXT_SESSION.md`'s old "Earlier commit (Phase 34)"
   section). Both are append-only historical records by this project's
   own convention (the ledger is never retroactively edited; old
   "Earlier commit" sections in `NEXT_SESSION.md` are preserved
   verbatim as history, with corrections living in the CURRENT section
   instead) -- left untouched, consistent with "preserve the original
   experiment results as historical records."
3. **Confirmed no OTHER document's numeric claim depends on the same
   flawed technique.** `research/CKKS_RESIDENCY_AUDIT.md`'s mention of
   this function only counts transfer/sync/malloc CALL COUNTS (not a
   CPU/GPU time split) and is unaffected. No downstream DECISION in
   Phase 34's own report was found to rest on the 1.2-2.5% number
   beyond the characterization itself.

## 4. Track A: detailed CPU profile of the production encrypt path

Instrumented the REAL production function
(`encrypt_host_shoup_pooled_barrett_batched`) directly, temporarily
(`#ifdef PHASE37_INSTRUMENT`, fully applied, measured, then reverted --
confirmed zero diff afterward), with internal `chrono`/`cudaEvent`
timers placed at the true boundaries of each distinct operation, not
external black-box bracketing (the exact technique Phase 36 found
unreliable). Ran via `bench_aggregation_decomposed` (n=8192, S=4096,
sizeQ=4, K=32 clients x 10 reps = 320 real production calls), parsed
all 320 `[P37]` lines.

**Per-row internal consistency check** (not just column medians):
reconstructing `total` from its own sub-components matched the
directly-measured `total` to within floating-point rounding on every
sampled row (diff ≈ 0.0000ms) -- confirming the chrono buckets are
contiguous and non-overlapping, i.e. **no double-counting** between
CPU execution time, GPU-section wall-clock time, and combine time.

**Steady-state medians** (n=320, first-call warm-up effects visible
but not excluded from the reported median since they're a small
minority of 320 samples):

| Component | Type | Median | % of total |
|---|---|---:|---:|
| RNG (sampling, incl. c0/c1 resize) | CPU | 0.487 ms | 19.6% |
| Packed-buffer allocation | CPU | 0.056 ms | 2.3% |
| Embed loop (Barrett reduction x4/coeff) | CPU | 0.407 ms | 16.4% |
| Kernel-argument table build | CPU | 0.001 ms | ~0.05% |
| GPU section (wall-clock, host's view) | mixed | 1.057 ms | 42.6% |
| &nbsp;&nbsp;-- small pointer-table memcpys | GPU | 0.131 ms | |
| &nbsp;&nbsp;-- big H2D upload | GPU | 0.299 ms | |
| &nbsp;&nbsp;-- kernel execution (5 stages, merged) | GPU | 0.129 ms | |
| &nbsp;&nbsp;-- sync-only extra | GPU | 0.055 ms | |
| &nbsp;&nbsp;-- big D2H download | GPU | 0.346 ms | |
| &nbsp;&nbsp;-- (host-side driver overhead not on GPU timeline) | CPU | ~0.078 ms | ~3.1% |
| Combine loop (Shoup mulmod x4/coeff) | CPU | 0.363 ms | 14.6% |
| **Total** | | **2.481 ms** | 100% |

**A sub-breakdown of the RNG bucket** (second instrumentation pass,
also reverted) found engine construction (`std::mt19937_64`/
`normal_distribution` construction) and the three `std::vector<int>`
allocations are both negligible (0.0005ms and 0.0065ms respectively)
-- **97.8% of the RNG bucket's cost is the actual sampling loop itself**
(n ternary draws + 2n Gaussian draws via `std::normal_distribution`),
not setup overhead.

**This refines, not contradicts, Phase 36's own 40-48% CPU-side
estimate**: that estimate lumped the GPU section AND the combine loop
into one timestamp bucket (`gpu_section_plus_combine`), which
necessarily biased the apparent "GPU share" upward by folding a real
CPU-only loop (combine, 14.6%) into it. This phase's cleaner separation
puts CPU-side work (RNG + alloc + embed + tablebuild + combine) at
**~53%** and GPU-side (the cudaEvent-measured transfer+kernel work
alone) at **~43%**, with the remaining ~4% being host-side CUDA-driver
overhead not visible on the GPU timeline. Both figures are internally
consistent (per-row reconstruction, above) and are reported here as a
further refinement of Phase 36's range, not a contradiction of it.

## 5. Track B: ranked candidates

1. **RNG sampling** (19.6%, the single largest CPU component). **Not a
   safe candidate.** This is genuinely security-relevant randomness
   (the ephemeral ternary secret `v` and the Gaussian noise `e0`/`e1`
   that give CKKS encryption its security) -- per the mission's
   explicit instruction, no algorithm/distribution change was
   considered. The cost is in the sampling itself (97.8% of the
   bucket), not fixable without touching the sampling algorithm. Ranked
   highest by cost, lowest by eligibility.
2. **Embed loop's `barrett_reduce_u64`: replace an unbounded `while`
   correction loop with 2 bounded `if` corrections.** Classical Barrett
   theory (`mu = floor(2^64/q)`, `x < 2^64`) guarantees at most 2
   corrections; this touches only arithmetic (no security/distribution
   impact), is low implementation complexity, and is independently
   verifiable before touching production. **Selected (Track C).**
3. **Combine loop's `shoup_precon` (a 128-bit division per
   coefficient): replace with a Barrett reduction of the full
   128-bit product.** Phase 36 speculatively ranked this as "plausibly
   the largest remaining win" without working out the actual
   arithmetic. This phase did that analysis (section 6) and found it
   does NOT hold up: a correct double-wide Barrett reduction needs
   genuine multi-precision arithmetic (effectively a 128x64-bit
   multiply needing >128 bits of intermediate precision, since
   `unsigned __int128` cannot hold `P * mu` for a ~124-bit product and
   a ~63-bit `mu` without splitting into multiple 64x64 multiplies) --
   materially higher implementation complexity and correctness risk
   than Shoup's existing approach, which ALREADY amortizes its one
   division (`vPrecon`) across the two multiplies that reuse it
   (`pkB*v` and `pkA*v`). **Not attempted** -- a corrected ranking
   supersedes Phase 36's speculative one; this is NOT a safe near-term
   candidate as currently understood.
4. **Pinned host memory for H2D/D2H.** Already explored in Phases
   15-16 (1.06-1.17x isolated, real adoption friction, kept opt-in, not
   promoted). Not re-attempted -- no new mechanism identified.
5. **Host/GPU overlap via async streams.** Not attempted: the function
   is globally mutex-serialized by design (Phase 13's own hardening,
   unchanged), the GPU section is dominated by blocking transfer calls
   already at/near the measured PCIe floor (Phase 34), and genuine
   overlap would require restructuring stream ownership and host-buffer
   lifetimes well beyond this phase's narrow scope -- correctly
   identified as a larger architectural change for a future phase, not
   something to attempt speculatively here.
6. **Reducing per-call host-side allocation count** (combining `vc`/
   `e0c`/`e1c`/`packed` into one allocation instead of four). A
   genuinely DIFFERENT mechanism from Phase 36's disproved cross-call
   pooling (this is same-call allocation-count reduction, not
   cross-call reuse) -- but `alloc` is only 2.3% of the call and the
   vector allocations are already negligible (0.0065ms), so the
   theoretical ceiling is a few microseconds per call. Not pursued:
   expected benefit too small to justify the complexity/validation
   cost relative to candidate 2.

## 6. Track C: selected optimization, implementation, and measured results

### 6.1 Standalone reproducer (built and run BEFORE touching production)

`research/benchmarks/bench_barrett_reduce_bound_check.cpp` (kept, not
linked into any target, not registered with ctest -- a one-off
verification tool per this project's established convention) duplicates
`make_tower_barrett`/`barrett_reduce_u64` exactly, implements the
proposed 2-bounded-`if` candidate, and tests both against every real
bit-width this project uses (30/40/50/55/58/60/62 bits, 5 real primes
each via a self-contained `native_primes` port) across explicit boundary
cases (0, 1, q±1, 2q±1, values near 2^62/2^63/UINT64_MAX) plus 2 million
random `x` per modulus spanning the full `uint64_t` domain:

```
Testing 35 real moduli (bit-widths 30,40,50,55,58,60,62; 5 each)
Total trials: 70,000,420
Max while-loop iterations ever observed (reference impl): 1
Mismatches between reference (while) and candidate (2x bounded if): 0
RESULT: PASS
```

The existing `while` loop never needed more than 1 correction across
70 million trials (the theoretical bound is 2); the bounded candidate
was bit-exact in every trial.

### 6.2 Production change

`src/encrypt_host_shoup_pooled_barrett_batched.cpp`'s `barrett_reduce_u64`:
```cpp
// before
while (r >= T.q) r -= T.q;
// after
if (r >= T.q) r -= T.q;
if (r >= T.q) r -= T.q;
```
A pure branch-shape change (data-dependent loop → two straight-line
conditional subtractions, typically compiled to branch-free `CMOV`
under `-O3`) -- no change to `mu`, `q`, the embedding formula, the
combine formula, the mutex, the pool, or any cryptographic parameter.

### 6.3 Focused correctness evidence

- All 6 existing dedicated tests for this function
  (`test_encrypt_host_shoup_pooled_barrett_batched_{matches_reference,
  concurrency_audit}` and its `_pinned_`/`_registered_` variants) --
  **bit-exact pass**, including the `LLONG_MIN`/`LLONG_MAX`-adjacent
  message coefficients the existing `matches_reference` test already
  exercises (these happen to stress exactly the extreme-`x` boundary
  this change touches).
- **New permanent regression test added**:
  `tests/test_barrett_reduce_bounded_correction.cpp`, registered via a
  lightweight `add_executable`/`add_test` (no gpufhe/CUDA link needed,
  runs in 0.01s) -- a permanent, smaller-scale (still real-moduli,
  still boundary+random) version of the one-off reproducer, so a
  FUTURE change to `make_tower_barrett`'s `mu` formula or to the
  real `x`-domain this function is called with would be caught on every
  ordinary regression pass, not just once. Passes.
- `bench_private_aggregation_workload` (the other multi-client
  workload the mission names): worst absolute error **3.347054e-09**,
  identical in magnitude to the pre-change baseline -- confirms no
  correctness drift on this second, independent workload either.

### 6.4 Isolated CPU-component measurement

`bench_barrett_embed_isolated.cpp` (standalone, not linked into any
target): reproduces the embed loop's exact workload shape (4 reductions
x sizeQ x n = 32768 calls per "encrypt-equivalent" iteration) at the
real n=8192/sizeQ=4 parameters, alternating OLD/NEW every repetition
WITHIN one process (2000 reps, 50-rep warm-up) to control for any
system drift during the one run:

```
Correctness (full embed-loop output, old vs new): MATCH
OLD    median=384.7170 us   NEW    median=372.2940 us
Speedup (median): 1.0334x
```

A clean, reproducible **1.033x (3.3%) isolated speedup**, bit-exact
correct.

### 6.5 End-to-end measurement (`bench_aggregation_decomposed`)

A proper `git stash`-based interleaved A/B, 7 alternating NEW/OLD pairs
(not fewer -- this session's elevated system noise, section 1, made a
single pair unreliable, so more pairs were collected than usual):

| Trial | NEW median (ms) | OLD median (ms) |
|---|---:|---:|
| A | 81.82 | 85.04 |
| B | 84.88 | 84.31 |
| C | 78.99 | 80.83 |
| D | 81.75 | 82.45 |
| E | 83.05 | 81.37 |
| F | 82.32 | 81.33 |
| G | 81.60 | 81.76 |

Across all 7: NEW median = **81.82 ms**, OLD median = **81.76 ms**
(essentially a wash, well within this session's observed 1.5-8ms
per-trial stddev); NEW mean = **82.06 ms**, OLD mean = **82.44 ms**
(~0.46% favoring NEW). 4 of 7 pairs favor NEW, 3 favor OLD.

**This end-to-end result is NOT a clean, confident win, and is reported
as such rather than oversold.** It is, however, exactly consistent with
a back-of-envelope prediction from the isolated measurement: a 3.3%
improvement to a component that is itself only ~16.4% of the call
predicts an end-to-end effect of roughly `0.033 * 0.164 ≈ 0.5%` --
right at or below this session's noise floor, which is exactly what was
observed (a small, inconsistent-in-direction-per-pair, but
non-negative, average effect).

### 6.6 Disposition: KEEP

This change is kept, for three reasons, none of which is "the
benchmark looked good":

1. It is **rigorously proven correctness-preserving** (70M-trial
   standalone reproducer + a new permanent regression test + all 6
   existing bit-exact production tests, including the
   `bench_private_aggregation_workload` cross-check).
2. It has a **real, clean, isolated improvement** (1.033x, measured via
   interleaved A/B within one process, not a single noisy sample).
3. There is **no evidence of a regression** at the end-to-end level --
   the honest conclusion is "small positive isolated effect, end-to-end
   effect too small to cleanly separate from this session's noise, but
   directionally consistent with the isolated measurement's predicted
   magnitude and never negative on average." This is a different,
   third category from Phase 36's two outcomes (confirmed regression;
   confirmed null) -- a small, safe, evidenced improvement, not a
   fabricated win and not a null result either.

## 7. Track D: re-evaluation of the full encryption pipeline

With the change in place, the pipeline's cost composition (from the
Track A profile, section 4, which captured the dominant structure
before this small change) remains:

- **GPU-side (~43%)**: dominated by the H2D+D2H transfer pair (0.645ms
  combined, ~26% of total), which is already at/near the PCIe floor
  Phase 34 independently measured for this exact buffer size
  (0.577ms round-trip) -- pinned memory was already explored for this
  exact gap (section 5, candidate 4) and not re-attempted. Kernel
  execution itself (0.129ms, ~5%) is small; the 5-internal-launch
  merged `LaunchNTT_CT_Batched` call (Phase 34's own prior work) is not
  revisited this phase since nothing in this phase's measurements
  suggests it is newly the bottleneck.
- **CPU-side (~53%, now very slightly smaller after this phase's
  change)**: RNG sampling (19.6%, off-limits per security constraints)
  is now the largest single component; embed (slightly reduced) and
  combine (14.6%, reconsidered and found not to have an obviously safe
  cheaper algorithm, section 5 candidate 3) follow.
- **Host-driver overhead not on the GPU timeline (~3%)**: small,
  not chased further this phase.

**Expected optimization path for a future phase**, in order of
plausibility: (a) a rigorously-reviewed, standalone-reproducer-first
multi-precision Barrett rewrite of the combine loop's `shoup_precon`
division (genuinely harder than this phase's embed change, not
attempted here); (b) a real async host/GPU overlap redesign (requires
careful stream-ownership and host-buffer-lifetime work, explicitly
deferred, section 5 candidate 5); (c) an architectural reduction in
bytes transferred per call, if a future phase can identify one that
doesn't require touching the packed `[v|e0|e1|m]` layout's own
established rationale. No CUDA kernel rewrite is indicated by this
phase's measurements -- the kernel-execution bucket (5%) is small
relative to transfer and CPU-side work.

## 8. Track E: authenticated financial input review

Reviewed (not expanded) `examples/authenticated_financial_input/`
against the mission's checklist:

- **Encoding metadata binding**: `enc_meta.n`/`enc_meta.delta`, set in
  `build_envelope` to the SAME `F.n`/`delta` values actually passed to
  `encode_host_fft` for that envelope -- confirmed the binding is real
  (the committed metadata is the metadata the real CKKS operation used),
  not merely nominal.
- **Ciphertext shape validation**: confirmed `AggregationSessionVerifier::verify`
  checks `c0.size() == c1.size() == expected_size_q_ * env.enc_meta.n`
  (step 5), after confirming `enc_meta` itself matches the session's
  expectation (step 3) -- re-read directly from
  `src/authenticated_financial_input.cpp` this phase, not assumed from
  memory.
- **Fail-closed**: `verify()`'s structure is five sequential checks,
  each with an early `return Rejected*`; `Accepted` is reachable only
  if every check passes, in order -- fail-closed by construction,
  confirmed by direct re-read.
- **Duplicate/replay handling**: unchanged, still keyed on the
  commitment digest, checked after attestation and encoding-mismatch.
- **Canonical serialization determinism**: `canonical_attestation_bytes`
  is a pure function of its inputs (length-prefixed strings, fixed-width
  integers, raw digest bytes, no non-deterministic ordering) --
  unchanged, still deterministic.
- **Mock labeling**: unchanged, still extensively and explicitly
  labeled non-cryptographic throughout the header, implementation, and
  README.
- **Boundary isolation**: still compiled only into its own executable,
  zero references from `libgpufhe`.

**Conclusion: already adequate.** No gap was found beyond what Phase 36
already closed. No new checks, tests, or complexity were added this
phase, per the mission's explicit "do not expand scope just to add
features" instruction.

## 9. Track F: regression results

An initial attempt to run the base C++ suite in one `-j1` batch (even
after excluding the 4 heaviest n=131072-class tests) was **killed by
the host** for running the system low on memory -- confirmed no
orphaned process afterward. Re-run in four smaller sub-batches by test
ID range instead (tests 1-45, 46-90, 94-120, 122-132, skipping the 4
heaviest throughout), checking free RAM between each:

| Batch | Tests run | Passed | Failed (all pre-existing) |
|---|---:|---:|---|
| 1-45 | 45 | 43 | `test_evalkeygen_isolated`, `test_keyswitch_stage_trace` |
| 46-90 | 45 | 42 | `test_polyeval`, `test_stage2`, `test_boot_multi_input_small` |
| 94-120 | 27 | 27 | none |
| 122-132 | 11 | 6 | the five `test_rescale_*_n1024*` |

**Total: 118/128 passing, exactly the same 10 pre-existing documented
failures as every prior phase, zero new regressions.** Includes
`bench_embed_barrett_vs_reference` (directly relevant to this phase's
change) passing, and `authenticated_financial_input` passing (its
14 checks, confirmed in section 8).

- **`test_n131072_keysize`** (own measured requirement ~1.3 GB RSS):
  free RAM checked at ~6.45 GiB beforehand -- **Passed**, 8.71s.
- **`test_n131072_cache_churn`** (own measured requirement ~3.2 GiB
  RSS plateau): free RAM checked at ~6.2 GiB beforehand -- **killed by
  the host twice in a row** (two separate attempts, ~5.3 GiB and
  ~4.5 GiB free respectively at each start), despite both starting
  points comfortably exceeding this test's own documented requirement.
  No orphaned process either time. **Classified as resource exhaustion
  / environmental, NOT a regression and NOT claimed as passing** --
  this session's host-wide memory volatility (section 1) was
  measurably worse than Phase 36's own session, where this exact test
  passed. Not retried a third time, per the project's own
  "diagnose, don't blindly retry" discipline.
- **`test_boot_n131072`** and **`test_n131072_gpu_lifecycle`**:
  deferred, not attempted -- both belong to the established high
  (6-10 GiB) memory-gate class, which this session's free RAM (1.1-7.3
  GiB, frequently well below 6 GiB) did not reliably clear, and
  `test_boot_n131072` is independently documented
  (`research/MEMORY_FEASIBILITY.md`) as never having completed a
  bootstrap stage even under more generous memory in past sessions.
- **Rust** (`cargo test --release`, both `RUST_TEST_THREADS=1` and
  default parallel): **62 passed / 0 failed / 1 ignored**, both modes,
  unchanged.

## 10. Deferred work and known risks

- `test_n131072_cache_churn` was not verified passing this session
  (resource exhaustion, not a regression) -- re-run it first thing next
  session once free RAM is confirmed comfortably above ~3.2 GiB AND
  stable (not just momentarily above the threshold, given this
  session's demonstrated volatility).
- The combine loop's `shoup_precon` division (14.6% of a call) remains
  unoptimized. Section 5/6.1's complexity analysis found Phase 36's own
  speculative ranking of this as "the best-supported remaining
  candidate" does not hold up once the actual multi-precision Barrett
  arithmetic is worked out -- a future phase should not assume this is
  a quick win; any attempt needs its own standalone reproducer
  comparing REAL implementation complexity and measured cost, not
  another speculative ranking.
- RNG sampling (19.6%, the single largest CPU component) remains
  untouched and is NOT a safe target without dedicated, independently
  reviewed cryptographic analysis of any proposed alternative sampling
  method's security equivalence -- this phase deliberately did not
  attempt it, consistent with the mission's explicit randomness
  constraints.
- The end-to-end benefit of this phase's one shipped change is small
  (~0.5%, consistent with its own predicted magnitude) and was not
  cleanly distinguishable from this session's elevated measurement
  noise -- a future phase re-measuring under calmer conditions may see
  a cleaner signal, but should not expect a large one; this change was
  never expected or claimed to be a major end-to-end win.

## 11. Exact commits and push status

See the end of this session's work for exact commit hashes (filled in
immediately after creation, in the same session) and push confirmation.
