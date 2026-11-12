# Phase 39: FHE textbook integration, RNS audit, K=32 investigation, and a verified RNS arithmetic improvement

## 1. Starting commit and environment

Starting commit: `d1dc2be` (Phase 38's final commit), confirmed matching
`origin/main` exactly (0 ahead / 0 behind) at Phase 39 start.

Hardware/software: RTX 2060 Max-Q, 6144 MiB VRAM (5942 MiB free at
session start), compute capability 7.5, driver 595.97, CUDA 13.2,
Release `-O3`. cmake 3.28.3. Rust/cargo 1.94.0. gcc 13.3.0, Ubuntu
24.04.4. 16 logical CPU cores. Windows host RAM started at ~1.25 GiB
free (very low) and fluctuated ~1.9-6.7 GiB through the session.

## 2. Phase 38 baseline (re-verified from source and the report, not assumed)

| Measurement | Phase 38 result |
|---|---:|
| Batched throughput improvement, K ≤ 16 | 1.07x-1.38x |
| Improvement around K=32 | 1.02x-1.04x |
| Sequential execution scaling | Approximately linear with K |
| VRAM scaling | Flat (pooled, reused buffers) |
| C++ tests | 130/134, same 10 documented failures |
| Rust (both modes) | 62 passed / 0 failed / 1 ignored |
| Both n=131072 tests | Passed |

Re-confirmed this phase (section 9): C++ matches exactly (same 130/134);
Rust matches after a fix this phase itself needed to apply (section 7.3).

## 3. Reference resources: access and scope

| Resource | File | Local path | Committed to repo? |
|---|---|---|---|
| A: Ronny Ko, *The Beginner's Textbook for Fully Homomorphic Encryption*, arXiv:2503.05136v26 (259 pp.) | `2503.05136v26.pdf` | `~/fhe-reference-material/2503.05136v26.pdf` | No -- copyrighted arXiv preprint, no confirmed redistribution license |
| B: Craig Gentry, *A Fully Homomorphic Encryption Scheme*, Stanford PhD dissertation, Sept. 2009 (209 pp.) | `craig-thesis.pdf` | `~/fhe-reference-material/craig-thesis.pdf` | No, same reasoning |
| C: TLShare (already reviewed Phase 35) | `3-3-1.pdf` | `~/fhe-reference-material/3-3-1.pdf` | No, same reasoning |

All three were found in `~/Downloads/` (the user's own machine, not
pre-placed in a WSL-accessible reference directory) and copied into a
dedicated non-repository directory, per the mission's own suggested
convention, before any reading began.

Sections actually studied (not merely headings), and the resulting
mapping, are in the companion document
`research/PHASE39_FHE_TEXTBOOK_IMPLEMENTATION_MAP.md`. Summary: A-14
(CRT/RNS) and A-16 (NTT) read in full; D-3 (the complete CKKS chapter,
pp. 176-199) read in full; D-5 (RNS-variant techniques, pp. 226-248)
read in full; Gentry's Abstract, full Introduction (pp. 1-22), and
Chapter 12 (Performance and Optimizations, pp. 115-124, per the
mission's own emphasis) read in full. Gentry's Chapters 4/7/10/11 were
confirmed, via their own opening pages, to formalize exactly what the
Introduction's informal sections 1.3-1.6 already explain -- not
re-read page-by-page in formal notation, since the mission explicitly
prohibits transplanting any of that scheme's specific machinery and the
informal treatment already suffices to extract general lessons (section
8). TLShare was not re-reviewed beyond the input-authenticity
cross-check in section 10.

## 4. Track A: the mapping's headline finding

The full table is in the companion map document. The single most
load-bearing result: `kernels/basis_convert.cu`'s `ApproxSwitchCRTBasis`
is confirmed to be exactly the textbook's `FastBConv` (D-5.1), and it is
genuinely, correctly APPROXIMATE -- it carries the `|u·q|_b` noise the
book describes (`|u| ≤ sizeQ/2+1`), with no `SmallMont`-style correction
applied, matching OpenFHE's own reference hybrid key-switching (whose
constants this file explicitly borrows). This gives a concrete, sourced
explanation for WHY this project's own established `P >= Q_part_max`
invariant must hold: the auxiliary modulus `P` must be large enough,
relative to the largest key-switch partition, to absorb exactly this
approximation's noise. No `SmallMont` or `FastBConvEx` equivalent exists
anywhere in this repository, and neither was added -- `SmallMont` only
matters here if it justified shrinking `P` (a security-sensitive change
out of scope), and this project's CKKS multiplication never needs
`FastBConvEx` in the first place (confirmed against D-5.10.5's own
stated reason: CKKS skips `ModRaise`/constant-multiply/`FastBConvEx`
entirely, needing only a modulus switch). A second, honest finding: NO
separate `ModDrop` operation exists in this codebase at all (confirmed
by a repository-wide search) -- only the full rescale (`rescale_resident`)
exists, which always changes the plaintext scale as well as the
modulus, unlike the book's own D-3.7/D-5.4 distinct, scale-preserving
ModDrop. Not a bug (this project's shallow, non-bootstrapping workloads
have never needed the distinction), but recorded per the mission's own
instruction to note what the textbook covers that the repository does
not implement.

## 5. Track B: auditing the Phase 38 batched implementation

Re-read `src/encrypt_host_shoup_pooled_barrett_batched_multiclient.cpp`
and `kernels/ntt_ct.cu`'s `ct_stage_b`/`ct_shared_tail_b` directly.
Confirmed:

- Shared across clients: `pkA`, `pkB`, `mod`, `root`, the per-tower NTT
  tables (`fr`/`fp`, cached by `enc_dev_tab_multi`). Duplicated per
  client: the per-tower table POINTERS in `d_fr`/`d_fp`/`d_mod` (4*sizeQ
  entries repeated M times, matching Phase 34's own already-proven
  pattern one level further). Independent per client: `v`/`e0`/`e1`
  randomness (own seed) and plaintext.
- `blockIdx.y` indexes `towers = 4*sizeQ*M` directly; each tower reads/
  writes its own disjoint `n`-length segment. At the real production
  `n=8192, sizeQ=4`: `towers` = 16, 32, 64, ..., 1024 for K=1..64 --
  nowhere near CUDA's 65535 grid-dimension-y limit at any tested K.
- The grid's x-dimension (`bl`, early-stage kernel) and span count
  (shared-tail kernel) are both independent of `K` entirely -- only
  grid.y scales with K. No code path in the batched function's own
  logic depends on K crossing any particular value; there is no
  branch, no fixed-size buffer, and no K-dependent constant anywhere in
  the function that could produce a K=32-specific effect.
- Non-power-of-two M (tested at M=17 in Phase 38's own correctness
  suite) produces bit-exact results identical to power-of-two M --
  confirmed again this phase by re-running
  `test_encrypt_multiclient_matches_sequential` (still passing,
  unmodified).

**Conclusion: nothing in the batched implementation's own design
depends on K=32 specifically.** This motivated Track C's actual
reproduction, below, rather than guessing at an occupancy or
register-pressure explanation from code reading alone.

## 6. Track C: the K=32 investigation (priority, as instructed)

Built `bench/bench_multiclient_k32_investigation.cpp` (new, kept as a
one-off diagnostic, not registered with ctest), extending Phase 38's own
K-sweep with K = {1,2,4,8,16,24,31,32,33,48,64}, interleaved SEQ/BATCH
every repetition, 16 reps/arm/K (first rep dropped as cold), reporting
per-client cost explicitly (not just total time) for both arms.

**The decisive observation**: the SEQUENTIAL arm's per-client cost --
which calls the UNCHANGED, pre-Phase-38 single-client encrypt function
identically at every K, with nothing in its own code depending on K at
all -- itself jumps sharply between K=31 and K=32, and stays elevated
through K=64. Reproduced across two independent runs:

| K | SEQ per-client (ms), run 1 | SEQ per-client (ms), run 2 |
|---:|---:|---:|
| 16 | 2.838 | 2.867 |
| 24 | 2.832 | 2.916 |
| 31 | 3.038 | 2.984 |
| 32 | 3.906 | 3.912 |
| 33 | 3.909 | 3.933 |
| 48 | 3.843 | 3.895 |
| 64 | 3.839 | 3.871 |

**This rules out a batched-kernel-specific cause.** The sequential arm's
grid geometry, occupancy, and register usage are IDENTICAL at every K
(it is the same function call, repeated K times) -- yet the SAME
threshold appears at the SAME K. Whatever causes this must be external
to both encrypt-path implementations.

**Supporting evidence (direct GPU clock sampling during a sustained
sweep)**: `nvidia-smi` queried four times during one full K-sweep run
showed SM clock fluctuating 555-975 MHz throughout -- never once
approaching the card's 2100 MHz boost clock, and never settling at a
single steady value either. This is consistent with (not, on its own,
definitive proof of) a Max-Q mobile GPU's dynamic power-management
algorithm transitioning between clock/power states based on cumulative
sustained load duration within one measurement round, rather than
anything about "32" as a number -- K=32 is simply the client count at
which THIS benchmark's own total per-rep GPU-busy duration happens to
cross whatever threshold this specific hardware's power management uses
on this specific day. This directly extends Phase 34's own earlier,
already-documented GPU-clock-drift finding: there, drift looked like
general noise between isolated and interleaved measurement; here, under
a controlled, interleaved, per-K sweep, it presents as a genuine
discrete step rather than pure noise, once per-rep duration crosses a
threshold.

**An attempt at continuous clock/throttle-reason logging via
`nvidia-smi --loop-ms` across the whole sweep did not succeed this
session** (the logging process could not be started reliably from a
background shell in this environment -- a tooling limitation, not a
measurement failure) -- the four point-sample readings above are the
evidence actually in hand, honestly reported as supporting, not
conclusive, evidence. A future phase with working continuous logging
could confirm this more rigorously.

**Conclusion: the K=32 weakness is not a defect in the batched
encryption implementation.** It is most plausibly a GPU clock/power-state
characteristic of this specific consumer Max-Q laptop, triggered by
cumulative per-round GPU-busy duration, affecting ALL GPU work in that
round equally regardless of how it is organized into kernel launches.
There is no code-level fix available for this -- it is environmental,
not architectural.

## 7. Track D/E: RNS optimization candidates, ranked, and the one implemented

### 7.1 Candidates considered

1. **`ApproxModDown`'s combine kernel** (`kernels/moddown.cu`): a raw
   128-bit `% q` division per coefficient, even though its multiplier
   `s` (`PInvModq[i]`) is a fixed per-tower scalar known before the
   kernel launches -- the exact shape Shoup's precomputed-multiplier
   technique (already used on this project's own encrypt hot path)
   targets. **Selected and implemented.**
2. **`rescale_fuse_kernel`'s combine step** (`kernels/rescale.cu`): the
   SAME shape, with TWO raw divisions per coefficient instead of one --
   and the file's own pre-existing comment literally reads "Shoup is a
   later opt," a direct, unprompted invitation from this project's own
   prior work. **Selected and implemented** (the same conceptual finding
   applied to a second, more heavily-exercised code path -- rescale is
   exercised by many more of this project's own tests than key-switch
   mod-down alone).
3. **A `SmallMont` tightening of `ApproxSwitchCRTBasis`'s own noise
   bound.** Ranked but NOT attempted: the only way this would matter is
   to justify shrinking `P`, a security-sensitive parameter change this
   phase's narrow scope does not justify (per the mission's own explicit
   caution against exactly this kind of transplant).
4. **A `FastBConvEx`-based exact rescale path.** Ranked but NOT
   attempted: this project's CKKS multiplication structurally does not
   need it (section 4), so adding it would be a speculative, unused
   addition, not a real improvement.
5. **The K=32 batching anomaly itself.** Ranked, investigated
   thoroughly (section 6), concluded to have NO code-level fix
   available -- correctly NOT "implemented," since there is nothing
   safe to implement.

### 7.2 Mathematical correctness argument

Both kernels compute `(value * fixed_scalar) mod q` where `value < q`
and `fixed_scalar < q` (a modular difference and a modular inverse,
respectively) -- exactly the domain Shoup's technique requires.
`shoup_mulmod(x,c,precon,q) = (x*c) mod q` for any `x,c < q`, given
`precon = floor(c<<64 / q)`, is a standard, already-proven identity
this project already relies on elsewhere (e.g. the production encrypt
path's own combine loop). Verified, not just asserted: a standalone
reproducer (`research/benchmarks/bench_moddown_shoup_check.cpp`, kept)
compared `shoup_mulmod` against the raw `%` operator across 35 million
trials (35 real moduli spanning every bit-width this project uses --
30 through 62 bits -- boundary values plus 200,000 random `x` per
`(q,s)` pair) -- zero mismatches, before either kernel was touched.

### 7.3 Implementation and a genuinely important catch

Changed `kernels/moddown.cu`'s `moddown_combine_kernel` and
`kernels/rescale.cu`'s `rescale_fuse_kernel` to accept precomputed
`precon` parameter(s) and use `shoup_mulmod` instead of `% q`. Updated
`include/moddown.h`/`include/rescale.h` and every call site: `src/
keyswitch.cpp`, `src/keyswitch_resident.cpp`, `src/ckks_ops.cpp` (2
production call sites across the two kernels), plus `tests/
test_moddown_combine_isolated.cpp`, `tests/test_keyswitch_stage_trace.cpp`,
`tests/test_moddown.cpp`, `tests/test_rescale_resident.cpp`,
`bench/bench_rescale_cpp_vs_rust.cpp` (5 test/bench call sites), each
computing `shoup_precon(s,q)` inline at the call site (the per-call
cost of one extra 128-bit division per tower per rescale/mod-down call
is negligible relative to the per-element division it eliminates, and
avoids touching any context-struct lifecycle for this narrow change).

**After the C++ side was fully updated and all C++ tests passed, running
the Rust regression suite caught a genuine, serious bug this phase would
otherwise have shipped**: `rust/crates/ckks-cuda/src/rescale.rs` has its
OWN separate `extern "C"` FFI declaration for `LaunchRescaleFuse` (Rust's
`build.rs` compiles `kernels/rescale.cu` independently of the C++ CMake
build) -- this declaration still had the OLD 7-parameter signature after
the kernel itself was changed to 9 parameters, and Rust's own
`real_encrypted_evalmult_via_registered_path_matches_expected_product`
test SEGFAULTED (signal 11) on the resulting ABI mismatch. Fixed by
updating `rescale.rs`'s declaration and its one call site
(`RescaleEngine`'s rescale method) to match, computing `s1_precon`/
`s2_precon` via Rust's own native `u128` arithmetic. **This is exactly
why this project's own established discipline insists on running BOTH
regression suites, not just the C++ one, before trusting any change
that touches a function with cross-language callers** -- recorded here
prominently, not glossed over, since it is a genuine near-miss this
phase's own process caught.

### 7.4 Focused correctness evidence (after the fix)

- All C++ tests touching either kernel: `test_moddown` (RNS-residue
  OpenFHE differential), `test_moddown_crt_basis_isolated`,
  `test_moddown_combine_isolated` (production-lambda mirror),
  `test_keyswitch` (full `ks_context_create`-driven key-switch),
  `test_rescale_resident`, `test_evalmult_rescale` (end-to-end
  EvalMult+rescale), `bench_keyswitch_cpp_vs_rust`,
  `bench_rescale_cpp_vs_rust` -- all pass, bit-exact. The 20-test
  rotation/key-switch/hoisting suite (`test_rotate*`, `test_ks_*`,
  `test_hoist_*`, `bench_ks_graph_perf`, `bench_hoist_multi`) -- all
  20/20 pass, including the memory-sensitive `test_hoist_allstages`
  (135.67s, ran cleanly this session).
- `test_keyswitch_stage_trace` still fails -- this is the SAME
  pre-existing, documented failure (a deliberately invalid-configuration
  test exercising the centralized relin-identity noise check, unrelated
  to the arithmetic this phase changed -- confirmed by reading its own
  failure output, which fails at a `~50% of q` noise-tolerance check,
  not at any value this phase's kernels compute).
- Rust: both `RUST_TEST_THREADS=1` and default-parallel modes, 62
  passed / 0 failed / 1 ignored, after the fix in section 7.3.

### 7.5 Performance: isolated, order-controlled, at two scales

A first, naive isolated GPU-only comparison (always OLD-then-NEW within
each repetition, 50-rep warm-up folded into the timed loop) showed an
implausible result -- NEW *slower* than OLD by more than 2x -- which did
not survive scrutiny: re-run with call order alternated every repetition
and 100 reps of dedicated warm-up BEFORE any timed repetition, the
result reversed to something consistent with theory. **This measurement
artifact, and its correction, is recorded explicitly** as a direct,
concrete illustration of why this project's own established interleaved-
A/B discipline exists -- an uncontrolled first measurement would have
led to reverting a correct, beneficial change.

Order-controlled results (median, GPU-event-timed, 3000 reps/kernel,
both scales this project actually uses):

| Kernel | n=8192 (shallow production chain) | n=131072 (this project's own large-ring tests) |
|---|---:|---:|
| ModDown combine | 1.034x | 1.435x |
| Rescale fuse | 1.026x | 1.875x |

**Honest interpretation**: at the project's primary, shallow-chain
production scale (n=8192), both kernels are so small (8192 threads,
~32 blocks on a 30-SM card) that launch overhead and memory latency
dominate -- the arithmetic saving is real but nearly invisible against
that fixed cost, essentially a wash (never a regression, consistently
~1.03x). At n=131072 -- a configuration this project already builds,
tests (`test_n131072_keysize`, `test_n131072_cache_churn`), and cares
about -- there is enough per-launch work for the arithmetic saving to
show through clearly: a genuine 1.44x-1.88x per-kernel speedup. This
phase does not claim a large end-to-end win for the CURRENT shallow,
non-bootstrapping, pure-sum aggregation workload (which never calls
key-switch or rescale at all) -- it claims a real, verified, bit-exact
improvement to two specific RNS primitives that benefits any workload
exercising key-switching or rescaling at larger ring dimensions,
consistent with the mission's own framing ("make the actual CKKS
implementation more correct, efficient, and scalable," not only the one
workload prior phases happened to benchmark).

## 8. Track F: Gentry's Chapter 12 lessons, applied honestly

Reviewed against the actual CKKS call graph, per the mission's explicit
questions:

- **"Are any intermediate values being recomputed unnecessarily?"**
  Yes -- `shoup_precon(s,q)` is now computed once per tower per call
  instead of the equivalent division being paid once PER COEFFICIENT
  (n times) -- directly analogous to Gentry's own Optimization 1
  (Remark 12.3.2: discard/avoid recomputation that provably cannot
  affect correctness, applied there to symmetric-polynomial
  coefficients, applied here to a per-tower modular inverse).
- **"Can a representation be simplified without changing the
  cryptographic construction?"** Yes, and this is exactly what was
  done: the SAME mathematical value (`x*s mod q`) computed via a
  cheaper equivalent representation (multiply-and-shift instead of
  division), with zero change to any ciphertext, key, or noise value --
  directly analogous to Gentry's own Optimization 2 (collapsing a
  representation to the minimum sufficient for the actual result,
  without changing what is computed).
- **"Is there computation that can be shared across clients without
  sharing secret or random material?"** This question was already
  answered by Phase 38's own batched design (section 5) -- re-examined
  this phase and confirmed unchanged: still no secret/random sharing,
  still only public per-tower constants shared.
- **"Can an expensive operation be deferred or avoided under a formally
  proven condition?"** Investigated for `SmallMont`/`FastBConvEx`
  (section 4) and concluded: not safely, for the reasons given there.
  Gentry's own Optimization 3 (Chapter 12) is the direct cautionary
  parallel here -- it reduces computation but EXPLICITLY WEAKENS THE
  SECURITY REDUCTION ("the drawback of this optimization is that its
  security is questionable"), which Gentry records honestly rather than
  hiding. This phase treats that as a direct instruction: do not take
  the analogous step here (shrinking `P` via a tighter noise bound)
  without the same honesty about what it would cost, which is exactly
  why it was not attempted.
- **No idea from Gentry's scheme-specific machinery (ideal lattices,
  squashing, the encrypted-secret-key `τ`, bootstrapping) was
  transplanted** -- none of it is mathematically applicable to a
  ring-based CKKS encryption/key-switching pipeline with no
  bootstrapping at any point in its history, and the mission's own
  explicit list of "do not transplant" items was respected in full.

## 9. Regression (Track G)

Focused tests first (all moddown/rescale/keyswitch/rotation/hoist
tests, section 7.4) -- all pass except the one pre-existing failure.
Then the broad suite, in 4 memory-checked sub-batches:

| Batch | Tests run | Passed | Failed (all pre-existing) |
|---|---:|---:|---|
| 1-45 | 45 | 43 | `test_evalkeygen_isolated`, `test_keyswitch_stage_trace` |
| 46-91 | 46 | 43 | `test_polyeval`, `test_stage2`, `test_boot_multi_input_small` |
| 95-122 | 28 | 28 | none |
| 124-134 | 11 | 6 | the five `test_rescale_*_n1024*` |

**Total: 130/134 passing, exactly the same 10 pre-existing documented
failures as every prior phase, zero new regressions** -- re-confirmed
AFTER both kernel changes were in place together.

- **`test_n131072_keysize`**: free RAM ~5.8-6.2 GiB beforehand --
  **Passed** (8.84s, 9.00s on two separate runs this session).
- **`test_n131072_cache_churn`**: killed by the host for low memory
  TWICE this session (first attempt at ~3.7 GiB free, second retry at
  ~3.7 GiB free after a brief recovery) -- **not verified passing this
  session**, classified as resource exhaustion, not a regression, not
  retried a third time per this project's own "diagnose, don't blindly
  retry" discipline. (This exact test passed cleanly in Phase 38's own
  session; this session's host memory was measurably more constrained
  throughout, starting at ~1.25 GiB free before any work began.)
- **`test_boot_n131072`**, **`test_n131072_gpu_lifecycle`**: deferred
  as usual, consistent with every prior phase.
- **Rust**: 62 passed / 0 failed / 1 ignored, both modes -- AFTER the
  FFI-signature fix in section 7.3 (the suite genuinely caught a real
  defect this phase introduced and then fixed, not a pre-existing one).

No orphaned/runaway process was left behind at any point this session.

## 10. Input-authenticity cross-check (brief, per the mission's own scope limit)

Confirmed, by direct search, that this phase's two kernel changes touch
nothing the `examples/authenticated_financial_input/` boundary depends
on -- that boundary still calls only the single-client
`encrypt_host_shoup_pooled_barrett_batched` function (unchanged this
phase) and has zero references to key-switching, rescaling, or the
Phase 38 multi-client function. TLShare's own conclusions (Phase 35, not
repeated here) are unaffected and not revisited.

## 11. Known limitations and deferred research

- The K=32 anomaly's root cause (section 6) is well-supported but not
  proven beyond reasonable doubt -- the planned continuous clock-logging
  confirmation did not succeed this session (a tooling limitation, not
  a measurement failure). A future phase with working continuous
  `nvidia-smi` logging across one full sweep could close this gap.
- `test_n131072_cache_churn` was not verified passing this session
  (resource exhaustion). Re-run first next session once memory is
  confirmed stable.
- `LaunchRescaleFuse`'s OpenFHE-differential coverage at n=131072 was
  not specifically re-run this phase (this project's existing
  `test_rescale_*` suite only covers n=1024/n=8192 configurations) --
  the isolated kernel-level comparison (section 7.5) is the only
  n=131072 evidence gathered this phase for this specific kernel; a
  future phase adding an n=131072 rescale/key-switch differential test
  would close this gap.
- `SmallMont` and `FastBConvEx` remain unimplemented, by deliberate
  choice (section 4/7.1) -- not a gap to close casually; any future
  attempt needs its own dedicated security review given the direct
  interaction with `P >= Q_part_max`.
- No separate `ModDrop` operation exists in this codebase (section 4) --
  not needed by any current workload, but worth knowing if a future
  phase's workload shape changes.

## 12. Exact commits and push status

Two commits, on top of Phase 38's `d1dc2be`:
- `0baae6b` -- Tracks D/E/F: the Shoup-ified ModDown and RescaleFuse kernels, the standalone reproducer, and the Rust FFI fix.
- `4550cc8` -- this report, the companion implementation map, the ledger entry, the K=32 investigation benchmark, and the NEXT_SESSION.md handoff.

Both verified fast-forward-pushable and pushed to `origin/main`, with
the remote confirmed matching local HEAD afterward.
