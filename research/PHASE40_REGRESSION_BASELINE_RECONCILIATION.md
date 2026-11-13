# Phase 40, Track A: Regression Baseline Reconciliation

## 1. The inconsistency, exactly stated

Phases 37, 38, and 39 each report a headline regression line of the
shape `"<passed>/<discovered> passing, same 10 pre-existing failures"`
(Phase 37: `118/128`; Phase 38 and Phase 39: `130/134`). Read literally,
`130/134 passing` plus `10 pre-existing failures` implies `130 + 10 =
140` accounted-for tests against a stated total of only `134` --
impossible. That is the exact arithmetic inconsistency this god prompt
flagged, and it is real: verified below by reading the actual per-batch
tables in both source reports, not by assuming either number was typo'd.

## 2. Root cause: "passing" was written as "run", not "passed"

Both Phase 38's and Phase 39's own batch tables (reproduced below from
source, byte-identical between the two phases since no tests were added
or removed in Phase 39) are internally consistent once `Passed + Failed
= Tests run` is checked per row:

| Batch (test-ID range) | Tests run | Passed | Failed |
|---|---:|---:|---:|
| 1-45 | 45 | 43 | 2 (`test_evalkeygen_isolated`, `test_keyswitch_stage_trace`) |
| 46-91 | 46 | 43 | 3 (`test_polyeval`, `test_stage2`, `test_boot_multi_input_small`) |
| 95-122 | 28 | 28 | 0 |
| 124-134 | 11 | 6 | 5 (`test_rescale_n1024`, `test_rescale_scalar_diff_n1024`, `test_rescale_vector_diff_n1024`, `test_rescale_double_call_n1024`, `test_rescale_n1024_matched_scale`) |
| **Sum** | **130** | **120** | **10** |

`120 + 10 = 130` -- the batch table itself is arithmetically correct.
The headline sentence in both reports ("Total: 130/134 passing") used
**130 (tests run in these four batches)** as the numerator where it
should have used **120 (tests that actually passed)**. The `134` in
the same sentence is a *different* quantity again: the full `ctest`
discovery count, which is 4 tests larger than the 130 run in these
batches because four large-n tests are deliberately excluded from the
batch ranges and handled separately (see the gaps: 91->95 skips test
IDs 92-94, and 122->124 skips test ID 123 -- 4 skipped IDs, matching
the four large-n tests named individually below each table in both
reports). So the sentence conflated two different pairs of numbers
("120 passed out of 130 run" and "130 run out of 134 discovered") into
one ratio that is neither.

**This is not a one-off typo.** Phase 37's own equivalent sentence
("118/128 passing") used the *same* two-number convention but
happened to be arithmetically self-consistent by coincidence, because
Phase 37 phrased its numerator correctly as the passed count: its own
batch table sums to `43+42+27+6 = 118` passed out of `45+45+27+11 =
128` run, and its denominator (128) *was* the run total, not a larger
discovered total (Phase 37 had only 4 large-n tests held out of 132
discovered, and wrote 128 as "run", consistently). Phase 38 then
introduced the inconsistent phrasing when it switched the sentence to
read "130/134" (using the *run* count as if it were the *passed*
count, and the *discovered* count as the denominator instead of the
*run* count), and Phase 39 copied that same sentence verbatim. Verified
by direct comparison of the three reports' source text, not inferred.

## 3. Corrected combined picture, verified from source

`ctest` discovery count growth, confirmed from each phase's own batch
tables plus the explicit "skipping the N heaviest" / large-n call-outs:

| Phase | Discovered (ctest -N) | Run in 4 batches | Passed | Failed (pre-existing) | Large-n tests (handled separately, not in the 4 batches) |
|---|---:|---:|---:|---:|---|
| 37 | 132 | 128 | 118 | 10 | 4: `test_n131072_keysize`, `test_n131072_cache_churn`, `test_boot_n131072`, `test_n131072_gpu_lifecycle` |
| 38 | 134 (+2: Phase 38 added `test_encrypt_multiclient_matches_sequential` and `bench_aggregation_multiclient_batched` as ctest-registered) | 130 | 120 | 10 | same 4 |
| 39 | 134 (unchanged -- CMakeLists.txt not modified this phase) | 130 | 120 | 10 | same 4 |

Large-n test outcomes by phase (never counted as pass unless actually
observed completing; resource exhaustion is its own category, not a
pass or a fail):

| Phase | `test_n131072_keysize` | `test_n131072_cache_churn` | `test_boot_n131072` | `test_n131072_gpu_lifecycle` |
|---|---|---|---|---|
| 37 | Passed (8.71s) | **Resource-exhausted** (killed by host twice, not retried) | Deferred (not run) | Deferred (not run) |
| 38 | Passed (8.51s) | Passed (182.74s) | Deferred (not run) | Deferred (not run) |
| 39 | Passed (8.84s, 9.00s, two runs) | **Resource-exhausted** (killed by host twice, not retried) | Deferred (not run) | Deferred (not run) |

**Honest totals out of 134 discovered, each phase (the number the
headline sentence should have reported):**

| Phase | Verified passing | Pre-existing failing | Resource-exhausted | Deferred / not run |
|---|---:|---:|---:|---:|
| 37 | 118 (batches) + 1 (`keysize`) = **119/132** | 10 | 1 (`cache_churn`) | 2 (`boot_n131072`, `gpu_lifecycle`) |
| 38 | 120 (batches) + 2 (`keysize`, `cache_churn`) = **122/134** | 10 | 0 | 2 |
| 39 | 120 (batches) + 1 (`keysize`) = **121/134** | 10 | 1 (`cache_churn`) | 2 |

None of this changes any phase's underlying finding: the same 10
named pre-existing failures recur in every phase, bit-exact, and zero
new regression was ever introduced. What was wrong was only the
*summary sentence's arithmetic*, not the underlying test results. This
document corrects the sentence, not the history -- the original
"130/134 passing" phrasing is left intact at its source in both the
Phase 38 and Phase 39 reports (not erased), with a pointer added to
this document for the corrected reading.

## 4. The 10 pre-existing failures, named exactly (unchanged across all three phases)

1. `test_evalkeygen_isolated`
2. `test_keyswitch_stage_trace`
3. `test_polyeval`
4. `test_stage2`
5. `test_boot_multi_input_small`
6. `test_rescale_n1024`
7. `test_rescale_scalar_diff_n1024`
8. `test_rescale_vector_diff_n1024`
9. `test_rescale_double_call_n1024`
10. `test_rescale_n1024_matched_scale`

(Section 5 below records this session's own fresh classification check
on each of these, since Track A also requires *this session* to
re-verify, not just cite history.)

## 5. This session's own fresh, independently-run numbers

Rebuilt from current source at `HEAD=05ada33` plus this session's own
changes (one new test, `tests/test_shoup_moddown_rescale_boundary.cpp`,
registered as test #40; and the Track E `ks_moddown` batching change in
`kernels/moddown.cu` / `include/moddown.h` / `src/keyswitch_resident.cpp`,
detailed in `PHASE40_PERFORMANCE_AND_NEXT_BOTTLENECK.md`). The build
itself was blocked for 19+ minutes by an environmental issue (WSL's
Windows-PATH import colliding with a full CMake reconfigure, not a
code problem) -- see `PHASE40_WSL_PERFORMANCE_DIAGNOSIS.md` for the
diagnosis and fix. All numbers below are from the rebuilt, fixed
environment, run AFTER the Track E change (the full suite was also run
once immediately before it, with identical results on every test this
change could possibly affect -- see that document's section on
correctness verification).

`ctest -N` -> **135 discovered** (134 carried forward + 1 new).

| Batch (test-ID range) | Tests run | Passed | Failed |
|---|---:|---:|---:|
| 1-46 | 46 | 44 | 2 (`test_evalkeygen_isolated`, `test_keyswitch_stage_trace`) |
| 47-92 | 46 | 43 | 3 (`test_polyeval`, `test_stage2`, `test_boot_multi_input_small`) |
| 96-123 | 28 | 28 | 0 |
| 125-135 | 11 | 6 | 5 (the five `test_rescale_*_n1024*`) |
| **Sum** | **131** | **121** | **10** |

Every failure independently re-verified THIS session by running the
binary directly (not just trusting the name match against history):
`test_rescale_n1024` throws `std::invalid_argument` from
`ks_context_create`'s own `P >= Q_part_max` guard (a deliberately
invalid parameter configuration, by design); `test_keyswitch_stage_trace`
reports `worst=49.99% of q` relin-identity error, the same signature
documented in `EXPERIMENT_LEDGER.md` entry #42. **Confirmed pre-existing
by direct inspection, not assumed from the name.**

Large-n tests (held out of the four batches above, run individually,
each gated on free host RAM first): `test_n131072_keysize` **passed**
(8.49-9.40s across repeat runs); `test_n131072_cache_churn` **passed**
(182-189s across repeat runs -- unlike Phase 37 and Phase 39, where
this exact test was killed by host memory pressure); `test_n131072_gpu_lifecycle`
**passed** (217.71s, run incidentally as part of a batch this session
and not reverted from); `test_boot_n131072` **deliberately not run**
(same standing classification as every prior phase: this test has
never completed a single bootstrap stage in this project's history
regardless of memory available, and this session's one accidental
inclusion of it via an overly broad `ctest -R` substring match was
caught via a host free-RAM check showing a drop to 1.23 GiB and killed
immediately -- see `PHASE40_WSL_PERFORMANCE_DIAGNOSIS.md` is NOT the
right pointer for this; it is recorded here and in
`PHASE40_PERFORMANCE_AND_NEXT_BOTTLENECK.md`'s correctness section).

**Honest total out of 135 discovered this session: 121 (batches) + 3
(keysize, cache_churn, gpu_lifecycle) = 124 verified passing, 10
pre-existing failing (named and independently re-verified), 1
deliberately not run (`test_boot_n131072`).** `124+10+1=135`. This is
the best result of any phase to date (every prior phase deferred or
lost at least one more large-n test to resource exhaustion or
deliberate non-attempt) -- not because anything was fixed, but because
this session's host memory happened to be sufficient when
`cache_churn` and `gpu_lifecycle` ran, and because `gpu_lifecycle`'s
inclusion in a batch was incidental, not a deliberate new policy to
always attempt it.

**Rust**: `cargo test --release`, both default-parallel and
`RUST_TEST_THREADS=1` modes, run twice (once before the Track E
change, once after, both forcing a genuine rebuild via
`touch crates/ckks-cuda/build.rs` first): **62 passed, 0 failed, 1
ignored, identically in all four runs.** (An earlier `tail -40`-truncated
capture of one of these runs misleadingly showed only 58 -- the
ckks-core crate's own 4-test block had scrolled out of the captured
window; re-run capturing the full untruncated log confirmed 62,
matching history exactly.)
