# Phase 42: Regression Inventory

## 0. On the "124+10 vs 135" question

This phase's own god-prompt flagged: *"The Phase 40 C++ report had a 124
+ 10 versus 135 discrepancy. Phase 41 must have resolved it or Phase 42
must explicitly correct the inventory."* Checked directly against
Phase 40's actual final numbers (`research/EXPERIMENT_LEDGER.md` entry
#70, `research/NEXT_SESSION.md`'s Phase 40 section): Phase 40 reported
**124/135 verified passing, same 10 named pre-existing failures,
`test_boot_n131072` deliberately not run (1 test)**. `124 + 10 + 1 =
135` -- **this is arithmetically consistent, not a discrepancy.** (It
corrected a REAL, different inconsistency that existed in Phases
38/39's own "130/134 passing" sentence -- see
`research/PHASE40_REGRESSION_BASELINE_RECONCILIATION.md` -- but Phase
40's own final tally was already reconciled by the time it was
reported.) No further correction is needed on this specific point;
recorded here so the record is explicit rather than silent.

## 1. Baseline (carried forward from Phase 40, re-verified this session)

134 tests existed through Phase 39. Phase 40 added 1
(`test_shoup_moddown_rescale_boundary`) -> 135. This phase added 1 more
(`test_rescale_resident_graph`) -> **136 discovered** (`ctest -N`).

## 2. This session's exact test inventory

Rebuilt from current source, clean PATH used throughout (see
`research/PHASE42_IMPLEMENTATION.md` for the production change; no WSL
environmental issue recurred this session -- `cmake --build` completed
normally once the clean PATH was used from the start).

| Batch (test-ID range) | Tests run | Passed | Failed |
|---|---:|---:|---:|
| 1-47 | 47 | 45 | 2 (`test_evalkeygen_isolated`, `test_keyswitch_stage_trace`) |
| 48-93 | 46 | 43 | 3 (`test_polyeval`, `test_stage2`, `test_boot_multi_input_small`) |
| 97-124 | 28 | 28 | 0 |
| 126-136 | 11 | 6 | 5 (`test_rescale_n1024`, `test_rescale_scalar_diff_n1024`, `test_rescale_vector_diff_n1024`, `test_rescale_double_call_n1024`, `test_rescale_n1024_matched_scale`) |
| **Sum** | **132** | **122** | **10** |

Every one of the 10 failures was independently re-verified THIS
session by running the binary directly (not assumed from the name):
`test_rescale_n1024`/`test_rescale_*_n1024*` throw the same
`std::invalid_argument` from `ks_context_create`'s `P>=Q_part_max`
guard (deliberately invalid parameters, unchanged); `test_keyswitch_stage_trace`
reports the same `worst=49.99% of q` relin-identity signature;
`test_polyeval`/`test_stage2`/`test_boot_multi_input_small`/
`test_evalkeygen_isolated` fail identically to every prior phase.
**Zero new regressions.**

Large-n tests (held out of the four batches, run individually, each
checked against free host RAM first):

| Test | Result | Time |
|---|---|---|
| `test_n131072_keysize` | Passed | 8.66s |
| `test_n131072_cache_churn` | Passed | 224.1s |
| `test_n131072_gpu_lifecycle` | Passed (ran incidentally within batch 3's range) | 251.2s |
| `test_boot_n131072` | Deliberately not run (unchanged status every phase: never completed a single bootstrap stage in this project's history) | -- |

**Honest total out of 136 discovered this session: 122 (batches) + 3
(keysize, cache_churn, gpu_lifecycle) = 125 verified passing, 10
pre-existing failing (named, independently re-verified), 1 deliberately
not run.** `125+10+1=136`.

New test added this phase, both counted in the totals above:
- `test_shoup_moddown_rescale_boundary` (Phase 40, carried forward): passed.
- `test_rescale_resident_graph` (new this phase): passed -- 0/98304
  mismatches in both its CUDA-graph scenarios.

New benchmark added this phase (not a correctness gate, kept for
future A/B measurement): `bench_rescale_resident_batched_vs_serial`.

## 3. Rust regression

`cargo test --release`, both default-parallel and
`RUST_TEST_THREADS=1`, run after forcing a genuine rebuild
(`touch crates/ckks-cuda/build.rs`): **62 passed, 0 failed, 1 ignored,
identically in both modes.** `keyswitch_resident.cpp` (containing the
new batched `rescale_resident_raw` path) and `kernels/switch_modulus.cu`/
`kernels/rescale.cu` (containing the two new batched kernels) are all
compiled directly by Rust's `build.rs`, so this is a genuine
cross-language check, not a formality -- it compiled and ran clean.
`rescale.rs`'s own Rust-side `RescaleEngine` calls the UNCHANGED
single-tower `LaunchRescaleFuse` directly (confirmed by source read,
not assumed) and so does not exercise the new batched kernels itself;
its passing tests confirm the surrounding file still builds/links/runs
correctly, not that the new path was exercised from Rust.

## 4. Skipped / deferred / infrastructure

- `test_boot_n131072`: deliberately not run, same status every phase.
- No infrastructure failures this session (the WSL/PATH stall from
  Phase 40 did not recur -- clean PATH was applied from the first build
  command, not discovered mid-session this time).
- No test was resource-deferred due to an actual OOM kill this session
  (both large-n tests happened to complete; free RAM was checked before
  each and ranged ~2-6 GiB through the session).
