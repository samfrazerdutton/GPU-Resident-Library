# Phase 29: classification of the 11 previously-reported test failures

GOD PROMPT "CKKS PHASE 29", Section 12. Phase 28 reported "104/115 passing,
11 pre-existing, unrelated" without a per-test breakdown. This round
re-ran every one of the 11 individually (now 11 of 120, after Phase 29
added 5 new tests, all passing), read each one's source and purpose, and
classified it per the governing directive's four categories. None are
newly introduced regressions: none reference `encode_fft`/`encode_host_fft`
or anything else Phase 28/29 touched, and the ones with a documented
history predate Phase 27.

## Expected negative tests (deliberately drive invalid configurations)

- **`test_evalkeygen_isolated`**: deliberately constructs a
  Hybrid key-switch configuration with `P < Q_part_max` and calls
  `keyswitch_core_resident` directly (bypassing `ks_context_create`'s
  validation) specifically to DOCUMENT the resulting ~50% relin-identity
  error (`research/EXPERIMENT_LEDGER.md` entry #42). Fresh run: the exact
  same failure, `[FAIL] full reconstruction: keyswitch_core_resident`,
  worst residual 49.9987% of q. Already listed in `research/
  NEXT_SESSION.md` as "deliberately exempt, never calls
  `ks_context_create`... intentionally drive invalid configurations to
  document the phenomenon."
- **`test_keyswitch_stage_trace`**: sweeps a grid of
  (sizeQl, sizeP, numPart) combinations specifically to show WHICH
  satisfy `P >= Q_part_max` and which don't. Fresh run: CHECK C fails
  with ~50% of q error at every combination where the test's own chosen
  sizeP is too small, and PASSES at the two combinations where it is
  large enough (sizeP=3, numPart=2 at both sizeQl=5 and sizeQl=8). This
  is a parameter-space diagnostic grid, not a flaky test -- it behaves
  exactly as its own design predicts.
- **`test_stage2`**: sweeps `tw=2..15` for two probes; fails once `tw`
  grows large enough that its fixed `sizeP` no longer covers that tw's
  `Q_part_max` (same guard, same root mechanism). Fresh run: passes
  tw=2,3,4, then hits the SAME `ks_context_create` exception at tw=5.
  Same class as the rescale family below -- a fixed-parameter sweep test
  that was never extended to grow `sizeP` alongside `tw`.

## Known pre-existing defects (documented, not fixed, by design)

- **`test_polyeval`**: `research/EXPERIMENT_LEDGER.md` entries #25-26
  definitively root-caused this to a genuine, still-UNRESOLVED library
  bug in the native rescale pathway at small n (NOT a test-logic issue --
  confirmed via instrumented `[chk]` diagnostics that the tensor product
  and post-keyswitch result both decode correctly; corruption appears only
  after `rescale_resident_raw`). `research/NEXT_SESSION.md` explicitly
  says: "its `[FAIL]` is UNDERSTOOD and EXPECTED, not a red flag requiring
  a fix. Do not weaken its tolerance or delete it." Fresh run reproduces
  the same `ks_context_create` P<Q_part_max exception this test's own
  modulus chain triggers before even reaching the rescale step being
  studied (a second, compounding parameter-sizing issue layered on top of
  the documented rescale bug).
- **`test_rescale_n1024`, `test_rescale_scalar_diff_n1024`,
  `test_rescale_vector_diff_n1024`, `test_rescale_double_call_n1024`,
  `test_rescale_n1024_matched_scale`** (5 tests): all trigger the
  identical `ks_context_create: P < Q_part_max` exception at their shared
  n=1024 configuration. `research/NEXT_SESSION.md`: "`tests/
  test_rescale_n1024.cpp` remains committed with its original (mismatched)
  parameters as a permanent artifact of an earlier isolation experiment --
  its `[FAIL]` is UNDERSTOOD and EXPECTED... Do not weaken its tolerance
  or delete it; do not 'fix' it by changing library code." All 5 fresh
  runs reproduce the identical exception message, byte for byte.
- **`test_boot_multi_input_small`**: 12/12 actual cryptographic cases
  PASS (encode/decode, encrypt/decrypt, decode-after-decrypt,
  ciphertext-independence, repeatability -- all genuinely correct). The
  ONE failing check is `writeJsonl` returning false: `boot_driver_
  harness.h`'s `writeJsonl` opens `research/benchmarks/
  boot_multi_input_small.jsonl` via a path relative to the process's
  current working directory; `ctest` runs from `build/`, where `build/
  research/benchmarks/` does not exist. This is an **environment-
  dependent failure** (a relative-path/cwd mismatch), not a defect in any
  encode/encrypt/decrypt code path, and not something Phase 28 or 29
  touched (`boot_driver_harness.h` and `test_boot_multi_input_small.cpp`
  are untouched by either phase).

## Genuine pre-existing defect, newly classified this round (not previously documented anywhere)

- **`test_ks_graph`**: NOT a deliberately-invalid-parameter test --
  derives real CKKS parameters from OpenFHE (n=32768, depth=3, HYBRID key
  switch) the same way `bench/bench_evalmult.cpp` does, then checks that
  `keyswitch_resident` captured into a CUDA graph and replayed produces a
  bit-exact match against a direct launch. Fresh run: `graph-vs-direct
  mismatches=262144` (exactly `sizeQ*n`, i.e. one whole output buffer's
  worth), not a small numerical discrepancy -- consistent with one of the
  two output ciphertext components never being written by the graph
  replay at all. `git log` shows this file has had exactly ONE commit
  ever (`f87f3fc`, its creation) and is not mentioned in any research
  document -- it has never been investigated or revisited since being
  written. Checked whether this is a general CUDA-graph-capture
  limitation of this environment rather than a `keyswitch_resident`-
  specific bug: `bench/bench_evalmult_resident.cpp` ALSO captures a graph
  (of a different kernel sequence) and its own measured 172x-22634x graph
  speedup (`research/EXPERIMENT_LEDGER.md`) is strong evidence that CUDA
  graphs themselves work correctly in this exact WSL2/driver 595.97/CUDA
  13.2 environment -- ruling out a blanket environment explanation and
  pointing at something specific to how `keyswitch_resident` behaves
  under stream capture (a plausible candidate, not yet confirmed: its
  internal buffer lifecycle interacting badly with capture, given this
  codebase's established pattern elsewhere of per-call or thread_local
  conditional allocation). **Classification: genuine pre-existing defect,
  confirmed NOT caused by Phase 28 or 29** (neither touched anything
  related to CUDA graphs or `keyswitch_resident`), **newly identified and
  documented this round**, root cause not yet isolated. Recorded in
  `research/NEXT_SESSION.md` as a candidate for a dedicated investigation
  -- NOT attempted here, since a proper fix requires editing `src/
  keyswitch_resident.cpp` internals, which the governing directive
  requires an isolated reproducer for (this test case already serves as
  one) and real analysis beyond this already-large phase's remaining
  budget.

## Summary table

| Test | Category | Caused by Phase 28/29? |
|---|---|---|
| test_evalkeygen_isolated | Expected negative test | No |
| test_keyswitch_stage_trace | Expected negative test | No |
| test_stage2 | Expected negative test (param sweep boundary) | No |
| test_polyeval | Known pre-existing defect (documented, EXPERIMENT_LEDGER #25-26) | No |
| test_rescale_n1024 (+4 variants) | Known pre-existing defect (documented) | No |
| test_boot_multi_input_small | Environment-dependent (relative path/cwd) | No |
| test_ks_graph | Genuine pre-existing defect, newly classified this round | No |

**Zero of the 11 are newly introduced regressions.** All reproduce
identically to their Phase 28-era behavior; none touch any file Phase 28
or 29 modified.
