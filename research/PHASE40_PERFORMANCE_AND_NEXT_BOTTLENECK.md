# Phase 40: Performance Reproduction and the Next Bottleneck

## 1. Repository state

`HEAD = 05ada33`, verified matching `origin/main` exactly at session
start. Working tree clean apart from pre-existing, untouched items
(`.claude/agents/`, `.claude/skills/scrum-orchestrator/`,
`.scrum-orchestrator/`, `tools/__pycache__/`, and unstaged
`graphify-out/` regeneration) -- none of these are part of this
project's CKKS source and none were touched.

## 2. Track C: independent reproduction of Phase 39's Shoup claims

**Build-flag sanity check first** (explicitly required before trusting
any number): `build/CMakeCache.txt` shows `CMAKE_CUDA_FLAGS:STRING=`
(empty) and `CMAKE_BUILD_TYPE:STRING=Release` with
`CMAKE_CUDA_FLAGS_RELEASE:STRING=-O3 -DNDEBUG`. No `-G` (CUDA device
debug) flag is present. `CMakeLists.txt` itself carries a prominent,
pre-existing comment recording a *prior* incident where `-G` leaked in
via a one-off `cmake` invocation and silently persisted across
reconfigures (CMake only appends `CMAKE_CUDA_FLAGS`, never replaces
it) -- a guard (`CACHE STRING ... FORCE`) was added specifically to
prevent a repeat. Confirmed clean this session.

**Methodology**: reused the Phase 39 call-order-alternating, pre-warmed
isolated-kernel harness (`research/PHASE39_...md` section 7.5's
corrected methodology), but did NOT reuse the old compiled binaries.
Both the "old" (division) and "new" (Shoup) kernel bodies are
hand-copied into a self-contained `.cu` file for this kind of isolated
microbenchmark (project convention, since Phase 9) -- verified the
copied "new" kernel body is byte-identical to the current
`kernels/moddown.cu` / `kernels/rescale.cu` source before compiling
(it is). Freshly compiled this session with `nvcc -O3 -arch=sm_75`
(this machine's RTX 2060 Max-Q is compute capability 7.5) directly
against current source, 3 independent runs each, 3000 reps/run,
100-rep dedicated warm-up before any timed rep, call order alternated
every repetition.

**Extended this phase**: Phase 39 only measured n=8192 and n=131072.
This session added n=16384, 32768, 65536 to see the actual transition,
not just the two endpoints.

### ModDown combine kernel

| n | OLD median | NEW median | Speedup | 3-run spread |
|---:|---:|---:|---:|---|
| 8192 | ~0.0125-0.0136 ms | ~0.0126-0.0137 ms | 0.985x-1.000x | noise-level, no reliable win |
| 16384 | 0.01242 ms | 0.01258 ms | 0.987x | (single run; same noise band expected) |
| 32768 | 0.01254 ms | 0.01229 ms | 1.021x | transition begins |
| 65536 | 0.01507 ms | 0.01248 ms | 1.208x | real, growing |
| 131072 | ~0.0213-0.0217 ms | ~0.0145-0.0147 ms | 1.463x-1.476x (3 runs) | real, reproducible |

### RescaleFuse kernel

| n | OLD median | NEW median | Speedup | 3-run spread |
|---:|---:|---:|---:|---|
| 8192 | ~0.0123-0.0126 ms | ~0.0123-0.0129 ms | 0.978x-1.000x | noise-level, no reliable win |
| 16384 | 0.01312 ms | 0.01264 ms | 1.038x | transition begins |
| 32768 | 0.01504 ms | 0.01267 ms | 1.187x | real |
| 65536 | 0.01990 ms | 0.01280 ms | 1.555x | real |
| 131072 | ~0.0306-0.0307 ms | ~0.0157-0.0160 ms | 1.922x-1.947x (3 runs) | real, reproducible, largest win |

**Verdict: Phase 39's claims reproduce, honestly, from fresh source.**
At n=8192 (this project's primary demo scale) the result is a wash --
some runs show the "new" kernel very slightly SLOWER (0.978x-0.990x),
all within the same noise band as the "old" kernel, consistent with
Phase 39's own "launch-overhead-dominated, not a reliable win here"
framing. Not exaggerated: the n=8192 number is reported as a wash, not
rounded up to a win. The win is real and grows monotonically with n,
reaching ~1.46-1.48x (ModDown) and ~1.92-1.95x (RescaleFuse) at
n=131072, slightly higher than Phase 39's single-run figures
(1.435x/1.875x) but the same order of magnitude and well within
plausible run-to-run GPU timing variance for microsecond-scale kernels
-- not a different result, a reproduction.

**Absolute time saved per kernel launch at n=131072**: ModDown
~0.007 ms, RescaleFuse ~0.015 ms. **This is on the critical path once
per surviving tower, per rescale or per key-switch call** -- real but
small in absolute terms; see section 4 for whether this is visible
end to end.

**Underlying primitive re-verified from current source**: re-ran
`research/benchmarks/bench_moddown_shoup_check.cpp` (kept, unmodified
since Phase 39) fresh this session: 35,000,000 trials across the same
35 real NTT-friendly moduli (bit-widths 30-62), 0 mismatches, `shoup_mulmod`
bit-exact against raw `(x*c)%q` for every tested x,s < q. Confirms the
arithmetic identity has not regressed.

## 3. Track B summary (full detail in `PHASE40_SHOUP_RNS_VALIDATION.md`)

- Re-derived the correctness argument for both kernels by hand: both
  `moddown_combine_kernel` and `rescale_fuse_kernel` call
  `shoup_mulmod(x, c, precon, q)` only with `x < q` and `c < q`
  (`diff` is always a mod-`q` difference; `s`/`s1`/`s2` are always
  precomputed modular inverses/constants `< q`), and all moduli this
  project uses are far below `2^63`, so the standard one-correction
  Shoup bound applies and a single `if (r >= q) r -= q;` is provably
  sufficient -- matching the project's own already-verified
  `shoup_mulmod` used elsewhere (the encrypt hot path).
- All three `shoup_precon` helper definitions
  (`src/ckks_ops.cpp`, `src/keyswitch.cpp`, `src/keyswitch_resident.cpp`)
  are byte-identical: `(uint64_t)(((unsigned __int128)c << 64) / q)`.
- Rust FFI audit: `rust/crates/ckks-cuda/src/rescale.rs`'s
  `extern "C" fn LaunchRescaleFuse` declaration matches the current
  9-parameter C++ signature exactly (verified parameter-by-parameter);
  its call site computes `s1_precon`/`s2_precon` via native `u128`
  arithmetic identically to the C++ side. `LaunchModDownCombine` has
  **no separate Rust FFI declaration at all** -- grep confirms it is
  never referenced from any `rust/crates/*/src/*.rs` file; it is only
  reached through Rust's `build.rs` compiling the C++
  `keyswitch.cpp`/`keyswitch_resident.cpp` sources wholesale and
  calling them as opaque bridge functions, so there is no second
  declaration that could go stale for this particular function.
- Added `tests/test_shoup_moddown_rescale_boundary.cpp` (new, kept,
  ctest-registered): launches both kernels directly with explicit
  boundary operands (0, 1, q-1, q/2, q/2+1, plus pseudorandom) across
  three real primes (30/50/62-bit), checked against an INDEPENDENT
  oracle (plain 128-bit `%`, sharing no code with `shoup_mulmod`), and
  explicitly counts how many trials hit each of `shoup_mulmod`'s two
  branches (corrected vs. not) to make the "every branch is covered"
  claim a verified fact rather than an assumption. Result: pending
  this session's full build (Track A/D section records exact
  pass/fail).

## 4. End-to-end impact: honestly small, not hidden

This project's only pure-sum aggregation benchmark path (the one
end-to-end number tracked across Phases 34-38) exercises neither
rescale nor key-switching, so it is UNAFFECTED by this optimization --
consistent with Phase 39's own finding, re-confirmed, not re-litigated.
Any CKKS workload that calls `rescale_resident` or key-switching
(rotation, relinearization, EvalMult) gains the per-tower savings in
section 2 once per call; at this project's primary n=8192 scale that
saving is at or below measurement noise, so no end-to-end claim is
made for that scale. At n=131072 (the large-ring class this project
already builds and tests via `test_n131072_*`), the per-call saving is
real (section 2) but still a small fraction of that call's total cost
once NTT/INTT and memory traffic are included -- this document does
not claim a measured end-to-end wall-clock improvement at n=131072
because isolating the rescale/key-switch call's own contribution from
a full pipeline run was not performed this session (would require
instrumenting `keyswitch_resident.cpp`/`ckks_ops.cpp` call boundaries,
not attempted here to avoid the exact external-bracketing mistake this
project's own history (Phase 34) warns against). **Honest status: a
real, reproducible kernel-level win, with an unverified and likely
small end-to-end effect at this project's current scales.**

## 5. Track D: next bottleneck (see also section 6)

Carried forward without re-litigating (per this phase's own
instruction not to repeat settled negative/null results):
- RNG generation in the encrypt path: off-limits (Phase 37 conclusion).
- Thread-level parallelism of small per-call CPU work: actively
  harmful here (Phase 36's confirmed regression).
- Scratch-buffer pooling beyond what already exists: reverted
  previously, not revisited.
- Rotation hoisting (decompose-once-per-stage): null result, a
  correctness failure (100% mismatch under this project's approximate
  basis conversion), not revisited.
- K=32 multi-client throughput dip: root-caused to GPU power
  management (Phase 39), environmental, no code-level fix exists.

## 6. Track D finding and Track E implementation

**Static call-graph review** of `src/keyswitch_resident.cpp` (the
production resident key-switch/rescale path) found one clear,
well-precedented candidate: `ks_moddown`'s final combine step looped
over `sizeQl` towers issuing one `LaunchModDownCombine` kernel launch
per tower -- a serial pattern, unlike the INTT/NTT steps directly
above it in the same function, which were ALREADY batched into one
launch each (`LaunchINTT_GS_Batched`/`LaunchNTT_CT_Batched`) using a
`blockIdx.y`-indexed grid, the exact technique this project has
repeatedly and successfully used before (Phases 9, 30, 34, 38). This
was a leftover, not a new idea -- the surrounding code already
demonstrates the pattern works here.

**Quantified before implementing anything**: a standalone, order-
alternated, pre-warmed isolated reproducer (not committed; hand-copied
kernel bodies verified byte-identical to production source, same
methodology as the Phase 39/Track C benchmarks above) measuring
exactly this shape -- `sizeQl` separate tiny launches vs. one batched
launch, at this project's own n=8192 scale -- found a **3.88x
(sizeQl=5) to 5.93x (sizeQl=8) median speedup**, substantially larger
than the Shoup kernels' own n=8192 result (section 2) because this
targets launch-overhead directly, at exactly the scale where it
dominates.

**Implemented** (the one Track E optimization this phase):
- `kernels/moddown.cu` / `include/moddown.h`: added
  `moddown_combine_batched_kernel` / `LaunchModDownCombineBatched`,
  batching across `towers` in one launch via `blockIdx.y`, bit-exact
  with the existing per-tower kernel (same `shoup_mulmod` call per
  (tower, coefficient), just indexed instead of hard-coded to one
  tower per launch). The original `LaunchModDownCombine` is UNCHANGED
  and still used by every other call site (`src/keyswitch.cpp`,
  `src/ckks_ops.cpp`'s via `rescale_resident`'s own towers loop is
  untouched -- `rescale_resident_raw`'s analogous per-tower loop over
  `LaunchSwitchModulus`/`LaunchNTT_CT`/`LaunchRescaleFuse` was
  identified as a second, likely similarly-sized opportunity but
  **deliberately NOT implemented this phase**, to respect the "at most
  ONE optimization" scope -- flagged below for Phase 41).
- `include/keyswitch_resident.h` / `src/keyswitch_resident.cpp`: added
  three small per-tower device buffers (`d_mdS`, `d_mdSPrecon`, `d_mdQ`,
  `sizeQl` entries) and their persistent host-side staging vectors
  (`h_mdS`, `h_mdSPrecon`, `h_mdQ`) to `DeviceKSWork`, following this
  struct's own documented, hard-won rule (see its own long comment,
  from an earlier phase's CUDA-graph-capture investigation,
  `research/KS_GRAPH_CAPTURE_INVESTIGATION.md`): host-side staging
  buffers for any `cudaMemcpyAsync` issued inside a potentially-
  captured function MUST be persistent fields on `W`, never function-
  local, and MAY safely be shared across `ks_moddown`'s two call sites
  (for `d_ba0`/`d_ba1`) because these specific values are a pure
  function of the fixed `DeviceKSContext C`, never of `d_res`/`d_out` --
  identical reasoning already applied to this same function's
  `h_towerMods_qsw`/`h_rootsA_qsw` role. `ks_moddown`'s per-tower loop
  was replaced with: fill the three host arrays once, three
  `cudaMemcpyAsync` uploads, one `LaunchModDownCombineBatched` call.

**Correctness verification** (full, before trusting the change):
rebuilt clean; ran `test_ks_resident`, `test_rot_resident`,
`test_moddown`, `test_moddown_combine_isolated`,
`test_shoup_moddown_rescale_boundary`, `test_evalmult_rescale`,
`test_ks_fastcore`, `test_keyswitch` -- all passed bit-exact (only the
pre-existing, unrelated `test_keyswitch_stage_trace` failure present,
confirmed independently as the same deliberately-invalid-parameter
case documented in section 5 of `PHASE40_REGRESSION_BASELINE_RECONCILIATION.md`).
Ran the full rotation-hoisting battery (`test_hoist_single`,
`test_hoist_multi`, `test_hoist_levelreduced`, `test_hoist_stage0`,
`test_hoist_allstages`, `bench_hoist_multi`) -- **100% passed**,
including the memory-heavy `test_hoist_allstages` (134.67s). Ran the
full 4-batch broad regression suite plus both large-n tests again
after the change -- identical results to the pre-change baseline in
`PHASE40_REGRESSION_BASELINE_RECONCILIATION.md` section 5, zero new
regressions. Ran the Rust suite twice more (both parallel and
`RUST_TEST_THREADS=1`, forcing a genuine rebuild each time): **62
passed / 0 failed / 1 ignored**, unchanged -- `keyswitch_resident.cpp`
is compiled directly by Rust's `build.rs`, so this is a real
cross-language correctness check, not a formality.

**An incident during this verification, reported rather than hidden**:
one correctness-test invocation used an overly broad `ctest -R`
substring pattern (`...|boot|...`) intended to match
`test_boot_multi_input_small`, which also matched
`test_boot_n131072` -- a test this project's history already
established NEVER COMPLETES and has an unbounded memory footprint.
It was caught via a Windows host free-RAM check (showing a drop to
1.23 GiB) within roughly a minute of starting, and stopped immediately
(`TaskStop` on the specific background task this session had started;
verified via `ps aux` that no process survived). No OOM occurred, no
data was lost, and the correctness tests that regex was meant to catch
were re-run immediately afterward with an exact-name list instead.
Recorded here as the mistake it was, not smoothed over.

**End-to-end effect: honestly small or not measurable in existing heavy
tests, exactly like the Shoup kernels' own finding.** `test_hoist_allstages`
measured 133.12s immediately before this change and 134.67s immediately
after, on the same machine, same session -- within this specific test's
own observed historical noise band (133-136s across this project's
prior phases, with no code change at all). `bench_hoist_multi` similarly
showed no visible change (22.49s before, 23.22s after). This is the
expected, honest outcome given the isolated savings are sub-millisecond
per call (roughly 0.033-0.061ms per `ks_moddown` call at sizeQl=5-8,
so ~0.07-0.12ms per `keyswitch_resident` call, which invokes it twice) --
real and bit-exact-verified at the kernel level, but swamped by
everything else (CPU-side decompose/setup, large memory operations,
other kernel launches) in these particular heavy, multi-stage tests.
**No end-to-end speedup is claimed for this change beyond the isolated,
order-controlled kernel-launch-count measurement itself.** This mirrors
Phase 39's own honest framing for the Shoup kernels and is reported the
same way deliberately, not as an afterthought.

## 7. Recommended Phase 41 direction

1. **`rescale_resident_raw`'s analogous per-tower loop** (`LaunchSwitchModulus`
   + `LaunchNTT_CT` + `LaunchRescaleFuse`, 3 launches per surviving
   tower) is a second, likely comparably-sized launch-overhead
   opportunity, deliberately deferred this phase to respect the
   "at most one" scope. It is structurally harder than the ModDown case
   (three different kernel types per tower, not one), so it deserves
   its own dedicated reproducer-first investigation, not a rushed
   extension of this phase's work.
2. **A tighter end-to-end measurement instrument** for small per-call
   savings would help future phases avoid the "real at the kernel
   level, invisible in a 130-second test" pattern recurring a third
   time (Shoup kernels, this phase's ModDown batching) -- e.g. a
   dedicated microbenchmark that calls `keyswitch_resident` alone, many
   times, with GPU-event timing and no surrounding decompose/setup
   cost, would make small per-call wins visible without needing to
   revert and re-measure production code.
3. Continue carrying forward all prior null/negative results (RNG
   off-limits, thread-parallel/scratch-pooling reverted, rotation-
   hoisting null result, K=32 GPU power-management finding) without
   re-litigating them.
