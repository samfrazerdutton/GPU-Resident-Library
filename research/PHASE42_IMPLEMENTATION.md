# Phase 42: Implementation — Batched `rescale_resident_raw`

## 1. The mathematical operation, unchanged

CKKS rescale drops the last RNS tower and fuses its (base-converted,
re-NTT'd) contribution into each surviving tower:
```
result[j] = (a[j]*s1[t] + b[j]*s2[t]) mod q_t     -- per surviving tower t
```
where `a` is the surviving tower's own coefficients, `b` is the dropped
tower switched (via `SwitchModulus`) into `q_t`'s representation and
NTT'd back to evaluation form, and `s1`/`s2` are OpenFHE's precomputed
`qlInvModq`/`QlQlInvModqlDivqlModq` constants. This is unchanged by
this phase -- the only change is HOW MANY kernel launches compute it.

## 2. Before: per-tower serial loop

```cpp
for (t = 0; t < last; ++t) {
    cudaMemcpyAsync(d_scratch, d_drop, n*8, DtoD, s);  // re-copy the SAME source
    LaunchSwitchModulus(d_scratch, qLast, q_t, n, s);  // 1 launch
    LaunchNTT_CT(d_scratch, roots[t], precon[t], n, q_t, s); // ~5 launches at n=8192-32768
    LaunchRescaleFuse(d_c+t*n, d_scratch, s1[t],...,s2[t],...,q_t, n, s); // 1 launch
}
```
`last` iterations x (1 + ~5 + 1) launches. At `last=10` (depth 10 in
this project's standard key-switch config): ~70 kernel launches for
one `rescale_resident_raw` call.

## 3. After: three batched launches

```cpp
LaunchSwitchModulusBatched(d_scratchBatch, d_drop, qLast, d_qNew, d_diff, d_newLarger, last, n, s);
LaunchNTT_CT_Batched(d_scratchBatch, d_rootsA, d_rootsB, d_towerMods, last, n, s); // ~5 launches total, not x last
LaunchRescaleFuseBatched(d_c, d_scratchBatch, d_s1, d_s1p, d_s2, d_s2p, d_q, last, n, s);
```
~7 launches total regardless of `last` (the NTT stage count is fixed
by `n`, not by `towers` -- it was already this way for
`LaunchNTT_CT_Batched`, proven and shipped in Phases 30/40 for
`ks_moddown`'s own INTT/NTT steps; this phase reuses it UNCHANGED).

### New kernels (the only new arithmetic surface)

- `kernels/switch_modulus.cu`: `switch_modulus_batched_kernel` /
  `LaunchSwitchModulusBatched`. Reads the SAME single source `d_drop`
  for every tower-block (eliminates the need for a separate broadcast
  step) and writes each tower's result into its own `[towers][n]`
  output slot. Formula identical to the existing, unchanged
  `switch_modulus_kernel`, just indexed by `blockIdx.y` (`q_new`/`diff`/
  `newLarger` read from per-tower arrays) instead of being fixed
  per-call scalars.
- `kernels/rescale.cu`: `rescale_fuse_batched_kernel` /
  `LaunchRescaleFuseBatched`. Identical to the existing, unchanged
  `rescale_fuse_kernel` (same `shoup_mulmod` calls, same addition/
  correction), just indexed by `blockIdx.y` with per-tower constant
  arrays instead of per-call scalars -- the exact same transformation
  Phase 40 already applied to `ModDownCombine`.
- `LaunchNTT_CT_Batched`: NOT new. Already existed, already proven
  (Phase 30 for `ks_accumulate_qp`'s complement NTT, Phase 40 for
  `ks_moddown`'s own INTT/NTT steps). Reused here unchanged.

Both new kernels are provably bit-exact with their already-trusted
single-tower counterparts by direct inspection: same arithmetic, same
operand ranges (re-derived, not assumed -- see
`research/PHASE40_SHOUP_RNS_VALIDATION.md` section 1 for the
`shoup_mulmod` bound this reuses unchanged; `SwitchModulus`'s own
branch structure is copied verbatim, only the per-tower constant
source changed from scalar to array-indexed).

## 4. The new internal state: `RescaleBatchWork`

`rescale_resident_raw`'s PUBLIC signature is UNCHANGED (it is called
from ~30 files; changing it would have meant updating every caller's
own buffer allocations). The batched path's extra per-tower arrays
(`d_scratchBatch`, `d_qNew`/`d_diff`/`d_newLarger`, the NTT root-table
pointer arrays, `d_s1`/`d_s1Precon`/`d_s2`/`d_s2Precon`/`d_q`) live in a
new `RescaleBatchWork` struct, cached process-lifetime by `(n, last)`
in `src/keyswitch_resident.cpp`'s existing anonymous namespace, guarded
by the SAME `g_ks_cache_mutex()` every other cache in this file already
shares (established convention: "lookups/inserts are cheap... a lock
here costs nothing that matters").

**CUDA graph-capture safety** (this project's own hard-learned lesson,
`research/KS_GRAPH_CAPTURE_INVESTIGATION.md`, re-applied deliberately
here, not rediscovered): the cache's HOST-side staging vectors are
persistent fields on the struct (not function-local), because a
captured `cudaMemcpyAsync` node stores the host pointer, re-read on
every future `cudaGraphLaunch` replay. They are refilled with the
CURRENT call's values on every call. The one correctness question this
raises -- what if two calls at the SAME `(n,last)` key but with
DIFFERENT `s1`/`s2` are captured inside the SAME graph? -- was checked
against this codebase's actual usage, not just argued: the only real
caller that captures `rescale_resident_raw`
(`bench/bench_evalmult_resident.cpp`) always calls it twice per
capture with the SAME `s1`/`s2` (CKKS's rescale constants depend only
on the level, never on which ciphertext component is being rescaled --
a mathematical invariant of this project's own usage, not a
coincidence of one file), and captures across different chains are
strictly sequential (one `cudaStreamBeginCapture`/`EndCapture` pair
fully completes before the next begins). **Verified, not just argued**:
`tests/test_rescale_resident_graph.cpp` (new, kept) directly tests both
the single-call and the two-call-same-(n,last)-same-s1/s2 scenarios
under real graph capture, bit-exact against direct (non-captured)
calls (0/98304 mismatches, both scenarios).

An A/B switch, `bool g_rescale_batched` (default `true`), mirrors the
project's own already-established `g_ks_batched` convention exactly --
the OLD serial loop is fully preserved (not deleted) behind
`if (!g_rescale_batched)`, giving a real, switchable reference
implementation for interleaved measurement rather than a git-stash
round trip.

## 5. Design decisions rejected

- **Changing `rescale_resident_raw`'s public signature** to accept a
  larger caller-owned scratch buffer: rejected -- ~30 call sites would
  each need their own allocation updated, a much larger and riskier
  change than an internal, additive cache.
- **A separate broadcast kernel** to replicate `d_drop` into each
  tower's scratch slot before switching modulus: rejected --
  `switch_modulus_batched_kernel` reads the single shared source
  directly, eliminating the extra launch entirely.
- **Batching `rescale_resident_raw` even at `last<2`**: the function
  already returns early for `towers<2`; no special-casing needed for
  `last==1` either -- the batched path's launch count does not
  increase relative to the serial path at `last==1` (still one launch
  per kernel type), so there was no regression to special-case around.
- **Generalizing the per-tower-loop-batching pattern into a shared
  utility used by both `ks_moddown` and `rescale_resident_raw`**:
  rejected as unjustified abstraction for two call sites with
  different per-stage shapes (one kernel type vs. three); matches this
  project's own stated preference for duplication over premature
  centralization.

## 6. Files changed

- `kernels/switch_modulus.cu`, `include/switch_modulus.h`: new batched
  kernel/launcher.
- `kernels/rescale.cu`, `include/rescale.h`: new batched kernel/launcher.
- `include/keyswitch_resident.h`, `src/keyswitch_resident.cpp`:
  `RescaleBatchWork` cache, `g_rescale_batched` flag,
  `rescale_resident_raw`'s rewritten (OLD preserved + NEW) body.
- `tests/test_rescale_resident_graph.cpp` (new): CUDA-graph-capture
  correctness gate.
- `bench/bench_rescale_resident_batched_vs_serial.cpp` (new): isolated
  A/B.
- `bench/bench_evalmult_resident.cpp` (modified, additively): one new
  A/B block measuring the direct-path end-to-end EvalMult effect via
  the same toggle; nothing existing removed.
- `CMakeLists.txt`: registers the two new files.
