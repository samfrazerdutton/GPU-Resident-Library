# Next session — exact continuation

## Current commit

Check `git log -3` on resume. As of this writing, HEAD is a new commit
on top of `f1978ab` (the n=131072 memory-investigation round — see
`MEMORY_FEASIBILITY.md`'s final `## UPDATE` section for the full story,
condensed in `MASTER_STATE.md`'s latest `## This round` section and
`EXPERIMENT_LEDGER.md` entry #22). Confirm push status — push again if
`git status` shows ahead of origin.

## Read first

`research/MEMORY_FEASIBILITY.md`'s final `## UPDATE (new round): KROT/KSC
cache mechanism DEFINITIVELY CLEARED...` section — it supersedes the
"something besides KROT/KSC grows with n, not yet identified" framing
used everywhere earlier in that file and in this document. Then
`research/MASTER_STATE.md`'s `## This round: production integration,
honest A/B, security re-review` section for the PRIOR round's findings
(still current), and `research/EXPERIMENT_LEDGER.md` entries #21-#22.

## Immediate next action (the one concrete, executable item left open)

**n=131072 feasibility is now believed to be blocked by this specific
machine's total physical RAM (~15.4GB), not by an unidentified
implementation bug.** This round isolated the KROT/KSC bounded-LRU
cache mechanism in a standalone experiment (`tests/
test_n131072_cache_churn.cpp`) and proved it plateaus at a bounded
~3.2GB through 22+11 real evictions — it is NOT the runaway cause.
Enhanced instrumentation on a 5th real attempt showed RSS was only
1436MB (safely low) immediately before the crash, which happened
inside the very first `keyAt()`/`stageRotBegin()` call of stage 0 —
code-reading ruled out that call's own allocations (device-side,
~350MB-1.35GB, all previously-established patterns) as the explanation.
Checking the Windows host directly found the real likely cause: this
machine has only ~15.42GB total physical RAM, `.wslconfig` caps WSL2 at
11GB, but Windows itself already uses ~11.8GB at near-idle (only ~3.6GB
free) — WSL2's internal `free -h` ("10Gi free") reflects a *promised*
allocation, not memory the host can actually back under real
contention. **Next concrete steps, in order**: (1) close other
Windows-side memory consumers and retry the exact same
`./test_boot_n131072` command to see if it survives further with
maximum real host headroom — the cheapest, most direct test of this
diagnosis, not yet done; (2) if it still fails with minimal other
load, treat ~15.4GB total physical RAM as a hard blocker for n=131072
on this specific machine and stop retrying cache-constant tweaks — the
realistic paths forward are more physical RAM or a streaming
(non-cached) key-material architecture, both out of scope for a single
session. **Do NOT further lower `KROT_CAP`/`KSC_CAP` expecting a
different result** — `test_n131072_cache_churn.cpp` already proved the
cache mechanism itself is not the limiting factor at the caps already
in use. Also note: `stageDiags`/`stageDiagsFwd` (O(S²), now measured at
268s/call at n=131072, ~35 CPU-minutes for all 8 C2S+S2C stages) is a
severe, separate, real performance cost independent of whether the
memory question resolves — worth fixing (precompute diagonal structure
analytically) before any future large-n attempt regardless. Full detail:
`MEMORY_FEASIBILITY.md`.

Secondary open items from this round (lower priority than the above):
- A larger, truly-randomized (not block-alternated) A/B trial series to
  settle whether rotation hoisting's end-to-end effect on the full
  bootstrap is real-but-small or statistically zero — the current
  8-trial, order-balanced series could not distinguish the two (explicit
  order-effect decomposition: median=-1.826s, mean=-2.564s, stdev=6.680s
  across the 4 pairs — a direction consistent with GPU warm-up but not
  statistically crisp at this sample size; see `MASTER_STATE.md`).
- The SageMath lattice-estimator cross-check and the direct
  eprint-2021/152 / eprint-2026/1060 parameter check named in
  `SECURITY_ASSESSMENT.md`'s updated `## Disposition`.

## Two pre-existing bugs newly discovered this round (NOT caused by
## this round's changes, NOT fixed — out of scope for this directive)

A full `ctest` run (excluding `test_boot_n131072`, the known-blocked
n=131072 feasibility target) was completed for the first time all
session — every prior attempt had been OOM-killed partway through, so
these two failures had never actually been observed before now:

1. **`test_polyeval` fails deterministically**: `deg-7 poly eval, final
   tw=1 scale=2^50 maxerr=2376.98` (`[FAIL]`), reproduced identically
   twice (same exact error values both times, in and out of the full
   ctest run). `git log -- tests/test_polyeval.cpp` shows it was last
   touched in `78d7407`, long before this session's hoisting work —
   this round touched none of the C2S/polynomial-evaluation code paths.
   Not investigated further (out of scope for the current directive);
   whoever picks this up next should treat it as a real, pre-existing,
   previously-undiscovered correctness bug, not a flake.
2. **`bench_evalmult_resident` aborts**: `terminate called after
   throwing an instance of 'std::runtime_error' what(): ShadowRegistry
   cudaMalloc failed (VRAM exhausted?)`, reproduced standalone with GPU
   VRAM confirmed otherwise idle (`nvidia-smi`: 10MiB/6144MiB used
   immediately before the retry). `git log -- bench/
   bench_evalmult_resident.cpp` shows it was last touched in `6efc812`
   (CUDA-graphs-era commit, well before this session). "ShadowRegistry"
   does not appear anywhere in this repo's own source — it is coming
   from CUDA's own graph-capture machinery, suggesting a driver/CUDA-
   graph resource limit specific to this old benchmark on this GPU,
   not a host-RAM or this-round issue. Not investigated further (out of
   scope); this is a benchmark, not a correctness test.

**Everything else in the full suite passed**, including all of this
round's own correctness evidence: `test_boot` (113.71s, matching
established timing), `test_hoist_single`/`test_hoist_multi`/
`test_hoist_levelreduced`/`test_hoist_stage0`/`test_hoist_allstages`
(the last one needed to be run standalone after the first full-suite
attempt OOM'd entering it — passed cleanly alone: `[PASS] all-stage C2S
hoisting validation`, maxDiff=35 vs maxAbs=5.26e15, speedup=1.297x in
that one specific sequential-pass comparison — consistent with entry
#20's prior result, not a new number to generalize from).

## Before trusting any new benchmark number

GPU thermal state matters on this machine (Max-Q mobile part):
```bash
wsl -d Ubuntu -- bash -lc 'export PATH=/usr/bin:/bin:/usr/sbin:/sbin; /usr/lib/wsl/lib/nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,power.draw --format=csv'
```
Use `research/benchmarks/run_trials.sh <label> <N>` or `interleaved_ab.sh`
for repeated/interleaved measurement — never trust a single sequential run.

## This round's real, delivered results (commits `93f2231`, `0043af3`)

1. **KSC bounded cache**: peak RSS 10.78GB → 8.12-8.32GB (~25%), bit-exact.
2. **Eval-domain automorphism** (ranked experiment #2): replaced
   `automorphism_eval_device`'s INTT→permute→NTT with a single precomputed
   gather kernel (`automorphism_eval_fast_device`, `kernels/automorph_eval.cu`).
   Derived AND independently validated (GPU-free Python reference, 20
   trials) before writing CUDA. Caught a real use-after-free bug (async
   copy from a stack-local vector) via the standalone test
   (`tests/test_automorph_eval_fast.cpp`) before it ever reached production.
   40/40 bit-exact at real scale (n=8192). **Measured: rotate bucket down
   to 55-60s (was 62-82s depending on thermal state), total bootstrap down
   to 108-114s — the lowest this entire session.**

## Priority queue (highest value first)

1. **Correct-order rotation hoisting (ranked experiment #3) — single-
   rotation correctness now RESOLVED (commit `46998b7`). The "100%
   mismatch" recorded in earlier revisions of this file was a TEST BUG
   (bit-exact ciphertext comparison, the wrong invariant), not an algebra
   bug — see `EXPERIMENT_LEDGER.md` entry #14 for the full explanation.
   Key-generation convention attempt 2 (permute `pkA_QP`/`pkB_QP` by
   `k^-1`, keep payload = unpermuted base secret) is CORRECT. Do not
   re-derive this from scratch — it is proven, move on to the next step.**

   **What's PROVEN correct now** (`tests/test_hoist_single.cpp`, promoted
   into the default test suite, `GPUFHE_BUILD_EXPERIMENTAL_TESTS` gate
   removed from `CMakeLists.txt` since it's unused): the full pipeline —
   `ks_accumulate_qp` on the unpermuted c1, folding `c0*(P mod q)` into
   the Q-tower range of the QP-basis accumulator via `LaunchScaledAddInto`
   (`kernels/stage_acc.cu`), applying `automorphism_eval_fast_device` to
   the QP-basis accumulator (sizeQlP towers), then `ks_moddown`, using a
   key generated by permuting `pkA_QP`/`pkB_QP` by `k^-1` while keeping
   the payload as the unpermuted base secret — matches the existing,
   proven `rotate_ct_resident` **after decryption, within CKKS's own
   noise budget, across 27/27 cases** (3 independently-encrypted random
   ciphertexts x 9 rotation indices including k=M-1, k=M/2-1 edge cases).
   Decrypted-coefficient error is a stable ~2000-3300 against a
   ~1e14-1e15 signal — consistent with ordinary keyswitch rounding noise.

   **Multi-rotation decompose-sharing mechanism: DONE, real measured win
   (commit pending at time of writing — see git log for the actual hash).**
   New, separate files (do NOT touch `src/keyswitch_resident.cpp`'s logic,
   per the hard constraint): `include/keyswitch_hoist.h` +
   `src/keyswitch_hoist.cpp` provide `hoist_digits_create`/
   `hoist_digits_destroy`, `ks_decompose_only` (steps 1-3 of
   `ks_accumulate_qp`, factored apart, storing per-part `d_compl` results
   in a `HoistedDigits` struct instead of a single shared scratch buffer)
   and `ks_accumulate_from_digits` (step 4 only, against a SPECIFIC
   context's av/bv, reusing a previously-computed `HoistedDigits`).
   `tests/test_hoist_multi.cpp` proves TWO things: (1) the decompose/
   accumulate split is BIT-EXACT identical to the existing fused
   `ks_accumulate_qp`, 0/524288 mismatches (the refactor itself changes
   nothing); (2) decomposing c1 ONCE and reusing it across 6 different
   rotation indices matches the proven ground truth within noise, same
   magnitude (~2000-3300) as the single-rotation test. `bench/
   bench_hoist_multi.cpp` measured the actual performance claim required
   before trusting any speedup: interleaved A/B (baseline and hoisted
   methods alternated every rep, 30 reps + 5 warmup, within one process
   so GPU thermal drift is common-mode and cancels in the paired
   difference, same statistical approach as `bench_resident_chain.cpp`) —
   **N=5 rotation indices: baseline (5 independent full keyswitches)
   median 94.49ms vs. hoisted (1 decompose + 5 cheap accumulates) median
   46.32ms — a ~2.04x speedup, bootstrap 95% CI on the paired difference
   [47.86, 48.60]ms, entirely excluding zero.** This is a real, clean win
   at the micro-benchmark level, not noise.

   **Re-measured at a REALISTIC bucket size (N=16, matching C2S stage
   0's actual measured diagonal count "diags=16" exactly, not an
   arbitrary N=5): baseline median 374.76ms vs hoisted median 153.62ms —
   ~2.44x, bootstrap 95% CI on the paired difference [220.0, 297.0]ms,
   entirely excludes zero.** (Raw per-rep times drifted upward 320ms→
   492ms across the run from GPU thermal throttling — expected and
   harmless here specifically because baseline/hoisted are measured
   back-to-back within each rep, so the ratio stays valid even as
   absolute numbers drift; this is exactly why the paired difference,
   not either raw mean, is the number to trust.) The relative win is
   LARGER at N=16 than at N=5 (2.44x vs 2.04x), as expected since more
   rotations amortize one shared decompose further.

   **Real end-to-end validation against an ACTUAL bootstrap stage: DONE,
   PASSED.** `tests/test_hoist_stage0.cpp` (a modified copy of
   `test_boot.cpp`, identical through ModRaise and C2S stage 0's own
   trusted computation, then ALSO computes a hoisted-mechanism version
   of that SAME real stage — own fresh hoisted-convention keys, no
   cache advantage over the trusted path, which is also generating
   fresh keys here on its first use via `keyAt`): **maxDiff=613 against
   maxAbs=5.76e17 (negligible, ~1e-15 relative — [MATCH within noise]).
   Hoisted total 7.53s (decompose-once=0.073s + 16 fresh keygens=6.71s +
   compute=0.75s) vs trusted 25.89s — a real ~3.4x speedup on the
   actual first C2S stage of a live bootstrap, both sides paying for
   fresh key generation.** This is a substantially larger win than the
   N=16 synthetic micro-benchmark's ~2.44x, plausibly because the real
   stage's trusted per-diagonal compute (full `ks_accumulate_qp` via
   `keyswitch_resident`, 16 times) carries more overhead than the
   synthetic benchmark isolated.

   Debugging note worth preserving: the first attempt at this
   validation produced a dramatic, nonsensical mismatch (maxDiff twice
   maxAbs) that turned out to be a TEST bug, not an algorithm bug —
   decrypting the PRE-`rescaleAt` accumulated sum (16 ciphertext×
   plaintext products summed, at a combined scale that exceeds what
   `decrypt_host`'s `int64_t` return represents) overflows silently.
   Isolating each diagonal's raw rotation (pre-multiply, still at the
   original encryption scale) showed perfect agreement throughout,
   which is what correctly pointed at the comparison itself rather than
   the mechanism. Always rescale before decrypting an intermediate
   stage's accumulator.

   **Reproduced**: re-ran stage 0's validation a second time independently
   — [PASS] again, maxDiff=613 (bit-identical to the first run, since
   decrypted noise from a deterministic pipeline is itself deterministic),
   ~3.0x-3.4x speedup across the two runs (normal thermal/timing variance,
   not a correctness concern). Extended `tests/test_hoist_stage0.cpp` to
   attempt ALL 3 C2S stages in one run (not just stage 0) — it OOM'd
   entering stage 1 (31 diagonals vs stage 0's 16) on BOTH attempts, after
   stage 0 itself passed cleanly both times. This is consistent with
   this session's established ~10GB ceiling being tight even at n=8192
   when stacking: the trusted path's own `KROT`/`KSC` caches accumulating
   across multiple stages' worth of keys, PLUS the hoisted path's fresh,
   uncached per-diagonal key generation (by design, for a fair
   no-cache-advantage comparison) adds real additional pressure on top.
   Not chased further via that design — the single-stage result (now
   reproduced twice) is strong enough evidence for this round.

   **Follow-up: `tests/test_hoist_allstages.cpp` — NOW PASSING, all 3
   real C2S stages, after fixing two bugs found in sequence** (new file,
   sequential-pass design: run the trusted path's existing, UNMODIFIED
   C2S loop to completion first, THEN reset to the same ModRaise output
   and run a SEPARATE hoisted-only pass over all 3 stages — never
   holding both paths' keys simultaneously, avoiding the memory
   coexistence problem above). Pass 1 (trusted) completed cleanly every
   time (matches known timings). Pass 2 (hoisted) surfaced TWO distinct
   bugs, in sequence (both now fixed, see below for the resolution and
   final result):
   1. The per-stage hoisted key construction (`Kh`) and the decompose
      shape reference (`Cshape`) both passed the FULL, un-sliced `mod`
      array (30 entries) to `compute_keyswitch_constants` even after
      `hctw` shrank to 29, 28 via rescale — a genuine bug (not just a
      stage-0 coincidence masking it): the trusted path's own
      `atLevel()` always slices `mod` to the current tower count first,
      which this code failed to do. Threw `std::runtime_error("no root
      table for modulus")` entering stage 1. **Fixed** by slicing to
      `hml` (already computed per-stage) before calling
      `compute_keyswitch_constants`, matching `atLevel`'s pattern.
   2. After that fix, pass 2 instead SEGFAULTS. **RESOLVED (later
      continuation round)**: ran under `gdb --batch -ex run -ex "bt full"`
      on a fast, memory-light repro (a scratch copy with pass 1 skipped
      entirely, `g<0` instead of `g<LST`, to avoid pass 1's own ~7.5GB
      RSS competing with gdb's overhead) — crash squarely inside
      `gpufhe::ks_context_create` (existing, proven, untouched). Root
      cause: a hand-built "shape-only" `KeySwitchConstants` (correct
      `sizeQl`/`numPart`/root-tables, but default-constructed EMPTY
      `av`/`bv`, since decompose doesn't read them) — but
      `ks_context_create` unconditionally uploads every part's `av`/`bv`
      regardless of caller intent, indexing the empty vectors. Fixed by
      never hand-rolling a shape-only key for anything passed to
      `ks_context_create`: use `atLevel(KREL,hctw)` instead (the trusted
      path's own existing mechanism for exactly this — re-shape a FULL
      key down to `hctw` towers while copying its REAL, non-empty
      `av`/`bv` across). Added `tests/test_hoist_levelreduced.cpp` as a
      permanent regression test for this exact defect class (builds a
      real level-reduced context via the correct pattern, asserts no
      CUDA error) — now in the default test suite.
   3. **A second, more subtle bug surfaced immediately after fixing #2**:
      with the segfault gone, pass 2 completed all 3 stages but produced
      a real correctness MISMATCH (`maxDiff` exceeding `maxAbs` — not
      noise, wrong). Cause (found by reasoning through
      `ks_accumulate_qp`'s own `fullQ`/`delta` indexing, not another
      gdb session): `evalkeygen_host_sold` always writes `av`/`bv` using
      the FULL, fixed 32-tower `modQP` layout (real Q at indices 0-29,
      real P at 30-31) — generating the hoisted rotation key directly at
      a REDUCED `sizeQl=hctw` (as the code did) does not shrink that
      layout, it only changes which towers get the "+P*s2" payload. The
      later accumulate step locates the P-range WITHIN `av`/`bv` using
      `sizeQl` as an offset into that fixed layout — correct only when
      `hctw==sizeQ` (stage 0 exclusively, which is exactly why ONLY
      stage 0 ever passed in every earlier single-stage test). Fixed by
      using the same two-step pattern the trusted path already uses for
      every rotation key at every level (`rotFull`+`atLevel`): build the
      hoisted key ONCE at full `sizeQl=sizeQ`, then take an
      `atLevel`-style view for the current `hctw` (`atLevel` already
      sets `K.fullQ=sizeQ`, which is what makes `ks_accumulate_qp`'s
      `delta=fullQ-sizeQl` correctly skip the now-irrelevant dropped
      towers and land on the real P-range).

   **Result after both fixes: `[PASS]` across ALL 3 real C2S stages**
   (not just stage 0) — `maxDiff=35` against `maxAbs=5.26e15` (negligible
   noise, ~6.7e-15 relative). This is the strongest correctness evidence
   yet for the hoisting mechanism: proven right across every stage of
   a real bootstrap's C2S phase, not just one stage or synthetic data.

   **Timing from this specific run: 0.93x (pass 2 slightly SLOWER than
   pass 1) — do NOT read this as "hoisting doesn't help."** This
   benchmark's sequential-pass design (deliberately, for a fair
   no-cache-advantage comparison) makes the hoisted path pay full fresh
   `evalkeygen_host_sold` cost on EVERY diagonal, while the trusted
   path's `keyAt` benefits from its own `KSC` view-cache (cheap
   re-copies, not fresh keygen, for most diagonals after the first few).
   This conflates "decompose-sharing vs. not" with "no cache vs. has a
   cache" — two different variables — unlike the EARLIER, still-valid
   `bench/bench_hoist_multi.cpp` (N=5/N=16, ~2.04x/~2.44x) and
   `tests/test_hoist_stage0.cpp` (stage 0 alone, ~3.4x) results, where
   BOTH sides paid fresh keygen fairly. A clean multi-stage speedup
   measurement would need the hoisted path to ALSO use a bounded view
   cache analogous to `KSC`, not fresh-every-diagonal — not yet built.

   **DONE, in a later round (see `MASTER_STATE.md`'s `## This round`
   section and `EXPERIMENT_LEDGER.md` entry #21 — this paragraph is
   preserved for history, do not re-plan it from scratch).** Extended
   from "validated on stage 0 in isolation" to "wired into the live
   full bootstrap, both C2S and S2C," via exactly the separate
   `KROT_hoist` cache design this paragraph called for, feature-flagged
   behind `HOIST_ROTATION_ENABLED` (off by default) with per-diagonal/
   per-stage CUDA-error fallback to the trusted path. Correctness is
   proven (byte-identical flag-off behavior; byte-identical `max slot
   err` with the flag on across 4 trials). **What that later round did
   NOT achieve: a clean, statistically solid performance win** — an
   8-trial order-balanced A/B series found the effect small and within
   noise once the warm-up order confound (the original design below
   didn't anticipate it) is controlled for. `HOIST_ROTATION_ENABLED`
   remains `false` by default and `test_boot_hoist_enabled` remains
   un-promoted (not in the default ctest run) pending a larger,
   fully-randomized trial series. The paragraph below (describing the
   design BEFORE it was built) stays for history:

   needs a real architectural decision, not a quick swap — `test_boot.cpp`'s
   `rotFull`/`keyAt`/`KROT`/`KSC` build and cache TEXTBOOK-convention
   keys (permute the payload secret by `k`); the hoisted mechanism needs
   the OTHER convention (permute `pkA_QP`/`pkB_QP` by `k^-1`, payload
   unpermuted) — a structurally different key, not a drop-in replacement
   for what's cached today. Correct, safe approach: add a SEPARATE,
   parallel `KROT_hoist`/`KSC_hoist` cache (same bounded-LRU pattern)
   feeding a `stageRotHoistDiag`, wired into ALL stages (not just stage
   0), with the existing trusted path kept available for the interleaved
   comparison until the full-bootstrap numbers are in — do not simply
   replace it. `tests/test_hoist_stage0.cpp` is the proven template for
   each stage's validation; extending it to loop over every `g` instead
   of returning after `g==0` is the mechanical part of this next step.

2. **`KROT`'s own memory scaling — lever analysis DONE, safety-net
   implemented, actual eviction tuning for n=131072 NOT yet done.** A
   delegated investigation ruled out zero-copy host/device sharing with
   direct code evidence (`ks_context_create` already re-uploads+frees
   av/bv to device on EVERY call — there is no persistent device copy to
   deduplicate; also VRAM 6GB < host RAM 10GB on this machine, so it
   wouldn't help anyway), and rejected both BSGS/LST retuning (collides
   with the fixed 1775-bit security budget; flagged as the highest
   long-term-value lever, needs a dedicated session to re-derive the
   scale-management chain safely) and numPart/alpha retuning (same
   budget collision, PLUS surfaced a real pre-existing latent correctness
   bug at alpha=10, `tests/test_alpha10.cpp`, previously undocumented).
   Full analysis: `MEMORY_FEASIBILITY.md`'s "KROT lever analysis" section.
   **Implemented**: `KROT_CAP` bounded LRU cache in `tests/test_boot.cpp`
   (same pattern as the already-merged `KSC_CAP`), defaulted to 96 —
   comfortably above the measured 77-entry n=8192 working set, so this
   causes ZERO evictions and ZERO behavior change today (verified:
   `test_boot` reproduces the exact same `max slot err = 5.47871e-4` and
   peak RSS ~8.11GB as before this change). This is the SAFETY NET, not
   yet a fix — at n=131072 the cap will need to be lowered well below the
   full working set (infeasible at ~1GB/key × 77 keys), and doing so has
   a real, measured regen cost (≈0.64s/key at n=8192, extrapolated
   ≈10.9s/key at n=131072 — NOT measured at that scale) that must be
   re-measured, not assumed free, once n=131072 is actually attempted
   (item 3 below). Do not claim n=131072 is memory-feasible until that
   measurement is done with a real cap chosen for that scale.

3. **Run n=131072 end to end — ATTEMPTED THREE TIMES this round, all
   OOM'd, NOT complete. Security target independently reconfirmed; real
   per-key cost measured; a genuine, unidentified memory-growth source
   beyond KROT/KSC still blocks it.** Keep the sparse secret (h=64) — no
   `keygen.cpp` change needed, re-confirmed fresh this round:
   `python3 tools/hybrid_security.py 131072 1775 64` → 134 classical
   bits (output preserved in `SECURITY_ASSESSMENT.md`).

   **What's real from this round**: `tests/test_n131072_keysize.cpp`
   measured the ACTUAL per-key cost at n=131072 (not extrapolated):
   1085MB RSS, 7.5s keygen time per rotation key (vs. 62.9MB/0.64s at
   n=8192). `tests/test_boot_n131072.cpp` is a parameterized copy of
   `test_boot.cpp` (default n=8192 test untouched) with `n=131072`,
   `LST=4` (lgS=16 needs a divisor other than n=8192's LST=3; LST=4
   keeps the same RST=4 per-stage radix the original choice was tuned
   around — NOT independently re-validated for numerical accuracy at
   this scale).

   **What's NOT done**: three full-run attempts, with `KROT_CAP`/
   `KSC_CAP` tightened each time (96/48 → 4/3 → 2/1), ALL OOM'd before
   completing even C2S stage 0 (confirmed: zero stdout/stderr ever
   printed — not even the first stage-completion line). The first
   failure is explained (left `KSC_CAP` unbounded, same ~1GB-per-entry
   magnitude as `KROT` at this scale — see `MEMORY_FEASIBILITY.md`).
   The second and third are NOT explained: attempt 3's observed RSS
   (up to 4.5GB) exceeded its own theoretical two-cache ceiling (~3.3GB)
   substantially, meaning something ELSE grows with n that neither cap
   bounds. **Do not guess at cache constants again as the next step** —
   add per-diagonal `[mem]` instrumentation inside the C2S/S2C stage
   loops first (the existing checkpoints only print at stage
   boundaries, never reached in any of the three attempts) to actually
   localize the growth before touching `KROT_CAP`/`KSC_CAP` further.
   Full detail, including the three-attempt table: `MEMORY_FEASIBILITY.md`.

## What NOT to do

- Do not re-attempt the OLD hoisting approach (permuting decomposed
  digits directly) — falsified via direct debug test AND via literature
  confirming the correct order is different (permute the QP-basis OUTPUT,
  not the digits).
- For experiment #3's key-generation convention: permuting the payload
  secret `s2` by `k^-1` while keeping `pkA_QP`/`pkB_QP` unpermuted is
  WRONG (100% mismatch, confirmed real, not a test artifact — this one
  really does encrypt the wrong operand). Permuting `pkA_QP`/`pkB_QP` by
  `k^-1` while keeping the payload as the unpermuted base secret is
  CORRECT (see priority queue item 1 above) — do not re-litigate this,
  move on to building the multi-rotation decompose-sharing mechanism.
- Do not switch to a uniform ternary secret — corrected security analysis
  shows it doesn't help for this codebase's circuit. Keep h=64.
- Do not re-attempt experiment #1 (resident-rotation context/stream
  hoisting) expecting a different result — properly validated as a null
  result via interleaved A/B post-thermal-equilibrium.
- Do not modify `keyswitch_resident.cpp`'s actual logic for experiment #3
  — the safe seam (`ks_accumulate_qp`/`ks_moddown`) already exists and is
  verified zero-behavior-change; keep building on top of it, don't touch
  its internals.
- Do not use CKKS for messaging — `MESSAGING_ARCHITECTURE.md` stands.

## Standing environment facts

RTX 2060 Max-Q, 6144MB VRAM, CUDA 13.2, driver 595.97, WSL2 Ubuntu, 10GB
host RAM visible to WSL2, 16 CPU threads. GPU clock throttling under WSL2
and thermal drift over long sessions are both real, measured, not fixed.
