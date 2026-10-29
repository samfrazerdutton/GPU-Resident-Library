# Master state

Last updated: this session, FOURTH execution-focused cycle (production
integration round, on top of commit `02e84eb`). **Read
`## This round: production integration, honest A/B, security re-review`
near the end of this file first** — it supersedes several claims in the
body below (notably the "classical security is solid" framing and the
un-corrected-for-order-bias performance numbers). The body below is
preserved for history and is otherwise still accurate.

Prior summary (third execution-focused cycle): delivered TWO real,
validated wins (KSC bound, eval-domain automorphism), one safe
prerequisite refactor (keyswitch_resident split), and attempted a third,
riskier optimization (correct-order rotation hoisting) that failed twice
on the key-generation convention specifically -- documented precisely
rather than abandoned or faked.

## Current verified state

- Repo: `github.com/samfrazerdutton/GPU-Resident-Library`, branch `main`,
  commit `2080aa0` (pushed to origin as of the prior checkpoint; confirm
  `git log -1` and push status on resume).
- Hardware: RTX 2060 Max-Q, 6144MB VRAM, CUDA 13.2, driver 595.97, WSL2
  Ubuntu, 10GB host RAM visible to WSL2, 16 CPU threads.
- Current demo bootstrap (n=8192, log2(QP)=1775, sparse secret h=64):
  **108-114s wall time** (down from 125-150s at the start of this round;
  see eval-domain automorphism below) — **still only ~51 classical / ~46
  quantum bits of security. Never secure.** Every speed number describes
  this insecure demo, not a security result.
- **Peak host RSS: 8.12-8.32GB** (down from 10.78GB — see KSC fix below).

## This cycle's real, validated outcomes

1. **Benchmark methodology fixed.** Built `research/benchmarks/run_trials.sh`
   and `interleaved_ab.sh` — repeated-trial harnesses that log GPU
   temp/clock/power per run and support true interleaved A/B comparison.
   Confirmed empirically: this session's GPU reached a stable thermal
   plateau around 85°C under sustained load (vs. ~78-82°C early-session),
   which shifts absolute wall-clock numbers run-to-run — use interleaved
   comparison, not sequential single runs, for any future A/B.

2. **Resident-rotation experiment (#1): validated NULL result, not
   inconclusive.** Interleaved A/B (3 rounds, separate `test_boot`/
   `test_boot_exp` binaries) showed apparent slowdowns in rounds 1-2 that
   were an order confound (experiment always ran second, inheriting the
   preceding run's heat). **Round 3, after thermal equilibrium: baseline
   81.94s vs. experiment 81.88s rotate-bucket — statistically identical.**
   Hoisting stream/work/ciphertext residency out of the per-diagonal loop
   does not meaningfully help, because the unavoidable per-key eval-key
   upload (~63MB) dominates either way. Not merged (adds complexity, zero
   benefit). See `EXPERIMENT_LEDGER.md`.

3. **KSC bounded cache: real ~25% peak-RSS win, committed (`93f2231`).**
   Capped the cheap-to-regenerate rotation-key VIEW cache at 48 entries
   (LRU eviction). Measured peak RSS: 10.78GB → 8.12-8.32GB (two
   independent measurement methods), bit-exact. `KROT` (the expensive-to-
   regenerate full-key cache, confirmed genuinely reused across the WHOLE
   bootstrap, not per-stage) deliberately left uncapped.

4. **Security recommendation corrected AGAIN, now self-consistently
   verified.** The previous "n=65536 + uniform ternary" recommendation
   reused the SPARSE secret's modulus budget (1775 bits) without
   accounting for the fact that this codebase's own EvalMod circuit needs
   MORE double-angle depth for a uniform (vs. sparse) secret — Kbud scales
   with √n for uniform secrets, adding 3-5 extra towers depending on n.
   Recomputed self-consistently: **n=65536 uniform only reaches 113 bits
   (short of 128). n=131072 with the EXISTING sparse secret (h=64, zero
   code change) reaches 134 bits using the exact same, already-validated
   circuit.** See `SECURITY_ASSESSMENT.md` for the full corrected analysis
   — this supersedes the prior revision's recommendation.

5. **Eval-domain automorphism: real win in `rotate_ct_resident` itself,
   committed (`0043af3`) — but this round found that function is NOT
   actually what `test_boot`'s hot loop calls, correcting a mis-
   attributed measurement.** Derived and INDEPENDENTLY validated (GPU-free
   Python reference, 20 trials) that this codebase's exact NTT convention
   makes Galois automorphism a pure eval-domain permutation (no sign flip,
   no modular arithmetic) — replaced the INTT→permute→NTT round trip with
   one gather kernel in `rotate_ct_resident`. Caught a real async-host-
   buffer-lifetime bug via the standalone test before it reached
   production. 40/40 bit-exact at n=8192.

   **Correction (this round):** direct code reading found `tests/
   test_boot.cpp`'s actual C2S/S2C hot loop never calls `rotate_ct_resident`
   at all — it has its own separate, already-existing `StageRotCtx`/
   `stageRotDiag` mechanism (stage-resident base ciphertext + stream,
   predates this session) that was still calling the OLD, slow
   `automorphism_eval_device` directly. The "rotate bucket down to 55-60s
   (was 62-82s), total bootstrap down to 108-114s" figure previously
   recorded here could not have come from this fix, since the hot loop
   never exercised the changed function — that number reflected something
   else (likely ordinary thermal variance, mis-attributed). Fixed
   `stageRotDiag` to call `automorphism_eval_fast_device` (and resized its
   scratch buffer from n-sized to towers*n-sized, which the fast path
   requires — the old n-sized buffer would have been a silent out-of-
   bounds bug if swapped in naively). Verified bit-exact (`max slot err =
   5.47871e-4`, identical to before) via a temporary side-by-side build of
   the pre-fix function, interleaved: **old=116.4s (1 run); new=114.3s,
   110.3s (2 runs) — a real but SMALL ~2-6s total / ~3-7s rotate-bucket
   improvement, not the dramatic figure previously claimed.** A clean
   same-run-position comparison (old run #2) was attempted but hit
   OOM three times in a row near this machine's ~10GB ceiling (test_boot's
   own ~8.1GB peak RSS leaves only ~2GB margin) — GPU/host memory were
   confirmed clean between attempts (not a stuck resource), so this is
   accepted as genuine near-ceiling flakiness, not a bug. The temporary
   comparison binary and its CMakeLists.txt entry were removed after use
   (not part of the lasting codebase).

6. **keyswitch_resident split into ks_accumulate_qp + ks_moddown,
   committed (`2080aa0`).** Safe mechanical extraction (zero behavior
   change, verified via the full regression suite), creating the seam
   needed to attempt correct-order rotation hoisting.

7. **Correct-order rotation hoisting (experiment #3): single-rotation
   correctness RESOLVED this round (commit `46998b7`).** The "100%
   mismatch" recorded in the previous revision of this file was a TEST
   BUG — comparing raw ciphertexts for bit-exactness instead of decrypted
   plaintexts within CKKS's noise budget (full explanation:
   `EXPERIMENT_LEDGER.md` entry #14). Fixed the test to decrypt-and-
   compare; the key-generation convention already tried (permute
   `pkA_QP`/`pkB_QP` by `k^-1`, keep payload as the unpermuted base
   secret) was correct all along. Expanded to 3 independently-encrypted
   random ciphertexts x 9 rotation indices (incl. k=M-1, k=M/2-1 edge
   cases): **27/27 pass**, stable ~2000-3300 decrypted-coefficient error
   against a ~1e14-1e15 signal (ordinary keyswitch noise). Promoted
   `tests/test_hoist_single.cpp` into the default test suite. **This is
   single-rotation correctness only — zero performance benefit by
   itself.** The actual win (sharing one `decompose(c1)` across multiple
   rotation indices in a bootstrap stage) is a separate mechanism, not
   yet built — see `NEXT_SESSION.md` priority queue item 1.

8. **Multi-rotation decompose-sharing: DONE, validated on ALL 3 real
   C2S stages of a live bootstrap, not just one stage or synthetic
   benchmarks.** New, separate files `include/keyswitch_hoist.h` +
   `src/keyswitch_hoist.cpp` (`keyswitch_resident.cpp` untouched).
   `tests/test_hoist_multi.cpp`: bit-exact vs. the existing fused
   `ks_accumulate_qp` (0/524288 mismatches), correct across 6 shared
   rotation indices. `bench/bench_hoist_multi.cpp`, interleaved A/B:
   N=5 ~2.04x, N=16 (matching the real C2S stage 0 bucket size) ~2.44x,
   both CIs exclude zero. `tests/test_hoist_stage0.cpp` validated the
   mechanism against test_boot.cpp's ACTUAL first C2S stage alone:
   maxDiff=613 vs maxAbs=5.76e17, **real timing: hoisted 7.53s vs
   trusted 25.89s — ~3.4x** (both sides paying real keygen cost).

   **`tests/test_hoist_allstages.cpp` then extended this to ALL 3 C2S
   stages** (sequential-pass design: trusted path's own unmodified loop
   to completion, then a separate hoisted-only pass over all 3 stages,
   so the two paths' keys never coexist in memory). Found and fixed
   TWO real bugs in sequence via careful diagnosis (gdb for the first,
   reasoning through `ks_accumulate_qp`'s own `fullQ`/`delta` indexing
   for the second — full story in `NEXT_SESSION.md`): (1) a segfault in
   `ks_context_create` from a hand-built shape-only key with empty
   `av`/`bv` (fixed by reusing the trusted path's own `atLevel()`
   pattern; added `tests/test_hoist_levelreduced.cpp` as a permanent
   regression test, now in the default suite); (2) a correctness bug
   from generating the hoisted key directly at a reduced tower count
   instead of building it full-size and taking an `atLevel` view (same
   root cause class as #1 — the trusted path's two-step key pattern
   exists for exactly this reason, and skipping it broke silently for
   any level other than the full one). **Result after both fixes:
   `[PASS]` across all 3 real C2S stages, maxDiff=35 against
   maxAbs=5.26e15 (negligible, ~6.7e-15 relative) — the strongest
   correctness evidence yet.** Timing from this run (0.93x, pass 2
   marginally slower) is NOT a speedup regression — this benchmark's
   fair, no-cache-advantage design makes the hoisted side pay fresh
   keygen every diagonal while the trusted side's own `KSC` view-cache
   makes most of ITS diagonals cheap re-copies, conflating "decompose-
   sharing" with "caching strategy." The earlier N=5/N=16/stage-0-alone
   results (both sides fresh-keygen, apples-to-apples) remain the valid
   speedup evidence. Not yet wired into the live, production bootstrap
   (needs a parallel hoisted-key cache analogous to `KSC`, since the
   hoisted convention is structurally different from today's `KROT`'s
   textbook convention) — scoped in `NEXT_SESSION.md` priority queue
   item 1.

9. **`KROT` memory-scaling: lever analysis DONE, bounded-cache safety
   net implemented.** Zero-copy ruled out with direct code evidence
   (`ks_context_create` re-uploads+frees av/bv to device on every call —
   no persistent device copy exists to deduplicate). BSGS/LST retuning
   and numPart/alpha retuning both rejected for now (both collide with
   the fixed 1775-bit security budget; alpha retuning additionally
   surfaced a real, previously undocumented pre-existing bug at alpha=10,
   `tests/test_alpha10.cpp`). `KROT_CAP=96` LRU bound added to
   `tests/test_boot.cpp` (same pattern as the merged `KSC_CAP` fix),
   verified ZERO behavior change at n=8192 (identical `max slot err =
   5.47871e-4`, identical ~8.11GB peak RSS — cap never triggers, working
   set stays at 77). This is the safety net for n=131072, not yet a
   working fix at that scale — see `MEMORY_FEASIBILITY.md`'s "KROT lever
   analysis" section for the measured regen-cost tradeoff numbers.

10. **Quantum-adjusted hybrid security estimate: derived, with an
    honest split verdict — the standing "`hybrid_security.py` has no
    quantum variant" gap is resolved, partially, not glossed over.**
    Both user-supplied reference PDFs actually read (not assumed):
    "Mosca" resolved to the Kaye/Laflamme/Mosca QC textbook (explicitly
    disclaims covering quantum cryptography), Nielsen & Chuang is the
    full comprehensive reference. Both verified to cover only the
    generic single-list Grover speedup — neither contains the
    specialized two-list quantum meet-in-the-middle algorithm
    (Brassard-Hoyer-Tapp 1997) this model's MITM phase actually needs.
    Implemented `hybrid_quantum()` (classical `hybrid()` untouched) with
    two honestly-separated bounds instead of one invented number: BKZ
    phase gets the same quantum sieve exponent `lwe_security.py` already
    uses; MITM phase stays classical by default (naive double-Grover
    would double-count a speedup that isn't there), with an opt-in
    BHT-exponent bound requiring a large-QRACM attacker (a real,
    contested assumption). New `tools/test_hybrid_security.py`, 10/10
    pass. **Real result for n=131072/h=64: 134 classical (unchanged) /
    130 quantum conservative (2-bit margin over 128 — thin) / 114
    quantum if large QRACM is credited (FAILS 128).** A genuine finding,
    not an artifact: full derivation and citations in
    `SECURITY_ASSESSMENT.md`.

## n=131072: real numbers now known, three real attempts made, still blocked by an UNIDENTIFIED cause

Not GPU VRAM (181MB of 6144MB used at n=8192 — huge headroom). Real,
MEASURED (not extrapolated) per-key cost at n=131072 this round:
**1085MB RSS, 7.5s keygen** per rotation key (`tests/
test_n131072_keysize.cpp`) — confirms the ~1GB/key, ~71-77GB-for-76-keys
estimate used throughout earlier revisions of this file was accurate,
not overblown. `KROT_CAP` (added this round) and `KSC_CAP` (both now
bounded — `KSC_CAP` was the ORIGINAL oversight: it caches full-size
`KROT`-entry-sized views and was left at its n=8192 default of 48,
which alone could reach ~48GB at this scale) together make the KEY
CACHES bounded and feasible in principle.

**But three actual end-to-end attempts this round, with progressively
tighter caps (96/48 → 4/3 → 2/1), ALL OOM'd before completing even C2S
stage 0** (confirmed: zero progress output ever printed). The tightest
attempt's observed RSS (up to 4.5GB) exceeded its OWN theoretical
two-cache ceiling (~3.3GB) substantially — **something besides
`KROT`/`KSC` also grows with n and is not yet identified.** Do not
re-attempt this by further shrinking cache constants; the next step is
per-diagonal `[mem]` instrumentation to actually localize the growth
(full detail and the three-attempt table: `MEMORY_FEASIBILITY.md`).
**n=131072 end-to-end has NOT been run successfully. Do not report it
as supported — this is a harder-than-modeled problem, not a solved one
with an untested implementation.**

## What's done vs. not done against the acceptance criteria

- [x] Clean build from documented instructions (confirmed repeatedly this
  session).
- [x] Core correctness tests pass (bit-exact, 14+ consecutive changes
  this session with zero numerical drift).
- [x] Full supported bootstrap executes end to end (at n=8192 demo params).
- [x] Reproducible before/after measurements for the KSC fix (two
  independent measurement methods, interleaved-confirmed null result for
  the rotation experiment).
- [x] Security parameters have an independent, self-correcting documented
  assessment (two real corrections made and recorded, not hidden).
- [x] Host RAM measured precisely; VRAM measured (with an acknowledged
  coarse-polling caveat).
- [ ] The selected (n=131072, sparse h=64) configuration has NOT yet been
  run end to end — blocked on KROT's memory scaling, not yet fixed.
- [x] Eval-domain automorphism permutation (ranked experiment #2) — DONE,
  real win, committed `0043af3`.
- [x] Correct-order rotation hoisting (ranked experiment #3), single-
  rotation correctness — DONE, `46998b7` (see above and
  `EXPERIMENT_LEDGER.md` entry #14).
- [x] Correct-order rotation hoisting, multi-rotation decompose-sharing
  — DONE. `include/keyswitch_hoist.h`+`src/keyswitch_hoist.cpp` (new,
  separate files; `keyswitch_resident.cpp` untouched). Bit-exact vs. the
  fused path (`tests/test_hoist_multi.cpp` check 1), correct across 6
  shared rotation indices (check 2), and a REAL measured speedup:
  **N=5 rotations, baseline median 94.49ms vs. hoisted median 46.32ms
  (~2.04x), bootstrap 95% CI on the paired difference [47.86,48.60]ms,
  excludes zero** (`bench/bench_hoist_multi.cpp`, interleaved A/B, 30
  reps). Not yet wired into the real bootstrap's C2S/S2C stages — that
  end-to-end measurement is the next step, see `NEXT_SESSION.md`.
- [x] `KROT` memory-scaling lever analysis — DONE (zero-copy ruled out
  with code evidence, BSGS/alpha retuning rejected for now, bounded-LRU
  `KROT_CAP` implemented as the safety net, verified zero behavior
  change at n=8192). NOT yet a fix for n=131072 — that needs the cap
  actually lowered and the regen-cost tradeoff measured at that scale.
  See `MEMORY_FEASIBILITY.md`.
- [ ] Messaging track: architecture proposal complete and committed
  (`MESSAGING_ARCHITECTURE.md`); no implementation/prototype built (not
  requested as a build task, scoped as a proposal this session).

## Environment constraints (confirmed, not assumed)

- GPU clock throttling under WSL2 and thermal drift over long sessions are
  both real and measured. Will not attempt privilege escalation to force
  clock locking.
- Host RAM ceiling (10GB in WSL2): confirmed binding; now partially
  relieved (25% reduction via the KSC fix) but KROT's own scaling at
  n=131072 remains unsolved and is the standing blocker.

## This round: production integration, honest A/B, security re-review

Built directly on `02e84eb` (correct-order rotation hoisting, proven
across all 3 C2S stages; quantum-adjusted security estimate derived).
Three pieces of real work, one still-open gap, honestly reported.

### 1. Hoisting wired into the live production bootstrap, behind a flag

`tests/test_boot.cpp` gained a `KROT_hoist` cache (same bounded-LRU
pattern as `KROT`/`KSC`) plus `HoistedStageCtx`/`hoistedStageBegin/Diag/
End` lambdas, feature-flagged via `constexpr bool
HOIST_ROTATION_ENABLED` (false in `test_boot.cpp`; a new parallel file
`tests/test_boot_hoist_enabled.cpp` sets it true — same CMake target,
NOT added to the default ctest run). `src/keyswitch_resident.cpp` was
not touched. Both the C2S loop and the S2C loop were wired to the
hoisted path, each with per-diagonal AND per-stage CUDA-error fail-safe
fallback to the trusted `StageRotCtx`/`stageRotDiag` path (`cudaOK()`
checks `cudaGetLastError()` and logs `[hoist-fallback]` before falling
back).

**Flag-off behavior is proven byte-identical to the pre-integration
baseline**: `max slot err = 0.000547871` matched exactly across all 4
flag-OFF trials in the A/B series below, and peak RSS matched (~8.3GB).
This is the correctness gate for "the fallback path is untouched," and
it passed.

**Real bug found and fixed mid-round: cross-phase cache-sharing side
effect.** The trusted path's `KROT` cache is populated during C2S and
then REUSED for free by S2C (confirmed: KROT size is flat from C2S's end
through S2C's end). A first attempt hoisted ONLY C2S, diverting its
keygen into the separate `KROT_hoist` cache and leaving the original
`KROT` empty — so S2C (still trusted-path) lost its free cache hits and
paid full fresh keygen per diagonal, ballooning 29.1s→77.8s and erasing
C2S's own real gain at the full-bootstrap level (167.8s vs 132.9s
baseline — SLOWER overall despite C2S alone being faster, 73.7s vs
86.3s). Fix: extend the identical hoisting pattern to S2C's `q2`-loop so
C2S and S2C share ONE `KROT_hoist` cache. After the fix, a single
end-to-end run reached 111.154s vs a 132.937s baseline — a real
improvement in that one comparison — but see the A/B series below for
why that single comparison should not be trusted as "the" speedup
number.

Cache telemetry from that run: `KROT_hoist` size=75, hits=... ,
hit-rate=58.3%, 0 evictions, ~0.595s/key keygen cost — a working,
correctly-instrumented cache (hit/miss/eviction counts all behave as
expected under repeated execution).

### 2. Honest, order-balanced interleaved A/B measurement — small, noisy effect, NOT a clean win

Directive required fair, interleaved, order-controlled A/B measurement
with reported sample count/median/variability — not a single comparison.
Ran 8 full-bootstrap trials (`test_boot` = OFF, `test_boot_hoist_enabled`
= ON), all passing correctness (`max slot err` matched the established
baseline exactly in every trial):

| # | order pos | flag | wall time |
|---|-----------|------|-----------|
| 1 | 1st | OFF | 124.203s |
| 2 | 2nd | ON  | 112.957s |
| 3 | 1st | OFF | 113.855s |
| 4 | 2nd | ON  | 110.372s |
| 5 | 1st | OFF | 108.375s |
| 6 | 2nd | ON  | 113.016s |
| 7 | 1st | ON  | 108.673s |
| 8 | 2nd | OFF | 108.504s |

First 3 pairs ran OFF-then-ON (paired diffs, OFF−ON: +11.246, +3.483,
−4.641). This is the SAME order-confound class this session's earlier
resident-rotation experiment was burned by: the condition that runs
SECOND within a pair inherits extra GPU warm-up, biasing it to look
faster. Trial 7-8 reversed the order (ON-first, OFF-second) specifically
to correct for this, and the diff collapsed to near zero (OFF−ON =
−0.169s).

**Aggregate, n=4 OFF / n=4 ON**: OFF mean=113.734s (stdev=7.432, range
108.4–124.2), ON mean=111.255s (stdev=2.117, range 108.7–113.0). Paired
diff median=+1.657s, mean=+2.480s, **stdev of the paired diff
(6.722s) exceeds the mean diff** — the apparent ON-is-faster effect is
NOT clearly distinguishable from run-to-run/order/thermal noise at this
sample size. The single order-balanced pair (7↔8) that most directly
isolates the true effect from the warm-up confound shows it near zero.

**Explicit order-effect decomposition (requested follow-up check:
"does order itself have a systematic effect, separate from the flag").**
The composition is 3 OFF-first pairs + 1 ON-first pair (not a balanced
2-and-2 design — only one pair breaks the order/flag confound). Computing
(2nd-run time − 1st-run time) regardless of which flag ran in which
position: `[−11.246, −3.483, +4.641, −0.169]` → median=−1.826s,
mean=−2.564s, stdev=6.680s. There IS a consistent-direction tendency (3
of 4 pairs) for whichever condition runs SECOND to finish faster,
matching the GPU-warm-up hypothesis — but its own stdev also exceeds its
mean, so it is not crisply established at n=4 either. Because OFF ran
first and ON ran second in 3 of the 4 pairs, order and flag are
confounded there and the apparent "ON is faster" signal in those three
pairs cannot be attributed to the flag with confidence. The single pair
that breaks the confound (7↔8: ON first, unfavored position; OFF
second, favored position) showed ON essentially matching OFF
(108.673s vs 108.504s, diff 0.169s) despite NOT having the order
advantage — consistent with either "no real flag effect" or "a small
real flag effect that roughly cancels the order disadvantage"; this
one data point cannot distinguish between those two readings. **A
properly randomized (not block-alternated) trial series with more
samples, ideally with GPU clock/temp logged per trial via `nvidia-smi`,
is required to resolve order vs. flag — not yet done.**

**Honest verdict: the hoisted path has NOT been shown to deliver a
clean, reproducible end-to-end speedup once order bias is controlled
for.** The ~16% win from the single first post-fix comparison
(111.154s vs 132.937s) was almost certainly inflated by factors this
8-trial series does not reproduce (likely a cold/warm cache or thermal
asymmetry in that specific one-off comparison, not a property of the
mechanism itself). Per the directive's explicit instruction, this is
reported as-is rather than cited as a win. **Performance acceptance
criteria have NOT cleanly passed — `HOIST_ROTATION_ENABLED` must stay
`false` in `test_boot.cpp` and `test_boot_hoist_enabled` must stay
un-promoted (not in the default `gpufhe_test`/ctest run).** A larger,
fully-randomized (not block-alternated) trial count, with GPU clocks
observed via `nvidia-smi` per trial, would be needed to resolve whether
a small real effect exists under the noise.

### 3. Security re-review: adversarial check found the prior "130-bit conservative" framing overconfident

A delegated adversarial review of `02e84eb`'s quantum-adjusted estimate
(`research/SECURITY_ASSESSMENT.md`, `## Independent adversarial review`
section) found two real issues, independently fact-checked via WebFetch
against the cited papers: (a) the classical 134-bit hybrid-attack
calibration is ~3-6 years stale against newer attack-family papers
(eprint 2021/152, eprint 2026/1060 — the latter explicitly FHE-targeted)
never checked against these exact parameters, plausibly costing 2-18
more bits; (b) the "130-bit conservative quantum" bound is internally
inconsistent — it credits QRAM to the sieve phase but declines to credit
it to MITM, when arXiv 2410.15565 shows the sieve speedup also needs
QRAM, so 130 bits does not describe a coherent attacker.
**`SECURITY_ASSESSMENT.md`'s `## Disposition` section has been updated
to state this as the current operative position: n=131072/h=64 must NOT
be presented as production-secure; the honest range is 114-134 bits
depending on unresolved model assumptions ("insufficient evidence either
way"), and the 114-bit large-QRAM figure remains a live, material
warning, not a resolved footnote.**

### What's still open

- **n=131072 feasibility/performance test (directive section 6): NOT
  attempted this round.** The pre-existing, still-unresolved state from
  before this round (3 failed OOM attempts, an unidentified
  memory-growth source beyond `KROT`/`KSC`, documented in
  `MEMORY_FEASIBILITY.md`/`NEXT_SESSION.md`) was read but not acted on —
  this round's time went to the production-integration/benchmark/
  security work above instead. **Exact next action**: add per-diagonal
  `[mem]` instrumentation inside the C2S/S2C stage loops (the existing
  checkpoints only print at stage boundaries, never reached in any of
  the three n=131072 attempts) to localize the growth before touching
  cache-capacity constants again.
- A larger, truly-randomized A/B trial series (not block-alternated) to
  settle whether hoisting's end-to-end effect is real-but-small or zero.
- The SageMath lattice-estimator cross-check and the direct
  eprint-2021/152 / eprint-2026/1060 parameter check named in
  `SECURITY_ASSESSMENT.md`'s updated Disposition.

## Later round: n=131072 memory investigation — cache mechanism cleared, real root cause found to be host physical RAM

A dedicated round on the standing n=131072 feasibility blocker (full
detail: `MEMORY_FEASIBILITY.md`'s final `## UPDATE` section;
condensed: `EXPERIMENT_LEDGER.md` entry #22). Three results:

1. **New isolated experiment (`tests/test_n131072_cache_churn.cpp`)
   definitively clears the KROT/KSC bounded-LRU cache mechanism as the
   runaway OOM cause.** Drove 24 real rotation-key generations through
   the same `KROT_CAP=2`/`KSC_CAP=1` logic as the real bootstrap, no
   ciphertext/ModRaise machinery involved: RSS plateaus at exactly
   3233MB from the 3rd call onward, through 22 KROT evictions and 11
   KSC evictions — zero growth. Confirms a real, previously
   undocumented mechanism (the actual code generates each new key
   BEFORE evicting the old one, so the true transient peak is `cap+1`
   keys, not `cap`) and refutes allocator-retention growth across
   repeated evictions.
2. **Enhanced instrumentation (GPU memory via `cudaMemGetInfo`, a probe
   right after `stageDiags`) localized the 5th real attempt's crash
   precisely**: `stageDiags` for ONE stage took a measured 268 seconds
   (confirming an O(S²) algorithmic concern flagged in an earlier
   round, now quantified — ~35 CPU-minutes for all 8 C2S+S2C stages at
   n=131072), and RSS was a safe 1436MB immediately before the crash,
   which happened inside the very first `keyAt()`/`stageRotBegin()`
   call — well short of any modeled danger threshold.
3. **Root cause: this machine's total physical RAM (~15.42GB), not
   WSL2's self-reported "10GB ceiling."** `.wslconfig` caps WSL2 at
   11GB, but Windows itself already uses ~11.8GB at near-idle (only
   ~3.6GB free) — WSL2's internal `free -h` reflects a *promised*
   Hyper-V dynamic-memory allocation, not memory the host can actually
   back under real contention. **This reframes all of this session's
   recurring OOM/stall friction (not just n=131072) as likely rooted in
   genuine host-level memory pressure, not purely WSL2-internal
   accounting.** Treated as an environment blocker, not an
   implementation bug: next step is retrying with other Windows
   processes closed to test this diagnosis directly, not further
   cache-constant tuning (already proven not the limiting factor).

## Later round still: kill mechanism precisely characterized; diagonal bottleneck fixed; 6th attempt reaches 8x further than ever before

Full detail: `MEMORY_FEASIBILITY.md`'s "UPDATE (next round)" section;
condensed: `EXPERIMENT_LEDGER.md` entry #23. Three results:

1. **Kill mechanism investigated directly** (`dmesg`/`journalctl`, twice,
   for two separate kills): PROVEN not a Linux kernel OOM-kill (zero
   matching log lines across the whole VM uptime) and PROVEN not an
   uncaught C++ exception/signal (WSL's crash-capture mechanism, shown
   to work via an unrelated real example, logged nothing for either
   kill). Well-supported but not directly provable: an external kill
   from the Windows host side (most likely this session's own tool
   harness). `ru_maxrss` confirmed (man page) to be a monotonic
   high-water mark, so it cannot be hiding a larger true memory use.
2. **The independently-flagged `stageDiags`/`stageDiagsFwd` O(S^2)
   bottleneck is fixed**: a new standalone test
   (`tests/test_c2s_diag_fast.cpp`) proves an O(S*2^RST) analytic
   replacement exactly equivalent to the old brute-force algorithm
   across 18 (n,LST,g,direction) combinations (RST=1..13, both
   directions), with measured speedups 2.5x-332x. Wired into
   `tests/test_boot_n131072.cpp` only (same signatures, `tests/
   test_boot.cpp` untouched). On a real run, `stageDiags` for one stage
   dropped from the previously-measured 268.047s to 0.110205s.
3. **A 6th real n=131072 attempt reached diagonal 8 of 16** (every
   prior attempt died within 1-2 diagonals) with host RSS plateaued at
   EXACTLY 4561.35MB for 7 consecutive diagonals — directly confirming,
   inside the real bootstrap, what the prior round's isolated
   cache-churn experiment showed synthetically. A small, unresolved
   GPU-memory climb (+0.5-1MB/diagonal over diagonals 5-8) is the one
   remaining open thread; it was observed over too few of the 128 total
   diagonals a full run needs to classify as growing or plateauing. The
   process was still killed, again with zero kernel trace. **n=131072
   end-to-end remains incomplete, but the diagnosis is now
   substantially sharper on every front this round touched.**

## Later round still: GPU-memory lifecycle isolated with real production APIs; external-kill correlation directly demonstrated; NO-GO decision for the next full attempt under current host conditions

Full detail: `MEMORY_FEASIBILITY.md`'s "UPDATE (next round)" section;
condensed: `EXPERIMENT_LEDGER.md` entry #24. Built
`tests/test_n131072_gpu_lifecycle.cpp`, driving the real
`ks_context_create`/`keyswitch_resident`/`ks_context_destroy` per
diagonal and `ks_work_create`/`ks_work_destroy` per stage lifecycle
(the same bounded-LRU KROT cache as production), isolated from the rest
of the bootstrap. Three results:

1. **Host RSS proven flat across 68 diagonals and 3 stage transitions**
   (exactly 4193MB, zero drift) — the most robust confirmation yet that
   KROT/KSC and the host side of this lifecycle are bounded.
2. **GPU memory shows two separable patterns**: a tiny, linear
   ~0.53MB/diagonal within-stage drift (matching the real run's earlier
   observation), and a much larger ~200MB NET growth per STAGE
   TRANSITION that is NOT explained by `ks_work_create`/`destroy`
   (pairs cleanly) or any identified host-side cache. Best-supported
   hypothesis: CUDA device-allocator pool retention across
   differently-sized per-stage allocations — not independently
   confirmed, since `compute-sanitizer` and per-process `nvidia-smi`
   tracking are both unavailable in this WSL2 CUDA-passthrough setup
   (checked directly). Extrapolating linearly across a full 8-stage run
   projects to ~3-3.5GB GPU usage — comfortably under the 6GB VRAM
   card, meaning this growth is unlikely to be the binding constraint
   by itself.
3. **External-kill correlation directly demonstrated, not just
   inferred**: a second, cheaper run (8 stages, 3 diagonals each) was
   killed after only 2 diagonals — but those 2 diagonals' RSS/GPU
   values were bit-for-bit identical to the first run's first 2.
   Checked real host memory: 5.41GB free after the first run's kill vs.
   only 3.66GB free before the second run's launch. The identical
   internal process state survived wildly different lengths purely as
   a function of external host memory at launch time.

Also fixed one real, confirmed, tiny (~128 bytes/call) leak found via
direct code reading: `ks_context_destroy` never freed `C.d_keyRow`.
Three to four orders of magnitude too small to explain the
~200MB/transition growth, but genuine — fixed and verified via
`test_ks_resident`/`test_rot_resident`/`test_hoist_single` (bit-exact)
plus `test_c2s_diag_fast` (unaffected).

**Decision: NO-GO for a 7th full n=131072 attempt right now** — real
host free memory was only 2.49GB at the time of this writeup, well
below the ~6-7GB window that let this workload get furthest in every
attempt that succeeded partially. Explicit criterion recorded for a
future attempt: real Windows host free memory >=6GB, checked via
`Get-CimInstance Win32_OperatingSystem` immediately before launch.

## GOD PROMPT round: large-n gate re-checked (still NO-GO), bench_evalmult_resident root-caused and fixed, test_polyeval investigated (inconclusive, reverted)

Full detail: `EXPERIMENT_LEDGER.md` entry #25. Re-checked the host
memory gate first (3.12-3.6GB free throughout this round, below the
6GB threshold) — no full bootstrap attempted, consistent with the
standing decision.

**`bench_evalmult_resident` fixed.** Binary-search instrumentation
localized its `ShadowRegistry cudaMalloc failed` abort to inside pure
OpenFHE CPU calls (`cc->EvalMult`/`cc->RescaleInPlace`), with every one
of this repo's own CUDA checkpoints clean up to that point. `ldd`+
`strings` traced it to an EXTERNAL dependency: this repo's "CPU-only
ground-truth oracle" OpenFHE build secretly links a separate GPU-
acceleration fork (`~/openfheNVDIA-GPU`) whose `ShadowRegistry`
singleton transparently mirrors CPU math onto the GPU — colliding with
this program's own concurrent direct CUDA usage. Fixed with a one-line
`setenv("OPENFHE_GPU","0",1)` (an env-var switch found in that same
external library) that makes the OpenFHE comparison genuinely CPU-only
— zero change to any cryptographic parameter, verified working at both
N=1 and the default N=16.

**`test_polyeval` investigated, not fixed.** The natural hypothesis
(RNS-prime rounding drift in this test's hand-rolled Chebyshev scale
bookkeeping) was directly measured and REFUTED — real primes deviate
from 2^50 by only ~1e-10 relative, far too small to explain the
observed 2377x error. An explicit scale-equalization fix was
implemented, reduced the error 2377→0.55, but hand-tracing found it
introduces its own overflow bug (a correction ratio compounds to
exactly the fix's encoding constant by the 3rd Chebyshev level,
overflowing the RNS moduli when encoded again) — the improvement was
likely coincidental, not genuine. **Reverted** rather than keep an
unverified fix. True root cause remains open; a specific next
hypothesis (keyswitch breakdown at the very low tower counts this
test's deep chain reaches) is recorded in `NEXT_SESSION.md`, not yet
tested.
