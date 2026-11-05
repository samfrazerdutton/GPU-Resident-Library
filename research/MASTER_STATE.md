# Master state

Last updated: this session, GOD PROMPT "CKKS PHASE 5" round (real
encode/encrypt/decrypt proven; the real encrypted EvalMult round trip
found BLOCKED, on top of commit `6a3cf64`). **Read `## GOD PROMPT "CKKS
PHASE 5"` near the end of this file first**, and `research/
MIGRATION_MATRIX.md` for the condensed per-operation status table
(including the new "Real encrypted round trip" column — do not confuse
it with "Independently tested," which is a weaker, RNS-level-only
claim). Everything below that is preserved for history; the
superseded-claims note this paragraph used to carry (about the
"production integration round") is itself superseded by the Rust/CUDA
evolution rounds — the main C++ engine's production/security state
described in the body below is unaffected by the Rust work, which is a
separate, not-yet-production-integrated exploration (see the matrix).

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

**Library-usability finding (Priority 6, code-reading only, no
refactor attempted this round).** `include/device_ciphertext.h`'s
`DeviceCiphertext` class is a genuinely well-designed RAII primitive
(constructor/destructor pair, move-only, deleted copy, clear
ownership) — but it is used ONLY in simpler, early single-op tests and
benchmarks (`test_mult_resident`, `test_ntt_resident`,
`bench_resident_chain`, `bench_batch`, `ckks_ops.cpp`). The actual
production bootstrap/keyswitch/hoisting code this entire session's
work has focused on (`tests/test_boot.cpp`, `tests/
test_boot_n131072.cpp`, `src/keyswitch_resident.cpp`'s
`DeviceKSContext`/`DeviceKSWork`) uses manual `cudaMalloc`/`cudaFree`
pairs with raw pointers instead — the LESS safe idiom used for the
MORE complex, higher-stakes code. This is exactly the architectural
gap that produced the real, confirmed `d_keyRow` leak fixed earlier
this session (`ks_context_destroy` forgetting one of several raw
pointers it owns) — a class of bug RAII wrapping would make
structurally impossible rather than something that has to be gotten
right by hand at every call site. **Recommended smallest safe
improvement (not implemented this round, given the correctness risk of
touching the carefully-preserved, heavily-tested bootstrap path under
the current tight host-memory conditions that make full re-
verification difficult)**: wrap `DeviceKSContext`/`DeviceKSWork` in
thin RAII types (constructor calls `ks_context_create`/
`ks_work_create`, destructor calls the matching `_destroy`) as a
dedicated, carefully-scoped follow-up task with its own full
regression pass once host memory conditions allow it.

## GOD PROMPT round: test_polyeval's real root cause found — a genuine library bug in the native rescale pathway at small n, never a test-logic issue

Full detail: `EXPERIMENT_LEDGER.md` entry #26. This supersedes entry
#25's "root cause unidentified" status with a DEFINITIVE localization.
Instrumented decrypt-and-compare checkpoints added directly inside
`test_polyeval.cpp`'s `mulCt` (kept, additive-only, confirmed
byte-identical `maxerr=2376.98` before/after) proved: the tensor
product and the post-keyswitch-pre-rescale result are BOTH correct at
the exact point of first failure — the corruption appears ONLY after
`rescale_resident_raw` runs. A new minimal, independent reproduction
(`tests/test_rescale_n1024.cpp`, zero Chebyshev/relin complexity)
confirms this directly: encrypting a known constant and rescaling ONCE
at n=1024 fails (maxerr 807-24602) regardless of modulus bit-size
(tested both 50-bit and 55-bit mid-towers — both fail, ruling out
bit-size as the variable).

**This is a real, previously-unknown bug in `rescale_resident_raw`/
`native_rescale_consts`/their dependent NTT kernels, specific to small
n — not a bug in test_polyeval's own hand-rolled scale bookkeeping as
the prior round suspected.** Confirmed via code search that no test in
this 50+ test codebase exercises this exact native pathway below
n=8192 (`test_boot.cpp`) or n=32768 with OpenFHE-sourced constants
(`test_rescale_resident.cpp`, a different code path) — this bug has
simply never been caught before. `test_rescale_n1024.cpp` is committed
as a permanent, honestly-failing regression test, per the governing
directive's explicit requirement, rather than shipping any unverified
fix. The exact faulty kernel (NTT, INTT, SwitchModulus, RescaleFuse, or
`native_rescale_consts`'s own arithmetic) is not yet identified; a
specific next experiment (pure NTT round-trip at n=1024, no rescale)
is recorded in `NEXT_SESSION.md`.

Note: the first attempt at the isolated reproduction itself segfaulted
inside `ks_context_create` — traced via `gdb` to a bug in the NEW TEST
(missing `evalkeygen_host` call before `ks_context_create`, which
unconditionally reads `K.av`/`K.bv` — the exact "empty av/bv"
anti-pattern already documented from the `02e84eb` segfault fix).
Fixed in the test; not a second library bug.

## GOD PROMPT round, continued: RAII wrappers for DeviceKSContext/DeviceKSWork — audited, designed, validated in isolation, all passing; production integration deliberately deferred

Full detail: `EXPERIMENT_LEDGER.md` entry #27. Completed all three
phases the directive specified: audit every resource `DeviceKSContext`/
`DeviceKSWork` own vs. borrow (now precisely documented in
`include/ks_raii.h`'s header comment — `DeviceKSWork`'s 12 fields
confirmed fully owned and correctly freed, no leak; `DeviceKSContext`'s
`d_av`/`d_bv`/`d_keyRow` owned, 14 other fields correctly borrowed from
two process-lifetime caches and correctly NOT freed); design thin,
move-only `ScopedKSContext`/`ScopedKSWork` wrappers delegating entirely
to the proven C API; validate them in a new ISOLATED test
(`tests/test_ks_raii.cpp`, n=8192, avoiding any confound with this
round's separate small-n rescale finding). **All 7 checks pass**,
including 20 repeated create/destroy cycles with GPU memory delta=0MB
(perfectly bounded) and clean move-semantics/double-reset safety.

**Per the directive's own explicit caution, these wrappers are NOT
integrated into `tests/test_boot.cpp`/`test_boot_n131072.cpp` this
round** — that is a separate, carefully-scoped future task needing its
own full regression pass, deferred until host memory conditions allow
repeated `test_boot`-class runs (currently ~3GB free, well under the
~8GB that test needs).

## GOD PROMPT round, continued: benchmark methodology — fix scope verified, randomized A/B runner built and dry-run validated

Full detail: `EXPERIMENT_LEDGER.md` entry #28. Re-confirmed the
`bench_evalmult_resident` fix (`OPENFHE_GPU=0`) does not silence the
GPU measurement it's meant to take. Found this repo's existing
`research/benchmarks/interleaved_ab.sh` runs baseline-then-experiment
in a fixed order every round — the exact order confound this project's
own prior work was burned by twice. Built
`research/benchmarks/randomized_ab.sh`: a true random permutation
(via `shuf`) of baseline/experiment labels, run one at a time, with
its own header comment documenting why the resulting data must be
analyzed as two independent samples rather than paired by position.
Dry-run validated with trivial dummy binaries (zero GPU/memory use) —
confirmed correct shuffling and JSONL output. **Not yet run with the
real `test_boot`/`test_boot_hoist_enabled` binaries** — blocked by the
same host-memory gate as the large-n bootstrap; the exact command is
recorded in `NEXT_SESSION.md` for whenever that clears.

## GOD PROMPT "CORRECTNESS BREAKTHROUGH" round: entry #26's "rescale library bug" conclusion REVERSED — rescale_resident_raw proven correct, real cause was a test-parameter mismatch + inherent CKKS noise

Full detail: `EXPERIMENT_LEDGER.md` entry #29. This round set out to
locate and fix the rescale defect entry #26 identified. Instead, four
independent differential experiments proved the library's rescale
pathway has NO defect:

1. **`tests/test_ntt_roundtrip_n1024.cpp`** — pure coefficient→NTT→INTT
   round-trip, exact modular equality, all 8 towers of the failing
   chain, 7 input patterns each (zero/sparse/dense/boundary/random).
   **56/56 pass.**
2. Hand-derived `LaunchRescaleFuse`'s formula from `kernels/rescale.cu`
   and confirmed via Python that `native_rescale_consts`'s `s2[i]` is
   numerically identical to `(qi-s1[i]) mod qi` — the precomputed
   constants are mathematically equivalent to the textbook single-
   modulus RNS rescale. Constant-formula bug ruled out.
3. **`tests/test_rescale_scalar_diff_n1024.cpp`** (constant polynomials,
   10 CRT-exact cases including both sides of the rounding boundary) —
   **10/10 pass**, every one of 1024 slots per tower.
4. **`tests/test_rescale_vector_diff_n1024.cpp`** (independent per-
   coefficient random residues — closes the gap that constant data
   can't reveal position-mixing bugs) — **5 trials, 0/35840 mismatches.**
5. **`tests/test_rescale_double_call_n1024.cpp`** (the exact production
   call pattern: two back-to-back `rescale_resident_raw` calls sharing
   `scr`/`drp` on the same stream) — **0/43008 mismatches.**

With the arithmetic exhaustively cleared, root-caused
`test_rescale_n1024.cpp`'s actual maxerr=24602.7 failure to its OWN
parameter choice: Delta=2^50 vs. qLast≈2^55 is a ~32x scale/modulus
mismatch (sound CKKS requires the dropped modulus to approximately
match the scale being removed), which amplifies rescale's own
unavoidable ±0.5-unit rounding noise by ~32x on decode. Built
**`tests/test_rescale_n1024_matched_scale.cpp`** (Delta=2^55, matching
qLast): maxerr dropped to 769.125 — almost exactly the predicted 32x
reduction, confirming the amplification mechanism. Instrumented
diagnostics in that test then showed the *remaining* ~769 is ALSO not
a defect: tower-0-only and tower-7-only pre-rescale decryption of the
real ciphertext agree exactly (no RNS inconsistency), and the
per-coefficient discrepancy traces to comparing against an invalid
reference — CKKS rescales ciphertext components c0/c1 INDEPENDENTLY
(each gets its own rounding), so `round(c0/qL)+round(c1/qL)*s !=
round((c0+c1*s)/qL)` in general. This O(1)-per-coefficient inherent
combining noise gets amplified by `decode_host`'s n=1024-term inverse
embedding sum into the observed few-hundred-unit decode deviation.

`test_polyeval.cpp` itself uses 50-bit mid-towers matched to its own
Delta=2^50 (no Delta/qLast mismatch there) — its original
maxerr=2376.98 is consistent with this same inherent noise
accumulating across multiple chained Chebyshev-level rescales, though
this has NOT yet been separately, quantitatively confirmed for the
multi-level case — flagged as the honest residual uncertainty for a
future round.

**No code in `rescale_resident_raw`, `native_rescale_consts`, or the
NTT/SwitchModulus/RescaleFuse kernels was changed — no defect was
demonstrated in any of them.** Per the directive's Step 4, a
speculative fix was deliberately NOT shipped. All test files above are
committed as permanent regression evidence; `test_rescale_n1024.cpp`'s
own `[FAIL]` is now understood and expected (it exercises a
deliberately-mismatched parameter set from an earlier round's own
isolation experiment), not an unexplained red test.

## GOD PROMPT "RESOLVE POLYEVAL CORRECTNESS" round: every library-defect hypothesis for test_polyeval refuted by direct evidence; one formula proven wrong, not the library

Full detail: `EXPERIMENT_LEDGER.md` entry #30. Re-verified
`test_polyeval`'s baseline first (maxerr=2376.98, byte-identical;
peak RSS 131MB, no memory risk). Extended `mulCt`'s diagnostics to
cover the SECOND mulCt call (`Tk[0]*one`, both constant 1.0 — not
trivially zero at any slot, unlike the original slot-0-only check on
`x*x`) with full-array statistics.

**Chased down what first looked like a severe tensor-product defect
and found it was an invalid diagnostic, not a bug**: `decrypt_host`'s
"tower-0-only, centered" recovery is only valid when the true value's
magnitude fits within q0 (~2^60); the raw tensor product's true
magnitude is ~Delta² (~2^100), so the centered interpretation was
reading a meaningless aliased residue — the apparent "~0 instead of
~1.0" result was noise from an out-of-range comparison. Verified the
tensor arithmetic IS correct by comparing the raw, uncentered residue
mod q0 directly against an independently-computed expected value
(built from the actually-measured per-ciphertext noise, confirmed via
a new `test_encode_impulse_check` that `encode_host` produces an exact
impulse for constant input) — matched to 12 significant figures, the
expected magnitude of a negligible noise cross-term.

**Spent most of this round trying to directly verify
`keyswitch_core_resident`'s algebraic contract
(`ba0+ba1*s =?= a*s²`) and could not** — three escalating eval-domain
differential tests (random input, numPart=1 and 2, and a REAL
tensor-product input) all show ~50% of q (pure noise). Decisively
ruled out a native-vs-OpenFHE discrepancy as the cause: the identical
~50%-of-q failure reproduces even applying the SAME check to OpenFHE's
own bit-exact-proven-correct `ba0`/`ba1` (appended directly to the
existing `test_keyswitch.cpp`), with secret-key format and tower
alignment independently verified correct. Also discovered and ruled
out an external GPU-patched OpenFHE fork's keyswitch shadow path
(`gpu_ks_enabled()`/`gpu_keyswitch_inner_product` in
`libopenfhe_cuda_hal.so`, gated by `OPENFHE_GPU`/`OPENFHE_GPU_KS` —
same class of issue as the earlier `bench_evalmult_resident` fix, now
also applied to `test_keyswitch.cpp`): disabling it changed the
measured secret key value but not the ~50% result. Read OpenFHE's own
`keyswitch-hybrid.cpp` source directly and confirmed the high-level
"add ba0/ba1 into c0/c1" convention matches this project's usage, but
did not resolve the exact per-digit/per-part scaling that would make
the elementwise identity hold. **This formula is an honestly
unresolved research question — clearly marked informational-only, not
used to allege any defect.**

Separately, and independently of the unresolved formula: confirmed
`test_polyeval`'s per-mulCt-call rebuilt `Kl` (at its first call,
tw=sizeQ) produces bit-exact identical `ba0`/`ba1` to a top-level `K`
built the same way (0/8192 mismatches) — ruling out the custom
Kl-rebuild logic as a defect source for that call.

**No code in `rescale_resident_raw`, `native_rescale_consts`, the
NTT/SwitchModulus/RescaleFuse kernels, or `keyswitch_core_resident`
was changed — no defect was demonstrated in any of them.** The two
new keyswitch-formula tests are built but deliberately NOT registered
with `add_test` (their failure is expected given the unconfirmed
formula, not a regression signal). All other new/existing tests
re-verified passing with zero regressions. **Residual honest
uncertainty, unchanged from entry #29**: whether test_polyeval's
maxerr growth across Chebyshev levels is quantitatively consistent
with inherent independent-component rescale noise compounding across
multiple chained mulCt calls — not yet confirmed, recorded as the
next concrete step in `NEXT_SESSION.md`.

## GOD PROMPT "CKKS RUST + CUDA EVOLUTION"/"FAIR BENCHMARKS": first Rust-to-CUDA vertical slice working, PLUS a major project-wide build-flag bug found and fixed

Full detail: `EXPERIMENT_LEDGER.md` entry #32. Two things happened this
round, one planned (the Rust evolution's first milestone) and one
unplanned but more consequential (a build-configuration bug affecting
every CUDA benchmark this project has ever run).

**The Rust vertical slice** — `rust/` workspace, `ckks-core` (minimal
validated `RingDimension`/`Modulus` + `CkksError`) and `ckks-cuda`
(the real work) — compiles and runs against this environment's actual
toolchain (RTX 2060 Max-Q, CUDA 13.2, `cudarc` 0.19.10 — 0.17.8 does
not recognize CUDA 13.2). `ckks-cuda::NttContext` calls the EXISTING,
unmodified `kernels/ntt_ct.cu`/`intt_ct.cu` via FFI (compiled directly
by a custom `build.rs`, bypassing the main CMake target's
`CUDA_SEPARABLE_COMPILATION` device-link step since both files are
self-contained), with primitive-root search delegated to the existing
`gpufhe::native_root` through a one-function C-ABI bridge rather than
reimplemented. Its API is deliberately split into one-time setup
(`NttContext::new`) and a reusable per-call hot path
(`NttContext::roundtrip`) — an earlier draft mixed these, which made an
initial benchmark comparison invalid before any number was trusted.
7/7 Rust tests pass, including the full 56-case-per-tower exact-equality
NTT round-trip suite via a REUSED context (proving no stale state leaks
between calls).

**The build-flag bug** — found while chasing down why that benchmark
first showed Rust ~2.3x faster than C++ for the identical kernel. Ruled
out, in order: host-vector reallocation asymmetry (fixed, barely moved
the number), GPU clock-state/run-order confound (reproduced identically
in both orders), relocatable-device-code (`-rdc=true`) compilation
(both RDC and non-RDC isolated builds ran at Rust's speed), the
external GPU-shadowing OpenFHE fork from earlier rounds (still slow
with OpenFHE unlinked), and "many other kernels sharing the binary"
(still slow using only the 3 exact objects needed, extracted from
`libgpufhe.a`). The actual cause: `CMakeCache.txt` had
`CMAKE_CUDA_FLAGS:STRING=-g -G` cached from some earlier, unrecorded
`cmake` invocation — CMake always appends this ON TOP OF the
build-type-specific flags rather than replacing them, so `-G` (CUDA
device debug, which disables nearly all device-code optimization) has
been silently present in every build this whole project's history,
**Release included**. Fixed in `CMakeLists.txt` itself (not just the
cache) so it cannot silently regress again. Rebuilt `gpufhe` and
re-ran 4 regression tests — all pass, confirming the fix is
correctness-neutral.

**After the fix, the honest result**: C++ and Rust host orchestration
of the identical CUDA kernel are equivalent within measurement noise
(steady-state 119.6us vs 108.7us median, 9 trials x 100 reps) — no
genuine language-level performance advantage either way for this
workload. This is the scientifically expected outcome for a thin host
layer calling the same GPU kernel, not a disappointing result to hide.

**This finding's blast radius is larger than this one benchmark**:
every CUDA kernel performance number this project has recorded
anywhere (CUDA graph capture benchmarks, `bench_evalmult_resident`,
`bench_resident_chain`, etc.) was very likely measured with
device-debug mode silently enabled, and should be treated as suspect
until re-run under the corrected flags.

## Immediate follow-up, same session: concrete before/after proof the `-G` bug is real, large, and kernel-dependent

Full detail: `EXPERIMENT_LEDGER.md` entry #33. Re-ran two historical
benchmarks' EXISTING (pre-fix) binaries for genuine "before" numbers,
then rebuilt under the corrected flags and re-ran for "after":

- **`bench_resident_chain`** (n=32768, multiply-rescale chain): the fix
  flipped the benchmark's own verdict. Before: GPU median 7.236ms vs
  CPU 6.947ms, 95% CI entirely above zero — **"GPU SLOWER"**. After:
  GPU median 5.313ms (a 27% drop) vs CPU 6.993ms, CI entirely below
  zero — **"GPU FASTER"**. Not a rounding correction; a reversed
  conclusion about whether GPU acceleration helps this operation at
  all.
- **`bench_evalmult_resident`** (N=16, ring=32768): the CUDA-graph-vs-
  direct speedup ratio was essentially UNCHANGED (186.7x before,
  192.4x after) — for this kernel chain, `-G` cost both the graph and
  direct paths roughly proportionally, so the existing "~190x graph
  speedup" claim survives.

**Takeaway for anyone touching this project's performance history**:
neither "every old number is garbage" nor "the bug didn't matter" is
correct — it is genuinely kernel-dependent, and each specific
historical claim needs its own re-run before being trusted.

## Same session, completing the list: `bench_batch` and `bench_hoist_multi` re-verified too — all four originally-flagged benchmarks now closed

Full detail: `EXPERIMENT_LEDGER.md` entry #34.

- **`bench_batch`**: robust to the fix (57.68ms→56.35ms gpu batch,
  ~2% shift, verdict unchanged — "GPU FASTER on throughput" both
  times). Same pattern as `bench_evalmult_resident`.
- **`bench_hoist_multi`**: the qualitative verdict ("hoisted path is a
  measurable win") survives, but the EXACT magnitude does not — both
  the baseline and hoisted paths got dramatically faster
  (304.15ms→167.96ms and 125.47ms→46.77ms respectively), and because
  they sped up non-proportionally, the relative speedup ratio actually
  **grew** (2.42x→3.59x) while the absolute ms-saved-per-call
  **shrank** (178.85ms→118.80ms). Citing the old absolute figure (or
  an implied 2.4x) after this fix would be wrong, even though "hoisting
  helps" remains true.

All four benchmarks `NEXT_SESSION.md` flagged after entry #32's
discovery are now re-verified under the corrected build flags. Any
OTHER standalone performance number in this project not from one of
these four remains unverified and should still be treated as suspect
until independently re-run.

## GOD PROMPT "CKKS RUST + CUDA PHASE 2": second operation (rescale) ported, a real benchmark bug caught and fixed, Rust/C++ remain equivalent within noise

Full detail: `EXPERIMENT_LEDGER.md` entry #35. Condensed per-operation
status: `research/MIGRATION_MATRIX.md` (new this round).

- Ported `rescale_resident_raw`'s algorithm (single-modulus RNS
  rescale) into `RescaleContext` (`rust/crates/ckks-cuda/src/
  rescale.rs`), the second operation to follow the setup/hot-path split
  `NttContext` established. Confirmed by fresh source re-inspection
  (not by trusting the prior round's notes) that this algorithm's only
  dependency on `DeviceKSContext` is plain moduli + cached NTT root
  tables, never the eval key — the same narrowing argument that made
  `NttContext` possible applies here too.
- Refactored the crate into `util`/`ntt`/`rescale`/`lib` modules to
  share `DeviceBuf` and table-building code between the two operations
  instead of duplicating it.
- Added one new FFI bridge (`cpp_bridge/rescale_consts_bridge.cpp`)
  wrapping the existing, unmodified `gpufhe::native_rescale_consts` —
  no cryptographic arithmetic reimplemented in Rust.
- 4 new Rust tests, validated against a FRESH independent RNS-only
  reference (not the C++ test's own formula, to avoid two
  implementations sharing one bug): 0 mismatches across 3 trials of
  independent random per-coefficient data, plus reused-context and
  invalid-input rejection tests. Full suite 11/11 passing, debug and
  `--release`.
- **Caught a real bug before trusting any benchmark number**: the first
  draft of the fair C++ comparison (`bench/bench_rescale_cpp_vs_rust.cpp`)
  fed synthetic residues into `rescale()` without forward-NTTing them
  into eval domain first (the algorithm's actual contract), producing a
  100% mismatch (7168/7168) against the correctness reference. Fixed by
  mirroring the Rust test's own forward-NTT-before-compare methodology;
  re-ran and got `0/7168 mismatches` before any timing was printed.
- **Benchmark result, run twice in reversed order**: cold-start,
  context-init, steady-state, and end-to-end are all within normal
  run-to-run noise between C++ and Rust (steady-state reused-context
  median: C++ 535-573us vs Rust 534-542us) — same conclusion as the NTT
  operation (entry #32): a thin Rust host layer calling the identical
  CUDA kernel costs the same as the C++ host layer calling it, no
  language-level speedup either way.
- GPU memory confirmed fully freed after both benchmark runs (0MiB
  used). Host free RAM checked via PowerShell before starting (2.37GB
  free) — well under the 6GB full-bootstrap gate, but this round's
  n=1024 workload does not need that gate's caution.
- **Explicitly NOT claimed**: production integration (nothing in the
  engine calls `RescaleContext` outside its own tests/benchmarks),
  anything about key-switching/evalmult/bootstrap, or any security
  re-certification. Two operations are now ported and benchmarked out
  of a large engine — see `MIGRATION_MATRIX.md` for exactly which ones.

## GOD PROMPT "CKKS RUST + CUDA PHASE 3": third operation (full Hybrid key switching) ported, bit-exact vs. the established reference, a real latent C++ bug found and guarded against, memory measured not guessed

Full detail: `EXPERIMENT_LEDGER.md` entry #36. Condensed per-operation
status, including this round's row: `research/MIGRATION_MATRIX.md`.

- Ported `keyswitch_resident` (full Hybrid key switching) as the third
  operation. Unlike NTT/rescale, its setup-time state is a genuine
  evaluation key, not bookkeeping — so instead of rebuilding fields in
  Rust, `KeySwitchContext::new` calls ONE opaque C++ bridge function
  (`cpp_bridge/keyswitch_bridge.cpp`) that runs the existing native
  (OpenFHE-free) pipeline end to end: `native_primes`/`native_root` →
  `keygen_host` → `compute_keyswitch_constants` → `evalkeygen_host` →
  `ks_context_create`/`ks_work_create`. No cryptographic logic is
  reimplemented in Rust; `build.rs` compiles five more existing,
  unmodified C++ host files and three more self-contained kernel files
  directly, the same way it already did for NTT/rescale.
- Re-ran an existing, previously-unregistered test
  (`test_keyswitch_core_native_diff`) BEFORE writing any Rust code, to
  confirm the native key-generation pipeline at this round's chosen
  configuration (n=1024, sizeQ=8, sizeP=2, numPart=2) still executes
  end to end without crashing.
- **Found a real latent bug while tracing the call graph** (not fixed
  in the C++ engine, out of scope — guarded against in Rust instead):
  `compute_keyswitch_constants` divides by `num_part` and does
  unchecked unsigned subtraction for `num_part` outside `[1, size_q]`.
  `KeySwitchContext::new` now validates this before ever calling into
  C++.
- 6 new Rust tests, all passing: bit-exact `keyswitch_resident` (the
  GPU path) vs. `keyswitch_core_resident` (the host reference) across
  random trials, 5 repeated calls on one reused context, zero/max-residue
  boundary inputs, and invalid-construction/invalid-length rejection.
  **This correctness evidence is explicitly weaker than rescale's**:
  both functions implement the same algorithm, so it is not a
  from-scratch independent reference — stated plainly in the code's own
  module doc comment, not hidden. The genuinely independent link
  (`keyswitch_core_resident` proven bit-exact vs. OpenFHE's own
  `KeySwitchCore`) has only ever been established at an OpenFHE-sourced
  n=32768 configuration, never at this round's native n=1024 one; the
  one attempted from-scratch algebraic shortcut for that gap was
  already proven invalid in an earlier round (entry #30) and is not
  reattempted. Full suite 17/17 passing, debug and release.
- **Benchmark, three tiers not four**: unlike NTT/rescale, there is no
  meaningful host-only "cold-start" sub-tier here — key generation
  itself launches GPU kernels, confirmed by reading the source. Fixture
  creation, steady-state keyswitch, and end-to-end are all within
  normal run-to-run noise between C++ and Rust, reproduced in both run
  orders — the same conclusion as NTT and rescale, for a third,
  structurally different operation.
- **Memory measured, not guessed, before any architecture change**:
  sampled GPU memory every 50ms across extended runs of all three
  operations (200,000-500,000 repeated calls each). All three peak at
  an IDENTICAL 85MiB with zero growth — this is the fixed CUDA
  runtime/driver context floor, not operation data (NTT's tables are
  ~40KB, rescale's ~200KB, key switching's full eval key + context +
  scratch a few hundred KB — all negligible against that floor at
  n=1024). Conclusion: no cross-context table-sharing work is justified
  in this crate right now. The C++ engine's own process-lifetime caches
  (`g_rtab`/`upc` in `src/keyswitch_resident.cpp`) already solve this
  problem at the one place it has mattered historically, and
  `KeySwitchContext` inherits that for free. Revisit only if/when a
  genuine multi-tower or bootstrap-scale Rust workload emerges.
- GPU memory confirmed fully freed after every run. Host free RAM
  checked via PowerShell throughout this round (2.37-3.08GB free) —
  well under the 6GB full-bootstrap gate, correctly not invoked since
  this round's entire workload stayed at n=1024.
- **Explicitly NOT claimed**: production integration, anything about
  EvalMult/bootstrap, resolution of `test_polyeval`'s multi-level noise
  question (untouched this round), or any security re-certification.
  Three operations are now ported and benchmarked out of a large engine
  — see `MIGRATION_MATRIX.md` for exactly which ones and the honest
  caveat on key switching's weaker correctness evidence.

## GOD PROMPT "CKKS RUST + CUDA PHASE 4": fourth operation (EvalMult) ported by COMPOSING the prior two, a real test-side domain bug caught and fixed, wider benchmark noise honestly flagged

Full detail: `EXPERIMENT_LEDGER.md` entry #37. Condensed per-operation
status: `research/MIGRATION_MATRIX.md`.

- Ported ciphertext-ciphertext EvalMult (tensor product + relin +
  combine + rescale x2) as the fourth operation, by COMPOSING the
  already-ported `KeySwitchContext` (relin backend) and `RescaleContext`
  (the two post-relin rescale steps) rather than rebuilding either
  algorithm — confirming the prior round's own prediction that
  `KeySwitchContext` would be directly reusable here.
- Their existing public APIs were host-slice-oriented; composing them
  as-is would force pointless host round trips the real C++ pipeline
  never pays. Added exactly two new `pub(crate)` device-pointer methods
  (`keyswitch_device_resident`, `rescale_device_resident`), each reusing
  100% of the existing tested kernel-launch logic — no duplication.
- Two new kernels (`LaunchTensor`/`LaunchCombine`, `kernels/tensor.cu`)
  confirmed self-contained the same way every prior kernel was; no new
  host `.cpp` files needed.
- **A real bug was caught in this round's OWN test, not in the
  production code, before it was trusted**: the first draft generated
  synthetic eval-domain-looking data and applied rescale's
  coefficient-domain-only "center the dropped tower" identity directly
  to it — the exact same domain-confusion bug CLASS as the rescale
  benchmark's own incident in an earlier round (entry #35), independently
  rediscovered here. Root-caused by cross-checking every intermediate
  stage (all matched before rescale) and by confirming the GPU path's
  own `rescale_device_resident` agreed with the already-proven `rescale`
  on the exact same data — together isolating the bug to the test's
  reference formula. Fixed by generating genuinely coefficient-domain
  inputs and NTT/INTT-ing around the rescale identity, mirroring
  `rescale`'s own established methodology.
- 3 new differential tests (genuinely independent reference this time,
  not sharing the earlier draft's bug), plus 2 new permanent regression
  tests added to `rescale.rs` while isolating the bug. Full suite 20/20
  passing, debug and release.
- **Benchmark**: 3 independent A/B comparisons, alternating which
  language ran first. Every tier's ranges overlap substantially with no
  consistent winner — statistically indistinguishable, same conclusion
  as every prior operation — but with notably WIDER run-to-run variance
  than the three simpler single-kernel-chain operations, honestly
  flagged (plausibly more kernel launches per call amplifying
  host-dispatch/thermal noise, not independently isolated as the cause).
- **Memory measured, not guessed**: peak 108MiB across 50,000 repeated
  calls, zero growth. This session's idle floor was 22MiB (not entry
  #36's 0MiB — a session-level WSL2/driver-reporting difference, not a
  regression, confirmed via `nvidia-smi --query-compute-apps` showing no
  attached process). EvalMult's own delta above its floor (86MiB)
  closely matches key-switching's historical delta (85MiB) — same
  conclusion as entry #36: no cross-context table-sharing work is
  justified by this measurement.
- GPU memory confirmed fully freed after every run. Host free RAM
  checked via PowerShell throughout (1.64-2.77GB free) — well under the
  6GB full-bootstrap gate, correctly not invoked since this round's
  entire workload stayed at n=1024.
- **Explicitly NOT claimed**: production integration, resolution of
  `test_polyeval`'s multi-level noise question (untouched), a real
  encode/encrypt/decrypt round trip (left as an explicit follow-up, not
  attempted), or any security re-certification. Four operations are now
  ported and benchmarked out of a large engine — see
  `MIGRATION_MATRIX.md` for exactly which ones.

## GOD PROMPT "CKKS PHASE 5": real CKKS encode/encrypt/decrypt proven correct; the real encrypted EvalMult round trip found genuinely BLOCKED, not merely unattempted

Full detail: `EXPERIMENT_LEDGER.md` entry #38. Condensed per-operation
status, with the new real-vs-RNS distinction: `research/
MIGRATION_MATRIX.md`.

- Built a REAL CKKS encode/encrypt/decrypt/decode pipeline
  (`codec.rs` + `cpp_bridge/encdec_bridge.cpp`, wrapping the EXISTING
  `encode_host`/`decode_host`/`encrypt_host`/`decrypt_host` verbatim)
  and two TEST-ONLY key-material accessors on `KeySwitchContext`
  (`secret_key`/`public_key_a`/`public_key_b`) so a test can encrypt
  and decrypt under the EXACT SAME key an eval key was built under.
  Both round trips (encode->decode; encrypt->decrypt, single
  ciphertext) PASS with errors far below derived (not fitted)
  tolerances — genuinely new, stronger-than-RNS-level evidence.
- Attempting the natural next step — route a real ciphertext through
  `EvalMultContext::multiply_relin_rescale` and decrypt the result —
  found a genuine, deeply-investigated blocker: the relin step's
  output does not satisfy the textbook decryption invariant
  (`ba0+ba1*s ≈ t2*s²`), checked directly in eval domain with no
  decode/rescale involved. This is a CONFIRMED, more consequential
  recurrence of an EARLIER round's own unresolved finding (entry #30)
  — previously thought to be an unconfirmed algebraic curiosity since
  `keyswitch_resident`'s output was still independently proven correct
  at the RNS level; this round shows it ALSO blocks any real
  decrypt-based verification, since `LaunchCombine`'s `r0=t0+ba0,
  r1=t1+ba1` IS that exact identity.
- Tried 4 sign/arithmetic variants, `numPart=1` (no digit decomposition
  at all) as well as `numPart=2`, and a P/P⁻¹ auxiliary-modulus
  scaling-factor hypothesis — ALL give ~50% of q0 (statistically
  uncorrelated), ruling out a simple sign or scale-factor bug.
- **What this does NOT mean**: no new doubt is cast on
  `keyswitch_resident`/`rescale_resident_raw`/`LaunchTensor`/
  `LaunchCombine` themselves — all remain independently proven at the
  RNS level by entries #35-#37, using references that don't depend on
  this specific identity. The leading hypothesis is that the identity
  itself is incomplete for this scheme's real per-digit scaling
  convention, not that the implementation is wrong — but this round
  did not find the missing piece.
- The 4 blocked tests are kept, not deleted, marked `#[ignore]` with
  the finding documented in the ignore reason and in code — `cargo
  test` reports them as `ignored`, not silently passing or left red.
  Full suite: 22 passed, 4 ignored, 0 failed, debug and release.
- **Explicit next experiment**: link OpenFHE briefly, run a REAL
  `cc->EvalMult`+`cc->Decrypt` round trip (not the algebraic shortcut)
  to confirm OpenFHE's own relin decrypts correctly via normal usage,
  then work backward from its actual `EvalMultCore` combine-step
  source to find the missing factor.
- GPU memory confirmed clean (22MiB session floor, no growth). Host
  free RAM checked via PowerShell throughout (1.64-3.27GB free) — well
  under the 6GB full-bootstrap gate, correctly not invoked.
- **Explicitly NOT claimed**: the full encrypted EvalMult round trip,
  resolution of entry #30 or `test_polyeval`'s SEPARATE multi-level
  noise question (untouched and must not be conflated with this
  entry's finding), a benchmark of the real-encryption primitives
  (deliberately deferred — benchmarking before the one operation that
  would make it meaningful is correctness-established would violate
  the directive's own ordering), or production integration.

## GOD PROMPT "CKKS PHASE 6": the entry #30/#38 relin-identity mystery ROOT-CAUSED AND RESOLVED — it was a test methodology bug, not a defect in `keyswitch_resident`/`LaunchCombine`; a SEPARATE rescale/composition bug found in the full round trip

Full detail: `EXPERIMENT_LEDGER.md` entry #39. Updated next step:
`research/NEXT_SESSION.md`'s "READ THIS FIRST" section.

- **The core finding**: `ba0+ba1*s ≈ t2*s²` DOES hold correctly — every
  prior check of this identity (entry #30's original
  `tests/test_keyswitch.cpp` check, entry #38's native recurrence, and
  this round's own first two attempts) compared, or even multiplied,
  RNS polynomials in the wrong domain. Ring multiplication is pointwise
  ONLY in `Format::EVALUATION` (NTT domain); NTT is bijective but
  highly mixing, so a tiny `Format::COEFFICIENT`-domain difference
  (genuine, small key-switch noise) spreads into what looks like a
  large, pseudo-random difference at every individual NTT point —
  making the identity LOOK like it fails at ~50% of q even when the
  underlying polynomials are nearly identical.
- Built a new OpenFHE-only reference test
  (`tests/test_evalmult_relin_openfhe_reference.cpp`) that: (1) proves
  a REAL `cc->EvalMult()`+`cc->Decrypt()` round trip is correct
  (max_err~1e-12); (2) re-checks the identity on OpenFHE's own
  `KeySwitchCore` output, reproducing the same ~50%-of-q "failure" when
  compared the old (wrong) way; (3) through a systematic elimination
  (byte-identical extraction, raw-tensor-identity-alone verification,
  accessor cross-check, explicit tower/root-of-unity alignment — each
  independently confirmed, none the culprit) isolates the true cause to
  the comparison domain; (4) confirms the fix: converting both sides to
  `Format::COEFFICIENT` before comparing drops the residual to 4.5e-11%
  of q.
- Applied the same investigation to this project's OWN original entry
  #30 check (`tests/test_keyswitch.cpp`): naively converting to
  `Format::COEFFICIENT` did NOT immediately fix it — forced a second
  look that found a DIFFERENT, newly-introduced bug (multiplying
  already-COEFFICIENT-domain values pointwise is a meaningless Hadamard
  product, not ring multiplication). A controlled real-vs-synthetic-data
  experiment (same context/eval key, swapping entry #30's synthetic
  random input for a real `t2=c1a*c1b`, plus a real-decrypt control)
  made the contradiction airtight and pinpointed the fix. After
  correcting the multiplication order (multiply first in
  `Format::EVALUATION`, convert the RESULT to `Format::COEFFICIENT`):
  synthetic data drops to 0.0825% of q; real ciphertext data drops to
  5.54e-11% of q.
- **Conclusion: `keyswitch_resident`'s `ba0`/`ba1` output and
  `LaunchCombine`'s combine formula were correct all along.** This
  closes entries #30/#38's open question definitively, via an
  independent, OpenFHE-cross-validated route — it was never a missing
  sign, scale factor, per-digit normalization, or basis-conversion term.
- **This does NOT mean the full Rust round trip now passes.** Re-ran
  the still-`#[ignore]`d `real_encrypted_evalmult_matches_expected_product`:
  it STILL fails, with a genuine, large, DIFFERENT decode-level error
  (slot 0: decoded=-370.86, expected=5). Since `codec::decrypt`/`decode`
  already go through proper domain conversion, this is NOT explained by
  the methodology bug above — it is a separate, real, not-yet-root-caused
  issue, most likely in `rescale_device_resident`'s composition within
  the fused `multiply_relin_rescale` pipeline, or specific to real
  CKKS-scaled magnitudes rather than the synthetic RNS data entries
  #35-37's and Phase 4's own differential tests already cover.
- Added one new TEST-ONLY diagnostic,
  `EvalMultContext::multiply_relin_no_rescale_for_test` (`#[cfg(test)]`,
  returns the un-rescaled combined result at full tower count), NOT YET
  used by a test — building a correct decode for its `delta²`-scaled
  value exceeds `codec::decrypt`'s documented tower-0-only convention.
  See `NEXT_SESSION.md` for the two concrete next-step options.
- No production C++/CUDA source file was modified this round
  (`src/keyswitch_resident.cpp` etc. untouched) — consistent with
  "do not invent a fix" when, as it turned out, none was needed for the
  identity question. Full Rust suite re-confirmed clean: 22 passed, 4
  ignored (unchanged, correctly still ignored), 0 failed.
- Also discovered and worked around this round: two SEPARATE, diverged
  git clones of this repo exist on this machine (a Windows-filesystem
  one with an unrelated CUDA-kernel-optimization history, and the WSL
  one, authoritative for this entire thread) — see
  `NEXT_SESSION.md`'s reminder.
- **Explicitly NOT claimed**: the full encrypted EvalMult round trip
  (still blocked, now on a DIFFERENT, narrower bug than previously
  thought), resolution of `test_polyeval`'s SEPARATE multi-level noise
  question (untouched), a benchmark of EvalMult (correctness still not
  fully established end-to-end), or production integration.

## GOD PROMPT "CKKS PHASE 7": the remaining EvalMult blocker is ISOLATED to a confirmed defect in `evalkeygen_host`'s own native eval-key generation — `LaunchTensor`, `keyswitch_device_resident`, `keyswitch_core_resident`, and `LaunchCombine` are all proven correct for real data this round

Full detail: `EXPERIMENT_LEDGER.md` entry #40. Updated next step:
`research/NEXT_SESSION.md`'s "READ THIS FIRST" section.

- **The core finding**: this round added a sequence of new,
  discriminating experiments that prove `LaunchTensor`,
  `keyswitch_device_resident`/`keyswitch_core_resident` (the
  decompose/accumulate/moddown algorithm), and `LaunchCombine` are ALL
  correct for REAL encrypted ciphertext data under this crate's own
  native (OpenFHE-free) pipeline — not merely bit-exact against a
  reference on synthetic RNS data, as before. The confirmed defect is
  narrowed specifically to `evalkeygen_host`'s/`evalkeygen_host_sold`'s
  own eval-key GENERATION (`src/evalkeygen.cpp`), which had NEVER been
  independently checked against any reference before this round —
  every prior check of the relin identity (entries #30, #36, #38, and
  this project's own `tests/test_keyswitch.cpp`) used OpenFHE-SOURCED
  key material extracted into `KeySwitchConstants`, never this crate's
  own `evalkeygen_host` output.
- **The evidence chain**: (1) a corrected, ZERO-TOLERANCE algebraic
  identity check (`t0+t1*s+t2*s^2` must EXACTLY equal
  `(c0a+c1a*s)*(c0b+c1b*s)`, both computed directly from real
  ciphertext data) passes with 0 mismatches across 8192 positions —
  `LaunchTensor` is exact. An EARLIER version of this check wrongly
  compared against the noise-free plaintext product with a tight
  tolerance, giving a spurious failure — the raw, un-rescaled tensor's
  cross-noise terms are genuinely large by design (this is exactly why
  CKKS rescales after every multiply), not a defect; this was caught
  and corrected before being trusted. (2) `keyswitch_device_resident`
  and `keyswitch_reference`/`keyswitch_core_resident` agree BIT-EXACT
  (0/8192 mismatches) on the same real `t2` — the key-switch algorithm
  itself is not the defect. (3) Combined with entry #39 (this same
  algorithm, given an OpenFHE eval key, already proven correct for real
  data), the defect must be in the eval key. (4) A trivial
  CRT-consistent constant input (not even a real ciphertext) still
  fails the relin identity through this crate's own eval key. (5) The
  MOST isolated reproduction: for Q-towers outside eval-key part 0's
  own digit range, where `evalkeygen_host_sold` adds NO `P*s2` term at
  all, the expected `bv[0]-av[0]*s ~= 0` (depending on nothing but the
  basic "encryption of zero" machinery) still fails at ~50% of q,
  confirmed via a direct C++/Rust cross-check of the raw values (ruling
  out an accessor bug).
- **Hypotheses tested and REFUTED, do not re-try without new
  evidence**: digit decomposition/`numPart` (1 and 2 fail identically);
  the batched-vs-per-tower NTT kernel launch mechanism
  (`LaunchNTT_CT_Batched` vs. a per-tower `LaunchNTT_CT` loop — swapped
  as a direct experiment, gave a bit-for-bit IDENTICAL result, then
  reverted since it had no benefit); an eval-key accessor/FFI/indexing
  bug (cross-checked exactly); a `ks_context_create`-introduced
  mutation of `K.av`/`K.bv` (structurally impossible — it takes a
  `const` reference).
- **Hypotheses NOT yet tested**: the exact per-coefficient arithmetic
  of `u`/`e0`/`e1`'s sampling-to-coefficient-embedding step inside
  `evalkeygen_host_sold`; whether `PModq_QP`'s computation in
  `cpp_bridge/keyswitch_bridge.cpp` is correct (not exercised by the
  "outside range" test); a from-scratch, hand-constructed-key C++ test
  of `evalkeygen_host_sold` in isolation. See `NEXT_SESSION.md` for
  full detail.
- Added 5 new PERMANENT PASSING regression tests this round
  (`control_a2_raw_tensor_matches_direct_ciphertext_product_exactly`,
  `fast_keyswitch_matches_reference_keyswitch_on_real_t2`,
  `sanity_check_single_ciphertext_eval_domain_methodology_test`,
  `sanity_check_both_control_a2_ciphertexts_decrypt_individually`,
  `keyswitch_reference_on_zero_input_gives_near_zero`) and 4 new
  `#[ignore]`d tests documenting the confirmed defect with clear
  reasons, alongside the 4 pre-existing ignored tests from entry #39
  (unchanged). New, additive-only C++ instrumentation:
  `KeySwitchContext::eval_key_part`/`eval_key_towers`/`alpha` accessors
  in `cpp_bridge/keyswitch_bridge.cpp` (read-only, exposes existing
  eval-key data, no computation changed). Full Rust suite: 27 passed, 8
  ignored, 0 failed.
- No production C++/CUDA source file ends this round modified — the
  diagnostic NTT-kernel-swap experiment in `src/evalkeygen.cpp` was
  reverted (`git checkout --`) since it gave no benefit and should not
  be left as an unverified, speculative change.
- **Explicitly NOT claimed**: the full encrypted EvalMult round trip
  (still blocked, now correctly understood to depend on fixing
  `evalkeygen_host` first — the rescale/composition question from
  entry #39 is deliberately NOT re-investigated until the eval key
  itself is fixed), resolution of `test_polyeval`'s SEPARATE multi-level
  noise question (untouched), or production integration.

## GOD PROMPT "CKKS PHASE 8": entry #40's `evalkeygen_host` diagnosis DISPROVEN (a sign bug in its own test) — the real EvalMult blocker is pinpointed to a `sizeP=2`-specific defect in the native key-switch reconstruction (`compute_keyswitch_constants`+`keyswitch_core_resident`), the exact P-tower count the real Rust fixture uses

Full detail: `EXPERIMENT_LEDGER.md` entry #41. Updated next step:
`research/NEXT_SESSION.md`'s "READ THIS FIRST" section (entirely
rewritten this round).

- **The core finding**: entry #40's conclusion that `evalkeygen_host`
  has a confirmed defect was WRONG. It was entry #40's own test,
  `eval_key_outside_own_part_range_is_near_zero`, that had a sign bug:
  it computed `bv0[idx]-av_s`, literally following
  `src/evalkeygen.cpp`'s own doc comment, which itself states the
  relation with the wrong sign. The independently re-derived and
  empirically confirmed TRUE relation is `bv+av*s ~= P*s2` (in a
  part's own range) / `~=0` (outside it) — PLUS, not MINUS. Derivation:
  `keygen_host` (`src/keygen.cpp:105`) computes `b=ns*e-a*s`, so
  `b+a*s=ns*e` (small); substituting into `evalkeygen_host_sold`'s
  `av=pkA*u+ns*e1`, `bv=pkB*u+ns*e0[+Ps2]` gives
  `bv+av*s = u*ns*e_keygen + ns*(e0+e1*s) + Ps2 ~= Ps2`.
- **Four independent confirmations, all built/rebuilt this round**:
  (1) `tests/test_evalkeygen_isolated.cpp` (new, fully native, no
  OpenFHE/Rust/FFI dependency): CHECK 1, a hand-constructed nonzero
  secret (`s_hat=1`, `pkA=7`, `pkB=-7`, giving exact zero-noise
  cancellation) checked across ALL QP towers including the P range
  (never covered before) — worst 1.51e-12% of q. CHECK 2, a real
  `keygen_host` keypair checked for `pkB+pkA*s~=0` across all QP
  towers — worst 9.77e-13% of q. CHECK 3, the real key through the
  real `evalkeygen_host` wrapper with real `PModq_QP` — worst
  3.46e-11% of q. (2) The sign bug in `evalmult.rs`'s
  `eval_key_outside_own_part_range_is_near_zero` fixed
  (`bv0[idx]-av_s` → `addmod(bv0[idx],av_s,q)`) and un-ignored — now
  passes at worst 3.16e-11% of q, where entry #40 reported ~50% using
  the wrong sign. (3) Two PRE-EXISTING, previously-overlooked July-era
  tests, rebuilt and re-run this round for the first time in this
  investigation thread: `tests/test_evalkeygen.cpp` ("Stage 6c",
  OpenFHE-sourced keypair, already uses the "+" convention) — PASS,
  maxmag=2924. `tests/test_ks_constants.cpp` ("Stage 6a", bit-exact vs.
  OpenFHE's own `GetPInvModq`/`GetPHatInvModp`/etc.) — PASS, bit-exact
  at `sizeQ=5,sizeP=3,numPart=2`.
- **The real, newly-pinpointed defect**: CHECK 4 of
  `tests/test_evalkeygen_isolated.cpp` runs the REAL
  `compute_keyswitch_constants`+`evalkeygen_host`+
  `keyswitch_core_resident` full reconstruction on a constant
  CRT-consistent input, at the real Rust fixture's exact configuration
  (`N=1024,SIZE_Q=8,SIZE_P=2,NUM_PART=2,alpha=4`,
  `rust/crates/ckks-cuda/src/evalmult.rs`'s `new_ctx`) — **FAILS at
  ~50% of q across all 8 towers**. Systematically varying the
  configuration isolated the trigger to **`sizeP=2` specifically**:
  `sizeP=3` PASSES cleanly (~1e-11% of q) at both `sizeQl=8` and
  `sizeQl=5` (ruling out `sizeQl`); `numPart=1` still FAILS at
  `sizeP=2` (ruling out digit decomposition). `K.pInvModq`/`K.pHatInv`
  individually satisfy their own defining modular-inverse relations
  exactly even at `sizeP=2` — the constant VALUES are correct, their
  downstream USE inside `keyswitch_core_resident`'s moddown step is
  not yet cleared. An unexplained tower-0-vs-towers-1-7 asymmetry
  (0.096% vs. ~50%) was observed but not understood.
- **A methodology trap caught and corrected before being reported**:
  an earlier version of the isolated test hand-reconstructed a known
  CRT-consistent constant from the P-basis via `K4.pHatInv`/
  `K4.pHatModq`, expecting EXACT equality, and "failed" by exactly
  `P` — confirmed via direct arithmetic (`5+1*P`). This is not a
  defect: `LaunchApproxSwitchCRTBasis` is an APPROXIMATE conversion by
  design (reconstructs `x+k*P` for bounded `k`), and the surrounding
  `LaunchModDownCombine` step is specifically designed to cancel this
  bounded term. Removed from the committed test, replaced with an
  explanatory comment.
- **Hypotheses tested and REFUTED this round, do not re-try without
  new evidence**: `evalkeygen_host`/`evalkeygen_host_sold` having any
  defect (entry #40's entire diagnosis, see above); digit
  decomposition/`numPart` as the `sizeP=2` trigger; `K.pInvModq`/
  `K.pHatInv`'s own modular-inverse correctness; the naive
  CRT-reconstruction-exactness framing.
- **Hypotheses NOT yet tested**: `K.pHatModq`'s correct use inside
  `keyswitch_core_resident`'s `moddown()` lambda specifically for a
  2-element P basis; an edge case in `LaunchApproxSwitchCRTBasis`'s
  Barrett/Shoup reduction for small source counts;
  `LaunchModDownCombine`'s own combine arithmetic; the tower-0
  asymmetry. See `NEXT_SESSION.md` for full detail.
- New test file `tests/test_evalkeygen_isolated.cpp`, registered in
  `CMakeLists.txt`. Rust changes: the sign fix and un-ignore in
  `eval_key_outside_own_part_range_is_near_zero`, plus 7 corrected
  `#[ignore]` reason strings pointing from entry #40 to entry #41 with
  the corrected explanation — the 7 `real_encrypted_evalmult_*`/
  diagnostic tests blocked by the `sizeP=2` defect correctly REMAIN
  `#[ignore]`d. Full Rust suite: **28 passed, 7 ignored, 0 failed**
  (one more pass than entry #40, as expected).
- No production C++/CUDA source file ends this round modified — a
  diagnostic NTT-kernel-swap experiment in `src/evalkeygen.cpp`
  (testing the now-refuted eval-key hypothesis) was tried, gave a
  bit-identical result, and was reverted (`git checkout --`).
- **Explicitly NOT claimed**: the exact line/mechanism causing the
  `sizeP=2`-specific failure (not pinpointed); the full encrypted
  EvalMult round trip (still blocked, now correctly understood to
  depend on fixing this reconstruction defect, not `evalkeygen_host`);
  resolution of `test_polyeval`'s SEPARATE multi-level noise question
  (untouched).

## GOD PROMPT "CKKS PHASE 9": the real encrypted EvalMult round trip PASSES end-to-end for the first time (since Phase 5) — entry #41's "sizeP=2-specific defect" framing is SUPERSEDED; the real cause was a test-fixture parameter choice (NUM_PART=2 violating the Hybrid key-switch's P>=Q_part_max requirement), not a code defect, and it is fixed

Full detail: `EXPERIMENT_LEDGER.md` entry #42. `NEXT_SESSION.md`'s
"READ THIS FIRST" section is entirely rewritten to reflect this
resolution.

- **The core finding**: `sizeP=2` was never the trigger. Holding
  `sizeP=2` FIXED and sweeping ONLY `numPart` (`tests/
  test_ks_pq_ratio_sweep.cpp`, calling the real, unmodified
  `keyswitch_core_resident` directly) shows the relin-identity
  residual flips from ~50% of q to ~6e-12% of q EXACTLY when
  `P - Q_part_max` (the auxiliary modulus's bit length minus the
  largest key-switch digit's own modulus-product bit length) crosses
  from negative to positive: `numPart=1,2,3` (deficits
  -290/-90/-40 bits) all FAIL; `numPart=4,8` (surpluses +10/+60 bits)
  both PASS cleanly. This is the standard Hybrid key-switch
  correctness requirement (P must be at least as large as the largest
  digit's own modulus product) — a property of RNS-CKKS Hybrid
  key-switching itself, not a defect anywhere in this codebase.
- **The evidence chain that got there**: (1) `tests/
  test_moddown_crt_basis_isolated.cpp` — isolated `LaunchApproxSwitchCRTBasis`
  in the exact ModDown P->Q configuration, with an INDEPENDENT CPU
  reference deriving the expected `(V+k*P) mod q_i` relation, searching
  for a consistent `k` across all target towers. PASSES cleanly for
  `sizeP` in {1,2,3} at both `sizeQl=5,8`. (2) `tests/
  test_decompose_crt_basis_isolated.cpp` — the SAME methodology
  applied to decompose's OWN, different constant family
  (`partQHatInv`/`partQHatModp`), never independently tested before.
  ALSO passes cleanly at every `sizeP`. (3) `tests/
  test_moddown_combine_isolated.cpp` — the full ModDown chain
  (INTT->basis-convert->NTT->combine) at real n=1024 scale, fed a
  hand-constructed input giving an EXACT zero-tolerance expected
  output. PASSES cleanly at every `sizeP`. (4) `tests/
  test_keyswitch_stage_trace.cpp` — instrumented the REAL end-to-end
  pipeline (real secret, real eval key) across the full `sizeQl x
  sizeP x numPart` matrix with THREE inline checks: decompose (exact)
  and accumulate (exact) ALWAYS pass; only the final relin identity
  (approximate, noise-tolerant) fails, and the failing set does NOT
  track `sizeP` alone — `numPart=1` fails even at `sizeP=3`, which
  entry #41's Phase 8 round never checked. (5) `tests/
  test_ks_pq_ratio_sweep.cpp` — the decisive discriminator described
  above.
- **Fix applied**: `rust/crates/ckks-cuda/src/evalmult.rs`'s
  `new_ctx`'s `NUM_PART` constant changed from `2` to `4` (alpha drops
  4->2, `Q_part_max` drops ~210->~110 bits, a +10-bit margin over
  `P`'s ~120 bits). `SIZE_Q`, `SIZE_P`, `n`, and the secret
  distribution (`h=64`) are ALL UNCHANGED — this is a digit-grouping
  choice with zero security-parameter impact (`log2(QP)` unaffected).
- **Result**: `real_encrypted_evalmult_matches_expected_product`
  passes with max abs error `2.5896e-11` (tolerance `1e-6`); its 3
  siblings (`zero_boundary`, `negative_and_fractional_values`,
  `repeated_calls_reused_context`) all pass too.
  `control_a_relin_combine_matches_raw_tensor_within_keyswitch_noise_test`
  and `keyswitch_reference_on_constant_crt_consistent_input` (both
  `#[ignore]`d since entry #40) now pass, un-ignored. Full Rust suite:
  **34 passed, 0 failed, 1 ignored** (down from 7 ignored), identical
  in both `--release` and debug profiles.
- **Correctly kept ignored**:
  `control_a_relin_combine_matches_raw_tensor_num_part_1_test`
  (forces `numPart=1`, `Q_part=410b` vs `P=120b` — a genuine, much
  larger deficit that `NUM_PART=4` does not fix; would need a far
  larger `SIZE_P`, a security-relevant change, out of scope).
- **No production C++/CUDA source file changed** — every constant
  family and kernel in `ks_constants.cpp`/`keyswitch.cpp`/
  `basis_convert.cu`/`moddown.cu` was independently proven correct in
  isolation this round, at every tested `sizeP`. `CMakeLists.txt`
  gained 5 new test registrations (additive only); `tests/
  test_evalkeygen_isolated.cpp`'s own header comment was corrected to
  explain the now-superseded `sizeP=2` framing rather than restate it
  as fact (its own CHECK 4, at its fixed `sizeQl=8,sizeP=2,numPart=2`,
  still correctly and intentionally fails — now understood, not an
  open mystery). Pre-existing OpenFHE-cross-validated tests
  (`test_ks_constants.cpp`, `test_evalkeygen.cpp`) re-verified passing
  (fresh rebuild, not a cached binary), confirming no regression.
- **Not attempted**: a full CMake rebuild (real Windows host free
  memory was 1.41GB at check time, far under the established 6GB
  gate) — all new isolated tests verified via direct `nvcc`/`g++`
  compilation instead, per the established pattern. No performance
  claim is made for `NUM_PART=4` vs `NUM_PART=2` (not measured, per
  the governing directive's explicit instruction not to claim a
  performance improvement from correctness instrumentation).
  `keyswitch.rs`'s own test fixture retains the identical
  `NUM_PART=2` combo — not currently broken (its tests check only
  self-consistency between two implementations of the same formula,
  unaffected by the P/Q_part ratio) but flagged as a dormant instance
  of the same pattern. `test_polyeval`'s own, separate multi-level
  noise question remains untouched.

## GOD PROMPT "CKKS PHASE 10": repository-wide parameter audit + centralized, exact enforcement of the Hybrid key-switch's P>=Q_part_max requirement — four more invalid fixtures found and fixed, keyswitch.rs's own fixture corrected with a new decrypt-based test, invalid configurations now REJECTED at construction time

Full detail: `EXPERIMENT_LEDGER.md` entry #43. `NEXT_SESSION.md`'s
"READ THIS FIRST" section rewritten to reflect this.

- **The core deliverable**: entry #42 established the `P>=Q_part_max`
  requirement as a derived FACT; this round implemented it as CODE —
  `include/ks_params_validate.h`/`src/ks_params_validate.cpp`'s
  `check_hybrid_keyswitch_params`, using true arbitrary-precision
  integer products (never a bit-length approximation, since
  `Q_part_max` can exceed 128 bits and two numbers of equal bit length
  can differ by orders of magnitude). Cross-checked independently
  against Python's native bigints (`tools/verify_ks_params_validate.py`)
  — byte-for-byte decimal agreement on every tested configuration, a
  completely different implementation stack from the C++ BigUint code.
- **Enforced at `ks_context_create`** (`src/keyswitch_resident.cpp`),
  immediately after its pre-existing `K.av.size()!=K.numPart` check —
  the single construction-time choke point every real caller passes
  through (the Rust FFI bridge, hence `KeySwitchContext::new` AND
  `EvalMultContext::new`; every benchmark/example building a usable
  context directly in C++). Deliberately NOT inside
  `compute_keyswitch_constants` (several permanent diagnostic tests
  intentionally drive it with invalid configs on purpose) and NOT
  duplicated in Rust (Rust has zero independent copy — it only learns
  of a rejection via the C++ exception's propagated message). Zero
  hot-path cost: the per-call functions never call `ks_context_create`
  again after construction.
- **Repository-wide audit found FOUR MORE invalid fixtures** entry
  #42's audit missed (it only checked `rust/crates/ckks-cuda/src/`):
  `examples/bench_evalmult.rs`, `examples/bench_keyswitch.rs` (both
  `NUM_PART=2`), and `bench/bench_evalmult_cpp_vs_rust.cpp`, `bench/
  bench_keyswitch_cpp_vs_rust.cpp` (both `numPart=2`, direct C++,
  bypassing the Rust FFI bridge entirely, and registered as auto-run
  `ctest`s — these would have broken the test suite outright once
  validation landed). All four fixed to `NUM_PART=4`/`numPart=4`.
  `tests/test_keyswitch_core_native_diff.cpp`'s default (`numPart=2`
  via argv, doesn't call `ks_context_create`, so not construction-
  blocking) was also bumped to 4 purely so running it with no
  arguments now reports a clean PASS.
- **Fixed `keyswitch.rs`'s own fixture** (the GOD PROMPT's explicitly
  named target): `NUM_PART` 2->4. Added THREE new tests:
  `invalid_num_part_at_old_value_is_now_rejected` (confirms the exact
  old default is now rejected with a message naming `Q_part_max`),
  `valid_num_part_with_surplus_still_constructs` (confirms the
  validation doesn't over-reject), and
  `real_decrypt_based_relin_round_trip` — THE decrypt-based correctness
  test this module's own doc comment flagged as missing: builds a real
  ciphertext pair under the fixture's own secret/public key, forms the
  degree-2 tensor product via pure host-side schoolbook arithmetic
  (entry #37's methodology, genuinely independent of the production
  `LaunchTensor` kernel), relinearizes through the REAL `keyswitch`
  call, rescales via the independently-proven `RescaleContext`,
  decrypts and decodes via `codec`, and compares against the plaintext
  product computed in the clear. **Result: max abs error `2.23e-11`**
  (tolerance `1e-6`). Also added `evalmult.rs::
  former_invalid_num_part_2_fixture_is_now_rejected` confirming the
  rejection propagates through `EvalMultContext::new` too.
- **Audited and confirmed VALID (no fix needed), via the exact
  validator, not assumption**: `tests/test_ks_raii.cpp`, `tests/
  test_ks_context_create_validation.cpp` (`n=8192,sizeQ=4,sizeP=2,
  numPart=2`); `tests/test_hoist_*.cpp`/`test_rot_resident.cpp`/
  `bench/bench_hoist_multi.cpp` (all `n=8192,sizeQ=30,sizeP=2,
  numPart=15`); `tests/test_ks_resident.cpp` (OpenFHE's own `HYBRID`
  parameter choice at `n=32768`, re-built fresh with validation
  active, still bit-exact). **Trusted, not independently re-derived**:
  `bench/bench_evalmult.cpp`/`bench_evalmult_resident.cpp`
  (OpenFHE-sourced `GetNumPartQ`/`GetParamsP`). **Deliberately exempt**
  (never call `ks_context_create`, intentionally drive invalid configs
  for diagnostic purposes): `tests/test_evalkeygen_isolated.cpp`,
  `tests/test_ks_pq_ratio_sweep.cpp`, `tests/
  test_keyswitch_stage_trace.cpp`, `tests/
  test_moddown_crt_basis_isolated.cpp`, `tests/
  test_decompose_crt_basis_isolated.cpp`, `tests/
  test_moddown_combine_isolated.cpp`. `tests/test_alpha10.cpp` checked
  and confirmed UNRELATED (no keyswitch involvement at all).
- **Result**: Full Rust suite **38 passed, 0 failed, 1 ignored** (up
  from 34/1 at entry #42), identical in `--release` and debug. The two
  fixed C++ benchmarks and two fixed Rust examples all rebuilt fresh
  and confirmed working (both C++ benchmarks' own internal correctness
  gates pass: 0/16384 and 0/14336 mismatches).
- **The general lesson, now stated as explicit project doctrine**: a
  passing RNS-level or self-consistency test does NOT prove a
  parameter configuration satisfies the real ciphertext decryption
  invariant. `keyswitch.rs`'s invalid fixture passed every one of its
  own tests for multiple rounds precisely because none of them checked
  a real decryption invariant — only `keyswitch_device_resident`
  against `keyswitch_reference`, two implementations of the identical
  formula.
- **Not attempted**: a full CMake rebuild (host free memory stayed
  under the established 6GB gate throughout) — every C++-side claim
  was verified via direct `nvcc`/`g++` compilation, each confirmed
  freshly built. No performance claim made for any fixture's corrected
  `NUM_PART`. Exact equality `P==Q_part_max` remains untested in
  practice (explained in `include/ks_params_validate.h`'s own honesty
  note — distinct-prime products make literal equality a measure-zero
  coincidence this library's prime generation does not produce).
  `test_polyeval`'s own, separate multi-level noise question remains
  untouched.

## GOD PROMPT "CKKS PHASE 11": first real-world per-stage performance baseline for the complete encode→encrypt→EvalMult→decrypt→decode pipeline — the GPU kernels are NOT the bottleneck (~1% of total latency); two host-side optimizations (parallelized encode/decode, batched encrypt_host) give a verified 2.6x–5.3x total speedup with zero correctness regression

Full detail: `EXPERIMENT_LEDGER.md` entry #44. `NEXT_SESSION.md`'s
"READ THIS FIRST" section rewritten to point future performance work at
the clear next experiment (an O(n log n) FFT-based encode/decode).

- **Environment confirmed**: GPU = RTX 2060 Max-Q (6GiB VRAM, driver
  595.97), CUDA 13.2, `CMAKE_CUDA_FLAGS` empty (no `-G`) in both
  `CMakeLists.txt` and the cached `build2/CMakeCache.txt`. Real Windows
  host free RAM 1.15GB at measurement time (well under the 6GB gate —
  `n=131072` correctly not attempted). Phase 10's `NUM_PART=4` fixes
  confirmed present in both `evalmult.rs` and `keyswitch.rs`.
- **New benchmark**: `bench/bench_evalmult_e2e_breakdown.cpp` —
  per-stage `cudaEvent` + host-clock timing for all 8 pipeline stages
  (context/eval-key creation, encode, encrypt, tensor, relin/keyswitch,
  rescale, decrypt, decode) plus end-to-end, with `cudaMemGetInfo`/
  `getrusage` memory reporting and min/median/p95/max/mean statistics
  — swept across n=1024/2048/4096/8192, every configuration validated
  via `check_hybrid_keyswitch_params` BEFORE running (never bypassed),
  with a correctness gate (RNS-level + real encrypted round trip)
  before trusting any timing. A first attempt failed with a huge error
  (`INT64_MIN`-range garbage) — root-caused to a bug in the NEW
  benchmark itself (calling `encode_host` with an un-padded slot
  vector, undefined behavior; the Rust FFI bridge normally pads it)
  — fixed and confirmed via a standalone diagnostic before trusting any
  further result.
- **Baseline, n=8192, single real round trip**: Encode (host, O(n²))
  310406us (59%); Decode (host, O(n²)) 153870us (29%); Encrypt
  (host+GPU) 50642us (10%); Relin/keyswitch (GPU) 4632us (0.9%);
  Rescale x2 (GPU) 1568us (0.3%); Decrypt 558us; Tensor 131us; Combine
  53us. **Total 522890us, of which host-side encode+decode+encrypt =
  98.5%** — directly confirming the governing directive's own warning
  not to assume the largest CUDA kernel dominates.
- **Optimization 1 (KEPT)**: parallelized `encode_host`/`decode_host`'s
  O(n²) outer loop via `std::thread` (`src/encode.cpp`) — each output
  element is an independent read-only reduction, so splitting across
  threads changes NOTHING about any element's arithmetic (same ops,
  same order, same rounding). Correctness: `tests/
  test_encode_decode_parallel_matches_sequential.cpp`, an INDEPENDENT
  fresh single-threaded reimplementation (not src/encode.cpp itself),
  EXACT bit-identical match required and confirmed across n in
  {4,...,8192} (both sides of the sequential-fallback threshold), real
  and complex slots. A/B (3 interleaved runs, n=8192): Encode
  309-318ms->49-56ms (~5.7-6.4x); Decode 155-157ms->23-26ms
  (~6.1-6.8x). Negligible memory impact.
- **Optimization 2 (KEPT)**: batched `encrypt_host`'s per-tower NTTs
  (`src/encdec.cpp`) — replaced 4*sizeQ separate per-tower-per-buffer
  GPU round trips (each also rebuilding its own root table from
  scratch) with ONE upload + ONE `LaunchNTT_CT_Batched` call + ONE
  download per buffer, root tables cached by `(n,q)` — the SAME pattern
  `src/evalkeygen.cpp` already uses successfully. Correctness: `tests/
  test_encrypt_host_fast_matches_sequential.cpp`, an INDEPENDENT fresh
  reimplementation of the exact pre-Phase-11 unbatched algorithm using
  the SAME seeded RNG, EXACT bit-identical `(c0,c1)` required and
  confirmed across n in {1024,2048,4096} x sizeQ in {1,4,8}. A/B (3
  interleaved runs, n=8192): Encrypt 46-52ms->18-23ms (~2.2-2.8x);
  real-round-trip error magnitude IDENTICAL in every run
  (`1.11944e-10`), confirming bit-exact identical final output.
  Negligible memory impact.
- **Combined result**: total end-to-end latency 30.5ms->11.8ms (2.6x)
  at n=1024; 48ms->16.6ms (2.9x) at n=2048; 154.7ms->38.2ms (4.1x) at
  n=4096; 522.9ms->98.9ms (5.3x) at n=8192 — speedup GROWS with n.
  Every real encrypted round trip passed at every n with the error
  magnitude UNCHANGED from before optimization (bit-exact identical
  cryptographic output). RNS-level checks also passed at every n (0
  mismatches). GPU memory: workload's own incremental footprint stays
  tiny (2-10MB) against a large (~1038-1066MB) CUDA-runtime baseline,
  unaffected by either optimization. Host RSS: 100-121MB, no
  meaningful change. Full Rust suite: **38 passed, 0 failed, 1
  ignored**, identical in `--release` and debug — zero regression.
- **POST-optimization ranked bottleneck, n=8192**: Encode 53.0ms (54%);
  Decode 23.2ms (23%); Encrypt 17.4ms (18%); Relin/keyswitch (GPU)
  1.96ms (2%); Rescale (GPU) 1.63ms (1.6%); Decrypt 0.54ms;
  Tensor+Combine (GPU) ~0.07ms. Host-side encode+decode+encrypt STILL
  ~95% of total latency even after both optimizations.
- **Experiments considered and NOT pursued, documented not dropped**:
  a persistent thread pool for encode/decode (estimated ~1-3% further
  gain, not worth the added lifecycle complexity); a true O(n log n)
  FFT-based replacement for encode/decode's O(n²) algorithm — correctly
  identified as the single largest remaining opportunity (would address
  the now-dominant ~77% of latency) but requires non-trivial
  re-derivation of the `r_k=5^k mod 2N` permuted-slot convention against
  a standard FFT, carrying meaningfully higher correctness risk — NOT
  attempted this round, recorded as the clear next highest-value
  experiment. Explicit CUDA-kernel-level optimization (fusion,
  occupancy, shared memory) was NOT pursued since profiling
  conclusively showed the GPU kernels are not the bottleneck at any
  tested n.
- `CMakeLists.txt` gained `bench_evalmult_e2e_breakdown`, `test_encode_
  decode_parallel_matches_sequential`, `test_encrypt_host_fast_matches_
  sequential`, plus `find_package(Threads REQUIRED)`/`Threads::Threads`
  linked into the `gpufhe` target. `rust/crates/ckks-cuda/build.rs`
  gained an explicit `pthread` link line. Full CMake rebuild NOT
  attempted (host free memory 1.15GB, under the 6GB gate) — every claim
  verified via direct `nvcc`/`g++` compilation, each confirmed freshly
  built.

## GOD PROMPT "CKKS PHASE 12": the O(n log n) FFT-based encode/decode flagged at the end of Phase 11 is derived, independently verified three ways, and now the real production path — 31-87x standalone speedup, 2.2-4.2x additional full-pipeline speedup, zero correctness regression

Full detail: `research/FFT_ENCODING_DERIVATION.md` (the dedicated
derivation/verification document this phase's directive required) and
`EXPERIMENT_LEDGER.md` entry #45. Starting commit `a561991` (Phase 11
end state), confirmed via `git log -1` before any change.

- **Derivation (Section 2 of the directive) done on paper first, then
  independently checked against numpy — not assumed correct and
  written straight into C++.** `src/encode.cpp`'s own existing
  `encode_host`/`decode_host` formula (unchanged, re-derivation target
  only) reduces to: decode = one bin of a size-`M=2N` forward DFT of
  the zero-padded coefficient array (`m(ζ^{-t})` for `ζ=e^{iπ/N}`,
  gathered at `t=(M-r_k)%M`); encode = scatter `z_k`/`conj(z_k)` at
  `(M-r_k)%M`/`r_k`, take the UNNORMALIZED inverse DFT, scale by
  `Delta/N` (**not** `2*Delta/N` — the scatter of conjugate pairs
  already doubles the raw sum; this factor-of-2 subtlety was caught by
  the numpy prototype on its first run, before any C++ existed, and is
  the one real mistake this derivation made and fixed in time**).
  Full mathematical mapping (slot count/conjugate symmetry, the
  `r_k=5^k mod 2N` generator-group structure, FFT sign convention,
  every scaling/permutation/conjugation step) recorded in
  `FFT_ENCODING_DERIVATION.md` §1-§3.
- **Standalone experiment (Section 3)**: new, separate file
  `src/encode_fft.cpp` — a standard cached-twiddle iterative radix-2
  Cooley-Tukey FFT plus the `r_k`/`gidx_k` gather/scatter derived
  above. `src/encode.cpp` was NOT touched and remains the reference/
  fallback, exactly as the directive requires.
- **Numerical correctness (Section 4)**: `tests/
  test_encode_fft_matches_reference.cpp` compares the new FFT against
  the UNTOUCHED O(n²) implementation (a genuinely independent oracle —
  different algorithm, not a different thread count) across n in
  {16,256,1024,2048,4096,8192} and 9 input categories including
  deliberately adversarial (alternating ±1, 1000x magnitude-spread)
  cases. Decode matches to ~1e-22..1e-26 absolute error everywhere.
  Encode's larger-looking discrepancies (up to 80 units at n=16) were
  investigated, not dismissed: independently computed via `mpmath` at
  60-digit precision, which proved the TRUE pre-rounding value is
  exactly/near zero for the near-cancellation cases and ~5e15 for the
  magnitude-spread case — both implementations are correct to many
  more significant figures than CKKS itself ever uses. A principled
  (not loosened) tolerance, `(N+1000)*DBL_EPSILON*Delta` (combining an
  N-term accumulation bound with the 1000x magnitude-spread
  conditioning bound this test's own worst case exercises), passes
  every case — this is ~1e-10 RELATIVE to Delta, far below the
  project's own 1e-6 real-round-trip tolerance.
- **Real cryptographic correctness (Section 5)**: `cpp_bridge/
  encdec_bridge.cpp`'s `gpufhe_encode`/`gpufhe_decode` — the actual
  production call site every Rust caller (`codec::encode_real`/
  `codec::decode`) reaches — now call `encode_host_fft`/
  `decode_host_fft`; `encode_host`/`decode_host` remain fully present
  and directly callable as reference/fallback. Full Rust suite after
  the swap: **38 passed, 0 failed, 1 ignored**, identical in
  `--release` and debug — zero regression, including every
  `real_encrypted_evalmult_*` test. Decoded error magnitude essentially
  unchanged (`2.60e-11` after vs. `2.59e-11` before).
- **Benchmarks (Section 6), interleaved A/B, old implementation kept
  runnable throughout**: standalone (`bench/
  bench_encode_fft_vs_parallel.cpp`, no GPU/encryption, 15 trials/N) —
  encode 31.4x(n=2048)-61.2x(n=8192), decode 29.6x(n=2048)-81.2x
  (n=8192); n=1024: encode 42.1x, decode 53.0x; n=4096: encode 43.2x,
  decode 54.3x. Full real-encrypted pipeline (`bench/
  bench_evalmult_e2e_breakdown.cpp`, extended this round with a
  `useFft` A/B flag) — total latency 2.19x(n=1024)-4.15x(n=8192) faster
  on top of Phase 11's already-applied optimizations. **Cumulative
  speedup since the pre-Phase-11 original baseline: 21.5x at n=8192**
  (522890us -> 24335us).
- **Implementation choice (Section 7)**: deliberately the SIMPLEST
  correct construction — zero-pad to `M=2N`, one standard FFT,
  gather/scatter half — accepting a ~2x constant-factor overhead vs. a
  tighter N-point negacyclic-twisted FFT in exchange for needing no
  mathematical object beyond a textbook DFT and the SAME `r_k` array
  the existing code already computes. The O(n log n) vs. O(n²) win
  dominates this constant factor at every n tested (31-87x, far more
  than a 2x loss could ever cost). Deliberately NOT thread-parallelized
  (a single FFT is an inherently sequential dependent-stage pipeline;
  marginal gain not worth the complexity when already 31-61x faster
  than the parallelized O(n²) baseline) — no interaction with Phase
  11's `std::thread` pool in the untouched `src/encode.cpp`.
- `CMakeLists.txt` gained `src/encode_fft.cpp` in the main library's
  source list, `test_encode_fft_matches_reference`, and
  `bench_encode_fft_vs_parallel`. `rust/crates/ckks-cuda/build.rs`
  gained compilation of `src/encode_fft.cpp`. Full CMake rebuild NOT
  attempted this round either (host free memory still well under the
  6GB gate) — every claim verified via direct `nvcc`/`g++` compilation
  to standalone binaries, each confirmed freshly built, plus the full
  `cargo test` suite (which DOES build and link everything through
  `build.rs`).
- **Post-Phase-12 bottleneck, n=8192**: `encrypt_host` (17.0ms) is now
  the single largest contributor to the real round trip, followed by
  GPU keyswitch (2.0ms) and rescale (1.9ms). Encode (1.16ms) and decode
  (0.29ms) are no longer meaningful contributors — the clear next
  highest-value experiment is profiling `encrypt_host`'s own internal
  cost breakdown (NTT vs. embedding/combine loops), not another pass
  over encode/decode.

## GOD PROMPT "CKKS PHASE 13": deep-profiled `encrypt_host` — NOT the already-batched NTT, but two host-side loops (embed + combine) dominate; combine's modular multiply was calling libgcc's slow `__umodti3`; fixed with Shoup modmul + a device-buffer pool, catching and fixing a real concurrency bug along the way — 1.10x-1.84x faster, zero correctness regression

Full detail: `research/ENCRYPT_PROFILING.md` (the dedicated profiling/
derivation document this phase's directive required) and
`EXPERIMENT_LEDGER.md` entry #46. Starting commit `4a62f0a` (Phase 12
end state), confirmed via `git log -1` before any change.

- **Fresh baseline re-established, not reused from Phase 12's closing
  estimate**: confirmed `HEAD=4a62f0a` matches `origin/main`, no `-G`,
  16 CPU threads, GPU 5955MiB free, host free RAM 2.53GB (gate not
  cleared, `n=131072` correctly not attempted). Re-ran the existing
  Rust suite clean (38/0/1) and `bench_evalmult_e2e_breakdown.cpp`
  before touching anything — encrypt_host's own single-call latency at
  n=8192 (~8.8-9.7ms/call) matched Phase 12's closing "~17.0ms total
  (x2 calls)" figure, confirming it was real, not stale.
- **Internal stage-level profiling (Section 4 of the directive) done
  via an instrumented BYTE-FOR-BYTE copy** (`bench/
  bench_encrypt_profile.cpp`'s `encrypt_host_profiled`), verified
  bit-exact against production before any timing was trusted. Found
  the ALREADY-batched NTT kernel is a SHRINKING share of total time as
  n grows (13.0%→5.7%) — the governing directive's own explicit warning
  not to assume otherwise was correct. The real bottlenecks are two
  host-side loops: **embed** (RNS embedding, 20-27% of total, the
  single largest stage at every n) and **combine** (final ciphertext
  assembly, 17-24%) — together ~37-45% of total time. **Device
  allocation+free is a third, size-dependent cost** (10-27% combined),
  far from the "small constant overhead" one might assume.
- **Root cause of combine's cost confirmed via direct assembly
  inspection, not assumed**: every host-side modular multiply in this
  ENTIRE codebase (`src/keygen.cpp`, `src/evalkeygen.cpp`, `src/
  keyswitch_resident.cpp`, `src/ks_constants.cpp`, and `encrypt_host`'s
  own `mm()`) is written as `(uint64_t)(((unsigned
  __int128)a*b)%q)` — compiling this expression in isolation and
  checking the object file with `nm` confirms it calls libgcc's
  `__umodti3`, a slow software 128-bit division, not a hardware
  instruction.
- **Experiment 1 (KEPT): Shoup's precomputed-constant modmul in the
  combine loop.** New file `src/encrypt_host_shoup.cpp`
  (`encrypt_host_shoup`), reusing the SAME technique already validated
  in this codebase's own CUDA kernels (`kernels/intt_ct.cu`'s/`kernels/
  basis_convert.cu`'s device `shoup_mulmod`). `ns%q`'s precon is
  computed once per tower (not per coefficient); `v`'s NTT-domain value
  is reused as the fixed multiplicand for both `pkB*v` and `pkA*v`,
  halving the precon computations needed. Bit-exact verified against
  the untouched `encrypt_host` across 60 (n,sizeQ,message)
  combinations (`tests/test_encrypt_host_shoup_matches_reference.cpp`)
  — zero tolerance anywhere, Shoup reduction is exact. Measured
  1.04x-1.19x faster than OLD at every n, interleaved A/B, consistent
  across multiple independent runs.
- **Experiment 2 (KEPT, after fixing a real concurrency bug): a
  process-lifetime device-buffer pool keyed by `(n,sizeQ)`.** New file
  `src/encrypt_host_shoup_pooled.cpp` (`encrypt_host_shoup_pooled`),
  builds on Experiment 1 by also eliminating the repeated
  `cudaMalloc`/`cudaFree` of the four per-call device buffers (same
  caching pattern already used for immutable root tables, applied here
  to mutable per-call DATA buffers that are always fully overwritten
  before use). **Found and fixed a real data race, not a theoretical
  one**: after wiring this into the production FFI bridge, `cargo test
  --release` (default PARALLEL execution) FAILED 4/38 tests with wrong
  decrypted values; `--test-threads=1` passed 38/38 cleanly, isolating
  the cause to a race (two threads corrupting the SAME shared pooled
  buffer), not an arithmetic defect. Fixed with one `std::mutex`
  serializing the function body — negligible overhead for the real
  (sequential) calling pattern. Verified with a dedicated multi-
  threaded C++ stress test (`tests/
  test_encrypt_host_shoup_pooled_thread_safety.cpp`, 8 threads x 6
  concurrent calls at the same key, run 3x): **0/48 mismatches every
  run.** After the fix, full Rust suite re-confirmed **38 passed, 0
  failed, 1 ignored** under cargo's default PARALLEL execution, in both
  `--release` and debug. Pooling's own incremental benefit (on top of
  Shoup) ranges 1.00x-1.66x — real and never negative, but strongly
  n-dependent (largest at n=2048, smallest at n=1024/4096), reproduced
  across independent runs; exact mechanism (likely CUDA allocator
  sub-allocation behavior) not independently confirmed — no per-process
  GPU allocator tracing is available in this WSL2 setup (same
  limitation `MEMORY_FEASIBILITY.md` already documented for GPU memory
  investigations).
- **Production swap**: `cpp_bridge/encdec_bridge.cpp`'s `gpufhe_encrypt`
  now calls `encrypt_host_shoup_pooled`; `encrypt_host` (`src/
  encdec.cpp`) is UNCHANGED and remains fully present, tested, and
  directly reachable as reference/fallback. Zero change to randomness,
  Gaussian sampling, ternary sampling, or the embedding formula anywhere
  — verified both by direct inspection and by the bit-exact correctness
  tests themselves (any divergence in sampling would show up as an
  immediate mismatch; none did). The concurrency bug was a race on
  shared DEVICE BUFFER CONTENT, never on RNG state (`std::mt19937_64
  rng(seed)` is always a per-call local variable) — no cryptographic
  property was ever at risk.
- **Combined result**: Shoup+pooled measured 1.10x-1.84x faster than
  OLD `encrypt_host` across n=1024-8192 (interleaved A/B/C, multiple
  independent runs, `bench/bench_encrypt_all_variants.cpp`). Full real-
  encrypted pipeline impact (5-sample medians per n, since single-call
  timing is cold-start-noisy): 1.02x-1.18x faster end-to-end at every n
  on top of Phase 12's own gains, never a regression. Every correctness
  gate passed: RNS-level (0 mismatches), real encrypted round trip
  (error unchanged from Phase 12, `~5e-12`-`~1e-10` vs. `1e-6`
  tolerance), full Rust suite (38/0/1, both build profiles, default
  parallel execution).
- `CMakeLists.txt` gained `src/encrypt_host_shoup.cpp` and `src/
  encrypt_host_shoup_pooled.cpp` in the main library's source list,
  3 new correctness tests, and 4 new benchmarks. `rust/crates/ckks-
  cuda/build.rs` gained compilation of `src/
  encrypt_host_shoup_pooled.cpp` (the one new file the production
  bridge actually calls). Full CMake rebuild NOT attempted this round
  either (host free memory 1.8-2.5GB throughout, under the 6GB gate) —
  every claim verified via direct `nvcc`/`g++` compilation to
  standalone binaries plus the full `cargo test` suite (which DOES
  build and link everything through `build.rs`).
- **Post-Phase-13 remaining opportunity**: `embed` (RNS embedding, NOT
  touched by either optimization this round) is now almost certainly
  the single largest remaining cost inside `encrypt_host_shoup_pooled`
  — same double-loop shape as `combine` but using signed hardware
  division (`%`) rather than `__umodti3`, a smaller but still real
  candidate for a future round. The GPU-resident tensor+keyswitch+
  combine+rescale chain remains ~1% of total pipeline latency
  (unchanged conclusion since Phase 11).

## GOD PROMPT "CKKS PHASE 14": audited Phase 13's device-buffer pool (found it already sound, hardened with 4 new targeted concurrency tests) and optimized `embed` — the remaining largest stage — with a per-tower Barrett reduction, derived for the FULL int64_t range including LONG_MIN, validated 22M+ times with zero mismatches — 1.44x-1.73x faster `encrypt_host`, 1.17x-1.36x faster full pipeline, zero correctness regression

Full detail: `research/EMBED_AND_POOL_HARDENING.md` and
`EXPERIMENT_LEDGER.md` entry #47. Starting commit `4d46420` (Phase 13
end state), confirmed via `git log -1` before any change.

- **Buffer-pool audit (Section 3 of the directive) done as a real
  line-by-line trace, not a rubber stamp**: ownership, lifetime, keys,
  mutex scope, CUDA stream association, sync-before-reuse, failure
  paths, shutdown, concurrent contexts, different key material,
  reentrancy, and exception paths were all traced through `src/
  encrypt_host_shoup_pooled.cpp`'s existing Phase 13 implementation.
  **Conclusion: the single global mutex (serializing the whole
  function body) already covers every scenario correctly** — no new
  safety issue was found, so the pool's existing semantics were REUSED
  unchanged for Phase 14's new function (per the directive's own "if a
  safety issue is discovered, fix it before expanding the pool's
  role" — none was).
- **4 new targeted concurrency tests built to get positive evidence,
  not just a paper audit** (`tests/
  test_encrypt_host_shoup_pooled_barrett_concurrency_audit.cpp`): same-
  key concurrent reuse (8 threads x 8 calls), concurrent pool growth on
  8 simultaneously-first-used fresh keys, 200-iteration single-key
  longevity stress, and a mixed-shared-key race (10 threads cycling 2
  keys). **5 independent runs: 0/396 mismatches every time.** (One bug
  caught in the TEST ITSELF along the way: several non-power-of-2 `n`
  values caused an immediate segfault in the NTT's bit-reversal logic
  — fixed before trusting any result; `n` must be a power of 2
  throughout this codebase, same as every other `n` used elsewhere.)
- **Main experiment: `embed`, encrypt_host's remaining largest stage
  (20-27% of total per Phase 13's own profile).** Assembly inspection
  (per the directive's explicit instruction, not assumed) confirmed
  the existing `emb(x)=(x%q+q)%q` compiles to TWO hardware `idiv`
  instructions per call — a real cost, but NOT a slow software call
  like Phase 13's `__umodti3` finding, since `q` here fits safely in
  signed 64-bit range and the one UB case (`INT64_MIN % -1`) can never
  trigger (the divisor is always a positive CKKS modulus).
- **Derivation**: since `q` is the SAME for all `n` coefficients within
  a tower, a Barrett reduction constant (`mu = floor(2^64/q)`) is
  precomputed ONCE PER TOWER (not per coefficient, unlike Shoup's
  per-coefficient precon in Phase 13's combine fix). The embedding
  loop's signed input `x` (which can span `m[k]`'s FULL `int64_t`
  range, not just small values) is reduced via an unsigned-negation
  trick (`ax = (x<0) ? (0ULL-(uint64_t)x) : (uint64_t)x`) that is
  well-defined for EVERY `int64_t` including `LONG_MIN` (no signed-
  overflow UB anywhere), then Barrett-reduces `ax mod q`, then restores
  `emb()`'s own sign convention. New file `src/
  encrypt_host_shoup_pooled_barrett.cpp`
  (`encrypt_host_shoup_pooled_barrett`), builds on Phase 13's already-
  verified Shoup-combine + buffer-pool implementation unchanged except
  for this one loop.
- **Independent differential-test validation BEFORE touching
  production** (directive Section 6): a standalone scratch harness
  (12,000,102 checks across synthetic + realistic moduli, boundary
  cases including `LONG_MIN`/`LONG_MAX`, 2M random full-range values
  per modulus) plus a production-moduli test (`tests/
  test_embed_barrett_matches_reference.cpp`, 10,400,884 checks against
  this project's own REAL `native_primes()`-generated moduli at every
  supported n/sizeQ) — **22,400,986 total checks, ZERO mismatches**,
  exact integer arithmetic, no tolerance anywhere.
- **Production-integration correctness** (`tests/
  test_encrypt_host_shoup_pooled_barrett_matches_reference.cpp`):
  bit-exact vs. the untouched `encrypt_host` across n/sizeQ combinations
  x message patterns including `LONG_MIN`/`LONG_MAX`-adjacent and dense
  mixed-sign/magnitude, plus the pool-reuse/interleaved-different-keys
  stress sections carried forward from Phase 13 — every case EXACT
  MATCH.
- **Production swap**: `cpp_bridge/encdec_bridge.cpp`'s `gpufhe_encrypt`
  now calls `encrypt_host_shoup_pooled_barrett`; `encrypt_host_shoup_
  pooled` (Phase 13) and `encrypt_host` (Phase 11) both remain fully
  present, tested, and directly reachable as reference/fallback. Full
  Rust suite re-confirmed **38 passed, 0 failed, 1 ignored** across 5
  separate runs under cargo's default PARALLEL execution — zero
  flakiness, both `--release` and debug.
- **Benchmarks, interleaved, old implementations kept runnable
  throughout**: embed loop alone 1.97x(n=8192)-5.13x(n=1024) faster
  (speedup shrinks with n, consistent with the arithmetic advantage
  mattering less as the working set becomes more memory/cache-bound).
  Full `encrypt_host`, unified 3-way interleaved, run twice
  independently: **1.44x-1.73x faster than the original Phase-11
  encrypt_host, 1.18x-1.28x faster than Phase 13 alone**, consistent
  and positive at every n both times. Full real-encrypted pipeline
  (5-sample medians per n, since single-call timing is cold-start-
  noisy): **1.17x-1.36x faster than OLD on top of Phase 12's own
  encode/decode gains**, never a regression. Every correctness gate
  passed: RNS-level (0 mismatches), real encrypted round trip (error
  magnitude IDENTICAL across all three encrypt variants at every n,
  `~5e-12`-`~1e-10` vs. `1e-6` tolerance).
- `CMakeLists.txt` gained `src/encrypt_host_shoup_pooled_barrett.cpp`
  in the main library's source list, 3 new correctness tests, and 2 new
  benchmarks. `rust/crates/ckks-cuda/build.rs` gained compilation of
  the one new file the production bridge actually calls. Full CMake
  rebuild NOT attempted this round either (host free memory 1.8-3.1GB
  throughout, under the 6GB gate) — every claim verified via direct
  `nvcc`/`g++` compilation to standalone binaries from a CLEAN,
  from-scratch object-cache rebuild, plus the full `cargo test` suite.
- **Post-Phase-14 remaining opportunity**: both of `encrypt_host`'s two
  largest host-side loops (embed and combine) are now optimized, and
  device alloc/free is already pooled. The clear next candidate is the
  upload/download transfer stage (19-31% combined, not yet touched) —
  specifically whether the four separate per-buffer `cudaMemcpy` calls
  (`d_v,d_e0,d_e1,d_m`) can be merged into fewer, larger transfers.

## GOD PROMPT "CKKS PHASE 15": merged encrypt_host's 8 separate transfer calls into 2 over one packed buffer (1.07x-1.17x faster) — then measured a SECOND pinned-memory experiment that looked even better in isolation, caught it was actually a net loss for real usage via the full-pipeline benchmark, and correctly did NOT ship it

Full detail: `research/TRANSFER_PROFILING.md` and
`EXPERIMENT_LEDGER.md` entry #48. Starting commit `7ea73fd` (Phase 14
end state), confirmed via `git log -1` before any change.

- **Transfer data flow traced (Section 4 of the directive) before
  changing anything**: `d_v/d_e0/d_e1/d_m` are four separate pooled
  device buffers, each uploaded/downloaded via its own blocking
  `cudaMemcpy` from/to a pageable `std::vector`. All four share
  identical size, lifetime, and transfer pattern (same per-tower
  modulus/root tables, identical `LaunchNTT_CT_Batched` calls differing
  only in which pointer is passed) — confirmed safe to pack together,
  not assumed.
- **Experiment 1 (KEPT): batched transfers.** `src/
  encrypt_host_shoup_pooled_barrett_batched.cpp` replaces the four
  separate device buffers with ONE pooled packed buffer
  (`[v|e0|e1|m]`, `4*sizeQ*n*8` bytes) and merges the 8 separate
  transfer calls into 2 (one H2D, one D2H) — the FOUR kernel launches
  themselves are completely unchanged, just receiving pointer offsets
  into the packed region instead of four separate buffers. Bit-exact
  verified against `encrypt_host`, including a dedicated
  packed-buffer-boundary stress case (distinct extreme values at the
  first/last tower of each sub-buffer, to catch an off-by-one in the
  offsets). The new pooled device buffer's concurrency safety was
  re-audited (not assumed to inherit Phase 14's proof) with 4 targeted
  multi-threaded scenarios, 5 independent runs, **0/396 mismatches
  every run**. Measured **1.57x-2.78x faster than the original
  encrypt_host, 1.07x-1.17x faster than Phase 14 alone**, run twice
  independently, consistent both times.
- **Experiment 2 (measured, NOT promoted): pinned host staging
  memory.** Built on top of Experiment 1 — `src/
  encrypt_host_shoup_pooled_barrett_batched_pinned.cpp` additionally
  pools the host-side packed buffer as pinned (`cudaMallocHost`)
  memory. Also bit-exact verified and concurrency-audited the same
  way (5 runs, 0/396 mismatches). A standalone, properly-warmed-up
  benchmark showed a genuine further **1.06x-1.17x steady-state
  speedup**. **But wiring it into the full-pipeline benchmark's
  single-call (cold) measurement showed the OPPOSITE result** —
  slower than Phase 14 — which was investigated directly rather than
  dismissed: a dedicated timing test found `cudaMallocHost` costs
  **~3-8ms per distinct `(n,sizeQ)` key** on first use (vs. ~0.4-0.5ms
  for ordinary `cudaMalloc`), larger than `encrypt_host`'s entire
  typical latency — meaning any usage pattern that doesn't repeat
  enough calls at one key to amortize this cost would see Experiment 2
  as a net LOSS overall, despite its real steady-state benefit. **Per
  the governing directive's own explicit instruction ("if pinned
  memory introduces more overhead than it saves, revert it"), this
  was NOT integrated into production** — kept in the repository as a
  fully tested, honestly-reported experiment rather than silently
  discarded.
- **Production swap**: `cpp_bridge/encdec_bridge.cpp`'s `gpufhe_encrypt`
  now calls `encrypt_host_shoup_pooled_barrett_batched` (Experiment 1
  only); `encrypt_host_shoup_pooled_barrett` (Phase 14) and
  `encrypt_host` (Phase 11) both remain fully present, tested, and
  directly reachable as reference/fallback. Full Rust suite
  re-confirmed **38 passed, 0 failed, 1 ignored** across 4 separate
  release runs under cargo's default PARALLEL execution — zero
  flakiness, both `--release` and debug.
- **Full real-encrypted pipeline impact** (5-sample medians per n,
  since single-call timing is cold-start-noisy — this exact noisiness
  is what exposed Experiment 2's cold-start cost in the first place):
  **1.14x-1.33x faster than OLD, 1.04x-1.05x faster than Phase 14
  alone**, on top of Phase 12's encode/decode gains, never a
  regression. Every correctness gate passed: RNS-level (0 mismatches),
  real encrypted round trip (error IDENTICAL across all five measured
  encrypt variants at every n, `~5e-12`-`~1e-10` vs. `1e-6` tolerance).
- `CMakeLists.txt` gained both new source files (Experiment 1 and
  Experiment 2, both kept for the historical record) in the main
  library's source list, 4 new correctness/concurrency tests, and 3
  new benchmarks. `rust/crates/ckks-cuda/build.rs` gained compilation
  of ONLY `src/encrypt_host_shoup_pooled_barrett_batched.cpp` (the one
  file the production bridge actually calls — the pinned variant was
  deliberately NOT added to `build.rs`, consistent with this
  project's established convention of only compiling production-
  reachable files into the Rust-linked object set). Full CMake rebuild
  NOT attempted this round either (host free memory 1.2-2.4GB
  throughout, under the 6GB gate) — every claim verified via direct
  `nvcc`/`g++` compilation to standalone binaries from a CLEAN,
  from-scratch object-cache rebuild, plus the full `cargo test` suite.
- **Post-Phase-15 remaining opportunity**: embed, combine, and
  transfer/allocation are all now optimized. The clear next candidate
  is revisiting Experiment 2 with an explicit pre-warming scheme —
  allocating the pinned pool entry at context-creation time (when a
  one-time cost is expected and acceptable) rather than lazily on the
  first real encryption call — which would let production capture the
  proven steady-state win without the cold-start regression risk this
  round identified.

## GOD PROMPT "CKKS PHASE 16": built, proved correct, and concurrency-hardened an explicit pinned-memory prewarming API — found the allocation cost itself is unpredictable in this environment, so retained it as an opt-in experimental feature rather than promoting it to production

Full detail: `research/TRANSFER_PROFILING.md`'s Phase 16 section and
`EXPERIMENT_LEDGER.md` entry #49. Starting commit `69d307a` (Phase 15
end state), confirmed via `git log -1` before any change. Direct
follow-up to Phase 15's own closing recommendation.

- **Lifecycle designed before implementation**: `encrypt_prewarm_
  pinned_buffer(n,sizeQ)` (explicit, idempotent, checks every CUDA
  allocation's return code — the one deliberate, narrowly-scoped
  exception to this codebase's "don't check CUDA errors" convention,
  justified because this function's whole contract is failure-safety)
  → `encrypt_pinned_buffer_is_ready(n,sizeQ)` (query-only) →
  `encrypt_host_shoup_pooled_barrett_prewarmed(...)` (adaptive encrypt:
  uses the pinned path ONLY if ready, otherwise transparently defers
  to the UNMODIFIED production pageable path, never allocating pinned
  memory itself). A failed prewarm rolls back whatever subset
  succeeded and is NEVER inserted into the cache as "ready" — no code
  path can leave a half-initialized buffer observable to encryption.
  An application that never calls `encrypt_prewarm_pinned_buffer` pays
  nothing extra and behaves exactly like today's production path.
- **Ownership model**: a SHARED, PROCESS-LEVEL pool keyed by
  `(n,sizeQ)` — the SAME model every other cache in this codebase
  already uses, deliberately NOT a new per-context model (this C++
  layer has no context object for `encrypt_host` to begin with).
- **Correctness**: bit-exact vs. the untouched `encrypt_host` across
  both the fallback path and the prewarmed path, incl. idempotency and
  pool-reuse stress — every case EXACT MATCH
  (`tests/test_encrypt_host_prewarm_matches_reference.cpp`).
- **Concurrency/lifecycle, 6 scenarios re-audited (not assumed from
  Phase 13-15's own proofs)**: same-key concurrent prewarm, different-
  key concurrent prewarm, simultaneous encrypt at a prewarmed key,
  prewarm/encrypt interleaved on a fresh key (both race outcomes
  verified correct), 200-cycle longevity, mixed shared-key race. 5
  independent runs: **0/372 mismatches, 0 prewarm-consistency
  failures, every run.**
- **An unexpected, reproducible finding**: the one-time `cudaMallocHost`
  cost itself is NOT simply a function of buffer size in this
  environment — n=1024/8192 consistently cost several milliseconds
  while n=2048/4096 consistently cost under 20 microseconds, across
  two independent full benchmark runs in the SAME order, plausibly
  tied to this session's persistent host-memory pressure interacting
  with WSL2's dynamic-memory ballooning (not independently
  confirmable with the tooling available — reported honestly as
  "plausible, not confirmed," not asserted as fact).
- **Workload profile comparison** (directive's three required
  profiles, computed from directly-measured quantities): Profile C
  (long-lived, 500-5000 ops) wins UNCONDITIONALLY at every tested n.
  Profiles A (cold-start, 1-3 ops) and B (typical, 20 ops) are MIXED —
  winning or losing depending on which unpredictable allocation-cost
  regime a given key happened to land in, not on the workload itself.
- **Memory overhead**: 6.00MB GPU + 6.25MB host RSS for 8 prewarmed
  keys — small and bounded.
- **Decision: RETAINED AS AN OPT-IN EXPERIMENTAL C++ FEATURE — NOT
  promoted to production, NOT wired into `cpp_bridge/encdec_bridge.cpp`,
  NOT exposed through the Rust API this round.** The implementation is
  fully correct and fully concurrency-safe (not the reason it wasn't
  promoted) — the reason is that its performance benefit depends on an
  allocation-cost variance this library cannot predict or control,
  which is a materially weaker claim than every other kept optimization
  in Phases 11-15 made. `encrypt_host_shoup_pooled_barrett_prewarmed`,
  `encrypt_prewarm_pinned_buffer`, and `encrypt_pinned_buffer_is_ready`
  remain in the repository, fully built, tested, and registered in
  `CMakeLists.txt`, available for a future round that either resolves
  the allocation-cost variance's root cause or accepts the risk
  explicitly for a specific, named long-lived-service deployment.
- `CMakeLists.txt` gained the new source file, 2 new tests, and 1 new
  benchmark. `rust/crates/ckks-cuda/build.rs` and `cpp_bridge/
  encdec_bridge.cpp` were deliberately left UNCHANGED this round — no
  production or Rust-API surface change. Full Rust suite unaffected:
  38 passed, 0 failed, 1 ignored, both build profiles. Full CMake
  rebuild not attempted (host free memory ~1.3GB throughout, under the
  6GB gate) — every claim verified via direct `nvcc`/`g++` compilation
  from a clean, from-scratch object-cache rebuild, plus the full
  `cargo test` suite.
- **Post-Phase-16 remaining opportunity**: if a future round wants to
  pursue pinned prewarming further, the highest-value next step is
  getting visibility into WSL2's VM memory-ballooning behavior
  directly (this round could only observe its plausible EFFECT, not
  confirm the mechanism) — or, more simply, accepting the current
  implementation as-is for a specific long-lived-service use case where
  Profile C's unconditional win is what matters. Otherwise, encrypt_
  host's host-side arithmetic and transfer stages are all now
  optimized (Phases 13-15); the GPU-resident tensor+keyswitch+combine+
  rescale chain remains ~1% of total pipeline latency, unchanged since
  Phase 11.

## GOD PROMPT "CKKS PHASE 17": diagnosed Phase 16's unexplained allocation-cost variance — root mechanism demonstrated to be the Linux kernel's own memory-locking (`mlock()`) path, NOT CUDA-specific; no production change justified, Phase 16's decision stands on stronger evidence

Full detail: `research/TRANSFER_PROFILING.md`'s Phase 17 section and
`EXPERIMENT_LEDGER.md` entry #50. Starting commit `baee7a1` (Phase 16
end state), confirmed via `git log -1` before any change. **No
production or experimental code was modified this round** — this was
a pure diagnostic investigation.

- **Found a real methodology flaw in Phase 16's OWN measurement
  first**: its "prewarm cost" bundled FIVE separate CUDA calls
  (`cudaMalloc` x4 + `cudaMallocHost` x1) into one timed number,
  exactly the kind of combined measurement the governing directive
  explicitly warned against. Built 3 new, properly-separated standalone
  diagnostics instead (`bench/bench_pinned_alloc_diagnostic{,2}.cpp`,
  `bench/bench_thp_diagnostic.cpp`).
- **Experiment 1 (fresh-allocate-then-free)**: with memory always
  freed immediately, `cudaMallocHost` costs a CONSISTENT 2.5-4.2ms for
  EVERY size — flatly contradicting Phase 16's reported "some sizes
  are cheap" pattern. This alone proves that pattern is not an
  intrinsic property of size.
- **Experiment 2 (never-free, matching real pooling behavior)**:
  reproduced Phase 16's exact slow/fast-by-POSITION pattern identically
  across 5 independent runs — real, but NOT simply "first call slow,
  rest fast" (position 4 slow despite 3 live allocations already
  existing) and NOT simply size-dependent (the SAME size, n=2048, was
  fast at one position and slow at another in the SAME process).
- **Experiment 3 (pure Linux `mmap`+`madvise`+`mlock`, ZERO CUDA code)**:
  reproduced the SAME qualitative variance (117us-3.6ms for identical
  sizes) with no NVIDIA driver involved at all — the single most
  important finding this round. The mechanism lives in the Linux
  kernel's own memory-locking path, not in CUDA's allocator
  (Hypothesis C effectively ruled out as primary).
- **Experiment 4**: refuted "exactly 2MiB triggers THP" as a clean,
  sole explanation — a 2MiB-minus-64KB allocation was equally slow once
  several live pinned allocations already existed.
- **Context gathered, not asserted as cause**: this session's live
  shells measure `ulimit -l` (max locked memory) at only 64MB despite
  `/etc/wsl.conf`'s boot-time "ulimit -l unlimited" intent — that
  boot-time setting does not appear to propagate to ordinary exec'd
  sessions. The tested buffer sizes stay far below 64MB even
  cumulatively, so this is reported as correctly-identified context,
  not claimed as the explanation. Windows host free memory (1910.9MB
  → 1825.3MB, an 85MB drop) was bracketed before/after a batch of
  allocations — consistent with, but not proof of, host memory
  pressure as a contributing amplifier (coarse correlation only).
- **Ranked root-cause assessment**: (1) Linux kernel `mlock()`-path
  variance — HIGH confidence, directly demonstrated. (2) Host/WSL2
  memory pressure as an amplifier — MEDIUM confidence, plausible,
  not isolated from confounds. (3) Allocation size combined with
  growing cumulative never-freed pinned memory — MEDIUM confidence,
  real but not a clean deterministic rule. (4) WSL2-specific Hyper-V
  ballooning as a mechanism DISTINCT from ordinary kernel memory
  pressure — NOT CONFIRMED, cannot be distinguished from (1)/(2) with
  available tooling (this round's finding makes a WSL2-SPECIFIC
  mechanism less necessary as an explanation than Phase 16 implied,
  since ordinary Linux `mlock()` under pressure already suffices as a
  candidate). (5) CUDA-driver-specific behavior — LOW confidence,
  effectively ruled out. (6) Benchmark/measurement artifact — RULED
  OUT, reproduced across three independently-written programs.
- **Decision: category 3 — the variance is externally controlled (now
  DEMONSTRATED, not merely suspected), so Phase 16's opt-in design
  remains unchanged.** No `.wslconfig` change, no WSL restart, no
  system setting was modified — all investigation was read-only, per
  the governing directive's explicit instruction.
- **Regression**: confirmed via `git diff --name-only` that no
  production or Phase 13-16 experimental source file was touched this
  round — only new `bench/*.cpp` diagnostic files plus documentation
  and `CMakeLists.txt` (3 new registrations for discoverability only;
  none of the new files call into any `gpufhe::` code). Full Rust
  suite re-confirmed unaffected: 38 passed, 0 failed, 1 ignored, both
  build profiles. Phase 15/16's own correctness tests re-run fresh
  from a clean object-cache rebuild: every case still EXACT MATCH / 0
  mismatches — the pageable, pinned, and prewarmed/fallback paths all
  remain bit-exact and the existing concurrency/lifecycle guarantees
  are intact.
- **Post-Phase-17 remaining opportunity**: if pinned prewarming is
  revisited, the clearest next experiment (identified but not
  performed this round due to time constraints) is a controlled sweep
  allocating many DIFFERENT sizes in a fixed, non-repeating order with
  memory usage tracked at every step, to more precisely characterize
  hypothesis 3 above (size + cumulative-live-memory interaction).
  Separately, disabling Transparent Huge Pages
  (`/sys/kernel/mm/transparent_hugepage/enabled`, currently `madvise`)
  as a controlled, reversible, documented-baseline-first experiment
  would help test whether THP compaction is a necessary ingredient of
  hypothesis 1, or whether plain `mlock()` page-table work under
  memory pressure is sufficient on its own. Otherwise, `encrypt_host`'s
  arithmetic, transfer, and allocation stages are all now optimized or
  thoroughly investigated (Phases 13-17); the GPU-resident
  tensor+keyswitch+combine+rescale chain remains ~1% of total pipeline
  latency, unchanged since Phase 11.

## GOD PROMPT "CKKS PHASE 18": controlled THP experiment — the system-wide toggle is blocked by permissions (correctly NOT worked around, per explicit user choice); a real but narrow ~5x THP-hint effect found at exactly 2MiB; Phase 17's "not CUDA-specific, HIGH confidence" conclusion REVISED to MEDIUM after same-session interleaved re-testing contradicted it

Full detail: `research/TRANSFER_PROFILING.md`'s Phase 18 section and
`EXPERIMENT_LEDGER.md` entry #51. Starting commit `ff81b97` (Phase 17
end state), confirmed via `git log -1`, `git status`, and
`origin/main` before any change. **No production or experimental
code was modified this round** — this was a pure diagnostic
investigation, per the directive's explicit framing ("a controlled
diagnostic experiment, not a production optimization").

- **Baseline recorded (Step 1)**: THP `enabled`=`madvise`,
  `defrag`=`madvise`; kernel `6.6.87.2-microsoft-standard-WSL2+`; page
  size 4096; `ulimit -l`=65536KB (64MB, both soft/hard, unchanged from
  Phase 17, kept deliberately untouched as a separate confound);
  swap 8GB mostly free; Windows host free RAM ~2.3GB at phase start,
  ~1.93GB at phase end (measured via `Get-CimInstance
  Win32_OperatingSystem`, not WSL `free -h`, per the directive's
  explicit requirement) — the n=131072 ≥6GiB gate was never met, so
  that size was never attempted.
- **The system-wide THP toggle is blocked by permissions**: a
  same-value no-op write (`echo madvise >
  .../transparent_hugepage/enabled`) returned `Permission denied`;
  confirmed no passwordless sudo is available (`sudo -n true`/`sudo
  -n -l` both require a password). This is exactly the blocker the
  directive's Step 3/safety constraint #5 anticipated. **No
  workaround was attempted.** `AskUserQuestion` was used to let the
  user decide; **the user explicitly chose "Skip the THP toggle,
  document the blocker (recommended)."** The system-wide setting was
  never changed, confirmed identical to baseline both immediately
  after the denied attempt and again at phase end.
- **Substituted the one THP variable controllable without root**:
  per-allocation `madvise(MADV_HUGEPAGE/MADV_NOHUGEPAGE)` hints (new
  diagnostics `bench/bench_pinned_alloc_diagnostic3.cpp`,
  `bench/bench_thp_hint_by_size.cpp` — pure Linux `mmap`+`mlock`,
  zero CUDA).
- **`bench_thp_hint_by_size.cpp` — the decisive, demonstrated
  result**: interleaved HUGEPAGE vs. NOHUGEPAGE hints across all four
  packed sizes, 8 repeats each, run twice independently. 256KB/512KB/
  1MB showed NO meaningful difference (ratios 0.90-1.01, i.e. noise).
  **Exactly 2MiB (2097152 bytes, the exact x86-64 huge-page size)
  showed a consistent, reproducible ~5x slowdown with the hint
  disabled** (HUGEPAGE median≈230-247us vs. NOHUGEPAGE
  median≈1217-1270us, ratio≈4.93-5.51, both runs). This is the first
  time this project has DEMONSTRATED (not merely correlated) a
  specific, mechanistic THP effect — but its magnitude (≈1ms worst
  case, one size only) is far too small to explain the multi-
  millisecond-to-hundreds-of-milliseconds variance Phase 16/17
  observed in `cudaMallocHost`.
- **`bench_pinned_alloc_diagnostic3.cpp`**: full per-sample data
  (timestamp, size, order, live-before count, `/proc/meminfo`
  snapshot, separately-timed mlock/touch/release) under the unchanged
  THP=madvise setting. Clean, monotonic-in-size `mlock()` latency —
  256KB≈150-310us, 512KB≈280-480us, 1MB≈550-840us, 2MB≈1100-1450us —
  with NO multi-millisecond outliers, across 5 runs and all live-count
  buckets (0, 1-3, 4-7, 8+).
- **Self-correction (the most important finding this round)**: Phase
  17 concluded HIGH confidence that pure Linux `mlock()` reproduces
  the SAME variance as `cudaMallocHost`, proving the mechanism is
  NOT CUDA-specific. This phase re-ran that comparison more
  rigorously — SAME session, INTERLEAVED, back-to-back for the
  identical size — and it does NOT hold up: pure Linux `mlock()` was
  clean and fast (174-1613us) while `cudaMallocHost`, interleaved in
  the same short window, remained consistently 2984-5079us — an
  order of magnitude slower. **Phase 17's "pure Linux reproduces the
  same variance" finding was most likely a coincidental timing
  artifact from that one investigation window, not a robust
  equivalence.** Reported here as an honest, explicit revision,
  following this project's own precedent (Phase 12's correctness-
  breakthrough reversal).
- **Updated ranked root-cause assessment**: (1) CUDA/WSL2-driver-
  passthrough overhead — PROMOTED to leading hypothesis, MEDIUM-HIGH
  confidence (this phase's interleaved same-session comparison).
  (2) Linux kernel `mlock()`-path variance — DOWNGRADED from Phase
  17's HIGH to MEDIUM (demonstrated once, not reproduced under
  matched conditions this phase; itself intermittent). (3) THP
  per-call hint effect — CONFIRMED, MEDIUM confidence, narrow (~5x,
  exactly-2MiB-only). (4) System-wide THP enabled/defrag setting —
  UNTESTED, blocked by permissions. (5) Host/WSL2 memory pressure —
  MEDIUM, unchanged from Phase 17. (6) WSL2-specific Hyper-V
  ballooning distinct from (1)/(2)/(5) — still NOT CONFIRMED.
  (7) Benchmark/measurement artifact — still RULED OUT (same two
  previously-validated binaries, interleaved, same gap).
- **Decision: category 3 continues — no production code or config
  change justified.** Phase 16's opt-in prewarming API design stands,
  now on a partially revised rationale (CUDA/driver overhead promoted
  over pure-kernel `mlock()` as the likely dominant mechanism, plus a
  real but narrow THP-hint contributor). No `.wslconfig`/kernel/
  Windows setting was changed; the system-wide THP setting was never
  successfully written to at all.
- **Regression**: confirmed via `git status`/`git diff` that no
  production or experimental source file was touched this round —
  only the 2 new standalone `bench/*.cpp` diagnostics plus
  documentation and `CMakeLists.txt` (2 new registrations for
  discoverability only). Full Rust suite re-confirmed unaffected: 4
  passed + 38 passed, 0 failed, 1 ignored, both debug and release.
- **Post-Phase-18 remaining opportunity**: if root-causing this
  further is revisited, the clearest next experiment is obtaining
  passwordless sudo (or running the toggle manually, as the user was
  offered this round) to finally exercise the originally-planned
  A/B/C system-wide THP comparison; separately, a GPU-passthrough-
  layer trace or a non-WSL2 Linux host with a directly-attached GPU
  would let hypothesis 1 (CUDA/driver overhead) be tested directly
  rather than inferred from an interleaved comparison.

## GOD PROMPT "CKKS PHASE 19": mechanistically ISOLATED CUDA pinned-allocation overhead -- proven, via NVIDIA's own tracer (not comparative timing), to be dominated by `ioctl()` calls into the driver on EVERY call, not a one-time cost; `cudaHostRegister` found 1.5x-4x cheaper than `cudaMallocHost`/`cudaHostAlloc` for the full lifecycle, reproduced across 4 independent process runs

Full detail: `research/TRANSFER_PROFILING.md`'s Phase 19 section and
`EXPERIMENT_LEDGER.md` entry #52. Starting commit `91ae35f` (Phase 18
end state), confirmed via `git log -1`/`git status`/`origin/main`
before any change. **No production or experimental code was
modified this round** — a pure diagnostic/investigation phase, per
the governing directive's own framing.

- **Methodology gap found in Phases 16-18's own measurements**: none
  of them separated CUDA context initialization from the allocation
  call being timed; none tested `cudaHostAlloc` with explicit flags
  or `cudaHostRegister`; and critically, Phase 17/18's plain-`mlock()`-
  vs-`cudaMallocHost` comparison compared two OPERATIONS WITH
  DIFFERENT SEMANTICS — `mlock()` only prevents swapping, while
  `cudaMallocHost` ALSO registers the page for GPU DMA with the
  NVIDIA driver. Confirmed via `cudaPointerGetAttributes`: plain
  `malloc`+`mlock()` memory reports `cudaMemoryTypeUnregistered`
  (CUDA genuinely does not see it as pinned), while every
  `cudaMallocHost`/`cudaHostAlloc`/`cudaHostRegister` buffer reports
  `cudaMemoryTypeHost`. This retroactively explains why Phase 17's
  "pure Linux reproduces the same variance" claim was unreliable
  independent of any timing confound.
- **New diagnostic (`bench/bench_alloc_path_comparison.cpp`)**:
  isolates CUDA context init as its OWN Stage 0 (`cudaFree(0)`,
  371-845ms, measured once, never leaking into any other path's
  numbers), then compares 5 allocation paths (A: pageable malloc;
  B: malloc+mlock pure Linux; C: `cudaMallocHost`; D:
  `cudaHostAlloc(cudaHostAllocDefault)`; D2:
  `cudaHostAlloc(cudaHostAllocWriteCombined)`; E: malloc+touch+
  `cudaHostRegister`) at 4 sizes, via a fixed-order first-use pass, a
  FIXED-SEED randomized interleaved steady-state round (10 rounds x
  6 paths x 4 sizes = 240 samples), 20-repeat tight-loop cycles, and
  a retained-vs-short-lived comparison — 425 individual samples per
  run, all written to raw CSV, none discarded.
- **Finding 1 — C and D are the SAME operation, empirically
  confirmed**: `cudaMallocHost` and `cudaHostAlloc(cudaHostAllocDefault)`
  are statistically indistinguishable at every size across 4
  independent fresh-process runs, confirming CUDA's documentation
  claim empirically rather than assuming it. `cudaHostAllocWriteCombined`
  (D2) is ALSO indistinguishable — the flag does not change pinning
  cost.
- **Finding 2 — the cost is syscall-level and mechanistic, confirmed
  by NVIDIA's own tracer (`nsys`, installed and usable WITHOUT root
  on this system — not previously known/used in Phases 16-18)**:
  the first `cudaMalloc` (context init) spent 86.1% of its 797ms in
  `ioctl()` calls on the SAME calling thread (many calls, the
  largest 122.6ms, consistent with a retry/poll pattern during
  paravirtualized GPU-channel setup). Every subsequent
  `cudaHostAlloc`/`cudaMallocHost` call is consistently 92-97%
  `ioctl`-bound via exactly 4 `ioctl` calls per call. `cudaHostRegister`
  shows the SAME structural profile (4 `ioctl` calls, 87-91% bound)
  but each call is ~3-4x faster, yielding ~3.1x cheaper total
  registration cost (aggregated via `nsys stats --report
  cuda_api_sum`: `cudaHostAlloc`/`cudaMallocHost` avg 3.2ms/call vs.
  `cudaHostRegister` avg 1.03ms/call). Release is asymmetric:
  `cudaFreeHost` is only 26-39% `ioctl`-bound (most cost is other
  CPU-side work), while `cudaHostUnregister` is both cheaper AND more
  `ioctl`-bound. **This is the most mechanistically decisive finding
  across Phases 16-19 — read directly from syscall-level trace data
  on the same thread/window, not inferred from comparative wall-clock
  timing.**
- **Finding 3 — `cudaHostRegister` on pre-touched memory is 1.5x-4x
  cheaper than `cudaMallocHost`/`cudaHostAlloc` for the FULL
  lifecycle**, reproduced across 4 independent fresh-process runs at
  2MiB (`cudaMallocHost`≈4822-5374us, `cudaHostAlloc`≈5094-5205us,
  `cudaHostRegister`≈3323-3475us) and with a LARGER gap at smaller
  sizes (256KB: `cudaHostAlloc`≈4700-5500us vs.
  `cudaHostRegister`≈1250-2400us, 2-4x cheaper).
- **New, unexplained-until-now finding**: plain pageable
  `malloc`+`memset` (path A, zero pinning, zero CUDA) ALSO jumps
  sharply at exactly 2MiB (20-66us at smaller sizes → ~1601-1650us at
  2MiB) — demonstrating Phase 18's THP-hint effect is not specific to
  `mlock()`/pinning at all, but a property of ordinary anonymous-
  mapping first-touch at this kernel's huge-page size boundary. This
  is a SEPARATE, ADDITIVE mechanism from the CUDA/`ioctl` cost (which
  is present at EVERY size, not just 2MiB) — the two should not be
  conflated.
- **Tracing capability assessed honestly (directive Section 5)**:
  `nsys` works without root; `strace`/`ltrace`/`nvprof`/`ncu` are NOT
  installed (checked, not installed to keep this phase read-only).
  `perf_event_paranoid=2` would likely block CPU-sampling traces, not
  tested since CUPTI-based API/OS-runtime tracing answered the
  question without needing it. Honesty note: the trace proves the
  cost is `ioctl`-bound; it does NOT by itself prove WSL2
  paravirtualization specifically (vs. ordinary native-Linux NVIDIA
  driver `ioctl` handling) is why — no tool available here can trace
  inside the driver or across the VM boundary. WSL2-passthrough
  remains the best-supported hypothesis, not a directly observed
  fact.
- **`cudaHostRegister` evaluated as a production alternative,
  honestly (directive Section 6/8)**: credible full-lifecycle win,
  but NOT yet justified to ship — not tested inside the real pooled/
  prewarmed production paths, not re-verified for correctness/
  concurrency, not re-measured on the full pipeline, repeated-
  registration behavior not tested, large-retained-pool behavior near
  the 64MB `RLIMIT_MEMLOCK` cap not stress-tested. Recommended next
  phase: prototype the substitution inside `src/encrypt_host_shoup_
  pooled_barrett_batched_pinned.cpp`/the Phase 16 prewarming API,
  re-verify from scratch, re-run the full-pipeline benchmark, THEN
  decide.
- **CMake/CTest registration audit (directive Section 7)**: found
  `gpufhe_test`'s `add_test(...)` call means every Phase 16-18
  pinned-memory/THP diagnostic (6 targets) is auto-picked-up by a
  plain `ctest` run, indistinguishable from a real correctness test —
  not yet a real incident (this project's regression gate has been
  `cargo test`, not `ctest`, throughout), but a latent landmine.
  **Fix**: new `gpufhe_benchmark(name src)` CMake function (same
  build target, no `add_test`) — still buildable/runnable directly,
  never auto-run by `ctest`. Applied to all 6 Phase 16-18 targets plus
  this phase's new `bench_alloc_path_comparison`. Deliberately NOT
  applied to other pre-existing `bench_*` targets elsewhere in the
  file (same latent characteristic likely applies, but auditing the
  entire project's benchmark history is out of this investigation's
  scope — flagged, not fixed, consistent with this project's Phase 10
  precedent).
- **Decision: no production or experimental code change this phase,
  by design.** The dominant mechanism is now mechanistically
  demonstrated (syscall-level, every call, not one-time). A credible
  alternative (`cudaHostRegister`) was identified but not implemented.
  THP/driver/`ulimit -l`/`.wslconfig` settings were never read-write
  touched — only already-established values re-read. Windows host
  free RAM 1.41-1.53GB throughout (measured via `Get-CimInstance`),
  n=131072's ≥6GiB gate never met, never attempted; no allocation
  exceeded 2MiB, no retained total exceeded ~3.75MB.
- **Regression**: confirmed via `git status`/`git diff` that only the
  new `bench/bench_alloc_path_comparison.cpp` file and `CMakeLists.txt`
  (the new function plus 7 registrations switched to it) changed —
  no production or experimental source file touched. Full Rust suite
  re-confirmed unaffected: 4 passed + 38 passed, 0 failed, 1 ignored
  (pre-existing `control_a_relin_combine_matches_raw_tensor_num_part_1_test`,
  unrelated, status unchanged), both debug and release.
- **Next recommended experiment**: implementation-focused, not further
  diagnosis — prototype `cudaHostRegister` inside the real production
  pinned/prewarmed code paths, re-verify correctness and concurrency
  from scratch, re-run the full-pipeline benchmark, and only then
  decide whether to ship it.

## GOD PROMPT "CKKS PHASE 20": CUDA-registered host memory PROTOTYPE built, correctness-verified (0 mismatches across 400+ checks), measured against the real encrypt/EvalMult pipeline — RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE, NOT promoted to production

Full detail: `research/TRANSFER_PROFILING.md`'s Phase 20 section and
`EXPERIMENT_LEDGER.md` entry #53. Starting commit `8d6dbe4` (Phase 19
end state), confirmed via `git log -1`/`git status`/`origin/main`
before any change. The first phase in this investigation to move from
diagnosis to a real implementation prototype.

- **New ownership model, designed before implementation**:
  `include/registered_host_buffer.h`'s `RegisteredHostBuffer` — a
  move-only RAII class (following this codebase's own established
  `include/ks_raii.h` pattern) with an explicit 4-state machine
  (`Empty`/`Allocated`/`Registered`/`Failed`) owning a page-aligned
  host allocation registered via `cudaHostRegister`. Two measured
  facts shaped the design: (1) a direct probe found `cudaHostRegister`
  is NOT gated by `RLIMIT_MEMLOCK` on this system (up to 500MB
  succeeded with zero error, despite the 64MB limit) — any budget cap
  this phase enforces is therefore self-imposed, not OS-enforced;
  (2) an invalid `flags` value (not an oversized request) is the real,
  safe, bounded way to induce a genuine `cudaHostRegister` failure on
  this system for testing.
- **Standalone lifecycle/concurrency audit**
  (`tests/test_registered_host_buffer.cpp`, 10 scenarios): 0 check
  failures on the normal build, covering successful lifecycle, 50
  repeated alloc/free cycles, 200 reuse cycles, 2 REAL induced failures
  (double-registration, invalid flags), pool destruction/eviction with
  live buffers, 200 concurrent independent-buffer cycles across 8
  threads, a REAL async-transfer-completion test (`release()`'s own
  internal stream sync proven to actually wait, not assumed), and a
  REAL induced async-failure cleanup test. **Genuinely new finding**:
  under AddressSanitizer, `cudaHostRegister` reliably fails with
  "out of memory" for every call on this system (an ASan-vs-CUDA-driver
  memory-pressure incompatibility, not a code defect) — after fixing a
  real test bug (assumed success, crashed on null `data()`), the ASan
  build completed with ZERO crashes/leaks across all 27 induced
  failures, validating the failure-handling paths specifically.
- **Production-shaped prototype**
  (`src/encrypt_host_shoup_pooled_barrett_batched_registered.cpp`):
  mirrors Phase 15's pinned variant EXACTLY except the host staging
  buffer's allocation mechanism. Registration happens ONCE per
  `(n,sizeQ)` key (never per-call). A self-imposed, strict 16MiB
  cumulative-bytes budget triggers a safe fallback to the unmodified
  production pageable path if exceeded or if registration fails —
  mirroring Phase 16's own prewarm-file fallback shape. Symmetric
  `encrypt_prewarm_registered_buffer`/`encrypt_registered_buffer_is_
  ready` API added alongside, sharing the same pool.
- **Correctness**: bit-exact vs. `encrypt_host`, the Phase 15 batched,
  AND Phase 15 pinned implementations, across every size/boundary/
  repeat/interleave this project's established test pattern covers —
  **0 mismatches**. Real encrypted pipeline (encode→encrypt→EvalMult→
  decrypt→decode, via Phase 11's own harness extended with a NEW
  `encryptVariant=5`): max abs error identical (`1.12912e-10`) across
  production/pinned/registered — **[PASS]** for all three.
- **Concurrency audit** (6 scenarios, 436 checks, **0 mismatches**):
  the 4 scenarios already established for the pinned variant, PLUS
  (5) explicit prewarm racing with real encryption at a fresh key
  (correctness held regardless of which side won the race), and
  (6) fallback-path correctness under GENUINE budget pressure — 16
  distinct large keys (~27MiB, deliberately over the 16MiB budget):
  12 registered, 4 fell back to pageable, 0 mismatches either way.
- **Full lifecycle measurement** (`bench/bench_encrypt_registered_
  lifecycle.cpp`, new): 5 distinct configurations (production, pinned,
  registered-fresh-per-call, registered-pool-reused, registered-
  prewarmed), first-use recorded separately from a randomized
  fixed-seed interleaved steady-state round, every sample to CSV.
  **D's (registered, pool-reused) cold-start (~9.0-9.4ms) is ~2.1-2.4x
  CHEAPER than B's (pinned) cold-start (~18.9-21.6ms)** — directly
  resolving the exact cold-start weakness that blocked the pinned
  variant's own promotion in Phase 15/16. **D consistently beat A
  (production) in steady-state across all 3 independent fresh-process
  runs, by ~7-14%**, and was competitive with B (pinned). **Unresolved,
  honestly flagged**: E (explicitly prewarmed) was consistently
  slightly SLOWER than D (lazy) in every run — a genuine open question,
  not investigated further this round.
- **Nsight Systems confirmation** (in the REAL production-shaped code,
  not just Phase 19's isolated diagnostic): `cudaHostRegister` averaged
  1.07ms/call vs. `cudaHostAlloc`'s 5.13ms/call (~4.8x cheaper),
  closely matching Phase 19's own isolated finding — the driver-call
  reduction genuinely carries over into real use.
- **CMake/CTest integration**: 3 new correctness/lifecycle tests via
  `gpufhe_test` (CTest SHOULD run these); the new timing benchmark via
  `gpufhe_benchmark` (never auto-run), per Phase 19's own established
  distinction. A live `ctest -N` reconfigure was attempted but aborted
  after Windows host free RAM dropped to ~1.2GB mid-attempt — this
  phase's CMake correctness rests on direct inspection of the
  (unmodified) macros Phase 19 already verified, not a live run.
- **Decision: Category B — RETAIN AS AN OPT-IN EXPERIMENTAL FEATURE,
  NOT promoted to production.** Correct, concurrency-safe, resolves
  the exact cold-start weakness that blocked Phase 15/16's pinned
  variant, and shows a real (if modest, ~7-14%) steady-state win — a
  STRONGER case than the pinned variant ever achieved, but not yet
  large enough, nor validated at the Rust/FFI layer, to justify
  promotion this round. `encrypt_host_shoup_pooled_barrett_batched`
  (production) and `_batched_pinned` (Phase 15, unshipped) remain
  entirely unchanged; the new registered variant joins Phase 16's
  prewarming API as a second opt-in experimental C++ feature, not
  wired into the Rust API or FFI bridge this round.
- **Regression**: full Rust suite re-confirmed unaffected: 4 passed +
  38 passed, 0 failed, 1 ignored (pre-existing, unrelated), both debug
  and release. `git status`/`git diff` confirmed changes limited to 2
  new production/header files, 5 new test/bench files, 1 minimal
  surgical extension to an existing bench file, `CMakeLists.txt`, and
  documentation — no existing production file's content was modified.
- **Next recommended experiment**: investigate the D-vs-E anomaly
  directly; run 5+ independent fresh-process repetitions (matching
  this project's own bar for its strongest claims) before trusting the
  magnitude more firmly; if the win holds up, prototype wiring the
  registered path through `cpp_bridge/encdec_bridge.cpp` and the Rust
  FFI layer, with its own dedicated Rust-side tests, before any
  production promotion decision.

## GOD PROMPT "CKKS PHASE 21": Graphify repository-intelligence integration — installed, validated against 12 real questions with honest gaps documented, zero CKKS code touched

Full detail: `research/GRAPHIFY_GUIDE.md` and `EXPERIMENT_LEDGER.md`
entry #54. **Not a CKKS research round** — a tooling/workflow
integration. Starting commit `f20fa88` (Phase 20 end state), confirmed
via `git log -1`/`git status`/`origin/main` before any change.

- **Installed**: Graphify (PyPI `graphifyy`, installed version
  `0.9.102`) via `pipx install graphifyy` (`uv` was not present on
  this machine; `pipx`, the documented alternative, was used). A
  project-scoped Claude Code integration
  (`graphify install --project --platform claude`) was installed
  deliberately WITHOUT `--strict` — verified non-blocking by directly
  testing `graphify hook-guard search`/`read` (both exit 0, no
  output).
- **Merged, not replaced**: this repo already had a `CLAUDE.md` with
  a Scrum-Orchestrator integration and a `.claude/` tree with 8 agent
  files. A backup was taken before installing; a diff afterward
  confirmed the install appended exactly one `## graphify` section at
  the end of the existing `CLAUDE.md` (the Scrum-Orchestrator block
  untouched, byte-for-byte) and added only new files
  (`.claude/skills/graphify/`, `.claude/CLAUDE.md`,
  `.claude/settings.json`) alongside, never inside, the existing
  agent/skill directories. One additional short "Session start"
  section was added to `CLAUDE.md` by hand, pointing to
  `research/NEXT_SESSION.md`/`MASTER_STATE.md`/`MIGRATION_MATRIX.md`
  and `research/GRAPHIFY_GUIDE.md` — CLAUDE.md previously had no
  pointer to the project-state docs at all.
- **Graph generated**: `GRAPHIFY_BACKEND=none` was set explicitly for
  every extraction/update/cluster run this phase (not merely relying
  on the absence of an API key) — zero LLM calls, zero external
  network use, confirmed via `GRAPH_REPORT.md`'s own "Token cost: 0
  input · 0 output" line. A `.graphifyignore` excludes `build2/`
  (stale CMake build dir), `.scrum-orchestrator/`, and
  `tools/__pycache__/`. Final graph (after this phase's own doc/config
  edits were themselves re-indexed): 2539 nodes, 5563 edges, 162
  communities, covering 208+ C++/CUDA/Rust source files. Communities
  are unlabeled placeholders (`--no-label` used, same zero-LLM
  rationale) — the underlying structure is still correct and useful
  (see below), only the human-readable community NAMES are absent.
- **Auto-refresh installed**: `graphify hook install` registered 4
  git hooks (`post-commit`/`post-checkout`/`post-merge`/
  `post-rewrite`) plus a union-merge driver for `graphify-out/
  graph.json`. Read and verified safe before trusting: the
  `post-commit` hook runs the rebuild as a **detached background
  process** (never blocks `git commit`), skips during an in-progress
  rebase/merge/cherry-pick, has a 600s timeout with proper subprocess
  cleanup, and logs to `~/.cache/graphify-rebuild.log` rather than the
  commit's own output. These hooks are LOCAL to this checkout only
  (`.git/hooks/` is never pushed); another clone must run `graphify
  hook install` itself.
- **Generated-file policy, decided deliberately**: `graphify-out/
  graph.json`, `GRAPH_REPORT.md`, and `.graphify_analysis.json` ARE
  committed (small, no absolute paths/machine info — verified via
  `grep` for the local home-directory path, zero hits in `graph.json`)
  so a fresh clone can browse the graph without re-running extraction.
  `graph.html` (2MB visualization), `cache/` (4.9MB AST cache),
  `manifest.json` (checkout-specific mtimes), and `.graphify_root`
  (a machine-specific absolute path) are gitignored — all three are
  regenerated by the next `graphify extract`/`update` run regardless.
- **Four-layer memory architecture documented** (`research/
  GRAPHIFY_GUIDE.md`): Layer A (authoritative state —
  `NEXT_SESSION.md`/`MASTER_STATE.md`/`MIGRATION_MATRIX.md`, unchanged
  in authority), Layer B (experimental evidence —
  `EXPERIMENT_LEDGER.md`/`TRANSFER_PROFILING.md`, unchanged), Layer C
  (navigable code graph — new, Graphify), Layer D (concise
  session-start guidance tying A-C together — new, `CLAUDE.md`'s
  `## graphify` section + the new "Session start" section + this
  guide). Explicit conflict-resolution order documented: current
  source/Git state > a re-run test > the original experiment record >
  update Layer A > refresh Layer C.
- **12-question validation suite, tested against real evidence, not
  assumed**: the graph correctly and usefully answers structural
  questions (which function implements X, who calls it, which tests
  reference it, which CUDA kernels a pipeline reaches — `EvalMultFixtureCpp`
  and `DeviceKSContext` surfaced as top god nodes, matching this
  project's own repeatedly-emphasized architectural hubs; the
  clustering ALREADY separates the production/pinned/prewarmed/
  registered encrypt variants into 4 distinct communities purely from
  code structure, with no hint about which labeled "production").
  **Confirmed real gaps, honestly documented, not hidden**: `CMakeLists.txt`
  produces ZERO graph nodes (CMake isn't a parsed language on this
  install) — CTest registration questions have no graph shortcut, grep
  the file directly. Research `*.md` docs appear only as lightweight
  local reference-pointer nodes (confirmed zero-LLM-cost, via code
  comments that cite them by path), not semantically searchable — Phase
  19/20's actual findings are correctly NOT graph-answerable, by
  design. Free-text `graphify query` mis-seeded on an enum value named
  `A_Production` (from this project's OWN Phase 20 benchmark file)
  instead of the real production function when asked "what is the
  current production encryption implementation" — `graphify explain
  "<exact name>"` proved far more reliable and was adopted as the
  recommended pattern. Cross-language Rust-extern/C++-definition FFI
  pairs (e.g. `gpufhe_encrypt`) appear as two separate same-named nodes
  with no edge between them — a human/Claude must notice the matching
  name.
- **Measured, not estimated**: `graphify benchmark` (the tool's own
  built-in measurement, run on this actual repo) reported 91.2x fewer
  tokens per query vs. naive full-corpus reading, averaged over 3
  representative questions (91.8x/90.2x/91.6x individually). A single
  `graphify query` took ~0.5s wall-clock; an incremental `graphify
  update .` after a 26-file delta completed in well under a minute,
  re-extracting only the changed files via its own AST cache.
- **Regression**: full Rust suite re-confirmed unaffected (no CKKS
  source file touched this phase): 4 passed + 38 passed, 0 failed, 1
  ignored (pre-existing, unrelated), both debug and release.
  `git status`/`git diff` confirmed changes limited to new Graphify
  config/output files, `.gitignore`, `CLAUDE.md` (additive section
  only), `.claude/` (additive files only), and documentation — zero
  `src/*.cpp`, `kernels/*.cu`, `rust/**`, or `CMakeLists.txt` changes.
- **Decision**: the integration is adopted, non-strict, with the
  graph committed and auto-refreshing locally via git hooks. No CKKS
  cryptographic behavior, security parameter, or research conclusion
  was touched, read incorrectly, or overridden by the graph at any
  point this phase.
- **Next recommended step, if revisited**: if CTest-registration
  questions become frequent, investigate whether a `.graphifyrc`
  `language.inc=` style declaration (or a future Graphify CMake
  grammar) can close that specific gap; if deep semantic search over
  `research/*.md` findings becomes valuable enough to justify it, a
  local Ollama backend (verified NOT currently running on this
  machine) would close the documentation-semantics gap without any
  cloud dependency — not attempted this phase, judged to be more
  infrastructure than the task required given Layer B already serves
  that need.
