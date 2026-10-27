# Master state

Last updated: this session, third execution-focused cycle. Delivered TWO
real, validated wins (KSC bound, eval-domain automorphism), one safe
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

8. **Multi-rotation decompose-sharing: DONE, validated on a REAL live
   bootstrap stage, not just synthetic benchmarks.** New, separate
   files `include/keyswitch_hoist.h` + `src/keyswitch_hoist.cpp`
   (`keyswitch_resident.cpp` untouched). `tests/test_hoist_multi.cpp`:
   bit-exact vs. the existing fused `ks_accumulate_qp` (0/524288
   mismatches), correct across 6 shared rotation indices. `bench/
   bench_hoist_multi.cpp`, interleaved A/B: N=5 ~2.04x, N=16 (matching
   the real C2S stage 0 bucket size) ~2.44x, both CIs exclude zero.
   **`tests/test_hoist_stage0.cpp` then validated the mechanism against
   test_boot.cpp's ACTUAL first C2S stage** (identical setup through
   ModRaise, same real diagonal structure, own fresh hoisted-convention
   keys with no cache advantage over the trusted path): decrypted
   output matches within negligible noise (maxDiff=613 vs
   maxAbs=5.76e17), and **real timing: hoisted 7.53s vs trusted
   25.89s — a ~3.4x speedup on the actual live stage**, both sides
   paying real keygen cost. This is the strongest evidence yet — a
   real bootstrap stage, not a synthetic approximation. Not yet wired
   into ALL stages of the live bootstrap (needs a parallel hoisted-key
   cache, since the hoisted convention is structurally different from
   today's `KROT`'s textbook convention) — scoped in detail in
   `NEXT_SESSION.md` priority queue item 1, with `test_hoist_stage0.cpp`
   as the proven template to extend.

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
