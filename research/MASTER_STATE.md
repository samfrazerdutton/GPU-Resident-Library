# Master state

Last updated: this session, after a second execution-focused cycle (proper
benchmark harness built, resident-rotation experiment validated as a true
null result, KSC cache bounded with a real measured win, and a significant
self-correction to the security recommendation).

## Current verified state

- Repo: `github.com/samfrazerdutton/GPU-Resident-Library`, branch `main`,
  commit `93f2231` (pushed to origin as of the prior checkpoint; confirm
  `git log -1` and push status on resume).
- Hardware: RTX 2060 Max-Q, 6144MB VRAM, CUDA 13.2, driver 595.97, WSL2
  Ubuntu, 10GB host RAM visible to WSL2, 16 CPU threads.
- Current demo bootstrap (n=8192, log2(QP)=1775, sparse secret h=64):
  ~114-150s wall time depending on GPU thermal state (see below) — **still
  only ~51 classical / ~46 quantum bits of security. Never secure.** Every
  speed number below describes this insecure demo, not a security result.
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

## The real, quantified blocker for n=131072

Not GPU VRAM (181MB of 6144MB used — huge headroom). Not KSC anymore
(fixed, see above). **It is `KROT`'s own footprint**: 77 entries × 62.9MB
at n=8192 = 4.84GB, scaling to **77.5GB at n=131072** — infeasible, and
NOT addressed by the KSC fix (KROT's entries are confirmed genuinely
reused across the whole bootstrap, so naive eviction there would force
real, expensive `evalkeygen_host` regeneration, not a cheap re-copy).
A real fix needs either a zero-copy av/bv sharing mechanism (requires
changing `KeySwitchConstants`'s field types — broad blast radius, not
attempted, needs a cryptographic-correctness review first) or a
different architectural approach (streaming av/bv to GPU without a full
host-resident copy). **n=131072 end-to-end has NOT been run. Do not
report it as supported.**

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
- [ ] Eval-domain automorphism permutation (ranked experiment #2, "largest
  available win" per research) — not yet attempted.
- [ ] Correct-order rotation hoisting (ranked experiment #3, now properly
  understood) — not yet attempted.
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
