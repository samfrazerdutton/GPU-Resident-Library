# Next session — exact continuation

## Current commit

`865b59a` on `main`, pushed to origin. (Research round after this commit —
`research/*.md` additions plus a kept RSS-instrumentation line in
`tests/test_boot.cpp` — is in a NEW commit on top; check `git log -3`
when you resume to confirm whether it landed before this session ended.)

## Read first

`research/MASTER_STATE.md` (current state), then
`research/EXPERIMENT_LEDGER.md` (what's been tried, what worked, what
didn't). Both are more current than `docs/performance-baseline.md`, which
predates this `research/` directory and is kept as historical detail only.

## FIRST thing to do, before any new experiment: diagnose/reset the GPU environment

This session's own measurements drifted substantially over many hours:
the proven, UNCHANGED `rotate_ct_resident` code measured 62-66s early in
the session and 71-76s late in the session, on the SAME commit, with no
code change in between. This made the last experiment of the session
(stage-scoped resident rotation, see ledger) impossible to validate.

Before trusting ANY new benchmark number:
```bash
wsl -d Ubuntu -- bash -lc 'export PATH=/usr/bin:/bin:/usr/sbin:/sbin; /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv'
```
Run this, let the GPU idle/cool for a few minutes if temperature is high,
and run `test_boot` TWICE in a row to confirm two consecutive runs agree
within a few seconds before trusting a third (experimental) run against
them. If starting a brand new session/reboot, the GPU should be at a cold
baseline — measure that baseline FIRST, before resuming any experiment.

## Priority queue (highest value first)

1. **Re-validate experiment #1** (stage-scoped resident rotation — hoist
   stream/DeviceKSWork/base-ciphertext residency out of the C2S/S2C
   per-diagonal loop). The code is NOT currently in the tree (reverted for
   inconclusive measurement, not disproven). To reapply: see
   `EXPERIMENT_LEDGER.md`'s entry for the exact design (StageRotCtx struct
   + stageRotBegin/stageRotDiag/stageRotEnd lambdas placed right before the
   C2S loop in `tests/test_boot.cpp`, wired into both the C2S loop and the
   S2C loop's `q2` sub-loop). Use `automorphism_eval_device` +
   `keyswitch_resident` directly (not the `rotate_ct_device` convenience
   wrapper — that has an unbatched per-tower `LaunchAddInto` loop for the
   c0+=ba0 combine; keep that combine on the CPU instead, consistent with
   this session's own profiling showing the CPU combine costs only ~0.6s
   total). Validate with AT LEAST 3 repeated runs of baseline and 3 of the
   experiment, interleaved (baseline, experiment, baseline, experiment...),
   not sequential blocks, to separate real signal from environmental drift.

2. **Eval-domain automorphism as a precomputed index permutation**
   (ranked experiment #2, not yet attempted). Eliminates the
   INTT→permute→NTT round-trip in `automorphism_eval_device` entirely,
   replacing it with one permutation kernel operating directly on
   eval-domain data (for a negacyclic NTT, ψ_k with odd k is a permutation
   of the evaluation vector composed with bit-reversal on both sides — see
   `LITERATURE_REVIEW.md`'s OpenFHE/FIDESlib citations for the precise
   construction: `PrecomputeAutoMap`/`AutomorphismTransform`). Flagged by
   the research agent as "the single largest win available." Needs a new
   kernel + a correctness test against the existing coefficient-domain
   path before touching the hot loop.

3. **Correct-order rotation hoisting** (ranked experiment #3). Per the
   research agent's reading of OpenFHE's actual source: decompose the
   UNPERMUTED ciphertext once; keyswitch; apply the automorphism to the
   QP-basis OUTPUT (ba0/ba1, sizeQlP towers) AFTER the key-switch inner
   product, not to the decomposed digits; fold c0 in as `ba0 += P*c0`
   before the permutation; ModDown last. Requires regenerating rotation
   keys as keyswitch-to-ψ_{k⁻¹}(s) (inverse automorphism convention), not
   today's ψ_k(s). This is a bigger, higher-risk change than #1/#2 — do it
   after they land, since it amortizes the (by-then smaller) per-rotation
   cost they leave behind, making its own benefit cleaner to measure. Build
   a STANDALONE correctness test (N independent rotations vs.
   decompose-once + N finishes, bit-exact comparison) before touching
   `test_boot.cpp`, exactly as the FIRST (failed) hoisting attempt should
   have scoped more carefully from the start.

4. **KROT/KSC bounded host-side key cache.** Required before ANY attempt
   to increase n. Current unbounded caching already uses ~10.78GB peak RSS
   at n=8192 — see `MEMORY_FEASIBILITY.md`. Design a bounded LRU or
   per-stage-scoped cache (keys needed only within the current C2S/S2C
   stage do not need to survive past it) before n is touched at all.

5. **Security parameter switch: sparse (h=64) → uniform ternary secret,
   n=8192 → n=65536.** Verified via this repo's own `lwe_security.py` to
   give 128/120-bit classical/quantum security at the existing 1775-bit
   modulus budget — see `SECURITY_ASSESSMENT.md`. This is a LARGER,
   separate effort: requires (a) the KROT/KSC fix above first (memory will
   not survive n=65536 otherwise), (b) changing `set_secret_hamming_weight`
   usage to a uniform-ternary secret generation path, (c) regenerating all
   moduli/roots at n=65536, (d) full correctness re-verification, (e) a
   fresh performance baseline (expect substantially slower than the
   current ~115-130s; not yet measured, do not guess the number).

## What NOT to do

- Do not re-attempt hoisting the OLD way (permuting the decomposed
  digits) — rigorously falsified this session via an isolated debug test,
  root-caused definitively by the literature research (wrong operation
  order + wrong key convention, not an approximation-math dead end — see
  `LITERATURE_REVIEW.md`).
- Do not adopt n=131072 — confirmed wasteful (280 bits vs. the 128 needed).
- Do not trust any single-run benchmark from a long-running session
  without a fresh, immediately-preceding control run.
- Do not use CKKS for the messaging use case — see
  `MESSAGING_ARCHITECTURE.md` for the settled recommendation
  (ML-KEM-768 + Double Ratchet + AEAD).

## Standing environment facts (re-verify, don't assume stale)

RTX 2060 Max-Q, 6144MB VRAM, CUDA 13.2, driver 595.97, WSL2 Ubuntu, 10GB
host RAM visible to WSL2, 16 CPU threads. GPU clock throttling under WSL2
is real and was NOT fixed (requires elevated privileges this session
would not escalate to obtain without the user doing it themselves).
