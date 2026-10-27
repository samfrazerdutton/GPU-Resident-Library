# Next session — exact continuation

## Current commit

Check `git log -3` on resume. As of this writing: `2080aa0` on `main`,
pushed to origin (confirm push status — push again if `git status` shows
ahead of origin).

## Read first

`research/MASTER_STATE.md` (current state, written AFTER this round's
real work), then `research/EXPERIMENT_LEDGER.md` for full history.
`docs/performance-baseline.md` predates `research/` and is historical only.

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

1. **Correct-order rotation hoisting (ranked experiment #3) — ATTEMPTED
   this round, FAILED TWICE on the key-generation convention specifically,
   precisely documented below. Do not retry the same two wrong attempts.**

   **What's PROVEN correct already** (`tests/test_hoist_single.cpp`,
   built but not in the default test suite — `GPUFHE_BUILD_EXPERIMENTAL_TESTS`
   gated in `CMakeLists.txt`): with k=1 (trivial identity rotation),
   the full pipeline — `ks_accumulate_qp` on the unpermuted c1, folding
   `c0*(P mod q)` into the Q-tower range of the QP-basis accumulator via
   the new `LaunchScaledAddInto` kernel (`kernels/stage_acc.cu`), applying
   `automorphism_eval_fast_device` to the QP-basis accumulator (sizeQlP
   towers), then `ks_moddown` — matches the existing, proven
   `rotate_ct_resident` **bit-exact, 0/491520 mismatches.** This proves
   the plumbing is right.

   **What's WRONG, two attempts, both ruled out:**
   - Attempt 1: permuted the key's payload secret (`s2`) by `k^-1`, kept
     the encrypting public key as `KPqp.pkA/pkB` (unpermuted). 100%
     mismatch. Wrong because `evalkeygen_host_sold` always encrypts under
     whichever secret the SUPPLIED public key corresponds to — it never
     reads the `sQP` parameter in its arithmetic — so this permuted the
     wrong operand entirely.
   - Attempt 2: permuted the PUBLIC KEY (`pkA_QP`,`pkB_QP`) by `k^-1`
     instead (verified algebraically that this is a valid public key for
     `sigma_{k^-1}(s)`: if `pkB ≈ -pkA*s+e` then `sigma(pkB) ≈
     -sigma(pkA)*sigma(s)+sigma(e)`), kept payload = unpermuted base
     secret `s`. STILL 100% mismatch.

   **Where to look next**, precisely: `ks_moddown`'s formula
   (`kernels/moddown.cu`) is `ans=(res_Q - res_P_switched)*PInvModq` —
   this assumes the QP-basis value it's given is a multiple of P. Trace
   through `evalkeygen_host_sold` (`src/evalkeygen.cpp`): `bv` gets an
   explicit `+P*s2` term ONLY within each part's own Q-tower range; `av`
   NEVER gets a P-scaling term at all. `keyswitch_resident`/
   `ks_accumulate_qp` accumulates `res0 = Σ digit_i*bv_i` and
   `res1 = Σ digit_i*av_i` — then calls the SAME `ks_moddown` on BOTH.
   If `res1` (built purely from `av`, which has no `+P*...` term) isn't
   actually a P-multiple the way `res0` is, applying the same "(a-b)*PInv"
   correction to it may itself be the bug, OR the decryption identity
   that relates `res0`,`res1`, and whichever secret multiplies `res1`
   needs to be re-derived line-by-line from THIS codebase's exact
   `LaunchKSMultAcc`/`LaunchKSMultAccBatched` kernels (check
   `kernels/ks_fastcore.cu` for the exact per-tower formula those use —
   NOT assumed by analogy with the high-level OpenFHE description from
   the earlier literature research round). Re-derive what
   `ba0 + ba1 * X ≈ ...` actually equals for THIS codebase's specific
   key structure, for some X, before trying a third key-convention variant.

   Original design notes (still relevant background, see the git history
   of this file for the original write-up):

   (The `PModq` exposure, `ks_moddown` extraction, and single-rotation
   reordering described in earlier revisions of this file are now DONE —
   see `ks_accumulate_qp`/`ks_moddown` in `keyswitch_resident.h`,
   `LaunchScaledAddInto` in `stage_acc.h`, and `tests/test_hoist_single.cpp`.
   What remains is fixing the key-generation convention per the precise
   notes above, then — only once a single rotation is bit-exact — building
   the actual multi-rotation decompose-sharing mechanism, which is a
   SEPARATE subsequent step not yet started. The `ks_decompose_once`/
   `PersistentDigits` structures from the FIRST (different, also-failed)
   hoisting attempt earlier this session were mathematically fine for
   decompose-sharing itself, just built around the wrong overall ordering
   — they are gone (reverted), but the pattern is a reasonable reference
   once the single-rotation case works.)

2. **Fix `KROT`'s own memory scaling — the blocker for n=131072.** 77
   entries × 62.9MB at n=8192 (4.84GB) → 77.5GB at n=131072. `KSC` is
   already fixed and is NOT the remaining problem. Needs a zero-copy
   av/bv sharing mechanism (changes `KeySwitchConstants`'s field types —
   broad blast radius, needs correctness review before trusting) or a
   bigger architectural change (streaming). Not yet scoped in as much
   detail as #1 above — do #1 first, it's better understood.

3. **Run n=131072 end to end, once #2 above is fixed.** Keep the sparse
   secret (h=64) — no `keygen.cpp` change needed, per the corrected
   security analysis (134 bits, verified). Full correctness re-
   verification before trusting any timing number.

## What NOT to do

- Do not re-attempt the OLD hoisting approach (permuting decomposed
  digits directly) — falsified via direct debug test AND via literature
  confirming the correct order is different (permute the QP-basis OUTPUT,
  not the digits).
- For experiment #3's key-generation convention specifically, do NOT
  retry either: (a) permuting the payload secret `s2` by `k^-1` while
  keeping `pkA_QP`/`pkB_QP` as `KPqp`'s own unpermuted public key, or
  (b) permuting `pkA_QP`/`pkB_QP` by `k^-1` while keeping the payload as
  the unpermuted base secret. Both measured 100% mismatch in
  `tests/test_hoist_single.cpp`. Re-derive from `ks_fastcore.cu`'s exact
  per-tower formula first (see the precise notes above).
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
