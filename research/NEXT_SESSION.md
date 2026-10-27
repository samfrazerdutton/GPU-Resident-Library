# Next session — exact continuation

## Current commit

Check `git log -3` on resume. As of this writing: `93f2231` on `main`,
pushed to origin (confirm push status — push again if `git status` shows
ahead of origin).

## Read first

`research/MASTER_STATE.md` (current state, written AFTER this round's
real work — supersedes anything said earlier in the same file's history
or in `docs/performance-baseline.md`). Then `research/EXPERIMENT_LEDGER.md`
for the full experiment history including this round's additions.

## Before trusting any new benchmark number

GPU thermal state matters on this machine (Max-Q mobile part). Check:
```bash
wsl -d Ubuntu -- bash -lc 'export PATH=/usr/bin:/bin:/usr/sbin:/sbin; /usr/lib/wsl/lib/nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,power.draw --format=csv'
```
Use `research/benchmarks/run_trials.sh <label> <N>` for repeated-trial
measurement, or `research/benchmarks/interleaved_ab.sh <rounds>` (requires
two binaries, `test_boot` + a second one built from a modified copy, e.g.
`test_boot_exp`) for a true A/B. Both log full GPU state per trial to
`research/benchmarks/*.jsonl`. **Do not trust a single sequential run** —
this session's own baseline drifted from 62-66s to 71-82s for the same
unchanged code purely from thermal buildup.

## Priority queue (highest value first, reflects this round's real results)

1. **Fix `KROT`'s own memory scaling — the actual, quantified blocker for
   n=131072.** `MEMORY_FEASIBILITY.md` has the exact numbers: 77 entries ×
   62.9MB at n=8192 (4.84GB) → 77.5GB at n=131072. `KSC` is already fixed
   (bounded LRU, cap=48, `93f2231`) and is NOT the remaining problem.
   Options, hardest-but-best first: (a) change `KeySwitchConstants::av`/
   `bv` from `std::vector<std::vector<uint64_t>>` to a shared/reference
   type so `KSC`'s views (and any other derived key) stop deep-copying
   `KROT`'s data — this is the structurally correct fix but touches a
   core type used across `src/*.cpp`; get a cryptographic-correctness
   review of the change before trusting it, and test exhaustively (every
   place `.av`/`.bv` is read AND written, not just `test_boot.cpp`'s own
   usage). (b) Verify whether `KROT`'s 77 entries are really all needed
   simultaneously, or whether per-stage working-set analysis could safely
   evict between C2S and S2C (current instrumentation says NO — S2C
   reuses C2S's rotation amounts almost completely — so this option is
   likely a dead end, but hasn't been rigorously disproven, just
   observationally discouraged). (c) Stream av/bv to GPU on-demand without
   a full host-resident copy (biggest architectural change, highest risk).

2. **Eval-domain automorphism as a precomputed permutation** (ranked
   experiment #2 — not yet attempted this round either). Eliminates the
   INTT→permute→NTT round-trip in `automorphism_eval_device`
   (`src/rotate.cpp`). Needs: (a) derive the exact permutation from this
   codebase's specific NTT convention (read `kernels/ntt_ct.cu`'s actual
   bit-reversal/ordering — the header comments say "standard-order in,
   bit-reversed out" for forward NTT but the EXACT index formula needs
   deriving from the kernel code itself, not assumed); (b) build a small,
   isolated correctness test (exhaustive small n, randomized larger n)
   BEFORE touching the hot loop; (c) only then benchmark, using the
   interleaved harness.

3. **Correct-order rotation hoisting** (ranked experiment #3). Per the
   OpenFHE-source-grounded research from the prior round: decompose the
   UNPERMUTED ciphertext once; keyswitch; permute the QP-basis OUTPUT
   (ba0/ba1) after the key-switch inner product, not the decomposed
   digits; regenerate rotation keys as keyswitch-to-ψ_{k⁻¹}(s). Build a
   standalone correctness test (N independent rotations vs.
   decompose-once + N finishes) before touching `test_boot.cpp`. Do this
   AFTER #2 lands, since #2 changes the per-rotation baseline #3 would
   amortize against.

4. **Run n=131072 end to end, once #1 above is actually fixed.** Keep the
   sparse secret (h=64) — no `keygen.cpp` change needed, per the corrected
   security analysis. Regenerate moduli/roots at n=131072 in
   `tests/test_boot.cpp`. Full correctness re-verification (bit-exact vs.
   OpenFHE oracle) before trusting any timing number. This is the
   "make the secure configuration run end to end" deliverable — currently
   NOT done, explicitly blocked on #1.

## What NOT to do

- Do not re-attempt the OLD hoisting approach (permuting decomposed
  digits) — falsified twice now (once via direct debug test, once via
  literature confirming the correct order is different).
- Do not switch to a uniform ternary secret — the corrected security
  analysis shows it does not help for THIS codebase's circuit (needs more
  EvalMod depth than the sparse secret, net worse at the n values that
  matter). Keep h=64.
- Do not re-attempt experiment #1 (resident-rotation context/stream
  hoisting as previously scoped) expecting a different result — it was
  PROPERLY validated as a null result this round (interleaved A/B,
  post-thermal-equilibrium comparison), not just inconclusive. A
  completely different design would be a new experiment, not a retry.
- Do not trust a single-run benchmark on this machine without checking
  GPU temperature first.
- Do not use CKKS for messaging — `MESSAGING_ARCHITECTURE.md`'s
  recommendation (ML-KEM-768 + Double Ratchet + AEAD) stands, unchanged
  this round.

## Standing environment facts

RTX 2060 Max-Q, 6144MB VRAM, CUDA 13.2, driver 595.97, WSL2 Ubuntu, 10GB
host RAM visible to WSL2, 16 CPU threads. GPU clock throttling under
WSL2 and thermal drift over long sessions are both real, measured, and
not fixed (will not escalate privileges to force clock locking).
