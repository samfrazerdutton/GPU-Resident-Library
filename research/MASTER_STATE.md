# Master state

Last updated: this session, after the master-directive research cycle
(hoisting root-cause research, memory feasibility measurement, security
re-assessment, messaging architecture proposal).

## Current verified state

- Repo: `github.com/samfrazerdutton/GPU-Resident-Library`, branch `main`,
  commit `865b59a` (pushed to origin).
- Hardware: RTX 2060 Max-Q, 6144MB VRAM, CUDA 13.2, driver 595.97, WSL2
  Ubuntu, **10GB host RAM visible to WSL2**, 16 CPU threads.
- Current demo bootstrap (n=8192, log2(QP)=1775, sparse secret h=64):
  ~114.6-129s wall time depending on run (GPU clock throttled under
  WSL2, 300-975MHz of 2100MHz max — confirmed, unresolved, environmental).
  **This parameter set is ~51 classical / ~46 quantum bits of security —
  not 128-bit. It has never been secure.** Treat every speed number
  measured against it as "how fast this insecure demo runs," not a
  security-relevant result.

## The two biggest findings this cycle

1. **Security target correction**: n=131072 (this session's earlier
   conclusion) was wrong. The actual fix is **n=65536 + switch the secret
   from sparse (h=64) to uniform ternary** — verified with this repo's own
   `lwe_security.py` to give 128/120-bit classical/quantum security at the
   existing 1775-bit modulus budget. n=131072 would be 2.2x more security
   than needed at 2x the cost. See `SECURITY_ASSESSMENT.md`.

2. **Memory feasibility inverted from initial assumption**: GPU VRAM is
   NOT the binding constraint for scaling n (only 181MB of 6144MB used at
   n=8192 today). **Host RAM is** — measured peak RSS of 10.78GB at the
   CURRENT n=8192, driven directly by `test_boot.cpp`'s unbounded
   KROT/KSC host-side rotation-key caches. This must be fixed (bounded
   cache/streaming) before n can be increased at all, independent of GPU
   capability. See `MEMORY_FEASIBILITY.md`.

## Current bottleneck (what's actually slow, and why)

Profiled (not guessed) breakdown of C2S+S2C's ~100s combined:
`rotate_ct_resident` ~62-66s, `pt_to_eval_host` ~29-30s, CPU elementwise
accumulate ~0.6s (negligible — explains why earlier CPU→GPU migration
attempts there were null results). Root cause of `rotate_ct_resident`'s
cost, per a delegated literature-grounded research pass reading OpenFHE's
actual source: it recreates the GPU key-switch context and re-uploads the
ENTIRE rotation key (~63MB at current params) over PCIe, plus host
round-trips, on EVERY SINGLE rotation — this is memory-bandwidth/data-
movement-bound, not raw-arithmetic-bound, consistent with published
findings (Jung et al., TCHES 2021, "Over 100x Faster Bootstrapping... with
GPUs").

## Active hypothesis / next action (see NEXT_SESSION.md for exact commands)

Three ranked, literature-grounded experiments queued (see
`EXPERIMENT_LEDGER.md`'s "Round 2" table and `NEXT_SESSION.md`):
1. Make `rotate_ct_resident` actually resident (hoist context/key out of
   the BSGS loop) — lowest risk, pure refactor, **in progress now**.
2. Eval-domain automorphism as a precomputed index permutation (removes
   the INTT→permute→NTT round trip entirely).
3. Correct-order rotation hoisting (OpenFHE's method: permute the QP-basis
   output after key-switch, not the decomposed digits; regenerate keys
   with the ψ_{k⁻¹}(s) convention) — now understood to require ZERO
   changes to the approximate basis conversion, unlike today's failed
   attempt #1.

Separately, and NOT yet started: the KROT/KSC bounded-cache redesign
(blocking any n-scaling work) and the n=65536/uniform-ternary parameter
switch + re-measurement.

## Environment constraints (confirmed, not assumed)

- GPU clock throttling under WSL2: real, measured, unresolved. Will not
  attempt privilege escalation (`nvidia-smi -lgc`) to fix it — that is a
  system-settings change outside what this session does without the user
  doing it themselves.
- Host RAM ceiling (10GB in WSL2) is now a CONFIRMED binding constraint,
  not a hypothesis.
- No actual n=16384/32768/65536/131072 staged allocation benchmark has
  been run yet (planned, not done — see Phase B in `MEMORY_FEASIBILITY.md`).
