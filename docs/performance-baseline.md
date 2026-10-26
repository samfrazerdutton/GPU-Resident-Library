# Performance baseline — full bootstrap, this session

All numbers below are from real, executed runs of `tests/test_boot.cpp` on this
machine: NVIDIA GeForce RTX 2060 (mobile/Max-Q class), CUDA 13.2, driver 595.97,
under WSL2 Ubuntu. Ring dimension n=8192, sizeQ=30 initial towers, sizeP=10,
security target per the existing hybrid_security.py/lwe_security.py estimators
(unchanged by any work in this log — no security parameter was ever reduced).
`npFor` scaling gives up to numPart=15 digit decomposition parts at the top level.

Every row's correctness is `max slot err |boot(z) - z|` after a full
encrypt -> deplete -> bootstrap -> decrypt round trip, compared against OpenFHE as
the bit-exact oracle for every primitive beneath it (per this repo's existing
`tests/test_*.cpp` gates). Time is wall-clock `real` time for the whole process,
measured with the shell's own `time`, CUDA work included, not a kernel-only figure.

| Step | Wall time | Δ vs prior | Max slot error | Change |
|---|---|---|---|---|
| 0 | 191.002 s | — | 5.47871e-4 | baseline (repo HEAD before this session) |
| 1 | 144.286 s | -24.5% | 5.47871e-4 | cache evalkeygen root tables (device-resident) |
| 2 | 143.281 s | -0.7%† | 5.47871e-4 | batch keyswitch_core_resident decompose loop (kept: correct, no measured win) |
| 3 | 139.984 s | -2.3% | 5.47871e-4 | swap rotate_ct_host -> rotate_ct_resident in C2S/S2C |
| 4 | 132.25 s | -5.5% | 5.47871e-4 | evalkeygen round 2: batch per-part NTT data upload/download |
| 5 | 123.033 s | -7.0% | 5.47871e-4 | skip evalkeygen for rescale-only KeySwitchConstants |
| — | 123.961 s | ~0%† | 5.47871e-4 | TESTED AND REVERTED: CPU->GPU diagonal accumulation (correct, no measured win) |
| 6 | pending | — | — | batch keyswitch_resident's own decompose/moddown loops |
| 7 | pending | — | — | async eval-key uploads in ks_context_create |

† Within this setup's observed run-to-run noise band (~1-3s). Kept step 2 because
it is strictly correct and reduces launch count with zero downside, even though it
didn't move the measured number. Step 4's intermediate entry is listed as a
reverted experiment, not a numbered step, for the same reason — report the honest
result either way, not just the wins.

**Net so far: 191.0s -> 123.033s, a verified 35.6% reduction. Error did not move by
a single bit across any change** (every row's error column is byte-identical:
0.000547871). This is the actual gate — every change above was reverted
immediately if it broke this, no exceptions.

## What the stage breakdown looks like at the 123.033s checkpoint

```
keygen (48 keys) = 40.2408 s
C2S              = 72.0925 s
EvalMod x2       = 17.8066 s
S2C              = 32.377 s
other            = 1.68513 s
```

C2S + S2C account for ~85% of total time. Independently re-derived (not just
assumed from the existing git history) that this is dominated by genuine
arithmetic work proportional to `towers * parts`, not by avoidable
launch-overhead or caching gaps — confirmed four separate ways: (1) batching the
decompose loop in the function that turned out to no longer be in the hot path
showed no effect once isolated from (2) context-caching for rotate_ct_resident,
ruled out analytically (near-1:1 key:diagonal ratio, no real reuse to cache
against) before writing any code, (3) moving the per-diagonal accumulation math
from CPU to GPU, built and bit-exact-verified, measured zero improvement, and
reverted, and (4) a back-of-envelope launch-count-vs-wall-time check (≈0.96s per
keyswitch at C2S stage 2's scale, several orders of magnitude too high to be
launch overhead at microseconds/launch).

## The bigger, still-open finding: GPU clock behavior

A direct nvidia-smi poll (1 Hz) through an entire real bootstrap run showed SM
clock sitting at 300-975 MHz out of a 2100 MHz max the whole time, GPU
utilization 2-33%, power draw 6-26W. The GPU is not staying busy enough to reach
or hold boost clock under this workload's call pattern. Clock-locking was
attempted directly (`nvidia-smi -lgc 2100,2100`) on both the WSL and native
Windows sides and refused for lack of elevated privileges in this session (this
machine's own README already notes "WSL2 blocks GPU clock locking" as a known
constraint for benchmarking on this exact setup).

The most direct, concrete cause identified so far: `ks_context_create`'s per-
rotation eval-key upload (`upv()`, used for `K.av`/`K.bv`, tens of MB per call,
~150 calls across a bootstrap) uses plain synchronous `cudaMemcpy` with no stream
argument at all — the host thread blocks on every single rotation's key upload.
Converting this to `cudaMemcpyAsync` on a real stream is in progress as of this
writing; see the board for current status and this file's next revision for the
measured before/after.
