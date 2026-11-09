# Mission 3 — Device automorphism and stage-residency validation

GOD PROMPT "CKKS PHASE 26", Mission 3. Validates whether
`rotate_ct_resident` (the device-automorphism-based rotation path Phase 25's
audit traced) genuinely removes host round trips, with the explicit warning
the directive states observed directly: "Do not label an operation
GPU-resident merely because one kernel runs on the GPU."

## 1. Correctness: re-confirmed fresh this round

`tests/test_rot_resident.cpp` is the existing standalone correctness test
against the trusted host path (it was not rebuilt from scratch — it already
meets the directive's requirement, and rebuilding an equivalent test would
have been pure duplication). Re-run fresh this round at the bootstrap's own
parameters:

```
./build2/test_rot_resident 8192 30 9
```

Result: **0 / 491520 mismatching coefficients. [PASS] resident rotation is
bit-exact.** Peak host RSS 253 MB. Automorphism device-vs-host: 0 mismatches.
`rotate_ct_device` (pure device-pointer path) vs `rotate_ct_resident`: 0
mismatches.

## 2. Function-level call-site tally — the real finding

Phase 25's audit counted transfers/syncs/allocations per FILE
(`src/rotate.cpp`: H2D=10, D2H=4, sync=3, malloc=8). This round counted them
per FUNCTION, comparing `rotate_ct_host` directly against
`rotate_ct_resident`, by reading both bodies in full (`src/rotate.cpp` lines
221-334) and `keyswitch_core_resident`'s full 166-line body
(`src/keyswitch.cpp`).

| Call site | `rotate_ct_host` (one rotation) | `rotate_ct_resident` (one rotation) |
|---|---|---|
| H2D transfers | 12 (4 from two `automorphism_eval_host` calls via `xf_multi`; 8 from `keyswitch_core_resident`) | **33** (31 from `ks_context_create`'s eval-key upload — 30 for `av`/`bv` across 15 parts, 1 for `d_keyRow`; 2 for the ciphertext) |
| D2H transfers | 10 (4 automorphism; 6 `keyswitch_core_resident`) | **3** (1 c0 download, 2 `ba0`/`ba1` download) |
| `cudaDeviceSynchronize`/`cudaStreamSynchronize` | 8 (2 per `xf_multi` call x4 calls, from both automorphisms; 4 inside `keyswitch_core_resident`) | **2** |
| `cudaMalloc` | 14 (all inside `keyswitch_core_resident`'s per-tower-per-part decompose/moddown buffers) | **47** (31 from `ks_context_create`'s fresh `av`/`bv`/`d_keyRow`; 11 from `ks_work_create`; 5 explicit in the function body) |
| `cudaFree` | 9 | **47** (matching the above) |

**The "resident" path has nearly 3x the H2D call sites and more than 3x the
malloc/free call sites of the "host" path it replaces** — every one of them
from re-uploading and re-allocating the eval key fresh, every single call.
What it genuinely wins is D2H count (3 vs 10) and synchronization count (2 vs
8), because the ciphertext and the keyswitch kernels themselves truly stay
resident. The automorphism kernels (`automorphism_eval_fast_device` x2) and
`keyswitch_resident` itself are confirmed, by direct reading, to add zero
transfers, zero allocations, and zero syncs of their own — exactly the
GPU-resident property the directive asks to verify rather than assume.

**So "GPU-resident" here is accurate for the ciphertext and the kernel
execution, and false for the eval key.** Labeling the whole rotation
"resident" without that qualification would be exactly the error Mission 3
warns against.

## 3. Where the time actually goes — measured, not inferred

`test_rot_resident`'s own phase-breakdown instrumentation (re-run fresh this
round, n=8192, tw=30, 9 reps, median):

| Phase | Time | Share of resident total (0.0479 s) |
|---|---|---|
| `ks_context_create` | 0.02046 s | 43% |
| `free/destroy` | 0.00990 s | 21% |
| automorphism x2 | 0.00988 s | 21% |
| KEYSWITCH KERNELS | 0.00587 s | 12% |
| `ks_work_create` | 0.00164 s | 3% |
| `cudaMalloc` x3 (explicit) | 0.00158 s | 3% |
| upload/download ciphertext | 0.00146 s | 3% |

**`ks_context_create` + `free/destroy` together are 63% of the resident
path's own total time** — more than the automorphism and the actual keyswitch
kernels combined (33%). This is the same conclusion Phase 25's static byte
count reached (eval-key traffic dominates), now confirmed from the function's
own measured phase timing rather than from reading `evalkeygen.cpp`'s
`resize()` calls.

Overall speedup vs the host-orchestrated path remains real and substantial:
**4.00x** (0.19174 s -> 0.04788 s, median of 9 reps) — but per the table
above, nearly half of even the FASTER path's own time is the thing Mission 4
targets for removal. The residency audit's O1 (cache `DeviceKSContext` per
`(k, tw)`) and this rotation benchmark are measuring the same bottleneck from
two different angles, and Mission 4's experiment (section 4 below) shows what
removing it is worth.

## 4. Connection to Mission 4

Mission 4's `bench/bench_ks_context_resident_ab.cpp` experiment (see
`research/EVAL_KEY_TRANSFER_VERIFICATION.md`) demonstrates that caching the
`DeviceKSContext` across repeated use of the same `(k, tw)` eliminates
exactly the 43%+21%=64% of this rotation's time that `ks_context_create` and
its matching `destroy` currently cost, while remaining bit-exact. That
experiment does not modify `rotate_ct_resident` or any production file; it is
offered as evidence for a future, explicitly-scoped integration decision.

## 5. What this does NOT establish

- This is still the KEY-SWITCH-ADJACENT rotation primitive in isolation, not
  the bootstrap. `tests/test_boot.cpp`'s actual stage loop (per
  `research/BOOTSTRAP_GPU_RESIDENCY_AUDIT.md`) does not even call
  `rotate_ct_resident` — it has its own inlined `stageRotBegin`/`stageRotDiag`/
  `stageRotEnd` sequence that ALSO calls `ks_context_create` fresh per
  diagonal, with the same cost structure. No claim is made here about the
  bootstrap's end-to-end timing.
- `keyswitch_core_resident`'s own 166-line body was read for call-site
  counting only, not independently re-verified for correctness beyond the
  bit-exact gate `test_rot_resident` already provides by comparing its output
  end to end.
- The 4.00x figure is this run's median at 9 repetitions on a machine with a
  documented ~30% run-to-run spread; `test_rot_resident`'s own min/max
  (0.045-0.074 s resident, 0.182-0.246 s host) reflects that spread directly.
