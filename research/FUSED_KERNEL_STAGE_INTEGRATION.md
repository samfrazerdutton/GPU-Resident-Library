# Fused pt-mul-accumulate kernel — end-to-end stage integration (Mission 8)

GOD PROMPT "CKKS PHASE 27", Mission 8. Phase 26 proved `LaunchPtMulAcc`
(`kernels/stage_acc.cu`) bit-exact across sizes/boundaries/multi-diagonal/
concurrency (26/26 checks, `tests/test_stage_acc_differential.cpp`) and
measured ~7.6x in isolation against the CPU `ct_mul_pt_host`+`ct_add_ct_host`
pair. Mission 8 requires the harder question: does wiring it into the
bootstrap's actual accumulate loop improve **end-to-end stage latency**, not
just the isolated operation — explicitly warning not to report kernel-only
speedup as an application-level number.

## Semantics, domain compatibility, aliasing — re-verified, not re-asserted

- **CPU reference** (`src/ptops.cpp`): `ct_mul_pt_host` then `ct_add_ct_host`,
  both plain 128-bit-intermediate modular arithmetic, no approximation.
- **Fused kernel** (`kernels/stage_acc.cu`): `acc = (acc + b*d) mod q`, same
  128-bit-intermediate arithmetic. Equality with the CPU pair is therefore
  EXACT, not noise-tolerant — confirmed by Phase 26's 26/26 and reconfirmed
  in this round's stage-level test (0/491520 limbs differ).
- **Domain**: both operate on EVAL-domain (post-NTT) data; this was never a
  coeff/eval mismatch risk since `LaunchPtMulAcc`'s own inputs and
  `pt_to_eval_host`'s output are both eval-domain by construction.
- **Aliasing**: `LaunchPtMulAcc`'s `acc0`/`acc1` output arguments are never
  aliased with its `b0`/`b1`/`d` inputs in either the CPU-reference call
  pattern or this round's integration test — each diagonal uses freshly
  uploaded `b0`/`b1`/`d` buffers distinct from the resident accumulator.
- **Buffer lifetime / synchronization**: the fused kernel takes an explicit
  `cudaStream_t` and holds no static/global device state (confirmed by
  reading `kernels/stage_acc.cu`: zero `static` declarations) — unlike
  `xf_multi`'s `scratch()` buffer, the actual root cause of Phase 25's
  concurrency bug. Chaining many diagonals on ONE stream with ONE
  synchronize at the end (rather than per-diagonal) is therefore safe by
  construction, and is exactly what `tests/test_stage_acc_differential.cpp`'s
  multi-diagonal test (Phase 26) already proved.

## The end-to-end stage integration test

New `bench/bench_stage_acc_integration.cpp` simulates one real transform
stage's diagonal loop (NDIAG=16, matching the LST=3/RST=4 structure
`research/BOOTSTRAP_GPU_RESIDENCY_AUDIT.md` documents), comparing:

- **Arm A** (current production pattern, unchanged): per diagonal —
  `encode_host` → `pt_to_eval_host` (full H2D/NTT/D2H round trip) →
  `ct_mul_pt_host` (CPU) → `ct_add_ct_host` (CPU).
- **Arm B** (GPU-resident accumulate, built from already-exported, already-
  tested primitives only): per diagonal — `encode_host` → `pt_to_eval_host`
  (the SAME round trip, deliberately UNCHANGED — see below) → upload the
  diagonal and the (simulated) rotated ciphertext → `LaunchPtMulAcc`
  directly into a device-resident accumulator (no per-diagonal download).
  Only ONE download at the very end of the stage.

`pt_to_eval_host` is left identical in both arms on purpose: it already does
its own complete round trip inside an unexported anonymous-namespace function
(`xf_multi`), and making IT resident too is a separate, larger change (the
residency audit's O3) needing its own seam — out of this experiment's scope,
which is specifically the accumulate step Mission 8 asks about. Rotation
itself is SIMULATED (identical fixed data in both arms) rather than computed,
since rotation correctness is already covered by `tests/test_rot_resident.cpp`
and using identical simulated data in both arms cannot bias the A-vs-B
comparison.

### Result

```
[correctness] 0 / 491520 limbs differ -> [PASS bit-exact]

current (CPU multiply+add, per-diagonal nothing extra) : 0.679054 [0.610734..0.694168] s
resident (fused kernel, ONE final download)             : 0.564999 [0.478834..0.578433] s
END-TO-END STAGE speedup: 1.2019x
```

**A real, bit-exact, end-to-end stage-level speedup of ~1.2x** — far more
modest than the isolated kernel's ~7.6x, because `pt_to_eval_host`'s own
round trip (unchanged in both arms) dominates the stage's total time. This is
exactly the correction Mission 8 asks for: the isolated-kernel number was
never a safe proxy for the application-level effect.

## Verdict

Both conditions Mission 8 sets for promotion are met: **correctness is
preserved** (bit-exact) and **end-to-end stage latency improves** (~1.2x,
real and measured). This is a small, low-risk, evidence-backed candidate for
production integration — unlike the resident-context cache (see
`research/KS_RESIDENT_CACHE_DESIGN.md`), whose narrow applicability argues
against it.

**Not done this round**: actually wiring this into `tests/test_boot.cpp`'s
real `stageRotDiag`/accumulate loop. That is a production-code change to a
file the governing directive requires extra care around, and doing it
properly means replacing the CPU accumulate call sites with the
device-resident pattern this benchmark validates — a small, mechanical,
but real edit to `tests/test_boot.cpp` (not `src/keyswitch_resident.cpp`,
so the "no internals without a reproducer" rule does not block it) that
was not attempted this round in the interest of keeping cache work (Missions
1-7) and kernel work (Mission 8) as two separate, independently-verifiable
changes, per the directive's own explicit "do not combine this experiment
with cache integration in the same initial A/B" instruction. The evidence
above is the justification for a FUTURE such change, not the change itself.

## Commands

```
cmake --build build2 --target bench_stage_acc_integration -j4
./build2/bench_stage_acc_integration            # defaults: n=8192 tw=30 NDIAG=16 trials=9
```
