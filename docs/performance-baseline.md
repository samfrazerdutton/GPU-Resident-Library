# Performance baseline — full bootstrap, this session

All numbers below are from real, executed runs of `tests/test_boot.cpp` on this
machine: NVIDIA GeForce RTX 2060 (mobile/Max-Q class), CUDA 13.2, driver 595.97,
under WSL2 Ubuntu. Ring dimension n=8192, sizeQ=30 towers (1 x 60-bit + 29 x
55-bit), sizeP=2 (2 x 60-bit), for log2(QP)=1775 total. **No security parameter
was ever changed by any work in this log** — but see "Security reality check"
below: this parameter set was never actually validated against this repo's own
estimators before this check, and it is not close to 128-bit secure. Every
number in this file is a correctness/performance measurement of a DEMO
parameter set, not a secure one. `npFor` scaling gives up to numPart=15 digit
decomposition parts at the top level.

## Security reality check (n=8192, log2(QP)=1775, sparse secret h=64)

Ran this repo's own two literature-calibrated estimators (`tools/lwe_security.py`,
`tools/hybrid_security.py` — both validated against published reference tables
and real cryptanalysis papers, see their own docstrings) against the actual
parameters `tests/test_boot.cpp` uses, not the toy sizes their built-in
`--validate`/`--calibrate` self-tests use:

```
$ python3 tools/hybrid_security.py 8192 1775 64
n=8192 log2(QP)=1775 h=64: 51 classical bits   (binding attack: hybrid dual+MITM)
$ python3 tools/lwe_security.py 8192 1775 64
n=8192 log2(QP)=1775 sparse h=64: 47 classical / 46 quantum bits (primal, upper bound)
```

**51 bits of classical security, ~46 quantum-adjusted.** Not 128-bit. Not even
80-bit. This is a real, concrete result from this repo's own validated tooling,
not a guess — and it is NOT a regression from any change in this log: these
parameters were never checked against this tooling before now. The cause is
basic LWE parameter scaling, nothing exotic: at n=8192 this repo's own
`lwe_security.py --validate` shows only ~218 bits of modulus budget gives
128-bit security for a UNIFORM secret; log2(QP)=1775 here is over 8x that
budget, chosen purely to give the bootstrap circuit enough levels to survive
CoeffsToSlots -> EvalMod -> SlotsToCoeffs, with no security check performed
against it. Checked what ring dimension WOULD actually secure this same
1775-bit budget at the same sparse-secret hardness:

```
n= 8192 -> 51 bits     n=32768 -> 55 bits      n=65536 -> 90 bits
n=16384 -> 52 bits     n=131072 -> 134 bits  (>= 128-bit threshold)
```

Reaching genuine 128-bit security at this circuit's depth needs roughly
**n=131072 — 16x this session's n=8192.** Checked the other direction too: even
at n=8192's own validated 218-bit "128-bit-uniform" budget, the sparse secret
(h=64) this repo actually uses only reaches 85 bits via the hybrid attack, and
it gets worse as the budget grows (69 bits at 300, 57 at 400, 51 at 500+) — so
a real fix at n=8192 would also need a much shallower circuit (far fewer
towers/levels), not just a different n.

**This does not invalidate any optimization work in this log.** Every
technique (NTT/INTT batching, resident keyswitch, root-table/key caching,
hoisted rotations) is parameter-size-agnostic — it applies identically at
n=131072, it is just more expensive there (NTT cost scales ~O(n log n), so
roughly a 16x-21x per-tower cost increase moving n=8192 -> 131072, before
counting that a properly-shallow or properly-sized circuit might also need a
different tower count). What this DOES invalidate is treating any wall-clock
number in this file as "how fast post-quantum encryption runs" — it is
currently "how fast this specific insecure demo circuit runs." A genuinely
secure full bootstrap at comparable depth will be substantially slower than
everything measured in this log, on any hardware, including a datacenter GPU
without this session's WSL2 clock-throttling constraint. That is cryptographic
reality, not an engineering gap to optimize away.

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
| — | 128.253 s | ~0%† | 5.47871e-4 | async evalkeygen data uploads (keygen phase) -- correct, kept, attribution error (clock-throttling evidence was from a different function) |
| 6+7 | 120.123 s | -2.4% vs step 5 | 5.47871e-4 | batch keyswitch_resident's own decompose/moddown loops + async ks_context_create uploads + stream threading in rotate_ct_resident |
| — | 125-129 s | ~0%† | 5.47871e-4 | TESTED AND REVERTED: pinned host staging buffers for ks_context_create's av/bv uploads -- 2-buffer alternation forced a host sync almost immediately with no work to overlap, net negative, not kept |
| 8 | 118-126 s | ~0%† | 5.47871e-4 | move EvalMod's ct*ct RNS tensor product from CPU to GPU -- honest no-op, the CPU loop (0.58s of total, see profiling below) was never the bottleneck |
| 9 | 114.6-126 s | real win in EvalMod | 5.47871e-4 | cache buildK(tw) relin keys across EvalMod's R/I branches (they walk an identical deterministic tw sequence) -- EvalMod x2 dropped from 17.7-18.4s to a consistent 14.9-15.1s across two runs |
| 10 | 114.6-114.9 s | ~consistent | 5.47871e-4 | batch xf_multi's (pt_to_eval_host) per-tower NTT/INTT loop -- correct and kept, but the targeted bucket itself barely moved (30.1s -> 29.5-29.6s); the total's drop may be partly ordinary clock-state variance, reported honestly rather than overclaimed |

† Within this setup's observed run-to-run noise band (now understood to be wider
than first thought — up to ±8s on identical code, tracked to WSL2 GPU clock
throttling, see below). Kept each † row because it is strictly correct and
either reduces launch count or removes genuinely CPU-bound work, even when the
measured number didn't move. Rows marked "TESTED AND REVERTED" are listed as
unnumbered entries, not steps, for the same reason — report the honest result
either way, not just the wins; a reverted step is not a completed one.

**Net so far: 191.0s -> ~114.6-115s most recently measured, roughly a 40% reduction.
Error did not move by a single bit across any change** (every row's error column
is byte-identical: 0.000547871, now twelve consecutive changes with zero drift).
This is the actual gate — every change above was reverted immediately if it broke
this, no exceptions.

## Where the time actually goes (profiled, not guessed)

Added standing timing instrumentation to C2S/S2C's per-diagonal loop body
(`g_tPtEval`/`g_tRot`/`g_tCpuAcc` in `tests/test_boot.cpp`) rather than keep
guessing at the next lever. Measured split of C2S+S2C's combined ~100s:

```
rotate_ct_resident (GPU keyswitch)        : 62.7-65.7 s
pt_to_eval_host (plaintext encode+GPU-NTT): 29.5-30.1 s
CPU ct_mul_pt_host + ct_add_ct_host       : 0.57-0.58 s
```

This directly explains the earlier CPU->GPU diagonal-accumulation revert: that
CPU loop was only ever 0.5-0.6% of the bootstrap's total time, so moving it to
the GPU could never have produced a measurable win. `rotate_ct_resident` is the
genuine dominant cost and is already on the fully-resident, batched, cached path
from earlier this session.

## What the stage breakdown looks like at the most recent checkpoint

```
keygen (48 keys) = 38-42 s
C2S              = 66-73 s
EvalMod x2       = 14.9-15.1 s   (down from 17.7-18.4 s after the buildK cache)
S2C              = 30-36 s
other            = ~1.7-1.9 s
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

`ks_context_create`'s per-rotation eval-key upload (`upv()`, used for
`K.av`/`K.bv`) was converted from synchronous `cudaMemcpy` to `cudaMemcpyAsync`
on a real stream (rows 6+7 above) — a real but modest 2.4% gain. A follow-up
attempt to make that upload *genuinely* async by staging through pinned
(`cudaMallocHost`) host memory (plain `cudaMemcpyAsync` on pageable memory still
blocks in practice) was built, verified bit-exact, and measured as a **real
regression** (125-129s vs the 120.123s baseline) — the 2-buffer alternating
design forced a host-side sync almost immediately with no intervening work to
hide the transfer behind, so it paid for an extra memcpy and event-sync overhead
for nothing. Reverted, not committed.

The clock-throttling root cause itself remains environmental and unresolved:
`nvidia-smi -lgc 2100,2100` still refuses for lack of elevated privileges on
both WSL and native Windows in this session, and was not retried. Every
subsequent round's honest results (above) are consistent with this: changes
that reduce genuinely redundant GPU work (the buildK cache) produce real wins;
changes that only restructure already-necessary GPU work (batching, async
uploads, pinned staging) produce marginal-to-negative results, because the
underlying compute is throttled regardless of how it's dispatched.
