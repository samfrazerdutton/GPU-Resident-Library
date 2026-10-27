# Memory feasibility — measured, not estimated

Machine: NVIDIA GeForce RTX 2060 Max-Q, 6144 MiB VRAM (5955 MiB free at idle),
CUDA 13.2, driver 595.97, WSL2 Ubuntu, **10 GiB host RAM visible inside WSL2**
(8 GiB swap), 16 CPU threads. All numbers below are from `/usr/bin/time -v`,
`getrusage(RUSAGE_SELF).ru_maxrss`, and `nvidia-smi --query-gpu=memory.used`
polling every 2s during a real `./build/test_boot` run, commit `865b59a`.

## The headline finding, reversed from initial assumption

Earlier analysis in `docs/performance-baseline.md` focused on GPU VRAM as the
blocking constraint for scaling to n=131072 (estimating up to ~960MB per
rotation-key context). That assumption was **wrong in emphasis**. Measured:

| Resource | Peak observed at CURRENT n=8192 | Hard ceiling (this machine) |
|---|---|---|
| **Host RAM (RSS)** | **10,780,944 KB ≈ 10.78 GB** | ~10 GB visible to WSL2 (+ 8 GB swap) |
| **GPU VRAM** | **181 MB** (2-second polling) | 6144 MB |

**GPU VRAM is barely touched (3% of the card) even at the current scale.**
**Host RAM is already AT or PAST this WSL2 session's visible 10 GB limit —
at n=8192, today, with zero changes.** The process survived only via swap
(8 GB available, ~43 MB used at idle).

## Root cause, confirmed by direct instrumentation

`tests/test_boot.cpp` caches generated rotation/relinearization keys in two
process-lifetime `std::map`s that are never evicted:

- `KROT` (line ~71): one entry per *distinct Galois rotation amount*, each a
  FULL top-level `KeySwitchConstants` (own `av`/`bv`, `numPart=15` parts,
  `sizeQlP=32` towers at n=8192).
- `KSC` (line ~246): one entry per *(rotation amount, tw-level)* pair — a
  "view" of a KROT entry's av/bv, re-copied (not shared) at a smaller
  `numPart` for lower `tw`, via `atLevel()`.

Added a one-line RSS/cache-size checkpoint after each bootstrap phase
(`rssMB()` via `getrusage`, printed after C2S/EvalMod/S2C). Real output:

```
[mem] after C2S:     KROT=75  KSC=75   peak RSS=9151.93 MB
[mem] after EvalMod: KROT=76  KSC=75   peak RSS=9685.81 MB
[mem] after S2C:     KROT=76  KSC=150  peak RSS=10527.8 MB
```

RSS tracks cache growth almost exactly: cache count roughly doubles
(150→226 combined entries) and RSS grows ~1.4 GB in step. This is not a
leak — it is the intended "cache everything, never free" doctrine applied
to host-side key material, and at n=8192 it already consumes essentially
the whole WSL2 memory budget.

## Why this reframes n=131072 (and even n=65536) entirely

Per-key cost scales as `2 (av,bv) * numPart * sizeQlP * n * 8 bytes`. At
n=8192 this is the dominant host allocator. Moving to n=131072 (16x) with
the SAME caching doctrine and a SIMILAR tower/numPart count would scale
host RAM by roughly 16x → **~170 GB**, utterly infeasible on any consumer
machine, independent of GPU VRAM (which has 30x headroom by comparison).

**Conclusion: the binding constraint for scaling n is host-side key
caching, not GPU VRAM.** Any attempt to run a deeper/larger-n bootstrap
MUST first replace `KROT`/`KSC`'s "cache every key forever" strategy with
one of:
- A bounded LRU cache (evict least-recently-used keys; re-generate on miss).
- Streaming: generate a rotation key, use it, discard it (no cache), accept
  the recompute cost — viable if `evalkeygen_host`'s cost is small relative
  to the keyswitch itself (currently it is NOT negligible — see
  `EXPERIMENT_LEDGER.md`'s buildK-cache entry, which found EvalMod's R/I
  branches DID benefit from caching identical keys — so blind "never cache"
  is also wrong; the right answer is bounded + content-aware, not binary).
- Keeping only the CURRENT level's rotation keys resident, since BSGS stage
  g's rotation amounts are not reused once stage g+1 starts (needs
  verification against the actual diagonal/offset reuse pattern per stage).

This is now the standing precondition for any n-scaling experiment, tracked
as the first item in `NEXT_SESSION.md`.

## UPDATE: KSC bounded (real win), KROT NOT yet solved (quantified gap)

Implemented a bounded LRU cache for `KSC` specifically (commit `93f2231`):
safe because `KSC` entries are cheap to regenerate on a miss (pure CPU copy
from an already-resident `KROT` entry, no RNG, no `evalkeygen_host` call).
Capped at 48 entries. **Measured result: peak RSS 10.78GB → 8.12-8.32GB
(two independent measurements, `getrusage` and `/usr/bin/time -v`), a real
~23-25% reduction, bit-exact.**

`KROT` was deliberately left uncapped — instrumentation confirmed its ~77
entries are genuinely reused across the ENTIRE bootstrap (count stayed
flat from C2S's end through S2C's end, meaning S2C reuses C2S's rotation
amounts almost completely), so naive LRU eviction there would force real,
expensive `evalkeygen_host` regeneration, not a cheap re-copy. **This
means `KROT` itself remains unsolved, and its own scaling is now the
precise, quantified blocker for n=131072:**

```
KROT entry size at n=8192:  2(av,bv) x 15(numPart) x 32(sizeQlP) x 8192(n) x 8B = 62.9 MB
KROT total at n=8192:       77 x 62.9 MB = 4.84 GB
KROT total at n=131072:     77 x (62.9 MB x 16) = 77.5 GB   <-- infeasible, unaddressed by the KSC fix
```

**n=131072 end-to-end is NOT yet achievable — this is the specific,
quantified reason, not a vague "memory might be tight."** A real fix needs
one of: (a) a zero-copy reference mechanism so `KSC`'s views share
`KROT`'s av/bv storage instead of deep-copying a subset of it (would also
shrink `KROT`'s OWN effective footprint if applied consistently, but
requires changing `KeySwitchConstants`'s `av`/`bv` field types from
`std::vector<std::vector<uint64_t>>` to a shared/reference-counted type —
a change with broad blast radius across `src/*.cpp`, not attempted this
session given the risk, but the single most promising fix a future
session should scope carefully with a cryptographic-correctness review
first); (b) determining whether the 77 KROT entries are REALLY all needed
simultaneously or whether a smarter per-stage working-set analysis (not
yet done) could safely bound it; (c) streaming av/bv from host to GPU
on-demand per rotation without a full host-resident copy at all (a bigger
architectural change). None of these have been implemented. Do not claim
n=131072 is memory-feasible until one of them is.

## GPU VRAM at n=131072 (for completeness, now secondary)

Scaling the *resident* GPU-side structures (`DeviceKSContext`'s per-part
`av`/`bv` device buffers, `DeviceKSWork` scratch) by n alone (16x, holding
tower/numPart count roughly fixed) from the measured 181 MB baseline gives a
rough estimate of ~2.9 GB peak-in-flight — within the 6 GB card, but this
estimate carries real uncertainty (181 MB was sampled every 2s over a
~128s run; a short spike between samples would be missed). This needs a
dedicated finer-grained measurement (CUDA's own peak-allocated-bytes
counter, not external polling) before being trusted — not yet done.

## What's NOT yet done (explicitly incomplete)

- No actual n=16384/32768/65536/131072 allocation-only staged benchmark has
  been run yet (Phase B of the master directive's Section 6 plan). The
  analysis above is the memory MODEL (Phase A) plus a real baseline
  measurement at n=8192; extrapolation to larger n is arithmetic, not yet
  empirically verified at those sizes.
- Fine-grained (sub-2-second) VRAM peak measurement not yet done.
- The bounded-cache redesign for KROT has not been implemented (see below
  for the analysis that settles WHICH design to implement).

## UPDATE: KROT lever analysis complete (delegated investigation) — zero-copy ruled out with code evidence, bounded-cache is the recommended next implementation

A dedicated investigation (this session) read `ks_context_create`/
`ks_context_destroy` directly and found: **every keyswitch call does a
fresh `cudaMalloc`+`cudaMemcpyAsync` of the ENTIRE av/bv to device, then
frees that device copy immediately after the call returns.** There is no
persistent device-resident copy of any key at any time — the host
`std::vector<std::vector<uint64_t>>` IS the only durable copy; the GPU
side is pure per-call scratch. This settles option (a) above
definitively: **zero-copy host/device sharing is not applicable — there
is no second copy to deduplicate**, and separately, VRAM (6GB) < host RAM
(10GB) on this machine anyway, so moving the single copy to device
wouldn't even raise the ceiling. Reject (a).

Two other levers were evaluated and rejected for now, both for reasons
that touch currently-fixed, security-load-bearing budgets rather than
being unsound in principle:

- **BSGS/giant-step retuning in C2S/S2C** (fewer, larger rotation groups
  → fewer distinct keys): a from-scratch Python/numpy simulation of this
  codebase's exact `twd`/`applyMerged`/`stageDiags` BSGS structure,
  cross-validated against the real measured run (predicted 76 rotation
  keys for the CURRENT LST=3 split; the real run's instrumented
  `evalkeygen calls total=77` = 76 rotation keys + 1 relin key, an exact
  match), shows LST=12 (vs. current LST=3) would cut distinct keys
  4096→24-ish down to **24 keys at the finest split, a ~3.2x reduction**.
  **But** this needs 24 rescale levels for C2S+S2C combined vs. the
  current 6, and `SECURITY_ASSESSMENT.md`'s n=131072/134-bit result holds
  the ENTIRE modulus-chain shape fixed (`sizeQ=30, sizeP=2, numPart=15,
  log2(QP)=1775 bits`) — covering 18 more rescale levels at the current
  per-tower bit size would add >1000 bits to `log2(QP)`, which is exactly
  the already-rejected "weaken security to save memory" move. Doing this
  WITHOUT growing `log2(QP)` (more, smaller towers at fixed total budget)
  is security-neutral in principle but requires re-deriving the entire
  CKKS scale-management chain (`Delta_in`/`Delta_pt`/`Delta_w`), which
  this codebase's own comments describe as extremely fragile to resize.
  **This is the highest long-term value lever, but needs a dedicated
  session to re-derive the scale chain safely — not attempted here.**
- **numPart/alpha (dnum) retuning** (fewer, bigger digit groups → smaller
  av/bv per key): directly collides with the same fixed 1775-bit budget
  (bigger alpha needs bigger `sizeP` to keep the rounding-error bound
  sound). Separately and more importantly: **`tests/test_alpha10.cpp`
  already exists in this repo, isolating a reproducible correctness
  failure at alpha=10 in a plaintext-multiply+rescale path with no
  keyswitch involved at all** — a known, real numerical break in exactly
  the direction this lever would push, previously undocumented in any
  `research/*.md` file (confirmed via grep: zero mentions before now).
  **Rejected for this reason, independent of the budget collision.**

**Recommended next implementation: bound KROT itself with LRU eviction,
same pattern as the already-merged `KSC_CAP` fix.** Measured this
session: `evalkeygen calls total=77` took **49.3256s → ≈0.640s/key
regen cost at n=8192** (sizeQl=30, numPart=15, sizeQlP=32). Every KROT
key is used exactly twice per bootstrap (once in C2S, once in the
mirrored S2C pass — by the end of C2S alone, 75 of the 77 keys are
already live), so **any cap below ~76 forces real `evalkeygen_host_sold`
regenerations, not cheap re-copies** — this is a genuine compute-for-
memory trade, not a free win:

| | n=8192 (measured regen cost) | n=131072 (extrapolated, NOT measured — assumes the same ~16x linear-in-n scaling already used elsewhere in this doc) |
|---|---|---|
| Per-key size | 62.9 MB | ≈1.0 GB |
| Regen cost/key | ≈0.64s | ≈10.9s (extrapolated) |
| Cap=40 example | saves (76-40)×62.9MB≈2.26GB, ≥36 forced regens ≈ +23s to a ~148s bootstrap | n/a |
| Cap=8 example | n/a | fits in ~8GB of the 10GB ceiling; ≈68 forced regens ≈ +12 minutes (extrapolated) |

This converts "OOM crash at n=131072" into "survives, but pays a large,
honestly-uncertain compute tax" — it makes n=131072 FEASIBLE, not fast.

## UPDATE: KROT_CAP implemented; REAL (not extrapolated) n=131072 per-key numbers measured

`KROT_CAP` is now implemented in `tests/test_boot.cpp` (bounded LRU,
same pattern as `KSC_CAP`), defaulted to 96 — comfortably above the
n=8192 working set, verified to cause ZERO evictions/behavior change at
that scale (identical `max slot err`, identical peak RSS).

A standalone test (`tests/test_n131072_keysize.cpp`) built the REAL
n=131072 modulus chain (same shape: sizeQ=30, sizeP=2, log2(QP)=1775 —
unchanged circuit) and generated ONE actual rotation key, measuring
`getrusage` RSS directly rather than extrapolating:

```
RSS before any rotation key: 228 MB
ONE rotation key (evalkeygen_host_sold) generated in 7.49601s
RSS after one rotation key: 1313 MB
REAL measured per-key delta: 1085 MB
Exact structural size (2*numPart*sizeQlP*n*8B): 960 MB
Extrapolated total for 76 such keys: 71 GB (structural only)
```

This CONFIRMS the earlier extrapolation was accurate (960MB structural,
~1085MB real including ~13% allocator overhead, vs. the ~1000MB assumed
above) — not an overestimate. Keygen time (7.5s/key) came in LOWER than
the ~10.9s/key extrapolated above (real scaling ≈11.7x over n=8192's
measured 0.64s/key, not the naive ~17x linear-in-n guess).

Based on these real numbers, `tests/test_boot_n131072.cpp` (a
parameterized copy of `test_boot.cpp`, default n=8192 test left
UNTOUCHED) sets `LST=4` (lgS=16 at n=131072 needs a divisor other than
the n=8192 default LST=3; LST=4 keeps the same RST=4 per-stage radix the
original choice was tuned around — the most conservative available
choice, NOT independently re-validated for numerical accuracy at this
new scale).

## UPDATE: three real attempts, all OOM'd — an unidentified memory
## growth source exists BEYOND KROT/KSC; Objective C is NOT complete

Three actual end-to-end attempts were made this round, each with
successively more conservative cache bounds, all ending in an OOM kill
before any stage completed (confirmed by empty stdout logs each time —
`std::cerr`'s per-stage progress lines never printed, meaning not even
C2S stage 0's first 16 diagonals finished):

| Attempt | `KROT_CAP` | `KSC_CAP` | Theoretical cache ceiling | Observed RSS before OOM |
|---|---|---|---|---|
| 1 | 6 | 48 (unchanged — the bug) | ~6.5GB + up to ~48GB(!) | grew past 3.25GB, killed before next check |
| 2 | 4 | 3 | ~4.3GB + ~3.3GB ≈ 7.6GB | grew to ~3.5GB then OOM'd shortly after |
| 3 | 2 | 1 | ~2.2GB + ~1.1GB ≈ 3.3GB | **fluctuated 1.5GB→4.5GB→4.0GB→4.5GB over ~15 CPU-minutes, then OOM'd** |

Attempt 1's failure is fully explained (KSC caches FULL-SIZE views of
KROT entries — see above — and was left unbounded for this scale).
Attempts 2 and 3 are **not** fully explained: attempt 3's observed RSS
(up to 4.5GB) exceeds its own theoretical two-cache ceiling (~3.3GB) by
a wide margin, meaning **some other allocation grows with n in a way
neither `KROT_CAP` nor `KSC_CAP` bounds** — plausible candidates, NOT
yet investigated: `DeviceKSContext`/`DeviceKSWork` host-side staging
vectors inside `ks_context_create`/`stageRotBegin` (sized by `towers*n`,
which is large at n=131072 even for a single instance, and several may
be transiently live at once during a stage transition), the per-
diagonal plaintext arrays (`dg[]`, `S=n/2=65536` complex doubles each,
times up to 16 diagonals held simultaneously per stage), or something
else entirely. **Do not assume it is KROT/KSC again — that hypothesis
is now exhausted; further guessing at cache constants is not a sound
next step.**

**Correct next step for whoever picks up Objective C**: add fine-grained
`[mem]` instrumentation INSIDE the C2S stage loop (per-diagonal, not
just per-stage — the existing checkpoints only print at stage
boundaries, which never even reached once in three attempts) to
localize exactly which allocation grows, before touching any cache
constant again. `tests/test_boot_n131072.cpp` is committed as-is (at
its last, most conservative `KROT_CAP=2`/`KSC_CAP=1` setting) as a
ready starting point for that diagnosis, not as a working configuration
— it has not yet completed a single bootstrap stage at n=131072.
**Objective C (full n=131072 end-to-end run) is NOT complete.** What
IS done and real: the security target is independently reconfirmed
(134 bits, see `SECURITY_ASSESSMENT.md`), and real (not extrapolated)
per-key cost at this scale is measured (1085MB RSS, 7.5s keygen —
`tests/test_n131072_keysize.cpp`).
