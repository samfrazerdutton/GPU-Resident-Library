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
- The bounded-cache redesign for KROT/KSC has not been implemented.
