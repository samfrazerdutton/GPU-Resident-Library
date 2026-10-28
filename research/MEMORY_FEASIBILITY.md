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

## UPDATE: per-diagonal instrumentation added, ONE real data point obtained before a 4th OOM

Added per-diagonal `std::cerr` (unbuffered — unlike the per-stage
`std::cout` checkpoints, guaranteed to survive a sudden kill) `[mem]`
probes inside the C2S and S2C diagonal loops. A 4th attempt (same
`KROT_CAP=2`/`KSC_CAP=1`) got MUCH further than the prior three before
dying — it printed `ModRaise` and the analytic-inverse self-check
lines (confirming setup completes in reasonable time, NOT itself a
multi-minute bottleneck) and reached the very first C2S diagonal:

```
ModRaise: congruence mismatches = 0 (want 0), max|I| = 11
analytic inverse self-check: diag err=0 offdiag=2.45581e-17
      [mem] stage 0 diag 0/16 RSS=3506.18MB KROT=1 KSC=1
```

Then it OOM'd before printing diagonal 1. **This is still not fully
explained**: baseline RSS with only ONE `KROT` entry + ONE `KSC` entry
live (~1085MB each) plus `KREL` (the relin key, same ~1085MB magnitude,
generated once at the very start, outside any bounded cache) sums to
~3.25GB — close to but not exactly the observed 3506MB (plausibly base
keypairs + working buffers account for the ~250MB gap, not concerning
on its own). **What IS concerning**: it died within the next few
diagonals, i.e. after growing by at most 1-2 more KROT/KSC entries
(~1-2GB more, which the 10GB ceiling should comfortably absorb from a
3.5GB baseline with ~6.5GB headroom) — a MUCH faster climb to the
ceiling than "+1085MB per new distinct key" predicts. Two live
hypotheses, NEITHER yet confirmed: (a) `evalkeygen_host_sold`'s own
per-part working buffers (`uBuf`/`e0Buf`/`e1Buf`, each `sizeQlP*n*8`
bytes ≈ 33.5MB at this scale, ×3, allocated/freed per PART inside a
15-part loop) have some transient peak larger than the steady-state
~1085MB this doc has been modeling, especially under aggressive
cap-driven eviction+regeneration churn; (b) glibc's allocator does not
promptly return freed medium-sized blocks to the OS under this
allocation pattern, so apparent "freed" memory (an evicted cache entry,
or `stageDiags`' S=65536-iteration temporary-vector churn — see the
`stageDiags`/`applyMerged` O(S²) cost noted for a future performance
pass, independent of this memory question) keeps inflating RSS well
past what's actually live. **Next concrete step**: run under
`/usr/bin/time -v` or a proper allocator-tracing tool (not just
`getrusage` RSS, which conflates "ever touched" peak with "currently
live") to distinguish these, and add a `[mem]` probe immediately AFTER
`keyAt`'s/`rotFull`'s eviction point specifically, not just at the end
of each diagonal, to see whether the spike happens AT eviction/regen
or elsewhere in the diagonal's own processing.

**Separate, independent observation from this same run (a performance
concern, not a memory one)**: this attempt spent roughly 5-10 CPU-
minutes between process start and the first C2S diagonal printing
(stable 1.46GB RSS, climbing CPU time, consistent with a CPU-bound not
memory-bound phase) — `tests/test_boot.cpp`'s `stageDiags`/
`stageDiagsFwd` (lines ~268-277 and ~675-684) compute each stage's
diagonal structure via a loop of `S` iterations (`S=n/2`), each doing
an O(S) `applyMerged` call AND an O(S) zero-init of a fresh `S`-sized
`std::vector<cd>` AND an O(S) scan — i.e. O(S²) total per `stageDiags`
call, called once per C2S/S2C stage. At n=8192 (S=4096) this is
~16.8M-scale work per call, imperceptible; at n=131072 (S=65536) it's
~4.3B-scale, a 256x increase — plausibly several seconds to low
minutes per call on a single CPU thread with no apparent
parallelization, and called 8 times total (4 C2S + 4 S2C stages at
`LST=4`). This was NOT precisely isolated from `ModRaise`'s own O(n)
setup cost in this run (no per-phase timestamp exists yet), but is
flagged here as a real, independent algorithmic scaling concern worth
fixing (e.g. precomputing diagonal structure analytically instead of
probing empirically) for ANY future large-n attempt, regardless of how
the memory question above resolves.

## UPDATE (new round): KROT/KSC cache mechanism DEFINITIVELY CLEARED;
## true root cause found to be the WINDOWS HOST's total physical RAM,
## not WSL2's self-reported 10GB ceiling

This round answered the two live hypotheses the previous update left
open (transient `cap+1` keygen-before-evict peak vs. allocator
retention across repeated evictions) with a direct, isolated
experiment, then went further and found the actual environmental root
cause, which had been mis-framed all session.

### Experiment 1: `tests/test_n131072_cache_churn.cpp` (new, committed)

Isolates the KROT+KSC bounded-LRU cache mechanism from the rest of the
bootstrap entirely (no ModRaise, no ciphertexts, no `stageDiags`) and
drives it through 24 distinct rotation-key generations at the real
n=131072 scale, with `KROT_CAP=2`/`KSC_CAP=1` (identical to
`tests/test_boot_n131072.cpp`), forcing 22 real KROT evictions and 11
real KSC evictions. Result, exactly reproduced across every call after
the third:

```
RSS plateaus at EXACTLY 3233MB from call #3 onward, through all 22
KROT evictions and 11 KSC evictions. GPU used stays flat at 1264MB
throughout. Zero growth with call index.
```

**This conclusively confirms hypothesis (a) (the real `rotFull()`/
`keyAt()` code in `tests/test_boot_n131072.cpp` generates the NEW key
via `evalkeygen_host_sold` BEFORE checking/evicting the cap — so the
true transient peak is `(cap+1)` keys resident, not `cap` keys; 3233MB
÷ 3 ≈ 1077MB/key, matching the independently measured per-key cost
almost exactly) and DEFINITIVELY REFUTES hypothesis (b) (allocator
retention across repeated evict+regenerate cycles) — glibc correctly
reuses freed same-sized blocks here; RSS does not ratchet up with
eviction count.** The KROT/KSC cache mechanism — including the
previously-undocumented `cap+1` transient — is bounded, well-behaved,
and NOT the runaway cause of the repeated OOMs. This is new, direct
evidence, not a re-confirmation of the prior round's single data point.

### Experiment 2: enhanced instrumentation, 5th real n=131072 attempt

Added `gpuUsedMB()` (via `cudaMemGetInfo`, no sync needed) alongside
the existing RSS probes in `tests/test_boot_n131072.cpp`, plus a new
probe immediately after `stageDiags`/`stageDiagsFwd` returns (before
the diagonal loop starts) — the existing per-diagonal probes only see
state AFTER that call already ran, so its own cost and any RSS it
leaves behind were previously invisible. Result:

```
ModRaise: congruence mismatches = 0 (want 0), max|I| = 11
analytic inverse self-check: diag err=0 offdiag=2.45581e-17
    [mem] C2S stage 0 stageDiags done: t=268.047s diags=16 RSS=1436.15MB GPU=1254.69MB
```
then killed (host-memory pressure) before the next line — i.e. inside
the FIRST `keyAt(offs[0],30)`/`stageRotBegin`/`stageRotDiag` call of
stage 0, before even diagonal 0's own per-diagonal probe could print.

Two real findings here:
1. **`stageDiags` for ONE stage costs 268 seconds** — a real, measured
   (not estimated) confirmation of the O(S²) scaling concern flagged in
   the previous update. At `LST=4` (8 stages total, 4 C2S + 4 S2C),
   this alone could cost **~35 minutes of single-threaded CPU time**
   before any bootstrap compute happens, independent of whether the
   memory question is ever resolved. This is now a measured, documented
   cost, not a guess.
2. **RSS was only 1436MB right before the crash** — far below any
   previously-modeled danger threshold (even the old cap-based ~3.3GB
   estimate, let alone the 10GB nominal ceiling). Reading
   `stageRotBegin`/`ks_work_create`/`ks_context_create`'s actual
   allocation sizes (`src/keyswitch_resident.cpp`,
   `tests/test_boot_n131072.cpp` lines ~372-427) confirms their device
   (GPU VRAM) allocations are modest at this scale (~350MB combined,
   `ks_context_create`'s per-call av/bv upload another ~1GB, freed
   immediately after each call per the established "fresh
   cudaMalloc+upload, freed after" pattern) — nothing in the code this
   round read explains a sudden multi-GB host-RAM spike from a 1436MB
   baseline. The expected `keyAt()` growth (one fresh KROT entry +
   one atLevel KSC copy, ≈2.15GB) would only bring RSS to ≈3.6GB,
   matching the EARLIER (less-instrumented) attempt's "diag 0" RSS of
   3506MB almost exactly — still nowhere near 10GB.

### The actual root cause: this machine's physical RAM, not WSL2's internal accounting

Given neither the cache mechanism (Experiment 1) nor the stage-setup
code's own allocation sizes (Experiment 2's code reading) explain a
sudden OOM from a 1436MB-3.6GB baseline, the investigation moved to the
Windows host itself:

```
PS> Get-CimInstance Win32_OperatingSystem | Select TotalVisibleMemorySize,FreePhysicalMemory
TotalVisibleMemorySize: 16,168,544 KB (~15.42 GB)
FreePhysicalMemory:      3,792,048 KB (~3.62 GB)   <- measured with vmmemWSL near-idle (434.8MB)
```

```
C:\Users\samca\.wslconfig:
[wsl2]
memory=11GB
swap=8GB
```

**This machine has only ~15.42GB of TOTAL physical RAM, and `vmmemWSL`
(the WSL2 VM) draws from that SAME shared pool as every other Windows
process — it is not a separate, independent 10-11GB allocation as this
entire session's prior analysis assumed.** `.wslconfig` explicitly caps
WSL2 at 11GB (matching the "~10GB visible to WSL2" figure used
throughout this doc), but even with the WSL2 VM nearly idle, Windows
itself already has only ~3.6GB free out of 15.42GB total — meaning
roughly 11.8GB is already committed to the Windows OS, background
services, and whatever else is running (this Claude Code session's own
Windows-side process included) BEFORE WSL2 grows into its 11GB
allowance at all. **WSL2's internal `free -h` (which has reported
"10Gi total, 10Gi free" throughout this whole session, including
immediately before every one of the five n=131072 attempts) reflects
memory Hyper-V's dynamic-memory mechanism has PROMISED the VM, not
memory the HOST can actually back under real contention.** When WSL2
tries to actually commit several more GB (generating large keys,
building ciphertext buffers), Windows' own memory manager has to find
that RAM from an already-tight ~3.6GB of real headroom — exactly the
kind of pressure that triggers an abrupt kill (or, in earlier attempts,
the "D-state disk-wait stalls" and "frequent OOM-kills, must retry"
pattern this whole session documented as standing environmental
friction, for ordinary n=8192 workloads too, not just n=131072).

**This reframes the entire feasibility question.** It is not that
n=131072's structural memory requirement (modeled at ~4.3-5GB including
KREL+KROT+KSC's bounded peak, now empirically tighter-bounded by
Experiment 1) exceeds a real 10GB WSL2 ceiling — it is that **this
specific machine's total physical RAM (~15.4GB) cannot reliably back
even the modeled ~4-5GB structural requirement once Windows' own
baseline usage and WSL2's configured-but-not-actually-available 11GB
allowance are both accounted for.** Lowering `KROT_CAP`/`KSC_CAP`
further (as attempts 1-4 did) was always fighting a host-level
constraint those caps cannot fix, which explains why progressively more
conservative caps never converged on a working configuration.

**Objective C (full n=131072 end-to-end run) is NOT complete, and this
round's evidence points to an ENVIRONMENT BLOCKER rather than a fixable
implementation bug**: this specific machine likely does not have enough
real, available physical RAM to run the n=131072 feasibility test to
completion, independent of further cache-tuning. Concrete next actions,
in order of how directly they test this diagnosis: (1) close other
Windows-side memory-consuming processes (browser, IDEs, etc.) and retry
with maximum real headroom, to see whether the SAME attempt survives
further when host contention is minimized — the cheapest, most direct
test of this round's diagnosis; (2) if it still fails, the machine's
~15.4GB total physical RAM is very likely a hard blocker for n=131072
on this hardware, and the honest path forward is either acquiring a
machine with more RAM, or redesigning the bootstrap to stream key
material rather than holding even 2-3 full rotation keys host-resident
at once (a bigger architectural change, not scoped here); (3) do NOT
re-attempt with further-lowered `KROT_CAP`/`KSC_CAP` values expecting a
different result — Experiment 1 already proved the cache mechanism
itself is not the limiting factor at the caps already in use.

### Caught in the act: an ordinary, previously-reliable `test_boot` (n=8192) run was itself OOM-killed minutes later, with WSL reporting "10Gi free" throughout

Immediately after writing up the diagnosis above, a routine post-change
regression check — plain `./test_boot` at n=8192, the SAME binary that
has completed successfully dozens of times this session, normally
peaking at ~8.3GB RSS with no history of failing — was itself
OOM-killed. `free -h` INSIDE WSL at that exact moment still reported
"10Gi total, 10Gi free." The Windows host, checked in the same moment:

```
PS> Get-CimInstance Win32_OperatingSystem | Select TotalVisibleMemorySize,FreePhysicalMemory
TotalGB: 15.42   FreeGB: 2.83

PS> Get-Process | Sort WorkingSet64 -Descending | Select -First 10
Memory Compression   1111.5 MB   <- Windows already compressing pages: real pressure
claude (x2)            963.5 MB   <- this session's own Windows-side processes
opera (x4)             1434.1 MB  <- browser, several tabs/processes
vmmemWSL                453.8 MB
MsMpEng                 320.1 MB
ChatGPT                 178.2 MB
```

This is a second, independent, live-caught data point for the SAME
mechanism this round diagnosed from the n=131072 crash: WSL2's internal
memory accounting ("10Gi free") is disconnected from whether the
Windows host can actually back that promise, and ordinary
Windows-side memory pressure from OTHER running applications (a
browser with multiple tabs, other app instances, routine background
services) is sufficient to OOM-kill even a modest, previously-reliable
n=8192 workload — not just the much larger n=131072 attempt. **This is
not something to fix in the codebase**: it is a statement about this
machine's current operating conditions. The most direct, lowest-cost
next action for anyone continuing this work is simply to close other
memory-heavy applications (the browser especially, at ~1.4GB across its
processes) before any further large-n attempt or even routine
regression run, and to treat intermittent OOM kills on ordinary
n=8192 workloads as a signal to check real host memory first, not as
evidence of a new code regression.

A retry of the identical `./test_boot` command a few minutes later was
ALSO OOM-killed, with Windows host free memory still only ~3.85GB at
that point — consistent with sustained, not momentary, host pressure
this specific session. No further retries were attempted (per this
session's own established practice: accept the data in hand rather than
loop indefinitely against a known environmental condition). This round's
code changes (`tests/test_boot_n131072.cpp`'s added instrumentation, the
new `tests/test_n131072_cache_churn.cpp`, `CMakeLists.txt`'s new target)
do not touch `tests/test_boot.cpp` at all, and that file has passed
consistently (`max slot err=0.000547871`, bit-exact) across dozens of
runs earlier in this same session under normal host conditions,
including as recently as the immediately prior commit (`f1978ab`) — the
two OOM kills here are classified as environmental/resource-related,
not a regression from this round's changes, on that basis.

## UPDATE (next round): precise kill-mechanism investigation + the diagonal bottleneck fixed + a 6th attempt reaches far further than ever before

Per a follow-up directive ("VERIFY OOM DIAGNOSIS AND REMOVE THE
LARGE-N DIAGONAL BOTTLENECK"), this round did NOT repeat the prior
investigation blindly — it (1) checked the actual kill mechanism
directly rather than inferring it from Windows-side free-memory numbers
alone, (2) fixed the independently-flagged `stageDiags`/`stageDiagsFwd`
O(S²) bottleneck with a proven-equivalent analytic replacement, and (3)
used that fix to re-attempt n=131072 under favorable host conditions,
getting dramatically further than any of the five prior attempts.

### Priority 2 result: what is PROVEN, what is HYPOTHESIS, what remains UNKNOWN about the kill mechanism

**Proven** (direct evidence, not inference):
- `dmesg` (covering this WSL2 VM's entire uptime, Oct 24 boot through
  the present, 1809+ lines) contains **zero** occurrences of `oom`,
  `out of memory`, or `killed process` — the standard Linux kernel
  OOM-killer's own log strings. `journalctl -k` returns no entries at
  all (kernel ring buffer is not persisted via systemd-journald in this
  setup; `dmesg` is the authoritative source and was checked directly).
  **The Linux kernel inside the WSL2 VM never invoked its own
  OOM-killer for any of these terminations.**
- WSL's own crash-capture mechanism (`WSL_ENABLE_CRASH_DUMP=1` in the
  kernel boot command line, confirmed via `dmesg`) DOES actively log
  genuine process crashes: a real example exists in this exact `dmesg`
  buffer — `bench_evalmult_resident`'s known, pre-existing,
  unrelated `std::runtime_error`/SIGABRT (signal 6) was captured with
  an explicit `WSL (... - CaptureCrash): Capturing crash for pid: ...,
  executable: ...bench_evalmult_resident, signal: 6` line. **No
  equivalent crash-capture entry exists for any `test_boot` or
  `test_boot_n131072` termination** — ruling out an uncaught C++
  exception (e.g. `std::bad_alloc` from a failed host allocation) or a
  signal-raising crash (e.g. `SIGSEGV`) as the mechanism for THOSE
  kills specifically; such an event would have produced a log entry
  exactly like the one that exists for the unrelated benchmark.
- `ru_maxrss` (used throughout this file's RSS figures) is documented
  (`man 2 getrusage`) as "the maximum resident set size used" — a
  monotonic high-water mark since process start, not the instantaneous
  RSS at the moment of the call. This means every printed RSS figure in
  this document is a valid upper bound on memory in use at that print
  point, never an understatement — the mystery growth this file
  describes is not an artifact of `ru_maxrss` hiding a larger transient
  spike that had already subsided; if anything, a transient spike
  elsewhere would make the printed number LARGER than the true
  instantaneous value, not smaller.

**Well-supported hypothesis, not directly provable from inside the WSL2
VM**: the terminations are most plausibly an external kill (most likely
SIGKILL, which cannot be caught or logged by the dying process, and is
not expected to appear in the crash-capture mechanism either, which
only captures catchable signals/exceptions) issued from OUTSIDE the
Linux VM — almost certainly the Claude Code tool harness's own
system-memory safety monitor on the Windows host (every kill this
session, across multiple distinct commands, produced the identical
notification wording "stopped because the system is running low on
memory," consistent with one centralized, Windows-side monitor rather
than varying organic in-VM events, which — per the dmesg evidence above
— did not occur). This is NOT something this session can verify with
certainty from inside WSL: the harness's exact trigger logic (whether
it watches Windows host free RAM, the WSL2 VM's own memory footprint,
or some combination) is not observable from here.

**Unknown**: the precise instant-by-instant memory trajectory between
the last successful `[mem]` print and the actual kill (the probes are
per-diagonal, not continuous) — so a genuine, very brief, large
transient spike in that specific window cannot be fully ruled out as a
contributing factor, even though the evidence above weighs against the
process's own steady-state memory need being the direct cause (see the
6th attempt below, where RSS was observed PLATEAUED, not climbing, for
seven consecutive diagonals before the kill).

### Priority 3 result: the `stageDiags`/`stageDiagsFwd` O(S²) bottleneck is FIXED, proven equivalent, and measured in production

Full detail, methodology, and the complete equivalence-proof data are in
`EXPERIMENT_LEDGER.md` entry #23; summary here. A new, independent,
self-contained test (`tests/test_c2s_diag_fast.cpp`) reproduces the
EXACT brute-force `applyMerged`/`stageDiags` (and `applyMergedFwd`/
`stageDiagsFwd`) algorithm as an oracle, implements an analytic
replacement (a symbolic dynamic-programming pass through the RST
butterfly stages — tracking, for every output index, the small list of
`(source_index, coefficient)` pairs it depends on, instead of probing
with `S` full unit basis vectors), and proves them **exactly equivalent
(within 1e-9) across 18 (n, LST, g, direction) combinations spanning
RST=1 through RST=13**, including RST=4 explicitly — the exact value
this file's own `LST=4` configuration uses at n=131072. Benchmarked at
n=2048/4096/8192/16384: speedups from 2.5x to **332x** depending on
RST (the production-relevant RST=4 case at n=8192 shows the 332x
figure). The old brute-force oracle is kept permanently in that
standalone test file (not duplicated inside any production test), per
the governing directive's "preserve the old implementation as a test
oracle" instruction.

**Wired directly into `tests/test_boot_n131072.cpp`** (same function
names/signatures for `stageDiags`/`stageDiagsFwd`, so no other code in
that file needed to change) — `tests/test_boot.cpp` (the n=8192 default
test) was deliberately left untouched, since O(S²) is never a
bottleneck at that scale. **Measured in the real production file on a
real run**: `stageDiags` for C2S stage 0 now takes **0.110205 seconds**
(was 268.047 seconds) — a ~2433x reduction for that single call, and by
extension an estimated ~35-CPU-minute total cost across all 8 C2S+S2C
stages reduced to well under one second.

### Priority 4 result: a 6th attempt, with favorable host memory, reaches diagonal 8 of 16 — far further than any of the five prior attempts, then is killed anyway with zero kernel trace

Checked real Windows host memory before launching (7.43GB free — a
materially better window than the ~2.8-3.9GB seen during the prior
round's attempts) and proceeded, consistent with the directive's
go/no-go gate. Result:

```
ModRaise: congruence mismatches = 0 (want 0), max|I| = 11
analytic inverse self-check: diag err=0 offdiag=2.45581e-17
    [mem] C2S stage 0 stageDiags done: t=0.110205s diags=16 RSS=1462.99MB GPU=1254.69MB
      [mem] stage 0 diag 0/16 RSS=3533.24MB GPU=1604.69MB KROT=1 KSC=1
      [mem] stage 0 diag 1/16 RSS=4523.34MB GPU=1732.69MB KROT=2 KSC=1
      [mem] stage 0 diag 2/16 RSS=4561.34MB GPU=1732.69MB KROT=2 KSC=1
      [mem] stage 0 diag 3/16 RSS=4561.35MB GPU=1732.69MB KROT=2 KSC=1
      [mem] stage 0 diag 4/16 RSS=4561.35MB GPU=1734.69MB KROT=2 KSC=1
      [mem] stage 0 diag 5/16 RSS=4561.35MB GPU=1734.69MB KROT=2 KSC=1
      [mem] stage 0 diag 6/16 RSS=4561.35MB GPU=1734.69MB KROT=2 KSC=1
      [mem] stage 0 diag 7/16 RSS=4561.35MB GPU=1734.69MB KROT=2 KSC=1
      [mem] stage 0 diag 8/16 RSS=4561.35MB GPU=1736.69MB KROT=2 KSC=1
(killed -- no further output)
```

Three real findings:
1. **The diagonal fix works in production, exactly as benchmarked
   standalone** — `stageDiags` cost collapsed from the previously
   measured 268s to 0.11s, confirmed on this live run, not just in the
   isolated test.
2. **The KROT/KSC cache mechanism is confirmed bounded in the REAL
   bootstrap context, not just the isolated synthetic experiment**:
   host RSS reached a steady 4561.35MB by diagonal 2 and stayed EXACTLY
   there (not even a KB of drift reported) through diagonal 8 — seven
   consecutive diagonals with zero RSS growth. This directly extends
   the prior round's isolated `test_n131072_cache_churn` finding into
   the actual bootstrap.
3. **A small, not-yet-explained residual GPU-memory climb exists and is
   NOT yet proven to plateau**: GPU used memory crept from 1732.69MB to
   1736.69MB over diagonals 5 through 8 (roughly +0.5-1MB/diagonal) --
   small in absolute terms, but observed over only 4 of the 128 total
   diagonals (16/stage x 8 stages) a complete run would need, so
   whether it continues, accelerates, or itself plateaus past diagonal
   8 is genuinely unknown, not just unmeasured.

Checked `dmesg` immediately after this kill too (current VM time vs.
dmesg's last entry, both confirmed): **again zero kernel-logged
events of any kind for this termination**, identical to the first
`test_boot` kill this round — the same external-kill signature,
reinforcing (not newly proving) the Priority 2 hypothesis above rather
than providing independent confirmation of a different mechanism.

**Honest verdict**: Objective C (n=131072 end-to-end) is still not
complete. But the diagnosis is now substantially sharper than before
this round: the diagonal bottleneck that was an independent, severe
problem is FIXED and verified; the KROT/KSC cache mechanism is now
TWICE confirmed bounded (isolated experiment, and now 8 real
diagonals in situ) rather than merely argued from code-reading; and the
termination itself still shows zero in-VM kernel evidence, consistent
with an external, host-level trigger rather than this specific
process's own memory need. The one genuinely open technical thread is
the small residual GPU-memory climb, observed over too few diagonals to
classify. Do not re-attempt the full n=131072 run repeatedly hoping for
a different result under similar host conditions (per the governing
directive); the next concrete, lower-cost step is to extend the
per-diagonal GPU-memory probe across MORE diagonals within a
cheaper, isolated harness (extending `tests/test_n131072_cache_churn.cpp`
to also exercise `ks_context_create`/`ks_work_create`/`stageRotBegin`'s
device allocations repeatedly, the one allocation path that isolated
test still does not touch) before attempting another full bootstrap run.

## UPDATE (next round): GPU-memory lifecycle isolated with real production APIs — a real, reproducible transition-correlated growth found; external-kill correlation directly demonstrated; one confirmed (tiny) leak fixed

Per the follow-up directive "ISOLATE GPU MEMORY GROWTH AND PROVE
LARGE-N FEASIBILITY." Built `tests/test_n131072_gpu_lifecycle.cpp`,
which drives the REAL production lifecycle — `ks_context_create`,
`keyswitch_resident`, `ks_context_destroy` per diagonal, `ks_work_create`/
`ks_work_destroy` per stage, the same bounded-LRU `KROT` rotation-key
cache as production, `stageRotBegin`/`stageRotDiag`/`stageRotEnd` with
identical shape — with none of the rest of the bootstrap (no ModRaise,
no ciphertext encode/decode, no `stageDiags`), to isolate exactly this
device-memory lifecycle from everything else.

### Run 1: 4 stages (tw=30,29,28,27) x 20 diagonals — 68 diagonals completed (far more than the 6th real attempt's 8) before an external kill

```
baseline: RSS=228MB GPU_used=1038.69MB
stage0 diag0:  RSS=3105MB GPU_used=1732.69MB
stage0 diag19: RSS=4193MB GPU_used=1742.69MB   (+10MB over 19 steps)
[stage0 end, after ks_work_destroy]:     GPU_used=1612.69MB   (-130MB)
stage1 diag0:  RSS=4193MB GPU_used=1952.69MB   (+340MB jump from stage0's end)
stage1 diag19: RSS=4193MB GPU_used=1962.69MB   (+10MB over 19 steps)
[stage1 end]:                            GPU_used=1832.69MB   (-130MB)
stage2 diag0:  RSS=4193MB GPU_used=2150.69MB   (+318MB jump)
stage2 diag19: RSS=4193MB GPU_used=2160.69MB   (+10MB)
[stage2 end]:                            GPU_used=2038.69MB   (-122MB)
stage3 diag0:  RSS=4193MB GPU_used=2356.69MB   (+318MB jump)
stage3 diag7:  RSS=4193MB GPU_used=2360.69MB   (killed shortly after)
```

Three distinct, precisely separable findings:

1. **Host RSS is PROVEN flat — more robustly than ever before.** Exactly
   4193MB from stage 0 diagonal 2 onward, through all 68 diagonals and 3
   full stage transitions, zero drift. This extends the prior round's
   8-diagonal in-situ observation and the isolated cache-churn
   experiment's 22-eviction observation to a much longer, harder test
   (68 diagonals, 4 distinct tower counts) with the identical result:
   the KROT/KSC mechanism and the host side of this lifecycle are
   genuinely bounded, not a contributing factor.
2. **A small, consistent, LINEAR within-stage GPU-memory drift: exactly
   +10MB over 19 diagonal-to-diagonal steps, identical in all three
   fully-observed stages** (≈0.53MB/diagonal). This reproduces, in a
   controlled setting, the +0.5-1MB/diagonal climb seen over only 4
   data points in the real 6th attempt — confirmed real, confirmed
   small, confirmed consistent in magnitude across stages.
3. **A much larger, NOT yet explained, consistent ~200MB NET growth per
   STAGE TRANSITION** (computed two ways, cross-checking: directly from
   consecutive stage-start values, 220MB/198MB/206MB; and as [jump into
   new stage] minus [[[drop at previous stage's end]], 340-130=210MB /
   318-130=188MB / 318-122=196MB — consistent to within measurement
   noise). This is NOT explained by:
   - `ks_work_create`/`ks_work_destroy`: these pair cleanly — the
     ~122-130MB drop at every stage's end matches `ks_work_create`'s own
     computed footprint (`maxPart*n*8 + 3*sizeQlP*n*8 + sizeP*n*8 +
     sizeQl*n*8` ≈ 134MB at this scale) almost exactly, confirming this
     specific create/destroy pair is NOT leaking.
   - Any of the per-stage-keyed host-side caches read directly in
     `src/keyswitch_resident.cpp` (`g_rtab`, keyed by `(n,q)` — moduli
     are stage-invariant, so this should not even grow after stage 0;
     `upc`, keyed by `(n,sizeQl,numPart,part,field)` — DOES get new
     entries each stage, but each entry is at most a few hundred bytes
     (`sizePart`/`sizeCompl`-sized, not `n`-sized), summing to roughly
     20KB of new entries per stage — three orders of magnitude too
     small).
   - `evalkeygen_host_sold`'s own per-part transient device buffers
     (`d_u`/`d_e0`/`d_e1`, `src/evalkeygen.cpp`): these are IDENTICALLY
     sized every single call (always built at full `sizeQlP=32`,
     regardless of the level the key will later be viewed at) and
     called on EVERY diagonal, not just at transitions — if they were
     the cause, growth would track diagonal count linearly throughout,
     not concentrate at transitions the way it actually does.

   **Best-supported hypothesis** (not independently confirmed — see
   tooling limitation below): CUDA's device memory allocator retains
   freed blocks in an internal pool rather than returning them to the
   device-wide free count `cudaMemGetInfo` reports, and because
   `ks_work_create`'s buffer sizes are tied to `sizeQl`/`sizeQlP`, which
   SHRINK every stage (30→29→28→27, never returning to a previously-seen
   size), freed blocks from an earlier, larger-sized stage may never be
   reused by a later, smaller-sized stage's allocation request —
   accumulating a permanently-retained (from this process's perspective)
   pool, one stage-sized increment at a time, without any single
   "missing cudaFree" in the application code. This is consistent with
   the magnitude (roughly `ks_work_create`'s own footprint, ~130-200MB)
   and the fact that it fires exactly at transitions, not per-diagonal.
   **Tooling limitation, checked directly**: `compute-sanitizer` is not
   installed in this WSL2 environment, and `nvidia-smi
   --query-compute-apps=pid,used_memory` returns no process rows at all
   (a known WSL2 CUDA-passthrough limitation on this setup) — there is
   no available tool in this environment to directly inspect CUDA's
   internal allocator state or attribute GPU memory to this process
   specifically beyond `cudaMemGetInfo`'s device-wide figure (which, with
   no other GPU workload running concurrently during these tests, is
   equivalent to this process's own usage by elimination, not by direct
   per-process measurement). **This hypothesis is well-supported, not
   proven.**

### Run 2: 8 stages (tw=30..23) x 3 diagonals — killed after only 2 diagonals, with an IDENTICAL internal trajectory to Run 1's first 2 diagonals

```
baseline: RSS=228MB GPU_used=1038.69MB
stage0 diag0: RSS=3105MB GPU_used=1732.69MB    <- bit-for-bit identical to Run 1
stage0 diag1: RSS=4065MB GPU_used=1732.69MB    <- bit-for-bit identical to Run 1
(killed)
```

Checked real Windows host free memory immediately before each run:
**5.41GB free after Run 1's kill; only 3.66GB free immediately before
launching Run 2** (unrelated to anything this test does — the gap
reflects other activity on this machine between the two launches). **The
SAME deterministic internal process state produced wildly different
survival times (68 diagonals vs. 2) purely as a function of EXTERNAL
host memory conditions at launch time, not this process's own
behavior.** This is direct, clean evidence — not inference from absence
of kernel log entries, as in the prior round, but a controlled
comparison of the same program run twice — supporting the
characterization that external host-level memory pressure, not this
specific workload's resource needs, determines when these kills happen.

### A real, tiny, confirmed leak found and fixed (not the explanation for the MB-scale growth above)

Direct reading of `ks_context_create`/`ks_context_destroy`
(`src/keyswitch_resident.cpp`) while building the harness above found:
`C.d_keyRow` is allocated fresh every call
(`cudaMalloc(&C.d_keyRow,C.sizeQlP*4)`, ~128 bytes at n=131072's
`sizeQlP=32`) but was never freed in `ks_context_destroy` — unlike
`d_av`/`d_bv`, which are correctly freed, and unlike the `g_rtab`/`upc`
caches, which are deliberately process-lifetime and correctly NOT freed
per-context. This is a genuine, confirmed per-call device-memory leak,
but at ~128 bytes/call it is three to four orders of magnitude too
small to explain the ~200MB/transition growth measured above — it would
take roughly 1.5 million calls to leak 200MB this way. **Fixed**
(`cudaFree(C.d_keyRow); C.d_keyRow=nullptr;` added to
`ks_context_destroy`) since it is real, in-scope, and zero-risk to
correctness (a pure resource-cleanup addition, no computation changed).
**Verified via the existing regression suite after the fix**:
`test_ks_resident` (bit-exact), `test_rot_resident` (bit-exact,
2.3x speedup intact), `test_hoist_single` (27/27) all pass identically;
`tests/test_c2s_diag_fast.cpp` (the diagonal-algorithm oracle, unrelated
to this fix) still passes. This is the only source-code change to
`src/keyswitch_resident.cpp` this round, made only after the leak was
directly confirmed by reading the code, not guessed.

### Go/no-go decision for the next full n=131072 attempt

Extrapolating the measured ~200MB/transition growth across a full
8-stage C2S+S2C run (7 transitions total after the first stage) would
add roughly 1.4GB beyond the ~1.7-2.4GB GPU usage already observed
within 4 stages, projecting to perhaps ~3-3.5GB GPU_used by the end of a
complete run — comfortably within the 6144MB VRAM card even under this
pessimistic linear extrapolation (not verified beyond 4 fully-observed
stages; the 8-stage/3-diagonal run that would have tested this directly
was itself killed too early by external pressure). Combined with host
RSS now proven flat at ~4.2GB across 68 diagonals, **GPU/host memory
growth intrinsic to this workload does not appear, on current evidence,
to be the binding constraint for a full run** — the binding constraint
remains external host memory availability at attempt time, exactly as
the prior round concluded, now with direct (not just inferential)
supporting evidence from the Run 1 vs. Run 2 comparison above.

**Decision: NO-GO for a 7th full bootstrap attempt right now.** Checked
real Windows host free memory immediately before writing this
section: **2.49GB free** — well below the ~7GB window that let the 6th
real attempt reach diagonal 8 and Run 1 above reach 68 diagonals.
**Explicit criteria for a justified future attempt**: real Windows host
free memory >= 6GB measured immediately before launch (matching the
two conditions under which this workload got furthest), via
`Get-CimInstance Win32_OperatingSystem | Select
FreePhysicalMemory` from PowerShell — not WSL2's internal `free -h`,
which has been shown repeatedly this session to misreport availability
under real contention. Do not launch a full attempt below that
threshold; do not retry repeatedly hoping conditions improve on their
own — check, and only proceed if they do.

**Honest summary of options if this machine's real host-memory headroom
remains the limiting factor going forward** (not implemented, stated
for completeness per the governing directive): (1) more physical RAM on
this machine — the direct fix; (2) a streaming, non-cached key-material
architecture that never holds more than one rotation key's worth of
host memory resident at a time (a substantial architectural change, not
attempted); (3) a smaller supported parameter set than n=131072 (would
need a fresh security re-derivation, not simply reusing n=131072's
existing 134-bit classical estimate). None of these are implemented;
stating them is not a claim that any has been built.
