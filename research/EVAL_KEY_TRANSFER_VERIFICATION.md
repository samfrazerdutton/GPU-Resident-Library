# Eval-key transfer verification and resident-context experiment

GOD PROMPT "CKKS PHASE 26", Mission 4. Independently verifies Phase 25's
static estimate of per-diagonal eval-key H2D traffic, and reports a
controlled, bit-exact A/B experiment for caching `DeviceKSContext` objects
across repeated use of the same rotation key.

## 1. The estimate being verified

`research/BOOTSTRAP_GPU_RESIDENCY_AUDIT.md` (Phase 25) derived, from reading
`src/evalkeygen.cpp`'s `K.av[part].resize((size_t)sizeQlP*n)` and
`src/keyswitch_resident.cpp`'s `C.d_av.push_back(upv(K.av[p], s))`, that one
`ks_context_create` call at the bootstrap's own parameters
(n=8192, sizeQ=30, sizeP=2 so sizeQlP=32, numPart=15) uploads approximately
60 MiB of eval key, fresh, every single call — regardless of whether that
exact key was already built and uploaded moments earlier.

That number was a static derivation from array dimensions, not a measurement.
This round measured it directly.

## 2. Independent measurement

New standalone benchmark `bench/bench_ks_context_resident_ab.cpp`. Before any
timing, it constructs a real `KeySwitchConstants` object at the bootstrap's
parameters (via the exact same `evalkeygen_host_sold` call path
`tests/test_hoist_multi.cpp` and `tests/test_boot.cpp` use) and sums the
ACTUAL `K.av[p].size()*8 + K.bv[p].size()*8` across all 15 parts of that live
object:

```
av+bv across 15 parts, sizeQlP=32, n=8192: 62914560 bytes = 60.000 MiB per ks_context_create call
```

**60.000 MiB — the static estimate confirmed exactly, from a live object.**
Also confirmed by direct code reading: `ks_context_destroy` frees only
`d_av`, `d_bv`, and `d_keyRow` — every other device array a `DeviceKSContext`
holds (root tables, per-part decompose constants, moddown constants) is
process-lifetime shared state cached by key (chain SHAPE, not the rotation
index), so `d_av`/`d_bv` are the ENTIRE marginal cost of rebuilding a context
for an already-used key.

## 3. The experiment

`ResidentContextCache`: a bounded LRU cache of `DeviceKSContext`, keyed on
**(k, tw)** — never `k` alone, per `research/EXPERIMENT_LEDGER.md` entry
#20's documented bug class (building/using a rotation key at the wrong tower
count). Lives entirely inside the benchmark file. `src/keyswitch_resident.cpp`
was not touched.

### Methodology

Same discipline as `bench/bench_hoist_ab.cpp` (this project's established A/B
practice): shared untimed setup, 2 warm-up iterations per arm, arm order
ALTERNATING every trial, median of 9 trials reported with min/max.

- **Arm A (current production behavior)**: for each of 48 simulated diagonal
  calls, `ks_context_create` → `ks_work_create` → accumulate → moddown ×2 →
  destroy both, EVERY call — exactly what `tests/test_boot.cpp`'s
  `stageRotDiag` does today, even when the same rotation key recurs (which it
  does: Phase 24's ledger measured KROT, the HOST-side key cache, growing to
  ~77 entries reused with zero evictions across a real bootstrap).
- **Arm B (resident cache)**: the same 48 calls draw from only 8 distinct
  keys (matching a bounded working set), with the device context looked up
  in the LRU cache — built once per key, reused on every subsequent hit.

### Correctness, checked before any timing

Each of the 48 calls' key-switched output (checksummed via XOR/multiply over
both result halves) is compared, call for call, between the two arms:

```
[correctness] arm B's per-call output must be BIT-EXACT vs arm A's, call for call
  0 / 48 diagonal calls differ -> [PASS bit-exact]
```

The benchmark refuses to report timings unless this passes.

### Results

```
[cache stats] hits=40 misses=8 (of 48 calls)
create-every-time : 2.061020 [1.939740..2.637216] s
resident cache     : 1.239994 [1.172898..1.310116] s
speedup (this NDIAG=48 / NKEYS=8 working set): 1.6621x

create-every-time: 48 uploads x 60.000 MiB = 2880.0 MiB H2D eval-key traffic
resident cache   : 8 uploads x 60.000 MiB = 480.0 MiB H2D eval-key traffic
reduction: 2400.0 MiB avoided (83.3% of the create-every-time traffic)
```

Peak host RSS 697 MB. VRAM: 1071 MiB consumed (6144-5073), of which ~480 MiB
is the 8 resident contexts' eval-key material — consistent with the measured
60 MiB/context figure.

**1.66x measured speedup on the key-switch-adjacent portion (accumulate +
moddown + context lifetime) alone, with an 83.3% measured reduction in
eval-key H2D traffic** — not an estimate; the byte count for each arm is
computed from the number of actual cache misses (`ks_context_create` calls)
times the measured 60 MiB/call figure.

## 4. Connection to the rotation-level measurement (Mission 3)

`tests/test_rot_resident`'s own phase breakdown (re-run fresh this round, see
`research/MISSION3_DEVICE_AUTOMORPHISM_RESIDENCY.md`) shows
`ks_context_create` + its matching `destroy` consuming 63% of
`rotate_ct_resident`'s own total time at these same parameters. This
benchmark's 1.66x figure and that 63% figure are two independent
measurements of the same underlying cost, from different angles (an isolated
accumulate/moddown loop here; the full rotation primitive there), and they
agree in direction and rough magnitude.

## 5. Explicit limitations

- **Single-threaded only.** The cache has no lock. Phase 25 found that
  unsynchronized shared device state corrupts ciphertexts under concurrency
  (`src/rotate.cpp`'s `scratch()` buffer, fixed with `thread_local`). This
  cache has NOT been tested under concurrent access and must not be assumed
  safe there — a production version serving a multi-threaded caller would
  need its own concurrency design and its own test, not an extrapolation
  from this single-threaded result.
- **NKEYS=8 exactly fits the cache cap** in this run (zero evictions,
  `misses=8=NKEYS`). A real bootstrap's KROT measured ~77 distinct entries
  reused — a production-sized cache (77 contexts × 60 MiB ≈ 4.6 GiB) would
  need its own cost/benefit study against the 6144 MiB card and against
  `KROT_CAP`/`KSC_CAP`'s existing eviction policy, not an assumption carried
  over from this smaller run.
- **Scope**: this measures the key-switch-adjacent portion only (context
  lifetime + accumulate + moddown), explicitly excluding the automorphism and
  the ciphertext transfers around it — the same scope discipline as
  `bench/bench_hoist_ab.cpp`. It is NOT a bootstrap or end-to-end figure.
- **No production change was made.** `src/keyswitch_resident.cpp` is
  untouched. This experiment is evidence for a future, explicitly-scoped
  integration decision (the residency audit's O1), not an integration itself.

## 6. Commands

```
cmake --build build2 --target bench_ks_context_resident_ab -j4
./build2/bench_ks_context_resident_ab            # defaults: n=8192 tw=30 keys=8 diag=48 trials=9
./build2/bench_ks_context_resident_ab 8192 30 16 64 9   # larger working set, if memory allows
```
