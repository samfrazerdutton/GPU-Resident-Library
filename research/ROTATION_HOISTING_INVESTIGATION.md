# Rotation-hoisting investigation

GOD PROMPT "CKKS PHASE 25", Mission 4D: *"Rotation hoisting is currently
disabled by default. Find out why."*

## 0. The answer, up front

Rotation hoisting is disabled because **its benefit is entirely an amortisation
effect, and at one rotation per decompose it is a net loss.** Measured this
round on the key-switch portion at the bootstrap's own parameters:

| rotations sharing one decompose | speedup (upper bound) run 1 | run 2 |
|---|---|---|
| 1 | **0.59× (slower)** | **0.76× (slower)** |
| 2 | 1.18× | 1.18× |
| 3 | 1.43× | 1.43× |
| 4 | 1.69× | 1.57× |
| 6 | 1.90× | 1.89× |

Correctness was never the problem — it is bit-exact (0 of 2,949,120 limbs differ
across 6 keys, both runs). The problem is that Phase 23/24 measured hoisting on
the *full bootstrap*, where this ratio applies to only a slice of the runtime
and is swamped by host-side diagonal processing and PCIe traffic that hoisting
does not touch. That is why the earlier result read as "small and noisy" and the
performance acceptance criteria never cleanly passed.

**The recommendation is to keep it disabled by default** — and the reason is now
a measured curve rather than an unexplained null result.

## 1. What hoisting actually changes

From `include/keyswitch_hoist.h`: `ks_accumulate_qp` splits into

* **steps 1–3, "decompose"** — per-part INTT → `ApproxSwitchCRTBasis` → NTT.
  Depends only on the input ciphertext half `d_a` and the decompose *shape*
  constants, which are identical for every context built from the same
  `(n, sizeQl, numPart)` chain.
* **step 4, "accumulate"** — against one specific key's `av`/`bv`. This is the
  only part that differs per rotation index.

Hoisting computes steps 1–3 **once** into a `HoistedDigits` and reuses it across
N per-index accumulates, replacing N decomposes with 1 decompose + N accumulates.

The decompose is the expensive half (numPart-many INTT/basis-convert/NTT
triples; numPart=15 at the bootstrap's sizeQ=30). So the arithmetic is simply:

```
trusted:  N x (decompose + accumulate)
hoisted:  1 x decompose + N x accumulate
```

Hoisting can only win when the decompose it saves outweighs the extra
bookkeeping, which requires N ≥ 2 **by construction**. There was never a
mechanism by which it could help a lone rotation.

## 2. The measurement

New benchmark `bench/bench_hoist_ab.cpp`, run on the authoritative checkout.

### Configuration

| | |
|---|---|
| Hardware | NVIDIA GeForce RTX 2060 with Max-Q Design, 6144 MiB |
| Driver / CUDA | 595.97 / nvcc 13.2 (V13.2.51) |
| Build | `CMAKE_BUILD_TYPE=Release`, `-O3 -DNDEBUG`, **no `-G`** |
| CKKS params | n=8192, sizeQ=30, sizeP=2, numPart=15, h=64, σ=3.19, ns=1 |
| Rotation indices | successive powers of five mod 2n, last replaced by M−1 (conjugation) — the shape a real BSGS stage uses |
| Trials | 9 (run 1) and 21 (run 2), median reported with min..max |
| Warm-up | 2 untimed iterations of **each** arm before any sample |
| Order control | arm order **alternates** every trial (A,B then B,A) |
| Peak host RSS | **548 MB** |
| VRAM | 1469 MiB in use for 6 contexts + work buffers; 4675 of 6144 MiB free |
| Wall clock | 5.7 s for the 9-trial sweep |

### Exact commands

```
cmake --build build2 --target bench_hoist_ab -j4
./build2/bench_hoist_ab 8192 30 9 6     # run 1
./build2/bench_hoist_ab 8192 30 21 6    # run 2 (confirmation)
```

Machine-readable rows are appended to `research/benchmarks/hoist_ab.jsonl`.

### Scope — what is timed, and what is deliberately not

**Timed:** only the work that differs between arms.

* trusted arm: for each key, `ks_accumulate_qp` (decompose+accumulate) + `ks_moddown` ×2
* hoisted arm: `ks_decompose_only` **once**, then per key `ks_accumulate_from_digits` + `ks_moddown` ×2

**Not timed, and identical in both arms:** the per-rotation automorphism
(`automorphism_eval_fast_device`), all host↔device transfers, the host-side
combine, key generation, and context creation.

Therefore the ratios in §0 are an **upper bound on the achievable rotation
speedup, and are NOT a bootstrap speedup.** A real rotation also pays the
automorphism and the transfers, which hoisting does not accelerate; a real
bootstrap additionally spends heavily on host-side diagonal processing (see
`research/BOOTSTRAP_GPU_RESIDENCY_AUDIT.md` §3). Quoting 1.9× as a bootstrap
number would be precisely the error the governing directive forbids.

### Correctness, asserted before any timing is printed

The benchmark refuses to report timings unless the hoisted arm is bit-exact:

```
[correctness] hoisted arm must be BIT-EXACT against the trusted arm
  0 / 2949120 limbs differ across 6 keys -> [PASS bit-exact]
```

### Reproducibility

The curve is stable where it matters. Between two independent runs (9 and 21
trials): R=2 gave 1.1777× and 1.1784×; R=3 gave 1.4332× and 1.4254×; R=6 gave
1.8992× and 1.8902×. R=1 is the noisiest point (0.59× then 0.76×) but is
consistently **below 1.0** in both — the loss at N=1 is not a measurement
artifact.

Per-trial min..max spreads are wide (e.g. R=6 trusted ranged 0.029–0.115 s
against a 0.034 s median), which is consistent with this machine's documented
~30% run-to-run spread, WSL2 GPU clock throttling, and thermal drift. This is
exactly why the median-of-many-trials-with-alternating-order methodology is
used, and why single-pair comparisons on this machine are worthless.

## 3. Correctness evidence, independent of the benchmark

Three pre-existing hoisting tests were executed this round (all previously
blocked only by the assumption that they were heavy — they are not):

| test | result | peak RSS | wall |
|---|---|---|---|
| `test_hoist_single` | **[PASS] 27/27 cases** vs proven `rotate_ct_resident`, decrypted-plaintext comparison | 275 MB | 23.3 s |
| `test_hoist_multi` | **[PASS]** check 1 bit-exact (0/524,288 mismatches) split-vs-fused; check 2 **6/6 indices** off ONE shared decompose | 570 MB | 6.5 s |
| `test_hoist_levelreduced` | **[PASS]** level-reduced `ks_context_create`/`ks_decompose_only`, reducedTw=28 vs full sizeQ=30, `cuda err=no error` | 261 MB | 1.3 s |

`test_hoist_multi`'s check 2 reports `maxDiff` of 2253–3260 against
`maxAbs ≈ 5.26e15`, i.e. ~6e-13 relative — noise, not error.

So the hoisting mechanism is correct at single-rotation, multi-rotation and
level-reduced granularity, and the decompose-sharing claim (one decompose, many
indices) holds. **The feature is disabled for performance reasons alone.**

## 4. Why Phase 23/24's measurement looked inconclusive

Three distinct reasons, now separable:

1. **Dilution.** The full bootstrap applies this ratio to the key-switch slice
   only. Everything else — host-side diagonal processing, the ~60 MiB
   per-diagonal key re-upload documented in the residency audit, the per-diagonal
   `cudaStreamSynchronize` — is untouched by hoisting.
2. **A methodology artifact already on record.** `research/EXPERIMENT_LEDGER.md`
   entry #20 documents that `test_hoist_allstages`'s 0.93× was an artifact of
   its fair-comparison design: the hoisted side paid fresh keygen every diagonal
   while the trusted side's `KSC` view-cache made most of its diagonals cheap
   re-copies — conflating caching strategy with the decompose-sharing question.
   That entry already flagged the earlier N=5/N=16 results as the valid
   evidence. This round's N-sweep is the clean version of that experiment.
3. **Noise exceeding the effect.** On the full bootstrap the end-to-end
   difference is small relative to this machine's ~30% spread, so no feasible
   number of 110-second runs would resolve it. Measuring the mechanism directly
   sidesteps the problem entirely: 5.7 s per sweep instead of ~37 minutes.

## 5. What was NOT done, and why

* **The full-bootstrap A/B was not run.** `research/benchmarks/randomized_ab.sh`
  over `test_boot` vs `test_boot_hoist_enabled` needs ~8 GB host RSS × 20 runs.
  Windows free RAM measured 2.35–3.19 GiB all session;
  `research/benchmarks/bootstrap_preflight.ps1` returned NO-GO (logged). No
  end-to-end hoisting claim is made.
* **`test_hoist_allstages` was not run** (~8 GB peak RSS, same gate).
* **Hoisting was NOT enabled globally.** `HOIST_ROTATION_ENABLED` stays `false`
  in `tests/test_boot.cpp`. No production file was modified by this
  investigation.
* **No crossover point below N=2 was sought.** None can exist: at N=1 hoisting
  does strictly more work than the trusted path.

## 6. Recommendation

1. **Keep `HOIST_ROTATION_ENABLED = false`.** It is the right default: it is a
   loss for any isolated rotation, and its gain on a stage is bounded by the
   fraction of bootstrap time spent in key-switch accumulate.
2. **Enable it per-site, not globally, if at all.** The natural site is a BSGS
   stage loop, where ~16 diagonals share one input ciphertext half — squarely in
   the N≥4 regime where the measured upper bound is 1.6–1.9×. A feature flag
   scoped to the stage loop, not a global one, matches the measured shape of the
   benefit.
3. **Do the residency audit's O1 first.** Caching `DeviceKSContext` across
   diagonals removes ~60 MiB of H2D key traffic *per diagonal* — by static byte
   count about 6× the diagonal's entire ciphertext traffic. O1 and hoisting are
   orthogonal (O1 removes key *uploads*, hoisting removes redundant
   *decomposes*) and compose, but O1 is the larger static target and does not
   need a feature flag to be correct.
4. **Re-run the full A/B only once the host-memory gate clears**, and treat
   `bench/bench_hoist_ab.cpp` as the cheap pre-check: if the N-sweep ever stops
   showing a gain at the stage's real N, the end-to-end run cannot show one
   either.
