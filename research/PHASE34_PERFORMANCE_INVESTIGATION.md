# Phase 34: closing the deferred large-n gaps, then a verified performance improvement

Starting commit `f95eb74` (Phase 33 end), verified matching `origin/main`
exactly (0 ahead/0 behind) before any change. Hardware re-confirmed
unchanged: RTX 2060 Max-Q, 6144 MiB, driver 595.97, CUDA 13.2 (`nvcc`
13.2.51), cmake 3.28.3, Rust/cargo 1.94.0.

## 1. The two deferred n=131072 tests: both closed

Exact names, confirmed from `research/NEXT_SESSION.md` rather than
guessed: `test_n131072_keysize`, `test_n131072_cache_churn`.

**Correction to Phase 33's own characterization**: Phase 33 described
these as "the same memory-gate class as `test_boot_n131072`" (the 6 GiB
gate). That was imprecise. This project's own prior work already
measured each test's SPECIFIC peak RSS directly:
`research/MEMORY_FEASIBILITY.md` records `test_n131072_keysize`'s
measured RSS at **1313-1314 MB** (one real rotation key), and
`research/EXPERIMENT_LEDGER.md` entry #22 records `test_n131072_cache_churn`
plateauing at exactly **3233 MB** through 24 real keygens and 33 real
evictions — bounded, not unbounded, and well under the blanket 6 GiB
figure that applies to the full-bootstrap scenario (`test_boot_n131072`,
which chains ModRaise + ciphertext buffers + O(S²) stageDiags on top).
Both of these lighter, standalone diagnostics have their own, much
smaller, specifically-measured requirements.

**Results, using the correct thresholds**:

| Test | Measured requirement | Host free RAM at attempt | Result |
|---|---|---|---|
| `test_n131072_keysize` | ~1.3 GB | 2.58 GiB | **PASSED**, 9.27-9.28s, peak RSS 1314 MB (matches history exactly) |
| `test_n131072_cache_churn` | ~3.2 GB (bounded) | 5.08 GiB | **PASSED**, 189.20-196.97s, peak RSS 3232-3233 MB (matches history exactly, both the plateau value AND the mechanism finding: RSS stops growing after the 3rd call, confirming hypothesis (a) from entry #22 — keygen-before-evict transient — with zero additional allocator-retention growth) |

Both confirmed via `ctest -R <name> --output-on-failure` for the
official record, in addition to direct `/usr/bin/time -v` runs.
**Both deferred tests are now genuinely closed, not merely attempted.**

## 2. Regression validation

Re-ran in small batches throughout the session (the Windows host's free
RAM fluctuated 2.08-5.1 GiB — competing desktop load, not this phase's
own commands; direct single-invocation full-suite attempts were killed
twice by the harness's own low-memory guard before completing, same
pattern as Phase 33).

**Baseline, before any production change** (C++, excluding the 5
always-excluded full-bootstrap tests): **115/125 passing**, both
n=131072 tests now included (not excluded) and passing, the exact same
10 pre-existing documented failures as every prior phase
(`test_evalkeygen_isolated`, `test_keyswitch_stage_trace`,
`test_polyeval`, `test_stage2`, `test_boot_multi_input_small`, the five
`test_rescale_*_n1024*` family). Rust: 62 passed/0 failed/1 ignored,
both debug-serial and release-parallel.

**After the production change (section 5 below)**: re-ran the full
suite again — **115/125 passing, identical failure set**, both
n=131072 tests re-confirmed passing a second time. Rust: 62/0/1, both
modes, unchanged. **Zero regressions from Phase 33's own baseline or
from this round's own production change.**

## 3. Performance investigation methodology

Located the production-path benchmark: `bench/bench_aggregation_decomposed.cpp`
(Phase 29), confirmed via direct source reading to call
`encrypt_host_shoup_pooled_barrett_batched` — the exact function the
Rust FFI bridge calls (`rust/crates/ckks-cuda/cpp_bridge/encdec_bridge.cpp`'s
own comment confirms this directly).

**Methodology inspected for contamination before trusting any number**:
uses `std::chrono::steady_clock` (correct for wall-clock), measures each
stage with its own timer boundary, excludes keygen from per-aggregation
stats (amortized once), interleaves the OLD and PRODUCTION arms across
repetitions specifically to control for thermal/clock drift (this
machine's own documented ~30% run-to-run spread), reports the median
(robust to outliers) over R=10 reps. **One minor, stated limitation**:
no discarded warm-up repetition before the measured loop, so the very
first occurrence of each arm pays a one-time pool-creation cost baked
into rep 0 — a small, bounded effect on the mean, negligible on the
reported median (1 of 10 samples).

## 4. Fresh measurements (re-established, not assumed)

Ran `bench_aggregation_decomposed` twice independently before any
change:

| | Run 1 | Run 2 | Historical (Phase 32) |
|---|---|---|---|
| END-TO-END speedup (median) | 1.482x | 1.470x | 1.518x |
| encrypt stage as % of end-to-end | 86.3% | 86.4% | 86.9% |

**The historical bottleneck is confirmed still the current bottleneck**:
encrypt remains ~86% of end-to-end, essentially unchanged from Phase
29/32's own figures (within this machine's own run-to-run noise band).
Nothing had silently shifted.

## 5. Decomposing WHY encrypt dominates (fresh, not historical)

Phase 15's own 2021-era closing note said "within
`encrypt_host_shoup_pooled_barrett_batched`, the remaining components
are RNG+sampling... and NTT kernel launches... both small" — true in
relative terms at the time, but never quantified freshly for the actual
CURRENT batched variant. This round did, with three new, standalone,
non-invasive probes (zero production code touched for the measurement
itself — only `cudaEvent` brackets placed OUTSIDE the function call, or
pure-CUDA micro-benchmarks calling the real exported kernel entry
point):

- **`research/benchmarks/bench_encrypt_cpu_gpu_split.cpp`**: brackets
  the real production call with `cudaEvent`s on the default stream.
  Result: GPU-side (H2D+kernel+D2H) is **97.5-98.8%** of one call's
  wall-clock; CPU-side (RNG+Barrett-embed+Shoup-combine) is **1.2-2.5%**
  — genuinely tiny now, confirming Phases 13/14's embed/combine
  optimizations succeeded completely (CPU work that was once 37-45% of
  the OLD `encrypt_host`'s total is now ~1-3% of the NEW function's much
  smaller total).
- **`research/benchmarks/bench_transfer_only.cpp`**: isolates pure PCIe
  H2D+D2H for a buffer the EXACT size of the production packed buffer
  (1,048,576 bytes, pageable memory, matching production exactly).
  Result: **0.577 ms round-trip** — a real, measured floor, not an
  estimate.
- **`research/benchmarks/bench_ntt_merge_experiment.cpp`**: reading
  `kernels/ntt_ct.cu` directly revealed `LaunchNTT_CT_Batched` issues
  **~5 internal kernel launches per call** (4 "early stage" + 1 "shared
  tail", the CT-NTT's own log2(n)-stage structure, already correctly
  batched across the `towers` dimension via a 2D grid). The production
  function calls it **4 separate times** (once each for v/e0/e1/m) —
  **20 total kernel launches per encrypt call**, not 4 as a surface
  read of the "batched" name might suggest.

This decomposition (transfer ~0.58ms known-floor, plus kernel-launch
count at 20/call) is the genuinely fresh finding this round adds to the
historical record.

## 6. The experiment: merging 4 kernel-launch sequences into 1

**Hypothesis**: `v`, `e0`, `e1`, `m` are already laid out contiguously
in one packed device buffer (`[v|e0|e1|m]`, Phase 15's own design).
Reading `kernels/ntt_ct.cu`'s `ct_stage_b`/`ct_shared_tail_b` directly:
tower index is `blockIdx.y`; each tower independently indexes
`x+tw*n`/`roots[tw]`/`precon[tw]`/`mods[tw]`. Repeating the 4 real
per-tower tables 4 times and calling `LaunchNTT_CT_Batched` ONCE with
`towers=4*sizeQ` over the whole packed buffer is therefore an EXACT
reorganization of identical work (same per-element formula, same
per-tower tables, just a wider grid), not an approximation.

**Verified bit-exact BEFORE touching production**: the standalone probe
confirmed 0/131072 limbs differ between 4-separate-calls and
1-merged-call, across the real exported `LaunchNTT_CT_Batched` entry
point.

**Measurement discipline learned the hard way**: the FIRST isolated
measurement (non-interleaved, 30 reps of one arm then 30 of the other)
showed wildly inconsistent speedups across repeated runs of the SAME
comparison — 5.458x, then 2.302x, then 2.368x — tracked down to this
laptop GPU's clock ramp-up/down behavior over a tight loop of
millisecond-scale kernels (a known, already-documented machine
characteristic, here found to be severe enough to swing a >2x factor
for operations this short). **Rebuilt as a properly interleaved A/B**
(alternate arms every repetition, controlling for drift within one
process, the same discipline `bench_aggregation_decomposed` already
uses) — this gave a **stable, reproducible ~1.2x** on the full
transfer+kernel sequence across two independent runs: **1.240x, 1.206x**.
This non-interleaved-vs-interleaved discrepancy is itself a useful,
previously-undocumented methodology lesson for any future sub-millisecond
GPU micro-benchmark on this hardware.

## 7. Production change implemented

`src/encrypt_host_shoup_pooled_barrett_batched.cpp`: `EncryptBufPoolBatched`'s
`d_fr`/`d_fp`/`d_mod` are now `4*sizeQ`-length (repeating the real
`sizeQ` per-tower tables 4 times), built ONCE per `(n,sizeQ)` pool entry
— no new per-call host-side work. The hot path's 4 separate
`LaunchNTT_CT_Batched` calls (20 total kernel launches) are replaced by
1 call with `towers=4*sizeQ` (5 total kernel launches). No change to
the mutex, its scope, the pool's keying, the embed/combine arithmetic,
or any cryptographic parameter.

**Correctness, re-verified immediately after the change** (all 4
directly-relevant existing tests, before anything else was touched):

| Test | Result |
|---|---|
| `test_encrypt_host_shoup_pooled_barrett_batched_matches_reference` | PASS, every dimension incl. `sizeQ=1` edge case |
| `test_encrypt_host_shoup_pooled_barrett_batched_registered_matches_reference` | PASS, every dimension |
| `test_encrypt_host_shoup_pooled_barrett_batched_concurrency_audit` | PASS, 396/396 checks, 0 mismatches |
| `test_encrypt_host_shoup_pooled_barrett_batched_registered_concurrency_audit` | PASS, 436/436 checks, 0 mismatches |

The concurrency audits in particular confirm the mutex-serialization
safety model (Phase 13's own audit) is completely untouched by this
change — only the table shape and launch count inside the already-locked
section changed.

## 8. Real end-to-end re-measurement (not an extrapolation)

`bench_aggregation_decomposed`, 2 fresh runs, AFTER the change:

| | Before (section 4) | After | Change |
|---|---|---|---|
| encrypt stage median | 83.76-84.61 ms | 75.05-75.64 ms | **~1.11-1.13x faster** |
| END-TO-END median | 97.05-97.91 ms | 87.84-88.82 ms | **~1.09-1.12x faster** |
| speedup vs OLD plain `encrypt_host` (unaffected by this change) | 1.47-1.48x | 1.643-1.646x | grew, as expected |

`bench_private_aggregation_workload` (independent benchmark, same
workload): mean runtime **98.97 ms → 91.13 ms** (~1.086x), correctness
unchanged (`worst abs err` identical to before, `3.347e-09`).

**This is a real, measured, reproducible, bit-exact-correctness-preserving
improvement — not an extrapolation from the isolated kernel-only
measurement**, which (correctly) predicted a smaller full-call effect
than the raw kernel-only ratio, since the fixed transfer cost dilutes
it.

## 9. DeviceBuffer<T> (Part 6)

No new genuine second caller was found or sought this round — the
kernel-merge change touches only table shape and launch count inside
`EncryptBufPoolBatched`, a pool type with the same multi-pointer,
per-(n,sizeQ) shape Phase 33 already found unsuitable for `DeviceBuffer<T>`
without a redesign. Not reopened, per the explicit instruction not to
force a migration without a fitting caller.

## 10. Remaining limitations

- The ~1.2x kernel-launch-merge idea was the ONE clear, safe, evidence-backed
  experiment this round identified and implemented. Encode (CPU FFT,
  ~10.5ms/K-calls) and decrypt+decode (~1ms combined) remain untouched —
  encode in particular is now, in relative terms, a LARGER share of
  end-to-end (10.5/88.8 ≈ 12% after this change, up from 10.5/97≈11%
  before) simply because encrypt shrank; it was not investigated this
  round (out of scope — a CPU-only FFT stage, not a GPU residency
  question).
- The non-interleaved-measurement clock-drift lesson (section 6) was
  learned empirically this round; it is not yet captured as a reusable
  methodology note anywhere more general than this report.
- `test_rescale_*_n1024*`'s underlying `P>=Q_part_max` rejection and the
  other 7 pre-existing failures remain exactly as documented in every
  prior phase; not re-investigated here (no new evidence this round
  bears on them).
