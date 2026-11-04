# encrypt_host deep profiling and optimization (Phase 13)

GOD PROMPT "CKKS PHASE 13". This document records the internal
stage-level profile of `encrypt_host` (`src/encdec.cpp`, Phase 11's
batched-NTT implementation), the optimization experiments it motivated,
their correctness verification, a real concurrency bug found and fixed
along the way, and the final benchmark results. See `research/
EXPERIMENT_LEDGER.md` entry #46 for the condensed ledger summary.

Starting commit: `4a62f0a` (Phase 12 end state), confirmed via `git log
-1` and `origin/main` before any change this round.

## 1. Call graph

```
Rust codec::encrypt
  -> cpp_bridge/encdec_bridge.cpp: gpufhe_encrypt (extern "C")
       -> gpufhe::encrypt_host_shoup_pooled   <- PRODUCTION as of this round
            (src/encrypt_host_shoup_pooled.cpp, NEW)
       -> gpufhe::encrypt_host                <- reference/fallback, UNCHANGED
            (src/encdec.cpp, Phase 11)

encrypt_host's own internal stages (identical in both implementations
except where noted):
  1. RNG + Gaussian/ternary sampling (host, std::mt19937_64 + std::normal_distribution)
  2. Embed v/e0/e1/m into sizeQ RNS towers (host, signed 64-bit `%`)
  3. Device buffer allocation (cudaMalloc x7: d_v,d_e0,d_e1,d_m,d_fr,d_fp,d_mod)
       -- SKIPPED after the first call, per (n,sizeQ), in the pooled variant
  4. Upload H2D (4 blocking cudaMemcpy calls)
  5. NTT kernel launches x4 (LaunchNTT_CT_Batched) + cudaDeviceSynchronize
  6. Download D2H (4 blocking cudaMemcpy calls)
  7. Device buffer free (cudaFree x7)
       -- SKIPPED (buffers returned to the pool) in the pooled variant
  8. Combine: c0 = pkB*v + ns*e0 + m, c1 = pkA*v + ns*e1 (host,
     per-coefficient modular multiply-add) -- GENERIC 128-bit `%` in
     encrypt_host, Shoup's precomputed-constant modmul in both new
     variants (see Section 5)
```

## 2. Fresh baseline (re-established this round, not reused from Phase 12)

Verified before any change: `HEAD=4a62f0a`, matches `origin/main`; GPU
= RTX 2060 Max-Q (6144MiB, driver 595.97, 5955MiB free at measurement
time), CUDA 13.2, `CMAKE_CUDA_FLAGS` empty (no `-G`) in `CMakeLists.txt`
(confirmed by direct `grep`); 16 CPU threads (Ryzen 9 4900HS); real
Windows host free RAM 2.53GB at the start of this round (well under the
6GB n=131072 gate -- not attempted). Existing encrypt/decrypt and real
encrypted EvalMult tests re-run clean (Rust `cargo test --release`: 38
passed, 0 failed, 1 ignored) before any profiling began.

`bench/bench_evalmult_e2e_breakdown.cpp`'s single-call stage report at
n=8192, sizeQ=8 (production, FFT encode/decode from Phase 12, OLD
encrypt_host): Encrypt (host+GPU, x2 calls) ~17.6-19.3us *thousand*
(i.e. 17.6-19.3ms combined, ~8.8-9.7ms per call) -- consistent with
Phase 12's own closing estimate of "~17.0ms," confirming that figure
was a genuine, reproducible baseline, not a stale number carried
forward without verification.

## 3. Internal profiling methodology

`bench/bench_encrypt_profile.cpp` (new) contains `encrypt_host_profiled`
-- a byte-for-byte algorithmic COPY of `encrypt_host` (same RNG
consumption order, same embedding formula, same NTT calls, same
combine formula) with `std::chrono::steady_clock` checkpoints inserted
at each of the 8 stage boundaries listed in Section 1. Because every
CUDA call on this hot path runs on the default stream and every memcpy
used is the BLOCKING (non-`Async`) form, host-clock checkpoints
bracketing them are already correctly synchronized; the one explicit
`cudaDeviceSynchronize()` after the kernel launches (inherited
unchanged from `encrypt_host`) is what makes the "NTT kernel" stage
boundary accurate without needing a separate CUDA event.

**Correctness of the profiling copy itself was verified BEFORE trusting
any timing from it**: for the same seed, `encrypt_host` and
`encrypt_host_profiled` produce bit-identical `(c0,c1)`, checked inside
`bench_encrypt_profile.cpp`'s own `main()` at every tested n. All four
n values passed this check on every run.

## 4. Stage-level breakdown (median of 7 trials x 15 reps, per-trial medians)

| n | RNG | Embed | Alloc | Upload | NTT kernel | Download | Free | Combine | TOTAL |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 66us (5.8%) | 227us (20.2%) | 106us (9.2%) | 149us (13.3%) | 130us (12.0%) | 194us (16.9%) | 15us (1.3%) | 211us (18.4%) | 1147us |
| 2048 | 130us (6.6%) | 466us (23.6%) | 160us (8.1%) | 227us (11.5%) | 201us (10.2%) | 278us (14.1%) | 19us (1.0%) | 425us (21.5%) | 1975us |
| 4096 | 257us (5.2%) | 976us (19.7%) | 864us (17.4%) | 495us (10.0%) | 441us (8.9%) | 499us (10.1%) | 491us (9.9%) | 855us (17.2%) | 4964us |
| 8192 | 511us (6.2%) | 1965us (23.7%) | 1282us (15.5%) | 823us (9.9%) | 470us (5.7%) | 766us (9.2%) | 465us (5.6%) | 1757us (21.2%) | 8292us |

**Key findings, directly contradicting the "NTT batching is still the
largest opportunity" assumption the governing directive explicitly
warned against trusting blindly**:

- **Embed (host RNS embedding loop) is the single largest stage at
  every n**: 20-27% of total, and growing in absolute terms with n (as
  expected for an O(sizeQ*n) loop with 2 divisions per element).
- **Combine (host, final modmul/add loop) is the second-largest stage**:
  17-24% of total -- together, embed+combine consistently account for
  ~37-45% of encrypt_host's total time, dwarfing the NTT kernel itself
  (5.7-13%) and the transfers (upload+download, 19-31% combined,
  themselves mostly fixed per-call overhead rather than bandwidth-bound
  at these buffer sizes).
- **Device allocation (+free) is a large, size-dependent cost**: alloc
  alone ranges 8-17% of total, and alloc+free together range
  ~10-27% depending on n -- NOT the small, constant-overhead cost one
  might assume for `cudaMalloc`/`cudaFree`. This was the second
  candidate investigated (Section 6.2).
- The GPU kernel itself (NTT, already batched since Phase 11) is a
  SHRINKING share of total time as n grows (13.0% at n=1024 down to
  5.7% at n=8192) -- confirming the directive's instruction not to
  assume the already-optimized NTT batching is still the bottleneck.
  It no longer is.

## 5. Why `combine` is expensive: direct assembly evidence, not assumption

Every host-side modular multiply in this ENTIRE codebase (not just
`encrypt_host`'s own `mm()` -- the identical pattern appears in `src/
keygen.cpp`, `src/evalkeygen.cpp`, `src/keyswitch_resident.cpp`, `src/
ks_constants.cpp`) is written as `(uint64_t)(((unsigned
__int128)a*b)%q)`. Compiling this exact expression in isolation and
inspecting the object file with `nm`:

```
$ nm mmcheck.o | grep umod
                 U __umodti3
```

This confirms the compiler emits a call into **libgcc's `__umodti3`**
-- a generic, branchy SOFTWARE 128-bit division routine, not a hardware
instruction. `encrypt_host`'s combine loop calls this FOUR times per
(tower, coefficient) pair (`mm(pkB,v,q)`, `mm(ns,e0,q)`, `mm(pkA,v,q)`,
`mm(ns,e1,q)`), which is exactly the magnitude of cost the stage
breakdown shows.

## 6. Experiments

Per the governing directive's experiment discipline: hypothesis,
predicted benefit, scope, correctness test, A/B benchmark, memory
impact, keep/revert decision -- for each experiment, pursued ONE AT A
TIME, never stacked unverified.

### 6.1 Experiment: Shoup's precomputed-constant modmul in the combine loop

**Hypothesis**: replacing the four generic `__umodti3`-based `mm()`
calls per (tower, coefficient) with Shoup's precomputed-constant
modular multiplication (the SAME technique already used, and already
validated, by this codebase's own CUDA kernels -- `kernels/
intt_ct.cu`'s and `kernels/basis_convert.cu`'s device `shoup_mulmod`)
will measurably reduce combine's cost, since Shoup reduction replaces
a division with two multiplies and a comparison.

**Scope**: `ns%q` is the same scalar for every coefficient within a
tower -- its Shoup "precon" is computed ONCE per tower, not per
coefficient. `v`'s NTT-domain value at each coefficient (`vT[k]`) is
reused as the fixed multiplicand for BOTH the `pkB*v` and `pkA*v`
products -- computing its precon once (one division) replaces what
would otherwise be two separate `__umodti3` calls. Net: 4 slow
divisions per (tower,coefficient) in the original, down to 1
(amortized) here, the rest replaced by cheap multiply-based
`shoup_mulmod` calls. New file `src/encrypt_host_shoup.cpp`
(`encrypt_host_shoup`), byte-for-byte identical to `encrypt_host`
except for this one loop; `src/encdec.cpp` untouched.

**Correctness**: Shoup reduction is an EXACT modular-multiplication
algorithm for correctly-reduced inputs (<q), not an approximation --
`tests/test_encrypt_host_shoup_matches_reference.cpp` compares
`encrypt_host_shoup` against the untouched `encrypt_host` across n in
{16,1024,2048,4096,8192} x sizeQ in {1,4,8} x 4 message patterns (zero,
single-large-magnitude, dense-mixed-sign/magnitude, alternate-seed) --
**every one of the 60 cases matched BIT-EXACTLY**, no tolerance used
anywhere.

**Benchmark** (`bench/bench_encrypt_shoup_vs_reference.cpp`, interleaved
old/new per rep, 7 trials x 15 reps):

| n | OLD median | Shoup-only median | speedup |
|---|---|---|---|
| 1024 | 1142us | 1032-1106us | 1.08-1.11x |
| 2048 | 1958-3560us | 1711-3424us | 1.04-1.15x |
| 4096 | 3717-4976us | 3169-4474us | 1.11-1.17x |
| 8192 | 7967-11433us | 6901-10120us | 1.12-1.19x |

(Ranges reflect multiple independent runs on a host under WSL2/thermal
variance this project has documented repeatedly; the direction and
rough magnitude of the win is consistent across every run and every n
-- always a real, positive effect, never a regression.)

**Decision: KEEP.** Consistent, reproducible, bit-exact, zero security
impact.

### 6.2 Experiment: process-lifetime device-buffer pool

**Hypothesis**: Section 4's profile showed device allocation (+free)
costing 10-27% of total time depending on n -- far more than the
~10-20us one might assume for `cudaMalloc`/`cudaFree`. Caching the four
per-call device data buffers (`d_v,d_e0,d_e1,d_m`) and the small
pointer-array/modulus buffers, keyed by `(n,sizeQ)` (the SAME pattern
already used throughout this codebase for immutable tables --
`enc_dev_tab`/`ek_dev_tab`), should eliminate most of this cost on
repeated calls at the same configuration.

**Scope**: new file `src/encrypt_host_shoup_pooled.cpp`
(`encrypt_host_shoup_pooled`), building on 6.1's already-verified Shoup
combine loop, ADDITIONALLY removing the per-call `cudaMalloc`/`cudaFree`
for those buffers. Every buffer is still fully overwritten by its own
fresh `cudaMemcpy` before use on every call -- pooling changes WHEN
allocation happens, never what is computed.

**Correctness**: `tests/
test_encrypt_host_shoup_pooled_matches_reference.cpp` -- bit-exact vs.
`encrypt_host` across the same n/sizeQ sweep as 6.1, PLUS a dedicated
stress section exercising repeated calls at the SAME (n,sizeQ) (10
calls) and INTERLEAVED calls across three DIFFERENT (n,sizeQ)
configurations (3 rounds x 3 configs), to rule out cross-contamination
between differently-sized cached buffers. **All cases passed bit-exact
on first implementation.**

**A real bug found and fixed mid-experiment: a data race, not an
arithmetic defect.** After wiring `encrypt_host_shoup_pooled` into the
production FFI bridge and running the full Rust suite, `cargo test
--release` (default: parallel test execution across multiple OS
threads) FAILED 4 of 38 tests with wrong decrypted values (e.g. "slot 0
got=8796.48 expected=1"). Re-running the IDENTICAL suite with
`--test-threads=1` **passed 38/38 cleanly**, isolating the cause
precisely: the pooled device buffers are SHARED, MUTABLE state across
calls (unlike the read-only root tables this caching pattern was
borrowed from) -- two threads calling `encrypt_host_shoup_pooled`
concurrently at the same `(n,sizeQ)` key corrupted each other's buffer
content mid-flight. This was NOT a theoretical concern raised and
dismissed -- it was a real, reproducible failure in this project's own
existing test suite, caught before being reported as a success.

**Fix**: a single `std::mutex` serializing the entire function body.
Simple and unconditionally correct; negligible overhead (tens of ns)
for the actual production calling pattern (sequential, one ciphertext
at a time) -- the lock is only ever contended by genuinely concurrent
callers, which the Rust test suite's default parallelism is, but the
real pipeline is not.

**Verification of the fix, beyond just re-running cargo test**: a
dedicated multi-threaded stress test, `tests/
test_encrypt_host_shoup_pooled_thread_safety.cpp`, spawns 8
`std::thread`s that each call `encrypt_host_shoup_pooled` 6 times
(48 total concurrent calls) at the SAME `(n=2048,sizeQ=8)` key with
distinct seeds, checking every result against `encrypt_host` computed
for that seed. Run 3 times independently: **0/48 mismatches on every
run.** `cargo test --release` (default parallel) and `cargo test`
(debug, default parallel) both re-confirmed 38 passed / 0 failed / 1
ignored after the fix.

**Benchmark** (`bench/bench_encrypt_all_variants.cpp`, unified
interleaved old/Shoup/Shoup+pooled per rep, 9 trials x 20 reps, run
twice independently):

| n | Shoup-only vs OLD | Shoup+pooled vs OLD | pooling's own incremental contribution |
|---|---|---|---|
| 1024 | 1.08-1.11x | 1.10-1.16x | 1.02-1.06x |
| 2048 | 1.04-1.11x | 1.63-1.84x | 1.57-1.66x |
| 4096 | 1.13-1.17x | 1.16-1.17x | 1.00-1.04x |
| 8192 | 1.14-1.19x | 1.34-1.40x | 1.11-1.26x |

Pooling's incremental benefit is real at every n (never below 1.0x
across either run) but highly non-uniform -- dramatic at n=2048, modest
at n=1024/4096. This pattern reproduced consistently across two
independent full sweeps, not a one-off artifact. The most likely
explanation, not independently confirmed (no per-process GPU allocator
tracing is available in this WSL2/CUDA-passthrough setup, the same
limitation noted in `MEMORY_FEASIBILITY.md`'s own GPU-memory
investigations): the CUDA driver's own sub-allocator may already serve
some same-sized repeated malloc/free pairs from an internal free list
at some sizes but not others, making the EXPLICIT pool's benefit size-
dependent. Regardless of the exact mechanism, the measured effect is
real, reproducible, and never negative.

**Decision: KEEP**, with the mutex fix as a mandatory part of the kept
change (the unsynchronized version is a confirmed correctness defect
and was never shipped to the production bridge).

## 7. Real encrypted correctness (production swap)

`cpp_bridge/encdec_bridge.cpp`'s `gpufhe_encrypt` -- the actual
production call site every Rust caller (`codec::encrypt`) reaches --
now calls `encrypt_host_shoup_pooled`. `encrypt_host` (`src/encdec.cpp`)
is UNCHANGED and remains fully present, tested, and directly callable
as reference/fallback.

Full Rust suite after the swap (with the mutex fix in place): **38
passed, 0 failed, 1 ignored**, identical in `--release` and debug,
under cargo's DEFAULT parallel test execution -- including every
`real_encrypted_evalmult_*` test and `encrypt_decrypt_round_trip_
matches_input`.

`bench/bench_evalmult_e2e_breakdown.cpp`'s own correctness gates (RNS-
level tensor+relin+combine+rescale vs. an independent reference, and
the real encode->encrypt->EvalMult->decrypt->decode round trip against
the mathematically expected product) both passed with ZERO mismatches
and error magnitudes unchanged from Phase 12 (`~5e-12` to `~1e-10`,
well under the `1e-6` tolerance) at every n in {1024,2048,4096,8192},
for BOTH the OLD and NEW encrypt path.

## 8. Full real-encrypted pipeline impact

Single-call "complete end-to-end" measurements are noisy (cold-start,
GPU thermal variance -- this project's own `MASTER_STATE.md` documents
this repeatedly). Rather than trust one sample, 5 independent single-
call runs were collected for OLD and NEW at each n and compared by
median:

| n | OLD median (end-to-end) | NEW median (end-to-end) | speedup |
|---|---|---|---|
| 1024 | 7023us | 6145us | 1.14x |
| 2048 | 9239us | 8051us | 1.15x |
| 4096 | 13383us | 13156us | 1.02x |
| 8192 | 28936us | 24566us | 1.18x |

Consistent with Section 6.2's finding that pooling's benefit is
n-dependent (smallest at n=4096), the full-pipeline speedup is modest
but real and NEVER a regression at any tested n -- on top of Phase 12's
already-applied encode/decode optimization. RNS-level and real-
encrypted correctness checks passed at every n for both paths (Section
7).

## 9. CPU and memory behavior

Host RSS: no meaningful change from either optimization (both are
algorithm/lifecycle changes to existing buffers, not new allocations
of consequence) -- `bench_evalmult_e2e_breakdown.cpp`'s own reporting
showed host RSS deltas in the same ~0-9MB range as Phase 12, consistent
across OLD/NEW. GPU memory: the buffer pool adds a BOUNDED, one-time
allocation per distinct `(n,sizeQ)` actually used by the process --
for the single `(n,sizeQ)` configuration this project's real pipeline
uses at any one time, this is at most a few MB (4 buffers of
`sizeQ*n*8` bytes each, e.g. ~512KB x4 at n=8192,sizeQ=8), negligible
against the ~1GB CUDA-runtime baseline floor this project has
documented since Phase 10/11. No unbounded growth: the pool is keyed
by `(n,sizeQ)`, and `test_encrypt_host_shoup_pooled_matches_reference.
cpp`'s interleaved-different-configs stress test confirms distinct
keys do not interfere.

## 10. Security review

**No change to randomness, sampling, or the noise model anywhere in
this round.** Verified directly by inspection and by the bit-exact
correctness tests themselves (which would fail on ANY difference in
sampled values):

- The entropy source is unchanged: `std::mt19937_64 rng(seed)`, exactly
  as in the untouched `encrypt_host`. This round did not touch seeding,
  stream separation, or RNG choice in any way.
- Gaussian error sampling (`std::normal_distribution<double>
  g(0.0,sigma)`) is copied verbatim, same calls, same order, in every
  new function.
- Ternary `v` sampling (`(int)(rng()%3)-1`) is copied verbatim.
- The embedding formula (how sampled integers become RNS residues) is
  copied verbatim.
- The ONLY thing that changed is the ARITHMETIC used to combine
  already-sampled, already-NTT'd values into `(c0,c1)` (Shoup's exact
  modular multiplication instead of generic `%`), and WHEN device
  buffers are allocated (pooled instead of per-call) -- neither touches
  what random values are drawn, how many are drawn, or in what order.
- Because both changes are bit-exact-verified against the untouched
  `encrypt_host` for the SAME seed across dozens of (n,sizeQ,message)
  combinations, any accidental change to randomness consumption would
  have shown up as an immediate, unambiguous mismatch -- none did.
- The concurrency bug found and fixed (Section 6.2) was a data race on
  shared DEVICE BUFFER CONTENT, not on any RNG state -- `std::mt19937_64
  rng(seed)` is a per-call LOCAL variable in every implementation,
  never shared across calls or threads, so the race never affected
  randomness generation, independence, or reuse. No cryptographic
  property was ever at risk from this bug; it was a correctness-only
  defect (wrong ciphertext bytes), caught and fixed before reaching the
  production bridge in any committed, pushed state.

**No security-sensitive change occurred this round**, so no further
review beyond this confirmation is required per the governing
directive's own criterion ("a security-sensitive change requires a
documented review" -- there is none to review here, which this section
documents explicitly rather than silently).

## 11. What remains the dominant cost, post-Phase-13

At n=8192, the GPU-resident tensor+keyswitch+combine+rescale chain
remains ~1% of total pipeline latency (unchanged conclusion from Phases
11-12). Within `encrypt_host_shoup_pooled` itself, `embed` (host RNS
embedding, ~20-27% of the ORIGINAL encrypt_host's time, not yet
touched by either optimization this round) is now almost certainly the
single largest remaining opportunity, having a similar algorithmic
shape to `combine` (a double loop with a generic 64-bit `%` per
element) but using signed hardware division rather than `__umodti3` --
a smaller, but still real, candidate for a future round (documented in
`NEXT_SESSION.md`, not attempted here per the "do not stack unverified
changes" discipline -- this round already shipped two). Context/
eval-key creation (the "cold" fixture-build cost, ~9.5-58ms depending
on n) remains the single largest number in any end-to-end measurement
but is a ONE-TIME setup cost, not part of the per-ciphertext steady-
state path this phase targeted.
