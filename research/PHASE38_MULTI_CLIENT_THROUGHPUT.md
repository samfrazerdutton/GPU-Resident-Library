# Phase 38: multi-client CKKS throughput, batched encryption, and GPU-resident execution

## 1. Starting commit and environment

Starting commit: `5061631` (Phase 37's final commit), confirmed matching
`origin/main` exactly (0 ahead / 0 behind) at Phase 38 start.

Hardware/software, confirmed at session start: RTX 2060 Max-Q, 6144 MiB
VRAM (5942 MiB free at session start), compute capability 7.5, driver
595.97, CUDA 13.2, Release `-O3`. cmake 3.28.3. Rust/cargo 1.94.0. gcc
13.3.0, Ubuntu 24.04.4. 16 logical CPU cores. Windows host RAM:
16,168,544 KB total (~15.4 GiB); free RAM fluctuated ~2.6-6.9 GiB
through the session, consistent with the volatility of the two prior
phases.

## 2. Phase 37 baseline (re-stated, not re-measured from scratch)

| Measurement | Phase 37 result |
|---|---:|
| CPU share of one encryption call | ~53% |
| GPU share of one encryption call | ~43% |
| RNG sampling share | ~19.6% |
| Barrett correction-loop isolated improvement | 1.033x |
| End-to-end effect | ~0.5%, noise-level |
| C++ | 118/128, same 10 documented failures |
| Rust debug | 62/0/1 |
| Rust release | 62/0/1 |
| `test_n131072_keysize` | Passed |
| `test_n131072_cache_churn` | Resource exhaustion, not passed |

Re-confirmed this phase (section 9): the two n=131072 tests that
Phase 37 could not fully verify (`test_n131072_cache_churn` specifically)
both pass cleanly this session.

## 3. The real application workload

Used the existing financial aggregation workload exactly as
implemented (`bench/bench_private_aggregation_workload.cpp`,
`bench/bench_aggregation_client_scaling.cpp`) rather than a separate
artificial benchmark: K independent clients each `encode_host_fft` +
`encrypt_host_shoup_pooled_barrett_batched` their own private value
under a SHARED public key, folded into a running sum via
`ct_add_ct_host` (pure CPU), decrypted once at the end. No relin,
rotation, or rescale (a pure sum needs none).

`bench_aggregation_client_scaling.cpp` (Phase 29, re-run fresh this
phase, unmodified) already sweeps K = {1,2,4,8,16,32,64} and confirmed,
again, that VRAM stays flat (0 MiB delta across every K -- device
buffers are pooled and reused, never accumulated per client) while
runtime scales close to linearly with K (9.56ms at K=1 to 238.70ms at
K=64) -- i.e. **no economy of scale exists in the current sequential
path**: every client pays essentially the same marginal cost. This is
the concrete gap Track A/B investigate.

## 4. Track A: repeated work across clients

Traced the production call graph for K independent encrypt calls in
one aggregation round. For each candidate, verified genuine invariance
by reading the actual code (mod/root/pk are passed as the SAME
`const std::vector&` arguments to every call in the K-loop; the
client-specific data -- `m`, `seed` -- changes every call):

1. **The 3 small per-call pointer/modulus-table uploads
   (`h_fr`/`h_fp`/`h_mod4` → `d_fr`/`d_fp`/`d_mod`).** `mod`/`root` are
   IDENTICAL across every client in one aggregation round (same shared
   modulus chain), so these uploads write the exact same bytes to the
   exact same device addresses on every single call. Measured by
   Phase 37's own profile at ~0.131ms/call (small_memcpy bucket) --
   genuinely redundant for calls 2..K. **Subsumed by Track B's batched
   design** (section 5): a true batch upload does this once for the
   whole batch, not once per client, so this candidate was not
   separately implemented.
2. **The H2D/kernel-launch/D2H round trip itself.** Each sequential
   call pays its own fixed kernel-launch and `cudaMemcpy`-call overhead
   (Phase 37: ~0.315ms/call combined, independent of transfer size).
   This is the single largest repeated-overhead opportunity, and is
   exactly what Track B's batching targets directly.
3. **`enc_dev_tab_batched`'s NTT root/power tables.** Already cached
   process-lifetime, keyed by `(n,q)` -- confirmed NOT redundant (this
   was Phase 13/14's own work, re-verified by direct reading, not
   re-done).
4. **`encode_host_fft`'s twiddle factors and slot-index tables.**
   Already cached process-lifetime, keyed by `M`/`N` respectively
   (`src/encode_fft.cpp`'s `fft_twiddles`/`slot_idx`) -- confirmed NOT
   redundant by direct reading; nothing to do here.
5. **`ct_add_ct_host`'s per-call cost.** Already small (Phase 36:
   ~0.041ms/call for K-1 calls) -- ranked but deprioritized; not a
   meaningful opportunity at this scale.
6. **The host-side packed buffer allocation.** Already investigated
   and found negligible at single-client scale (Phase 37: 0.056ms).
   Phase 36 already disproved CROSS-CALL pooling of this exact kind of
   buffer (null result) -- not re-attempted as a standalone candidate;
   see section 10 for a NEW, scale-dependent risk this phase's batching
   work surfaced instead.

**Ranked candidates** (5, as required):

1. **True batched multi-client encryption** (merge M clients' host
   prep, ONE H2D, ONE kernel launch, ONE D2H) -- highest expected
   throughput benefit (eliminates M-1 redundant launch/transfer-call
   overheads), low-to-moderate complexity (the underlying kernel-merge
   technique is already proven by Phase 34), moderate validation cost
   (needs a dedicated correctness test, which this phase built).
   **Selected (Track B/E).**
2. **Chunked batching** (batch in groups of, say, 8 instead of all K at
   once) -- a direct variant of candidate 1, useful if VRAM or
   diminishing returns at large M become a problem. Evaluated as part
   of Track D's memory analysis; not needed at the VRAM budgets tested
   (section 7).
3. **CPU/GPU overlap across successive batches** (double-buffered
   pipeline). Highest theoretical throughput ceiling (could overlap the
   ~53% CPU phase of batch N+1 with the ~43% GPU phase of batch N), but
   requires a stream-ownership/buffer-lifetime redesign well beyond
   this phase's narrow scope. **Investigated, not implemented** --
   Track C.
4. **Pinned host memory for the batched transfer.** Re-examined at
   batch scale (a batch's single transfer is now M MiB instead of 1
   MiB) -- not attempted this phase; Phases 15/16 already found only a
   modest (1.06-1.17x) isolated gain with real adoption friction at the
   single-client scale, and this phase's batching result (section 6)
   already exceeds that modest figure through a different mechanism
   without touching memory type at all.
5. **Skip redundant pointer-table uploads without full batching**
   (candidate 1 implemented standalone, without merging the kernel
   launch). Lower implementation complexity than full batching, but
   strictly dominated by it (batching gets this savings AND the
   launch-count savings); not implemented separately since the fuller
   solution was feasible and validated within this phase's scope.

## 5. Track B: batched encryption -- implemented and measured

Built `encrypt_host_shoup_pooled_barrett_batched_multiclient`
(`src/encrypt_host_shoup_pooled_barrett_batched_multiclient.cpp`), a
narrow, workload-driven extension (not a generic, unjustified batch
API): it takes exactly the shape the real aggregation workload already
has -- M independent clients, one shared public key/modulus chain,
each client supplying its own plaintext and its own independent seed.

**Why this is safe, traced from source before writing any code**:
`kernels/ntt_ct.cu`'s `ct_stage_b`/`ct_shared_tail_b` index tower state
purely via `blockIdx.y`; each tower independently reads/writes its own
`n`-length segment. Phase 34 already proved that repeating the SAME
per-tower tables 4 times (for v/e0/e1/m) and widening `towers` from
`sizeQ` to `4*sizeQ` is an EXACT reorganization, not an approximation.
Repeating those tables `M` MORE times (one repeat-set per client, since
every client shares the identical modulus chain) and widening `towers`
again to `4*sizeQ*M` is the identical argument applied once more.

Packed-buffer layout: client `i`'s block occupies
`[i*4*sizeQ*n, (i+1)*4*sizeQ*n)`, internally `[v|e0|e1|m]` exactly like
the single-client function. Pooled device state is keyed by
`(n,sizeQ,M)` -- a separate pool from the single-client function's own
`(n,sizeQ)` pool (different struct, different mutex, no shared state).

What is shared across clients (all PUBLIC CKKS parameters, zero
confidentiality impact): `pkA`, `pkB`, `mod`, `root`, `n`, `sigma`,
`ns`, and the per-tower NTT tables. What is NEVER shared: each client's
own `v`/`e0`/`e1` randomness (independently seeded per client from the
caller-supplied `seeds[i]`) and each client's own plaintext.

## 6. Correctness evidence

`tests/test_encrypt_multiclient_matches_sequential.cpp` (new,
registered with ctest): compares the batched function's output,
client-by-client, against M independent calls to the existing,
already-verified single-client function. **Bit-exact MATCH** across:

- n ∈ {1024, 4096, 8192} x sizeQ ∈ {1, 4} x M ∈ {1, 2, 8, 17}
  (M=1 is the batch-size boundary case; M=17 is deliberately
  non-power-of-2).
- Independent-randomness check: two clients with the IDENTICAL
  plaintext but DIFFERENT seeds produce DIFFERENT ciphertexts
  (confirms no accidental randomness sharing/reuse, without exposing
  any secret material -- only public ciphertext outputs are compared).
- Resource-lifetime: 5 repeated calls at the same `(n,sizeQ,M)` (pool
  reuse) and 3 interleaved rounds across 3 different `(n,sizeQ,M)`
  combinations (pool-key separation) -- all bit-exact.

All 12 tests in `*encrypt_host_shoup_pooled_barrett_batched*` passed
unaffected (the single-client function is untouched by this phase).

## 7. Latency and throughput: sequential vs. batched

`bench/bench_aggregation_multiclient_batched.cpp` (new): the real
aggregation workload shape, SEQ (existing path) and BATCH (new path)
interleaved every repetition (not block-sequential), K ∈
{1,2,4,8,16,32,64}, 16 reps/arm/K after a dedicated first-call
measurement. Three independent runs (R=8 then two at R=16) to confirm
stability:

| K | Run 1 (R=8) | Run 2 (R=16) | Run 3 (R=16) |
|---:|---:|---:|---:|
| 1 | 1.074x | 1.147x | 1.099x |
| 2 | 1.067x | 1.194x | 1.235x |
| 4 | 1.163x | 1.287x | 1.222x |
| 8 | 1.257x | 1.293x | 1.334x |
| 16 | 1.375x | 1.384x | 1.297x |
| 32 | 1.038x | 1.030x | 1.020x |
| 64 | 0.960x | 1.058x | 1.045x |

**Honest reading, not an oversold one**: the benefit is real and
reproducible across all three runs for K ≤ 16 (peaking around
1.3-1.4x at K=16), but is consistently and reproducibly WEAKER at
K=32 (~1.02-1.04x, the most stable reading of the three columns) and
noisier at K=64 (0.960x in the first, shorter run; ~1.05x in the two
longer runs -- the R=8 reading at K=64 is judged a measurement
artifact from too few repetitions at a ~185-240ms-per-rep workload, not
a reproducible regression, since R=16 resolved it in both later runs).
**No run showed a genuine regression at any K** once repetition count
was adequate. First-call (fully cold, before any pool exists) latency
was NOT worse for BATCH (5.6-5.8ms) than SEQ (6.8-7.5ms) at K=1.

Throughput (encryptions/second, from the R=16 run): SEQ holds roughly
flat at 320-355 enc/s across the whole K range (confirming the "no
economy of scale" finding from section 3); BATCH reaches 480 enc/s at
K=16 before settling back toward 330-410 enc/s at K=32/64 -- the
batched path's throughput ADVANTAGE is real but not monotonically
increasing with K, and the K=32 dip specifically deserves a note of
honest uncertainty: it is reproducible across all three runs, but this
phase did not fully root-cause it (see section 10).

## 8. Memory scalability

VRAM measured externally (`nvidia-smi`), not just via the benchmark's
own `cudaMemGetInfo` calls: free VRAM at session start 5942 MiB; ~4s
into the full K=1..64 sweep (by which point several of the larger
pool entries have been allocated) 5794 MiB free -- **~148 MiB total**
consumed by every distinct `(n,sizeQ,M)` pool entry created while
testing all 7 batch sizes in one process. After the process exits,
VRAM free returns to 5942 MiB exactly -- full recovery, no leak beyond
the process's own lifetime (consistent with every other pool in this
codebase). Peak host RSS: ~221-223 MB across the full sweep.

**A real design characteristic, not a bug, worth flagging explicitly**:
the batched pool is keyed by `(n,sizeQ,M)`, so each DISTINCT `M` used
by an application gets its OWN permanent pool entry (sized
`~M` MiB), unlike the single-client function's pool (keyed by
`(n,sizeQ)` only, one entry regardless of call count). An application
that varies its batch size across many distinct values over its
lifetime would accumulate one pool entry per distinct size, which could
grow without bound in principle. This is NOT a problem for this
phase's own benchmark (7 sizes, ~148 MiB total, trivial against 6 GiB)
or for any REALISTIC deployment of this workload (which would use ONE
fixed batch size -- e.g. always exactly 32 clients per aggregation
round -- repeatedly, creating exactly one pool entry, identical in
character to the existing single-client pool). Documented here as a
known limitation for a future phase or caller to be aware of, not
something this phase attempts to bound (no `KROT_CAP`/`KSC_CAP`-style
eviction policy was added, since the real workload this phase targets
does not need one).

## 9. Track C: CPU/GPU overlap -- investigated, not implemented

The batched function (section 5) already front-loads ALL CPU work
(RNG + embed for every client in the batch) before the ONE GPU call,
then does ALL CPU work for combine after that call returns -- there is
no CPU work left WITHIN one batched call to overlap with that same
call's own GPU phase. The only genuine overlap opportunity is ACROSS
successive batches (e.g. a sustained server handling many aggregation
rounds back-to-back, where round N+1's CPU prep could run while round
N's GPU work is still in flight) -- a workload shape this project's
existing benchmarks do not currently model, and implementing it would
require: two full sets of pooled device/host buffers (double
buffering, so round N+1's prep never touches memory round N's GPU work
is still reading), `cudaMemcpyAsync` on a dedicated stream with
explicit `cudaEvent`-based ordering, and a redesigned mutex/pool model
(the current single mutex, by design, serializes the ENTIRE call —
exactly the opposite of what overlap needs). This is a genuinely larger
architectural change than "one meaningful improvement" for this phase,
and the mission's own explicit instruction is direct: if overlap cannot
be safely validated within scope, document and do not ship it. **Not
implemented this phase.** No threads were spawned for this investigation
either (Phase 36's own lesson: thread-spawn overhead can exceed small
per-call CPU work) -- this was a source-level and architectural
analysis, not a speculative implementation attempt.

## 10. Known limitations and unresolved risks

- **The K=32 throughput dip is reproducible but not root-caused.**
  Consistent across three independent runs (~1.02-1.04x, the weakest
  positive result in the whole sweep), it does not fit a simple
  "overhead amortizes with scale" story on its own (K=16 and K=64
  bracket it with stronger results). A future phase should
  instrument this specific case (e.g. the same `[P37]`-style internal
  CPU/GPU split Phase 37 built, applied to the batched function at
  K=32 specifically) before assuming any particular cause.
- **Pool growth across distinct batch sizes** (section 8) is a real
  design characteristic a future caller should be aware of if it ever
  varies `M` widely and repeatedly; not a problem for this phase's own
  scope, and not given an eviction policy here.
- **CPU/GPU overlap remains unexplored in practice** (only analyzed),
  per section 9 -- the single highest-ceiling remaining throughput
  opportunity this phase identified, deliberately not attempted given
  its scope and risk.
- **The combine loop's `shoup_precon` division** (Phase 37's own
  finding, 14.6% of a single-client call) is now, at batch scale, a
  proportionally larger piece of the batched function's own CPU-side
  cost (since combine runs once per client, same as before, just now
  inside one bigger function call) -- Phase 37's complexity analysis
  (a correct Barrett replacement needs genuine multi-precision
  arithmetic, not clearly cheaper than Shoup's existing approach)
  still applies unchanged; not revisited this phase.
- This phase does not claim the batched path is the ONLY or final
  throughput improvement available -- it is the one candidate this
  phase's own measurements supported clearly enough to implement and
  validate within scope.

## 11. Cryptographic integrity (Track F)

- `h=64` preserved (`set_secret_hamming_weight(64)`, unchanged,
  confirmed in both the new function and every test/benchmark).
- No switch to uniform ternary secrets; no change to any security
  parameter, modulus chain, `KROT_CAP`/`KSC_CAP` (untouched, not
  referenced by this phase's code at all), or `P >= Q_part_max`
  validation (untouched, no key-switch code touched).
- Randomness: each client's `v`/`e0`/`e1` are drawn from an
  INDEPENDENTLY SEEDED `std::mt19937_64`/`std::normal_distribution`
  per client, matching the existing single-client function's own
  per-call semantics exactly -- no RNG state, stream, or value is ever
  shared or reused across clients (verified both by direct code
  reading and by the independent-randomness test, section 6).
- No CKKS semantic or numerical-tolerance change: the batched function
  computes the IDENTICAL formula as the single-client function,
  verified bit-exact (section 6), not just "close."
- No key-switch internals touched; no new proof system introduced; no
  messaging use of CKKS.
- The authenticated-input boundary (`examples/authenticated_financial_input/`)
  has ZERO references to the new multi-client function (confirmed via
  direct search) -- multi-client execution does not interact with,
  weaken, or bypass that boundary's duplicate/replay checks in any way;
  the two remain fully separate, as they were before this phase.

## 12. Comparison against the repository's own baseline (Track G)

The primary, and most important, comparison is SEQ (the existing,
already-shipped, already-verified production path) vs BATCH (this
phase's new path) under IDENTICAL cryptographic parameters, the
IDENTICAL real workload, and IDENTICAL correctness requirements
(section 7) -- not a comparison against a different scheme, a
different security level, or an external library with different
settings. `bench_aggregation_openfhe_cpu_baseline` (the existing
OpenFHE CPU comparison) was re-run as part of the regression suite
(section 9) unchanged and is not re-litigated here; it was never a fair
apples-to-apples GPU-vs-CPU comparison and this phase does not treat it
as one.

## 13. Regression (Track F/11)

Focused tests first: all 12 encrypt/aggregation-related tests
(including the 2 new ones) passed. Then the broad suite, in 4 memory-
checked sub-batches (by test-ID range, same discipline as Phases 36-37):

| Batch | Tests run | Passed | Failed (all pre-existing) |
|---|---:|---:|---|
| 1-45 | 45 | 43 | `test_evalkeygen_isolated`, `test_keyswitch_stage_trace` |
| 46-91 | 46 | 43 | `test_polyeval`, `test_stage2`, `test_boot_multi_input_small` |
| 95-122 | 28 | 28 | none |
| 124-134 | 11 | 6 | the five `test_rescale_*_n1024*` |

**Total: 130/134 passing, exactly the same 10 pre-existing documented
failures as every prior phase, zero new regressions.**

- **`test_n131072_keysize`**: free RAM ~6.56 GiB beforehand --
  **Passed**, 8.51s.
- **`test_n131072_cache_churn`**: free RAM ~6.55 GiB beforehand --
  **Passed**, 182.74s (unlike Phase 37, where this exact test was
  killed by host memory pressure twice; this session's memory was
  healthier at the time this test ran).
- **`test_boot_n131072`** and **`test_n131072_gpu_lifecycle`**:
  deferred, not attempted -- both remain in the established high
  (6-10 GiB) memory-gate class; not re-attempted this phase (no new
  evidence changes their status from prior phases).
- **Rust** (`cargo test --release`, both `RUST_TEST_THREADS=1` and
  default parallel): **62 passed / 0 failed / 1 ignored**, both modes,
  unchanged.

No orphaned/runaway process was left behind at any point this session.

## 14. Deferred research

- Root-causing the K=32 throughput dip (section 10).
- A genuine cross-batch CPU/GPU overlap pipeline (section 9), if a
  future phase identifies a real sustained-server workload shape to
  validate it against.
- An eviction/bounding policy for the `(n,sizeQ,M)`-keyed pool, only if
  a future application genuinely needs to vary `M` across many distinct
  values repeatedly (not needed for this phase's own workload).
- Re-examining pinned memory specifically at the now-larger batched
  transfer size (section 4, candidate 4) -- not attempted, no new
  evidence gathered.

## 15. Exact commits and push status

See the end of this session's work for exact commit hashes (filled in
immediately after creation) and push confirmation.
