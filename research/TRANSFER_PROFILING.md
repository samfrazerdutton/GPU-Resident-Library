# Batched CUDA transfer optimization for encrypt_host (Phase 15)

GOD PROMPT "CKKS PHASE 15". This document records the transfer data
flow this phase traced, two experiments built on top of it (one kept,
one measured and deliberately NOT integrated), and the resulting
correctness/benchmark evidence. See `research/EXPERIMENT_LEDGER.md`
entry #48 for the condensed ledger summary.

Starting commit: `7ea73fd` (Phase 14 end state), confirmed via `git
log -1` and `origin/main` before any change this round.

## 1. Existing transfer data flow (traced before changing anything)

`src/encrypt_host_shoup_pooled_barrett.cpp` (Phase 14's production
implementation at the start of this round):

```
Host (pageable std::vector<uint64_t>, 8-byte aligned)   Device (cudaMalloc, 256-byte aligned, POOLED by (n,sizeQ))
  vBuf[sizeQ*n]  --cudaMemcpy H2D-->  d_v[sizeQ*n]  --LaunchNTT_CT_Batched-->  (in place)
  e0Buf[sizeQ*n] --cudaMemcpy H2D-->  d_e0[sizeQ*n] --LaunchNTT_CT_Batched-->  (in place)
  e1Buf[sizeQ*n] --cudaMemcpy H2D-->  d_e1[sizeQ*n] --LaunchNTT_CT_Batched-->  (in place)
  mBuf[sizeQ*n]  --cudaMemcpy H2D-->  d_m[sizeQ*n]  --LaunchNTT_CT_Batched-->  (in place)
                                       cudaDeviceSynchronize()
  vBuf  <--cudaMemcpy D2H--  d_v
  e0Buf <--cudaMemcpy D2H--  d_e0
  e1Buf <--cudaMemcpy D2H--  d_e1
  mBuf  <--cudaMemcpy D2H--  d_m
```

**Exact properties, determined not assumed** (directive Section 4):
- Source/destination: `vBuf/e0Buf/e1Buf/mBuf` are LOCAL (call-scoped)
  `std::vector<uint64_t>`, size `sizeQ*n` each; `d_v/d_e0/d_e1/d_m`
  are SEPARATE `cudaMalloc` buffers, size `sizeQ*n*8` bytes each,
  POOLED (process-lifetime, keyed by `(n,sizeQ)`, same pattern as
  Phase 13/14's `EncryptBufPool`).
- Alignment: host side is 8-byte aligned (uint64_t's natural
  alignment, guaranteed by the default allocator); device side is
  256-byte aligned (CUDA's own guarantee for `cudaMalloc`). No
  alignment problem exists in the original code.
- Lifetimes: host buffers are call-scoped (destroyed at function
  return); device buffers are pooled (never freed during the
  process's lifetime, exactly like every other table/buffer cache in
  this codebase).
- CUDA stream: all operations use the default/null stream (`0`).
- Synchronous vs. asynchronous: every `cudaMemcpy` here is the
  BLOCKING form (no `Async` suffix) -- each call blocks the host
  until its own copy completes.
- Host memory type: PAGEABLE (plain `std::vector`, no
  `cudaHostAlloc`/`cudaMallocHost`). This matters: pageable-memory
  transfers are internally staged by the CUDA driver (the GPU's DMA
  engine can only read/write pinned memory), which both caps
  achievable bandwidth and adds an invisible extra host-side copy.
- Independent later use: v/e0/e1/m are each referenced independently
  in the final combine loop (v twice, e0/e1/m once each) -- but all
  four are the SAME size, use the SAME per-tower modulus/root tables,
  and are processed by IDENTICAL `LaunchNTT_CT_Batched` calls
  differing ONLY in which buffer pointer is passed. This is exactly
  the "same per-element size, same lifetime, same transfer pattern"
  condition that makes packing them together safe and low-risk.

**Call count**: 4 H2D + 4 D2H = 8 transfer calls per `encrypt_host`
call (plus 3 small calls for `d_fr`/`d_fp`/`d_mod`, unaffected by
either experiment below). **Total bytes**: `4*sizeQ*n*8` bytes each
direction (e.g. 2MB H2D + 2MB D2H at n=8192,sizeQ=8).

## 2. Experiment 1 (KEPT): batched transfers, pageable memory

**Hypothesis**: since all four sub-buffers share identical size,
lifetime, and transfer pattern, merging them into ONE contiguous
packed region -- one pooled device allocation of `4*sizeQ*n*8` bytes,
one local host buffer of the same layout -- would cut the transfer
CALL COUNT from 8 to 2 (one H2D, one D2H) while moving the exact same
total bytes, reducing per-call fixed overhead (driver dispatch,
synchronization bookkeeping) without touching kernel-launch logic at
all.

**Design** (directive Section 5, option B/E): `src/
encrypt_host_shoup_pooled_barrett_batched.cpp`'s
`encrypt_host_shoup_pooled_barrett_batched`. Host side: ONE local
(call-scoped, not pooled, not shared) `std::vector<uint64_t> packed`
of `4*sizeQ*n` elements, laid out `[v | e0 | e1 | m]` (each sub-region
starts already 8-byte aligned -- uint64_t's natural alignment needs no
padding). Device side: the pooled `EncryptBufPool` is replaced with
`EncryptBufPoolBatched`, holding ONE `d_packed` buffer of
`4*sizeQ*n*8` bytes instead of four separate buffers (same
`(n,sizeQ)`-keyed, process-lifetime pool pattern). The FOUR
`LaunchNTT_CT_Batched` calls are **completely unchanged** -- same
signature, same `d_fr`/`d_fp`/`d_mod` tables, same `sizeQ` towers --
they simply receive pointer OFFSETS into `d_packed`
(`d_packed`, `d_packed+sizeQ*n`, `d_packed+2*sizeQ*n`,
`d_packed+3*sizeQ*n`) instead of four separate buffers. This keeps
kernel-launch-level risk at zero.

**Concurrency** (directive Section 6): the packed DEVICE buffer is
pooled, shared, mutable state -- protected by the exact same single
`std::mutex` pattern Phase 13/14 already established and Phase 14's
audit (`research/EMBED_AND_POOL_HARDENING.md`) proved sound. The
packed HOST buffer is local/call-scoped, introducing no new
concurrency surface at all. Per the directive's explicit "add
concurrency tests for any new pooled or reusable resource"
instruction, a dedicated 4-scenario audit (`tests/
test_encrypt_host_shoup_pooled_barrett_batched_concurrency_audit.cpp`
-- same-key reuse, concurrent pool growth on fresh keys, 200-iteration
longevity, mixed-key race) was run 5 independent times: **0/396
mismatches every run.** (One bug caught in the TEST ITSELF: Scenario 2
initially used non-power-of-2 `n` values, which this codebase's NTT
requires to be a power of 2 -- caused an immediate segfault, fixed
before trusting any result, same class of mistake Phase 14's own audit
caught once already.)

**Correctness** (directive Section 7): `tests/
test_encrypt_host_shoup_pooled_barrett_batched_matches_reference.cpp`
-- bit-exact vs. BOTH the untouched `encrypt_host` AND Phase 14's
`encrypt_host_shoup_pooled_barrett`, across the full n/sizeQ/message
matrix used by every prior phase's test PLUS a dedicated
**packed-buffer-boundary stress case** (distinct extreme values placed
at the first/last tower of the message sub-buffer, to catch an
off-by-one in the `[v|e0|e1|m]` offsets) and the repeated-same-key/
interleaved-different-keys pool stress sections. **Every case: EXACT
MATCH** against both oracles.

**Benchmark** (`bench/bench_encrypt_phase15_all_variants.cpp`, unified
4-way interleaved old/Phase13/Phase14/Phase15, 9 trials x 20 reps, run
twice independently):

| n | Phase15(Exp1) vs OLD | Phase15(Exp1) vs Phase14 | Phase15(Exp1) vs Phase13 |
|---|---|---|---|
| 1024 | 1.57-1.63x | 1.07-1.17x | 1.36-1.40x |
| 2048 | 2.67-2.78x | 1.15-1.16x | 1.47-1.50x |
| 4096 | 1.66-1.69x | 1.09-1.13x | 1.40-1.45x |
| 8192 | 1.99-2.00x | 1.13-1.15x | 1.39-1.43x |

(n=2048's unusually large "vs OLD" ratio reproduced identically across
both runs -- consistent with the SAME n-dependent pattern Phase 13's
own buffer-pooling experiment showed, likely related to this GPU's
allocator sub-allocation behavior at that specific size; not
independently re-investigated this round, same limitation noted in
Phase 13's own report.)

Full real-encrypted pipeline (5-sample medians per n, since single-call
timing is cold-start-noisy): **1.14x-1.33x faster than OLD, 1.04x-1.05x
faster than Phase 14 alone**, on top of Phase 12's encode/decode gains.
Every RNS-level correctness check (0 mismatches) and real-round-trip
error magnitude (IDENTICAL across all variants at every n, `~5e-12` to
`~1e-10`, tolerance `1e-6`) passed before any timing was trusted.

**Decision: KEEP.** Real, reproducible, never a regression, zero added
concurrency risk beyond what was already proven safe.

## 3. Experiment 2 (MEASURED, NOT PROMOTED): pinned host staging memory

**Hypothesis**: pageable host memory requires the CUDA driver to stage
transfers internally; pinned (page-locked) memory should be faster and
achieve higher bandwidth for the SAME packed-buffer transfer
Experiment 1 already introduced.

**First estimate, per the directive's "estimate expected savings
before implementing"**: a standalone microbenchmark comparing
pageable vs. pinned H2D+D2H round-trip time at the EXACT packed-buffer
sizes this codebase uses (256KB/2MB/4MB, matching `4*sizeQ*n*8` at
n=1024..8192,sizeQ=8) showed a real **1.24x-1.42x** speedup from
pinned memory -- large enough to justify building it.

**Design**: `src/encrypt_host_shoup_pooled_barrett_batched_pinned.cpp`.
Builds directly on Experiment 1 -- the packed buffer is now ALSO
pinned on the host side (`cudaMallocHost`, pooled by `(n,sizeQ)`
alongside the device buffer, same mutex-serialized lifetime model).
The embed loop writes DIRECTLY into the pinned buffer (no intermediate
`std::vector`); the combine loop reads directly from it after
download -- zero extra host-side copying versus Experiment 1, only the
packed buffer's MEMORY TYPE changes.

**Concurrency**: re-audited with the SAME 4-scenario test design
(`tests/
test_encrypt_host_shoup_pooled_barrett_batched_pinned_concurrency_audit.cpp`),
5 independent runs, **0/396 mismatches every run.**

**Correctness**: `tests/
test_encrypt_host_shoup_pooled_barrett_batched_pinned_matches_reference.cpp`
-- bit-exact vs. both `encrypt_host` and Experiment 1, across the same
matrix as Experiment 1's test. **Every case: EXACT MATCH.**

**Isolated steady-state benchmark** (`bench/
bench_encrypt_phase15_pinned_vs_batched.cpp`, interleaved, BOTH
variants warmed up before any timed measurement, 9 trials x 20 reps,
run twice independently): pinned measured **1.06x-1.17x faster than
Experiment 1** at every n, both runs -- a real, reproducible
steady-state win.

### The real cost this round found: `cudaMallocHost` is NOT free, and the full-pipeline benchmark exposed it

When Experiment 2 was wired into `bench_evalmult_e2e_breakdown.cpp`'s
single-call "complete end-to-end" measurement (which does NOT warm up
`encrypt_host` before taking its first reading, unlike the isolated
benchmark above), it showed the OPPOSITE result: **SLOWER than Phase
14** at every n (e.g. n=1024: 8249-9447us vs. Phase 14's 5297-5437us).
This contradicted the isolated benchmark and was investigated directly
rather than dismissed or averaged away.

**Root cause, measured directly, not assumed**: a standalone
`cudaMallocHost` timing test, with CUDA context initialization paid
separately (so as not to conflate the two), found pinned host
allocation costs **~3-8ms** for the packed-buffer sizes this project
uses (256KB-4MB) -- roughly 6-15x more expensive than the equivalent
`cudaMalloc` (device) allocation, which costs only ~0.4-0.5ms for the
same sizes. Since this cost is paid ONCE per distinct `(n,sizeQ)` key
(the pool amortizes it across subsequent calls at that key), it is
invisible in a benchmark that warms up first -- but it is **larger
than encrypt_host's entire typical latency** (0.7-5ms across the
tested range), so a calling pattern that does only a handful of calls
at a given key before the process ends would see the first call's
`cudaMallocHost` cost dominate, making Experiment 2 a NET LOSS
overall despite its genuine steady-state improvement.

**This is exactly the scenario the governing directive's Section 8
explicitly warned about** ("Do not compare a cold new implementation
against a warm old implementation" -- and, implicitly, do not let a
warmed-up microbenchmark conceal a cold-start regression the full
pipeline would actually experience) **and Section 14 explicitly
anticipated** ("If pinned memory introduces more overhead than it
saves, revert it").

**Decision: NOT PROMOTED to production.** Experiment 2's steady-state
benefit is real and was not fabricated or dismissed -- it is recorded
here in full, with its own passing correctness and concurrency tests,
exactly as the directive requires for a failed/conditional experiment
("record failed experiments and their causes," not "silently discard
failed experiments"). The deciding factor is that its benefit is
CONDITIONAL on amortizing a large one-time cost across many repeated
calls at the same `(n,sizeQ)`, unlike every other optimization kept in
this project (Phases 11-15's other changes), which were all
unconditional improvements with no equivalent regression risk. Since
this project cannot assume every future caller of this library will
make enough repeated calls to amortize the cost, and the directive's
own priority order ranks "reproducible end-to-end performance" above
a performance win that depends on usage pattern, Experiment 1 (the
unconditional win) remains the production path. The source, tests,
and benchmarks for Experiment 2 are kept in the repository as a fully
documented, independently verified, honestly-reported experiment --
not deleted -- should a future round want to revisit it (e.g., with an
explicit pre-warming step at context-creation time, decoupling the
one-time cost from the first real encryption call).

## 4. Security review

**No change to randomness, sampling, or the noise model in either
experiment.** The RNG, Gaussian error sampling, and ternary `v`
sampling are copied verbatim into both new functions -- the ONLY
changes are (a) how the embed/NTT staging buffers are laid out and
transferred (packed vs. four separate buffers) and (b) in Experiment 2
only, which memory TYPE the packed host buffer uses (pinned vs.
pageable). Neither touches what random values are drawn, how many,
or in what order. Both changes were verified bit-exact against the
untouched `encrypt_host` across dozens of `(n,sizeQ,message)`
combinations including the packed-buffer-boundary stress case and
messages spanning the full `int64_t` range -- any accidental
divergence would have shown up as an immediate mismatch; none did.

## 5. What remains the dominant cost, post-Phase-15

Embed, combine, and now the transfer/allocation stages have all been
optimized (Phases 13-15). The GPU-resident tensor+keyswitch+combine+
rescale chain remains ~1% of total pipeline latency, unchanged since
Phase 11. Within `encrypt_host_shoup_pooled_barrett_batched` itself,
the remaining components are the RNG+sampling loop (a small, fixed
percentage per Phase 13's original profile) and the NTT kernel launches
themselves (already batched since Phase 11, confirmed not the
bottleneck). The clear next highest-value experiment, if performance
work continues, is revisiting Experiment 2 (pinned memory) with an
explicit pre-warming scheme -- allocating the pinned pool entry at
context-creation time (when a one-time cost is expected and acceptable)
rather than lazily on the first real encryption call, which would let
production capture the proven 1.06x-1.17x steady-state win without the
cold-start regression risk this round identified.

## Phase 16: explicit pinned-memory prewarming -- built, proven correct and safe, measured honestly, RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE (not promoted)

GOD PROMPT "CKKS PHASE 16". Starting commit `69d307a` (Phase 15 end
state), confirmed via `git log -1` and `origin/main` before any change.
This section directly follows up on Phase 15's closing recommendation
above.

### Verified starting state

Re-read the actual repository rather than trusting the prior report:
confirmed `cpp_bridge/encdec_bridge.cpp`'s `gpufhe_encrypt` calls
`encrypt_host_shoup_pooled_barrett_batched` (the pageable production
path, NOT the pinned experiment) -- Phase 15's report was accurate.
Re-read `src/encrypt_host_shoup_pooled_barrett_batched_pinned.cpp` in
full: its pinned buffer is allocated LAZILY on first use, keyed by
`(n,sizeQ)`, in a process-lifetime `std::map` protected by one
`std::mutex` that serializes the entire encrypt call; there is no
explicit cleanup/destroy anywhere in this codebase's pooling pattern
(every cache -- `enc_dev_tab`, `KROT`/`KSC`, every Phase 13-15 buffer
pool -- lives for the process lifetime, bounded only by the number of
distinct keys actually used).

### Lifecycle design (directive Step 2, decided BEFORE implementation)

```
1. Caller (anywhere -- no Rust-level "context" object touches
   encrypt_host today) optionally calls
   encrypt_prewarm_pinned_buffer(n, sizeQ).
2. That call allocates (or confirms already-allocated) the pinned
   host + device packed buffers for exactly this (n,sizeQ), checking
   EVERY CUDA allocation's return code (a deliberate, narrow exception
   to this codebase's otherwise-universal "don't check CUDA errors"
   convention, justified because this function's entire contract is
   about failure-safety). Returns true iff a FULLY valid buffer is
   now available.
3. encrypt_pinned_buffer_is_ready(n, sizeQ) lets a caller query
   status without allocating.
4. encrypt_host_shoup_pooled_barrett_prewarmed(...) is the adaptive
   encrypt call: it checks readiness FIRST. If ready, it uses the
   pinned path (identical arithmetic to Phase 15's pinned experiment).
   If NOT ready (prewarm never called, or it failed), it transparently
   defers to the UNMODIFIED, already-production
   encrypt_host_shoup_pooled_barrett_batched -- never attempting its
   own lazy pinned allocation. An application that never calls
   encrypt_prewarm_pinned_buffer pays NOTHING extra and behaves
   EXACTLY like today's production path -- ordinary use is not made
   unexpectedly expensive.
5. A failed prewarm (any one of the 5 allocation calls failing) rolls
   back whatever subset succeeded and marks the entry not-ready BEFORE
   it is ever inserted into the cache -- there is no code path that
   can leave encryption able to observe, let alone use, a half-
   initialized buffer.
6. No explicit "destroy" exists, by design, matching every other
   resource in this codebase (see "Verified starting state" above).
```

**Ownership model chosen**: a SHARED, PROCESS-LEVEL pool keyed by
`(n,sizeQ)` -- the SAME model every other cache in this codebase
already uses. This C++ layer has no per-context object for
`encrypt_host` to begin with (Rust's `EvalMultContext`/
`KeySwitchContext` are a separate crate layer that never touches
`encrypt_host`), so introducing a NEW per-context ownership model here
would be a bigger, less consistent, and riskier change than reusing
the architecture that already exists -- exactly what the directive's
"do not introduce a new global ownership model merely for
convenience" steers away from in the other direction (toward reusing
what exists). Multiple concurrent "contexts" at the same `(n,sizeQ)`
safely SHARE one prewarmed buffer, identically to how every earlier
phase's device pool is already shared and already proven safe.

### Correctness (directive Step 3/7)

`tests/test_encrypt_host_prewarm_matches_reference.cpp`: bit-exact vs.
the untouched `encrypt_host` across BOTH code paths this new function
can take -- (a) the fallback path (no prewarm ever requested, every
`(n,sizeQ)` x message pattern tested, confirmed each key starts
NOT-ready before the test exercises it) and (b) the prewarmed path
(explicit prewarm requested first, confirmed ready, then encrypted,
across the same n/sizeQ/message matrix PLUS idempotency -- prewarming
twice must not fail or reallocate -- and a 10-call pool-reuse stress).
**Every case: EXACT MATCH.**

### Concurrency and lifecycle (directive Step 4)

`tests/test_encrypt_host_prewarm_concurrency_audit.cpp`, run 5
independent times, **0/372 mismatches, 0 prewarm-consistency failures,
every run**:

1. Same-key contexts prewarming CONCURRENTLY (10 threads racing to be
   the one that allocates) -- all report the same final ready state.
2. Different-key contexts prewarming CONCURRENTLY (6 distinct fresh
   keys, concurrent map-mutation race) -- all become ready.
3. Multiple threads ENCRYPTING simultaneously at an already-prewarmed
   key -- the exact class of bug Phase 13's original, unsynchronized
   pool had.
4. Prewarm and encrypt INTERLEAVED on the SAME fresh key from
   different threads -- some encrypt calls race the prewarm and
   correctly take the fallback path, others see it ready; both paths
   verified bit-exact regardless of which one a given call took.
5. 200 sequential prewarmed-cycle reuses at one key (longevity).
6. A mixed shared-key race across 2 keys, 10 threads x 10 iterations,
   randomly prewarming and encrypting.

"Allocation failure and fallback" and "partial initialization
failure" were NOT forced via deliberate OOM -- consistent with this
codebase's established practice (Phase 14's audit noted the same
limitation), deliberately engineering an OOM condition on a shared
6GiB GPU under this session's own tight host-memory pressure was
judged a disproportionate risk to the session for a defect class no
other test in this 50+ test codebase has ever forced either. The
rollback path was instead verified by direct code reading: every
allocation step is gated on the previous one's success, and the
resulting entry is inserted into the cache only AFTER success/failure
is fully resolved -- `ready=true` and `ready=false` are the only two
observable states, both fully initialized or fully rolled back, never
partial.

### Workload-level benchmark (directive Step 3/5)

`bench/bench_encrypt_prewarm_workloads.cpp`, run twice independently,
n in {1024, 2048, 4096, 8192}, sizeQ=8, both steady-state series
properly warmed up before any timed measurement:

| n | prewarm cost (fresh key, 2 runs) | steady-state savings (run 1) | break-even ops |
|---|---|---|---|
| 1024 | 4555-4825us | 63us (~1.10x) | ~72 |
| 2048 | 16-16us | 122us (~1.12x) | ~0.13 |
| 4096 | 18-19us | 278us (~1.15x) | ~0.07 |
| 8192 | 5358-5897us | 422us (~1.12x) | ~13 |

**A real, reproducible, and unexpected finding**: the one-time
`cudaMallocHost` cost itself is NOT a simple function of buffer size --
n=1024 and n=8192 consistently cost several MILLISECONDS across two
independent runs, while n=2048 and n=4096 consistently cost under 20
MICROSECONDS, every time, in the same run order. This is almost
certainly related to this session's own persistent host memory
pressure (Windows free RAM measured at 1.2-2.4GB throughout this
round, well under comfortable headroom) interacting with WSL2's
dynamic-memory ballooning (documented extensively elsewhere in this
project, `research/MEMORY_FEASIBILITY.md`) -- a pinned-memory request
that requires the WSL2 VM to negotiate MORE memory from the Windows
host is plausibly much slower than one that fits within already-
committed slack, and which buffer sizes trigger that negotiation is
not something this round could directly instrument or confirm (no
per-process WSL2 VM memory-ballooning trace is available). This is
reported as an OBSERVED, REPRODUCIBLE pattern with a PLAUSIBLE but NOT
independently confirmed mechanism, consistent with this project's own
established practice of flagging "not independently confirmed" rather
than asserting a cause it cannot directly verify.

**This finding matters for the decision below**: it means prewarming's
own upfront cost is itself unpredictable in this environment, ranging
over two full orders of magnitude (16us to 5.9ms) for reasons tied to
host memory conditions, not to anything the caller controls. A
break-even calculation that can range from "essentially free" to
"~70 operations needed" depending on conditions outside the library's
control is not the kind of clean, unconditional result this project's
other kept optimizations (Phases 11-15) have delivered.

### Workload profile comparison (directive Step 5)

Using the directly-measured quantities above, combined analytically
(total time = prewarm_cost + N x steady_state_us, vs. N x pageable_us):

| Profile | n=1024 | n=2048 | n=4096 | n=8192 |
|---|---|---|---|---|
| A: cold-start (1 op) | LOSES (-4.49ms) | WINS (+0.11ms) | WINS (+0.26ms) | LOSES (-4.94ms) |
| A: cold-start (3 ops) | LOSES (-4.37ms) | WINS (+0.35ms) | WINS (+0.82ms) | LOSES (-4.09ms) |
| B: typical (20 ops) | LOSES (-3.30ms) | WINS (+2.43ms) | WINS (+5.54ms) | WINS, marginal (+3.07ms, ~3.8% of total) |
| C: long-lived (500 ops) | WINS (+26.9ms) | WINS (+61.1ms) | WINS (+138.9ms) | WINS (+205.4ms) |
| C: long-lived (5000 ops) | WINS (+310.2ms) | WINS (+611.1ms) | WINS (+1389.1ms) | WINS (+2102.3ms) |

Profile C (long-lived service, many repeated calls at one
configuration) is the ONLY profile that wins UNCONDITIONALLY across
every tested `n`, consistent with Phase 15's own original hypothesis.
Profile A (cold-start, 1-3 operations) LOSES whenever the prewarm cost
happens to be large (n=1024, n=8192 in this environment) and WINS when
it happens to be small (n=2048, n=4096) -- i.e., its outcome is
determined by the SAME unpredictable allocation-cost variance just
described, not by the workload itself. Profile B (typical, 20 ops) is
mixed, winning at 3 of 4 tested `n` but only marginally at n=8192.

### Memory overhead (directive Step 8)

Measured directly: prewarming 8 distinct `(n,sizeQ)` keys (every
`n` in {1024,2048,4096,8192} x `sizeQ` in {4,8}) added **6.00MB** to
GPU memory usage and **6.25MB** to host RSS -- small and bounded, each
key's packed buffer capped at `4*sizeQ*n*8` bytes (at most 2MB device
+ 2MB pinned host at the largest tested configuration, n=8192,
sizeQ=8).

### Decision (directive Step 8's three categories)

**RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE. NOT promoted to
production, and NOT exposed through the Rust API this round.**

This is neither a clean win (category 1) nor a clear rejection
(category 3) -- it sits in category 2 deliberately:

- The implementation is fully correct (bit-exact, every path), fully
  concurrency-safe (0/372 mismatches across 5 independent runs
  covering 6 distinct lifecycle scenarios), and has small, bounded
  memory overhead. These are NOT reasons to reject it.
- But the governing directive's own decision gate requires a
  "repeatable end-to-end improvement for a clearly defined workload,"
  and this round found the improvement is repeatable ONLY for
  Profile C (long-lived services) -- exactly the profile Phase 15
  already flagged as the target. For Profiles A and B, the outcome
  this round measured DEPENDS on an allocation-cost variance this
  library cannot predict or control, which is a materially different,
  weaker claim than every other kept optimization in Phases 11-15 has
  made.
- Per directive Step 6, no public Rust API is shipped this round:
  "Do not ship a public API until its ownership and failure semantics
  are unambiguous" -- the OWNERSHIP and FAILURE semantics ARE
  unambiguous (documented above, tested above), but the PERFORMANCE
  semantics (when does this actually help?) are not yet clean enough
  to commit to a public contract. A future round that wants to
  promote this should first either (a) confirm the allocation-cost
  variance's root cause (ideally with WSL2 VM memory-ballooning
  visibility this round did not have) and find a way to bound or
  avoid it, or (b) accept the risk explicitly for a specific, named
  long-lived-service deployment that calls `encrypt_prewarm_pinned_
  buffer` once at startup and performs many thousands of encryptions
  afterward, where Profile C's unconditional win applies regardless of
  the allocation-cost variance.
- `encrypt_host_shoup_pooled_barrett_prewarmed`,
  `encrypt_prewarm_pinned_buffer`, and `encrypt_pinned_buffer_is_ready`
  remain in the repository, fully built, tested, and registered in
  `CMakeLists.txt` -- available for any future C++-level caller or
  Rust FFI work that wants to build on a proven-correct, proven-safe
  foundation, without this round over-claiming a production-ready
  performance win it did not cleanly establish.

## Phase 17: diagnosing the allocation-cost variance -- the mechanism is the Linux kernel's memory-locking path, NOT CUDA-specific, with host memory pressure a plausible but unconfirmed amplifier

GOD PROMPT "CKKS PHASE 17". Starting commit `baee7a1` (Phase 16 end
state), confirmed via `git log -1` and `origin/main` before any change.
This section directly investigates the open question Phase 16 raised
and explicitly flagged as "not independently confirmed."

**No production or experimental code was modified this round.** This
is a pure diagnostic investigation: three new standalone benchmark
files were added (none touch `src/`, none touch the FFI bridge, none
touch any pool/cache), and all existing tests were re-run to confirm
zero regression.

### Methodology critique of Phase 16's own measurement (directive Step 1/2)

Re-reading `src/encrypt_host_shoup_pooled_barrett_prewarm.cpp`'s
`prewarm_locked()` found that Phase 16's single "prewarm cost" number
BUNDLED five separate calls into one timed region: `cudaMalloc`
(device, packed buffer), `cudaMallocHost` (pinned host, packed
buffer), and three small `cudaMalloc` calls (`d_fr`, `d_fp`, `d_mod`).
The governing directive's Step 2 explicitly requires these NOT be
combined. `bench/bench_pinned_alloc_diagnostic.cpp` (new) separates
every stage: CUDA init, `cudaMalloc`, `cudaMallocHost`, first-touch
(`memset`), H2D, D2H, `cudaDeviceSynchronize`, and release -- each
with min/median/p90/p95/max/mean/stddev over multiple repeats, for
both the pinned and pageable paths.

### Experiment 1: fresh-allocate-then-free, 10 repeats per size, one process

Every repeat does a genuine `cudaMalloc`+`cudaMallocHost`+...+
`cudaFree`+`cudaFreeHost` (no caching). Result: `cudaMallocHost` costs
**consistently 2.5-4.2ms for EVERY size** (n=1024 through n=8192), with
NO size-dependent pattern and NO "some sizes are cheap" behavior at
all -- flatly contradicting Phase 16's "n=2048/4096 cheap, n=1024/8192
expensive" result. This first experiment alone proves Phase 16's
pattern is NOT an intrinsic property of `cudaMallocHost` as a function
of size when memory is always immediately freed.

### Experiment 2: never-free, matching production's actual behavior

`bench/bench_pinned_alloc_diagnostic2.cpp` (new) never calls
`cudaFreeHost` between allocations -- exactly mirroring the real
pooled implementations (`encrypt_buf_pool_pinned`/`PrewarmEntry`),
which never free. Allocating n=1024,2048,4096,8192 in that order
(matching Phase 16's exact benchmark), THEN repeating the same four
sizes again:

```
call #1 n=1024  (0 live before): ~400-680ms  (confounded with CUDA context init -- first-ever CUDA call)
call #2 n=2048  (1 live before): ~12-15us    FAST
call #3 n=4096  (2 live before): ~5-9us      FAST
call #4 n=8192  (3 live before): ~2.8-4.9ms  SLOW
call #5 n=1024  (4 live before): ~19-39us    FAST
call #6 n=2048  (5 live before): ~2.9-4.5ms  SLOW
call #7 n=4096  (6 live before): ~10-35us    FAST
call #8 n=8192  (7 live before): ~3.2-6.2ms  SLOW
```

**This exact pattern reproduced identically across 5 independent full
runs** (slow at positions 1,4,6,8; fast at 2,3,5,7, every single time)
-- confirming Phase 16's original pattern IS real and reproducible,
and additionally revealing that it is NOT simply "first call slow,
rest fast" (call #4 is slow despite 3 live allocations already
existing) and NOT simply "small=fast, large=slow" (n=2048 is fast at
position 2 but slow at position 6 -- the SAME size, same process,
different outcome).

### Experiment 3: isolating CUDA from the underlying OS mechanism

`bench/bench_thp_diagnostic.cpp` (new) uses raw Linux `mmap` +
`madvise(MADV_HUGEPAGE/NOHUGEPAGE)` + `mlock()` + first-touch, with
**NO CUDA code involved at all**, on the exact same byte sizes. Result:
**the same qualitative variance reproduces** -- `mlock()` alone costs
anywhere from ~117us to ~3.6ms for the SAME 2MiB size across different
calls within the SAME process, and even the smallest tested size
(256KB) showed real variance (154-603us). The `MADV_HUGEPAGE` hint
did NOT deterministically correlate with slow/fast outcomes (both
hinted and non-hinted 2MiB calls showed both fast and slow instances).

**This is the single most important finding of this phase**: the
variance reproduces with ZERO CUDA or NVIDIA driver code running,
using only the standard Linux `mlock()` syscall. Whatever `cudaMallocHost`
does under the hood to pin memory, the variance it exhibits is
downstream of a MORE FUNDAMENTAL, CUDA-independent Linux kernel
behavior -- not a CUDA allocator quirk (Hypothesis C is effectively
ruled out as the primary mechanism).

### Experiment 4: the exact-2MiB boundary is not special; "large and never-freed" is what correlates

A follow-up test allocated six buffers via `cudaMallocHost`, NEVER
freeing any of them, at sizes `2097152` (exactly 2MiB),
`2097152-65536` (2MiB minus 64KB, NOT huge-page-aligned), and
`2097152+65536`, interleaved. Result: after the first (context-init-
confounded) call, **every subsequent call was slow** (2.7-8.5ms),
REGARDLESS of exact 2MiB alignment -- directly refuting a clean
"exactly-2MiB-triggers-THP" rule as the sole mechanism (Experiment 2's
n=2048/4096 sub-1MiB calls being fast, contrasted with this all-~2MiB
sequence being uniformly slow, instead points toward allocation SIZE
in a looser sense -- roughly "around or above ~1-2MiB" -- combined
with GROWING cumulative live (never-freed) pinned memory, as the
better-supported pattern, though not a clean deterministic rule
either; see the ranked assessment below).

### Windows-side memory bracketing (Hypothesis A)

Windows free physical memory was measured immediately before and
after a batch of 5 repeated diagnostic-2 runs (10 `cudaMallocHost`
calls each, 50 total): **1910.9MB before, 1825.3MB after** (a ~85MB
drop). This is CONSISTENT with host memory pressure being a real,
concurrent factor, but is CORRELATIONAL ONLY -- this coarse
before/after bracketing cannot isolate causation from confounds (other
Windows background processes, the measurement overhead of invoking
PowerShell itself, etc.), and no tool was available to sample Windows
memory AT THE EXACT MOMENT of each individual slow allocation (the
diagnostic runs are WSL-side C++ processes; invoking PowerShell
per-sample would itself take longer than the allocation being
measured). Windows free memory was consistently LOW throughout this
entire investigation (1.2-2.4GB, consistent with every prior phase's
own measurements) -- this is circumstantial support for Hypothesis A
as a background condition, not a demonstrated causal link.

### WSL2 configuration inspected (Hypothesis B)

`.wslconfig`: `memory=11GB` (a CAP, not a fixed pre-committed
reservation -- WSL2's Hyper-V VM grows its actual committed memory
dynamically up to this cap by default). `/etc/wsl.conf`: `[boot]
command = "ulimit -l unlimited"`. **A real, separate finding**: this
boot-time command does NOT appear to apply to ordinary interactive/
exec'd sessions reached via `wsl.exe -d Ubuntu -- bash -lc "..."` --
every such session measured `ulimit -l` (max locked memory) at
**65536 KB (64MB), both soft AND hard limit**, meaning an unprivileged
shell cannot raise it further. This is notable context (a process-
wide cap on how much memory ANY process can `mlock()`) but the
packed-buffer sizes tested (256KB-2MiB, cumulatively under 8MiB even
across all 4 sizes prewarmed) stay far below 64MB, so exceeding
`RLIMIT_MEMLOCK` outright is NOT a plausible direct explanation for
the observed slowness -- it is reported here as a correctly-identified
but ultimately NON-explanatory finding, not asserted as the cause.
No `.wslconfig` change, WSL restart, or VM setting was made -- per the
directive's explicit instruction, this was investigated read-only.

### Benchmark artifact check (Hypothesis E)

Ruled out as the PRIMARY cause: `std::chrono::steady_clock` (monotonic,
high-resolution) was used throughout; the SAME qualitative variance
reproduced across THREE independently-written test programs (the
original bundled Phase 16 benchmark, the isolated CUDA diagnostic, and
the pure-Linux non-CUDA diagnostic) with different code structure,
different timer placement, and (for the pure-Linux test) a completely
different API surface -- a measurement artifact specific to any ONE
program's timer or structure cannot explain a pattern that persists
across three independently-written programs.

### Ranked root-cause assessment (directive Step 5)

1. **Linux kernel memory-locking (`mlock()`) path variance -- HIGH
   confidence, directly demonstrated.** Evidence: Experiment 3
   reproduced the SAME qualitative variance (hundreds of
   microseconds to several milliseconds, for the same sizes) using
   raw `mlock()` with zero CUDA involvement. Contradictory evidence:
   none found. Falsifying experiment: if a kernel/mlock-level fix
   (e.g., disabling THP via `/sys/kernel/mm/transparent_hugepage/
   enabled`, which was inspected read-only and found set to
   `madvise` -- not changed) eliminated the variance entirely, that
   would strengthen this; this round did not make that change (per
   the directive's "do not change... without first recording the
   baseline" instruction, and because it is a system-wide setting
   change, a separate, reversible experiment left for a future
   round if pursued). Location: inside the Linux kernel / Windows-
   WSL2 VM, NOT inside this project's code, NOT CUDA-specific.
2. **Host/WSL2 memory pressure as a contributing amplifier -- MEDIUM
   confidence, plausible but not isolated.** Evidence: Windows free
   memory was consistently low (1.2-2.4GB) throughout, and dropped
   further during a batch of allocations. Contradictory evidence:
   none directly contradicts it, but the correlation is coarse
   (before/after bracketing, not per-call) and confounded with other
   background activity. Falsifying experiment: repeating the same
   diagnostic on a machine/session with several GB of free host RAM
   and comparing variance magnitude -- not performed this round (no
   such environment was available). Location: Windows host /
   Hyper-V / WSL2 VM boundary, outside this project's control either
   way.
3. **Allocation size (roughly "large," not a clean power-of-two or
   exact-2MiB-THP rule) combined with growing cumulative never-freed
   pinned memory -- MEDIUM confidence.** Evidence: Experiment 4
   showed ALL ~2MiB-class allocations slow once several were already
   live, regardless of exact THP alignment; Experiment 2 showed
   sub-1MiB sizes fast in the SAME sequence shape. Contradictory
   evidence: Experiment 2's n=2048 call was fast on its FIRST
   occurrence (position 2) despite being the "large-ish" 512KB size,
   and slow on its SECOND occurrence (position 6) -- so size alone,
   independent of history, is not a sufficient predictor either.
   Falsifying experiment: a controlled sweep allocating many
   DIFFERENT sizes in a fixed, non-repeating order with memory usage
   tracked at each step -- not performed this round due to time
   constraints; flagged as the clearest next experiment if this
   question is revisited.
4. **WSL2-specific Hyper-V dynamic-memory ballooning as a DISTINCT
   mechanism from ordinary Linux kernel memory pressure -- NOT
   CONFIRMED, cannot be distinguished from Hypothesis 1/2 with the
   tooling available.** This was Phase 16's own original (hedged)
   guess. No tool exists inside this WSL2 guest to directly observe
   Hyper-V's memory-balloon negotiation events, so this round can
   neither confirm nor refute it as a mechanism DISTINCT from
   ordinary kernel-level memory pressure (#1/#2 above) -- it may well
   be the SAME underlying phenomenon Hypothesis 1/2 already capture,
   just described with a WSL2-specific vocabulary Phase 16 reached
   for without evidence. This round's finding (the variance
   reproduces identically with zero CUDA/driver code, via plain
   `mlock()`) makes a WSL2-ballooning-SPECIFIC mechanism less
   necessary as an explanation than Phase 16 implied, since ordinary
   Linux `mlock()` under memory pressure is already a sufficient,
   demonstrated candidate.
5. **CUDA-driver-specific allocator behavior -- LOW confidence,
   effectively ruled out as the PRIMARY cause.** Evidence against:
   Experiment 3's pure-Linux, zero-CUDA reproduction. It remains
   possible CUDA's own wrapper adds SOME overhead on top of the
   underlying `mlock()` variance, but it is not the dominant or
   necessary mechanism.
6. **Benchmark/measurement artifact -- RULED OUT as the explanation
   for the core phenomenon.** Reproduced across three independently-
   written programs with different timer placement and different
   APIs; a single flawed benchmark cannot explain that.

### Engineering response (directive Step 6)

**No production code change is justified.** The root mechanism (Linux
kernel `mlock()` path variance, Hypothesis 1, HIGH confidence) is
outside this project's code and outside CUDA's own allocator --
nothing in `src/` can fix a kernel-level or Hyper-V-level memory-
locking cost. The Phase 16 decision (opt-in experimental feature, not
promoted, pageable path remains production, applications that never
call `encrypt_prewarm_pinned_buffer` pay nothing extra) remains the
CORRECT engineering response, now for a more precisely understood
reason: Phase 16 could only say "the cost is unpredictable, mechanism
unconfirmed"; this phase upgrades that to "the cost is unpredictable
because it reproduces at the Linux `mlock()` level independent of
CUDA, most likely amplified by this host's persistent low free memory
-- a mechanism no application-level code change in this repository can
reliably control." The one incremental improvement worth making
available (not a promotion, an honesty improvement): a future caller
of `encrypt_prewarm_pinned_buffer` should be told, via this
documentation, that its own cost is unpredictable and NOT simply
proportional to `n` -- which this document now states directly, where
Phase 16 could only speculate.

### Decision (directive Step 9)

**Category 3: the variance is externally controlled (Linux kernel
`mlock()` path, plausibly amplified by host memory pressure), so the
current opt-in design should remain.** This is a STRONGER version of
category 3 than Phase 16 could claim, because the external mechanism
is now directly DEMONSTRATED (via a CUDA-free reproduction), not
merely suspected. No code was changed. The opt-in prewarming API
(`encrypt_prewarm_pinned_buffer`/`encrypt_pinned_buffer_is_ready`/
`encrypt_host_shoup_pooled_barrett_prewarmed`) remains exactly as
Phase 16 left it: fully correct, fully concurrency-safe, retained as
an experimental feature, not wired into production or the Rust API.

## Phase 18: controlled THP experiment -- the system-wide toggle is blocked by permissions (correctly not worked around); a real but small THP-hint effect found at exactly 2MiB; Phase 17's "not CUDA-specific" conclusion is REVISED based on same-session interleaved re-testing

GOD PROMPT "CKKS PHASE 18". Starting commit `ff81b97` (Phase 17 end
state), confirmed via `git log -1`, `git status`, and `origin/main`
before any change: HEAD and `origin/main` both at `ff81b97`, working
tree clean apart from pre-existing unrelated untracked items
(`.claude/`, `.scrum-orchestrator/`, `CLAUDE.md`, `build2/`,
`tools/__pycache__/`), none of which belong to this project's tracked
history and none of which were touched this round.

**This is a controlled diagnostic experiment, not a production
optimization. No production or experimental encryption code was
modified.** Two new standalone diagnostic files were added
(`bench/bench_pinned_alloc_diagnostic3.cpp`,
`bench/bench_thp_hint_by_size.cpp`); none touch `src/`, the FFI
bridge, or any pool/cache. Direct inspection confirmed
`src/encrypt_host_shoup_pooled_barrett_batched.cpp`,
`src/encrypt_host_shoup_pooled_barrett_batched_pinned.cpp`,
`src/encrypt_host_shoup_pooled_barrett_prewarm.cpp`,
`rust/crates/ckks-cuda/build.rs`, and
`rust/crates/ckks-cuda/cpp_bridge/encdec_bridge.cpp` were never
written to this phase.

### Step 1: baseline environment (read-only)

Recorded before any experiment, and re-verified identical afterward:

- `/sys/kernel/mm/transparent_hugepage/enabled`: `always [madvise]
  never` (madvise active, selected).
- `/sys/kernel/mm/transparent_hugepage/defrag`: `madvise` (selected).
- Kernel: `6.6.87.2-microsoft-standard-WSL2+`.
- Page size: 4096 bytes (standard 4KB base page; THP would promote to
  2MiB = 2097152 bytes, matching the n=8192 packed-buffer size
  exactly).
- `ulimit -l` (both soft and hard): 65536 KB (64MB) -- unchanged from
  Phase 17's own finding; kept as a separate, untouched confound per
  the directive's explicit Step 5 instruction.
- Swap: 8GB configured, mostly free throughout.
- `/proc/meminfo`: `MemFree`/`MemAvailable`/`AnonHugePages`/
  `SwapFree` all sampled per-allocation by the new diagnostics (see
  below); no anomalous reading at baseline.
- Windows host free RAM (`(Get-CimInstance
  Win32_OperatingSystem).FreePhysicalMemory`, NOT WSL's own `free
  -h`, per the directive's explicit requirement): approximately
  2.3GB at the start of this phase, approximately 1.93GB at the end
  -- consistent with every prior phase's own low-free-memory
  readings, and confirming the constraint #13 gate (≥6GiB required
  for any n=131072 run) was never met, so no such run was attempted.

### Step 2/3: the system-wide THP toggle is blocked by permissions -- correctly NOT worked around

Attempted the least invasive possible change: a same-value,
effectively no-op write, `echo madvise >
/sys/kernel/mm/transparent_hugepage/enabled`. Result: `bash:
/sys/kernel/mm/transparent_hugepage/enabled: Permission denied`.
Confirmed via `id` that the session user (`samfrazerdutton`) is in
the `sudo` group, but `sudo -n true` and `sudo -n -l` both report "a
password is required" -- there is no passwordless sudo available
non-interactively. This is exactly the blocker the governing
directive's Step 3 and safety constraint #5 anticipated ("If
permissions do not permit a reversible runtime change, stop and
report the blocker. Do not work around it using destructive system
modifications").

No workaround was attempted (no password guessing, no alternative
privilege escalation, no editing of sudoers, no attempt to disable
the permission check). `AskUserQuestion` was used to let the user
decide between documenting the blocker or providing sudo credentials
themselves. **The user explicitly chose "Skip the THP toggle, document
the blocker (recommended)."** Per that decision, the system-wide THP
setting was NEVER changed at any point in this phase -- confirmed by
re-reading `/sys/kernel/mm/transparent_hugepage/enabled` both
immediately after the denied write attempt and again at the end of
the phase: both readings identical to the original baseline
(`madvise` selected, unchanged).

Because the system-wide toggle could not be exercised, Step 2's
three-condition (A/B/C) design as originally specified could not be
run. Instead, the investigation used the ONE THP-related variable
that IS controllable without root: the per-allocation
`madvise(addr, len, MADV_HUGEPAGE | MADV_NOHUGEPAGE)` hint, which
requires no special privilege on memory a process has mapped itself.
This substitution is reported here explicitly rather than silently
presented as equivalent to the originally planned experiment --it
tests whether THP participation on a per-allocation basis changes
cost, not whether disabling THP system-wide does.

### Step 4/5: `bench_pinned_alloc_diagnostic3.cpp` -- per-sample methodology under the unchanged THP=madvise setting

This diagnostic fulfills the directive's per-sample data
requirements (timestamp, allocation index, size, order, live-before
count, full `/proc/meminfo` snapshot, separately-timed
mlock/first-touch/release stages) using pure `mmap`+`mlock`, zero
CUDA, run with ascending/descending/randomized order and both
never-free and free-immediately modes. Representative run (`asc
neverfree 3`, 12 samples, packed sizes 256KB/512KB/1MB/2MB): `mlock()`
latency was clean and monotonic in size --
256KB≈150-310us, 512KB≈280-480us, 1MB≈550-840us, 2MB≈1100-1450us --
with **no multi-millisecond outliers of the kind Phase 17's own
`bench_pinned_alloc_diagnostic2.cpp` (CUDA path) and
`bench_thp_diagnostic.cpp` (pure-Linux path) both reported**.
Reproduced identically across 5 total runs this session, across all
tested orders and live-count buckets (0, 1-3, 4-7, 8+ concurrently
live allocations) -- no bucket showed the multi-millisecond spikes
Phase 17 recorded for pure Linux `mlock()`.

### Step 4: `bench_thp_hint_by_size.cpp` -- the decisive, demonstrated THP-hint effect

This diagnostic interleaves `MADV_HUGEPAGE` and `MADV_NOHUGEPAGE`
calls, 8 repeats each, across all four packed-buffer sizes, reporting
median latency and the NOHUGEPAGE/HUGEPAGE ratio per size. Run twice
independently, both runs highly consistent:

```
n=1024 (256KB):  HUGEPAGE median~124-125us   NOHUGEPAGE median~121-125us   ratio~0.96-1.01  (no meaningful difference)
n=2048 (512KB):  HUGEPAGE median~252-262us   NOHUGEPAGE median~243-255us   ratio~0.93-1.01  (no meaningful difference)
n=4096 (1MB):    HUGEPAGE median~518-589us   NOHUGEPAGE median~518-532us   ratio~0.90-1.00  (no meaningful difference)
n=8192 (2MB):    HUGEPAGE median~230-247us   NOHUGEPAGE median~1217-1270us ratio~4.93-5.51  (CONSISTENT ~5x SLOWDOWN)
```

**A real, demonstrated, reproducible THP-hint effect exists, but it
is narrowly specific to allocations of exactly 2097152 bytes (2MiB --
the exact x86-64 huge-page size)**: disabling the per-call hugepage
hint makes `mlock()` on an exactly-2MiB region roughly 5x slower
(≈230-247us → ≈1217-1270us), reproduced across both independent runs.
No such effect exists at 256KB, 512KB, or 1MB -- at those sizes the
hint makes no meaningful difference either way (ratios within
0.90-1.01 of each other, i.e. noise). This is the first time this
project has DEMONSTRATED (not merely correlated) a specific,
mechanistic, reproducible THP effect, rather than a generic
"variance exists" observation.

Critically, this effect's magnitude (≈1ms worst case, at exactly one
size) is far too small to explain the multi-millisecond-to-several-
hundred-millisecond variance Phase 16/17 observed in
`cudaMallocHost`. THP participation is a real but minor factor, not
the dominant driver of the originally observed variance.

### Step 6/self-correction: Phase 17's "not CUDA-specific" conclusion does NOT hold up under same-session interleaved re-testing -- REVISED

Phase 17 concluded (HIGH confidence): "the variance reproduces
identically with zero CUDA code, via plain `mlock()`, proving the
mechanism is the Linux kernel's own memory-locking path, NOT
CUDA-specific." This phase re-ran that comparison more rigorously --
in the SAME session, INTERLEAVED, back-to-back for the identical
size -- and the result directly contradicts it:

- Pure Linux `mlock()`, re-run fresh this phase via both
  `bench_pinned_alloc_diagnostic3.cpp` and a fresh re-run of Phase
  17's own `bench_thp_diagnostic.cpp`: clean, fast, largely
  size-proportional, sub-2ms for every sample, NO multi-hundred-
  millisecond outliers.
- `cudaMallocHost`, re-run in the SAME session via Phase 17's own
  `bench_pinned_alloc_diagnostic2.cpp`, interleaved in the same short
  time window with the pure-Linux tests above: consistently
  **2984-5079us**, and in earlier Phase 16/17 runs, occasionally
  several hundred milliseconds for the very first call -- an order of
  magnitude slower than the worst pure-Linux `mlock()` sample
  measured in the identical window (174-1613us for the same size).

**This means Phase 17's "pure Linux reproduces the same variance"
finding was most likely a coincidental timing artifact -- both test
types happened to be slow during that one investigation window, not
a robust, repeatable equivalence.** This is an honest revision, not a
hidden one, following this project's own established precedent (e.g.
Phase 12's correctness-breakthrough round reversing an earlier
library-bug claim). The correct reading of the combined Phase
17+18 evidence is: plain Linux `mlock()` CAN exhibit variance (Phase
17 observed it once; this phase did not reproduce it under otherwise-
identical conditions, suggesting it is itself environment/timing-
dependent rather than deterministic), but `cudaMallocHost` carries
its OWN, additional, consistently-reproducible overhead ON TOP of
whatever the underlying `mlock()` call costs -- an overhead this
phase's interleaved testing shows is NOT explained by the pure-Linux
path alone.

### Updated ranked root-cause assessment (supersedes Phase 17's Step 5 ranking)

1. **CUDA/driver-specific overhead, most plausibly WSL2's
   paravirtualized GPU-passthrough layer forwarding `cudaMallocHost`
   to the Windows host driver -- PROMOTED to the leading hypothesis,
   MEDIUM-HIGH confidence.** Evidence: this phase's same-session,
   interleaved comparison showed `cudaMallocHost` consistently
   2-20x slower than pure-Linux `mlock()` for the identical size,
   measured back-to-back in the same short window where confounds
   (host memory state, time-of-day load, kernel state) are held
   maximally constant. Contradictory evidence: Phase 17 itself once
   observed pure-Linux `mlock()` variance of a similar order of
   magnitude, so a CUDA/driver-only explanation is not yet airtight.
   Falsifying experiment: capturing a GPU-passthrough-layer trace (no
   tool for this was available in this WSL2 guest) or testing on a
   non-WSL2 Linux host with a directly-attached GPU to see if the gap
   vanishes -- not performed this round, no such environment
   available.
2. **Linux kernel `mlock()` path variance -- DOWNGRADED from Phase
   17's HIGH to MEDIUM confidence; demonstrated to occur at least
   once, but NOT reproduced under otherwise-matched same-session
   conditions this phase.** Evidence: Phase 17's own `bench_thp_
   diagnostic.cpp` showed multi-millisecond `mlock()` outliers in its
   original run; this phase's fresh re-run of the SAME binary showed
   clean sub-2ms behavior throughout. This is not a contradiction of
   Phase 17's raw data (the outliers were real when they happened),
   but it means the mechanism is not a deterministic, always-present
   property of plain `mlock()` on this system -- it is itself
   intermittent, and its intermittency has not been isolated from
   host-level load at the time of each specific run.
3. **THP per-call hint effect, demonstrated, narrow -- CONFIRMED,
   MEDIUM confidence, but minor in magnitude.** A real ~5x effect at
   exactly 2MiB (≈230us → ≈1217us), absent at other sizes. Too small
   (≈1ms worst case) to be the dominant driver of the multi-
   millisecond-to-hundreds-of-milliseconds `cudaMallocHost` variance,
   but a real, now-demonstrated contributor specifically for
   exactly-2MiB-sized allocations when the hugepage hint is unset or
   defeated.
4. **System-wide THP enabled/defrag setting as a contributor --
   UNTESTED, blocked by permissions.** Cannot be confirmed or refuted
   this phase; the per-call hint test above is the closest available
   substitute and shows a real but narrow effect, suggesting (but not
   proving) that toggling the system-wide setting would show a
   similarly narrow, size-specific effect rather than a broad one.
5. **Host/WSL2 memory pressure as a contributing amplifier -- MEDIUM
   confidence, unchanged from Phase 17; still plausible, still not
   isolated.** Windows free RAM remained consistently low (1.93-2.3GB)
   throughout this phase, consistent with Phase 16/17's own readings.
6. **WSL2-specific Hyper-V ballooning as a mechanism DISTINCT from
   ordinary kernel memory pressure or CUDA/driver passthrough -- still
   NOT CONFIRMED,** unchanged from Phase 17; no new tooling was found
   to distinguish it from hypotheses 1/2/5 above.
7. **Benchmark/measurement artifact -- still effectively RULED OUT**
   for the core `cudaMallocHost`-vs-`mlock()` gap: the SAME two
   binaries (`bench_pinned_alloc_diagnostic2.cpp` for CUDA,
   `bench_thp_diagnostic.cpp`/`bench_pinned_alloc_diagnostic3.cpp` for
   pure Linux) were used in both Phase 17 and this phase, with
   identical timer code (`std::chrono::steady_clock`), yet produced
   a consistent, reproducible, order-of-magnitude gap when run
   interleaved -- a single benchmark's artifact cannot explain a gap
   between two independently-written, previously-validated programs
   measured side by side.

### Step 7: production implications

**No production code or configuration change is justified by this
phase's findings.** The Phase 16 decision (opt-in experimental
pinned-prewarming API, not promoted, pageable path remains
production) remains correct, and is now supported by a MORE
PRECISELY understood (if partially revised) picture: `cudaMallocHost`
carries a consistently-reproducible overhead over plain Linux
`mlock()` for the same size, most plausibly rooted in WSL2's GPU-
passthrough layer, with a real but narrow (~5x, ~1ms-scale, exactly-
2MiB-specific) additional THP-hint contribution layered on top. None
of this is controllable from `src/` or the Rust API. The one
documentation-level improvement made here (not a code or config
change): this section itself, recommending that a future WSL2 CKKS
deployment on a host with persistently low free RAM should expect
`cudaMallocHost` costs in the low-single-digit-millisecond range as
a NORMAL floor (not a bug), with the opt-in prewarming API existing
specifically to absorb that cost once, off the hot path, for
applications that choose to pay it.

### Step 8: regression confirmation

Full Rust suite re-run from the current working tree (no production
file touched): **debug -- 4 passed, 38 passed, 0 failed, 1 ignored
(two empty 0-test crates); release -- identical: 4 passed, 38 passed,
0 failed, 1 ignored.** Zero regression, as expected since no
production or experimental source file was modified this phase.

### Step 9: THP/ulimit restoration verification

Trivial by construction: no system setting was ever successfully
changed (the one write attempted was denied by the kernel itself), so
there is nothing to restore. Final verification re-read of
`/sys/kernel/mm/transparent_hugepage/enabled` and `/sys/kernel/mm/
transparent_hugepage/defrag` at the end of this phase matched the
Step 1 baseline exactly (`madvise` selected for both). `ulimit -l`
was never touched, per the directive's explicit instruction to treat
it as a separate confound; still 65536 KB (64MB), both soft and hard,
unchanged.

### Decision

**Category 3 continues to apply, now with a partially revised
rationale.** The dominant externally-controlled mechanism is most
plausibly CUDA/WSL2-driver-passthrough overhead (promoted this
phase), not purely the Linux kernel `mlock()` path (as Phase 17
concluded with HIGH confidence -- now downgraded to MEDIUM, since
this phase could not reproduce Phase 17's own pure-Linux variance
under matched, interleaved, same-session conditions). A real,
demonstrated, but narrow THP-hint effect (~5x, ~1ms-scale, exactly-
2MiB-specific) was found and is reported as a genuine new result. No
production or experimental code was changed. The opt-in prewarming
API remains exactly as Phases 16/17 left it.
