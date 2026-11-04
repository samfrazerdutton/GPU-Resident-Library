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
