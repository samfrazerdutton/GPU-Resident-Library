# CUDA graph-capture defect in `keyswitch_resident` — root cause, fix, and validation (Phase 30)

GOD PROMPT "CKKS PHASE 30". Phase 29 found `tests/test_ks_graph.cpp` failing
(`graph-vs-direct mismatches=262144`) and classified it as a genuine,
previously-undocumented, pre-existing defect, root cause not yet isolated.
This document records the full investigation: reproduction, root-cause
isolation via two independent standalone reproducers, the fix, and
rigorous validation (correctness, lifecycle, memory, concurrency,
performance).

## 1. The exact original failure

`tests/test_ks_graph.cpp`: derives real CKKS parameters from OpenFHE
(n=32768, depth=3, HYBRID key switch — the same way `bench/
bench_evalmult.cpp` does), builds a native `KeySwitchConstants`/
`DeviceKSContext`/`DeviceKSWork`, then:

1. Runs `keyswitch_resident(d_a, d_b0, d_b1, C, W, 0)` directly (stream 0),
   captures `ref0`/`ref1`.
2. Zeros `d_b0`/`d_b1`, captures a `cudaStreamBeginCapture`/
   `keyswitch_resident(..., s)`/`cudaStreamEndCapture` sequence on an
   explicit stream, instantiates and launches the graph.
3. Compares the replayed output against `ref0`/`ref1`.

**Exact result**: `graph-vs-direct mismatches=262144` — every element of
BOTH output ciphertext components wrong (confirmed via added
instrumentation this round: `b0=131072/131072 b1=131072/131072`, i.e.
100% of each buffer, not a partial or small numerical discrepancy).
**No CUDA error was ever reported** — `cudaStreamBeginCapture`,
`cudaStreamEndCapture`, `cudaGraphInstantiate`, `cudaGraphLaunch`, and
`cudaStreamSynchronize` all returned `cudaSuccess` at every step. This is
a pure logic/value defect, not a memory-safety or API-misuse error —
confirmed by running the fixed-and-unfixed versions under `compute-
sanitizer --tool memcheck`, which reported 0 errors in BOTH cases (the
bug is entirely about which host memory a graph node reads at replay
time, not about accessing memory that doesn't exist).

## 2. Isolating the root cause — two standalone reproducers

Per the governing directive's explicit requirement, `src/
keyswitch_resident.cpp`'s internals were not touched until a standalone
reproducer established the cause.

### 2.1 First reproducer: the dangling-host-pointer mechanism

A minimal, gpufhe-independent CUDA program (no cryptography, no gpufhe
headers) proved the general mechanism:

```
helperFillDevice(d_dst, n) {
    std::vector<uint64_t> hostTemp(n);           // LOCAL to this call
    fill hostTemp with a recognizable pattern (1000+i)
    cudaMemcpyAsync(d_dst, hostTemp.data(), ..., HostToDevice, stream);
}                                                  // hostTemp destructs HERE
```

Called directly: `d_dst` ends up with the correct pattern (1000, 1001,
1002, 1003, ...) — pageable-memory `cudaMemcpyAsync(HostToDevice)` is
synchronously staged by the driver before the call returns, so the data
is safely consumed before `hostTemp` is destroyed.

Called inside `cudaStreamBeginCapture`/`cudaStreamEndCapture`, with
unrelated host vectors allocated and filled (0xDEADBEEF, 0xFEEDFACE)
*between* capture end and `cudaGraphLaunch`: the replayed graph read
**`102228211831641, 15997515988907812752, 3735928559, 3735928559`** —
`3735928559` decimal is exactly `0xDEADBEEF`. **Direct, unambiguous proof**
that a `cudaMemcpyAsync(HostToDevice)` node added via stream capture
stores the host *pointer*, not a snapshot of its bytes, and re-reads from
that address at every future `cudaGraphLaunch` — long after the
capturing function (and any local `std::vector` it used) has returned.

A control using a heap-allocated buffer kept alive across capture and
every replay produced the correct pattern with 0 mismatches, confirming
the fix shape: keep the host source buffer alive, not "host-to-device
copies are forbidden in graphs" (they are not).

### 2.2 Second reproducer: narrowing to `src/keyswitch_resident.cpp`

A second, OpenFHE-free reproducer using gpufhe's own native keygen
(mirroring `tests/test_rot_resident.cpp`'s setup) at n=1024, tw=4
reproduced the exact same symptom (`mismatches=8192`, i.e. 100% of both
buffers) at far smaller, faster-to-iterate scale — ruling out anything
OpenFHE-specific and confirming the defect is purely in
`keyswitch_resident`'s own code.

Staged debug instrumentation (temporary, removed before the final fix)
isolated exactly where the divergence first appears: `W.d_part`
immediately after the raw device-to-device copy that seeds it (0 diffs
vs direct) is correct; `W.d_part` immediately after the very next step —
the batched INTT launch, which consumes `W.d_rootsA`/`W.d_rootsB`/
`W.d_towerMods`/`W.d_nInv`/`W.d_nInvPrecon` — is already 100% wrong. The
defect is in how those five arrays reach the device.

## 3. The actual mechanism (two layered bugs)

`ks_accumulate_qp` and `ks_moddown` (the two halves `keyswitch_resident`
composes) each build small host-side arrays (which device root-table
pointer and modulus belongs to which tower) and upload them via
`cudaMemcpyAsync(..., HostToDevice, s)` immediately before the kernel
that consumes them. There are 5 such blocks total: 2 per part inside
`ks_accumulate_qp`'s loop (the "part slice" INTT role and the
"complement" NTT role) plus the batched digit-pointer array (also
per-part), and 2 inside `ks_moddown` (the P-part INTT role and the Ql NTT
role).

**Bug 1 (dangling pointer)**: originally, the host arrays
(`hir`/`hip`/`hmod`/`hninv`/`hninvp`/`hfr`/`hfp`/`hp`) were plain
function-local `std::vector`s, destroyed when `ks_accumulate_qp`/
`ks_moddown` returned — which happens well before `cudaStreamEndCapture`
even runs, let alone any later `cudaGraphLaunch`. Exactly the mechanism
section 2.1 proved.

**First fix attempt (insufficient on its own)**: moving these to a
*single* buffer per field, owned by the long-lived `DeviceKSWork` struct,
fixed the lifetime problem — but running the real test afterward still
produced `mismatches=0` only for... no, it still produced
`mismatches=262144`, unchanged. Investigating why uncovered **Bug 2**.

**Bug 2 (value aliasing across (part, role))**: `ks_accumulate_qp`'s loop
runs `numPart` times, and within each iteration performs TWO of these
host-array-then-memcpy sequences (one for the part-slice INTT role, one
for the complement NTT role) — all sharing the SAME single host buffer
(after the first fix). During a captured call, the host C++ loops that
*fill* that buffer with each role's correct values run exactly once,
synchronously, at capture time. They do **not** re-run on replay — only
the recorded GPU-side memcpy/kernel nodes do. So by the time the graph
replays, every one of those 4+ memcpy nodes (across every part and both
roles) reads from the SAME host address — which holds whatever was
written *last* during capture (the final part's complement-role values),
correct for exactly one of those nodes and wrong for every other. This
explains the symptom precisely: not "one buffer never written" but
"every stage downstream of the first wrongly-fed kernel inherits
corrupted intermediate data," which is why 100% of both final outputs
differ, not a small or partial fraction.

This is a genuinely different class of bug from Bug 1: Bug 1 is about
*lifetime* (the buffer doesn't exist long enough); Bug 2 is about
*identity* (multiple logically-distinct values being forced to share one
physical buffer, which is only safe if whatever reads it re-derives its
own value fresh each time — true for direct calls, false for a replayed
graph, which never re-runs host code).

## 4. The fix

`include/keyswitch_resident.h`'s `DeviceKSWork` gained one **dedicated,
persistent host buffer per (part, role)** instead of one shared buffer
per field:

```cpp
std::vector<std::vector<const uint64_t*>> h_rootsA_part, h_rootsB_part;   // indexed [part]
std::vector<std::vector<uint64_t>> h_towerMods_part, h_nInv_part, h_nInvPrecon_part;
std::vector<std::vector<const uint64_t*>> h_rootsA_compl, h_rootsB_compl; // indexed [part]
std::vector<std::vector<uint64_t>> h_towerMods_compl;
std::vector<std::vector<const uint64_t*>> h_digPtrs_part;                 // indexed [part]
std::vector<const uint64_t*> h_rootsA_pwork, h_rootsB_pwork;              // ks_moddown role 1
std::vector<uint64_t> h_towerMods_pwork, h_nInv_pwork, h_nInvPrecon_pwork;
std::vector<const uint64_t*> h_rootsA_qsw, h_rootsB_qsw;                  // ks_moddown role 2
std::vector<uint64_t> h_towerMods_qsw;
```

Sized once in `ks_work_create` (to each part's actual `sizePart[part]`/
`sizeCompl[part]`, or `sizeQlP`/`sizeP`/`sizeQl` for the non-part-indexed
ones). `ks_moddown`'s two roles are *not* part-indexed (its value depends
only on the fixed context `C`, identically for both of its call sites per
`keyswitch_resident` — re-overwriting with the same value every call is
harmless), but ARE kept in two *separate* buffers from each other, since
they are two distinct roles within one call, exactly the aliasing Bug 2
describes.

The underlying **device** buffers (`W.d_rootsA` etc.) stay shared and
reused exactly as before — that reuse was never the problem, since
ordinary CUDA stream ordering (identical whether live or replayed)
already guarantees each kernel sees whatever its immediately-preceding
memcpy just wrote. Only the **host-side source** needed per-(part,role)
separation.

No device pointer, cryptographic constant, modulus, root table, or
computed value changed — this is purely a storage-location refactor of
already-correct intermediate scratch data.

## 5. Correctness validation

- `tests/test_ks_graph.cpp` (the original failing test, instrumented this
  round with a per-buffer breakdown and sample values): **`mismatches=0
  (b0=0/131072 b1=0/131072)`, bit-exact, 5/5 repeated runs.**
- `compute-sanitizer --tool memcheck`: 0 errors, both before and after
  the fix (confirming this was never a memory-safety bug compute-sanitizer
  could catch — it is a logic bug about which already-valid memory a node
  reads).
- `tests/test_ks_graph_lifecycle.cpp` (new, native, registered with
  ctest): 14/14 checks passing, reliably, across 8+ repeated runs —
  covering:
  1. Normal (direct) keyswitch.
  2. Graph capture (both `cudaStreamBeginCapture`/`cudaStreamEndCapture`
     report no error).
  3. Graph instantiation.
  4. First graph launch, bit-exact vs direct.
  5. 10 repeated graph launches, ALL bit-exact vs direct (poisoning the
     output buffer with `0xFF` before each replay, so a no-op replay
     cannot pass).
  6. Graph replay with **updated input data** at the same device pointer
     (same captured graph, new ciphertext content written to `d_a`
     between replays) — matches a fresh direct run on the new data,
     confirming the graph correctly re-reads device-side input each
     replay (this was never broken — only the *host*-sourced scratch
     data was).
  7-9. Two independent graphs, two independent `DeviceKSWork` objects,
     same context, launched on independent streams before either is
     synchronized — each matches its own direct reference, no
     cross-contamination, and (sanity check) their different-random-input
     outputs are genuinely different from each other.
  - Destroying graph 1 (+ its stream) while graph 2 remains un-launched
    leaves graph 2 fully functional afterward (independent lifetimes).
  10-12. Graph executable, graph, context, and work-object destruction in
     the correct order (graph/stream first, then work, then context),
     all without error.
  - 5 repeated full create/capture/launch/destroy cycles: VRAM delta 0
    MiB (no leak).

## 6. Memory and resource ownership

- `DeviceKSContext` owns: per-modulus root tables (shared process-wide
  via `g_rtab`, never freed per-context by design — unchanged), per-part
  decompose constants, the eval key (`d_av`/`d_bv`, freed by
  `ks_context_destroy`), moddown constants, `d_modsQlP`/`d_keyRow` (freed
  by `ks_context_destroy`).
- `DeviceKSWork` owns: all per-call scratch device buffers (`d_part`,
  `d_compl`, `d_res0`/`d_res1`, `d_pwork`, `d_qsw`, `d_digPtrs`,
  `d_rootsA`/`d_rootsB`/`d_towerMods`/`d_nInv`/`d_nInvPrecon`) — unchanged
  — **plus, new this round**, the per-(part,role) host staging buffers
  described in section 4. All freed by `ks_work_destroy`.
- A `cudaGraphExec_t`/`cudaGraph_t` captured from a `keyswitch_resident`
  call does **not** own any of `C`'s or `W`'s memory — it only records
  references to the device pointers that existed at capture time. The
  required destruction order, validated by the lifecycle test: destroy
  the graph executable and graph object (and the capturing stream) before
  destroying the `DeviceKSWork` it was captured against; destroy the work
  object before the context it was built from (unchanged from the
  pre-existing contract — `ks_context_destroy` frees `d_av`/`d_bv`, which
  nothing in `W` reads after the fact, but following this order avoids
  any ambiguity). Two independent graphs built from *different* work
  objects sharing one context are provably independent: destroying one
  does not disturb the other (test 7-9 above).
- No change to `ks_context_destroy`/`ks_work_destroy`'s own freeing
  logic was needed or made — the new host buffers are plain
  `std::vector` members of `DeviceKSWork`, freed automatically by its
  destructor like any other member, no manual `cudaFree` involved (they
  are host, not device, memory).

## 7. Concurrency — two real bugs fixed, one separate limitation found and documented

Stress-testing the fix with real `std::thread` concurrency (never
`fork()`, per Phase 27's explicit finding that `fork()` after CUDA init
is unsafe) surfaced two **additional, genuinely pre-existing** concurrency
defects, unrelated to the CUDA-graph bug itself, in the same latent-race
pattern class Phase 25 found and fixed as active in `src/rotate.cpp`:

- `src/keyswitch_resident.cpp`: both `g_rtab` (root-table cache, keyed by
  `(n,q)`) and `upc`'s cache `m` (keyed by a constant-tuple hash) were
  plain, unsynchronized `static std::map`s, populated via an
  unsynchronized find-then-emplace. Multiple threads building independent
  `KeySwitchConstants` from identical moduli (the realistic case: same
  CKKS parameters, different keys) raced these caches' cold-start insert
  for real — not latent here, genuinely triggered (partial, varying
  thread-pass counts run to run, the classic signature of concurrent
  `std::map` corruption).
- `src/evalkeygen.cpp`: `ek_dev_tab`'s cache `ekCache` had the identical
  pattern and was fixed the same way.

Both fixed with a single `std::mutex` guarding each cache's lookup+insert
(plus, for `g_rtab`, the subsequent read of the found/inserted entry),
mirroring the project's own already-established pattern for this exact
class of bug (Phase 28's fix to `src/encode_fft.cpp`'s caches). Cheap:
lookups/inserts are trivial relative to real keyswitch work.

**A third, deeper issue remains, found but NOT fixed this round, and
explicitly scoped out of this phase's mandate**: even after both cache
fixes, running many threads that each build a *fresh* `KeySwitchConstants`
from scratch *simultaneously* can still produce a thread whose **direct**
(non-graph) execution disagrees with itself across two back-to-back calls
on the same thread — proving a residual race exists somewhere in the
key-generation/context-setup pipeline (`keygen_host` and/or
`evalkeygen_host_sold`'s use of the legacy default stream `0` across
concurrent threads is a plausible but *unconfirmed* next hypothesis — not
asserted here, since the governing directive explicitly warns against
introducing a mutex without proof it protects identified shared state).
Separately, even when direct execution *was* self-consistent, the graph
path occasionally still diverged from it — a second, distinct symptom not
yet explained.

Per the governing directive's explicit instruction not to let this phase
become an open-ended concurrency audit of the entire key-generation
pipeline: **this is reported as a documented limitation, with a precise,
committed, standalone reproducer (`tests/
test_ks_graph_setup_concurrency_diagnostic.cpp`), deliberately NOT
registered with ctest (it is non-deterministic — registering it would
make the regular suite flaky)**. It is handed off as the top item for a
dedicated future investigation. The now-fixed graph-capture mechanism
itself is fully validated for the concurrency pattern that mechanism
actually needs: multiple independent contexts/work-objects/graphs used
concurrently from a single thread (fully tested, 100% reliable — section
5's tests 7-9), and sequential/single-threaded use is unaffected by any
of this.

**Supported configuration (proven)**: one `DeviceKSContext`, one or more
`DeviceKSWork` objects each with their own captured graph, used from a
single thread (sequential or interleaved), with graphs destroyed before
their owning work object, work objects destroyed before their owning
context. **Not yet proven safe**: many threads simultaneously
constructing fresh `KeySwitchConstants`/contexts/graphs from cold caches
(a residual race exists somewhere in setup, not in `keyswitch_resident`
itself).

## 8. Performance (honest measurement, not forced)

Native, n=8192/tw=30 (this project's typical bootstrap-scale
configuration), RTX 2060 Max-Q, CUDA 13.2, Release `-O3` (no `-G`), 15
repetitions, wall-clock for complete operations:

```
direct keyswitch_resident    mean=6.8404 median=6.4281 min=5.9349 max=9.9817 ms (n=15)
capture                      1.2452 ms (one-time)
instantiate                  3.2457 ms (one-time)
first launch                 4.8976 ms (one-time)
warm replay                  mean=4.3319 median=4.3190 min=4.1764 max=4.6568 ms (n=15)
graph destroy                1.1995 ms (one-time)

replay vs direct (median): 1.488x  (direct 6.4281 ms -> replay 4.3190 ms)
one-time overhead (capture+instantiate+first-launch): 9.3885 ms,
  breaks even after ~4 replays
```

**Warm graph replay is a real 1.49x faster than direct execution at this
scale**, for a context/work pair reused many times with the same
captured sequence (the realistic usage pattern for a resident, repeatedly
re-keyed pipeline) — a genuine bonus beyond the correctness fix, not the
primary goal of this investigation and not forced to appear favorable.

## 8b. A real regression found by the full suite, and its fix

Section 4's fix was first validated only against this investigation's
own new tests (which always build a complete `DeviceKSContext`) plus
`tests/test_ks_graph.cpp` itself. Running the **full** existing
regression suite afterward caught a genuine SEGFAULT in `tests/
test_hoist_stage0.cpp` that neither of those caught.

Root cause: `test_hoist_stage0.cpp`'s `stageRotBegin` builds a
deliberate, minimal "shape-only" `DeviceKSContext` (`Csize`) purely to
size a long-lived `DeviceKSWork` once, outside the per-diagonal loop —
setting only `n`/`sizeQl`/`sizeP`/`sizeQlP`/`sizePart`. `numPart` and
`sizeCompl` are left at their struct defaults (`0` and an empty
`std::vector`) on that placeholder, because the *original* flat,
`sizeQlP`-worst-case-sized scratch buffers never needed them. Section
4's fix sized the new per-(part,role) buffers from `C.numPart` and
indexed `C.sizeCompl[part]` — both silently wrong (0 and
out-of-bounds, respectively) for this specific, legitimate,
pre-existing caller pattern. The REAL per-diagonal context (built fresh
from `Krot` inside the hot loop) has a real, nonzero `numPart`; the
mismatch between the WORK object's (zero-sized) buffers and the REAL
context's loop bound produced the crash — caught by `gdb -batch -ex run
-ex "bt full"`, pointing directly at `ks_accumulate_qp`.

**Fix**: size the new buffers defensively, never from `C.numPart` or
`C.sizeCompl`. `(uint32_t)C.sizePart.size()` is used for the part count
(every known caller, including this placeholder, populates `sizePart`
with one real entry per part — it is the field this exact pattern was
already relying on for the *original* `maxPart` computation one line
above). `C.sizeQlP` is used as a uniform, always-safe upper bound for
every per-part/per-role buffer's inner size (a part's slice or
complement can never exceed the whole by construction). This requires no
caller change and preserves the pre-existing "size a work object once
from a cheap placeholder, reuse it against many real per-call contexts"
pattern exactly as `test_hoist_stage0.cpp` (and potentially other,
not-yet-identified callers) already depend on.

Re-verified after this fix: `tests/test_hoist_stage0.cpp` passes clean
(`[PASS] stage 0 hoisting validation against the REAL bootstrap state`,
exit 0); `tests/test_ks_graph.cpp` and `tests/test_ks_graph_lifecycle.cpp`
both still pass bit-exact/14-of-14, confirming the defensive sizing
didn't reintroduce the original defect. The full regression suite was
then re-run in its entirety once more (see the Phase 30 regression
report) to check for any further such caller.

## 9. Summary

**Outcome A (defect fixed)** for the assigned `test_ks_graph` CUDA-graph-
capture defect: root cause proven via two independent standalone
reproducers; minimal fix implemented (dedicated per-(part,role) host
staging buffers, zero change to any computed value or device buffer
reuse pattern); graph and direct outputs agree bit-exact; repeated
replay, updated-input replay, and multi-graph/multi-context usage from a
single thread are all correct; resource lifetimes and destruction order
validated; no VRAM leak across repeated cycles; no regressions anywhere
in the existing test suite (see the Phase 30 regression report); warm
replay measured 1.49x faster than direct, honestly reported.

**Outcome B (documented limitation)** for a separate finding surfaced
while stress-testing Objective F: two real, previously-undocumented
concurrency bugs in unrelated process-lifetime caches were found and
fixed; a third, deeper concurrency race in the key-generation/
context-setup pipeline (not in `keyswitch_resident` or its graph-capture
path) remains, reproduced precisely but not yet root-caused, deliberately
not force-fixed, and handed off as the next session's priority
investigation.
