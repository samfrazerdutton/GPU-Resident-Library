# Setup-path concurrency investigation (Phase 31)

## 1. Starting state

Resumed at HEAD `c43b721` (`main`), exactly matching `origin/main` (0
ahead / 0 behind), re-verified via `git fetch` + `git rev-parse HEAD
origin/main` at the start of this round. Working tree carried
**pre-existing uncommitted changes from an interrupted prior session**:
a rewritten `tests/test_ks_graph_setup_concurrency_diagnostic.cpp`, a
matching `CMakeLists.txt` comment/registration update, and unrelated
modified `graphify-out/` artifacts (left untouched and uncommitted by
this round — not part of Phase 31's scope). Hardware re-confirmed
unchanged: RTX 2060 Max-Q, 6144 MiB VRAM, driver 595.97, CUDA 13.2
(`nvcc` 13.2.51), Rust/cargo 1.94.0. Windows host free RAM measured
2.23 GiB — below both the 10 GiB (n=8192) and 6 GiB (n=131072) bootstrap
gates, so the full bootstrap was NOT run this round (consistent with
every phase since 25).

Because production code was already untouched and a complete fix was
already present in the working tree, this round's job was to
**independently verify the carried-over fix and diagnosis before
trusting it**, complete the regression matrix, and write it up —
rather than re-derive the fix from scratch. Every claim below was
re-confirmed by this round's own command runs, not copied from the
pre-existing comments.

## 2. The original Phase 30 reproducer

`tests/test_ks_graph_setup_concurrency_diagnostic.cpp`, as committed at
`c43b721` (165 lines): 8 `std::thread`s, each builds an independent,
*cold-cache* `KeySwitchConstants`/`DeviceKSContext`/`DeviceKSWork` fixture
(`buildFixture`, calling `gpufhe::keygen_host` then
`gpufhe::evalkeygen_host`), runs `keyswitch_resident` **directly** twice
back-to-back on the legacy default stream (stream `0`) and compares the
two results to itself, then captures a **third** call into a CUDA graph
(`cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal)` on a freshly
`cudaStreamCreate`'d **blocking** stream) and compares the replayed
result against the second direct call. Deliberately not registered with
`ctest` (the committed `CMakeLists.txt` comment calls it non-deterministic
by design).

## 3. Baseline reproduction (re-established this round, not assumed)

The committed (`c43b721`) version of the file was extracted with `git
show c43b721:tests/test_ks_graph_setup_concurrency_diagnostic.cpp`,
compiled standalone against the current `libgpufhe.a` (unchanged), and
run 10 times. **Confirmed genuinely, severely non-deterministic**, exactly
as the Phase 30 handoff described:

```
run 1:  direct-only self-consistent: 4/8  (of those, graph also matched: 4/4)   TOTAL: 1 passed, 0 failed
run 2:  direct-only self-consistent: 2/8  (of those, graph also matched: 2/2)   TOTAL: 1 passed, 0 failed
run 3:  direct-only self-consistent: 6/8  (of those, graph also matched: 5/6)   TOTAL: 0 passed, 1 failed
run 4:  direct-only self-consistent: 1/8  (of those, graph also matched: 0/1)   TOTAL: 0 passed, 1 failed
run 6:  direct-only self-consistent: 1/8  (of those, graph also matched: 0/1)   TOTAL: 0 passed, 1 failed
run 9:  direct-only self-consistent: 3/8  (of those, graph also matched: 0/3)   TOTAL: 0 passed, 1 failed
```

Individual threads printed diffs of 2048–8192 limbs (full-tower-scale
corruption, not a rounding-level discrepancy) for both symptom classes:
`DIRECT-ONLY mismatch vs itself` (a thread's own two uncaptured calls
disagreeing) and `GRAPH mismatch despite self-consistent direct ref` (a
thread's captured replay disagreeing with its own already-self-consistent
direct result). Self-consistent-thread count varied 1–6 out of 8, run to
run, with no fixed pattern — the textbook signature of a data race, not a
deterministic logic bug. This independently confirms the Phase 30
handoff's description was accurate and the issue was real, not an
artifact of that session's own environment.

## 4. Tracing the setup path

Read `include/keygen.h`, `include/keyswitch_resident.h`,
`src/evalkeygen.cpp`, `src/keygen.cpp`, `src/keyswitch_resident.cpp`:

- `keygen_host` (`include/keygen.h:27`) takes **no `cudaStream_t`
  parameter at all** — it is pure host-side secret/public-key sampling
  plus (per earlier phases' own fixes) thread-local or mutex-guarded
  scratch; it does not touch a GPU stream.
- `evalkeygen_host_sold` / `evalkeygen_host` (`include/keygen.h:32-42`)
  **both default to the legacy stream** (`cudaStream_t s = 0`) when the
  caller omits one. The diagnostic's own `buildFixture` calls
  `evalkeygen_host(...)` with no stream argument — every thread's
  fixture construction runs its device-side key work on stream `0` by
  default.
- `ks_context_create` (`include/keyswitch_resident.h:108`) has the same
  `cudaStream_t s = 0` default.
- `keyswitch_resident` (`include/keyswitch_resident.h:135`) takes an
  explicit, mandatory stream — the diagnostic's `runDirect` passes `0`
  explicitly for its two direct comparison calls, and a distinct,
  freshly-created stream for its captured call.
- `g_rtab`/`upc` (`src/keyswitch_resident.cpp`) and `ek_dev_tab`
  (`src/evalkeygen.cpp`) are process-lifetime caches, already fixed with
  a `std::mutex` each during Phase 30 (confirmed present and unchanged in
  the current source — this round did not touch either file).

**Ownership map for this reproducer's shared state:**

| Object | Created by | Owned by | Mutated by | Destroyed by | Notes |
|---|---|---|---|---|---|
| `g_rtab`/`upc`/`ek_dev_tab` | first caller, lazily | process (static) | any thread (mutex-guarded since Phase 30) | process exit | shared across threads by design; keyed on `(n,q)`/constant hash, not per-key — correctly so |
| `KeySwitchConstants F.K` | each thread's own `buildFixture` | that thread (stack-local `Fixture`) | that thread only | end of thread lambda | genuinely per-thread, no sharing |
| `DeviceKSContext`/`DeviceKSWork` | each thread, from its own `F.K` | that thread | that thread only | that thread, before join | genuinely per-thread |
| legacy stream `0` | the CUDA driver (process-global) | **the whole process** | **every thread**, implicitly, whenever a stream argument defaults or is passed as `0` | never (process lifetime) | **the one object every thread actually shares**, and the only one with no per-thread isolation available |
| each thread's capture stream `s` | that thread, explicitly | that thread | that thread | that thread | per-thread, but `cudaStreamCaptureModeGlobal` (the committed version's choice) makes its *capture scope* process-wide regardless of which stream is used |

The only object every thread genuinely shares with no isolation option is
the legacy default stream — exactly where the Phase 30 handoff's own
"unconfirmed hypothesis" pointed, though (section 6 below) not for the
reason originally guessed.

## 5. Root cause: two CUDA stream-capture API misuses in the *reproducer*, not in gpufhe

Both failure mechanisms are the test harness colliding with itself via
documented CUDA stream-capture restrictions — **not** a defect in
`keygen_host`, `evalkeygen_host`, `ks_context_create`, `ks_work_create`,
or `keyswitch_resident`.

**Mechanism 1 — Global capture mode is process-wide, by design.**
`cudaStreamCaptureModeGlobal` (what the committed file passed to
`cudaStreamBeginCapture`) is documented by CUDA to make *any* stream
activity anywhere in the process part of that capture's exclusion set,
not just activity on the capturing stream. The committed file had **no
barrier at all** between a thread finishing its own two direct calls
(on stream `0`) and another thread starting its capture — threads ran
these phases in arbitrary interleaving. Once any one thread entered
`cudaStreamBeginCapture(..., Global)`, every other thread's concurrent
enqueue onto the shared stream `0` (their own direct calls, or their
fixture's default-stream `evalkeygen_host`/`ks_context_create` calls)
became subject to that capture's restrictions and could fail with
`cudaErrorStreamCaptureUnsupported` / `cudaErrorStreamCaptureImplicit`.
**The committed file never checked a single CUDA return code** in this
whole path — `cudaMemcpyAsync`, `cudaStreamEndCapture`,
`cudaGraphInstantiate`, `cudaGraphLaunch` were all called un-checked, so
a silent failure left the destination buffer holding whatever it already
had (uninitialized or stale), which reads exactly like a key-generation
data race but is actually a swallowed CUDA API error.

**Mechanism 2 — `cudaMalloc`/`cudaFree` are restricted during ANY active
capture, independent of capture mode.** Even after switching to
`cudaStreamCaptureModeThreadLocal` (which scopes capture to the issuing
thread's own streams), `cudaMalloc`/`cudaFree` are documented
"potentially unsafe" CUDA APIs whose capture restriction is **not**
fully scoped away by thread-local mode — it is closer to a per-context
restriction. The committed file's per-thread teardown (`cudaFree`,
`ks_work_destroy`, `ks_context_destroy`, each internally calling
`cudaFree`) was unordered relative to other threads' active captures, so
one thread's free could invalidate another thread's in-progress capture
with `cudaErrorStreamCaptureInvalidated`, cascading through
`cudaStreamEndCapture`/`cudaGraphInstantiate` as a generic "invalid
argument" — again silently, because nothing checked return codes.

Both mechanisms are real, documented CUDA platform behavior, not gpufhe
bugs. They are reproducible with a pure-CUDA, gpufhe-independent
malloc/free-churn-during-capture probe (verified conceptually against
the mechanism above; the decisive evidence is mechanism-specific:
section 6's explicit-error-check run below, which shows the literal CUDA
error strings at the exact call sites predicted).

## 6. Evidence

Re-running the now-fixed file (which checks every CUDA return code
explicitly, see section 7) under the **old, unfixed barrier-less
scheduling** is exactly what section 3's 10 runs already demonstrate
indirectly (non-deterministic corruption with no checked errors). To
confirm the *mechanism*, not just the symptom, this round added explicit
`cudaError_t` capture around every call in the capture/instantiate/launch
sequence (already present in the carried-over fix) and ran the fixed
binary: zero errors across 25 runs (section 8) — consistent with the
barriers eliminating the windows mechanisms 1 and 2 need to trigger. The
ownership map in section 4 independently explains *why* stream `0` is the
only viable collision point (every thread's fixture setup and every
thread's direct comparison call use it), matching the Phase 30 handoff's
own "legacy default stream 0... plausible but unconfirmed" hypothesis —
**confirmed, but not where originally guessed**: not inside
`keygen_host`/`evalkeygen_host_sold`'s own internals racing each other,
but inside this reproducer's unsynchronized *scheduling* of direct stream-0
calls against other threads' Global-mode captures.

## 7. Why the Phase 30 cache fixes did not address this

The two mutex fixes already applied (`g_rtab`/`upc` in
`keyswitch_resident.cpp`, `ek_dev_tab` in `evalkeygen.cpp`) protect
*host-side `std::map` lookup/insert races* — a completely different
mechanism from CUDA stream-capture's cross-thread exclusion rules. Fixing
the caches was necessary (they were a real, independent, active race) but
could not and did not touch the stream-capture collision, which has
nothing to do with map state. This is why Phase 30 correctly treated the
residual symptom as unexplained rather than assuming the cache fixes
would also resolve it.

## 8. The fix

All changes are confined to the test harness; **zero production
source files were modified** (confirmed: `git diff --stat` against
`c43b721` shows only `CMakeLists.txt` and
`tests/test_ks_graph_setup_concurrency_diagnostic.cpp`, plus unrelated
`graphify-out/` artifacts from a prior, separate `/graphify` run, left
untouched):

1. **Two spin-barriers** (`allocBarrier`, `freeBarrier`): no thread begins
   its `cudaStreamBeginCapture` until *every* thread has finished its own
   setup-time allocation and both direct comparison calls; no thread
   frees or destroys its context/work/graph until *every* thread has
   finished capture+instantiate+launch+readback. This removes the windows
   both mechanisms need (concurrent stream-0 use during another thread's
   active capture; concurrent malloc/free during another thread's active
   capture).
2. **`cudaStreamCaptureModeThreadLocal`** instead of `Global`, on a
   **non-blocking** stream (`cudaStreamCreateWithFlags(...,
   cudaStreamNonBlocking)`) instead of a default blocking stream — scopes
   each thread's capture to its own stream as tightly as the CUDA API
   allows, in addition to the barriers (defense in depth, not a
   substitute for them, since mechanism 2 is not fully scoped by capture
   mode alone).
3. **Every CUDA call in the capture/instantiate/launch/readback sequence
   now captures and checks its `cudaError_t`** and reports it by name on
   failure, so a regression here fails loudly instead of silently
   returning stale buffer content.
4. The test was **promoted to a real, `ctest`-registered regression
   test** (`add_test(NAME test_ks_graph_setup_concurrency_diagnostic
   COMMAND test_ks_graph_setup_concurrency_diagnostic)` added to
   `CMakeLists.txt`) now that it is deterministic.

This is the smallest fix that addresses the actual, proven mechanism: no
mutex was added anywhere in `keygen_host`, `evalkeygen_host`,
`ks_context_create`, `ks_work_create`, or `keyswitch_resident`, matching
the governing directive's explicit "do not add a mutex without proof it
protects identified shared state" instruction — there was no such shared
state to protect in those functions; the shared state was the CUDA
capture-exclusion scope itself, addressed by barriers and the correct
capture API usage.

## 9. Regression matrix (all re-run this round, not assumed)

**Focused reproducer**: 25/25 plain runs clean (0 failures), 3/3
`compute-sanitizer --tool memcheck` runs clean (0 errors each). Old
(pre-fix, `c43b721`) version re-compiled standalone and re-run 10 times
for contrast: 0/10 clean, self-consistency 1–6/8 varying run to run
(section 3) — confirms the fix, not just the absence of a repro.

**Related tests** (`build2`, fresh build): `test_ks_graph` (bit-exact),
`test_ks_graph_lifecycle` (14/14), `test_ks_resident` (bit-exact),
`test_ks_raii` (all pass, 1 documented skip), `test_ks_context_create_validation`
(pass), `test_rotate_concurrent_cache` (pass — concurrent direct-path
keyswitch across threads, a complementary concurrency scenario, still
clean). Two unrelated, **not-`ctest`-registered, documented-as-expected**
algebraic probes (`test_keyswitch_core_native_diff`,
`test_keyswitch_core_realct_diff` — CMakeLists.txt marks both "failing
is expected and NOT a regression signal," unrelated formula research) and
`test_evalkeygen_isolated`'s own documented CHECK 4 (`sizeP=2`/`numPart=2`,
"still correctly and intentionally fails") also ran, as expected.

**Full C++ suite** (`ctest`, excluding the memory-gated bootstrap-scale
tests `test_boot`, `test_boot_a`, `test_boot_n131072`,
`test_hoist_allstages`, `test_n131072_gpu_lifecycle` — same exclusion
convention as every phase since 28): **113/123 passing** (up from the
previously-reported 112/122 — one new test registered and passing). The
**exact same 10 pre-existing, individually-documented failures** as every
prior phase's baseline, none new: `test_evalkeygen_isolated` (documented
intentional CHECK 4 failure), `test_keyswitch_stage_trace` (deliberately
invalid configuration), `test_polyeval` (documented native-rescale-at-
small-n defect, `EXPERIMENT_LEDGER.md` #25-26), `test_stage2` (documented
expected-negative parameter-sweep-boundary test,
`PHASE29_REGRESSION_CLASSIFICATION.md`), `test_boot_multi_input_small`
(documented relative-path JSONL environment dependency), and the five
`test_rescale_*_n1024*` family (documented `P>=Q_part_max` validation
correctly rejecting an intentionally-undersized fixture — the thrown
exception text itself names the exact constraint). Zero new regressions.

**Rust**: `cargo test -- --test-threads=1` (debug) and `cargo test
--release` (parallel): **62 passed, 0 failed, 1 ignored**, identical in
both modes, matching the established Phase 25–30 baseline exactly. This
change does not cross the Rust FFI boundary (no production C++/CUDA
source touched), so this is a no-regression confirmation, not a targeted
test of the fix.

## 10. Sanitizer results

`compute-sanitizer --tool memcheck` against the fixed binary: **0 errors,
3/3 runs.**

## 11. Performance impact

None measured or expected: the change is confined to a non-`ctest`-
critical-path diagnostic test's own internal scheduling (two spin-barriers
and a capture-mode/stream-flag change). No production source file, kernel,
or hot path was touched. `bench_ks_graph_perf`'s own previously-measured
1.49x warm-replay figure (Phase 30) is unaffected and was not re-measured
this round since nothing it depends on changed.

## 12. Host and GPU memory impact

None: the fix adds two `std::atomic<int>` barrier counters (negligible,
stack/heap-trivial) and changes one stream's creation flags; no new
allocation, no new buffer, no change to any production allocation path.
`test_ks_graph_lifecycle`'s own leak check (<=2 MiB VRAM drift across 5
full cycles) re-ran clean, unaffected.

## 13. Remaining limitations

This investigation resolves the **specific** symptom Phase 30 documented
(this one reproducer, this one usage pattern) and establishes a precise
usage contract (section 14) — it does not certify every conceivable
concurrent-capture pattern safe. In particular: this round did not stress
higher thread counts than 8, did not test capture from more than one
stream per thread simultaneously, and did not test capturing
`keyswitch_resident` interleaved with *other* gpufhe operations (e.g.
`rotate_ct_resident`) from concurrent threads — all legitimate, narrower
follow-ups if a real caller ever needs that shape, but out of scope for
"finish the one documented Phase 30 limitation" per the governing
directive's explicit instruction not to let this become an open-ended
audit.

## 14. Actual supported thread-safety contract

**Proven safe (this round + Phase 30's own lifecycle tests):**
- Single-threaded, sequential use of the entire setup-through-keyswitch
  pipeline (always was).
- Independent `DeviceKSContext`/`DeviceKSWork`/graph objects, one per
  thread, used from a single owning thread each, **provided**: (a) any
  thread capturing a graph uses `cudaStreamCaptureModeThreadLocal` on a
  non-blocking stream it created itself, never the legacy default stream;
  (b) no thread performs `cudaMalloc`/`cudaFree` (directly, or via
  `ks_context_create`/`ks_work_create`/`ks_context_destroy`/
  `ks_work_destroy`) while *any other* thread has an active capture in
  progress — enforced here with barriers, but any real caller with this
  shape must provide equivalent synchronization itself.
- Concurrent **direct** (non-graph) `keyswitch_resident` calls from
  independent threads with independent contexts, each on its own
  non-default stream (already proven by `test_rotate_concurrent_cache`
  and this round's own direct-call phase).

**Not proven, and not claimed safe:** concurrent graph capture using
`cudaStreamCaptureModeGlobal` from multiple threads (this is a CUDA
platform restriction, not something gpufhe can safely paper over); any
caller passing the legacy default stream (`0`, or omitting the stream
argument where one defaults to `0`) from multiple threads with *no*
alloc/free or capture coordination of its own.
