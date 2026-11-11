# Cache concurrency audit, proof, and repair (Phase 33)

Resolves the four files Phase 32's architecture audit flagged as
containing unguarded, process-lifetime `static std::map` caches
(`research/GPU_RESIDENT_ARCHITECTURE.md` section 1: "4 of 13... `src/
rotate.cpp` (two caches, lines 29 and 165), `src/keyswitch.cpp` (line
47), `src/encrypt_host_shoup.cpp` (line 72), `src/encdec.cpp` (line
71)"). That is **4 files, 5 distinct cache instances** (`rotate.cpp`
contributes two) — this document tracks all 5 individually rather than
forcing a false "exactly four" count.

## 1. Inventory

| # | Cache | File:line (at Phase 33 start) | Key | Value | Readers/writers |
|---|---|---|---|---|---|
| 1 | `dev_tab()`'s `dcache` | `src/rotate.cpp:24` | `(n,q)` | `DevTab` (4 device root-table pointers) | `automorphism_eval_device` (host-orchestrated automorphism, 2 call sites in-file) |
| 2 | `autoPermMap()`'s `cache` | `src/rotate.cpp:79` | `(n,k)` | `uint32_t*` (device permutation-index array) | `automorphism_eval_fast_device` (1 call site in-file) |
| 3 | `dev_tab()`'s `dcache` | `src/keyswitch.cpp:46` | `(n,q)` | `DevTab` | `keyswitch_core_resident` (4 call sites, same function) |
| 4 | `enc_dev_tab_shoup()`'s `cache` | `src/encrypt_host_shoup.cpp:189` | `(n,q)` | `EncDevTab` (2 device root-table pointers) | `encrypt_host_shoup` |
| 5 | `enc_dev_tab()`'s `cache` | `src/encdec.cpp:244` | `(n,q)` | `EncDevTab` | `encrypt_host` |

All five share the IDENTICAL shape: a function-local `static std::map<Key,Value>`,
populated via `find()` then, on miss, `emplace()`, returning a reference
into the map — no mutex, no atomic, no synchronization of any kind. All
five hold process-lifetime, never-freed device pointers (correct choice
for a never-evicted cache; the concurrency question is purely about the
`std::map` container itself, not the device memory it points to).

## 2. Call-graph tracing (real reachability, not assumption)

Traced every caller of each cache-bearing function across `src/`,
`tests/`, `bench/`, `examples/`, and the Rust FFI bridge
(`rust/crates/ckks-cuda/cpp_bridge/`):

- **Cache 1** (`rotate.cpp` `dev_tab`): called by `automorphism_eval_device`,
  reached by `automorphism_eval_host`, reached by `xf_multi`'s callers —
  `pt_to_eval_host` and `rotate_ct_host`. `rotate_ct_host` is itself
  called concurrently by `tests/test_rotate_concurrent_cache.cpp`'s
  **Phase D** (9 threads, 3 distinct `n` values x 3 fixtures each).
- **Cache 2** (`rotate.cpp` `autoPermMap`): called by
  `automorphism_eval_fast_device`, reached by `rotate_ct_resident`.
  Called concurrently by the SAME test's **Phase B** (9 threads). Because
  the fixture's rotation amount `k` is fixed at `5` for every fixture
  while only `n` varies across 3 groups of 3, **this phase drives a
  genuine identical-key collision**: 3 threads per group call
  `autoPermMap(n, 5, ...)` with the exact same key simultaneously.
- **Cache 3** (`keyswitch.cpp` `dev_tab`, inside `keyswitch_core_resident`):
  this is a DIFFERENT function from `keyswitch_resident.cpp`'s
  `keyswitch_resident()` (already mutex-guarded since Phase 30) — an
  initial hypothesis that `rotate_ct_resident` reaches this cache was
  checked directly against `rotate_ct_resident`'s body and found FALSE
  (it calls `keyswitch_resident`, not `keyswitch_core_resident`). The
  REAL reachability: `rotate_ct_host` (Phase D, above) calls
  `keyswitch_core_resident` directly, AND the Rust FFI bridge's
  `gpufhe_ks_fixture_run_reference` (`cpp_bridge/keyswitch_bridge.cpp`)
  calls it as a correctness oracle for Rust-side tests.
- **Cache 4** (`encrypt_host_shoup.cpp` `enc_dev_tab_shoup`): called by
  `encrypt_host_shoup`. Confirmed NOT reachable from the Rust FFI
  production path (the bridge's own comment: `encrypt_host_shoup_pooled`
  is the production call, "NOT called from here anymore" for the plain
  variants). Grepped every test and benchmark calling
  `encrypt_host_shoup` directly: **zero existing multi-threaded callers**
  before this phase — the one cache with no prior concurrent-reachability
  evidence at all.
- **Cache 5** (`encdec.cpp` `enc_dev_tab`): called by `encrypt_host`.
  Also confirmed NOT reachable from the Rust FFI production path (same
  bridge comment, for `encrypt_host` specifically: "NOT called from here
  anymore"). BUT: `tests/test_encrypt_host_shoup_pooled_thread_safety.cpp`
  (written in Phase 13 to stress a DIFFERENT cache —
  `encrypt_host_shoup_pooled`'s buffer pool) calls `encrypt_host` as its
  single-threaded-reference computation **from inside its own 8-thread
  concurrent loop**, at the IDENTICAL `(n=2048, sizeQ=8)` key every
  single call (48 total concurrent calls, same key) — a real, pre-existing,
  if incidental, concurrent reproducer for this exact cache.

**Bottom line on reachability**: caches 1, 2, 3, and 5 all have genuine,
real, already-existing concurrent call paths (three via one established
Phase 25 test, one via a Phase 13 test's incidental side effect, and
cache 3 additionally via the Rust FFI bridge). Cache 4 had none — the
only one requiring a brand-new reproducer (section 5).

## 3. Mechanism-level proof (not just "it hasn't visibly broken")

Re-ran `tests/test_rotate_concurrent_cache.cpp` and `tests/
test_encrypt_host_shoup_pooled_thread_safety.cpp` fresh, 6 times each,
against the UNMODIFIED (pre-fix) source: both passed bit-exact every
time, exactly matching the historical record (Phase 25/30/31's own
re-runs). **A passing value-level check is not proof of absence of a
data race** — the test file's own comment says so explicitly
("concurrency defects are probabilistic by nature. A pass is evidence,
not proof").

To get decisive mechanism-level proof, ThreadSanitizer was the right
tool — but it cannot run directly against this project's CUDA-linked
binaries in this WSL2 environment (`FATAL: ThreadSanitizer: unexpected
memory mapping`, reproduced even against a trivial CUDA-free program
under default WSL2 ASLR settings — resolved by running under `setarch
$(uname -m) -R` to disable address-space-layout randomization for the
one invocation; a documented TSan/high-entropy-ASLR interaction on
modern kernels, not a CUDA-specific issue, and not a defect in this
environment that required any system-wide configuration change).

Extracted the EXACT pattern (function-local `static std::map<Key,Value>`,
`find()` then `emplace()`, no synchronization) into a minimal, fully
standalone, CUDA-free reproducer mirroring `test_rotate_concurrent_cache.cpp`'s
own 9-thread/3-distinct-n/3-threads-per-n layout (so the same identical-key
collision shape cache 2 above actually exercises). **ThreadSanitizer
reported multiple confirmed data races** in `std::map`'s internal
`_Rb_tree`: a `find()`'s read racing another thread's `emplace()`-driven
tree-node insertion/rebalancing, and a heap-allocation race on the newly
created tree node — the textbook signature of concurrent, unsynchronized
`std::map` mutation. Re-ran the identical reproducer with ONLY a
`std::mutex` added around the whole `find`+`emplace` sequence (the exact
fix applied to production in section 4): **zero races reported**,
confirming both that the unfixed pattern is genuinely unsafe and that the
chosen fix actually resolves it.

This constitutes the "prove any race that exists" evidence the governing
directive requires — not a theoretical C++-memory-model argument, an
actual race detector confirming the mechanism under the real access
pattern.

## 4. The fix

All five cache instances were fixed identically, using the SAME pattern
this project already established and proved low-risk in Phases 28/30
(`src/encode_fft.cpp`, `src/keyswitch_resident.cpp`'s `g_ks_cache_mutex()`
guarding `g_rtab`/`upc`): a single `std::mutex` per file (shared across
both of `rotate.cpp`'s two caches, matching
`keyswitch_resident.cpp`'s own one-mutex-guards-multiple-caches
precedent), taken as a `std::lock_guard` around the entire
`find`-then-possibly-`build`-then-`emplace`-then-return sequence.

No change was made to `keyswitch_resident.cpp`'s already-fixed `g_rtab`/
`upc`, to any cryptographic parameter, or to any cache's KEY SHAPE,
VALUE SHAPE, or eviction policy (there is none — these remain
intentionally never-freed, process-lifetime caches, unchanged from their
original design).

Files changed: `src/rotate.cpp`, `src/keyswitch.cpp`,
`src/encrypt_host_shoup.cpp`, `src/encdec.cpp` (one `#include <mutex>`
and one mutex-guard addition each; zero lines of existing cache-building
logic altered).

## 5. New regression test (closing the one real gap)

`tests/test_encrypt_host_shoup_concurrency.cpp`: 8 threads x 8 iterations,
each thread calling `encrypt_host_shoup` TWICE with the identical seed
at the identical `(n=4096, sizeQ=6)` key, checking the two calls agree
with EACH OTHER (not just a separately-computed reference) — directly
closing cache 4's previously-nonexistent concurrent-reachability
coverage. Registered with `ctest`. 5/5 fresh runs clean, 0 mismatches
each (0 total mismatches across 8 threads x 8 iters x 2 calls = 128
comparisons per run, 640 across the 5 runs).

## 6. Performance impact (measured, not assumed)

A real before/after A/B, not a theoretical argument: `git stash`d the
four fixed files, rebuilt, and re-ran `bench_evalmult` (exercises cache
3 via `keyswitch_core_resident`), `bench_hoist_ab` (exercises caches 1/2
via rotation), and `bench_encrypt_shoup_vs_reference` (exercises caches
4/5), 3 runs each, then restored the fix and re-ran the identical 3
benchmarks 3 times each for direct comparison:

| Benchmark | Before (unfixed) | After (fixed) |
|---|---|---|
| `bench_evalmult` GPU relin median | 62.87 / 59.20 / 49.48 ms | 63.37 / 49.61 / 60.46 ms |
| `bench_hoist_ab` | [PASS], bit-exact, all 3 runs | [PASS], bit-exact, all 3 runs |
| `bench_encrypt_shoup_vs_reference` speedup | 1.109x / 1.150x / 1.143x | 1.163x / 1.141x / 1.178x |

No systematic difference in either direction — both sets of numbers sit
inside this machine's own already-documented ~30% run-to-run timing
spread (Phase 25's environment notes). This matches the expectation
stated in every prior phase's identical fix (Phase 28/30): the lock
guards a cheap, cold-path lookup-or-one-time-build, never the actual GPU
compute, so its cost is unmeasurable against real work.

## 7. Second-caller evaluation for `DeviceBuffer<T>`

Per the god-prompt's secondary objective: audited `include/device_buffer.h`
(Phase 32) for a second production-worthy caller. `DeviceCiphertext`
(`src/device_ciphertext.cpp`) remains the only existing production
device-buffer-shaped type, and it is NOT a mechanical drop-in replacement
candidate — it owns exactly two fixed-role arrays (`c0`/`c1`) with
synchronous-only transfers by design, a narrower contract than
`DeviceBuffer<T>`'s general single-pointer/async-capable shape; forcibly
migrating it would not simplify anything and risks disturbing a
proven-correct, widely-used type for no measurable benefit. The five
caches fixed in this phase hold RAW device pointers inside hand-rolled
structs (`DevTab`, `EncDevTab`) — each holds MULTIPLE distinct pointers
per entry (e.g. `DevTab`'s `fr`/`fp`/`ir`/`ip`), not a single buffer, so
wrapping them in `DeviceBuffer<T>` would need either four separate
`DeviceBuffer` members per struct (a real, non-trivial refactor of
working code with no test currently covering that specific shape change)
or a `DeviceBuffer<DevTab>`-of-one semantics that doesn't fit
`DeviceBuffer`'s current single-element-type-array design. **No safe,
small, genuinely production-value migration was found this phase** —
deferred, per the god-prompt's own explicit "if it requires a broad
redesign, defer it and document the exact missing capability" instruction.
The missing capability, if ever pursued: `DeviceBuffer<T>` would need a
documented convention for "a cache entry holding N related device arrays"
(e.g. a small fixed-size array of `DeviceBuffer<T>` members, or a
separate `DeviceBufferGroup` type) before migrating any of these five
caches' value types onto it.

## 8. Regression (re-run this round)

See `research/NEXT_SESSION.md` for the exact current totals (C++ `ctest`
and Rust debug/release), re-run fresh against the fixed state. The new
`test_encrypt_host_shoup_concurrency` test is included and passing.

## 9. Remaining limitations

- ThreadSanitizer could not be run directly against the production
  binaries in this environment; the proof is from a faithful, pattern-
  identical, CUDA-free extraction, not the literal production object
  code. The extraction was verified byte-for-byte pattern-identical to
  each cache's `find`/`emplace` logic, but this is still one level of
  indirection from direct instrumentation of `src/rotate.cpp` itself.
- `src/keyswitch_resident.cpp`'s `g_rtab`/`upc` (already fixed in Phase
  30) were NOT re-verified under this round's TSan reproducer — they use
  the identical pattern and were not re-litigated since Phase 30 already
  established their fix with its own evidence.
- The second-caller `DeviceBuffer<T>` migration (section 7) remains
  unimplemented, by evidence-based choice, not oversight.
