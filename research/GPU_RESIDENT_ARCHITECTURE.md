# GPU-resident execution architecture — audit and proposal (Phase 32, Work Package A)

Audit-only: no production source file was modified while writing this
document. Every claim below is grounded in a direct read of the current
source tree or a grep count run against it this round — nothing here is
inferred from memory of prior phases without re-checking.

## 1. Current architecture

There is **no general-purpose GPU-resident execution abstraction** in
this codebase today. What exists is a set of independently-evolved,
operation-specific patterns, each reinvented by hand:

- **`DeviceCiphertext`** (`include/device_ciphertext.h`,
  `src/device_ciphertext.cpp`): the cleanest existing example. RAII,
  move-only, owns exactly two `uint64_t*` device arrays
  (`c0_`/`c1_`), synchronous `cudaMemcpy` upload/download, a local
  `ck()` helper that throws `std::runtime_error` on any CUDA failure.
  Fixed shape (ciphertext-specific: always 2 towers-worth arrays), no
  stream parameter at all — every transfer is synchronous. This is the
  **only file in the entire `src/`/`include/` tree with a consistent,
  checked-error-on-every-call convention** (verified: grepping for a
  local `ck`/`checkCuda`/`check_cuda` helper across `src/` + `include/`
  returns exactly one file).
- **`RegisteredHostBuffer`** (`include/registered_host_buffer.h`): a
  genuinely well-designed host-side pinned-buffer RAII primitive with
  an explicit `Empty→Allocated→Registered→(release)` state machine,
  idempotent release, and a stream-aware `release(cudaStream_t)` that
  synchronizes before unregistering if an async transfer might still be
  touching the memory. This is the project's best existing example of
  the "async guarantee" pattern Work Package A.D asks about — but it
  only covers the *host* side of a transfer, not a device buffer.
- **`ScopedKSContext`/`ScopedKSWork`** (`include/ks_raii.h`): thin
  RAII wrappers around the keyswitch-specific C API
  (`ks_context_create`/`ks_context_destroy`,
  `ks_work_create`/`ks_work_destroy`). The header's own comment is
  explicit about what it does and does not fix: it delegates entirely to
  functions that still don't report `cudaMalloc` failure mid-construction,
  and `cudaFree`'s return value is still never checked by the underlying
  destroy functions. Tied one-to-one to `DeviceKSContext`/`DeviceKSWork`
  — not a general buffer or context abstraction, a specific wrapper for
  one feature's specific structs.
- **`KsResidentCache`** (`include/ks_resident_cache.h`): the most
  sophisticated existing resource-lifecycle primitive in the project —
  a bounded (by count AND by resident bytes) LRU cache of
  `ScopedKSContext`, fails closed on projected VRAM pressure via
  `cudaMemGetInfo` *before* attempting allocation, records a
  `cudaStream_t` per entry and synchronizes it before any evict/destroy
  to avoid racing in-flight kernels against `cudaFree`, and aborts the
  process on cross-thread use rather than silently corrupting state.
  Per Phase 27's own `KS_RESIDENT_CACHE_DESIGN.md` (section 7), this was
  evaluated and **explicitly NOT integrated into production** — it
  exists, is tested (34/34 checks), but nothing calls it today. It is
  also hard-coded to `DeviceKSContext`, not generic over buffer/resource
  type.
- **Nine independent copies of the same unsynchronized-then-ad-hoc-fixed
  cache pattern**: `grep -rn 'static std::map' src/` finds 13 files using
  a function-local `static std::map<Key,Value>` as a process-lifetime
  device-resource cache (root tables, eval-key device tables, encrypt
  buffer pools, FFT twiddle tables, rotation device tables). Of these,
  9 now have a `std::mutex` guarding the lookup+insert
  (`encode_fft.cpp`, `evalkeygen.cpp`, `keyswitch_resident.cpp`, and
  all four `encrypt_host_shoup_pooled*` variants) — each fixed
  **independently, after being found actively racing**, in Phases 28 and
  30. **Four do not**: `src/rotate.cpp` (two caches, lines 29 and 165),
  `src/keyswitch.cpp` (line 47), `src/encrypt_host_shoup.cpp` (line 72),
  `src/encdec.cpp` (line 71). `rotate.cpp`'s case is a *documented,
  deliberate* decision (Phase 25's handoff: "latent, not active" —
  `test_rotate_concurrent_cache`'s Phase B exercises exactly this path
  and passes 6/6, so it was left as a control, not re-investigated since).
  The other three were never flagged either way in any phase's
  handoff — they appear to be superseded-but-still-compiled earlier
  implementations (the project's own Phase 29 handoff names
  `encrypt_host_shoup_pooled_barrett_batched` as "what the Rust bridge
  actually calls," implying `encrypt_host_shoup.cpp`/`encdec.cpp` are
  not on the hot production path), but this was not independently
  re-verified this round and should not be assumed safe.
  **Update (Phase 33): all 13/13 are now mutex-guarded.** The remaining
  four were independently re-traced (not assumed): `rotate.cpp`'s two
  caches and `keyswitch.cpp`'s are reachable via real, already-existing
  concurrent test paths (`tests/test_rotate_concurrent_cache.cpp`'s
  Phases B/D) plus, for `keyswitch.cpp`'s cache specifically, the Rust
  FFI bridge's `gpufhe_ks_fixture_run_reference`; `encdec.cpp`'s is
  reachable via `tests/test_encrypt_host_shoup_pooled_thread_safety.cpp`'s
  own concurrent reference computation; `encrypt_host_shoup.cpp`'s had
  no prior concurrent reproducer and got a new one
  (`tests/test_encrypt_host_shoup_concurrency.cpp`). The exact
  find()-then-emplace() pattern was proven racing by ThreadSanitizer
  (a faithful, CUDA-free standalone extraction, since TSan cannot run
  directly against this project's CUDA-linked binaries in this
  environment) and proven fixed by the identical mutex pattern already
  used for the other 9. Zero measurable performance impact (real
  before/after A/B, not assumed). Full evidence:
  `research/PHASE33_CACHE_CONCURRENCY_AUDIT.md`.
- **Raw, unwrapped CUDA elsewhere**: 89 `cudaMalloc`, 40 `cudaFree`, 103
  synchronous `cudaMemcpy`, and 53 `cudaMemcpyAsync` call sites across
  `src/`+`include/` (counts from this round's own grep), the large
  majority with no RAII wrapper and no shared error-checking convention.
  20 `cudaStreamCreate` call sites across `src/`+`tests/`.
- **Default-stream-0 parameters**: `ks_context_create`,
  `evalkeygen_host`, `evalkeygen_host_sold`, `ScopedKSContext`'s
  constructor, and `KsResidentCache::get` all default their
  `cudaStream_t` parameter to `0` (the legacy/blocking stream) when a
  caller omits it. Phase 31 (`research/SETUP_PATH_CONCURRENCY_INVESTIGATION.md`)
  found this is exactly the implicit shared state that broke a
  multi-threaded CUDA-graph-capture reproducer — not because any of
  these functions are buggy, but because nothing in their signature
  warns a multi-threaded caller that omitting the stream means sharing
  the one stream every other thread also defaults to.
- **CUDA graphs**: used in exactly one place —
  `keyswitch_resident`, validated by `tests/test_ks_graph.cpp` and
  `tests/test_ks_graph_lifecycle.cpp` (Phase 30), and the multi-threaded
  capture contract proven by `tests/test_ks_graph_setup_concurrency_diagnostic.cpp`
  (Phase 31). No other operation in the codebase captures a graph.

## 2. Ownership model (what exists today)

| Resource | Owner today | Lifetime | Thread-safety |
|---|---|---|---|
| `DeviceCiphertext`'s `d_c0_`/`d_c1_` | the `DeviceCiphertext` object | object lifetime (RAII) | single-owner only; no sharing model exists |
| `DeviceKSContext`'s per-part `d_av`/`d_bv`/`d_keyRow` | the context (via `ks_context_create`/`destroy`, or `ScopedKSContext`) | context lifetime | per-context single-thread (Phase 31 contract) |
| `DeviceKSContext`'s `d_fr`/`d_fp`/.../root-table pointers | `g_rtab`/`upc` static caches in `keyswitch_resident.cpp` | process lifetime, never freed | mutex-guarded lookup (Phase 30) |
| `DeviceKSWork`'s 12 owned pointers | the work object | work-object lifetime | per-object single-thread |
| `KsResidentCache` entries | the cache object | LRU-evicted or cache destruction | single-thread enforced by `abort()` on violation; not integrated anywhere |
| `RegisteredHostBuffer` | the buffer object | object lifetime, stream-aware release | single-owner; move-only |
| most `encrypt_host_*`/`encode_fft`/`evalkeygen` static caches | the cache itself (process-lifetime) | process lifetime | 13/13 mutex-guarded as of Phase 33 (see above) |

**No object in this codebase currently represents "the application's
handle to a unit of GPU work" in a reusable, cross-feature way.** Each
feature (keyswitch, encrypt, encode, rotate) built its own cache and its
own ownership rules, converging independently on similar shapes
(RAII move-only wrapper + process-lifetime cache + eventual mutex) but
never factored into a shared primitive.

## 3. Execution model

There is no execution-context object at all. A "context" in this
codebase always means a specific feature's device-resident constant set
(`DeviceKSContext`), never a reusable device/stream/scratch handle an
application would hold once and pass across many unrelated operations.
Every operation that needs a stream either takes one as an explicit
parameter (`keyswitch_resident`, `automorphism_eval_device`,
`rotate_ct_device`) or silently defaults to the legacy stream
(`evalkeygen_host`, `ks_context_create`). There is no concept of a
per-application "default context" that bundles a chosen device, a
non-default stream, and a scratch allocator — every call site manages
its own stream lifetime by hand.

## 4. Thread-safety contract (incorporating Phase 31 verbatim)

Quoting `research/SETUP_PATH_CONCURRENCY_INVESTIGATION.md` section 14
directly, since Phase 31 is the authoritative, tested source for this:

> **Proven safe:** Single-threaded, sequential use of the entire
> setup-through-keyswitch pipeline. Independent `DeviceKSContext`/
> `DeviceKSWork`/graph objects, one per thread, used from a single
> owning thread each, **provided**: (a) any thread capturing a graph
> uses `cudaStreamCaptureModeThreadLocal` on a non-blocking stream it
> created itself, never the legacy default stream; (b) no thread
> performs `cudaMalloc`/`cudaFree` while any other thread has an active
> capture in progress. Concurrent **direct** (non-graph) calls from
> independent threads with independent contexts, each on its own
> non-default stream.
>
> **Not proven, and not claimed safe:** concurrent graph capture using
> `cudaStreamCaptureModeGlobal` from multiple threads; any caller passing
> the legacy default stream from multiple threads with no coordination
> of its own.

This round adds one generalization, grounded in section 1's cache
inventory above: the same "proven safe only per-object, single-thread
unless explicitly fixed" pattern applies to every one of the 13
static-map caches, not just the keyswitch path — 9 are proven
mutex-safe, 4 are unaudited or deliberately left as documented latent
risk. **Thread-safety in this codebase is currently established
call-site-by-call-site, not architecturally** — there is no mechanism
that would make a *new* cache automatically safe; a developer must
remember to add the mutex, exactly as 4 of the existing 13 did not.

## 5. Stream model

- **Per-operation explicit streams** (`keyswitch_resident`,
  `automorphism_eval_device`, `rotate_ct_device`,
  `automorphism_eval_fast_device`): the caller must supply a stream;
  there is no default. This is the safer pattern, but each operation
  that follows it does so by individual convention, not a shared type
  (`cudaStream_t` is passed as a bare handle everywhere — nothing
  wraps "a stream I own and must destroy" vs "a stream someone lent me").
- **Default-to-legacy-stream parameters** (`ks_context_create`,
  `evalkeygen_host`, `evalkeygen_host_sold`): convenient for
  single-threaded sequential callers, a correctness hazard for
  multi-threaded callers who don't realize they're sharing stream 0
  (exactly Phase 31's root cause).
- **No stream-ownership type exists.** Nothing distinguishes "this
  `cudaStream_t` is mine to destroy" from "this `cudaStream_t` was
  handed to me by a caller" — every call site that creates a stream
  (20 `cudaStreamCreate` sites) must remember to destroy it, with no
  RAII enforcement anywhere in the codebase for streams specifically
  (contrast with the RAII discipline `DeviceCiphertext`/
  `RegisteredHostBuffer`/`ks_raii.h` DO apply to allocations).

## 6. Buffer lifecycle

Three different lifecycle disciplines coexist, with no shared contract:
1. **RAII-owned** (`DeviceCiphertext`, `RegisteredHostBuffer`,
   `ScopedKSContext`/`ScopedKSWork`): allocate in constructor, free in
   destructor, move-only, safe against double-free and (for
   `RegisteredHostBuffer`) async-transfer-outliving-release via an
   explicit state machine.
2. **Manual pair discipline** (`DeviceKSContext`/`DeviceKSWork`'s raw C
   API, most `encrypt_host_*` buffer pools): the caller must call a
   matching `_destroy`/free function; `ks_raii.h`'s own header comment
   documents a real historical bug (`ks_context_destroy` once forgot to
   free `d_keyRow`) caused by exactly this discipline.
3. **Process-lifetime, never freed** (the static-map caches): correct
   by construction for a never-freed cache (nothing to double-free), but
   means VRAM for every distinct `(shape, key)` ever requested is held
   until process exit — acceptable for the current small working sets
   (measured ~60 MiB/eval-key context, Phase 25-27), a real risk if this
   pattern is reused for a larger or more varied working set without the
   bounded-eviction discipline `KsResidentCache` already built but never
   wired in.

No buffer type in the codebase currently supports the god-prompt's
stated goal ("allocate a buffer, populate it, perform multiple GPU
operations, and retrieve results without unnecessary intermediate
copies") for an arbitrary shape — only `DeviceCiphertext` gets close,
and only for the one fixed ciphertext shape.

## 7. Graph execution model

CUDA graph capture/replay is proven correct and performant for exactly
one operation (`keyswitch_resident`, Phase 30: 1.49x warm-replay speedup
at bootstrap scale; Phase 31: safe multi-threaded capture contract). It
is not a reusable facility — `cudaStreamBeginCapture`/
`cudaStreamEndCapture`/`cudaGraphInstantiate`/`cudaGraphLaunch` calls
live directly in `tests/test_ks_graph*.cpp`, not behind any abstraction
a second operation could reuse without re-deriving Phase 30's two
hard-won lessons (stable buffers for every host-side array fed to a
captured `cudaMemcpyAsync`; one dedicated buffer per distinct
`(part, role)` rather than a shared scratch array). **Generalizing this
would require, at minimum**: (a) a documented "capturable operation"
contract — no function-local host buffer ever passed to an async copy
inside the captured region, every buffer the operation touches stable
for the graph's entire replay lifetime; (b) a reusable graph-lifecycle
wrapper analogous to `ScopedKSContext` (capture/instantiate/destroy as
RAII, mirroring `tests/test_ks_graph_lifecycle.cpp`'s already-proven
14-check lifecycle, which currently lives in a test file, not a header);
(c) explicit, checked `cudaError_t` handling at every step (Phase 31's
single biggest lesson: the original bug was invisible specifically
because nothing checked a return code). None of this exists as a
reusable component today.

## 8. Existing limitations (summary)

1. No general device-buffer abstraction — `DeviceCiphertext` is the
   only RAII device buffer, and it is shape-locked to ciphertexts.
2. No stream-ownership type — raw `cudaStream_t` handles everywhere,
   no RAII, no "mine vs. borrowed" distinction.
3. No execution-context object bundling device/stream/scratch for an
   application to hold once and reuse across unrelated operations.
4. The thread-safety-via-mutex pattern has been applied 9 times and
   skipped (or not yet audited) 4 times, with no mechanism preventing a
   10th cache from repeating the unguarded mistake.
5. Error-checking is inconsistent: exactly one file (`device_ciphertext.cpp`)
   has a reusable `ck()` helper; the Phase 31 root cause was precisely a
   capture/launch sequence that checked zero of its ~7 CUDA calls.
6. `KsResidentCache` — the best-designed lifecycle primitive in the
   project — is tested but unused; its design effort is not amortized
   across any other feature.
7. CUDA graph capture is validated for one operation only and is not a
   reusable facility.

## 9. Proposed architecture (smallest useful addition)

Given the above, a large new framework is not justified — the codebase
already contains several correct, hard-won individual primitives; what
is missing is generalizing the ones that have already paid for
themselves at least twice, and nothing more. Two primitives are proposed,
each directly justified by a specific, already-observed pain point
(not speculative):

### 9.1 `DeviceBuffer<T>` — generalizes `DeviceCiphertext`'s proven pattern
Justified by: `DeviceCiphertext` already proves the RAII-move-only-owned
-device-pointer pattern is correct and used in production
(`src/ckks_ops.cpp` and friends build on it); the other 89 `cudaMalloc`
sites have no equivalent and have, historically, been where real bugs
lived (`ks_raii.h`'s own documented `d_keyRow` leak). A single-pointer,
single-shape, move-only RAII wrapper over `cudaMalloc`/`cudaFree` with
an explicit stream parameter for upload/download (sync by default,
matching `DeviceCiphertext`'s own current behavior, with an async
variant that requires the caller to pass a stream and accepts
responsibility for ordering — do not silently make every call async)
would let new code stop hand-rolling malloc/free pairs without forcing
any existing, working call site to change. This is strictly additive.

### 9.2 `cuda_check(cudaError_t, const char* what)` — generalizes `device_ciphertext.cpp`'s `ck()`
Justified by: it is the single biggest contributor to Phase 31's root
cause being silent instead of loud, and it is a ~10-line change with zero
risk (it already exists, duplicated, in one file — promoting it to a
shared header is not new design, it's deduplication). Should become the
standard the project's own `CLAUDE.md`/contribution convention
recommends for any new CUDA call site.

**Deliberately NOT proposed this round** (speculative or not yet
justified by a second real use case):
- A generic `GpuExecutionContext` bundling device+stream+scratch: no
  current caller has asked for one, and `KsResidentCache`'s own
  unintegrated status (Phase 27) is a cautionary tale about building
  infrastructure ahead of a confirmed integration target. Revisit if/when
  Work Package B's application workload (a genuinely new, non-FHE
  caller) turns out to need it.
- Generalizing CUDA graph capture into a reusable facility: only one
  operation uses it; a second real caller should come first.
- A generic synchronized-cache template to replace the 13 hand-rolled
  `static std::map` caches: Phase 33 closed the actual safety gap (all
  13/13 now individually mutex-guarded, proven by ThreadSanitizer, not
  assumed) without this generalization — rewriting 13 existing,
  mostly-working call sites into a shared template remains a larger,
  riskier change than either phase's budget justified, and is no longer
  blocking on a correctness gap, only a maintainability one. Still a
  reasonable future follow-up, now lower priority.

## 10. Migration strategy

Neither proposed primitive requires touching a single existing call
site to adopt. `DeviceBuffer<T>` and `cuda_check` are additive headers;
existing code (including `DeviceCiphertext` itself) can migrate
opportunistically or not at all. The recommended adoption path is: use
both in the new non-FHE application workload (Work Package B) where
there is no existing convention to disturb, prove them there, and only
then consider whether any existing `src/*.cpp` file benefits from
migrating. Do not mass-migrate 89 call sites in one phase.
