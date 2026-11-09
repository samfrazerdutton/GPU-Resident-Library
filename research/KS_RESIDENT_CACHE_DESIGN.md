# Resident eval-key context cache — design, memory model, and verdict

GOD PROMPT "CKKS PHASE 27". Determines whether a bounded, safe,
production-grade resident `DeviceKSContext` cache can deliver Phase 26's
promised benefit under the ACTUAL bootstrap workload — not a toy benchmark.

## 0. Verdict, up front

**Phase 26's 1.66-1.69x figure does not generalize to the real bootstrap.**
A faithful replay of the actual rotation-key access sequence (not a
synthetic round-robin) shows that of 181 total diagonal rotation requests
across a full n=8192 bootstrap, only **one of its six BSGS passes** has any
cacheable reuse at all — the other five have exactly zero. A cache helps
there and nowhere else. On that one pass, a correctly-sized cache (~30
entries, ~600 MiB) delivers a REAL, bit-exact, honestly-measured **1.5-1.8x**
speedup (depending on accounting choices — see section 2). A cache sized to
the naive "~77-entry KROT working set" Phase 26's own limitations section
cited would be **both the wrong quantity and still not enough** — the real
device-context working set is 151 distinct `(k, tw)` pairs, requiring ~5.6
GiB, which was calculated, not allocated, to avoid exactly the OOM risk
Mission 2 warns against.

**The cache built this round (`include/ks_resident_cache.h`) is correct,
tested (34/34 checks, repeatable), and safe** — but given how narrow the real
benefit is, **production integration is not recommended as a general-purpose
change.** It is offered as verified, bounded, opt-in infrastructure for a
future, explicitly-scoped integration into the ONE stage pattern that
benefits, should that be judged worth the added complexity.

## 1. Reconstructing and verifying Phase 26's experiment (Mission 1)

`bench/bench_ks_context_resident_ab.cpp` was rebuilt and rerun this round.
Every core claim reproduced:

| Claim | Phase 26 | This round |
|---|---|---|
| Eval-key size per context | 60.000 MiB | 60.000 MiB (identical) |
| Bit-exact | 0/48 differ | 0/48 differ |
| Speedup | 1.6621x | 1.6871x (same order, expected ~30% machine spread) |
| Transfer reduction | 2400 MiB / 83.3% | identical (same arithmetic) |

**One real discrepancy found and quantified** (Mission 1 point 7, "were setup
and cache-construction costs included"): Arm A's measured interval includes
48 `ks_context_create` **and** 48 matching `ks_context_destroy` calls. Arm
B's measured interval includes only the 8 `ks_context_create` calls on cache
miss — the 8 matching destroys happen in the cache object's destructor,
**after** the timer has already stopped (the destructor runs when the
`ResidentContextCache` local variable goes out of scope, which is after the
`return elapsed;` expression is evaluated). A new diagnostic,
`bench/probe_create_destroy_cost.cpp`, measured `ks_context_destroy`
directly: **it costs 36% of `ks_context_create`'s time** (0.0080s vs 0.0219s,
median of 15 trials, same parameters). Correcting Arm B's reported median
upward by `8 * 0.0080s ≈ 0.064s` moves the speedup from ~1.69x to **~1.61x**.
This is a real but modest overstatement — the core finding (a cache helps)
survives; the precise multiplier does not.

Every other Mission 1 verification point held: the cache key is `(k, tw)`,
never `k` alone (checked by reading the source); all 48 calls used
compatible, correctly-constructed keys; the comparison used identical
ciphertexts/keys/operations/correctness checks in both arms; the result was
stable across 2 independent runs with alternating order and 9 trials each.

## 2. The real working set (Mission 2)

### 2.1 The access sequence, extracted without running the bootstrap

`tests/test_boot.cpp`'s diagonal-finding logic (`stageDiags`,
`stageDiagsFwd`, `applyMerged`, `applyMergedFwd`, `rotAmt`) is pure
complex-arithmetic bookkeeping over `S=n/2`-sized vectors — it touches no
ciphertext, no device memory, no CUDA. `research/benchmarks/
extract_real_key_sequence.cpp` replicates this logic verbatim (same n, LST,
RST, tower-decrement pattern, EvalMod depth arithmetic) to produce the ACTUAL
ordered sequence of `(rotation key, tower count)` requests the real bootstrap
generates, without executing it.

```
Total diagonal rotation accesses (excluding off==0 identity diagonals): 181
Distinct (k, tw) pairs [the device-context working set]  : 151
Distinct k alone (ignoring tw) [KROT's host-side working set]: 76
```

**The 76-distinct-k figure independently cross-validates against a
previously INSTRUMENTED (not simulated) measurement**, found in
`research/EXPERIMENT_LEDGER.md`: *"predicted 76 rotation keys for LST=3,
matching the instrumented `evalkeygen calls total=77` = 76+1 relin key"*.
This extraction tool reproduces that exact historical number from pure
source-logic replay, which is strong evidence it is faithful.

**The "~77 entries" figure Phase 26's limitations section cited is this
76-distinct-k number** — KROT's host-side cache, which stores a full
`KeySwitchConstants` keyed on the rotation index alone and is reused
whenever that SAME `k` recurs (saving real `evalkeygen_host_sold` cost).
**It is the wrong quantity for sizing a DEVICE-context cache**, which must be
keyed on `(k, tw)` (per `EXPERIMENT_LEDGER.md` entry #20's documented bug
class). The real device-context working set is 151, not 76 — nearly double.

### 2.2 Per-tower-count breakdown

```
tw=7:   15 accesses,  15 distinct keys   (S2C stage, g=0)
tw=8:   30 accesses,  30 distinct keys   (S2C stage, g=1)
tw=9:   60 accesses,  30 distinct keys   (S2C stage, g=2 -- SEE BELOW)
tw=27:   1 access,     1 distinct key    (the fixed M-1 conjugation)
tw=28:  30 accesses,  30 distinct keys   (C2S stage, g=2)
tw=29:  30 accesses,  30 distinct keys   (C2S stage, g=1)
tw=30:  15 accesses,  15 distinct keys   (C2S stage, g=0)
```

**Only tw=9 shows ANY repeat** (60 accesses, 30 distinct keys — exactly 2x).
Every other tower count's access count EQUALS its distinct-key count: every
single diagonal in those five passes uses a rotation key that is NEVER
needed again at that tower count, anywhere else in the bootstrap. The 30
"extra" accesses at tw=9 are the entire source of the 151-vs-181 gap, and
they come from one specific, deliberate algorithmic structure: `test_boot.cpp`'s
S2C stage at `g == LST-1` runs the SAME set of diagonal offsets **twice** —
once folding in the real branch (`oR`), once the imaginary branch (`oI`) —
before the two merge. Confirmed directly from the source
(`tests/test_boot.cpp` lines ~875-885): `stageDiagsFwd` is called ONCE, and
the resulting `offs`/`dg` are shared across a `for (q2 = 0; q2 < srcs0.size(); ++q2)`
loop that runs twice at `g == LST-1`.

**This is the entire mechanism available for a resident-context cache to
exploit in this algorithm as currently structured.** Nothing else recurs.

### 2.3 Hit-rate simulation against the real sequence (not a round-robin)

An LRU simulation (`extract_real_key_sequence.cpp`'s own output) against the
real 181-access sequence, at a range of capacities:

```
cap      hits     misses   hit_rate
1-24     0        181      0.0%
32       30       151      16.6%
40-151   30       151      16.6%    <- FLAT. Scaling the cache past 32 buys NOTHING.
```

**The hit rate is a hard ceiling at 16.6%, independent of cache size, for
any cap ≥ 32.** A cache of 151 entries (the full working set) performs
identically to a cache of exactly 32. This directly contradicts the implicit
assumption that a bigger cache captures more reuse — in this algorithm, it
does not, because the only reuse available is confined to one stage's own
internal double-pass, with a reuse DISTANCE of exactly that stage's width
(confirmed directly: a bounded A/B capacity sweep, section 4 below, shows 0%
hit rate at cap=4/8/16 and the full 50%-of-that-stage hit rate only once cap
reaches the real stage width of 30).

### 2.4 Why a reduced-tower context is NOT smaller than the full one

`tests/test_boot.cpp`'s `atLevel` function (the real mechanism behind every
reduced-`tw` key, e.g. `KSC`'s view cache) does this:

```cpp
K.av.assign(K.numPart, {}); K.bv.assign(K.numPart, {});
for (uint32_t p = 0; p < K.numPart; ++p) { K.av[p] = KF.av[p]; K.bv[p] = KF.bv[p]; }
```

It copies `av[p]`/`bv[p]` **unchanged** from the full key `KF` — each is
still `sizeQlP(FULL=32) * n` elements. Only the **number** of parts
(`K.numPart`, from `npFor(tw)`) shrinks with `tw`. A reduced-tw context is
therefore smaller than the full one only because it uses *fewer* full-size
part-arrays, not because each array scales down. Verified directly, by
building the real atLevel-style views in
`bench/bench_realistic_ks_cache.cpp`:

```
full key   (tw=30, numPart=15): 60.000 MiB per context
atLevel view (tw=9, numPart=5): 20.000 MiB per context
```

### 2.5 Full-working-set memory requirement — calculated, not allocated

Per the directive's explicit instruction, the full 151-entry working set's
memory requirement is **calculated analytically**, never attempted as a real
allocation:

```
entry size(tw) = numPart(tw) * 4 MiB   (2 arrays x 2 MiB each, full sizeQlP=32)
npFor(tw) = tw<=2 ? 1 : (tw+1)/2

  tw=7:  15 keys * 16 MiB = 240 MiB
  tw=8:  30 keys * 16 MiB = 480 MiB
  tw=9:  30 keys * 20 MiB = 600 MiB
  tw=27:  1 key  * 56 MiB =  56 MiB
  tw=28: 30 keys * 56 MiB = 1680 MiB
  tw=29: 30 keys * 60 MiB = 1800 MiB
  tw=30: 15 keys * 60 MiB =  900 MiB
  -----------------------------------
  TOTAL                   = 5756 MiB  ≈ 5.62 GiB
```

**This would not safely coexist** with everything else this project
allocates on the measured 6144 MiB card (driver overhead, `DeviceKSWork`'s
per-stream buffers, root-table caches, the ciphertext itself). It is also
**pointless** even if it did fit: section 2.3's hit-rate ceiling shows caching
beyond the one stage's own 30-key width buys nothing. Both facts point the
same direction: **do not build a large persistent cache.**

### 2.6 Can the bootstrap process keys in batches without changing semantics?

Yes, trivially, for the ONE stage that benefits: the double real/imaginary
pass already visits the SAME 30-key set twice, in the SAME order, with no
dependency between the two passes beyond "the second pass needs the same
keys the first pass used." A cache scoped to just that stage (built at the
stage's start, torn down at its end) captures the entire available benefit
without ever holding more than ~600 MiB resident. The other five passes have
no reuse to batch.

## 3. Cache architecture (Mission 3)

`include/ks_resident_cache.h` — `gpufhe::KsResidentCache`. Built on top of
the project's own proven RAII primitive, `gpufhe::ScopedKSContext`
(`include/ks_raii.h`, already verified in isolation: "20 create/destroy
cycles with zero GPU-memory drift").

### 3.1 Identity and bounding

- Cache key is `{k, tw}` (a named struct, not a raw `pair`, to prevent
  transposition bugs) — **never `k` alone**.
- Bounded by **both** entry count (`maxEntries`, default 32 — slightly above
  the real measured 30-key single-stage width, matching this project's own
  convention of sizing a cap slightly above measured need: `KROT_CAP=96`
  over a measured ~76) **and** total resident bytes (`maxResidentBytes`,
  default 768 MiB — comfortably covers the real 600 MiB single-stage working
  set with headroom, deliberately far short of the 5.62 GiB full-working-set
  figure section 2.5 shows would not fit).
- LRU eviction.

### 3.2 Fail-closed on resource pressure

`include/ks_raii.h`'s own documentation flags a real gap in the underlying C
API: *"Neither `ks_context_create` nor `ks_work_create` reports allocation
failures today... A failed `cudaMalloc` partway through the per-part loop...
would leave a PARTIALLY populated context with no signal to the caller."*
`KsResidentCache` closes this at the POLICY layer (the only layer available
without touching `src/keyswitch_resident.cpp` internals, forbidden without a
standalone reproducer): before any miss's allocation, it queries
`cudaMemGetInfo` and projects whether eviction would free enough room; if
not, it throws `KsResidentCacheOOM` **before** attempting the allocation,
leaving cache state completely unchanged. Verified in testing (Mission 5, B3):
an impossible safety margin correctly throws, the cache's `entriesNow` stays
at 0, and the cache remains fully usable afterward (B3b).

### 3.3 Enforced single-threaded access

`DeviceKSContext`'s own documentation says a built context "serves any
number of streams concurrently" as **read-only** state, which is true. But
`KsResidentCache`'s internal `std::map`/`std::list` are mutable shared state
with no lock — unsafe from multiple threads. Rather than merely documenting
this (Phase 26's approach) and hoping it is honored, every public entry point
records the constructing thread's id and **aborts the process** if a
different thread ever calls in — converting a silent-corruption risk (the
exact failure class Phase 25 found in `src/rotate.cpp`'s `scratch()` buffer)
into an immediate, loud failure. Verified: `tests/
ks_cache_thread_violation_helper.cpp`, invoked as a genuinely fresh process
via `system()` (see section 5.3 for why a bare `fork()` was tried first and
found unsafe), confirms the abort actually fires (`SIGABRT`), not merely that
it is documented.

### 3.4 In-flight safety — found during this round's own design review

A hazard neither Phase 26's prototype nor this cache's first draft handled:
if a context is evicted while a kernel launched against its `av`/`bv` is
still in flight on some stream, freeing that memory races with the pending
kernel — undefined behavior, the same silent-corruption shape as the
`scratch()` bug. **Found and closed in this round**: `get()` takes an
optional `cudaStream_t`, records it per entry, and every path that can
destroy an entry (`evict`, `clear`, eviction-on-insert, the destructor)
calls `cudaStreamSynchronize()` on that entry's last-recorded stream FIRST.
Verified directly (Mission 5, C5): a context evicted immediately after
launching async work on it, with no explicit sync by the caller, still
produces the bit-exact correct result.

The flip side is an explicit caller contract, documented in `get()`'s own
comment: a non-default stream passed to `get()` must remain valid until the
cache either evicts that entry, is `clear()`'d, or is destroyed — whichever
comes first. Violating this (destroying the stream before the cache) is
undefined behavior the cache cannot detect from its own side. **This was not
a hypothetical worry**: writing `tests/test_ks_resident_cache.cpp` hit this
exact ordering mistake once, producing a real `SIGSEGV`, fixed by reordering
(`cache.clear()` before `cudaStreamDestroy(s)`) and then documented precisely
in the header so the next caller does not repeat it.

## 4. Test results (Mission 5)

`tests/test_ks_resident_cache.cpp` — **34/34 checks pass**, confirmed stable
across 5 repeated runs from two different working directories (see section
5.4 for a real intermittent failure this surfaced and fixed).

| Category | Checks | Highlights |
|---|---|---|
| A. Correctness | 10 | cached vs uncached bit-exact; 4 distinct keys; SAME k at 2 different tw (proves `(k,tw)` keying, not k-only); eviction + re-insertion still bit-exact |
| B. Cache behavior | 14 | cold miss → warm hit → repeated hits; capacity-enforced eviction; simulated OOM refusal + cache stays usable after; "disabled" (bypass at call site); `clear()` teardown + reuse; 20 construct/destroy cycles |
| C. Concurrency | 5 + 1 N/A | 5 independent keys; 2 DIFFERENT streams reading ONE cached context correctly; the enforced thread-violation abort (via a separate helper process); in-flight eviction safety |
| D. Memory | 4 + 1 N/A | residency tracking self-consistent; eviction/`clear()` correctly zero it; 10 cycles drift 0.0 MiB |

No test relies on timing alone for a correctness claim.

## 5. Methodology notes — things that went wrong and were fixed, kept visible

### 5.1 `fork()` is unsafe once CUDA is active

The first attempt at testing the thread-violation enforcement used a bare
`fork()` from inside the already-CUDA-active test process. This produced
`cudaMemGetInfo failed (initialization error)` in the child and a `SIGSEGV`
in the whole run — CUDA explicitly does not support `fork()` once a context
is active (the child inherits invalid driver state). `fork()+exec()`
(exactly what `system()` does) IS safe, because `exec()` replaces the process
image before any CUDA call happens. Fixed by moving the violation scenario
into its own executable, `tests/ks_cache_thread_violation_helper.cpp`,
invoked via `system()`.

### 5.2 `system()`'s signal encoding

`system()` runs the command via `/bin/sh -c`; when the invoked program dies
by signal, the WRAPPING SHELL reports that to its own caller as a **normal
exit with status 128+signal** — not `WIFSIGNALED`. An earlier version of the
check assumed `WIFSIGNALED` applied directly and consequently misreported a
confirmed-working abort as a test failure. Fixed by checking both encodings.

### 5.3 stdout buffering hid a working program behind an apparently-empty crash

While debugging 5.1/5.2, every crashed run appeared to produce NO output at
all — deeply misleading, since dozens of `[PASS]` lines had actually already
run. Cause: `stdout` is fully buffered (not line-buffered) whenever it is not
attached to a TTY, true of every captured/piped run; a hard crash discards
every unflushed line. Fixed with `setvbuf(stdout, nullptr, _IOLBF, 0)` at the
top of `main()`.

### 5.4 A genuine intermittent failure, caught by running tests from a
different working directory (exactly as Mission 5 instructs)

The full regression suite was run once from `build2/` (34/34 pass) and once
via `./build2/test_ks_resident_cache` from the repo root — which failed
(32/34). Cause: the thread-violation test invoked its helper via a bare
`"./ks_cache_thread_violation_helper"`, relative to the CURRENT WORKING
DIRECTORY, not the test binary's own location — silently resolving to
nothing when run from elsewhere (shell "command not found", exit 127,
correctly NOT mistaken for a signal death, but also not finding the real
helper). Fixed by resolving the helper's path via `/proc/self/exe`,
confirmed stable across 2 working directories x 5 repeated runs each.

## 6. Recommendation

1. **Do not integrate a general-purpose resident eval-key cache into
   production.** The measured benefit (1.5-1.8x on one isolated pass) does
   not justify the complexity and memory risk for an optimization that helps
   exactly one of six BSGS passes and nothing else.
2. **If ever pursued, scope it narrowly**: a cache built at the START of the
   S2C stage that processes both real and imaginary branches, sized to that
   stage's own key count (~30-32 entries, ~600 MiB), torn down at the END of
   that stage. This is the one place the evidence supports.
3. **Keep `include/ks_resident_cache.h` and its test suite** as verified,
   reusable infrastructure. No production file was modified to build it —
   `src/keyswitch_resident.cpp`, `src/rotate.cpp`, and `tests/test_boot.cpp`
   remain untouched.
4. Any future integration must run the FULL correctness/memory/concurrency
   gate this round built (Mission 5's suite) against the real production
   call site, not assume this round's isolated fixture-level results
   transfer unchanged.
