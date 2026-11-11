# LLM KV-cache residency research (Phase 32, Work Package E)

**Status: exploratory research + one small, standalone, host-only
prototype. This is NOT an LLM engine, does not touch any production
source file, and gpufhe does not implement a production KV-cache manager
today.** Nothing here claims otherwise.

## 1. Why this is relevant to GPU-Resident-Library at all

This project's actual production experience with exactly one bounded,
GPU-resident cache — `gpufhe::KsResidentCache` (`include/ks_resident_cache.h`,
designed in `research/KS_RESIDENT_CACHE_DESIGN.md`) — is the single most
directly transferable piece of evidence available, more useful here than
any external survey, because it already encodes real lessons this exact
codebase paid for in bugs:

- **Composite keys only, never a partial identity.** `KsResidentCacheKey{k, tw}`
  deliberately never keys on the rotation element alone, because
  `research/EXPERIMENT_LEDGER.md` entry #20 records a real, shipped bug
  from exactly that mistake (a reduced-tower-count key reused at the
  wrong tower count, silently wrong ciphertexts). A KV-cache block keyed
  on `(sequence_id, block_index)` alone and reused across a different
  `layer` or `kv_head_group` would reproduce this bug class exactly —
  wrong tensor content, no crash, no error.
- **Bound by bytes AND by count.** Entry size is not uniform in that
  cache (an `atLevel`-reduced-`tw` context carries unchanged `av`/`bv`
  sizes — "doesn't shrink the way you'd guess"); a pure entry-count cap
  can silently blow VRAM. KV-cache blocks are normally uniform-sized (by
  design, in every production system reviewed below), which actually
  makes this easier than gpufhe's own case — but the general principle
  (size variance must be accounted for, not assumed away) still applies
  if a future design ever mixes block sizes (e.g. different quantization
  levels per block).
- **Fail closed on memory pressure, before attempting the allocation.**
  `KsResidentCache` calls `cudaMemGetInfo` and refuses (throwing
  `KsResidentCacheOOM`, cache state unchanged) rather than risking a
  partially-populated allocation with no signal — closing a gap
  `ks_raii.h` itself documents in the underlying `ks_context_create` C
  API. A KV-cache allocator has the identical failure mode: a half-filled
  block from a failed `cudaMalloc` is worse than a clean rejection.
- **In-flight safety: synchronize before you free.** Every eviction path
  calls `cudaStreamSynchronize` on the entry's last-recorded stream
  *before* its destructor can free device memory, closing a real,
  reproduced-in-testing race class (freeing memory a kernel might still
  be reading) that is this project's own version of the "evict a KV block
  an attention kernel is still reading" bug every paged-attention-style
  system must also avoid.
- **Explicit single-thread enforcement, not a documented-and-hoped-for
  contract.** `KsResidentCache` records its owner thread and calls
  `std::abort()` on a violation, rather than Phase 26's earlier
  weaker approach of merely documenting a limitation. The same philosophy
  (enforce visibly, don't just document) is directly transferable to any
  KV-cache allocator that is not meant to be called concurrently from
  multiple serving threads without its own synchronization.

## 2. What production LLM-serving systems actually do (general, no
   project-specific claim)

This section draws on well-established, publicly documented
LLM-serving memory-management techniques, reasoned from general
knowledge rather than any live lookup:

- **Block/paged allocation (the vLLM "PagedAttention" approach).** Rather
  than pre-allocating each sequence's KV cache for its maximum possible
  length (which wastes enormous memory on padding for variable-length
  generations), the cache is split into fixed-size blocks (commonly
  8–32 tokens per block). Each sequence holds a "block table" — a list of
  block IDs — analogous to a page table. New tokens simply append to the
  sequence's block list, allocating a new block only when the current one
  fills. This is architecturally the same idea as `KsResidentCache`'s
  entry-keyed pool, generalized from "one entry per rotation key" to "one
  entry per (sequence, layer, block) tuple."
- **Eviction / preemption under memory pressure.** When the block pool is
  exhausted, serving systems either (a) preempt a lower-priority sequence
  (free its blocks, mark it for later recomputation from scratch), or
  (b) swap its blocks to host RAM and restore them later (what this
  phase's prototype calls "spill/restore"). Recomputation is simpler but
  wastes GPU compute; host-swap is cheaper to resume but costs PCIe
  bandwidth and host RAM. Both require the identical in-flight-safety
  discipline `KsResidentCache` already enforces: never free/overwrite a
  block a kernel might still be reading.
- **Quantized KV caches.** Storing K/V tensors in int8 (or lower) instead
  of fp16 roughly halves (or quarters) the cache's memory footprint at
  the cost of a dequantization step and some accuracy loss — directly
  analogous to a "small vs full" cache-entry-size tradeoff, the same
  shape of decision this project already made explicitly when it chose
  NOT to build a cache sized for the full 151-entry/~5.6 GiB working set
  (`research/KS_RESIDENT_CACHE_DESIGN.md`) and instead scoped it to the
  realistic ~30-entry/~768 MiB working set that actually benefits.
- **Sliding-window / attention-sink eviction.** Some architectures bound
  total KV-cache size regardless of sequence length by evicting the
  oldest blocks beyond a fixed window (keeping a few "sink" blocks at the
  very start for stability). This is a pure LRU-with-a-twist policy,
  structurally identical to the LRU eviction `KsResidentCache` already
  implements via its `lru_`/`lruPos_` list+map pair.
- **Stream ordering.** Every serving engine that uses CUDA graphs or
  multiple streams for batched decode must guarantee a block is not
  reused (evicted and reallocated to a different sequence) while any
  in-flight kernel from a previous step still reads or writes it — the
  exact discipline Phase 30/31 of this project already fought for in a
  different context (CUDA graph capture, stream-0 collisions). The
  lesson generalizes directly: **any GPU-resident cache in this
  architecture should adopt `KsResidentCache`'s "record the stream,
  synchronize before free" pattern as a hard rule, not a per-case
  judgment call.**

## 3. Memory pressure, illustrated for this project's actual hardware

The project's reference GPU is a 6144 MiB RTX 2060 Max-Q. To make the
scale of the problem concrete (illustrative order-of-magnitude math only
— no specific real model is implied or benchmarked):

For one sequence, one layer, fp16 K+V: `2 (K and V) x heads x head_dim x
seq_len x 2 bytes`. A toy-scale illustrative model (32 layers, 32 heads,
128 head_dim — deliberately small, "could plausibly fit the parameter
budget of a small quantized model on this card," not a specific real
model) at a 2048-token context:

```
per layer: 2 * 32 * 128 * 2048 * 2 bytes = 33,554,432 bytes ≈ 32 MiB
32 layers:                                 ≈ 1024 MiB (~1 GiB) for ONE sequence
```

One GiB for a single sequence's KV cache, on a 6 GiB card that must also
hold model weights, activation workspace, and (for this project) CKKS's
own resident eval-key/cache state if ever co-resident — leaves almost no
room for batching multiple concurrent sequences without either (a) a much
smaller model/context, (b) quantization, or (c) the block/eviction
machinery in section 2. This is the same shape of constraint this
project already lived through and solved once: `KS_RESIDENT_CACHE_DESIGN.md`
explicitly rejected a cache sized for the full ~5.6 GiB theoretical
working set in favor of a ~768 MiB bounded one, because the full set
"would not safely coexist with everything else this project allocates on
the measured 6144 MiB card." The same reasoning applies directly to any
future KV-cache integration: **scope it to a measured working set, never
a theoretical maximum, on this hardware tier.**

## 4. What transfers directly vs. what doesn't

| `KsResidentCache` design choice | Transfers to KV-cache blocks? |
|---|---|
| Composite key (never partial identity) | **Yes, directly.** Key must be at minimum `(sequence_id, layer, block_index)`; a real system also needs `kv_head_group` under grouped-query attention. |
| Bound by both entry count and bytes | **Yes, though less critical** if block size is fixed (the common real-world design) — still worth keeping as a belt-and-suspenders check. |
| Fail closed before allocating (VRAM preflight) | **Yes, directly** — arguably more important here, since LLM serving workloads are far more memory-pressured than this project's own keyswitch working set. |
| Stream-synchronize before evicting/freeing | **Yes, directly** — this is the load-bearing correctness rule for ANY GPU-resident cache with async consumers, independent of domain. |
| Single-thread-enforced, abort on violation | **Partially.** Real serving engines ARE often multi-threaded/batched by design (many concurrent sequences processed together in one batched kernel launch) — the transferable principle is not "forbid concurrency" but "make the actual concurrency contract explicit and enforced," which for a block pool likely means "single writer thread owns block-table mutation; the batched kernel itself reads many sequences' blocks in one call, which is fine since that's data-parallelism within one call, not concurrent cache mutation." |
| LRU eviction policy | **Partially** — real systems often need priority-aware eviction (protect blocks for in-progress generations over completed ones), not pure recency; LRU is a reasonable starting policy, not necessarily the final one. |
| Host spill instead of discard | **Yes, directly**, and is exactly what this phase's prototype demonstrates (section 5). |

## 5. Prototype built this round

A small, standalone, **host-only (no CUDA at all)** block-pool simulator:
`research/kv_cache_prototype/kv_block_cache_sim.cpp`. Deliberately
CPU-only — avoids any GPU contention with the other Phase 32 work
packages' real GPU benchmarks running concurrently, and a block
allocator's core logic (keying, bounding, eviction, spill/restore
bookkeeping) is independent of whether the payload bytes live in VRAM or
host RAM; the identical logic would sit underneath a real
`cudaMallocAsync`/`cudaMemcpyAsync`-based implementation.

Design, directly modeled on `KsResidentCache`'s own structure (map + LRU
list + LRU-position map + stats struct): a `KvBlockPool` bounded by both
block count and bytes, keyed on the composite `(seqId, layer, blockIdx)`
— deliberately avoiding entry #20's bug class — with explicit
`evictToHost`, LRU-triggered automatic eviction-to-host-rather-than-discard,
and a deterministic per-block expected-content function so every read
(fresh allocation, LRU hit, or host-restored) can be checked bit-exact
against what it should contain.

**Build** (standalone, no CMake, no CUDA):
```
g++ -O2 -std=c++17 -o /tmp/kv_block_cache_sim kv_block_cache_sim.cpp
```
**Run:** `/tmp/kv_block_cache_sim`

**Workload**: 12 synthetic sequences x 4 layers x 8 blocks/sequence = 384
distinct blocks of demand, against a pool capacity of 64 blocks — a
deliberate 6.0x oversubscription (384 blocks / 64 capacity), forcing real
eviction and restore traffic rather than a trivially-fitting pool.

**Actual measured result** (one real run, this round):
```
Correctness: 768/768 block reads bit-exact (0 failures)
Hits: 0  Misses: 768  Evictions: 704  Spills: 704  Restores: 384
Resident now: 64 blocks (0.12 MiB)  Peak resident: 64 blocks (0.12 MiB)
Bytes spilled to host: 1.38 MiB   Bytes restored from host: 0.75 MiB
Total wall time for 768 block operations: 1.594 ms (2075.5 ns/op)
```
768/768 reads (384 initial allocations + 384 second-pass re-touches)
reconstructed bit-exact content, including every one of the 384
second-pass touches that required a host-restore (the pool's 64-entry
capacity against 384 total blocks meant almost nothing survived between
the first and second pass — zero LRU hits in this particular access
pattern, all second-pass reads went through restore). This is the one
property that actually matters for a real system: **a restored or
reused block must never hand back the wrong bytes**, and in this
synthetic run, across 704 evictions and 384 restores, it never did.

Honest limitations of this prototype, stated plainly: it is pure
bookkeeping over `std::vector<uint8_t>` payloads with no real tensor
math, no CUDA, no actual attention kernel, and no real model — it
validates the ALLOCATOR's correctness contract, not anything about
inference quality, throughput under real batched kernels, or real PCIe
transfer timing (host-only `memcpy` stands in for what would be
`cudaMemcpyAsync` in a real implementation). The near-zero hit rate in
this particular synthetic access pattern is a property of the
deliberately adversarial 6x-oversubscribed test, not a claim about real
workload locality (real decode workloads re-touch the SAME sequence's
most recent blocks far more often than this synthetic round-robin
pattern does).

## 6. Recommendation for a future phase

**Do not build a production KV-cache manager or any LLM inference path
in gpufhe yet.** The honest next step, if this direction is ever pursued
for real:

1. Port this prototype's allocator logic onto actual device memory
   (`cudaMallocAsync`/a pool allocator) and real `cudaMemcpyAsync` spill/
   restore, inside Work Package A's proposed `GpuExecutionContext`/
   `DeviceBuffer` primitives (if/once those land) rather than as a new,
   unrelated abstraction — this is exactly the kind of second, clearly
   distinct use case that would validate (or invalidate) those
   primitives' generality, the same way `KsResidentCache` is this
   project's existing proof that its `ScopedKSContext`/`ks_raii.h`
   primitive generalizes beyond its first use site.
2. Only then, attach it to an actual small inference workload (a toy
   attention kernel over synthetic Q/K/V, NOT a downloaded real model) to
   get a first real GPU-timed measurement of spill/restore cost against
   PCIe bandwidth on this specific hardware.
3. Revisit the single-thread-vs-batched-concurrency question (section 4)
   with a concrete target batching API in mind, since that is a real
   design fork, not a detail.

This is a reasonable, bounded, evidence-informed starting point for a
future phase — not a commitment this phase is making on behalf of a
later one.
