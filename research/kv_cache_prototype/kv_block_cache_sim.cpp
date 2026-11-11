// Phase 32, Work Package E: LLM KV-cache residency research prototype.
//
// EXPLORATORY ONLY -- not an LLM engine, not wired into gpufhe, no real
// model, no real tensors. Pure host-side (no CUDA) simulation of a
// block-based KV-cache allocator, modeled directly on this project's own
// existing gpufhe::KsResidentCache (include/ks_resident_cache.h): bounded
// by both block COUNT and total BYTES, LRU eviction, explicit eviction,
// fail-closed-on-pressure instead of silent overcommit, and a correctly
// composite cache key (never a partial identity -- see
// research/EXPERIMENT_LEDGER.md entry #20's bug class: a cache keyed on
// only part of an object's real identity silently returns/reuses the
// WRONG entry the moment that missing dimension varies).
//
// This demonstrates, synthetically: fixed-size block allocation, block
// reuse via LRU, explicit eviction, host-spill-and-restore (simulating
// "move a cold sequence's blocks to host RAM, reconstruct on demand"),
// and verifies BIT-EXACT reconstruction of every block's content after a
// spill/restore cycle -- the one property that actually matters: a
// reused or restored block must never hand back the wrong bytes.
//
// Build (standalone, no CMake, no CUDA):
//   g++ -O2 -std=c++17 -o /tmp/kv_block_cache_sim kv_block_cache_sim.cpp
// Run:
//   /tmp/kv_block_cache_sim
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <list>
#include <map>
#include <unordered_map>
#include <string>
#include <random>
#include <chrono>
#include <cassert>

// A "block" holds TOKENS_PER_BLOCK tokens' worth of K+V for one (sequence,
// layer) pair. Real systems use real tensor dtypes; this uses a synthetic
// byte payload whose content is a deterministic function of
// (sequenceId, layer, blockIndexWithinSequence, slotWithinBlock) so any
// restored block's content can be checked bit-exact without storing a
// separate "expected" copy.
static constexpr size_t TOKENS_PER_BLOCK = 16;
static constexpr size_t BYTES_PER_TOKEN  = 128; // e.g. head_dim-ish, synthetic
static constexpr size_t BLOCK_BYTES      = TOKENS_PER_BLOCK * BYTES_PER_TOKEN;

static void fillExpected(uint8_t* buf, uint32_t seqId, uint32_t layer, uint32_t blockIdx) {
    for (size_t i = 0; i < BLOCK_BYTES; ++i) {
        buf[i] = (uint8_t)((seqId * 2654435761u) ^ (layer * 40503u) ^ (blockIdx * 2246822519u) ^ (uint32_t)i);
    }
}

// Composite key: (sequenceId, layer, blockIndex). Dropping ANY of these
// three reproduces entry #20's bug class the moment that dropped dimension
// varies -- e.g. keying on (sequenceId, blockIndex) alone would silently
// alias two different layers' blocks onto the same cache slot.
struct BlockKey {
    uint32_t seqId, layer, blockIdx;
    bool operator<(const BlockKey& o) const {
        if (seqId != o.seqId) return seqId < o.seqId;
        if (layer != o.layer) return layer < o.layer;
        return blockIdx < o.blockIdx;
    }
};

struct Stats {
    uint64_t hits = 0, misses = 0, evictions = 0, spills = 0, restores = 0;
    uint64_t bytesResident = 0, bytesSpilled = 0, bytesRestored = 0;
    size_t entriesNow = 0;
    size_t peakEntries = 0, peakBytes = 0;
};

// Thrown (reported, not a real exception here for simplicity) when even
// after evicting everything evictable, a new block would not fit -- the
// fail-closed contract KsResidentCache enforces via cudaMemGetInfo; here
// there is no GPU, so the analog is simply "the pool, even fully evicted
// down to nothing, is smaller than one block," which cannot happen by
// construction but is checked anyway to mirror the real contract.
struct PoolExhausted {};

// A bounded, byte-aware, LRU-evicting block pool, directly modeled on
// gpufhe::KsResidentCache's structure (map + LRU list + LRU-position map +
// stats), with an explicit host-spill path standing in for
// cudaMemGetInfo-gated VRAM pressure in the real GPU case.
class KvBlockPool {
public:
    KvBlockPool(size_t maxEntries, size_t maxBytes)
        : maxEntries_(maxEntries), maxBytes_(maxBytes) {}

    // Returns true if (key) is already resident (a "hit" -- the whole
    // point of residency: no re-copy, no re-compute). On a miss, allocates
    // a fresh block (evicting LRU victims to host spill if needed) and
    // fills it with its deterministic expected content (standing in for
    // "the attention kernel just wrote this block's new K/V").
    bool getOrAllocate(BlockKey key, uint8_t* outPtr /* caller-owned BLOCK_BYTES buffer, filled on return */) {
        auto it = resident_.find(key);
        if (it != resident_.end()) {
            ++stats_.hits;
            touch(key);
            std::memcpy(outPtr, it->second.data(), BLOCK_BYTES);
            return true;
        }
        ++stats_.misses;

        // Is this key cold-spilled on the host? Restore it instead of
        // recomputing -- this is the "host spill and retrieval" path.
        auto spilledIt = spilled_.find(key);
        std::vector<uint8_t> payload(BLOCK_BYTES);
        if (spilledIt != spilled_.end()) {
            payload = spilledIt->second;
            spilled_.erase(spilledIt);
            ++stats_.restores;
            stats_.bytesRestored += BLOCK_BYTES;
        } else {
            fillExpected(payload.data(), key.seqId, key.layer, key.blockIdx);
        }

        evictIfNeeded();
        if (resident_.size() >= maxEntries_ || stats_.bytesResident + BLOCK_BYTES > maxBytes_) {
            // Even after evicting everything evictable this doesn't fit --
            // fail closed rather than silently overcommitting.
            throw PoolExhausted{};
        }

        resident_.emplace(key, payload);
        lru_.push_front(key);
        lruPos_[key] = lru_.begin();
        stats_.bytesResident += BLOCK_BYTES;
        stats_.entriesNow = resident_.size();
        stats_.peakEntries = std::max(stats_.peakEntries, stats_.entriesNow);
        stats_.peakBytes = std::max(stats_.peakBytes, stats_.bytesResident);
        std::memcpy(outPtr, payload.data(), BLOCK_BYTES);
        return false;
    }

    // Explicit eviction (e.g. a sequence finished generating) -- spills to
    // host rather than discarding, so a later restore is possible (unlike
    // real transient scratch, KV content for a paused/preempted sequence
    // is exactly what a real serving engine needs to get back later).
    void evictToHost(BlockKey key) {
        auto it = resident_.find(key);
        if (it == resident_.end()) return;
        spilled_[key] = it->second;
        stats_.bytesSpilled += BLOCK_BYTES;
        ++stats_.spills;
        stats_.bytesResident -= BLOCK_BYTES;
        lru_.erase(lruPos_[key]); lruPos_.erase(key);
        resident_.erase(it);
        stats_.entriesNow = resident_.size();
    }

    const Stats& stats() const { return stats_; }

private:
    void touch(const BlockKey& key) {
        lru_.erase(lruPos_[key]);
        lru_.push_front(key);
        lruPos_[key] = lru_.begin();
    }

    void evictIfNeeded() {
        while ((resident_.size() >= maxEntries_ || stats_.bytesResident + BLOCK_BYTES > maxBytes_) && !lru_.empty()) {
            BlockKey victim = lru_.back(); lru_.pop_back();
            lruPos_.erase(victim);
            auto it = resident_.find(victim);
            if (it != resident_.end()) {
                spilled_[victim] = it->second;     // LRU eviction spills, doesn't discard
                stats_.bytesSpilled += BLOCK_BYTES;
                ++stats_.spills;
                stats_.bytesResident -= it->second.size();
                resident_.erase(it);
                ++stats_.evictions;
            }
        }
    }

    std::map<BlockKey, std::vector<uint8_t>> resident_;
    std::map<BlockKey, std::vector<uint8_t>> spilled_;
    std::list<BlockKey> lru_;
    std::map<BlockKey, std::list<BlockKey>::iterator> lruPos_;
    size_t maxEntries_, maxBytes_;
    Stats stats_;
};

int main() {
    std::printf("=== Phase 32 Work Package E: KV-cache block-pool simulation (host-only, no CUDA) ===\n");
    std::printf("Block size: %zu bytes (%zu tokens x %zu bytes/token)\n\n", BLOCK_BYTES, TOKENS_PER_BLOCK, BYTES_PER_TOKEN);

    // Pool deliberately smaller than total demand, forcing real eviction +
    // restore traffic (mirrors a 6GB card: KV cache pool is a small
    // resident fraction of total demand across many concurrent sequences).
    const size_t POOL_MAX_ENTRIES = 64;
    const size_t POOL_MAX_BYTES = POOL_MAX_ENTRIES * BLOCK_BYTES;
    KvBlockPool pool(POOL_MAX_ENTRIES, POOL_MAX_BYTES);

    const uint32_t NUM_SEQS = 12;
    const uint32_t LAYERS = 4;
    const uint32_t BLOCKS_PER_SEQ = 8; // e.g. 8*16=128 tokens of context per sequence

    // Total demand if everything were resident at once:
    size_t totalDemandBlocks = (size_t)NUM_SEQS * LAYERS * BLOCKS_PER_SEQ;
    std::printf("Pool capacity: %zu blocks (%.2f MiB). Total demand if fully resident: %zu blocks (%.2f MiB) -- %.1fx oversubscribed.\n\n",
        POOL_MAX_ENTRIES, POOL_MAX_BYTES / 1048576.0, totalDemandBlocks,
        totalDemandBlocks * BLOCK_BYTES / 1048576.0, (double)totalDemandBlocks / POOL_MAX_ENTRIES);

    auto t0 = std::chrono::steady_clock::now();

    // Phase 1: simulate decode -- each sequence appends its blocks in
    // order, across all layers, interleaved (round-robin) the way a real
    // batched decode step touches every active sequence's newest block
    // each step.
    uint8_t scratch[BLOCK_BYTES];
    size_t correctnessChecks = 0, correctnessFailures = 0;
    std::vector<std::vector<BlockKey>> seqBlocks(NUM_SEQS);

    for (uint32_t b = 0; b < BLOCKS_PER_SEQ; ++b) {
        for (uint32_t s = 0; s < NUM_SEQS; ++s) {
            for (uint32_t l = 0; l < LAYERS; ++l) {
                BlockKey key{s, l, b};
                try {
                    pool.getOrAllocate(key, scratch);
                } catch (const PoolExhausted&) {
                    std::printf("[FAIL] pool exhausted even after eviction at seq=%u layer=%u block=%u (should not happen by construction)\n", s, l, b);
                    return 1;
                }
                seqBlocks[s].push_back(key);
                // Immediately verify the just-written/just-restored block's
                // content is exactly what it should be.
                uint8_t expected[BLOCK_BYTES];
                fillExpected(expected, key.seqId, key.layer, key.blockIdx);
                ++correctnessChecks;
                if (std::memcmp(scratch, expected, BLOCK_BYTES) != 0) ++correctnessFailures;
            }
        }
        // Every third decode step, explicitly evict one finished-ish
        // sequence's oldest-touched layer's blocks to host (simulating a
        // preempted/paused sequence being spilled to make room) -- then
        // immediately try to retrieve one of them again later, exercising
        // restore.
        if (b % 3 == 2 && b >= 2) {
            uint32_t victimSeq = b % NUM_SEQS;
            for (uint32_t l = 0; l < LAYERS; ++l)
                pool.evictToHost(BlockKey{victimSeq, l, b - 2});
        }
    }

    // Phase 2: re-touch every block of every sequence once more (simulating
    // a second decode pass / re-attending to full context), forcing
    // restores for anything that got evicted above, and re-verify
    // correctness on every restore.
    for (uint32_t s = 0; s < NUM_SEQS; ++s) {
        for (const BlockKey& key : seqBlocks[s]) {
            pool.getOrAllocate(key, scratch);
            uint8_t expected[BLOCK_BYTES];
            fillExpected(expected, key.seqId, key.layer, key.blockIdx);
            ++correctnessChecks;
            if (std::memcmp(scratch, expected, BLOCK_BYTES) != 0) ++correctnessFailures;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const Stats& st = pool.stats();
    std::printf("Correctness: %zu/%zu block reads bit-exact (%zu failures)\n", correctnessChecks - correctnessFailures, correctnessChecks, correctnessFailures);
    std::printf("Hits: %llu  Misses: %llu  Evictions: %llu  Spills: %llu  Restores: %llu\n",
        (unsigned long long)st.hits, (unsigned long long)st.misses, (unsigned long long)st.evictions,
        (unsigned long long)st.spills, (unsigned long long)st.restores);
    std::printf("Resident now: %zu blocks (%.2f MiB)  Peak resident: %zu blocks (%.2f MiB)\n",
        st.entriesNow, st.bytesResident / 1048576.0, st.peakEntries, st.peakBytes / 1048576.0);
    std::printf("Bytes spilled to host: %.2f MiB   Bytes restored from host: %.2f MiB\n",
        st.bytesSpilled / 1048576.0, st.bytesRestored / 1048576.0);
    std::printf("Total wall time for %zu block operations: %.3f ms (%.1f ns/op)\n",
        correctnessChecks, ms, (ms * 1e6) / correctnessChecks);
    std::printf("\n=== %s ===\n", correctnessFailures == 0 ? "PASS" : "FAIL");
    return correctnessFailures == 0 ? 0 : 1;
}
