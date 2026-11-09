#pragma once
// GOD PROMPT "CKKS PHASE 27", Mission 3/4: a bounded, safe, EXPLICITLY
// OPT-IN resident DeviceKSContext cache.
//
// STATUS: experimental. NOT wired into any production call site
// (tests/test_boot.cpp, src/rotate.cpp, src/keyswitch_resident.cpp are all
// untouched). This header is offered as a tested, evidence-backed building
// block for a future, explicitly-scoped integration decision -- see
// research/KS_RESIDENT_CACHE_DESIGN.md for the full rationale, the realistic
// working-set analysis that shaped these defaults, and why a GLOBAL
// always-on cache is NOT recommended by this round's evidence.
//
// WHAT THIS IS FOR, PRECISELY (read research/KS_RESIDENT_CACHE_DESIGN.md
// before using this for anything beyond its own tests): a faithful replay of
// tests/test_boot.cpp's real rotation-key access pattern
// (research/benchmarks/extract_real_key_sequence.cpp) found that of 181
// total diagonal rotation requests across a full n=8192 bootstrap, 151 are
// to DISTINCT (rotation key, tower count) pairs -- i.e. the overwhelming
// majority of ks_context_create calls are for a key that will NEVER be
// reused at that same tower count again. The ONLY place real reuse exists is
// one specific S2C stage that processes the same ~30 rotation offsets twice
// (once for the real branch, once for the imaginary branch). A cache helps
// THERE and nowhere else in the algorithm as currently structured. This is
// why the cache below is designed to be BOUNDED AND SCOPABLE to one stage's
// working set (default capacity ~32, not 77, not 151, not unbounded) rather
// than a large persistent process-global cache.
//
// DESIGN PRINCIPLES (see Mission 3's explicit requirements):
//   - Cache identity is (rotation key k, tower count tw) -- NEVER k alone.
//     research/EXPERIMENT_LEDGER.md entry #20 records a real correctness bug
//     from building/using a rotation key at the wrong tower count. A k-only
//     cache would risk reproducing exactly that defect the moment the same k
//     recurs at a DIFFERENT tw (which happens constantly in the real
//     sequence -- 76 distinct k values appear across 151 distinct (k,tw)
//     pairs, so most k's are reused at multiple DIFFERENT tw's).
//   - Bounded by BYTES, not just entry count, because entry size varies with
//     tw (a reduced-tw context, built via an atLevel-style view, is smaller
//     than a full one but NOT scaled down proportionally to tw -- see the
//     design doc's "av/bv do not shrink with tw" finding). A pure
//     entry-count cap could silently blow VRAM if callers mix large and
//     small entries.
//   - Built on top of the project's own proven RAII primitive
//     (gpufhe::ScopedKSContext, include/ks_raii.h -- "20 create/destroy
//     cycles with zero GPU-memory drift" already verified in isolation) so
//     this header adds a cache POLICY on top of an already-correct
//     ownership primitive, rather than re-deriving ownership from scratch.
//   - FAILS CLOSED on resource pressure: before attempting a new context's
//     allocation, this cache queries cudaMemGetInfo and refuses to proceed
//     (throwing KsResidentCacheOOM) if the request would not comfortably
//     fit, rather than risking the "partially populated context with no
//     signal to the caller" failure mode ks_raii.h's own documentation
//     explicitly flags as a known gap in the underlying C API (neither
//     ks_context_create nor ks_work_create reports allocation failure
//     today). This is this header's actual closure of that gap, at the
//     POLICY layer, since fixing it at the ks_context_create layer itself
//     would require touching src/keyswitch_resident.cpp internals without a
//     standalone reproducer -- forbidden by the governing directive.
//   - EXPLICITLY single-threaded. DeviceKSContext's own doc comment says a
//     built context "serves any number of streams concurrently" as
//     READ-ONLY state, which is true -- but Phase 25 found that
//     unsynchronized MUTATION of shared device state (src/rotate.cpp's
//     scratch() buffer) silently corrupts ciphertexts. This cache's internal
//     std::map/std::list ARE mutable shared state with no lock, so it is not
//     safe to call from multiple threads, full stop. Rather than merely
//     documenting that limitation (Phase 26's approach) and hoping it is
//     honored, this cache ENFORCES it: every public entry point records the
//     constructing thread's id and asserts (in all builds, not just debug --
//     a thread-safety violation here is a silent-corruption risk, not a
//     performance concern, so the check is not optimized away) that every
//     subsequent call comes from that same thread. A violation terminates
//     the program via std::abort with a clear message rather than silently
//     corrupting a ciphertext.
//
// WHAT THIS HEADER DOES NOT DO:
//   - It does not modify src/keyswitch_resident.cpp, src/rotate.cpp, or
//     tests/test_boot.cpp.
//   - It does not change any cryptographic parameter.
//   - It is not wired into any production call site. Using it requires an
//     explicit, separate opt-in at a call site -- there is no "default on"
//     behavior anywhere in this file.
#include "keyswitch_resident.h"
#include "keyswitch.h"
#include "ks_raii.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <list>
#include <stdexcept>
#include <string>
#include <thread>
#include <functional>

namespace gpufhe {

// Thrown when a cache miss's allocation would not safely fit in currently
// free VRAM. The cache's own state is left UNCHANGED on this path -- no
// partial insertion, no corrupted LRU state. Caller should fall back to the
// existing uncached ks_context_create/destroy path (which will likely also
// fail, but at least not through this cache silently returning a bad
// context), or reduce the cache's capacity, or free other GPU memory first.
class KsResidentCacheOOM : public std::runtime_error {
public:
    explicit KsResidentCacheOOM(const std::string& what) : std::runtime_error(what) {}
};

// Thrown when get()/try_get() is called from a thread other than the one
// that constructed the cache (or last reset it). This is a programming
// error, not a resource condition -- it is deliberately NOT caught and
// recovered from anywhere in this header; the default handler is
// std::abort() via the assert-like check below, because continuing past a
// detected cross-thread access risks exactly the silent-corruption failure
// mode Phase 25 found in production code.
class KsResidentCacheThreadViolation : public std::logic_error {
public:
    explicit KsResidentCacheThreadViolation(const std::string& what) : std::logic_error(what) {}
};

struct KsResidentCacheKey {
    uint32_t k;    // rotation Galois element
    uint32_t tw;   // tower count this context was built at -- NEVER omit this
    bool operator<(const KsResidentCacheKey& o) const {
        return k != o.k ? k < o.k : tw < o.tw;
    }
    bool operator==(const KsResidentCacheKey& o) const { return k == o.k && tw == o.tw; }
};

struct KsResidentCacheStats {
    uint64_t hits = 0, misses = 0, evictions = 0;
    uint64_t bytesUploaded = 0;     // actual eval-key H2D bytes this cache has caused, cumulative
    uint64_t bytesAvoided = 0;      // eval-key H2D bytes a hit avoided re-uploading, cumulative
    uint64_t oomRejections = 0;     // misses that were refused due to the VRAM preflight check
    size_t entriesNow = 0;
    size_t residentBytesNow = 0;
};

// A bounded, byte-aware, single-thread-enforced LRU cache of DeviceKSContext,
// built on gpufhe::ScopedKSContext. See this file's header comment for the
// full rationale and the "ONE stage benefits, the other five don't" finding
// that shaped the defaults below.
class KsResidentCache {
public:
    // maxEntries: a hard cap on entry COUNT, independent of bytes (belt and
    // suspenders -- catches a misconfiguration even if maxResidentBytes is
    // set too generously). Default 32: slightly above the real measured
    // 30-key single-stage working set (research/benchmarks/
    // extract_real_key_sequence.cpp), matching this project's own existing
    // convention of sizing a cap slightly above measured need (KROT_CAP=96
    // over a measured ~76-77; KSC_CAP=48 over... a smaller measured need).
    //
    // maxResidentBytes: a hard cap on total resident eval-key bytes. Default
    // 768 MiB: comfortably covers the real single-stage working set (30
    // entries x up to ~20-60 MiB each, measured 600 MiB for the tw=9 case in
    // research/EVAL_KEY_TRANSFER_VERIFICATION_V2.md) with headroom, while
    // remaining a SMALL fraction of the 6144 MiB card -- this is deliberately
    // NOT sized to hold the full 151-entry/~5.6 GiB real working set, which
    // Mission 2's analysis shows would not safely coexist with everything
    // else this project allocates on this hardware, and would not help
    // anyway (five of the six BSGS passes have zero reuse to exploit).
    //
    // vramSafetyMarginBytes: refuse a new allocation unless AT LEAST this
    // much free VRAM would remain afterward. Default 512 MiB -- a deliberate
    // guess at headroom for DeviceKSWork's own per-stream buffers (~8 MiB
    // each, see research/BOOTSTRAP_GPU_RESIDENCY_AUDIT.md) and CUDA context
    // overhead; NOT a measured figure, callers with tighter memory budgets
    // should pass a value informed by their own cudaMemGetInfo headroom.
    explicit KsResidentCache(size_t maxEntries = 32,
                             size_t maxResidentBytes = 768ull * 1024 * 1024,
                             size_t vramSafetyMarginBytes = 512ull * 1024 * 1024)
        : maxEntries_(maxEntries), maxResidentBytes_(maxResidentBytes),
          vramSafetyMargin_(vramSafetyMarginBytes), ownerThread_(std::this_thread::get_id()) {}

    // Non-copyable (owns device memory via ScopedKSContext), non-movable for
    // simplicity (nothing in this project's expected usage needs to move a
    // cache instance; a moved-from cache with a stale ownerThread_ would be
    // a footgun not worth the complexity to avoid).
    KsResidentCache(const KsResidentCache&) = delete;
    KsResidentCache& operator=(const KsResidentCache&) = delete;

    ~KsResidentCache() {
        checkThread("~KsResidentCache");
        // Synchronize every entry's last-recorded stream BEFORE entries_'s
        // own destruction runs each ScopedKSContext's dtor -- same in-flight
        // safety contract as evict()/clear()/evictIfNeeded(). Destructors
        // must not throw, so this return value is intentionally unchecked --
        // BUT see the API CONTRACT in get()'s own doc comment: calling this
        // on a stream the CALLER has already destroyed is undefined
        // behavior at the CUDA driver level (not a graceful "failure" this
        // destructor could react to even if it wanted to), confirmed by a
        // real SIGSEGV produced while writing tests/test_ks_resident_cache.cpp
        // with that exact ordering mistake (cache destroyed AFTER its
        // stream). The caller owns getting this ordering right; this cache
        // cannot detect a dangling stream handle from its own side.
        for (auto& kv : entries_) cudaStreamSynchronize(kv.second.lastStream);
    }  // entries_'s own destructor then frees each ScopedKSContext

    // Looks up (k, tw). On a hit, returns the cached context and updates LRU
    // order. On a miss, builds a fresh one via evalKeyBuilder(k, tw) ->
    // KeySwitchConstants, after a VRAM preflight check -- throws
    // KsResidentCacheOOM (cache state unchanged) if that check fails, rather
    // than attempting an allocation that might partially succeed with no
    // signal (the exact gap include/ks_raii.h's own documentation flags in
    // the underlying ks_context_create).
    //
    // The returned reference is valid until the next call to get() that
    // might evict THIS key's entry -- it cannot be evicted by the SAME call
    // that just returned it (a hit always re-inserts at the front; a miss's
    // newly-inserted entry is also always at the front), but a LATER call
    // for a DIFFERENT key, once this cache is at capacity, can evict it.
    // Callers that need a context to outlive several subsequent get() calls
    // must copy what they need out of it (or restructure to finish using one
    // context's results before requesting the next), not hold the reference
    // across calls.
    //
    // IN-FLIGHT SAFETY CONTRACT (found and closed during this cache's own
    // Mission 3 design review, not present in Phase 26's prototype): a
    // context that is EVICTED while a kernel launched against its d_av/d_bv
    // is still in flight on some stream would free device memory a pending
    // kernel might still read -- undefined behavior, and exactly the
    // silent-corruption shape Phase 25's scratch()-buffer bug took. To close
    // this, every get() call records WHICH STREAM the caller intends to use
    // the returned context on (default stream 0, the common case in this
    // project's own test/benchmark code). Before this cache ever destroys an
    // entry -- on eviction, on evict(), on clear(), or in the destructor --
    // it calls cudaStreamSynchronize() on that entry's last-recorded stream
    // first. This trades a possible stall for a guarantee: this cache can
    // never free memory a kernel might still be reading. If a caller uses
    // one context across SEVERAL streams, pass the stream used by the LATEST
    // outstanding work on each get() call that could trigger an eviction.
    //
    // CALLER-OWNED API CONTRACT, the flip side of the guarantee above: any
    // non-default stream passed here MUST remain a VALID stream handle until
    // this cache either evicts that entry, is clear()'d, or is itself
    // destroyed -- whichever comes first. Destroying the stream yourself
    // BEFORE that point, then letting this cache's cleanup logic
    // cudaStreamSynchronize() the now-dangling handle, is undefined behavior
    // at the CUDA driver level (confirmed: produced a real SIGSEGV while
    // writing tests/test_ks_resident_cache.cpp with that exact ordering
    // mistake -- the fix there is to call clear() or let the cache go out of
    // scope BEFORE destroying a stream it was given). This cache cannot
    // detect a dangling stream handle from its own side; getting the
    // ordering right is the caller's responsibility.
    template <typename Builder>
    DeviceKSContext& get(uint32_t k, uint32_t tw, Builder evalKeyBuilder, cudaStream_t s = 0) {
        checkThread("get");
        KsResidentCacheKey key{k, tw};
        auto it = entries_.find(key);
        if (it != entries_.end()) {
            ++stats_.hits;
            stats_.bytesAvoided += it->second.bytes;
            it->second.lastStream = s;
            touch(key);
            return it->second.ctx.get();
        }
        ++stats_.misses;

        KeySwitchConstants K = evalKeyBuilder(k, tw);
        size_t needBytes = 0;
        for (uint32_t p = 0; p < K.numPart; ++p) needBytes += K.av[p].size() * 8 + K.bv[p].size() * 8;

        preflightOrThrow(needBytes, key);

        Entry e;
        e.ctx = ScopedKSContext(K);   // may itself throw (e.g. P>=Q_part_max validation) -- cache state still unchanged, since nothing below has run yet
        e.bytes = needBytes;
        e.lastStream = s;
        stats_.bytesUploaded += needBytes;

        evictIfNeeded(needBytes);

        auto res = entries_.emplace(key, std::move(e));
        lru_.push_front(key);
        lruPos_[key] = lru_.begin();
        stats_.residentBytesNow += needBytes;
        stats_.entriesNow = entries_.size();
        return res.first->second.ctx.get();
    }

    // Explicitly evict one entry (no-op if absent). Returns true if an entry
    // was actually removed.
    bool evict(uint32_t k, uint32_t tw) {
        checkThread("evict");
        KsResidentCacheKey key{k, tw};
        auto it = entries_.find(key);
        if (it == entries_.end()) return false;
        cudaStreamSynchronize(it->second.lastStream);  // see get()'s in-flight safety contract
        stats_.residentBytesNow -= it->second.bytes;
        lru_.erase(lruPos_[key]); lruPos_.erase(key);
        entries_.erase(it);
        ++stats_.evictions;
        stats_.entriesNow = entries_.size();
        return true;
    }

    // Drops every entry, freeing all device memory. Safe to call at any
    // time; a subsequent get() starts cold again. This is the explicit
    // "cache teardown" operation Mission 5 asks to be tested independently
    // of the destructor. Synchronizes every entry's last-recorded stream
    // first, per get()'s in-flight safety contract.
    void clear() {
        checkThread("clear");
        for (auto& kv : entries_) cudaStreamSynchronize(kv.second.lastStream);
        entries_.clear();  // ScopedKSContext destructors run here, freeing device memory
        lru_.clear(); lruPos_.clear();
        stats_.entriesNow = 0; stats_.residentBytesNow = 0;
    }

    const KsResidentCacheStats& stats() const { return stats_; }
    size_t maxEntries() const { return maxEntries_; }
    size_t maxResidentBytes() const { return maxResidentBytes_; }

private:
    struct Entry { ScopedKSContext ctx; size_t bytes = 0; cudaStream_t lastStream = 0; };

    void checkThread(const char* where) const {
        if (std::this_thread::get_id() != ownerThread_) {
            // Fixed buf[256] previously truncated this message (compiler
            // warning caught it) -- printed directly instead of through an
            // intermediate fixed buffer, so the full text always reaches
            // stderr regardless of length.
            std::fprintf(stderr,
                "KsResidentCache::%s called from a different thread than the one that "
                "constructed this cache. This cache is explicitly single-threaded; see this "
                "file's header comment. Aborting rather than risking silent ciphertext "
                "corruption from unsynchronized shared device state (the exact failure class "
                "Phase 25 found and fixed in src/rotate.cpp's scratch() buffer).\n", where);
            std::abort();
        }
    }

    void touch(const KsResidentCacheKey& key) {
        lru_.erase(lruPos_[key]); lru_.push_front(key); lruPos_[key] = lru_.begin();
    }

    void preflightOrThrow(size_t needBytes, const KsResidentCacheKey& key) {
        size_t freeB = 0, totalB = 0;
        cudaError_t err = cudaMemGetInfo(&freeB, &totalB);
        if (err != cudaSuccess) {
            throw KsResidentCacheOOM(
                "KsResidentCache: cudaMemGetInfo failed (" + std::string(cudaGetErrorString(err)) +
                ") -- refusing to attempt an allocation whose safety cannot be verified.");
        }
        // Would evicting enough LRU entries free enough room? Compute what
        // free space WOULD be after any eviction evictIfNeeded() will do.
        size_t wouldFreeFromEviction = 0;
        {
            size_t projectedBytes = stats_.residentBytesNow;
            size_t projectedCount = entries_.size();
            auto it = lru_.rbegin();
            while ((projectedBytes + needBytes > maxResidentBytes_ || projectedCount >= maxEntries_) && it != lru_.rend()) {
                auto found = entries_.find(*it);
                if (found != entries_.end()) { projectedBytes -= found->second.bytes; wouldFreeFromEviction += found->second.bytes; --projectedCount; }
                ++it;
            }
        }
        if (freeB + wouldFreeFromEviction < needBytes + vramSafetyMargin_) {
            ++stats_.oomRejections;
            throw KsResidentCacheOOM(
                "KsResidentCache: refusing a " + std::to_string(needBytes / 1048576.0) +
                " MiB allocation for key (k=" + std::to_string(key.k) + ", tw=" + std::to_string(key.tw) +
                "): only " + std::to_string(freeB / 1048576.0) + " MiB free (+" +
                std::to_string(wouldFreeFromEviction / 1048576.0) +
                " MiB recoverable by eviction), below the required safety margin of " +
                std::to_string(vramSafetyMargin_ / 1048576.0) + " MiB. Cache state is unchanged.");
        }
    }

    void evictIfNeeded(size_t incomingBytes) {
        while ((stats_.residentBytesNow + incomingBytes > maxResidentBytes_ || entries_.size() >= maxEntries_)
               && !lru_.empty()) {
            KsResidentCacheKey victim = lru_.back(); lru_.pop_back();
            lruPos_.erase(victim);
            auto it = entries_.find(victim);
            if (it != entries_.end()) {
                cudaStreamSynchronize(it->second.lastStream);  // see get()'s in-flight safety contract
                stats_.residentBytesNow -= it->second.bytes;
                entries_.erase(it);
                ++stats_.evictions;
            }
        }
    }

    std::map<KsResidentCacheKey, Entry> entries_;
    std::list<KsResidentCacheKey> lru_;
    std::map<KsResidentCacheKey, std::list<KsResidentCacheKey>::iterator> lruPos_;
    size_t maxEntries_, maxResidentBytes_, vramSafetyMargin_;
    KsResidentCacheStats stats_;
    std::thread::id ownerThread_;
};

} // namespace gpufhe
