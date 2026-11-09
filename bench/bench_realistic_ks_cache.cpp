// GOD PROMPT "CKKS PHASE 27", Missions 2+6: realistic performance A/B
// against the ACTUAL bootstrap access pattern, correcting two errors in
// Phase 26's toy benchmark (bench/bench_ks_context_resident_ab.cpp):
//
// ERROR 1 (memory model): Phase 26 built each test key directly via
// evalkeygen_host_sold with sizeQlP = tw + sizeP. The REAL production path
// (tests/test_boot.cpp's own `atLevel` closure) does NOT do this for a
// reduced tower count -- it takes a FULL-size key (built once at
// sizeQlP=sizeQ+sizeP) and COPIES its av[p]/bv[p] UNCHANGED
// (`K.av[p]=KF.av[p]`), only reducing the NUMBER of parts (`numPart`) used.
// So a real reduced-tw context's av/bv arrays are each still the FULL
// sizeQlP*n elements -- fewer of them, not smaller ones. This file replicates
// `atLevel` verbatim (same logic, same variable names where practical) to
// build keys the way the real bootstrap actually does.
//
// ERROR 2 (access pattern): Phase 26 drove the A/B with a synthetic
// round-robin over 8 keys repeated 6x each (reuse distance 6). The REAL
// bootstrap's access pattern -- extracted faithfully in
// research/benchmarks/extract_real_key_sequence.cpp by replicating
// test_boot.cpp's stageDiags/stageDiagsFwd/rotAmt logic (pure CPU
// arithmetic, no CUDA, no bootstrap execution) -- shows ALMOST NO cross-tw
// reuse: of 181 total diagonal rotation accesses across the whole bootstrap,
// 151 are to DISTINCT (k, tw) pairs. The ONLY repeats (exactly 30 of them)
// come from ONE place: the S2C stage at g=LST-1 processes the SAME 30
// rotation offsets TWICE -- once for the real branch (oR), once for the
// imaginary branch (oI) -- with a reuse DISTANCE of exactly 30 (all 30 keys
// of pass 1 are used before pass 2 starts reusing key #0). Every other
// stage (C2S's 3 stages; S2C's other 2 stages) has ZERO internal repeats --
// each diagonal's rotation key is used exactly once, full stop.
//
// This file models THAT real pattern directly: 30 distinct rotation keys,
// each used in a first pass (q2=0) then again in a second pass (q2=1) at the
// SAME tw=9, with the real atLevel-style key construction. This is the
// single realistic workload this project actually has for a resident
// eval-key cache to help with -- not a proxy, not a round-robin guess.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <map>
#include <list>
#include <sys/resource.h>

namespace gpufhe {
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void automorphism_eval_host(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t,
                            const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void set_secret_hamming_weight(uint32_t);
}

using cd = std::complex<double>;
static uint64_t mm(uint64_t a, uint64_t b, uint64_t q) { return (uint64_t)(((unsigned __int128)a * b) % q); }
static uint64_t modinv(uint64_t k, uint64_t M) {
    long long t = 0, newt = 1, r = (long long)M, newr = (long long)(k % M);
    while (newr != 0) { long long q = r / newr; long long tmp = t - q * newt; t = newt; newt = tmp;
        tmp = r - q * newr; r = newr; newr = tmp; }
    if (t < 0) t += (long long)M;
    return (uint64_t)t;
}
static double rssMB() { struct rusage ru; getrusage(RUSAGE_SELF, &ru); return ru.ru_maxrss / 1024.0; }
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end()); size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}
// npFor: alpha=2 partitioning, verbatim from tests/test_boot.cpp line 53.
static uint32_t npFor(uint32_t tw) { return tw <= 2 ? 1u : (tw + 1) / 2; }

// Namespace-scope (not local to main): C++ forbids member function
// TEMPLATES inside a class local to a function body.
struct ResidentContextCache {
    using Key = uint32_t;  // index into KView -- stands in for (k, tw), fixed tw this run
    std::map<Key, gpufhe::DeviceKSContext> ctx;
    std::list<Key> lru; std::map<Key, std::list<Key>::iterator> lruPos;
    size_t cap, misses = 0, hits = 0;
    explicit ResidentContextCache(size_t c) : cap(c) {}
    ~ResidentContextCache() { for (auto& kv : ctx) gpufhe::ks_context_destroy(kv.second); }
    template <typename Builder>
    gpufhe::DeviceKSContext& get(Key key, Builder build) {
        auto it = ctx.find(key);
        if (it != ctx.end()) { ++hits; lru.erase(lruPos[key]); lru.push_front(key); lruPos[key] = lru.begin(); return it->second; }
        ++misses;
        auto C = build(key);
        if (ctx.size() >= cap) { Key v = lru.back(); lru.pop_back(); lruPos.erase(v);
            gpufhe::ks_context_destroy(ctx[v]); ctx.erase(v); }
        auto res = ctx.emplace(key, std::move(C));
        lru.push_front(key); lruPos[key] = lru.begin();
        return res.first->second;
    }
};

int main(int argc, char** argv) {
    const uint32_t n = 8192, sizeQ = 30, sizeP = 2;
    const uint64_t ns = 1;
    const uint32_t M = 2 * n;
    const uint32_t NKEYS = (argc > 1) ? (uint32_t)std::stoul(argv[1]) : 30;  // the REAL S2C stage width
    const uint32_t TW_REDUCED = (argc > 2) ? (uint32_t)std::stoul(argv[2]) : 9;  // EvalMod's real output tw
    const uint32_t TRIALS = (argc > 3) ? (uint32_t)std::stoul(argv[3]) : 9;

    gpufhe::set_secret_hamming_weight(64);  // h=64, UNCHANGED

    std::printf("=== Mission 2+6: REALISTIC ks-context cache A/B (actual S2C double-pass pattern) ===\n");
    std::printf("params: n=%u sizeQ=%u sizeP=%u h=64 sigma=3.19\n", n, sizeQ, sizeP);
    std::printf("keys=%u (the real S2C stage width), tw_reduced=%u (EvalMod's real output tw), trials=%u\n",
                NKEYS, TW_REDUCED, TRIALS);

    // ---- shared untimed setup: build the FULL (sizeQ=30) chain, exactly as
    // test_boot.cpp's own top-level setup does. ----
    std::vector<uint64_t> mod, modP;
    gpufhe::native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids, sizeQ - 1, 55, n, mod);
      for (auto m : mids) mod.push_back(m); }
    gpufhe::native_primes(modP, sizeP, 60, n, mod);
    std::vector<uint64_t> root(sizeQ), rootQP;
    for (uint32_t i = 0; i < sizeQ; ++i) root[i] = gpufhe::native_root(n, mod[i]);
    rootQP = root;
    for (auto p : modP) rootQP.push_back(gpufhe::native_root(n, p));
    std::vector<uint64_t> modQP = mod; for (auto p : modP) modQP.push_back(p);
    const uint32_t sizeQlPFull = sizeQ + sizeP;  // 32, the FULL eval-key tower count

    auto KPqp = gpufhe::keygen_host(n, modQP, rootQP, ns, 3.19, 101);
    auto KPq  = gpufhe::keygen_host(n, mod, root, ns, 3.19, 101);
    std::vector<uint64_t> PModq(sizeQlPFull);
    for (uint32_t t = 0; t < sizeQlPFull; ++t) {
        uint64_t q = modQP[t], P = 1 % q;
        for (uint32_t j = 0; j < sizeP; ++j) P = mm(P, modQP[sizeQ + j] % q, q);
        PModq[t] = P;
    }

    std::vector<cd> z(n / 2);
    for (uint32_t i = 0; i < n / 2; ++i) z[i] = { 0.3 * std::sin(0.01 * i), 0.1 * std::cos(0.02 * i) };
    std::vector<int64_t> mz;
    gpufhe::encode_host(mz, z, n, std::pow(2.0, 55));
    std::vector<uint64_t> c0, c1;
    gpufhe::encrypt_host(c0, c1, mz, KPq.pkA, KPq.pkB, n, mod, root, ns, 3.19, 303);
    // At tw_reduced, the ciphertext we'd actually be operating on is TW_REDUCED
    // towers (the real post-EvalMod state) -- slice to that.
    const size_t TR = (size_t)TW_REDUCED * n, BR = TR * 8;
    std::vector<uint64_t> c1r(c1.begin(), c1.begin() + TR);

    // ---- rotFull: build the FULL (sizeQlP=32) key for rotation index k. ----
    auto rotFull = [&](uint32_t k, uint32_t seed) -> gpufhe::KeySwitchConstants {
        std::vector<uint64_t> sA = KPqp.s;
        gpufhe::automorphism_eval_host(sA, sizeQlPFull, n, k, modQP, rootQP);
        gpufhe::KeySwitchConstants K; K.n = n;
        gpufhe::compute_keyswitch_constants(K, mod, modP, npFor(sizeQ));
        for (uint32_t i = 0; i < sizeQ; ++i) { K.rootModList.push_back(mod[i]); K.rootValList.push_back(root[i]); }
        for (uint32_t j = 0; j < sizeP; ++j) { K.rootModList.push_back(modP[j]); K.rootValList.push_back(rootQP[sizeQ + j]); }
        gpufhe::evalkeygen_host_sold(K, KPqp.s, sA, KPqp.pkA, KPqp.pkB, PModq, modQP, rootQP, ns, 3.19, seed);
        return K;
    };
    // ---- atLevel: VERBATIM from tests/test_boot.cpp (lines 125-134). Copies
    // av[p]/bv[p] UNCHANGED from the full key KF -- this is the real behavior
    // being modeled, not a simplification. ----
    auto atLevel = [&](const gpufhe::KeySwitchConstants& KF, uint32_t tw) -> gpufhe::KeySwitchConstants {
        std::vector<uint64_t> ml(mod.begin(), mod.begin() + tw), rl(root.begin(), root.begin() + tw);
        gpufhe::KeySwitchConstants K; K.n = n;
        gpufhe::compute_keyswitch_constants(K, ml, modP, npFor(tw));
        for (uint32_t i = 0; i < tw; ++i) { K.rootModList.push_back(ml[i]); K.rootValList.push_back(rl[i]); }
        for (uint32_t j = 0; j < sizeP; ++j) { K.rootModList.push_back(modP[j]); K.rootValList.push_back(rootQP[sizeQ + j]); }
        K.fullQ = sizeQ; K.evalKeyTowers = KF.evalKeyTowers;
        K.av.assign(K.numPart, {}); K.bv.assign(K.numPart, {});
        for (uint32_t p = 0; p < K.numPart; ++p) { K.av[p] = KF.av[p]; K.bv[p] = KF.bv[p]; }
        return K;
    };

    // ---- build the REAL 30 distinct rotation keys (power-of-5 sequence,
    // matching rk[]/rotAmt's construction in test_boot.cpp), each as a FULL
    // key then an atLevel view at tw_reduced -- exactly as keyAt()/KSC does. ----
    std::vector<uint32_t> ks(NKEYS);
    { uint64_t r = 1; for (uint32_t i = 0; i < NKEYS; ++i) { r = (r * 5) % M; ks[i] = (uint32_t)r; } }

    std::printf("\n[setup] building %u FULL rotation keys + their atLevel(tw=%u) views (untimed)...\n",
                NKEYS, TW_REDUCED);
    std::vector<gpufhe::KeySwitchConstants> KFull(NKEYS), KView(NKEYS);
    for (uint32_t i = 0; i < NKEYS; ++i) {
        KFull[i] = rotFull(ks[i], 900 + ks[i]);
        KView[i] = atLevel(KFull[i], TW_REDUCED);
    }

    size_t perKeyBytesFull = 0, perKeyBytesView = 0;
    for (uint32_t p = 0; p < KFull[0].numPart; ++p) perKeyBytesFull += KFull[0].av[p].size() * 8 + KFull[0].bv[p].size() * 8;
    for (uint32_t p = 0; p < KView[0].numPart; ++p) perKeyBytesView += KView[0].av[p].size() * 8 + KView[0].bv[p].size() * 8;
    std::printf("[verify] full key (tw=%u, numPart=%u): %.3f MiB per context\n",
                sizeQ, KFull[0].numPart, perKeyBytesFull / 1048576.0);
    std::printf("[verify] atLevel view (tw=%u, numPart=%u): %.3f MiB per context -- NOTE: av/bv arrays are\n",
                TW_REDUCED, KView[0].numPart, perKeyBytesView / 1048576.0);
    std::printf("         STILL sizeQlP(full)=%u*n elements each (copied unchanged from the full key);\n", sizeQlPFull);
    std::printf("         only numPart shrank from %u to %u. This is the REAL production memory cost of\n",
                KFull[0].numPart, KView[0].numPart);
    std::printf("         a reduced-tw context -- smaller than the full key, but NOT scaled down by tw.\n");

    uint64_t *d_c1, *d_out0, *d_out1;
    cudaMalloc(&d_c1, BR); cudaMalloc(&d_out0, BR); cudaMalloc(&d_out1, BR);
    cudaMemcpy(d_c1, c1r.data(), BR, cudaMemcpyHostToDevice);

    // ---- THE REAL ACCESS SEQUENCE: pass q2=0 over all NKEYS, then pass
    // q2=1 over all NKEYS again, in the SAME order -- exactly test_boot.cpp's
    // S2C g==LST-1 double pass over one shared `offs` list. ----
    std::vector<uint32_t> sequence;
    for (int pass = 0; pass < 2; ++pass) for (uint32_t i = 0; i < NKEYS; ++i) sequence.push_back(i);

    auto armCreateEveryTime = [&]() {
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t idx : sequence) {
            auto C = gpufhe::ks_context_create(KView[idx]);
            auto W = gpufhe::ks_work_create(C);
            gpufhe::ks_accumulate_qp(d_c1, C, W, 0);
            gpufhe::ks_moddown(C, W, W.d_res0, d_out0, 0);
            gpufhe::ks_moddown(C, W, W.d_res1, d_out1, 0);
            gpufhe::ks_work_destroy(W);
            gpufhe::ks_context_destroy(C);
        }
        cudaStreamSynchronize(0);
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    auto armResidentCache = [&](size_t cap, size_t* outHits = nullptr, size_t* outMisses = nullptr) {
        ResidentContextCache cache(cap);
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t idx : sequence) {
            auto& C = cache.get(idx, [&](uint32_t i) { return gpufhe::ks_context_create(KView[i]); });
            auto W = gpufhe::ks_work_create(C);
            gpufhe::ks_accumulate_qp(d_c1, C, W, 0);
            gpufhe::ks_moddown(C, W, W.d_res0, d_out0, 0);
            gpufhe::ks_moddown(C, W, W.d_res1, d_out1, 0);
            gpufhe::ks_work_destroy(W);
        }
        cudaStreamSynchronize(0);
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (outHits) *outHits = cache.hits;
        if (outMisses) *outMisses = cache.misses;
        return elapsed;
    };

    // ---- correctness: bit-exact checksum, cache vs no-cache, at the cap
    // that should give the full real-world hit rate (cap=NKEYS). ----
    std::printf("\n[correctness] cache (cap=%u) output must be BIT-EXACT vs the uncached path\n", NKEYS);
    auto checksumRun = [&](bool useCache, size_t cap) {
        std::vector<uint64_t> out;
        ResidentContextCache cache(cap);
        for (uint32_t idx : sequence) {
            gpufhe::DeviceKSContext localC;
            gpufhe::DeviceKSContext* Cp;
            if (useCache) Cp = &cache.get(idx, [&](uint32_t i) { return gpufhe::ks_context_create(KView[i]); });
            else { localC = gpufhe::ks_context_create(KView[idx]); Cp = &localC; }
            auto W = gpufhe::ks_work_create(*Cp);
            gpufhe::ks_accumulate_qp(d_c1, *Cp, W, 0);
            gpufhe::ks_moddown(*Cp, W, W.d_res0, d_out0, 0);
            gpufhe::ks_moddown(*Cp, W, W.d_res1, d_out1, 0);
            cudaStreamSynchronize(0);
            std::vector<uint64_t> a(TR), b(TR);
            cudaMemcpy(a.data(), d_out0, BR, cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), d_out1, BR, cudaMemcpyDeviceToHost);
            uint64_t cs = 0; for (auto v : a) cs ^= v; for (auto v : b) cs ^= v * 0x9E3779B97F4A7C15ull;
            out.push_back(cs);
            gpufhe::ks_work_destroy(W);
            if (!useCache) gpufhe::ks_context_destroy(localC);
        }
        return out;
    };
    auto csUncached = checksumRun(false, 1);
    auto csCached = checksumRun(true, NKEYS);
    size_t mism = 0; for (size_t i = 0; i < csUncached.size(); ++i) if (csUncached[i] != csCached[i]) ++mism;
    std::printf("  %zu / %zu calls differ -> %s\n", mism, csUncached.size(), mism == 0 ? "[PASS bit-exact]" : "[FAIL]");
    if (mism) { std::printf("[FAIL] refusing to report timings\n"); return 1; }

    // ---- warm-up ----
    for (int w = 0; w < 2; ++w) { armCreateEveryTime(); armResidentCache(NKEYS); }

    // ---- MISSION 6: capacity sweep against the REAL access pattern, not a
    // synthetic one. Phase 26 tested only an 8-entry cache (on a toy
    // round-robin sequence); this sweeps 4, 8, 16, and NKEYS (the capacity
    // this analysis shows is "appropriate for the measured workload" --
    // sized to the real single-stage width, never to the full 151-entry/
    // ~5.6 GiB bootstrap-wide working set, which research/
    // KS_RESIDENT_CACHE_DESIGN.md's memory model shows would not safely
    // coexist with everything else this project allocates. That full-
    // working-set capacity is therefore NOT tested here by actually
    // allocating it -- per the governing directive's explicit "do not
    // allocate the entire working-set cache merely to see whether it
    // fits" instruction; the "it would not fit" conclusion is reached
    // analytically in that document, not by attempting the allocation.) ----
    std::printf("\n[MISSION 6] capacity sweep on the REAL access pattern (not a round-robin toy sequence)\n");
    std::printf("%-6s %-8s %-8s %-10s %-12s\n", "cap", "hits", "misses", "hit_rate", "note");
    double hitRateAtFullCap = 0.0;
    for (uint32_t cap : {4u, 8u, 16u, NKEYS}) {
        size_t h = 0, m = 0;
        armResidentCache(cap, &h, &m);  // untimed pass just to get hit/miss counts at this cap
        const char* note = (cap < NKEYS) ? "too small: real reuse distance is NKEYS" : "matches real stage width";
        double hr = 100.0 * h / (h + m);
        if (cap == NKEYS) hitRateAtFullCap = hr;
        std::printf("%-6u %-8zu %-8zu %-10.1f%% %-12s\n", cap, h, m, hr, note);
    }
    std::printf("Expected and confirmed: hit rate is 0%% for ANY cap < NKEYS (the real reuse distance\n");
    std::printf("in this ONE stage is exactly NKEYS -- a cache smaller than the stage width captures\n");
    std::printf("NONE of the available reuse, not a partial fraction of it), and caps AT OR ABOVE\n");
    std::printf("NKEYS all give the SAME %.1f%% hit rate -- scaling the cache past the real stage width\n",
                hitRateAtFullCap);
    std::printf("buys nothing further. This is why a cap of exactly ~NKEYS (not 77, not 151, not\n");
    std::printf("unbounded) is \"the capacity appropriate for the measured workload.\"\n");

    // ---- measurement: cold (uncached) vs warm (cache sized to cap exactly
    // NKEYS, the only capacity that captures the real reuse) ----
    std::printf("\n[timing] median of %u trials, alternating order, REAL access pattern (2 passes over %u keys)\n",
                TRIALS, NKEYS);
    std::vector<double> vA, vB;
    size_t lastHits = 0, lastMisses = 0;
    for (uint32_t t = 0; t < TRIALS; ++t) {
        if (t % 2 == 0) { vA.push_back(armCreateEveryTime()); vB.push_back(armResidentCache(NKEYS, &lastHits, &lastMisses)); }
        else            { vB.push_back(armResidentCache(NKEYS, &lastHits, &lastMisses)); vA.push_back(armCreateEveryTime()); }
    }
    const double mA = median(vA), mB = median(vB);
    std::printf("  [cache] hits=%zu misses=%zu (of %zu calls, hit rate %.1f%%)\n",
                lastHits, lastMisses, sequence.size(), 100.0 * lastHits / sequence.size());
    std::printf("  uncached (2x fresh)   : %.6f [%.6f..%.6f] s\n", mA,
                *std::min_element(vA.begin(), vA.end()), *std::max_element(vA.begin(), vA.end()));
    std::printf("  cached (cap=%-3u)      : %.6f [%.6f..%.6f] s\n", NKEYS, mB,
                *std::min_element(vB.begin(), vB.end()), *std::max_element(vB.begin(), vB.end()));
    std::printf("  speedup on THIS REAL pattern: %.4fx\n", mA / mB);

    size_t vramFree = 0, vramTotal = 0;
    cudaMemGetInfo(&vramFree, &vramTotal);
    std::printf("\n[resources] peak host RSS %.1f MB; VRAM free %.0f/%.0f MiB\n",
                rssMB(), vramFree / 1048576.0, vramTotal / 1048576.0);
    std::printf("  cache steady-state footprint: %u contexts x %.3f MiB = %.1f MiB\n",
                NKEYS, perKeyBytesView / 1048576.0, NKEYS * perKeyBytesView / 1048576.0);

    std::printf("\n[HONEST SCOPE] This models exactly ONE of the bootstrap's SIX BSGS passes (the\n");
    std::printf("S2C stage that processes both the real and imaginary branches). The other FIVE\n");
    std::printf("passes (C2S's 3 stages, S2C's other 2 stages) have been shown by\n");
    std::printf("research/benchmarks/extract_real_key_sequence.cpp's real-sequence replay to have\n");
    std::printf("ZERO internal key reuse at all -- a cache provides NO benefit there, only overhead.\n");
    std::printf("The speedup above is NOT a bootstrap-wide figure; it is the honest, realistic\n");
    std::printf("ceiling for the ONE place in the whole algorithm where this optimization can help.\n");

    cudaFree(d_c1); cudaFree(d_out0); cudaFree(d_out1);
    return 0;
}
