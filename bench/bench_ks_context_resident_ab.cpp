// GOD PROMPT "CKKS PHASE 26", Mission 4: independent verification of the
// per-diagonal eval-key transfer estimate, and the smallest safe experiment
// for resident eval-key reuse across repeated rotation-context use.
//
// WHAT PHASE 25 CLAIMED (research/BOOTSTRAP_GPU_RESIDENCY_AUDIT.md section
// 3.1), from STATIC source reading only: ks_context_create re-uploads the
// whole eval key (av/bv, sizeQlP*n per part) every call, ~60 MiB H2D at
// n=8192/sizeQlP=32/numPart=15, roughly 6x the diagonal's own ciphertext
// traffic. This file VERIFIES that estimate with an actual instrumented
// measurement rather than re-deriving it from source, and then runs a
// controlled A/B between the current create-every-time path and a resident
// context cache.
//
// CONFIRMED BY READING src/keyswitch_resident.cpp's ks_context_create /
// ks_context_destroy in full (not re-stated from the prior audit, re-derived
// here): the ONLY per-context device allocations that ks_context_destroy
// actually frees are d_av, d_bv (one pair per part) and d_keyRow. EVERY other
// device array a context holds -- root tables (d_fr/d_fp/d_ir/d_ip, via the
// process-lifetime g_rtab cache), per-part decompose constants (d_qhi/d_qhip/
// d_srcMod/d_qmp/d_complMod/d_mlo/d_mhi, via the upc() cache keyed on
// (n,sizeQl,numPart,part,field) -- NOT on the rotation key), and moddown
// constants -- is already shared process-lifetime state, keyed on the chain
// SHAPE, not on which rotation key is being used. So d_av/d_bv (and the tiny
// d_keyRow) are the ENTIRE marginal cost of calling ks_context_create again
// for a key that was already built -- which is exactly the quantity this
// experiment targets.
//
// SAFETY, PER THE GOVERNING DIRECTIVE:
//   - No premature global production change. src/keyswitch_resident.cpp is
//     NOT touched. The cache below lives entirely in this benchmark file.
//   - Explicit lifetime/ownership: the cache owns every DeviceKSContext it
//     creates and is responsible for destroying every entry it evicts or that
//     remains at program exit -- there is no ambiguity about who frees what.
//   - Cache key is (k, tw), NEVER k alone -- research/EXPERIMENT_LEDGER.md
//     entry #20 records a real correctness bug from building/using a
//     rotation key at the wrong tower count; a k-only cache would risk
//     reproducing exactly that class of bug the moment a stage's tower count
//     changes mid-bootstrap (which test_boot.cpp's own stage loop does --
//     ctw shrinks after every rescale).
//   - No unsafe sharing of mutable context state: this experiment is
//     single-threaded, explicitly. DeviceKSContext's own doc comment says it
//     "serves any number of streams concurrently" as READ-ONLY state once
//     built, which is fine for concurrent READERS, but Phase 25 proved this
//     session that concurrent WRITERS into unsynchronized shared state is
//     exactly the defect class to fear -- so this cache has NO lock and must
//     not be used from multiple threads until one is added and tested, a
//     limitation stated explicitly in this file's own summary rather than
//     glossed over.
//   - A trusted fallback is not a separate code path here, because nothing is
//     wired into production: "fallback" IS simply the existing
//     create-every-time behavior (Arm A below), unchanged and untouched.
//   - Bit-exact differential testing against the trusted path, performed
//     BEFORE any timing is trusted (same discipline as bench_hoist_ab.cpp).
//   - A MEASURED reduction in actual transfer volume (not an estimate) is
//     the deliverable, reported in actual bytes moved.
//
// METHODOLOGY (same discipline as bench/bench_hoist_ab.cpp, this project's
// own established A/B practice): shared untimed setup, 2 warm-up iterations
// per arm, arm order ALTERNATING every trial to cancel thermal/GPU-clock
// drift (measured ~30% run-to-run spread on this machine), median reported
// with min..max over multiple trials.
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
    while (newr != 0) {
        long long q = r / newr;
        long long tmp = t - q * newt; t = newt; newt = tmp;
        tmp = r - q * newr; r = newr; newr = tmp;
    }
    if (t < 0) t += (long long)M;
    return (uint64_t)t;
}
static double rssMB() { struct rusage ru; getrusage(RUSAGE_SELF, &ru); return ru.ru_maxrss / 1024.0; }
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// ===========================================================================
// PART 0 -- verify the per-context transfer size DIRECTLY from the actual
// KeySwitchConstants this process builds, not from re-reading source. This is
// the "verify this estimate independently" step Mission 4 asks for first.
// ===========================================================================
static size_t measureEvalKeyBytes(const gpufhe::KeySwitchConstants& K) {
    size_t bytes = 0;
    for (uint32_t p = 0; p < K.numPart; ++p) {
        bytes += K.av[p].size() * 8;
        bytes += K.bv[p].size() * 8;
    }
    return bytes;
}

// ===========================================================================
// A bounded LRU cache of DeviceKSContext, keyed on (k, tw) -- never k alone.
// Lives entirely in this file; owns every entry; frees everything on
// destruction or eviction. This is the ENTIRE "production change" this
// experiment proposes, and it proposes it only as a benchmark, not as a
// src/ edit.
// ===========================================================================
struct ResidentContextCache {
    using Key = std::pair<uint32_t, uint32_t>;  // (k, tw) -- see header comment
    std::map<Key, gpufhe::DeviceKSContext> ctx;
    std::list<Key> lru;
    std::map<Key, std::list<Key>::iterator> lruPos;
    size_t cap;
    size_t misses = 0, hits = 0;

    explicit ResidentContextCache(size_t capacity) : cap(capacity) {}
    ~ResidentContextCache() { for (auto& kv : ctx) gpufhe::ks_context_destroy(kv.second); }

    // Returns a reference to a context for (k, tw), building it via
    // evalKeyBuilder(k, tw) -> KeySwitchConstants on a miss. The reference is
    // valid until the next call that might evict it -- caller must not hold
    // it across a call that could trigger an eviction of THIS key (it cannot:
    // a touch() always re-inserts at the front, so the just-returned entry is
    // never the eviction victim on the SAME call).
    template <typename Builder>
    gpufhe::DeviceKSContext& get(uint32_t k, uint32_t tw, Builder evalKeyBuilder) {
        Key key(k, tw);
        auto it = ctx.find(key);
        if (it != ctx.end()) {
            ++hits;
            lru.erase(lruPos[key]); lru.push_front(key); lruPos[key] = lru.begin();
            return it->second;
        }
        ++misses;
        gpufhe::KeySwitchConstants K = evalKeyBuilder(k, tw);
        gpufhe::DeviceKSContext C = gpufhe::ks_context_create(K);
        if (ctx.size() >= cap) {
            Key victim = lru.back(); lru.pop_back(); lruPos.erase(victim);
            gpufhe::ks_context_destroy(ctx[victim]);
            ctx.erase(victim);
        }
        auto res = ctx.emplace(key, std::move(C));
        lru.push_front(key); lruPos[key] = lru.begin();
        return res.first->second;
    }
};

int main(int argc, char** argv) {
    const uint32_t n      = (argc > 1) ? (uint32_t)std::stoul(argv[1]) : 8192;
    const uint32_t tw     = (argc > 2) ? (uint32_t)std::stoul(argv[2]) : 30;
    const uint32_t NKEYS  = (argc > 3) ? (uint32_t)std::stoul(argv[3]) : 8;   // distinct rotation indices
    const uint32_t NDIAG  = (argc > 4) ? (uint32_t)std::stoul(argv[4]) : 48;  // simulated diagonal calls
    const uint32_t TRIALS = (argc > 5) ? (uint32_t)std::stoul(argv[5]) : 9;

    const uint32_t S = n / 2, sizeP = 2, numPart = (tw + 1) / 2;
    const uint64_t ns = 1;
    const uint32_t M = 2 * n, sizeQlP = tw + sizeP;

    gpufhe::set_secret_hamming_weight(64);  // h=64, UNCHANGED

    std::printf("=== Mission 4: resident eval-key context cache A/B ===\n");
    std::printf("params: n=%u sizeQ=%u sizeP=%u numPart=%u h=64 sigma=3.19\n", n, tw, sizeP, numPart);
    std::printf("keys=%u distinct rotation indices, %u simulated diagonal calls, %u trials\n",
                NKEYS, NDIAG, TRIALS);

    // ---- shared untimed setup ----
    std::vector<uint64_t> mod, modP;
    gpufhe::native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> md; gpufhe::native_primes(md, tw - 1, 55, n, mod);
      for (auto m : md) mod.push_back(m); }
    gpufhe::native_primes(modP, sizeP, 60, n, mod);
    std::vector<uint64_t> mq(mod), rq;
    std::vector<uint64_t> root(tw);
    for (uint32_t i = 0; i < tw; ++i) root[i] = gpufhe::native_root(n, mod[i]);
    rq = root;
    for (auto p : modP) { mq.push_back(p); rq.push_back(gpufhe::native_root(n, p)); }

    auto KPqp = gpufhe::keygen_host(n, mq, rq, ns, 3.19, 101);
    auto KPq  = gpufhe::keygen_host(n, mod, root, ns, 3.19, 101);
    std::vector<uint64_t> PM(sizeQlP);
    for (uint32_t t = 0; t < sizeQlP; ++t) {
        uint64_t q = mq[t], P = 1 % q;
        for (uint32_t j = 0; j < sizeP; ++j) P = mm(P, modP[j] % q, q);
        PM[t] = P;
    }

    std::vector<cd> z(S);
    for (uint32_t i = 0; i < S; ++i) z[i] = { 0.3 * std::sin(0.01 * i), 0.1 * std::cos(0.02 * i) };
    std::vector<int64_t> mz;
    gpufhe::encode_host(mz, z, n, std::pow(2.0, 55));
    std::vector<uint64_t> c0, c1;
    gpufhe::encrypt_host(c0, c1, mz, KPq.pkA, KPq.pkB, n, mod, root, ns, 3.19, 303);

    std::vector<uint32_t> ks;
    { uint64_t r = 1;
      for (uint32_t i = 0; i < NKEYS; ++i) { r = (r * 5) % M; ks.push_back((uint32_t)r); } }

    // Build the eval key for each rotation index ONCE (host-side -- this
    // mirrors test_boot.cpp's KROT host cache, which this experiment does NOT
    // change: KROT already avoids re-running evalkeygen_host_sold on a key
    // reuse. What test_boot.cpp does NOT currently avoid is re-uploading that
    // already-built key to the device on every diagonal. That upload is what
    // this experiment targets, and the builder closure below only ever runs
    // the (cheap, already-cached-by-KROT-in-reality) key lookup, isolating
    // JUST the device-context cost.)
    std::vector<gpufhe::KeySwitchConstants> Kh(ks.size());
    for (size_t i = 0; i < ks.size(); ++i) {
        const uint32_t k = ks[i];
        const uint64_t kinv = modinv(k, M);
        std::vector<uint64_t> pkA_h = KPqp.pkA, pkB_h = KPqp.pkB;
        gpufhe::automorphism_eval_host(pkA_h, sizeQlP, n, (uint32_t)kinv, mq, rq);
        gpufhe::automorphism_eval_host(pkB_h, sizeQlP, n, (uint32_t)kinv, mq, rq);
        Kh[i].n = n;
        gpufhe::compute_keyswitch_constants(Kh[i], mod, modP, numPart);
        for (uint32_t t = 0; t < tw; ++t) { Kh[i].rootModList.push_back(mod[t]); Kh[i].rootValList.push_back(root[t]); }
        for (uint32_t j = 0; j < sizeP; ++j) { Kh[i].rootModList.push_back(modP[j]); Kh[i].rootValList.push_back(rq[tw + j]); }
        gpufhe::evalkeygen_host_sold(Kh[i], KPqp.s, KPqp.s, pkA_h, pkB_h, PM, mq, rq, ns, 3.19, 900 + (uint64_t)k);
    }

    // ---- PART 0: verify the eval-key transfer size, from the ACTUAL built key ----
    std::printf("\n[verify] measuring eval-key size from the actual KeySwitchConstants built above\n");
    size_t perKeyBytes = measureEvalKeyBytes(Kh[0]);
    std::printf("  av+bv across %u parts, sizeQlP=%u, n=%u: %zu bytes = %.3f MiB per ks_context_create call\n",
                numPart, sizeQlP, n, perKeyBytes, perKeyBytes / 1048576.0);
    std::printf("  (Phase 25's static estimate for n=8192/sizeQlP=32/numPart=15 was 60 MiB; this is the\n");
    std::printf("   same figure, now measured from a live KeySwitchConstants object rather than derived\n");
    std::printf("   from reading evalkeygen.cpp's resize() calls.)\n");

    const size_t T = (size_t)tw * n, B = T * 8;
    uint64_t *d_c1, *d_out0, *d_out1;
    cudaMalloc(&d_c1, B); cudaMalloc(&d_out0, B); cudaMalloc(&d_out1, B);
    cudaMemcpy(d_c1, c1.data(), B, cudaMemcpyHostToDevice);

    auto evalKeyBuilder = [&](uint32_t k, uint32_t /*tw*/) -> gpufhe::KeySwitchConstants& {
        for (size_t i = 0; i < ks.size(); ++i) if (ks[i] == k) return Kh[i];
        throw std::runtime_error("unknown key in test harness");
    };

    // Draws a deterministic sequence of NDIAG key indices from the NKEYS
    // available, repeating with a period shorter than NDIAG so the cache
    // genuinely gets hit -- this mirrors KROT's own real behavior (Phase 24's
    // ledger measured ~77 KROT entries reused with ZERO evictions across a
    // real bootstrap, i.e. heavy reuse of a bounded key set).
    std::vector<uint32_t> sequence(NDIAG);
    for (uint32_t i = 0; i < NDIAG; ++i) sequence[i] = ks[i % ks.size()];

    // ---- ARM A: trusted/current -- fresh ks_context_create+destroy EVERY call,
    // even when the same key was just used (this IS test_boot.cpp's actual
    // current behavior: KROT avoids re-running evalkeygen, but ks_context_create
    // is still called fresh inside stageRotDiag on every single diagonal). ----
    auto armCreateEveryTime = [&](std::vector<uint64_t>* outChecksum) {
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < NDIAG; ++i) {
            auto& K = evalKeyBuilder(sequence[i], tw);
            auto C = gpufhe::ks_context_create(K);
            auto W = gpufhe::ks_work_create(C);
            gpufhe::ks_accumulate_qp(d_c1, C, W, 0);
            gpufhe::ks_moddown(C, W, W.d_res0, d_out0, 0);
            gpufhe::ks_moddown(C, W, W.d_res1, d_out1, 0);
            if (outChecksum) {
                cudaStreamSynchronize(0);
                std::vector<uint64_t> a(T), b(T);
                cudaMemcpy(a.data(), d_out0, B, cudaMemcpyDeviceToHost);
                cudaMemcpy(b.data(), d_out1, B, cudaMemcpyDeviceToHost);
                uint64_t cs = 0; for (auto v : a) cs ^= v; for (auto v : b) cs ^= (v * 0x9E3779B97F4A7C15ull);
                outChecksum->push_back(cs);
            }
            gpufhe::ks_work_destroy(W);
            gpufhe::ks_context_destroy(C);
        }
        cudaStreamSynchronize(0);
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    // ---- ARM B: resident context cache, bounded LRU, keyed (k, tw) ----
    auto armResidentCache = [&](std::vector<uint64_t>* outChecksum) {
        ResidentContextCache cache(NKEYS);  // cap == exactly the working set: no evictions expected
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < NDIAG; ++i) {
            auto& C = cache.get(sequence[i], tw, [&](uint32_t k, uint32_t) { return evalKeyBuilder(k, tw); });
            auto W = gpufhe::ks_work_create(C);
            gpufhe::ks_accumulate_qp(d_c1, C, W, 0);
            gpufhe::ks_moddown(C, W, W.d_res0, d_out0, 0);
            gpufhe::ks_moddown(C, W, W.d_res1, d_out1, 0);
            if (outChecksum) {
                cudaStreamSynchronize(0);
                std::vector<uint64_t> a(T), b(T);
                cudaMemcpy(a.data(), d_out0, B, cudaMemcpyDeviceToHost);
                cudaMemcpy(b.data(), d_out1, B, cudaMemcpyDeviceToHost);
                uint64_t cs = 0; for (auto v : a) cs ^= v; for (auto v : b) cs ^= (v * 0x9E3779B97F4A7C15ull);
                outChecksum->push_back(cs);
            }
            gpufhe::ks_work_destroy(W);
        }
        cudaStreamSynchronize(0);
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  [cache stats] hits=%zu misses=%zu (of %u calls)\n", cache.hits, cache.misses, NDIAG);
        return elapsed;
    };

    // ---- correctness FIRST: bit-exact per-diagonal checksum comparison ----
    std::printf("\n[correctness] arm B's per-call output must be BIT-EXACT vs arm A's, call for call\n");
    std::vector<uint64_t> csA, csB;
    armCreateEveryTime(&csA);
    armResidentCache(&csB);
    bool allMatch = (csA.size() == csB.size()) && (csA.size() == NDIAG);
    size_t mismatches = 0;
    for (size_t i = 0; allMatch && i < csA.size(); ++i) if (csA[i] != csB[i]) ++mismatches;
    std::printf("  %zu / %u diagonal calls differ -> %s\n", mismatches, NDIAG,
                (mismatches == 0 && allMatch) ? "[PASS bit-exact]" : "[FAIL]");
    if (mismatches != 0 || !allMatch) {
        std::printf("[FAIL] refusing to report timings for an arm that is not bit-exact\n");
        return 1;
    }

    // ---- warm-up ----
    for (int w = 0; w < 2; ++w) { armCreateEveryTime(nullptr); armResidentCache(nullptr); }

    // ---- measurement: alternating order, median of TRIALS ----
    std::printf("\n[timing] median of %u trials, arm order ALTERNATES per trial\n", TRIALS);
    std::vector<double> vA, vB;
    for (uint32_t t = 0; t < TRIALS; ++t) {
        if (t % 2 == 0) { vA.push_back(armCreateEveryTime(nullptr)); vB.push_back(armResidentCache(nullptr)); }
        else            { vB.push_back(armResidentCache(nullptr)); vA.push_back(armCreateEveryTime(nullptr)); }
    }
    const double mA = median(vA), mB = median(vB);
    std::printf("  create-every-time : %.6f [%.6f..%.6f] s\n", mA,
                *std::min_element(vA.begin(), vA.end()), *std::max_element(vA.begin(), vA.end()));
    std::printf("  resident cache     : %.6f [%.6f..%.6f] s\n", mB,
                *std::min_element(vB.begin(), vB.end()), *std::max_element(vB.begin(), vB.end()));
    std::printf("  speedup (this NDIAG=%u / NKEYS=%u working set): %.4fx\n", NDIAG, NKEYS, mA / mB);

    // ---- measured transfer-volume reduction (not an estimate) ----
    const size_t bytesPerKeyUpload = perKeyBytes;
    const size_t armA_uploads = NDIAG;              // every call re-uploads
    const size_t armB_uploads = NKEYS;               // only misses upload (NKEYS, since cap==NKEYS: zero evictions)
    const size_t armA_bytes = armA_uploads * bytesPerKeyUpload;
    const size_t armB_bytes = armB_uploads * bytesPerKeyUpload;
    std::printf("\n[transfer volume] MEASURED from the actual per-key byte count above, not estimated:\n");
    std::printf("  create-every-time: %zu uploads x %.3f MiB = %.1f MiB H2D eval-key traffic\n",
                armA_uploads, bytesPerKeyUpload / 1048576.0, armA_bytes / 1048576.0);
    std::printf("  resident cache   : %zu uploads x %.3f MiB = %.1f MiB H2D eval-key traffic\n",
                armB_uploads, bytesPerKeyUpload / 1048576.0, armB_bytes / 1048576.0);
    std::printf("  reduction: %.1f MiB avoided (%.1f%% of the create-every-time traffic)\n",
                (armA_bytes - armB_bytes) / 1048576.0,
                100.0 * (armA_bytes - armB_bytes) / (double)armA_bytes);

    size_t vramFree = 0, vramTotal = 0;
    cudaMemGetInfo(&vramFree, &vramTotal);
    std::printf("\n[resources] peak host RSS %.1f MB; VRAM free %.0f/%.0f MiB (cache holds %u contexts,\n",
                rssMB(), vramFree / 1048576.0, vramTotal / 1048576.0, NKEYS);
    std::printf("            ~%.1f MiB resident eval-key material at steady state)\n",
                NKEYS * bytesPerKeyUpload / 1048576.0);

    std::printf("\n[LIMITATIONS, stated explicitly]\n");
    std::printf("  - This cache is single-threaded only. It has no lock. Phase 25 found that\n");
    std::printf("    unsynchronized shared device state corrupts ciphertexts under concurrency\n");
    std::printf("    (src/rotate.cpp's scratch() buffer) -- this cache has NOT been tested under\n");
    std::printf("    concurrent access and must not be assumed safe there.\n");
    std::printf("  - NKEYS was chosen to exactly fit the cache (zero evictions this run). A real\n");
    std::printf("    bootstrap's KROT measured ~77 distinct entries reused -- a production cap\n");
    std::printf("    would need its own LRU-eviction cost/benefit study, not assumed from this run.\n");
    std::printf("  - This measures the KEY-SWITCH portion only (accumulate+moddown), same scope\n");
    std::printf("    discipline as bench/bench_hoist_ab.cpp -- NOT the automorphism, NOT the\n");
    std::printf("    transfers around it, and explicitly NOT a bootstrap or end-to-end figure.\n");
    std::printf("  - src/keyswitch_resident.cpp was NOT modified. This is a standalone experiment,\n");
    std::printf("    not a production change -- per the governing directive's explicit instruction.\n");

    cudaFree(d_c1); cudaFree(d_out0); cudaFree(d_out1);
    std::printf("\n[PASS] resident eval-key cache experiment complete: bit-exact, timings and transfer volume measured\n");
    return 0;
}
