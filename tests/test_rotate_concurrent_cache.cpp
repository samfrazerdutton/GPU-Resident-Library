// GOD PROMPT "CKKS PHASE 25": standalone reproducer for a CONCURRENCY defect
// in the process-global device-table caches on the resident rotation path.
//
// WHY THIS TEST EXISTS
//
// Phase 24 recorded two Rust test failures and concluded they were
// "deterministic test-order artifacts" from
// `budget_exhaustion_falls_back_safely_and_stays_correct`'s deliberate
// 16MiB-budget-exceeding design, and that "skipping that one test makes
// everything else pass unconditionally".
//
// Phase 25 re-ran that suite and found the conclusion is only HALF right.
// There are TWO independent defects, which Phase 24 conflated:
//
//   (1) A genuine test-ISOLATION defect in the registered-memory budget (the
//       global counter is monotonic with no release path). This is what
//       Phase 24 actually diagnosed, and it reproduces EXACTLY as Phase 24
//       described -- but only when the suite runs SERIALLY
//       (`--test-threads=1`), where the test order is deterministic.
//
//   (2) A CONCURRENCY defect on the rotation path, which is what fails in the
//       default (parallel) `cargo test` run, and which has nothing to do with
//       the registered-memory budget at all. Measured this round:
//
//         cargo test --release rotation:: -- --test-threads=1   -> 10 passed, 0 failed
//         cargo test --release rotation::                       ->  6 passed, 4 failed
//         cargo test --release -- --skip budget_exhaustion      -> 3 rotation tests STILL fail
//
//       So Phase 24's "skipping that one test makes everything else pass
//       unconditionally" is FALSE in the default parallel mode. The rotation
//       failures survive skipping it, vanish under serial execution, and the
//       failing SET varies run to run (3 failures, then 2, then 4) -- the
//       signature of a data race, not of an ordering artifact.
//
// THE MECHANISM -- AND A HYPOTHESIS THIS TEST ITSELF FALSIFIED
//
// The first hypothesis was the four unsynchronized process-global `std::map`
// caches (there is not one mutex in either file):
//
//   src/keyswitch_resident.cpp:30   upc()         static std::map<uint64_t,uint64_t*>
//   src/keyswitch_resident.cpp:106  g_rtab        static std::map<pair<uint32_t,uint64_t>,RTab>
//   src/rotate.cpp:29               dev_tab()     static std::map<pair<uint32_t,uint64_t>,DevTab>
//   src/rotate.cpp:130              autoPermMap() static std::map<pair<uint32_t,uint32_t>,uint32_t*>
//
// Those ARE a latent data race (C++11 guarantees thread-safe INITIALIZATION of
// a function-local static, nothing about concurrent MUTATION of the object
// afterwards). But PHASE B below drives exactly that path -- 9 threads calling
// `rotate_ct_resident`, hence `ks_context_create`, on cold caches with distinct
// moduli -- and it passed 6 runs out of 6, bit-exact. So the map race is NOT
// what the Rust rotation tests are hitting, and this test keeps Phase B
// permanently as the CONTROL that says so. Recording a falsified hypothesis
// rather than deleting it: the next person to see these maps should know they
// were suspected, tested, and found not to be the active defect.
//
// The ACTUAL mechanism is a shared device SCRATCH BUFFER, not a map:
//
//   src/rotate.cpp:48  uint64_t* scratch(size_t bytes){
//                          static uint64_t* buf=nullptr; static size_t cap=0;
//                          if(bytes>cap){ if(buf) cudaFree(buf); cudaMalloc(&buf,bytes); cap=bytes; }
//                          return buf; }
//
// ONE buffer, process-wide, handed out to every caller. `xf_multi` (the batched
// all-towers NTT/INTT) stages its data through it, together with a matching set
// of file-scope metadata arrays (d_rA/d_rB/d_md/d_ni/d_nip via scratchPtrArr /
// scratchU64Arr). Two threads in `xf_multi` at once therefore upload into the
// SAME device memory and read back each other's data. This is not a subtle
// ordering window -- it is unconditional corruption whenever the calls overlap.
//
// Who reaches it:
//   pt_to_eval_host      -> xf_multi             (1 call)
//   automorphism_eval_host -> xf_multi           (2 calls: to coeff, back to eval)
//   rotate_ct_host       -> automorphism_eval_host + keyswitch_core_resident
//
// And who does NOT: `rotate_ct_resident`, which allocates its own `d_sc` per
// call and permutes via the fully device-side `automorphism_eval_fast_device`.
// That asymmetry is precisely why Phase B passes while Phases C and D fail, and
// it is the single most useful fact in this whole investigation: the GPU-
// RESIDENT path is clean; the HOST-orchestrated path is the one that corrupts
// under concurrency.
//
// For contrast, the registered-memory pool added in Phase 20/22 DOES guard its
// global state with `registered_mutex()`. The pattern is understood in this
// codebase; these performance globals (added to stop per-call cudaMalloc/Free
// from regressing the bootstrap 53 -> 101 s) simply never got it.
//
// WHAT EACH PHASE DOES
//
//   Phase A  Build fixtures and compute every reference SERIALLY. These are the
//            ground truth, computed with no concurrency anywhere.
//   Phase B  CONTROL: concurrent `rotate_ct_resident`. Exercises the global
//            std::map caches on a cold process. Expected to PASS -- and if it
//            ever starts failing, the falsified map hypothesis is back on the
//            table and should be re-opened, not explained away.
//   Phase C  Concurrent `pt_to_eval_host`. The tightest possible isolation of
//            `xf_multi` + `scratch()`: pure function, no keys, no ciphertext,
//            deterministic output. If this fails, the shared scratch buffer is
//            demonstrated with nothing else in the frame.
//   Phase D  Concurrent `rotate_ct_host`. The real-world shape of the defect,
//            and the same thing the failing Rust rotation tests do.
//
// Distinct (n, towers) fixtures are used throughout so threads also contend on
// different cache keys, keeping Phase B a meaningful control.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cstdio>
#include <thread>
#include <atomic>
#include <string>

namespace gpufhe {
void pt_to_eval_host(std::vector<uint64_t>&, const std::vector<int64_t>&, uint32_t, uint32_t,
                     const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void automorphism_eval_host(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t,
                            const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void rotate_ct_host(std::vector<uint64_t>&, std::vector<uint64_t>&, uint32_t,
                    const KeySwitchConstants&, uint32_t, uint32_t,
                    const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void rotate_ct_resident(std::vector<uint64_t>&, std::vector<uint64_t>&, uint32_t,
                        const KeySwitchConstants&, uint32_t, uint32_t,
                        const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void set_secret_hamming_weight(uint32_t);
}

static uint64_t mm(uint64_t a, uint64_t b, uint64_t q) {
    return (uint64_t)(((unsigned __int128)a * b) % q);
}
using cd = std::complex<double>;

struct Fixture {
    uint32_t n = 0, tw = 0, k = 5;
    std::vector<uint64_t> mod, root;
    gpufhe::KeySwitchConstants K;
    std::vector<uint64_t> c0, c1;          // fresh ciphertext (input)
    std::vector<uint64_t> ref0, ref1;      // host-path reference output (serial)
    std::vector<uint64_t> got0, got1;      // resident-path output (concurrent)
    std::vector<int64_t>  mz;              // encoded plaintext, for pt_to_eval_host
    std::vector<uint64_t> ptRef;           // pt_to_eval_host reference (serial)
    std::vector<uint64_t> ptGot;           // pt_to_eval_host output (concurrent)
    std::vector<uint64_t> h0, h1;          // rotate_ct_host output (concurrent)
};

// Counts differing limbs between two equal-length buffers. A short/empty
// buffer is reported as fully different rather than as "0 diffs", so a missing
// result can never read as a pass.
static size_t diffLimbs(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b, size_t len) {
    if (a.size() < len || b.size() < len) return len;
    size_t d = 0;
    for (size_t x = 0; x < len; ++x) if (a[x] != b[x]) ++d;
    return d;
}

// Releases every thread at the same instant, so overlapping use of the shared
// scratch buffer is as likely as it can be made.
template <typename Fn>
static void runConcurrently(size_t count, Fn&& body) {
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> th;
    th.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        th.emplace_back([&, i]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) { /* spin */ }
            body(i);
        });
    }
    while (ready.load(std::memory_order_acquire) < (int)count) { /* wait */ }
    go.store(true, std::memory_order_release);
    for (auto& t : th) t.join();
}

// Builds one fixture end to end. Every cryptographic parameter matches the
// project's existing conventions exactly (sizeP=2, numPart=(tw+1)/2, sigma=3.19,
// h=64, ns=1) -- nothing here is relaxed to make the test easier to construct.
static bool buildFixture(Fixture& F, uint32_t n, uint32_t tw, uint64_t seedBase) {
    const uint32_t sizeP = 2, numPart = (tw + 1) / 2;
    const uint64_t ns = 1;
    F.n = n; F.tw = tw;

    std::vector<uint64_t> modP;
    gpufhe::native_primes(F.mod, 1, 60, n, {});
    { std::vector<uint64_t> md;
      gpufhe::native_primes(md, tw - 1, 55, n, F.mod);
      for (auto m : md) F.mod.push_back(m); }
    gpufhe::native_primes(modP, sizeP, 60, n, F.mod);

    std::vector<uint64_t> mq(F.mod);
    for (auto p : modP) mq.push_back(p);
    F.root.resize(tw);
    for (uint32_t i = 0; i < tw; ++i) F.root[i] = gpufhe::native_root(n, F.mod[i]);
    std::vector<uint64_t> rq = F.root;
    for (auto p : modP) rq.push_back(gpufhe::native_root(n, p));

    const uint32_t sizeQlP = tw + sizeP;
    auto KPqp = gpufhe::keygen_host(n, mq, rq, ns, 3.19, 101);
    auto KPq  = gpufhe::keygen_host(n, F.mod, F.root, ns, 3.19, 101);

    std::vector<uint64_t> PM(sizeQlP);
    for (uint32_t t = 0; t < sizeQlP; ++t) {
        uint64_t q = mq[t], P = 1 % q;
        for (uint32_t j = 0; j < sizeP; ++j) P = mm(P, modP[j] % q, q);
        PM[t] = P;
    }

    std::vector<uint64_t> sA = KPqp.s;
    gpufhe::automorphism_eval_host(sA, sizeQlP, n, F.k, mq, rq);
    F.K.n = n;
    gpufhe::compute_keyswitch_constants(F.K, F.mod, modP, numPart);
    for (uint32_t i = 0; i < tw; ++i) { F.K.rootModList.push_back(F.mod[i]); F.K.rootValList.push_back(F.root[i]); }
    for (uint32_t j = 0; j < sizeP; ++j) { F.K.rootModList.push_back(modP[j]); F.K.rootValList.push_back(rq[tw + j]); }
    gpufhe::evalkeygen_host_sold(F.K, KPqp.s, sA, KPqp.pkA, KPqp.pkB, PM, mq, rq, ns, 3.19, (uint32_t)(900 + seedBase));

    const uint32_t S = n / 2;
    std::vector<cd> z(S);
    for (uint32_t i = 0; i < S; ++i) z[i] = { 0.3 * std::sin(0.01 * i + 0.1 * (double)seedBase), 0 };
    gpufhe::encode_host(F.mz, z, n, std::pow(2.0, 50));
    gpufhe::encrypt_host(F.c0, F.c1, F.mz, KPq.pkA, KPq.pkB, n, F.mod, F.root, ns, 3.19, (uint32_t)(303 + seedBase));
    return true;
}

int main(int argc, char** argv) {
    gpufhe::set_secret_hamming_weight(64);   // h=64, UNCHANGED

    // Distinct (n, tw) pairs -> distinct cache keys in all four global caches.
    // tw=5 is avoided deliberately: `size_q=5` with `num_part=4` is a separate,
    // pre-existing `std::bad_alloc` defect already recorded in the Rust suite's
    // own comments. This test uses numPart=(tw+1)/2, a different configuration,
    // but there is no reason to go near a known-bad neighbourhood when any tw
    // serves this test's purpose equally well.
    struct Spec { uint32_t n, tw; };
    const std::vector<Spec> specs = {
        {1024, 3}, {1024, 4}, {1024, 6},
        {2048, 3}, {2048, 4}, {2048, 6},
        {4096, 3}, {4096, 4}, {4096, 6},
    };

    std::printf("=== rotate_ct_resident concurrent global-cache reproducer ===\n");
    std::printf("fixtures: %zu distinct (n,towers) pairs, h=64, sigma=3.19, sizeP=2, numPart=(tw+1)/2\n",
                specs.size());

    std::vector<Fixture> F(specs.size());

    // ---------------- PHASE A: serial build + HOST reference ----------------
    // rotate_ct_host goes through keyswitch_core_resident and never touches
    // ks_context_create, so g_rtab / upc / autoPermMap stay COLD after this.
    std::printf("\n[phase A] building fixtures and host-path references (serial)...\n");
    for (size_t i = 0; i < specs.size(); ++i) {
        if (!buildFixture(F[i], specs[i].n, specs[i].tw, (uint64_t)i)) {
            std::printf("  [FAIL] fixture %zu (n=%u tw=%u) could not be built\n",
                        i, specs[i].n, specs[i].tw);
            return 1;
        }
        F[i].ref0 = F[i].c0; F[i].ref1 = F[i].c1;
        gpufhe::rotate_ct_host(F[i].ref0, F[i].ref1, F[i].k, F[i].K, F[i].tw, F[i].n, F[i].mod, F[i].root);
        gpufhe::pt_to_eval_host(F[i].ptRef, F[i].mz, F[i].tw, F[i].n, F[i].mod, F[i].root);
        std::printf("  fixture %zu: n=%-5u tw=%u  serial references ready\n", i, specs[i].n, specs[i].tw);
    }

    const size_t NF = F.size();
    int phaseFailures = 0;

    // Reports one phase and returns the number of mismatching fixtures.
    auto report = [&](const char* phase, const char* what,
                      std::vector<size_t>& diffs, size_t limbsPerFixture) {
        size_t bad = 0, total = 0;
        for (size_t i = 0; i < NF; ++i) {
            if (diffs[i]) { ++bad; total += diffs[i]; }
            std::printf("  fixture %zu (n=%-5u tw=%u): %-9s (%zu differing limbs of %zu)\n",
                        i, F[i].n, F[i].tw, diffs[i] ? "MISMATCH" : "bit-exact",
                        diffs[i], limbsPerFixture ? limbsPerFixture : (size_t)F[i].tw * F[i].n);
        }
        if (bad) {
            std::printf("  => [%s FAIL] %zu/%zu fixtures returned WRONG data from %s (%zu limbs total)\n",
                        phase, bad, NF, what, total);
            ++phaseFailures;
        } else {
            std::printf("  => [%s PASS] all %zu fixtures bit-exact from %s\n", phase, NF, what);
        }
        return bad;
    };

    // ------- PHASE B (CONTROL): concurrent rotate_ct_resident --------------
    // Drives the unsynchronized global std::map caches on a cold process. This
    // is EXPECTED TO PASS. It is kept permanently as the control that falsified
    // the original map-race hypothesis -- if it ever starts failing, that
    // hypothesis must be re-opened rather than explained away.
    std::printf("\n[phase B / CONTROL] %zu threads in rotate_ct_resident (GPU-resident path,\n"
                "                    per-call scratch) -- exercises the global map caches\n", NF);
    runConcurrently(NF, [&](size_t i) {
        Fixture& f = F[i];
        f.got0 = f.c0; f.got1 = f.c1;
        gpufhe::rotate_ct_resident(f.got0, f.got1, f.k, f.K, f.tw, f.n, f.mod, f.root);
    });
    {
        std::vector<size_t> d(NF);
        for (size_t i = 0; i < NF; ++i) {
            const size_t len = (size_t)F[i].tw * F[i].n;
            d[i] = diffLimbs(F[i].got0, F[i].ref0, len) + diffLimbs(F[i].got1, F[i].ref1, len);
        }
        report("B", "rotate_ct_resident", d, 0);
    }

    // ------- PHASE C: concurrent pt_to_eval_host ---------------------------
    // Tightest isolation of xf_multi + the shared scratch() buffer: a pure
    // function of (plaintext, mod, root) with no keys and no ciphertext in the
    // frame at all. If this mismatches, the shared device scratch buffer is
    // demonstrated with nothing else that could explain it.
    std::printf("\n[phase C] %zu threads in pt_to_eval_host (host path -> xf_multi -> shared scratch())\n", NF);
    runConcurrently(NF, [&](size_t i) {
        Fixture& f = F[i];
        gpufhe::pt_to_eval_host(f.ptGot, f.mz, f.tw, f.n, f.mod, f.root);
    });
    {
        std::vector<size_t> d(NF);
        for (size_t i = 0; i < NF; ++i)
            d[i] = diffLimbs(F[i].ptGot, F[i].ptRef, (size_t)F[i].tw * F[i].n);
        report("C", "pt_to_eval_host", d, 0);
    }

    // ------- PHASE D: concurrent rotate_ct_host ----------------------------
    // The real-world shape: the same host-orchestrated rotation the failing
    // Rust rotation tests compute their references with.
    std::printf("\n[phase D] %zu threads in rotate_ct_host (automorphism_eval_host -> xf_multi -> shared scratch())\n", NF);
    runConcurrently(NF, [&](size_t i) {
        Fixture& f = F[i];
        f.h0 = f.c0; f.h1 = f.c1;
        gpufhe::rotate_ct_host(f.h0, f.h1, f.k, f.K, f.tw, f.n, f.mod, f.root);
    });
    {
        std::vector<size_t> d(NF);
        for (size_t i = 0; i < NF; ++i) {
            const size_t len = (size_t)F[i].tw * F[i].n;
            d[i] = diffLimbs(F[i].h0, F[i].ref0, len) + diffLimbs(F[i].h1, F[i].ref1, len);
        }
        report("D", "rotate_ct_host", d, 0);
    }

    std::printf("\n=== RESULT ===\n");
    if (phaseFailures == 0) {
        std::printf("[PASS] every concurrent phase is bit-exact against its serial reference\n");
        std::printf("NOTE: concurrency defects are probabilistic by nature. A pass is evidence,\n"
                    "      not proof -- run this repeatedly (the recorded loop does).\n");
        return 0;
    }
    std::printf("[FAIL] %d of 3 concurrent phases produced silently WRONG results\n", phaseFailures);
    std::printf("Phases C/D failing while control phase B passes localises the defect to the\n"
                "shared process-global device scratch buffer in src/rotate.cpp (scratch() at :48,\n"
                "plus the scratchPtrArr/scratchU64Arr metadata arrays), reached via xf_multi from\n"
                "pt_to_eval_host and automorphism_eval_host. The failure mode is silently\n"
                "incorrect ciphertext data, not a crash.\n");
    return 1;
}
