// GOD PROMPT "CKKS PHASE 25": controlled A/B benchmark for rotation hoisting
// (decompose-sharing), at a scale this machine can actually run.
//
// # The question
//
// Rotation hoisting is `HOIST_ROTATION_ENABLED = false` in tests/test_boot.cpp.
// Phase 24 recorded why: correctness is proven (byte-identical max slot error
// to the trusted path), but "the measured speedup is small and noisy once
// run-order bias is corrected for", so the performance acceptance criteria
// never cleanly passed. The A/B that would settle it
// (research/benchmarks/randomized_ab.sh over test_boot vs
// test_boot_hoist_enabled) needs ~8GB host RSS per run x 20 runs and has been
// blocked by host memory since Phase 23.
//
// This benchmark answers the mechanism-level question instead, and it runs in
// ~600MB:
//
//   Hoisting replaces N decomposes with ONE decompose plus N accumulates.
//   So its benefit MUST scale with N -- the number of rotations that share a
//   single decompose. At N=1 there is nothing to amortise and hoisting can
//   only lose (it adds bookkeeping). The interesting quantity is therefore not
//   "is hoisting faster" but "at what N does it start to pay, and by how much".
//
// Phase 24's "small and noisy" measurement was taken on the full bootstrap,
// where that ratio is diluted by everything else the pipeline does. Measuring
// the ratio directly, as a function of N, is both cheaper and more informative.
//
// # What is timed, and what is NOT -- read this before quoting any number
//
// Timed: exactly the work that DIFFERS between the two arms.
//   ARM A (trusted):  for each of N keys:  ks_accumulate_qp  (decompose+accumulate)
//                                        + ks_moddown x2
//   ARM B (hoisted):  ks_decompose_only ONCE, then for each of N keys:
//                                          ks_accumulate_from_digits
//                                        + ks_moddown x2
//
// NOT timed, and identical in both arms: the per-rotation automorphism
// (automorphism_eval_fast_device), the ciphertext upload/download, the final
// host-side combine, key generation, and context creation.
//
// Consequently the ratio reported below is an **UPPER BOUND on the achievable
// end-to-end rotation speedup, and is NOT a bootstrap speedup**. A real
// rotation also pays the automorphism and the transfers, which hoisting does
// not accelerate at all; a real bootstrap additionally spends most of its time
// in host-side diagonal processing (see research/BOOTSTRAP_GPU_RESIDENCY_AUDIT.md).
// Quoting this figure as a bootstrap number would be exactly the error the
// governing directive forbids.
//
// # Methodology
//
//   * Shared setup: ONE modulus chain, ONE key pair, ONE ciphertext, and all N
//     rotation contexts built up front -- the realistic model, matching a
//     bootstrap stage where KROT is already warm. None of it is timed.
//   * Both arms consume the SAME ciphertext and the SAME key material.
//   * Warm-up: 2 untimed iterations of each arm before any measurement, so
//     first-touch allocation and cache population never land in a sample.
//   * Run-order bias: the arm order ALTERNATES per trial (A,B then B,A). This
//     machine shows real thermal drift and GPU clock throttling under WSL2
//     over a long run, which systematically favours whichever arm runs first.
//   * Multiple trials, and the MEDIAN is reported with min/max so the spread is
//     visible. On this machine the run-to-run spread has been measured at ~30%,
//     wider than most effects being chased, so a single pair of numbers is
//     worthless.
//   * Correctness is verified independently of timing: arm B's outputs must be
//     BIT-EXACT against arm A's for every key, and that is asserted before any
//     timing result is printed. A faster-but-wrong arm is not a result.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keyswitch_hoist.h"
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

static uint64_t mm(uint64_t a, uint64_t b, uint64_t q) {
    return (uint64_t)(((unsigned __int128)a * b) % q);
}
// Modular inverse of k mod M by extended Euclid (k is odd, M a power of two).
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
static double rssMB() {
    struct rusage ru; getrusage(RUSAGE_SELF, &ru); return ru.ru_maxrss / 1024.0;
}
static double median(std::vector<double> v) {
    if (v.empty()) return -1.0;
    std::sort(v.begin(), v.end());
    const size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

int main(int argc, char** argv) {
    const uint32_t n      = (argc > 1) ? (uint32_t)std::stoul(argv[1]) : 8192;
    const uint32_t tw     = (argc > 2) ? (uint32_t)std::stoul(argv[2]) : 30;
    const uint32_t TRIALS = (argc > 3) ? (uint32_t)std::stoul(argv[3]) : 9;
    const uint32_t RMAX   = (argc > 4) ? (uint32_t)std::stoul(argv[4]) : 6;

    const uint32_t S = n / 2, sizeP = 2, numPart = (tw + 1) / 2;
    const uint64_t ns = 1;
    const uint32_t M = 2 * n, sizeQlP = tw + sizeP;

    gpufhe::set_secret_hamming_weight(64);   // h=64, UNCHANGED

    std::printf("=== rotation-hoisting A/B (decompose-sharing) ===\n");
    std::printf("params: n=%u sizeQ=%u sizeP=%u numPart=%u h=64 sigma=3.19 trials=%u Rmax=%u\n",
                n, tw, sizeP, numPart, TRIALS, RMAX);

    // ---------------------- shared setup (NOT timed) ----------------------
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

    // RMAX hoisted-convention rotation keys, built once. Rotation indices are
    // the power-of-five Galois elements a real BSGS stage uses, plus M-1
    // (conjugation), so the key set resembles an actual stage's rather than
    // being arbitrary.
    std::vector<uint32_t> ks;
    { uint64_t r = 1;
      for (uint32_t i = 0; i < RMAX && ks.size() < RMAX; ++i) { r = (r * 5) % M; ks.push_back((uint32_t)r); } }
    if (!ks.empty()) ks.back() = M - 1;

    std::printf("\n[setup] building %zu hoisted rotation keys (untimed)...\n", ks.size());
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

    std::vector<gpufhe::DeviceKSContext> C(ks.size());
    for (size_t i = 0; i < ks.size(); ++i) C[i] = gpufhe::ks_context_create(Kh[i]);
    auto W = gpufhe::ks_work_create(C[0]);
    auto D = gpufhe::hoist_digits_create(C[0]);

    const size_t T = (size_t)tw * n, B = T * 8;
    const size_t TQP = (size_t)sizeQlP * n, BQP = TQP * 8;
    uint64_t *d_c1, *d_out0, *d_out1;
    cudaMalloc(&d_c1, B); cudaMalloc(&d_out0, B); cudaMalloc(&d_out1, B);
    cudaMemcpy(d_c1, c1.data(), B, cudaMemcpyHostToDevice);

    size_t vramFree = 0, vramTotal = 0;
    cudaMemGetInfo(&vramFree, &vramTotal);
    std::printf("[setup] done. peak RSS so far %.1f MB; VRAM free %.0f/%.0f MiB\n",
                rssMB(), vramFree / 1048576.0, vramTotal / 1048576.0);

    // One arm's work over the first R keys. Returns elapsed seconds.
    // `collect` optionally captures each key's moddown output for comparison.
    auto armTrusted = [&](uint32_t R, std::vector<std::vector<uint64_t>>* collect) {
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < R; ++i) {
            gpufhe::ks_accumulate_qp(d_c1, C[i], W, 0);
            gpufhe::ks_moddown(C[i], W, W.d_res0, d_out0, 0);
            gpufhe::ks_moddown(C[i], W, W.d_res1, d_out1, 0);
            if (collect) {
                cudaStreamSynchronize(0);
                std::vector<uint64_t> a(T), b(T);
                cudaMemcpy(a.data(), d_out0, B, cudaMemcpyDeviceToHost);
                cudaMemcpy(b.data(), d_out1, B, cudaMemcpyDeviceToHost);
                collect->push_back(a); collect->push_back(b);
            }
        }
        cudaStreamSynchronize(0);
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    auto armHoisted = [&](uint32_t R, std::vector<std::vector<uint64_t>>* collect) {
        auto t0 = std::chrono::steady_clock::now();
        gpufhe::ks_decompose_only(d_c1, C[0], D, 0);     // *** exactly ONCE ***
        for (uint32_t i = 0; i < R; ++i) {
            gpufhe::ks_accumulate_from_digits(d_c1, D, C[i], W, 0);
            gpufhe::ks_moddown(C[i], W, W.d_res0, d_out0, 0);
            gpufhe::ks_moddown(C[i], W, W.d_res1, d_out1, 0);
            if (collect) {
                cudaStreamSynchronize(0);
                std::vector<uint64_t> a(T), b(T);
                cudaMemcpy(a.data(), d_out0, B, cudaMemcpyDeviceToHost);
                cudaMemcpy(b.data(), d_out1, B, cudaMemcpyDeviceToHost);
                collect->push_back(a); collect->push_back(b);
            }
        }
        cudaStreamSynchronize(0);
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    // ------------- correctness FIRST, before any timing is reported -------------
    std::printf("\n[correctness] hoisted arm must be BIT-EXACT against the trusted arm\n");
    {
        std::vector<std::vector<uint64_t>> ra, rb;
        armTrusted((uint32_t)ks.size(), &ra);
        armHoisted((uint32_t)ks.size(), &rb);
        if (ra.size() != rb.size() || ra.empty()) {
            std::printf("  [FAIL] collected %zu vs %zu buffers -- harness error\n", ra.size(), rb.size());
            return 1;
        }
        size_t bad = 0, limbs = 0;
        for (size_t b2 = 0; b2 < ra.size(); ++b2) {
            for (size_t x = 0; x < ra[b2].size() && x < rb[b2].size(); ++x) { ++limbs; if (ra[b2][x] != rb[b2][x]) ++bad; }
        }
        std::printf("  %zu / %zu limbs differ across %zu keys -> %s\n",
                    bad, limbs, ks.size(), bad == 0 ? "[PASS bit-exact]" : "[FAIL]");
        if (bad) {
            std::printf("  [FAIL] refusing to report timings for an arm that is not bit-exact\n");
            return 1;
        }
    }

    // --------------------------- warm-up (untimed) ---------------------------
    for (int w = 0; w < 2; ++w) { armTrusted((uint32_t)ks.size(), nullptr); armHoisted((uint32_t)ks.size(), nullptr); }

    // ------------------------------ measurement ------------------------------
    std::printf("\n[timing] median of %u trials, arm order ALTERNATES per trial.\n", TRIALS);
    std::printf("         Timed: decompose+accumulate+moddown only. NOT timed and identical in\n");
    std::printf("         both arms: automorphism, transfers, host combine. The ratio is an\n");
    std::printf("         UPPER BOUND on rotation speedup, and is NOT a bootstrap speedup.\n\n");
    std::printf("  %-4s %-26s %-26s %-10s\n", "R", "trusted med [min..max] s", "hoisted med [min..max] s", "speedup");

    std::vector<uint32_t> Rs;
    for (uint32_t r = 1; r <= (uint32_t)ks.size(); ++r)
        if (r == 1 || r == 2 || r == 3 || r == 4 || r == (uint32_t)ks.size()) Rs.push_back(r);

    std::string jsonl;
    for (uint32_t R : Rs) {
        std::vector<double> va, vb;
        for (uint32_t t = 0; t < TRIALS; ++t) {
            if (t % 2 == 0) { va.push_back(armTrusted(R, nullptr)); vb.push_back(armHoisted(R, nullptr)); }
            else            { vb.push_back(armHoisted(R, nullptr)); va.push_back(armTrusted(R, nullptr)); }
        }
        const double ma = median(va), mb = median(vb);
        const double lo_a = *std::min_element(va.begin(), va.end()), hi_a = *std::max_element(va.begin(), va.end());
        const double lo_b = *std::min_element(vb.begin(), vb.end()), hi_b = *std::max_element(vb.begin(), vb.end());
        const double sp = (mb > 0) ? ma / mb : -1.0;
        char bufA[64], bufB[64];
        std::snprintf(bufA, sizeof bufA, "%.5f [%.5f..%.5f]", ma, lo_a, hi_a);
        std::snprintf(bufB, sizeof bufB, "%.5f [%.5f..%.5f]", mb, lo_b, hi_b);
        std::printf("  %-4u %-26s %-26s %.4fx\n", R, bufA, bufB, sp);

        char line[640];
        std::snprintf(line, sizeof line,
            "{\"record\":\"hoist_ab\",\"n\":%u,\"size_q\":%u,\"size_p\":%u,\"num_part\":%u,"
            "\"rotations\":%u,\"trials\":%u,\"trusted_median_s\":%.9g,\"trusted_min_s\":%.9g,"
            "\"trusted_max_s\":%.9g,\"hoisted_median_s\":%.9g,\"hoisted_min_s\":%.9g,"
            "\"hoisted_max_s\":%.9g,\"speedup_upper_bound\":%.9g,\"bit_exact\":true,"
            "\"scope\":\"decompose+accumulate+moddown only; NOT an end-to-end or bootstrap speedup\"}\n",
            n, tw, sizeP, numPart, R, TRIALS, ma, lo_a, hi_a, mb, lo_b, hi_b, sp);
        jsonl += line;
    }

    cudaMemGetInfo(&vramFree, &vramTotal);
    std::printf("\n[resources] peak host RSS %.1f MB; VRAM free %.0f/%.0f MiB\n",
                rssMB(), vramFree / 1048576.0, vramTotal / 1048576.0);

    const char* envp = std::getenv("BOOT_DRIVER_JSONL");
    const std::string out = (envp && *envp) ? std::string(envp)
                                            : std::string("research/benchmarks/hoist_ab.jsonl");
    if (FILE* f = std::fopen(out.c_str(), "a")) {
        std::fputs(jsonl.c_str(), f);
        std::fclose(f);
        std::printf("[jsonl] appended %zu rows to %s\n", Rs.size(), out.c_str());
    } else {
        std::printf("[jsonl] WARNING: could not open %s for append\n", out.c_str());
    }

    gpufhe::hoist_digits_destroy(D);
    gpufhe::ks_work_destroy(W);
    for (auto& c : C) gpufhe::ks_context_destroy(c);
    cudaFree(d_c1); cudaFree(d_out0); cudaFree(d_out1);
    std::printf("\n[PASS] A/B complete: hoisted arm bit-exact, timings recorded\n");
    return 0;
}
