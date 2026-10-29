// GOD PROMPT "CKKS CORRECTNESS BREAKTHROUGH" round, Priority 1 Step 3
// (continued, targeted hypothesis). test_rescale_vector_diff_n1024 (this
// round, PASSED 5/5 trials, 0/35840 mismatches) proved rescale_resident_raw
// correct for a SINGLE call on independent random per-coefficient data.
// But the ORIGINAL failing reproduction, test_rescale_n1024.cpp, calls
// rescale_resident_raw TWICE in a row -- once for ciphertext component c0,
// once for c1 -- REUSING THE SAME scr/drp scratch buffers on the SAME
// (default, null) stream:
//   rescale_resident_raw(d0, sizeQ, C, s1, s2, scr, drp, 0);
//   rescale_resident_raw(d1, sizeQ, C, s1, s2, scr, drp, 0);
// Nothing in test_rescale_vector_diff_n1024 or test_rescale_scalar_diff_
// n1024 exercised that back-to-back scratch-buffer-reuse pattern -- each
// called rescale_resident_raw exactly once. This test isolates exactly
// that: two independent random ciphertext-component-like inputs, rescaled
// back-to-back sharing scr/drp, each checked against its OWN independent
// RNS reference (same reference method as test_rescale_vector_diff_n1024).
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include "ntt.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <array>

namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void native_rescale_consts(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t);
}

namespace {
uint64_t powmod(uint64_t b, uint64_t e, uint64_t q) {
    unsigned __int128 r = 1, bb = b % q;
    while (e) { if (e & 1) r = (r * bb) % q; bb = (bb * bb) % q; e >>= 1; }
    return (uint64_t)r;
}
uint32_t bitrev(uint32_t v, uint32_t b) { uint32_t r = 0; for (uint32_t k = 0; k < b; ++k) { r = (r << 1) | (v & 1); v >>= 1; } return r; }
uint64_t* upload(const std::vector<uint64_t>& h) {
    uint64_t* d; cudaMalloc(&d, h.size() * 8); cudaMemcpy(d, h.data(), h.size() * 8, cudaMemcpyHostToDevice); return d;
}
std::vector<uint64_t> forward_ntt(const std::vector<uint64_t>& coeff, uint32_t n, uint32_t logn, uint64_t q, uint64_t root) {
    std::vector<uint64_t> pw(n); pw[0] = 1;
    for (uint32_t i = 1; i < n; ++i) pw[i] = (uint64_t)(((unsigned __int128)pw[i-1] * root) % q);
    std::vector<uint64_t> tr(n), tp(n);
    for (uint32_t i = 0; i < n; ++i) { uint64_t r = pw[bitrev(i, logn)]; tr[i] = r; tp[i] = (uint64_t)(((__uint128_t)r << 64) / q); }
    uint64_t* dx = upload(coeff); uint64_t* dr = upload(tr); uint64_t* dp = upload(tp);
    LaunchNTT_CT(dx, dr, dp, n, q, 0);
    cudaDeviceSynchronize();
    std::vector<uint64_t> out(n);
    cudaMemcpy(out.data(), dx, (size_t)n * 8, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dr); cudaFree(dp);
    return out;
}
// Build genuinely-random coefficient data + its RNS rescale reference.
void make_case(std::mt19937_64& rng, const std::vector<uint64_t>& mod, const std::vector<uint64_t>& s1,
                uint32_t n, uint32_t sizeQ,
                std::vector<std::vector<uint64_t>>& c, std::vector<std::vector<uint64_t>>& yref) {
    uint64_t qLast = mod[sizeQ - 1];
    c.assign(sizeQ, std::vector<uint64_t>(n));
    for (uint32_t t = 0; t < sizeQ; ++t) {
        std::uniform_int_distribution<uint64_t> d(0, mod[t] - 1);
        for (uint32_t k = 0; k < n; ++k) c[t][k] = d(rng);
    }
    yref.assign(sizeQ - 1, std::vector<uint64_t>(n));
    for (uint32_t k = 0; k < n; ++k) {
        uint64_t xL = c[sizeQ - 1][k];
        uint64_t half = qLast >> 1;
        bool neg = xL > half;
        uint64_t vabs = neg ? (qLast - xL) : xL;
        for (uint32_t i = 0; i < sizeQ - 1; ++i) {
            uint64_t qi = mod[i];
            uint64_t vmodqi = neg ? ((qi - (vabs % qi)) % qi) : (vabs % qi);
            uint64_t diff = (c[i][k] + qi - vmodqi) % qi;
            yref[i][k] = (uint64_t)(((unsigned __int128)diff * s1[i]) % qi);
        }
    }
}
int check(const std::vector<uint64_t>& out, const std::vector<std::vector<uint64_t>>& yref,
          const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root,
          uint32_t n, uint32_t logn, uint32_t sizeQ, const char* label) {
    int mismatches = 0; uint32_t bt = UINT32_MAX, bi = UINT32_MAX; uint64_t got = 0, want = 0;
    for (uint32_t i = 0; i < sizeQ - 1; ++i) {
        auto expected_eval = forward_ntt(yref[i], n, logn, mod[i], root[i]);
        for (uint32_t k = 0; k < n; ++k) {
            uint64_t v = out[(size_t)i * n + k];
            if (v != expected_eval[k]) { ++mismatches; if (bt == UINT32_MAX) { bt = i; bi = k; got = v; want = expected_eval[k]; } }
        }
    }
    std::cout << (mismatches == 0 ? "[PASS] " : "[FAIL] ") << label << " mismatches=" << mismatches << "/" << ((sizeQ - 1) * n);
    if (mismatches) std::cout << " first_bad_tower=" << bt << " idx=" << bi << " got=" << got << " want=" << want;
    std::cout << "\n";
    return mismatches;
}
} // anon

int main() {
    const uint32_t n = 1024, sizeQ = 8, sizeP = 2; const uint64_t ns = 1;
    uint32_t logn = 0; while ((1u << logn) < n) ++logn;
    std::vector<uint64_t> mod;
    gpufhe::native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids, sizeQ - 1, 55, n, mod); for (auto m : mids) mod.push_back(m); }
    std::vector<uint64_t> modP; gpufhe::native_primes(modP, sizeP, 60, n, mod);
    std::vector<uint64_t> root(sizeQ); for (uint32_t i = 0; i < sizeQ; ++i) root[i] = gpufhe::native_root(n, mod[i]);
    std::vector<uint64_t> rootP(sizeP); for (uint32_t j = 0; j < sizeP; ++j) rootP[j] = gpufhe::native_root(n, modP[j]);

    std::vector<uint64_t> modQP(mod); for (auto p : modP) modQP.push_back(p);
    std::vector<uint64_t> rootQP(root); for (auto rp : rootP) rootQP.push_back(rp);
    auto KPqp = gpufhe::keygen_host(n, modQP, rootQP, ns, 3.19, 101);
    std::vector<uint64_t> PModq_QP(sizeQ + sizeP);
    for (uint32_t t = 0; t < sizeQ + sizeP; ++t) { uint64_t q = modQP[t], P = 1 % q; for (uint32_t j = 0; j < sizeP; ++j) P = (uint64_t)(((unsigned __int128)P * (modQP[sizeQ + j] % q)) % q); PModq_QP[t] = P; }
    gpufhe::KeySwitchConstants K; K.n = n;
    gpufhe::compute_keyswitch_constants(K, mod, modP, 2);
    for (uint32_t i = 0; i < sizeQ; ++i) { K.rootModList.push_back(mod[i]); K.rootValList.push_back(root[i]); }
    for (uint32_t j = 0; j < sizeP; ++j) { K.rootModList.push_back(modP[j]); K.rootValList.push_back(rootP[j]); }
    gpufhe::evalkeygen_host(K, KPqp.s, KPqp.pkA, KPqp.pkB, PModq_QP, modQP, rootQP, ns, 3.19, 202);
    auto C = gpufhe::ks_context_create(K);
    std::vector<uint64_t> s1, s2; gpufhe::native_rescale_consts(s1, s2, mod, sizeQ - 1);

    std::mt19937_64 rng(42);
    size_t T = (size_t)sizeQ * n;
    uint64_t *d0, *d1, *scr, *drp;
    cudaMalloc(&d0, T * 8); cudaMalloc(&d1, T * 8); cudaMalloc(&scr, (size_t)n * 8); cudaMalloc(&drp, (size_t)n * 8);

    int total_mm = 0;
    for (int trial = 0; trial < 3; ++trial) {
        std::vector<std::vector<uint64_t>> c0, yref0, c1, yref1;
        make_case(rng, mod, s1, n, sizeQ, c0, yref0);
        make_case(rng, mod, s1, n, sizeQ, c1, yref1);

        std::vector<uint64_t> h0(T), h1(T);
        for (uint32_t t = 0; t < sizeQ; ++t) {
            auto a0 = forward_ntt(c0[t], n, logn, mod[t], root[t]);
            auto a1 = forward_ntt(c1[t], n, logn, mod[t], root[t]);
            for (uint32_t k = 0; k < n; ++k) { h0[(size_t)t * n + k] = a0[k]; h1[(size_t)t * n + k] = a1[k]; }
        }
        cudaMemcpy(d0, h0.data(), T * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(d1, h1.data(), T * 8, cudaMemcpyHostToDevice);

        // Exactly test_rescale_n1024.cpp's call pattern: back-to-back,
        // sharing scr/drp, same (null) stream.
        gpufhe::rescale_resident_raw(d0, sizeQ, C, s1, s2, scr, drp, 0);
        gpufhe::rescale_resident_raw(d1, sizeQ, C, s1, s2, scr, drp, 0);
        cudaDeviceSynchronize();

        std::vector<uint64_t> out0(T), out1(T);
        cudaMemcpy(out0.data(), d0, T * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(out1.data(), d1, T * 8, cudaMemcpyDeviceToHost);

        char lab0[64], lab1[64];
        snprintf(lab0, sizeof(lab0), "trial=%d d0(first call) ", trial);
        snprintf(lab1, sizeof(lab1), "trial=%d d1(second call)", trial);
        total_mm += check(out0, yref0, mod, root, n, logn, sizeQ, lab0);
        total_mm += check(out1, yref1, mod, root, n, logn, sizeQ, lab1);
    }
    std::cout << "---\n" << (total_mm == 0 ? "[PASS]" : "[FAIL]")
              << " test_rescale_double_call_n1024 (back-to-back scr/drp reuse pattern)\n";

    cudaFree(d0); cudaFree(d1); cudaFree(scr); cudaFree(drp);
    gpufhe::ks_context_destroy(C);
    return total_mm == 0 ? 0 : 1;
}
