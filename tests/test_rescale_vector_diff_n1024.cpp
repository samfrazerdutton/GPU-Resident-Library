// GOD PROMPT "CKKS CORRECTNESS BREAKTHROUGH" round, Priority 1 Step 3
// (continued). test_rescale_scalar_diff_n1024 (this round, PASSED 10/10)
// used CONSTANT polynomials -- every one of the n coefficients equal. That
// is a degenerate case: NTT(constant) = constant, so coefficient-domain and
// eval-domain data coincide trivially, and the test cannot distinguish
// "correct per-position RNS rescale" from "a bug that only manifests when
// coefficients genuinely differ across positions" (e.g. an NTT/INTT
// position-mixing or buffer-aliasing defect that a uniform value can't
// reveal). This test closes that gap with INDEPENDENT per-coefficient
// random residues.
//
// Reference, computed directly in RNS arithmetic (no big-integer CRT
// reconstruction needed -- the single-modulus rescale identity is already
// position-wise over residues):
//   v[k]        = center(c_qLast[k], qLast)            (signed lift, same
//                 convention as kernels/switch_modulus.cu: xL<=floor(qL/2)
//                 stays non-negative, else xL-qL)
//   y_i[k]      = (c_i[k] - v[k]) * (qLast^-1 mod q_i) mod q_i,  i=0..6
// c_i[k] (i=0..7) are independently random per (tower, position) --
// genuinely non-constant coefficient-domain data. Both the real ciphertext-
// like input (forward-NTT of c_i[]) and the expected output (forward-NTT of
// y_i[]) are produced by this test's OWN forward-NTT helper (mirroring, not
// importing, encdec.cpp's xform/Step 2's build_tables -- already proven
// bit-exact at n=1024 for all 8 towers in test_ntt_roundtrip_n1024). The
// comparison is then eval-domain vs eval-domain, exactly where
// rescale_resident_raw leaves its output, over ALL n positions.
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
uint64_t invmod(uint64_t a, uint64_t q) { return powmod(a % q, q - 2, q); }
uint32_t bitrev(uint32_t v, uint32_t b) { uint32_t r = 0; for (uint32_t k = 0; k < b; ++k) { r = (r << 1) | (v & 1); v >>= 1; } return r; }
uint64_t* upload(const std::vector<uint64_t>& h) {
    uint64_t* d; cudaMalloc(&d, h.size() * 8); cudaMemcpy(d, h.data(), h.size() * 8, cudaMemcpyHostToDevice); return d;
}
// Forward NTT only (coefficient -> eval), self-built tables (same convention
// proven exact in test_ntt_roundtrip_n1024: standard order in, bit-reversed
// out), used both to build the real eval-domain input AND the expected
// eval-domain reference output.
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
} // anon

int main() {
    const uint32_t n = 1024, sizeQ = 8, sizeP = 2; const uint64_t ns = 1;
    uint32_t logn = 0; while ((1u << logn) < n) ++logn;
    std::vector<uint64_t> mod;
    gpufhe::native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids, sizeQ - 1, 55, n, mod); for (auto m : mids) mod.push_back(m); }
    const std::array<uint64_t,8> expectMod = {1152921504606830593ULL,36028797018820609ULL,36028797018802177ULL,
        36028797018789889ULL,36028797018746881ULL,36028797018667009ULL,36028797018652673ULL,36028797018646529ULL};
    for (uint32_t i = 0; i < sizeQ; ++i) {
        if (mod[i] != expectMod[i]) { std::cerr << "[ABORT] modulus chain drifted from reference\n"; return 2; }
    }
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

    const int NUM_TRIALS = 5;
    std::mt19937_64 rng(90210);
    int total_mismatches = 0;
    uint64_t qLast = mod[sizeQ - 1];

    size_t T = (size_t)sizeQ * n;
    uint64_t *d0, *scr, *drp;
    cudaMalloc(&d0, T * 8); cudaMalloc(&scr, (size_t)n * 8); cudaMalloc(&drp, (size_t)n * 8);

    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
        std::vector<std::vector<uint64_t>> c(sizeQ, std::vector<uint64_t>(n));
        for (uint32_t t = 0; t < sizeQ; ++t) {
            std::uniform_int_distribution<uint64_t> d(0, mod[t] - 1);
            for (uint32_t k = 0; k < n; ++k) c[t][k] = d(rng);
        }
        // Reference, pure RNS, position-wise.
        std::vector<std::vector<uint64_t>> yref(sizeQ - 1, std::vector<uint64_t>(n));
        for (uint32_t k = 0; k < n; ++k) {
            uint64_t xL = c[sizeQ - 1][k];
            uint64_t half = qLast >> 1;
            bool neg = xL > half;
            uint64_t vabs = neg ? (qLast - xL) : xL;   // |v|, v = xL or xL-qLast
            for (uint32_t i = 0; i < sizeQ - 1; ++i) {
                uint64_t qi = mod[i];
                uint64_t vmodqi = neg ? ((qi - (vabs % qi)) % qi) : (vabs % qi);
                uint64_t diff = (c[i][k] + qi - vmodqi) % qi;
                yref[i][k] = (uint64_t)(((unsigned __int128)diff * s1[i]) % qi);
            }
        }
        // Build real eval-domain input (forward NTT of c[]), upload.
        std::vector<uint64_t> h(T);
        for (uint32_t t = 0; t < sizeQ; ++t) {
            auto a = forward_ntt(c[t], n, logn, mod[t], root[t]);
            for (uint32_t k = 0; k < n; ++k) h[(size_t)t * n + k] = a[k];
        }
        cudaMemcpy(d0, h.data(), T * 8, cudaMemcpyHostToDevice);

        gpufhe::rescale_resident_raw(d0, sizeQ, C, s1, s2, scr, drp, 0);
        cudaDeviceSynchronize();

        std::vector<uint64_t> out(T);
        cudaMemcpy(out.data(), d0, T * 8, cudaMemcpyDeviceToHost);

        int mismatches = 0; uint32_t first_bad_tower = UINT32_MAX, first_bad_idx = UINT32_MAX; uint64_t got = 0, want = 0;
        for (uint32_t i = 0; i < sizeQ - 1; ++i) {
            auto expected_eval = forward_ntt(yref[i], n, logn, mod[i], root[i]);
            for (uint32_t k = 0; k < n; ++k) {
                uint64_t v = out[(size_t)i * n + k];
                if (v != expected_eval[k]) {
                    ++mismatches;
                    if (first_bad_tower == UINT32_MAX) { first_bad_tower = i; first_bad_idx = k; got = v; want = expected_eval[k]; }
                }
            }
        }
        total_mismatches += mismatches;
        std::cout << (mismatches == 0 ? "[PASS] " : "[FAIL] ") << "trial=" << trial
                  << " mismatches=" << mismatches << "/" << ((sizeQ - 1) * n);
        if (mismatches) std::cout << " first_bad_tower=" << first_bad_tower << " idx=" << first_bad_idx
                                   << " got=" << got << " want=" << want;
        std::cout << "\n";
    }
    std::cout << "---\n" << (total_mismatches == 0 ? "[PASS]" : "[FAIL]")
              << " test_rescale_vector_diff_n1024 (" << NUM_TRIALS << " trials, independent random per-coefficient residues)\n";

    cudaFree(d0); cudaFree(scr); cudaFree(drp);
    gpufhe::ks_context_destroy(C);
    return total_mismatches == 0 ? 0 : 1;
}
