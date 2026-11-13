// GOD PROMPT "CKKS PHASE 40" Track B: dedicated boundary-value and
// correction-branch coverage test for the two Shoup-ified kernels added in
// Phase 39 (kernels/moddown.cu's moddown_combine_kernel, kernels/rescale.cu's
// rescale_fuse_kernel). Distinct from tests/test_moddown.cpp and
// tests/test_moddown_combine_isolated.cpp (which already exercise
// LaunchModDownCombine end to end against OpenFHE / a hand-derived exact
// identity with realistic random inputs): this test instead launches both
// kernels DIRECTLY with explicit, hand-picked operand values -- 0, 1, q-1,
// q/2, and a spread of pseudorandom values -- across several real NTT-
// friendly primes this project actually uses, and checks every result
// against an INDEPENDENT oracle that does NOT reuse shoup_mulmod: a plain
// 128-bit multiply-and-% (the exact definition of modular multiplication),
// computed on the host with __int128, not with the Shoup identity. It also
// explicitly classifies and reports how many trials hit each of
// shoup_mulmod's two possible outcomes (no correction needed vs. the single
// `r -= q` correction), so "every correction branch is covered" is a
// verified fact, not an assumption.
#include "moddown.h"
#include "rescale.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstdint>

namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
}

namespace {
uint64_t shoup_precon(uint64_t c, uint64_t q) { return (uint64_t)(((unsigned __int128)c << 64) / q); }
// Independent oracle: the literal definition of modular multiplication,
// computed via 128-bit multiply + remainder -- shares no code or formula
// with shoup_mulmod (no precomputed constant, no multiply-high trick).
uint64_t oracle_mulmod(uint64_t x, uint64_t c, uint64_t q) {
    return (uint64_t)(((unsigned __int128)x * (unsigned __int128)c) % q);
}
uint64_t oracle_submod(uint64_t a, uint64_t b, uint64_t q) {
    return (a >= b) ? (a - b) : (a - b + q);
}
uint64_t oracle_addmod(uint64_t a, uint64_t b, uint64_t q) {
    uint64_t s = a + b; // a,b < q < 2^63, no overflow
    return (s >= q) ? s - q : s;
}
// Classifies which branch the device shoup_mulmod takes, replicated exactly
// (same formula as kernels/moddown.cu / kernels/rescale.cu's __device__
// function) so branch coverage can be counted honestly, not guessed at.
bool shoup_needed_correction(uint64_t x, uint64_t c, uint64_t precon, uint64_t q) {
    unsigned __int128 hi128 = ((unsigned __int128)x * precon) >> 64;
    uint64_t hi = (uint64_t)hi128;
    uint64_t r = x * c - hi * q; // wraps mod 2^64, matches device uint64_t arithmetic
    return r >= q;
}

uint64_t* up1(uint64_t v) { uint64_t* d; cudaMalloc(&d, 8); cudaMemcpy(d, &v, 8, cudaMemcpyHostToDevice); return d; }
uint64_t down1(uint64_t* d) { uint64_t v; cudaMemcpy(&v, d, 8, cudaMemcpyDeviceToHost); return v; }
} // namespace

int main() {
    using namespace gpufhe;
    bool allPass = true;
    uint64_t correctionHit = 0, noCorrectionHit = 0, totalTrials = 0;

    std::vector<uint64_t> primes;
    for (uint32_t bits : {30u, 50u, 62u}) {
        std::vector<uint64_t> p;
        native_primes(p, 2, bits, 8192, primes);
        for (auto q : p) primes.push_back(q);
    }

    std::mt19937_64 rng(0xC0FFEE);

    for (uint64_t q : primes) {
        // Boundary operand set: 0, 1, q-1, q/2, q/2+1, plus 10 pseudorandom
        // values spanning the full [0,q) domain -- every pairing below is
        // tested, so every boundary combination (including both operands at
        // their extremes simultaneously) is exercised, not just each value
        // in isolation.
        std::vector<uint64_t> vals = {0, 1, q - 1, q / 2, q / 2 + 1};
        std::uniform_int_distribution<uint64_t> dist(0, q - 1);
        for (int i = 0; i < 10; ++i) vals.push_back(dist(rng));

        const uint32_t n = (uint32_t)(vals.size() * vals.size());
        std::vector<uint64_t> aBuf, bBuf; // every (a,b) pairing, one per coefficient slot
        aBuf.reserve(n); bBuf.reserve(n);
        for (uint64_t a : vals) for (uint64_t b : vals) { aBuf.push_back(a); bBuf.push_back(b); }

        uint64_t *dA, *dB, *dAns;
        cudaMalloc(&dA, n * 8); cudaMalloc(&dB, n * 8); cudaMalloc(&dAns, n * 8);
        cudaMemcpy(dA, aBuf.data(), n * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, bBuf.data(), n * 8, cudaMemcpyHostToDevice);

        // --- ModDown combine: ans = (a - b mod q) * s mod q, for several s ---
        for (uint64_t s : std::vector<uint64_t>{0, 1, q - 1, q / 2}) {
            uint64_t sPrecon = shoup_precon(s, q);
            LaunchModDownCombine(dAns, dA, dB, s, sPrecon, q, n, 0);
            cudaDeviceSynchronize();
            std::vector<uint64_t> ans(n);
            cudaMemcpy(ans.data(), dAns, n * 8, cudaMemcpyDeviceToHost);
            for (uint32_t k = 0; k < n; ++k) {
                uint64_t diff = oracle_submod(aBuf[k], bBuf[k], q);
                uint64_t expect = oracle_mulmod(diff, s, q);
                if (ans[k] != expect) {
                    allPass = false;
                    std::cout << "[FAIL] ModDown q=" << q << " a=" << aBuf[k] << " b=" << bBuf[k]
                              << " s=" << s << " got=" << ans[k] << " expect=" << expect << "\n";
                }
                if (shoup_needed_correction(diff, s, sPrecon, q)) ++correctionHit; else ++noCorrectionHit;
                ++totalTrials;
            }
        }

        // --- RescaleFuse: a[idx] = (a*s1 + b*s2) mod q, in place, for several (s1,s2) ---
        for (auto sp : std::vector<std::pair<uint64_t,uint64_t>>{{0,0},{1,1},{q-1,q-1},{1,q-1},{q/2,q/2+1}}) {
            uint64_t s1 = sp.first, s2 = sp.second;
            uint64_t s1Precon = shoup_precon(s1, q), s2Precon = shoup_precon(s2, q);
            cudaMemcpy(dA, aBuf.data(), n * 8, cudaMemcpyHostToDevice); // a is overwritten in place
            LaunchRescaleFuse(dA, dB, s1, s1Precon, s2, s2Precon, q, n, 0);
            cudaDeviceSynchronize();
            std::vector<uint64_t> ans(n);
            cudaMemcpy(ans.data(), dA, n * 8, cudaMemcpyDeviceToHost);
            for (uint32_t k = 0; k < n; ++k) {
                uint64_t t1 = oracle_mulmod(aBuf[k], s1, q);
                uint64_t t2 = oracle_mulmod(bBuf[k], s2, q);
                uint64_t expect = oracle_addmod(t1, t2, q);
                if (ans[k] != expect) {
                    allPass = false;
                    std::cout << "[FAIL] RescaleFuse q=" << q << " a=" << aBuf[k] << " b=" << bBuf[k]
                              << " s1=" << s1 << " s2=" << s2 << " got=" << ans[k] << " expect=" << expect << "\n";
                }
                if (shoup_needed_correction(aBuf[k], s1, s1Precon, q)) ++correctionHit; else ++noCorrectionHit;
                if (shoup_needed_correction(bBuf[k], s2, s2Precon, q)) ++correctionHit; else ++noCorrectionHit;
                ++totalTrials;
            }
        }

        cudaFree(dA); cudaFree(dB); cudaFree(dAns);
    }

    std::cout << "Total coefficient-level trials: " << totalTrials
              << ", shoup_mulmod correction branch taken: " << correctionHit
              << " times, not taken: " << noCorrectionHit << " times.\n";
    if (correctionHit == 0 || noCorrectionHit == 0) {
        std::cout << "[FAIL] one of shoup_mulmod's two branches was never exercised -- "
                     "coverage claim would be false.\n";
        allPass = false;
    }
    std::cout << (allPass ? "[OVERALL PASS]\n" : "[OVERALL FAIL]\n");
    return allPass ? 0 : 1;
}
