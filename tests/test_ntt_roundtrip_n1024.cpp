// GOD PROMPT "CKKS CORRECTNESS BREAKTHROUGH" round, Priority 1 Step 2.
//
// test_rescale_n1024 (committed earlier) proved rescale_resident_raw +
// native_rescale_consts together produce a wrong result at n=1024 with
// test_polyeval's own modulus chain (1x60-bit + 7x55-bit). That pipeline
// calls, per survivor tower: LaunchINTT_GS (once, on the dropped tower),
// LaunchSwitchModulus, LaunchNTT_CT, LaunchRescaleFuse. Before trusting any
// hypothesis about SwitchModulus/RescaleFuse/native_rescale_consts, this
// test isolates the NTT/INTT transform kernels themselves: do
// LaunchNTT_CT/LaunchINTT_GS round-trip EXACTLY at n=1024, for every
// modulus actually used in that chain?
//
// This test owns its own root-table construction rather than calling any
// production helper, deliberately mirroring (not reusing) the exact
// convention documented in kernels/ntt_ct.cu / kernels/intt_ct.cu and
// implemented by src/encdec.cpp's anonymous-namespace `xform` helper:
//   - forward table: tr[i] = root^(bitreverse_logn(i)) mod q   (standard
//     order in, bit-reversed out -- the CT/DIF structure)
//   - inverse table: built the same way from root^{-1} mod q (bit-reversed
//     in, standard order out -- the GS structure), plus a separate n^{-1}
//     scale pass fused by LaunchINTT_GS itself.
//   - Shoup precomputation: precon[i] = floor(root_pow[i] * 2^64 / q).
// Building this independently (not importing xform) means a bug shared by
// both the production root-table code and this test's own copy could still
// hide -- that risk is noted in the mathematical reference doc, and is
// exactly why Step 2 only rules in/out the NTT *kernels*, not the whole
// root-table-generation pathway, which native_rescale_consts and keygen_host
// exercise via a DIFFERENT function (native_root/native_primes, read from
// src/params.cpp, not reimplemented here).
#include "ntt.h"
#include "intt.h"
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <iostream>
#include <cstdint>

namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
}

namespace {
uint64_t powmod(uint64_t b, uint64_t e, uint64_t q) {
    unsigned __int128 r = 1, bb = b % q;
    while (e) { if (e & 1) r = (r * bb) % q; bb = (bb * bb) % q; e >>= 1; }
    return (uint64_t)r;
}
uint64_t invmod(uint64_t a, uint64_t q) { return powmod(a % q, q - 2, q); }
uint32_t bitrev(uint32_t v, uint32_t b) { uint32_t r = 0; for (uint32_t k = 0; k < b; ++k) { r = (r << 1) | (v & 1); v >>= 1; } return r; }

// Build (root_pow_bitreversed, shoup_precon) tables for either direction.
void build_tables(std::vector<uint64_t>& tr, std::vector<uint64_t>& tp,
                   uint32_t n, uint32_t logn, uint64_t q, uint64_t root, bool inv) {
    uint64_t base = inv ? invmod(root, q) : root;
    std::vector<uint64_t> pw(n); pw[0] = 1;
    for (uint32_t i = 1; i < n; ++i) pw[i] = (uint64_t)(((unsigned __int128)pw[i-1] * base) % q);
    tr.resize(n); tp.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint64_t r = pw[bitrev(i, logn)];
        tr[i] = r;
        tp[i] = (uint64_t)(((__uint128_t)r << 64) / q);
    }
}

uint64_t* upload(const std::vector<uint64_t>& h) {
    uint64_t* d; cudaMalloc(&d, h.size() * 8); cudaMemcpy(d, h.data(), h.size() * 8, cudaMemcpyHostToDevice); return d;
}

// Forward NTT then inverse NTT in place on device buffer; returns host result.
std::vector<uint64_t> roundtrip(const std::vector<uint64_t>& coeff, uint32_t n, uint32_t logn, uint64_t q, uint64_t root) {
    std::vector<uint64_t> frt, fpt, irt, ipt;
    build_tables(frt, fpt, n, logn, q, root, false);
    build_tables(irt, ipt, n, logn, q, root, true);
    uint64_t ni = invmod(n, q), nip = (uint64_t)(((__uint128_t)ni << 64) / q);

    uint64_t* dx = upload(coeff);
    uint64_t* dfr = upload(frt); uint64_t* dfp = upload(fpt);
    uint64_t* dir = upload(irt); uint64_t* dip = upload(ipt);

    LaunchNTT_CT(dx, dfr, dfp, n, q, 0);
    LaunchINTT_GS(dx, dir, dip, n, q, ni, nip, 0);
    cudaDeviceSynchronize();

    std::vector<uint64_t> out(n);
    cudaMemcpy(out.data(), dx, (size_t)n * 8, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dfr); cudaFree(dfp); cudaFree(dir); cudaFree(dip);
    return out;
}

struct Case { const char* name; std::vector<uint64_t> poly; };

std::vector<Case> build_cases(uint32_t n, uint64_t q, std::mt19937_64& rng) {
    std::vector<Case> cs;
    cs.push_back({"zero", std::vector<uint64_t>(n, 0)});
    { std::vector<uint64_t> p(n, 0); p[0] = 1; cs.push_back({"sparse_dc", p}); }
    { std::vector<uint64_t> p(n, 0); p[n/2] = 1; cs.push_back({"sparse_mid", p}); }
    { std::vector<uint64_t> p(n, 0); p[n-1] = q - 1; cs.push_back({"sparse_last_boundary", p}); }
    { std::vector<uint64_t> p(n, q - 1); cs.push_back({"dense_boundary", p}); }
    { std::vector<uint64_t> p(n); std::uniform_int_distribution<uint64_t> d(0, q - 1); for (auto& v : p) v = d(rng); cs.push_back({"dense_random", p}); }
    { std::vector<uint64_t> p(n); for (uint32_t i = 0; i < n; ++i) p[i] = (i % 2 == 0) ? 0 : (q - 1); cs.push_back({"alternating_boundary", p}); }
    return cs;
}
} // anon

int main() {
    const uint32_t n = 1024; uint32_t logn = 0; while ((1u << logn) < n) ++logn;
    // Exact modulus chain test_rescale_n1024 uses: 1x60-bit + 7x55-bit.
    std::vector<uint64_t> mod;
    gpufhe::native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids, 7, 55, n, mod); for (auto m : mids) mod.push_back(m); }

    std::mt19937_64 rng(777);
    int total = 0, failed = 0;
    for (size_t ti = 0; ti < mod.size(); ++ti) {
        uint64_t q = mod[ti];
        uint64_t root = gpufhe::native_root(n, q);
        auto cases = build_cases(n, q, rng);
        for (auto& c : cases) {
            auto out = roundtrip(c.poly, n, logn, q, root);
            bool ok = true; uint32_t first_bad = UINT32_MAX;
            for (uint32_t i = 0; i < n; ++i) if (out[i] != c.poly[i]) { ok = false; if (first_bad == UINT32_MAX) first_bad = i; }
            ++total; if (!ok) ++failed;
            std::cout << (ok ? "[PASS] " : "[FAIL] ") << "tower=" << ti << " q=" << q
                      << " case=" << c.name;
            if (!ok) std::cout << " first_bad_idx=" << first_bad
                                << " got=" << out[first_bad] << " want=" << c.poly[first_bad];
            std::cout << "\n";
        }
    }
    std::cout << "---\n" << (total - failed) << "/" << total << " NTT round-trip cases passed at n=1024\n";
    std::cout << (failed == 0 ? "[PASS]" : "[FAIL]") << " test_ntt_roundtrip_n1024\n";
    return failed == 0 ? 0 : 1;
}
