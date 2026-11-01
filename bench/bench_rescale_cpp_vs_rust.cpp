// GOD PROMPT "CKKS RUST + CUDA PHASE 2" round. Path A (existing CUDA C++
// kernels, minimal from-scratch host orchestration) of a fair comparison
// against rust/crates/ckks-cuda::RescaleContext (Path B). Same n=1024,
// same 8-tower modulus chain as this round's Rust test, same algorithm as
// src/keyswitch_resident.cpp's rescale_resident_raw -- but DELIBERATELY
// NOT using DeviceKSContext/KeySwitchConstants/ks_context_create: this
// benchmark's RescaleContextCpp builds its OWN root tables independently
// (same approach as bench_ntt_roundtrip_cpp_vs_rust.cpp's NttContextCpp
// last round), mirroring rust::RescaleContext field-for-field, so BOTH
// languages are measured doing the SAME minimal setup work, not "C++
// using the full production context machinery vs Rust using a narrower
// one" -- that would confound the language comparison with a setup-
// scope difference, exactly the class of mistake the governing directive
// warns about. See research/EXPERIMENT_LEDGER.md for why rescale_
// resident_raw's OWN dependencies (a DeviceKSContext) reduce, for this
// algorithm specifically, to nothing more than per-tower modulus values
// and cached NTT root tables -- neither tied to key-switching -- which is
// exactly what both RescaleContextCpp and rust::RescaleContext rebuild
// independently instead.
#include "ntt.h"
#include "intt.h"
#include "switch_modulus.h"
#include "rescale.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace gpufhe {
uint64_t native_root(uint32_t, uint64_t);
void native_rescale_consts(std::vector<uint64_t>&, std::vector<uint64_t>&,
                            const std::vector<uint64_t>&, uint32_t);
}

static uint64_t powmod(uint64_t b, uint64_t e, uint64_t q) {
    unsigned __int128 r = 1 % q, bb = b % q;
    while (e) { if (e & 1) r = (r * bb) % q; bb = (bb * bb) % q; e >>= 1; }
    return (uint64_t)r;
}
static uint64_t invmod(uint64_t a, uint64_t q) { return powmod(a % q, q - 2, q); }
static uint32_t bitrev(uint32_t v, uint32_t b) { uint32_t r = 0; for (uint32_t k = 0; k < b; ++k) { r = (r << 1) | (v & 1); v >>= 1; } return r; }
static void build_tables(std::vector<uint64_t>& tr, std::vector<uint64_t>& tp,
                          uint32_t n, uint32_t logn, uint64_t q, uint64_t root, bool inv) {
    uint64_t base = inv ? invmod(root, q) : root;
    std::vector<uint64_t> pw(n); pw[0] = 1 % q;
    for (uint32_t i = 1; i < n; ++i) pw[i] = (uint64_t)(((unsigned __int128)pw[i-1] * base) % q);
    tr.resize(n); tp.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint64_t r = pw[bitrev(i, logn)];
        tr[i] = r; tp[i] = (uint64_t)(((__uint128_t)r << 64) / q);
    }
}

// Test-only: forward-NTTs one tower's COEFFICIENT-domain residues into
// EVAL domain, exactly mirroring rust/crates/ckks-cuda::rescale::tests::
// forward_ntt. Needed because rescale_resident_raw's contract (see
// src/keyswitch_resident.cpp and this crate's rescale.rs module doc)
// assumes its input is already in EVAL domain -- a synthetic buffer of
// arbitrary residues is NOT a valid eval-domain ciphertext, so the
// correctness reference must be computed on (and compared against) the
// SAME domain the kernels actually operate on.
static std::vector<uint64_t> forward_ntt_host(const std::vector<uint64_t>& coeff,
                                               uint32_t n, uint32_t logn,
                                               uint64_t q, uint64_t root) {
    std::vector<uint64_t> fr, fp;
    build_tables(fr, fp, n, logn, q, root, false);
    size_t bytes = (size_t)n * 8;
    uint64_t *d_x, *d_fr, *d_fp;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_fr, bytes); cudaMalloc(&d_fp, bytes);
    cudaMemcpy(d_x, coeff.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fr, fr.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fp, fp.data(), bytes, cudaMemcpyHostToDevice);
    LaunchNTT_CT(d_x, d_fr, d_fp, n, q, 0);
    cudaDeviceSynchronize();
    std::vector<uint64_t> out(n);
    cudaMemcpy(out.data(), d_x, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_fr); cudaFree(d_fp);
    return out;
}

// Mirrors rust/crates/ckks-cuda::RescaleContext field-for-field.
struct RescaleContextCpp {
    uint32_t n, logn, towers;
    std::vector<uint64_t> mods, s1, s2;
    std::vector<uint64_t*> d_fr, d_fp;      // per-survivor-tower forward tables
    uint64_t *d_ir, *d_ip;                   // dropped tower's inverse table
    uint64_t n_inv_last, n_inv_precon_last;
    uint64_t *d_c, *d_scratch, *d_drop;

    RescaleContextCpp(uint32_t n_, const std::vector<uint64_t>& mods_) : n(n_), mods(mods_) {
        logn = 0; while ((1u << logn) < n) ++logn;
        towers = (uint32_t)mods.size();
        uint32_t last = towers - 1;
        uint64_t qLast = mods[last];

        gpufhe::native_rescale_consts(s1, s2, mods, last);

        size_t bytes = (size_t)n * 8;
        d_fr.resize(last); d_fp.resize(last);
        for (uint32_t t = 0; t < last; ++t) {
            uint64_t qi = mods[t];
            uint64_t root = gpufhe::native_root(n, qi);
            std::vector<uint64_t> fr, fp;
            build_tables(fr, fp, n, logn, qi, root, false);
            cudaMalloc(&d_fr[t], bytes); cudaMalloc(&d_fp[t], bytes);
            cudaMemcpy(d_fr[t], fr.data(), bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_fp[t], fp.data(), bytes, cudaMemcpyHostToDevice);
        }
        uint64_t rootLast = gpufhe::native_root(n, qLast);
        std::vector<uint64_t> ir, ip;
        build_tables(ir, ip, n, logn, qLast, rootLast, true);
        cudaMalloc(&d_ir, bytes); cudaMalloc(&d_ip, bytes);
        cudaMemcpy(d_ir, ir.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ip, ip.data(), bytes, cudaMemcpyHostToDevice);
        n_inv_last = invmod(n, qLast);
        n_inv_precon_last = (uint64_t)(((__uint128_t)n_inv_last << 64) / qLast);

        cudaMalloc(&d_c, bytes * towers);
        cudaMalloc(&d_scratch, bytes);
        cudaMalloc(&d_drop, bytes);
    }
    ~RescaleContextCpp() {
        for (auto p : d_fr) cudaFree(p);
        for (auto p : d_fp) cudaFree(p);
        cudaFree(d_ir); cudaFree(d_ip);
        cudaFree(d_c); cudaFree(d_scratch); cudaFree(d_drop);
    }

    // Steady-state per-call path, EXACTLY rescale_resident_raw's own
    // sequence (see src/keyswitch_resident.cpp), with locally-owned
    // tables instead of a DeviceKSContext.
    void rescale(const std::vector<uint64_t>& coeffs_in, std::vector<uint64_t>& out) {
        uint32_t last = towers - 1;
        size_t bytes = (size_t)n * 8;
        uint64_t qLast = mods[last];

        cudaMemcpy(d_c, coeffs_in.data(), bytes * towers, cudaMemcpyHostToDevice);

        cudaMemcpyAsync(d_drop, d_c + (size_t)last * n, bytes, cudaMemcpyDeviceToDevice, 0);
        LaunchINTT_GS(d_drop, d_ir, d_ip, n, qLast, n_inv_last, n_inv_precon_last, 0);
        for (uint32_t t = 0; t < last; ++t) {
            uint64_t qi = mods[t];
            cudaMemcpyAsync(d_scratch, d_drop, bytes, cudaMemcpyDeviceToDevice, 0);
            LaunchSwitchModulus(d_scratch, qLast, qi, n, 0);
            LaunchNTT_CT(d_scratch, d_fr[t], d_fp[t], n, qi, 0);
            LaunchRescaleFuse(d_c + (size_t)t * n, d_scratch, s1[t], s2[t], qi, n, 0);
        }
        cudaDeviceSynchronize();

        out.resize((size_t)last * n);
        cudaMemcpy(out.data(), d_c, bytes * last, cudaMemcpyDeviceToHost);
    }
};

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n/2] : 0.5 * (v[n/2-1] + v[n/2]);
}
static void report(const char* label, std::vector<double>& per_trial) {
    double mn = *std::min_element(per_trial.begin(), per_trial.end());
    double mx = *std::max_element(per_trial.begin(), per_trial.end());
    std::vector<double> copy = per_trial;
    double m = median(copy);
    std::cout << label << ": median=" << m << "us min=" << mn << "us max=" << mx << "us (n=" << per_trial.size() << " trials)\n";
}

int main(int argc, char** argv) {
    const uint32_t n = 1024;
    // Same 8-tower chain as rust/crates/ckks-cuda::rescale::tests::MODULI.
    const std::vector<uint64_t> mods = {
        1152921504606830593ULL, 36028797018820609ULL, 36028797018802177ULL,
        36028797018789889ULL, 36028797018746881ULL, 36028797018667009ULL,
        36028797018652673ULL, 36028797018646529ULL,
    };
    const uint32_t towers = (uint32_t)mods.size();

    size_t reps = argc > 1 ? (size_t)std::atoi(argv[1]) : 50;
    size_t trials = argc > 2 ? (size_t)std::atoi(argv[2]) : 7;

    std::vector<uint64_t> input(towers * n);
    for (uint32_t t = 0; t < towers; ++t)
        for (uint32_t k = 0; k < n; ++k)
            input[(size_t)t * n + k] = ((uint64_t)(t * 97 + k) * 1000003ULL + 7ULL) % mods[t];

    // --- Tier 1: cold-start, host-only (native_root x towers + native_rescale_consts + table builds). ---
    std::vector<double> cold_start_us;
    for (size_t trial = 0; trial < trials; ++trial) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<uint64_t> s1, s2;
        gpufhe::native_rescale_consts(s1, s2, mods, towers - 1);
        volatile uint64_t sink = s1[0] + s2[0];
        for (uint32_t t = 0; t < towers; ++t) {
            uint64_t root = gpufhe::native_root(n, mods[t]);
            uint32_t logn = 0; while ((1u << logn) < n) ++logn;
            std::vector<uint64_t> tr, tp;
            build_tables(tr, tp, n, logn, mods[t], root, t == towers - 1);
            sink += tr[0] + tp[0];
        }
        (void)sink;
        auto t1 = std::chrono::steady_clock::now();
        cold_start_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    report("cold-start (host-only: native_rescale_consts + native_root x8 + table builds)", cold_start_us);

    // --- Tier 2: context init. ---
    std::vector<double> init_us;
    for (size_t trial = 0; trial < trials; ++trial) {
        auto t0 = std::chrono::steady_clock::now();
        RescaleContextCpp ctx(n, mods);
        auto t1 = std::chrono::steady_clock::now();
        init_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    report("context init (RescaleContextCpp constructor, incl. cold-start)", init_us);

    // --- Correctness gate, BEFORE any timing is trusted: this file's
    // RescaleContextCpp is a SEPARATE implementation from the production
    // rescale_resident_raw (deliberately, to match rust::RescaleContext's
    // DeviceKSContext-free setup -- see this file's header comment), so
    // it needs its OWN independent check, not a borrowed assumption from
    // other tests. Same RNS-only reference as
    // rust/crates/ckks-cuda/src/rescale.rs's test module and the earlier
    // round's tests/test_rescale_vector_diff_n1024.cpp. ---
    {
        RescaleContextCpp checkCtx(n, mods);
        uint32_t last = towers - 1;
        uint32_t logn = 0; while ((1u << logn) < n) ++logn;

        // `input` above is COEFFICIENT-domain synthetic data (plain
        // residues, not a valid eval-domain ciphertext). rescale_
        // resident_raw's contract assumes EVAL domain input for every
        // tower (see rescale.rs's module doc / RescaleContext::rescale
        // doc comment: the dropped tower is INTT'd FROM eval domain, and
        // survivor towers are fused directly against freshly-forward-
        // NTT'd scratch, also in eval domain). Forward-NTT every tower's
        // coefficient residues into eval domain before calling rescale(),
        // and forward-NTT the RNS reference's output the same way before
        // comparing -- exactly mirroring rust/crates/ckks-cuda/src/
        // rescale.rs's test module, which this implementation's earlier
        // version failed to do (100% mismatch, caught before any timing
        // was trusted).
        std::vector<uint64_t> eval_input((size_t)towers * n);
        for (uint32_t t = 0; t < towers; ++t) {
            uint64_t qt = mods[t];
            uint64_t root = gpufhe::native_root(n, qt);
            std::vector<uint64_t> coeff(
                input.begin() + (size_t)t * n, input.begin() + (size_t)(t + 1) * n);
            std::vector<uint64_t> eval = forward_ntt_host(coeff, n, logn, qt, root);
            std::copy(eval.begin(), eval.end(), eval_input.begin() + (size_t)t * n);
        }

        std::vector<uint64_t> actual;
        checkCtx.rescale(eval_input, actual);

        uint64_t qLast = mods[last];
        uint64_t half = qLast >> 1;
        std::vector<uint64_t> yref((size_t)last * n);
        for (uint32_t k = 0; k < n; ++k) {
            uint64_t xLast = input[(size_t)last * n + k];
            bool neg = xLast > half;
            uint64_t vabs = neg ? (qLast - xLast) : xLast;
            for (uint32_t i = 0; i < last; ++i) {
                uint64_t qi = mods[i];
                uint64_t vmodqi = neg ? ((qi - (vabs % qi)) % qi) : (vabs % qi);
                uint64_t diff = (input[(size_t)i * n + k] + qi - vmodqi) % qi;
                yref[(size_t)i * n + k] = (uint64_t)(((unsigned __int128)diff * checkCtx.s1[i]) % qi);
            }
        }

        int mismatches = 0;
        for (uint32_t i = 0; i < last; ++i) {
            uint64_t qi = mods[i];
            uint64_t root = gpufhe::native_root(n, qi);
            std::vector<uint64_t> yref_i(yref.begin() + (size_t)i * n, yref.begin() + (size_t)(i + 1) * n);
            std::vector<uint64_t> expected_eval = forward_ntt_host(yref_i, n, logn, qi, root);
            for (uint32_t k = 0; k < n; ++k) {
                if (actual[(size_t)i * n + k] != expected_eval[k]) ++mismatches;
            }
        }
        if (mismatches != 0) {
            std::cerr << "CORRECTNESS CHECK FAILED: " << mismatches << " mismatches -- refusing to report timing\n";
            return 1;
        }
        std::cout << "correctness check: 0/" << ((size_t)last * n) << " mismatches vs independent RNS reference\n";

        // Tiers 3/4 below time the SAME kernels on arbitrary-but-valid-
        // range residues; whether that buffer happens to be a "real" NTT
        // output does not change the kernels' arithmetic cost, so `input`
        // (not `eval_input`) remains the timed workload, matching the
        // already-proven-fair NTT benchmark's own precedent.
    }

    // --- Tier 3: steady-state, ONE context reused across all reps/trials. ---
    RescaleContextCpp ctx(n, mods);
    std::vector<uint64_t> out;
    out.reserve((size_t)(towers - 1) * n);
    std::vector<double> per_trial_median;
    for (size_t trial = 0; trial < trials; ++trial) {
        std::vector<double> per_call;
        for (size_t r = 0; r < reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            ctx.rescale(input, out);
            auto t1 = std::chrono::steady_clock::now();
            per_call.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        per_trial_median.push_back(median(per_call));
    }
    report("steady-state rescale (reused context)", per_trial_median);

    // --- Tier 4: end-to-end, fresh context + first rescale, combined. ---
    std::vector<double> e2e_us;
    for (size_t trial = 0; trial < trials; ++trial) {
        auto t0 = std::chrono::steady_clock::now();
        RescaleContextCpp fresh(n, mods);
        std::vector<uint64_t> out2;
        fresh.rescale(input, out2);
        e2e_us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    report("end-to-end (fresh context + first rescale)", e2e_us);

    return 0;
}
