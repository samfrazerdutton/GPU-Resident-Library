// GOD PROMPT "CKKS RUST + CUDA EVOLUTION" round. Path A of the three-way
// comparison (existing C++ host + existing CUDA C++ kernel), structured
// to mirror rust/crates/ckks-cuda's NttContext EXACTLY: same n=1024, same
// tower-0 modulus, same deterministic input, and the SAME four-tier
// measurement split (cold-start table build, device-side context init,
// steady-state per-call roundtrip, end-to-end), so the two languages'
// numbers are comparing the SAME workload at each tier, not different
// amounts of setup work hidden inside a single number.
//
// An earlier version of this file reallocated and re-uploaded the table
// buffers on every call (same methodological bug the Rust side had, just
// a smaller version of it: no per-call primitive-root search, but still
// a per-call cudaMalloc+upload for 4 buffers that are input-independent
// and should be reused). Fixed by introducing NttContextCpp below, with
// the identical "everything reusable happens once in the constructor"
// structure as NttContext::new in Rust.
#include "ntt.h"
#include "intt.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace gpufhe { uint64_t native_root(uint32_t, uint64_t); }

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

// Mirrors rust/crates/ckks-cuda's NttContext field-for-field: everything
// reusable (device selection implicit via first CUDA call, tables, their
// device-side upload, and the scratch buffer) is set up exactly once in
// the constructor; roundtrip() does only the input-dependent work.
struct NttContextCpp {
    uint32_t n, logn; uint64_t q, n_inv, n_inv_precon;
    uint64_t *d_fr, *d_fp, *d_ir, *d_ip, *d_x;

    NttContextCpp(uint32_t n_, uint64_t q_) : n(n_), q(q_) {
        logn = 0; while ((1u << logn) < n) ++logn;
        uint64_t root = gpufhe::native_root(n, q);
        std::vector<uint64_t> fr, fp, ir, ip;
        build_tables(fr, fp, n, logn, q, root, false);
        build_tables(ir, ip, n, logn, q, root, true);
        n_inv = invmod(n, q);
        n_inv_precon = (uint64_t)(((__uint128_t)n_inv << 64) / q);
        size_t bytes = (size_t)n * 8;
        cudaMalloc(&d_fr, bytes); cudaMalloc(&d_fp, bytes);
        cudaMalloc(&d_ir, bytes); cudaMalloc(&d_ip, bytes);
        cudaMalloc(&d_x, bytes);
        cudaMemcpy(d_fr, fr.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_fp, fp.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ir, ir.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ip, ip.data(), bytes, cudaMemcpyHostToDevice);
    }
    ~NttContextCpp() {
        cudaFree(d_fr); cudaFree(d_fp); cudaFree(d_ir); cudaFree(d_ip); cudaFree(d_x);
    }

    // Steady-state per-call path: upload input into the already-allocated
    // scratch buffer, launch both kernels, synchronize, read back. No
    // allocation, no table rebuild -- matches NttContext::roundtrip.
    void roundtrip(const std::vector<uint64_t>& coeff, std::vector<uint64_t>& out) {
        size_t bytes = (size_t)n * 8;
        cudaMemcpy(d_x, coeff.data(), bytes, cudaMemcpyHostToDevice);
        LaunchNTT_CT(d_x, d_fr, d_fp, n, q, 0);
        LaunchINTT_GS(d_x, d_ir, d_ip, n, q, n_inv, n_inv_precon, 0);
        cudaDeviceSynchronize();
        out.resize(n);
        cudaMemcpy(out.data(), d_x, bytes, cudaMemcpyDeviceToHost);
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
    const uint64_t q = 1152921504606830593ULL; // same tower-0 modulus the Rust bench uses

    size_t reps = argc > 1 ? (size_t)std::atoi(argv[1]) : 50;
    size_t trials = argc > 2 ? (size_t)std::atoi(argv[2]) : 7;

    std::vector<uint64_t> input(n);
    for (uint32_t k = 0; k < n; ++k) input[k] = ((uint64_t)k * 1000003ULL + 7ULL) % q;

    // --- Tier 1: cold-start, host-only (native_root FFI-equivalent call +
    // table build), no GPU touched yet. Measured standalone, `trials` times. ---
    std::vector<double> cold_start_us;
    for (size_t t = 0; t < trials; ++t) {
        auto t0 = std::chrono::steady_clock::now();
        uint32_t logn = 0; while ((1u << logn) < n) ++logn;
        uint64_t root = gpufhe::native_root(n, q);
        std::vector<uint64_t> fr, fp, ir, ip;
        build_tables(fr, fp, n, logn, q, root, false);
        build_tables(ir, ip, n, logn, q, root, true);
        volatile uint64_t sink = fr[0] + fp[0] + ir[0] + ip[0]; (void)sink;
        auto t1 = std::chrono::steady_clock::now();
        cold_start_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    report("cold-start (host-only: native_root + table build)", cold_start_us);

    // --- Tier 2: context init (device malloc + table upload), `trials` times. ---
    std::vector<double> init_us;
    for (size_t t = 0; t < trials; ++t) {
        auto t0 = std::chrono::steady_clock::now();
        NttContextCpp ctx(n, q);
        auto t1 = std::chrono::steady_clock::now();
        init_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    report("context init (NttContextCpp constructor, incl. cold-start)", init_us);

    // --- Tier 3: steady-state, ONE context reused across all reps/trials. ---
    // `out` is allocated ONCE, outside the loop, and reused every call --
    // an earlier version of this benchmark declared a FRESH, empty vector
    // inside the per-call loop, so `roundtrip`'s internal `out.resize(n)`
    // paid a heap allocation on every single call while the Rust
    // benchmark's equivalent `out` buffer was already preallocated once.
    // That was a real, found host-allocation asymmetry (exactly the class
    // of hidden cost Section 2.3 of the governing directive calls out),
    // not a genuine C++-vs-Rust difference -- fixed here to match.
    NttContextCpp ctx(n, q);
    std::vector<uint64_t> out(n);
    std::vector<double> per_trial_median;
    for (size_t t = 0; t < trials; ++t) {
        std::vector<double> per_call;
        for (size_t r = 0; r < reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            ctx.roundtrip(input, out);
            auto t1 = std::chrono::steady_clock::now();
            if (out != input) { std::cerr << "correctness check failed mid-benchmark\n"; return 1; }
            per_call.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        per_trial_median.push_back(median(per_call));
    }
    report("steady-state roundtrip (reused context)", per_trial_median);

    // --- Tier 4: end-to-end, fresh context + first roundtrip, `trials` times. ---
    std::vector<double> e2e_us;
    for (size_t t = 0; t < trials; ++t) {
        auto t0 = std::chrono::steady_clock::now();
        NttContextCpp fresh(n, q);
        std::vector<uint64_t> out;
        fresh.roundtrip(input, out);
        auto t1 = std::chrono::steady_clock::now();
        if (out != input) { std::cerr << "correctness check failed mid-benchmark\n"; return 1; }
        e2e_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    report("end-to-end (fresh context + first roundtrip)", e2e_us);

    return 0;
}
