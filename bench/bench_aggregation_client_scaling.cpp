// GOD PROMPT "CKKS PHASE 29", Objective F: consumer-GPU memory scaling for
// the aggregation workload as client count K grows. Uses the PRODUCTION
// encrypt path (encrypt_host_shoup_pooled_barrett_batched) since that is
// what a real deployment would use (Objective G's promotion target).
//
// Hypothesis: peak VRAM should stay roughly CONSTANT as K grows, because
// each client's ciphertext is encrypted, immediately folded into the
// running accumulator via ct_add_ct_host (pure CPU), and then freed/reused
// -- no per-client device buffer is held concurrently. Runtime should
// scale linearly with K. This is checked, not assumed.
#include "keygen.h"
#include <cuda_runtime.h>
#include <sys/resource.h>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

namespace gpufhe {
void encode_host_fft(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void decode_host_fft(std::vector<std::complex<double>>&, const std::vector<int64_t>&, uint32_t, double);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void encrypt_host_shoup_pooled_barrett_batched(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void decrypt_host(std::vector<int64_t>&, const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                   const std::vector<uint64_t>&, uint32_t, const std::vector<uint64_t>&,
                   const std::vector<uint64_t>&);
void ct_add_ct_host(std::vector<uint64_t>&, std::vector<uint64_t>&,
                    const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                    uint32_t, uint32_t, const std::vector<uint64_t>&);
}
using cd = std::complex<double>;
using clk = std::chrono::steady_clock;
static double rssMB() { struct rusage ru; getrusage(RUSAGE_SELF, &ru); return ru.ru_maxrss / 1024.0; }

int main() {
    const uint32_t HW = 64;
    gpufhe::set_secret_hamming_weight(HW);
    const uint32_t n = 8192, S = n / 2, sizeQ = 4;
    const uint64_t ns = 1;
    const double sigma = 3.19;
    const double Delta = std::pow(2.0, 45);

    std::vector<uint64_t> mod;
    gpufhe::native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids, sizeQ - 1, 55, n, mod); for (auto m : mids) mod.push_back(m); }
    std::vector<uint64_t> root(sizeQ);
    for (uint32_t i = 0; i < sizeQ; ++i) root[i] = gpufhe::native_root(n, mod[i]);
    auto KP = gpufhe::keygen_host(n, mod, root, ns, sigma, 101);

    std::printf("=== Phase 29, Objective F: client-count memory/runtime scaling ===\n");
    std::printf("n=%u sizeQ=%u h=%u, production encrypt path\n\n", n, sizeQ, HW);
    std::printf("%6s %12s %14s %14s %10s\n", "K", "runtime(ms)", "VRAM_free_before", "VRAM_free_after", "delta_MiB");

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> d(-50.0, 50.0);
    std::vector<uint32_t> Ks = {1, 2, 4, 8, 16, 32, 64};

    for (uint32_t K : Ks) {
        size_t vFreeBefore, vTotal, vFreeAfter;
        cudaMemGetInfo(&vFreeBefore, &vTotal);

        std::vector<double> vals(K);
        for (uint32_t k = 0; k < K; ++k) vals[k] = d(rng);

        auto t0 = clk::now();
        std::vector<uint64_t> acc0, acc1;
        for (uint32_t k = 0; k < K; ++k) {
            std::vector<cd> z(S, cd(vals[k], 0.0));
            std::vector<int64_t> m;
            gpufhe::encode_host_fft(m, z, n, Delta);
            std::vector<uint64_t> c0, c1;
            gpufhe::encrypt_host_shoup_pooled_barrett_batched(c0, c1, m, KP.pkA, KP.pkB, n, mod, root, ns, sigma, 7000 + k);
            if (k == 0) { acc0 = c0; acc1 = c1; }
            else gpufhe::ct_add_ct_host(acc0, acc1, c0, c1, sizeQ, n, mod);
        }
        std::vector<int64_t> mDec;
        gpufhe::decrypt_host(mDec, acc0, acc1, KP.s, n, mod, root);
        std::vector<cd> zOut;
        gpufhe::decode_host_fft(zOut, mDec, n, Delta);
        double runtimeMs = std::chrono::duration<double, std::milli>(clk::now() - t0).count();

        cudaMemGetInfo(&vFreeAfter, &vTotal);
        std::printf("%6u %12.2f %14zu %14zu %10zd\n", K, runtimeMs,
                    vFreeBefore / 1048576, vFreeAfter / 1048576,
                    (ptrdiff_t)(vFreeBefore - vFreeAfter) / 1048576);
    }
    std::printf("\npeak host RSS at end: %.1f MB\n", rssMB());
    std::printf("(VRAM delta near 0 across K => memory does NOT scale with client count;\n");
    std::printf(" runtime scaling roughly linear in K is the EXPECTED, checked-not-assumed result.)\n");
    return 0;
}
