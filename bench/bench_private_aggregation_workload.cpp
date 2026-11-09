// GOD PROMPT "CKKS PHASE 28", Objective E: the first real consumer-GPU CKKS
// use case, through the actual library API only (keygen_host/encode_host_fft
// /encrypt_host/ct_add_ct_host/decrypt_host/decode_host_fft -- no private
// kernels, no direct NTT/rotation/keyswitch calls).
//
// WORKLOAD: privacy-preserving aggregation of K client-held values (e.g. a
// secure sum over K parties' private readings -- the building block behind
// federated-learning secure aggregation and private survey tallying). Each
// client encodes+encrypts its OWN value independently (own RNG, own
// encryption randomness); the "server" never decrypts any individual
// ciphertext, only the homomorphic SUM of all K, via repeated
// ct_add_ct_host. This needs NO relin, NO rotation keys, NO rescale (a pure
// sum consumes no multiplicative depth), making it the lightest genuinely
// useful workload this library supports -- deliberately chosen over a
// weighted-sum/EvalMult variant, which would need rescale (a pathway
// research/NEXT_SESSION.md documents a real, unresolved bug in at n=1024;
// sidestepping it here, rather than risking it, is a deliberate scope
// choice for this workload, not an oversight).
//
// Repeated R times with FRESH randomized inputs each time (own message seed
// and own per-client encryption seed), reporting error against the
// plaintext ground truth, wall-clock runtime, and peak host/device memory,
// per the governing directive's Objective E requirements.
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
void encrypt_host(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
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
static double rssMB() { struct rusage ru; getrusage(RUSAGE_SELF, &ru); return ru.ru_maxrss / 1024.0; }
using clk = std::chrono::steady_clock;

int main() {
    const uint32_t HW = 64;                       // h=64, UNCHANGED per the governing directive
    gpufhe::set_secret_hamming_weight(HW);

    const uint32_t n = 8192, S = n / 2, sizeQ = 4; // shallow chain: pure sum consumes no depth
    const uint64_t ns = 1;
    const double sigma = 3.19;
    const double Delta = std::pow(2.0, 45);
    const uint32_t K = 32;   // clients contributing to the aggregate
    const uint32_t R = 10;   // repetitions, fresh randomized inputs each time

    std::vector<uint64_t> mod;
    gpufhe::native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> mids;
      gpufhe::native_primes(mids, sizeQ - 1, 55, n, mod);
      for (auto m : mids) mod.push_back(m); }
    std::vector<uint64_t> root(sizeQ);
    for (uint32_t i = 0; i < sizeQ; ++i) root[i] = gpufhe::native_root(n, mod[i]);

    auto t0 = clk::now();
    auto KP = gpufhe::keygen_host(n, mod, root, ns, sigma, 101);   // ONE key pair, shared (realistic deployment)
    double keygenMs = std::chrono::duration<double, std::milli>(clk::now() - t0).count();

    std::printf("=== Phase 28, Objective E: private aggregation over %u clients, n=%u, sizeQ=%u ===\n",
                K, n, sizeQ);
    std::printf("keygen: %.2f ms (once, amortized across every aggregation below)\n\n", keygenMs);

    std::mt19937_64 msgRng(42), encRng(999);
    double worstErr = 0.0, sumRuntimeMs = 0.0;

    for (uint32_t rep = 0; rep < R; ++rep) {
        // Each client draws its OWN fresh private value this repetition.
        std::vector<double> clientValues(K);
        std::uniform_real_distribution<double> d(-50.0, 50.0);
        for (uint32_t k = 0; k < K; ++k) clientValues[k] = d(msgRng);

        double plainSum = 0.0;
        for (double v : clientValues) plainSum += v;

        auto tStart = clk::now();
        std::vector<uint64_t> acc0, acc1;
        bool first = true;
        for (uint32_t k = 0; k < K; ++k) {
            // Each client packs its scalar into every slot (a real deployment
            // would pack K DIFFERENT clients' scalars into one ciphertext's
            // slots instead and reduce via rotation -- deliberately avoided
            // here, see file header, so this uses only already-verified,
            // rotation-free primitives).
            std::vector<cd> z(S, cd(clientValues[k], 0.0));
            std::vector<int64_t> m;
            gpufhe::encode_host_fft(m, z, n, Delta);
            std::vector<uint64_t> c0, c1;
            gpufhe::encrypt_host(c0, c1, m, KP.pkA, KP.pkB, n, mod, root, ns, sigma,
                                  (uint32_t)(5000 + rep * 100 + k));  // own encryption randomness per client
            if (first) { acc0 = c0; acc1 = c1; first = false; }
            else gpufhe::ct_add_ct_host(acc0, acc1, c0, c1, sizeQ, n, mod);
        }
        std::vector<int64_t> mDec;
        gpufhe::decrypt_host(mDec, acc0, acc1, KP.s, n, mod, root);
        std::vector<cd> zOut;
        gpufhe::decode_host_fft(zOut, mDec, n, Delta);
        double runtimeMs = std::chrono::duration<double, std::milli>(clk::now() - tStart).count();

        double decryptedSum = zOut[0].real();   // every slot holds the same aggregate by construction
        double err = std::abs(decryptedSum - plainSum);
        worstErr = std::max(worstErr, err);
        sumRuntimeMs += runtimeMs;

        std::printf("  rep %2u: plaintext sum=%10.4f  decrypted sum=%10.4f  abs err=%.6e  runtime=%.2f ms\n",
                    rep, plainSum, decryptedSum, err, runtimeMs);
    }

    size_t vramFree = 0, vramTotal = 0;
    cudaMemGetInfo(&vramFree, &vramTotal);
    std::printf("\n=== Summary ===\n");
    std::printf("  worst abs error across %u repetitions: %.6e\n", R, worstErr);
    std::printf("  mean per-aggregation runtime (K=%u clients): %.2f ms\n", K, sumRuntimeMs / R);
    std::printf("  peak host RSS: %.1f MB; VRAM free: %.0f/%.0f MiB\n",
                rssMB(), vramFree / 1048576.0, vramTotal / 1048576.0);
    std::printf("  [%s] all %u repetitions within a generous 1e-3 absolute-error sanity bound\n",
                worstErr < 1e-3 ? "PASS" : "FAIL", R);
    return worstErr < 1e-3 ? 0 : 1;
}
