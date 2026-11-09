// GOD PROMPT "CKKS PHASE 27", Mission 1: quantify the create-vs-destroy
// timing asymmetry identified by reading bench_ks_context_resident_ab.cpp.
// In that benchmark, Arm A's measured interval includes 48 ks_context_create
// AND 48 ks_context_destroy calls; Arm B's measured interval includes only
// the 8 ks_context_create calls on cache miss -- the matching 8 destroys
// happen in the cache's destructor, AFTER the timer has already stopped.
// This probe measures ks_context_create and ks_context_destroy SEPARATELY,
// at the same bootstrap parameters, to determine whether that asymmetry is
// large enough to matter for the reported 1.66-1.69x figure.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <algorithm>

namespace gpufhe {
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void automorphism_eval_host(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t,
                            const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void set_secret_hamming_weight(uint32_t);
}
static uint64_t mm(uint64_t a, uint64_t b, uint64_t q) { return (uint64_t)(((unsigned __int128)a * b) % q); }
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

int main() {
    const uint32_t n = 8192, tw = 30, sizeP = 2, numPart = 15;
    const uint64_t ns = 1;
    const uint32_t sizeQlP = tw + sizeP;
    gpufhe::set_secret_hamming_weight(64);

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
    std::vector<uint64_t> PM(sizeQlP);
    for (uint32_t t = 0; t < sizeQlP; ++t) {
        uint64_t q = mq[t], P = 1 % q;
        for (uint32_t j = 0; j < sizeP; ++j) P = mm(P, modP[j] % q, q);
        PM[t] = P;
    }
    uint32_t k = 5;
    std::vector<uint64_t> pkA_h = KPqp.pkA, pkB_h = KPqp.pkB;
    gpufhe::automorphism_eval_host(pkA_h, sizeQlP, n, k, mq, rq);
    gpufhe::automorphism_eval_host(pkB_h, sizeQlP, n, k, mq, rq);
    gpufhe::KeySwitchConstants K; K.n = n;
    gpufhe::compute_keyswitch_constants(K, mod, modP, numPart);
    for (uint32_t t = 0; t < tw; ++t) { K.rootModList.push_back(mod[t]); K.rootValList.push_back(root[t]); }
    for (uint32_t j = 0; j < sizeP; ++j) { K.rootModList.push_back(modP[j]); K.rootValList.push_back(rq[tw + j]); }
    gpufhe::evalkeygen_host_sold(K, KPqp.s, KPqp.s, pkA_h, pkB_h, PM, mq, rq, ns, 3.19, 900);

    // warm up the shared caches (g_rtab, upc) so only the per-context cost
    // (av/bv upload + keyRow + frees) is measured, matching what a SECOND
    // and later ks_context_create/destroy call for a DIFFERENT key would cost
    // once the chain-shape caches are already warm -- exactly the steady-state
    // condition both arms of bench_ks_context_resident_ab run under after the
    // first miss.
    { auto C = gpufhe::ks_context_create(K); gpufhe::ks_context_destroy(C); }

    const int TRIALS = 15;
    std::vector<double> vCreate, vDestroy;
    for (int t = 0; t < TRIALS; ++t) {
        auto t0 = std::chrono::steady_clock::now();
        auto C = gpufhe::ks_context_create(K);
        cudaDeviceSynchronize();
        double tCreate = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        auto t1 = std::chrono::steady_clock::now();
        gpufhe::ks_context_destroy(C);
        cudaDeviceSynchronize();
        double tDestroy = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

        vCreate.push_back(tCreate);
        vDestroy.push_back(tDestroy);
    }
    double mCreate = median(vCreate), mDestroy = median(vDestroy);
    std::printf("ks_context_create : median %.6f s  [%.6f..%.6f]\n", mCreate,
                *std::min_element(vCreate.begin(), vCreate.end()), *std::max_element(vCreate.begin(), vCreate.end()));
    std::printf("ks_context_destroy: median %.6f s  [%.6f..%.6f]\n", mDestroy,
                *std::min_element(vDestroy.begin(), vDestroy.end()), *std::max_element(vDestroy.begin(), vDestroy.end()));
    std::printf("destroy/create ratio: %.4f (fraction of create's cost that destroy adds)\n", mDestroy / mCreate);
    std::printf("\nBIAS ESTIMATE for bench_ks_context_resident_ab's 48-call/8-key run:\n");
    std::printf("  Arm A's measured interval includes 48 creates + 48 destroys.\n");
    std::printf("  Arm B's measured interval includes  8 creates +  0 destroys (destroys deferred past the timer).\n");
    std::printf("  If Arm A's TRUE combined cost is 48*(create+destroy) and its reported time credits exactly that,\n");
    std::printf("  while Arm B's reported time is underclaimed by 8*destroy (excluded), the corrected comparison\n");
    std::printf("  would adjust Arm B's reported median UP by roughly 8*%.6f = %.6f s before comparing.\n",
                mDestroy, 8 * mDestroy);
    return 0;
}
