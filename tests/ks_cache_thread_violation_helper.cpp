// GOD PROMPT "CKKS PHASE 27", Mission 5 (C3): standalone helper process for
// testing KsResidentCache's enforced single-thread violation.
//
// WHY A SEPARATE EXECUTABLE, NOT fork() FROM WITHIN THE MAIN TEST BINARY:
// CUDA explicitly does not support fork() once a process has an active CUDA
// context -- the child inherits a COPY of driver state that is not valid,
// and in practice this corrupts BOTH processes (confirmed this round: a
// bare fork() from inside test_ks_resident_cache.cpp, after that process had
// already touched CUDA for its correctness tests, produced
// "cudaMemGetInfo failed (initialization error)" in the child AND a SIGSEGV
// in the overall run, not the clean SIGABRT the test intended to observe).
// fork()+exec() (which is what invoking a SEPARATE already-linked executable
// via system()/posix_spawn does) is the standard-safe pattern: exec()
// replaces the process image entirely before any CUDA call happens, so the
// new process starts with NO inherited CUDA state at all.
//
// This executable does exactly one thing: build a KsResidentCache in its
// main thread, then deliberately call get() from a SECOND thread. The
// single-thread enforcement in include/ks_resident_cache.h must abort the
// process. The calling test (tests/test_ks_resident_cache.cpp) invokes this
// via system() and checks that the process died by SIGABRT, not that it
// exited cleanly.
#include "keyswitch.h"
#include "ks_resident_cache.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <thread>
#include <vector>
#include <cstdint>

namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void automorphism_eval_host(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t,
                            const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void set_secret_hamming_weight(uint32_t);
}

static uint64_t mm(uint64_t a, uint64_t b, uint64_t q) { return (uint64_t)(((unsigned __int128)a * b) % q); }

int main() {
    const uint32_t n = 1024, tw = 4, sizeP = 2, numPart = (tw + 1) / 2;
    const uint64_t ns = 1;
    gpufhe::set_secret_hamming_weight(64);
    std::vector<uint64_t> mod, modP;
    gpufhe::native_primes(mod, 1, 60, n, {});
    { std::vector<uint64_t> md; gpufhe::native_primes(md, tw - 1, 55, n, mod); for (auto m : md) mod.push_back(m); }
    gpufhe::native_primes(modP, sizeP, 60, n, mod);
    std::vector<uint64_t> mq(mod), rq;
    std::vector<uint64_t> root(tw);
    for (uint32_t i = 0; i < tw; ++i) root[i] = gpufhe::native_root(n, mod[i]);
    rq = root;
    for (auto p : modP) { mq.push_back(p); rq.push_back(gpufhe::native_root(n, p)); }
    const uint32_t sizeQlP = tw + sizeP;
    auto KPqp = gpufhe::keygen_host(n, mq, rq, ns, 3.19, 101);
    std::vector<uint64_t> PM(sizeQlP);
    for (uint32_t t = 0; t < sizeQlP; ++t) { uint64_t q = mq[t], P = 1 % q;
        for (uint32_t j = 0; j < sizeP; ++j) P = mm(P, modP[j] % q, q); PM[t] = P; }

    auto buildKey = [&](uint32_t k, uint32_t seed) {
        std::vector<uint64_t> sA = KPqp.s;
        gpufhe::automorphism_eval_host(sA, sizeQlP, n, k, mq, rq);
        gpufhe::KeySwitchConstants K; K.n = n;
        gpufhe::compute_keyswitch_constants(K, mod, modP, numPart);
        for (uint32_t t = 0; t < tw; ++t) { K.rootModList.push_back(mod[t]); K.rootValList.push_back(root[t]); }
        for (uint32_t j = 0; j < sizeP; ++j) { K.rootModList.push_back(modP[j]); K.rootValList.push_back(rq[tw + j]); }
        gpufhe::evalkeygen_host_sold(K, KPqp.s, sA, KPqp.pkA, KPqp.pkB, PM, mq, rq, ns, 3.19, seed);
        return K;
    };

    gpufhe::KsResidentCache cache(4);
    cache.get(5, tw, [&](uint32_t k, uint32_t) { return buildKey(k, 900); });  // fine, same (main) thread

    std::thread t([&]() {
        cache.get(25, tw, [&](uint32_t k, uint32_t) { return buildKey(k, 925); });  // VIOLATION: different thread
    });
    t.join();

    return 0;  // should never be reached -- the violation above must abort first
}
