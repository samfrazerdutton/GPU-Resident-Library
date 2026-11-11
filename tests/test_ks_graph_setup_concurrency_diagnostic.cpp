// GOD PROMPT "CKKS PHASE 30/31": originally a non-ctest diagnostic for a
// seemingly separate "setup/keygen-path" concurrency race, surfaced while
// testing the keyswitch_resident CUDA-graph fix (tests/test_ks_graph.cpp --
// see research/KS_GRAPH_CAPTURE_INVESTIGATION.md). Phase 31 root-caused it:
// there is NO race in keygen_host/evalkeygen_host/ks_context_create. Both
// symptoms this file used to show were artifacts of this file's OWN
// multi-threaded use of CUDA stream capture, not a defect in gpufhe:
//
//   1. "DIRECT-ONLY mismatch vs itself" (non-captured keyswitch_resident
//      disagreeing with itself on the SAME thread) was caused by ANOTHER
//      thread's cudaStreamBeginCapture(..., cudaStreamCaptureModeGlobal)
//      (on a plain blocking stream). Global capture mode's documented
//      semantics treat ALL stream activity in the WHOLE PROCESS as
//      relevant to that capture; concurrent direct (non-captured) use of
//      the shared legacy stream 0 by OTHER threads then fails with
//      cudaErrorStreamCaptureUnsupported / cudaErrorStreamCaptureImplicit.
//      This file never checked CUDA return codes, so the failure was
//      silent: the destination buffer simply kept its pre-existing
//      content, which reads exactly like a key-generation data race but
//      isn't one. Fix: cudaStreamCaptureModeThreadLocal + a non-blocking
//      capture stream (cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)).
//
//   2. "GRAPH mismatch despite self-consistent direct ref" persisted even
//      after fix (1), via a SECOND, independent mechanism: cudaMalloc/
//      cudaFree (including the ones inside ks_context_create/ks_work_create/
//      ks_context_destroy/ks_work_destroy, and this file's own b0/b1/d_a)
//      are "potentially unsafe" CUDA APIs whose restriction during stream
//      capture is NOT fully scoped away by ThreadLocal capture mode -- it
//      is a platform-level restriction independent of capture mode.
//      Another thread's cudaMalloc/cudaFree landing inside THIS thread's
//      capture window produces cudaErrorStreamCaptureInvalidated, which
//      cascades through cudaStreamEndCapture/cudaGraphInstantiate/
//      cudaGraphLaunch as "invalid argument" -- again silent, again
//      reading like a real-work regression. Fix: two spin-barriers so no
//      thread's allocate/free calls can execute while ANY other thread has
//      an active capture in progress.
//
// Evidence for both: see research/SETUP_PATH_CONCURRENCY_INVESTIGATION.md
// (bisected every setup-pipeline stage for self-consistency under the same
// 8-way load -- all clean; cross-thread-compared native_primes/native_root
// -- clean; a gpufhe-independent pure-CUDA malloc/free churn probe against
// a stable buffer -- clean; then used compute-sanitizer plus explicit
// cudaError_t capture around every call in the capture/instantiate/launch
// sequence to catch the exact two silently-failing CUDA API calls above).
// With both fixes applied, this file is now deterministic: 25/25 plain
// runs and 3/3 compute-sanitizer --tool memcheck runs, 8/8 threads clean
// every time (was previously 3-8/8 direct-self-consistent and 4-8/8
// graph-matches, nondeterministic run to run). Promoted to a real,
// ctest-registered regression test; no changes were needed to keygen_host,
// evalkeygen_host, ks_context_create, ks_work_create, or keyswitch_resident
// -- the entire fix is in how THIS FILE uses the CUDA stream-capture API,
// which is also the real-world usage contract: any caller that captures
// keyswitch_resident from multiple concurrent host threads must use
// cudaStreamCaptureModeThreadLocal with a non-blocking stream, and must not
// allocate/free device memory (directly, or via ks_context_create/
// ks_work_create/ks_context_destroy/ks_work_destroy) on any thread while
// another thread has an active capture in progress.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <atomic>
namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void set_secret_hamming_weight(uint32_t);
}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
static int g_pass=0, g_fail=0;
static void check(bool ok, const std::string& what){
    if(ok){++g_pass; std::cout<<"  [PASS] "<<what<<"\n";}
    else  {++g_fail; std::cout<<"  [FAIL] "<<what<<"\n";}
}

struct Fixture {
    uint32_t n, tw, sizeP, numPart, sizeQlP;
    std::vector<uint64_t> mod, modP, root, rq;
    gpufhe::KeySwitchConstants K;
};

static Fixture buildFixture(uint32_t n, uint32_t tw, uint64_t keySeed){
    Fixture F; F.n=n; F.tw=tw; F.sizeP=2; F.numPart=(tw+1)/2;
    const uint64_t ns=1;
    gpufhe::native_primes(F.mod,1,60,n,{});
    { std::vector<uint64_t> md; gpufhe::native_primes(md,tw-1,55,n,F.mod); for(auto m:md)F.mod.push_back(m); }
    gpufhe::native_primes(F.modP,F.sizeP,60,n,F.mod);
    std::vector<uint64_t> mq(F.mod); for(auto p:F.modP)mq.push_back(p);
    F.root.resize(tw); for(uint32_t i=0;i<tw;++i)F.root[i]=gpufhe::native_root(n,F.mod[i]);
    F.rq=F.root; for(auto p:F.modP)F.rq.push_back(gpufhe::native_root(n,p));
    F.sizeQlP=tw+F.sizeP;
    auto KPqp=gpufhe::keygen_host(n,mq,F.rq,ns,3.19,keySeed);
    std::vector<uint64_t> PM(F.sizeQlP);
    for(uint32_t t=0;t<F.sizeQlP;++t){uint64_t q=mq[t],P=1%q;for(uint32_t j=0;j<F.sizeP;++j)P=mm(P,F.modP[j]%q,q);PM[t]=P;}
    F.K.n=n;
    gpufhe::compute_keyswitch_constants(F.K,F.mod,F.modP,F.numPart);
    for(uint32_t i=0;i<tw;++i){F.K.rootModList.push_back(F.mod[i]);F.K.rootValList.push_back(F.root[i]);}
    for(uint32_t j=0;j<F.sizeP;++j){F.K.rootModList.push_back(F.modP[j]);F.K.rootValList.push_back(F.rq[tw+j]);}
    gpufhe::evalkeygen_host(F.K,KPqp.s,KPqp.pkA,KPqp.pkB,PM,mq,F.rq,ns,3.19,keySeed+100);
    return F;
}

static std::vector<uint64_t> randomInput(const Fixture& F, uint64_t seed){
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> a((size_t)F.tw*F.n);
    for(uint32_t t=0;t<F.tw;++t)for(uint32_t k=0;k<F.n;++k) a[(size_t)t*F.n+k]=rng()%F.mod[t];
    return a;
}

static bool runDirect(const Fixture& F, const gpufhe::DeviceKSContext& C, gpufhe::DeviceKSWork& W,
                       uint64_t* d_a, std::vector<uint64_t>& out0, std::vector<uint64_t>& out1){
    uint64_t *d_b0,*d_b1;
    cudaMalloc(&d_b0,(size_t)F.tw*F.n*8); cudaMalloc(&d_b1,(size_t)F.tw*F.n*8);
    gpufhe::keyswitch_resident(d_a,d_b0,d_b1,C,W,0);
    cudaError_t err = cudaDeviceSynchronize();
    out0.resize((size_t)F.tw*F.n); out1.resize((size_t)F.tw*F.n);
    cudaMemcpy(out0.data(),d_b0,out0.size()*8,cudaMemcpyDeviceToHost);
    cudaMemcpy(out1.data(),d_b1,out1.size()*8,cudaMemcpyDeviceToHost);
    cudaFree(d_b0); cudaFree(d_b1);
    return err==cudaSuccess;
}

static uint32_t countDiff(const std::vector<uint64_t>&a,const std::vector<uint64_t>&b){
    uint32_t d=0; for(size_t i=0;i<a.size()&&i<b.size();++i) if(a[i]!=b[i])++d; return d;
}


int main(){
    const uint32_t n=1024, tw=4;
    std::cout<<"=== Phase 31: keyswitch_resident setup-path + multi-threaded graph-capture concurrency (fixed, deterministic) ===\n";
    // --- Concurrency: independent (C,W) pairs from independent THREADS ---
    // See file header: both symptoms this test used to show (direct-path
    // self-mismatch; graph-path mismatch vs. a self-consistent direct ref)
    // were CUDA stream-capture API misuse in this file, not a gpufhe data
    // race. Fixed here via (1) cudaStreamCaptureModeThreadLocal + a
    // non-blocking capture stream, and (2) barriers keeping every thread's
    // cudaMalloc/cudaFree (direct and via ks_context_create/ks_work_create/
    // ks_context_destroy/ks_work_destroy) out of every OTHER thread's
    // active capture window. Every CUDA call in the capture/instantiate/
    // launch sequence is now explicitly checked; any regression here must
    // fail loudly instead of silently returning stale buffer content.
    {
        std::cout<<"\n-- Concurrency: independent context+work pairs, real std::thread (never fork()) --\n";
        const int NTHREADS=8;
        std::atomic<int> directSelfConsistent{0}, graphMatchesWhenDirectWas{0};
        // Spin-barriers enforcing the usage contract documented above: no
        // thread may allocate/free device memory while another thread has
        // an active stream capture in progress (cudaMalloc/cudaFree are
        // "potentially unsafe" CUDA APIs whose capture restriction is NOT
        // scoped away by ThreadLocal capture mode -- see file header).
        std::atomic<int> allocBarrier{0}, freeBarrier{0};
        auto spinBarrier=[&](std::atomic<int>& b){ b.fetch_add(1); while(b.load()<NTHREADS) std::this_thread::yield(); };
        std::atomic<int> cudaErrors{0};
        std::vector<std::thread> threads;
        for(int t=0;t<NTHREADS;++t){
            threads.emplace_back([&,t](){
                Fixture F = buildFixture(n,tw,300+(uint64_t)t);
                auto C = gpufhe::ks_context_create(F.K);
                auto W = gpufhe::ks_work_create(C);
                auto a = randomInput(F,500+(uint64_t)t);
                uint64_t* d_a; cudaMalloc(&d_a,a.size()*8); cudaMemcpy(d_a,a.data(),a.size()*8,cudaMemcpyHostToDevice);
                std::vector<uint64_t> ref0,ref1; runDirect(F,C,W,d_a,ref0,ref1);
                std::vector<uint64_t> ref0b,ref1b; runDirect(F,C,W,d_a,ref0b,ref1b);
                uint32_t directSelfDiff = countDiff(ref0,ref0b)+countDiff(ref1,ref1b);
                if (directSelfDiff) std::printf("  [thread %d] DIRECT-ONLY mismatch vs itself: %u diffs (unexpected -- would be a real regression)\n", t, directSelfDiff);
                else directSelfConsistent.fetch_add(1);

                uint64_t *b0,*b1; cudaMalloc(&b0,(size_t)tw*n*8); cudaMalloc(&b1,(size_t)tw*n*8);
                cudaStream_t s; cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking);
                cudaGraph_t g; cudaGraphExec_t ge;
                spinBarrier(allocBarrier); // no thread begins capture until ALL threads' setup-time allocs are done
                cudaError_t e1=cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal);
                gpufhe::keyswitch_resident(d_a,b0,b1,C,W,s);
                cudaError_t e2=cudaStreamEndCapture(s,&g);
                cudaError_t e3=cudaGraphInstantiate(&ge,g,nullptr,nullptr,0);
                cudaError_t e4=cudaGraphLaunch(ge,s);
                cudaError_t e5=cudaStreamSynchronize(s);
                std::vector<uint64_t> o0((size_t)tw*n),o1((size_t)tw*n);
                cudaError_t e6=cudaMemcpy(o0.data(),b0,o0.size()*8,cudaMemcpyDeviceToHost);
                cudaError_t e7=cudaMemcpy(o1.data(),b1,o1.size()*8,cudaMemcpyDeviceToHost);
                if (e1||e2||e3||e4||e5||e6||e7) {
                    cudaErrors.fetch_add(1);
                    std::printf("  [thread %d] CUDA error in capture/launch sequence: begin=%s end=%s inst=%s launch=%s sync=%s cpy0=%s cpy1=%s\n", t,
                        cudaGetErrorString(e1), cudaGetErrorString(e2), cudaGetErrorString(e3),
                        cudaGetErrorString(e4), cudaGetErrorString(e5), cudaGetErrorString(e6), cudaGetErrorString(e7));
                }
                uint32_t graphDiff = countDiff(ref0b,o0)+countDiff(ref1b,o1);
                if (!directSelfDiff) { if (!graphDiff) graphMatchesWhenDirectWas.fetch_add(1);
                                       else std::printf("  [thread %d] GRAPH mismatch despite self-consistent direct ref: %u diffs (unexpected -- would be a real regression)\n", t, graphDiff); }

                spinBarrier(freeBarrier); // no thread frees/destroys until ALL threads finished their own capture+launch+readback
                cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(s);
                gpufhe::ks_work_destroy(W); gpufhe::ks_context_destroy(C);
                cudaFree(d_a); cudaFree(b0); cudaFree(b1);
            });
        }
        for(auto& th:threads) th.join();
        std::cout<<"   direct-only self-consistent: "<<directSelfConsistent.load()<<"/"<<NTHREADS
                 <<"  (of those, graph also matched: "<<graphMatchesWhenDirectWas.load()<<"/"<<directSelfConsistent.load()<<")\n";
        check(cudaErrors.load()==0, "no CUDA API errors in the multi-threaded capture/instantiate/launch sequence");
        check(directSelfConsistent.load()==NTHREADS,
              "setup path: direct (non-captured) keyswitch_resident is self-consistent on every thread ("
              + std::to_string(directSelfConsistent.load()) + "/" + std::to_string(NTHREADS) + ")");
        check(graphMatchesWhenDirectWas.load()==directSelfConsistent.load(),
              "concurrency: whenever direct execution was self-consistent, the graph path ALSO matched it "
              "(" + std::to_string(graphMatchesWhenDirectWas.load()) + "/" + std::to_string(directSelfConsistent.load()) + ")");
    }

    std::cout<<"\n=== TOTAL: "<<g_pass<<" passed, "<<g_fail<<" failed ===\n";
    return g_fail==0 ? 0 : 1;
}
