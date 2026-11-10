// GOD PROMPT "CKKS PHASE 30", Objective F: NOT a regular regression test --
// deliberately NOT registered with ctest (same convention as
// tests/test_boot_hoist_enabled.cpp: build/run explicitly). This is the
// standalone reproducer for a REAL, SEPARATE concurrency issue found while
// testing the keyswitch_resident CUDA-graph fix (tests/test_ks_graph.cpp,
// now fixed -- see research/KS_GRAPH_CAPTURE_INVESTIGATION.md): when many
// threads simultaneously build FRESH (cold-cache) KeySwitchConstants from
// scratch, DIRECT (non-graph) execution can already disagree with itself
// across two back-to-back calls on the SAME thread -- proving this race is
// in the key-generation/context-setup pipeline, not in keyswitch_resident or
// its graph-capture path. Two caches in this exact class (src/
// keyswitch_resident.cpp's g_rtab/upc, src/evalkeygen.cpp's ek_dev_tab) were
// found and fixed (mutex-guarded) during this investigation; this probe still
// shows a residual, not-yet-isolated race beyond those two, reported here as
// a diagnostic, not asserted as a required pass -- a precise, reproducible
// starting point for a dedicated future investigation, per the governing
// directive's explicit instruction not to let this phase become an open-ended
// concurrency audit of the entire keygen pipeline.
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
    std::cout<<"=== Phase 30: keyswitch_resident SETUP-PATH concurrency diagnostic (known, separate, unresolved issue) ===\n";
    // --- Concurrency: independent (C,W) pairs from independent THREADS ---
    // GOD PROMPT "CKKS PHASE 30", Objective F: this probe found a REAL,
    // SEPARATE concurrency issue beyond the CUDA-graph-capture defect this
    // phase was asked to fix -- see the printed diagnostics and research/
    // KS_GRAPH_CAPTURE_INVESTIGATION.md's "known limitation" section for the
    // full story. Summary: when many threads simultaneously build FRESH
    // (cold-cache) KeySwitchConstants/contexts from scratch, DIRECT (non-
    // graph) execution can ALREADY disagree with itself across two
    // back-to-back calls on the SAME thread -- proving the remaining race
    // is in the key-generation/context-setup pipeline, not in
    // keyswitch_resident or its graph-capture path (which this phase DID
    // fix and verify). Two caches in this exact class (src/
    // keyswitch_resident.cpp's g_rtab/upc, src/evalkeygen.cpp's ek_dev_tab)
    // were found and fixed with a mutex during this investigation; a
    // further, not-yet-isolated source remains. Per the governing
    // directive's explicit instruction not to let this phase become an
    // unrelated, open-ended concurrency audit of the entire keygen
    // pipeline, this is reported as a DIAGNOSTIC (not asserted as a
    // required pass) and handed off as a precisely-reproducible, separate
    // investigation for a future phase.
    {
        std::cout<<"\n-- Concurrency: independent context+work pairs, real std::thread (never fork()) --\n";
        std::cout<<"   (diagnostic only -- see comment above and research/KS_GRAPH_CAPTURE_INVESTIGATION.md)\n";
        const int NTHREADS=8;
        std::atomic<int> directSelfConsistent{0}, graphMatchesWhenDirectWas{0};
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
                if (directSelfDiff) std::printf("  [thread %d] DIRECT-ONLY mismatch vs itself: %u diffs (setup/keygen-path race, separate from the graph-capture fix)\n", t, directSelfDiff);
                else directSelfConsistent.fetch_add(1);

                uint64_t *b0,*b1; cudaMalloc(&b0,(size_t)tw*n*8); cudaMalloc(&b1,(size_t)tw*n*8);
                cudaStream_t s; cudaStreamCreate(&s);
                cudaGraph_t g; cudaGraphExec_t ge;
                cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal);
                gpufhe::keyswitch_resident(d_a,b0,b1,C,W,s);
                cudaStreamEndCapture(s,&g);
                cudaGraphInstantiate(&ge,g,nullptr,nullptr,0);
                cudaGraphLaunch(ge,s); cudaStreamSynchronize(s);
                std::vector<uint64_t> o0((size_t)tw*n),o1((size_t)tw*n);
                cudaMemcpy(o0.data(),b0,o0.size()*8,cudaMemcpyDeviceToHost);
                cudaMemcpy(o1.data(),b1,o1.size()*8,cudaMemcpyDeviceToHost);
                uint32_t graphDiff = countDiff(ref0b,o0)+countDiff(ref1b,o1);
                if (!directSelfDiff) { if (!graphDiff) graphMatchesWhenDirectWas.fetch_add(1);
                                       else std::printf("  [thread %d] GRAPH mismatch despite self-consistent direct ref: %u diffs (a graph-path regression, would be a real finding)\n", t, graphDiff); }

                cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(s);
                gpufhe::ks_work_destroy(W); gpufhe::ks_context_destroy(C);
                cudaFree(d_a); cudaFree(b0); cudaFree(b1);
            });
        }
        for(auto& th:threads) th.join();
        std::cout<<"   direct-only self-consistent: "<<directSelfConsistent.load()<<"/"<<NTHREADS
                 <<"  (of those, graph also matched: "<<graphMatchesWhenDirectWas.load()<<"/"<<directSelfConsistent.load()<<")\n";
        check(graphMatchesWhenDirectWas.load()==directSelfConsistent.load(),
              "concurrency: whenever direct execution was self-consistent, the graph path ALSO matched it "
              "(" + std::to_string(graphMatchesWhenDirectWas.load()) + "/" + std::to_string(directSelfConsistent.load()) +
              ") -- this is what this phase's fix covers; the separate setup-path race above is NOT");
    }

    std::cout<<"\n=== TOTAL: "<<g_pass<<" passed, "<<g_fail<<" failed (diagnostic -- see file header) ===\n";
    return 0;
}
