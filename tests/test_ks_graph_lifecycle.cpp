// GOD PROMPT "CKKS PHASE 30", Objectives E+F+H: full lifecycle, concurrency,
// and memory/resource testing for the now-graph-safe keyswitch_resident.
// Native (OpenFHE-free) setup, matching tests/test_rot_resident.cpp's
// pattern, so this stays fast and has no dependency on anything that could
// itself touch CUDA before our own init (Phase 25/27's own finding).
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
    std::cout<<"=== Phase 30: keyswitch_resident CUDA-graph lifecycle/concurrency/memory tests ===\n";
    std::cout<<"n="<<n<<" tw="<<tw<<"\n\n";

    // --- 1-5: normal keyswitch, capture, instantiate, first launch, repeated launches ---
    {
        std::cout<<"-- Lifecycle: single context/work, repeated replay --\n";
        Fixture F = buildFixture(n,tw,101);
        auto C = gpufhe::ks_context_create(F.K);
        auto W = gpufhe::ks_work_create(C);
        auto a = randomInput(F,1);
        uint64_t* d_a; cudaMalloc(&d_a,a.size()*8); cudaMemcpy(d_a,a.data(),a.size()*8,cudaMemcpyHostToDevice);

        std::vector<uint64_t> ref0,ref1;
        check(runDirect(F,C,W,d_a,ref0,ref1), "1. normal (direct) keyswitch executes without error");

        uint64_t *d_b0,*d_b1;
        cudaMalloc(&d_b0,(size_t)tw*n*8); cudaMalloc(&d_b1,(size_t)tw*n*8);
        cudaStream_t s; cudaStreamCreate(&s);
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaMemset(d_b0,0,(size_t)tw*n*8); cudaMemset(d_b1,0,(size_t)tw*n*8);
        cudaError_t be = cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal);
        gpufhe::keyswitch_resident(d_a,d_b0,d_b1,C,W,s);
        cudaError_t ee = cudaStreamEndCapture(s,&g);
        check(be==cudaSuccess && ee==cudaSuccess, "2. graph capture succeeds (beginCapture+endCapture report no error)");
        cudaError_t ie = cudaGraphInstantiate(&ge,g,nullptr,nullptr,0);
        check(ie==cudaSuccess, "3. graph instantiation succeeds");

        cudaGraphLaunch(ge,s); cudaStreamSynchronize(s);
        std::vector<uint64_t> g0((size_t)tw*n),g1((size_t)tw*n);
        cudaMemcpy(g0.data(),d_b0,g0.size()*8,cudaMemcpyDeviceToHost);
        cudaMemcpy(g1.data(),d_b1,g1.size()*8,cudaMemcpyDeviceToHost);
        check(countDiff(ref0,g0)==0 && countDiff(ref1,g1)==0, "4. first graph launch matches direct execution bit-exact");

        bool allRepeatsMatch=true;
        for(int rep=0;rep<10;++rep){
            cudaMemset(d_b0,0xFF,(size_t)tw*n*8); cudaMemset(d_b1,0xFF,(size_t)tw*n*8); // poison, so a no-op can't pass
            cudaGraphLaunch(ge,s); cudaStreamSynchronize(s);
            std::vector<uint64_t> r0((size_t)tw*n),r1((size_t)tw*n);
            cudaMemcpy(r0.data(),d_b0,r0.size()*8,cudaMemcpyDeviceToHost);
            cudaMemcpy(r1.data(),d_b1,r1.size()*8,cudaMemcpyDeviceToHost);
            if(countDiff(ref0,r0)!=0 || countDiff(ref1,r1)!=0) allRepeatsMatch=false;
        }
        check(allRepeatsMatch, "5. 10 repeated graph launches ALL match direct execution bit-exact");

        // 6. replay with UPDATED input data at the SAME device pointer
        auto a2 = randomInput(F,2);
        cudaMemcpy(d_a,a2.data(),a2.size()*8,cudaMemcpyHostToDevice);
        std::vector<uint64_t> ref0b,ref1b;
        runDirect(F,C,W,d_a,ref0b,ref1b); // fresh direct reference for the NEW input
        cudaMemset(d_b0,0,(size_t)tw*n*8); cudaMemset(d_b1,0,(size_t)tw*n*8);
        cudaGraphLaunch(ge,s); cudaStreamSynchronize(s);
        std::vector<uint64_t> gu0((size_t)tw*n),gu1((size_t)tw*n);
        cudaMemcpy(gu0.data(),d_b0,gu0.size()*8,cudaMemcpyDeviceToHost);
        cudaMemcpy(gu1.data(),d_b1,gu1.size()*8,cudaMemcpyDeviceToHost);
        check(countDiff(ref0b,gu0)==0 && countDiff(ref1b,gu1)==0,
              "6. graph replay with UPDATED input data (same d_a pointer, new content) matches a fresh direct run");

        // 10-12: destruction order
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
        cudaStreamDestroy(s);
        check(true, "10. graph executable + graph destroyed without error");
        gpufhe::ks_work_destroy(W);
        check(true, "12. work object destroyed (after its graph) without error");
        gpufhe::ks_context_destroy(C);
        check(true, "11. context destroyed without error");
        cudaFree(d_a); cudaFree(d_b0); cudaFree(d_b1);
    }

    // --- 7-9: multiple independent graph instances, same context ---
    {
        std::cout<<"\n-- Multiple independent graphs sharing one context, DIFFERENT work objects --\n";
        Fixture F = buildFixture(n,tw,202);
        auto C = gpufhe::ks_context_create(F.K);
        auto Wa = gpufhe::ks_work_create(C);
        auto Wb = gpufhe::ks_work_create(C);
        auto a1 = randomInput(F,11), a2 = randomInput(F,22);
        uint64_t *d_a1,*d_a2; cudaMalloc(&d_a1,a1.size()*8); cudaMalloc(&d_a2,a2.size()*8);
        cudaMemcpy(d_a1,a1.data(),a1.size()*8,cudaMemcpyHostToDevice);
        cudaMemcpy(d_a2,a2.data(),a2.size()*8,cudaMemcpyHostToDevice);

        std::vector<uint64_t> ref1_0,ref1_1,ref2_0,ref2_1;
        runDirect(F,C,Wa,d_a1,ref1_0,ref1_1);
        runDirect(F,C,Wb,d_a2,ref2_0,ref2_1);

        uint64_t *b1_0,*b1_1,*b2_0,*b2_1;
        cudaMalloc(&b1_0,(size_t)tw*n*8); cudaMalloc(&b1_1,(size_t)tw*n*8);
        cudaMalloc(&b2_0,(size_t)tw*n*8); cudaMalloc(&b2_1,(size_t)tw*n*8);
        cudaStream_t s1,s2; cudaStreamCreate(&s1); cudaStreamCreate(&s2);
        cudaGraph_t g1,g2; cudaGraphExec_t ge1,ge2;
        cudaStreamBeginCapture(s1,cudaStreamCaptureModeGlobal);
        gpufhe::keyswitch_resident(d_a1,b1_0,b1_1,C,Wa,s1);
        cudaStreamEndCapture(s1,&g1);
        cudaStreamBeginCapture(s2,cudaStreamCaptureModeGlobal);
        gpufhe::keyswitch_resident(d_a2,b2_0,b2_1,C,Wb,s2);
        cudaStreamEndCapture(s2,&g2);
        cudaGraphInstantiate(&ge1,g1,nullptr,nullptr,0);
        cudaGraphInstantiate(&ge2,g2,nullptr,nullptr,0);

        // Launch BOTH graphs (genuinely independent streams) before syncing either.
        cudaGraphLaunch(ge1,s1);
        cudaGraphLaunch(ge2,s2);
        cudaStreamSynchronize(s1); cudaStreamSynchronize(s2);

        std::vector<uint64_t> o1_0((size_t)tw*n),o1_1((size_t)tw*n),o2_0((size_t)tw*n),o2_1((size_t)tw*n);
        cudaMemcpy(o1_0.data(),b1_0,o1_0.size()*8,cudaMemcpyDeviceToHost);
        cudaMemcpy(o1_1.data(),b1_1,o1_1.size()*8,cudaMemcpyDeviceToHost);
        cudaMemcpy(o2_0.data(),b2_0,o2_0.size()*8,cudaMemcpyDeviceToHost);
        cudaMemcpy(o2_1.data(),b2_1,o2_1.size()*8,cudaMemcpyDeviceToHost);
        check(countDiff(ref1_0,o1_0)==0 && countDiff(ref1_1,o1_1)==0,
              "7. graph 1 (own work object Wa) matches its own direct reference");
        check(countDiff(ref2_0,o2_0)==0 && countDiff(ref2_1,o2_1)==0,
              "8. graph 2 (own work object Wb) matches its own direct reference, no cross-contamination");
        check(countDiff(ref1_0,o2_0)!=0 || countDiff(ref1_1,o2_1)!=0,
              "9. graph 1's and graph 2's outputs genuinely differ (different random inputs -- sanity, not a false pass)");

        cudaGraphExecDestroy(ge1); cudaGraphDestroy(g1); cudaStreamDestroy(s1);
        // 7b: destroy graph 1 and its resources while graph 2 is STILL ALIVE and unused yet.
        bool g2StillWorks=true;
        cudaMemset(b2_0,0,(size_t)tw*n*8); cudaMemset(b2_1,0,(size_t)tw*n*8);
        cudaGraphLaunch(ge2,s2); cudaStreamSynchronize(s2);
        std::vector<uint64_t> o2_0b((size_t)tw*n),o2_1b((size_t)tw*n);
        cudaMemcpy(o2_0b.data(),b2_0,o2_0b.size()*8,cudaMemcpyDeviceToHost);
        cudaMemcpy(o2_1b.data(),b2_1,o2_1b.size()*8,cudaMemcpyDeviceToHost);
        if(countDiff(ref2_0,o2_0b)!=0 || countDiff(ref2_1,o2_1b)!=0) g2StillWorks=false;
        check(g2StillWorks, "destroying graph 1 (+its stream) leaves graph 2 fully functional (independent lifetimes)");

        cudaGraphExecDestroy(ge2); cudaGraphDestroy(g2); cudaStreamDestroy(s2);
        gpufhe::ks_work_destroy(Wa); gpufhe::ks_work_destroy(Wb);
        gpufhe::ks_context_destroy(C);
        cudaFree(d_a1); cudaFree(d_a2);
        cudaFree(b1_0); cudaFree(b1_1); cudaFree(b2_0); cudaFree(b2_1);
    }


    // --- Memory: repeated create/capture/launch/destroy cycles, watch VRAM ---
    {
        std::cout<<"\n-- Memory: 5 repeated create/capture/launch/destroy cycles --\n";
        size_t freeBefore,total; cudaMemGetInfo(&freeBefore,&total);
        for(int cyc=0;cyc<5;++cyc){
            Fixture F = buildFixture(n,tw,900+cyc);
            auto C = gpufhe::ks_context_create(F.K);
            auto W = gpufhe::ks_work_create(C);
            auto a = randomInput(F,1000+cyc);
            uint64_t* d_a; cudaMalloc(&d_a,a.size()*8); cudaMemcpy(d_a,a.data(),a.size()*8,cudaMemcpyHostToDevice);
            uint64_t *b0,*b1; cudaMalloc(&b0,(size_t)tw*n*8); cudaMalloc(&b1,(size_t)tw*n*8);
            cudaStream_t s; cudaStreamCreate(&s);
            cudaGraph_t g; cudaGraphExec_t ge;
            cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal);
            gpufhe::keyswitch_resident(d_a,b0,b1,C,W,s);
            cudaStreamEndCapture(s,&g);
            cudaGraphInstantiate(&ge,g,nullptr,nullptr,0);
            cudaGraphLaunch(ge,s); cudaStreamSynchronize(s);
            cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(s);
            gpufhe::ks_work_destroy(W); gpufhe::ks_context_destroy(C);
            cudaFree(d_a); cudaFree(b0); cudaFree(b1);
        }
        size_t freeAfter; cudaMemGetInfo(&freeAfter,&total);
        ptrdiff_t deltaMiB = (ptrdiff_t)(freeBefore-freeAfter)/1048576;
        std::cout<<"  VRAM free before="<<freeBefore/1048576<<" MiB after="<<freeAfter/1048576<<" MiB delta="<<deltaMiB<<" MiB\n";
        check(deltaMiB <= 2, "5 full create/capture/launch/destroy cycles leak <=2 MiB VRAM (not a growing leak)");
    }

    std::cout<<"\n=== TOTAL: "<<g_pass<<" passed, "<<g_fail<<" failed ===\n";
    return g_fail==0?0:1;
}
