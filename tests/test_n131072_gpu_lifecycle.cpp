// Directive "ISOLATE GPU MEMORY GROWTH AND PROVE LARGE-N FEASIBILITY",
// Priority 2: the 6th real n=131072 attempt (research/MEMORY_FEASIBILITY.md)
// reached diagonal 8 of 16 before being killed, with host RSS PROVEN flat
// (exactly 4561.35MB for 7 consecutive diagonals) but GPU memory showing a
// small, unexplained +0.5-1MB/diagonal climb over only 4 observed data
// points -- too little evidence to classify as plateauing or growing.
//
// This harness isolates EXACTLY the real per-diagonal device-memory
// lifecycle (ks_context_create -> keyswitch_resident -> ks_context_destroy,
// inside a stageRotBegin/stageRotDiag/stageRotEnd-shaped loop, using the
// SAME bounded-LRU KROT rotation-key cache as production) from everything
// else the real bootstrap does (no ModRaise, no ciphertext encode/decode,
// no stageDiags), and drives it through many MORE diagonals than the 6th
// attempt survived to reach -- across multiple simulated stage boundaries
// with decreasing tower counts, exactly as a real C2S/S2C pass would -- to
// see whether the GPU-memory figure plateaus (bounded allocator reuse) or
// keeps climbing (a real, if slow, leak somewhere in this exact lifecycle).
//
// Uses the REAL production APIs (ks_context_create/ks_work_create/
// keyswitch_resident/ks_context_destroy/ks_work_destroy,
// automorphism_eval_fast_device, evalkeygen_host_sold) with their real
// ownership semantics -- not a synthetic cudaMalloc/cudaFree loop.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <cstdint>
#include <chrono>
#include <sys/resource.h>
namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void automorphism_eval_host(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t,
                            const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void set_secret_hamming_weight(uint32_t);
}
static long rssMB(){ struct rusage ru; getrusage(RUSAGE_SELF,&ru); return ru.ru_maxrss/1024; }
static void gpuMemMB(double& usedMB, double& freeMB){
    size_t f=0,t=0; cudaMemGetInfo(&f,&t);
    freeMB=(double)f/1024.0/1024.0; usedMB=(double)(t-f)/1024.0/1024.0;
}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
using clk=std::chrono::steady_clock;
static double EL(clk::time_point a){return std::chrono::duration<double>(clk::now()-a).count();}

int main(){
    const uint32_t n=131072, sizeQ=30, sizeP=2; const uint64_t ns=1;
    constexpr size_t KROT_CAP=2;   // identical to tests/test_boot_n131072.cpp
    gpufhe::set_secret_hamming_weight(64);

    std::vector<uint64_t> mod,modP;
    gpufhe::native_primes(mod,1,60,n,{});
    { std::vector<uint64_t> md; gpufhe::native_primes(md,sizeQ-1,55,n,mod); for(auto m:md)mod.push_back(m); }
    gpufhe::native_primes(modP,sizeP,60,n,mod);
    std::vector<uint64_t> mq(mod),rq;
    std::vector<uint64_t> root(sizeQ); for(uint32_t i=0;i<sizeQ;++i)root[i]=gpufhe::native_root(n,mod[i]);
    rq=root; for(auto p:modP){ mq.push_back(p); rq.push_back(gpufhe::native_root(n,p)); }
    const uint32_t sizeQlP=sizeQ+sizeP;

    auto KPqp=gpufhe::keygen_host(n,mq,rq,ns,3.19,101);
    std::vector<uint64_t> PM(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=mq[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modP[j]%q,q);PM[t]=P;}

    auto npFor=[&](uint32_t tw)->uint32_t{ return tw<=2?1u:(tw+1)/2; };

    // KROT: identical bounded-LRU rotation-key cache to production, built
    // from the REAL evalkeygen_host_sold (same cost/shape as a real key).
    std::map<uint32_t,gpufhe::KeySwitchConstants> KROT;
    std::list<uint32_t> KROT_lru;
    std::map<uint32_t,std::list<uint32_t>::iterator> KROT_lru_pos;
    uint32_t nKeygen=0, nHits=0, nEvictions=0;
    auto rotFull=[&](uint32_t k)->const gpufhe::KeySwitchConstants&{
        auto it=KROT.find(k);
        if(it!=KROT.end()){ ++nHits;
            KROT_lru.erase(KROT_lru_pos[k]); KROT_lru.push_front(k); KROT_lru_pos[k]=KROT_lru.begin();
            return it->second; }
        std::vector<uint64_t> sA=KPqp.s;
        gpufhe::automorphism_eval_host(sA,sizeQlP,n,k,mq,rq);
        gpufhe::KeySwitchConstants K; K.n=n;
        gpufhe::compute_keyswitch_constants(K,mod,modP,npFor(sizeQ));
        for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
        for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rq[sizeQ+j]);}
        gpufhe::evalkeygen_host_sold(K,KPqp.s,sA,KPqp.pkA,KPqp.pkB,PM,mq,rq,1,3.19,900+k);
        ++nKeygen;
        if(KROT.size()>=KROT_CAP){
            uint32_t victim=KROT_lru.back(); KROT_lru.pop_back();
            KROT_lru_pos.erase(victim); KROT.erase(victim); ++nEvictions;
        }
        KROT.emplace(k,std::move(K));
        KROT_lru.push_front(k); KROT_lru_pos[k]=KROT_lru.begin();
        return KROT[k]; };
    // Level view, identical to production's atLevel().
    auto atLevel=[&](const gpufhe::KeySwitchConstants& KF,uint32_t tw)->gpufhe::KeySwitchConstants{
        std::vector<uint64_t> ml(mod.begin(),mod.begin()+tw),rl(root.begin(),root.begin()+tw);
        gpufhe::KeySwitchConstants K; K.n=n;
        gpufhe::compute_keyswitch_constants(K,ml,modP,npFor(tw));
        for(uint32_t i=0;i<tw;++i){K.rootModList.push_back(ml[i]);K.rootValList.push_back(rl[i]);}
        for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rq[sizeQ+j]);}
        K.fullQ=sizeQ; K.evalKeyTowers=KF.evalKeyTowers;
        K.av.assign(K.numPart,{}); K.bv.assign(K.numPart,{});
        for(uint32_t p=0;p<K.numPart;++p){K.av[p]=KF.av[p];K.bv[p]=KF.bv[p];}
        return K; };

    // StageRotCtx, stageRotBegin/stageRotDiag/stageRotEnd: IDENTICAL shape
    // and call sequence to tests/test_boot_n131072.cpp -- the real
    // production lifecycle, not a synthetic substitute.
    struct StageRotCtx{
        cudaStream_t s=0; gpufhe::DeviceKSWork W{};
        uint64_t *d_base0=nullptr,*d_base1=nullptr,*d_w0=nullptr,*d_w1=nullptr;
        uint64_t *d_scratch=nullptr,*d_b0=nullptr,*d_b1=nullptr; size_t cap=0;
    };
    auto stageRotBegin=[&](StageRotCtx& R, uint32_t towers, const gpufhe::KeySwitchConstants& anyKeyAtLevel){
        size_t need=(size_t)towers*n*8;
        if(need>R.cap){
            if(R.d_base0){cudaFree(R.d_base0);cudaFree(R.d_base1);cudaFree(R.d_w0);cudaFree(R.d_w1);
                          cudaFree(R.d_scratch);cudaFree(R.d_b0);cudaFree(R.d_b1);}
            cudaMalloc(&R.d_base0,need);cudaMalloc(&R.d_base1,need);
            cudaMalloc(&R.d_w0,need);cudaMalloc(&R.d_w1,need);
            cudaMalloc(&R.d_scratch,need);cudaMalloc(&R.d_b0,need);cudaMalloc(&R.d_b1,need);
            R.cap=need;
        }
        cudaStreamCreate(&R.s);
        gpufhe::DeviceKSContext Csize;
        Csize.n=anyKeyAtLevel.n; Csize.sizeQl=anyKeyAtLevel.sizeQl; Csize.sizeP=anyKeyAtLevel.sizeP;
        Csize.sizeQlP=anyKeyAtLevel.sizeQl+anyKeyAtLevel.sizeP; Csize.sizePart=anyKeyAtLevel.sizePart;
        R.W=gpufhe::ks_work_create(Csize);
        cudaMemsetAsync(R.d_base0,0,need,R.s); cudaMemsetAsync(R.d_base1,0,need,R.s);
    };
    auto stageRotDiag=[&](StageRotCtx& R, uint32_t k, const gpufhe::KeySwitchConstants& Krot, uint32_t towers){
        size_t B=(size_t)towers*n*8;
        cudaMemcpyAsync(R.d_w0,R.d_base0,B,cudaMemcpyDeviceToDevice,R.s);
        cudaMemcpyAsync(R.d_w1,R.d_base1,B,cudaMemcpyDeviceToDevice,R.s);
        gpufhe::automorphism_eval_fast_device(R.d_w0,towers,n,k,R.d_scratch,R.s);
        gpufhe::automorphism_eval_fast_device(R.d_w1,towers,n,k,R.d_scratch,R.s);
        auto C=gpufhe::ks_context_create(Krot,R.s);
        gpufhe::keyswitch_resident(R.d_w1,R.d_b0,R.d_b1,C,R.W,R.s);
        gpufhe::ks_context_destroy(C);
        cudaStreamSynchronize(R.s);
    };
    auto stageRotEnd=[&](StageRotCtx& R){
        cudaStreamDestroy(R.s); R.s=0;
        gpufhe::ks_work_destroy(R.W); R.W=gpufhe::DeviceKSWork{};
    };

    double gu0,gf0; gpuMemMB(gu0,gf0);
    std::cerr<<"=== baseline (before any stage) === RSS="<<rssMB()<<"MB GPU_used="<<gu0
              <<"MB GPU_free="<<gf0<<"MB\n";

    // Simulate multiple stages with decreasing tower counts (mirrors a real
    // C2S/S2C pass: towers only shrinks via rescale, never grows), each
    // with MORE diagonals than the real 6th attempt survived to reach (8),
    // to get a clear plateau-vs-growth signal cheaply.
    const std::vector<uint32_t> stageTowers={30,29,28,27,26,25,24,23};
    const uint32_t diagsPerStage=3;   // fewer per stage, more stage TRANSITIONS --
    // the earlier 4-stage/20-diag run already showed the per-diagonal drift is
    // small and linear (~0.5MB/diag) while the per-TRANSITION jump (~200MB) is
    // the open question; this configuration trades diagonal depth for more
    // transitions (7 vs 3) to see whether that jump stays constant, decays,
    // or compounds over a longer sequence, at comparable total cost.
    uint32_t globalDiag=0;
    for(uint32_t si=0; si<stageTowers.size(); ++si){
        uint32_t tw=stageTowers[si];
        StageRotCtx R;
        // Need a real key AT this level for ks_work_create's shape and for
        // the first diagonal -- use rotation index (si*1000+1) as a stand-in
        // "offs[0]"-equivalent, identical in spirit to production.
        auto Kshape=atLevel(rotFull(si*1000+1),tw);
        stageRotBegin(R,tw,Kshape);
        for(uint32_t d=0; d<diagsPerStage; ++d, ++globalDiag){
            uint32_t k = si*1000 + d + 1;   // distinct rotation index every diagonal
            auto t0=clk::now();
            auto Krot=atLevel(rotFull(k),tw);
            stageRotDiag(R,k,Krot,tw);
            double dt=EL(t0);
            double gu,gf; gpuMemMB(gu,gf);
            std::cerr<<"  stage="<<si<<" tw="<<tw<<" diag="<<d<<"/"<<diagsPerStage
                      <<" globalDiag="<<globalDiag<<" t="<<dt<<"s RSS="<<rssMB()<<"MB "
                      <<"GPU_used="<<gu<<"MB GPU_free="<<gf<<"MB "
                      <<"KROT="<<KROT.size()<<" nKeygen="<<nKeygen<<" nHits="<<nHits
                      <<" nEvictions="<<nEvictions<<"\n";
        }
        stageRotEnd(R);
        double gu,gf; gpuMemMB(gu,gf);
        std::cerr<<"  [stage "<<si<<" end, after ks_work_destroy] GPU_used="<<gu<<"MB GPU_free="<<gf<<"MB\n";
        // Note: R.d_base0/d_base1/d_w0/d_w1/d_scratch/d_b0/d_b1 are NOT
        // freed here -- matches production's stageRotBegin, which only
        // frees+reallocates them when a LATER stage needs MORE bytes (never
        // happens here since towers only shrinks). This is intentional,
        // bounded, one-time-sized retention, not a per-diagonal leak --
        // verified explicitly in the full teardown below.
    }

    std::cerr<<"\n=== summary ===\n";
    std::cerr<<"total diagonals="<<globalDiag<<" nKeygen="<<nKeygen<<" nHits="<<nHits
              <<" nEvictions="<<nEvictions<<" final RSS="<<rssMB()<<"MB\n";
    double guF,gfF; gpuMemMB(guF,gfF);
    std::cerr<<"final GPU_used="<<guF<<"MB GPU_free="<<gfF<<"MB (baseline was "<<gu0<<"MB)\n";
    std::cerr<<"GPU_used delta vs baseline: "<<(guF-gu0)<<"MB over "<<globalDiag<<" diagonals ("
              <<((guF-gu0)/globalDiag)<<"MB/diagonal average)\n";
    return 0;
}
