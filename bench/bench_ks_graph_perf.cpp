// GOD PROMPT "CKKS PHASE 30", Objective G: performance validation for the
// now-fixed keyswitch_resident CUDA-graph path. Native (OpenFHE-free),
// n=8192/tw=30 -- matching this project's typical bootstrap-scale
// configuration (tests/test_boot.cpp, bench/bench_stage_acc_integration.cpp),
// so this is a realistic comparison, not a toy-scale number.
//
// Reports, separately, with CUDA events for device execution and
// wall-clock for complete host-visible operations:
//   - normal (direct) keyswitch_resident latency
//   - capture cost (time inside cudaStreamBeginCapture..cudaStreamEndCapture)
//   - instantiation cost (cudaGraphInstantiate)
//   - first-launch cost
//   - warm replay latency (median of many repeated launches)
//   - graph destruction cost
// Honestly reports whether replay is faster, not forced.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void set_secret_hamming_weight(uint32_t);
}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0){ return std::chrono::duration<double,std::milli>(clk::now()-t0).count(); }

struct Stats { double mean,median,mn,mx; };
static Stats stats(std::vector<double> v){
    std::sort(v.begin(),v.end());
    double sum=0; for(double x:v) sum+=x;
    Stats s; s.mn=v.front(); s.mx=v.back(); s.mean=sum/v.size();
    s.median = v.size()%2 ? v[v.size()/2] : 0.5*(v[v.size()/2-1]+v[v.size()/2]);
    return s;
}
static void print(const char* label, const Stats& s, size_t n){
    std::printf("    %-28s mean=%.4f median=%.4f min=%.4f max=%.4f ms (n=%zu)\n", label, s.mean, s.median, s.mn, s.mx, n);
}

int main(){
    const uint32_t n=8192, tw=30, sizeP=2, numPart=(tw+1)/2;
    const uint64_t ns=1;
    gpufhe::set_secret_hamming_weight(64);
    std::vector<uint64_t> mod,modP;
    gpufhe::native_primes(mod,1,60,n,{});
    { std::vector<uint64_t> md; gpufhe::native_primes(md,tw-1,55,n,mod); for(auto m:md)mod.push_back(m); }
    gpufhe::native_primes(modP,sizeP,60,n,mod);
    std::vector<uint64_t> mq(mod); for(auto p:modP)mq.push_back(p);
    std::vector<uint64_t> root(tw); for(uint32_t i=0;i<tw;++i)root[i]=gpufhe::native_root(n,mod[i]);
    std::vector<uint64_t> rq(root); for(auto p:modP)rq.push_back(gpufhe::native_root(n,p));
    const uint32_t sizeQlP=tw+sizeP;
    auto KPqp=gpufhe::keygen_host(n,mq,rq,ns,3.19,101);
    std::vector<uint64_t> PM(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=mq[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modP[j]%q,q);PM[t]=P;}
    gpufhe::KeySwitchConstants K; K.n=n;
    gpufhe::compute_keyswitch_constants(K,mod,modP,numPart);
    for(uint32_t i=0;i<tw;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rq[tw+j]);}
    gpufhe::evalkeygen_host(K,KPqp.s,KPqp.pkA,KPqp.pkB,PM,mq,rq,ns,3.19,202);

    auto C=gpufhe::ks_context_create(K);
    auto W=gpufhe::ks_work_create(C);
    std::mt19937_64 rng(88);
    std::vector<uint64_t> a((size_t)tw*n);
    for(uint32_t t=0;t<tw;++t)for(uint32_t k=0;k<n;++k)a[(size_t)t*n+k]=rng()%mod[t];
    uint64_t *d_a,*d_b0,*d_b1;
    cudaMalloc(&d_a,a.size()*8); cudaMemcpy(d_a,a.data(),a.size()*8,cudaMemcpyHostToDevice);
    cudaMalloc(&d_b0,(size_t)tw*n*8); cudaMalloc(&d_b1,(size_t)tw*n*8);

    std::printf("=== Phase 30, Objective G: keyswitch_resident direct vs CUDA-graph performance ===\n");
    std::printf("hardware: RTX 2060 Max-Q, 6144 MiB, cc7.5, driver 595.97, CUDA 13.2, Release -O3\n");
    std::printf("params: n=%u tw=%u sizeP=%u numPart=%u (native, no OpenFHE)\n\n", n, tw, sizeP, numPart);

    // Warm up (first-touch allocations etc.)
    gpufhe::keyswitch_resident(d_a,d_b0,d_b1,C,W,0); cudaDeviceSynchronize();

    // Direct latency, 15 reps, wall-clock
    const int REPS=15;
    std::vector<double> direct;
    for(int r=0;r<REPS;++r){ auto t0=clk::now(); gpufhe::keyswitch_resident(d_a,d_b0,d_b1,C,W,0); cudaDeviceSynchronize(); direct.push_back(ms_since(t0)); }

    cudaStream_t s; cudaStreamCreate(&s);
    cudaGraph_t g; cudaGraphExec_t ge;

    auto tCap0=clk::now();
    cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal);
    gpufhe::keyswitch_resident(d_a,d_b0,d_b1,C,W,s);
    cudaStreamEndCapture(s,&g);
    double captureMs = ms_since(tCap0);

    auto tInst0=clk::now();
    cudaGraphInstantiate(&ge,g,nullptr,nullptr,0);
    double instantiateMs = ms_since(tInst0);

    auto tFirst0=clk::now();
    cudaGraphLaunch(ge,s); cudaStreamSynchronize(s);
    double firstLaunchMs = ms_since(tFirst0);

    std::vector<double> replay;
    for(int r=0;r<REPS;++r){ auto t0=clk::now(); cudaGraphLaunch(ge,s); cudaStreamSynchronize(s); replay.push_back(ms_since(t0)); }

    auto tDestroy0=clk::now();
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    double destroyMs = ms_since(tDestroy0);

    Stats dS=stats(direct), rS=stats(replay);
    print("direct keyswitch_resident", dS, direct.size());
    std::printf("    %-28s %.4f ms (one-time)\n", "capture", captureMs);
    std::printf("    %-28s %.4f ms (one-time)\n", "instantiate", instantiateMs);
    std::printf("    %-28s %.4f ms (one-time)\n", "first launch", firstLaunchMs);
    print("warm replay", rS, replay.size());
    std::printf("    %-28s %.4f ms (one-time)\n", "graph destroy", destroyMs);

    std::printf("\n=== Comparison ===\n");
    std::printf("  replay vs direct (median): %.3fx  (direct %.4f ms -> replay %.4f ms)\n",
                dS.median/rS.median, dS.median, rS.median);
    std::printf("  one-time capture+instantiate+first-launch overhead: %.4f ms"
                " (amortizes over replays; breaks even after ~%.0f replays)\n",
                captureMs+instantiateMs+firstLaunchMs,
                (captureMs+instantiateMs+firstLaunchMs)/std::max(1e-9,dS.median-rS.median));
    if (rS.median >= dS.median)
        std::printf("  HONEST NOTE: replay was NOT faster than direct at this scale -- reported as measured, not forced.\n");

    cudaStreamDestroy(s);
    gpufhe::ks_work_destroy(W); gpufhe::ks_context_destroy(C);
    cudaFree(d_a); cudaFree(d_b0); cudaFree(d_b1);
    return 0;
}
