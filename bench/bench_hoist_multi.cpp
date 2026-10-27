// Benchmark: does sharing one decompose(c1) across N rotation indices
// (tests/test_hoist_multi.cpp's proven-correct mechanism) actually beat
// doing N independent full keyswitches (today's rotate_ct_resident,
// i.e. N full decomposes)? Per the governing directive: "keep only if
// end-to-end results support it" -- this measures that, interleaved,
// before any claim of a win.
//
// WSL can't lock GPU clocks (thermal drift is real and measured this
// session) -- alternate method A/B every rep WITHIN one process so
// thermal drift is common-mode and cancels in the paired difference,
// same statistical approach as bench_resident_chain.cpp (median of
// per-rep A-B difference, bootstrap CI), rather than trusting either
// method's raw mean.
//
// N=5 is a representative BSGS inner-loop width (a typical giant-step
// group size in this codebase's C2S/S2C linear transforms) -- the
// number of rotation indices that would plausibly be processed against
// the SAME base ciphertext within one hoisting opportunity.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keyswitch_hoist.h"
#include "keygen.h"
#include "stage_acc.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <vector>
namespace gpufhe {
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void automorphism_eval_host(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t,
                            const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void automorphism_eval_fast_device(uint64_t*, uint32_t, uint32_t, uint32_t, uint64_t*, cudaStream_t);
void set_secret_hamming_weight(uint32_t);
}
using clk = std::chrono::steady_clock;
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
static uint64_t modinv(uint64_t a, uint64_t m){
    int64_t old_r=(int64_t)a, r=(int64_t)m, old_s=1, s=0;
    while(r!=0){ int64_t q=old_r/r; int64_t tr=old_r-q*r; old_r=r; r=tr;
        int64_t ts=old_s-q*s; old_s=s; s=ts; }
    int64_t res=old_s % (int64_t)m; if(res<0) res+=(int64_t)m; return (uint64_t)res;
}
static double median(std::vector<double> v){
    if(v.empty())return 0; std::sort(v.begin(),v.end());
    size_t n=v.size(); return n%2? v[n/2] : 0.5*(v[n/2-1]+v[n/2]);
}
static double quantile(std::vector<double> v,double q){
    if(v.empty())return 0; std::sort(v.begin(),v.end());
    double pos=q*(v.size()-1); size_t lo=(size_t)pos, hi=std::min(lo+1,v.size()-1);
    return v[lo]+(pos-lo)*(v[hi]-v[lo]);
}
static std::pair<double,double> bootstrap(const std::vector<double>&d,int it=4000){
    if(d.size()<3)return{0,0}; std::mt19937 rng(7);
    std::uniform_int_distribution<size_t> pick(0,d.size()-1);
    std::vector<double> meds; meds.reserve(it); std::vector<double> s(d.size());
    for(int k=0;k<it;++k){ for(size_t i=0;i<d.size();++i)s[i]=d[pick(rng)]; meds.push_back(median(s)); }
    return {quantile(meds,0.025),quantile(meds,0.975)};
}

int main(){
    const uint32_t n=8192, tw=30, sizeP=2, numPart=(tw+1)/2; const uint64_t ns=1;
    gpufhe::set_secret_hamming_weight(64);
    std::vector<uint64_t> mod,modP;
    gpufhe::native_primes(mod,1,60,n,{});
    { std::vector<uint64_t> md; gpufhe::native_primes(md,tw-1,55,n,mod); for(auto m:md)mod.push_back(m); }
    gpufhe::native_primes(modP,sizeP,60,n,mod);
    std::vector<uint64_t> mq(mod),rq;
    std::vector<uint64_t> root(tw); for(uint32_t i=0;i<tw;++i)root[i]=gpufhe::native_root(n,mod[i]);
    rq=root; for(auto p:modP){ mq.push_back(p); rq.push_back(gpufhe::native_root(n,p)); }
    const uint32_t sizeQlP=tw+sizeP;

    auto KPqp=gpufhe::keygen_host(n,mq,rq,ns,3.19,101);
    auto KPq =gpufhe::keygen_host(n,mod,root,ns,3.19,101);
    std::vector<uint64_t> PM(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=mq[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modP[j]%q,q);PM[t]=P;}

    std::vector<std::complex<double>> z(n/2); for(uint32_t i=0;i<n/2;++i) z[i]={0.3,0};
    std::vector<int64_t> mz; gpufhe::encode_host(mz,z,n,std::pow(2.0,55));
    std::vector<uint64_t> c0,c1;
    gpufhe::encrypt_host(c0,c1,mz,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,303);

    const std::vector<uint32_t> ks = {5u,7u,11u,13u,99u};   // N=5
    std::vector<gpufhe::KeySwitchConstants> K(ks.size());
    for(size_t ki=0; ki<ks.size(); ++ki){
        uint32_t k=ks[ki]; uint64_t kinv=modinv(k,2*n);
        std::vector<uint64_t> pkA_h=KPqp.pkA, pkB_h=KPqp.pkB;
        gpufhe::automorphism_eval_host(pkA_h,sizeQlP,n,(uint32_t)kinv,mq,rq);
        gpufhe::automorphism_eval_host(pkB_h,sizeQlP,n,(uint32_t)kinv,mq,rq);
        K[ki].n=n;
        gpufhe::compute_keyswitch_constants(K[ki],mod,modP,numPart);
        for(uint32_t i=0;i<tw;++i){K[ki].rootModList.push_back(mod[i]);K[ki].rootValList.push_back(root[i]);}
        for(uint32_t j=0;j<sizeP;++j){K[ki].rootModList.push_back(modP[j]);K[ki].rootValList.push_back(rq[tw+j]);}
        gpufhe::evalkeygen_host_sold(K[ki],KPqp.s,KPqp.s,pkA_h,pkB_h,PM,mq,rq,1,3.19,900+(uint64_t)k);
    }
    // KROT/KSC in a real bootstrap already hold these keys resident -- build
    // every DeviceKSContext up front, OUTSIDE the timed region, for both
    // methods (this benchmark is about decompose cost, not key upload cost).
    std::vector<gpufhe::DeviceKSContext> C(ks.size());
    for(size_t ki=0; ki<ks.size(); ++ki) C[ki]=gpufhe::ks_context_create(K[ki]);

    size_t T=(size_t)tw*n, B=T*8, TQP=(size_t)sizeQlP*n, BQP=TQP*8;
    uint64_t *d_c0,*d_c1,*d_scratch_qp,*d_ba0,*d_ba1;
    cudaMalloc(&d_c0,B); cudaMalloc(&d_c1,B); cudaMalloc(&d_scratch_qp,BQP);
    cudaMalloc(&d_ba0,B); cudaMalloc(&d_ba1,B);
    cudaMemcpy(d_c0,c0.data(),B,cudaMemcpyHostToDevice);
    cudaMemcpy(d_c1,c1.data(),B,cudaMemcpyHostToDevice);

    auto run_baseline=[&](){   // N full keyswitches: today's rotate_ct_resident-equivalent cost
        double total=0;
        for(size_t ki=0; ki<ks.size(); ++ki){
            auto W=gpufhe::ks_work_create(C[ki]);
            cudaDeviceSynchronize();
            auto t0=clk::now();
            gpufhe::ks_accumulate_qp(d_c1,C[ki],W,0);
            for(uint32_t t=0;t<tw;++t)
                LaunchScaledAddInto(W.d_res0+(size_t)t*n, d_c0+(size_t)t*n, PM[t], mod[t], n, 0);
            gpufhe::automorphism_eval_fast_device(W.d_res0,sizeQlP,n,ks[ki],d_scratch_qp,0);
            gpufhe::automorphism_eval_fast_device(W.d_res1,sizeQlP,n,ks[ki],d_scratch_qp,0);
            gpufhe::ks_moddown(C[ki],W,W.d_res0,d_ba0,0);
            gpufhe::ks_moddown(C[ki],W,W.d_res1,d_ba1,0);
            cudaStreamSynchronize(0);
            total+=std::chrono::duration<double,std::milli>(clk::now()-t0).count();
            gpufhe::ks_work_destroy(W);
        }
        return total;
    };
    auto run_hoisted=[&](){   // decompose ONCE, N cheap accumulates
        auto W=gpufhe::ks_work_create(C[0]);
        auto D=gpufhe::hoist_digits_create(C[0]);
        cudaDeviceSynchronize();
        auto t0=clk::now();
        gpufhe::ks_decompose_only(d_c1,C[0],D,0);
        for(size_t ki=0; ki<ks.size(); ++ki){
            gpufhe::ks_accumulate_from_digits(d_c1,D,C[ki],W,0);
            for(uint32_t t=0;t<tw;++t)
                LaunchScaledAddInto(W.d_res0+(size_t)t*n, d_c0+(size_t)t*n, PM[t], mod[t], n, 0);
            gpufhe::automorphism_eval_fast_device(W.d_res0,sizeQlP,n,ks[ki],d_scratch_qp,0);
            gpufhe::automorphism_eval_fast_device(W.d_res1,sizeQlP,n,ks[ki],d_scratch_qp,0);
            gpufhe::ks_moddown(C[ki],W,W.d_res0,d_ba0,0);
            gpufhe::ks_moddown(C[ki],W,W.d_res1,d_ba1,0);
        }
        cudaStreamSynchronize(0);
        double total=std::chrono::duration<double,std::milli>(clk::now()-t0).count();
        gpufhe::hoist_digits_destroy(D);
        gpufhe::ks_work_destroy(W);
        return total;
    };

    const int warmup=5, reps=30;
    for(int i=0;i<warmup;++i){ run_baseline(); run_hoisted(); }
    std::vector<double> ta, th, diff;
    for(int i=0;i<reps;++i){
        double a=run_baseline(), h=run_hoisted();   // interleaved per-rep, baseline first each time
        ta.push_back(a); th.push_back(h); diff.push_back(a-h);
        std::cout<<"  rep "<<i<<": baseline="<<a<<"ms hoisted="<<h<<"ms diff="<<(a-h)<<"ms\n";
    }
    auto ci=bootstrap(diff);
    std::cout<<"\nbaseline (N="<<ks.size()<<" full keyswitches) median: "<<median(ta)<<"ms"
              <<" [p25="<<quantile(ta,0.25)<<", p75="<<quantile(ta,0.75)<<"]\n";
    std::cout<<"hoisted  (1 decompose + N cheap accumulates) median: "<<median(th)<<"ms"
              <<" [p25="<<quantile(th,0.25)<<", p75="<<quantile(th,0.75)<<"]\n";
    std::cout<<"paired diff (baseline-hoisted) median: "<<median(diff)<<"ms"
              <<" 95% CI [bootstrap]: ["<<ci.first<<", "<<ci.second<<"]\n";
    bool clearWin = ci.first>0;   // CI entirely positive => hoisted reliably faster
    std::cout<<(clearWin? "[RESULT] hoisted path is a measurable win (CI excludes zero)\n"
                         : "[RESULT] NOT a clearly measurable win at this N/scale (CI includes zero or is negative) -- do not report a speedup claim from this run\n");
    return 0;
}
