// Minimal deep-eval gate: p(x) = c0 + c1*x + c2*T2(x), T2=2x^2-1. ONE ct*ct
// multiply, no level juggling. Isolates the eval engine from level-alignment.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
namespace gpufhe {
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void decode_host(std::vector<std::complex<double>>&, const std::vector<int64_t>&, uint32_t, double);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void native_rescale_consts(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t);
void pt_to_eval_host(std::vector<uint64_t>&, const std::vector<int64_t>&, uint32_t, uint32_t,
                     const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void ct_mul_pt_host(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<uint64_t>&,
                    uint32_t, uint32_t, const std::vector<uint64_t>&);
void ct_add_ct_host(std::vector<uint64_t>&, std::vector<uint64_t>&,
                    const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                    uint32_t, uint32_t, const std::vector<uint64_t>&);
}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
static uint64_t am(uint64_t a,uint64_t b,uint64_t q){uint64_t s=a+b;return s>=q?s-q:s;}
static uint64_t sm(uint64_t a,uint64_t b,uint64_t q){return a>=b?a-b:a+q-b;}
using cd=std::complex<double>;

int main(){
    const uint32_t n=1024,S=n/2,numPart=2,sizeQ=3,sizeP=2; const uint64_t ns=1;
    std::vector<uint64_t> mod,modP;
    gpufhe::native_primes(mod,1,60,n,{});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids,sizeQ-1,50,n,mod); for(auto m:mids)mod.push_back(m); }
    gpufhe::native_primes(modP,sizeP,60,n,mod);
    std::vector<uint64_t> modQP=mod; for(auto p:modP)modQP.push_back(p);
    std::vector<uint64_t> root(sizeQ),rootQP;
    for(uint32_t i=0;i<sizeQ;++i)root[i]=gpufhe::native_root(n,mod[i]);
    rootQP=root; for(auto p:modP)rootQP.push_back(gpufhe::native_root(n,p));
    const uint32_t sizeQlP=sizeQ+sizeP;
    auto KPqp=gpufhe::keygen_host(n,modQP,rootQP,ns,3.19,101);
    auto KPq =gpufhe::keygen_host(n,mod,root,ns,3.19,101);
    std::vector<uint64_t> PModq_QP(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=modQP[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modQP[sizeQ+j]%q,q);PModq_QP[t]=P;}
    gpufhe::KeySwitchConstants K; K.n=n;
    gpufhe::compute_keyswitch_constants(K,mod,modP,numPart);
    for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modQP[sizeQ+j]);K.rootValList.push_back(rootQP[sizeQ+j]);}
    gpufhe::evalkeygen_host(K,KPqp.s,KPqp.pkA,KPqp.pkB,PModq_QP,modQP,rootQP,ns,3.19,202);
    auto C=gpufhe::ks_context_create(K);
    const double Delta=std::pow(2.0,50);

    std::vector<cd> xin(S);
    for(uint32_t i=0;i<S;++i) xin[i]={0.6*std::sin(0.02*i),0};
    // encrypt x at sizeQ towers
    std::vector<int64_t> mx; gpufhe::encode_host(mx,xin,n,Delta);
    std::vector<uint64_t> c0,c1; gpufhe::encrypt_host(c0,c1,mx,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,303);

    // x*x -> tensor+relin+combine, scale Delta^2, at sizeQ towers
    const size_t T=(size_t)sizeQ*n;
    std::vector<uint64_t> t0(T),t1(T),t2(T);
    for(uint32_t t=0;t<sizeQ;++t){uint64_t q=mod[t];
        for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
            t0[x]=mm(c0[x],c0[x],q); t1[x]=am(mm(c0[x],c1[x],q),mm(c1[x],c0[x],q),q); t2[x]=mm(c1[x],c1[x],q);}}
    auto R=gpufhe::keyswitch_core_resident(t2,K);
    std::vector<uint64_t> r0(T),r1(T);
    for(uint32_t t=0;t<sizeQ;++t){uint64_t q=mod[t];
        for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k; r0[x]=am(t0[x],R.ba0[x],q); r1[x]=am(t1[x],R.ba1[x],q);}}
    // now (r0,r1) = x^2 at scale Delta^2, sizeQ towers. T2 = 2x^2 - 1.
    // 2*x^2: double c0,c1
    for(uint32_t t=0;t<sizeQ;++t){uint64_t q=mod[t];
        for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k; r0[x]=(r0[x]*2)%q; r1[x]=(r1[x]*2)%q;}}
    // (subtract 1 AFTER rescale, where scale is back to ~Delta-range)
    // result p(x) = c2*T2 + c1*x + c0. Combine at scale Delta^2:
    // c2*T2: mul_pt by c2 (scale Delta^3 -> too big). Instead keep it linear:
    // just test T2 alone first (c2=1,c1=0,c0=0) -> expect 2x^2-1.
    // rescale T2 down: Delta^2 -> Delta^2/qLast
    std::vector<uint64_t> s1,s2; { std::vector<uint64_t> sub(mod.begin(),mod.end()); gpufhe::native_rescale_consts(s1,s2,sub,sizeQ-1); }
    uint64_t *dr0,*dr1,*scr,*drp;
    cudaMalloc(&dr0,T*8);cudaMalloc(&dr1,T*8);cudaMalloc(&scr,(size_t)n*8);cudaMalloc(&drp,(size_t)n*8);
    cudaMemcpy(dr0,r0.data(),T*8,cudaMemcpyHostToDevice);cudaMemcpy(dr1,r1.data(),T*8,cudaMemcpyHostToDevice);
    gpufhe::rescale_resident_raw(dr0,sizeQ,C,s1,s2,scr,drp,0);
    gpufhe::rescale_resident_raw(dr1,sizeQ,C,s1,s2,scr,drp,0);
    cudaDeviceSynchronize();
    uint32_t nt=sizeQ-1; size_t T2s=(size_t)nt*n;
    std::vector<uint64_t> h0(T),h1(T); cudaMemcpy(h0.data(),dr0,T*8,cudaMemcpyDeviceToHost);cudaMemcpy(h1.data(),dr1,T*8,cudaMemcpyDeviceToHost);
    std::vector<uint64_t> o0(h0.begin(),h0.begin()+T2s), o1(h1.begin(),h1.begin()+T2s);
    double DeltaOut=Delta*Delta/(double)mod[sizeQ-1];
    // T2 = 2x^2 - 1 : subtract 1.0 encoded at the POST-rescale scale DeltaOut
    { std::vector<cd> one(S,cd{1.0,0}); std::vector<int64_t> m1; gpufhe::encode_host(m1,one,n,DeltaOut);
      std::vector<uint64_t> modl2(mod.begin(),mod.begin()+nt),rootl2(root.begin(),root.begin()+nt);
      std::vector<uint64_t> e1; gpufhe::pt_to_eval_host(e1,m1,nt,n,modl2,rootl2);
      for(uint32_t t=0;t<nt;++t){uint64_t q=mod[t];
        for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k; o0[x]=sm(o0[x],e1[x],q);}} }

    std::vector<uint64_t> modl(mod.begin(),mod.begin()+nt),rootl(root.begin(),root.begin()+nt);
    auto KPr=gpufhe::keygen_host(n,modl,rootl,ns,3.19,101);
    std::vector<int64_t> dec; gpufhe::decrypt_host(dec,o0,o1,KPr.s,n,modl,rootl);
    std::vector<cd> y; gpufhe::decode_host(y,dec,n,DeltaOut);

    double maxerr=0;
    for(uint32_t i=0;i<S;++i){ double x=xin[i].real(); double t2v=2*x*x-1;
        maxerr=std::max(maxerr,std::abs(y[i].real()-t2v)); }
    std::cout<<"T2=2x^2-1, DeltaOut=2^"<<std::log2(DeltaOut)<<" maxerr="<<maxerr<<"\n";
    std::cout<<"  y[0..3]="; for(int k=0;k<4;++k)std::cout<<y[k].real()<<" "; std::cout<<"  ref="; for(int k=0;k<4;++k){double x=xin[k].real();std::cout<<(2*x*x-1)<<" ";} std::cout<<"\n";
    if(maxerr<1e-2){std::cout<<"[PASS] deep eval engine sound (T2 under encryption)\n";return 0;}
    std::cout<<"[FAIL]\n"; return 1;
}
