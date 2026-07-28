// KEY REUSE ACROSS LEVELS: one relin key generated ONCE over full QP, then used
// at every level via the eval-key index offset delta = fullQ - sizeQl.
// Every prior test rebuilt a key per level, so delta was ALWAYS 0 and the offset
// path (idx = i+delta for the P towers) has never been exercised. Reuse is
// mandatory for a bootstrap benchmark and for n=32768 (keys are ~250MB there).
// alpha is fixed at 2 towers/part so part boundaries agree across levels;
// at level tw only the first ceil(tw/2) parts of the key are used.
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
}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
static uint64_t am(uint64_t a,uint64_t b,uint64_t q){uint64_t s=a+b;return s>=q?s-q:s;}
using cd=std::complex<double>;

int main(){
    const uint32_t n=1024,S=n/2,sizeQF=15,sizeP=2; const uint64_t ns=1;
    auto npFor=[](uint32_t tw)->uint32_t{ return tw<=2?1u:(tw+1)/2; };   // alpha = 2
    std::vector<uint64_t> mod,modP;
    gpufhe::native_primes(mod,1,60,n,{});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids,sizeQF-1,50,n,mod); for(auto m:mids)mod.push_back(m); }
    gpufhe::native_primes(modP,sizeP,60,n,mod);
    std::vector<uint64_t> modQP=mod; for(auto p:modP)modQP.push_back(p);
    std::vector<uint64_t> root(sizeQF),rootQP;
    for(uint32_t i=0;i<sizeQF;++i)root[i]=gpufhe::native_root(n,mod[i]);
    rootQP=root; for(auto p:modP)rootQP.push_back(gpufhe::native_root(n,p));
    const uint32_t sizeQlPF=sizeQF+sizeP;
    auto KPqp=gpufhe::keygen_host(n,modQP,rootQP,ns,3.19,101);
    std::vector<uint64_t> PModq(sizeQlPF);
    for(uint32_t t=0;t<sizeQlPF;++t){uint64_t q=modQP[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modP[j]%q,q);PModq[t]=P;}

    // ---- ONE relin key over full QP
    gpufhe::KeySwitchConstants KF; KF.n=n;
    gpufhe::compute_keyswitch_constants(KF,mod,modP,npFor(sizeQF));
    for(uint32_t i=0;i<sizeQF;++i){KF.rootModList.push_back(mod[i]);KF.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){KF.rootModList.push_back(modP[j]);KF.rootValList.push_back(rootQP[sizeQF+j]);}
    gpufhe::evalkeygen_host(KF,KPqp.s,KPqp.pkA,KPqp.pkB,PModq,modQP,rootQP,ns,3.19,202);
    std::cout<<"full-QP relin key: parts="<<KF.numPart<<" evalKeyTowers="<<KF.evalKeyTowers<<"\n";

    const double Delta=std::pow(2.0,50);
    std::vector<cd> xin(S);
    for(uint32_t i=0;i<S;++i) xin[i]={0.6*std::sin(0.02*i),0};

    uint32_t fails=0;
    for(uint32_t tw : {15u,13u,11u,9u,7u,5u,3u}){
        std::vector<uint64_t> ml(mod.begin(),mod.begin()+tw),rl(root.begin(),root.begin()+tw);
        // level constants, but the SAME key, with delta = sizeQF - tw
        gpufhe::KeySwitchConstants K; K.n=n;
        gpufhe::compute_keyswitch_constants(K,ml,modP,npFor(tw));
        for(uint32_t i=0;i<tw;++i){K.rootModList.push_back(ml[i]);K.rootValList.push_back(rl[i]);}
        for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rootQP[sizeQF+j]);}
        K.fullQ=sizeQF;                      // <-- delta = fullQ - sizeQl becomes NONZERO
        K.evalKeyTowers=KF.evalKeyTowers;
        K.av.assign(K.numPart,{}); K.bv.assign(K.numPart,{});
        for(uint32_t p=0;p<K.numPart;++p){ K.av[p]=KF.av[p]; K.bv[p]=KF.bv[p]; }

        auto KPq=gpufhe::keygen_host(n,ml,rl,ns,3.19,101);
        std::vector<int64_t> mx; gpufhe::encode_host(mx,xin,n,Delta);
        std::vector<uint64_t> c0,c1;
        gpufhe::encrypt_host(c0,c1,mx,KPq.pkA,KPq.pkB,n,ml,rl,ns,3.19,303);
        size_t T=(size_t)tw*n;
        std::vector<uint64_t> t0(T),t1(T),t2(T);
        for(uint32_t t=0;t<tw;++t){uint64_t q=ml[t];
            for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
                t0[x]=mm(c0[x],c0[x],q); t1[x]=am(mm(c0[x],c1[x],q),mm(c1[x],c0[x],q),q); t2[x]=mm(c1[x],c1[x],q);}}
        auto R=gpufhe::keyswitch_core_resident(t2,K);
        std::vector<uint64_t> r0(T),r1(T);
        for(uint32_t t=0;t<tw;++t){uint64_t q=ml[t];
            for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k; r0[x]=am(t0[x],R.ba0[x],q); r1[x]=am(t1[x],R.ba1[x],q);}}
        auto C=gpufhe::ks_context_create(K);
        std::vector<uint64_t> s1,s2; gpufhe::native_rescale_consts(s1,s2,ml,tw-1);
        uint64_t *d0,*d1,*sc,*dp;
        cudaMalloc(&d0,T*8);cudaMalloc(&d1,T*8);cudaMalloc(&sc,(size_t)n*8);cudaMalloc(&dp,(size_t)n*8);
        cudaMemcpy(d0,r0.data(),T*8,cudaMemcpyHostToDevice);cudaMemcpy(d1,r1.data(),T*8,cudaMemcpyHostToDevice);
        gpufhe::rescale_resident_raw(d0,tw,C,s1,s2,sc,dp,0);
        gpufhe::rescale_resident_raw(d1,tw,C,s1,s2,sc,dp,0);
        cudaDeviceSynchronize();
        std::vector<uint64_t> h0(T),h1(T);
        cudaMemcpy(h0.data(),d0,T*8,cudaMemcpyDeviceToHost);cudaMemcpy(h1.data(),d1,T*8,cudaMemcpyDeviceToHost);
        cudaFree(d0);cudaFree(d1);cudaFree(sc);cudaFree(dp);gpufhe::ks_context_destroy(C);
        uint32_t nt=tw-1; size_t T2=(size_t)nt*n;
        std::vector<uint64_t> o0(h0.begin(),h0.begin()+T2),o1(h1.begin(),h1.begin()+T2);
        std::vector<uint64_t> mf(ml.begin(),ml.begin()+nt),rf(rl.begin(),rl.begin()+nt);
        auto KPr=gpufhe::keygen_host(n,mf,rf,ns,3.19,101);
        std::vector<int64_t> dec; gpufhe::decrypt_host(dec,o0,o1,KPr.s,n,mf,rf);
        std::vector<cd> y; gpufhe::decode_host(y,dec,n,Delta*Delta/(double)ml[tw-1]);
        double e=0; for(uint32_t i=0;i<S;++i){double x=xin[i].real(); e=std::max(e,std::abs(y[i].real()-x*x));}
        bool ok=e<1e-3; if(!ok)++fails;
        std::cout<<"tw="<<tw<<" parts="<<K.numPart<<" delta="<<(sizeQF-tw)
                 <<"  x^2 err="<<e<<(ok?"   OK":"   <-- BAD")<<"\n"<<std::flush;
    }
    if(fails==0){std::cout<<"[PASS] one full-QP key reused at every level (delta offset works)\n";return 0;}
    std::cout<<"[FAIL] "<<fails<<" levels wrong\n"; return 1;
}
