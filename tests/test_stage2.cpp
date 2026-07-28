// TWO ISOLATED PROBES across tw=2..15:
//  RELIN  : tensor+keyswitch, decrypted PRE-rescale at Delta^2 (Delta=2^25 so
//           Delta^2=2^50 < q0/2). No rescale involved at all.
//  RESCALE: fresh encryption, NO keyswitch, last tower is a SMALL 20-bit prime
//           so post-rescale scale = 2^45/2^20 = 2^25 stays precise. Run with
//           np=2 AND np=tw: if they differ, rescale wrongly depends on numPart.
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
    const uint32_t n=1024,S=n/2,sizeP=2; const uint64_t ns=1;
    const uint32_t MAXQ=15;
    std::vector<uint64_t> modAll,modP,smallP;
    gpufhe::native_primes(modAll,1,60,n,{});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids,MAXQ-1,50,n,modAll); for(auto m:mids)modAll.push_back(m); }
    gpufhe::native_primes(modP,sizeP,60,n,modAll);
    gpufhe::native_primes(smallP,1,20,n,modAll);
    std::vector<uint64_t> rootAll(MAXQ),rootP(sizeP);
    for(uint32_t i=0;i<MAXQ;++i)rootAll[i]=gpufhe::native_root(n,modAll[i]);
    for(uint32_t j=0;j<sizeP;++j)rootP[j]=gpufhe::native_root(n,modP[j]);
    uint64_t qSmall=smallP[0], rSmall=gpufhe::native_root(n,qSmall);

    std::vector<cd> xin(S);
    for(uint32_t i=0;i<S;++i) xin[i]={0.5*std::sin(0.02*i),0};

    auto mkK=[&](const std::vector<uint64_t>&modl,const std::vector<uint64_t>&rootl,uint32_t np,bool withKey)
              ->gpufhe::KeySwitchConstants{
        uint32_t tw=modl.size();
        gpufhe::KeySwitchConstants K; K.n=n;
        gpufhe::compute_keyswitch_constants(K,modl,modP,np);
        for(uint32_t i=0;i<tw;++i){K.rootModList.push_back(modl[i]);K.rootValList.push_back(rootl[i]);}
        for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rootP[j]);}
        if(withKey){
            std::vector<uint64_t> modQPl(modl); for(auto p:modP)modQPl.push_back(p);
            std::vector<uint64_t> rootQPl(rootl); for(auto r:rootP)rootQPl.push_back(r);
            auto KPqp=gpufhe::keygen_host(n,modQPl,rootQPl,ns,3.19,101);
            std::vector<uint64_t> PModq(tw+sizeP);
            for(uint32_t t=0;t<tw+sizeP;++t){uint64_t q=modQPl[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modP[j]%q,q);PModq[t]=P;}
            gpufhe::evalkeygen_host(K,KPqp.s,KPqp.pkA,KPqp.pkB,PModq,modQPl,rootQPl,ns,3.19,202);
        } else {
            K.av.assign(K.numPart,{});K.bv.assign(K.numPart,{});K.evalKeyTowers=tw+sizeP;
            for(uint32_t p=0;p<K.numPart;++p){K.av[p].assign((size_t)(tw+sizeP)*n,0);K.bv[p].assign((size_t)(tw+sizeP)*n,0);}
        }
        return K; };

    std::cout<<"tw  RELIN(pre-rescale)      RESCALE np=2         RESCALE np=tw\n";
    for(uint32_t tw=2;tw<=MAXQ;++tw){
        // ---- RELIN probe
        double eRel=0;
        { const double D=std::pow(2.0,25);
          std::vector<uint64_t> modl(modAll.begin(),modAll.begin()+tw),rootl(rootAll.begin(),rootAll.begin()+tw);
          auto K=mkK(modl,rootl,(tw+1)/2,true);
          auto KPq=gpufhe::keygen_host(n,modl,rootl,ns,3.19,101);
          std::vector<int64_t> mx; gpufhe::encode_host(mx,xin,n,D);
          std::vector<uint64_t> c0,c1; gpufhe::encrypt_host(c0,c1,mx,KPq.pkA,KPq.pkB,n,modl,rootl,ns,3.19,303);
          size_t T=(size_t)tw*n;
          std::vector<uint64_t> t0(T),t1(T),t2(T);
          for(uint32_t t=0;t<tw;++t){uint64_t q=modl[t];
              for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
                  t0[x]=mm(c0[x],c0[x],q); t1[x]=am(mm(c0[x],c1[x],q),mm(c1[x],c0[x],q),q); t2[x]=mm(c1[x],c1[x],q);}}
          cudaGetLastError();
          auto R=gpufhe::keyswitch_core_resident(t2,K);
          cudaDeviceSynchronize();
          { cudaError_t ce=cudaGetLastError();
            if(ce!=cudaSuccess) printf("  [CUDA ERR tw=%u] %s\n",tw,cudaGetErrorString(ce)); }
          std::vector<uint64_t> r0(T),r1(T);
          for(uint32_t t=0;t<tw;++t){uint64_t q=modl[t];
              for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k; r0[x]=am(t0[x],R.ba0[x],q); r1[x]=am(t1[x],R.ba1[x],q);}}
          std::vector<int64_t> d; gpufhe::decrypt_host(d,r0,r1,KPq.s,n,modl,rootl);
          std::vector<cd> v; gpufhe::decode_host(v,d,n,D*D);
          for(uint32_t i=0;i<S;++i){double x=xin[i].real(); eRel=std::max(eRel,std::abs(v[i].real()-x*x));} }

        // ---- RESCALE probe (no keyswitch), small last prime, two np values
        auto rescProbe=[&](uint32_t np)->double{
            const double D=std::pow(2.0,45);
            std::vector<uint64_t> modl(modAll.begin(),modAll.begin()+(tw-1)),rootl(rootAll.begin(),rootAll.begin()+(tw-1));
            modl.push_back(qSmall); rootl.push_back(rSmall);
            auto K=mkK(modl,rootl,np,false);
            auto C=gpufhe::ks_context_create(K);
            auto KPq=gpufhe::keygen_host(n,modl,rootl,ns,3.19,101);
            std::vector<int64_t> mx; gpufhe::encode_host(mx,xin,n,D);
            std::vector<uint64_t> c0,c1; gpufhe::encrypt_host(c0,c1,mx,KPq.pkA,KPq.pkB,n,modl,rootl,ns,3.19,303);
            std::vector<uint64_t> s1,s2; gpufhe::native_rescale_consts(s1,s2,modl,tw-1);
            size_t T=(size_t)tw*n; uint64_t *d0,*d1,*scr,*drp;
            cudaMalloc(&d0,T*8);cudaMalloc(&d1,T*8);cudaMalloc(&scr,(size_t)n*8);cudaMalloc(&drp,(size_t)n*8);
            cudaMemcpy(d0,c0.data(),T*8,cudaMemcpyHostToDevice);cudaMemcpy(d1,c1.data(),T*8,cudaMemcpyHostToDevice);
            gpufhe::rescale_resident_raw(d0,tw,C,s1,s2,scr,drp,0);
            gpufhe::rescale_resident_raw(d1,tw,C,s1,s2,scr,drp,0);
            cudaDeviceSynchronize();
            std::vector<uint64_t> h0(T),h1(T);
            cudaMemcpy(h0.data(),d0,T*8,cudaMemcpyDeviceToHost);cudaMemcpy(h1.data(),d1,T*8,cudaMemcpyDeviceToHost);
            cudaFree(d0);cudaFree(d1);cudaFree(scr);cudaFree(drp); gpufhe::ks_context_destroy(C);
            uint32_t nt=tw-1; size_t T2=(size_t)nt*n;
            std::vector<uint64_t> o0(h0.begin(),h0.begin()+T2),o1(h1.begin(),h1.begin()+T2);
            std::vector<uint64_t> mf(modl.begin(),modl.begin()+nt),rf(rootl.begin(),rootl.begin()+nt);
            auto KPr=gpufhe::keygen_host(n,mf,rf,ns,3.19,101);
            std::vector<int64_t> d; gpufhe::decrypt_host(d,o0,o1,KPr.s,n,mf,rf);
            std::vector<cd> v; gpufhe::decode_host(v,d,n,D/(double)qSmall);
            double e=0; for(uint32_t i=0;i<S;++i) e=std::max(e,std::abs(v[i].real()-xin[i].real()));
            return e; };
        double eR2=rescProbe(2), eRt=rescProbe(tw);
        auto tag=[](double e){return e<1e-3?"OK ":"BAD";};
        printf("%2u  %s %-12.4g  %s %-12.4g  %s %-12.4g\n",tw,tag(eRel),eRel,tag(eR2),eR2,tag(eRt),eRt);
        fflush(stdout);
    }
    return 0;
}
