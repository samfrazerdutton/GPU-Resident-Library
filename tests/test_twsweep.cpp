// BISECT: at which tower count does square+relin+rescale break?
// Holds the PROVEN sizeP=2, parts <=2 towers. Sweeps tw=3..8. x^2 vs plaintext.
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
    const uint32_t MAXQ=8;
    std::vector<uint64_t> modAll,modP;
    gpufhe::native_primes(modAll,1,60,n,{});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids,MAXQ-1,50,n,modAll); for(auto m:mids)modAll.push_back(m); }
    gpufhe::native_primes(modP,sizeP,60,n,modAll);
    std::vector<uint64_t> rootAll(MAXQ),rootP(sizeP);
    for(uint32_t i=0;i<MAXQ;++i)rootAll[i]=gpufhe::native_root(n,modAll[i]);
    for(uint32_t j=0;j<sizeP;++j)rootP[j]=gpufhe::native_root(n,modP[j]);
    const double Delta=std::pow(2.0,50);
    std::vector<cd> xin(S);
    for(uint32_t i=0;i<S;++i) xin[i]={0.6*std::sin(0.02*i),0};

    for(uint32_t tw=3;tw<=MAXQ;++tw){
        uint32_t np=(tw+1)/2;                       // parts <= 2 towers
        std::vector<uint64_t> modl(modAll.begin(),modAll.begin()+tw),rootl(rootAll.begin(),rootAll.begin()+tw);
        std::vector<uint64_t> modQPl(modl); for(auto p:modP)modQPl.push_back(p);
        std::vector<uint64_t> rootQPl(rootl); for(auto r:rootP)rootQPl.push_back(r);
        double qbits=0; for(auto q:modl)qbits+=std::log2((double)q);
        double pbits=0; for(auto q:modP)pbits+=std::log2((double)q);

        auto KPqp=gpufhe::keygen_host(n,modQPl,rootQPl,ns,3.19,101);
        auto KPq =gpufhe::keygen_host(n,modl,rootl,ns,3.19,101);
        gpufhe::KeySwitchConstants K; K.n=n;
        gpufhe::compute_keyswitch_constants(K,modl,modP,np);
        for(uint32_t i=0;i<tw;++i){K.rootModList.push_back(modl[i]);K.rootValList.push_back(rootl[i]);}
        for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rootP[j]);}
        std::vector<uint64_t> PModq(tw+sizeP);
        for(uint32_t t=0;t<tw+sizeP;++t){uint64_t q=modQPl[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modP[j]%q,q);PModq[t]=P;}
        gpufhe::evalkeygen_host(K,KPqp.s,KPqp.pkA,KPqp.pkB,PModq,modQPl,rootQPl,ns,3.19,202);

        std::vector<int64_t> mx; gpufhe::encode_host(mx,xin,n,Delta);
        std::vector<uint64_t> c0,c1;
        gpufhe::encrypt_host(c0,c1,mx,KPq.pkA,KPq.pkB,n,modl,rootl,ns,3.19,303);

        size_t T=(size_t)tw*n;
        std::vector<uint64_t> t0(T),t1(T),t2(T);
        for(uint32_t t=0;t<tw;++t){uint64_t q=modl[t];
            for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
                t0[x]=mm(c0[x],c0[x],q); t1[x]=am(mm(c0[x],c1[x],q),mm(c1[x],c0[x],q),q); t2[x]=mm(c1[x],c1[x],q);}}
        auto R=gpufhe::keyswitch_core_resident(t2,K);
        std::vector<uint64_t> r0(T),r1(T);
        for(uint32_t t=0;t<tw;++t){uint64_t q=modl[t];
            for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k; r0[x]=am(t0[x],R.ba0[x],q); r1[x]=am(t1[x],R.ba1[x],q);}}

        auto C=gpufhe::ks_context_create(K);
        std::vector<uint64_t> s1,s2; gpufhe::native_rescale_consts(s1,s2,modl,tw-1);
        uint64_t *dr0,*dr1,*scr,*drp;
        cudaMalloc(&dr0,T*8);cudaMalloc(&dr1,T*8);cudaMalloc(&scr,(size_t)n*8);cudaMalloc(&drp,(size_t)n*8);
        cudaMemcpy(dr0,r0.data(),T*8,cudaMemcpyHostToDevice);cudaMemcpy(dr1,r1.data(),T*8,cudaMemcpyHostToDevice);
        gpufhe::rescale_resident_raw(dr0,tw,C,s1,s2,scr,drp,0);
        gpufhe::rescale_resident_raw(dr1,tw,C,s1,s2,scr,drp,0);
        cudaDeviceSynchronize();
        std::vector<uint64_t> h0(T),h1(T);
        cudaMemcpy(h0.data(),dr0,T*8,cudaMemcpyDeviceToHost);cudaMemcpy(h1.data(),dr1,T*8,cudaMemcpyDeviceToHost);
        cudaFree(dr0);cudaFree(dr1);cudaFree(scr);cudaFree(drp); gpufhe::ks_context_destroy(C);

        uint32_t nt=tw-1; size_t T2s=(size_t)nt*n;
        std::vector<uint64_t> o0(h0.begin(),h0.begin()+T2s),o1(h1.begin(),h1.begin()+T2s);
        std::vector<uint64_t> mf(modl.begin(),modl.begin()+nt),rf(rootl.begin(),rootl.begin()+nt);
        auto KPr=gpufhe::keygen_host(n,mf,rf,ns,3.19,101);
        std::vector<int64_t> dec; gpufhe::decrypt_host(dec,o0,o1,KPr.s,n,mf,rf);
        double DO=Delta*Delta/(double)modl[tw-1];
        std::vector<cd> y; gpufhe::decode_host(y,dec,n,DO);
        double e=0; for(uint32_t i=0;i<S;++i){double x=xin[i].real(); e=std::max(e,std::abs(y[i].real()-x*x));}
        std::cout<<"tw="<<tw<<" np="<<np<<" Qbits="<<(int)qbits<<" Pbits="<<(int)pbits
                 <<"  x^2 err="<<e<<(e<1e-3?"   OK":"   <-- BREAKS")<<"\n";
    }
    return 0;
}
