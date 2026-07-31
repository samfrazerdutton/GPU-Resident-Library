// Minimal reproducer for the numPart chain bug. test_keyreuse does ONE relin
// per level from FRESH encryptions and passes everywhere; EvalMod's 8th
// sequential multiply (the first at numPart=2) is destroyed. This walks a
// chain: multiply by Enc(1.0) at each level so the value stays put and any
// deviation is pure error, mirroring the Chebyshev recurrence's structure
// (operand re-encrypted at the current level, accumulator carried down).
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
void set_secret_hamming_weight(uint32_t);
}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
static uint64_t am(uint64_t a,uint64_t b,uint64_t q){uint64_t s=a+b;return s>=q?s-q:s;}
using cd=std::complex<double>;

int main(int argc,char**argv){
    const uint32_t ALPHA=(argc>1)?std::stoul(argv[1]):10;
    const uint32_t SZP  =(argc>2)?std::stoul(argv[2]):10;
    const uint32_t MIDB =(argc>3)?std::stoul(argv[3]):55;
    const uint32_t TRUNC=(argc>4)?std::stoul(argv[4]):0;  // 1 = truncated operand
    const uint32_t n=1024,S=n/2,sizeQ=30; const uint64_t ns=1;
    auto npFor=[&](uint32_t tw){ return (tw+ALPHA-1)/ALPHA; };
    gpufhe::set_secret_hamming_weight(64);
    std::vector<uint64_t> mod,modP;
    gpufhe::native_primes(mod,1,60,n,{});
    { std::vector<uint64_t> md; gpufhe::native_primes(md,sizeQ-1,MIDB,n,mod); for(auto m:md)mod.push_back(m); }
    gpufhe::native_primes(modP,SZP,60,n,mod);
    std::vector<uint64_t> root(sizeQ),rootP(SZP);
    for(uint32_t i=0;i<sizeQ;++i)root[i]=gpufhe::native_root(n,mod[i]);
    for(uint32_t j=0;j<SZP;++j)rootP[j]=gpufhe::native_root(n,modP[j]);
    const double D=std::pow(2.0,MIDB);

    std::vector<cd> xv(S),one(S,cd{1.0,0});
    for(uint32_t i=0;i<S;++i) xv[i]={0.7*std::sin(0.01*i),0};

    auto encAt=[&](const std::vector<cd>& v,uint32_t tw,uint32_t seed,
                   std::vector<uint64_t>& c0,std::vector<uint64_t>& c1){
        std::vector<uint64_t> ml(mod.begin(),mod.begin()+tw),rl(root.begin(),root.begin()+tw);
        auto KP=gpufhe::keygen_host(n,ml,rl,ns,3.19,101);
        std::vector<int64_t> m; gpufhe::encode_host(m,v,n,D);
        gpufhe::encrypt_host(c0,c1,m,KP.pkA,KP.pkB,n,ml,rl,ns,3.19,seed); };

    std::vector<uint64_t> a0,a1; encAt(xv,sizeQ,303,a0,a1);
    std::vector<uint64_t> oneFull0,oneFull1; encAt(one,sizeQ,777,oneFull0,oneFull1);
    uint32_t tw=sizeQ; double sc=D;
    std::cout<<"ALPHA="<<ALPHA<<" sizeP="<<SZP<<" midbits="<<MIDB<<(TRUNC?"  TRUNCATED operand":"  fresh operand")<<"\n";
    uint32_t bad=0;
    for(uint32_t step=1; step<=15; ++step){
        std::vector<uint64_t> ml(mod.begin(),mod.begin()+tw),rl(root.begin(),root.begin()+tw);
        std::vector<uint64_t> mq(ml); for(auto p:modP)mq.push_back(p);
        std::vector<uint64_t> rq(rl); for(auto r:rootP)rq.push_back(r);
        // EvalMod multiplies by lvl(U, tw) -- a ciphertext carried DOWN from a
        // high level by tower truncation, not re-encrypted. That is the last
        // structural difference from test_keyreuse and the plain chain.
        std::vector<uint64_t> b0,b1;
        if(TRUNC){ b0.assign(oneFull0.begin(),oneFull0.begin()+(size_t)tw*n);
                   b1.assign(oneFull1.begin(),oneFull1.begin()+(size_t)tw*n); }
        else encAt(one,tw,900+step,b0,b1);
        // fresh relin key at this level
        // LAST DIFFERENCE vs test_boot: it SLICES a keypair generated over the
        // FULL sizeQ+sizeP towers rather than generating one at this level.
        // Same secret (sparse sampling precedes the tower loop) but keygen_host
        // draws the uniform pubkey per tower, so a and e differ between paths.
        std::vector<uint64_t> mqF(mod), rqF(root);
        for(auto p2:modP) mqF.push_back(p2);
        for(auto r2:rootP) rqF.push_back(r2);
        auto KPfull=gpufhe::keygen_host(n,mqF,rqF,ns,3.19,101);
        auto slice=[&](const std::vector<uint64_t>& src){
            std::vector<uint64_t> d((size_t)(tw+SZP)*n);
            for(uint32_t t=0;t<tw;++t)
                std::copy(src.begin()+(size_t)t*n,src.begin()+(size_t)(t+1)*n,d.begin()+(size_t)t*n);
            for(uint32_t j=0;j<SZP;++j)
                std::copy(src.begin()+(size_t)(sizeQ+j)*n,src.begin()+(size_t)(sizeQ+j+1)*n,
                          d.begin()+(size_t)(tw+j)*n);
            return d; };
        gpufhe::KeyPairHost KPqp;
        KPqp.s=slice(KPfull.s); KPqp.pkA=slice(KPfull.pkA); KPqp.pkB=slice(KPfull.pkB);
        gpufhe::KeySwitchConstants K; K.n=n;
        gpufhe::compute_keyswitch_constants(K,ml,modP,npFor(tw));
        for(uint32_t i=0;i<tw;++i){K.rootModList.push_back(ml[i]);K.rootValList.push_back(rl[i]);}
        for(uint32_t j=0;j<SZP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rootP[j]);}
        std::vector<uint64_t> PM(tw+SZP);
        for(uint32_t t=0;t<tw+SZP;++t){uint64_t q=mq[t],P=1%q;for(uint32_t j=0;j<SZP;++j)P=mm(P,modP[j]%q,q);PM[t]=P;}
        gpufhe::evalkeygen_host(K,KPqp.s,KPqp.pkA,KPqp.pkB,PM,mq,rq,ns,3.19,202);
        size_t T=(size_t)tw*n;
        std::vector<uint64_t> t0(T),t1(T),t2(T);
        for(uint32_t t=0;t<tw;++t){uint64_t q=ml[t];
            for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
                t0[x]=mm(a0[x],b0[x],q);
                t1[x]=am(mm(a0[x],b1[x],q),mm(a1[x],b0[x],q),q);
                t2[x]=mm(a1[x],b1[x],q);}}
        auto R=gpufhe::keyswitch_core_resident(t2,K);
        std::vector<uint64_t> r0(T),r1(T);
        for(uint32_t t=0;t<tw;++t){uint64_t q=ml[t];
            for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
                r0[x]=am(t0[x],R.ba0[x],q); r1[x]=am(t1[x],R.ba1[x],q);}}
        auto Cx=gpufhe::ks_context_create(K);
        std::vector<uint64_t> s1,s2; gpufhe::native_rescale_consts(s1,s2,ml,tw-1);
        uint64_t *d0,*d1,*sc2,*dp;
        cudaMalloc(&d0,T*8);cudaMalloc(&d1,T*8);cudaMalloc(&sc2,(size_t)n*8);cudaMalloc(&dp,(size_t)n*8);
        cudaMemcpy(d0,r0.data(),T*8,cudaMemcpyHostToDevice);cudaMemcpy(d1,r1.data(),T*8,cudaMemcpyHostToDevice);
        gpufhe::rescale_resident_raw(d0,tw,Cx,s1,s2,sc2,dp,0);
        gpufhe::rescale_resident_raw(d1,tw,Cx,s1,s2,sc2,dp,0);
        cudaDeviceSynchronize();
        std::vector<uint64_t> h0(T),h1(T);
        cudaMemcpy(h0.data(),d0,T*8,cudaMemcpyDeviceToHost);cudaMemcpy(h1.data(),d1,T*8,cudaMemcpyDeviceToHost);
        cudaFree(d0);cudaFree(d1);cudaFree(sc2);cudaFree(dp);gpufhe::ks_context_destroy(Cx);
        sc=sc*D/(double)ml[tw-1]; --tw;
        size_t T2=(size_t)tw*n;
        a0.assign(h0.begin(),h0.begin()+T2); a1.assign(h1.begin(),h1.begin()+T2);
        std::vector<uint64_t> mf(mod.begin(),mod.begin()+tw),rf(root.begin(),root.begin()+tw);
        auto KR=gpufhe::keygen_host(n,mf,rf,ns,3.19,101);
        std::vector<int64_t> dec; gpufhe::decrypt_host(dec,a0,a1,KR.s,n,mf,rf);
        std::vector<cd> y; gpufhe::decode_host(y,dec,n,sc);
        double e=0; for(uint32_t i=0;i<S;++i) e=std::max(e,std::abs(y[i].real()-xv[i].real()));
        bool ok=e<1e-6; if(!ok)++bad;
        std::cout<<"  step "<<step<<" -> tw="<<tw<<" parts="<<npFor(tw+1)
                 <<" err="<<e<<(ok?"   OK":"   <-- BAD")<<"\n";
    }
    std::cout<<(bad?"[FAIL] ":"[PASS] ")<<bad<<" bad steps\n"; return bad?1:0;
}
