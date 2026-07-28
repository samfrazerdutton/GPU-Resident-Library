// EVALMOD: homomorphically strip the integer part of x = m + I.
//   sin(2*pi*x) = cos(2*pi*(x-1/4));  y = (pi/8)(x-1/4);  cos(16y) = sin(2 pi x)
//   -> low-degree Chebyshev cos(y) on |y|<1, then 4x double-angle c <- 2c^2-1.
// Since sin(2 pi (m+I)) = sin(2 pi m) ~ 2 pi m, output/(2 pi) recovers m.
// Depth: 1 affine + 7 (T2..T8) + 1 (sum rescale) + 4 (doublings) = 13.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <functional>
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
struct Ct { std::vector<uint64_t> c0,c1; uint32_t tw; double scale; };

int main(){
    const uint32_t n=1024,S=n/2,sizeQ=15,sizeP=2; const uint64_t ns=1;
    auto npFor=[](uint32_t tw)->uint32_t{ return tw<=2?1u:(tw+1)/2; };  // parts <= 2 towers
    std::vector<uint64_t> mod,modP;
    gpufhe::native_primes(mod,1,60,n,{});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids,sizeQ-1,50,n,mod); for(auto m:mids)mod.push_back(m); }
    gpufhe::native_primes(modP,sizeP,60,n,mod);
    std::vector<uint64_t> modQP=mod; for(auto p:modP)modQP.push_back(p);
    std::vector<uint64_t> root(sizeQ),rootQP;
    for(uint32_t i=0;i<sizeQ;++i)root[i]=gpufhe::native_root(n,mod[i]);
    rootQP=root; for(auto p:modP)rootQP.push_back(gpufhe::native_root(n,p));
    auto KPqp=gpufhe::keygen_host(n,modQP,rootQP,ns,3.19,101);
    auto KPq =gpufhe::keygen_host(n,mod,root,ns,3.19,101);
    const double Delta=std::pow(2.0,50);

    auto buildK=[&](uint32_t tw)->gpufhe::KeySwitchConstants{
        gpufhe::KeySwitchConstants Kl; Kl.n=n;
        std::vector<uint64_t> modl(mod.begin(),mod.begin()+tw), rootl(root.begin(),root.begin()+tw);
        gpufhe::compute_keyswitch_constants(Kl,modl,modP,npFor(tw));
        for(uint32_t i=0;i<tw;++i){Kl.rootModList.push_back(modl[i]);Kl.rootValList.push_back(rootl[i]);}
        for(uint32_t j=0;j<sizeP;++j){Kl.rootModList.push_back(modP[j]);Kl.rootValList.push_back(rootQP[sizeQ+j]);}
        std::vector<uint64_t> modQPl(modl); for(auto p:modP)modQPl.push_back(p);
        std::vector<uint64_t> rootQPl(rootl); for(uint32_t j=0;j<sizeP;++j)rootQPl.push_back(rootQP[sizeQ+j]);
        auto slice=[&](const std::vector<uint64_t>&src){ std::vector<uint64_t> d((size_t)(tw+sizeP)*n);
            for(uint32_t t=0;t<tw;++t) std::copy(src.begin()+(size_t)t*n,src.begin()+(size_t)(t+1)*n,d.begin()+(size_t)t*n);
            for(uint32_t j=0;j<sizeP;++j) std::copy(src.begin()+(size_t)(sizeQ+j)*n,src.begin()+(size_t)(sizeQ+j+1)*n,d.begin()+(size_t)(tw+j)*n);
            return d; };
        std::vector<uint64_t> PModq(tw+sizeP);
        for(uint32_t t=0;t<tw+sizeP;++t){uint64_t q=modQPl[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modP[j]%q,q);PModq[t]=P;}
        gpufhe::evalkeygen_host(Kl,slice(KPqp.s),slice(KPqp.pkA),slice(KPqp.pkB),PModq,modQPl,rootQPl,ns,3.19,202);
        return Kl; };

    auto rescaleInPlace=[&](std::vector<uint64_t>&a0,std::vector<uint64_t>&a1,uint32_t tw,
                            const gpufhe::KeySwitchConstants&Kx){
        auto Cl=gpufhe::ks_context_create(Kx);
        std::vector<uint64_t> s1,s2; { std::vector<uint64_t> sub(mod.begin(),mod.begin()+tw); gpufhe::native_rescale_consts(s1,s2,sub,tw-1); }
        size_t T=(size_t)tw*n; uint64_t *d0,*d1,*scr,*drp;
        cudaMalloc(&d0,T*8);cudaMalloc(&d1,T*8);cudaMalloc(&scr,(size_t)n*8);cudaMalloc(&drp,(size_t)n*8);
        cudaMemcpy(d0,a0.data(),T*8,cudaMemcpyHostToDevice);cudaMemcpy(d1,a1.data(),T*8,cudaMemcpyHostToDevice);
        gpufhe::rescale_resident_raw(d0,tw,Cl,s1,s2,scr,drp,0);
        gpufhe::rescale_resident_raw(d1,tw,Cl,s1,s2,scr,drp,0);
        cudaDeviceSynchronize();
        std::vector<uint64_t> h0(T),h1(T);
        cudaMemcpy(h0.data(),d0,T*8,cudaMemcpyDeviceToHost);cudaMemcpy(h1.data(),d1,T*8,cudaMemcpyDeviceToHost);
        cudaFree(d0);cudaFree(d1);cudaFree(scr);cudaFree(drp); gpufhe::ks_context_destroy(Cl);
        size_t T2=(size_t)(tw-1)*n;
        a0.assign(h0.begin(),h0.begin()+T2); a1.assign(h1.begin(),h1.begin()+T2); };

    auto mulCt=[&](const Ct&A,const Ct&B)->Ct{
        uint32_t tw=A.tw; size_t T=(size_t)tw*n;
        std::vector<uint64_t> t0(T),t1(T),t2(T);
        for(uint32_t t=0;t<tw;++t){uint64_t q=mod[t];
            for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
                t0[x]=mm(A.c0[x],B.c0[x],q);
                t1[x]=am(mm(A.c0[x],B.c1[x],q),mm(A.c1[x],B.c0[x],q),q);
                t2[x]=mm(A.c1[x],B.c1[x],q);}}
        auto Kl=buildK(tw);
        auto R=gpufhe::keyswitch_core_resident(t2,Kl);
        std::vector<uint64_t> r0(T),r1(T);
        for(uint32_t t=0;t<tw;++t){uint64_t q=mod[t];
            for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k; r0[x]=am(t0[x],R.ba0[x],q); r1[x]=am(t1[x],R.ba1[x],q);}}
        rescaleInPlace(r0,r1,tw,Kl);
        Ct o; o.tw=tw-1; o.scale=A.scale*B.scale/(double)mod[tw-1]; o.c0=r0; o.c1=r1;
        return o; };

    auto levelReduce=[&](const Ct&A,uint32_t tw)->Ct{
        Ct o; o.tw=tw; o.scale=A.scale; size_t T=(size_t)tw*n;
        o.c0.assign(A.c0.begin(),A.c0.begin()+T); o.c1.assign(A.c1.begin(),A.c1.begin()+T); return o; };
    auto scaleBy2=[&](Ct&A){ for(uint32_t t=0;t<A.tw;++t){uint64_t q=mod[t];
        for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k; A.c0[x]=(A.c0[x]*2)%q; A.c1[x]=(A.c1[x]*2)%q;}} };
    auto subCt=[&](const Ct&A,const Ct&B)->Ct{
        Ct o; o.tw=A.tw; o.scale=A.scale; size_t T=(size_t)A.tw*n; o.c0.resize(T);o.c1.resize(T);
        for(uint32_t t=0;t<A.tw;++t){uint64_t q=mod[t];
            for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
                o.c0[x]=sm(A.c0[x],B.c0[x],q); o.c1[x]=sm(A.c1[x],B.c1[x],q);}}
        return o; };
    auto subConst=[&](Ct&A,double v){
        std::vector<cd> z(S,cd{v,0}); std::vector<int64_t> m; gpufhe::encode_host(m,z,n,A.scale);
        std::vector<uint64_t> ml(mod.begin(),mod.begin()+A.tw),rl(root.begin(),root.begin()+A.tw);
        std::vector<uint64_t> e; gpufhe::pt_to_eval_host(e,m,A.tw,n,ml,rl);
        for(uint32_t t=0;t<A.tw;++t){uint64_t q=mod[t];
            for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k; A.c0[x]=sm(A.c0[x],e[x],q);}} };
    // multiply by a plaintext scalar, then rescale (drops one level)
    auto mulConst=[&](const Ct&A,double v)->Ct{
        Ct o=A; std::vector<cd> z(S,cd{v,0}); std::vector<int64_t> m; gpufhe::encode_host(m,z,n,Delta);
        std::vector<uint64_t> ml(mod.begin(),mod.begin()+A.tw),rl(root.begin(),root.begin()+A.tw);
        std::vector<uint64_t> e; gpufhe::pt_to_eval_host(e,m,A.tw,n,ml,rl);
        gpufhe::ct_mul_pt_host(o.c0,o.c1,e,A.tw,n,ml);
        auto Kl=buildK(A.tw); rescaleInPlace(o.c0,o.c1,A.tw,Kl);
        o.tw=A.tw-1; o.scale=A.scale*Delta/(double)mod[A.tw-1]; return o; };

    // ---- input: x = m + I,  |m| small, I integer in [-2,2]
    std::vector<cd> xin(S),mtrue(S);
    for(uint32_t i=0;i<S;++i){
        double m=0.04*std::sin(0.021*i);
        int I=(int)(i%5)-2;
        mtrue[i]={m,0}; xin[i]={m+(double)I,0}; }
    std::vector<int64_t> mx; gpufhe::encode_host(mx,xin,n,Delta);
    Ct X; X.tw=sizeQ; X.scale=Delta;
    gpufhe::encrypt_host(X.c0,X.c1,mx,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,303);

    auto chk=[&](const char*lbl,const Ct&A,const std::vector<double>&exp){
        std::vector<uint64_t> ml(mod.begin(),mod.begin()+A.tw),rl(root.begin(),root.begin()+A.tw);
        auto KP=gpufhe::keygen_host(n,ml,rl,ns,3.19,101);
        std::vector<int64_t> d; gpufhe::decrypt_host(d,A.c0,A.c1,KP.s,n,ml,rl);
        std::vector<cd> v; gpufhe::decode_host(v,d,n,A.scale);
        double e=0; for(uint32_t i=0;i<S;++i) e=std::max(e,std::abs(v[i].real()-exp[i]));
        std::cerr<<"    "<<lbl<<" tw="<<A.tw<<" err="<<e<<"\n"; };

    // ---- affine:  y = (pi/8) x - pi/32
    const double A_=M_PI/8.0, B_=-M_PI/32.0;
    Ct U=mulConst(X,A_); subConst(U,-B_);
    { std::vector<double> ey(S); for(uint32_t i=0;i<S;++i) ey[i]=A_*xin[i].real()+B_; chk("affine y",U,ey); }

    // ---- Chebyshev cos(y), degree 8
    const uint32_t deg=8;
    std::vector<double> c(deg+1,0.0);
    { const uint32_t NQ=4*(deg+1);
      for(uint32_t k=0;k<=deg;++k){ double s=0;
          for(uint32_t j=0;j<NQ;++j){ double th=M_PI*(j+0.5)/NQ; s+=std::cos(std::cos(th))*std::cos(k*th); }
          c[k]=2.0*s/NQ; }
      c[0]*=0.5; }

    std::vector<Ct> T(deg+1); T[1]=U;
    { Ct sq=mulCt(U,U); scaleBy2(sq); subConst(sq,1.0); T[2]=sq; }
    for(uint32_t k=3;k<=deg;++k){
        Ct ul=levelReduce(U,T[k-1].tw);
        Ct P=mulCt(ul,T[k-1]); scaleBy2(P);
        Ct km2=levelReduce(T[k-2],P.tw);
        T[k]=subCt(P,km2); }

    uint32_t low=T[deg].tw;
    std::vector<uint64_t> ml(mod.begin(),mod.begin()+low),rl(root.begin(),root.begin()+low);
    Ct acc; acc.tw=low; acc.scale=0; bool first=true;
    for(uint32_t k=1;k<=deg;++k){
        Ct t=levelReduce(T[k],low);
        std::vector<cd> zc(S,cd{c[k],0}); std::vector<int64_t> mc; gpufhe::encode_host(mc,zc,n,Delta);
        std::vector<uint64_t> cE; gpufhe::pt_to_eval_host(cE,mc,low,n,ml,rl);
        gpufhe::ct_mul_pt_host(t.c0,t.c1,cE,low,n,ml);
        if(first){acc.c0=t.c0;acc.c1=t.c1;acc.scale=t.scale*Delta;first=false;}
        else gpufhe::ct_add_ct_host(acc.c0,acc.c1,t.c0,t.c1,low,n,ml); }
    { auto Kd=buildK(low); rescaleInPlace(acc.c0,acc.c1,low,Kd);
      acc.scale=acc.scale/(double)mod[low-1]; acc.tw=low-1; }
    subConst(acc,-c[0]);
    { std::vector<double> ec(S); for(uint32_t i=0;i<S;++i){double y=A_*xin[i].real()+B_; ec[i]=std::cos(y);} chk("cos(y)",acc,ec); }

    // ---- 4 double-angle steps: c <- 2c^2 - 1   =>  cos(16 y) = sin(2 pi x)
    for(uint32_t r=1;r<=4;++r){
        Ct sq=mulCt(acc,acc); scaleBy2(sq); subConst(sq,1.0); acc=sq;
        std::vector<double> ed(S); double f=std::pow(2.0,(double)r);
        for(uint32_t i=0;i<S;++i){double y=A_*xin[i].real()+B_; ed[i]=std::cos(f*y);}
        char lbl[32]; snprintf(lbl,32,"double x%u",r); chk(lbl,acc,ed); }

    // ---- decrypt: result = sin(2 pi x) ; m ~ result/(2 pi)
    std::vector<uint64_t> mf(mod.begin(),mod.begin()+acc.tw),rf(root.begin(),root.begin()+acc.tw);
    auto KPr=gpufhe::keygen_host(n,mf,rf,ns,3.19,101);
    std::vector<int64_t> dec; gpufhe::decrypt_host(dec,acc.c0,acc.c1,KPr.s,n,mf,rf);
    std::vector<cd> y; gpufhe::decode_host(y,dec,n,acc.scale);

    double errSin=0,errMod=0;
    for(uint32_t i=0;i<S;++i){
        errSin=std::max(errSin,std::abs(y[i].real()-std::sin(2*M_PI*xin[i].real())));
        errMod=std::max(errMod,std::abs(y[i].real()/(2*M_PI)-mtrue[i].real())); }
    std::cout<<"EvalMod: final tw="<<acc.tw<<" scale=2^"<<std::log2(acc.scale)<<"\n";
    std::cout<<"  err vs sin(2 pi x) = "<<errSin<<"\n";
    std::cout<<"  err vs m (integer part STRIPPED) = "<<errMod<<"\n";
    std::cout<<"  x[0..3]="; for(int k=0;k<4;++k)std::cout<<xin[k].real()<<" ";
    std::cout<<"\n  m[0..3]="; for(int k=0;k<4;++k)std::cout<<mtrue[k].real()<<" ";
    std::cout<<"\n  got    ="; for(int k=0;k<4;++k)std::cout<<y[k].real()/(2*M_PI)<<" "; std::cout<<"\n";
    if(errMod<1e-3){std::cout<<"[PASS] EvalMod: homomorphic mod-reduction (integer part removed)\n";return 0;}
    std::cout<<"[FAIL]\n"; return 1;
}
