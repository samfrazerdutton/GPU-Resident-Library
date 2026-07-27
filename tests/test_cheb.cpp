// EvalMod CORE: degree-7 Chebyshev evaluation under encryption.
//   T_0=1, T_1=x, T_{k+1} = 2 x T_k - T_{k-1};  p(x) = sum_k c_k T_k(x)
// Level alignment by RNS TOWER TRUNCATION (scale-preserving, noise-free) — NOT
// by multiplying by an encrypted 1. Constants always encoded at the ct's ACTUAL
// current scale. 6 sequential ct*ct mults => 6 rescales. OpenFHE-free, n=1024.
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
struct Ct { std::vector<uint64_t> c0,c1; uint32_t tw; double scale; };

int main(){
    const uint32_t n=1024,S=n/2,sizeQ=8,sizeP=2; const uint64_t ns=1;
    // PROVEN CONFIG (test_twsweep: clean 8.8e-12 at every tw=3..8): sizeP=2 with
    // parts <= 2 towers. numPart must ADAPT as towers drop -- fixed numPart=2 makes
    // parts 210 bits vs P=120 and the keyswitch noise swamps the message.
    auto npFor=[](uint32_t tw)->uint32_t{ return tw<=2?1u:(tw+1)/2; };
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

    // per-level keyswitch constants + relin key (built on demand)
    auto buildK=[&](uint32_t tw)->gpufhe::KeySwitchConstants{
        gpufhe::KeySwitchConstants Kl; Kl.n=n;
        std::vector<uint64_t> modl(mod.begin(),mod.begin()+tw), rootl(root.begin(),root.begin()+tw);
        gpufhe::compute_keyswitch_constants(Kl,modl,modP,npFor(tw));
        for(uint32_t i=0;i<tw;++i){Kl.rootModList.push_back(modl[i]);Kl.rootValList.push_back(rootl[i]);}
        for(uint32_t j=0;j<sizeP;++j){Kl.rootModList.push_back(modP[j]);Kl.rootValList.push_back(rootQP[sizeQ+j]);}
        std::vector<uint64_t> modQPl(modl); for(auto p:modP)modQPl.push_back(p);
        std::vector<uint64_t> rootQPl(rootl); for(uint32_t j=0;j<sizeP;++j)rootQPl.push_back(rootQP[sizeQ+j]);
        auto slice=[&](const std::vector<uint64_t>&src)->std::vector<uint64_t>{
            std::vector<uint64_t> d((size_t)(tw+sizeP)*n);
            for(uint32_t t=0;t<tw;++t) std::copy(src.begin()+(size_t)t*n,src.begin()+(size_t)(t+1)*n,d.begin()+(size_t)t*n);
            for(uint32_t j=0;j<sizeP;++j) std::copy(src.begin()+(size_t)(sizeQ+j)*n,src.begin()+(size_t)(sizeQ+j+1)*n,d.begin()+(size_t)(tw+j)*n);
            return d; };
        std::vector<uint64_t> PModq(tw+sizeP);
        for(uint32_t t=0;t<tw+sizeP;++t){uint64_t q=modQPl[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modP[j]%q,q);PModq[t]=P;}
        gpufhe::evalkeygen_host(Kl,slice(KPqp.s),slice(KPqp.pkA),slice(KPqp.pkB),PModq,modQPl,rootQPl,ns,3.19,202);
        return Kl; };

    // ct*ct -> tensor + relin + rescale (drops one tower)
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
        auto Cl=gpufhe::ks_context_create(Kl);
        std::vector<uint64_t> s1,s2; { std::vector<uint64_t> sub(mod.begin(),mod.begin()+tw); gpufhe::native_rescale_consts(s1,s2,sub,tw-1); }
        uint64_t *dr0,*dr1,*scr,*drp;
        cudaMalloc(&dr0,T*8);cudaMalloc(&dr1,T*8);cudaMalloc(&scr,(size_t)n*8);cudaMalloc(&drp,(size_t)n*8);
        cudaMemcpy(dr0,r0.data(),T*8,cudaMemcpyHostToDevice);cudaMemcpy(dr1,r1.data(),T*8,cudaMemcpyHostToDevice);
        gpufhe::rescale_resident_raw(dr0,tw,Cl,s1,s2,scr,drp,0);
        gpufhe::rescale_resident_raw(dr1,tw,Cl,s1,s2,scr,drp,0);
        cudaDeviceSynchronize();
        std::vector<uint64_t> h0(T),h1(T);
        cudaMemcpy(h0.data(),dr0,T*8,cudaMemcpyDeviceToHost);cudaMemcpy(h1.data(),dr1,T*8,cudaMemcpyDeviceToHost);
        cudaFree(dr0);cudaFree(dr1);cudaFree(scr);cudaFree(drp); gpufhe::ks_context_destroy(Cl);
        Ct o; o.tw=tw-1; o.scale=A.scale*B.scale/(double)mod[tw-1];
        size_t T2s=(size_t)o.tw*n;
        o.c0.assign(h0.begin(),h0.begin()+T2s); o.c1.assign(h1.begin(),h1.begin()+T2s);
        return o; };

    // LEVEL REDUCE by truncation: residues mod Q' with Q'|Q. Scale unchanged, no noise.
    auto levelReduce=[&](const Ct&A,uint32_t tw)->Ct{
        Ct o; o.tw=tw; o.scale=A.scale; size_t T=(size_t)tw*n;
        o.c0.assign(A.c0.begin(),A.c0.begin()+T); o.c1.assign(A.c1.begin(),A.c1.begin()+T);
        return o; };
    auto scaleBy2=[&](Ct&A){ for(uint32_t t=0;t<A.tw;++t){uint64_t q=mod[t];
        for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k; A.c0[x]=(A.c0[x]*2)%q; A.c1[x]=(A.c1[x]*2)%q;}} };
    auto subCt=[&](const Ct&A,const Ct&B)->Ct{   // same tw, scales treated equal (FIXEDMANUAL)
        Ct o; o.tw=A.tw; o.scale=A.scale; size_t T=(size_t)A.tw*n; o.c0.resize(T);o.c1.resize(T);
        for(uint32_t t=0;t<A.tw;++t){uint64_t q=mod[t];
            for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
                o.c0[x]=sm(A.c0[x],B.c0[x],q); o.c1[x]=sm(A.c1[x],B.c1[x],q);}}
        return o; };
    auto subConst=[&](Ct&A,double v){            // encode v at A's ACTUAL scale
        std::vector<cd> z(S,cd{v,0}); std::vector<int64_t> m; gpufhe::encode_host(m,z,n,A.scale);
        std::vector<uint64_t> modl(mod.begin(),mod.begin()+A.tw),rootl(root.begin(),root.begin()+A.tw);
        std::vector<uint64_t> e; gpufhe::pt_to_eval_host(e,m,A.tw,n,modl,rootl);
        for(uint32_t t=0;t<A.tw;++t){uint64_t q=mod[t];
            for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k; A.c0[x]=sm(A.c0[x],e[x],q);}} };

    // ---- input + coefficients
    const uint32_t deg=7;
    std::vector<double> c={0.1,-0.3,0.2,0.15,-0.1,0.05,0.08,-0.04};
    std::vector<cd> xin(S);
    for(uint32_t i=0;i<S;++i) xin[i]={0.6*std::sin(0.02*i),0};
    std::vector<int64_t> mx; gpufhe::encode_host(mx,xin,n,Delta);
    Ct X; X.tw=sizeQ; X.scale=Delta;
    gpufhe::encrypt_host(X.c0,X.c1,mx,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,303);

    // per-step verifier: decrypt T_k at its own level and compare to plaintext T_k(x)
    auto chk=[&](uint32_t k,const Ct&A){
        std::vector<uint64_t> ml(mod.begin(),mod.begin()+A.tw),rl(root.begin(),root.begin()+A.tw);
        auto KP=gpufhe::keygen_host(n,ml,rl,ns,3.19,101);
        std::vector<int64_t> d; gpufhe::decrypt_host(d,A.c0,A.c1,KP.s,n,ml,rl);
        std::vector<cd> v; gpufhe::decode_host(v,d,n,A.scale);
        double e=0;
        for(uint32_t i=0;i<S;++i){ double x=xin[i].real(), Tp=1, Tc=x;
            for(uint32_t j=2;j<=k;++j){ double Tn=2*x*Tc-Tp; Tp=Tc; Tc=Tn; }
            e=std::max(e,std::abs(v[i].real()-Tc)); }
        std::cerr<<"    CHECK T"<<k<<" tw="<<A.tw<<" err="<<e<<"  got["<<v[3].real()<<"]\n"; };

    // ---- Chebyshev recurrence
    std::vector<Ct> T(deg+1);
    T[1]=X;
    { Ct sq=mulCt(X,X); scaleBy2(sq); subConst(sq,1.0); T[2]=sq; }   // 2x^2-1
    chk(1,T[1]); chk(2,T[2]);
    for(uint32_t k=3;k<=deg;++k){
        Ct xl=levelReduce(X,T[k-1].tw);
        Ct P=mulCt(xl,T[k-1]);           // 2 x T_{k-1} (pre-double)
        scaleBy2(P);
        Ct km2=levelReduce(T[k-2],P.tw);
        T[k]=subCt(P,km2);
        chk(k,T[k]);
    }

    // ---- sum c_k T_k at the common lowest level (k>=1), then one rescale, then + c_0
    uint32_t low=T[deg].tw;
    std::vector<uint64_t> modl(mod.begin(),mod.begin()+low),rootl(root.begin(),root.begin()+low);
    Ct acc; acc.tw=low; acc.scale=0; bool first=true;
    for(uint32_t k=1;k<=deg;++k){
        Ct t=levelReduce(T[k],low);
        std::vector<cd> zc(S,cd{c[k],0}); std::vector<int64_t> mc; gpufhe::encode_host(mc,zc,n,Delta);
        std::vector<uint64_t> cE; gpufhe::pt_to_eval_host(cE,mc,low,n,modl,rootl);
        gpufhe::ct_mul_pt_host(t.c0,t.c1,cE,low,n,modl);      // scale -> t.scale*Delta
        if(first){acc.c0=t.c0;acc.c1=t.c1;acc.scale=t.scale*Delta;first=false;}
        else gpufhe::ct_add_ct_host(acc.c0,acc.c1,t.c0,t.c1,low,n,modl);
    }
    // one rescale: scale -> acc.scale/mod[low-1]
    {
        gpufhe::KeySwitchConstants Kd; Kd.n=n;
        gpufhe::compute_keyswitch_constants(Kd,modl,modP,npFor(low));
        for(uint32_t i=0;i<low;++i){Kd.rootModList.push_back(modl[i]);Kd.rootValList.push_back(rootl[i]);}
        for(uint32_t j=0;j<sizeP;++j){Kd.rootModList.push_back(modP[j]);Kd.rootValList.push_back(rootQP[sizeQ+j]);}
        Kd.av.assign(Kd.numPart,{});Kd.bv.assign(Kd.numPart,{});Kd.evalKeyTowers=low+sizeP;
        for(uint32_t p=0;p<Kd.numPart;++p){Kd.av[p].assign((size_t)(low+sizeP)*n,0);Kd.bv[p].assign((size_t)(low+sizeP)*n,0);}
        auto Cd=gpufhe::ks_context_create(Kd);
        std::vector<uint64_t> s1,s2; { std::vector<uint64_t> sub(modl); gpufhe::native_rescale_consts(s1,s2,sub,low-1); }
        size_t T=(size_t)low*n; uint64_t *dr0,*dr1,*scr,*drp;
        cudaMalloc(&dr0,T*8);cudaMalloc(&dr1,T*8);cudaMalloc(&scr,(size_t)n*8);cudaMalloc(&drp,(size_t)n*8);
        cudaMemcpy(dr0,acc.c0.data(),T*8,cudaMemcpyHostToDevice);cudaMemcpy(dr1,acc.c1.data(),T*8,cudaMemcpyHostToDevice);
        gpufhe::rescale_resident_raw(dr0,low,Cd,s1,s2,scr,drp,0);
        gpufhe::rescale_resident_raw(dr1,low,Cd,s1,s2,scr,drp,0);
        cudaDeviceSynchronize();
        std::vector<uint64_t> h0(T),h1(T);
        cudaMemcpy(h0.data(),dr0,T*8,cudaMemcpyDeviceToHost);cudaMemcpy(h1.data(),dr1,T*8,cudaMemcpyDeviceToHost);
        cudaFree(dr0);cudaFree(dr1);cudaFree(scr);cudaFree(drp); gpufhe::ks_context_destroy(Cd);
        double ns2=acc.scale/(double)modl[low-1]; acc.tw=low-1; acc.scale=ns2;
        size_t T2s=(size_t)acc.tw*n;
        acc.c0.assign(h0.begin(),h0.begin()+T2s); acc.c1.assign(h1.begin(),h1.begin()+T2s);
    }
    subConst(acc,-c[0]);   // + c_0 (T_0 = 1), encoded at acc's actual scale

    // ---- decrypt + compare
    std::vector<uint64_t> modf(mod.begin(),mod.begin()+acc.tw),rootf(root.begin(),root.begin()+acc.tw);
    auto KPr=gpufhe::keygen_host(n,modf,rootf,ns,3.19,101);
    std::vector<int64_t> dec; gpufhe::decrypt_host(dec,acc.c0,acc.c1,KPr.s,n,modf,rootf);
    std::vector<cd> y; gpufhe::decode_host(y,dec,n,acc.scale);

    double maxerr=0;
    for(uint32_t i=0;i<S;++i){ double x=xin[i].real();
        double Tp=1,Tc=x,p=c[0]+c[1]*x;
        for(uint32_t k=2;k<=deg;++k){ double Tn=2*x*Tc-Tp; p+=c[k]*Tn; Tp=Tc; Tc=Tn; }
        maxerr=std::max(maxerr,std::abs(y[i].real()-p)); }
    std::cout<<"deg-7 Chebyshev, final tw="<<acc.tw<<" scale=2^"<<std::log2(acc.scale)<<" maxerr="<<maxerr<<"\n";
    std::cout<<"  y[0..3]="; for(int k=0;k<4;++k)std::cout<<y[k].real()<<" ";
    std::cout<<"\n  ref   ="; for(int k=0;k<4;++k){double x=xin[k].real();double Tp=1,Tc=x,p=c[0]+c[1]*x;
        for(uint32_t j=2;j<=deg;++j){double Tn=2*x*Tc-Tp;p+=c[j]*Tn;Tp=Tc;Tc=Tn;} std::cout<<p<<" ";} std::cout<<"\n";
    if(maxerr<1e-3){std::cout<<"[PASS] degree-7 Chebyshev evaluation under encryption (EvalMod engine)\n";return 0;}
    std::cout<<"[FAIL]\n"; return 1;
}
