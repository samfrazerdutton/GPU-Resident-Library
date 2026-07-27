// SlotsToCoeffs: inverse of C2S, ANALYTIC matrices:
//   A'[i][k] = (W[i][k] - i*W[i][k+S])/2 ; B'[i][k] = (W[i][k] + i*W[i][k+S])/2
// Gate: slots y (arbitrary complex) -> homomorphic A'*y + B'*conj(y) -> slots
// must equal the numeric A'y+B'conj(y) (== decode of the unpacked coeff poly).
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
void automorphism_eval_host(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t,
                            const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void rotate_ct_host(std::vector<uint64_t>&, std::vector<uint64_t>&, uint32_t,
                    const KeySwitchConstants&, uint32_t, uint32_t,
                    const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void pt_to_eval_host(std::vector<uint64_t>&, const std::vector<int64_t>&, uint32_t, uint32_t,
                     const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void ct_mul_pt_host(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<uint64_t>&,
                    uint32_t, uint32_t, const std::vector<uint64_t>&);
void ct_add_ct_host(std::vector<uint64_t>&, std::vector<uint64_t>&,
                    const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                    uint32_t, uint32_t, const std::vector<uint64_t>&);
}
static uint64_t mmu(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
using cd=std::complex<double>;

int main(){
    const uint32_t n=1024,S=n/2,sizeQ=4,sizeP=2,numPart=2,M=2*n; const uint64_t ns=1;
    std::vector<uint64_t> mod,modP;
    gpufhe::native_primes(mod,sizeQ,50,n,{});
    gpufhe::native_primes(modP,sizeP,51,n,mod);
    std::vector<uint64_t> modQP=mod; for(auto p:modP)modQP.push_back(p);
    std::vector<uint64_t> root(sizeQ),rootQP;
    for(uint32_t i=0;i<sizeQ;++i)root[i]=gpufhe::native_root(n,mod[i]);
    rootQP=root; for(auto p:modP)rootQP.push_back(gpufhe::native_root(n,p));
    const uint32_t sizeQlP=sizeQ+sizeP;
    auto KPqp=gpufhe::keygen_host(n,modQP,rootQP,ns,3.19,101);
    auto KPq =gpufhe::keygen_host(n,mod,root,ns,3.19,101);
    std::vector<uint64_t> PModq_QP(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=modQP[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mmu(P,modQP[sizeQ+j]%q,q);PModq_QP[t]=P;}
    auto mkKidx=[&](uint32_t k,uint32_t seed)->gpufhe::KeySwitchConstants{
        std::vector<uint64_t> sA=KPqp.s;
        gpufhe::automorphism_eval_host(sA,sizeQlP,n,k,modQP,rootQP);
        gpufhe::KeySwitchConstants K; K.n=n;
        gpufhe::compute_keyswitch_constants(K,mod,modP,numPart);
        for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
        for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modQP[sizeQ+j]);K.rootValList.push_back(rootQP[sizeQ+j]);}
        gpufhe::evalkeygen_host_sold(K,KPqp.s,sA,KPqp.pkA,KPqp.pkB,PModq_QP,modQP,rootQP,ns,3.19,seed);
        return K; };
    const double Delta=std::pow(2.0,35);

    // W row entries + analytic A'/B'
    std::vector<uint64_t> rk(S); {uint64_t r=1;for(uint32_t k=0;k<S;++k){rk[k]=r;r=(r*5)%M;}}
    auto Went=[&](uint32_t i,uint32_t j)->cd{ return std::polar(1.0, M_PI*(double)(((uint64_t)j*rk[i])%M)/(double)n); };
    auto Ap=[&](uint32_t i,uint32_t k)->cd{ return (Went(i,k)-cd{0,1}*Went(i,k+S))*0.5; };
    auto Bp=[&](uint32_t i,uint32_t k)->cd{ return (Went(i,k)+cd{0,1}*Went(i,k+S))*0.5; };

    // input slots y + numeric reference z = A'y + B'conj(y)
    std::vector<cd> y(S);
    for(uint32_t i=0;i<S;++i) y[i]={0.02*std::sin(0.01*i),0.02*std::cos(0.017*i)};
    std::vector<cd> zref(S);
    for(uint32_t i=0;i<S;++i){ cd acc{0,0};
        for(uint32_t k=0;k<S;++k) acc+=Ap(i,k)*y[k]+Bp(i,k)*std::conj(y[k]);
        zref[i]=acc; }

    std::vector<int64_t> my; gpufhe::encode_host(my,y,n,Delta);
    std::vector<uint64_t> c0,c1;
    gpufhe::encrypt_host(c0,c1,my,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,404);
    std::vector<uint64_t> j0=c0,j1=c1;
    { auto Kc=mkKidx(M-1,7100); gpufhe::rotate_ct_host(j0,j1,M-1,Kc,sizeQ,n,mod,root); }

    const size_t T=(size_t)sizeQ*n;
    std::vector<uint64_t> acc0(T,0),acc1(T,0); bool first=true;
    auto addLT=[&](const std::vector<uint64_t>& s0,const std::vector<uint64_t>& s1,bool isB){
        for(uint32_t d=0;d<S;++d){
            std::vector<cd> diag(S); double nz=0;
            for(uint32_t i=0;i<S;++i){ uint32_t c=(i+d)%S; diag[i]=isB?Bp(i,c):Ap(i,c); nz+=std::abs(diag[i]); }
            if(nz<1e-9) continue;
            std::vector<int64_t> md; gpufhe::encode_host(md,diag,n,Delta);
            std::vector<uint64_t> dE; gpufhe::pt_to_eval_host(dE,md,sizeQ,n,mod,root);
            std::vector<uint64_t> b0=s0,b1=s1;
            if(d>0){ uint64_t kk=1;for(uint32_t t=0;t<d;++t)kk=(kk*5)%M;
                auto Kj=mkKidx((uint32_t)kk,(isB?9500:8500)+d);
                gpufhe::rotate_ct_host(b0,b1,(uint32_t)kk,Kj,sizeQ,n,mod,root); }
            gpufhe::ct_mul_pt_host(b0,b1,dE,sizeQ,n,mod);
            if(first){acc0=b0;acc1=b1;first=false;}
            else gpufhe::ct_add_ct_host(acc0,acc1,b0,b1,sizeQ,n,mod);
        } };
    addLT(c0,c1,false); addLT(j0,j1,true);

    gpufhe::KeySwitchConstants Kr; Kr.n=n;
    gpufhe::compute_keyswitch_constants(Kr,mod,modP,numPart);
    for(uint32_t i=0;i<sizeQ;++i){Kr.rootModList.push_back(mod[i]);Kr.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){Kr.rootModList.push_back(modQP[sizeQ+j]);Kr.rootValList.push_back(rootQP[sizeQ+j]);}
    Kr.av.assign(Kr.numPart,{});Kr.bv.assign(Kr.numPart,{});Kr.evalKeyTowers=sizeQlP;
    for(uint32_t p=0;p<Kr.numPart;++p){Kr.av[p].assign((size_t)sizeQlP*n,0);Kr.bv[p].assign((size_t)sizeQlP*n,0);}
    auto C=gpufhe::ks_context_create(Kr);
    std::vector<uint64_t> rs1,rs2; gpufhe::native_rescale_consts(rs1,rs2,mod,sizeQ-1);
    uint64_t *dr0,*dr1,*scr,*drp;
    cudaMalloc(&dr0,T*8);cudaMalloc(&dr1,T*8);cudaMalloc(&scr,(size_t)n*8);cudaMalloc(&drp,(size_t)n*8);
    cudaMemcpy(dr0,acc0.data(),T*8,cudaMemcpyHostToDevice);cudaMemcpy(dr1,acc1.data(),T*8,cudaMemcpyHostToDevice);
    gpufhe::rescale_resident_raw(dr0,sizeQ,C,rs1,rs2,scr,drp,0);
    gpufhe::rescale_resident_raw(dr1,sizeQ,C,rs1,rs2,scr,drp,0);
    cudaDeviceSynchronize();
    cudaMemcpy(acc0.data(),dr0,T*8,cudaMemcpyDeviceToHost);cudaMemcpy(acc1.data(),dr1,T*8,cudaMemcpyDeviceToHost);
    std::vector<uint64_t> modR(mod.begin(),mod.end()-1),rootR(root.begin(),root.end()-1);
    auto KPr=gpufhe::keygen_host(n,modR,rootR,ns,3.19,101);
    std::vector<int64_t> dec; gpufhe::decrypt_host(dec,acc0,acc1,KPr.s,n,modR,rootR);
    double DeltaOut=Delta*Delta/(double)mod[sizeQ-1];
    std::vector<cd> z; gpufhe::decode_host(z,dec,n,DeltaOut);

    double maxerr=0;
    for(uint32_t i=0;i<S;++i) maxerr=std::max(maxerr,std::abs(z[i]-zref[i]));
    std::cout<<"S2C max err="<<maxerr<<"  z[0]="<<z[0]<<" ref="<<zref[0]<<"\n";
    if(maxerr<5e-3){std::cout<<"[PASS] SlotsToCoeffs: slots homomorphically moved back to coefficients\n";return 0;}
    std::cout<<"[FAIL]\n"; return 1;
}
