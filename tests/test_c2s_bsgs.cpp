// REAL CoeffsToSlots: y = A*z + B*conj(z), [A B] = P * inv([W; conj(W)]).
// Stage 1 gates homomorphic CONJUGATION (automorphism k=M-1). Stage 2 runs the
// full transform (2x 512-diagonal LTs + conj) and checks slots against the
// packed coefficients y_k=(m_k + i*m_{k+S})/Delta. OpenFHE-free, n=1024.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <map>
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

    // ---- STAGE 1: conjugation gate (k = M-1)
    {
        std::vector<cd> z(S);
        for(uint32_t i=0;i<S;++i) z[i]={0.3*std::sin(0.01*i),0.2*std::cos(0.013*i)};
        std::vector<int64_t> mz; gpufhe::encode_host(mz,z,n,Delta);
        std::vector<uint64_t> c0,c1;
        gpufhe::encrypt_host(c0,c1,mz,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,303);
        auto Kc=mkKidx(M-1,7000);
        gpufhe::rotate_ct_host(c0,c1,M-1,Kc,sizeQ,n,mod,root);
        std::vector<int64_t> dec; gpufhe::decrypt_host(dec,c0,c1,KPq.s,n,mod,root);
        std::vector<cd> zc; gpufhe::decode_host(zc,dec,n,Delta);
        double e=0; for(uint32_t i=0;i<S;++i) e=std::max(e,std::abs(zc[i]-std::conj(z[i])));
        std::cout<<"conjugation gate err="<<e<<"\n";
        if(e>1e-3){std::cout<<"[FAIL] conjugation\n";return 1;}
        std::cout<<"[PASS stage1] homomorphic conjugation\n";
    }

    // ---- STAGE 2: full C2S
    // build W and invert [W; conj W] (N x N complex Gaussian elimination)
    std::vector<uint64_t> rk(S); {uint64_t r=1;for(uint32_t k=0;k<S;++k){rk[k]=r;r=(r*5)%M;}}
    const uint32_t N=n;
    std::vector<cd> Wf((size_t)N*N);  // rows 0..S-1: W; rows S..N-1: conj(W)
    for(uint32_t k=0;k<S;++k)for(uint32_t j=0;j<N;++j){
        cd w=std::polar(1.0, M_PI*(double)((j*rk[k])%M)/(double)N);
        Wf[(size_t)k*N+j]=w; Wf[(size_t)(k+S)*N+j]=std::conj(w); }
    // invert via Gauss-Jordan into I
    std::vector<cd> Inv((size_t)N*N,{0,0}); for(uint32_t i=0;i<N;++i)Inv[(size_t)i*N+i]={1,0};
    for(uint32_t col=0;col<N;++col){
        uint32_t piv=col; double best=std::abs(Wf[(size_t)col*N+col]);
        for(uint32_t r=col+1;r<N;++r){double a=std::abs(Wf[(size_t)r*N+col]); if(a>best){best=a;piv=r;}}
        if(piv!=col){ for(uint32_t j=0;j<N;++j){std::swap(Wf[(size_t)col*N+j],Wf[(size_t)piv*N+j]);std::swap(Inv[(size_t)col*N+j],Inv[(size_t)piv*N+j]);} }
        cd d=Wf[(size_t)col*N+col];
        for(uint32_t j=0;j<N;++j){Wf[(size_t)col*N+j]/=d;Inv[(size_t)col*N+j]/=d;}
        for(uint32_t r=0;r<N;++r){ if(r==col)continue; cd f=Wf[(size_t)r*N+col]; if(std::abs(f)<1e-14)continue;
            for(uint32_t j=0;j<N;++j){Wf[(size_t)r*N+j]-=f*Wf[(size_t)col*N+j];Inv[(size_t)r*N+j]-=f*Inv[(size_t)col*N+j];} }
    }
    // [A B] = P * Inv : A[i][c]=Inv[c... wait P (SxN): row i has 1 at col i, i at col i+S.
    // (P*Inv)[i][c] = Inv[i][c] + i*Inv[i+S][c]; split c<S -> A, c>=S -> B? No:
    // y = [A B][z; conj z], columns 0..S-1 multiply z, S..N-1 multiply conj z.
    auto Aent=[&](uint32_t i,uint32_t c)->cd{ return Inv[(size_t)i*N+c]+cd{0,1}*Inv[(size_t)(i+S)*N+c]; };
    // A[i][c] = Aent(i,c) for c<S; B[i][c] = Aent(i,c+S)

    // input + reference
    std::vector<cd> z(S);
    for(uint32_t i=0;i<S;++i) z[i]={0.3*std::sin(0.01*i),0.15*std::cos(0.017*i)};
    std::vector<int64_t> mz; gpufhe::encode_host(mz,z,n,Delta);
    std::vector<cd> yref(S);
    for(uint32_t k=0;k<S;++k) yref[k]={(double)mz[k]/Delta,(double)mz[k+S]/Delta};

    std::vector<uint64_t> c0,c1;
    gpufhe::encrypt_host(c0,c1,mz,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,404);
    // conj branch
    std::vector<uint64_t> j0=c0,j1=c1;
    { auto Kc=mkKidx(M-1,7001); gpufhe::rotate_ct_host(j0,j1,M-1,Kc,sizeQ,n,mod,root); }

    const size_t T=(size_t)sizeQ*n;
    // ---- BSGS: j = g*n1 + b ; keys needed = b in [1,n1) and g*n1 for g in [1,n2)
    const uint32_t n1=32, n2=S/n1;
    std::map<uint32_t,gpufhe::KeySwitchConstants> Kc;   // rotation-amount -> key
    auto keyFor=[&](uint32_t r)->gpufhe::KeySwitchConstants&{
        auto it=Kc.find(r); if(it!=Kc.end()) return it->second;
        uint64_t k=1; for(uint32_t t=0;t<r;++t)k=(k*5)%M;
        Kc.emplace(r,mkKidx((uint32_t)k,4000+r)); return Kc[r]; };
    auto rotAmt=[&](uint32_t r)->uint32_t{ uint64_t k=1; for(uint32_t t=0;t<r;++t)k=(k*5)%M; return (uint32_t)k; };

    std::vector<uint64_t> acc0(T,0),acc1(T,0); bool first=true;
    auto bsgsLT=[&](const std::vector<uint64_t>& x0,const std::vector<uint64_t>& x1,bool isB){
        // baby steps rot_b(x), b=0..n1-1
        std::vector<std::vector<uint64_t>> B0(n1),B1(n1);
        B0[0]=x0; B1[0]=x1;
        for(uint32_t b=1;b<n1;++b){ B0[b]=x0; B1[b]=x1;
            gpufhe::rotate_ct_host(B0[b],B1[b],rotAmt(b),keyFor(b),sizeQ,n,mod,root); }
        for(uint32_t g=0;g<n2;++g){
            std::vector<uint64_t> in0(T,0),in1(T,0); bool ifirst=true;
            for(uint32_t b=0;b<n1;++b){
                std::vector<cd> d(S); double nz=0;
                for(uint32_t i=0;i<S;++i){
                    uint32_t r=(i+S-((g*n1)%S))%S, c=(i+b)%S;
                    d[i]=isB?Aent(r,c+S):Aent(r,c); nz+=std::abs(d[i]); }
                if(nz<1e-12) continue;
                std::vector<int64_t> md; gpufhe::encode_host(md,d,n,Delta);
                std::vector<uint64_t> dE; gpufhe::pt_to_eval_host(dE,md,sizeQ,n,mod,root);
                std::vector<uint64_t> t0=B0[b],t1=B1[b];
                gpufhe::ct_mul_pt_host(t0,t1,dE,sizeQ,n,mod);
                if(ifirst){in0=t0;in1=t1;ifirst=false;}
                else gpufhe::ct_add_ct_host(in0,in1,t0,t1,sizeQ,n,mod); }
            if(ifirst) continue;
            if(g>0) gpufhe::rotate_ct_host(in0,in1,rotAmt(g*n1),keyFor(g*n1),sizeQ,n,mod,root);
            if(first){acc0=in0;acc1=in1;first=false;}
            else gpufhe::ct_add_ct_host(acc0,acc1,in0,in1,sizeQ,n,mod); } };
    bsgsLT(c0,c1,false);
    bsgsLT(j0,j1,true);
    std::cout<<"BSGS n1="<<n1<<" n2="<<n2<<" rotation keys built = "<<Kc.size()<<"\n";

    // rescale + decrypt + decode
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
    std::vector<cd> y; gpufhe::decode_host(y,dec,n,DeltaOut);

    double maxerr=0;
    for(uint32_t k=0;k<S;++k) maxerr=std::max(maxerr,std::abs(y[k]-yref[k]));
    std::cout<<"C2S(BSGS) max err="<<maxerr<<"  y[0]="<<y[0]<<" ref="<<yref[0]<<"\n";
    if(maxerr<5e-3){std::cout<<"[PASS] CoeffsToSlots via BSGS: coefficients homomorphically moved into slots\n";return 0;}
    std::cout<<"[FAIL]\n"; return 1;
}
