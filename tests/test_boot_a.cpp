// COMPOSED BOOTSTRAP, FRONT HALF: ModRaise -> C2S -> real/imag split.
// Declared input scale s_in = q0, so C2S slots come out as y_k = a_k/q0 =
// (mz_k/q0) + I_k -- integer part I_k is exactly what EvalMod strips.
// Split: ctR = ct+conj(ct) (slots 2*Re y), ctI = ct-conj(ct) (slots 2i*Im y);
// the 1/2 and 1/(2i) get absorbed into EvalMod's affine multiply => 0 levels.
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
static uint64_t smu(uint64_t a,uint64_t b,uint64_t q){return a>=b?a-b:a+q-b;}
using cd=std::complex<double>;

int main(){
    const uint32_t n=1024,S=n/2,sizeQ=4,sizeP=2,M=2*n; const uint64_t ns=1;
    auto npFor=[](uint32_t tw)->uint32_t{ return tw<=2?1u:(tw+1)/2; };
    std::vector<uint64_t> mod,modP;
    gpufhe::native_primes(mod,1,60,n,{});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids,sizeQ-1,50,n,mod); for(auto m:mids)mod.push_back(m); }
    gpufhe::native_primes(modP,sizeP,60,n,mod);
    std::vector<uint64_t> modQP=mod; for(auto p:modP)modQP.push_back(p);
    std::vector<uint64_t> root(sizeQ),rootQP;
    for(uint32_t i=0;i<sizeQ;++i)root[i]=gpufhe::native_root(n,mod[i]);
    rootQP=root; for(auto p:modP)rootQP.push_back(gpufhe::native_root(n,p));
    const uint32_t sizeQlP=sizeQ+sizeP; const uint64_t q0=mod[0];
    std::vector<uint64_t> mod1{mod[0]},root1{root[0]};

    auto KPqp=gpufhe::keygen_host(n,modQP,rootQP,ns,3.19,101);
    auto KPq =gpufhe::keygen_host(n,mod,root,ns,3.19,101);
    auto KP1 =gpufhe::keygen_host(n,mod1,root1,ns,3.19,101);
    std::vector<uint64_t> PModq(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=modQP[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mmu(P,modQP[sizeQ+j]%q,q);PModq[t]=P;}
    auto mkKidx=[&](uint32_t k,uint32_t seed)->gpufhe::KeySwitchConstants{
        std::vector<uint64_t> sA=KPqp.s;
        gpufhe::automorphism_eval_host(sA,sizeQlP,n,k,modQP,rootQP);
        gpufhe::KeySwitchConstants K; K.n=n;
        gpufhe::compute_keyswitch_constants(K,mod,modP,npFor(sizeQ));
        for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
        for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rootQP[sizeQ+j]);}
        gpufhe::evalkeygen_host_sold(K,KPqp.s,sA,KPqp.pkA,KPqp.pkB,PModq,modQP,rootQP,ns,3.19,seed);
        return K; };

    const double Delta_in=std::pow(2.0,54), Delta_pt=std::pow(2.0,40);

    // ---- message, encrypt at the BOTTOM level (q0 only)
    std::vector<cd> z(S);
    for(uint32_t i=0;i<S;++i) z[i]={0.35*std::sin(0.02*i),0.25*std::cos(0.017*i)};
    std::vector<int64_t> mz; gpufhe::encode_host(mz,z,n,Delta_in);
    std::vector<uint64_t> c0,c1;
    gpufhe::encrypt_host(c0,c1,mz,KP1.pkA,KP1.pkB,n,mod1,root1,ns,3.19,303);

    // ---- MODRAISE: centered coeffs -> reduce into every tower + NTT
    std::vector<uint64_t> zero((size_t)n,0);
    std::vector<int64_t> a0,a1;
    gpufhe::decrypt_host(a0,c0,zero,KP1.s,n,mod1,root1);
    gpufhe::decrypt_host(a1,c1,zero,KP1.s,n,mod1,root1);
    std::vector<uint64_t> R0,R1;
    gpufhe::pt_to_eval_host(R0,a0,sizeQ,n,mod,root);
    gpufhe::pt_to_eval_host(R1,a1,sizeQ,n,mod,root);

    // reference a_j: per-tower decrypt of the raised ct, Garner over towers 0,1
    // (|a| <= 2^68 << (q0*q1)/2, and q0*q1 = 2^110 fits __int128)
    std::vector<__int128> aref(n);   // |a| ~ 2^68 -- MUST NOT be double or uint64
    auto d128=[](__int128 v)->double{ bool ng=v<0;
        unsigned __int128 u=ng?(unsigned __int128)(-v):(unsigned __int128)v;
        double r=(double)(uint64_t)(u>>64)*18446744073709551616.0+(double)(uint64_t)u;
        return ng?-r:r; };
    {
        std::vector<std::vector<uint64_t>> res(2);
        for(uint32_t t=0;t<2;++t){
            std::vector<uint64_t> A0(R0.begin()+(size_t)t*n,R0.begin()+(size_t)(t+1)*n);
            std::vector<uint64_t> A1(R1.begin()+(size_t)t*n,R1.begin()+(size_t)(t+1)*n);
            std::vector<uint64_t> st(KPq.s.begin()+(size_t)t*n,KPq.s.begin()+(size_t)(t+1)*n);
            std::vector<uint64_t> mt{mod[t]},rt{root[t]};
            std::vector<int64_t> d; gpufhe::decrypt_host(d,A0,A1,st,n,mt,rt);
            res[t].resize(n);
            for(uint32_t j=0;j<n;++j){long long v=d[j],q=(long long)mod[t];long long r=v%q;if(r<0)r+=q;res[t][j]=(uint64_t)r;} }
        auto inv=[](unsigned long long a,unsigned long long m)->unsigned long long{
            long long g=m,x=0,x1=1,a1=a%m;
            while(a1){long long qq=g/a1,t2=g-qq*a1;g=a1;a1=t2;long long t3=x-qq*x1;x=x1;x1=t3;}
            long long r=x%(long long)m; if(r<0)r+=m; return (unsigned long long)r; };
        unsigned __int128 prod=(unsigned __int128)mod[0]*mod[1], half=prod>>1;
        for(uint32_t j=0;j<n;++j){
            unsigned __int128 x=res[0][j];
            unsigned long long q1=mod[1], xm=(unsigned long long)(x%q1);
            unsigned long long diff=(res[1][j]+q1-xm)%q1;
            unsigned long long k=(unsigned long long)(((unsigned __int128)diff*inv(mod[0]%q1,q1))%q1);
            x += (unsigned __int128)mod[0]*k;
            aref[j] = (x>half)? -(__int128)(prod-x) : (__int128)x; }
        uint32_t bad=0; __int128 maxI=0;
        for(uint32_t j=0;j<n;++j){ __int128 d=aref[j]-(__int128)mz[j];
            if(d%(__int128)q0!=0) ++bad;
            else { __int128 I=d/(__int128)q0; if(I<0)I=-I; if(I>maxI)maxI=I; } }
        std::cout<<"ModRaise: congruence mismatches = "<<bad<<" (want 0), max|I| = "<<(double)(long long)maxI<<"\n";
    }

    // ---- C2S matrices [A B] = P * inv([W; conj W])
    std::vector<uint64_t> rk(S); {uint64_t r=1;for(uint32_t k=0;k<S;++k){rk[k]=r;r=(r*5)%M;}}
    const uint32_t N=n;
    std::vector<cd> Wf((size_t)N*N);
    for(uint32_t k=0;k<S;++k)for(uint32_t j=0;j<N;++j){
        cd w=std::polar(1.0, M_PI*(double)((j*rk[k])%M)/(double)N);
        Wf[(size_t)k*N+j]=w; Wf[(size_t)(k+S)*N+j]=std::conj(w); }
    std::vector<cd> Inv((size_t)N*N,{0,0}); for(uint32_t i=0;i<N;++i)Inv[(size_t)i*N+i]={1,0};
    for(uint32_t col=0;col<N;++col){
        uint32_t piv=col; double best=std::abs(Wf[(size_t)col*N+col]);
        for(uint32_t r=col+1;r<N;++r){double a=std::abs(Wf[(size_t)r*N+col]);if(a>best){best=a;piv=r;}}
        if(piv!=col)for(uint32_t j=0;j<N;++j){std::swap(Wf[(size_t)col*N+j],Wf[(size_t)piv*N+j]);std::swap(Inv[(size_t)col*N+j],Inv[(size_t)piv*N+j]);}
        cd d=Wf[(size_t)col*N+col];
        for(uint32_t j=0;j<N;++j){Wf[(size_t)col*N+j]/=d;Inv[(size_t)col*N+j]/=d;}
        for(uint32_t r=0;r<N;++r){ if(r==col)continue; cd f=Wf[(size_t)r*N+col]; if(std::abs(f)<1e-14)continue;
            for(uint32_t j=0;j<N;++j){Wf[(size_t)r*N+j]-=f*Wf[(size_t)col*N+j];Inv[(size_t)r*N+j]-=f*Inv[(size_t)col*N+j];} } }
    auto Aent=[&](uint32_t i,uint32_t c)->cd{ return Inv[(size_t)i*N+c]+cd{0,1}*Inv[(size_t)(i+S)*N+c]; };

    // ---- BSGS linear transform (46 keys, shared across branches)
    const uint32_t n1=32,n2=S/n1;
    std::map<uint32_t,gpufhe::KeySwitchConstants> Kc;
    auto rotAmt=[&](uint32_t r)->uint32_t{uint64_t k=1;for(uint32_t t=0;t<r;++t)k=(k*5)%M;return (uint32_t)k;};
    auto keyFor=[&](uint32_t r)->gpufhe::KeySwitchConstants&{
        auto it=Kc.find(r); if(it!=Kc.end())return it->second;
        Kc.emplace(r,mkKidx(rotAmt(r),4000+r)); return Kc[r]; };
    const size_t T=(size_t)sizeQ*n;
    std::vector<uint64_t> acc0(T,0),acc1(T,0); bool first=true;
    auto bsgsAdd=[&](const std::vector<uint64_t>& x0,const std::vector<uint64_t>& x1,bool isB){
        std::vector<std::vector<uint64_t>> B0(n1),B1(n1);
        B0[0]=x0;B1[0]=x1;
        for(uint32_t b=1;b<n1;++b){B0[b]=x0;B1[b]=x1;
            gpufhe::rotate_ct_host(B0[b],B1[b],rotAmt(b),keyFor(b),sizeQ,n,mod,root);}
        for(uint32_t g=0;g<n2;++g){
            std::vector<uint64_t> in0(T,0),in1(T,0); bool ifirst=true;
            for(uint32_t b=0;b<n1;++b){
                std::vector<cd> d(S); double nz=0;
                for(uint32_t i=0;i<S;++i){uint32_t r=(i+S-((g*n1)%S))%S,c=(i+b)%S;
                    d[i]=isB?Aent(r,c+S):Aent(r,c); nz+=std::abs(d[i]);}
                if(nz<1e-12)continue;
                std::vector<int64_t> md; gpufhe::encode_host(md,d,n,Delta_pt);
                std::vector<uint64_t> dE; gpufhe::pt_to_eval_host(dE,md,sizeQ,n,mod,root);
                std::vector<uint64_t> t0=B0[b],t1=B1[b];
                gpufhe::ct_mul_pt_host(t0,t1,dE,sizeQ,n,mod);
                if(ifirst){in0=t0;in1=t1;ifirst=false;}
                else gpufhe::ct_add_ct_host(in0,in1,t0,t1,sizeQ,n,mod);}
            if(ifirst)continue;
            if(g>0)gpufhe::rotate_ct_host(in0,in1,rotAmt(g*n1),keyFor(g*n1),sizeQ,n,mod,root);
            if(first){acc0=in0;acc1=in1;first=false;}
            else gpufhe::ct_add_ct_host(acc0,acc1,in0,in1,sizeQ,n,mod);} };

    // conj branch of the input
    std::vector<uint64_t> j0=R0,j1=R1;
    { auto Kc2=mkKidx(M-1,7300); gpufhe::rotate_ct_host(j0,j1,M-1,Kc2,sizeQ,n,mod,root); }
    bsgsAdd(R0,R1,false);
    bsgsAdd(j0,j1,true);
    std::cout<<"BSGS rotation keys built = "<<Kc.size()<<"\n";

    // one rescale -> declared scale q0*Delta_pt/qLast
    gpufhe::KeySwitchConstants Kr; Kr.n=n;
    gpufhe::compute_keyswitch_constants(Kr,mod,modP,npFor(sizeQ));
    for(uint32_t i=0;i<sizeQ;++i){Kr.rootModList.push_back(mod[i]);Kr.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){Kr.rootModList.push_back(modP[j]);Kr.rootValList.push_back(rootQP[sizeQ+j]);}
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
    std::vector<uint64_t> h0(T),h1(T);
    cudaMemcpy(h0.data(),dr0,T*8,cudaMemcpyDeviceToHost);cudaMemcpy(h1.data(),dr1,T*8,cudaMemcpyDeviceToHost);
    uint32_t nt=sizeQ-1; size_t T2=(size_t)nt*n;
    std::vector<uint64_t> y0(h0.begin(),h0.begin()+T2),y1(h1.begin(),h1.begin()+T2);
    const double sOut=(double)q0*Delta_pt/(double)mod[sizeQ-1];
    std::vector<uint64_t> mf(mod.begin(),mod.begin()+nt),rf(root.begin(),root.begin()+nt);
    auto KPr=gpufhe::keygen_host(n,mf,rf,ns,3.19,101);

    auto peek=[&](const std::vector<uint64_t>&p0,const std::vector<uint64_t>&p1,double sc)->std::vector<cd>{
        std::vector<int64_t> d; gpufhe::decrypt_host(d,p0,p1,KPr.s,n,mf,rf);
        std::vector<cd> v; gpufhe::decode_host(v,d,n,sc); return v; };

    // ---- CHECK C2S: slots must be y_k = (a_k + i a_{k+S})/q0
    { auto v=peek(y0,y1,sOut);
      double e=0,mx=0; for(uint32_t k=0;k<S;++k){
          cd r{d128(aref[k])/(double)q0, d128(aref[k+S])/(double)q0};
          e=std::max(e,std::abs(v[k]-r)); mx=std::max(mx,std::abs(r)); }
      std::cout<<"C2S slots err = "<<e<<"  (max|y| = "<<mx<<" -> K budget)\n";
      std::cout<<"  y[0]="<<v[0]<<" ref=("<<d128(aref[0])/(double)q0<<","<<d128(aref[S])/(double)q0<<")\n"; }

    // ---- SPLIT: ctR = ct + conj(ct) (slots 2Re y) ; ctI = ct - conj(ct) (slots 2i Im y)
    std::vector<uint64_t> k0=y0,k1=y1;
    { auto Kcj=mkKidx(M-1,7400);
      gpufhe::KeySwitchConstants Kl; Kl.n=n;
      gpufhe::compute_keyswitch_constants(Kl,mf,modP,npFor(nt));
      for(uint32_t i=0;i<nt;++i){Kl.rootModList.push_back(mf[i]);Kl.rootValList.push_back(rf[i]);}
      for(uint32_t j=0;j<sizeP;++j){Kl.rootModList.push_back(modP[j]);Kl.rootValList.push_back(rootQP[sizeQ+j]);}
      std::vector<uint64_t> sA=KPqp.s;
      gpufhe::automorphism_eval_host(sA,sizeQlP,n,M-1,modQP,rootQP);
      std::vector<uint64_t> sQPl((size_t)(nt+sizeP)*n),pAl((size_t)(nt+sizeP)*n),pBl((size_t)(nt+sizeP)*n),sOl((size_t)(nt+sizeP)*n);
      auto sl=[&](const std::vector<uint64_t>&src,std::vector<uint64_t>&dst){
          for(uint32_t t=0;t<nt;++t)std::copy(src.begin()+(size_t)t*n,src.begin()+(size_t)(t+1)*n,dst.begin()+(size_t)t*n);
          for(uint32_t j=0;j<sizeP;++j)std::copy(src.begin()+(size_t)(sizeQ+j)*n,src.begin()+(size_t)(sizeQ+j+1)*n,dst.begin()+(size_t)(nt+j)*n); };
      sl(KPqp.s,sQPl); sl(KPqp.pkA,pAl); sl(KPqp.pkB,pBl); sl(sA,sOl);
      std::vector<uint64_t> mq(mf); for(auto p:modP)mq.push_back(p);
      std::vector<uint64_t> rq(rf); for(uint32_t j=0;j<sizeP;++j)rq.push_back(rootQP[sizeQ+j]);
      std::vector<uint64_t> PM(nt+sizeP);
      for(uint32_t t=0;t<nt+sizeP;++t){uint64_t q=mq[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mmu(P,modP[j]%q,q);PM[t]=P;}
      gpufhe::evalkeygen_host_sold(Kl,sQPl,sOl,pAl,pBl,PM,mq,rq,ns,3.19,7400);
      gpufhe::rotate_ct_host(k0,k1,M-1,Kl,nt,n,mf,rf); }

    std::vector<uint64_t> r0=y0,r1=y1,i0=y0,i1=y1;
    gpufhe::ct_add_ct_host(r0,r1,k0,k1,nt,n,mf);           // 2*Re(y)
    for(uint32_t t=0;t<nt;++t){uint64_t q=mf[t];
        for(uint32_t j=0;j<n;++j){size_t x=(size_t)t*n+j;
            i0[x]=smu(i0[x],k0[x],q); i1[x]=smu(i1[x],k1[x],q);}}   // 2i*Im(y)

    double eR=0,eI=0;
    { auto vr=peek(r0,r1,sOut), vi=peek(i0,i1,sOut);
      for(uint32_t k=0;k<S;++k){
          eR=std::max(eR,std::abs(vr[k].real()-2.0*d128(aref[k])/(double)q0));
          eI=std::max(eI,std::abs(vi[k].imag()-2.0*d128(aref[k+S])/(double)q0)); } }
    std::cout<<"split: ctR err(vs 2*Re y) = "<<eR<<"   ctI err(vs 2i*Im y) = "<<eI<<"\n";
    std::cout<<"declared scales: s_in=q0=2^"<<std::log2((double)q0)
             <<"  Delta_pt=2^"<<std::log2(Delta_pt)<<"  sOut=2^"<<std::log2(sOut)<<"\n";
    if(eR<5e-2 && eI<5e-2){std::cout<<"[PASS] bootstrap front half: ModRaise -> C2S -> real/imag split\n";return 0;}
    std::cout<<"[FAIL]\n"; return 1;
}
