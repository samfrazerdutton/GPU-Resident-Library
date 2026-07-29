// FULL COMPOSED BOOTSTRAP: ModRaise -> C2S -> real/imag split.
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
#include <functional>
#include <chrono>
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
    const uint32_t HW=64;
    gpufhe::set_secret_hamming_weight(HW);   // sparse secret: pins K independent of n
    const uint32_t n=1024,S=n/2,sizeQ=30,sizeP=2,M=2*n; const uint64_t ns=1;
    // dnum: alpha=10 towers per part => 3 parts at tw=30 (was alpha=2 => 15 parts).
    // P must cover a part (10x50=500 bits), hence sizeP=9 x 60 = 540 bits.
    // At n=32768 this is what makes a rotation key ~61MB instead of ~251MB.
    auto npFor=[](uint32_t tw)->uint32_t{ return tw<=2?1u:(tw+1)/2; };  // alpha=2: the partitioning everything was gated at
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
    // ---- FULL-QP KEYS: generated ONCE, viewed at every level via delta ----
    gpufhe::KeySwitchConstants KREL; KREL.n=n;
    gpufhe::compute_keyswitch_constants(KREL,mod,modP,npFor(sizeQ));
    for(uint32_t i=0;i<sizeQ;++i){KREL.rootModList.push_back(mod[i]);KREL.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){KREL.rootModList.push_back(modP[j]);KREL.rootValList.push_back(rootQP[sizeQ+j]);}
    gpufhe::evalkeygen_host(KREL,KPqp.s,KPqp.pkA,KPqp.pkB,PModq,modQP,rootQP,ns,3.19,202);
    std::map<uint32_t,gpufhe::KeySwitchConstants> KROT;
    uint32_t nKeygen=1; double tKeygen=0,tC2S=0,tEM=0,tS2C=0;
    auto NOW=[](){return std::chrono::steady_clock::now();};
    auto EL=[](auto a){return std::chrono::duration<double>(std::chrono::steady_clock::now()-a).count();};
    auto rotFull=[&](uint32_t k,uint32_t seed)->const gpufhe::KeySwitchConstants&{
        auto it=KROT.find(k); if(it!=KROT.end())return it->second;
        auto _t=NOW();
        std::vector<uint64_t> sA=KPqp.s;
        gpufhe::automorphism_eval_host(sA,sizeQlP,n,k,modQP,rootQP);
        gpufhe::KeySwitchConstants K; K.n=n;
        gpufhe::compute_keyswitch_constants(K,mod,modP,npFor(sizeQ));
        for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
        for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rootQP[sizeQ+j]);}
        gpufhe::evalkeygen_host_sold(K,KPqp.s,sA,KPqp.pkA,KPqp.pkB,PModq,modQP,rootQP,ns,3.19,seed);
        ++nKeygen; tKeygen+=EL(_t); KROT.emplace(k,std::move(K)); return KROT[k]; };
    auto atLevel=[&](const gpufhe::KeySwitchConstants& KF,uint32_t tw)->gpufhe::KeySwitchConstants{
        std::vector<uint64_t> ml(mod.begin(),mod.begin()+tw),rl(root.begin(),root.begin()+tw);
        gpufhe::KeySwitchConstants K; K.n=n;
        gpufhe::compute_keyswitch_constants(K,ml,modP,npFor(tw));
        for(uint32_t i=0;i<tw;++i){K.rootModList.push_back(ml[i]);K.rootValList.push_back(rl[i]);}
        for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rootQP[sizeQ+j]);}
        K.fullQ=sizeQ; K.evalKeyTowers=KF.evalKeyTowers;
        K.av.assign(K.numPart,{}); K.bv.assign(K.numPart,{});
        for(uint32_t p=0;p<K.numPart;++p){K.av[p]=KF.av[p];K.bv[p]=KF.bv[p];}
        return K; };

    auto mkKidx=[&](uint32_t k,uint32_t seed)->gpufhe::KeySwitchConstants{
        return atLevel(rotFull(k,seed),sizeQ); };

    // Encoded coeffs scale as |mz| ~ Delta_in*|z|*sqrt(2/N), so at fixed
    // Delta_in the per-coefficient signal SHRINKS like 1/sqrt(n) -- and the
    // EvalMod signal is 2pi*mz/q0, so SNR degrades with the ring. Scale
    // Delta_in with sqrt(n) to hold |mz| constant. Ceiling is the sine
    // approximation: need mz/q0 << 1 (cubic term), and we sit at ~2^-9 here
    // versus a tolerance around 0.02, so there is ample room.
    const double Delta_in=std::pow(2.0,54)*std::sqrt((double)n/1024.0)*8.0;
    const double Delta_pt=std::pow(2.0,40);

    // ---- message, encrypt at the BOTTOM level (q0 only)
    std::vector<cd> z(S);
    for(uint32_t i=0;i<S;++i) z[i]={0.35*std::sin(0.02*i),0.25*std::cos(0.017*i)};
    std::vector<int64_t> mz; gpufhe::encode_host(mz,z,n,Delta_in);
    std::vector<uint64_t> c0,c1;
    gpufhe::encrypt_host(c0,c1,mz,KP1.pkA,KP1.pkB,n,mod1,root1,ns,3.19,303);

    auto tBoot=std::chrono::steady_clock::now();
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
        std::vector<int64_t> dec1; gpufhe::decrypt_host(dec1,c0,c1,KP1.s,n,mod1,root1);
        uint32_t bad=0; __int128 maxI=0;
        for(uint32_t j=0;j<n;++j){ __int128 d=aref[j]-(__int128)dec1[j];
            if(d%(__int128)q0!=0) ++bad;
            else { __int128 I=d/(__int128)q0; if(I<0)I=-I; if(I>maxI)maxI=I; } }
        std::cout<<"ModRaise: congruence mismatches = "<<bad<<" (want 0), max|I| = "<<(double)(long long)maxI<<"\n";
    }

    // ---- C2S matrices [A B] = P * inv([W; conj W])
    std::vector<uint64_t> rk(S); {uint64_t r=1;for(uint32_t k=0;k<S;++k){rk[k]=r;r=(r*5)%M;}}
    const uint32_t N=n;
    // The rows of [W; conj W] are ORTHOGONAL: for two odd powers r,r',
    // sum_j e^{i*pi*j(r-r')/N} = 0 (r-r' is even so the numerator vanishes) and
    // = N when r=r'. So the matrix is sqrt(N)*unitary and its inverse is just
    // (1/N)*conjugate transpose -- ANALYTIC, O(1) per entry, zero storage.
    // The old NxN complex Gauss-Jordan was 3.5e13 ops and 17GB at n=32768.
    auto Wrow=[&](uint32_t r,uint32_t j)->cd{        // row r, col j of [W; conj W]
        uint32_t k=(r<S)? r : r-S;
        cd w=std::polar(1.0, M_PI*(double)(((uint64_t)j*rk[k])%M)/(double)N);
        return (r<S)? w : std::conj(w); };
    auto Aent=[&](uint32_t i,uint32_t c)->cd{        // = Inv[i][c] + i*Inv[i+S][c]
        return (std::conj(Wrow(c,i)) + cd{0,1}*std::conj(Wrow(c,i+S)))/(double)N; };
    { // self-check: (Wf * Inv)[a][b] must be delta_ab
      double offmax=0, diagerr=0;
      for(uint32_t t=0;t<6;++t){
        uint32_t a=(t*37)%N, b=(t*91+5)%N;
        cd sd{0,0}, so{0,0};
        for(uint32_t c=0;c<N;++c){ sd+=Wrow(a,c)*std::conj(Wrow(a,c))/(double)N;
                                   so+=Wrow(a,c)*std::conj(Wrow(b,c))/(double)N; }
        diagerr=std::max(diagerr,std::abs(sd-cd{1,0}));
        if(a!=b) offmax=std::max(offmax,std::abs(so)); }
      std::cout<<"analytic inverse self-check: diag err="<<diagerr<<" offdiag="<<offmax<<"\n"; }

    // ---- BSGS linear transform (46 keys, shared across branches)
    const uint32_t n1=32,n2=S/n1;
    std::map<uint32_t,gpufhe::KeySwitchConstants> Kc;
    auto rotAmt=[&](uint32_t r)->uint32_t{uint64_t k=1;for(uint32_t t=0;t<r;++t)k=(k*5)%M;return (uint32_t)k;};
    auto keyFor=[&](uint32_t r)->gpufhe::KeySwitchConstants&{
        auto it=Kc.find(r); if(it!=Kc.end())return it->second;
        Kc.emplace(r,mkKidx(rotAmt(r),4000+r)); return Kc[r]; };
    const size_t T=(size_t)sizeQ*n; (void)T;
    // ---- C2S via L merged FFT stages (replaces the dense S-diagonal transform).
    // Stage s: h=S>>(s+1), blocks of 2h, twiddle at position k is pts[k]^(2^s).
    // Merging r=lgS/L consecutive stages flips r consecutive index bits => 2^r
    // diagonals for the first merged stage, 2^(r+1)-1 after. Gated against the
    // dense matrix in test_c2s_stages (1.5e-16). Output is BIT-REVERSED in slot
    // order; the permutation is never applied because everything up to S2C is
    // elementwise in slot index, so S2C absorbs it via brev() on its columns.
    uint32_t lgS=0; while((1u<<lgS)<S) ++lgS;
    const uint32_t LST=3, RST=lgS/LST;
    auto brev=[&](uint32_t i){uint32_t r=0;for(uint32_t b=0;b<lgS;++b) if(i&(1u<<b)) r|=1u<<(lgS-1-b); return r;};
    auto twd=[&](uint32_t k,uint32_t st)->cd{ uint64_t e=((uint64_t)rk[k]<<st)%M;
        return std::polar(1.0, 2*M_PI*(double)e/(double)M); };
    auto applyMerged=[&](std::vector<cd>& v,uint32_t g){
        for(uint32_t st=g*RST;st<(g+1)*RST;++st){
            uint32_t h=S>>(st+1); std::vector<cd> o(S);
            for(uint32_t base=0;base<S;base+=2*h)
                for(uint32_t k=0;k<h;++k){
                    cd a=v[base+k],b=v[base+k+h],t=twd(k,st);
                    o[base+k]=(a+b)*0.5; o[base+k+h]=(a-b)/(2.0*t);}
            v.swap(o);} };
    auto stageDiags=[&](uint32_t g,std::vector<uint32_t>& offs,std::vector<std::vector<cd>>& dg){
        std::vector<std::vector<cd>> acc(S); std::vector<uint8_t> used(S,0);
        for(uint32_t jj=0;jj<S;++jj){
            std::vector<cd> u(S,cd{0,0}); u[jj]=1; applyMerged(u,g);
            for(uint32_t ii=0;ii<S;++ii) if(std::abs(u[ii])>1e-12){
                uint32_t d=(jj+S-ii)%S;
                if(!used[d]){used[d]=1;acc[d].assign(S,cd{0,0});}
                acc[d][ii]=u[ii]; } }
        offs.clear(); dg.clear();
        for(uint32_t d=0;d<S;++d) if(used[d]){offs.push_back(d);dg.push_back(acc[d]);} };
    std::map<uint64_t,gpufhe::KeySwitchConstants> KSC;
    auto keyAt=[&](uint32_t r,uint32_t tw)->gpufhe::KeySwitchConstants&{
        uint64_t kk=((uint64_t)r<<8)|tw; auto it=KSC.find(kk); if(it!=KSC.end())return it->second;
        // alpha=2 tolerates key reuse across levels (the last part mismatches by
        // at most 1 tower); it is alpha=10 that breaks. So take a level VIEW of
        // one full-QP key instead of regenerating -- fresh keys per level cost
        // up to 105 keygens here with 14 parts.
        KSC.emplace(kk,atLevel(rotFull(rotAmt(r),4000+r),tw)); return KSC[kk]; };
    auto rescaleAt=[&](std::vector<uint64_t>& a0,std::vector<uint64_t>& a1,uint32_t tw){
        std::vector<uint64_t> ml(mod.begin(),mod.begin()+tw),rl(root.begin(),root.begin()+tw);
        gpufhe::KeySwitchConstants Kd; Kd.n=n;
        gpufhe::compute_keyswitch_constants(Kd,ml,modP,npFor(tw));
        for(uint32_t i2=0;i2<tw;++i2){Kd.rootModList.push_back(ml[i2]);Kd.rootValList.push_back(rl[i2]);}
        for(uint32_t j2=0;j2<sizeP;++j2){Kd.rootModList.push_back(modP[j2]);Kd.rootValList.push_back(rootQP[sizeQ+j2]);}
        Kd.av.assign(Kd.numPart,{});Kd.bv.assign(Kd.numPart,{});Kd.evalKeyTowers=tw+sizeP;
        for(uint32_t p2=0;p2<Kd.numPart;++p2){Kd.av[p2].assign((size_t)(tw+sizeP)*n,0);Kd.bv[p2].assign((size_t)(tw+sizeP)*n,0);}
        auto Cc=gpufhe::ks_context_create(Kd);
        std::vector<uint64_t> s1,s2; gpufhe::native_rescale_consts(s1,s2,ml,tw-1);
        size_t TT=(size_t)tw*n; uint64_t *d0,*d1,*sc2,*dp2;
        cudaMalloc(&d0,TT*8);cudaMalloc(&d1,TT*8);cudaMalloc(&sc2,(size_t)n*8);cudaMalloc(&dp2,(size_t)n*8);
        cudaMemcpy(d0,a0.data(),TT*8,cudaMemcpyHostToDevice);cudaMemcpy(d1,a1.data(),TT*8,cudaMemcpyHostToDevice);
        gpufhe::rescale_resident_raw(d0,tw,Cc,s1,s2,sc2,dp2,0);
        gpufhe::rescale_resident_raw(d1,tw,Cc,s1,s2,sc2,dp2,0);
        cudaDeviceSynchronize();
        std::vector<uint64_t> g0(TT),g1(TT);
        cudaMemcpy(g0.data(),d0,TT*8,cudaMemcpyDeviceToHost);cudaMemcpy(g1.data(),d1,TT*8,cudaMemcpyDeviceToHost);
        cudaFree(d0);cudaFree(d1);cudaFree(sc2);cudaFree(dp2);gpufhe::ks_context_destroy(Cc);
        size_t T2b=(size_t)(tw-1)*n; a0.assign(g0.begin(),g0.begin()+T2b); a1.assign(g1.begin(),g1.begin()+T2b); };

    std::vector<uint64_t> cs0=R0, cs1=R1;
    uint32_t ctw=sizeQ; double csc=(double)q0;      // declared input scale
    uint32_t ndiagTot=0;
    { auto _t=NOW();
      for(uint32_t g=0;g<LST;++g){
        std::vector<uint32_t> offs; std::vector<std::vector<cd>> dg;
        stageDiags(g,offs,dg); ndiagTot+=(uint32_t)offs.size();
        std::vector<uint64_t> ml(mod.begin(),mod.begin()+ctw),rl(root.begin(),root.begin()+ctw);
        size_t TT=(size_t)ctw*n;
        std::vector<uint64_t> a0(TT,0),a1(TT,0); bool fst=true;
        // Each stage costs a rescale, so encoding every stage's diagonals at
        // Delta_pt would divide the scale by (Delta_pt/q) per stage and collapse
        // it (2^60 -> 2^30 at L=3), pushing it below Delta_w so that EvalMod's
        // ct*ct multiplies shrink it further into denormals -> inf. Hold the
        // scale FLAT through the intermediate stages by encoding at the modulus
        // being dropped; only the last stage does the 2^60 -> 2^50 step.
        const double dpt = (g+1<LST) ? (double)mod[ctw-1] : Delta_pt;
        for(size_t x=0;x<offs.size();++x){
            std::vector<int64_t> md; gpufhe::encode_host(md,dg[x],n,dpt);
            std::vector<uint64_t> dE; gpufhe::pt_to_eval_host(dE,md,ctw,n,ml,rl);
            std::vector<uint64_t> t0=cs0,t1=cs1;
            if(offs[x]) gpufhe::rotate_ct_host(t0,t1,rotAmt(offs[x]),keyAt(offs[x],ctw),ctw,n,ml,rl);
            gpufhe::ct_mul_pt_host(t0,t1,dE,ctw,n,ml);
            if(fst){a0=t0;a1=t1;fst=false;} else gpufhe::ct_add_ct_host(a0,a1,t0,t1,ctw,n,ml); }
        rescaleAt(a0,a1,ctw);
        csc=csc*dpt/(double)mod[ctw-1]; --ctw; cs0=a0; cs1=a1; }
      tC2S=EL(_t); }
    std::cout<<"C2S: "<<LST<<" FFT stages (radix 2^"<<RST<<"), "<<ndiagTot
             <<" diagonals vs "<<S<<" dense, rotation keys="<<KSC.size()<<"\n";
    uint32_t nt=ctw; size_t T2=(size_t)nt*n; (void)T2;
    std::vector<uint64_t> y0=cs0,y1=cs1;
    const double sOut=csc;
    std::vector<uint64_t> mf(mod.begin(),mod.begin()+nt),rf(root.begin(),root.begin()+nt);
    auto KPr=gpufhe::keygen_host(n,mf,rf,ns,3.19,101);

    auto peek=[&](const std::vector<uint64_t>&p0,const std::vector<uint64_t>&p1,double sc)->std::vector<cd>{
        std::vector<int64_t> d; gpufhe::decrypt_host(d,p0,p1,KPr.s,n,mf,rf);
        std::vector<cd> v; gpufhe::decode_host(v,d,n,sc); return v; };

    // ---- CHECK C2S: slots must be y_k = (a_k + i a_{k+S})/q0
    { auto v=peek(y0,y1,sOut);
      double e=0,mx=0; for(uint32_t k=0;k<S;++k){
          cd r{d128(aref[k])/(double)q0, d128(aref[k+S])/(double)q0};
          e=std::max(e,std::abs(v[brev(k)]-r)); mx=std::max(mx,std::abs(r)); }
      std::cout<<"C2S slots err = "<<e<<"  (max|y| = "<<mx<<" -> K budget)\n";
      std::cout<<"  y[0]="<<v[brev(0)]<<" ref=("<<d128(aref[0])/(double)q0<<","<<d128(aref[S])/(double)q0<<")\n"; }

    // ---- SPLIT: ctR = ct + conj(ct) (slots 2Re y) ; ctI = ct - conj(ct) (slots 2i Im y)
    std::vector<uint64_t> k0=y0,k1=y1;
    { auto Kl=atLevel(rotFull(M-1,7400),nt);
      gpufhe::rotate_ct_host(k0,k1,M-1,Kl,nt,n,mf,rf); }

    std::vector<uint64_t> r0=y0,r1=y1,i0=y0,i1=y1;
    gpufhe::ct_add_ct_host(r0,r1,k0,k1,nt,n,mf);           // 2*Re(y)
    for(uint32_t t=0;t<nt;++t){uint64_t q=mf[t];
        for(uint32_t j=0;j<n;++j){size_t x=(size_t)t*n+j;
            i0[x]=smu(i0[x],k0[x],q); i1[x]=smu(i1[x],k1[x],q);}}   // 2i*Im(y)


    // ================== BACK HALF: EvalMod x2 -> S2C -> recover ==================
    const double Delta_w=std::pow(2.0,50);
    struct Ct { std::vector<uint64_t> c0,c1; uint32_t tw; double scale; };

    auto mkKlvl=[&](uint32_t k,uint32_t tw,uint32_t seed)->gpufhe::KeySwitchConstants{
        return atLevel(rotFull(k,seed),tw); };
    // CONTROL: regenerate the relin key FRESH at each level (no reuse) to
    // separate "alpha=10 is broken" from "reuse across a numPart change is broken".
    auto buildK=[&](uint32_t tw)->gpufhe::KeySwitchConstants{
        std::vector<uint64_t> ml(mod.begin(),mod.begin()+tw),rl(root.begin(),root.begin()+tw);
        std::vector<uint64_t> mq(ml); for(auto p:modP)mq.push_back(p);
        std::vector<uint64_t> rq(rl); for(uint32_t j=0;j<sizeP;++j)rq.push_back(rootQP[sizeQ+j]);
        auto sl=[&](const std::vector<uint64_t>&src){std::vector<uint64_t> d((size_t)(tw+sizeP)*n);
            for(uint32_t t=0;t<tw;++t)std::copy(src.begin()+(size_t)t*n,src.begin()+(size_t)(t+1)*n,d.begin()+(size_t)t*n);
            for(uint32_t j=0;j<sizeP;++j)std::copy(src.begin()+(size_t)(sizeQ+j)*n,src.begin()+(size_t)(sizeQ+j+1)*n,d.begin()+(size_t)(tw+j)*n);
            return d;};
        gpufhe::KeySwitchConstants K; K.n=n;
        gpufhe::compute_keyswitch_constants(K,ml,modP,npFor(tw));
        for(uint32_t i=0;i<tw;++i){K.rootModList.push_back(ml[i]);K.rootValList.push_back(rl[i]);}
        for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rootQP[sizeQ+j]);}
        std::vector<uint64_t> PM(tw+sizeP);
        for(uint32_t t=0;t<tw+sizeP;++t){uint64_t q=mq[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mmu(P,modP[j]%q,q);PM[t]=P;}
        gpufhe::evalkeygen_host(K,sl(KPqp.s),sl(KPqp.pkA),sl(KPqp.pkB),PM,mq,rq,ns,3.19,202);
        return K; };
    auto rescIP=[&](std::vector<uint64_t>&a0,std::vector<uint64_t>&a1,uint32_t tw,const gpufhe::KeySwitchConstants&Kx){
        auto Cl=gpufhe::ks_context_create(Kx);
        std::vector<uint64_t> s1,s2; {std::vector<uint64_t> sub(mod.begin(),mod.begin()+tw);
            gpufhe::native_rescale_consts(s1,s2,sub,tw-1);}
        size_t TT=(size_t)tw*n; uint64_t *d0,*d1,*sc,*dp;
        cudaMalloc(&d0,TT*8);cudaMalloc(&d1,TT*8);cudaMalloc(&sc,(size_t)n*8);cudaMalloc(&dp,(size_t)n*8);
        cudaMemcpy(d0,a0.data(),TT*8,cudaMemcpyHostToDevice);cudaMemcpy(d1,a1.data(),TT*8,cudaMemcpyHostToDevice);
        gpufhe::rescale_resident_raw(d0,tw,Cl,s1,s2,sc,dp,0);
        gpufhe::rescale_resident_raw(d1,tw,Cl,s1,s2,sc,dp,0);
        cudaDeviceSynchronize();
        std::vector<uint64_t> g0(TT),g1(TT);
        cudaMemcpy(g0.data(),d0,TT*8,cudaMemcpyDeviceToHost);cudaMemcpy(g1.data(),d1,TT*8,cudaMemcpyDeviceToHost);
        cudaFree(d0);cudaFree(d1);cudaFree(sc);cudaFree(dp);gpufhe::ks_context_destroy(Cl);
        size_t T2b=(size_t)(tw-1)*n; a0.assign(g0.begin(),g0.begin()+T2b); a1.assign(g1.begin(),g1.begin()+T2b); };
    auto mulCt=[&](const Ct&A,const Ct&B)->Ct{
        uint32_t tw=A.tw; size_t TT=(size_t)tw*n;
        std::vector<uint64_t> t0(TT),t1(TT),t2(TT);
        for(uint32_t t=0;t<tw;++t){uint64_t q=mod[t];
            for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
                t0[x]=mmu(A.c0[x],B.c0[x],q);
                uint64_t p1=mmu(A.c0[x],B.c1[x],q),p2=mmu(A.c1[x],B.c0[x],q);
                uint64_t sm2=p1+p2; t1[x]=(sm2>=q)?sm2-q:sm2;
                t2[x]=mmu(A.c1[x],B.c1[x],q);}}
        auto Kl=buildK(tw); auto R=gpufhe::keyswitch_core_resident(t2,Kl);
        std::vector<uint64_t> r0b(TT),r1b(TT);
        for(uint32_t t=0;t<tw;++t){uint64_t q=mod[t];
            for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
                uint64_t u0=t0[x]+R.ba0[x]; r0b[x]=(u0>=q)?u0-q:u0;
                uint64_t u1=t1[x]+R.ba1[x]; r1b[x]=(u1>=q)?u1-q:u1;}}
        rescIP(r0b,r1b,tw,Kl);
        Ct o; o.tw=tw-1; o.scale=A.scale*B.scale/(double)mod[tw-1]; o.c0=r0b; o.c1=r1b; return o; };
    auto lvl=[&](const Ct&A,uint32_t tw)->Ct{ Ct o; o.tw=tw; o.scale=A.scale; size_t TT=(size_t)tw*n;
        o.c0.assign(A.c0.begin(),A.c0.begin()+TT); o.c1.assign(A.c1.begin(),A.c1.begin()+TT); return o; };
    auto x2=[&](Ct&A){ for(uint32_t t=0;t<A.tw;++t){uint64_t q=mod[t];
        for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k; A.c0[x]=(A.c0[x]*2)%q; A.c1[x]=(A.c1[x]*2)%q;}} };
    auto subC=[&](const Ct&A,const Ct&B)->Ct{ Ct o; o.tw=A.tw; o.scale=A.scale;
        size_t TT=(size_t)A.tw*n; o.c0.resize(TT);o.c1.resize(TT);
        for(uint32_t t=0;t<A.tw;++t){uint64_t q=mod[t];
            for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
                o.c0[x]=smu(A.c0[x],B.c0[x],q); o.c1[x]=smu(A.c1[x],B.c1[x],q);}} return o; };
    auto subK=[&](Ct&A,cd v){ std::vector<cd> zc(S,v); std::vector<int64_t> m2;
        gpufhe::encode_host(m2,zc,n,A.scale);
        std::vector<uint64_t> ml(mod.begin(),mod.begin()+A.tw),rl(root.begin(),root.begin()+A.tw);
        std::vector<uint64_t> e2; gpufhe::pt_to_eval_host(e2,m2,A.tw,n,ml,rl);
        for(uint32_t t=0;t<A.tw;++t){uint64_t q=mod[t];
            for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k; A.c0[x]=smu(A.c0[x],e2[x],q);}} };
    auto mulK=[&](const Ct&A,cd v)->Ct{ Ct o=A; std::vector<cd> zc(S,v); std::vector<int64_t> m2;
        gpufhe::encode_host(m2,zc,n,Delta_w);
        std::vector<uint64_t> ml(mod.begin(),mod.begin()+A.tw),rl(root.begin(),root.begin()+A.tw);
        std::vector<uint64_t> e2; gpufhe::pt_to_eval_host(e2,m2,A.tw,n,ml,rl);
        gpufhe::ct_mul_pt_host(o.c0,o.c1,e2,A.tw,n,ml);
        auto Kl=buildK(A.tw); rescIP(o.c0,o.c1,A.tw,Kl);
        o.tw=A.tw-1; o.scale=A.scale*Delta_w/(double)mod[A.tw-1]; return o; };

    // K = max|a_k/q0| grows like sqrt(n): uniform ternary s has ~n/3 nonzero
    // coeffs, so |I| ~ sqrt(n) and K doubles per 4x ring (29.2@n=1024,
    // 69.6@n=4096). The double-angle count must track it, since Chebyshev is
    // only valid for |y|<=1 and y = 2pi(x-1/4)/2^r:
    //     r = ceil(log2(2*pi*(K+0.25)))
    // Costs one extra level per 4x ring. (A SPARSE ternary secret, fixed
    // hamming weight ~64, would pin K independent of n — that is why standard
    // CKKS bootstrapping uses one.)
    // K = max|a_k/q0| is set by the SECRET's hamming weight, NOT by n: |I| ~
    // sqrt(h). Uniform ternary (h ~ n/3) makes K grow like sqrt(n) -- 29.2@1024,
    // 69.6@4096 -- but with fixed h it is n-independent (measured 9.49 at h=64).
    // Using the sqrt(n) law with a sparse secret picks r=9 where 7 suffices, and
    // each surplus doubling amplifies error ~4x (that cost a run: err 1.30).
    const double Kbud=(HW? 2.5*std::sqrt((double)HW) : 40.0*std::sqrt((double)n/1024.0));
    const uint32_t rDbl=(uint32_t)std::ceil(std::log2(2*M_PI*(Kbud+0.25))), degC=10;
    std::cout<<"K budget="<<Kbud<<" -> "<<rDbl<<" double-angle steps; depth="
             <<(1+1+(degC-1)+1+rDbl+1)<<" levels, output tw="<<((int)sizeQ-(int)(4+degC+rDbl))<<"\n";
    const double Aaff=2*M_PI/std::pow(2.0,rDbl), Baff=-(M_PI/2)/std::pow(2.0,rDbl);
    std::vector<double> cc(degC+1,0.0);
    { const uint32_t NQ=4*(degC+1);
      for(uint32_t k=0;k<=degC;++k){ double s2=0;
        for(uint32_t j=0;j<NQ;++j){double th=M_PI*(j+0.5)/NQ; s2+=std::cos(std::cos(th))*std::cos(k*th);}
        cc[k]=2.0*s2/NQ; } cc[0]*=0.5; }

    auto evalMod=[&](const Ct& X, cd aff)->Ct{
        Ct U=mulK(X,aff); subK(U,cd{-Baff,0});
        std::vector<Ct> Tk(degC+1); Tk[1]=U;
        { Ct sq=mulCt(U,U); x2(sq); subK(sq,cd{1,0}); Tk[2]=sq; }
        for(uint32_t k=3;k<=degC;++k){ Ct ul=lvl(U,Tk[k-1].tw);
            Ct P=mulCt(ul,Tk[k-1]); x2(P);
            Ct km2=lvl(Tk[k-2],P.tw); Tk[k]=subC(P,km2);
 }
        uint32_t low=Tk[degC].tw;
        std::vector<uint64_t> ml(mod.begin(),mod.begin()+low),rl(root.begin(),root.begin()+low);
        Ct acc; acc.tw=low; acc.scale=0; bool f1=true;
        for(uint32_t k=1;k<=degC;++k){ Ct t=lvl(Tk[k],low);
            std::vector<cd> zc(S,cd{cc[k],0}); std::vector<int64_t> mc;
            gpufhe::encode_host(mc,zc,n,Delta_w);
            std::vector<uint64_t> cE; gpufhe::pt_to_eval_host(cE,mc,low,n,ml,rl);
            gpufhe::ct_mul_pt_host(t.c0,t.c1,cE,low,n,ml);
            if(f1){acc.c0=t.c0;acc.c1=t.c1;acc.scale=t.scale*Delta_w;f1=false;}
            else gpufhe::ct_add_ct_host(acc.c0,acc.c1,t.c0,t.c1,low,n,ml); }
        { auto Kd=buildK(low); rescIP(acc.c0,acc.c1,low,Kd);
          acc.scale/=(double)mod[low-1]; acc.tw=low-1; }
        subK(acc,cd{-cc[0],0});
        for(uint32_t j=1;j<=rDbl;++j){ Ct sq=mulCt(acc,acc); x2(sq); subK(sq,cd{1,0}); acc=sq; }
        return acc; };

    { auto vr=peek(r0,r1,sOut), vi=peek(i0,i1,sOut);
      double mr=0,mi=0,er=0,ei=0;
      for(uint32_t k=0;k<S;++k){
          mr=std::max(mr,std::abs(vr[k])); mi=std::max(mi,std::abs(vi[k]));
          er=std::max(er,std::abs(vr[k].real()-2.0*d128(aref[brev(k)])/(double)q0));
          ei=std::max(ei,std::abs(vi[k].imag()-2.0*d128(aref[brev(k)+S])/(double)q0)); }
      std::cout<<"  SPLIT ctR max|slot|="<<mr<<" err="<<er
               <<" | ctI max|slot|="<<mi<<" err="<<ei<<"\n"; }
    Ct ctR; ctR.c0=r0; ctR.c1=r1; ctR.tw=nt; ctR.scale=sOut;
    Ct ctI; ctI.c0=i0; ctI.c1=i1; ctI.tw=nt; ctI.scale=sOut;
    std::cout<<"EvalMod on both branches (depth "<<(1+ (degC-1) +1+rDbl)<<" levels each)...\n"<<std::flush;
    auto _tem=NOW();
    Ct oR=evalMod(ctR, cd{Aaff/2.0,0});
    Ct oI=evalMod(ctI, cd{0,-Aaff/2.0});
    tEM=EL(_tem);
    std::cout<<"  EvalMod out tw="<<oR.tw<<" scale=2^"<<std::log2(oR.scale)<<"\n";
    { std::vector<uint64_t> mo(mod.begin(),mod.begin()+oR.tw),ro(root.begin(),root.begin()+oR.tw);
      auto KO=gpufhe::keygen_host(n,mo,ro,ns,3.19,101);
      std::vector<int64_t> d; gpufhe::decrypt_host(d,oR.c0,oR.c1,KO.s,n,mo,ro);
      std::vector<cd> v; gpufhe::decode_host(v,d,n,oR.scale);
      // slots are BIT-REVERSED after the staged C2S: slot k holds a_{brev(k)}
      double e=0; for(uint32_t k=0;k<S;++k) e=std::max(e,std::abs(v[k].real()-std::sin(2*M_PI*d128(aref[brev(k)])/(double)q0)));
      std::cout<<"  EvalMod(R) err vs sin(2pi a_k/q0) = "<<e<<"\n"; }

    // ---- S2C: z_i = sum_k W[i][k]*oR_k + sum_k W[i][k+S]*oI_k
    auto Went=[&](uint32_t i,uint32_t j)->cd{
        return std::polar(1.0, M_PI*(double)(((uint64_t)j*rk[i])%M)/(double)n); };
    // ---- S2C via the FORWARD FFT stages (mirror of C2S): stage order runs
    // L-1 -> 0, sub-stages high -> low, butterfly a +/- t*b. It consumes the
    // bit-reversed order C2S produces, so no brev() indexing is needed.
    // Both branches ride the SAME stages: fft(oR + i*oI) = fft(oR) + i*fft(oI),
    // so the i is folded into the I-branch's first-stage diagonals -> the two
    // merge during stage L-1 and the rest run on one ciphertext. Cost is L+1
    // transform passes, depth L. Round trip S2C(C2S(z))=z gated at 9.6e-16.
    auto applyMergedFwd=[&](std::vector<cd>& v,uint32_t g){
        for(int st=(int)((g+1)*RST)-1; st>=(int)(g*RST); --st){
            uint32_t h=S>>(st+1); std::vector<cd> o(S);
            for(uint32_t base=0;base<S;base+=2*h)
                for(uint32_t k=0;k<h;++k){
                    cd a=v[base+k],b=v[base+k+h],t=twd(k,(uint32_t)st);
                    o[base+k]=a+t*b; o[base+k+h]=a-t*b;}
            v.swap(o);} };
    auto stageDiagsFwd=[&](uint32_t g,std::vector<uint32_t>& offs,std::vector<std::vector<cd>>& dg){
        std::vector<std::vector<cd>> acc(S); std::vector<uint8_t> used(S,0);
        for(uint32_t jj=0;jj<S;++jj){
            std::vector<cd> u(S,cd{0,0}); u[jj]=1; applyMergedFwd(u,g);
            for(uint32_t ii=0;ii<S;++ii) if(std::abs(u[ii])>1e-12){
                uint32_t d=(jj+S-ii)%S;
                if(!used[d]){used[d]=1;acc[d].assign(S,cd{0,0});}
                acc[d][ii]=u[ii]; } }
        offs.clear(); dg.clear();
        for(uint32_t d=0;d<S;++d) if(used[d]){offs.push_back(d);dg.push_back(acc[d]);} };

    uint32_t stw=oR.tw; double ssc=oR.scale; uint32_t s2cdiag=0;
    std::vector<uint64_t> sa0,sa1;
    { auto _t=NOW();
      for(int g=(int)LST-1; g>=0; --g){
        std::vector<uint32_t> offs; std::vector<std::vector<cd>> dg;
        stageDiagsFwd((uint32_t)g,offs,dg); s2cdiag+=(uint32_t)offs.size();
        std::vector<uint64_t> ml(mod.begin(),mod.begin()+stw),rl(root.begin(),root.begin()+stw);
        size_t TT=(size_t)stw*n;
        std::vector<uint64_t> a0(TT,0),a1(TT,0); bool fst=true;
        const double dpt = (g>0) ? (double)mod[stw-1] : Delta_pt;   // hold scale flat
        std::vector<std::pair<const std::vector<uint64_t>*,cd>> srcs0, srcs1;
        if((uint32_t)g==LST-1){ srcs0={{&oR.c0,cd{1,0}},{&oI.c0,cd{0,1}}};
                                srcs1={{&oR.c1,cd{1,0}},{&oI.c1,cd{0,1}}}; }
        else { srcs0={{&sa0,cd{1,0}}}; srcs1={{&sa1,cd{1,0}}}; }
        for(size_t q2=0;q2<srcs0.size();++q2)
          for(size_t x=0;x<offs.size();++x){
            std::vector<cd> dd(S);
            for(uint32_t i2=0;i2<S;++i2) dd[i2]=dg[x][i2]*srcs0[q2].second;
            std::vector<int64_t> md; gpufhe::encode_host(md,dd,n,dpt);
            std::vector<uint64_t> dE; gpufhe::pt_to_eval_host(dE,md,stw,n,ml,rl);
            std::vector<uint64_t> t0=*srcs0[q2].first, t1=*srcs1[q2].first;
            if(offs[x]) gpufhe::rotate_ct_host(t0,t1,rotAmt(offs[x]),keyAt(offs[x],stw),stw,n,ml,rl);
            gpufhe::ct_mul_pt_host(t0,t1,dE,stw,n,ml);
            if(fst){a0=t0;a1=t1;fst=false;} else gpufhe::ct_add_ct_host(a0,a1,t0,t1,stw,n,ml); }
        rescaleAt(a0,a1,stw);
        ssc=ssc*dpt/(double)mod[stw-1]; --stw; sa0=a0; sa1=a1; }
      tS2C=EL(_t); }
    std::cout<<"S2C: "<<LST<<" FFT stages, "<<s2cdiag<<" diagonals vs "<<S<<" dense\n";
    uint32_t twF=stw;
    double sFinal=ssc;
    double Dfinal=sFinal*2*M_PI*Delta_in/(double)q0;
    std::vector<uint64_t> mF(mod.begin(),mod.begin()+twF),rF(root.begin(),root.begin()+twF);
    auto KF=gpufhe::keygen_host(n,mF,rF,ns,3.19,101);
    std::vector<int64_t> dF; gpufhe::decrypt_host(dF,sa0,sa1,KF.s,n,mF,rF);
    std::vector<cd> zF; gpufhe::decode_host(zF,dF,n,Dfinal);

    double eFin=0; for(uint32_t k=0;k<S;++k) eFin=std::max(eFin,std::abs(zF[k]-z[k]));
    double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-tBoot).count();
    std::cout<<"\n=== BOOTSTRAP RESULT ===\n";
    std::cout<<"evalkeygen calls total = "<<nKeygen<<" (was ~130 with per-level regeneration)\n";
    std::cout<<"bootstrap wall time = "<<secs<<" s\n";
    std::cout<<"  keygen (48 keys) = "<<tKeygen<<" s\n";
    std::cout<<"  C2S              = "<<tC2S<<" s\n";
    std::cout<<"  EvalMod x2       = "<<tEM<<" s\n";
    std::cout<<"  S2C              = "<<tS2C<<" s\n";
    std::cout<<"  other            = "<<(secs-tC2S-tEM-tS2C)<<" s\n";
    std::cout<<"input level tw=1 (q0 only)  ->  output level tw="<<twF<<"\n";
    std::cout<<"final decode scale 2^"<<std::log2(Dfinal)<<"\n";
    std::cout<<"max slot err |boot(z) - z| = "<<eFin<<"\n";
    std::cout<<"  z[0]="<<z[0]<<"  got="<<zF[0]<<"\n";
    std::cout<<"  z[1]="<<z[1]<<"  got="<<zF[1]<<"\n";
    if(eFin<2e-2){std::cout<<"[PASS] FULL CKKS BOOTSTRAP: depleted ct refreshed to a higher level\n";return 0;}
    std::cout<<"[FAIL]\n"; return 1;
}
