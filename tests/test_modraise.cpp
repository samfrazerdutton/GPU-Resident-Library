// MODRAISE: lift a ciphertext living mod q0 only into the full tower set Q.
// c0,c1 -> centered integer coefficients -> reduce mod every q_t -> NTT.
// The plaintext is preserved MOD q0, and the raised ct decrypts to
//   Delta*m + e + q0*I      (I a small integer polynomial)
// which is exactly the input EvalMod consumes. Gate: every coefficient of the
// raised decryption is congruent to the original mod q0 (exact integer check).
// 40-bit primes / L=3 so the centered value (<=2^49) stays inside int64.
#include "keyswitch.h"
#include "keygen.h"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cstdlib>
namespace gpufhe {
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void decode_host(std::vector<std::complex<double>>&, const std::vector<int64_t>&, uint32_t, double);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void pt_to_eval_host(std::vector<uint64_t>&, const std::vector<int64_t>&, uint32_t, uint32_t,
                     const std::vector<uint64_t>&, const std::vector<uint64_t>&);
}
using cd=std::complex<double>;

int main(){
    const uint32_t n=1024,S=n/2,L=3; const uint64_t ns=1;
    std::vector<uint64_t> mod; gpufhe::native_primes(mod,L,40,n,{});
    std::vector<uint64_t> root(L);
    for(uint32_t i=0;i<L;++i) root[i]=gpufhe::native_root(n,mod[i]);
    const uint64_t q0=mod[0];
    std::vector<uint64_t> mod1{mod[0]}, root1{root[0]};
    const double Delta=std::pow(2.0,25);

    auto KP1=gpufhe::keygen_host(n,mod1,root1,ns,3.19,101);
    auto KPL=gpufhe::keygen_host(n,mod,root,ns,3.19,101);

    std::vector<cd> z(S);
    for(uint32_t i=0;i<S;++i) z[i]={0.4*std::sin(0.02*i),0};
    std::vector<int64_t> mz; gpufhe::encode_host(mz,z,n,Delta);
    std::vector<uint64_t> c0,c1;
    gpufhe::encrypt_host(c0,c1,mz,KP1.pkA,KP1.pkB,n,mod1,root1,ns,3.19,303);

    // baseline: decrypt at the single tower
    std::vector<int64_t> dec1; gpufhe::decrypt_host(dec1,c0,c1,KP1.s,n,mod1,root1);
    std::vector<cd> y1; gpufhe::decode_host(y1,dec1,n,Delta);
    double e1=0; for(uint32_t i=0;i<S;++i) e1=std::max(e1,std::abs(y1[i].real()-z[i].real()));
    std::cout<<"baseline (tw=1) decode err = "<<e1<<"\n";

    // ---- MODRAISE ----
    // centered coefficients of each component: decrypt with a zero partner
    std::vector<uint64_t> zero((size_t)n,0);
    std::vector<int64_t> a0,a1;
    gpufhe::decrypt_host(a0,c0,zero,KP1.s,n,mod1,root1);
    gpufhe::decrypt_host(a1,c1,zero,KP1.s,n,mod1,root1);
    // reduce into every tower + NTT
    std::vector<uint64_t> C0,C1;
    gpufhe::pt_to_eval_host(C0,a0,L,n,mod,root);
    gpufhe::pt_to_eval_host(C1,a1,L,n,mod,root);

    // decrypt the raised ct PER TOWER, then CRT-reconstruct the true integer.
    // (decrypt_host on all L towers returns only the mod-q0 representative, so
    // the q0*I term is invisible to it -- reconstruct explicitly with Garner.)
    std::vector<std::vector<uint64_t>> res(L);
    for(uint32_t t=0;t<L;++t){
        std::vector<uint64_t> C0t(C0.begin()+(size_t)t*n, C0.begin()+(size_t)(t+1)*n);
        std::vector<uint64_t> C1t(C1.begin()+(size_t)t*n, C1.begin()+(size_t)(t+1)*n);
        std::vector<uint64_t> st (KPL.s.begin()+(size_t)t*n, KPL.s.begin()+(size_t)(t+1)*n);
        std::vector<uint64_t> mt{mod[t]}, rt{root[t]};
        std::vector<int64_t> d; gpufhe::decrypt_host(d,C0t,C1t,st,n,mt,rt);
        res[t].resize(n);
        for(uint32_t j=0;j<n;++j){ long long v=d[j]; long long q=(long long)mod[t];
            long long r=v%q; if(r<0)r+=q; res[t][j]=(uint64_t)r; }
    }
    auto invmod=[](unsigned long long a,unsigned long long m)->unsigned long long{
        long long g=m,x=0,x1=1,a1=a%m;
        while(a1){ long long qq=g/a1; long long t2=g-qq*a1; g=a1; a1=t2;
                   long long t3=x-qq*x1; x=x1; x1=t3; }
        long long r=x%(long long)m; if(r<0)r+=m; return (unsigned long long)r; };
    // Garner: x = r0 + q0*(...) ; intermediates stay under 128 bits
    std::vector<int64_t> vfull(n);
    for(uint32_t j=0;j<n;++j){
        unsigned __int128 x=res[0][j];
        unsigned __int128 prod=mod[0];
        for(uint32_t t=1;t<L;++t){
            unsigned long long qt=mod[t];
            unsigned long long xm=(unsigned long long)(x%qt);
            unsigned long long diff=(res[t][j]+qt-xm)%qt;
            unsigned long long pinv=invmod((unsigned long long)(prod%qt),qt);
            unsigned long long k=(unsigned long long)(((unsigned __int128)diff*pinv)%qt);
            x += prod*(unsigned __int128)k;
            prod *= qt;
        }
        // center mod Q
        unsigned __int128 half=prod>>1;
        long long centered = (x>half)? -(long long)(unsigned long long)(prod-x) : (long long)(unsigned long long)x;
        vfull[j]=centered;
    }

    uint32_t bad=0; long long maxI=0; uint32_t nonzeroI=0; long long maxV=0;
    for(uint32_t j=0;j<n;++j){
        long long d=vfull[j]-(long long)dec1[j];
        if(llabs(vfull[j])>maxV) maxV=llabs(vfull[j]);
        if(d%(long long)q0!=0){ if(bad<5) std::cout<<"  MISMATCH j="<<j<<" diff="<<d<<"\n"; ++bad; }
        else { long long I=d/(long long)q0; if(I!=0)++nonzeroI; if(llabs(I)>maxI)maxI=llabs(I); }
    }
    std::cout<<"q0=2^"<<std::log2((double)q0)<<"  Q=2^"
             <<(std::log2((double)mod[0])+std::log2((double)mod[1])+std::log2((double)mod[2]))<<"\n";
    std::cout<<"raised |value| max = 2^"<<std::log2((double)maxV)<<"  (mod-q0 rep was ~2^"<<std::log2((double)llabs(dec1[0]))<<")\n";
    std::cout<<"congruence-mod-q0 mismatches = "<<bad<<"\n";
    std::cout<<"q0*I term: "<<nonzeroI<<"/"<<n<<" coeffs nonzero, max|I| = "<<maxI<<"\n";
    std::cout<<"  dec1[0..2]="; for(int k=0;k<3;++k)std::cout<<dec1[k]<<" ";
    std::cout<<"\n  raised[0..2]="; for(int k=0;k<3;++k)std::cout<<vfull[k]<<" "; std::cout<<"\n";
    if(bad==0 && nonzeroI>0){
        std::cout<<"[PASS] ModRaise: plaintext preserved mod q0, q0*I term created (EvalMod's input)\n"; return 0; }
    if(bad==0){ std::cout<<"[PASS-weak] congruent but I==0 everywhere (no raise happened?)\n"; return 1; }
    std::cout<<"[FAIL]\n"; return 1;
}
