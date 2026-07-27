// Native CKKS encrypt (pk) + decrypt, poly-arithmetic level (no complex
// encode/decode -- messages are integer coefficient polys, validated directly).
// Encrypt: c0 = p0*v + ns*e0 + m ; c1 = p1*v + ns*e1  (v ternary, e Gaussian).
// Decrypt: c0 + c1*s (+ c2*s^2 ... for >2 components). All eval form.
#include "keygen.h"
#include "ntt.h"
#include "intt.h"
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>

namespace gpufhe {
namespace {
uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
uint64_t am(uint64_t a,uint64_t b,uint64_t q){uint64_t s=a+b;return s>=q?s-q:s;}
uint64_t sm(uint64_t a,uint64_t b,uint64_t q){return a>=b?a-b:a+q-b;}
uint64_t* up(const std::vector<uint64_t>&h){uint64_t*d;cudaMalloc(&d,h.size()*8);cudaMemcpy(d,h.data(),h.size()*8,cudaMemcpyHostToDevice);return d;}
uint64_t powmod(uint64_t b,uint64_t e,uint64_t q){unsigned __int128 r=1,bb=b%q;while(e){if(e&1)r=(r*bb)%q;bb=(bb*bb)%q;e>>=1;}return(uint64_t)r;}
uint64_t invmod(uint64_t a,uint64_t q){return powmod(a%q,q-2,q);}
void xform(std::vector<uint64_t>&h,uint32_t n,uint64_t q,uint64_t root,bool inv){
    auto br=[](uint32_t v,uint32_t b){uint32_t r=0;for(uint32_t k=0;k<b;k++){r=(r<<1)|(v&1);v>>=1;}return r;};
    uint32_t logn=0;while((1u<<logn)<n)++logn;
    uint64_t base=inv?invmod(root,q):root;
    std::vector<uint64_t> tr(n),tp(n),pw(n);pw[0]=1;
    for(uint32_t i=1;i<n;++i)pw[i]=mm(pw[i-1],base,q);
    for(uint32_t i=0;i<n;++i){uint64_t r=pw[br(i,logn)];tr[i]=r;tp[i]=(uint64_t)(((__uint128_t)r<<64)/q);}
    uint64_t*dx=up(h),*dr=up(tr),*dp=up(tp);
    if(inv){ uint64_t ni=invmod(n,q),nip=(uint64_t)(((__uint128_t)ni<<64)/q);
        LaunchINTT_GS(dx,dr,dp,n,q,ni,nip,0); }
    else LaunchNTT_CT(dx,dr,dp,n,q,0);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(),dx,(size_t)n*8,cudaMemcpyDeviceToHost);
    cudaFree(dx);cudaFree(dr);cudaFree(dp);
}
} // anon

// Encrypt an integer-coefficient message m (coeff form, small ints centered) under
// pubkey (pkA,pkB). Returns (c0,c1) eval form, sizeQ*n each.
void encrypt_host(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1,
                  const std::vector<int64_t>& m,           // n coeffs (same across towers)
                  const std::vector<uint64_t>& pkA, const std::vector<uint64_t>& pkB,
                  uint32_t n, const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root,
                  uint64_t ns, double sigma, uint64_t seed)
{
    uint32_t sizeQ=(uint32_t)mod.size();
    std::mt19937_64 rng(seed); std::normal_distribution<double> g(0.0,sigma);
    std::vector<int> vc(n),e0c(n),e1c(n);
    for(uint32_t k=0;k<n;++k){ vc[k]=(int)(rng()%3)-1; e0c[k]=(int)std::lround(g(rng)); e1c[k]=(int)std::lround(g(rng)); }
    c0.resize((size_t)sizeQ*n); c1.resize((size_t)sizeQ*n);
    for(uint32_t t=0;t<sizeQ;++t){ uint64_t q=mod[t], rt=root[t];
        std::vector<uint64_t> vT(n),e0T(n),e1T(n),mT(n);
        for(uint32_t k=0;k<n;++k){ auto emb=[&](long x){return (uint64_t)((x%(long)q+(long)q)%(long)q);};
            vT[k]=emb(vc[k]); e0T[k]=emb(e0c[k]); e1T[k]=emb(e1c[k]); mT[k]=emb((long)m[k]); }
        xform(vT,n,q,rt,false); xform(e0T,n,q,rt,false); xform(e1T,n,q,rt,false); xform(mT,n,q,rt,false);
        for(uint32_t k=0;k<n;++k){
            uint64_t b=am(mm(pkB[(size_t)t*n+k],vT[k],q), mm(ns%q,e0T[k],q), q);
            uint64_t a=am(mm(pkA[(size_t)t*n+k],vT[k],q), mm(ns%q,e1T[k],q), q);
            c0[(size_t)t*n+k]=am(b,mT[k],q);   // c0 = p0*v + ns*e0 + m
            c1[(size_t)t*n+k]=a;               // c1 = p1*v + ns*e1
        }
    }
}

// Decrypt (c0,c1) -> message coeff poly (centered ints). phase = c0 + c1*s eval,
// INTT to coeff, center. s is sizeQ*n eval.
void decrypt_host(std::vector<int64_t>& mout,
                  const std::vector<uint64_t>& c0, const std::vector<uint64_t>& c1,
                  const std::vector<uint64_t>& s,
                  uint32_t n, const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root)
{
    // Use tower 0 only for recovery (message is same mod each q_i; tower 0 has
    // the largest modulus, so centered recovery there gives the true small int).
    uint64_t q=mod[0], rt=root[0];
    std::vector<uint64_t> phase(n);
    for(uint32_t k=0;k<n;++k) phase[k]=am(c0[k], mm(c1[k],s[k],q), q);
    xform(phase,n,q,rt,true); // -> coeff
    mout.resize(n);
    for(uint32_t k=0;k<n;++k){ uint64_t c=phase[k]; mout[k]=(c>q/2)?((int64_t)c-(int64_t)q):(int64_t)c; }
}

} // namespace gpufhe
