// Native eval (relin) key generation. Reproduces KeySwitchGenInternal
// (keyswitch-hybrid.cpp:144): per part p over the extended QP basis,
//   av[p] = newp1*u + ns*e1
//   bv[p] = newp0*u + ns*e0 + (P mod q_i)*s2   [only in part p's Q-tower range]
// where (newp0,newp1) is the pubkey of s over QP, u fresh ternary, e0/e1 fresh
// Gaussian, s2 = s*s. Validated by the eval-key decryption identity
//   bv[p] - av[p]*s == (P*s2 in part's range) + small noise.
#include "keyswitch.h"
#include "keygen.h"
#include "ntt.h"
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include <map>
#include <cuda_runtime.h>

namespace gpufhe {
namespace {
uint64_t mulmod(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
uint64_t addmod(uint64_t a,uint64_t b,uint64_t q){uint64_t s=a+b;return s>=q?s-q:s;}
uint64_t submod(uint64_t a,uint64_t b,uint64_t q){return a>=b?a-b:a+q-b;}
uint64_t* up(const std::vector<uint64_t>&h){uint64_t*d;cudaMalloc(&d,h.size()*8);cudaMemcpy(d,h.data(),h.size()*8,cudaMemcpyHostToDevice);return d;}
uint64_t powmod(uint64_t b,uint64_t e,uint64_t q){unsigned __int128 r=1,bb=b%q;while(e){if(e&1)r=(r*bb)%q;bb=(bb*bb)%q;e>>=1;}return(uint64_t)r;}
uint64_t invmod(uint64_t a,uint64_t q){return powmod(a%q,q-2,q);}

// PERF: root/precon table cache by (n,q) for the process lifetime, mirroring
// keyswitch.cpp's DevTab/dev_tab pattern (kept as a separate, private cache
// here -- this file never does INTT, only LaunchNTT_CT forward transforms).
// The data path is also batched: per part, u/e0/e1 are each uploaded once,
// transformed once across all sizeQlP towers via LaunchNTT_CT_Batched, and
// downloaded once, instead of one upload/sync/download per tower per buffer.
struct EkDevTab { uint64_t* fr=nullptr; uint64_t* fp=nullptr; };
EkDevTab build_ek_tab(uint32_t n, uint64_t q, uint64_t root){
    auto br=[](uint32_t v,uint32_t b){uint32_t r=0;for(uint32_t k=0;k<b;k++){r=(r<<1)|(v&1);v>>=1;}return r;};
    uint32_t logn=0;while((1u<<logn)<n)++logn;
    std::vector<uint64_t> fr(n),fp(n),pw(n);pw[0]=1;
    for(uint32_t i=1;i<n;++i)pw[i]=mulmod(pw[i-1],root,q);
    for(uint32_t i=0;i<n;++i){uint64_t r=pw[br(i,logn)];fr[i]=r;fp[i]=(uint64_t)(((__uint128_t)r<<64)/q);}
    EkDevTab T; T.fr=up(fr); T.fp=up(fp);
    return T;
}
EkDevTab& ek_dev_tab(uint32_t n, uint64_t q, uint64_t root){
    static std::map<std::pair<uint32_t,uint64_t>,EkDevTab> ekCache;
    auto key=std::make_pair(n,q);
    auto it=ekCache.find(key);
    if(it!=ekCache.end()) return it->second;
    return ekCache.emplace(key,build_ek_tab(n,q,root)).first->second;
}
} // anon

// Fills K.av / K.bv (per part, over QP towers, eval form) from a native relin
// key. Requires: s over QP (sizeQlP*n eval), pubkey (pkA,pkB) over QP, the
// PModq value per Q tower (K.pInvModq's sibling -- pass P mod q_i), moduli+roots
// for QP. Writes evalKeyTowers = sizeQlP.
void evalkeygen_host_sold(KeySwitchConstants& K,
                     const std::vector<uint64_t>& sQP,
                     const std::vector<uint64_t>& sOld,       // the key being switched (eval, QP)
                     const std::vector<uint64_t>& pkA_QP,     // sizeQlP*n eval
                     const std::vector<uint64_t>& pkB_QP,
                     const std::vector<uint64_t>& PModq_QP,   // P mod (each QP modulus) -- but P*s2 only added in Q range
                     const std::vector<uint64_t>& modQP,
                     const std::vector<uint64_t>& rootQP,
                     uint64_t ns, double sigma, uint64_t seed)
{
    const uint32_t n=K.n, sizeQlP=(uint32_t)modQP.size();
    const uint32_t sizeQl=K.sizeQl, alpha=K.alpha, numPart=K.numPart;
    K.evalKeyTowers=sizeQlP;
    K.av.assign(numPart,{}); K.bv.assign(numPart,{});

    const std::vector<uint64_t>& s2 = sOld;   // generalized: caller supplies the switched key

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0,sigma);

    std::vector<uint64_t*> h_fr(sizeQlP), h_fp(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){
        EkDevTab& T = ek_dev_tab(n, modQP[t], rootQP[t]);
        h_fr[t]=T.fr; h_fp[t]=T.fp;
    }
    uint64_t **d_fr=nullptr, **d_fp=nullptr;
    cudaMalloc(&d_fr, sizeQlP*sizeof(uint64_t*));
    cudaMalloc(&d_fp, sizeQlP*sizeof(uint64_t*));
    cudaMemcpy(d_fr, h_fr.data(), sizeQlP*sizeof(uint64_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fp, h_fp.data(), sizeQlP*sizeof(uint64_t*), cudaMemcpyHostToDevice);
    uint64_t* d_modQP = up(modQP);

    for(uint32_t part=0; part<numPart; ++part){
        uint32_t startIdx=alpha*part;
        uint32_t sizePart=(sizeQl>startIdx+alpha)?alpha:(sizeQl-startIdx);
        uint32_t endIdx=startIdx+sizePart;
        K.av[part].resize((size_t)sizeQlP*n);
        K.bv[part].resize((size_t)sizeQlP*n);

        // fresh u (ternary), e0,e1 (Gaussian) as small integer polys, coeff form.
        std::vector<int> uc(n),e0c(n),e1c(n);
        for(uint32_t k=0;k<n;++k){ uc[k]=(int)(rng()%3)-1;
            e0c[k]=(int)std::lround(gauss(rng)); e1c[k]=(int)std::lround(gauss(rng)); }

        // embed u,e0,e1 into every tower (coeff), batched into contiguous buffers
        std::vector<uint64_t> uBuf((size_t)sizeQlP*n), e0Buf((size_t)sizeQlP*n), e1Buf((size_t)sizeQlP*n);
        for(uint32_t t=0;t<sizeQlP;++t){ uint64_t q=modQP[t];
            uint64_t* uT=uBuf.data()+(size_t)t*n; uint64_t* e0T=e0Buf.data()+(size_t)t*n; uint64_t* e1T=e1Buf.data()+(size_t)t*n;
            for(uint32_t k=0;k<n;++k){ long uv=uc[k]; uT[k]=(uint64_t)((uv%(long)q+(long)q)%(long)q);
                long a0=e0c[k]; e0T[k]=(uint64_t)((a0%(long)q+(long)q)%(long)q);
                long a1=e1c[k]; e1T[k]=(uint64_t)((a1%(long)q+(long)q)%(long)q); }
        }
        // NTT to eval: one upload/transform/download per buffer, across all towers
        uint64_t *d_u=up(uBuf), *d_e0=up(e0Buf), *d_e1=up(e1Buf);
        LaunchNTT_CT_Batched(d_u, (const uint64_t* const*)d_fr,(const uint64_t* const*)d_fp,d_modQP,sizeQlP,n,0);
        LaunchNTT_CT_Batched(d_e0,(const uint64_t* const*)d_fr,(const uint64_t* const*)d_fp,d_modQP,sizeQlP,n,0);
        LaunchNTT_CT_Batched(d_e1,(const uint64_t* const*)d_fr,(const uint64_t* const*)d_fp,d_modQP,sizeQlP,n,0);
        cudaDeviceSynchronize();
        cudaMemcpy(uBuf.data(), d_u, (size_t)sizeQlP*n*8,cudaMemcpyDeviceToHost);
        cudaMemcpy(e0Buf.data(),d_e0,(size_t)sizeQlP*n*8,cudaMemcpyDeviceToHost);
        cudaMemcpy(e1Buf.data(),d_e1,(size_t)sizeQlP*n*8,cudaMemcpyDeviceToHost);
        cudaFree(d_u); cudaFree(d_e0); cudaFree(d_e1);

        for(uint32_t t=0;t<sizeQlP;++t){ uint64_t q=modQP[t];
            const uint64_t* uT=uBuf.data()+(size_t)t*n; const uint64_t* e0T=e0Buf.data()+(size_t)t*n; const uint64_t* e1T=e1Buf.data()+(size_t)t*n;
            for(uint32_t k=0;k<n;++k){
                uint64_t p1u=mulmod(pkA_QP[(size_t)t*n+k],uT[k],q);   // newp1 = a
                uint64_t p0u=mulmod(pkB_QP[(size_t)t*n+k],uT[k],q);   // newp0 = b
                uint64_t av=addmod(p1u, mulmod(ns%q,e1T[k],q), q);
                uint64_t bv=addmod(p0u, mulmod(ns%q,e0T[k],q), q);
                // add P*s2 only in this part's Q-tower range
                if(t>=startIdx && t<endIdx){
                    uint64_t Ps2=mulmod(PModq_QP[t]%q, s2[(size_t)t*n+k], q);
                    bv=addmod(bv,Ps2,q);
                }
                K.av[part][(size_t)t*n+k]=av;
                K.bv[part][(size_t)t*n+k]=bv;
            }
        }
    }
    cudaFree(d_fr); cudaFree(d_fp); cudaFree(d_modQP);
}

// relin wrapper: sOld = s^2
void evalkeygen_host(KeySwitchConstants& K,
                     const std::vector<uint64_t>& sQP,
                     const std::vector<uint64_t>& pkA_QP,
                     const std::vector<uint64_t>& pkB_QP,
                     const std::vector<uint64_t>& PModq_QP,
                     const std::vector<uint64_t>& modQP,
                     const std::vector<uint64_t>& rootQP,
                     uint64_t ns, double sigma, uint64_t seed)
{
    const uint32_t n=K.n, sizeQlP=(uint32_t)modQP.size();
    std::vector<uint64_t> s2(sQP.size());
    for(uint32_t t=0;t<sizeQlP;++t){ uint64_t q=modQP[t];
        for(uint32_t k=0;k<n;++k) s2[(size_t)t*n+k]=mulmod(sQP[(size_t)t*n+k],sQP[(size_t)t*n+k],q); }
    evalkeygen_host_sold(K,sQP,s2,pkA_QP,pkB_QP,PModq_QP,modQP,rootQP,ns,sigma,seed);
}

} // namespace gpufhe
