// Native key generation for CKKS/RLWE (the sampling layer). Secret s (ternary),
// public key (b = ns*e - a*s). Sampling uses our own RNG -- can't match
// OpenFHE's draws, so keys are validated by the decryption identity
// b + a*s = ns*e (small), which is THE correctness criterion for an RLWE key
// regardless of which noise was drawn. All polys are per-tower RNS over Q,
// stored as sizeQ*n host arrays, EVALUATION form (NTT'd) unless noted.
#include "keyswitch.h"
#include "keygen.h"
#include "ntt.h"
#include "intt.h"
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>

namespace gpufhe {
namespace { uint32_t g_hw=0; }   // 0 = uniform ternary (default)
namespace {
uint64_t mulmod(uint64_t a,uint64_t b,uint64_t q){return (uint64_t)(((unsigned __int128)a*b)%q);}
uint64_t addmod(uint64_t a,uint64_t b,uint64_t q){uint64_t s=a+b; return s>=q?s-q:s;}
uint64_t submod(uint64_t a,uint64_t b,uint64_t q){return a>=b?a-b:a+q-b;}
uint64_t* up(const std::vector<uint64_t>& h){uint64_t*d;size_t B=h.size()*8;cudaMalloc(&d,B);cudaMemcpy(d,h.data(),B,cudaMemcpyHostToDevice);return d;}

// Build fwd/inv root tables for modulus q using the borrowed OpenFHE root.
struct Tab{std::vector<uint64_t>fr,fp,ir,ip;uint64_t ninv,ninv_p;};
uint64_t powmod(uint64_t b,uint64_t e,uint64_t q){unsigned __int128 r=1,bb=b%q;while(e){if(e&1)r=(r*bb)%q;bb=(bb*bb)%q;e>>=1;}return(uint64_t)r;}
uint64_t invmod(uint64_t a,uint64_t q){return powmod(a%q,q-2,q);}
Tab mk_tab(uint32_t n,uint64_t q,uint64_t root){
    uint64_t rootInv=invmod(root,q);
    auto br=[](uint32_t v,uint32_t b){uint32_t r=0;for(uint32_t k=0;k<b;k++){r=(r<<1)|(v&1);v>>=1;}return r;};
    uint32_t logn=0;while((1u<<logn)<n)++logn;
    Tab T;T.fr.resize(n);T.fp.resize(n);T.ir.resize(n);T.ip.resize(n);
    auto fill=[&](uint64_t base,std::vector<uint64_t>&r,std::vector<uint64_t>&p){
        std::vector<uint64_t>pw(n);pw[0]=1;
        for(uint32_t i=1;i<n;++i)pw[i]=mulmod(pw[i-1],base,q);
        for(uint32_t i=0;i<n;++i){uint64_t ri=pw[br(i,logn)];r[i]=ri;p[i]=(uint64_t)(((__uint128_t)ri<<64)/q);}};
    fill(root,T.fr,T.fp);fill(rootInv,T.ir,T.ip);
    T.ninv=invmod(n,q);T.ninv_p=(uint64_t)(((__uint128_t)T.ninv<<64)/q);
    return T;
}
// NTT a single coeff-form tower to eval (in place on host vector).
void ntt_tower(std::vector<uint64_t>& h,uint32_t n,uint64_t q,const Tab& T){
    uint64_t*dx=up(h),*dr=up(T.fr),*dp=up(T.fp);
    LaunchNTT_CT(dx,dr,dp,n,q,0);cudaDeviceSynchronize();
    cudaMemcpy(h.data(),dx,(size_t)n*8,cudaMemcpyDeviceToHost);
    cudaFree(dx);cudaFree(dr);cudaFree(dp);
}
} // anon

// Generate a keypair over Q. Returns s, and pubkey (b,a), all sizeQ*n eval form.
// moduli/roots borrowed. sigma = Gaussian stddev (OpenFHE default 3.19).
void set_secret_hamming_weight(uint32_t h){ g_hw=h; }

KeyPairHost keygen_host(uint32_t n, const std::vector<uint64_t>& moduli,
                        const std::vector<uint64_t>& roots, uint64_t ns, double sigma,
                        uint64_t seed)
{
    const uint32_t sizeQ=(uint32_t)moduli.size();
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0,sigma);

    // Sample ONE ternary secret and ONE Gaussian error in coefficient form as
    // small signed integers, then CRT-embed + NTT into each tower. Same small
    // poly across towers (that's what CRT of a small integer poly means).
    std::vector<int> sc(n), ec(n);
    if(g_hw==0){
        for(uint32_t k=0;k<n;++k){ int r=(int)(rng()%3); sc[k]=r-1; }    // uniform {-1,0,1}
    } else {
        // SPARSE ternary: exactly g_hw nonzero coeffs, random positions/signs.
        // Uniform ternary has ~n/3 nonzeros, so |I| in ModRaise grows like
        // sqrt(n) and K=max|a_k/q0| doubles per 4x ring (29.2@1024 ->
        // 69.6@4096), which forces more double-angle steps AND shrinks the
        // message's share of the EvalMod range. Fixed hamming weight pins K
        // independent of n. Sampled BEFORE the tower loop, so the same seed
        // still yields the same secret at any tower count (the reduced-level
        // decrypt trick depends on that).
        std::vector<uint32_t> idx(n); for(uint32_t k=0;k<n;++k) idx[k]=k;
        for(uint32_t k=n-1;k>0;--k){ uint32_t j=(uint32_t)(rng()%(k+1)); std::swap(idx[k],idx[j]); }
        uint32_t h=(g_hw<n)?g_hw:n;
        for(uint32_t k=0;k<h;++k) sc[idx[k]] = (rng()&1ull)? 1 : -1;
    }
    for(uint32_t k=0;k<n;++k){ long e=std::lround(gauss(rng)); ec[k]=(int)e; }
    // uniform a is independent per tower (uniform mod q_i).

    KeyPairHost K;
    K.s.resize((size_t)sizeQ*n); K.pkA.resize((size_t)sizeQ*n); K.pkB.resize((size_t)sizeQ*n);

    for(uint32_t t=0;t<sizeQ;++t){
        uint64_t q=moduli[t]; Tab T=mk_tab(n,q,roots[t]);
        // s tower: embed signed sc into [0,q), coeff form, then NTT.
        std::vector<uint64_t> sT(n), eT(n), aT(n);
        for(uint32_t k=0;k<n;++k){ long v=sc[k]; sT[k]=(uint64_t)((v%(long)q+(long)q)%(long)q);
            long ev=ec[k]; eT[k]=(uint64_t)((ev%(long)q+(long)q)%(long)q);
            aT[k]=rng()%q; }                     // a already "uniform"; treat as eval directly
        ntt_tower(sT,n,q,T);                      // s -> eval
        ntt_tower(eT,n,q,T);                      // e -> eval
        // a is sampled uniform; OpenFHE's DugType gives eval-form uniform, so
        // treat aT as already eval (uniform in each slot is uniform either way).
        // b = ns*e - a*s   (all eval, pointwise)
        std::vector<uint64_t> bT(n);
        for(uint32_t k=0;k<n;++k){
            uint64_t as=mulmod(aT[k],sT[k],q);
            uint64_t nse=mulmod(ns%q,eT[k],q);
            bT[k]=submod(nse,as,q);
        }
        std::copy(sT.begin(),sT.end(),K.s.begin()+(size_t)t*n);
        std::copy(aT.begin(),aT.end(),K.pkA.begin()+(size_t)t*n);
        std::copy(bT.begin(),bT.end(),K.pkB.begin()+(size_t)t*n);
    }
    return K;
}

} // namespace gpufhe
