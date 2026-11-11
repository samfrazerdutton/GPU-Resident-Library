// Native CKKS encrypt (pk) + decrypt, poly-arithmetic level (no complex
// encode/decode -- messages are integer coefficient polys, validated directly).
// Encrypt: c0 = p0*v + ns*e0 + m ; c1 = p1*v + ns*e1  (v ternary, e Gaussian).
// Decrypt: c0 + c1*s (+ c2*s^2 ... for >2 components). All eval form.
#include "keygen.h"
#include "ntt.h"
#include "intt.h"
#include <cstdint>
#include <vector>
#include <map>
#include <utility>
#include <random>
#include <cmath>
#include <mutex>
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

// GOD PROMPT "CKKS PHASE 11": encrypt_host's forward-NTT root tables,
// cached by (n,q) for the PROCESS LIFETIME, mirroring src/evalkeygen.cpp's
// own ek_dev_tab/build_ek_tab pattern (that file's own established,
// already-proven fix for the identical problem). Measured as the
// dominant component of `encrypt_host`'s own cost (bench/
// bench_evalmult_e2e_breakdown.cpp: ~39-51ms at n=4096/8192 for a single
// call, vs ~6ms for the ENTIRE tensor+keyswitch+combine+rescale GPU chain
// -- research/EXPERIMENT_LEDGER.md entry #44) -- the ORIGINAL per-tower
// `xform` calls above each rebuilt an O(n) root table FROM SCRATCH and
// paid their own cudaMalloc/upload/kernel/download/free round trip, FOUR
// TIMES per tower (v,e0,e1,m), i.e. 4*sizeQ separate GPU round trips for
// one encrypt_host call. encrypt_host_fast below batches all sizeQ
// towers of each buffer into ONE upload + ONE LaunchNTT_CT_Batched call +
// ONE download (4 GPU round trips total, not 4*sizeQ), reusing cached
// root tables instead of rebuilding them -- the EXACT same technique
// evalkeygen_host_sold already uses successfully. The underlying
// arithmetic (embedding formula, NTT, and the b=pkB*v+ns*e0+m / a=pkA*v+
// ns*e1 combine) is UNCHANGED -- this is a batching/caching change, not
// an algorithmic one -- verified bit-exact against the original
// `xform`-based path in tests/test_encrypt_host_fast_matches_sequential.cpp.
struct EncDevTab { uint64_t* fr=nullptr; uint64_t* fp=nullptr; };
EncDevTab build_enc_tab(uint32_t n, uint64_t q, uint64_t root){
    auto br=[](uint32_t v,uint32_t b){uint32_t r=0;for(uint32_t k=0;k<b;k++){r=(r<<1)|(v&1);v>>=1;}return r;};
    uint32_t logn=0;while((1u<<logn)<n)++logn;
    std::vector<uint64_t> fr(n),fp(n),pw(n);pw[0]=1;
    for(uint32_t i=1;i<n;++i)pw[i]=mm(pw[i-1],root,q);
    for(uint32_t i=0;i<n;++i){uint64_t r=pw[br(i,logn)];fr[i]=r;fp[i]=(uint64_t)(((__uint128_t)r<<64)/q);}
    EncDevTab T; T.fr=up(fr); T.fp=up(fp);
    return T;
}
// GOD PROMPT "CKKS PHASE 33": PROVEN-unsafe find()-then-emplace() pattern
// (research/PHASE33_CACHE_CONCURRENCY_AUDIT.md). Confirmed REACHABLE under
// real concurrency: tests/test_encrypt_host_shoup_pooled_thread_safety.cpp
// calls encrypt_host (which reaches this cache) from 8 threads x 6 iters at
// the IDENTICAL (n,sizeQ) key, as its own single-threaded reference
// computation inside the concurrent loop.
std::mutex& g_encdec_cache_mutex(){ static std::mutex m; return m; }
EncDevTab& enc_dev_tab(uint32_t n, uint64_t q, uint64_t root){
    std::lock_guard<std::mutex> lock(g_encdec_cache_mutex());
    static std::map<std::pair<uint32_t,uint64_t>,EncDevTab> cache;
    auto key=std::make_pair(n,q);
    auto it=cache.find(key);
    if(it!=cache.end()) return it->second;
    return cache.emplace(key,build_enc_tab(n,q,root)).first->second;
}
} // anon

// Encrypt an integer-coefficient message m (coeff form, small ints centered) under
// pubkey (pkA,pkB). Returns (c0,c1) eval form, sizeQ*n each.
//
// GOD PROMPT "CKKS PHASE 11": batched across all sizeQ towers -- ONE
// upload + ONE LaunchNTT_CT_Batched + ONE download per buffer (v,e0,e1,m),
// using cached root tables (enc_dev_tab), instead of the previous
// per-tower loop's 4*sizeQ separate `xform` calls (each its own
// cudaMalloc/upload/kernel/download/free AND its own from-scratch O(n)
// root-table rebuild). The embedding formula, the NTT itself, and the
// b=pkB*v+ns*e0+m / a=pkA*v+ns*e1 combine are ALL UNCHANGED -- this is a
// batching/caching change only, verified bit-exact against the original
// per-tower implementation in tests/
// test_encrypt_host_fast_matches_sequential.cpp.
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

    // Embed v/e0/e1/m into every tower (coeff domain), batched into
    // contiguous buffers -- same embedding formula as before, per tower.
    std::vector<uint64_t> vBuf((size_t)sizeQ*n), e0Buf((size_t)sizeQ*n), e1Buf((size_t)sizeQ*n), mBuf((size_t)sizeQ*n);
    for(uint32_t t=0;t<sizeQ;++t){ uint64_t q=mod[t];
        uint64_t* vT=vBuf.data()+(size_t)t*n; uint64_t* e0T=e0Buf.data()+(size_t)t*n;
        uint64_t* e1T=e1Buf.data()+(size_t)t*n; uint64_t* mT=mBuf.data()+(size_t)t*n;
        for(uint32_t k=0;k<n;++k){ auto emb=[&](long x){return (uint64_t)((x%(long)q+(long)q)%(long)q);};
            vT[k]=emb(vc[k]); e0T[k]=emb(e0c[k]); e1T[k]=emb(e1c[k]); mT[k]=emb((long)m[k]); }
    }

    // One batched NTT per buffer across all sizeQ towers, root tables
    // cached by (n,q) for the process lifetime.
    std::vector<uint64_t*> h_fr(sizeQ), h_fp(sizeQ);
    for(uint32_t t=0;t<sizeQ;++t){ EncDevTab& T=enc_dev_tab(n,mod[t],root[t]); h_fr[t]=T.fr; h_fp[t]=T.fp; }
    uint64_t **d_fr=nullptr,**d_fp=nullptr;
    cudaMalloc(&d_fr, sizeQ*sizeof(uint64_t*));
    cudaMalloc(&d_fp, sizeQ*sizeof(uint64_t*));
    cudaMemcpy(d_fr, h_fr.data(), sizeQ*sizeof(uint64_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fp, h_fp.data(), sizeQ*sizeof(uint64_t*), cudaMemcpyHostToDevice);
    uint64_t* d_mod = up(mod);

    uint64_t *d_v,*d_e0,*d_e1,*d_m;
    const size_t T8=(size_t)sizeQ*n*8;
    cudaMalloc(&d_v,T8); cudaMalloc(&d_e0,T8); cudaMalloc(&d_e1,T8); cudaMalloc(&d_m,T8);
    cudaMemcpy(d_v, vBuf.data(),T8,cudaMemcpyHostToDevice);
    cudaMemcpy(d_e0,e0Buf.data(),T8,cudaMemcpyHostToDevice);
    cudaMemcpy(d_e1,e1Buf.data(),T8,cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, mBuf.data(),T8,cudaMemcpyHostToDevice);
    LaunchNTT_CT_Batched(d_v, (const uint64_t* const*)d_fr,(const uint64_t* const*)d_fp,d_mod,sizeQ,n,0);
    LaunchNTT_CT_Batched(d_e0,(const uint64_t* const*)d_fr,(const uint64_t* const*)d_fp,d_mod,sizeQ,n,0);
    LaunchNTT_CT_Batched(d_e1,(const uint64_t* const*)d_fr,(const uint64_t* const*)d_fp,d_mod,sizeQ,n,0);
    LaunchNTT_CT_Batched(d_m, (const uint64_t* const*)d_fr,(const uint64_t* const*)d_fp,d_mod,sizeQ,n,0);
    cudaDeviceSynchronize();
    cudaMemcpy(vBuf.data(), d_v, T8,cudaMemcpyDeviceToHost);
    cudaMemcpy(e0Buf.data(),d_e0,T8,cudaMemcpyDeviceToHost);
    cudaMemcpy(e1Buf.data(),d_e1,T8,cudaMemcpyDeviceToHost);
    cudaMemcpy(mBuf.data(), d_m, T8,cudaMemcpyDeviceToHost);
    cudaFree(d_v);cudaFree(d_e0);cudaFree(d_e1);cudaFree(d_m);cudaFree(d_fr);cudaFree(d_fp);cudaFree(d_mod);

    for(uint32_t t=0;t<sizeQ;++t){ uint64_t q=mod[t];
        const uint64_t* vT=vBuf.data()+(size_t)t*n; const uint64_t* e0T=e0Buf.data()+(size_t)t*n;
        const uint64_t* e1T=e1Buf.data()+(size_t)t*n; const uint64_t* mT=mBuf.data()+(size_t)t*n;
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
