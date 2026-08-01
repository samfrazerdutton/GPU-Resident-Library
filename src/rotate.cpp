// Galois automorphism sigma_k on an eval-form RNS poly: x -> x^k.
// Coefficient-domain (correct-first): per tower INTT -> permute coeff j to
// index (j*k mod 2n), negating when it lands in [n,2n) (x^n = -1) -> NTT.
#include <cstdint>
#include <vector>
#include <map>
#include <utility>
#include <cuda_runtime.h>
#include "ntt.h"
#include "intt.h"
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "automorph.h"
#include "stage_acc.h"
namespace gpufhe {
namespace {
uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
uint64_t powmod(uint64_t b,uint64_t e,uint64_t q){unsigned __int128 r=1,bb=b%q;while(e){if(e&1)r=(r*bb)%q;bb=(bb*bb)%q;e>>=1;}return(uint64_t)r;}
uint64_t invmod(uint64_t a,uint64_t q){return powmod(a%q,q-2,q);}
uint64_t* up(const std::vector<uint64_t>&h){uint64_t*d;cudaMalloc(&d,h.size()*8);cudaMemcpy(d,h.data(),h.size()*8,cudaMemcpyHostToDevice);return d;}

// PERF: the old xf() rebuilt the root tables (n modmuls + n 128-bit divides)
// and did 3 malloc + 3 memcpy + sync + 3 free on EVERY call, PER TOWER. With
// pt_to_eval_host called once per BSGS diagonal that was ~42k such calls per
// CoeffsToSlots. Now: tables cached on device for the process lifetime, and
// all towers ride one upload / launch-loop / download.
struct DevTab { uint64_t *fr=nullptr,*fp=nullptr,*ir=nullptr,*ip=nullptr; uint64_t ninv=0,ninv_p=0; };
DevTab& dev_tab(uint32_t n, uint64_t q, uint64_t root){
    static std::map<std::pair<uint32_t,uint64_t>,DevTab> dcache;
    auto key=std::make_pair(n,q);
    auto it=dcache.find(key); if(it!=dcache.end()) return it->second;
    auto br=[](uint32_t v,uint32_t b){uint32_t r=0;for(uint32_t k=0;k<b;k++){r=(r<<1)|(v&1);v>>=1;}return r;};
    uint32_t logn=0; while((1u<<logn)<n)++logn;
    std::vector<uint64_t> fr(n),fp(n),ir(n),ip(n),pw(n);
    auto fill=[&](uint64_t base,std::vector<uint64_t>&r,std::vector<uint64_t>&pp){
        pw[0]=1; for(uint32_t i=1;i<n;++i)pw[i]=mm(pw[i-1],base,q);
        for(uint32_t i=0;i<n;++i){uint64_t v=pw[br(i,logn)];r[i]=v;pp[i]=(uint64_t)(((__uint128_t)v<<64)/q);}};
    fill(root,fr,fp); fill(invmod(root,q),ir,ip);
    DevTab D; D.fr=up(fr); D.fp=up(fp); D.ir=up(ir); D.ip=up(ip);
    D.ninv=invmod(n,q); D.ninv_p=(uint64_t)(((__uint128_t)D.ninv<<64)/q);
    return dcache.emplace(key,D).first->second;
}
// Transform every tower of a towers*n poly: one upload, per-tower launches, one download.
// Hoisted device scratch: batching made allocations BIGGER (towers*n instead of
// n) and per-call cudaMalloc/Free of a ~250KB block costs far more than many
// small ones -- that regressed the bootstrap 53 -> 101 s. Allocate once, grow
// on demand, never free.
uint64_t* scratch(size_t bytes){
    static uint64_t* buf=nullptr; static size_t cap=0;
    if(bytes>cap){ if(buf) cudaFree(buf); cudaMalloc(&buf,bytes); cap=bytes; }
    return buf;
}
void xf_multi(std::vector<uint64_t>& h, uint32_t towers, uint32_t n,
              const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root, bool inv){
    const size_t T=(size_t)towers*n;
    uint64_t* dx=scratch(T*8);
    cudaMemcpy(dx,h.data(),T*8,cudaMemcpyHostToDevice);
    for(uint32_t t=0;t<towers;++t){ uint64_t q=mod[t]; DevTab&D=dev_tab(n,q,root[t]);
        if(inv) LaunchINTT_GS(dx+(size_t)t*n,D.ir,D.ip,n,q,D.ninv,D.ninv_p,0);
        else    LaunchNTT_CT (dx+(size_t)t*n,D.fr,D.fp,n,q,0); }
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(),dx,T*8,cudaMemcpyDeviceToHost);
}
} // anon

// DEVICE automorphism: INTT -> permute kernel -> NTT, all on device pointers.
// Needs one n-sized scratch buffer (the permute cannot be done in place).
// Root tables come from the same process-lifetime dev_tab cache the host
// version uses, so this adds no setup cost.
void automorphism_eval_device(uint64_t* d_v, uint32_t towers, uint32_t n,
                              uint32_t k, const std::vector<uint64_t>& mod,
                              const std::vector<uint64_t>& root,
                              uint64_t* d_scratch, cudaStream_t s)
{
    for(uint32_t t=0;t<towers;++t){
        uint64_t q=mod[t]; DevTab&D=dev_tab(n,q,root[t]);
        uint64_t* col=d_v+(size_t)t*n;
        LaunchINTT_GS(col,D.ir,D.ip,n,q,D.ninv,D.ninv_p,s);
        cudaMemsetAsync(d_scratch,0,(size_t)n*8,s);
        LaunchAutomorphPermute(d_scratch,col,n,(uint64_t)k,q,s);
        cudaMemcpyAsync(col,d_scratch,(size_t)n*8,cudaMemcpyDeviceToDevice,s);
        LaunchNTT_CT(col,D.fr,D.fp,n,q,s);
    }
}

// In-place automorphism on a towers*n eval-form poly.
void automorphism_eval_host(std::vector<uint64_t>& v, uint32_t towers, uint32_t n,
                            uint32_t k, const std::vector<uint64_t>& mod,
                            const std::vector<uint64_t>& root)
{
    const uint32_t M=2*n;
    xf_multi(v,towers,n,mod,root,true);                 // all towers -> coeff
    std::vector<uint64_t> o((size_t)towers*n,0);
    for(uint32_t t=0;t<towers;++t){ uint64_t q=mod[t];
        const uint64_t* c=v.data()+(size_t)t*n;
        uint64_t* ot=o.data()+(size_t)t*n;
        for(uint32_t j=0;j<n;++j){ uint64_t idx=((uint64_t)j*k)%M; uint64_t cv=c[j];
            if(idx<n) ot[idx]=cv; else ot[idx-n]=cv? q-cv : 0; } }   // x^n = -1
    v.swap(o);
    xf_multi(v,towers,n,mod,root,false);                // all towers -> eval
}
// Embed an encoded integer-coeff poly into towers*n EVAL form (per tower:
// center-embed mod q, NTT).
void pt_to_eval_host(std::vector<uint64_t>& out, const std::vector<int64_t>& m,
                     uint32_t towers, uint32_t n, const std::vector<uint64_t>& mod,
                     const std::vector<uint64_t>& root)
{
    out.resize((size_t)towers*n);
    for(uint32_t t=0;t<towers;++t){ uint64_t q=mod[t];
        uint64_t* c=out.data()+(size_t)t*n;
        for(uint32_t k=0;k<n;++k){ long v=(long)m[k]; c[k]=(uint64_t)((v%(long)q+(long)q)%(long)q); } }
    xf_multi(out,towers,n,mod,root,false);
}
// PURE DEVICE-POINTER rotation: nothing crosses PCIe, no allocation, no sync.
// Caller owns the context/work/scratch and the ciphertext already lives on the
// GPU. This is what lets a whole transform stage stay resident: every diagonal
// rotates the SAME uploaded ciphertext, so the loop can run entirely on device.
void rotate_ct_device(uint64_t* d_c0, uint64_t* d_c1, uint32_t k,
                      const DeviceKSContext& C, DeviceKSWork& W,
                      uint32_t towers, uint32_t n,
                      const std::vector<uint64_t>& mod,
                      const std::vector<uint64_t>& root,
                      uint64_t* d_scratch, uint64_t* d_b0, uint64_t* d_b1,
                      cudaStream_t s)
{
    automorphism_eval_device(d_c0,towers,n,k,mod,root,d_scratch,s);
    automorphism_eval_device(d_c1,towers,n,k,mod,root,d_scratch,s);
    keyswitch_resident(d_c1,d_b0,d_b1,C,W,s);
    for(uint32_t t=0;t<towers;++t)
        LaunchAddInto(d_c0+(size_t)t*n, d_b0+(size_t)t*n, mod[t], n, s);
    cudaMemcpyAsync(d_c1,d_b1,(size_t)towers*n*8,cudaMemcpyDeviceToDevice,s);
}

// RESIDENT rotation. Same result as rotate_ct_host but the keyswitch runs on
// the fully device-resident path (DeviceKSContext/DeviceKSWork, zero malloc,
// pure kernel launches) instead of keyswitch_core_resident, which despite its
// name round-trips host<->device between stages and mallocs per tower per part.
// The context (eval key + constants + root tables) is uploaded once per call
// here; if that dominates, cache contexts across calls next.
void rotate_ct_resident(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1,
                        uint32_t k, const KeySwitchConstants& Krot,
                        uint32_t towers, uint32_t n,
                        const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root)
{
    // Fully device-side now: both automorphisms run as kernels on data that
    // was uploaded once, instead of two host permutes (~19 ms) bracketing the
    // keyswitch. c0 needs its automorphism too, so it rides along on device.
    auto C=ks_context_create(Krot);
    auto W=ks_work_create(C);
    const size_t T=(size_t)towers*n, B=T*8;
    uint64_t *d_a,*d_b0,*d_b1,*d_c0,*d_sc;
    cudaMalloc(&d_a,B); cudaMalloc(&d_b0,B); cudaMalloc(&d_b1,B);
    cudaMalloc(&d_c0,B); cudaMalloc(&d_sc,(size_t)n*8);
    cudaMemcpy(d_a,c1.data(),B,cudaMemcpyHostToDevice);
    cudaMemcpy(d_c0,c0.data(),B,cudaMemcpyHostToDevice);
    automorphism_eval_device(d_a ,towers,n,k,mod,root,d_sc,0);
    automorphism_eval_device(d_c0,towers,n,k,mod,root,d_sc,0);
    cudaMemcpy(c0.data(),d_c0,B,cudaMemcpyDeviceToHost);
    cudaFree(d_c0); cudaFree(d_sc);
    keyswitch_resident(d_a,d_b0,d_b1,C,W,0);
    cudaDeviceSynchronize();
    std::vector<uint64_t> ba0(T),ba1(T);
    cudaMemcpy(ba0.data(),d_b0,B,cudaMemcpyDeviceToHost);
    cudaMemcpy(ba1.data(),d_b1,B,cudaMemcpyDeviceToHost);
    cudaFree(d_a);cudaFree(d_b0);cudaFree(d_b1);
    ks_work_destroy(W); ks_context_destroy(C);

    for(uint32_t t=0;t<towers;++t){ uint64_t q=mod[t];
        for(uint32_t kk=0;kk<n;++kk){ size_t x=(size_t)t*n+kk;
            uint64_t s2=c0[x]+ba0[x]; c0[x]=(s2>=q)?s2-q:s2; } }
    c1=ba1;
}

// Full homomorphic rotation: sigma_k on (c0,c1), keyswitch sigma(c1) with the
// rotation key in Krot, output (sigma(c0)+ba0, ba1). k must match Krot's sOld.
void rotate_ct_host(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1,
                    uint32_t k, const KeySwitchConstants& Krot,
                    uint32_t towers, uint32_t n,
                    const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root)
{
    automorphism_eval_host(c0,towers,n,k,mod,root);
    automorphism_eval_host(c1,towers,n,k,mod,root);
    auto R=keyswitch_core_resident(c1,Krot);
    for(uint32_t t=0;t<towers;++t){ uint64_t q=mod[t];
        for(uint32_t kk=0;kk<n;++kk){ size_t x=(size_t)t*n+kk;
            uint64_t s2=c0[x]+R.ba0[x]; c0[x]=(s2>=q)?s2-q:s2; } }
    c1=R.ba1;
}
} // namespace gpufhe
