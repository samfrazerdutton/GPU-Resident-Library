#include "keyswitch.h"
#include "ntt.h"
#include "intt.h"
#include "basis_convert.h"
#include "ks_fastcore.h"
#include "moddown.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <map>
#include <vector>

namespace gpufhe {

// Per-modulus root tables, built once and cached by modulus value.
namespace {
struct RootTab { std::vector<uint64_t> fr,fp,ir,ip; uint64_t ninv,ninv_p; };

uint64_t powmod(uint64_t b, uint64_t e, uint64_t q){ unsigned __int128 r=1,bb=b%q;
    while(e){ if(e&1) r=(r*bb)%q; bb=(bb*bb)%q; e>>=1;} return (uint64_t)r; }
uint64_t invmod(uint64_t a, uint64_t q){ return powmod(a,q-2,q); } // q prime

RootTab build_tab(uint32_t n, uint64_t q, uint64_t root){
    uint64_t rootInv=invmod(root,q);
    auto br=[](uint32_t v,uint32_t b){uint32_t r=0;for(uint32_t k=0;k<b;k++){r=(r<<1)|(v&1);v>>=1;}return r;};
    uint32_t logn=0; while((1u<<logn)<n)++logn;
    RootTab T; T.fr.resize(n);T.fp.resize(n);T.ir.resize(n);T.ip.resize(n);
    auto fill=[&](uint64_t base,std::vector<uint64_t>&r,std::vector<uint64_t>&p){
        std::vector<uint64_t> pw(n); pw[0]=1;
        for(uint32_t i=1;i<n;++i) pw[i]=(uint64_t)(((unsigned __int128)pw[i-1]*base)%q);
        for(uint32_t i=0;i<n;++i){ uint64_t ri=pw[br(i,logn)];
            r[i]=ri; p[i]=(uint64_t)(((__uint128_t)ri<<64)/q); }};
    fill(root,T.fr,T.fp); fill(rootInv,T.ir,T.ip);
    T.ninv=invmod(n,q); T.ninv_p=(uint64_t)(((__uint128_t)T.ninv<<64)/q);
    return T;
}

uint64_t* up(const std::vector<uint64_t>& h){ uint64_t* d; size_t B=h.size()*8;
    cudaMalloc(&d,B); cudaMemcpy(d,h.data(),B,cudaMemcpyHostToDevice); return d; }

// PERF: root tables for a given (n,q) are IDENTICAL on every call, but the old
// cache was function-local -- so build_tab (4n modmuls + 2n 128-bit divides)
// re-ran and the tables were re-uploaded per transform, ~230 times per
// bootstrap. Cache them on the DEVICE for the process lifetime instead.
// (single-threaded use; concurrent first-touch would need a lock)
struct DevTab { uint64_t *fr=nullptr,*fp=nullptr,*ir=nullptr,*ip=nullptr; uint64_t ninv=0,ninv_p=0; };
DevTab& dev_tab(uint32_t n, uint64_t q, uint64_t root){
    static std::map<std::pair<uint32_t,uint64_t>,DevTab> dcache;
    auto key=std::make_pair(n,q);
    auto it=dcache.find(key);
    if(it!=dcache.end()) return it->second;
    RootTab T=build_tab(n,q,root);
    DevTab D; D.ninv=T.ninv; D.ninv_p=T.ninv_p;
    D.fr=up(T.fr); D.fp=up(T.fp); D.ir=up(T.ir); D.ip=up(T.ip);
    return dcache.emplace(key,D).first->second;
}
}

KeySwitchResult keyswitch_core_resident(
    const std::vector<uint64_t>& aTowers, const KeySwitchConstants& K)
{
    const uint32_t n=K.n, sizeQl=K.sizeQl, sizeP=K.sizeP;
    const uint32_t sizeQlP=sizeQl+sizeP;
    const uint32_t delta=K.fullQ - sizeQl;
    const size_t B=(size_t)n*8;

    std::map<uint64_t,RootTab> cache;
    auto root_for=[&](uint64_t q)->uint64_t{
        for(size_t r=0;r<K.rootModList.size();++r) if(K.rootModList[r]==q) return K.rootValList[r];
        return 0; };
    auto tab=[&](uint64_t q)->RootTab&{ auto it=cache.find(q);
        if(it==cache.end()) it=cache.emplace(q,build_tab(n,q,root_for(q))).first; return it->second; };

    // === STAGE 2: decompose aTowers into K.numPart digits over QlP ===
    // digit[part] = sizeQlP * n, eval form.
    std::vector<std::vector<uint64_t>> digits(K.numPart);
    for (uint32_t part=0; part<K.numPart; ++part){
        const uint32_t sizePart=K.sizePart[part], sizeCompl=K.sizeCompl[part];
        const uint32_t startIdx=K.startIdx[part], endIdx=startIdx+sizePart;

        // part's tower slice -> coeff via INTT
        std::vector<uint64_t> partCoeff((size_t)sizePart*n);
        for (uint32_t i=0;i<sizePart;++i){
            uint64_t q=K.partSrcMod[part][i]; DevTab&T=dev_tab(n,q,root_for(q));
            std::vector<uint64_t> h(aTowers.begin()+(size_t)(startIdx+i)*n,
                                    aTowers.begin()+(size_t)(startIdx+i+1)*n);
            uint64_t *dx=up(h);
            LaunchINTT_GS(dx,T.ir,T.ip,n,q,T.ninv,T.ninv_p,0);
            cudaMemcpy(partCoeff.data()+(size_t)i*n,dx,B,cudaMemcpyDeviceToHost);
            cudaFree(dx);
        }
        // ApproxSwitchCRTBasis partQ -> complement
        uint64_t *dsrc=up(partCoeff),
                 *dqhi=up(K.partQHatInv[part]),*dqhip=up(K.partQHatInvPrec[part]),
                 *dq=up(K.partSrcMod[part]),*dqmp=up(K.partQHatModp[part]),
                 *dp=up(K.partComplMod[part]),*dmlo=up(K.partBMuLo[part]),*dmhi=up(K.partBMuHi[part]);
        uint64_t* ddst; cudaMalloc(&ddst,(size_t)sizeCompl*n*8);
        LaunchApproxSwitchCRTBasis(dsrc,ddst,dqhi,dqhip,dq,dqmp,dp,dmlo,dmhi,sizePart,sizeCompl,n,0);
        cudaDeviceSynchronize();
        std::vector<uint64_t> complCoeff((size_t)sizeCompl*n);
        cudaMemcpy(complCoeff.data(),ddst,(size_t)sizeCompl*n*8,cudaMemcpyDeviceToHost);
        cudaFree(dsrc);cudaFree(dqhi);cudaFree(dqhip);cudaFree(dq);cudaFree(dqmp);
        cudaFree(dp);cudaFree(dmlo);cudaFree(dmhi);cudaFree(ddst);
        // NTT complement back to eval
        std::vector<uint64_t> complEval((size_t)sizeCompl*n);
        for(uint32_t j=0;j<sizeCompl;++j){
            uint64_t q=K.partComplMod[part][j]; DevTab&T=dev_tab(n,q,root_for(q));
            std::vector<uint64_t> h(complCoeff.begin()+(size_t)j*n, complCoeff.begin()+(size_t)(j+1)*n);
            uint64_t *dx=up(h);
            LaunchNTT_CT(dx,T.fr,T.fp,n,q,0);
            cudaMemcpy(complEval.data()+(size_t)j*n,dx,B,cudaMemcpyDeviceToHost);
            cudaFree(dx);
        }
        // reassemble sizeQlP: [0,startIdx)=complEval[i]; [startIdx,endIdx)=a[i]; [endIdx,..)=complEval[i-sizePart]
        std::vector<uint64_t>& d=digits[part]; d.resize((size_t)sizeQlP*n);
        for(uint32_t i=0;i<sizeQlP;++i){
            if(i<startIdx)
                std::copy(complEval.begin()+(size_t)i*n, complEval.begin()+(size_t)(i+1)*n, d.begin()+(size_t)i*n);
            else if(i<endIdx)
                std::copy(aTowers.begin()+(size_t)i*n, aTowers.begin()+(size_t)(i+1)*n, d.begin()+(size_t)i*n);
            else
                std::copy(complEval.begin()+(size_t)(i-sizePart)*n, complEval.begin()+(size_t)(i-sizePart+1)*n, d.begin()+(size_t)i*n);
        }
    }

    // === STAGE 3a: fast-core inner product over QlP ===
    // result0/result1 = sizeQlP * n, accumulate over parts.
    // QlP modulus per tower: Q towers use qMod, P towers use pMod.
    auto qlp_mod=[&](uint32_t i)->uint64_t{ return (i<sizeQl)? K.qMod[i] : K.pMod[i-sizeQl]; };
    std::vector<uint64_t> res0((size_t)sizeQlP*n), res1((size_t)sizeQlP*n);
    {
        // PERF: hoist every allocation and upload key+digits ONCE. The previous
        // version did 3 cudaMalloc + 3 memcpy + a full cudaDeviceSynchronize +
        // 3 cudaFree per (tower,part) -- ~5300 CUDA calls per keyswitch at
        // sizeQlP=32/numPart=15, which measured ~1.2 s per rotation. Now it is
        // pure kernel launches over preloaded buffers with ONE sync.
        const size_t DP=(size_t)sizeQlP*n, KP=(size_t)K.evalKeyTowers*n, OT=(size_t)sizeQlP*n;
        uint64_t *d_dig=nullptr,*d_ka=nullptr,*d_kb=nullptr,*d_o0=nullptr,*d_o1=nullptr;
        cudaMalloc(&d_dig,(size_t)K.numPart*DP*8);
        cudaMalloc(&d_ka ,(size_t)K.numPart*KP*8);
        cudaMalloc(&d_kb ,(size_t)K.numPart*KP*8);
        cudaMalloc(&d_o0 ,OT*8);
        cudaMalloc(&d_o1 ,OT*8);
        for(uint32_t p=0;p<K.numPart;++p){
            cudaMemcpy(d_dig+(size_t)p*DP,digits[p].data(),DP*8,cudaMemcpyHostToDevice);
            cudaMemcpy(d_ka +(size_t)p*KP,K.av[p].data(),  KP*8,cudaMemcpyHostToDevice);
            cudaMemcpy(d_kb +(size_t)p*KP,K.bv[p].data(),  KP*8,cudaMemcpyHostToDevice);
        }
        cudaMemset(d_o0,0,OT*8);
        cudaMemset(d_o1,0,OT*8);
        for (uint32_t i=0;i<sizeQlP;++i){
            uint64_t q=qlp_mod(i);
            uint32_t idx=(i>=sizeQl)? i+delta : i;   // eval-key tower index
            for(uint32_t part=0;part<K.numPart;++part){
                LaunchKSMultAcc(d_o0+(size_t)i*n, d_o1+(size_t)i*n,
                                d_dig+(size_t)part*DP+(size_t)i*n,
                                d_kb +(size_t)part*KP+(size_t)idx*n,
                                d_ka +(size_t)part*KP+(size_t)idx*n, q, n, 0);
            }
        }
        cudaDeviceSynchronize();
        cudaMemcpy(res0.data(),d_o0,OT*8,cudaMemcpyDeviceToHost);
        cudaMemcpy(res1.data(),d_o1,OT*8,cudaMemcpyDeviceToHost);
        cudaFree(d_dig);cudaFree(d_ka);cudaFree(d_kb);cudaFree(d_o0);cudaFree(d_o1);
    }

    // === STAGE 3b: ApproxModDown QlP -> Q on each of res0, res1 ===
    auto moddown=[&](const std::vector<uint64_t>& res)->std::vector<uint64_t>{
        // P towers (high [sizeQl, sizeQlP)) -> coeff
        std::vector<uint64_t> pCoeff((size_t)sizeP*n);
        for(uint32_t j=0;j<sizeP;++j){ uint64_t q=K.pMod[j]; DevTab&T=dev_tab(n,q,root_for(q));
            std::vector<uint64_t> h(res.begin()+(size_t)(sizeQl+j)*n, res.begin()+(size_t)(sizeQl+j+1)*n);
            uint64_t *dx=up(h);
            LaunchINTT_GS(dx,T.ir,T.ip,n,q,T.ninv,T.ninv_p,0);
            cudaMemcpy(pCoeff.data()+(size_t)j*n,dx,B,cudaMemcpyDeviceToHost);
            cudaFree(dx); }
        // ApproxSwitchCRTBasis P -> Q
        uint64_t *dsrc=up(pCoeff),*dqhi=up(K.pHatInv),*dqhip=up(K.pHatInvPrec),
                 *dq=up(K.pMod),*dqmp=up(K.pHatModq),*dp=up(K.qMod),*dmlo=up(K.mdBMuLo),*dmhi=up(K.mdBMuHi);
        uint64_t* ddst; cudaMalloc(&ddst,(size_t)sizeQl*n*8);
        LaunchApproxSwitchCRTBasis(dsrc,ddst,dqhi,dqhip,dq,dqmp,dp,dmlo,dmhi,sizeP,sizeQl,n,0);
        cudaDeviceSynchronize();
        std::vector<uint64_t> qSw((size_t)sizeQl*n);
        cudaMemcpy(qSw.data(),ddst,(size_t)sizeQl*n*8,cudaMemcpyDeviceToHost);
        cudaFree(dsrc);cudaFree(dqhi);cudaFree(dqhip);cudaFree(dq);cudaFree(dqmp);cudaFree(dp);cudaFree(dmlo);cudaFree(dmhi);cudaFree(ddst);
        // NTT back + combine per Q tower
        std::vector<uint64_t> out((size_t)sizeQl*n);
        for(uint32_t i=0;i<sizeQl;++i){ uint64_t q=K.qMod[i]; DevTab&T=dev_tab(n,q,root_for(q));
            std::vector<uint64_t> h(qSw.begin()+(size_t)i*n, qSw.begin()+(size_t)(i+1)*n);
            uint64_t *dx=up(h);
            LaunchNTT_CT(dx,T.fr,T.fp,n,q,0);
            std::vector<uint64_t> ha(res.begin()+(size_t)i*n, res.begin()+(size_t)(i+1)*n);
            uint64_t *da=up(ha),*dans; cudaMalloc(&dans,B);
            LaunchModDownCombine(dans,da,dx,K.pInvModq[i],q,n,0);
            cudaMemcpy(out.data()+(size_t)i*n,dans,B,cudaMemcpyDeviceToHost);
            cudaFree(dx);cudaFree(da);cudaFree(dans); }
        return out;
    };

    KeySwitchResult R;
    R.ba0 = moddown(res0);
    R.ba1 = moddown(res1);
    return R;
}

} // namespace gpufhe
