#include "keyswitch_resident.h"
#include "ntt.h"
#include "intt.h"
#include "basis_convert.h"
#include "ks_fastcore.h"
#include "moddown.h"
#include "switch_modulus.h"
#include "rescale.h"
#include <cstring>
#include <stdexcept>
#include <map>
#include <map>
#include <utility>

namespace gpufhe {
bool g_ks_batched=true;   // A/B switch for the batched fastcore
namespace {
uint64_t mulmod(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
uint64_t powmod(uint64_t b,uint64_t e,uint64_t q){unsigned __int128 r=1,bb=b%q;while(e){if(e&1)r=(r*bb)%q;bb=(bb*bb)%q;e>>=1;}return(uint64_t)r;}
uint64_t invmod(uint64_t a,uint64_t q){return powmod(a%q,q-2,q);}
uint64_t* upv(const std::vector<uint64_t>& h, cudaStream_t s=0){ uint64_t* d;
    cudaMalloc(&d,h.size()*8);
    cudaMemcpyAsync(d,h.data(),h.size()*8,cudaMemcpyHostToDevice,s);
    return d; }
// These constants depend only on (n,sizeQl,numPart) -- NOT on the key -- yet
// every rotation rebuilt ~135 identical allocations. Cache by that tuple; only
// av/bv stay per-key. Contexts do not own these (destroy must not free them).
uint64_t* upc(uint64_t key, const std::vector<uint64_t>& h){
    static std::map<uint64_t,uint64_t*> m;
    auto it=m.find(key); if(it!=m.end()) return it->second;
    uint64_t* d=upv(h); m.emplace(key,d); return d; }
} // anon

DeviceKSContext ks_context_create(const KeySwitchConstants& K, cudaStream_t s){
    DeviceKSContext C;
    C.n=K.n; C.sizeQl=K.sizeQl; C.sizeP=K.sizeP; C.sizeQlP=K.sizeQl+K.sizeP;
    C.numPart=K.numPart; C.fullQ=K.fullQ;
    const uint32_t n=K.n;

    // root tables for every modulus in the root list (built like keyswitch.cpp)
    auto br=[](uint32_t v,uint32_t b){uint32_t r=0;for(uint32_t k=0;k<b;k++){r=(r<<1)|(v&1);v>>=1;}return r;};
    uint32_t logn=0; while((1u<<logn)<n)++logn;
    // Root tables depend only on (n,q) and are IDENTICAL across every context,
    // but each context was rebuilding and re-uploading all of them: 4 cudaMallocs
    // per modulus x 32 moduli = 128 allocations per rotation, and the measured
    // ks_context_create cost was 54 ms of a 170 ms rotation. Cache them for the
    // process lifetime and share the pointers; contexts no longer own them
    // (ks_context_destroy must not free these).
    struct RTab { uint64_t *fr,*fp,*ir,*ip, ninv, ninv_p; };
    static std::map<std::pair<uint32_t,uint64_t>,RTab> g_rtab;
    for(size_t r=0;r<K.rootModList.size();++r){
        uint64_t q=K.rootModList[r], root=K.rootValList[r];
        auto key=std::make_pair(n,q);
        auto it=g_rtab.find(key);
        if(it==g_rtab.end()){
            uint64_t rootInv=invmod(root,q);
            std::vector<uint64_t> fr(n),fp(n),ir(n),ip(n),pw(n);
            auto fill=[&](uint64_t base,std::vector<uint64_t>&tr,std::vector<uint64_t>&tp){
                pw[0]=1; for(uint32_t i=1;i<n;++i)pw[i]=mulmod(pw[i-1],base,q);
                for(uint32_t i=0;i<n;++i){uint64_t ri=pw[br(i,logn)];tr[i]=ri;tp[i]=(uint64_t)(((__uint128_t)ri<<64)/q);}};
            fill(root,fr,fp); fill(rootInv,ir,ip);
            uint64_t ni=invmod(n,q);
            RTab t{upv(fr),upv(fp),upv(ir),upv(ip),ni,(uint64_t)(((__uint128_t)ni<<64)/q)};
            it=g_rtab.emplace(key,t).first;
        }
        C.modList.push_back(q);
        C.d_fr.push_back(it->second.fr); C.d_fp.push_back(it->second.fp);
        C.d_ir.push_back(it->second.ir); C.d_ip.push_back(it->second.ip);
        C.ninv.push_back(it->second.ninv); C.ninv_p.push_back(it->second.ninv_p);
    }

    // per-part constants
    const uint64_t KB=((uint64_t)n<<40)|((uint64_t)K.sizeQl<<32)|((uint64_t)K.numPart<<24);
    C.sizePart=K.sizePart; C.sizeCompl=K.sizeCompl; C.startIdx=K.startIdx;
    for(uint32_t p=0;p<K.numPart;++p){
        C.complModHost.push_back(K.partComplMod[p]);
        C.d_qhi.push_back(upc(KB|((uint64_t)p<<8)|0,K.partQHatInv[p]));
        C.d_qhip.push_back(upc(KB|((uint64_t)p<<8)|1,K.partQHatInvPrec[p]));
        C.d_srcMod.push_back(upc(KB|((uint64_t)p<<8)|2,K.partSrcMod[p]));
        C.d_qmp.push_back(upc(KB|((uint64_t)p<<8)|3,K.partQHatModp[p]));
        C.d_complMod.push_back(upc(KB|((uint64_t)p<<8)|4,K.partComplMod[p]));
        C.d_mlo.push_back(upc(KB|((uint64_t)p<<8)|5,K.partBMuLo[p]));
        C.d_mhi.push_back(upc(KB|((uint64_t)p<<8)|6,K.partBMuHi[p]));
        C.d_av.push_back(upv(K.av[p], s));
        C.d_bv.push_back(upv(K.bv[p], s));
    }

    // moddown constants
    C.d_pHatInv=upc(KB|100,K.pHatInv); C.d_pHatInvPrec=upc(KB|101,K.pHatInvPrec);
    C.d_pMod=upc(KB|102,K.pMod); C.d_pHatModq=upc(KB|103,K.pHatModq);
    C.d_qMod=upc(KB|104,K.qMod); C.d_mdMuLo=upc(KB|105,K.mdBMuLo); C.d_mdMuHi=upc(KB|106,K.mdBMuHi);
    C.pInvModqHost=K.pInvModq; C.qModHost=K.qMod; C.pModHost=K.pMod;
    {   // constants for the batched fastcore launch (same every call)
        std::vector<uint64_t> mq(C.sizeQlP); std::vector<uint32_t> kr(C.sizeQlP);
        const uint32_t dl=C.fullQ-C.sizeQl;
        for(uint32_t i=0;i<C.sizeQlP;++i){
            mq[i]=(i<C.sizeQl)? K.qMod[i] : K.pMod[i-C.sizeQl];
            kr[i]=(i>=C.sizeQl)? i+dl : i; }
        C.d_modsQlP=upc(KB|200,mq);
        cudaMalloc(&C.d_keyRow,C.sizeQlP*4);
        // kr is pageable; cudaMemcpyAsync falls back to sync here -- do not pin kr without adding an explicit cudaStreamSynchronize first.
        cudaMemcpyAsync(C.d_keyRow,kr.data(),C.sizeQlP*4,cudaMemcpyHostToDevice,s); }
    return C;
}

DeviceKSWork ks_work_create(const DeviceKSContext& C){
    DeviceKSWork W; const size_t n=C.n;
    uint32_t maxPart=0; for(auto s:C.sizePart) maxPart=std::max(maxPart,s);
    cudaMalloc(&W.d_part,(size_t)maxPart*n*8);
    cudaMalloc(&W.d_compl,(size_t)C.sizeQlP*n*8);
    cudaMalloc(&W.d_res0,(size_t)C.sizeQlP*n*8);
    cudaMalloc(&W.d_res1,(size_t)C.sizeQlP*n*8);
    cudaMalloc(&W.d_pwork,(size_t)C.sizeP*n*8);
    cudaMalloc(&W.d_qsw,(size_t)C.sizeQl*n*8);
    cudaMalloc(&W.d_digPtrs,(size_t)C.sizeQlP*sizeof(uint64_t*));
    cudaMalloc(&W.d_rootsA,(size_t)C.sizeQlP*sizeof(uint64_t*));
    cudaMalloc(&W.d_rootsB,(size_t)C.sizeQlP*sizeof(uint64_t*));
    cudaMalloc(&W.d_towerMods,(size_t)C.sizeQlP*8);
    cudaMalloc(&W.d_nInv,(size_t)C.sizeQlP*8);
    cudaMalloc(&W.d_nInvPrecon,(size_t)C.sizeQlP*8);
    return W;
}

void ks_context_destroy(DeviceKSContext& C){
    // NOTE: d_fr/d_fp/d_ir/d_ip point into the process-lifetime root-table
    // cache and are SHARED across contexts -- freeing them here would corrupt
    // every context created later. Only the per-context arrays are freed.
    
    
    
    for(auto p:C.d_av)cudaFree(p);
    for(auto p:C.d_bv)cudaFree(p);
    
    
}
void ks_work_destroy(DeviceKSWork& W){
    cudaFree(W.d_part);cudaFree(W.d_compl);cudaFree(W.d_res0);cudaFree(W.d_res1);
    cudaFree(W.d_pwork);cudaFree(W.d_qsw);cudaFree(W.d_digPtrs);
    cudaFree(W.d_rootsA);cudaFree(W.d_rootsB);cudaFree(W.d_towerMods);cudaFree(W.d_nInv);cudaFree(W.d_nInvPrecon);
}

// modulus -> root-table row
static int row_of(const DeviceKSContext& C, uint64_t q){
    for(size_t r=0;r<C.modList.size();++r) if(C.modList[r]==q) return (int)r;
    throw std::runtime_error("no root table for modulus");
}

void keyswitch_resident(const uint64_t* d_a, uint64_t* d_ba0, uint64_t* d_ba1,
                        const DeviceKSContext& C, DeviceKSWork& W, cudaStream_t s)
{
    const uint32_t n=C.n, sizeQl=C.sizeQl, sizeP=C.sizeP, sizeQlP=C.sizeQlP;
    const uint32_t delta=C.fullQ - sizeQl;
    const size_t B=(size_t)n*8;

    cudaMemsetAsync(W.d_res0,0,(size_t)sizeQlP*n*8,s);
    cudaMemsetAsync(W.d_res1,0,(size_t)sizeQlP*n*8,s);

    for(uint32_t part=0;part<C.numPart;++part){
        const uint32_t sp=C.sizePart[part], sc=C.sizeCompl[part], st=C.startIdx[part];
        const uint32_t en=st+sp;

        // 1. copy part slice (keep d_a intact), INTT each tower in place
        cudaMemcpyAsync(W.d_part, d_a+(size_t)st*n, (size_t)sp*n*8,
                        cudaMemcpyDeviceToDevice, s);
        {
            std::vector<const uint64_t*> hir(sp),hip(sp);
            std::vector<uint64_t> hmod(sp),hninv(sp),hninvp(sp);
            for(uint32_t i=0;i<sp;++i){
                uint64_t q=C.qModHost[st+i]; int r=row_of(C,q);
                hir[i]=C.d_ir[r]; hip[i]=C.d_ip[r];
                hmod[i]=q; hninv[i]=C.ninv[r]; hninvp[i]=C.ninv_p[r];
            }
            cudaMemcpyAsync(W.d_rootsA,hir.data(),sp*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(W.d_rootsB,hip.data(),sp*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(W.d_towerMods,hmod.data(),sp*8,cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(W.d_nInv,hninv.data(),sp*8,cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(W.d_nInvPrecon,hninvp.data(),sp*8,cudaMemcpyHostToDevice,s);
            LaunchINTT_GS_Batched(W.d_part, W.d_rootsA, W.d_rootsB, W.d_towerMods,
                                  W.d_nInv, W.d_nInvPrecon, sp, n, s);
        }
        // 2. basis convert partQ -> complement
        LaunchApproxSwitchCRTBasis(W.d_part, W.d_compl,
            C.d_qhi[part], C.d_qhip[part], C.d_srcMod[part],
            C.d_qmp[part], C.d_complMod[part], C.d_mlo[part], C.d_mhi[part],
            sp, sc, n, s);
        // 3. NTT each complement tower back to eval
        {
            std::vector<const uint64_t*> hfr(sc),hfp(sc);
            std::vector<uint64_t> hmod(sc);
            for(uint32_t j=0;j<sc;++j){
                uint64_t q=C.complModHost[part][j]; int r=row_of(C,q);
                hfr[j]=C.d_fr[r]; hfp[j]=C.d_fp[r]; hmod[j]=q;
            }
            cudaMemcpyAsync(W.d_rootsA,hfr.data(),sc*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(W.d_rootsB,hfp.data(),sc*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(W.d_towerMods,hmod.data(),sc*8,cudaMemcpyHostToDevice,s);
            LaunchNTT_CT_Batched(W.d_compl, W.d_rootsA, W.d_rootsB, W.d_towerMods, sc, n, s);
        }
        // 4. accumulate: digit tower i = d_a (in range) or d_compl (else).
        // BATCHED: was sizeQlP launches per part (32*15 = 480 per keyswitch at
        // n=8192/tw=30, each on a single 8192-element tower => launch-bound).
        // Now one launch per part with grid.y = tower, as tensor.cu does.
        // The digit source differs per tower (inside the part's range it is d_a,
        // otherwise d_compl with a shifted index), so pass a pointer array
        // rather than branching in the kernel.
        if(!g_ks_batched){
            for(uint32_t i=0;i<sizeQlP;++i){
                uint64_t q=(i<sizeQl)? C.qModHost[i] : C.pModHost[i-sizeQl];
                uint32_t idx=(i>=sizeQl)? i+delta : i;
                const uint64_t* dig = (i>=st && i<en) ? d_a+(size_t)i*n
                                    : (i<st) ? W.d_compl+(size_t)i*n
                                             : W.d_compl+(size_t)(i-sp)*n;
                LaunchKSMultAcc(W.d_res0+(size_t)i*n, W.d_res1+(size_t)i*n,
                                dig, C.d_bv[part]+(size_t)idx*n, C.d_av[part]+(size_t)idx*n,
                                q, n, s);
            }
        } else
        {   std::vector<const uint64_t*> hp(sizeQlP);
            for(uint32_t i=0;i<sizeQlP;++i)
                hp[i] = (i>=st && i<en) ? d_a+(size_t)i*n
                      : (i<st)          ? W.d_compl+(size_t)i*n
                                        : W.d_compl+(size_t)(i-sp)*n;
            cudaMemcpyAsync(W.d_digPtrs,hp.data(),sizeQlP*sizeof(uint64_t*),
                            cudaMemcpyHostToDevice,s);
            LaunchKSMultAccBatched(W.d_res0, W.d_res1, W.d_digPtrs,
                                   C.d_bv[part], C.d_av[part],
                                   C.d_keyRow, C.d_modsQlP, sizeQlP, n, s);
        }
    }

    // moddown QlP->Q on res0 -> ba0, res1 -> ba1
    auto moddown=[&](uint64_t* d_res, uint64_t* d_out){
        cudaMemcpyAsync(W.d_pwork, d_res+(size_t)sizeQl*n, (size_t)sizeP*n*8,
                        cudaMemcpyDeviceToDevice, s);
        {
            std::vector<const uint64_t*> hir(sizeP),hip(sizeP);
            std::vector<uint64_t> hmod(sizeP),hninv(sizeP),hninvp(sizeP);
            for(uint32_t j=0;j<sizeP;++j){
                uint64_t q=C.pModHost[j]; int r=row_of(C,q);
                hir[j]=C.d_ir[r]; hip[j]=C.d_ip[r];
                hmod[j]=q; hninv[j]=C.ninv[r]; hninvp[j]=C.ninv_p[r];
            }
            cudaMemcpyAsync(W.d_rootsA,hir.data(),sizeP*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(W.d_rootsB,hip.data(),sizeP*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(W.d_towerMods,hmod.data(),sizeP*8,cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(W.d_nInv,hninv.data(),sizeP*8,cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(W.d_nInvPrecon,hninvp.data(),sizeP*8,cudaMemcpyHostToDevice,s);
            LaunchINTT_GS_Batched(W.d_pwork, W.d_rootsA, W.d_rootsB, W.d_towerMods,
                                  W.d_nInv, W.d_nInvPrecon, sizeP, n, s);
        }
        LaunchApproxSwitchCRTBasis(W.d_pwork, W.d_qsw,
            C.d_pHatInv, C.d_pHatInvPrec, C.d_pMod,
            C.d_pHatModq, C.d_qMod, C.d_mdMuLo, C.d_mdMuHi,
            sizeP, sizeQl, n, s);
        {
            std::vector<const uint64_t*> hfr(sizeQl),hfp(sizeQl);
            std::vector<uint64_t> hmod(sizeQl);
            for(uint32_t i=0;i<sizeQl;++i){
                uint64_t q=C.qModHost[i]; int r=row_of(C,q);
                hfr[i]=C.d_fr[r]; hfp[i]=C.d_fp[r]; hmod[i]=q;
            }
            cudaMemcpyAsync(W.d_rootsA,hfr.data(),sizeQl*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(W.d_rootsB,hfp.data(),sizeQl*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(W.d_towerMods,hmod.data(),sizeQl*8,cudaMemcpyHostToDevice,s);
            LaunchNTT_CT_Batched(W.d_qsw, W.d_rootsA, W.d_rootsB, W.d_towerMods, sizeQl, n, s);
        }
        for(uint32_t i=0;i<sizeQl;++i){
            uint64_t q=C.qModHost[i];
            LaunchModDownCombine(d_out+(size_t)i*n, d_res+(size_t)i*n,
                                 W.d_qsw+(size_t)i*n, C.pInvModqHost[i], q, n, s);
        }
    };
    moddown(W.d_res0, d_ba0);
    moddown(W.d_res1, d_ba1);
}

void rescale_resident_raw(uint64_t* d_c, uint32_t towers,
                          const DeviceKSContext& C,
                          const std::vector<uint64_t>& s1, const std::vector<uint64_t>& s2,
                          uint64_t* d_scratch, uint64_t* d_drop, cudaStream_t s)
{
    const uint32_t n=C.n; if(towers<2) return;
    const uint32_t last=towers-1;
    uint64_t qLast=C.qModHost[last]; int rl=row_of(C,qLast);
    cudaMemcpyAsync(d_drop, d_c+(size_t)last*n, (size_t)n*8, cudaMemcpyDeviceToDevice, s);
    LaunchINTT_GS(d_drop, C.d_ir[rl], C.d_ip[rl], n, qLast, C.ninv[rl], C.ninv_p[rl], s);
    for(uint32_t t=0;t<last;++t){
        uint64_t qi=C.qModHost[t]; int r=row_of(C,qi);
        cudaMemcpyAsync(d_scratch, d_drop, (size_t)n*8, cudaMemcpyDeviceToDevice, s);
        LaunchSwitchModulus(d_scratch, qLast, qi, n, s);
        LaunchNTT_CT(d_scratch, C.d_fr[r], C.d_fp[r], n, qi, s);
        LaunchRescaleFuse(d_c+(size_t)t*n, d_scratch, s1[t], s2[t], qi, n, s);
    }
}

} // namespace gpufhe
