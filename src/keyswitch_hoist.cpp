// Decompose-sharing for correct-order rotation hoisting. A NEW, separate
// implementation path (research/NEXT_SESSION.md priority queue item 1) --
// does not modify src/keyswitch_resident.cpp's logic. The per-part
// decompose body below is a deliberate duplicate of ks_accumulate_qp's
// steps 1-3 (same Launch* kernel calls, same indexing), reading from
// the already-exported DeviceKSContext fields rather than touching that
// file -- see keyswitch_resident.h's comment on that struct for why the
// decompose SHAPE constants (not av/bv) are safe to reuse across any
// context sharing the same modulus chain.
#include "keyswitch_hoist.h"
#include "ntt.h"
#include "intt.h"
#include "basis_convert.h"
#include "ks_fastcore.h"
#include <stdexcept>
#include <vector>

namespace gpufhe {
namespace {
int row_of_h(const DeviceKSContext& C, uint64_t q){
    for(size_t r=0;r<C.modList.size();++r) if(C.modList[r]==q) return (int)r;
    throw std::runtime_error("ks_decompose_only: no root table for modulus");
}
}

HoistedDigits hoist_digits_create(const DeviceKSContext& C){
    HoistedDigits D; D.numPart=C.numPart; D.n=C.n; D.sizeQlP=C.sizeQlP;
    D.d_compl.resize(D.numPart, nullptr);
    for(uint32_t p=0;p<D.numPart;++p)
        cudaMalloc(&D.d_compl[p], (size_t)D.sizeQlP*D.n*8);
    return D;
}

void hoist_digits_destroy(HoistedDigits& D){
    for(auto* p : D.d_compl) if(p) cudaFree(p);
    D.d_compl.clear(); D.numPart=D.n=D.sizeQlP=0;
}

void ks_decompose_only(const uint64_t* d_a, const DeviceKSContext& C,
                       HoistedDigits& D, cudaStream_t s)
{
    const uint32_t n=C.n, sizeQlP=C.sizeQlP;
    uint64_t* d_part; cudaMalloc(&d_part, (size_t)sizeQlP*n*8);
    uint64_t **d_rootsA,**d_rootsB; uint64_t *d_towerMods,*d_nInv,*d_nInvPrecon;
    cudaMalloc(&d_rootsA, sizeQlP*sizeof(uint64_t*));
    cudaMalloc(&d_rootsB, sizeQlP*sizeof(uint64_t*));
    cudaMalloc(&d_towerMods, sizeQlP*8);
    cudaMalloc(&d_nInv, sizeQlP*8);
    cudaMalloc(&d_nInvPrecon, sizeQlP*8);

    for(uint32_t part=0; part<C.numPart; ++part){
        const uint32_t sp=C.sizePart[part], sc=C.sizeCompl[part], st=C.startIdx[part];

        // 1. copy part slice, INTT each tower in place (same as
        // ks_accumulate_qp step 1).
        cudaMemcpyAsync(d_part, d_a+(size_t)st*n, (size_t)sp*n*8,
                        cudaMemcpyDeviceToDevice, s);
        {
            std::vector<const uint64_t*> hir(sp),hip(sp);
            std::vector<uint64_t> hmod(sp),hninv(sp),hninvp(sp);
            for(uint32_t i=0;i<sp;++i){
                uint64_t q=C.qModHost[st+i]; int r=row_of_h(C,q);
                hir[i]=C.d_ir[r]; hip[i]=C.d_ip[r];
                hmod[i]=q; hninv[i]=C.ninv[r]; hninvp[i]=C.ninv_p[r];
            }
            cudaMemcpyAsync(d_rootsA,hir.data(),sp*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(d_rootsB,hip.data(),sp*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(d_towerMods,hmod.data(),sp*8,cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(d_nInv,hninv.data(),sp*8,cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(d_nInvPrecon,hninvp.data(),sp*8,cudaMemcpyHostToDevice,s);
            LaunchINTT_GS_Batched(d_part, d_rootsA, d_rootsB, d_towerMods,
                                  d_nInv, d_nInvPrecon, sp, n, s);
        }
        // 2. basis convert partQ -> complement, into THIS part's owned
        // buffer (not a shared scratch -- that's the whole point: all
        // numPart results must stay live simultaneously for reuse).
        LaunchApproxSwitchCRTBasis(d_part, D.d_compl[part],
            C.d_qhi[part], C.d_qhip[part], C.d_srcMod[part],
            C.d_qmp[part], C.d_complMod[part], C.d_mlo[part], C.d_mhi[part],
            sp, sc, n, s);
        // 3. NTT each complement tower back to eval.
        {
            std::vector<const uint64_t*> hfr(sc),hfp(sc);
            std::vector<uint64_t> hmod(sc);
            for(uint32_t j=0;j<sc;++j){
                uint64_t q=C.complModHost[part][j]; int r=row_of_h(C,q);
                hfr[j]=C.d_fr[r]; hfp[j]=C.d_fp[r]; hmod[j]=q;
            }
            cudaMemcpyAsync(d_rootsA,hfr.data(),sc*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(d_rootsB,hfp.data(),sc*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(d_towerMods,hmod.data(),sc*8,cudaMemcpyHostToDevice,s);
            LaunchNTT_CT_Batched(D.d_compl[part], d_rootsA, d_rootsB, d_towerMods, sc, n, s);
        }
    }
    // The scratch buffers below are reused across all numPart iterations
    // above via async copies on the SAME stream `s`, so stream ordering
    // alone guarantees every launch that reads them has already completed
    // relative to the next iteration's overwrite. The explicit sync here
    // is only to make freeing them immediately after return safe (the
    // caller may free/reuse d_a right away otherwise).
    cudaStreamSynchronize(s);
    cudaFree(d_part); cudaFree(d_rootsA); cudaFree(d_rootsB);
    cudaFree(d_towerMods); cudaFree(d_nInv); cudaFree(d_nInvPrecon);
}

void ks_accumulate_from_digits(const uint64_t* d_a, const HoistedDigits& D,
                               const DeviceKSContext& C, DeviceKSWork& W,
                               cudaStream_t s)
{
    const uint32_t n=C.n, sizeQlP=C.sizeQlP;
    cudaMemsetAsync(W.d_res0,0,(size_t)sizeQlP*n*8,s);
    cudaMemsetAsync(W.d_res1,0,(size_t)sizeQlP*n*8,s);
    for(uint32_t part=0;part<C.numPart;++part){
        const uint32_t sp=C.sizePart[part], st=C.startIdx[part], en=st+sp;
        std::vector<const uint64_t*> hp(sizeQlP);
        for(uint32_t i=0;i<sizeQlP;++i)
            hp[i] = (i>=st && i<en) ? d_a+(size_t)i*n
                  : (i<st)          ? D.d_compl[part]+(size_t)i*n
                                    : D.d_compl[part]+(size_t)(i-sp)*n;
        cudaMemcpyAsync(W.d_digPtrs,hp.data(),sizeQlP*sizeof(uint64_t*),
                        cudaMemcpyHostToDevice,s);
        LaunchKSMultAccBatched(W.d_res0, W.d_res1, W.d_digPtrs,
                               C.d_bv[part], C.d_av[part],
                               C.d_keyRow, C.d_modsQlP, sizeQlP, n, s);
    }
}

} // namespace gpufhe
