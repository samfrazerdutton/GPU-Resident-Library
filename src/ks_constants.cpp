// Native computation of the Hybrid keyswitch constant families, from borrowed
// moduli (moduliQ, moduliP) + roots. Reproduces rns-cryptoparameters.cpp
// PrecomputeCRTTables (48-320) WITHOUT bignum: every constant is `X mod q_i`,
// computed by incremental modular folding (fold each prime into the running
// residue mod q_i), so no product ever overflows. Fills KeySwitchConstants.
#include "keyswitch.h"
#include <cstdint>
#include <vector>
#include <cmath>

namespace gpufhe {
namespace {

uint64_t mulmod(uint64_t a, uint64_t b, uint64_t q){ return (uint64_t)(((unsigned __int128)a*b)%q); }
uint64_t powmod(uint64_t b, uint64_t e, uint64_t q){ unsigned __int128 r=1,bb=b%q;
    while(e){ if(e&1)r=(r*bb)%q; bb=(bb*bb)%q; e>>=1;} return (uint64_t)r; }
uint64_t invmod(uint64_t a, uint64_t q){ return powmod(a%q, q-2, q); } // q prime
// Shoup precon = floor(val * 2^64 / q)  (== NativeInteger::PrepModMulConst)
uint64_t precon(uint64_t val, uint64_t q){ return (uint64_t)(((unsigned __int128)val<<64)/q); }

// product of all moduli in `mods` (except optional skip index), reduced mod q.
uint64_t prod_mod(const std::vector<uint64_t>& mods, uint64_t q, int skip=-1){
    uint64_t r=1%q;
    for(size_t k=0;k<mods.size();++k){ if((int)k==skip) continue; r=mulmod(r, mods[k]%q, q); }
    return r;
}
} // anon

// Fills the ModDown + decompose constant families of K, given moduliQ/moduliP.
// (Root list is filled separately by the caller from borrowed roots.)
void compute_keyswitch_constants(
    KeySwitchConstants& K,
    const std::vector<uint64_t>& moduliQ,
    const std::vector<uint64_t>& moduliP,
    uint32_t numPart)
{
    const uint32_t sizeQ=(uint32_t)moduliQ.size();
    const uint32_t sizeP=(uint32_t)moduliP.size();
    const uint32_t alpha=(uint32_t)std::ceil((double)sizeQ/numPart);

    K.n=K.n; // caller sets n
    K.sizeQl=sizeQ; K.sizeP=sizeP; K.numPart=numPart; K.alpha=alpha; K.fullQ=sizeQ;
    K.qMod=moduliQ; K.pMod=moduliP;

    // ---- ModDown family (P->Q) ----
    // PModq[i] = P mod q_i ; PInvModq[i] = P^{-1} mod q_i (+precon)
    K.pInvModq.resize(sizeQ);
    for(uint32_t i=0;i<sizeQ;++i){ uint64_t q=moduliQ[i];
        uint64_t Pmodq=prod_mod(moduliP,q);
        K.pInvModq[i]=invmod(Pmodq,q); }
    // PHatInvModp[j]=(P/p_j)^{-1} mod p_j (+precon); PHatModq[j][i]=(P/p_j) mod q_i
    K.pHatInv.resize(sizeP); K.pHatInvPrec.resize(sizeP);
    K.pHatModq.resize((size_t)sizeP*sizeQ);
    for(uint32_t j=0;j<sizeP;++j){ uint64_t pj=moduliP[j];
        uint64_t PHatModpj=prod_mod(moduliP,pj,(int)j);   // (P/p_j) mod p_j
        K.pHatInv[j]=invmod(PHatModpj,pj);
        K.pHatInvPrec[j]=precon(K.pHatInv[j],pj);
        for(uint32_t i=0;i<sizeQ;++i){ uint64_t q=moduliQ[i];
            K.pHatModq[(size_t)j*sizeQ+i]=prod_mod(moduliP,q,(int)j); } }
    // ModDown Barrett mu per Q tower = floor(2^128 / q_i)
    K.mdBMuLo.resize(sizeQ); K.mdBMuHi.resize(sizeQ);
    for(uint32_t i=0;i<sizeQ;++i){ unsigned __int128 mu=(~(unsigned __int128)0)/moduliQ[i];
        K.mdBMuLo[i]=(uint64_t)mu; K.mdBMuHi[i]=(uint64_t)(mu>>64); }

    // ---- Decompose families (per part), full sublevel (l=0), matching what
    // keyswitch_core_resident consumes. Part p owns towers [alpha*p, +sizePart).
    K.sizePart.resize(numPart); K.sizeCompl.resize(numPart); K.startIdx.resize(numPart);
    K.partSrcMod.resize(numPart); K.partComplMod.resize(numPart);
    K.partQHatInv.resize(numPart); K.partQHatInvPrec.resize(numPart);
    K.partQHatModp.resize(numPart); K.partBMuLo.resize(numPart); K.partBMuHi.resize(numPart);
    for(uint32_t part=0; part<numPart; ++part){
        uint32_t startIdx=alpha*part;
        uint32_t sizePart=(sizeQ>startIdx+alpha)?alpha:(sizeQ-startIdx);
        K.startIdx[part]=startIdx; K.sizePart[part]=sizePart;
        // source moduli = this part's Q towers
        std::vector<uint64_t> src(sizePart);
        for(uint32_t i=0;i<sizePart;++i) src[i]=moduliQ[startIdx+i];
        K.partSrcMod[part]=src;
        // PartQHatInvModq[i] = (partQ/q_i)^{-1} mod q_i (full sublevel)
        K.partQHatInv[part].resize(sizePart); K.partQHatInvPrec[part].resize(sizePart);
        for(uint32_t i=0;i<sizePart;++i){ uint64_t q=src[i];
            uint64_t QHat=prod_mod(src,q,(int)i);       // (partQ/q_i) mod q_i
            K.partQHatInv[part][i]=invmod(QHat,q);
            K.partQHatInvPrec[part][i]=precon(K.partQHatInv[part][i],q); }
        // complement basis = all OTHER Q towers (not this part) + all P towers.
        // Order must match OpenFHE's GetParamsComplPartQ(sizeQ-1,part):
        //   first the non-part Q towers in index order, then the P towers.
        std::vector<uint64_t> compl_mod;
        for(uint32_t i=0;i<sizeQ;++i) if(i<startIdx || i>=startIdx+sizePart) compl_mod.push_back(moduliQ[i]);
        for(uint32_t j=0;j<sizeP;++j) compl_mod.push_back(moduliP[j]);
        uint32_t sizeCompl=(uint32_t)compl_mod.size();
        K.sizeCompl[part]=sizeCompl; K.partComplMod[part]=compl_mod;
        // PartQHatModp[i][j] = (partQ/q_i) mod compl_j   [sizePart x sizeCompl]
        K.partQHatModp[part].resize((size_t)sizePart*sizeCompl);
        for(uint32_t i=0;i<sizePart;++i)for(uint32_t j=0;j<sizeCompl;++j)
            K.partQHatModp[part][(size_t)i*sizeCompl+j]=prod_mod(src, compl_mod[j], (int)i);
        // complement Barrett mu = floor(2^128 / compl_j)
        K.partBMuLo[part].resize(sizeCompl); K.partBMuHi[part].resize(sizeCompl);
        for(uint32_t j=0;j<sizeCompl;++j){ unsigned __int128 mu=(~(unsigned __int128)0)/compl_mod[j];
            K.partBMuLo[part][j]=(uint64_t)mu; K.partBMuHi[part][j]=(uint64_t)(mu>>64); }
    }
}

} // namespace gpufhe
