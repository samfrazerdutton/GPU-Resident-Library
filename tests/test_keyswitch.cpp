// Stage 5 capstone: full Hybrid KeySwitchCore composed as a library function.
// Extract KeySwitchConstants from OpenFHE (borrowed key), run
// keyswitch_core_resident, compare ba0/ba1 against OpenFHE's KeySwitchCore
// (decompose->fastcore->ModDown) bit-exact. All sub-ops proven; this proves
// the composition + the OpenFHE-free core interface.

#include "openfhe.h"
#include "keyswitch.h"
#include <iostream>
#include <random>
#include <vector>

using namespace lbcrypto;
using gpufhe::KeySwitchConstants;

int main() {
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(3); params.SetScalingModSize(50);
    params.SetRingDim(32768); params.SetBatchSize(16384);
    params.SetKeySwitchTechnique(HYBRID);
    auto cc = GenCryptoContext(params);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE); cc->Enable(ADVANCEDSHE);
    auto kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);

    auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto ep = cp->GetElementParams();
    const uint32_t n = ep->GetRingDimension();
    const uint32_t sizeQl = (uint32_t)ep->GetParams().size();
    auto paramsP = cp->GetParamsP();
    const uint32_t sizeP = (uint32_t)paramsP->GetParams().size();
    const uint32_t sizeQlP = sizeQl + sizeP;
    const uint32_t fullQ = sizeQl; // fresh ct at full level

    // Input a = random poly over Ql (stands in for cv.back()), eval form.
    std::mt19937_64 rng(20260731);
    std::vector<uint64_t> aTowers((size_t)sizeQl*n);
    using Poly = DCRTPoly::PolyType;
    std::vector<Poly> tv; tv.reserve(sizeQl);
    for (uint32_t i=0;i<sizeQl;++i){ auto pp=ep->GetParams()[i];
        uint64_t q=pp->GetModulus().ConvertToInt();
        NativeVector v(n,pp->GetModulus());
        for(uint32_t k=0;k<n;++k){ uint64_t x=rng()%q; v[k]=NativeInteger(x); aTowers[(size_t)i*n+k]=x; }
        Poly p(pp,Format::EVALUATION,true); p.SetValues(std::move(v),Format::EVALUATION); tv.push_back(std::move(p)); }
    DCRTPoly a(tv);

    // Borrow eval key.
    auto evalKeyVec = CryptoContextImpl<DCRTPoly>::GetEvalMultKeyVector(kp.secretKey->GetKeyTag());
    auto evalKey = evalKeyVec[0];
    const auto& av = evalKey->GetAVector();
    const auto& bv = evalKey->GetBVector();

    // OpenFHE reference: KeySwitchCore(a, evalKey).
    auto ref = cc->GetScheme()->KeySwitchCore(a, evalKey);

    // ===== extract KeySwitchConstants =====
    KeySwitchConstants K;
    K.n=n; K.sizeQl=sizeQl; K.sizeP=sizeP; K.fullQ=fullQ;
    K.alpha = cp->GetNumPerPartQ();
    uint32_t numPart = (uint32_t)std::ceil((double)sizeQl/K.alpha);
    if (numPart > cp->GetNumberOfQPartitions()) numPart = cp->GetNumberOfQPartitions();
    K.numPart = numPart;
    for(uint32_t i=0;i<sizeQl;++i) K.qMod.push_back(ep->GetParams()[i]->GetModulus().ConvertToInt());
    for(uint32_t j=0;j<sizeP;++j) K.pMod.push_back(paramsP->GetParams()[j]->GetModulus().ConvertToInt());

    // Per-part decompose constants.
    K.sizePart.resize(numPart); K.sizeCompl.resize(numPart); K.startIdx.resize(numPart);
    K.partQHatInv.resize(numPart); K.partQHatInvPrec.resize(numPart); K.partSrcMod.resize(numPart);
    K.partQHatModp.resize(numPart); K.partComplMod.resize(numPart);
    K.partBMuLo.resize(numPart); K.partBMuHi.resize(numPart);
    for (uint32_t part=0; part<numPart; ++part){
        uint32_t startIdx=K.alpha*part;
        uint32_t sizePart=(sizeQl>startIdx+K.alpha)?K.alpha:(sizeQl-startIdx);
        auto paramsPartQ=cp->GetParamsPartQ(part);
        auto paramsCompl=cp->GetParamsComplPartQ(sizeQl-1,part);
        uint32_t sizeCompl=(uint32_t)paramsCompl->GetParams().size();
        K.startIdx[part]=startIdx; K.sizePart[part]=sizePart; K.sizeCompl[part]=sizeCompl;

        const auto& qhi=cp->GetPartQlHatInvModq(part,sizePart-1);
        const auto& qhip=cp->GetPartQlHatInvModqPrecon(part,sizePart-1);
        const auto& qmp=cp->GetPartQlHatModp(sizeQl-1,part);
        const auto& bmu=cp->GetmodComplPartqBarrettMu(sizeQl-1,part);
        for(uint32_t i=0;i<sizePart;++i){ K.partQHatInv[part].push_back(qhi[i].ConvertToInt());
            K.partQHatInvPrec[part].push_back(qhip[i].ConvertToInt());
            K.partSrcMod[part].push_back(paramsPartQ->GetParams()[i]->GetModulus().ConvertToInt()); }
        K.partQHatModp[part].resize((size_t)sizePart*sizeCompl);
        for(uint32_t i=0;i<sizePart;++i)for(uint32_t j=0;j<sizeCompl;++j)
            K.partQHatModp[part][(size_t)i*sizeCompl+j]=qmp[i][j].ConvertToInt();
        for(uint32_t j=0;j<sizeCompl;++j){ K.partComplMod[part].push_back(paramsCompl->GetParams()[j]->GetModulus().ConvertToInt());
            unsigned __int128 mu=(unsigned __int128)bmu[j];
            K.partBMuLo[part].push_back((uint64_t)mu); K.partBMuHi[part].push_back((uint64_t)(mu>>64)); }
    }

    // Eval key av/bv flattened per part over full QP towers.
    K.evalKeyTowers = (uint32_t)av[0].GetAllElements().size();
    K.av.resize(numPart); K.bv.resize(numPart);
    for (uint32_t part=0; part<numPart; ++part){
        uint32_t kt=K.evalKeyTowers;
        K.av[part].resize((size_t)kt*n); K.bv[part].resize((size_t)kt*n);
        for(uint32_t t=0;t<kt;++t)for(uint32_t k=0;k<n;++k){
            K.av[part][(size_t)t*n+k]=av[part].GetAllElements()[t][k].ConvertToInt();
            K.bv[part][(size_t)t*n+k]=bv[part].GetAllElements()[t][k].ConvertToInt(); }
    }

    // ModDown constants (arg-less).
    for(uint32_t j=0;j<sizeP;++j){ K.pHatInv.push_back(cp->GetPHatInvModp()[j].ConvertToInt());
        K.pHatInvPrec.push_back(cp->GetPHatInvModpPrecon()[j].ConvertToInt()); }
    K.pHatModq.resize((size_t)sizeP*sizeQl);
    for(uint32_t i=0;i<sizeP;++i)for(uint32_t j=0;j<sizeQl;++j)
        K.pHatModq[(size_t)i*sizeQl+j]=cp->GetPHatModq()[i][j].ConvertToInt();
    for(uint32_t j=0;j<sizeQl;++j){ unsigned __int128 mu=(unsigned __int128)cp->GetModqBarrettMu()[j];
        K.mdBMuLo.push_back((uint64_t)mu); K.mdBMuHi.push_back((uint64_t)(mu>>64)); }
    for(uint32_t i=0;i<sizeQl;++i){ K.pInvModq.push_back(cp->GetPInvModq()[i].ConvertToInt()); }

    // OpenFHE's exact stored root of unity per modulus (Q and P), so the
    // resident INTT/NTT match OpenFHE's transform convention.
    for(uint32_t i=0;i<sizeQl;++i){
        K.rootModList.push_back(ep->GetParams()[i]->GetModulus().ConvertToInt());
        K.rootValList.push_back(ep->GetParams()[i]->GetRootOfUnity().ConvertToInt()); }
    for(uint32_t j=0;j<sizeP;++j){
        K.rootModList.push_back(paramsP->GetParams()[j]->GetModulus().ConvertToInt());
        K.rootValList.push_back(paramsP->GetParams()[j]->GetRootOfUnity().ConvertToInt()); }

    // ===== run resident =====
    auto R = gpufhe::keyswitch_core_resident(aTowers, K);

    // Compare ba0/ba1 vs OpenFHE ref (both sizeQl towers, eval).
    uint32_t bad0=0,bad1=0; int f0=-1,f1=-1;
    for(uint32_t i=0;i<sizeQl;++i)for(uint32_t k=0;k<n;++k){
        uint64_t r0=(*ref)[0].GetAllElements()[i][k].ConvertToInt();
        uint64_t r1=(*ref)[1].GetAllElements()[i][k].ConvertToInt();
        if(R.ba0[(size_t)i*n+k]!=r0){ if(f0<0)f0=(int)i; ++bad0; }
        if(R.ba1[(size_t)i*n+k]!=r1){ if(f1<0)f1=(int)i; ++bad1; }
    }
    std::cout<<"sizeQl="<<sizeQl<<" sizeP="<<sizeP<<" numPart="<<numPart<<" evalKeyTowers="<<K.evalKeyTowers<<"\n";
    std::cout<<"  ba0: "<<(bad0==0?"ok":"BAD"); if(bad0)std::cout<<" "<<bad0<<" first tower "<<f0; std::cout<<"\n";
    std::cout<<"  ba1: "<<(bad1==0?"ok":"BAD"); if(bad1)std::cout<<" "<<bad1<<" first tower "<<f1; std::cout<<"\n";
    if(bad0==0&&bad1==0){ std::cout<<"[PASS] full resident KeySwitchCore bit-exact vs OpenFHE\n"; return 0; }
    std::cout<<"[FAIL]\n"; return 1;
}
