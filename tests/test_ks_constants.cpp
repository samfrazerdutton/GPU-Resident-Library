// Stage 6a: native computation of the ModDown constant family, bit-exact vs
// OpenFHE. Borrows moduli only; computes PInvModq, PHatInvModp, PHatModq, and
// the Q Barrett mu by incremental modular folding (no bignum).
#include "openfhe.h"
#include "keyswitch.h"
#include <iostream>
#include <vector>
using namespace lbcrypto;

int main(){
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(3); params.SetScalingModSize(50);
    params.SetRingDim(32768); params.SetBatchSize(16384);
    params.SetKeySwitchTechnique(HYBRID);
    auto cc=GenCryptoContext(params);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    auto cp=std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto ep=cp->GetElementParams(); auto paramsP=cp->GetParamsP();
    uint32_t sizeQ=ep->GetParams().size(), sizeP=paramsP->GetParams().size();
    uint32_t numPart=cp->GetNumPartQ();

    std::vector<uint64_t> mq(sizeQ), mp(sizeP);
    for(uint32_t i=0;i<sizeQ;++i) mq[i]=ep->GetParams()[i]->GetModulus().ConvertToInt();
    for(uint32_t j=0;j<sizeP;++j) mp[j]=paramsP->GetParams()[j]->GetModulus().ConvertToInt();

    gpufhe::KeySwitchConstants K; K.n=ep->GetRingDimension();
    gpufhe::compute_keyswitch_constants(K, mq, mp, numPart);

    uint32_t bad=0;
    // PInvModq
    for(uint32_t i=0;i<sizeQ;++i){ uint64_t r=cp->GetPInvModq()[i].ConvertToInt();
        if(K.pInvModq[i]!=r){ if(bad<5)std::cout<<"  PInvModq["<<i<<"] mine="<<K.pInvModq[i]<<" ref="<<r<<"\n"; ++bad; } }
    // PHatInvModp
    for(uint32_t j=0;j<sizeP;++j){ uint64_t r=cp->GetPHatInvModp()[j].ConvertToInt();
        if(K.pHatInv[j]!=r){ if(bad<5)std::cout<<"  PHatInvModp["<<j<<"] mine="<<K.pHatInv[j]<<" ref="<<r<<"\n"; ++bad; } }
    // PHatModq
    for(uint32_t j=0;j<sizeP;++j)for(uint32_t i=0;i<sizeQ;++i){ uint64_t r=cp->GetPHatModq()[j][i].ConvertToInt();
        if(K.pHatModq[(size_t)j*sizeQ+i]!=r){ if(bad<5)std::cout<<"  PHatModq["<<j<<"]["<<i<<"] mine="<<K.pHatModq[(size_t)j*sizeQ+i]<<" ref="<<r<<"\n"; ++bad; } }
    // Barrett mu
    for(uint32_t i=0;i<sizeQ;++i){ unsigned __int128 r=(unsigned __int128)cp->GetModqBarrettMu()[i];
        unsigned __int128 mine=((unsigned __int128)K.mdBMuHi[i]<<64)|K.mdBMuLo[i];
        if(mine!=r){ if(bad<5)std::cout<<"  ModqBMu["<<i<<"] differs\n"; ++bad; } }

    // ---- decompose families per part vs OpenFHE ----
    for(uint32_t part=0; part<numPart; ++part){
        uint32_t sizePart=K.sizePart[part], sizeCompl=K.sizeCompl[part];
        const auto& qhi=cp->GetPartQlHatInvModq(part,sizePart-1);
        const auto& qhip=cp->GetPartQlHatInvModqPrecon(part,sizePart-1);
        const auto& qmp=cp->GetPartQlHatModp(sizeQ-1,part);
        auto paramsCompl=cp->GetParamsComplPartQ(sizeQ-1,part);
        uint32_t refCompl=paramsCompl->GetParams().size();
        if(refCompl!=sizeCompl){ std::cout<<"  part "<<part<<" sizeCompl mine="<<sizeCompl<<" ref="<<refCompl<<"\n"; ++bad; continue; }
        for(uint32_t i=0;i<sizePart;++i){
            if(K.partQHatInv[part][i]!=qhi[i].ConvertToInt()){ if(bad<8)std::cout<<"  part"<<part<<" QHatInv["<<i<<"] mine="<<K.partQHatInv[part][i]<<" ref="<<qhi[i].ConvertToInt()<<"\n"; ++bad; }
            if(K.partQHatInvPrec[part][i]!=qhip[i].ConvertToInt()){ ++bad; } }
        // complement moduli order
        for(uint32_t j=0;j<sizeCompl;++j){ uint64_t rc=paramsCompl->GetParams()[j]->GetModulus().ConvertToInt();
            if(K.partComplMod[part][j]!=rc){ if(bad<8)std::cout<<"  part"<<part<<" complMod["<<j<<"] mine="<<K.partComplMod[part][j]<<" ref="<<rc<<"\n"; ++bad; } }
        // PartQHatModp [sizePart][sizeCompl]
        for(uint32_t i=0;i<sizePart;++i)for(uint32_t j=0;j<sizeCompl;++j){
            uint64_t r=qmp[i][j].ConvertToInt();
            if(K.partQHatModp[part][(size_t)i*sizeCompl+j]!=r){ if(bad<8)std::cout<<"  part"<<part<<" QHatModp["<<i<<"]["<<j<<"] mine="<<K.partQHatModp[part][(size_t)i*sizeCompl+j]<<" ref="<<r<<"\n"; ++bad; } }
    }

    std::cout<<"sizeQ="<<sizeQ<<" sizeP="<<sizeP<<" numPart="<<numPart<<"\n";
    if(bad==0){ std::cout<<"[PASS] native ModDown constants bit-exact vs OpenFHE\n"; return 0; }
    std::cout<<"[FAIL] "<<bad<<" mismatches\n"; return 1;
}
