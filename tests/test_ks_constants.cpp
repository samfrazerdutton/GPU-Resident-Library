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

    std::cout<<"sizeQ="<<sizeQ<<" sizeP="<<sizeP<<" numPart="<<numPart<<"\n";
    if(bad==0){ std::cout<<"[PASS] native ModDown constants bit-exact vs OpenFHE\n"; return 0; }
    std::cout<<"[FAIL] "<<bad<<" mismatches\n"; return 1;
}
