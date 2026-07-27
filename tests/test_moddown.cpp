// Stage 3b: ApproxModDown QP->Q (CKKS, t=0), the last new keyswitch arithmetic.
// Take the P towers (high indices) of a QlP poly, INTT to coeff, ApproxSwitchCRTBasis
// P->Q (the proven kernel, ModDown constants), NTT back, then per Q tower:
// ans = (Qpart - Pswitched)*PInvModq mod q. Bit-exact vs OpenFHE's ApproxModDown.

#include "openfhe.h"
#include "math/hal/intnat/transformnat.h"
#include "ntt.h"
#include "intt.h"
#include "basis_convert.h"
#include "moddown.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace lbcrypto;

struct Tab { std::vector<uint64_t> fr,fp,ir,ip; uint64_t ninv,ninv_p; };
static Tab mk_tab(uint32_t n, NativeInteger q){
    using NI=NativeInteger; NI root=RootOfUnity<NI>(2*n,q), rootInv=root.ModInverse(q);
    auto br=[](uint32_t v,uint32_t b){uint32_t r=0;for(uint32_t k=0;k<b;k++){r=(r<<1)|(v&1);v>>=1;}return r;};
    uint32_t logn=0; while((1u<<logn)<n)++logn;
    Tab T; T.fr.resize(n);T.fp.resize(n);T.ir.resize(n);T.ip.resize(n);
    auto fill=[&](NI base,std::vector<uint64_t>&r,std::vector<uint64_t>&p){
        std::vector<NI> pw(n); pw[0]=NI(1);
        for(uint32_t i=1;i<n;++i)pw[i]=pw[i-1].ModMul(base,q);
        for(uint32_t i=0;i<n;++i){NI ri=pw[br(i,logn)];
            r[i]=ri.ConvertToInt();
            p[i]=(uint64_t)(((__uint128_t)ri.ConvertToInt()<<64)/(__uint128_t)q.ConvertToInt());}};
    fill(root,T.fr,T.fp); fill(rootInv,T.ir,T.ip);
    NI ni=NI(n).ModInverse(q); T.ninv=ni.ConvertToInt();
    T.ninv_p=(uint64_t)(((__uint128_t)T.ninv<<64)/(__uint128_t)q.ConvertToInt());
    return T;
}
static uint64_t* up(const std::vector<uint64_t>& h){ uint64_t* d; size_t B=h.size()*8;
    cudaMalloc(&d,B); cudaMemcpy(d,h.data(),B,cudaMemcpyHostToDevice); return d; }

int main() {
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(3); params.SetScalingModSize(50);
    params.SetRingDim(32768); params.SetBatchSize(16384);
    params.SetKeySwitchTechnique(HYBRID);
    auto cc = GenCryptoContext(params);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);

    auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto paramsQ = cp->GetElementParams();
    auto paramsP = cp->GetParamsP();
    const uint32_t n = paramsQ->GetRingDimension();
    const uint32_t sizeQ = (uint32_t)paramsQ->GetParams().size();
    const uint32_t sizeP = (uint32_t)paramsP->GetParams().size();

    // Build a random QlP poly (sizeQ + sizeP towers, eval form) as ApproxModDown input.
    auto paramsQlP = paramsQ->GetParams(); // vector of Q param ptrs
    std::mt19937_64 rng(20260730);
    using Poly = DCRTPoly::PolyType;
    std::vector<Poly> tv; tv.reserve(sizeQ+sizeP);
    std::vector<NativeInteger> allMod;
    for (uint32_t i=0;i<sizeQ;++i){ auto pp=paramsQ->GetParams()[i];
        allMod.push_back(pp->GetModulus());
        NativeVector v(n,pp->GetModulus());
        for(uint32_t k=0;k<n;++k) v[k]=NativeInteger(rng()%pp->GetModulus().ConvertToInt());
        Poly p(pp,Format::EVALUATION,true); p.SetValues(std::move(v),Format::EVALUATION); tv.push_back(std::move(p)); }
    for (uint32_t j=0;j<sizeP;++j){ auto pp=paramsP->GetParams()[j];
        allMod.push_back(pp->GetModulus());
        NativeVector v(n,pp->GetModulus());
        for(uint32_t k=0;k<n;++k) v[k]=NativeInteger(rng()%pp->GetModulus().ConvertToInt());
        Poly p(pp,Format::EVALUATION,true); p.SetValues(std::move(v),Format::EVALUATION); tv.push_back(std::move(p)); }
    // Build the QlP param set OpenFHE expects (extended basis).
    // Construct DCRTPoly over the extended QlP basis by using the Q poly's ExtendedCRTBasis.
    std::vector<Poly> qtowers(tv.begin(), tv.begin()+sizeQ);
    DCRTPoly Aq(qtowers);
    auto paramsExt = Aq.GetExtendedCRTBasis(paramsP);
    DCRTPoly A(paramsExt, Format::EVALUATION, true);
    for (uint32_t i=0;i<sizeQ+sizeP;++i) A.SetElementAtIndex(i, tv[i]);

    // Reference: OpenFHE ApproxModDown.
    NativeInteger tzero(0);
    DCRTPoly ref = A.ApproxModDown(paramsQ, paramsP,
        cp->GetPInvModq(), cp->GetPInvModqPrecon(),
        cp->GetPHatInvModp(), cp->GetPHatInvModpPrecon(),
        cp->GetPHatModq(), cp->GetModqBarrettMu(),
        {}, {}, tzero, {});

    // Resident: INTT the P towers, ApproxSwitchCRTBasis P->Q, NTT back, combine.
    const size_t B=(size_t)n*8;
    // 1. P towers to coeff.
    std::vector<uint64_t> pCoeff((size_t)sizeP*n);
    for (uint32_t j=0;j<sizeP;++j){
        NativeInteger q=paramsP->GetParams()[j]->GetModulus(); Tab T=mk_tab(n,q);
        std::vector<uint64_t> h(n);
        for(uint32_t k=0;k<n;++k) h[k]=A.GetAllElements()[sizeQ+j][k].ConvertToInt();
        uint64_t *dx=up(h),*dir=up(T.ir),*dip=up(T.ip);
        LaunchINTT_GS(dx,dir,dip,n,q.ConvertToInt(),T.ninv,T.ninv_p,0); cudaDeviceSynchronize();
        cudaMemcpy(pCoeff.data()+(size_t)j*n,dx,B,cudaMemcpyDeviceToHost);
        cudaFree(dx);cudaFree(dir);cudaFree(dip);
    }
    // 2. ApproxSwitchCRTBasis P (sizeP) -> Q (sizeQ), ModDown constants.
    const auto& PHatInvModp = cp->GetPHatInvModp();
    const auto& PHatInvModpPrecon = cp->GetPHatInvModpPrecon();
    const auto& PHatModq = cp->GetPHatModq();          // [sizeP][sizeQ]
    const auto& modqBMu = cp->GetModqBarrettMu();
    std::vector<uint64_t> hqhi(sizeP),hqhip(sizeP),hpq(sizeP);
    for(uint32_t i=0;i<sizeP;++i){ hqhi[i]=PHatInvModp[i].ConvertToInt();
        hqhip[i]=PHatInvModpPrecon[i].ConvertToInt();
        hpq[i]=paramsP->GetParams()[i]->GetModulus().ConvertToInt(); }
    std::vector<uint64_t> hqmp((size_t)sizeP*sizeQ);
    for(uint32_t i=0;i<sizeP;++i)for(uint32_t j=0;j<sizeQ;++j)
        hqmp[(size_t)i*sizeQ+j]=PHatModq[i][j].ConvertToInt();
    std::vector<uint64_t> hp(sizeQ),hmlo(sizeQ),hmhi(sizeQ);
    for(uint32_t j=0;j<sizeQ;++j){ hp[j]=paramsQ->GetParams()[j]->GetModulus().ConvertToInt();
        unsigned __int128 mu=(unsigned __int128)modqBMu[j]; hmlo[j]=(uint64_t)mu; hmhi[j]=(uint64_t)(mu>>64); }
    uint64_t *dsrc=up(pCoeff),*dqhi=up(hqhi),*dqhip=up(hqhip),*dq=up(hpq),
             *dqmp=up(hqmp),*dp=up(hp),*dmlo=up(hmlo),*dmhi=up(hmhi);
    uint64_t* ddst; cudaMalloc(&ddst,(size_t)sizeQ*n*8);
    LaunchApproxSwitchCRTBasis(dsrc,ddst,dqhi,dqhip,dq,dqmp,dp,dmlo,dmhi,sizeP,sizeQ,n,0);
    cudaDeviceSynchronize();
    std::vector<uint64_t> qSwitchedCoeff((size_t)sizeQ*n);
    cudaMemcpy(qSwitchedCoeff.data(),ddst,(size_t)sizeQ*n*8,cudaMemcpyDeviceToHost);
    cudaFree(dsrc);cudaFree(dqhi);cudaFree(dqhip);cudaFree(dq);cudaFree(dqmp);cudaFree(dp);cudaFree(dmlo);cudaFree(dmhi);cudaFree(ddst);

    // 3. NTT each switched Q tower back to eval, then 4. combine.
    const auto& PInvModq = cp->GetPInvModq();
    std::vector<uint64_t> got((size_t)sizeQ*n);
    for (uint32_t i=0;i<sizeQ;++i){
        NativeInteger q=paramsQ->GetParams()[i]->GetModulus(); Tab T=mk_tab(n,q);
        std::vector<uint64_t> h(qSwitchedCoeff.begin()+(size_t)i*n, qSwitchedCoeff.begin()+(size_t)(i+1)*n);
        uint64_t *dx=up(h),*dr=up(T.fr),*dpp=up(T.fp);
        LaunchNTT_CT(dx,dr,dpp,n,q.ConvertToInt(),0); cudaDeviceSynchronize();
        // combine: (A_q[i] - switched)*PInvModq[i]
        std::vector<uint64_t> ha(n);
        for(uint32_t k=0;k<n;++k) ha[k]=A.GetAllElements()[i][k].ConvertToInt();
        uint64_t *da=up(ha), *dans; cudaMalloc(&dans,B);
        LaunchModDownCombine(dans,da,dx,PInvModq[i].ConvertToInt(),q.ConvertToInt(),n,0);
        cudaDeviceSynchronize();
        cudaMemcpy(got.data()+(size_t)i*n,dans,B,cudaMemcpyDeviceToHost);
        cudaFree(dx);cudaFree(dr);cudaFree(dpp);cudaFree(da);cudaFree(dans);
    }

    uint32_t bad=0; int ft=-1;
    for (uint32_t i=0;i<sizeQ;++i)for(uint32_t k=0;k<n;++k)
        if(got[(size_t)i*n+k]!=ref.GetAllElements()[i][k].ConvertToInt()){ if(ft<0)ft=(int)i; ++bad; }
    std::cout << "sizeQ="<<sizeQ<<" sizeP="<<sizeP<<"\n";
    std::cout << "  ApproxModDown: "<<(bad==0?"ok":"BAD");
    if(bad) std::cout<<" "<<bad<<" first tower "<<ft; std::cout<<"\n";
    if(bad==0){ std::cout<<"[PASS] resident ApproxModDown bit-exact vs OpenFHE\n"; return 0; }
    std::cout<<"[FAIL]\n"; return 1;
}
