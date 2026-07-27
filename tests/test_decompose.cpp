// Stage 2: digit decompose (EvalKeySwitchPrecomputeCore) host-orchestrated.
// Per part: INTT the part's tower slice -> ApproxSwitchCRTBasis (proven kernel)
// -> NTT back -> reassemble into the sizeQlP layout via three loops. Validates
// the composition + the fiddly per-part reassembly indexing bit-exact vs
// OpenFHE. No new arithmetic -- all proven sub-ops.

#include "openfhe.h"
#include "math/hal/intnat/transformnat.h"
#include "ntt.h"
#include "intt.h"
#include "basis_convert.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace lbcrypto;

// Build fwd/inv root + precon tables + n_inv for modulus q (host, one-off).
struct Tab { std::vector<uint64_t> fr,fp,ir,ip; uint64_t ninv,ninv_p; };
static Tab mk_tab(uint32_t n, NativeInteger q) {
    using NI=NativeInteger;
    NI root=RootOfUnity<NI>(2*n,q), rootInv=root.ModInverse(q);
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
    params.SetMultiplicativeDepth(3);
    params.SetScalingModSize(50);
    params.SetRingDim(32768);
    params.SetBatchSize(16384);
    params.SetKeySwitchTechnique(HYBRID);
    auto cc = GenCryptoContext(params);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);

    auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto ep = cp->GetElementParams();
    const uint32_t n = ep->GetRingDimension();

    // Input c = a random DCRTPoly over the full current Q basis (eval form),
    // standing in for cv.back(). The decompose is a linear op on it.
    const uint32_t sizeQl = (uint32_t)ep->GetParams().size();
    std::mt19937_64 rng(20260729);
    using Poly = DCRTPoly::PolyType;
    std::vector<Poly> tv; tv.reserve(sizeQl);
    for (uint32_t i=0;i<sizeQl;++i){ auto pp=ep->GetParams()[i];
        auto q=pp->GetModulus().ConvertToInt();
        NativeVector v(n,pp->GetModulus());
        for(uint32_t k=0;k<n;++k) v[k]=NativeInteger(rng()%q);
        Poly p(pp,Format::EVALUATION,true); p.SetValues(std::move(v),Format::EVALUATION);
        tv.push_back(std::move(p)); }
    DCRTPoly c(tv);

    // OpenFHE reference.
    auto ref = cc->GetScheme()->EvalKeySwitchPrecomputeCore(c, cp);

    const uint32_t alpha  = cp->GetNumPerPartQ();
    uint32_t numPartQl = (uint32_t)std::ceil((double)sizeQl/alpha);
    if (numPartQl > cp->GetNumberOfQPartitions()) numPartQl = cp->GetNumberOfQPartitions();
    auto paramsP = cp->GetParamsP();
    const uint32_t sizeP = (uint32_t)paramsP->GetParams().size();
    const uint32_t sizeQlP = sizeQl + sizeP;

    std::cout << "sizeQl="<<sizeQl<<" alpha="<<alpha<<" numPartQl="<<numPartQl
              << " sizeP="<<sizeP<<" sizeQlP="<<sizeQlP<<"\n";

    const size_t B=(size_t)n*8;
    uint32_t grand_bad=0;

    for (uint32_t part=0; part<numPartQl; ++part) {
        const uint32_t startPartIdx = alpha*part;
        const uint32_t sizePartQl = (sizeQl > startPartIdx+alpha) ? alpha : (sizeQl-startPartIdx);
        const uint32_t endPartIdx = startPartIdx + sizePartQl;

        auto paramsPartQ = cp->GetParamsPartQ(part);
        auto paramsCompl = cp->GetParamsComplPartQ(sizeQl-1, part);
        const uint32_t sizeCompl = (uint32_t)paramsCompl->GetParams().size();

        // 1. Part's tower slice -> coefficient form via INTT (per tower).
        std::vector<uint64_t> partCoeff((size_t)sizePartQl*n);
        for (uint32_t i=0;i<sizePartQl;++i){
            uint32_t gt = startPartIdx+i;                 // global tower index
            NativeInteger q = ep->GetParams()[gt]->GetModulus();
            Tab T = mk_tab(n,q);
            std::vector<uint64_t> h(n);
            for(uint32_t k=0;k<n;++k) h[k]=c.GetAllElements()[gt][k].ConvertToInt();
            uint64_t *dx=up(h), *dir=up(T.ir), *dip=up(T.ip);
            LaunchINTT_GS(dx,dir,dip,n,q.ConvertToInt(),T.ninv,T.ninv_p,0);
            cudaDeviceSynchronize();
            cudaMemcpy(partCoeff.data()+(size_t)i*n,dx,B,cudaMemcpyDeviceToHost);
            cudaFree(dx);cudaFree(dir);cudaFree(dip);
        }

        // 2. ApproxSwitchCRTBasis: partQ (sizePartQl) -> compl (sizeCompl).
        const auto& QHatInv  = cp->GetPartQlHatInvModq(part, sizePartQl-1);
        const auto& QHatInvP = cp->GetPartQlHatInvModqPrecon(part, sizePartQl-1);
        const auto& QHatModp = cp->GetPartQlHatModp(sizeQl-1, part);           // [sizePartQl][sizeCompl]
        const auto& BMu      = cp->GetmodComplPartqBarrettMu(sizeQl-1, part);

        std::vector<uint64_t> hqhi(sizePartQl),hqhip(sizePartQl),hq(sizePartQl);
        for(uint32_t i=0;i<sizePartQl;++i){ hqhi[i]=QHatInv[i].ConvertToInt();
            hqhip[i]=QHatInvP[i].ConvertToInt();
            hq[i]=paramsPartQ->GetParams()[i]->GetModulus().ConvertToInt(); }
        std::vector<uint64_t> hqmp((size_t)sizePartQl*sizeCompl);
        for(uint32_t i=0;i<sizePartQl;++i)for(uint32_t j=0;j<sizeCompl;++j)
            hqmp[(size_t)i*sizeCompl+j]=QHatModp[i][j].ConvertToInt();
        std::vector<uint64_t> hp(sizeCompl),hmlo(sizeCompl),hmhi(sizeCompl);
        for(uint32_t j=0;j<sizeCompl;++j){ hp[j]=paramsCompl->GetParams()[j]->GetModulus().ConvertToInt();
            unsigned __int128 mu=(unsigned __int128)BMu[j]; hmlo[j]=(uint64_t)mu; hmhi[j]=(uint64_t)(mu>>64); }

        uint64_t *dsrc=up(partCoeff),*dqhi=up(hqhi),*dqhip=up(hqhip),*dq=up(hq),
                 *dqmp=up(hqmp),*dp=up(hp),*dmlo=up(hmlo),*dmhi=up(hmhi);
        uint64_t* ddst; cudaMalloc(&ddst,(size_t)sizeCompl*n*8);
        LaunchApproxSwitchCRTBasis(dsrc,ddst,dqhi,dqhip,dq,dqmp,dp,dmlo,dmhi,
                                   sizePartQl,sizeCompl,n,0);
        cudaDeviceSynchronize();
        std::vector<uint64_t> complCoeff((size_t)sizeCompl*n);
        cudaMemcpy(complCoeff.data(),ddst,(size_t)sizeCompl*n*8,cudaMemcpyDeviceToHost);
        cudaFree(dsrc);cudaFree(dqhi);cudaFree(dqhip);cudaFree(dq);cudaFree(dqmp);
        cudaFree(dp);cudaFree(dmlo);cudaFree(dmhi);cudaFree(ddst);

        // 3. NTT the complement back to eval, per complement tower.
        // Complement tower j maps to a global QlP modulus; its modulus comes
        // from paramsCompl. NTT each with that modulus's tables.
        std::vector<uint64_t> complEval((size_t)sizeCompl*n);
        for(uint32_t j=0;j<sizeCompl;++j){
            NativeInteger q=paramsCompl->GetParams()[j]->GetModulus();
            Tab T=mk_tab(n,q);
            std::vector<uint64_t> h(complCoeff.begin()+(size_t)j*n, complCoeff.begin()+(size_t)(j+1)*n);
            uint64_t *dx=up(h),*dr=up(T.fr),*dpp=up(T.fp);
            LaunchNTT_CT(dx,dr,dpp,n,q.ConvertToInt(),0);
            cudaDeviceSynchronize();
            cudaMemcpy(complEval.data()+(size_t)j*n,dx,B,cudaMemcpyDeviceToHost);
            cudaFree(dx);cudaFree(dr);cudaFree(dpp);
        }

        // 4. Reassemble sizeQlP result and compare against ref[part] per tower.
        // [0,startPartIdx): complEval[i]; [startPartIdx,endPartIdx): c[i] (eval);
        // [endPartIdx,sizeQlP): complEval[i - sizePartQl].
        uint32_t bad=0; int firstT=-1;
        for (uint32_t i=0;i<sizeQlP;++i){
            std::vector<uint64_t> mine(n);
            if (i<startPartIdx)            for(uint32_t k=0;k<n;++k) mine[k]=complEval[(size_t)i*n+k];
            else if (i<endPartIdx)         for(uint32_t k=0;k<n;++k) mine[k]=c.GetAllElements()[i][k].ConvertToInt();
            else                           for(uint32_t k=0;k<n;++k) mine[k]=complEval[(size_t)(i-sizePartQl)*n+k];
            uint32_t m=0;
            for(uint32_t k=0;k<n;++k) if(mine[k]!=(*ref)[part].GetAllElements()[i][k].ConvertToInt()){++m;}
            if(m && firstT<0) firstT=(int)i;
            bad+=m;
        }
        std::cout << "  part "<<part<<" (sizePartQl="<<sizePartQl<<" sizeCompl="<<sizeCompl
                  <<"): "<<(bad==0?"ok":"BAD");
        if(bad) std::cout<<" "<<bad<<" mism, first bad tower "<<firstT;
        std::cout<<"\n";
        grand_bad+=bad;
    }

    if(grand_bad==0){ std::cout<<"[PASS] resident digit decompose bit-exact vs OpenFHE\n"; return 0; }
    std::cout<<"[FAIL]\n"; return 1;
}
