// Stage 3a: fast-keyswitch multiply-accumulate (EvalFastKeySwitchCoreExt).
// For each digit j, each QP tower i: out0[i] += digit[j][i]*bv[j][idx],
// out1[i] += digit[j][i]*av[j][idx], idx=(i>=sizeQl)?i+delta:i. Digits from
// OpenFHE's precompute (isolates 3a from stage 2); eval key av/bv borrowed.
// Bit-exact vs OpenFHE's EvalFastKeySwitchCoreExt. No ModDown yet (stage 4).

#include "openfhe.h"
#include "ks_fastcore.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace lbcrypto;

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
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE); cc->Enable(ADVANCEDSHE);
    auto kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);

    auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto ep = cp->GetElementParams();
    const uint32_t n = ep->GetRingDimension();
    const uint32_t sizeQl = (uint32_t)ep->GetParams().size();

    // Input c (stands in for c2), decomposed via OpenFHE's precompute.
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

    auto digits = cc->GetScheme()->EvalKeySwitchPrecomputeCore(c, cp);
    const uint32_t limit = (uint32_t)digits->size();
    auto paramsQlP = (*digits)[0].GetParams();
    const uint32_t sizeQlP = (uint32_t)paramsQlP->GetParams().size();
    const uint32_t delta = sizeQl - sizeQl; // full Q == current Ql here (fresh ct)
    // NOTE: delta = fullQ - sizeQl; here the ct is at full level so delta such
    // that (i>=sizeQl?i+delta:i). Compute from crypto params' full Q size.
    const uint32_t fullQ = (uint32_t)cp->GetElementParams()->GetParams().size();
    const uint32_t realDelta = fullQ - sizeQl;

    // Borrow the relin eval key.
    auto evalKeyVec = CryptoContextImpl<DCRTPoly>::GetEvalMultKeyVector(kp.secretKey->GetKeyTag());
    auto evalKey = evalKeyVec[0];
    const auto& av = evalKey->GetAVector();
    const auto& bv = evalKey->GetBVector();

    // OpenFHE reference.
    auto ref = cc->GetScheme()->EvalFastKeySwitchCoreExt(digits, evalKey, paramsQlP);

    // Resident accumulate. out0/out1 zero-init per tower, accumulate over digits.
    const size_t B=(size_t)n*8;
    std::vector<uint64_t> got0((size_t)sizeQlP*n,0), got1((size_t)sizeQlP*n,0);
    std::vector<uint64_t> zero(n,0);

    for (uint32_t i=0;i<sizeQlP;++i){
        uint64_t q = paramsQlP->GetParams()[i]->GetModulus().ConvertToInt();
        uint32_t idx = (i>=sizeQl)? i+realDelta : i;
        uint64_t *d_o0=up(zero), *d_o1=up(zero);
        for (uint32_t j=0;j<limit;++j){
            std::vector<uint64_t> hd(n),hb(n),ha(n);
            for(uint32_t k=0;k<n;++k){
                hd[k]=(*digits)[j].GetAllElements()[i][k].ConvertToInt();
                hb[k]=bv[j].GetAllElements()[idx][k].ConvertToInt();
                ha[k]=av[j].GetAllElements()[idx][k].ConvertToInt();
            }
            uint64_t *d_d=up(hd), *d_b=up(hb), *d_a=up(ha);
            LaunchKSMultAcc(d_o0,d_o1,d_d,d_b,d_a,q,n,0);
            cudaDeviceSynchronize();
            cudaFree(d_d);cudaFree(d_b);cudaFree(d_a);
        }
        cudaMemcpy(got0.data()+(size_t)i*n,d_o0,B,cudaMemcpyDeviceToHost);
        cudaMemcpy(got1.data()+(size_t)i*n,d_o1,B,cudaMemcpyDeviceToHost);
        cudaFree(d_o0);cudaFree(d_o1);
    }

    uint32_t bad0=0,bad1=0; int f0=-1,f1=-1;
    for (uint32_t i=0;i<sizeQlP;++i)
        for (uint32_t k=0;k<n;++k){
            if(got0[(size_t)i*n+k]!=(*ref)[0].GetAllElements()[i][k].ConvertToInt()){ if(f0<0)f0=(int)i; ++bad0; }
            if(got1[(size_t)i*n+k]!=(*ref)[1].GetAllElements()[i][k].ConvertToInt()){ if(f1<0)f1=(int)i; ++bad1; }
        }
    std::cout << "sizeQl="<<sizeQl<<" sizeQlP="<<sizeQlP<<" limit="<<limit
              <<" delta="<<realDelta<<"\n";
    std::cout << "  result[0] (bv): "<<(bad0==0?"ok":"BAD");
    if(bad0) std::cout<<" "<<bad0<<" first tower "<<f0; std::cout<<"\n";
    std::cout << "  result[1] (av): "<<(bad1==0?"ok":"BAD");
    if(bad1) std::cout<<" "<<bad1<<" first tower "<<f1; std::cout<<"\n";

    if(bad0==0 && bad1==0){ std::cout<<"[PASS] resident fast-keyswitch inner product bit-exact\n"; return 0; }
    std::cout<<"[FAIL]\n"; return 1;
}
