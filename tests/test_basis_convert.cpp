// Feasibility gate for Hybrid keyswitch: resident ApproxSwitchCRTBasis must be
// bit-exact vs OpenFHE's own, using OpenFHE's borrowed constants (rescale
// playbook). If this passes, resident keyswitch is viable and the Barrett-128
// port is exact; if not, the mismatch pattern says whether it's the Barrett or
// the constant plumbing. Highest bug-surface primitive in the project.

#include "openfhe.h"
#include "basis_convert.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace lbcrypto;

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
    const uint32_t sizeQ_full = (uint32_t)ep->GetParams().size();

    // Use part 0. Source basis = paramsPartQ(0); target = complementary basis.
    const uint32_t part = 0;
    auto paramsPartQ = cp->GetParamsPartQ(part);
    const uint32_t sizeQ = (uint32_t)paramsPartQ->GetParams().size();
    // complementary basis at full level (numTowers index = sizeQ_full - 1)
    auto paramsCompl = cp->GetParamsComplPartQ(sizeQ_full - 1, part);
    const uint32_t sizeP = (uint32_t)paramsCompl->GetParams().size();
    std::cout << "part=" << part << " sizeQ=" << sizeQ << " sizeP=" << sizeP
              << " n=" << n << "\n";
    if (sizeP > 8) { std::cout << "sizeP>8, kernel local cap exceeded\n"; return 1; }

    // Build a random source DCRTPoly in the partQ basis, COEFFICIENT form
    // (ApproxSwitchCRTBasis is called on coefficient-form input in the decompose).
    using Poly = DCRTPoly::PolyType;
    std::mt19937_64 rng(20260729);
    std::vector<Poly> tv; tv.reserve(sizeQ);
    for (uint32_t i=0;i<sizeQ;++i){
        auto pp = paramsPartQ->GetParams()[i];
        auto q = pp->GetModulus().ConvertToInt();
        NativeVector v(n, pp->GetModulus());
        for (uint32_t k=0;k<n;++k) v[k]=NativeInteger(rng()%q);
        Poly p(pp, Format::COEFFICIENT, true);
        p.SetValues(std::move(v), Format::COEFFICIENT);
        tv.push_back(std::move(p));
    }
    DCRTPoly A(tv);

    // Borrowed constants for this part/level.
    const auto& QHatInvModq       = cp->GetPartQlHatInvModq(part, sizeQ-1);
    const auto& QHatInvModqPrecon = cp->GetPartQlHatInvModqPrecon(part, sizeQ-1);
    const auto& QHatModp          = cp->GetPartQlHatModp(sizeQ_full-1, part); // [sizeQ][sizeP]
    const auto& modpBarrettMu     = cp->GetmodComplPartqBarrettMu(sizeQ_full-1, part);

    // OpenFHE reference.
    DCRTPoly ref = A.ApproxSwitchCRTBasis(paramsPartQ, paramsCompl,
                                          QHatInvModq, QHatInvModqPrecon,
                                          QHatModp, modpBarrettMu);

    // Flatten inputs + constants for the kernel.
    std::vector<uint64_t> hsrc((size_t)sizeQ*n);
    for (uint32_t i=0;i<sizeQ;++i) for (uint32_t k=0;k<n;++k)
        hsrc[(size_t)i*n+k] = A.GetAllElements()[i][k].ConvertToInt();

    std::vector<uint64_t> hQHatInv(sizeQ), hQHatInvP(sizeQ), hq(sizeQ);
    for (uint32_t i=0;i<sizeQ;++i){
        hQHatInv[i]=QHatInvModq[i].ConvertToInt();
        hQHatInvP[i]=QHatInvModqPrecon[i].ConvertToInt();
        hq[i]=paramsPartQ->GetParams()[i]->GetModulus().ConvertToInt();
    }
    std::vector<uint64_t> hQHatModp((size_t)sizeQ*sizeP);
    for (uint32_t i=0;i<sizeQ;++i) for (uint32_t j=0;j<sizeP;++j)
        hQHatModp[(size_t)i*sizeP+j]=QHatModp[i][j].ConvertToInt();
    std::vector<uint64_t> hp(sizeP), hmu_lo(sizeP), hmu_hi(sizeP);
    for (uint32_t j=0;j<sizeP;++j){
        hp[j]=paramsCompl->GetParams()[j]->GetModulus().ConvertToInt();
        unsigned __int128 mu = (unsigned __int128)modpBarrettMu[j];
        hmu_lo[j]=(uint64_t)mu; hmu_hi[j]=(uint64_t)(mu>>64);
    }

    // Upload, run, download.
    auto up=[&](const std::vector<uint64_t>& h){ uint64_t* d; size_t B=h.size()*8;
        cudaMalloc(&d,B); cudaMemcpy(d,h.data(),B,cudaMemcpyHostToDevice); return d; };
    uint64_t *d_src=up(hsrc), *d_qhi=up(hQHatInv), *d_qhip=up(hQHatInvP), *d_q=up(hq),
             *d_qmp=up(hQHatModp), *d_p=up(hp), *d_mlo=up(hmu_lo), *d_mhi=up(hmu_hi);
    uint64_t* d_dst; cudaMalloc(&d_dst,(size_t)sizeP*n*8);

    LaunchApproxSwitchCRTBasis(d_src,d_dst,d_qhi,d_qhip,d_q,d_qmp,d_p,d_mlo,d_mhi,
                               sizeQ,sizeP,n,0);
    cudaDeviceSynchronize();
    std::vector<uint64_t> got((size_t)sizeP*n);
    cudaMemcpy(got.data(),d_dst,(size_t)sizeP*n*8,cudaMemcpyDeviceToHost);

    uint32_t total=0;
    for (uint32_t j=0;j<sizeP;++j){
        uint32_t m=0; int first=-1;
        for (uint32_t k=0;k<n;++k)
            if (got[(size_t)j*n+k]!=ref.GetAllElements()[j][k].ConvertToInt()){
                if(first<0)first=(int)k; ++m; }
        std::cout << "  target tower " << j << ": " << (m==0?"ok":"BAD");
        if(m) std::cout << " " << m << " mism first@" << first
                        << " got=" << got[(size_t)j*n+first]
                        << " ref=" << ref.GetAllElements()[j][first].ConvertToInt();
        std::cout << "\n";
        total+=m;
    }

    if(total==0){ std::cout << "[PASS] resident ApproxSwitchCRTBasis bit-exact vs OpenFHE\n"; return 0; }
    std::cout << "[FAIL]\n"; return 1;
}
