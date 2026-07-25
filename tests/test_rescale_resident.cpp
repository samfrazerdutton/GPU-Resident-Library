// Rescale composition test. Wires the validated sub-ops (INTT, SwitchModulus,
// forward NTT, fused correction) and compares against OpenFHE's OWN
// DropLastElementAndScale -- the exact primitive the GPU path replicates,
// called directly on copied elements with the same constants. This sidesteps
// the scale-gated Rescale/ModReduce wrappers (which drop zero towers on a
// ciphertext they don't consider ready) and makes bit-exact the right bar:
// same operation, same input, same constants, on both paths.

#include "openfhe.h"
#include "math/hal/intnat/transformnat.h"
#include "ntt.h"
#include "intt.h"
#include "switch_modulus.h"
#include "rescale.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace lbcrypto;

struct Tables { std::vector<uint64_t> fr, fp, ir, ip; uint64_t ninv, ninv_p; };
static Tables build_tables(uint32_t n, NativeInteger q) {
    using NI = NativeInteger;
    NI root = RootOfUnity<NI>(2*n, q);
    NI rootInv = root.ModInverse(q);
    auto bitrev = [](uint32_t v, uint32_t b){ uint32_t r=0;
        for(uint32_t k=0;k<b;k++){ r=(r<<1)|(v&1); v>>=1; } return r; };
    uint32_t logn=0; while((1u<<logn)<n) ++logn;
    Tables T; T.fr.resize(n); T.fp.resize(n); T.ir.resize(n); T.ip.resize(n);
    auto fill=[&](NI base,std::vector<uint64_t>&r,std::vector<uint64_t>&p){
        std::vector<NI> pw(n); pw[0]=NI(1);
        for(uint32_t i=1;i<n;++i) pw[i]=pw[i-1].ModMul(base,q);
        for(uint32_t i=0;i<n;++i){ NI ri=pw[bitrev(i,logn)];
            r[i]=ri.ConvertToInt();
            p[i]=(uint64_t)(((__uint128_t)ri.ConvertToInt()<<64)/(__uint128_t)q.ConvertToInt()); }
    };
    fill(root,T.fr,T.fp); fill(rootInv,T.ir,T.ip);
    NI ninv=NI(n).ModInverse(q);
    T.ninv=ninv.ConvertToInt();
    T.ninv_p=(uint64_t)(((__uint128_t)T.ninv<<64)/(__uint128_t)q.ConvertToInt());
    return T;
}

// GPU rescale of one component. Returns surviving-tower-major host buffer.
static std::vector<uint64_t> gpu_rescale_component(
    const DCRTPoly& comp, uint32_t n, uint32_t towers,
    const std::vector<uint64_t>& s1, const std::vector<uint64_t>& s2)
{
    const uint32_t surviving = towers - 1;
    const size_t B = (size_t)n * sizeof(uint64_t);

    NativeInteger qLast = comp.GetAllElements()[towers-1].GetModulus();
    std::vector<uint64_t> dropped(n);
    for (uint32_t i=0;i<n;++i) dropped[i]=comp.GetAllElements()[towers-1][i].ConvertToInt();

    Tables TL = build_tables(n, qLast);
    uint64_t *d_drop=nullptr,*d_ir=nullptr,*d_ip=nullptr;
    cudaMalloc(&d_drop,B); cudaMalloc(&d_ir,B); cudaMalloc(&d_ip,B);
    cudaMemcpy(d_drop,dropped.data(),B,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ir,TL.ir.data(),B,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ip,TL.ip.data(),B,cudaMemcpyHostToDevice);
    LaunchINTT_GS(d_drop,d_ir,d_ip,n,qLast.ConvertToInt(),TL.ninv,TL.ninv_p,0);
    cudaDeviceSynchronize();
    std::vector<uint64_t> droppedCoeff(n);
    cudaMemcpy(droppedCoeff.data(),d_drop,B,cudaMemcpyDeviceToHost);
    cudaFree(d_drop); cudaFree(d_ir); cudaFree(d_ip);

    std::vector<uint64_t> out(n * surviving);
    for (uint32_t t=0; t<surviving; ++t) {
        NativeInteger qi = comp.GetAllElements()[t].GetModulus();
        uint64_t Qi = qi.ConvertToInt();
        std::vector<uint64_t> bcoeff = droppedCoeff;
        uint64_t *d_b=nullptr,*d_fr=nullptr,*d_fp=nullptr,*d_a=nullptr;
        cudaMalloc(&d_b,B); cudaMalloc(&d_fr,B); cudaMalloc(&d_fp,B); cudaMalloc(&d_a,B);
        cudaMemcpy(d_b,bcoeff.data(),B,cudaMemcpyHostToDevice);
        LaunchSwitchModulus(d_b,qLast.ConvertToInt(),Qi,n,0);
        Tables Ti = build_tables(n, qi);
        cudaMemcpy(d_fr,Ti.fr.data(),B,cudaMemcpyHostToDevice);
        cudaMemcpy(d_fp,Ti.fp.data(),B,cudaMemcpyHostToDevice);
        LaunchNTT_CT(d_b,d_fr,d_fp,n,Qi,0);
        std::vector<uint64_t> acoeff(n);
        for (uint32_t i=0;i<n;++i) acoeff[i]=comp.GetAllElements()[t][i].ConvertToInt();
        cudaMemcpy(d_a,acoeff.data(),B,cudaMemcpyHostToDevice);
        LaunchRescaleFuse(d_a,d_b,s1[t],s2[t],Qi,n,0);
        cudaDeviceSynchronize();
        cudaMemcpy(out.data()+t*n,d_a,B,cudaMemcpyDeviceToHost);
        cudaFree(d_b); cudaFree(d_fr); cudaFree(d_fp); cudaFree(d_a);
    }
    return out;
}

int main() {
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(3);
    params.SetScalingModSize(50);
    params.SetRingDim(32768);
    params.SetBatchSize(16384);
    auto cc = GenCryptoContext(params);
    cc->Enable(PKE); cc->Enable(LEVELEDSHE);
    auto kp = cc->KeyGen();

    auto ep = cc->GetCryptoParameters()->GetElementParams();
    const uint32_t n = ep->GetRingDimension();

    std::vector<double> vals(16384);
    for (size_t i=0;i<vals.size();++i) vals[i]=0.1*((i%7)+1);
    auto ct = cc->Encrypt(kp.publicKey, cc->MakeCKKSPackedPlaintext(vals));
    const uint32_t towers = ct->GetElements()[0].GetNumOfElements();
    std::cout << "towers=" << towers << "\n";

    auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(
        cc->GetCryptoParameters());
    const auto& qlInvV = cryptoParams->GetqlInvModq(0);
    const auto& QlV    = cryptoParams->GetQlQlInvModqlDivqlModq(0);
    std::vector<uint64_t> s1(towers-1), s2(towers-1);
    for (uint32_t t=0;t<towers-1;++t){ s1[t]=qlInvV[t].ConvertToInt(); s2[t]=QlV[t].ConvertToInt(); }

    const uint32_t ncomp = ct->GetElements().size();
    const uint32_t surviving = towers-1;

    auto cmp=[&](const std::vector<uint64_t>& g, const DCRTPoly& r)->uint32_t{
        uint32_t mism=0;
        for (uint32_t t=0;t<surviving;++t)
            for (uint32_t i=0;i<n;++i)
                if (g[t*n+i]!=r.GetAllElements()[t][i].ConvertToInt()) ++mism;
        return mism;
    };

    uint32_t total=0;
    for (uint32_t c=0;c<ncomp;++c) {
        // GPU path.
        auto g = gpu_rescale_component(ct->GetElements()[c], n, towers, s1, s2);
        // Reference: OpenFHE's OWN DropLastElementAndScale on a copy, same constants.
        DCRTPoly ref = ct->GetElements()[c];
        ref.DropLastElementAndScale(QlV, qlInvV);
        uint32_t m = cmp(g, ref);
        if (c==0) {
            std::cout << "comp0 tower0 gpu vs ref:\n";
            for (uint32_t i=0;i<4;++i)
                std::cout << "  [" << i << "] gpu=" << g[i]
                          << " ref=" << ref.GetAllElements()[0][i].ConvertToInt() << "\n";
        }
        std::cout << "comp" << c << " mismatches: " << m << "\n";
        total += m;
    }

    if (total==0) {
        std::cout << "[PASS] resident rescale bit-exact vs OpenFHE DropLastElementAndScale\n";
        return 0;
    }
    std::cout << "[FAIL]\n";
    return 1;
}
