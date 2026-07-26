// Barrett multiply must be bit-exact vs plain 128-bit modulo. The Barrett
// estimate can be off by up to 2, corrected by two conditional subtractions;
// this verifies the correction is exact across real CKKS tower moduli,
// including the 60-bit prime where the estimate precision is tightest.

#include "openfhe.h"
#include "rns_arith.h"
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
    auto cc = GenCryptoContext(params);
    auto ep = cc->GetCryptoParameters()->GetElementParams();
    const uint32_t n = ep->GetRingDimension();
    const uint32_t towers = (uint32_t)ep->GetParams().size();

    std::mt19937_64 rng(20260727);
    const size_t B = (size_t)n*sizeof(uint64_t);
    uint64_t *d_a=nullptr,*d_b=nullptr,*d_r=nullptr;
    cudaMalloc(&d_a,B); cudaMalloc(&d_b,B); cudaMalloc(&d_r,B);

    uint32_t total_mism = 0;
    for (uint32_t t=0;t<towers;++t) {
        uint64_t q = ep->GetParams()[t]->GetModulus().ConvertToInt();
        std::vector<uint64_t> a(n), b(n), ref(n), got(n);
        for (uint32_t i=0;i<n;++i){
            a[i]=rng()%q; b[i]=rng()%q;
            ref[i]=(uint64_t)(((unsigned __int128)a[i]*b[i])%q);  // plain-modulo reference
        }
        cudaMemcpy(d_a,a.data(),B,cudaMemcpyHostToDevice);
        cudaMemcpy(d_b,b.data(),B,cudaMemcpyHostToDevice);
        LaunchRNSMultTower(d_a,d_b,d_r,q,n,0);   // Barrett kernel
        cudaDeviceSynchronize();
        cudaMemcpy(got.data(),d_r,B,cudaMemcpyDeviceToHost);

        uint32_t m=0; int first=-1;
        for (uint32_t i=0;i<n;++i) if(got[i]!=ref[i]){ if(first<0)first=(int)i; ++m; }
        uint32_t bits=0; { uint64_t qq=q; while(qq){++bits;qq>>=1;} }
        std::cout << "  tower " << t << " q=" << q << " (" << bits << "-bit): "
                  << (m==0?"ok":"BAD");
        if(m) std::cout << " " << m << " mism first@" << first
                        << " got=" << got[first] << " ref=" << ref[first];
        std::cout << "\n";
        total_mism += m;
    }
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_r);

    if(total_mism==0){ std::cout << "[PASS] Barrett multiply bit-exact vs plain modulo\n"; return 0; }
    std::cout << "[FAIL]\n"; return 1;
}
