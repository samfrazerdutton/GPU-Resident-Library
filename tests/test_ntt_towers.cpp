// Validates the (shared-memory) forward NTT bit-exact vs OpenFHE across a real
// depth-10 ring-32768 context's tower moduli. n=32768 is where the early/late
// stage split engages -- n=4096 (test_ntt_resident) doesn't stress it. This is
// the gate for the shared-memory NTT: the omega-index mapping in the fused
// shared kernel must be exact across all towers incl. the 60-bit prime.

#include "openfhe.h"
#include "math/hal/intnat/transformnat.h"
#include "ntt.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace lbcrypto;

int main() {
    using NI = NativeInteger;
    using NV = intnat::NativeVectorT<NI>;

    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(10);
    params.SetScalingModSize(50);
    params.SetRingDim(32768);
    params.SetBatchSize(16384);
    auto cc = GenCryptoContext(params);
    auto ep = cc->GetCryptoParameters()->GetElementParams();
    const uint32_t n = ep->GetRingDimension();
    const uint32_t towers = (uint32_t)ep->GetParams().size();
    const uint32_t cyclo = 2*n;

    auto bitrev = [](uint32_t v, uint32_t b){ uint32_t r=0;
        for(uint32_t k=0;k<b;k++){ r=(r<<1)|(v&1); v>>=1; } return r; };
    uint32_t logn=0; while((1u<<logn)<n) ++logn;

    const size_t B = (size_t)n*sizeof(uint64_t);
    uint64_t *d_x=nullptr,*d_r=nullptr,*d_p=nullptr;
    cudaMalloc(&d_x,B); cudaMalloc(&d_r,B); cudaMalloc(&d_p,B);

    std::mt19937_64 rng(20260728);
    uint32_t total_bad = 0;
    for (uint32_t t=0;t<towers;++t) {
        NI q = ep->GetParams()[t]->GetModulus();
        NI root = RootOfUnity<NI>(cyclo, q);

        NV in(n, q);
        for (uint32_t i=0;i<n;++i) in[i]=NI(rng()%q.ConvertToInt());
        NV ref = in;
        intnat::ChineseRemainderTransformFTTNat<NV> crt;
        crt.ForwardTransformToBitReverseInPlace(root, cyclo, &ref);

        std::vector<uint64_t> roots(n), precon(n), h(n);
        std::vector<NI> pw(n); pw[0]=NI(1);
        for (uint32_t i=1;i<n;++i) pw[i]=pw[i-1].ModMul(root,q);
        for (uint32_t i=0;i<n;++i){ NI ri=pw[bitrev(i,logn)];
            roots[i]=ri.ConvertToInt();
            precon[i]=(uint64_t)(((__uint128_t)ri.ConvertToInt()<<64)/(__uint128_t)q.ConvertToInt());
            h[i]=in[i].ConvertToInt(); }

        cudaMemcpy(d_x,h.data(),B,cudaMemcpyHostToDevice);
        cudaMemcpy(d_r,roots.data(),B,cudaMemcpyHostToDevice);
        cudaMemcpy(d_p,precon.data(),B,cudaMemcpyHostToDevice);
        LaunchNTT_CT(d_x,d_r,d_p,n,q.ConvertToInt(),0);
        cudaDeviceSynchronize();
        std::vector<uint64_t> got(n);
        cudaMemcpy(got.data(),d_x,B,cudaMemcpyDeviceToHost);

        uint32_t m=0; int first=-1;
        for (uint32_t i=0;i<n;++i) if(got[i]!=ref[i].ConvertToInt()){ if(first<0)first=(int)i; ++m; }
        uint32_t bits=0; { uint64_t qq=q.ConvertToInt(); while(qq){++bits;qq>>=1;} }
        std::cout << "  tower " << t << " (" << bits << "-bit): " << (m==0?"ok":"BAD");
        if(m) std::cout << " " << m << " mism first@" << first
                        << " got=" << got[first] << " ref=" << ref[first].ConvertToInt();
        std::cout << "\n";
        total_bad += m;
    }
    cudaFree(d_x); cudaFree(d_r); cudaFree(d_p);

    if(total_bad==0){ std::cout << "[PASS] shared-mem NTT bit-exact vs OpenFHE, all towers (n="
                                << n << ")\n"; return 0; }
    std::cout << "[FAIL]\n"; return 1;
}
