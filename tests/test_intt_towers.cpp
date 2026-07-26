// Validates the shared-memory INTT bit-exact vs OpenFHE's inverse transform
// across a real depth-10 ring-32768 context's tower moduli. n=32768 stresses
// the shared/global stage split that n=4096 (test_intt_resident) doesn't. This
// is the gate for the shared-memory INTT before it's benchmarked.

#include "openfhe.h"
#include "math/hal/intnat/transformnat.h"
#include "intt.h"
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
    uint64_t *d_x=nullptr,*d_ir=nullptr,*d_ip=nullptr;
    cudaMalloc(&d_x,B); cudaMalloc(&d_ir,B); cudaMalloc(&d_ip,B);

    std::mt19937_64 rng(20260728);
    uint32_t total_bad = 0;
    for (uint32_t t=0;t<towers;++t) {
        NI q = ep->GetParams()[t]->GetModulus();
        NI root = RootOfUnity<NI>(cyclo, q);
        NI rootInv = root.ModInverse(q);

        // Bit-reversed input (INTT expects bit-reversed order in).
        NV in(n, q);
        for (uint32_t i=0;i<n;++i) in[i]=NI(rng()%q.ConvertToInt());
        NV ref = in;
        intnat::ChineseRemainderTransformFTTNat<NV> crt;
        crt.InverseTransformFromBitReverseInPlace(root, cyclo, &ref);

        std::vector<uint64_t> iroots(n), iprec(n), h(n);
        std::vector<NI> pw(n); pw[0]=NI(1);
        for (uint32_t i=1;i<n;++i) pw[i]=pw[i-1].ModMul(rootInv,q);
        for (uint32_t i=0;i<n;++i){ NI ri=pw[bitrev(i,logn)];
            iroots[i]=ri.ConvertToInt();
            iprec[i]=(uint64_t)(((__uint128_t)ri.ConvertToInt()<<64)/(__uint128_t)q.ConvertToInt());
            h[i]=in[i].ConvertToInt(); }
        NI ninv = NI(n).ModInverse(q);
        uint64_t n_inv = ninv.ConvertToInt();
        uint64_t n_inv_p = (uint64_t)(((__uint128_t)n_inv<<64)/(__uint128_t)q.ConvertToInt());

        cudaMemcpy(d_x,h.data(),B,cudaMemcpyHostToDevice);
        cudaMemcpy(d_ir,iroots.data(),B,cudaMemcpyHostToDevice);
        cudaMemcpy(d_ip,iprec.data(),B,cudaMemcpyHostToDevice);
        LaunchINTT_GS(d_x,d_ir,d_ip,n,q.ConvertToInt(),n_inv,n_inv_p,0);
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
    cudaFree(d_x); cudaFree(d_ir); cudaFree(d_ip);

    if(total_bad==0){ std::cout << "[PASS] shared-mem INTT bit-exact vs OpenFHE, all towers (n="
                                << n << ")\n"; return 0; }
    std::cout << "[FAIL]\n"; return 1;
}
