// Inverse NTT validation, two ways, both bit-exact:
//  (1) LaunchINTT_GS output == OpenFHE's InverseTransformFromBitReverseInPlace
//  (2) round-trip: INTT(NTT(x)) == x exactly (catches any n^{-1} scaling or
//      stage-ordering error instantly -- a wrong scale makes it off by a factor)
// Completes the forward/inverse transform pair the rescale needs.

#include "openfhe.h"
#include "math/hal/intnat/transformnat.h"
#include "ntt.h"
#include "intt.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace lbcrypto;

int main() {
    using NI = NativeInteger;
    using NV = intnat::NativeVectorT<NI>;

    const uint32_t n = 4096;
    const uint32_t cyclo = 2 * n;
    NI q = FirstPrime<NI>(50, cyclo);
    NI root = RootOfUnity<NI>(cyclo, q);
    NI rootInv = root.ModInverse(q);

    // Forward + inverse bit-reversed root tables + Shoup precons, reconstructed
    // publicly (fwd from root, inv from root^{-1}).
    auto bitrev = [](uint32_t v, uint32_t b){ uint32_t r=0;
        for(uint32_t k=0;k<b;k++){ r=(r<<1)|(v&1); v>>=1; } return r; };
    uint32_t logn = 0; while ((1u<<logn) < n) ++logn;

    auto build = [&](NI base, std::vector<uint64_t>& roots, std::vector<uint64_t>& prec){
        roots.resize(n); prec.resize(n);
        std::vector<NI> pw(n); pw[0] = NI(1);
        for (uint32_t i = 1; i < n; ++i) pw[i] = pw[i-1].ModMul(base, q);
        for (uint32_t i = 0; i < n; ++i) {
            NI r_i = pw[bitrev(i, logn)];
            roots[i] = r_i.ConvertToInt();
            prec[i]  = (uint64_t)(((__uint128_t)r_i.ConvertToInt() << 64)
                                  / (__uint128_t)q.ConvertToInt());
        }
    };
    std::vector<uint64_t> froots, fprec, iroots, iprec;
    build(root,    froots, fprec);
    build(rootInv, iroots, iprec);

    // n^{-1} mod q and its Shoup precon.
    NI nInv = NI(n).ModInverse(q);
    uint64_t n_inv = nInv.ConvertToInt();
    uint64_t n_inv_precon = (uint64_t)(((__uint128_t)n_inv << 64)
                                       / (__uint128_t)q.ConvertToInt());

    // Random standard-order input.
    std::mt19937_64 rng(20260725);
    std::vector<uint64_t> orig(n);
    NV in(n, q);
    for (uint32_t i = 0; i < n; ++i) {
        uint64_t v = rng() % q.ConvertToInt();
        orig[i] = v; in[i] = NI(v);
    }

    // OpenFHE reference: forward then inverse should give back orig.
    NV fwd = in;
    intnat::ChineseRemainderTransformFTTNat<NV> crt;
    crt.ForwardTransformToBitReverseInPlace(root, cyclo, &fwd);
    NV invref = fwd;
    crt.InverseTransformFromBitReverseInPlace(root, cyclo, &invref);

    // Device buffers.
    uint64_t *d_x=nullptr,*d_fr=nullptr,*d_fp=nullptr,*d_ir=nullptr,*d_ip=nullptr;
    const size_t B = (size_t)n*sizeof(uint64_t);
    cudaMalloc(&d_x,B); cudaMalloc(&d_fr,B); cudaMalloc(&d_fp,B);
    cudaMalloc(&d_ir,B); cudaMalloc(&d_ip,B);
    cudaMemcpy(d_fr,froots.data(),B,cudaMemcpyHostToDevice);
    cudaMemcpy(d_fp,fprec.data(), B,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ir,iroots.data(),B,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ip,iprec.data(), B,cudaMemcpyHostToDevice);

    uint64_t Q = q.ConvertToInt();

    // Check 1: run GPU forward NTT, then GPU INTT, expect orig back.
    cudaMemcpy(d_x, orig.data(), B, cudaMemcpyHostToDevice);
    LaunchNTT_CT(d_x, d_fr, d_fp, n, Q, 0);
    LaunchINTT_GS(d_x, d_ir, d_ip, n, Q, n_inv, n_inv_precon, 0);
    cudaDeviceSynchronize();
    std::vector<uint64_t> rt(n);
    cudaMemcpy(rt.data(), d_x, B, cudaMemcpyDeviceToHost);

    uint32_t rt_mism = 0; int rt_first = -1;
    for (uint32_t i = 0; i < n; ++i)
        if (rt[i] != orig[i]) { if(rt_first<0) rt_first=(int)i; ++rt_mism; }

    // Check 2: GPU INTT of the SAME bit-reversed input OpenFHE inverted,
    // compared to OpenFHE's inverse output directly.
    std::vector<uint64_t> fwd_h(n);
    for (uint32_t i = 0; i < n; ++i) fwd_h[i] = fwd[i].ConvertToInt();
    cudaMemcpy(d_x, fwd_h.data(), B, cudaMemcpyHostToDevice);
    LaunchINTT_GS(d_x, d_ir, d_ip, n, Q, n_inv, n_inv_precon, 0);
    cudaDeviceSynchronize();
    std::vector<uint64_t> gi(n);
    cudaMemcpy(gi.data(), d_x, B, cudaMemcpyDeviceToHost);

    uint32_t di_mism = 0; int di_first = -1;
    for (uint32_t i = 0; i < n; ++i)
        if (gi[i] != invref[i].ConvertToInt()) { if(di_first<0) di_first=(int)i; ++di_mism; }

    cudaFree(d_x); cudaFree(d_fr); cudaFree(d_fp); cudaFree(d_ir); cudaFree(d_ip);

    std::cout << "round-trip INTT(NTT(x))==x : "
              << (rt_mism==0 ? "PASS" : "FAIL") ;
    if (rt_mism) std::cout << " (" << rt_mism << " mism, first@" << rt_first
                           << " got=" << rt[rt_first] << " exp=" << orig[rt_first] << ")";
    std::cout << "\nGPU INTT vs OpenFHE inverse: "
              << (di_mism==0 ? "PASS" : "FAIL");
    if (di_mism) std::cout << " (" << di_mism << " mism, first@" << di_first
                           << " got=" << gi[di_first] << " exp=" << invref[di_first].ConvertToInt() << ")";
    std::cout << "\n";

    if (rt_mism==0 && di_mism==0) {
        std::cout << "[PASS] inverse NTT bit-exact (n=" << n << ")\n";
        return 0;
    }
    std::cout << "[FAIL]\n";
    return 1;
}
