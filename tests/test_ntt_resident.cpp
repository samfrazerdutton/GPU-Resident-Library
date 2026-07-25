// Proves the device-resident architecture: a DeviceCiphertext owns its tower
// data in VRAM, the NTT runs on it in place, and the result is bit-exact vs
// OpenFHE's transform. No HAL, no cache, no host-pointer keying -- the object
// owns the data start to finish. This is what the OpenFHE-HAL approach could
// not do.

#include "openfhe.h"
#include "math/hal/intnat/transformnat.h"
#include "device_ciphertext.h"
#include "ntt.h"
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

    // Random coefficient-form input.
    intnat::ChineseRemainderTransformFTTNat<NV> crt;
    NV in(n, q);
    std::mt19937_64 rng(20260725);
    for (uint32_t i = 0; i < n; ++i) in[i] = NI(rng() % q.ConvertToInt());

    // OpenFHE reference transform.
    NV ref = in;
    crt.ForwardTransformToBitReverseInPlace(root, cyclo, &ref);

    // Reconstruct OpenFHE's bit-reversed root + Shoup precon tables publicly.
    auto bitrev = [](uint32_t v, uint32_t b){ uint32_t r=0;
        for(uint32_t k=0;k<b;k++){ r=(r<<1)|(v&1); v>>=1; } return r; };
    uint32_t logn = 0; while ((1u<<logn) < n) ++logn;
    std::vector<uint64_t> roots(n), precon(n);
    std::vector<NI> pw(n); pw[0] = NI(1);
    for (uint32_t i = 1; i < n; ++i) pw[i] = pw[i-1].ModMul(root, q);
    for (uint32_t i = 0; i < n; ++i) {
        NI r_i = pw[bitrev(i, logn)];
        roots[i]  = r_i.ConvertToInt();
        precon[i] = (uint64_t)(((__uint128_t)r_i.ConvertToInt() << 64)
                               / (__uint128_t)q.ConvertToInt());
    }

    // Load the input into a device-resident ciphertext (1 tower, c0 = data).
    std::vector<uint64_t> h_c0(n), h_c1(n, 0);
    for (uint32_t i = 0; i < n; ++i) h_c0[i] = in[i].ConvertToInt();

    gpufhe::DeviceCiphertext ct(gpufhe::Scheme::CKKS, n, 1);
    ct.set_format(gpufhe::Format::COEFFICIENT);
    ct.upload(h_c0, h_c1);

    // Upload the root tables and run the NTT on the RESIDENT buffer in place.
    uint64_t *d_r=nullptr, *d_p=nullptr;
    cudaMalloc(&d_r, n*sizeof(uint64_t));
    cudaMalloc(&d_p, n*sizeof(uint64_t));
    cudaMemcpy(d_r, roots.data(),  n*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, precon.data(), n*sizeof(uint64_t), cudaMemcpyHostToDevice);

    LaunchNTT_CT(ct.c0(), d_r, d_p, n, q.ConvertToInt(), 0);
    cudaDeviceSynchronize();
    ct.set_format(gpufhe::Format::EVALUATION);

    // Bring it back and compare bit-exact.
    std::vector<uint64_t> out_c0, out_c1;
    ct.to_host(out_c0, out_c1);
    cudaFree(d_r); cudaFree(d_p);

    uint32_t mism = 0; int first = -1;
    for (uint32_t i = 0; i < n; ++i)
        if (out_c0[i] != ref[i].ConvertToInt()) { if(first<0) first=(int)i; ++mism; }

    if (mism == 0) {
        std::cout << "[PASS] resident NTT bit-exact vs OpenFHE (n=" << n
                  << ", q=" << q.ConvertToInt() << ")\n";
        return 0;
    }
    std::cout << "[FAIL] " << mism << " mismatches, first@" << first
              << " gpu=" << out_c0[first] << " ref=" << ref[first].ConvertToInt() << "\n";
    return 1;
}
