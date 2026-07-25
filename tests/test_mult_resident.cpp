// Second resident operation: coefficient-wise multiply of two ciphertexts,
// entirely in VRAM, bit-exact vs OpenFHE's eval-form DCRTPoly multiply.
// Multi-tower from the start (each tower has its own modulus), so a per-modulus
// bug surfaces here rather than deep in a chain.

#include "openfhe.h"
#include "device_ciphertext.h"
#include "rns_arith.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace lbcrypto;

int main() {
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(2);
    params.SetScalingModSize(50);
    params.SetRingDim(32768);
    params.SetBatchSize(16384);
    auto cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);

    auto ep = cc->GetCryptoParameters()->GetElementParams();
    const uint32_t n = ep->GetRingDimension();
    const uint32_t towers = (uint32_t)ep->GetParams().size();

    // Build two random DCRTPolys in EVALUATION form from tower vectors.
    using Poly = DCRTPoly::PolyType;
    std::mt19937_64 rng(20260725);
    std::vector<Poly> towersA, towersB;
    towersA.reserve(towers); towersB.reserve(towers);
    for (uint32_t t = 0; t < towers; ++t) {
        auto pp = ep->GetParams()[t];
        auto q = pp->GetModulus().ConvertToInt();
        NativeVector va(n, pp->GetModulus()), vb(n, pp->GetModulus());
        for (uint32_t i = 0; i < n; ++i) {
            va[i] = NativeInteger(rng() % q);
            vb[i] = NativeInteger(rng() % q);
        }
        Poly pa(pp, Format::EVALUATION, true);
        Poly pb(pp, Format::EVALUATION, true);
        pa.SetValues(std::move(va), Format::EVALUATION);
        pb.SetValues(std::move(vb), Format::EVALUATION);
        towersA.push_back(std::move(pa));
        towersB.push_back(std::move(pb));
    }
    DCRTPoly A(towersA);
    DCRTPoly B(towersB);

    // OpenFHE reference: eval-form multiply.
    DCRTPoly REF = A * B;

    // Flatten A, B into tower-major host buffers and upload to resident cts.
    std::vector<uint64_t> hA(n * towers), hB(n * towers);
    for (uint32_t t = 0; t < towers; ++t)
        for (uint32_t i = 0; i < n; ++i) {
            hA[t*n + i] = A.GetAllElements()[t][i].ConvertToInt();
            hB[t*n + i] = B.GetAllElements()[t][i].ConvertToInt();
        }

    std::vector<uint64_t> zero(n * towers, 0);
    gpufhe::DeviceCiphertext ctA(gpufhe::Scheme::CKKS, n, towers);
    gpufhe::DeviceCiphertext ctB(gpufhe::Scheme::CKKS, n, towers);
    gpufhe::DeviceCiphertext ctR(gpufhe::Scheme::CKKS, n, towers);
    ctA.upload(hA, zero);
    ctB.upload(hB, zero);

    // Resident per-tower multiply: c0 of A * c0 of B -> c0 of R, all on device.
    for (uint32_t t = 0; t < towers; ++t) {
        auto q = ep->GetParams()[t]->GetModulus().ConvertToInt();
        LaunchRNSMultTower(ctA.c0() + t*n, ctB.c0() + t*n, ctR.c0() + t*n,
                           q, n, 0);
    }
    cudaDeviceSynchronize();

    std::vector<uint64_t> hR, hRc1;
    ctR.to_host(hR, hRc1);

    uint32_t mism = 0; int ft = -1, fi = -1;
    for (uint32_t t = 0; t < towers && mism == 0; ++t)
        for (uint32_t i = 0; i < n; ++i)
            if (hR[t*n + i] != REF.GetAllElements()[t][i].ConvertToInt()) {
                ft = (int)t; fi = (int)i; ++mism; break;
            }

    if (mism == 0) {
        std::cout << "[PASS] resident multiply bit-exact vs OpenFHE (n=" << n
                  << ", towers=" << towers << ")\n";
        return 0;
    }
    std::cout << "[FAIL] first mismatch tower " << ft << " idx " << fi
              << " gpu=" << hR[ft*n + fi]
              << " ref=" << REF.GetAllElements()[ft][fi].ConvertToInt() << "\n";
    return 1;
}
