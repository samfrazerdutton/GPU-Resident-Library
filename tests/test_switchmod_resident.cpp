#include "openfhe.h"
#include "switch_modulus.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace lbcrypto;

static uint32_t check_switch(uint32_t n, NativeInteger q_old, NativeInteger q_new,
                             int& first, uint64_t& gotv, uint64_t& expv) {
    using NV = intnat::NativeVectorT<NativeInteger>;
    std::mt19937_64 rng(20260726);
    NV ref(n, q_old);
    std::vector<uint64_t> host(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint64_t v = rng() % q_old.ConvertToInt();
        ref[i] = NativeInteger(v);
        host[i] = v;
    }
    ref.SwitchModulus(q_new);

    uint64_t* d = nullptr;
    const size_t B = (size_t)n * sizeof(uint64_t);
    cudaMalloc(&d, B);
    cudaMemcpy(d, host.data(), B, cudaMemcpyHostToDevice);
    LaunchSwitchModulus(d, q_old.ConvertToInt(), q_new.ConvertToInt(), n, 0);
    cudaDeviceSynchronize();
    std::vector<uint64_t> got(n);
    cudaMemcpy(got.data(), d, B, cudaMemcpyDeviceToHost);
    cudaFree(d);

    uint32_t mism = 0; first = -1;
    for (uint32_t i = 0; i < n; ++i)
        if (got[i] != ref[i].ConvertToInt()) {
            if (first < 0) { first = (int)i; gotv = got[i]; expv = ref[i].ConvertToInt(); }
            ++mism;
        }
    return mism;
}

int main() {
    const uint32_t n = 4096;
    const uint32_t cyclo = 2 * n;
    NativeInteger qBig   = FirstPrime<NativeInteger>(55, cyclo);
    NativeInteger qSmall = FirstPrime<NativeInteger>(50, cyclo);

    int f1, f2; uint64_t g1=0, e1=0, g2=0, e2=0;
    uint32_t m1 = check_switch(n, qBig, qSmall, f1, g1, e1);
    uint32_t m2 = check_switch(n, qSmall, qBig, f2, g2, e2);

    std::cout << "larger->smaller: " << (m1==0 ? "PASS" : "FAIL");
    if (m1) std::cout << " (" << m1 << " mism, first@" << f1
                      << " got=" << g1 << " exp=" << e1 << ")";
    std::cout << "\nsmaller->larger: " << (m2==0 ? "PASS" : "FAIL");
    if (m2) std::cout << " (" << m2 << " mism, first@" << f2
                      << " got=" << g2 << " exp=" << e2 << ")";
    std::cout << "\n";

    if (m1==0 && m2==0) {
        std::cout << "[PASS] SwitchModulus bit-exact vs OpenFHE, both directions (n="
                  << n << ")\n";
        return 0;
    }
    std::cout << "[FAIL]\n";
    return 1;
}
