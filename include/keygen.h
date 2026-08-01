#pragma once
#include <cuda_runtime.h>   // cudaStream_t for the device automorphism
#include <cstdint>
#include <vector>
namespace gpufhe {
struct KeyPairHost {
    std::vector<uint64_t> s;        // sizeQ*n eval
    std::vector<uint64_t> pkA, pkB; // sizeQ*n eval
};
void set_secret_hamming_weight(uint32_t h);
void automorphism_eval_device(uint64_t*, uint32_t, uint32_t, uint32_t,
                              const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                              uint64_t*, cudaStream_t);
void rotate_ct_resident(std::vector<uint64_t>&, std::vector<uint64_t>&, uint32_t,
                        const struct KeySwitchConstants&, uint32_t, uint32_t,
                        const std::vector<uint64_t>&, const std::vector<uint64_t>&);   // 0 = uniform ternary
KeyPairHost keygen_host(uint32_t n, const std::vector<uint64_t>& moduli,
                        const std::vector<uint64_t>& roots, uint64_t ns, double sigma,
                        uint64_t seed);

struct KeySwitchConstants; // fwd
void evalkeygen_host_sold(struct KeySwitchConstants& K,
    const std::vector<uint64_t>& sQP, const std::vector<uint64_t>& sOld,
    const std::vector<uint64_t>& pkA_QP, const std::vector<uint64_t>& pkB_QP,
    const std::vector<uint64_t>& PModq_QP, const std::vector<uint64_t>& modQP,
    const std::vector<uint64_t>& rootQP, uint64_t ns, double sigma, uint64_t seed);
void evalkeygen_host(struct KeySwitchConstants& K,
    const std::vector<uint64_t>& sQP, const std::vector<uint64_t>& pkA_QP,
    const std::vector<uint64_t>& pkB_QP, const std::vector<uint64_t>& PModq_QP,
    const std::vector<uint64_t>& modQP, const std::vector<uint64_t>& rootQP,
    uint64_t ns, double sigma, uint64_t seed);


void encrypt_host(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1,
    const std::vector<int64_t>& m, const std::vector<uint64_t>& pkA,
    const std::vector<uint64_t>& pkB, uint32_t n, const std::vector<uint64_t>& mod,
    const std::vector<uint64_t>& root, uint64_t ns, double sigma, uint64_t seed);
void decrypt_host(std::vector<int64_t>& mout, const std::vector<uint64_t>& c0,
    const std::vector<uint64_t>& c1, const std::vector<uint64_t>& s,
    uint32_t n, const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root);
#include <cstdint>

}
