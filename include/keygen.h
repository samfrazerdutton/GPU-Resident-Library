#pragma once
#include <cstdint>
#include <vector>
namespace gpufhe {
struct KeyPairHost {
    std::vector<uint64_t> s;        // sizeQ*n eval
    std::vector<uint64_t> pkA, pkB; // sizeQ*n eval
};
KeyPairHost keygen_host(uint32_t n, const std::vector<uint64_t>& moduli,
                        const std::vector<uint64_t>& roots, uint64_t ns, double sigma,
                        uint64_t seed);

struct KeySwitchConstants; // fwd
void evalkeygen_host(struct KeySwitchConstants& K,
    const std::vector<uint64_t>& sQP, const std::vector<uint64_t>& pkA_QP,
    const std::vector<uint64_t>& pkB_QP, const std::vector<uint64_t>& PModq_QP,
    const std::vector<uint64_t>& modQP, const std::vector<uint64_t>& rootQP,
    uint64_t ns, double sigma, uint64_t seed);

}
