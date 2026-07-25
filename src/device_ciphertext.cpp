#include "device_ciphertext.h"
using namespace gpufhe;
#include <stdexcept>
#include <string>

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("DeviceCiphertext: ") + what +
                                 ": " + cudaGetErrorString(e));
}

DeviceCiphertext::DeviceCiphertext(Scheme scheme, uint32_t n, uint32_t num_towers)
    : scheme_(scheme), n_(n), towers_(num_towers) {
    const size_t bytes = (size_t)n_ * towers_ * sizeof(uint64_t);
    ck(cudaMalloc(&d_c0_, bytes), "cudaMalloc c0");
    ck(cudaMalloc(&d_c1_, bytes), "cudaMalloc c1");
}

DeviceCiphertext::~DeviceCiphertext() {
    if (d_c0_) cudaFree(d_c0_);
    if (d_c1_) cudaFree(d_c1_);
}

DeviceCiphertext::DeviceCiphertext(DeviceCiphertext&& o) noexcept
    : scheme_(o.scheme_), format_(o.format_), n_(o.n_), towers_(o.towers_),
      d_c0_(o.d_c0_), d_c1_(o.d_c1_) {
    o.d_c0_ = nullptr;
    o.d_c1_ = nullptr;
}

DeviceCiphertext& DeviceCiphertext::operator=(DeviceCiphertext&& o) noexcept {
    if (this != &o) {
        if (d_c0_) cudaFree(d_c0_);
        if (d_c1_) cudaFree(d_c1_);
        scheme_ = o.scheme_; format_ = o.format_;
        n_ = o.n_; towers_ = o.towers_;
        d_c0_ = o.d_c0_; d_c1_ = o.d_c1_;
        o.d_c0_ = nullptr; o.d_c1_ = nullptr;
    }
    return *this;
}

void DeviceCiphertext::upload(const std::vector<uint64_t>& c0,
                              const std::vector<uint64_t>& c1) {
    const size_t need = (size_t)n_ * towers_;
    if (c0.size() != need || c1.size() != need)
        throw std::runtime_error("DeviceCiphertext::upload size mismatch");
    const size_t bytes = need * sizeof(uint64_t);
    ck(cudaMemcpy(d_c0_, c0.data(), bytes, cudaMemcpyHostToDevice), "upload c0");
    ck(cudaMemcpy(d_c1_, c1.data(), bytes, cudaMemcpyHostToDevice), "upload c1");
}

void DeviceCiphertext::to_host(std::vector<uint64_t>& c0,
                               std::vector<uint64_t>& c1) const {
    const size_t need = (size_t)n_ * towers_;
    c0.resize(need);
    c1.resize(need);
    const size_t bytes = need * sizeof(uint64_t);
    ck(cudaMemcpy(c0.data(), d_c0_, bytes, cudaMemcpyDeviceToHost), "to_host c0");
    ck(cudaMemcpy(c1.data(), d_c1_, bytes, cudaMemcpyDeviceToHost), "to_host c1");
}
