#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace gpufhe {
enum class Scheme : uint8_t { CKKS, BFV, BGV };
enum class Format : uint8_t { COEFFICIENT, EVALUATION };

// Ciphertext whose RNS tower data lives in VRAM and is OWNED here. No cache,
// no host-pointer keying: the object is the owner, identity is object identity.
// Data crosses PCIe twice total -- up at upload(), down at to_host(). Every op
// between mutates device memory in place; the host is never consulted. This is
// the property OpenFHE's SetValues-realloc made impossible at the HAL boundary.
class DeviceCiphertext {
public:
    DeviceCiphertext(Scheme scheme, uint32_t n, uint32_t num_towers);
    ~DeviceCiphertext();
    DeviceCiphertext(const DeviceCiphertext&) = delete;
    DeviceCiphertext& operator=(const DeviceCiphertext&) = delete;
    DeviceCiphertext(DeviceCiphertext&&) noexcept;
    DeviceCiphertext& operator=(DeviceCiphertext&&) noexcept;

    void upload(const std::vector<uint64_t>& c0, const std::vector<uint64_t>& c1);
    void to_host(std::vector<uint64_t>& c0, std::vector<uint64_t>& c1) const;

    uint64_t* c0() { return d_c0_; }
    uint64_t* c1() { return d_c1_; }
    uint32_t n()          const { return n_; }
    uint32_t num_towers() const { return towers_; }
    Format   format()     const { return format_; }
    void set_format(Format f)   { format_ = f; }
    void drop_tower()           { if (towers_) --towers_; }
    Scheme scheme()       const { return scheme_; }

private:
    Scheme   scheme_;
    Format   format_ = Format::EVALUATION;
    uint32_t n_;
    uint32_t towers_;
    uint64_t* d_c0_ = nullptr;
    uint64_t* d_c1_ = nullptr;
};
}  // namespace gpufhe
