// GOD PROMPT "CKKS RUST + CUDA PHASE 2" round: minimal, additive C ABI
// bridge exposing the EXISTING, unmodified src/params.cpp's
// gpufhe::native_rescale_consts to Rust. That function takes/returns
// std::vector<uint64_t>& -- not a stable C ABI type -- so this bridge
// takes raw pointers + lengths instead, constructs the vectors
// internally, calls the real (unmodified) implementation, and copies the
// results into caller-owned output buffers. Adds ZERO new cryptographic
// or numeric logic: s1/s2 (the per-tower rescale-fuse constants) are
// computed EXACTLY as native_rescale_consts always has -- see
// src/params.cpp for the actual math (already independently verified in
// an earlier round: s2[i] proven numerically identical to
// (qi-s1[i]) mod qi for the real modulus chains this project uses).
//
// Ownership: `mod_` is read-only, `mod_len` elements, not retained.
// `s1_out`/`s2_out` must each have at least `last` elements of writable
// storage -- the caller allocates and owns them; this function only
// writes into them, never (re)allocates or frees anything the caller
// didn't hand it.
#include <cstdint>
#include <vector>
#include <algorithm>

namespace gpufhe {
void native_rescale_consts(std::vector<uint64_t>&, std::vector<uint64_t>&,
                            const std::vector<uint64_t>&, uint32_t);
}

extern "C" void gpufhe_native_rescale_consts(
    const uint64_t* mod_, uint32_t mod_len, uint32_t last,
    uint64_t* s1_out, uint64_t* s2_out)
{
    std::vector<uint64_t> mod_vec(mod_, mod_ + mod_len);
    std::vector<uint64_t> s1, s2;
    gpufhe::native_rescale_consts(s1, s2, mod_vec, last);
    // native_rescale_consts always resizes s1/s2 to exactly `last`
    // elements -- see src/params.cpp -- so this copy is exactly as large
    // as the caller was told to allocate.
    std::copy(s1.begin(), s1.end(), s1_out);
    std::copy(s2.begin(), s2.end(), s2_out);
}
