// GOD PROMPT "CKKS RUST + CUDA EVOLUTION" round: minimal, additive C ABI
// bridge exposing the EXISTING, unmodified src/params.cpp's
// gpufhe::native_root (primitive 2n-th root of unity search, q=1 mod 2n
// NTT-friendly prime math) to Rust via extern "C" linkage. This file adds
// ZERO new cryptographic logic -- it forward-declares the real function
// exactly the way this project's own test files already do (see e.g.
// tests/test_rescale_n1024.cpp's own `namespace gpufhe { uint64_t
// native_root(uint32_t, uint64_t); }` forward declaration) and wraps it.
// Per the governing directive's Section 5D ("ckks-ffi... stable C ABI for
// interoperability with the existing C++ engine... narrow boundaries,
// document ownership"): this boundary is intentionally narrow -- one
// pure, side-effect-free, no-throw-in-practice numeric function, no
// pointers, no ownership transfer, nothing that can unwind a C++
// exception across the FFI boundary under the exact inputs this project
// actually passes it (n a power of two, q a native_primes-selected NTT-
// friendly prime). native_root is a pure host-side function (no CUDA),
// so this file is compiled with a plain C++ compiler, not nvcc -- no
// device-link step is needed for it.
#include <cstdint>

namespace gpufhe {
uint64_t native_root(uint32_t n, uint64_t q);
}

extern "C" uint64_t gpufhe_native_root(uint32_t n, uint64_t q) {
    return gpufhe::native_root(n, q);
}
