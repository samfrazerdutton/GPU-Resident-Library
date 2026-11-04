// FFI bridge for the native CKKS encode/decode/encrypt/decrypt primitives
// (GOD PROMPT "CKKS PHASE 5": the real encrypted EvalMult round trip).
// Every function below forwards to the EXISTING, unmodified src/encode.cpp
// / src/encdec.cpp implementations -- none of their arithmetic (canonical-
// embedding encode/decode, the ternary-secret/Gaussian-error encrypt
// convention, the tower-0-centered decrypt recovery) is reimplemented
// here, only the calling convention is adapted: `std::complex<double>` and
// `std::vector<T>` have no stable FFI ABI, so complex slot vectors cross
// the boundary as separate real/imaginary `f64` arrays, and every other
// vector as a raw pointer + length pair.
//
// Unlike cpp_bridge/keyswitch_bridge.cpp, none of these functions need an
// opaque fixture -- encode_host/decode_host/encrypt_host/decrypt_host are
// already stateless, pure host functions (confirmed by reading
// src/encode.cpp and src/encdec.cpp: no global state, no device context
// retained across calls), so each is wrapped as a single direct call.
#include <cstdint>
#include <cstring>
#include <complex>
#include <vector>

namespace gpufhe {
// GOD PROMPT "CKKS PHASE 12": encode_host/decode_host (src/encode.cpp,
// Phase 11's std::thread-parallelized O(N^2)) remain fully present and
// tested, as the reference/fallback the governing directive requires --
// NOT called from here anymore. This bridge (the actual production call
// site every Rust caller goes through via codec::encode_real/
// codec::decode) now calls encode_host_fft/decode_host_fft (src/
// encode_fft.cpp, the new O(N log N) FFT-based implementation),
// verified bit-close (see research/FFT_ENCODING_DERIVATION.md and
// tests/test_encode_fft_matches_reference.cpp) against encode_host/
// decode_host across every supported dimension and a battery of
// adversarial inputs, and measured 31-61x (encode) / 40-81x (decode)
// faster, before this swap was made. encode_host/decode_host are still
// reachable directly by any test or future caller that explicitly wants
// the old path.
void encode_host_fft(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void decode_host_fft(std::vector<std::complex<double>>&, const std::vector<int64_t>&, uint32_t, double);
void encrypt_host(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                   const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void decrypt_host(std::vector<int64_t>&, const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                   const std::vector<uint64_t>&, uint32_t, const std::vector<uint64_t>&, const std::vector<uint64_t>&);
}

extern "C" {

// Encodes `num_slots` complex slots (`z_re[k] + i*z_im[k]`, `num_slots <=
// n/2`, the rest implicitly zero-padded) into `n` integer coefficients at
// scale `delta`. `m_out` must have exactly `n` elements.
void gpufhe_encode(const double* z_re, const double* z_im, uint32_t num_slots,
                    uint32_t n, double delta, int64_t* m_out)
{
    std::vector<std::complex<double>> z(n / 2, {0.0, 0.0});
    for (uint32_t k = 0; k < num_slots && k < n / 2; ++k) z[k] = {z_re[k], z_im[k]};
    std::vector<int64_t> m;
    gpufhe::encode_host_fft(m, z, n, delta);
    std::memcpy(m_out, m.data(), (size_t)n * 8);
}

// Decodes `n` integer coefficients at scale `delta` back into `n/2`
// complex slots. `z_re_out`/`z_im_out` must each have `n/2` elements.
void gpufhe_decode(const int64_t* m, uint32_t n, double delta, double* z_re_out, double* z_im_out)
{
    std::vector<int64_t> mv(m, m + n);
    std::vector<std::complex<double>> z;
    gpufhe::decode_host_fft(z, mv, n, delta);
    for (size_t k = 0; k < z.size(); ++k) { z_re_out[k] = z[k].real(); z_im_out[k] = z[k].imag(); }
}

// Encrypts `m` (n integer coefficients) under public key (pkA,pkB) over
// the modulus chain `mod`/`root` (size_q entries each). c0_out/c1_out
// must each have size_q*n elements.
void gpufhe_encrypt(const int64_t* m, uint32_t n,
                     const uint64_t* pkA, const uint64_t* pkB,
                     const uint64_t* mod, const uint64_t* root, uint32_t size_q,
                     uint64_t ns, double sigma, uint64_t seed,
                     uint64_t* c0_out, uint64_t* c1_out)
{
    std::vector<int64_t> mv(m, m + n);
    std::vector<uint64_t> pkAv(pkA, pkA + (size_t)size_q * n), pkBv(pkB, pkB + (size_t)size_q * n);
    std::vector<uint64_t> modv(mod, mod + size_q), rootv(root, root + size_q);
    std::vector<uint64_t> c0, c1;
    gpufhe::encrypt_host(c0, c1, mv, pkAv, pkBv, n, modv, rootv, ns, sigma, seed);
    std::memcpy(c0_out, c0.data(), (size_t)size_q * n * 8);
    std::memcpy(c1_out, c1.data(), (size_t)size_q * n * 8);
}

// Decrypts (c0,c1) (size_q*n each) under secret key `s` (size_q*n) over
// `mod`/`root` (size_q entries -- only tower 0 is actually used by
// decrypt_host's own tower-0-centered recovery, see src/encdec.cpp's own
// comment; the full-length arrays are still required since that is
// decrypt_host's own signature). `m_out` must have `n` elements.
void gpufhe_decrypt(const uint64_t* c0, const uint64_t* c1, const uint64_t* s,
                     uint32_t n, const uint64_t* mod, const uint64_t* root, uint32_t size_q,
                     int64_t* m_out)
{
    std::vector<uint64_t> c0v(c0, c0 + (size_t)size_q * n), c1v(c1, c1 + (size_t)size_q * n);
    std::vector<uint64_t> sv(s, s + (size_t)size_q * n);
    std::vector<uint64_t> modv(mod, mod + size_q), rootv(root, root + size_q);
    std::vector<int64_t> m;
    gpufhe::decrypt_host(m, c0v, c1v, sv, n, modv, rootv);
    std::memcpy(m_out, m.data(), (size_t)n * 8);
}

} // extern "C"
