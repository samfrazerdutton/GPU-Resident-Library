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
// GOD PROMPT "CKKS PHASE 13": encrypt_host (src/encdec.cpp, the Phase
// 11 batched-NTT version) remains fully present and tested, as the
// reference/fallback the governing directive requires -- NOT called
// from here anymore. This bridge (the actual production call site
// every Rust caller goes through via codec::encrypt) now calls
// encrypt_host_shoup_pooled (src/encrypt_host_shoup_pooled.cpp), which
// combines two independently verified, bit-exact optimizations over
// encrypt_host's own combine loop and device-buffer lifecycle: Shoup's
// precomputed-constant modular multiplication (replacing calls that
// profiling proved compile to libgcc's slow __umodti3 software 128-bit
// division -- see research/ENCRYPT_PROFILING.md) and a process-
// lifetime device-buffer pool keyed by (n,sizeQ) (eliminating repeated
// cudaMalloc/cudaFree). Verified bit-exact against encrypt_host across
// every supported dimension, multiple seeds, and multiple message
// patterns (tests/test_encrypt_host_shoup_matches_reference.cpp, tests/
// test_encrypt_host_shoup_pooled_matches_reference.cpp -- the latter
// also stress-tests interleaved calls across different (n,sizeQ)
// configurations to rule out pool cross-contamination), and measured
// 1.04x-1.19x (Shoup alone) / 1.13x-1.88x (combined) faster before this
// swap was made. encrypt_host is still reachable directly by any test
// or future caller that explicitly wants the old path.
void encrypt_host_shoup_pooled(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                                const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                                const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
// GOD PROMPT "CKKS PHASE 14": encrypt_host_shoup_pooled (Phase 13)
// remains fully present and tested, as the reference/fallback the
// governing directive requires -- NOT called from here anymore. This
// bridge now calls encrypt_host_shoup_pooled_barrett (src/
// encrypt_host_shoup_pooled_barrett.cpp), which ADDITIONALLY replaces
// the embedding loop's signed-64-bit `emb(x)=(x%q+q)%q` -- confirmed
// via assembly inspection to compile to two hardware `idiv`
// instructions per call, and measured as encrypt_host's single LARGEST
// stage (20-27% of total) -- with a Barrett reduction precomputed once
// per tower (not per coefficient). Verified bit-exact against
// encrypt_host across every supported dimension, multiple seeds, and
// messages spanning the full int64_t range including LONG_MIN/
// LONG_MAX (tests/
// test_encrypt_host_shoup_pooled_barrett_matches_reference.cpp), and
// the derivation itself independently proven via 22M+ differential-
// test cases against both synthetic and this project's own real
// native_primes()-generated moduli (tests/
// test_embed_barrett_matches_reference.cpp) before any production code
// was touched. The buffer-pool concurrency hardening from Phase 13
// (one mutex serializing the whole call) is preserved unchanged and
// re-verified under direct multi-threaded stress (tests/
// test_encrypt_host_shoup_pooled_barrett_concurrency_audit.cpp).
// Measured 1.18x-1.28x faster than Phase 13 alone, 1.44x-1.73x faster
// than the original encrypt_host, before this swap was made.
// encrypt_host_shoup_pooled and encrypt_host are both still reachable
// directly by any test or future caller that wants an earlier path.
// See research/EMBED_AND_POOL_HARDENING.md.
void encrypt_host_shoup_pooled_barrett(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                                        const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                                        const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
// GOD PROMPT "CKKS PHASE 15": encrypt_host_shoup_pooled_barrett
// (Phase 14) remains fully present and tested, as the reference/
// fallback the governing directive requires -- NOT called from here
// anymore. This bridge now calls encrypt_host_shoup_pooled_barrett_
// batched (src/encrypt_host_shoup_pooled_barrett_batched.cpp), which
// ADDITIONALLY merges the four separate per-buffer H2D/D2H cudaMemcpy
// calls (for d_v,d_e0,d_e1,d_m) into ONE H2D and ONE D2H call each,
// over a single packed device buffer -- the four kernel launches
// themselves are UNCHANGED (same signature, same tables, same tower
// count; they now receive pointer offsets into one packed allocation
// instead of four separate ones). Verified bit-exact against
// encrypt_host, including the packed-buffer boundaries specifically
// (tests/
// test_encrypt_host_shoup_pooled_barrett_batched_matches_reference.cpp),
// and the new pooled resource's concurrency safety re-audited with 4
// dedicated multi-threaded scenarios (tests/
// test_encrypt_host_shoup_pooled_barrett_batched_concurrency_audit.cpp,
// 5 runs, 0/396 mismatches). Measured 1.04x-1.17x faster than Phase 14
// alone, 1.57x-2.78x faster than the original encrypt_host, before
// this swap was made.
//
// A SECOND experiment this round (pinned host staging memory, src/
// encrypt_host_shoup_pooled_barrett_batched_pinned.cpp) was built,
// verified bit-exact, and measured a further REAL 1.06x-1.17x
// steady-state speedup -- but was NOT promoted here: `cudaMallocHost`
// was measured to cost ~3-8ms per distinct (n,sizeQ) key on first use
// (vs. ~0.4-0.5ms for the ordinary `cudaMalloc` this batched version
// already uses), large enough to exceed a single encrypt call's ENTIRE
// latency and make the pinned version a net LOSS for any usage pattern
// that does not repeat many calls at the same key before the process
// ends. See research/TRANSFER_PROFILING.md for the full measurement
// and this round's keep/revert reasoning.
void encrypt_host_shoup_pooled_barrett_batched(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
                                                const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
                                                const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t);
void decrypt_host(std::vector<int64_t>&, const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                   const std::vector<uint64_t>&, uint32_t, const std::vector<uint64_t>&, const std::vector<uint64_t>&);
// GOD PROMPT "CKKS PHASE 22": Phase 20's CUDA-registered host-memory
// EXPERIMENTAL encrypt path (src/encrypt_host_shoup_pooled_barrett_
// batched_registered.cpp), now reachable from Rust for the first time,
// via a NEW, separate FFI entry point below -- gpufhe_encrypt (above)
// keeps calling the unchanged production pageable path; nothing about
// the default Rust `codec::encrypt` changes. The `_diag` variant
// reports which path actually executed (the C++-only Phase 20 API had
// no way to tell a caller this without a SEPARATE query call).
void encrypt_host_shoup_pooled_barrett_batched_registered_diag(
    std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<int64_t>&,
    const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t,
    const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t, double, uint64_t,
    bool&, bool&, int&);
bool encrypt_prewarm_registered_buffer_diag(uint32_t, uint32_t, bool&, int&);
bool encrypt_registered_buffer_is_ready(uint32_t, uint32_t);
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
    gpufhe::encrypt_host_shoup_pooled_barrett_batched(c0, c1, mv, pkAv, pkBv, n, modv, rootv, ns, sigma, seed);
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

// GOD PROMPT "CKKS PHASE 22": EXPERIMENTAL -- CUDA-registered host
// memory (Phase 20), reachable from Rust for the first time. Same
// calling convention/ownership shape as gpufhe_encrypt above (every
// argument copied into/out of std::vector on the C++ side; nothing
// retained across the call; the underlying C++ function is fully
// synchronous -- it calls cudaDeviceSynchronize() before returning, so
// there is no asynchronous GPU work still touching this memory once
// this function returns, and no pointer crosses back into Rust that
// Rust could use after the C++-owned pool entry is gone). out_used_
// registered/out_budget_skipped/out_last_error_code are ALWAYS written
// (never left unspecified) so the Rust caller can observe exactly
// which path executed and, on fallback, distinguish a deliberate
// budget skip from a genuine cudaHostRegister failure.
void gpufhe_encrypt_registered(const int64_t* m, uint32_t n,
                                const uint64_t* pkA, const uint64_t* pkB,
                                const uint64_t* mod, const uint64_t* root, uint32_t size_q,
                                uint64_t ns, double sigma, uint64_t seed,
                                uint64_t* c0_out, uint64_t* c1_out,
                                uint8_t* out_used_registered, uint8_t* out_budget_skipped, int32_t* out_last_error_code)
{
    std::vector<int64_t> mv(m, m + n);
    std::vector<uint64_t> pkAv(pkA, pkA + (size_t)size_q * n), pkBv(pkB, pkB + (size_t)size_q * n);
    std::vector<uint64_t> modv(mod, mod + size_q), rootv(root, root + size_q);
    std::vector<uint64_t> c0, c1;
    bool usedRegistered = false, budgetSkipped = false; int lastErrorCode = 0;
    gpufhe::encrypt_host_shoup_pooled_barrett_batched_registered_diag(
        c0, c1, mv, pkAv, pkBv, n, modv, rootv, ns, sigma, seed,
        usedRegistered, budgetSkipped, lastErrorCode);
    std::memcpy(c0_out, c0.data(), (size_t)size_q * n * 8);
    std::memcpy(c1_out, c1.data(), (size_t)size_q * n * 8);
    *out_used_registered = usedRegistered ? 1 : 0;
    *out_budget_skipped = budgetSkipped ? 1 : 0;
    *out_last_error_code = (int32_t)lastErrorCode;
}

// Explicit, idempotent prewarm for the registered pool (mirrors
// gpufhe_encrypt_registered's diagnostic shape). Returns 1 iff the
// (n,size_q) key is now registered and ready; always writes the two
// out-params so a caller can tell a budget skip from a real failure
// even when ready==0.
uint8_t gpufhe_prewarm_registered_buffer(uint32_t n, uint32_t size_q, uint8_t* out_budget_skipped, int32_t* out_last_error_code)
{
    bool budgetSkipped = false; int lastErrorCode = 0;
    bool ready = gpufhe::encrypt_prewarm_registered_buffer_diag(n, size_q, budgetSkipped, lastErrorCode);
    *out_budget_skipped = budgetSkipped ? 1 : 0;
    *out_last_error_code = (int32_t)lastErrorCode;
    return ready ? 1 : 0;
}

// Query-only: is the registered pool already ready for (n,size_q)?
// Never allocates. Mirrors the existing C++-only
// encrypt_registered_buffer_is_ready exactly.
uint8_t gpufhe_registered_buffer_is_ready(uint32_t n, uint32_t size_q)
{
    return gpufhe::encrypt_registered_buffer_is_ready(n, size_q) ? 1 : 0;
}

} // extern "C"
