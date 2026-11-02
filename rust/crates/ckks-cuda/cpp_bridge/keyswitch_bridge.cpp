// FFI bridge for the THIRD ported operation (GOD PROMPT "CKKS RUST + CUDA
// PHASE 3"): full Hybrid key switching. Unlike native_root_bridge.cpp /
// rescale_consts_bridge.cpp (one pure function each), key switching needs a
// whole chain: a native (OpenFHE-free) modulus chain + secret/public key +
// KeySwitchConstants (decompose/moddown constants) + eval key (av/bv), THEN
// the device-resident context/scratch (DeviceKSContext/DeviceKSWork) built
// from that. None of this crypto/number-theory machinery is reimplemented
// here -- every call below forwards to the EXISTING, unmodified
// gpufhe:: functions (src/keygen.cpp, src/evalkeygen.cpp,
// src/ks_constants.cpp, src/keyswitch.cpp, src/keyswitch_resident.cpp,
// src/params.cpp), exactly the same recipe already proven to execute by
// tests/test_keyswitch_core_native_diff.cpp and tests/test_ks_resident.cpp
// (confirmed by re-running test_keyswitch_core_native_diff this round
// before writing this file).
//
// Ownership: one opaque KsFixture bundles the KeySwitchConstants (host,
// including the eval key av/bv), the device-resident DeviceKSContext
// (device-resident eval key + cached root tables) and DeviceKSWork (mutable
// per-call scratch -- NOT safe to share across concurrent in-flight calls,
// same as the C++ engine's own documented contract for DeviceKSWork). A
// fixture is built ONCE (gpufhe_ks_fixture_create) and reused for every
// keyswitch call until gpufhe_ks_fixture_destroy; the eval key is never
// rebuilt per call.
//
// Error handling: every exported function wraps its body in try/catch and
// NEVER lets a C++ exception (ks_context_create throws std::invalid_argument
// on a malformed K; native_primes/native_root throw std::runtime_error on
// search exhaustion) propagate across the FFI boundary into Rust, where it
// would be undefined behavior. Failures are reported as a null handle / a
// nonzero return code plus a bounded, caller-provided error-message buffer.
// A failed gpufhe_ks_fixture_create cleans up any context/work it had
// already constructed before the failure, in reverse order.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>

// Forward-declared, not redeclared in any header -- same pattern as
// native_root_bridge.cpp for gpufhe::native_root (src/params.cpp defines
// both, deliberately header-free; existing test files forward-declare them
// the same way).
namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
}

namespace {

uint64_t mulmod_bridge(uint64_t a, uint64_t b, uint64_t q) {
    return (uint64_t)(((unsigned __int128)a * b) % q);
}

void set_err(char* err_buf, size_t err_buf_len, const char* msg) {
    if (!err_buf || err_buf_len == 0) return;
    size_t len = std::strlen(msg);
    if (len >= err_buf_len) len = err_buf_len - 1;
    std::memcpy(err_buf, msg, len);
    err_buf[len] = '\0';
}

// Bundles everything one key-switch fixture owns. See file header comment.
// secretKey/pubKeyA/pubKeyB (GOD PROMPT "CKKS PHASE 5", new): the first
// K.sizeQl*n elements of the keypair gpufhe_ks_fixture_create generates
// internally via keygen_host, retained ONLY so a caller can perform a REAL
// encrypt/decrypt round trip under the EXACT SAME key the eval key was
// built for (see gpufhe_ks_fixture_secret_key's own doc comment below for
// why this is test-only, not a production API).
struct KsFixture {
    gpufhe::KeySwitchConstants K;
    gpufhe::DeviceKSContext C;
    gpufhe::DeviceKSWork W;
    std::vector<uint64_t> secretKey, pubKeyA, pubKeyB;
    bool context_created = false;
    bool work_created = false;
};

} // namespace

extern "C" {

// Builds a fully native (OpenFHE-free) key-switch fixture: an NTT-friendly
// modulus chain of size_q Q-towers (first 60-bit, remainder 50-bit, matching
// tests/test_keyswitch_core_native_diff.cpp's own convention) plus size_p
// P-towers, a native secret/public key pair (gpufhe::keygen_host), the
// decompose/moddown constant families (gpufhe::compute_keyswitch_constants),
// the relin-style eval key (gpufhe::evalkeygen_host, "sOld = s^2" per its own
// source comment), and the device-resident context + scratch
// (gpufhe::ks_context_create / gpufhe::ks_work_create). Returns an opaque
// handle, or nullptr with err_buf filled on any failure (parameter
// validation is the CALLER's responsibility -- e.g. num_part==0 would divide
// by zero inside compute_keyswitch_constants; this function does not
// second-guess its arguments beyond catching whatever exception results).
void* gpufhe_ks_fixture_create(uint32_t n, uint32_t size_q, uint32_t size_p,
                                uint32_t num_part, uint64_t seed,
                                char* err_buf, size_t err_buf_len)
{
    KsFixture* fx = nullptr;
    try {
        fx = new KsFixture();
        gpufhe::KeySwitchConstants& K = fx->K;

        std::vector<uint64_t> mod, modP;
        gpufhe::native_primes(mod, 1, 60, n, {});
        {
            std::vector<uint64_t> mids;
            gpufhe::native_primes(mids, size_q - 1, 50, n, mod);
            for (auto m : mids) mod.push_back(m);
        }
        gpufhe::native_primes(modP, size_p, 60, n, mod);

        std::vector<uint64_t> modQP = mod;
        for (auto p : modP) modQP.push_back(p);
        std::vector<uint64_t> root(size_q), rootQP;
        for (uint32_t i = 0; i < size_q; ++i) root[i] = gpufhe::native_root(n, mod[i]);
        rootQP = root;
        for (auto p : modP) rootQP.push_back(gpufhe::native_root(n, p));

        auto KPqp = gpufhe::keygen_host(n, modQP, rootQP, /*ns=*/1, /*sigma=*/3.19, seed);
        // Retain the Q-restricted (first size_q*n elements) prefix of the
        // keypair -- see KsFixture's own doc comment. KPqp itself (the
        // full QP-sized vectors) still goes out of scope normally; only
        // this narrower copy survives.
        fx->secretKey.assign(KPqp.s.begin(), KPqp.s.begin() + (size_t)size_q * n);
        fx->pubKeyA.assign(KPqp.pkA.begin(), KPqp.pkA.begin() + (size_t)size_q * n);
        fx->pubKeyB.assign(KPqp.pkB.begin(), KPqp.pkB.begin() + (size_t)size_q * n);

        std::vector<uint64_t> PModq_QP(size_q + size_p);
        for (uint32_t t = 0; t < size_q + size_p; ++t) {
            uint64_t q = modQP[t], P = 1 % q;
            for (uint32_t j = 0; j < size_p; ++j) P = mulmod_bridge(P, modQP[size_q + j] % q, q);
            PModq_QP[t] = P;
        }

        K.n = n;
        gpufhe::compute_keyswitch_constants(K, mod, modP, num_part);
        for (uint32_t i = 0; i < size_q; ++i) { K.rootModList.push_back(mod[i]); K.rootValList.push_back(root[i]); }
        for (uint32_t j = 0; j < size_p; ++j) { K.rootModList.push_back(modQP[size_q + j]); K.rootValList.push_back(rootQP[size_q + j]); }

        gpufhe::evalkeygen_host(K, KPqp.s, KPqp.pkA, KPqp.pkB, PModq_QP, modQP, rootQP,
                                /*ns=*/1, /*sigma=*/3.19, seed + 10007);

        fx->C = gpufhe::ks_context_create(K);
        fx->context_created = true;
        fx->W = gpufhe::ks_work_create(fx->C);
        fx->work_created = true;

        cudaError_t cerr = cudaGetLastError();
        if (cerr != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(cerr));
        }
        return fx;
    } catch (const std::exception& e) {
        set_err(err_buf, err_buf_len, e.what());
    } catch (...) {
        set_err(err_buf, err_buf_len, "unknown C++ exception in gpufhe_ks_fixture_create");
    }
    if (fx) {
        if (fx->work_created) gpufhe::ks_work_destroy(fx->W);
        if (fx->context_created) gpufhe::ks_context_destroy(fx->C);
        delete fx;
    }
    return nullptr;
}

void gpufhe_ks_fixture_destroy(void* handle) {
    if (!handle) return;
    KsFixture* fx = static_cast<KsFixture*>(handle);
    gpufhe::ks_work_destroy(fx->W);
    gpufhe::ks_context_destroy(fx->C);
    delete fx;
}

uint32_t gpufhe_ks_fixture_size_ql(void* handle)   { return static_cast<KsFixture*>(handle)->K.sizeQl; }
uint32_t gpufhe_ks_fixture_size_p(void* handle)    { return static_cast<KsFixture*>(handle)->K.sizeP; }
uint32_t gpufhe_ks_fixture_n(void* handle)         { return static_cast<KsFixture*>(handle)->K.n; }
uint32_t gpufhe_ks_fixture_num_part(void* handle)  { return static_cast<KsFixture*>(handle)->K.numPart; }

// Per-tower Q modulus, so a caller can generate VALID in-range residues
// without reimplementing this fixture's own native_primes/native_root
// derivation.
uint64_t gpufhe_ks_fixture_modulus_q(void* handle, uint32_t tower) {
    return static_cast<KsFixture*>(handle)->K.qMod[tower];
}

// GOD PROMPT "CKKS PHASE 5", new: exposes the Q-restricted secret/public
// key this fixture's eval key was built under, so a caller can perform a
// REAL gpufhe::encrypt_host/decrypt_host round trip under the EXACT SAME
// key -- without this, "EvalMult produces a correct result" could only be
// checked at the raw RNS level (entries #35-#37), never as actual CKKS
// ciphertext arithmetic with real encryption noise.
//
// TEST-ONLY, explicitly not a production API: a real deployment would
// never hand its own secret key back to a caller. This exists because
// `gpufhe_ks_fixture_create`'s `keygen_host` call is the ONLY place this
// fixture's secret is ever generated (see KsFixture's own doc comment) --
// there is no separate "regenerate the same key" path, and fabricating a
// different key here would silently test the wrong thing. `out_len` must
// equal `size_ql*n` exactly (returns false, writes nothing, otherwise);
// each copies `size_ql*n` u64 values, never more. Never logs or prints
// the copied values itself -- what the CALLER does with them afterward is
// the caller's own responsibility, same as any other secret material.
bool gpufhe_ks_fixture_secret_key(void* handle, uint64_t* out, size_t out_len) {
    if (!handle) return false;
    KsFixture* fx = static_cast<KsFixture*>(handle);
    if (out_len != fx->secretKey.size()) return false;
    std::memcpy(out, fx->secretKey.data(), out_len * 8);
    return true;
}
bool gpufhe_ks_fixture_public_key_a(void* handle, uint64_t* out, size_t out_len) {
    if (!handle) return false;
    KsFixture* fx = static_cast<KsFixture*>(handle);
    if (out_len != fx->pubKeyA.size()) return false;
    std::memcpy(out, fx->pubKeyA.data(), out_len * 8);
    return true;
}
bool gpufhe_ks_fixture_public_key_b(void* handle, uint64_t* out, size_t out_len) {
    if (!handle) return false;
    KsFixture* fx = static_cast<KsFixture*>(handle);
    if (out_len != fx->pubKeyB.size()) return false;
    std::memcpy(out, fx->pubKeyB.data(), out_len * 8);
    return true;
}

// Runs gpufhe::keyswitch_resident (the device-resident path this operation
// ports) on preallocated device buffers: d_a (sizeQl*n, eval domain) ->
// d_ba0/d_ba1 (sizeQl*n each, caller-preallocated). Synchronizes the default
// stream before returning so the caller can safely download the result
// immediately after. Returns 0 on success, nonzero + err_buf on failure.
int gpufhe_ks_fixture_run_resident(void* handle, const uint64_t* d_a,
                                    uint64_t* d_ba0, uint64_t* d_ba1,
                                    char* err_buf, size_t err_buf_len)
{
    if (!handle) { set_err(err_buf, err_buf_len, "null fixture handle"); return 1; }
    KsFixture* fx = static_cast<KsFixture*>(handle);
    try {
        gpufhe::keyswitch_resident(d_a, d_ba0, d_ba1, fx->C, fx->W, 0);
        cudaError_t cerr = cudaDeviceSynchronize();
        if (cerr != cudaSuccess) { set_err(err_buf, err_buf_len, cudaGetErrorString(cerr)); return 2; }
        return 0;
    } catch (const std::exception& e) {
        set_err(err_buf, err_buf_len, e.what());
        return 3;
    } catch (...) {
        set_err(err_buf, err_buf_len, "unknown C++ exception in gpufhe_ks_fixture_run_resident");
        return 4;
    }
}

// Runs gpufhe::keyswitch_core_resident (the host-orchestrated reference
// implementation, independently proven bit-exact against OpenFHE's own
// KeySwitchCore in tests/test_keyswitch.cpp, at a DIFFERENT, OpenFHE-sourced
// configuration) on the SAME KeySwitchConstants this fixture holds, entirely
// on the host. a/ba0_out/ba1_out are plain host arrays, sizeQl*n each. This
// exists ONLY as a correctness oracle for the resident path above -- it is
// not the fast path and is not meant to be benchmarked as one.
int gpufhe_ks_fixture_run_reference(void* handle, const uint64_t* a, size_t a_len,
                                     uint64_t* ba0_out, uint64_t* ba1_out, size_t out_len,
                                     char* err_buf, size_t err_buf_len)
{
    if (!handle) { set_err(err_buf, err_buf_len, "null fixture handle"); return 1; }
    KsFixture* fx = static_cast<KsFixture*>(handle);
    try {
        std::vector<uint64_t> aTowers(a, a + a_len);
        gpufhe::KeySwitchResult R = gpufhe::keyswitch_core_resident(aTowers, fx->K);
        if (R.ba0.size() != out_len || R.ba1.size() != out_len) {
            set_err(err_buf, err_buf_len, "reference output length mismatch");
            return 5;
        }
        std::memcpy(ba0_out, R.ba0.data(), out_len * 8);
        std::memcpy(ba1_out, R.ba1.data(), out_len * 8);
        return 0;
    } catch (const std::exception& e) {
        set_err(err_buf, err_buf_len, e.what());
        return 3;
    } catch (...) {
        set_err(err_buf, err_buf_len, "unknown C++ exception in gpufhe_ks_fixture_run_reference");
        return 4;
    }
}

} // extern "C"
