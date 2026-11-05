// FFI bridge for the FIFTH ported operation (Phase 23: "FULL CKKS CAPABILITY
// COMPLETION" -- rotation selected as the one principal implementation
// target; see research/EXPERIMENT_LEDGER.md and research/MIGRATION_MATRIX.md
// for the full target-selection justification). Homomorphic rotation
// (Galois automorphism sigma_k + key switch) had ZERO Rust/FFI exposure
// before this file -- confirmed by grepping rust/crates/ckks-cuda/ for
// "rotate"/"automorph" before writing any of this.
//
// Architecture: this file does NOT reimplement any cryptographic or
// number-theoretic logic. It composes EXISTING, unmodified gpufhe::
// functions exactly the way tests/test_boot.cpp's own `rotFull`/`mkKidx`
// lambdas already do to build a rotation key, and the way
// src/rotate.cpp's own `rotate_ct_resident` already does to apply one --
// the one genuine addition is caching the DeviceKSContext/DeviceKSWork
// across calls instead of rebuilding them per call, which
// rotate_ct_resident's own source comment flags as a known opportunity
// ("if that dominates, cache contexts across calls next").
//
// Key insight that makes this possible without touching keyswitch_
// resident.cpp or evalkeygen.cpp: evalkeygen_host_sold (src/evalkeygen.cpp)
// already takes an explicit `sOld` parameter -- "generalized: caller
// supplies the switched key" per its own source comment -- so a ROTATION
// eval key (sOld = sigma_k(s), the Galois-automorphism-permuted secret)
// is built by the exact same function evalkeygen_host's relin wrapper
// calls with sOld = s^2, just with a different sOld. No new key-generation
// algorithm, only a different argument to an already-general function.
//
// Rotation amount convention: this bridge takes a SLOT rotation count `r`
// (0 = identity, same slot layout every other operation in this crate
// uses) and derives the Galois element internally as k = 5^r mod 2n -- the
// exact convention tests/test_boot.cpp's own `rotAmt` lambda uses
// (`uint64_t k=1;for(uint32_t t=0;t<r;++t)k=(k*5)%M;`), confirmed by
// reading that file before writing this one. Rust callers never handle
// the Galois element directly.
//
// Ownership: one opaque RotFixture bundles the ROTATION KeySwitchConstants
// (host, including the rotation eval key av/bv -- built from sigma_k(s),
// not s^2), the device-resident DeviceKSContext + DeviceKSWork (built
// ONCE, reused by every run call), and three persistent device scratch
// buffers (d_scratch/d_b0/d_b1, sizeQl*n each, grown once at create time,
// mirroring src/rotate.cpp's own "allocate once, never free" scratch
// doctrine -- its own comment notes that per-call malloc/free of exactly
// this kind of buffer regressed the bootstrap 53->101s).
//
// Error handling: same convention as keyswitch_bridge.cpp -- every
// exported function wraps its body in try/catch, never lets a C++
// exception cross the FFI boundary; failures are a null handle / nonzero
// return code plus a bounded caller-provided error buffer.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include "stage_acc.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>

// Forward-declared, not redeclared in any header -- same situation
// native_root_bridge.cpp/keyswitch_bridge.cpp document for
// native_primes/native_root: src/rotate.cpp defines automorphism_eval_host
// deliberately header-free (only the CUDA-kernel-launcher declarations in
// include/automorph.h and the device-pointer variants in include/keygen.h
// are declared anywhere).
namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void automorphism_eval_host(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t,
                            const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void rotate_ct_host(std::vector<uint64_t>&, std::vector<uint64_t>&, uint32_t,
                    const struct KeySwitchConstants&, uint32_t, uint32_t,
                    const std::vector<uint64_t>&, const std::vector<uint64_t>&);
}

namespace {

uint64_t mulmod_bridge(uint64_t a, uint64_t b, uint64_t q) {
    return (uint64_t)(((unsigned __int128)a * b) % q);
}

// k = 5^r mod 2n -- the CKKS rotation-group generator convention this
// codebase already uses (src/encode.cpp's own slot ordering comment,
// tests/test_boot.cpp's `rotAmt` lambda). r=0 maps to k=1, the identity
// automorphism.
uint64_t galois_element_for_rotation(uint32_t n, uint32_t r) {
    const uint64_t M = 2ull * n;
    uint64_t k = 1;
    for (uint32_t t = 0; t < r; ++t) k = (k * 5) % M;
    return k;
}

void set_err(char* err_buf, size_t err_buf_len, const char* msg) {
    if (!err_buf || err_buf_len == 0) return;
    size_t len = std::strlen(msg);
    if (len >= err_buf_len) len = err_buf_len - 1;
    std::memcpy(err_buf, msg, len);
    err_buf[len] = '\0';
}

// See file header comment for what each field is and why it is cached
// rather than rebuilt per call.
struct RotFixture {
    gpufhe::KeySwitchConstants K;
    gpufhe::DeviceKSContext C;
    gpufhe::DeviceKSWork W;
    std::vector<uint64_t> secretKey, pubKeyA, pubKeyB;
    uint64_t galoisElement = 1;
    uint64_t* d_scratch = nullptr;
    uint64_t* d_b0 = nullptr;
    uint64_t* d_b1 = nullptr;
    bool context_created = false;
    bool work_created = false;

    ~RotFixture() {
        if (d_scratch) cudaFree(d_scratch);
        if (d_b0) cudaFree(d_b0);
        if (d_b1) cudaFree(d_b1);
    }
};

} // namespace

extern "C" {

// Builds a fully native (OpenFHE-free) rotation fixture for slot-rotation
// amount `rot_amount` (0 = identity): the same modulus chain + native
// secret/public key recipe gpufhe_ks_fixture_create uses (so a rotation
// fixture and a relin fixture built with the same n/size_q/size_p/seed
// share the identical base key), PLUS the Galois-permuted secret
// sigma_k(s) (k = 5^rot_amount mod 2n) via gpufhe::automorphism_eval_host
// applied to a copy of the full QP-range secret, used as `sOld` in
// evalkeygen_host_sold to build a ROTATION eval key instead of a relin
// one. Returns an opaque handle, or nullptr with err_buf filled on
// failure.
void* gpufhe_rot_fixture_create(uint32_t n, uint32_t size_q, uint32_t size_p,
                                 uint32_t num_part, uint64_t seed,
                                 uint32_t rot_amount,
                                 char* err_buf, size_t err_buf_len)
{
    RotFixture* fx = nullptr;
    try {
        fx = new RotFixture();
        gpufhe::KeySwitchConstants& K = fx->K;
        fx->galoisElement = galois_element_for_rotation(n, rot_amount);
        const uint64_t k = fx->galoisElement;

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
        fx->secretKey.assign(KPqp.s.begin(), KPqp.s.begin() + (size_t)size_q * n);
        fx->pubKeyA.assign(KPqp.pkA.begin(), KPqp.pkA.begin() + (size_t)size_q * n);
        fx->pubKeyB.assign(KPqp.pkB.begin(), KPqp.pkB.begin() + (size_t)size_q * n);

        // sigma_k(s) over the FULL QP range -- the rotation eval key's
        // "sOld", exactly matching the pattern tests/test_boot.cpp's
        // `rotFull` lambda uses (automorphism_eval_host on a secret copy,
        // then evalkeygen with that as sOld). A fresh copy of KPqp.s is
        // permuted in place; KPqp.s itself is left untouched so the
        // eval-key generation below still uses the real secret.
        std::vector<uint64_t> sSigma = KPqp.s;
        gpufhe::automorphism_eval_host(sSigma, size_q + size_p, n, (uint32_t)k, modQP, rootQP);

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

        // ROTATION eval key: sOld = sigma_k(s), not s^2. evalkeygen_host_sold
        // is the same function evalkeygen_host (the relin wrapper) forwards
        // to -- "generalized: caller supplies the switched key" per its own
        // source comment in src/evalkeygen.cpp.
        gpufhe::evalkeygen_host_sold(K, KPqp.s, sSigma, KPqp.pkA, KPqp.pkB, PModq_QP, modQP, rootQP,
                                    /*ns=*/1, /*sigma=*/3.19, seed + 10007);

        fx->C = gpufhe::ks_context_create(K);
        fx->context_created = true;
        fx->W = gpufhe::ks_work_create(fx->C);
        fx->work_created = true;

        const size_t bytes = (size_t)K.sizeQl * n * 8;
        cudaMalloc(&fx->d_scratch, bytes);
        cudaMalloc(&fx->d_b0, bytes);
        cudaMalloc(&fx->d_b1, bytes);

        cudaError_t cerr = cudaGetLastError();
        if (cerr != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(cerr));
        }
        return fx;
    } catch (const std::exception& e) {
        set_err(err_buf, err_buf_len, e.what());
    } catch (...) {
        set_err(err_buf, err_buf_len, "unknown C++ exception in gpufhe_rot_fixture_create");
    }
    if (fx) {
        if (fx->work_created) gpufhe::ks_work_destroy(fx->W);
        if (fx->context_created) gpufhe::ks_context_destroy(fx->C);
        delete fx;
    }
    return nullptr;
}

void gpufhe_rot_fixture_destroy(void* handle) {
    if (!handle) return;
    RotFixture* fx = static_cast<RotFixture*>(handle);
    gpufhe::ks_work_destroy(fx->W);
    gpufhe::ks_context_destroy(fx->C);
    delete fx;
}

uint32_t gpufhe_rot_fixture_size_ql(void* handle)  { return static_cast<RotFixture*>(handle)->K.sizeQl; }
uint32_t gpufhe_rot_fixture_size_p(void* handle)   { return static_cast<RotFixture*>(handle)->K.sizeP; }
uint32_t gpufhe_rot_fixture_n(void* handle)        { return static_cast<RotFixture*>(handle)->K.n; }
uint32_t gpufhe_rot_fixture_num_part(void* handle) { return static_cast<RotFixture*>(handle)->K.numPart; }
uint64_t gpufhe_rot_fixture_galois_element(void* handle) { return static_cast<RotFixture*>(handle)->galoisElement; }

uint64_t gpufhe_rot_fixture_modulus_q(void* handle, uint32_t tower) {
    return static_cast<RotFixture*>(handle)->K.qMod[tower];
}

// TEST-ONLY, same rationale/caveats as gpufhe_ks_fixture_secret_key in
// cpp_bridge/keyswitch_bridge.cpp (not duplicated here): lets a caller
// perform a real encrypt/decrypt round trip under the exact key this
// fixture's rotation eval key was built for.
bool gpufhe_rot_fixture_secret_key(void* handle, uint64_t* out, size_t out_len) {
    if (!handle) return false;
    RotFixture* fx = static_cast<RotFixture*>(handle);
    if (out_len != fx->secretKey.size()) return false;
    std::memcpy(out, fx->secretKey.data(), out_len * 8);
    return true;
}
bool gpufhe_rot_fixture_public_key_a(void* handle, uint64_t* out, size_t out_len) {
    if (!handle) return false;
    RotFixture* fx = static_cast<RotFixture*>(handle);
    if (out_len != fx->pubKeyA.size()) return false;
    std::memcpy(out, fx->pubKeyA.data(), out_len * 8);
    return true;
}
bool gpufhe_rot_fixture_public_key_b(void* handle, uint64_t* out, size_t out_len) {
    if (!handle) return false;
    RotFixture* fx = static_cast<RotFixture*>(handle);
    if (out_len != fx->pubKeyB.size()) return false;
    std::memcpy(out, fx->pubKeyB.data(), out_len * 8);
    return true;
}

// Steady-state rotation on preallocated device buffers: d_c0/d_c1
// (sizeQl*n each, eval domain) hold the input ciphertext and are
// OVERWRITTEN in place with the rotated result. Internally: eval-domain
// batched-gather automorphism on both (gpufhe::automorphism_eval_fast_
// device, the path src/rotate.cpp's own rotate_ct_resident already
// validated bit-exact against the slower INTT/permute/NTT path at real
// scale -- see tests/test_automorph_eval_fast.cpp), key-switch the
// permuted c1 through this fixture's CACHED DeviceKSContext/DeviceKSWork
// (built once in gpufhe_rot_fixture_create, not rebuilt here), then
// combine (d_c0 += keyswitched.ba0, d_c1 = keyswitched.ba1) via the
// existing LaunchAddInto kernel, exactly matching src/rotate.cpp's own
// rotate_ct_device combine step. Synchronizes the default stream before
// returning, so the caller can safely download the result immediately
// after. Returns 0 on success, nonzero + err_buf on failure.
int gpufhe_rot_fixture_run_resident(void* handle, uint64_t* d_c0, uint64_t* d_c1,
                                    char* err_buf, size_t err_buf_len)
{
    if (!handle) { set_err(err_buf, err_buf_len, "null fixture handle"); return 1; }
    RotFixture* fx = static_cast<RotFixture*>(handle);
    try {
        const uint32_t n = fx->K.n;
        const uint32_t towers = fx->K.sizeQl;

        gpufhe::automorphism_eval_fast_device(d_c0, towers, n, (uint32_t)fx->galoisElement, fx->d_scratch, 0);
        gpufhe::automorphism_eval_fast_device(d_c1, towers, n, (uint32_t)fx->galoisElement, fx->d_scratch, 0);

        gpufhe::keyswitch_resident(d_c1, fx->d_b0, fx->d_b1, fx->C, fx->W, 0);

        for (uint32_t t = 0; t < towers; ++t) {
            LaunchAddInto(d_c0 + (size_t)t * n, fx->d_b0 + (size_t)t * n, fx->K.qMod[t], n, 0);
        }
        cudaMemcpyAsync(d_c1, fx->d_b1, (size_t)towers * n * 8, cudaMemcpyDeviceToDevice, 0);

        cudaError_t cerr = cudaDeviceSynchronize();
        if (cerr != cudaSuccess) { set_err(err_buf, err_buf_len, cudaGetErrorString(cerr)); return 2; }
        return 0;
    } catch (const std::exception& e) {
        set_err(err_buf, err_buf_len, e.what());
        return 3;
    } catch (...) {
        set_err(err_buf, err_buf_len, "unknown C++ exception in gpufhe_rot_fixture_run_resident");
        return 4;
    }
}

// Host-orchestrated reference path: gpufhe::rotate_ct_host, the slower
// coefficient-domain (INTT->permute->NTT) automorphism + keyswitch_core_
// resident recipe, entirely on the host, on the SAME KeySwitchConstants
// this fixture holds. Exists ONLY as this bridge's correctness oracle for
// the resident path above -- mirrors gpufhe_ks_fixture_run_reference's
// role in cpp_bridge/keyswitch_bridge.cpp exactly.
int gpufhe_rot_fixture_run_reference(void* handle, const uint64_t* c0_in, const uint64_t* c1_in,
                                     size_t len, uint64_t* c0_out, uint64_t* c1_out,
                                     char* err_buf, size_t err_buf_len)
{
    if (!handle) { set_err(err_buf, err_buf_len, "null fixture handle"); return 1; }
    RotFixture* fx = static_cast<RotFixture*>(handle);
    try {
        const uint32_t n = fx->K.n;
        const uint32_t towers = fx->K.sizeQl;
        const size_t expected = (size_t)towers * n;
        if (len != expected) { set_err(err_buf, err_buf_len, "input length mismatch"); return 5; }

        std::vector<uint64_t> c0(c0_in, c0_in + len), c1(c1_in, c1_in + len);
        std::vector<uint64_t> modQ(fx->K.qMod.begin(), fx->K.qMod.begin() + towers);
        std::vector<uint64_t> rootQ(fx->K.rootValList.begin(), fx->K.rootValList.begin() + towers);

        gpufhe::rotate_ct_host(c0, c1, (uint32_t)fx->galoisElement, fx->K, towers, n, modQ, rootQ);

        std::memcpy(c0_out, c0.data(), len * 8);
        std::memcpy(c1_out, c1.data(), len * 8);
        return 0;
    } catch (const std::exception& e) {
        set_err(err_buf, err_buf_len, e.what());
        return 3;
    } catch (...) {
        set_err(err_buf, err_buf_len, "unknown C++ exception in gpufhe_rot_fixture_run_reference");
        return 4;
    }
}

} // extern "C"
