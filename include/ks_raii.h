#pragma once
// Thin RAII wrappers around DeviceKSContext/DeviceKSWork's existing C API
// (ks_context_create/ks_context_destroy, ks_work_create/ks_work_destroy).
//
// Motivation (GOD PROMPT "CORRECTNESS FIRST, RESOURCE-SAFE ARCHITECTURE"
// round): this repository has already hit one real ownership bug --
// ks_context_destroy forgot to free C.d_keyRow, a raw pointer the context
// owns (found and fixed in an earlier round; see research/EXPERIMENT_LEDGER.md).
// DeviceKSContext/DeviceKSWork use manual cudaMalloc/cudaFree pairs the
// caller must get right at every call site, while include/device_ciphertext.h's
// DeviceCiphertext already demonstrates this codebase's own RAII pattern
// (ctor/dtor, move-only, deleted copy) for a different type. These wrappers
// bring the SAME pattern to DeviceKSContext/DeviceKSWork, delegating entirely
// to the proven C functions -- no cryptographic arithmetic is touched here.
//
// Resource audit this round (full detail in research/EXPERIMENT_LEDGER.md):
//   DeviceKSContext fields and who owns them:
//     - d_av, d_bv (per-part): OWNED (fresh cudaMalloc per ks_context_create
//       call); freed by ks_context_destroy.
//     - d_keyRow: OWNED (fresh cudaMalloc per call); freed by
//       ks_context_destroy (fixed this session -- was previously leaked).
//     - d_fr, d_fp, d_ir, d_ip (per modulus) and d_qhi/d_qhip/d_srcMod/
//       d_qmp/d_complMod/d_mlo/d_mhi/d_pHatInv/d_pHatInvPrec/d_pMod/
//       d_pHatModq/d_qMod/d_mdMuLo/d_mdMuHi/d_modsQlP: BORROWED from two
//       process-lifetime static caches (g_rtab keyed by (n,q); upc keyed by
//       a shape tuple) inside src/keyswitch_resident.cpp -- ks_context_destroy
//       must NOT free these (and does not), since they are shared across
//       every context built with the same (n,q)/shape.
//     - All other fields (modList, complModHost, sizePart, sizeCompl,
//       startIdx, pInvModqHost, qModHost, pModHost, ninv, ninv_p) are plain
//       host std::vector/scalar data -- self-managing, no ownership concern.
//   DeviceKSWork fields: ALL 12 pointer fields (d_part, d_compl, d_res0,
//     d_res1, d_pwork, d_qsw, d_digPtrs, d_rootsA, d_rootsB, d_towerMods,
//     d_nInv, d_nInvPrecon) are OWNED (fresh cudaMalloc per ks_work_create
//     call) and freed 1:1 by ks_work_destroy -- confirmed complete, no leak
//     found here.
//
// Known limitations NOT solved by this wrapper (documented, not hidden):
//   - Neither ks_context_create nor ks_work_create reports allocation
//     failures today (no exceptions, no error return; a failed cudaMalloc
//     partway through the per-part loop in ks_context_create would leave a
//     PARTIALLY populated context with no signal to the caller). This
//     wrapper does not change that -- it cannot add error detection the
//     underlying C API does not expose. A real fix would need
//     cudaGetLastError() checks added inside ks_context_create/
//     ks_work_create themselves, which is a separate, larger change than a
//     thin ownership wrapper, and is out of scope here.
//   - cudaFree's return value is not checked by ks_context_destroy/
//     ks_work_destroy; this wrapper's destructor does not change that
//     either (destructors must not throw, and a non-fatal log-on-failure
//     could be added later without touching ownership semantics).
#include "keyswitch_resident.h"
#include "keyswitch.h"
#include <cuda_runtime.h>
#include <utility>

namespace gpufhe {

// Move-only owner of a DeviceKSContext. Construction calls ks_context_create;
// destruction calls ks_context_destroy exactly once. Default-constructed
// (or moved-from) instances are "invalid" and safe to destroy (no-op).
class ScopedKSContext {
public:
    ScopedKSContext() noexcept = default;
    explicit ScopedKSContext(const KeySwitchConstants& K, cudaStream_t s = 0)
        : ctx_(ks_context_create(K, s)), valid_(true) {}

    ScopedKSContext(const ScopedKSContext&) = delete;
    ScopedKSContext& operator=(const ScopedKSContext&) = delete;

    ScopedKSContext(ScopedKSContext&& o) noexcept
        : ctx_(std::move(o.ctx_)), valid_(o.valid_) { o.valid_ = false; }
    ScopedKSContext& operator=(ScopedKSContext&& o) noexcept {
        if (this != &o) { reset(); ctx_ = std::move(o.ctx_); valid_ = o.valid_; o.valid_ = false; }
        return *this;
    }

    ~ScopedKSContext() { reset(); }

    // Explicitly release the owned context now (idempotent -- safe to call
    // more than once, or on an already-invalid/moved-from instance).
    void reset() noexcept {
        if (valid_) { ks_context_destroy(ctx_); valid_ = false; }
    }

    bool valid() const noexcept { return valid_; }
    const DeviceKSContext& get() const noexcept { return ctx_; }
    DeviceKSContext& get() noexcept { return ctx_; }
    const DeviceKSContext& operator*() const noexcept { return ctx_; }
    DeviceKSContext& operator*() noexcept { return ctx_; }

private:
    DeviceKSContext ctx_{};
    bool valid_ = false;
};

// Move-only owner of a DeviceKSWork. Construction calls ks_work_create;
// destruction calls ks_work_destroy exactly once.
class ScopedKSWork {
public:
    ScopedKSWork() noexcept = default;
    explicit ScopedKSWork(const DeviceKSContext& C)
        : work_(ks_work_create(C)), valid_(true) {}

    ScopedKSWork(const ScopedKSWork&) = delete;
    ScopedKSWork& operator=(const ScopedKSWork&) = delete;

    ScopedKSWork(ScopedKSWork&& o) noexcept
        : work_(std::move(o.work_)), valid_(o.valid_) { o.valid_ = false; }
    ScopedKSWork& operator=(ScopedKSWork&& o) noexcept {
        if (this != &o) { reset(); work_ = std::move(o.work_); valid_ = o.valid_; o.valid_ = false; }
        return *this;
    }

    ~ScopedKSWork() { reset(); }

    void reset() noexcept {
        if (valid_) { ks_work_destroy(work_); valid_ = false; }
    }

    bool valid() const noexcept { return valid_; }
    const DeviceKSWork& get() const noexcept { return work_; }
    DeviceKSWork& get() noexcept { return work_; }
    const DeviceKSWork& operator*() const noexcept { return work_; }
    DeviceKSWork& operator*() noexcept { return work_; }

private:
    DeviceKSWork work_{};
    bool valid_ = false;
};

} // namespace gpufhe
