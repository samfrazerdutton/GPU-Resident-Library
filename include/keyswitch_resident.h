#pragma once
#include "keyswitch.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace gpufhe {

// Shared read-only device state: constants, eval key, root tables. Built ONCE
// from a KeySwitchConstants; serves any number of streams concurrently.
struct DeviceKSContext {
    uint32_t n=0, sizeQl=0, sizeP=0, sizeQlP=0, numPart=0, fullQ=0;
    // per-modulus root tables, indexed parallel to modList
    std::vector<uint64_t> modList;                 // QP moduli (host)
    std::vector<uint64_t*> d_fr,d_fp,d_ir,d_ip;    // device tables per modulus
    std::vector<uint64_t> ninv, ninv_p;            // host scalars per modulus
    // per-part decompose constants (device)
    std::vector<uint32_t> sizePart, sizeCompl, startIdx;      // host
    std::vector<std::vector<uint64_t>> complModHost;          // host, for NTT launches
    std::vector<uint64_t*> d_qhi,d_qhip,d_srcMod,d_qmp,d_complMod,d_mlo,d_mhi;
    // eval key (device, per part, sizeQlP*n each)
    std::vector<uint64_t*> d_av, d_bv;
    // moddown constants (device)
    uint64_t *d_pHatInv=nullptr,*d_pHatInvPrec=nullptr,*d_pMod=nullptr,
             *d_pHatModq=nullptr,*d_qMod=nullptr,*d_mdMuLo=nullptr,*d_mdMuHi=nullptr;
    std::vector<uint64_t> pInvModqHost, qModHost, pModHost;   // host scalars
    // for the batched fastcore launch: per-QP-tower modulus and eval-key row
    uint64_t* d_modsQlP=nullptr; uint32_t* d_keyRow=nullptr;
};

// Per-stream scratch. One per concurrent keyswitch-in-flight.
struct DeviceKSWork {
    uint64_t *d_part=nullptr;      // maxSizePart * n
    uint64_t *d_compl=nullptr;     // sizeQlP * n (upper bound on sizeCompl)
    uint64_t *d_res0=nullptr, *d_res1=nullptr;   // sizeQlP * n
    uint64_t *d_pwork=nullptr;     // sizeP * n
    uint64_t *d_qsw=nullptr;       // sizeQl * n
    const uint64_t** d_digPtrs=nullptr;   // sizeQlP digit pointers (batched fastcore)
    const uint64_t** d_rootsA=nullptr;   // roots/invroots ptr array, sizeQlP entries (upper bound)
    const uint64_t** d_rootsB=nullptr;   // precon/invprecon ptr array, sizeQlP entries
    uint64_t* d_towerMods=nullptr;       // per-tower modulus, sizeQlP entries
    uint64_t* d_nInv=nullptr;            // per-tower n_inv (INTT only), sizeQlP entries
    uint64_t* d_nInvPrecon=nullptr;      // per-tower n_inv_precon (INTT only), sizeQlP entries
    // GOD PROMPT "CKKS PHASE 40" Track E: ks_moddown's final combine step's
    // per-tower constants (PInvModq[i], shoup_precon(PInvModq[i],q_i), q_i),
    // sizeQl entries -- feeds ONE LaunchModDownCombineBatched call instead
    // of sizeQl separate LaunchModDownCombine launches. Same role-buffer
    // reuse rule as d_rootsA/d_rootsB/d_towerMods above: these values are a
    // pure function of the fixed DeviceKSContext C, never of d_res/d_out,
    // so sharing one buffer across ks_moddown's two call sites (d_ba0,
    // d_ba1) is safe, including under CUDA graph capture.
    uint64_t *d_mdS=nullptr, *d_mdSPrecon=nullptr, *d_mdQ=nullptr;  // sizeQl entries

    // GOD PROMPT "CKKS PHASE 30": host-side staging buffers for the
    // cudaMemcpyAsync(..., HostToDevice, s) calls ks_accumulate_qp/
    // ks_moddown issue into d_digPtrs/d_rootsA/d_rootsB/d_towerMods/d_nInv/
    // d_nInvPrecon above, EVERY call.
    //
    // History: these were ORIGINALLY function-local std::vectors, destroyed
    // when the function returned -- harmless for a direct (non-captured)
    // launch (a pageable-memory H2D cudaMemcpyAsync is synchronously staged
    // by the driver before the call returns), but a cudaMemcpyAsync node
    // added via STREAM CAPTURE stores the HOST POINTER, not a snapshot of
    // its bytes -- the pointer is re-read at every future cudaGraphLaunch,
    // long after the capturing function (and its local vectors) returned.
    // Proven with a minimal, gpufhe-independent reproducer (research/
    // KS_GRAPH_CAPTURE_INVESTIGATION.md).
    //
    // FIRST FIX ATTEMPT (insufficient, kept here as a documented dead end):
    // moving these to a SINGLE shared buffer per field, owned by W, fixed
    // the dangling-pointer/use-after-free half of the problem but exposed a
    // SECOND, deeper bug: ks_accumulate_qp's per-part loop (and, within
    // each part, its two roles -- INTT-the-part-slice and NTT-the-
    // complement) all reused the SAME host buffer, overwriting it with a
    // NEW value for each (part, role) combination. During a captured
    // call, ALL of those host-side overwrites happen ONCE, synchronously,
    // while the graph is being recorded -- they do NOT re-run on replay
    // (only the recorded GPU-side memcpy/kernel nodes do). So by the time
    // the graph is replayed, every one of the 4+ memcpy nodes that shared
    // that one host buffer reads the SAME, single, LAST-WRITTEN value
    // (whichever (part, role) happened to run last during capture) --
    // correct for that one node, wrong for every other. This is why the
    // first fix compiled, ran, reported no CUDA errors, and STILL produced
    // 100% wrong output: a memory-safety bug was fixed, but a distinct
    // VALUE-aliasing bug remained, invisible to compute-sanitizer (every
    // access was in-bounds and initialized -- just not the value the
    // caller intended for that specific node).
    //
    // ACTUAL FIX: give every (part, role) combination ks_accumulate_qp's
    // loop produces its OWN dedicated, persistent host buffer -- indexed
    // by part for the per-part roles, so each part's captured memcpy node
    // keeps reading ITS OWN frozen-correct value on every future replay,
    // never another part's. ks_moddown's two roles (P-part INTT, Ql NTT)
    // are NOT part-indexed but ARE still two DISTINCT roles sharing no
    // buffer with each other; they MAY safely share one buffer EACH across
    // ks_moddown's two call sites per keyswitch_resident (once for d_ba0,
    // once for d_ba1), because both calls compute the IDENTICAL value for
    // a given role every time (purely a function of the fixed context C,
    // never of d_res/d_out) -- re-overwriting a buffer with the SAME value
    // is harmless. The underlying DEVICE buffers (W.d_rootsA etc.) stay
    // shared/reused exactly as before: that reuse was never the problem,
    // since ordinary CUDA stream ordering (preserved identically whether
    // live or replayed) already guarantees each kernel sees whatever its
    // immediately-preceding memcpy just wrote -- only the HOST-side source
    // needed per-(part,role) separation.
    std::vector<std::vector<const uint64_t*>> h_rootsA_part, h_rootsB_part;   // role: INTT the part slice, indexed [part]
    std::vector<std::vector<uint64_t>> h_towerMods_part, h_nInv_part, h_nInvPrecon_part;
    std::vector<std::vector<const uint64_t*>> h_rootsA_compl, h_rootsB_compl; // role: NTT the complement, indexed [part]
    std::vector<std::vector<uint64_t>> h_towerMods_compl;
    std::vector<std::vector<const uint64_t*>> h_digPtrs_part;                 // batched digit-pointer array, indexed [part]
    std::vector<const uint64_t*> h_rootsA_pwork, h_rootsB_pwork;              // ks_moddown role: P-part INTT
    std::vector<uint64_t> h_towerMods_pwork, h_nInv_pwork, h_nInvPrecon_pwork;
    std::vector<const uint64_t*> h_rootsA_qsw, h_rootsB_qsw;                  // ks_moddown role: Ql NTT
    std::vector<uint64_t> h_towerMods_qsw;
    // GOD PROMPT "CKKS PHASE 40" Track E: ks_moddown's batched-combine role;
    // persistent host staging for d_mdS/d_mdSPrecon/d_mdQ above (same CUDA-
    // graph-capture rule as every other host buffer in this struct).
    std::vector<uint64_t> h_mdS, h_mdSPrecon, h_mdQ;
};

DeviceKSContext ks_context_create(const KeySwitchConstants& K, cudaStream_t s=0);
DeviceKSWork    ks_work_create(const DeviceKSContext& C);
void ks_context_destroy(DeviceKSContext& C);
void ks_work_destroy(DeviceKSWork& W);

// The two halves keyswitch_resident (below) composes unchanged. Exposed as
// a seam for correct-order rotation hoisting (see research/NEXT_SESSION.md):
// a caller can run ks_accumulate_qp, modify W.d_res0/d_res1 in place (e.g.
// apply an automorphism -- the result is QP-basis, sizeQlP towers, eval
// domain), then call ks_moddown -- composing to something OTHER than plain
// keyswitch_resident's behavior, while keyswitch_resident itself stays
// byte-for-byte identical to before this seam existed.
void ks_accumulate_qp(const uint64_t* d_a, const DeviceKSContext& C, DeviceKSWork& W, cudaStream_t s);
void ks_moddown(const DeviceKSContext& C, DeviceKSWork& W,
                uint64_t* d_res, uint64_t* d_out, cudaStream_t s);

// Fully resident Hybrid keyswitch: d_a (sizeQl*n eval, DEVICE) -> d_ba0/d_ba1
// (sizeQl*n, DEVICE, preallocated by caller). Pure kernel launches + DtoD
// copies on `s`; no malloc, no host transfer. Caller syncs the stream.
// Resident rescale on a raw component buffer (towers*n, eval): drop the last
// tower, fuse into survivors. s1/s2 = OpenFHE's qlInvModq / QlQlInvModqlDivqlModq
// at the current level. scratch/drop = n-sized device buffers.
void rescale_resident_raw(uint64_t* d_c, uint32_t towers,
                          const DeviceKSContext& C,
                          const std::vector<uint64_t>& s1, const std::vector<uint64_t>& s2,
                          uint64_t* d_scratch, uint64_t* d_drop, cudaStream_t s);

void keyswitch_resident(const uint64_t* d_a, uint64_t* d_ba0, uint64_t* d_ba1,
                        const DeviceKSContext& C, DeviceKSWork& W, cudaStream_t s);

} // namespace gpufhe
