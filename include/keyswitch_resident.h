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
};

// Per-stream scratch. One per concurrent keyswitch-in-flight.
struct DeviceKSWork {
    uint64_t *d_part=nullptr;      // maxSizePart * n
    uint64_t *d_compl=nullptr;     // sizeQlP * n (upper bound on sizeCompl)
    uint64_t *d_res0=nullptr, *d_res1=nullptr;   // sizeQlP * n
    uint64_t *d_pwork=nullptr;     // sizeP * n
    uint64_t *d_qsw=nullptr;       // sizeQl * n
};

DeviceKSContext ks_context_create(const KeySwitchConstants& K);
DeviceKSWork    ks_work_create(const DeviceKSContext& C);
void ks_context_destroy(DeviceKSContext& C);
void ks_work_destroy(DeviceKSWork& W);

// Fully resident Hybrid keyswitch: d_a (sizeQl*n eval, DEVICE) -> d_ba0/d_ba1
// (sizeQl*n, DEVICE, preallocated by caller). Pure kernel launches + DtoD
// copies on `s`; no malloc, no host transfer. Caller syncs the stream.
void keyswitch_resident(const uint64_t* d_a, uint64_t* d_ba0, uint64_t* d_ba1,
                        const DeviceKSContext& C, DeviceKSWork& W, cudaStream_t s);

} // namespace gpufhe
