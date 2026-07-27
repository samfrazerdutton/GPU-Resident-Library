#pragma once
#include <cstdint>
#include <vector>

namespace gpufhe {

// All constants + eval-key data the Hybrid keyswitch core needs, as plain
// arrays. A test-side extractor pulls these from OpenFHE (borrowed-key stage);
// stage 6 (native keygen) will produce them without OpenFHE. This struct is
// the seam: keyswitch_core_resident is OpenFHE-free and only sees this.
struct KeySwitchConstants {
    uint32_t n = 0;
    uint32_t sizeQl = 0;      // current Q towers
    uint32_t sizeP = 0;       // P (auxiliary) towers
    uint32_t numPart = 0;     // digit groups
    uint32_t alpha = 0;       // towers per part
    uint32_t fullQ = 0;       // full Q size (for eval-key idx = i>=sizeQl? i+delta : i)

    std::vector<uint64_t> qMod;      // sizeQl : modulus per Q tower
    std::vector<uint64_t> pMod;      // sizeP  : modulus per P tower

    // Decompose (per part): ApproxSwitchCRTBasis partQ -> complement.
    // Flattened per part; part p has sizePart[p] source towers, sizeCompl[p] targets.
    std::vector<uint32_t> sizePart;              // numPart
    std::vector<uint32_t> sizeCompl;             // numPart
    std::vector<uint32_t> startIdx;              // numPart : alpha*p
    std::vector<std::vector<uint64_t>> partQHatInv;      // [part][sizePart]
    std::vector<std::vector<uint64_t>> partQHatInvPrec;  // [part][sizePart]
    std::vector<std::vector<uint64_t>> partSrcMod;       // [part][sizePart] : partQ moduli
    std::vector<std::vector<uint64_t>> partQHatModp;     // [part][sizePart*sizeCompl] flat
    std::vector<std::vector<uint64_t>> partComplMod;     // [part][sizeCompl] : target moduli
    std::vector<std::vector<uint64_t>> partBMuLo;        // [part][sizeCompl]
    std::vector<std::vector<uint64_t>> partBMuHi;        // [part][sizeCompl]

    // Eval key: av[part], bv[part] over full QP, flattened [part][sizeQlP * n].
    // sizeQlP = sizeQl + sizeP. Indexed at idx=(i>=sizeQl)?i+delta:i per tower.
    std::vector<std::vector<uint64_t>> av;   // [part][ (sizeQl+sizeP) * n ]... see note
    std::vector<std::vector<uint64_t>> bv;
    uint32_t evalKeyTowers = 0;              // stride: full QP tower count of the key

    // ModDown P->Q (arg-less OpenFHE constants).
    std::vector<uint64_t> pHatInv;       // sizeP
    std::vector<uint64_t> pHatInvPrec;   // sizeP
    std::vector<uint64_t> pHatModq;      // sizeP*sizeQl flat
    std::vector<uint64_t> mdBMuLo;       // sizeQl
    std::vector<uint64_t> mdBMuHi;       // sizeQl
    std::vector<uint64_t> pInvModq;      // sizeQl

    // OpenFHE's exact 2n-th root of unity per modulus (parallel to a modulus
    // list), so the resident INTT/NTT match OpenFHE's transform convention.
    // build_tab in keyswitch.cpp uses these instead of searching for a root.
    std::vector<uint64_t> rootModList;   // moduli
    std::vector<uint64_t> rootValList;   // OpenFHE root for each modulus

};

struct KeySwitchResult {
    std::vector<uint64_t> ba0;   // sizeQl * n, eval form, ModDown'd to Q
    std::vector<uint64_t> ba1;
};

// Full Hybrid KeySwitchCore, host-orchestrated over device kernels.
// aTowers = input poly (cv.back()) as sizeQl*n eval-form host data.
// Root tables are built internally per modulus (cached).
KeySwitchResult keyswitch_core_resident(
    const std::vector<uint64_t>& aTowers, const KeySwitchConstants& K);

} // namespace gpufhe
