#pragma once
#include "keyswitch_resident.h"
#include <vector>
#include <cstdint>

namespace gpufhe {

// Decompose-sharing for rotation hoisting (see research/NEXT_SESSION.md,
// priority queue item 1). A NEW, separate path -- does not modify or call
// into any function in keyswitch_resident.cpp other than the already-
// exported ks_moddown; ks_accumulate_qp's own logic is untouched.
//
// ks_accumulate_qp's work splits into: steps 1-3 (per-part INTT ->
// ApproxSwitchCRTBasis -> NTT, "decompose") depend ONLY on the input
// ciphertext half d_a and the decompose SHAPE constants (sizePart/
// sizeCompl/startIdx/root tables), which are identical for any
// DeviceKSContext built from the same (n,sizeQl,numPart) modulus chain --
// confirmed by keyswitch_resident.cpp's own `upc` memoization of exactly
// those constants across different keys sharing a chain. Step 4
// (accumulate against a SPECIFIC key's av/bv) is the only part that
// differs per rotation index. HoistedDigits holds the result of steps
// 1-3, computed ONCE, reusable across N different per-index contexts
// that share that chain -- paying the (expensive, numPart-many INTT/
// basis-convert/NTT) decompose cost once instead of N times.
struct HoistedDigits {
    uint32_t numPart=0, n=0, sizeQlP=0;
    std::vector<uint64_t*> d_compl;   // owned, one per part, sizeQlP*n each (upper bound)
};

HoistedDigits hoist_digits_create(const DeviceKSContext& C);
void hoist_digits_destroy(HoistedDigits& D);

// Steps 1-3 only: decompose d_a (sizeQl*n, eval, device) into D. Does not
// touch any DeviceKSWork -- call ks_accumulate_from_digits for step 4.
void ks_decompose_only(const uint64_t* d_a, const DeviceKSContext& C,
                       HoistedDigits& D, cudaStream_t s);

// Step 4 only: accumulate W.d_res0/d_res1 from a PREVIOUSLY computed
// HoistedDigits (from ks_decompose_only, possibly against a different
// context sharing the same chain) against THIS context's av/bv/keyRow.
// Resets W.d_res0/d_res1 to zero first, matching ks_accumulate_qp.
// Caller is responsible for ensuring D was decomposed against a context
// sharing the same (n,sizeQl,numPart,moduli) chain as C -- undefined
// results otherwise (sizes are checked; chain identity is not).
void ks_accumulate_from_digits(const uint64_t* d_a, const HoistedDigits& D,
                               const DeviceKSContext& C, DeviceKSWork& W,
                               cudaStream_t s);

} // namespace gpufhe
