// GOD PROMPT "CKKS PHASE 10": exact arbitrary-precision implementation of
// the Hybrid key-switch's P >= Q_part_max requirement (see
// include/ks_params_validate.h for the full derivation and rationale).
// Deliberately pure host C++ with no CUDA dependency, so it can be
// unit-tested on its own (no GPU needed to check a multiplication and a
// comparison) and called cheaply at context-construction time without
// touching the device at all.
//
// Moduli (both Q and P) are PUBLIC CKKS parameters, not secret key
// material -- rendering their exact products as decimal strings for
// diagnostics is safe and, unlike a bit-length summary, lets a human
// verify the comparison by hand if ever needed.
#include "ks_params_validate.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace gpufhe {
namespace {

// Little-endian base-2^64 arbitrary-precision unsigned integer.
using BigUint = std::vector<uint64_t>;

void trim(BigUint& a) {
    while (a.size() > 1 && a.back() == 0) a.pop_back();
}

// a *= m, exact, grows `a` as needed.
void mul_scalar(BigUint& a, uint64_t m) {
    if (m == 0) { a.assign(1, 0); return; }
    unsigned __int128 carry = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        unsigned __int128 prod = (unsigned __int128)a[i] * m + carry;
        a[i] = (uint64_t)prod;
        carry = prod >> 64;
    }
    while (carry) {
        a.push_back((uint64_t)carry);
        carry >>= 64;
    }
}

// Exact product of a list of 64-bit moduli, as a BigUint. Empty list -> 1
// (the identity, matching compute_keyswitch_constants's own prod_mod
// convention for an empty/degenerate partition).
BigUint product_of(const std::vector<uint64_t>& mods) {
    BigUint r(1, 1);
    for (uint64_t m : mods) mul_scalar(r, m);
    trim(r);
    return r;
}

// -1 if a<b, 0 if a==b, 1 if a>b. Both assumed trimmed (no leading-zero
// limbs beyond a single `0` for the zero value).
int compare_big(const BigUint& a, const BigUint& b) {
    if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
    for (size_t i = a.size(); i-- > 0;) {
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    }
    return 0;
}

// Decimal rendering via repeated division by 10^18 (fits in a uint64
// remainder), most-significant chunk first. O(limbs^2) but these numbers
// are at most a few hundred bits for any realistic CKKS parameter set, so
// this is negligible next to any GPU work the caller will do afterward.
std::string to_decimal(BigUint a) {
    trim(a);
    if (a.size() == 1 && a[0] == 0) return "0";
    const uint64_t CHUNK = 1000000000000000000ULL; // 10^18
    std::vector<uint64_t> chunks; // base-10^18, little-endian
    while (!(a.size() == 1 && a[0] == 0)) {
        unsigned __int128 rem = 0;
        for (size_t i = a.size(); i-- > 0;) {
            unsigned __int128 cur = (rem << 64) | a[i];
            a[i] = (uint64_t)(cur / CHUNK);
            rem = cur % CHUNK;
        }
        trim(a);
        chunks.push_back((uint64_t)rem);
    }
    std::string out = std::to_string(chunks.back());
    for (size_t i = chunks.size() - 1; i-- > 0;) {
        std::string piece = std::to_string(chunks[i]);
        out += std::string(18 - piece.size(), '0') + piece;
    }
    return out;
}

} // namespace

HybridKsValidation check_hybrid_keyswitch_params(
    const std::vector<uint64_t>& moduliQ,
    const std::vector<uint64_t>& moduliP,
    const std::vector<uint32_t>& partStartIdx,
    const std::vector<uint32_t>& partSize)
{
    HybridKsValidation out;
    BigUint P = product_of(moduliP);

    BigUint worst(1, 0);
    uint32_t worstPart = 0;
    bool any = false;
    for (size_t p = 0; p < partStartIdx.size(); ++p) {
        uint32_t start = partStartIdx[p], sz = partSize[p];
        std::vector<uint64_t> partMods;
        for (uint32_t i = 0; i < sz; ++i) {
            if (start + i < moduliQ.size()) partMods.push_back(moduliQ[start + i]);
        }
        BigUint qPart = product_of(partMods);
        if (!any || compare_big(qPart, worst) > 0) {
            worst = qPart;
            worstPart = (uint32_t)p;
            any = true;
        }
    }

    out.ok = compare_big(P, worst) >= 0; // P >= Q_part_max, non-strict
    out.worst_part = worstPart;
    out.p_decimal = to_decimal(P);
    out.q_part_max_decimal = to_decimal(worst);
    return out;
}

HybridKsValidation check_hybrid_keyswitch_params(
    const std::vector<uint64_t>& moduliQ,
    const std::vector<uint64_t>& moduliP,
    uint32_t numPart)
{
    const uint32_t sizeQ = (uint32_t)moduliQ.size();
    // EXACT same derivation as compute_keyswitch_constants (src/ks_constants.cpp):
    // alpha=ceil(sizeQ/numPart); part p owns [alpha*p, alpha*p+sizePart).
    const uint32_t alpha = numPart > 0 ? (uint32_t)std::ceil((double)sizeQ / numPart) : 0;
    std::vector<uint32_t> startIdx(numPart), sizePart(numPart);
    for (uint32_t part = 0; part < numPart; ++part) {
        uint32_t s = alpha * part;
        uint32_t sz = (sizeQ > s + alpha) ? alpha : (sizeQ > s ? sizeQ - s : 0);
        startIdx[part] = s;
        sizePart[part] = sz;
    }
    return check_hybrid_keyswitch_params(moduliQ, moduliP, startIdx, sizePart);
}

} // namespace gpufhe
