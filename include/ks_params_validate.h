#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace gpufhe {

// GOD PROMPT "CKKS PHASE 10": the single authoritative check of the
// Hybrid key-switch's required correctness condition, established
// empirically in Phase 9 (research/EXPERIMENT_LEDGER.md entry #42) and
// derived here precisely: for every key-switch digit/partition p, raising
// that digit from its own Q towers to the complement basis (via
// ApproxSwitchCRTBasis, see kernels/basis_convert.cu) reconstructs the
// digit's true integer value plus an unknown multiple of Q_part[p] (the
// product of partition p's own Q moduli) -- this is the "approximate" in
// "ApproxSwitchCRTBasis": it never performs the final quotient subtraction
// an exact CRT reconstruction would need. That bounded error is divided
// away by P (the product of the auxiliary moduli) in the final ModDown
// combine step (LaunchModDownCombine, kernels/moddown.cu: `(a-b)*PInvModq
// mod q`). For that division to bring the residual error down to
// cryptographically negligible noise rather than a value comparable to
// q_i itself, P must be AT LEAST AS LARGE as the largest digit's own
// modulus product -- i.e. P >= max_p(Q_part[p]). Phase 9's
// tests/test_ks_pq_ratio_sweep.cpp confirmed this empirically and
// precisely: holding sizeP fixed and sweeping only numPart, the relin
// identity's residual flips from ~50% of q (a deficit) to ~1e-11% of q
// (a surplus) exactly at this boundary.
//
// This header exposes that check as EXACT arbitrary-precision integer
// arithmetic -- never a bit-length approximation -- since Q_part can
// exceed 128 bits for realistic configurations (e.g. 4 towers of ~50-60
// bits each), and a bit-length comparison can disagree with the true
// product comparison near the boundary (two numbers with the same bit
// length can differ by orders of magnitude in actual value).
//
// Both overloads use NON-STRICT `P >= Q_part_max` (equality is accepted
// as valid). This is the natural reading of the requirement as stated in
// Phase 9 (entry #42) and of the underlying algebra (the error term
// after ModDown's division is bounded by the SAME small integer k
// regardless of whether P exceeds or exactly equals Q_part_max -- nothing
// in the derivation needs a STRICT surplus). Honesty check: exact
// equality P==Q_part_max was NOT independently empirically verified this
// round, because P and Q_part_max are each products of DISTINCT primes
// drawn from disjoint search spaces (native_primes never lets a Q prime
// equal a P prime), making literal equality of their PRODUCTS a
// measure-zero coincidence this library's prime generation does not
// naturally produce. Phase 9's sweep only bracketed the boundary (a
// -40-bit deficit failed, a +10-bit surplus passed); if a future round
// ever needs the exact boundary's behavior pinned down further, that is
// the next experiment, not assumed here.

struct HybridKsValidation {
    bool ok = false;
    uint32_t worst_part = 0;          // index of the partition achieving Q_part_max
    std::string p_decimal;             // P, exact decimal (moduli are PUBLIC parameters)
    std::string q_part_max_decimal;    // Q_part_max, exact decimal
};

// Primary form: takes the EXACT partition `compute_keyswitch_constants`
// already derived (its own K.startIdx/K.sizePart), so this check can
// never silently drift out of sync with the actual production
// partitioning logic. Prefer this overload whenever a KeySwitchConstants
// is already available (e.g. inside ks_context_create).
HybridKsValidation check_hybrid_keyswitch_params(
    const std::vector<uint64_t>& moduliQ,
    const std::vector<uint64_t>& moduliP,
    const std::vector<uint32_t>& partStartIdx,
    const std::vector<uint32_t>& partSize);

// Convenience form: derives the partition itself, using the IDENTICAL
// formula compute_keyswitch_constants uses (alpha=ceil(sizeQ/numPart);
// per-part startIdx=alpha*part, sizePart=min(alpha, sizeQ-startIdx)) --
// for callers that only have numPart, not a built KeySwitchConstants
// (e.g. a standalone validation entry point before any expensive
// construction work has run).
HybridKsValidation check_hybrid_keyswitch_params(
    const std::vector<uint64_t>& moduliQ,
    const std::vector<uint64_t>& moduliP,
    uint32_t numPart);

} // namespace gpufhe
