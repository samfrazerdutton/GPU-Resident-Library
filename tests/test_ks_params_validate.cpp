// GOD PROMPT "CKKS PHASE 10": standalone correctness test for
// check_hybrid_keyswitch_params (include/ks_params_validate.h), the new
// single authoritative implementation of the Hybrid key-switch's
// P >= Q_part_max requirement, enforced at ks_context_create (src/
// keyswitch_resident.cpp).
//
// Prints each configuration's exact decimal P/Q_part_max and verdict to
// stdout; a companion script (scripts/verify_ks_params_validate.py, run
// separately, see research/EXPERIMENT_LEDGER.md entry #43) independently
// recomputes the SAME products using Python's native arbitrary-precision
// integers -- a completely different implementation stack -- and diffs
// the decimal strings byte-for-byte, so this test's own BigUint
// arithmetic is never its own oracle.
//
// No CUDA/GPU use at all -- check_hybrid_keyswitch_params is pure host
// C++, deliberately so it can be exercised this cheaply.
#include "ks_params_validate.h"
#include <iostream>
#include <vector>
#include <cstdint>

namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
}

int main() {
    using namespace gpufhe;
    const uint32_t n = 1024;
    int failures = 0;

    struct Case { uint32_t sizeQl, sizeP, numPart; bool expectOk; const char* label; };
    // Expected outcomes are from Phase 9's directly-measured results
    // (tests/test_ks_pq_ratio_sweep.cpp, entry #42) -- this test exists to
    // confirm the NEW exact-bigint validator agrees with those
    // already-established, independently-measured verdicts, not to
    // establish them fresh.
    std::vector<Case> cases = {
        {8, 2, 1, false, "sizeQl=8 sizeP=2 numPart=1 (Q_part=410b > P=120b, Phase9 FAIL 49.99%)"},
        {8, 2, 2, false, "sizeQl=8 sizeP=2 numPart=2 (the old, invalid evalmult.rs/keyswitch.rs default)"},
        {8, 2, 3, false, "sizeQl=8 sizeP=2 numPart=3"},
        {8, 2, 4, true,  "sizeQl=8 sizeP=2 numPart=4 (the NEW evalmult.rs default, Phase9 PASS 6e-12%)"},
        {8, 2, 8, true,  "sizeQl=8 sizeP=2 numPart=8"},
        {5, 3, 2, true,  "sizeQl=5 sizeP=3 numPart=2 (OpenFHE's own test_ks_constants.cpp config)"},
        {8, 1, 1, false, "sizeQl=8 sizeP=1 numPart=1"},
    };

    for (auto& c : cases) {
        std::vector<uint64_t> modQ, modP;
        native_primes(modQ, 1, 60, n, {});
        { std::vector<uint64_t> mids; native_primes(mids, c.sizeQl - 1, 50, n, modQ);
          for (auto m : mids) modQ.push_back(m); }
        native_primes(modP, c.sizeP, 60, n, modQ);

        HybridKsValidation v = check_hybrid_keyswitch_params(modQ, modP, c.numPart);

        std::cout << c.label << "\n"
                  << "  P            = " << v.p_decimal << "\n"
                  << "  Q_part_max   = " << v.q_part_max_decimal << " (part " << v.worst_part << ")\n"
                  << "  ok (P>=Qmax) = " << (v.ok ? "true" : "false")
                  << " (expected " << (c.expectOk ? "true" : "false") << ")\n";

        if (v.ok != c.expectOk) {
            std::cout << "  [FAIL] verdict disagrees with Phase 9's directly-measured result\n";
            ++failures;
        } else {
            std::cout << "  [PASS]\n";
        }

        // Dump the raw moduli too, so the companion Python script can
        // recompute P/Q_part_max from the IDENTICAL input and diff the
        // decimal strings -- this is the actual independent check; the
        // expectOk comparison above only re-confirms Phase 9's finding.
        std::cout << "  modQ =";
        for (auto q : modQ) std::cout << " " << q;
        std::cout << "\n  modP =";
        for (auto p : modP) std::cout << " " << p;
        std::cout << "\n  numPart = " << c.numPart << "\n\n";
    }

    std::cout << (failures == 0 ? "[OVERALL PASS]\n" : "[OVERALL FAIL]\n");
    return failures == 0 ? 0 : 1;
}
