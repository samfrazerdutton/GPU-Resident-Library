#!/usr/bin/env python3
"""GOD PROMPT "CKKS PHASE 10": independent cross-check of
include/ks_params_validate.h's check_hybrid_keyswitch_params against
Python's native arbitrary-precision integers -- a completely different
implementation stack from the C++ BigUint arithmetic it verifies, so
that implementation is never its own oracle.

Usage: run tests/test_ks_params_validate.cpp first (it prints each
case's modQ/modP/numPart and the C++-computed P/Q_part_max decimal
strings); this script recomputes the SAME products independently
(using the IDENTICAL partition formula: alpha=ceil(sizeQ/numPart),
startIdx=alpha*part, sizePart=min(alpha, sizeQ-startIdx)) and reports
whether the decimal strings match byte-for-byte.

This file is also directly runnable on its own with the hardcoded cases
below (captured from an actual run of test_ks_params_validate.cpp), so
it does not depend on re-running the C++ binary to serve as a record.
"""
import math

CASES = [
    # (modQ, modP, numPart, cpp_P_decimal, cpp_qpartmax_decimal, cpp_ok)
    (
        [1152921504606830593, 1125899906826241, 1125899906820097, 1125899906738177,
         1125899906732033, 1125899906701313, 1125899906676737, 1125899906629633],
        [1152921504606791681, 1152921504606748673],
        1,
        "1329227995784738786466542463234189313",
        "2644223873342863112522324824238330471874439364081169710002229274389675398133207505628559162274433186183309550006330981609473",
        False,
    ),
    (
        [1152921504606830593, 1125899906826241, 1125899906820097, 1125899906738177,
         1125899906732033, 1125899906701313, 1125899906676737, 1125899906629633],
        [1152921504606791681, 1152921504606748673],
        2,
        "1329227995784738786466542463234189313",
        "1645504557111666169338930791607747679834627529780037133312823297",
        False,
    ),
    (
        [1152921504606830593, 1125899906826241, 1125899906820097, 1125899906738177,
         1125899906732033, 1125899906701313, 1125899906676737, 1125899906629633],
        [1152921504606791681, 1152921504606748673],
        4,
        "1329227995784738786466542463234189313",
        "1298074214614800148504476573990913",
        True,
    ),
    (
        [1152921504606830593, 1125899906826241, 1125899906820097, 1125899906738177,
         1125899906732033],
        [1152921504606791681, 1152921504606748673, 1152921504606683137],
        2,
        "1532495540865466912211049086090921374103311890062714881",
        "1461501637280374082592929972623646716454403778561",
        True,
    ),
]


def partition(size_q: int, num_part: int):
    alpha = math.ceil(size_q / num_part)
    parts = []
    for p in range(num_part):
        start = alpha * p
        size = min(alpha, max(0, size_q - start))
        parts.append((start, size))
    return parts


def product(mods):
    r = 1
    for m in mods:
        r *= m
    return r


def main():
    failures = 0
    for mod_q, mod_p, num_part, cpp_p, cpp_qmax, cpp_ok in CASES:
        p_exact = product(mod_p)
        parts = partition(len(mod_q), num_part)
        q_part_products = [product(mod_q[s:s + sz]) for s, sz in parts]
        q_part_max = max(q_part_products) if q_part_products else 1
        ok = p_exact >= q_part_max

        label = f"sizeQ={len(mod_q)} sizeP={len(mod_p)} numPart={num_part}"
        print(label)
        print(f"  python P          = {p_exact}")
        print(f"  cpp    P          = {cpp_p}")
        print(f"  python Q_part_max = {q_part_max}")
        print(f"  cpp    Q_part_max = {cpp_qmax}")
        print(f"  python ok={ok} cpp ok={cpp_ok}")

        agree = (str(p_exact) == cpp_p) and (str(q_part_max) == cpp_qmax) and (ok == cpp_ok)
        if not agree:
            print("  [MISMATCH] independent Python computation disagrees with the C++ validator")
            failures += 1
        else:
            print("  [MATCH]")
        print()

    print("[OVERALL PASS]" if failures == 0 else "[OVERALL FAIL]")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
