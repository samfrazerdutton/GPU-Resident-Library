#!/usr/bin/env python3
"""Standalone tests for hybrid_security.py's classical and quantum-adjusted
estimators. No external test framework in this repo's tools/ -- matches the
existing plain-assert, print-PASS/FAIL style of hybrid_security.py's own
--calibrate mode.
"""
import sys
from hybrid_security import hybrid, hybrid_quantum

failures = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not cond:
        failures.append(name)

print("1. Classical calibration (existing published-result comparison, unchanged)")
for name, n, lq, ref, tol in (("Son-Cheon 2019/1019", 65536, 1240, 113.0, 6.0),
                               ("Cheon+ 2019/1114",    32768,  628, 112.5, 6.0)):
    c, _ = hybrid(n, lq, 64)
    check(f"{name}: model {c:.1f} within {tol} bits of published {ref}",
          abs(c - ref) <= tol, f"model={c:.1f} published={ref} diff={c-ref:+.1f}")
    # Conservative-by-design: the docstring claims the model reports LESS
    # security than published, never more.
    check(f"{name}: model is conservative (<= published)", c <= ref + 0.05,
          f"model={c:.1f} published={ref}")

print("\n2. Quantum-adjusted estimate on the real n=131072/h=64 candidate")
n, logq, h = 131072, 1775, 64
c_classical, _ = hybrid(n, logq, h)
c_q_floor, _ = hybrid_quantum(n, logq, h, credit_mitm_qracm=False)
c_q_qracm, _ = hybrid_quantum(n, logq, h, credit_mitm_qracm=True)
check("classical estimate matches the documented 134-bit figure",
      abs(c_classical - 134) < 1.0, f"got {c_classical:.1f}")
print(f"  classical={c_classical:.1f}  quantum(no-QRACM)={c_q_floor:.1f}"
      f"  quantum(QRACM)={c_q_qracm:.1f}")

print("\n3. Sanity: quantum estimates must never exceed the classical one")
check("no-QRACM quantum bound <= classical", c_q_floor <= c_classical,
      f"{c_q_floor:.1f} vs {c_classical:.1f}")
check("QRACM quantum bound <= classical", c_q_qracm <= c_classical,
      f"{c_q_qracm:.1f} vs {c_classical:.1f}")
print("\n4. Sanity: crediting MITM with a quantum speedup can only cost MORE security")
check("QRACM bound <= no-QRACM bound (crediting more quantum power never helps the defender)",
      c_q_qracm <= c_q_floor, f"{c_q_qracm:.1f} vs {c_q_floor:.1f}")

print("\n5. Limiting case: h=0 (no sparsity at all -- degenerate, but must not crash)")
try:
    c0, _ = hybrid(n, logq, 0)
    q0, _ = hybrid_quantum(n, logq, 0)
    check("h=0 does not crash and returns finite values",
          c0 < 1e9 and q0 < 1e9, f"classical={c0:.1f} quantum={q0:.1f}")
except Exception as e:
    check("h=0 does not crash", False, f"raised {type(e).__name__}: {e}")

print("\n6. Limiting case: small n (sanity -- function should still terminate and return something)")
try:
    cs, _ = hybrid(512, 100, 8)
    qs, _ = hybrid_quantum(512, 100, 8)
    check("small n (512) does not crash", True, f"classical={cs:.1f} quantum={qs:.1f}")
except Exception as e:
    check("small n (512) does not crash", False, f"raised {type(e).__name__}: {e}")

print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILURE(S): ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
