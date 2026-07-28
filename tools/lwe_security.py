#!/usr/bin/env python3
"""Primal-uSVP LWE security estimate (Bai-Galbraith embedding, secret rescaled),
BDGL16 sieve cost model: log2(cost) = 0.292*beta + log2(8d) + 16.4.

VALIDATED against the Homomorphic Encryption Security Standard (2018) 128-bit
classical table for uniform ternary secrets: reproduces it to within 0.2 bits
for n>=4096 (3.6 bits worst case at n=1024).  Run with --validate to re-check.

LIMITATION: models the PRIMAL attack only.  For SPARSE secrets the binding
attack is the hybrid dual/MITM (Cheon et al.), which is NOT implemented here --
sparse numbers from this script are upper bounds, not guarantees.  Use the
SageMath lattice-estimator before calling any sparse parameter set secure.
"""
import math, sys

def delta_bkz(beta):
    return ((math.pi*beta)**(1.0/beta) * beta/(2*math.pi*math.e))**(1.0/(2.0*(beta-1)))

def primal_usvp(n, logq, sigma_e=3.19, sigma_s=None, hw=None):
    if sigma_s is None:
        sigma_s = math.sqrt(hw/n) if hw else math.sqrt(2.0/3.0)
    nu = sigma_e/sigma_s
    for beta in range(50, 4000):
        ld  = math.log2(delta_bkz(beta))
        lhs = math.log2(sigma_e) + 0.5*math.log2(beta)
        for m in range(max(64, n//4), 3*n, max(1, n//128)):
            d = m + n + 1
            if lhs <= (2*beta - d - 1)*ld + (m*logq + n*math.log2(nu))/d:
                return beta, d, 0.292*beta + math.log2(8*d) + 16.4, \
                                0.265*beta + math.log2(8*d) + 16.4
    return None

def validate():
    REF = {1024:27, 2048:54, 4096:109, 8192:218, 16384:438, 32768:881}
    print(f"{'n':>7} {'logq':>6} {'beta':>6} {'bits':>7} {'err':>6}")
    worst = 0.0
    for n, lq in sorted(REF.items()):
        b, d, c, _ = primal_usvp(n, lq)
        print(f"{n:7d} {lq:6d} {b:6d} {c:7.1f} {c-128:+6.1f}")
        worst = max(worst, abs(c-128))
    print(f"worst deviation from 128 bits: {worst:.1f}")
    return worst < 5.0

if __name__ == "__main__":
    if "--validate" in sys.argv:
        sys.exit(0 if validate() else 1)
    n    = int(sys.argv[1]); logqp = int(sys.argv[2])
    hw   = int(sys.argv[3]) if len(sys.argv) > 3 else None
    b, d, c, q = primal_usvp(n, logqp, hw=hw)
    kind = f"sparse h={hw}" if hw else "uniform ternary"
    print(f"n={n} log2(QP)={logqp} {kind}: beta={b} -> {c:.0f} classical / {q:.0f} quantum bits")
    if hw: print("  NOTE: primal only; hybrid MITM not modelled -- upper bound.")
