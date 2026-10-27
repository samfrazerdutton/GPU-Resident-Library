#!/usr/bin/env python3
"""Hybrid dual + Odlyzko-MITM attack estimate for SPARSE ternary secret LWE.

The primal estimate in lwe_security.py barely notices sparsity; for sparse
secrets THIS is the binding attack.  Structure: dual attack on (n-k) dimensions
plus a meet-in-the-middle search over a k-dimensional guessing window.  The
attacker may bet the window holds only w <= E[w] nonzeros -- cheaper to
enumerate, lower success probability -- but a failed bet means redoing the
WHOLE attack including lattice reduction, so 1/P multiplies (BKZ + sqrt(N)).

CALIBRATED against published results (conservative: reports LESS security):
  Son-Cheon 2019/1019   n=65536 logq=1240 h=64 -> published 113.0, model 109.8
  Cheon+   2019/1114    n=32768 logq= 628 h=64 -> published 112.5, model 106.9
"""
import math, sys
from math import lgamma

def log2C(n, k):
    if k < 0 or k > n: return -1e9
    return (lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)) / math.log(2)

def delta_bkz(b):
    return ((math.pi*b)**(1.0/b) * b/(2*math.pi*math.e))**(1.0/(2.0*(b-1)))

def hybrid(n, logq, h, sigma=3.2):
    best = (1e9, None)
    for k in range(n//64, n, max(1, n//80)):
        npr = n - k
        if npr < 64: break
        Ew = h*k/n
        opts = []
        for w in range(0, int(Ew)+3):
            lN = max(log2C(k,j)+j for j in range(0, w+1))
            ps = [log2C(k,j)+log2C(n-k,h-j)-log2C(n,h) for j in range(0, w+1)]
            mx = max(ps); lp = mx + math.log2(sum(2**(p-mx) for p in ps))
            opts.append((-lp, 0.5*lN))
        for b in range(60, 1400, 5):
            ld = math.log2(delta_bkz(b))
            for m in range(npr//2, 3*npr, max(1, npr//20)):
                d = m + npr
                tau = d*ld + (npr*logq)/d + math.log2(sigma) - logq
                if tau > 0: continue
                lR = 4*(math.pi**2)*(2**(2*tau))/math.log(2)
                if lR > 200: continue
                bk = 0.292*b + math.log2(8*d) + 16.4
                tot = min(lR + ip + max(bk, ln) for ip, ln in opts)
                if tot < best[0]: best = (tot, (k, b))
    return best

def hybrid_quantum(n, logq, h, sigma=3.2, credit_mitm_qracm=False):
    """Quantum-adjusted variant of hybrid(). Two DIFFERENT attack phases,
    two DIFFERENT and NOT-interchangeable quantum treatments -- see the
    derivation notes in research/SECURITY_ASSESSMENT.md before trusting
    either number blindly.

    BKZ/lattice-reduction half (`bk`): substitutes the quantum sieving
    exponent 0.265*beta (vs. classical 0.292*beta, BDGL16) for the SAME
    SVP-solving subroutine lwe_security.py already quantum-adjusts this
    way -- this is the standard CORE-SVP methodology (used identically by,
    e.g., Kyber/Dilithium's own specifications) and is NOT a generic
    "divide by something" substitution: it is literature-established for
    exactly this subroutine, reused unchanged whether the subroutine sits
    inside a primal, dual, or hybrid attack.

    MITM/guessing-window half (`opts`'s lN term): by DEFAULT
    (credit_mitm_qracm=False) NOT quantum-adjusted -- stays at the
    classical exponent 1/2 (sqrt(N) via the classical meet-in-the-middle
    time-memory tradeoff). Applying Grover's generic quadratic speedup
    AGAIN on top of an ALREADY sqrt(N)-optimized classical MITM would
    double-count a speedup Grover does not actually provide here: Grover
    speeds up brute-force O(N) unstructured search to O(sqrt(N)); the
    classical MITM technique already achieves O(sqrt(N)) through a
    completely different mechanism (two-list sort/hash collision, not
    search), so Grover has nothing further to act on in the naive
    reading. The ACTUAL quantum algorithm for this two-list
    meet-in-the-middle/collision-finding structure is Brassard-Hoyer-Tapp
    (1997), giving O(N^(1/3)) time -- but ONLY with O(N^(1/3))
    quantum-accessible quantum RAM (QRACM), an assumption many
    post-quantum security analyses explicitly decline to credit an
    attacker with (this exact caveat appears in NIST PQC methodology
    discussions and in submissions such as Kyber/Dilithium's own spec
    documents, which decline BHT-style credit for their own
    combinatorial/MITM-style attack components for the same reason).
    Pass credit_mitm_qracm=True for a SEPARATE, more aggressive bound
    (exponent 1/3 instead of 1/2) under that stronger, contested
    assumption. Report BOTH bounds, not just one -- the literature does
    not settle which hardware model an attacker is entitled to, so
    neither number alone is the whole picture.

    NOT adjusted, and NOT invented: the dual-attack's own repeat/sample
    count (`lR`, a statistical-averaging term, not a search) and the
    sparse-window-weight-guess retry amplification (-log2(P_w), "redo the
    WHOLE lattice-reduction-plus-MITM attack because the guess about how
    many nonzeros fell in the window was wrong"). No literature
    establishing a sound quantum (e.g. amplitude-amplification) treatment
    of either term, for this specific hybrid-dual-MITM-for-sparse-secret
    construction, was found in the references consulted (Kaye/Laflamme/
    Mosca and Nielsen & Chuang -- both cover ONLY the generic
    single-list Grover speedup and amplitude amplification in the
    abstract, not this attack's specific two-list/retry structure; see
    SECURITY_ASSESSMENT.md for exactly what was and was not found in
    each). Left classical rather than guessed.
    """
    best = (1e9, None)
    mitm_exp = (1.0/3.0) if credit_mitm_qracm else 0.5
    for k in range(n//64, n, max(1, n//80)):
        npr = n - k
        if npr < 64: break
        Ew = h*k/n
        opts = []
        for w in range(0, int(Ew)+3):
            lN = max(log2C(k,j)+j for j in range(0, w+1))
            ps = [log2C(k,j)+log2C(n-k,h-j)-log2C(n,h) for j in range(0, w+1)]
            mx = max(ps); lp = mx + math.log2(sum(2**(p-mx) for p in ps))
            opts.append((-lp, mitm_exp*lN))
        for b in range(60, 1400, 5):
            ld = math.log2(delta_bkz(b))
            for m in range(npr//2, 3*npr, max(1, npr//20)):
                d = m + npr
                tau = d*ld + (npr*logq)/d + math.log2(sigma) - logq
                if tau > 0: continue
                lR = 4*(math.pi**2)*(2**(2*tau))/math.log(2)
                if lR > 200: continue
                bk = 0.265*b + math.log2(8*d) + 16.4   # quantum sieve exponent
                tot = min(lR + ip + max(bk, ln) for ip, ln in opts)
                if tot < best[0]: best = (tot, (k, b))
    return best

if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        for name, n, lq, ref in (("Son-Cheon 2019/1019", 65536, 1240, 113.0),
                                 ("Cheon+ 2019/1114",    32768,  628, 112.5)):
            c, _ = hybrid(n, lq, 64)
            print(f"{name:<22} published {ref:6.1f}  model {c:6.1f}  ({c-ref:+.1f})")
        sys.exit(0)
    n, logqp, h = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    c, info = hybrid(n, logqp, h)
    print(f"n={n} log2(QP)={logqp} h={h}: {c:.0f} classical bits  (k={info[0]}, beta={info[1]})")
    if "--quantum" in sys.argv:
        qc, qinfo = hybrid_quantum(n, logqp, h, credit_mitm_qracm=False)
        print(f"  quantum (no large QRACM credited to MITM, conservative/NIST-floor):"
              f" {qc:.0f} bits  (k={qinfo[0]}, beta={qinfo[1]})")
        qa, qainfo = hybrid_quantum(n, logqp, h, credit_mitm_qracm=True)
        print(f"  quantum (BHT-style N^(1/3) MITM credited, assumes large QRACM):"
              f" {qa:.0f} bits  (k={qainfo[0]}, beta={qainfo[1]})")
        print("  NOT adjusted (no literature basis found): dual-attack repeat count,"
              " sparse-window-guess retry amplification -- see this function's docstring.")
