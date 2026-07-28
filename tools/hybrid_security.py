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
