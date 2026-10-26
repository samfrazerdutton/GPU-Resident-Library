# Security assessment — GPU-Resident-Library

## Method and tooling

Two literature-calibrated estimators already exist in this repo:
`tools/lwe_security.py` (primal uSVP, BDGL16 sieve cost model, validated to
within 0.2-3.6 bits against the Homomorphic Encryption Standard's published
128-bit reference table for uniform ternary secrets) and
`tools/hybrid_security.py` (hybrid dual+MITM attack for sparse secrets,
calibrated against two published cryptanalysis results: Son-Cheon 2019/1019
and Cheon et al. 2019/1114, within ~3-6 bits). Both are classical-attack
cost models with a quantum-adjusted sieve exponent (0.265 vs 0.292 β) for
the primal estimator; neither is a substitute for the SageMath
lattice-estimator or independent third-party review — this is stated
explicitly in their own docstrings and repeated here deliberately.

## Current parameters, measured

`tests/test_boot.cpp`: n=8192, sizeQ=30 towers (1×60-bit + 29×55-bit),
sizeP=2 (2×60-bit) → log2(QP)=1775 bits. Secret: **sparse ternary, Hamming
weight h=64** (`set_secret_hamming_weight(64)`).

```
$ python3 tools/hybrid_security.py 8192 1775 64
n=8192 log2(QP)=1775 h=64: 51 classical bits
$ python3 tools/lwe_security.py 8192 1775 64
n=8192 log2(QP)=1775 sparse h=64: 47 classical / 46 quantum bits (primal, upper bound)
```

**~51 classical bits, ~46 quantum-adjusted.** Not 128-bit by either
estimator. This was never checked before this session and is not a
regression from any specific change — it is the parameter set's inherent
security, undiscovered until measured.

## Why: budget vs. ring dimension, and independently reproduced

`lwe_security.py --validate`'s own table (uniform ternary, 128-bit
classical): n=8192 supports ~218 bits of modulus budget. This repo's
actual log2(QP)=1775 is **~8.1x** that budget. An independently-delegated
research agent, working from the published Homomorphic Encryption Standard
tables (eprint 2019/939) and not from this repo's own tooling, derived the
same ratio (1775/218 ≈ 8.1x) from a different source — cross-validating
that this repo's security-estimator implementation is not itself the
source of the discrepancy.

## Correction to the first-pass remediation estimate: n=131072 was the wrong target

The first-pass analysis in `docs/performance-baseline.md` (and the
now-superseded framing in this session) concluded ~n=131072 (16x) would be
needed to reach 128-bit security at the current 1775-bit budget, based on
sweeping `hybrid_security.py` against the SPARSE secret (h=64) at
increasing n. That sweep was correct given its own assumption (sparse
secret held fixed) — but the assumption itself was never questioned.

**A dedicated literature-research pass found the real fix: switch from a
sparse secret to a non-sparse (uniform ternary) secret, not just a bigger
ring.** Citation: Bossuat, Mouchet, Troncoso-Pastoriza, Hubaux, *"Efficient
Bootstrapping for Approximate Homomorphic Encryption with Non-Sparse
Keys"*, EUROCRYPT 2021 (eprint 2020/1203) — a paper that exists
specifically because sparse secrets lose meaningful security margin to
dual/hybrid attacks relative to uniform ternary secrets at the same
(n, logQP), and whose stated contribution is a bootstrapping design that
is secure precisely because it uses a non-sparse key. OpenFHE and Lattigo
both default to uniform ternary today for exactly this reason, per the
same research pass.

Against the Homomorphic Encryption Standard's table for a **uniform
ternary** secret:

| n | max log2(q) for 128-bit classical |
|---|---|
| 8192 | ~218 |
| 16384 | ~438 |
| 32768 | ~881 |
| 65536 | ~1747 |

**1775 bits is essentially exactly the n=65536 limit for a uniform ternary
secret — not n=131072.** The extra 8x this session originally estimated
was the cost of keeping a sparse secret, not an inherent property of the
circuit's depth requirement. n=8192 and n=16384 remain not viable (218 and
438 bits respectively support only ~3-7 towers, versus the ~20-30 a
CoeffsToSlots→EvalMod→SlotsToCoeffs bootstrap circuit needs) — "shrink the
circuit to fit a small n" is not a realistic path at this depth; "grow n
and drop the sparse-secret assumption" is.

## n=65536 + uniform ternary — now directly verified, not just cited

Ran this repo's own `lwe_security.py` (primal uSVP, validated to within
0.2-3.6 bits of the published 128-bit reference table) at the literal
current modulus budget with no other change but the secret distribution:

```
$ python3 tools/lwe_security.py 65536 1775
n=65536 log2(QP)=1775 uniform ternary: beta=314 -> 128 classical / 120 quantum bits
$ python3 tools/lwe_security.py 32768 1775
n=32768 log2(QP)=1775 uniform ternary: beta=97 -> 64 classical / 61 quantum bits
$ python3 tools/lwe_security.py 131072 1775
n=131072 log2(QP)=1775 uniform ternary: beta=831 -> 280 classical / 258 quantum bits
```

**n=65536 lands almost exactly on 128 classical / 120 quantum bits at this
circuit's existing 1775-bit budget.** n=32768 is insufficient (64 bits).
n=131072 is massive, wasteful overkill (280 bits — 2.2x more security than
needed, at 2x the n, memory, and NTT cost of n=65536 for zero required
benefit) once the sparse-secret assumption is dropped. **n=65536 is the
correct target, confirmed with this repo's own tooling, not just the cited
paper's general argument.**

## Standing, unresolved caveats (explicit, not glossed over)

- Neither estimator here has been cross-checked against the SageMath
  lattice-estimator (the de facto standard tool) for THESE specific
  parameters. The validation that exists is against published reference
  tables and two specific prior cryptanalysis results, not an
  apples-to-apples run of the same (n, logQP, h) through an independent
  tool. This remains open.
- Switching from sparse to uniform ternary secret has its OWN cost
  implications (larger Hamming weight generally means more non-zero
  secret-key coefficients, which can affect noise growth and certain
  circuit costs) that have not been analyzed here.
- No independent, external cryptographic review has occurred. Nothing in
  this repository should be described as "production-grade" or "secure
  for real data" on the basis of this assessment alone.

## Disposition

Do not adopt n=131072 (confirmed wasteful: 280 bits, 2.2x the needed
security, at 2x n=65536's cost). Do not continue treating the current
n=8192/sparse(h=64)/1775-bit parameter set as anything other than an
explicitly insecure (~51-bit) demo/research configuration — every
benchmark number in `docs/performance-baseline.md` and
`EXPERIMENT_LEDGER.md` describes this demo configuration, not a secure
one. The candidate secure profile is now: **n=65536, log2(QP)≈1775,
uniform ternary secret, 128/120-bit classical/quantum per this repo's own
primal estimator.** The next concrete security task is: (1) decide and
implement the secret-distribution switch (sparse h=64 -> uniform ternary)
in the actual keygen code, correctness-test it, (2) re-run the hybrid/MITM
estimator too for completeness (expected inapplicable/non-binding for a
uniform secret, but confirm rather than assume), (3) plan a
re-measurement of bootstrap performance at n=65536 (which will be slower
than the current ~115s, likely
substantially — not yet measured).
