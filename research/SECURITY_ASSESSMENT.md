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

## SUPERSEDED: n=65536 + uniform ternary — the flagged caveat turned out to matter

The previous revision of this document concluded n=65536 + uniform ternary
gives 128/120-bit security "at the existing 1775-bit budget," and
explicitly flagged as an open caveat: *"Switching from sparse to uniform
ternary secret has its own cost implications... that have not been
analyzed here."* That caveat was correct to raise, and resolving it
reverses the conclusion.

**The 1775-bit budget is not secret-distribution-independent.** This
codebase's own EvalMod circuit (`tests/test_boot.cpp`) sizes its
double-angle iteration count from a secret-dependent "K budget":

```cpp
const double Kbud=(HW? 2.5*std::sqrt((double)HW) : 40.0*std::sqrt((double)n/1024.0));
const uint32_t rDbl=(uint32_t)std::ceil(std::log2(2*M_PI*(Kbud+0.25))), degC=10;
```

For the sparse secret (HW=64, fixed), Kbud=20 regardless of n, giving
rDbl=7 double-angle steps — this is what the current, already-validated
circuit (1775-bit budget) actually implements. For a **uniform** ternary
secret (HW=0 branch), Kbud scales with √n, and grows the required
double-angle count substantially at the n values needed for security:

| n (uniform secret) | Kbud | rDbl | extra towers vs. current (rDbl=7) circuit |
|---|---|---|---|
| 8192 | 113.1 | 10 | +3 |
| 32768 | 226.3 | 11 | +4 |
| 65536 | 320.0 | 11 | +4 |
| 131072 | 452.5 | 12 | +5 |

Each extra double-angle step costs one more tower (~55 bits) in the actual
modulus chain — this is not a rough guess, it is how `rDbl` is consumed
in the existing, already-bit-exact-verified circuit. Re-ran the security
estimator at the **self-consistent** budget (1775 + extra_towers×55) for
each candidate n:

```
n=65536  self-consistent budget=1995: 113 classical / 106 quantum bits  -- SHORT of 128
n=131072 self-consistent budget=2050: 237 classical / 219 quantum bits  -- massive overkill
```

**n=65536 + uniform ternary does NOT reach 128 bits once the circuit's
own extra depth requirement is correctly priced in.** The 128-bit number
in the previous revision was real but computed against the wrong
(sparse-secret-sized) budget — an honest methodology error, caught by
actually implementing the self-consistent check rather than reusing a
borrowed budget figure across a secret-distribution change.

## Corrected recommendation: n=131072, KEEP the existing sparse secret (h=64)

Checked the alternative directly: sparse secret (h=64, **no code change,
no circuit change, no extra EvalMod depth** — Kbud stays fixed at 20
regardless of n since the sparse branch doesn't depend on n) at n=131072,
same 1775-bit budget as the circuit already implements and has validated
bit-exact 13 consecutive times this session:

```
$ python3 tools/hybrid_security.py 131072 1775 64
n=131072 log2(QP)=1775 h=64: 134 classical bits
$ python3 tools/hybrid_security.py 65536 1775 64
n=65536 log2(QP)=1775 h=64: 90 classical bits   -- confirms n=65536 is short either way
```

**n=131072 with the existing sparse secret reaches 134 classical bits,
using the exact same bootstrap circuit (same rDbl=7, same degC=10, same
everything except n/moduli/sizeQ) already extensively validated.** This
dominates the uniform-ternary path on every axis that matters here: it
meets the 128-bit target (uniform-at-65536 does not), it does not require
the larger, previously-unvalidated circuit depth uniform secrets need at
large n (uniform-at-131072 does, wastefully), and it requires zero changes
to `src/keygen.cpp`'s secret-sampling code (the sparse path is already
implemented and tested).

**n=131072 is the corrected target. Switching to uniform ternary is not
needed and should not be pursued for this specific implementation** — the
general argument in Bossuat et al. (sparse secrets lose margin to
dual/hybrid attacks) remains true in the abstract, but in THIS codebase's
concrete circuit, going uniform costs more in required depth than it saves
in attack resistance, at the n values actually in play.

## Quantum-adjusted hybrid estimate: derived, with an honest split verdict

The prior gap: `hybrid_security.py`'s hybrid dual+MITM model is the
BINDING attack for this codebase's sparse secret (h=64) — tighter than
the primal model, which doesn't capture the sparse-specific MITM
leverage — but it reported only a classical bit count. Deriving a sound
quantum-adjusted version requires treating its TWO distinct attack
phases separately; they do not share a single quantum speedup.

**The model's exact structure** (reconstructed from `tools/
hybrid_security.py`'s `hybrid()`, confirmed by reading the whole file):
for a guessing-window size `k` (0 to n, searched), split the n secret
coordinates into a `k`-sized MITM-guessing window and an `n-k`-sized
dual-attack window. For each bet `w` on how many of the window's
coordinates are actually nonzero (cheaper to enumerate for small `w`,
but wrong with probability governed by the hypergeometric distribution
`P[w'<=w] = sum_j C(k,j)*C(n-k,h-j)/C(n,h)`), the attack cost is:

```
cost(k,b,m,w) = lR(b,m)                [dual-attack repeat/sample count, statistical]
              + (-log2 P_w)            [expected restarts if the w-bet was wrong]
              + max( BKZ_cost(b,d),    [lattice-reduction phase, dimension d=m+(n-k)]
                     MITM_cost(k,w) )  [[combinatorial enumeration, classical sqrt(N)]]
```
minimized over `(k,b,m,w)`. `BKZ_cost(b,d) = 0.292*b + log2(8d) + 16.4`
is the SAME classical BDGL16 sieve-cost formula `lwe_security.py` already
uses for its own (unrelated, primal-attack) lattice reduction step.
`MITM_cost(k,w) = 0.5*log2(N)` for `N` = the size of the "at most w
nonzero positions among k, each +-1" enumeration space — i.e. the
classical meet-in-the-middle time-memory tradeoff (O(sqrt(N)) time via
a two-list sort/hash split, not a search algorithm).

**Reference verification (both supplied PDFs actually read, not
assumed from title):**
- The "Mosca" link resolves to Kaye, Laflamme & Mosca, *An Introduction
  to Quantum Computing* (Oxford, 2007) — a full co-authored textbook,
  not a standalone paper. Its own preface states it deliberately
  "avoided... the mathematical formalism of quantum information theory
  and quantum cryptography." Chapter 8 ("Algorithms Based on Amplitude
  Amplification," p152) covers Grover's search (8.1, p152) and
  amplitude amplification (8.2, p163) in the abstract. The index (p270)
  has no entry for "collision," "claw," "meet-in-the-middle," "lattice,"
  "LWE," or "hybrid attack."
- Nielsen & Chuang, *Quantum Computation and Quantum Information* (10th
  Anniversary Ed., Cambridge) — the standard comprehensive QC reference.
  Chapter 6 ("Quantum search algorithms," p248) covers Grover's
  algorithm (6.1-6.2), quantum counting (6.3), and explicitly "Speeding
  up the solution of NP-complete problems" (6.4, p263) — read directly:
  it applies Grover's GENERIC quadratic speedup to a SINGLE-LIST
  brute-force witness search (worked example: the Hamiltonian cycle
  problem, O(2^n) classical -> O(2^(n/2)) quantum). This is the exact
  same single-list speedup already reflected in `lwe_security.py`'s
  existing 0.292->0.265 sieve-exponent substitution. Chapter 6 contains
  no two-list/collision/meet-in-the-middle-specific algorithm.

**Conclusion from both references**: they establish the FOUNDATIONAL
Grover/amplitude-amplification material, confirming that a generic
single-list O(N)->O(sqrt(N)) quantum speedup is standard and
well-established — but NEITHER contains the specialized quantum
algorithm needed for this model's actual MITM structure (a two-list
meet-in-the-middle/collision search, not a single-list brute-force
search). That specific algorithm is Brassard-Hoyer-Tapp (1997),
"Quantum Cryptanalysis of Hash and Claw-Free Functions" (outside both
supplied references — general lattice/post-quantum cryptanalysis
literature, not covered by either introductory/comprehensive QC
textbook), giving O(N^(1/3)) time for two-list collision-finding — a
REAL but SMALLER speedup than naively squaring the classical sqrt(N)
exponent down to N^(1/4) would suggest (that naive move is EXACTLY the
"substituting a generic quantum speedup" the governing directive warns
against — rejected). Critically, BHT's N^(1/3) result REQUIRES
O(N^(1/3)) quantum-accessible quantum RAM (QRACM) — an assumption many
post-quantum security analyses (including NIST's own PQC methodology
discussions, and submissions such as Kyber/Dilithium's specifications)
explicitly decline to credit an attacker with for exactly this kind of
combinatorial/MITM attack component, treating the no-large-QRACM cost
as the conservative floor.

**What was implemented** (`tools/hybrid_security.py`'s new
`hybrid_quantum(n, logq, h, sigma, credit_mitm_qracm)`, classical
`hybrid()` left completely unchanged for regression comparison):
- BKZ/lattice-reduction phase: quantum sieve exponent 0.265 substituted
  for 0.292, unconditionally — same subroutine, same substitution
  `lwe_security.py` already makes, not new reasoning.
- MITM phase: classical exponent 1/2 by DEFAULT (`credit_mitm_qracm=
  False`) — Grover's generic speedup does not cleanly apply a second
  time on top of an already-sqrt(N)-optimized classical MITM. Set
  `credit_mitm_qracm=True` for a separate, more aggressive bound using
  BHT's 1/3 exponent, under the explicit, contested large-QRACM
  assumption.
- NOT adjusted, deliberately: the dual-attack's own repeat/sample count
  (`lR`, a statistical-averaging term, not a search problem) and the
  sparse-window-guess retry amplification (`-log2(P_w)`, "redo the
  WHOLE attack because the w-bet was wrong"). No literature establishing
  a sound quantum treatment of either, for this specific construction,
  was found in the two references consulted or in general knowledge of
  the hybrid-attack literature. Left classical rather than guessed —
  this means both reported quantum bounds below are themselves slightly
  CONSERVATIVE (a fully-quantum-adjusted attacker, if these two terms
  do turn out to admit some speedup, could do marginally better than
  either number states).

**Real output** (`python3 tools/hybrid_security.py 131072 1775 64
--quantum`, and `tools/test_hybrid_security.py`, 10/10 checks pass
including: classical calibration unchanged, quantum bounds both <=
classical, QRACM bound <= no-QRACM bound, h=0 and small-n edge cases):

```
n=131072 log2(QP)=1775 h=64: 134 classical bits  (k=79034, beta=225)
  quantum (no large QRACM credited to MITM, conservative/NIST-floor): 130 bits  (k=79034, beta=225)
  quantum (BHT-style N^(1/3) MITM credited, assumes large QRACM): 114 bits  (k=83948, beta=190)
```

**This is a genuine, non-trivial finding that must not be glossed
over: under the conservative/no-QRACM assumption, n=131072/h=64 still
clears the 128-bit target — but only by 2 bits, a much thinner margin
than the classical 134-bit figure suggested.** Under the more aggressive
QRACM-crediting assumption, it does NOT clear 128 bits (114 < 128). The
honest position is: this candidate's quantum security against the
BINDING sparse-secret attack is somewhere in the 114-130 bit range
depending on an attacker's access to large quantum-accessible quantum
memory, a genuinely unsettled assumption in the literature — not a
single confident number. The previously-reported 257-quantum-bit figure
(from the loose primal model) should NOT be read as this candidate's
quantum security margin; it significantly overstates it by not
modeling the MITM attack at all.

## Standing, unresolved caveats (explicit, not glossed over)

- Neither estimator here has been cross-checked against the SageMath
  lattice-estimator (the de facto standard tool) for THESE specific
  parameters. This remains open regardless of which secret distribution is
  chosen.
- `hybrid_security.py` now HAS a quantum-adjusted variant
  (`hybrid_quantum`, see the dedicated section below) — this caveat is
  RESOLVED, not fully, but with an honest range rather than a missing
  number. Previously: "reports only a classical-bit estimate... no
  corresponding quantum number from the tighter model." Now: two bounds
  under two different, explicitly labeled hardware assumptions (130 bits
  conservative, 114 bits if a large quantum-RAM attacker is credited) —
  see below for the full derivation and what remains genuinely
  unresolved within even the new quantum estimator.
- The Bossuat et al. paper's concern about sparse-secret attacks specific
  to bootstrapping evaluation keys (not just generic LWE hardness) may not
  be fully captured by `hybrid_security.py`'s calibration (validated
  against 2 published dual/MITM results, not against a bootstrapping-
  specific attack). This is a real residual uncertainty for the sparse
  path, not fully resolved by this assessment.
- No independent, external cryptographic review has occurred. Nothing in
  this repository should be described as "production-grade" or "secure
  for real data" on the basis of this assessment alone.
- The new `hybrid_quantum` estimator (see dedicated section above) only
  partially closes the quantum gap: it quantum-adjusts the BKZ/sieve
  phase (well-established) and offers two bounds for the MITM phase
  under two different, both-defensible QRACM assumptions, but leaves
  the dual-attack's repeat-count term and the sparse-window retry
  amplification UNADJUSTED, because no literature basis for a sound
  quantum treatment of either was found. If such a basis is later
  found, both reported quantum bit counts (130 and 114) could move
  LOWER, not higher — they are not final numbers, they are the best
  currently-derivable bounds. Neither has been cross-checked against
  the SageMath lattice-estimator or published hybrid-attack papers'
  own quantum variants (if any exist) — this was a literature search
  within the scope of one delegated investigation, not an exhaustive
  survey.

## Disposition

**n=131072, sparse secret h=64 (unchanged), log2(QP)=1775 (unchanged
circuit) is the corrected candidate secure profile — not n=65536 uniform
ternary.** Do not implement the secret-distribution switch that the
previous revision of this document recommended; it has been superseded by
the analysis above. Do not continue treating the current n=8192 instance
as anything other than an explicitly insecure (~51-bit) demo/research
configuration.

**Updated, with the quantum derivation above factored in**: classical
security is solid (134 bits, 6-bit margin over the 128-bit target).
Quantum security is NOT a single comfortable number — it is 130 bits
under a conservative, no-large-quantum-memory attacker (2-bit margin,
thin) and 114 bits (short of 128) under a more aggressive attacker
credited with large quantum-accessible quantum RAM. This is a real,
literature-grounded range, not an artifact of this repo's tooling — do
not report "134 bits" or "257 bits" as this candidate's quantum margin
without the caveats above.

The next concrete security task is: (1) regenerate moduli/
roots at n=131072 with the existing sparse-secret code path (no keygen.cpp
change needed), (2) fix the KROT/KSC bounded-cache prerequisite (see
`MEMORY_FEASIBILITY.md` — mandatory before n=131072, which will multiply
host RAM pressure ~16x over the already-near-ceiling n=8192 measurement),
(3) run the full bootstrap end to end at n=131072 and measure real
performance (expected substantially slower than the current ~115-140s;
do not guess the number), (4) revisit the residual sparse-secret caveat
above if an independent security review becomes available, (5) if a
wider quantum margin is later judged necessary (e.g. if the project's
threat model must credit a large-QRACM attacker), re-run
`hybrid_quantum(..., credit_mitm_qracm=True)` against candidate n values
above 131072 to find where the 114-bit figure clears 128 — not yet done,
out of scope for this round since the conservative bound already meets
the target.

## UPDATE: estimator output independently re-run and preserved (this round)

Re-ran both tools fresh for the exact candidate, output preserved
verbatim:

```
$ python3 tools/hybrid_security.py 131072 1775 64
n=131072 log2(QP)=1775 h=64: 134 classical bits  (k=79034, beta=225)

$ python3 tools/lwe_security.py 131072 1775 64
n=131072 log2(QP)=1775 sparse h=64: beta=828 -> 279 classical / 257 quantum bits
  NOTE: primal only; hybrid MITM not modelled -- upper bound.
```

Confirms the 134-classical-bit figure above (hybrid model, the binding
attack for a sparse secret) exactly, independent of the prior revision's
number. The primal model's 279/257 classical/quantum is a LOOSE upper
bound (does not model the sparse-specific MITM attack) — do not quote
257 quantum bits as the actual quantum security margin; no tight
quantum number exists for this candidate (see the standing caveats
above, unchanged). Target (128 bits) is met by the tight model with a
6-bit margin, not a wide one.
