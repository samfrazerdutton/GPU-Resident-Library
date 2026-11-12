# Phase 39: FHE textbook-to-source implementation map

This document maps the mathematical operations described in Ronny Ko's
*The Beginner's Textbook for Fully Homomorphic Encryption* (arXiv:2503.05136v26,
June 2, 2026) to this repository's actual C++/CUDA implementation. It is a
companion to `research/PHASE39_FHE_TEXTBOOK_RNS_AND_BATCHED_ENCRYPTION.md`,
which contains the full phase narrative, measurements, and the one
implemented change this mapping work led to.

An operation is only marked "implemented" below when direct source reading
confirmed an equivalent, not merely because a similarly-named function
exists.

## Reference materials

- **Resource A**: Ronny Ko, *The Beginner's Textbook for Fully Homomorphic
  Encryption*, arXiv:2503.05136v26 (259 pages). Locally accessible at
  `~/fhe-reference-material/2503.05136v26.pdf` (copied from
  `~/Downloads/2503.05136v26.pdf`; not committed to the repository --
  copyrighted arXiv preprint, no redistribution license confirmed).
- **Resource B**: Craig Gentry, *A Fully Homomorphic Encryption Scheme*,
  Stanford PhD dissertation, September 2009 (209 pages). Locally accessible
  at `~/fhe-reference-material/craig-thesis.pdf`. Not committed, same
  reasoning.
- **Resource C**: TLShare (already reviewed in Phase 35; not re-reviewed
  this phase beyond the input-authenticity cross-check in section 8 of the
  main report). Locally accessible at
  `~/fhe-reference-material/3-3-1.pdf`.

## Sections actually studied (not merely headings)

- **Part I**: A-14 (Chinese Remainder Theorem / RNS, pp. 64-66) and A-16
  (Efficient Polynomial Multiplication by FFT and NTT, pp. 68-73) read in
  full. A-5, A-7, A-8, A-11, A-13 reviewed via their direct use inside the
  CKKS chapter (D-3.1's own "Required Background" cross-references) rather
  than read as separate standalone sections -- their content (polynomial
  rings, roots of unity, vector/matrix isomorphisms, modulo rescaling) is
  fully subsumed by D-3.1's own worked example, which this document reads
  in full.
- **Part II**: B-1, B-3, B-4 reviewed via their direct instantiation inside
  D-3.2 (CKKS Encryption and Decryption), which states explicitly "CKKS's
  encryption and decryption schemes are similar to BFV's" and gives the
  concrete RLWE formulas directly -- the generic GLWE treatment in Part II
  is the same mathematical content at one level of abstraction higher.
- **Part III**: C-1 through C-5 (generic GLWE ciphertext operations)
  reviewed via their CKKS-specific instantiation in D-3.3 through D-3.8,
  for the same reason.
- **Part IV**: D-3 (CKKS Scheme, pp. 176-199) read in full, including
  D-3.1 (Encoding/Decoding, with its full worked numerical example),
  D-3.2 (Encryption/Decryption), D-3.3-D-3.6 (addition/multiplication),
  D-3.5.1-D-3.5.6 (synthetic ciphertext derivation, both relinearization
  methods, rescaling), D-3.7 (ModDrop, including its explicit "difference
  between modulus switch and ModDrop" subsection), D-3.8 (key switching),
  D-3.9 (rotation, with worked example), D-3.10-D-3.12 (encoding
  contemplation, conjugation, sparse packing). D-3.13 (bootstrapping) was
  **not** read in the same depth -- this repository implements no
  bootstrapping at any point in its history (confirmed by its own
  `research/` record across every phase), so the detailed CoeffToSlot/
  EvalExp/SlotToCoeff machinery has no corresponding source to map against;
  its high-level conclusion (CKKS bootstrapping eliminates the ModRaise
  garbage value via sine-function evaluation rather than digit extraction)
  is noted in section 6 below for completeness only.
- **D-5** (RNS-variant FHE Schemes, pp. 226-248) read in full: D-5.1
  (FastBConv), D-5.2 (SmallMont), D-5.3 (ModRaiseRNS), D-5.4 (ModDropRNS),
  D-5.5 (ModSwitchRNS), D-5.8 (FastBConvEx), D-5.9 (DecompMultRNS), D-5.10
  (applying RNS techniques to FHE operations, including D-5.10.5's CKKS-
  specific multiplication and D-5.10.10's noise-impact summary).
- **Gentry's dissertation**: Abstract and the full Introduction (sections
  1.1-1.8, pp. 1-22) read in full. Chapter 12 (Performance and
  Optimizations, pp. 115-124) read in full, per the mission's explicit
  emphasis. Chapters 4, 7, 10, 11 confirmed, via their own opening pages,
  to formalize exactly what sections 1.3-1.6 already explain informally
  (Recrypt/bootstrappability, the ideal-lattice construction, squashing,
  security) -- not re-read page-by-page in their formal notation, since
  the mission explicitly prohibits transplanting any of this scheme's
  specific machinery and the informal treatment already captures the
  content needed to extract general lessons (section 7 of the main
  report).

## The mapping table

| Textbook concept | Textbook section | Repository implementation | Equivalence / difference |
|---|---|---|---|
| CKKS encoding | D-3.1 | `src/encode_fft.cpp`: `encode_host_fft` | Same canonical-embedding + Delta-scaling + rounding idea; this repo uses an FFT-based (not the book's Vandermonde-matrix-literal) evaluation, over the identical `X^N+1` cyclotomic ring and the identical `r_k=5^k mod 2N` rotation-group slot ordering the book's own `J(h)=5^h mod 2n` convention describes. |
| CKKS decoding | D-3.1 | `src/encode_fft.cpp`: `decode_host_fft` | Same relationship, inverse direction. |
| RLWE encryption | D-3.2 | `src/encrypt_host_shoup_pooled_barrett_batched.cpp` (production path) and `src/encrypt_host_shoup_pooled_barrett_batched_multiclient.cpp` (Phase 38) | `B = -A*S + Delta*M + E` matches the book's `Summary D-3.2` formula exactly (this project's own ternary-secret, Gaussian-noise construction, `h=64`, unaffected by the book's own generic ternary-secret mention -- no change made, per the mission's explicit instruction). |
| NTT / inverse NTT | A-16 | `kernels/ntt_ct.cu` (forward, Cooley-Tukey, decimation-in-time), `kernels/intt_ct.cu` (inverse, Gentleman-Sande) | Same mathematical operation (evaluate/interpolate at the `2n`-th roots of unity of `X^n+1`) via the same recursive-splitting principle the book derives in A-16.2; this repo's kernel is an iterative, in-place, bit-reversed-output CT butterfly (standard production NTT shape), not the book's literal recursive pseudocode -- mathematically equivalent, independently verified by this project's own long-standing `test_ntt_*`/`test_intt_*` suite, not re-derived this phase. |
| RNS representation | A-14, D-5 | Every `encrypt_host_*`/`keyswitch*`/`rescale*` source file's `mod`/`q`/tower-indexed arrays | Standard CRT/RNS: each ciphertext coefficient is represented by its residues modulo a chain of co-prime NTT-friendly primes (`mod[0..sizeQ)`), exactly Theorem A-14.1/A-14.2's isomorphism. |
| Fast base conversion (`FastBConv`) | D-5.1 | `kernels/basis_convert.cu`: `LaunchApproxSwitchCRTBasis` (declared `include/basis_convert.h`) | **Confirmed present and APPROXIMATE, correctly so.** The kernel computes, per coefficient: `x_i = shoup_mulmod(src[i], QHatInvModq[i], ...)` then `sum[j] += x_i * QHatModp[i][j]` (128-bit accumulation) then a Barrett reduction mod `p_j` -- this is `FastBConv`'s own formula (`Σ_i |x_i·z_i|_{q_i}·y_i mod b_j`) verbatim, with NO SmallMont-style correction applied, so it carries the book's documented `|u·q|_b` noise (`|u| ≤ sizeQ/2+1`). The function's own name ("Approx") is an honest, correct acknowledgment of this, matching OpenFHE's own reference hybrid key-switching implementation (whose constants this file explicitly borrows). |
| Exact fast base conversion (`FastBConvEx`) | D-5.8 | **Not present.** | No equivalent operation found anywhere in this repository. This project's RNS pipeline never needs an exact base conversion, because (matching D-5.10.5's own stated reason) this repo's CKKS multiplication does not use BFV-style `ModRaise`/constant-multiply-by-`t`/`FastBConvEx` at all -- it only needs a modulus switch (rescale) to the next lower level, which this project implements directly via its own per-tower rescale machinery, not via `FastBConvEx`. |
| `SmallMont` | D-5.2 | **Not present.** | No equivalent correction technique found. Consistent with the `FastBConv` finding above: this project accepts `ApproxSwitchCRTBasis`'s full `|u|≤sizeQ/2+1` noise bound rather than tightening it to `{-1,0,1}`, and compensates by sizing the auxiliary modulus `P` large enough relative to the decomposed partition size to absorb it -- see this document's own section on `P >= Q_part_max` below. **Not added this phase**: the mission explicitly warns against transplanting `SmallMont` into an unrelated conversion routine merely because the formulas look similar, and reducing `P` on the strength of a `SmallMont`-tightened noise bound would be a security-parameter change this phase's narrow scope does not justify. |
| ModRaise (RNS) | D-5.3 | Used internally by `ApproxSwitchCRTBasis`'s own base-extension role (raising `Q`'s residues into the auxiliary `P` basis during key-switch decomposition) | Same `FastBConv(x,q,b)` call, reinterpreted as mod-raising rather than mod-switching -- the book's own D-5.3 derivation shows `ModRaiseRNS` IS `FastBConvRNS` under a different name; this repository's single `ApproxSwitchCRTBasis` kernel genuinely serves both roles (decompose: `Q→P`; combine: `P→Q`), exactly as the book's own cross-reference between D-5.3 and D-5.1 implies. |
| ModDrop (RNS) | D-5.4 | **Not present as a separate operation.** | Confirmed by direct search (`grep -rl 'moddrop\|ModDrop'` across the whole repository returns zero matches) and by reading `src/ckks_ops.cpp`'s `rescale_resident` in full: this project has exactly ONE level-dropping operation, and it always performs the FULL rescale (switch-modulus + `LaunchRescaleFuse` with scaling factors `s1`/`s2`) -- there is no lighter, zero-noise, scale-preserving ModDrop path the book's own D-3.7/D-3.7.1 describes as a distinct, cheaper operation for when a ciphertext is known to need no further multiplications. This is a genuine "textbook covers an operation the repository does not implement" finding, recorded per the mission's own instruction, not a bug (this project's workloads -- shallow, non-bootstrapping, a pure-sum aggregation path -- have never needed to drop a level without also rescaling, so the gap has had no practical consequence so far). |
| Modulus switching / rescaling | D-3.5.4, D-5.5 | `src/ckks_ops.cpp`'s `rescale_resident` (reviewed, not modified) | Matches `ModSwitchRNS`'s own structure: INTT the dropped tower, `LaunchSwitchModulus` (base conversion into each surviving tower's modulus) then `LaunchRescaleFuse` (combine with the scaling relation) -- architecturally the SAME SHAPE of operation as the `ApproxModDown` combine step this phase actually optimized (see below), just applied in the rescale context rather than key-switch mod-down. Not modified this phase (out of narrow scope; `LaunchRescaleFuse`'s own internals were not read closely enough to confirm whether it has the same raw-division-vs-Shoup opportunity -- flagged as deferred research, not assumed either way). |
| Key switching | D-3.8, D-5.10.2 | `src/keyswitch.cpp`, `src/keyswitch_resident.cpp`, `src/keyswitch_hoist.cpp` | `RLWE_{S',σ}(ΔM) = B + ⟨Decomp^{β,l}(A), RLev^{β,l}_{S',σ}(S)⟩` (book's own D-3.8 summary) matches this project's own hybrid key-switching architecture: decompose `A` into per-partition pieces (this project's `numPart`), base-convert each partition into the auxiliary `P` basis via `ApproxSwitchCRTBasis` (= `FastBConv`/`DecompMultRNS`'s per-partition pre-generated key-switching keys, D-5.9), accumulate, then `ApproxModDown` (= `ModSwitchRNS`) back to `Q`. This explains, with a direct textbook citation, WHY this project's own established `P >= Q_part_max` invariant must hold: `ApproxSwitchCRTBasis`'s approximate noise is bounded in terms of the SOURCE partition's tower count, and the hybrid key-switching's overall correctness proof requires the auxiliary modulus `P` to be large enough, relative to the largest single partition (`Q_part_max`), to absorb that noise without corrupting the decrypted result -- a concrete, sourced explanation for a rule this project's prior phases have enforced without this specific derivation on record. |
| Relinearization | D-3.5.1-D-3.5.3, D-5.9 (`DecompMultRNS`) | Same key-switching machinery above, applied to the squared-secret-key synthetic ciphertext `D_2` | Identical structure to key switching (the book's own D-5.9 explicitly derives relinearization as `DecompMultRNS` applied to `S^2` rather than `S→S'`) -- this project reuses the SAME key-switch code path for both, matching the book's own observation that they are the same primitive. |
| Slot rotation | D-3.9, D-5.10.3 | `src/rotate.cpp`, `src/automorph*.cpp`, `src/keyswitch_hoist.cpp` (hoisted rotation) | `RLWE_{S(X^{J(h)}),σ}(...) → ` key-switch to `S(X)` (book's own D-3.9/D-5.10.3 two-step structure) matches this project's own rotation implementation: apply the Galois automorphism to the ciphertext polynomials, then key-switch back to the original secret -- same `J(h)=5^h mod 2n` convention the book uses throughout. Not modified this phase. |

## A concrete finding this mapping produced: the `ApproxModDown` combine step

`kernels/moddown.cu`'s `moddown_combine_kernel` computes, per coefficient,
`ans[k] = (a[k]-b[k]) * s mod q`, where `s` (`PInvModq[i]`) is a FIXED
per-tower constant known before the kernel ever launches (the book's own
`ModSwitchRNS` summary, D-5.5, derives exactly this formula: `|b^{-1}|_q *
(χ - χ̂) mod q`). Before this phase, the kernel computed this via a raw
128-bit `% q` division per coefficient -- the SAME shape of computation
this project already uses Shoup's precomputed-multiplier technique for
everywhere else on its hot paths (e.g. the production encrypt path's own
`shoup_mulmod`), but not here. This phase implemented that substitution;
see the main report for the full correctness verification and
performance measurement.

## What this mapping did NOT lead to

- **No `SmallMont` addition.** Reducing `ApproxSwitchCRTBasis`'s noise
  bound would only be useful if it justified shrinking `P`, a security-
  sensitive parameter change out of this phase's scope (and the mission's
  explicit instruction: "do not add SmallMont to an unrelated conversion
  routine merely because the names or formulas appear similar").
- **No `FastBConvEx` addition.** This project's CKKS multiplication
  already avoids needing it (per D-5.10.5's own stated reason, confirmed
  against this repository's actual rescale-only multiplication structure).
- **No rewrite of the rescale/ModDrop machinery.** Reviewed and confirmed
  semantically distinct and correctly separated, per D-3.7.1/D-5.5.1's own
  explicit discussion -- no gap found to fix.
- **No bootstrapping work of any kind.** Not implemented in this project;
  D-3.13's machinery has no corresponding source to audit or improve.
