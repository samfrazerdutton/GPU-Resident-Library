# Literature review

Honesty note: entries below marked **[agent-sourced]** were produced by a
delegated research agent with live web access, which fetched and read
primary sources directly (including OpenFHE's actual source code, not just
its documentation) — citations and URLs are as the agent reported them,
not independently re-verified by the primary session except where noted
"re-verified." Entries marked **[training knowledge]** draw on the
assistant's own pretrained knowledge of well-established, canonical results
(not independently looked up this session) and should be treated as
lower-confidence than a fetched primary source until checked. No entry
claims to have read either of the two Google Books links in full — only
bibliographic metadata was fetched for one; the standard content they cover
(QFT, Shor/Grover, quantum cryptography, PQC migration) is well-established
and cited below via primary papers/standards instead of the books
themselves.

| Citation | Year | Contribution | Parameters/conditions | Hardware/benchmark | Reproducible evidence? | Applies to this project | Open questions |
|---|---|---|---|---|---|---|---|
| Cheon, Kim, Kim, Song, *"Homomorphic Encryption for Arithmetic of Approximate Numbers"* (CKKS original), ASIACRYPT 2017 | 2017 | Defines the approximate HE scheme this entire project implements | n/a (foundational) | n/a | Yes, widely implemented | This project IS an implementation of this scheme | — |
| Cheon, Han, Kim, Kim, Song, *"Bootstrapping for Approximate Homomorphic Encryption"*, EUROCRYPT 2018 | 2018 | The CoeffsToSlots/EvalMod/SlotsToCoeffs bootstrap structure this project's `test_boot.cpp` implements | n/a | n/a | Yes | Direct architectural basis for this project's bootstrap | — |
| **[agent-sourced]** Bossuat, Mouchet, Troncoso-Pastoriza, Hubaux, *"Efficient Bootstrapping for Approximate Homomorphic Encryption with Non-Sparse Keys"*, EUROCRYPT 2021, eprint 2020/1203 | 2021 | Establishes that SPARSE secrets lose meaningful security margin to dual/hybrid attacks relative to uniform ternary at the same (n, logQP); bootstrapping design secure specifically because it uses a non-sparse key | Compares sparse vs. uniform ternary secret security at matched (n, logQP) | n/a (theory + estimator) | Yes (published, cited by OpenFHE/Lattigo defaults) | **Directly resolves this project's n=131072-vs-n=65536 question** — switching this project's secret from sparse (h=64) to uniform ternary is the correct fix, confirmed independently against this repo's own `lwe_security.py` (128/120 bits at n=65536, log2(QP)=1775) | Noise-growth/circuit-cost implications of switching secret distribution not yet analyzed for this specific codebase |
| **[agent-sourced]** OpenFHE source, `src/pke/lib/schemebase/base-leveledshe.cpp`, `EvalAutomorphismCore`/`EvalAutomorphismKeyGen`/`EvalFastRotationPrecompute` | current (main branch, fetched live) | The CORRECT hoisting implementation: decompose+ModUp the UNPERMUTED ciphertext once; key-switch; apply automorphism to the QP-basis OUTPUT (not the decomposed digits); rotation keys generated as keyswitch-to-ψ_{k⁻¹}(s), not the textbook ψ_k(s) | n/a (implementation) | n/a | Yes — primary source code read directly | **Directly explains and fixes this session's hoisting failure.** The approximate-basis-conversion commutativity problem this session found is real but irrelevant to the correct algorithm — OpenFHE never permutes anything the approximation touches | This project has NOT yet implemented this corrected ordering; it is the top-ranked next experiment (#3 in the ranking, after two lower-risk prerequisites) |
| **[agent-sourced]** *"Triple-Hoisted Baby-Step Giant-Step Linear Transformation over CKKS"*, arXiv 2605.17222 | recent | States explicitly: hoisting delays applying the automorphism until after the switching-key inner product; inverse automorphism applied to the keys | BSGS linear transforms (i.e. exactly this project's C2S/S2C structure) | n/a | Published preprint | Confirms the OpenFHE-sourced mechanism is the established, current technique, not an outdated approach | — |
| **[agent-sourced]** FIDESlib, *"A Fully-Fledged Open-Source FHE Library for Efficient CKKS on GPUs"*, arXiv 2507.04775 | recent | Published default bootstrap parameter set `[N=2^16, L=29, Δ=59, dnum=4]` — ring dimension and tower count directly comparable to this project's | n=65536, ~29 towers, ~59-bit moduli, dnum(numPart)=4 (alpha≈8) | GPU-accelerated CKKS, open-source | Published, open-source | Confirms this project's tower/bit-width SHAPE (30 towers × ~55-60 bits) is normal for published bootstrap work, not unusually large; suggests this project's numPart=15/alpha=2 is far more fine-grained than the published default (alpha≈8/dnum=4) — a retuning candidate, but coupled to noise-budget re-analysis | Not yet tested whether alpha=8 fits this project's current sizeP=2 (likely does not — sizeP typically must scale with alpha) |
| **[agent-sourced]** Jung, Kim, Ahn, Cheon, Lee, *"Over 100x Faster Bootstrapping… Memory-centric Optimization with GPUs"*, TCHES 2021, eprint 2021/508 | 2021 | Identifies memory bandwidth / data movement, NOT raw arithmetic, as the dominant FHE-on-GPU bottleneck | GPU CKKS bootstrapping | Published, TCHES (peer-reviewed) | Directly supports this session's own measured finding that `rotate_ct_resident` re-uploads the full rotation key (~63MB) and does host round-trips on EVERY rotation — exactly the kind of data-movement cost this paper identifies as dominant | This project has not yet applied the fix (hoist context/key residency out of the BSGS loop) — ranked as experiment #1, lowest risk |
| Homomorphic Encryption Standard, eprint 2019/939 and v1.1 (homomorphicencryption.org) | 2019/ongoing | Published reference security tables (ring dimension vs. max modulus budget for 128-bit security, uniform ternary secret) | Various n | n/a (standard) | Yes, the standard this project's own `lwe_security.py --validate` is calibrated against | Used throughout this session's security assessment; independently reproduced by the delegated agent from the paper itself (not this repo's tool), cross-validating the tool | — |
| **[training knowledge]** NIST FIPS 203 (ML-KEM, formerly Kyber) and FIPS 204 (ML-DSA, formerly Dilithium) | 2024 (final) | Standardized post-quantum KEM and signature schemes | n/a | n/a | Yes, official NIST standards | Basis for the messaging-architecture proposal (`MESSAGING_ARCHITECTURE.md`) — this project should use these directly rather than CKKS for messaging | — |
| **[agent-sourced]** Signal, *The PQXDH Key Agreement Protocol* / *The Double Ratchet Algorithm* specifications | ongoing | Deployed, real-world hybrid classical+PQ key agreement and ratcheting design | Production messaging | Yes — deployed at scale in Signal | Direct architectural template for the messaging proposal | KEM-ratchet amortization interval (every N messages vs. every message) is a tunable this project has not fixed |

## What remains uncertain / explicitly not yet checked

- Whether this project's specific `ApproxSwitchCRTBasis` implementation, once
  hoisting is redone in OpenFHE's correct order (permute the QP-basis
  OUTPUT, not the decomposed digits), passes bit-exact verification here.
  Not yet attempted — this is the gated next experiment.
- The exact noise-growth cost of switching from sparse (h=64) to uniform
  ternary secret has not been derived or measured for this codebase.
- No SageMath lattice-estimator cross-check has been run against either
  the current (insecure) or candidate (n=65536, uniform ternary) parameter
  sets.
- The `dnum`/alpha retuning (FIDESlib's alpha≈8 vs. this project's alpha=2)
  has not been attempted and is explicitly coupled to the secret-switch and
  n-change work, not an independent experiment.
