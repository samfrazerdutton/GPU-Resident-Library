# Experiment ledger

This is a structured index. The full narrative for each entry (exact
commands, phase-by-phase timing breakdowns, root-cause reasoning) lives in
`docs/performance-baseline.md`, which predates this `research/` directory
and is kept as the detailed log rather than duplicated here. All
entries below are at the CURRENT (insecure, ~51-bit) demo parameter set —
n=8192, log2(QP)=1775, sparse secret h=64 — unless noted. Baseline before
any optimization: **191.002s**, max slot err 5.47871e-4 (bit-exact oracle:
OpenFHE).

| # | Commit / disposition | Change | Measured result | Correctness |
|---|---|---|---|---|
| 1 | cd4b2a7 | Cache evalkeygen root tables, device-resident | 191.0s → 144.3s (-24.5%) | bit-exact |
| 2 | 80c5359 | Batch keyswitch_core_resident decompose loop | 144.3s → 143.3s (~0%, kept: correct, no regression) | bit-exact |
| 3 | aa7bb14 | rotate_ct_host → rotate_ct_resident in C2S/S2C | 143.3s → 140.0s (-2.3%) | bit-exact |
| 4 | 5239477 | evalkeygen round 2: batch per-part NTT upload/download | 140.0s → 132.3s (-5.5%) | bit-exact |
| 5 | 77c4a05 | Skip evalkeygen for rescale-only KeySwitchConstants | 132.3s → 123.0s (-7.0%) | bit-exact |
| — | REVERTED, no commit | CPU→GPU diagonal accumulation (fused kernel) | ~0% (123.0 → 124.0s, noise) | bit-exact but no win; reverted |
| — | 2dcc6f0 | Async evalkeygen data uploads (keygen phase) | 123.0 → 128.3s (noise-band, kept: correct, attribution error — real throttling evidence was a different function) | bit-exact |
| 6+7 | 8067bad + 9d0bc4f | Batch keyswitch_resident's own decompose/moddown loops + async ks_context_create uploads + stream threading in rotate_ct_resident | → 120.123s (-2.4% vs. step 5) | bit-exact |
| — | REVERTED, no commit | Pinned host staging buffers for ks_context_create av/bv uploads | 120.1s → 125-129s (**real regression** — 2-buffer alternation forced an immediate host sync with no work to overlap) | bit-exact; reverted for being net-negative |
| 8 | e4ce458 | EvalMod ct*ct tensor product: CPU loop → batched GPU kernel | Honest no-op (~118-126s, within noise); but PROFILING showed the CPU loop was only 0.58s of ~118s total — explains why GPU-ifying it could never have mattered | bit-exact |
| 9 | cb07e09 | Cache EvalMod's buildK(tw) relin keys across R/I branches (they walk an identical deterministic tw sequence) | **Real win**: EvalMod phase 17.7-18.4s → consistent 14.9-15.1s across 2 runs | bit-exact |
| 10 | 0adccf1 | Batch xf_multi's (pt_to_eval_host) per-tower NTT/INTT loop | Ambiguous: targeted bucket barely moved (30.1→29.5-29.6s) but total dropped to 114.6-114.9s across 2 runs — attribution uncertain, reported as such, not oversold | bit-exact |
| — | REVERTED, no commit, CANCELLED on board | Rotation hoisting attempt #1: decompose once per stage, permute the decomposed digits per rotation | **100% mismatch**, isolated via standalone debug test to decompose+permute non-commutativity under THIS codebase's approximate basis conversion | n/a — failed before correctness could even be claimed |

**Net after round 1 (commits through 0adccf1): 191.0s → ~114.6-115s (~40%), zero bit-exactness drift across 12 consecutive changes.**

## Round 2 — post-master-directive, in progress

| # | Status | Change | Notes |
|---|---|---|---|
| — | **Root cause of hoisting attempt #1 now understood, not yet re-attempted** | — | A delegated algorithm-research pass read OpenFHE's actual `EvalAutomorphismCore`/`EvalAutomorphismKeyGen` source and found the real, published, correct hoisting order: decompose+ModUp the UNPERMUTED ciphertext once; apply automorphism to the QP-basis output AFTER the key-switch inner product, not to the decomposed digits; generate rotation keys as keyswitch-to-ψ_{k⁻¹}(s), not ψ_k(s). This requires zero changes to ApproxSwitchCRTBasis and is ranked experiment #3 (see `NEXT_SESSION.md`), gated behind two lower-risk prerequisites. |
| — | **Not yet started** | Make `rotate_ct_resident` actually resident (currently recreates `DeviceKSContext` and re-uploads the full rotation key over PCIe on EVERY rotation — ~63MB/rotation at current params — plus host round-trips for the final c0+=ba0 combine) | Ranked experiment #1: lowest risk, pure refactor using the already-existing `rotate_ct_device` path, numerically identical. This is the next concrete action. |
| — | **Not yet started** | Eval-domain automorphism as a precomputed index permutation (eliminate the INTT→permute→NTT round-trip entirely) | Ranked experiment #2: "single largest win available" per the research pass — removes ~128 transforms + ~1600 kernel launches per rotation from the ~62-66s `rotate_ct_resident` hot spot. |
| — | **Not yet started** | Correct-order rotation hoisting (OpenFHE's method) | Ranked experiment #3, gated on #1/#2 landing first (shrinks the per-rotation cost #3 would amortize, so its benefit is cleaner to measure afterward) and on regenerating rotation keys with the inverse-automorphism convention. |
| — | **Not yet started** | Replace KROT/KSC's unbounded host-side key cache with a bounded strategy | Required before ANY n-scaling attempt — see `MEMORY_FEASIBILITY.md`. Peak host RSS already at ~10.78GB at the CURRENT n=8192, driven directly and measurably by this cache. |
| — | **Not yet started** | Switch secret distribution from sparse (h=64) to uniform ternary; adopt n=65536 | See `SECURITY_ASSESSMENT.md` — verified via this repo's own estimator to give 128/120-bit classical/quantum security at the existing 1775-bit modulus budget. Will require re-measuring bootstrap performance from scratch (expected substantially slower than 115s; not yet measured). |

## Standing discipline (unchanged from round 1, still enforced)

Every change above was bit-exact verified (`max slot err` unchanged at
5.47871e-4) before being kept; every regression or null result was
reverted or reported honestly, never hidden; negative results are kept in
this ledger, not deleted.
