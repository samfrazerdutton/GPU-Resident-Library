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
| — | **VALIDATED NULL RESULT (properly controlled this time)** | Experiment #1: hoist stream/DeviceKSWork/base-ciphertext residency out of the per-diagonal loop (`StageRotCtx` design, built as a separate `test_boot_exp` binary alongside unmodified `test_boot` for true A/B). Built, bit-exact. | First attempt (single sequential runs) was contaminated by GPU thermal drift and had to be discarded as unmeasurable (see prior ledger entry, superseded). **Fixed methodology**: built `research/benchmarks/interleaved_ab.sh`, ran baseline/experiment as separate binaries interleaved 3 rounds, with GPU temp/clock/power logged per trial. Rounds 1-2 still showed experiment slower (order confound: experiment always ran second in each pair, inheriting the preceding baseline run's heat — a design flaw, noted for next time: alternate order, don't fix it). **Round 3, after the GPU reached thermal equilibrium (85°C stable): baseline rot=81.94s vs experiment rot=81.88s, total 149.16s vs 149.14s — statistically indistinguishable.** This is a clean, honest null result, not inconclusive: the per-key eval-key upload (~60MB, unavoidable, unaffected by this refactor) dominates the per-diagonal cost; the stream/work/ciphertext-residency churn this experiment eliminated was already a small fraction of it. **Not merged** (adds `StageRotCtx` complexity for zero measured benefit) — cleaned up, `test_boot_exp.cpp` removed, `test_boot.cpp` untouched. Full raw data: `research/benchmarks/interleaved_20261026T235349Z.jsonl`. |
| — | **Not yet started** | Eval-domain automorphism as a precomputed index permutation (eliminate the INTT→permute→NTT round-trip entirely) | Ranked experiment #2: "single largest win available" per the research pass — removes ~128 transforms + ~1600 kernel launches per rotation from the ~62-66s `rotate_ct_resident` hot spot. |
| — | **Not yet started** | Correct-order rotation hoisting (OpenFHE's method) | Ranked experiment #3, gated on #1/#2 landing first (shrinks the per-rotation cost #3 would amortize, so its benefit is cleaner to measure afterward) and on regenerating rotation keys with the inverse-automorphism convention. |
| 11 | 93f2231 | **DONE, real win.** Bound KSC (the cheap-to-regenerate rotation-key view cache) with LRU eviction, cap=48. KROT (expensive-to-regenerate, confirmed genuinely reused whole-bootstrap) deliberately left uncapped. | Peak RSS 10.78GB -> 8.12-8.32GB (two independent measurements: `getrusage` and `/usr/bin/time -v`), a real ~23-25% reduction. Bit-exact, 2 consecutive runs. **Does NOT by itself fix n=131072** — KROT alone would need 77.5GB at n=131072 (quantified in `MEMORY_FEASIBILITY.md`), unaddressed by this fix. |
| — | **SUPERSEDED, corrected** | Security target: was "switch to uniform ternary + n=65536" | Found via self-consistent re-derivation (accounting for uniform secrets needing more EvalMod double-angle depth in THIS codebase's circuit) that n=65536 uniform only reaches 113 bits, short of 128. **Corrected target: n=131072, KEEP the existing sparse secret (h=64) — reaches 134 bits with zero code/circuit changes**, using the already-validated circuit. See `SECURITY_ASSESSMENT.md`. |
| — | **Not yet started** | Fix KROT's own scaling (zero-copy av/bv sharing, or per-stage-scoped working set, or GPU-side streaming) | Now the precise, quantified blocker for n=131072 (77.5GB at n=131072 if unaddressed). Requires either a `KeySwitchConstants` field-type change (broad blast radius, needs correctness review) or a bigger architectural change. Not yet scoped in detail. |
| — | **Not yet started** | Eval-domain automorphism as a precomputed index permutation (eliminate the INTT→permute→NTT round-trip entirely) | Ranked experiment #2: "single largest win available" per the research pass — removes ~128 transforms + ~1600 kernel launches per rotation from the ~62-82s `rotate_ct_resident` hot spot (range reflects thermal-state-dependent measurement this session). |
| — | **Not yet started** | Correct-order rotation hoisting (OpenFHE's method) | Ranked experiment #3, gated on #2 landing first and on regenerating rotation keys with the inverse-automorphism convention. Note: experiment #1 (resident-rotation context/stream hoisting) is now a VALIDATED NULL — do not expect #3 to inherit a speedup from #1 that doesn't exist; measure #3 independently. |

## Standing discipline (unchanged from round 1, still enforced)

Every change above was bit-exact verified (`max slot err` unchanged at
5.47871e-4) before being kept; every regression or null result was
reverted or reported honestly, never hidden; negative results are kept in
this ledger, not deleted.
