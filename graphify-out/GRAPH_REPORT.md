# Graph Report - GPU-Resident-Library  (2026-11-04)

## Corpus Check
- 234 files · ~300,040 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 7 file(s) not represented in the graph (top: (none) 4, .jsonl 3)

## Summary
- 2539 nodes · 5563 edges · 162 communities (156 shown, 6 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 672 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f20fa882`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- keyswitch_resident.h
- keyswitch.rs
- evalmult.rs
- keyswitch.h
- cuda_runtime
- bench_rescale_cpp_vs_rust.cpp
- DeviceCiphertext
- Tab
- DeviceKSContext
- encrypt_host_shoup_pooled_barrett.cpp
- bench_encrypt_profile.cpp
- cstdint
- RingDimension
- rescale.rs
- EvalMultFixtureCpp
- keyswitch_bridge.cpp
- CkksError
- KeySwitchConstants
- bench_evalmult.rs
- bench_encrypt_registered_lifecycle.cpp
- rotate.cpp
- test_registered_host_buffer.cpp
- EvalMultFixtureCpp
- encrypt_host_shoup_pooled_barrett_prewarm.cpp
- Problem
- Security assessment — GPU-Resident-Library
- NttContextCpp
- bench_pinned_alloc_diagnostic3.cpp
- encrypt_host_shoup_pooled_barrett_batched_registered.cpp
- bench_embed_barrett_vs_reference.cpp
- bench_thp_diagnostic.cpp
- LaunchNTT_CT
- native_primes
- ScopedKSWork
- encrypt_host_shoup_pooled_barrett_batched
- encrypt_host_shoup_pooled_barrett_batched_pinned.cpp
- ks_context_create
- check_hybrid_keyswitch_params
- Sample
- test_keyswitch_stage_trace.cpp
- Stats
- bench_pinned_alloc_diagnostic.cpp
- intt_ct.cu
- test_evalkeygen_isolated.cpp
- test_rescale_double_call_n1024.cpp
- bench_alloc_path_comparison.cpp
- bench_encrypt_prewarm_workloads.cpp
- xform
- Fixture
- KeySwitchFixtureCpp
- Sample
- Priority queue (highest value first)
- Tab
- encrypt_host_shoup_pooled
- Fixture
- Fixture
- Fixture
- Fixture
- test_ntt_roundtrip_n1024.cpp
- Stats
- DeviceKSWork
- ErrStats
- Fixture
- Fixture
- Fixture
- Fixture
- test_twsweep.cpp
- Master state
- encrypt_host
- rns_arith.cu
- decode_host
- encrypt_host_shoup.cpp
- Memory feasibility — measured, not estimated
- test_cheb.cpp
- Tab
- test_embed_barrett_matches_reference.cpp
- Stats
- Stats
- Stats
- Stats
- Stats
- Stats
- Stats
- Stats
- LaunchScaledAddInto
- compute_keyswitch_constants
- Tab
- build.rs
- ks_fastcore.cu
- Next session — exact continuation
- EncryptBufPool
- test_evalkeygen.cpp
- Sample
- EventTimer
- bench_hoist_multi.cpp
- RegisteredHostBuffer
- 2. Main experiment: the embed loop
- EncryptBufPoolRegistered
- Graphify setup, usage, and validation (Phase 21)
- test_encrypt_host_fast_matches_sequential.cpp
- test_n131072_cache_churn.cpp
- gpu_rescale_component
- Path
- bench_evalmult.cpp
- boot
- basis_convert.cu
- Phase 19: isolating CUDA pinned-allocation overhead -- mechanistically PROVEN to be dominated by a single `ioctl()` into the NVIDIA driver on every call (not a one-time cost); `cudaHostRegister` on pre-touched memory is 1.5x-4x cheaper than `cudaMallocHost`/`cudaHostAlloc` for the full lifecycle, independently confirmed via NVIDIA's own tracer
- EncryptBufPoolPinned
- DevTab
- DevTab
- test_keygen.cpp
- RootTab
- Retained
- boot
- Fixture
- ckks-cuda/src/lib.rs
- automorph_eval.cu
- EncryptBufPoolBatched
- ptops.cpp
- DeviceTables
- LaunchINTT_GS
- Ct
- Phase 16: explicit pinned-memory prewarming -- built, proven correct and safe, measured honestly, RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE (not promoted)
- keygen.h
- Stats
- Ct
- device_ciphertext.cpp
- Phase 20: CUDA-registered host memory prototype -- built, correctness-verified (0 mismatches across 400+ checks), measured against the real encrypt/EvalMult pipeline, and RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE (not promoted)
- Performance baseline — full bootstrap, this session
- Phase 18: controlled THP experiment -- the system-wide toggle is blocked by permissions (correctly not worked around); a real but small THP-hint effect found at exactly 2MiB; Phase 17's "not CUDA-specific" conclusion is REVISED based on same-session interleaved re-testing
- test_keyreuse.cpp
- Scheme
- LaunchSwitchModulus
- interleaved_ab.sh
- randomized_ab.sh
- run_trials.sh
- test_moddown_crt_basis_isolated.cpp
- GPU-Resident-Library
- Post-quantum instant messaging — architecture proposal
- test_decompose_crt_basis_isolated.cpp
- EncryptBufPool
- This round: production integration, honest A/B, security re-review
- test_chain.cpp
- build_device_tables
- run_config
- ckks-core
- Experiment ledger
- Architect — Planning Specialist
- Code Reviewer — Adversarial Specialist
- Scrum Master — Team Lead
- Scrum-Orchestrator session
- Literature review
- test_rescale_vector_diff_n1024.cpp
- Backend Developer — Systems Specialist
- Frontend Developer — Interface Specialist
- QA Engineer — Test Specialist
- Release Engineer — Delivery Specialist
- Security Engineer — Defensive Specialist
- test_keyswitch_core_realct_diff.cpp
- CLAUDE.md
- .claude/CLAUDE.md

## God Nodes (most connected - your core abstractions)
1. `Round 2 — post-master-directive, in progress` - 61 edges
2. `encrypt_host()` - 56 edges
3. `DeviceKSContext` - 40 edges
4. `Master state` - 39 edges
5. `CkksError` - 37 edges
6. `EvalMultFixtureCpp` - 34 edges
7. `native_primes()` - 33 edges
8. `KeySwitchConstants` - 31 edges
9. `ks_context_create()` - 31 edges
10. `EvalMultFixtureCpp` - 28 edges

## Surprising Connections (you probably didn't know these)
- `Round 2 — post-master-directive, in progress` --references--> `gpufhe_ks_fixture_destroy()`  [INFERRED]
  research/EXPERIMENT_LEDGER.md → rust/crates/ckks-cuda/cpp_bridge/keyswitch_bridge.cpp
- `Round 2 — post-master-directive, in progress` --references--> `gpufhe_ks_fixture_secret_key()`  [INFERRED]
  research/EXPERIMENT_LEDGER.md → rust/crates/ckks-cuda/cpp_bridge/keyswitch_bridge.cpp
- `Known extraction gaps (honestly documented, not hidden)` --references--> `encrypt_host_shoup_pooled_barrett_batched()`  [INFERRED]
  research/GRAPHIFY_GUIDE.md → src/encrypt_host_shoup_pooled_barrett_batched.cpp
- `GOD PROMPT "CKKS PHASE 17": diagnosed Phase 16's unexplained allocation-cost variance — root mechanism demonstrated to be the Linux kernel's own memory-locking (`mlock()`) path, NOT CUDA-specific; no production change justified, Phase 16's decision stands on stronger evidence` --references--> `encrypt_host()`  [INFERRED]
  research/MASTER_STATE.md → src/encdec.cpp
- `Earlier commit (Phase 20)` --references--> `encrypt_host()`  [INFERRED]
  research/NEXT_SESSION.md → src/encdec.cpp

## Import Cycles
- None detected.

## Communities (162 total, 6 thin omitted)

### Community 0 - "keyswitch_resident.h"
Cohesion: 0.07
Nodes (35): cd, cd, main(), mmu(), rssMB(), smu(), main(), mmu() (+27 more)

### Community 1 - "keyswitch.rs"
Cohesion: 0.09
Nodes (32): addmod(), ERR_BUF_LEN, err_from_buf(), gpufhe_ks_fixture_alpha(), gpufhe_ks_fixture_create(), gpufhe_ks_fixture_destroy(), gpufhe_ks_fixture_eval_key_part(), gpufhe_ks_fixture_eval_key_towers() (+24 more)

### Community 2 - "evalmult.rs"
Cohesion: 0.13
Nodes (48): encode_real(), encrypt(), addmod(), assert_slots_close(), ckks_mods_and_roots(), coeff_to_eval(), control_a2_raw_tensor_matches_direct_ciphertext_product(), control_a2_raw_tensor_matches_direct_ciphertext_product_exactly() (+40 more)

### Community 3 - "keyswitch.h"
Cohesion: 0.04
Nodes (46): cd, cd, main(), mmu(), smu(), cd, main(), mm() (+38 more)

### Community 4 - "cuda_runtime"
Cohesion: 0.12
Nodes (6): clk, clk, main(), mm(), main(), mm()

### Community 5 - "bench_rescale_cpp_vs_rust.cpp"
Cohesion: 0.10
Nodes (24): bitrev(), build_tables(), forward_ntt_host(), invmod(), main(), median(), powmod(), report() (+16 more)

### Community 6 - "DeviceCiphertext"
Cohesion: 0.17
Nodes (10): DeviceCiphertext, d_c0_, d_c1_, towers_, Format, COEFFICIENT, EVALUATION, mul_resident() (+2 more)

### Community 7 - "Tab"
Cohesion: 0.20
Nodes (14): build_tab(), invmod(), main(), mulmod(), P_mod(), powmod(), Tab, fp (+6 more)

### Community 8 - "DeviceKSContext"
Cohesion: 0.05
Nodes (39): DeviceKSContext, complModHost, d_av, d_bv, d_complMod, d_fp, d_fr, d_ip (+31 more)

### Community 9 - "encrypt_host_shoup_pooled_barrett.cpp"
Cohesion: 0.18
Nodes (17): 4. A test bug caught and fixed (reported honestly, not hidden), am(), barrett_reduce_u64(), build_enc_tab_pooled_barrett(), emb_barrett(), EncDevTab, fp, fr (+9 more)

### Community 10 - "bench_encrypt_profile.cpp"
Cohesion: 0.08
Nodes (34): am(), build_enc_tab(), Clock, EncDevTab, fp, fr, encrypt_host_profiled(), EncryptStageTimes (+26 more)

### Community 11 - "cstdint"
Cohesion: 0.11
Nodes (3): am(), main(), mm()

### Community 12 - "RingDimension"
Cohesion: 0.10
Nodes (20): Modulus, RingDimension, cases(), cold_start_cost_probe(), gpufhe_native_root(), LaunchINTT_GS(), LaunchNTT_CT(), MODULI (+12 more)

### Community 13 - "rescale.rs"
Cohesion: 0.18
Nodes (18): cold_start_cost_probe(), forward_ntt(), gpufhe_native_rescale_consts(), invalid_input_length_is_rejected(), LaunchINTT_GS(), LaunchNTT_CT(), LaunchRescaleFuse(), LaunchSwitchModulus() (+10 more)

### Community 14 - "EvalMultFixtureCpp"
Cohesion: 0.06
Nodes (27): EvalMultFixtureCpp, ba0, ba1, C, c0a, c0b, c1a, c1b (+19 more)

### Community 15 - "keyswitch_bridge.cpp"
Cohesion: 0.07
Nodes (16): gpufhe_ks_fixture_create(), gpufhe_ks_fixture_destroy(), gpufhe_ks_fixture_run_reference(), gpufhe_ks_fixture_run_resident(), gpufhe_ks_fixture_secret_key(), KsFixture, C, context_created (+8 more)

### Community 16 - "CkksError"
Cohesion: 0.11
Nodes (12): CkksError, Cuda, InvalidModulus, InvalidRingDimension, decode(), decrypt(), gpufhe_decode(), gpufhe_decrypt() (+4 more)

### Community 17 - "KeySwitchConstants"
Cohesion: 0.06
Nodes (33): KeySwitchConstants, alpha, av, bv, evalKeyTowers, fullQ, mdBMuHi, mdBMuLo (+25 more)

### Community 18 - "bench_evalmult.rs"
Cohesion: 0.12
Nodes (20): deterministic_input(), main(), median(), N, NUM_PART, report(), SIZE_P, SIZE_Q (+12 more)

### Community 19 - "bench_encrypt_registered_lifecycle.cpp"
Cohesion: 0.11
Nodes (24): Clock, encrypt_registered_fresh_per_call(), Fixture, m, mod, n, pkA, pkB (+16 more)

### Community 20 - "rotate.cpp"
Cohesion: 0.11
Nodes (21): LaunchAutomorphPermute(), This cycle's real, validated outcomes, Root cause, confirmed by direct instrumentation, This round's real, delivered results (commits `93f2231`, `0043af3`), automorphism_eval_device(), automorphism_eval_fast_device(), automorphism_eval_host(), autoPermMap() (+13 more)

### Community 21 - "test_registered_host_buffer.cpp"
Cohesion: 0.30
Nodes (13): GOD PROMPT "CKKS PHASE 20": CUDA-registered host memory PROTOTYPE built, correctness-verified (0 mismatches across 400+ checks), measured against the real encrypt/EvalMult pipeline — RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE, NOT promoted to production, Ownership model, designed before implementation (directive Section 2), main(), scenario10_shutdown_while_draining(), scenario12_cleanup_after_failed_async_op(), scenario1_basic_lifecycle(), scenario2_repeated_alloc_free_cycles(), scenario3_repeated_reuse() (+5 more)

### Community 22 - "EvalMultFixtureCpp"
Cohesion: 0.07
Nodes (26): EvalMultFixtureCpp, ba0, ba1, C, c0a, c0b, c1a, c1b (+18 more)

### Community 23 - "encrypt_host_shoup_pooled_barrett_prewarm.cpp"
Cohesion: 0.05
Nodes (56): GOD PROMPT "CKKS PHASE 16": built, proved correct, and concurrency-hardened an explicit pinned-memory prewarming API — found the allocation cost itself is unpredictable in this environment, so retained it as an opt-in experimental feature rather than promoting it to production, READ THIS SIXTH if resuming into performance work: a pinned-memory PREWARMING API exists, fully correct and concurrency-safe, but was NOT promoted because its own allocation cost is unpredictable in this environment -- Phase 20 (top of this file) built a registered-memory alternative that resolves this exact weakness, Benchmark artifact check (Hypothesis E), Decision (directive Step 8's three categories), Decision (directive Step 9), Engineering response (directive Step 6), Experiment 1: fresh-allocate-then-free, 10 repeats per size, one process, Experiment 2: never-free, matching production's actual behavior (+48 more)

### Community 24 - "Problem"
Cohesion: 0.18
Nodes (11): cd, clk, compareResult(), EL(), main(), Problem, lgS, M (+3 more)

### Community 25 - "Security assessment — GPU-Resident-Library"
Cohesion: 0.07
Nodes (28): 1. Re-derivation of `hybrid()`'s math against the published literature — WEAKER than stated, 2. The 114-bit large-QRAM result — material, and the "conservative" 130-bit bound has an inconsistency problem, 3. SageMath lattice-estimator cross-check — NOT done, genuinely unresolved, 4. Modulus-chain correctness and precision — theoretically sound, EMPIRICALLY UNVERIFIED at n=131072, 5. Resource feasibility vs. the security requirement — real tension, not yet resolved, Corrected recommendation: n=131072, KEEP the existing sparse secret (h=64), Correction to the first-pass remediation estimate: n=131072 was the wrong target, Current parameters, measured (+20 more)

### Community 26 - "NttContextCpp"
Cohesion: 0.12
Nodes (18): bitrev(), build_tables(), invmod(), main(), median(), NttContextCpp, d_fp, d_fr (+10 more)

### Community 27 - "bench_pinned_alloc_diagnostic3.cpp"
Cohesion: 0.13
Nodes (16): Clock, main(), meminfo_field(), print_sample(), report(), Stats, max_, mean (+8 more)

### Community 28 - "encrypt_host_shoup_pooled_barrett_batched_registered.cpp"
Cohesion: 0.17
Nodes (18): am(), barrett_reduce_u64(), build_enc_tab_registered(), emb_barrett(), EncDevTab, fp, fr, encrypt_host_shoup_pooled_barrett_batched_registered() (+10 more)

### Community 29 - "bench_embed_barrett_vs_reference.cpp"
Cohesion: 0.29
Nodes (11): emb_barrett(), emb_old(), main(), make_tb(), report(), stats_of(), TowerBarrett, mu (+3 more)

### Community 30 - "bench_thp_diagnostic.cpp"
Cohesion: 0.16
Nodes (8): Clock, main(), test_size(), us_between(), Clock, main(), one_call(), us()

### Community 31 - "LaunchNTT_CT"
Cohesion: 0.26
Nodes (11): ct_shared_tail(), ct_shared_tail_b(), ct_stage(), ct_stage_b(), ilog2_u(), LaunchNTT_CT(), LaunchNTT_CT_Batched(), shoup_mulmod() (+3 more)

### Community 32 - "native_primes"
Cohesion: 0.09
Nodes (28): LaunchApproxSwitchCRTBasis(), LaunchModDownCombine(), GOD PROMPT "CKKS PHASE 8": entry #40's `evalkeygen_host` diagnosis DISPROVEN (a sign bug in its own test) — the real EvalMult blocker is pinpointed to a `sizeP=2`-specific defect in the native key-switch reconstruction (`compute_keyswitch_constants`+`keyswitch_core_resident`), the exact P-tower count the real Rust fixture uses, GOD PROMPT "CKKS PHASE 9": the real encrypted EvalMult round trip PASSES end-to-end for the first time (since Phase 5) — entry #41's "sizeP=2-specific defect" framing is SUPERSEDED; the real cause was a test-fixture parameter choice (NUM_PART=2 violating the Hybrid key-switch's P>=Q_part_max requirement), not a code defect, and it is fixed, GOD PROMPT "CKKS RUST + CUDA PHASE 3": third operation (full Hybrid key switching) ported, bit-exact vs. the established reference, a real latent C++ bug found and guarded against, memory measured not guessed, GOD PROMPT "RESOLVE POLYEVAL CORRECTNESS" round: every library-defect hypothesis for test_polyeval refuted by direct evidence; one formula proven wrong, not the library, Memory and benchmarks: deferred this round for the real-encryption primitives (entry #38), Memory: what this round measured (entry #36) (+20 more)

### Community 33 - "ScopedKSWork"
Cohesion: 0.16
Nodes (7): ScopedKSContext, ctx_, ScopedKSWork, work_, gpuMem(), main(), mm()

### Community 34 - "encrypt_host_shoup_pooled_barrett_batched"
Cohesion: 0.20
Nodes (16): am(), barrett_reduce_u64(), build_enc_tab_batched(), emb_barrett(), EncDevTab, fp, fr, encrypt_host_shoup_pooled_barrett_batched() (+8 more)

### Community 35 - "encrypt_host_shoup_pooled_barrett_batched_pinned.cpp"
Cohesion: 0.17
Nodes (17): Correctness, independently verified (directive Section 5), am(), barrett_reduce_u64(), build_enc_tab_pinned(), emb_barrett(), EncDevTab, fp, fr (+9 more)

### Community 36 - "ks_context_create"
Cohesion: 0.15
Nodes (31): The bigger, still-open finding: GPU clock behavior, LaunchINTT_GS_Batched(), Round 2 — post-master-directive, in progress, GOD PROMPT round: large-n gate re-checked (still NO-GO), bench_evalmult_resident root-caused and fixed, test_polyeval investigated (inconclusive, reverted), Later round still: GPU-memory lifecycle isolated with real production APIs; external-kill correlation directly demonstrated; NO-GO decision for the next full attempt under current host conditions, A real, tiny, confirmed leak found and fixed (not the explanation for the MB-scale growth above), Experiment 2: enhanced instrumentation, 5th real n=131072 attempt, Go/no-go decision for the next full n=131072 attempt (+23 more)

### Community 37 - "check_hybrid_keyswitch_params"
Cohesion: 0.23
Nodes (13): HybridKsValidation, ok, p_decimal, q_part_max_decimal, worst_part, BigUint, check_hybrid_keyswitch_params(), compare_big() (+5 more)

### Community 38 - "Sample"
Cohesion: 0.12
Nodes (16): Sample, alloc_us, bytes, cudaPtrType, errorCode, free_us, memAvail_kB, memFree_kB (+8 more)

### Community 39 - "test_keyswitch_stage_trace.cpp"
Cohesion: 0.09
Nodes (32): invmod(), keygen_host(), mk_tab(), mulmod(), ntt_tower(), powmod(), submod(), Tab (+24 more)

### Community 40 - "Stats"
Cohesion: 0.29
Nodes (7): Stats, max_, mean, min_, n, p50, p95

### Community 41 - "bench_pinned_alloc_diagnostic.cpp"
Cohesion: 0.08
Nodes (25): Clock, main(), measure_size(), report(), Stats, max_, mean, min_ (+17 more)

### Community 42 - "intt_ct.cu"
Cohesion: 0.46
Nodes (7): gs_shared_head(), gs_shared_head_b(), gs_stage(), gs_stage_b(), scale_ninv(), scale_ninv_b(), shoup_mulmod()

### Community 43 - "test_evalkeygen_isolated.cpp"
Cohesion: 0.23
Nodes (14): addmod(), build_inv_tab(), inverse_ntt(), invmod(), InvTab, ip, ir, ninv (+6 more)

### Community 44 - "test_rescale_double_call_n1024.cpp"
Cohesion: 0.40
Nodes (6): bitrev(), check(), forward_ntt(), main(), make_case(), upload()

### Community 45 - "bench_alloc_path_comparison.cpp"
Cohesion: 0.24
Nodes (11): Clock, main(), meminfo_field(), path_name(), query_pointer_type(), release_retained(), report(), run_one() (+3 more)

### Community 46 - "bench_encrypt_prewarm_workloads.cpp"
Cohesion: 0.19
Nodes (11): main(), median_of(), report(), Stats, max_, mean, min_, n (+3 more)

### Community 47 - "xform"
Cohesion: 0.27
Nodes (11): am(), build_enc_tab(), decrypt_host(), EncDevTab, fp, fr, invmod(), mm() (+3 more)

### Community 48 - "Fixture"
Cohesion: 0.20
Nodes (11): checkOne(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 49 - "KeySwitchFixtureCpp"
Cohesion: 0.18
Nodes (8): KeySwitchFixtureCpp, C, K, W, main(), median(), mulmod_b(), report()

### Community 50 - "Sample"
Cohesion: 0.17
Nodes (12): Sample, allocIndex, anonHuge_kB, bytes, firsttouch_us, liveBefore, memAvailable_kB, memFree_kB (+4 more)

### Community 51 - "Priority queue (highest value first)"
Cohesion: 0.21
Nodes (11): HoistedDigits, d_compl, n, numPart, sizeQlP, Priority queue (highest value first), hoist_digits_create(), hoist_digits_destroy() (+3 more)

### Community 52 - "Tab"
Cohesion: 0.18
Nodes (10): main(), mk_tab(), Tab, fp, fr, ip, ir, ninv (+2 more)

### Community 53 - "encrypt_host_shoup_pooled"
Cohesion: 0.27
Nodes (10): am(), build_enc_tab_pooled(), EncDevTab, fp, fr, encrypt_host_shoup_pooled(), mulmod_cold(), shoup_mulmod() (+2 more)

### Community 54 - "Fixture"
Cohesion: 0.18
Nodes (11): checkOne(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 55 - "Fixture"
Cohesion: 0.20
Nodes (11): checkOne(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 56 - "Fixture"
Cohesion: 0.20
Nodes (11): checkOne(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 57 - "Fixture"
Cohesion: 0.20
Nodes (11): checkOne(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 58 - "test_ntt_roundtrip_n1024.cpp"
Cohesion: 0.29
Nodes (11): bitrev(), build_cases(), build_tables(), Case, name, poly, invmod(), main() (+3 more)

### Community 59 - "Stats"
Cohesion: 0.18
Nodes (11): cd, main(), report(), Stats, max_, mean, min_, n (+3 more)

### Community 60 - "DeviceKSWork"
Cohesion: 0.15
Nodes (13): DeviceKSWork, d_compl, d_digPtrs, d_nInv, d_nInvPrecon, d_part, d_pwork, d_qsw (+5 more)

### Community 61 - "ErrStats"
Cohesion: 0.29
Nodes (9): checkDecode(), checkEncode(), compare_complex(), compare_int64(), ErrStats, maxAbs, maxRel, rms (+1 more)

### Community 62 - "Fixture"
Cohesion: 0.20
Nodes (11): check(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 63 - "Fixture"
Cohesion: 0.22
Nodes (11): check(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 64 - "Fixture"
Cohesion: 0.22
Nodes (11): check(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 65 - "Fixture"
Cohesion: 0.21
Nodes (11): check(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 66 - "test_twsweep.cpp"
Cohesion: 0.60
Nodes (4): am(), cd, main(), mm()

### Community 67 - "Master state"
Cohesion: 0.07
Nodes (35): LaunchRescaleFuse(), combine_kernel(), LaunchCombine(), LaunchTensor(), mulmod64(), tensor_kernel(), Measured, not estimated: token-reduction and timing, Validation: 12 questions, tested against real evidence (+27 more)

### Community 68 - "encrypt_host"
Cohesion: 0.09
Nodes (32): 1. Buffer-pool safety audit (directive Section 3), 3. Independent differential-test results (directive Section 6: built BEFORE touching production), 5. Benchmarks, 6. Security review, 7. What remains the dominant cost, post-Phase-14, Embed loop alone (`bench/bench_embed_barrett_vs_reference.cpp`, no GPU, interleaved, 9 trials x 20 reps), Embed modular arithmetic derivation and buffer-pool safety audit (Phase 14), Full `encrypt_host` (unified 3-way interleaved, `bench/bench_encrypt_phase14_all_variants.cpp`, 9 trials x 20 reps, run twice independently) (+24 more)

### Community 69 - "rns_arith.cu"
Cohesion: 0.19
Nodes (14): ct_tensor_mont_kernel(), LaunchCtTensorBatched(), LaunchRNSMultTower(), mont_mul(), mont_qinv(), mont_r2(), mont_reduce(), rns_mult_mont_kernel() (+6 more)

### Community 70 - "decode_host"
Cohesion: 0.10
Nodes (31): 1. The exact convention already in this repository (authoritative), 2. Derivation: why a single size-`M=2N` FFT suffices, 3. Why this is NOT the same permutation as a textbook FFT's bit-reversal, 4. Independent verification BEFORE writing any C++ (Python/numpy), 5. C++ implementation (`src/encode_fft.cpp`), 6. Numerical correctness results (`tests/test_encode_fft_matches_reference.cpp`), 7. Real cryptographic correctness, 8. Benchmark results (+23 more)

### Community 71 - "encrypt_host_shoup.cpp"
Cohesion: 0.23
Nodes (11): GOD PROMPT "CKKS PHASE 13": deep-profiled `encrypt_host` — NOT the already-batched NTT, but two host-side loops (embed + combine) dominate; combine's modular multiply was calling libgcc's slow `__umodti3`; fixed with Shoup modmul + a device-buffer pool, catching and fixing a real concurrency bug along the way — 1.10x-1.84x faster, zero correctness regression, am(), build_enc_tab_shoup(), EncDevTab, fp, fr, encrypt_host_shoup(), mulmod_cold() (+3 more)

### Community 72 - "Memory feasibility — measured, not estimated"
Cohesion: 0.07
Nodes (33): GOD PROMPT round: test_polyeval's real root cause found — a genuine library bug in the native rescale pathway at small n, never a test-logic issue, Caught in the act: an ordinary, previously-reliable `test_boot` (n=8192) run was itself OOM-killed minutes later, with WSL reporting "10Gi free" throughout, Experiment 1: `tests/test_n131072_cache_churn.cpp` (new, committed), GPU VRAM at n=131072 (for completeness, now secondary), growth source exists BEYOND KROT/KSC; Objective C is NOT complete, Memory feasibility — measured, not estimated, not WSL2's self-reported 10GB ceiling, Priority 2 result: what is PROVEN, what is HYPOTHESIS, what remains UNKNOWN about the kill mechanism (+25 more)

### Community 73 - "test_cheb.cpp"
Cohesion: 0.21
Nodes (10): am(), cd, Ct, c0, c1, scale, tw, main() (+2 more)

### Community 74 - "Tab"
Cohesion: 0.18
Nodes (10): main(), mk_tab(), Tab, fp, fr, ip, ir, ninv (+2 more)

### Community 75 - "test_embed_barrett_matches_reference.cpp"
Cohesion: 0.38
Nodes (11): barrett_reduce_u64(), checkModulus(), emb_fast(), emb_ref(), main(), make_tb(), TowerBarrett, mu (+3 more)

### Community 76 - "Stats"
Cohesion: 0.18
Nodes (11): Stats, cv, max_, mean, min_, n, p50, p90 (+3 more)

### Community 77 - "Stats"
Cohesion: 0.20
Nodes (10): main(), report(), Stats, max_, mean, min_, n, stats_of() (+2 more)

### Community 78 - "Stats"
Cohesion: 0.20
Nodes (10): main(), report(), Stats, max_, mean, min_, n, stats_of() (+2 more)

### Community 79 - "Stats"
Cohesion: 0.20
Nodes (10): main(), report(), Stats, max_, mean, min_, n, stats_of() (+2 more)

### Community 80 - "Stats"
Cohesion: 0.20
Nodes (10): main(), report(), Stats, max_, mean, min_, n, stats_of() (+2 more)

### Community 81 - "Stats"
Cohesion: 0.18
Nodes (11): Stats, cv, max_, mean, min_, n, p50, p90 (+3 more)

### Community 82 - "Stats"
Cohesion: 0.20
Nodes (10): main(), report(), Stats, max_, mean, min_, n, stats_of() (+2 more)

### Community 83 - "Stats"
Cohesion: 0.20
Nodes (10): main(), report(), Stats, max_, mean, min_, n, stats_of() (+2 more)

### Community 84 - "LaunchScaledAddInto"
Cohesion: 0.28
Nodes (4): LaunchAddInto(), LaunchPtMulAcc(), LaunchScaledAddInto(), main()

### Community 85 - "compute_keyswitch_constants"
Cohesion: 0.33
Nodes (7): GOD PROMPT "CKKS PHASE 10": repository-wide parameter audit + centralized, exact enforcement of the Hybrid key-switch's P>=Q_part_max requirement — four more invalid fixtures found and fixed, keyswitch.rs's own fixture corrected with a new decrypt-based test, invalid configurations now REJECTED at construction time, compute_keyswitch_constants(), invmod(), mulmod(), powmod(), precon(), prod_mod()

### Community 86 - "Tab"
Cohesion: 0.22
Nodes (10): build_inv_tab(), inverse_ntt(), invmod(), powmod(), Tab, ip, ir, ninv (+2 more)

### Community 87 - "build.rs"
Cohesion: 0.33
Nodes (5): compile_cpp(), compile_cuda(), main(), repo_root(), run_or_panic()

### Community 88 - "ks_fastcore.cu"
Cohesion: 0.24
Nodes (6): ks_macc_kernel(), LaunchKSMultAcc(), LaunchKSMultAccBatched(), mulmod64(), main(), up()

### Community 89 - "Next session — exact continuation"
Cohesion: 0.11
Nodes (17): Before trusting any new benchmark number, Current commit, Earlier commit (Phase 20), Earlier commit still (Phase 19), `-G` build-flag bug: CLOSED for the four originally-flagged benchmarks; one remaining sweep still open, Next session — exact continuation, RAII wrappers: isolated validation complete, production integration is the next concrete step (once memory allows), Randomized hoisting A/B: script ready, exact command to run once the memory gate clears (+9 more)

### Community 90 - "EncryptBufPool"
Cohesion: 0.22
Nodes (9): EncryptBufPool, d_e0, d_e1, d_fp, d_fr, d_m, d_mod, d_v (+1 more)

### Community 91 - "test_evalkeygen.cpp"
Cohesion: 0.39
Nodes (7): addmod_local(), invmod(), main(), mm(), powmod(), sub(), up()

### Community 92 - "Sample"
Cohesion: 0.25
Nodes (7): Sample, n, path, phase, sizeQ, total_us, wallclock_ms

### Community 93 - "EventTimer"
Cohesion: 0.25
Nodes (3): EventTimer, a, b

### Community 94 - "bench_hoist_multi.cpp"
Cohesion: 0.18
Nodes (12): bootstrap(), clk, main(), median(), mm(), modinv(), quantile(), cd (+4 more)

### Community 95 - "RegisteredHostBuffer"
Cohesion: 0.20
Nodes (8): RegisteredBufferState, Allocated, Empty, Failed, Registered, RegisteredHostBuffer, lastError_, ptr_

### Community 96 - "2. Main experiment: the embed loop"
Cohesion: 0.40
Nodes (5): 2. Main experiment: the embed loop, Assembly inspection (directive: "inspect optimized assembly rather than assuming"), Derivation: per-tower Barrett reduction, Operand ranges (determined, not assumed), Signedness and overflow analysis

### Community 97 - "EncryptBufPoolRegistered"
Cohesion: 0.25
Nodes (8): EncryptBufPoolRegistered, d_fp, d_fr, d_mod, d_packed, h_packed, ready, sizeQ

### Community 98 - "Graphify setup, usage, and validation (Phase 21)"
Cohesion: 0.18
Nodes (10): Generating and refreshing the graph, Graphify setup, usage, and validation (Phase 21), How Claude Code should use this (Layer D, kept short deliberately), How to disable or remove the integration, If Graphify is missing or broken in a future session, Installation and integration performed this phase, Known extraction gaps (honestly documented, not hidden), Privacy decision: code-only, zero LLM backend (+2 more)

### Community 99 - "test_encrypt_host_fast_matches_sequential.cpp"
Cohesion: 0.39
Nodes (7): am(), invmod(), main(), mm(), oracle_encrypt_host(), powmod(), up()

### Community 100 - "test_n131072_cache_churn.cpp"
Cohesion: 0.39
Nodes (6): clk, EL(), gpuMemMB(), main(), mm(), rssMB()

### Community 101 - "gpu_rescale_component"
Cohesion: 0.17
Nodes (10): build_tables(), gpu_rescale_component(), main(), Tables, fp, fr, ip, ir (+2 more)

### Community 102 - "Path"
Cohesion: 0.29
Nodes (7): Path, A_Pageable, B_MlockPure, C_MallocHost, D2_HostAllocWC, D_HostAllocDefault, E_HostRegister

### Community 103 - "bench_evalmult.cpp"
Cohesion: 0.43
Nodes (5): am(), clk, main(), median(), mm()

### Community 104 - "boot"
Cohesion: 0.42
Nodes (6): boot(), clk, main(), median(), mm(), quantile()

### Community 106 - "basis_convert.cu"
Cohesion: 0.83
Nodes (3): approx_switch_kernel(), barrett_u128_mod_u64(), shoup_mulmod()

### Community 107 - "Phase 19: isolating CUDA pinned-allocation overhead -- mechanistically PROVEN to be dominated by a single `ioctl()` into the NVIDIA driver on every call (not a one-time cost); `cudaHostRegister` on pre-touched memory is 1.5x-4x cheaper than `cudaMallocHost`/`cudaHostAlloc` for the full lifecycle, independently confirmed via NVIDIA's own tracer"
Cohesion: 0.15
Nodes (13): A genuinely new, unexplained finding: plain pageable `malloc`+`memset` ALSO jumps sharply at exactly 2MiB, Decision, Decisive finding 1: `cudaMallocHost` and `cudaHostAlloc(cudaHostAllocDefault)` are statistically indistinguishable -- confirmed empirically, not merely assumed from documentation, Decisive finding 2: the cost is syscall-level, mechanistic, and almost entirely inside a single `ioctl()` -- confirmed by NVIDIA's own tracer, not by this project's own timers, Decisive finding 3: `cudaHostRegister` on pre-touched memory is 1.5x-4x cheaper than `cudaMallocHost`/`cudaHostAlloc` for the FULL lifecycle -- reproduced across 4 independent fresh-process runs, Phase 19: isolating CUDA pinned-allocation overhead -- mechanistically PROVEN to be dominated by a single `ioctl()` into the NVIDIA driver on every call (not a one-time cost); `cudaHostRegister` on pre-touched memory is 1.5x-4x cheaper than `cudaMallocHost`/`cudaHostAlloc` for the full lifecycle, independently confirmed via NVIDIA's own tracer, Regression and environment, Section 2 audit: a real methodology gap in Phases 16-18's own measurements (+5 more)

### Community 108 - "EncryptBufPoolPinned"
Cohesion: 0.29
Nodes (7): EncryptBufPoolPinned, d_fp, d_fr, d_mod, d_packed, h_packed, sizeQ

### Community 109 - "DevTab"
Cohesion: 0.29
Nodes (7): DevTab, fp, fr, ip, ir, ninv, ninv_p

### Community 110 - "DevTab"
Cohesion: 0.29
Nodes (7): DevTab, fp, fr, ip, ir, ninv, ninv_p

### Community 111 - "test_keygen.cpp"
Cohesion: 0.48
Nodes (5): invmod(), main(), mm(), powmod(), up()

### Community 112 - "RootTab"
Cohesion: 0.20
Nodes (10): build_tab(), invmod(), powmod(), RootTab, fp, fr, ip, ir (+2 more)

### Community 113 - "Retained"
Cohesion: 0.33
Nodes (6): Retained, bytes, cudaFreeHostNotPlainFree, path, ptr, wasRegistered

### Community 114 - "boot"
Cohesion: 0.53
Nodes (4): boot(), main(), median(), quantile()

### Community 115 - "Fixture"
Cohesion: 0.22
Nodes (11): check(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 116 - "ckks-cuda/src/lib.rs"
Cohesion: 0.31
Nodes (3): main(), median(), report()

### Community 118 - "EncryptBufPoolBatched"
Cohesion: 0.17
Nodes (11): 1. Existing transfer data flow (traced before changing anything), 2. Experiment 1 (KEPT): batched transfers, pageable memory, 4. Security review, 5. What remains the dominant cost, post-Phase-15, Batched CUDA transfer optimization for encrypt_host (Phase 15), EncryptBufPoolBatched, d_fp, d_fr (+3 more)

### Community 119 - "ptops.cpp"
Cohesion: 0.53
Nodes (4): am(), ct_add_ct_host(), ct_mul_pt_host(), mm()

### Community 120 - "DeviceTables"
Cohesion: 0.22
Nodes (9): DeviceTables, d_fprecon, d_froots, d_iprecon, d_iroots, n, ninv, ninv_precon (+1 more)

### Community 121 - "LaunchINTT_GS"
Cohesion: 0.25
Nodes (7): ilog2_u(), LaunchINTT_GS(), main(), mulmod(), powmod(), main(), main()

### Community 122 - "Ct"
Cohesion: 0.33
Nodes (5): Ct, c0, c1, scale, tw

### Community 123 - "Phase 16: explicit pinned-memory prewarming -- built, proven correct and safe, measured honestly, RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE (not promoted)"
Cohesion: 0.25
Nodes (8): Concurrency and lifecycle (directive Step 4), Correctness (directive Step 3/7), Lifecycle design (directive Step 2, decided BEFORE implementation), Memory overhead (directive Step 8), Phase 16: explicit pinned-memory prewarming -- built, proven correct and safe, measured honestly, RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE (not promoted), Verified starting state, Workload-level benchmark (directive Step 3/5), Workload profile comparison (directive Step 5)

### Community 124 - "keygen.h"
Cohesion: 0.10
Nodes (22): DeviceKSContext, DeviceKSWork, KeyPairHost, pkA, pkB, s, KeySwitchConstants, am() (+14 more)

### Community 125 - "Stats"
Cohesion: 0.29
Nodes (7): Stats, max_, mean, min_, n, p50, p95

### Community 126 - "Ct"
Cohesion: 0.33
Nodes (5): Ct, c0, c1, scale, tw

### Community 127 - "device_ciphertext.cpp"
Cohesion: 0.48
Nodes (4): ck(), DeviceCiphertext::DeviceCiphertext(), DeviceCiphertext::to_host(), DeviceCiphertext::upload()

### Community 128 - "Phase 20: CUDA-registered host memory prototype -- built, correctness-verified (0 mismatches across 400+ checks), measured against the real encrypt/EvalMult pipeline, and RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE (not promoted)"
Cohesion: 0.18
Nodes (11): CMake/CTest integration (directive Section 9), Concurrency audit (directive Section 4/8), Environment re-verified (not assumed), Full lifecycle measurement (directive Section 6), Nsight Systems confirmation of the actual driver-call reduction (directive Section 6/12, completion criterion 8), Phase 20: CUDA-registered host memory prototype -- built, correctness-verified (0 mismatches across 400+ checks), measured against the real encrypt/EvalMult pipeline, and RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE (not promoted), Production impact assessment and decision (directive Section 10), Production-shaped prototype (directive Section 3) (+3 more)

### Community 129 - "Performance baseline — full bootstrap, this session"
Cohesion: 0.40
Nodes (4): Performance baseline — full bootstrap, this session, Security reality check (n=8192, log2(QP)=1775, sparse secret h=64), What the stage breakdown looks like at the most recent checkpoint, Where the time actually goes (profiled, not guessed)

### Community 130 - "Phase 18: controlled THP experiment -- the system-wide toggle is blocked by permissions (correctly not worked around); a real but small THP-hint effect found at exactly 2MiB; Phase 17's "not CUDA-specific" conclusion is REVISED based on same-session interleaved re-testing"
Cohesion: 0.18
Nodes (11): Decision, Phase 18: controlled THP experiment -- the system-wide toggle is blocked by permissions (correctly not worked around); a real but small THP-hint effect found at exactly 2MiB; Phase 17's "not CUDA-specific" conclusion is REVISED based on same-session interleaved re-testing, Step 1: baseline environment (read-only), Step 2/3: the system-wide THP toggle is blocked by permissions -- correctly NOT worked around, Step 4/5: `bench_pinned_alloc_diagnostic3.cpp` -- per-sample methodology under the unchanged THP=madvise setting, Step 4: `bench_thp_hint_by_size.cpp` -- the decisive, demonstrated THP-hint effect, Step 6/self-correction: Phase 17's "not CUDA-specific" conclusion does NOT hold up under same-session interleaved re-testing -- REVISED, Step 7: production implications (+3 more)

### Community 131 - "test_keyreuse.cpp"
Cohesion: 0.60
Nodes (4): am(), cd, main(), mm()

### Community 132 - "Scheme"
Cohesion: 0.40
Nodes (4): Scheme, BFV, BGV, CKKS

### Community 133 - "LaunchSwitchModulus"
Cohesion: 0.40
Nodes (3): LaunchSwitchModulus(), check_switch(), main()

### Community 134 - "interleaved_ab.sh"
Cohesion: 0.67
Nodes (3): PATH, run_one(), interleaved_ab.sh script

### Community 135 - "randomized_ab.sh"
Cohesion: 0.67
Nodes (3): PATH, run_one(), randomized_ab.sh script

### Community 136 - "run_trials.sh"
Cohesion: 0.67
Nodes (3): gpu_snapshot(), PATH, run_trials.sh script

### Community 137 - "test_moddown_crt_basis_isolated.cpp"
Cohesion: 0.35
Nodes (7): addmod(), main(), mulmod(), P_mod(), run_conversion(), submod(), up()

### Community 138 - "GPU-Resident-Library"
Cohesion: 0.20
Nodes (9): Benchmark methodology, Building, Design discipline, GPU-Resident-Library, Performance (RTX 2060 Max-Q, ring 32768), Proven components, Status: complete, self-contained, numerically-proven CKKS multiply, What's next (optional) (+1 more)

### Community 139 - "Post-quantum instant messaging — architecture proposal"
Cohesion: 0.20
Nodes (9): 0. The distinction that governs everything, 1. Key establishment: hybrid X25519 + ML-KEM-768, 2. Payload encryption: ChaCha20-Poly1305 or AES-256-GCM via HKDF, 3. Session design: Double Ratchet + KEM ratchet, 4. Authentication, 5. Operational concerns, 6. Latency: the quantitative case against CKKS, 7. Required before production (+1 more)

### Community 140 - "test_decompose_crt_basis_isolated.cpp"
Cohesion: 0.40
Nodes (7): addmod(), main(), mulmod(), Q_mod(), run_conversion(), submod(), up()

### Community 141 - "EncryptBufPool"
Cohesion: 0.22
Nodes (9): EncryptBufPool, d_e0, d_e1, d_fp, d_fr, d_m, d_mod, d_v (+1 more)

### Community 142 - "This round: production integration, honest A/B, security re-review"
Cohesion: 0.40
Nodes (5): 1. Hoisting wired into the live production bootstrap, behind a flag, 2. Honest, order-balanced interleaved A/B measurement — small, noisy effect, NOT a clean win, 3. Security re-review: adversarial check found the prior "130-bit conservative" framing overconfident, This round: production integration, honest A/B, security re-review, What's still open

### Community 143 - "test_chain.cpp"
Cohesion: 0.60
Nodes (4): am(), cd, main(), mm()

### Community 144 - "build_device_tables"
Cohesion: 0.83
Nodes (3): build_device_tables(), dev_upload(), host_tables()

### Community 145 - "run_config"
Cohesion: 0.39
Nodes (7): gpu_used_mb(), host_rss_mb(), main(), mulmod_b(), report(), run_config(), stats_of()

### Community 148 - "Architect — Planning Specialist"
Cohesion: 0.40
Nodes (4): Architect — Planning Specialist, Completion criteria, Deliverable, Responsibilities

### Community 149 - "Code Reviewer — Adversarial Specialist"
Cohesion: 0.40
Nodes (4): Code Reviewer — Adversarial Specialist, Completion criteria, Responsibilities, What you must not do

### Community 150 - "Scrum Master — Team Lead"
Cohesion: 0.40
Nodes (4): Board commands you will use, Completion criteria for your own work, Responsibilities, Scrum Master — Team Lead

### Community 151 - "Scrum-Orchestrator session"
Cohesion: 0.40
Nodes (4): Before anything else, Degraded mode, Running the session, Scrum-Orchestrator session

### Community 154 - "test_rescale_vector_diff_n1024.cpp"
Cohesion: 0.19
Nodes (10): Case, name, xs, ys, bitrev(), forward_ntt(), invmod(), main() (+2 more)

### Community 155 - "Backend Developer — Systems Specialist"
Cohesion: 0.50
Nodes (3): Backend Developer — Systems Specialist, Completion criteria, Responsibilities

### Community 156 - "Frontend Developer — Interface Specialist"
Cohesion: 0.50
Nodes (3): Completion criteria, Frontend Developer — Interface Specialist, Responsibilities

### Community 157 - "QA Engineer — Test Specialist"
Cohesion: 0.50
Nodes (3): Completion criteria, QA Engineer — Test Specialist, Responsibilities

### Community 158 - "Release Engineer — Delivery Specialist"
Cohesion: 0.50
Nodes (3): Completion criteria, Release Engineer — Delivery Specialist, Responsibilities

### Community 159 - "Security Engineer — Defensive Specialist"
Cohesion: 0.50
Nodes (3): Completion criteria, Responsibilities, Security Engineer — Defensive Specialist

### Community 160 - "test_keyswitch_core_realct_diff.cpp"
Cohesion: 0.83
Nodes (3): am(), main(), mm()

### Community 162 - "CLAUDE.md"
Cohesion: 0.50
Nodes (3): graphify, Scrum-Orchestrator, Session start (GOD PROMPT CKKS phases)

## Knowledge Gaps
- **619 isolated node(s):** `fp`, `fr`, `alloc_us`, `combine_us`, `download_us` (+614 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1055 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Round 2 — post-master-directive, in progress` connect `ks_context_create` to `bench_rescale_cpp_vs_rust.cpp`, `DeviceCiphertext`, `LaunchSwitchModulus`, `encrypt_host_shoup_pooled_barrett.cpp`, `keyswitch_bridge.cpp`, `CkksError`, `Experiment ledger`, `rotate.cpp`, `Security assessment — GPU-Resident-Library`, `LaunchNTT_CT`, `native_primes`, `check_hybrid_keyswitch_params`, `test_keyswitch_stage_trace.cpp`, `bench_alloc_path_comparison.cpp`, `xform`, `Priority queue (highest value first)`, `Master state`, `encrypt_host`, `decode_host`, `Memory feasibility — measured, not estimated`, `LaunchScaledAddInto`, `compute_keyswitch_constants`, `ks_fastcore.cu`, `test_evalkeygen.cpp`, `LaunchINTT_GS`?**
  _High betweenness centrality (0.268) - this node is a cross-community bridge._
- **What connects `fp`, `fr`, `alloc_us` to the rest of the system?**
  _619 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `keyswitch_resident.h` be split into smaller, more focused modules?**
  _Cohesion score 0.06711915535444947 - nodes in this community are weakly interconnected._
- **Why does `former_invalid_num_part_2_fixture_is_now_rejected()` connect `CkksError` to `evalmult.rs`, `ks_context_create`?**
  _High betweenness centrality (0.178) - this node is a cross-community bridge._
- **Should `keyswitch.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.08636363636363636 - nodes in this community are weakly interconnected._
- **Why does `encrypt_host()` connect `encrypt_host` to `test_registered_host_buffer.cpp`, `encrypt_host_shoup_pooled_barrett_prewarm.cpp`, `LaunchNTT_CT`, `native_primes`, `encrypt_host_shoup_pooled_barrett_batched_pinned.cpp`, `ks_context_create`, `xform`, `Fixture`, `Fixture`, `Fixture`, `Fixture`, `Fixture`, `Fixture`, `Fixture`, `Fixture`, `Fixture`, `Master state`, `decode_host`, `encrypt_host_shoup.cpp`, `Next session — exact continuation`, `test_encrypt_host_fast_matches_sequential.cpp`, `Fixture`, `EncryptBufPoolBatched`, `Phase 16: explicit pinned-memory prewarming -- built, proven correct and safe, measured honestly, RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE (not promoted)`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Should `evalmult.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.13208649912331968 - nodes in this community are weakly interconnected._