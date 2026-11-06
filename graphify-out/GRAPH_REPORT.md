# Graph Report - GPU-Resident-Library  (2026-11-05)

## Corpus Check
- 237 files · ~322,320 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 7 file(s) not represented in the graph (top: (none) 4, .jsonl 3)

## Summary
- 2689 nodes · 5933 edges · 159 communities (156 shown, 3 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 711 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b724cb1d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- keyswitch_resident.h
- keyswitch.rs
- evalmult.rs
- keygen.h
- test_boot_hoist_enabled.cpp
- bench_rescale_cpp_vs_rust.cpp
- DeviceCiphertext
- Tab
- DeviceKSContext
- encrypt_host_shoup_pooled_barrett.cpp
- bench_encrypt_profile.cpp
- vector
- RingDimension
- rescale.rs
- EvalMultFixtureCpp
- keyswitch_bridge.cpp
- encrypt_host
- KeySwitchConstants
- encrypt_host_shoup.cpp
- bench_encrypt_registered_lifecycle.cpp
- Round 2 — post-master-directive, in progress
- test_registered_host_buffer.cpp
- EvalMultFixtureCpp
- encrypt_host_shoup_pooled_barrett_prewarm.cpp
- Problem
- Security assessment — GPU-Resident-Library
- NttContextCpp
- bench_pinned_alloc_diagnostic3.cpp
- encrypt_host_shoup_pooled_barrett_batched_registered.cpp
- encdec.cpp
- bench_thp_diagnostic.cpp
- Fixture
- native_primes
- Fixture
- encrypt_host_shoup_pooled_barrett_batched
- encrypt_host_shoup_pooled_barrett_batched_pinned.cpp
- ks_context_create
- codec.rs
- Sample
- test_keyswitch_stage_trace.cpp
- Fixture
- bench_pinned_alloc_diagnostic.cpp
- intt_ct.cu
- test_evalkeygen_isolated.cpp
- test_rescale_double_call_n1024.cpp
- bench_alloc_path_comparison.cpp
- bench_encrypt_prewarm_workloads.cpp
- Phase 22: Rust/FFI integration of CUDA-registered host memory -- correctness/concurrency PROVEN at the FFI boundary for the first time; Phase 20's prewarming anomaly RESOLVED (a benchmark confound, not a real effect); registered path confirmed faster through the real Rust API too -- RETAINED AS OPT-IN, promotion criteria now fully satisfied except magnitude
- Fixture
- KeySwitchFixtureCpp
- rotation.rs
- Priority queue (highest value first)
- LaunchNTT_CT
- build_enc_tab_pooled
- Fixture
- Sample
- Fixture
- Fixture
- test_ntt_roundtrip_n1024.cpp
- Stats
- DeviceKSWork
- ErrStats
- Fixture
- bench_embed_barrett_vs_reference.cpp
- bootstrap
- Fixture
- Fixture
- Master state
- encrypt_host_shoup_pooled
- test_automorph_eval_fast.cpp
- FFT-based CKKS canonical-embedding encode/decode — derivation, implementation, and verification
- cuda_runtime
- Memory feasibility — measured, not estimated
- Phase 17: diagnosing the allocation-cost variance -- the mechanism is the Linux kernel's memory-locking path, NOT CUDA-specific, with host memory pressure a plausible but unconfirmed amplifier
- Tab
- Ct
- Stats
- Stats
- Stats
- Stats
- Stats
- Stats
- Stats
- Stats
- test_encode_decode_parallel_matches_sequential.cpp
- test_embed_barrett_matches_reference.cpp
- Tab
- build.rs
- KeyPairHost
- Next session — exact continuation
- EncryptBufPool
- test_evalkeygen.cpp
- Sample
- Rust/CUDA migration matrix
- rotation_bridge.cpp
- test_moddown_crt_basis_isolated.cpp
- test_decompose_crt_basis_isolated.cpp
- Fixture
- Graphify setup, usage, and validation (Phase 21)
- test_encrypt_host_fast_matches_sequential.cpp
- boot
- Tables
- EncryptBufPoolRegistered
- bench_evalmult.cpp
- test_cheb.cpp
- Performance baseline — full bootstrap, this session
- Tab
- Phase 19: isolating CUDA pinned-allocation overhead -- mechanistically PROVEN to be dominated by a single `ioctl()` into the NVIDIA driver on every call (not a one-time cost); `cudaHostRegister` on pre-touched memory is 1.5x-4x cheaper than `cudaMallocHost`/`cudaHostAlloc` for the full lifecycle, independently confirmed via NVIDIA's own tracer
- main
- DevTab
- bench_evalmult.rs
- test_keygen.cpp
- RootTab
- cstring
- boot
- PrewarmEntry
- DevTab
- automorph_eval.cu
- Phase 16: explicit pinned-memory prewarming -- built, proven correct and safe, measured honestly, RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE (not promoted)
- ptops.cpp
- test_evalmult.cpp
- test_keyswitch_core_native_diff.cpp
- Ct
- Path
- test_polyeval.cpp
- test_poly2.cpp
- Ct
- Stats
- Phase 20: CUDA-registered host memory prototype -- built, correctness-verified (0 mismatches across 400+ checks), measured against the real encrypt/EvalMult pipeline, and RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE (not promoted)
- ks_fastcore.cu
- Phase 18: controlled THP experiment -- the system-wide toggle is blocked by permissions (correctly not worked around); a real but small THP-hint effect found at exactly 2MiB; Phase 17's "not CUDA-specific" conclusion is REVISED based on same-session interleaved re-testing
- encrypt_prewarm_pinned_buffer
- EncryptBufPoolBatched
- EncryptBufPoolPinned
- interleaved_ab.sh
- randomized_ab.sh
- run_trials.sh
- Retained
- GPU-Resident-Library
- Post-quantum instant messaging — architecture proposal
- 2. Main experiment: the embed loop
- EncryptBufPool
- This round: production integration, honest A/B, security re-review
- test_keyswitch_core_realct_diff.cpp
- test_stagediag.cpp
- basis_convert.cu
- ckks-core
- Architect — Planning Specialist
- Code Reviewer — Adversarial Specialist
- Scrum Master — Team Lead
- Scrum-Orchestrator session
- test_rescale_vector_diff_n1024.cpp
- Backend Developer — Systems Specialist
- Frontend Developer — Interface Specialist
- QA Engineer — Test Specialist
- Release Engineer — Delivery Specialist
- Security Engineer — Defensive Specialist
- CLAUDE.md
- .claude/CLAUDE.md

## God Nodes (most connected - your core abstractions)
1. `Round 2 — post-master-directive, in progress` - 67 edges
2. `encrypt_host()` - 56 edges
3. `CkksError` - 42 edges
4. `Master state` - 40 edges
5. `DeviceKSContext` - 40 edges
6. `EvalMultFixtureCpp` - 34 edges
7. `native_primes()` - 34 edges
8. `Rust/CUDA migration matrix` - 31 edges
9. `Next session — exact continuation` - 31 edges
10. `KeySwitchConstants` - 31 edges

## Surprising Connections (you probably didn't know these)
- `Round 2 — post-master-directive, in progress` --references--> `gpufhe_ks_fixture_destroy()`  [INFERRED]
  research/EXPERIMENT_LEDGER.md → rust/crates/ckks-cuda/cpp_bridge/keyswitch_bridge.cpp
- `Round 2 — post-master-directive, in progress` --references--> `gpufhe_ks_fixture_secret_key()`  [INFERRED]
  research/EXPERIMENT_LEDGER.md → rust/crates/ckks-cuda/cpp_bridge/keyswitch_bridge.cpp
- `GOD PROMPT "CKKS PHASE 9": the real encrypted EvalMult round trip PASSES end-to-end for the first time (since Phase 5) — entry #41's "sizeP=2-specific defect" framing is SUPERSEDED; the real cause was a test-fixture parameter choice (NUM_PART=2 violating the Hybrid key-switch's P>=Q_part_max requirement), not a code defect, and it is fixed` --references--> `keyswitch_core_resident()`  [INFERRED]
  research/MASTER_STATE.md → src/keyswitch.cpp
- `GOD PROMPT "CKKS PHASE 17": diagnosed Phase 16's unexplained allocation-cost variance — root mechanism demonstrated to be the Linux kernel's own memory-locking (`mlock()`) path, NOT CUDA-specific; no production change justified, Phase 16's decision stands on stronger evidence` --references--> `encrypt_host()`  [INFERRED]
  research/MASTER_STATE.md → src/encdec.cpp
- `Earlier commit (Phase 21)` --references--> `encrypt_host()`  [INFERRED]
  research/NEXT_SESSION.md → src/encdec.cpp

## Import Cycles
- None detected.

## Communities (159 total, 3 thin omitted)

### Community 0 - "keyswitch_resident.h"
Cohesion: 0.07
Nodes (34): invmod(), powmod(), cd, main(), mmu(), cd, main(), mmu() (+26 more)

### Community 1 - "keyswitch.rs"
Cohesion: 0.09
Nodes (32): addmod(), ERR_BUF_LEN, err_from_buf(), gpufhe_ks_fixture_alpha(), gpufhe_ks_fixture_create(), gpufhe_ks_fixture_destroy(), gpufhe_ks_fixture_eval_key_part(), gpufhe_ks_fixture_eval_key_towers() (+24 more)

### Community 2 - "evalmult.rs"
Cohesion: 0.13
Nodes (47): encode_real(), addmod(), assert_slots_close(), ckks_mods_and_roots(), coeff_to_eval(), control_a2_raw_tensor_matches_direct_ciphertext_product(), control_a2_raw_tensor_matches_direct_ciphertext_product_exactly(), control_a_relin_combine_matches_raw_tensor_num_part_1_test() (+39 more)

### Community 3 - "keygen.h"
Cohesion: 0.04
Nodes (51): cd, cd, main(), mmu(), smu(), cd, main(), mmu() (+43 more)

### Community 4 - "test_boot_hoist_enabled.cpp"
Cohesion: 0.11
Nodes (20): cd, main(), mmu(), rssMB(), smu(), am(), cd, main() (+12 more)

### Community 5 - "bench_rescale_cpp_vs_rust.cpp"
Cohesion: 0.12
Nodes (24): bitrev(), build_tables(), forward_ntt_host(), invmod(), main(), median(), powmod(), report() (+16 more)

### Community 6 - "DeviceCiphertext"
Cohesion: 0.05
Nodes (40): DeviceTables, d_fprecon, d_froots, d_iprecon, d_iroots, n, ninv, ninv_precon (+32 more)

### Community 7 - "Tab"
Cohesion: 0.14
Nodes (13): build_tab(), invmod(), mulmod(), P_mod(), powmod(), Tab, fp, fr (+5 more)

### Community 8 - "DeviceKSContext"
Cohesion: 0.05
Nodes (39): DeviceKSContext, complModHost, d_av, d_bv, d_complMod, d_fp, d_fr, d_ip (+31 more)

### Community 9 - "encrypt_host_shoup_pooled_barrett.cpp"
Cohesion: 0.20
Nodes (16): am(), barrett_reduce_u64(), build_enc_tab_pooled_barrett(), emb_barrett(), EncDevTab, fp, fr, encrypt_host_shoup_pooled_barrett() (+8 more)

### Community 10 - "bench_encrypt_profile.cpp"
Cohesion: 0.05
Nodes (41): am(), build_enc_tab(), Clock, EncDevTab, fp, fr, encrypt_host_profiled(), EncryptStageTimes (+33 more)

### Community 11 - "vector"
Cohesion: 0.10
Nodes (13): clk, clk, DeviceKSContext, DeviceKSWork, KeySwitchConstants, clk, cd, main() (+5 more)

### Community 12 - "RingDimension"
Cohesion: 0.06
Nodes (25): CkksError, Cuda, InvalidModulus, InvalidRingDimension, Modulus, former_invalid_num_part_2_fixture_is_now_rejected(), cases(), cold_start_cost_probe() (+17 more)

### Community 13 - "rescale.rs"
Cohesion: 0.11
Nodes (19): RingDimension, cold_start_cost_probe(), forward_ntt(), gpufhe_native_rescale_consts(), invalid_input_length_is_rejected(), LaunchINTT_GS(), LaunchNTT_CT(), LaunchRescaleFuse() (+11 more)

### Community 14 - "EvalMultFixtureCpp"
Cohesion: 0.04
Nodes (41): EvalMultFixtureCpp, ba0, ba1, C, c0a, c0b, c1a, c1b (+33 more)

### Community 15 - "keyswitch_bridge.cpp"
Cohesion: 0.07
Nodes (19): GOD PROMPT "CKKS PHASE 22": Rust/FFI integration of CUDA-registered host memory — correctness/concurrency PROVEN at the FFI boundary; Phase 20's prewarming anomaly RESOLVED (a benchmark confound); still RETAINED AS OPT-IN (magnitude, not safety, is the remaining question), Earlier commit (Phase 22), READ THIS SECOND if resuming into pinned-memory work: CUDA-registered host memory is now Rust/FFI-ACCESSIBLE (`codec::encrypt_registered`), fully correctness/concurrency-verified at BOTH the C++ and Rust layers, and the prewarming anomaly is RESOLVED (it was a benchmark confound) -- STILL RETAINED AS OPT-IN; the only remaining question is magnitude-vs-complexity, not safety, gpufhe_ks_fixture_create(), gpufhe_ks_fixture_destroy(), gpufhe_ks_fixture_run_reference(), gpufhe_ks_fixture_run_resident(), gpufhe_ks_fixture_secret_key() (+11 more)

### Community 16 - "encrypt_host"
Cohesion: 0.14
Nodes (18): 1. Buffer-pool safety audit (directive Section 3), 3. Independent differential-test results (directive Section 6: built BEFORE touching production), 5. Benchmarks, 6. Security review, 7. What remains the dominant cost, post-Phase-14, Embed loop alone (`bench/bench_embed_barrett_vs_reference.cpp`, no GPU, interleaved, 9 trials x 20 reps), Embed modular arithmetic derivation and buffer-pool safety audit (Phase 14), Full `encrypt_host` (unified 3-way interleaved, `bench/bench_encrypt_phase14_all_variants.cpp`, 9 trials x 20 reps, run twice independently) (+10 more)

### Community 17 - "KeySwitchConstants"
Cohesion: 0.06
Nodes (33): KeySwitchConstants, alpha, av, bv, evalKeyTowers, fullQ, mdBMuHi, mdBMuLo (+25 more)

### Community 18 - "encrypt_host_shoup.cpp"
Cohesion: 0.21
Nodes (12): am(), build_enc_tab_shoup(), EncDevTab, fp, fr, encrypt_host_shoup(), mulmod_cold(), shoup_mulmod() (+4 more)

### Community 19 - "bench_encrypt_registered_lifecycle.cpp"
Cohesion: 0.11
Nodes (23): Clock, Fixture, m, mod, n, pkA, pkB, root (+15 more)

### Community 20 - "Round 2 — post-master-directive, in progress"
Cohesion: 0.10
Nodes (25): LaunchAutomorphPermute(), Literature review, What remains uncertain / explicitly not yet checked, This cycle's real, validated outcomes, Root cause, confirmed by direct instrumentation, Earlier commit (Phase 23), This round's real, delivered results (commits `93f2231`, `0043af3`), automorphism_eval_device() (+17 more)

### Community 21 - "test_registered_host_buffer.cpp"
Cohesion: 0.21
Nodes (17): encrypt_registered_fresh_per_call(), RegisteredHostBuffer, lastError_, ptr_, GOD PROMPT "CKKS PHASE 20": CUDA-registered host memory PROTOTYPE built, correctness-verified (0 mismatches across 400+ checks), measured against the real encrypt/EvalMult pipeline — RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE, NOT promoted to production, Ownership model, designed before implementation (directive Section 2), main(), scenario10_shutdown_while_draining() (+9 more)

### Community 22 - "EvalMultFixtureCpp"
Cohesion: 0.07
Nodes (26): EvalMultFixtureCpp, ba0, ba1, C, c0a, c0b, c1a, c1b (+18 more)

### Community 23 - "encrypt_host_shoup_pooled_barrett_prewarm.cpp"
Cohesion: 0.20
Nodes (16): am(), barrett_reduce_u64(), build_enc_tab_prewarm(), emb_barrett(), EncDevTab, fp, fr, encrypt_host_shoup_pooled_barrett_prewarmed() (+8 more)

### Community 24 - "Problem"
Cohesion: 0.18
Nodes (10): cd, compareResult(), EL(), main(), Problem, lgS, M, n (+2 more)

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
Cohesion: 0.14
Nodes (21): Production impact assessment and decision (directive Section 10), Reconstructing Phase 20 (directive Section 1) -- confirmed, not assumed, am(), barrett_reduce_u64(), build_enc_tab_registered(), emb_barrett(), EncDevTab, fp (+13 more)

### Community 29 - "encdec.cpp"
Cohesion: 0.27
Nodes (11): am(), build_enc_tab(), decrypt_host(), EncDevTab, fp, fr, invmod(), mm() (+3 more)

### Community 30 - "bench_thp_diagnostic.cpp"
Cohesion: 0.14
Nodes (8): Clock, main(), test_size(), us_between(), Clock, main(), one_call(), us()

### Community 31 - "Fixture"
Cohesion: 0.20
Nodes (11): checkOne(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 32 - "native_primes"
Cohesion: 0.13
Nodes (23): 4. A test bug caught and fixed (reported honestly, not hidden), GOD PROMPT "CKKS PHASE 10": repository-wide parameter audit + centralized, exact enforcement of the Hybrid key-switch's P>=Q_part_max requirement — four more invalid fixtures found and fixed, keyswitch.rs's own fixture corrected with a new decrypt-based test, invalid configurations now REJECTED at construction time, GOD PROMPT "CKKS RUST + CUDA PHASE 3": third operation (full Hybrid key switching) ported, bit-exact vs. the established reference, a real latent C++ bug found and guarded against, memory measured not guessed, compute_keyswitch_constants(), invmod(), mulmod(), powmod(), precon() (+15 more)

### Community 33 - "Fixture"
Cohesion: 0.22
Nodes (11): check(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 34 - "encrypt_host_shoup_pooled_barrett_batched"
Cohesion: 0.20
Nodes (16): am(), barrett_reduce_u64(), build_enc_tab_batched(), emb_barrett(), EncDevTab, fp, fr, encrypt_host_shoup_pooled_barrett_batched() (+8 more)

### Community 35 - "encrypt_host_shoup_pooled_barrett_batched_pinned.cpp"
Cohesion: 0.17
Nodes (17): Correctness, independently verified (directive Section 5), am(), barrett_reduce_u64(), build_enc_tab_pinned(), emb_barrett(), EncDevTab, fp, fr (+9 more)

### Community 36 - "ks_context_create"
Cohesion: 0.13
Nodes (31): GOD PROMPT round: large-n gate re-checked (still NO-GO), bench_evalmult_resident root-caused and fixed, test_polyeval investigated (inconclusive, reverted), Later round still: GPU-memory lifecycle isolated with real production APIs; external-kill correlation directly demonstrated; NO-GO decision for the next full attempt under current host conditions, A real, tiny, confirmed leak found and fixed (not the explanation for the MB-scale growth above), Experiment 2: enhanced instrumentation, 5th real n=131072 attempt, Go/no-go decision for the next full n=131072 attempt, Priority 2 result: what is PROVEN, what is HYPOTHESIS, what remains UNKNOWN about the kill mechanism, Priority 3 result: the `stageDiags`/`stageDiagsFwd` O(S²) bottleneck is FIXED, proven equivalent, and measured in production, Priority 4 result: a 6th attempt, with favorable host memory, reaches diagonal 8 of 16 — far further than any of the five prior attempts, then is killed anyway with zero kernel trace (+23 more)

### Community 37 - "codec.rs"
Cohesion: 0.18
Nodes (26): budget_exhaustion_falls_back_safely_and_stays_correct(), concurrent_rust_callers_same_and_distinct_keys(), decode(), decrypt(), encrypt(), encrypt_registered(), EncryptRegisteredResult, Fixture (+18 more)

### Community 38 - "Sample"
Cohesion: 0.12
Nodes (16): Sample, alloc_us, bytes, cudaPtrType, errorCode, free_us, memAvail_kB, memFree_kB (+8 more)

### Community 39 - "test_keyswitch_stage_trace.cpp"
Cohesion: 0.09
Nodes (32): invmod(), keygen_host(), mk_tab(), mulmod(), ntt_tower(), powmod(), submod(), Tab (+24 more)

### Community 40 - "Fixture"
Cohesion: 0.22
Nodes (11): check(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 41 - "bench_pinned_alloc_diagnostic.cpp"
Cohesion: 0.06
Nodes (41): Clock, main(), measure_size(), report(), Stats, max_, mean, min_ (+33 more)

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

### Community 47 - "Phase 22: Rust/FFI integration of CUDA-registered host memory -- correctness/concurrency PROVEN at the FFI boundary for the first time; Phase 20's prewarming anomaly RESOLVED (a benchmark confound, not a real effect); registered path confirmed faster through the real Rust API too -- RETAINED AS OPT-IN, promotion criteria now fully satisfied except magnitude"
Cohesion: 0.16
Nodes (15): GOD PROMPT "CKKS PHASE 14": audited Phase 13's device-buffer pool (found it already sound, hardened with 4 new targeted concurrency tests) and optimized `embed` — the remaining largest stage — with a per-tower Barrett reduction, derived for the FULL int64_t range including LONG_MIN, validated 22M+ times with zero mismatches — 1.44x-1.73x faster `encrypt_host`, 1.17x-1.36x faster full pipeline, zero correctness regression, GOD PROMPT "CKKS PHASE 15": merged encrypt_host's 8 separate transfer calls into 2 over one packed buffer (1.07x-1.17x faster) — then measured a SECOND pinned-memory experiment that looked even better in isolation, caught it was actually a net loss for real usage via the full-pipeline benchmark, and correctly did NOT ship it, GOD PROMPT "CKKS PHASE 21": Graphify repository-intelligence integration — installed, validated against 12 real questions with honest gaps documented, zero CKKS code touched, READ THIS SEVENTH if resuming into performance work on the OLDER (Phase 15) material: transfer batching is DONE and shipped; a pinned-memory follow-up was built, verified, measured FASTER in isolation, then found to be a net loss for real usage and correctly NOT shipped, FFI interface design (directive Section 3): Approach B, smallest safe interface, Git, Graphify refresh and verification (directive Section 9/11), Graphify used, not just read (directive Section 0/9) (+7 more)

### Community 48 - "Fixture"
Cohesion: 0.20
Nodes (11): checkOne(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 49 - "KeySwitchFixtureCpp"
Cohesion: 0.18
Nodes (8): KeySwitchFixtureCpp, C, K, W, main(), median(), mulmod_b(), report()

### Community 50 - "rotation.rs"
Cohesion: 0.10
Nodes (29): cold_start_vs_steady_state_timing(), ERR_BUF_LEN, err_from_buf(), gpufhe_rot_fixture_create(), gpufhe_rot_fixture_destroy(), gpufhe_rot_fixture_galois_element(), gpufhe_rot_fixture_modulus_q(), gpufhe_rot_fixture_num_part() (+21 more)

### Community 51 - "Priority queue (highest value first)"
Cohesion: 0.19
Nodes (10): HoistedDigits, d_compl, n, numPart, sizeQlP, hoist_digits_create(), hoist_digits_destroy(), ks_accumulate_from_digits() (+2 more)

### Community 52 - "LaunchNTT_CT"
Cohesion: 0.08
Nodes (31): LaunchApproxSwitchCRTBasis(), ilog2_u(), LaunchINTT_GS(), LaunchINTT_GS_Batched(), LaunchModDownCombine(), ct_shared_tail(), ct_shared_tail_b(), ct_stage() (+23 more)

### Community 53 - "build_enc_tab_pooled"
Cohesion: 0.29
Nodes (6): build_enc_tab_pooled(), EncDevTab, fp, fr, mulmod_cold(), up()

### Community 54 - "Fixture"
Cohesion: 0.20
Nodes (11): checkOne(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 55 - "Sample"
Cohesion: 0.17
Nodes (12): Sample, allocIndex, anonHuge_kB, bytes, firsttouch_us, liveBefore, memAvailable_kB, memFree_kB (+4 more)

### Community 56 - "Fixture"
Cohesion: 0.20
Nodes (11): check(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

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
Cohesion: 0.22
Nodes (11): check(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 63 - "bench_embed_barrett_vs_reference.cpp"
Cohesion: 0.29
Nodes (11): emb_barrett(), emb_old(), main(), make_tb(), report(), stats_of(), TowerBarrett, mu (+3 more)

### Community 64 - "bootstrap"
Cohesion: 0.53
Nodes (4): bootstrap(), main(), median(), quantile()

### Community 65 - "Fixture"
Cohesion: 0.21
Nodes (11): check(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 66 - "Fixture"
Cohesion: 0.20
Nodes (11): checkOne(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 67 - "Master state"
Cohesion: 0.08
Nodes (31): LaunchRescaleFuse(), 6. Numerical correctness results (`tests/test_encode_fft_matches_reference.cpp`), Mapping every item the governing directive asked for, Current verified state, Environment constraints (confirmed, not assumed), GOD PROMPT "CKKS PHASE 11": first real-world per-stage performance baseline for the complete encode→encrypt→EvalMult→decrypt→decode pipeline — the GPU kernels are NOT the bottleneck (~1% of total latency); two host-side optimizations (parallelized encode/decode, batched encrypt_host) give a verified 2.6x–5.3x total speedup with zero correctness regression, GOD PROMPT "CKKS PHASE 17": diagnosed Phase 16's unexplained allocation-cost variance — root mechanism demonstrated to be the Linux kernel's own memory-locking (`mlock()`) path, NOT CUDA-specific; no production change justified, Phase 16's decision stands on stronger evidence, GOD PROMPT "CKKS PHASE 18": controlled THP experiment — the system-wide toggle is blocked by permissions (correctly NOT worked around, per explicit user choice); a real but narrow ~5x THP-hint effect found at exactly 2MiB; Phase 17's "not CUDA-specific, HIGH confidence" conclusion REVISED to MEDIUM after same-session interleaved re-testing contradicted it (+23 more)

### Community 68 - "encrypt_host_shoup_pooled"
Cohesion: 0.11
Nodes (19): 10. Security review, 11. What remains the dominant cost, post-Phase-13, 1. Call graph, 2. Fresh baseline (re-established this round, not reused from Phase 12), 3. Internal profiling methodology, 4. Stage-level breakdown (median of 7 trials x 15 reps, per-trial medians), 5. Why `combine` is expensive: direct assembly evidence, not assumption, 6.1 Experiment: Shoup's precomputed-constant modmul in the combine loop (+11 more)

### Community 69 - "test_automorph_eval_fast.cpp"
Cohesion: 0.47
Nodes (3): gcd64(), main(), run_case()

### Community 70 - "FFT-based CKKS canonical-embedding encode/decode — derivation, implementation, and verification"
Cohesion: 0.14
Nodes (24): 1. The exact convention already in this repository (authoritative), 2. Derivation: why a single size-`M=2N` FFT suffices, 3. Why this is NOT the same permutation as a textbook FFT's bit-reversal, 4. Independent verification BEFORE writing any C++ (Python/numpy), 5. C++ implementation (`src/encode_fft.cpp`), 7. Real cryptographic correctness, 8. Benchmark results, 9. What remains the dominant cost, post-Phase-12 (+16 more)

### Community 71 - "cuda_runtime"
Cohesion: 0.09
Nodes (6): clk, clk, main(), up(), check_switch(), main()

### Community 72 - "Memory feasibility — measured, not estimated"
Cohesion: 0.08
Nodes (28): GOD PROMPT round: test_polyeval's real root cause found — a genuine library bug in the native rescale pathway at small n, never a test-logic issue, Caught in the act: an ordinary, previously-reliable `test_boot` (n=8192) run was itself OOM-killed minutes later, with WSL reporting "10Gi free" throughout, Experiment 1: `tests/test_n131072_cache_churn.cpp` (new, committed), GPU VRAM at n=131072 (for completeness, now secondary), growth source exists BEYOND KROT/KSC; Objective C is NOT complete, Memory feasibility — measured, not estimated, not WSL2's self-reported 10GB ceiling, The actual root cause: this machine's physical RAM, not WSL2's internal accounting (+20 more)

### Community 73 - "Phase 17: diagnosing the allocation-cost variance -- the mechanism is the Linux kernel's memory-locking path, NOT CUDA-specific, with host memory pressure a plausible but unconfirmed amplifier"
Cohesion: 0.17
Nodes (12): Benchmark artifact check (Hypothesis E), Experiment 1: fresh-allocate-then-free, 10 repeats per size, one process, Experiment 3: isolating CUDA from the underlying OS mechanism, Experiment 4: the exact-2MiB boundary is not special; "large and never-freed" is what correlates, Methodology critique of Phase 16's own measurement (directive Step 1/2), Phase 17: diagnosing the allocation-cost variance -- the mechanism is the Linux kernel's memory-locking path, NOT CUDA-specific, with host memory pressure a plausible but unconfirmed amplifier, Ranked root-cause assessment (directive Step 5), Windows-side memory bracketing (Hypothesis A) (+4 more)

### Community 74 - "Tab"
Cohesion: 0.18
Nodes (9): mk_tab(), Tab, fp, fr, ip, ir, ninv, ninv_p (+1 more)

### Community 75 - "Ct"
Cohesion: 0.33
Nodes (5): Ct, c0, c1, scale, tw

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

### Community 84 - "test_encode_decode_parallel_matches_sequential.cpp"
Cohesion: 0.67
Nodes (3): main(), oracle_decode(), oracle_encode()

### Community 85 - "test_embed_barrett_matches_reference.cpp"
Cohesion: 0.38
Nodes (11): barrett_reduce_u64(), checkModulus(), emb_fast(), emb_ref(), main(), make_tb(), TowerBarrett, mu (+3 more)

### Community 86 - "Tab"
Cohesion: 0.22
Nodes (10): build_inv_tab(), inverse_ntt(), invmod(), powmod(), Tab, ip, ir, ninv (+2 more)

### Community 87 - "build.rs"
Cohesion: 0.33
Nodes (5): compile_cpp(), compile_cuda(), main(), repo_root(), run_or_panic()

### Community 88 - "KeyPairHost"
Cohesion: 0.40
Nodes (4): KeyPairHost, pkA, pkB, s

### Community 89 - "Next session — exact continuation"
Cohesion: 0.10
Nodes (19): Before trusting any new benchmark number, Blocked this round, Blocked this round (carried forward from Phase 23, now with a sharper gate), Current commit, Earlier commit (Phase 21), Earlier commit still (Phase 19), `-G` build-flag bug: CLOSED for the four originally-flagged benchmarks; one remaining sweep still open, Next session — exact continuation (+11 more)

### Community 90 - "EncryptBufPool"
Cohesion: 0.22
Nodes (9): EncryptBufPool, d_e0, d_e1, d_fp, d_fr, d_m, d_mod, d_v (+1 more)

### Community 91 - "test_evalkeygen.cpp"
Cohesion: 0.39
Nodes (7): addmod_local(), invmod(), main(), mm(), powmod(), sub(), up()

### Community 92 - "Sample"
Cohesion: 0.25
Nodes (7): Sample, n, path, phase, sizeQ, total_us, wallclock_ms

### Community 93 - "Rust/CUDA migration matrix"
Cohesion: 0.10
Nodes (20): EventTimer, a, b, combine_kernel(), LaunchCombine(), LaunchTensor(), mulmod64(), tensor_kernel() (+12 more)

### Community 94 - "rotation_bridge.cpp"
Cohesion: 0.06
Nodes (23): LaunchAddInto(), LaunchPtMulAcc(), galois_element_for_rotation(), gpufhe_rot_fixture_create(), gpufhe_rot_fixture_run_reference(), gpufhe_rot_fixture_run_resident(), KeySwitchConstants, mulmod_bridge() (+15 more)

### Community 95 - "test_moddown_crt_basis_isolated.cpp"
Cohesion: 0.35
Nodes (7): addmod(), main(), mulmod(), P_mod(), run_conversion(), submod(), up()

### Community 96 - "test_decompose_crt_basis_isolated.cpp"
Cohesion: 0.40
Nodes (7): addmod(), main(), mulmod(), Q_mod(), run_conversion(), submod(), up()

### Community 97 - "Fixture"
Cohesion: 0.22
Nodes (11): check(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 98 - "Graphify setup, usage, and validation (Phase 21)"
Cohesion: 0.17
Nodes (11): Generating and refreshing the graph, Graphify setup, usage, and validation (Phase 21), How Claude Code should use this (Layer D, kept short deliberately), How to disable or remove the integration, If Graphify is missing or broken in a future session, Installation and integration performed this phase, Known extraction gaps (honestly documented, not hidden), Phase 22 field validation: used for real engineering work, not just the validation suite (+3 more)

### Community 99 - "test_encrypt_host_fast_matches_sequential.cpp"
Cohesion: 0.39
Nodes (7): am(), invmod(), main(), mm(), oracle_encrypt_host(), powmod(), up()

### Community 100 - "boot"
Cohesion: 0.43
Nodes (5): boot(), main(), median(), mm(), quantile()

### Community 101 - "Tables"
Cohesion: 0.20
Nodes (8): build_tables(), Tables, fp, fr, ip, ir, ninv, ninv_p

### Community 102 - "EncryptBufPoolRegistered"
Cohesion: 0.18
Nodes (11): Investigating the prewarming anomaly (directive Section 6) -- RESOLVED: a benchmark confound, not a real effect, EncryptBufPoolRegistered, budgetSkipped, d_fp, d_fr, d_mod, d_packed, h_packed (+3 more)

### Community 103 - "bench_evalmult.cpp"
Cohesion: 0.43
Nodes (5): am(), clk, main(), median(), mm()

### Community 104 - "test_cheb.cpp"
Cohesion: 0.53
Nodes (5): am(), cd, main(), mm(), sm()

### Community 105 - "Performance baseline — full bootstrap, this session"
Cohesion: 0.33
Nodes (5): Performance baseline — full bootstrap, this session, Security reality check (n=8192, log2(QP)=1775, sparse secret h=64), The bigger, still-open finding: GPU clock behavior, What the stage breakdown looks like at the most recent checkpoint, Where the time actually goes (profiled, not guessed)

### Community 106 - "Tab"
Cohesion: 0.18
Nodes (9): mk_tab(), Tab, fp, fr, ip, ir, ninv, ninv_p (+1 more)

### Community 107 - "Phase 19: isolating CUDA pinned-allocation overhead -- mechanistically PROVEN to be dominated by a single `ioctl()` into the NVIDIA driver on every call (not a one-time cost); `cudaHostRegister` on pre-touched memory is 1.5x-4x cheaper than `cudaMallocHost`/`cudaHostAlloc` for the full lifecycle, independently confirmed via NVIDIA's own tracer"
Cohesion: 0.15
Nodes (13): A genuinely new, unexplained finding: plain pageable `malloc`+`memset` ALSO jumps sharply at exactly 2MiB, Decision, Decisive finding 1: `cudaMallocHost` and `cudaHostAlloc(cudaHostAllocDefault)` are statistically indistinguishable -- confirmed empirically, not merely assumed from documentation, Decisive finding 2: the cost is syscall-level, mechanistic, and almost entirely inside a single `ioctl()` -- confirmed by NVIDIA's own tracer, not by this project's own timers, Decisive finding 3: `cudaHostRegister` on pre-touched memory is 1.5x-4x cheaper than `cudaMallocHost`/`cudaHostAlloc` for the FULL lifecycle -- reproduced across 4 independent fresh-process runs, Phase 19: isolating CUDA pinned-allocation overhead -- mechanistically PROVEN to be dominated by a single `ioctl()` into the NVIDIA driver on every call (not a one-time cost); `cudaHostRegister` on pre-touched memory is 1.5x-4x cheaper than `cudaMallocHost`/`cudaHostAlloc` for the full lifecycle, independently confirmed via NVIDIA's own tracer, Regression and environment, Section 2 audit: a real methodology gap in Phases 16-18's own measurements (+5 more)

### Community 108 - "main"
Cohesion: 0.39
Nodes (6): bootstrap(), main(), median(), mm(), modinv(), quantile()

### Community 109 - "DevTab"
Cohesion: 0.29
Nodes (7): DevTab, fp, fr, ip, ir, ninv, ninv_p

### Community 110 - "bench_evalmult.rs"
Cohesion: 0.11
Nodes (23): deterministic_input(), main(), median(), N, NUM_PART, report(), SIZE_P, SIZE_Q (+15 more)

### Community 111 - "test_keygen.cpp"
Cohesion: 0.48
Nodes (5): invmod(), main(), mm(), powmod(), up()

### Community 112 - "RootTab"
Cohesion: 0.20
Nodes (10): build_tab(), invmod(), powmod(), RootTab, fp, fr, ip, ir (+2 more)

### Community 113 - "cstring"
Cohesion: 0.25
Nodes (5): RegisteredBufferState, Allocated, Empty, Failed, Registered

### Community 114 - "boot"
Cohesion: 0.53
Nodes (4): boot(), main(), median(), quantile()

### Community 115 - "PrewarmEntry"
Cohesion: 0.22
Nodes (9): Experiment 2: never-free, matching production's actual behavior, PrewarmEntry, d_fp, d_fr, d_mod, d_packed, h_packed, ready (+1 more)

### Community 116 - "DevTab"
Cohesion: 0.29
Nodes (7): DevTab, fp, fr, ip, ir, ninv, ninv_p

### Community 118 - "Phase 16: explicit pinned-memory prewarming -- built, proven correct and safe, measured honestly, RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE (not promoted)"
Cohesion: 0.25
Nodes (8): Concurrency and lifecycle (directive Step 4), Correctness (directive Step 3/7), Lifecycle design (directive Step 2, decided BEFORE implementation), Memory overhead (directive Step 8), Phase 16: explicit pinned-memory prewarming -- built, proven correct and safe, measured honestly, RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE (not promoted), Verified starting state, Workload-level benchmark (directive Step 3/5), Workload profile comparison (directive Step 5)

### Community 119 - "ptops.cpp"
Cohesion: 0.53
Nodes (4): am(), ct_add_ct_host(), ct_mul_pt_host(), mm()

### Community 120 - "test_evalmult.cpp"
Cohesion: 0.83
Nodes (3): am(), main(), mm()

### Community 121 - "test_keyswitch_core_native_diff.cpp"
Cohesion: 0.83
Nodes (3): am(), main(), mm()

### Community 122 - "Ct"
Cohesion: 0.33
Nodes (5): Ct, c0, c1, scale, tw

### Community 123 - "Path"
Cohesion: 0.29
Nodes (7): Path, A_Pageable, B_MlockPure, C_MallocHost, D2_HostAllocWC, D_HostAllocDefault, E_HostRegister

### Community 124 - "test_polyeval.cpp"
Cohesion: 0.53
Nodes (5): am(), cd, main(), mm(), sm()

### Community 125 - "test_poly2.cpp"
Cohesion: 0.53
Nodes (5): am(), cd, main(), mm(), sm()

### Community 126 - "Ct"
Cohesion: 0.33
Nodes (5): Ct, c0, c1, scale, tw

### Community 127 - "Stats"
Cohesion: 0.29
Nodes (7): Stats, max_, mean, min_, n, p50, p95

### Community 128 - "Phase 20: CUDA-registered host memory prototype -- built, correctness-verified (0 mismatches across 400+ checks), measured against the real encrypt/EvalMult pipeline, and RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE (not promoted)"
Cohesion: 0.17
Nodes (12): CMake/CTest integration (directive Section 9), Concurrency audit (directive Section 4/8), Environment re-verified (not assumed), Full lifecycle measurement (directive Section 6), Implementation (directive Section 4), Nsight Systems confirmation of the actual driver-call reduction (directive Section 6/12, completion criterion 8), Phase 20: CUDA-registered host memory prototype -- built, correctness-verified (0 mismatches across 400+ checks), measured against the real encrypt/EvalMult pipeline, and RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE (not promoted), Production-shaped prototype (directive Section 3) (+4 more)

### Community 129 - "ks_fastcore.cu"
Cohesion: 0.21
Nodes (8): ks_macc_kernel(), LaunchKSMultAcc(), LaunchKSMultAccBatched(), mulmod64(), LaunchScaledAddInto(), Experiment ledger, Round 2 — post-master-directive, in progress, Standing discipline (unchanged from round 1, still enforced)

### Community 130 - "Phase 18: controlled THP experiment -- the system-wide toggle is blocked by permissions (correctly not worked around); a real but small THP-hint effect found at exactly 2MiB; Phase 17's "not CUDA-specific" conclusion is REVISED based on same-session interleaved re-testing"
Cohesion: 0.18
Nodes (11): Decision, Phase 18: controlled THP experiment -- the system-wide toggle is blocked by permissions (correctly not worked around); a real but small THP-hint effect found at exactly 2MiB; Phase 17's "not CUDA-specific" conclusion is REVISED based on same-session interleaved re-testing, Step 1: baseline environment (read-only), Step 2/3: the system-wide THP toggle is blocked by permissions -- correctly NOT worked around, Step 4/5: `bench_pinned_alloc_diagnostic3.cpp` -- per-sample methodology under the unchanged THP=madvise setting, Step 4: `bench_thp_hint_by_size.cpp` -- the decisive, demonstrated THP-hint effect, Step 6/self-correction: Phase 17's "not CUDA-specific" conclusion does NOT hold up under same-session interleaved re-testing -- REVISED, Step 7: production implications (+3 more)

### Community 131 - "encrypt_prewarm_pinned_buffer"
Cohesion: 0.38
Nodes (7): GOD PROMPT "CKKS PHASE 16": built, proved correct, and concurrency-hardened an explicit pinned-memory prewarming API — found the allocation cost itself is unpredictable in this environment, so retained it as an opt-in experimental feature rather than promoting it to production, Decision (directive Step 8's three categories), Decision (directive Step 9), Engineering response (directive Step 6), encrypt_pinned_buffer_is_ready(), encrypt_prewarm_pinned_buffer(), PrewarmKey

### Community 132 - "EncryptBufPoolBatched"
Cohesion: 0.29
Nodes (7): 2. Experiment 1 (KEPT): batched transfers, pageable memory, EncryptBufPoolBatched, d_fp, d_fr, d_mod, d_packed, sizeQ

### Community 133 - "EncryptBufPoolPinned"
Cohesion: 0.29
Nodes (7): EncryptBufPoolPinned, d_fp, d_fr, d_mod, d_packed, h_packed, sizeQ

### Community 134 - "interleaved_ab.sh"
Cohesion: 0.67
Nodes (3): PATH, run_one(), interleaved_ab.sh script

### Community 135 - "randomized_ab.sh"
Cohesion: 0.67
Nodes (3): PATH, run_one(), randomized_ab.sh script

### Community 136 - "run_trials.sh"
Cohesion: 0.67
Nodes (3): gpu_snapshot(), PATH, run_trials.sh script

### Community 137 - "Retained"
Cohesion: 0.33
Nodes (6): Retained, bytes, cudaFreeHostNotPlainFree, path, ptr, wasRegistered

### Community 138 - "GPU-Resident-Library"
Cohesion: 0.20
Nodes (9): Benchmark methodology, Building, Design discipline, GPU-Resident-Library, Performance (RTX 2060 Max-Q, ring 32768), Proven components, Status: complete, self-contained, numerically-proven CKKS multiply, What's next (optional) (+1 more)

### Community 139 - "Post-quantum instant messaging — architecture proposal"
Cohesion: 0.20
Nodes (9): 0. The distinction that governs everything, 1. Key establishment: hybrid X25519 + ML-KEM-768, 2. Payload encryption: ChaCha20-Poly1305 or AES-256-GCM via HKDF, 3. Session design: Double Ratchet + KEM ratchet, 4. Authentication, 5. Operational concerns, 6. Latency: the quantitative case against CKKS, 7. Required before production (+1 more)

### Community 140 - "2. Main experiment: the embed loop"
Cohesion: 0.40
Nodes (5): 2. Main experiment: the embed loop, Assembly inspection (directive: "inspect optimized assembly rather than assuming"), Derivation: per-tower Barrett reduction, Operand ranges (determined, not assumed), Signedness and overflow analysis

### Community 141 - "EncryptBufPool"
Cohesion: 0.22
Nodes (9): EncryptBufPool, d_e0, d_e1, d_fp, d_fr, d_m, d_mod, d_v (+1 more)

### Community 142 - "This round: production integration, honest A/B, security re-review"
Cohesion: 0.40
Nodes (5): 1. Hoisting wired into the live production bootstrap, behind a flag, 2. Honest, order-balanced interleaved A/B measurement — small, noisy effect, NOT a clean win, 3. Security re-review: adversarial check found the prior "130-bit conservative" framing overconfident, This round: production integration, honest A/B, security re-review, What's still open

### Community 143 - "test_keyswitch_core_realct_diff.cpp"
Cohesion: 0.83
Nodes (3): am(), main(), mm()

### Community 144 - "test_stagediag.cpp"
Cohesion: 0.60
Nodes (4): am(), cd, main(), mm()

### Community 145 - "basis_convert.cu"
Cohesion: 0.83
Nodes (3): approx_switch_kernel(), barrett_u128_mod_u64(), shoup_mulmod()

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

### Community 162 - "CLAUDE.md"
Cohesion: 0.50
Nodes (3): graphify, Scrum-Orchestrator, Session start (GOD PROMPT CKKS phases)

## Knowledge Gaps
- **640 isolated node(s):** `fp`, `fr`, `alloc_us`, `combine_us`, `download_us` (+635 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1113 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Round 2 — post-master-directive, in progress` connect `ks_fastcore.cu` to `bench_rescale_cpp_vs_rust.cpp`, `DeviceCiphertext`, `encrypt_host_shoup_pooled_barrett.cpp`, `RingDimension`, `keyswitch_bridge.cpp`, `encrypt_host`, `Round 2 — post-master-directive, in progress`, `Security assessment — GPU-Resident-Library`, `encdec.cpp`, `native_primes`, `ks_context_create`, `test_keyswitch_stage_trace.cpp`, `bench_pinned_alloc_diagnostic.cpp`, `bench_alloc_path_comparison.cpp`, `Priority queue (highest value first)`, `LaunchNTT_CT`, `Master state`, `FFT-based CKKS canonical-embedding encode/decode — derivation, implementation, and verification`, `Memory feasibility — measured, not estimated`, `test_evalkeygen.cpp`, `Rust/CUDA migration matrix`, `rotation_bridge.cpp`, `ptops.cpp`?**
  _High betweenness centrality (0.158) - this node is a cross-community bridge._
- **Are the 65 inferred relationships involving `Round 2 — post-master-directive, in progress` (e.g. with `run_one()` and `forward_ntt_host()`) actually correct?**
  _`Round 2 — post-master-directive, in progress` has 65 INFERRED edges - model-reasoned connections that need verification._
- **What connects `fp`, `fr`, `alloc_us` to the rest of the system?**
  _640 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `keyswitch_resident.h` be split into smaller, more focused modules?**
  _Cohesion score 0.0673758865248227 - nodes in this community are weakly interconnected._
- **Why does `former_invalid_num_part_2_fixture_is_now_rejected()` connect `RingDimension` to `ks_fastcore.cu`, `evalmult.rs`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **Should `keyswitch.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.09154437456324249 - nodes in this community are weakly interconnected._
- **Why does `encrypt_host()` connect `encrypt_host` to `ks_fastcore.cu`, `encrypt_prewarm_pinned_buffer`, `EncryptBufPoolBatched`, `encrypt_host_shoup.cpp`, `test_registered_host_buffer.cpp`, `encdec.cpp`, `Fixture`, `native_primes`, `Fixture`, `encrypt_host_shoup_pooled_barrett_batched_pinned.cpp`, `Fixture`, `Phase 22: Rust/FFI integration of CUDA-registered host memory -- correctness/concurrency PROVEN at the FFI boundary for the first time; Phase 20's prewarming anomaly RESOLVED (a benchmark confound, not a real effect); registered path confirmed faster through the real Rust API too -- RETAINED AS OPT-IN, promotion criteria now fully satisfied except magnitude`, `Fixture`, `LaunchNTT_CT`, `Fixture`, `Fixture`, `Fixture`, `Fixture`, `Fixture`, `Fixture`, `Master state`, `encrypt_host_shoup_pooled`, `FFT-based CKKS canonical-embedding encode/decode — derivation, implementation, and verification`, `Next session — exact continuation`, `Rust/CUDA migration matrix`, `Fixture`, `test_encrypt_host_fast_matches_sequential.cpp`, `Phase 16: explicit pinned-memory prewarming -- built, proven correct and safe, measured honestly, RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE (not promoted)`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._