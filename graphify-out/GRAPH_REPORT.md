# Graph Report - GPU-Resident-Library  (2026-11-09)

## Corpus Check
- 250 files · ~350,220 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 10 file(s) not represented in the graph (top: .jsonl 6, (none) 4)

## Summary
- 2903 nodes · 6346 edges · 165 communities (158 shown, 7 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 782 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `3e98cd9c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- keyswitch_resident.h
- keyswitch.rs
- evalmult.rs
- keygen.h
- test_hoist_multi.cpp
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
- CaseResult
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
- decrypt_host
- bench_thp_diagnostic.cpp
- Fixture
- KeySwitchContext
- main
- encrypt_host_shoup_pooled_barrett_batched
- encrypt_host_shoup_pooled_barrett_batched_pinned.cpp
- ks_context_create
- codec.rs
- Sample
- ckks-cuda/src/lib.rs
- bench_embed_barrett_vs_reference.cpp
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
- HoistedDigits
- ks_fastcore.cu
- build_enc_tab_pooled
- Fixture
- Fixture
- Tab
- Fixture
- test_ntt_roundtrip_n1024.cpp
- Stats
- DeviceKSWork
- ErrStats
- Graphify setup, usage, and validation (Phase 21)
- ParamRecord
- bootstrap
- Fixture
- Fixture
- .multiply_relin_rescale_breakdown
- encrypt_host_shoup_pooled
- test_automorph_eval_fast.cpp
- FFT-based CKKS canonical-embedding encode/decode — derivation, implementation, and verification
- cuda_runtime
- Memory feasibility — measured, not estimated
- Stats
- Tab
- Ct
- Stats
- Stats
- Stats
- Stats
- Stats
- DeviceBuf
- Stats
- Stats
- test_encode_decode_parallel_matches_sequential.cpp
- CaseRecord
- Tab
- build.rs
- KeyPairHost
- Next session — exact continuation
- EncryptBufPool
- test_evalkeygen.cpp
- Sample
- test_boot_multi_input.cpp
- rotation_bridge.cpp
- test_moddown_crt_basis_isolated.cpp
- test_decompose_crt_basis_isolated.cpp
- run_config
- Phase 18: controlled THP experiment -- the system-wide toggle is blocked by permissions (correctly not worked around); a real but small THP-hint effect found at exactly 2MiB; Phase 17's "not CUDA-specific" conclusion is REVISED based on same-session interleaved re-testing
- test_encrypt_host_fast_matches_sequential.cpp
- boot
- Tables
- ScopedKSWork
- bench_evalmult.cpp
- cmath
- DevTab
- Tab
- Phase 19: isolating CUDA pinned-allocation overhead -- mechanistically PROVEN to be dominated by a single `ioctl()` into the NVIDIA driver on every call (not a one-time cost); `cudaHostRegister` on pre-touched memory is 1.5x-4x cheaper than `cudaMallocHost`/`cudaHostAlloc` for the full lifecycle, independently confirmed via NVIDIA's own tracer
- main
- DevTab
- LaunchScaledAddInto
- test_keygen.cpp
- RootTab
- Performance baseline — full bootstrap, this session
- boot
- LaunchNTT_CT
- 2. Main experiment: the embed loop
- automorph_eval.cu
- test_boot_multi_input_small.cpp
- atLevel
- test_chain.cpp
- boot_driver_harness.h
- Ct
- Path
- bench_evalmult.rs
- test_boot_a.cpp
- Ct
- test_n131072_cache_churn.cpp
- MessageProfile
- basis_convert.cu
- Batched CUDA transfer optimization for encrypt_host (Phase 15)
- StageRecord
- interleaved_ab.sh
- randomized_ab.sh
- run_trials.sh
- GPU-Resident-Library
- Post-quantum instant messaging — architecture proposal
- native_primes
- EncryptBufPool
- Master state
- test_keyswitch_stage_trace.cpp
- Retained
- ckks-core
- bench_hoist_ab.cpp
- Architect — Planning Specialist
- Code Reviewer — Adversarial Specialist
- Scrum Master — Team Lead
- Scrum-Orchestrator session
- test_polyeval.cpp
- bootstrap_preflight.ps1
- test_rescale_vector_diff_n1024.cpp
- Backend Developer — Systems Specialist
- Frontend Developer — Interface Specialist
- QA Engineer — Test Specialist
- Release Engineer — Delivery Specialist
- Security Engineer — Defensive Specialist
- CLAUDE.md
- .claude/CLAUDE.md
- test_evalmult.cpp
- test_keyswitch_core_native_diff.cpp
- test_keyreuse.cpp
- Benchmark / driver baselines

## God Nodes (most connected - your core abstractions)
1. `Round 2 — post-master-directive, in progress` - 72 edges
2. `encrypt_host()` - 56 edges
3. `CkksError` - 42 edges
4. `DeviceKSContext` - 40 edges
5. `Master state` - 40 edges
6. `Next session — exact continuation` - 35 edges
7. `ks_context_create()` - 35 edges
8. `EvalMultFixtureCpp` - 34 edges
9. `native_primes()` - 34 edges
10. `CaseRecord` - 31 edges

## Surprising Connections (you probably didn't know these)
- `Round 2 — post-master-directive, in progress` --references--> `gpufhe_ks_fixture_destroy()`  [INFERRED]
  research/EXPERIMENT_LEDGER.md → rust/crates/ckks-cuda/cpp_bridge/keyswitch_bridge.cpp
- `Round 2 — post-master-directive, in progress` --references--> `gpufhe_ks_fixture_secret_key()`  [INFERRED]
  research/EXPERIMENT_LEDGER.md → rust/crates/ckks-cuda/cpp_bridge/keyswitch_bridge.cpp
- `Earlier commit (Phase 21)` --references--> `encrypt_host()`  [INFERRED]
  research/NEXT_SESSION.md → src/encdec.cpp
- `Operand ranges (determined, not assumed)` --references--> `native_primes()`  [INFERRED]
  research/EMBED_AND_POOL_HARDENING.md → src/params.cpp
- `GOD PROMPT "CKKS RUST + CUDA PHASE 2": second operation (rescale) ported, a real benchmark bug caught and fixed, Rust/C++ remain equivalent within noise` --references--> `rescale_resident_raw()`  [INFERRED]
  research/MASTER_STATE.md → src/keyswitch_resident.cpp

## Import Cycles
- None detected.

## Communities (165 total, 7 thin omitted)

### Community 0 - "keyswitch_resident.h"
Cohesion: 0.06
Nodes (43): invmod(), powmod(), cd, cd, main(), mmu(), rssMB(), smu() (+35 more)

### Community 1 - "keyswitch.rs"
Cohesion: 0.10
Nodes (21): addmod(), ERR_BUF_LEN, gpufhe_ks_fixture_alpha(), gpufhe_ks_fixture_create(), gpufhe_ks_fixture_destroy(), gpufhe_ks_fixture_eval_key_part(), gpufhe_ks_fixture_eval_key_towers(), gpufhe_ks_fixture_modulus_q() (+13 more)

### Community 2 - "evalmult.rs"
Cohesion: 0.14
Nodes (45): encode_real(), addmod(), assert_slots_close(), ckks_mods_and_roots(), coeff_to_eval(), control_a2_raw_tensor_matches_direct_ciphertext_product(), control_a2_raw_tensor_matches_direct_ciphertext_product_exactly(), control_a_relin_combine_matches_raw_tensor_within_keyswitch_noise_test() (+37 more)

### Community 3 - "keygen.h"
Cohesion: 0.07
Nodes (24): clk, DeviceKSContext, DeviceKSWork, KeySwitchConstants, cd, cd, main(), mmu() (+16 more)

### Community 4 - "test_hoist_multi.cpp"
Cohesion: 0.20
Nodes (9): cd, main(), mm(), modinv(), cd, main(), mm(), modinv() (+1 more)

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
Cohesion: 0.08
Nodes (34): am(), build_enc_tab(), Clock, EncDevTab, fp, fr, encrypt_host_profiled(), EncryptStageTimes (+26 more)

### Community 11 - "vector"
Cohesion: 0.11
Nodes (4): clk, clk, main(), mm()

### Community 12 - "RingDimension"
Cohesion: 0.08
Nodes (24): CkksError, Cuda, InvalidModulus, InvalidRingDimension, Modulus, RingDimension, cases(), cold_start_cost_probe() (+16 more)

### Community 13 - "rescale.rs"
Cohesion: 0.16
Nodes (18): cold_start_cost_probe(), forward_ntt(), gpufhe_native_rescale_consts(), invalid_input_length_is_rejected(), LaunchINTT_GS(), LaunchNTT_CT(), LaunchRescaleFuse(), LaunchSwitchModulus() (+10 more)

### Community 14 - "EvalMultFixtureCpp"
Cohesion: 0.06
Nodes (27): EvalMultFixtureCpp, ba0, ba1, C, c0a, c0b, c1a, c1b (+19 more)

### Community 15 - "keyswitch_bridge.cpp"
Cohesion: 0.07
Nodes (19): GOD PROMPT "CKKS PHASE 22": Rust/FFI integration of CUDA-registered host memory — correctness/concurrency PROVEN at the FFI boundary; Phase 20's prewarming anomaly RESOLVED (a benchmark confound); still RETAINED AS OPT-IN (magnitude, not safety, is the remaining question), Earlier commit (Phase 22), READ THIS SECOND if resuming into pinned-memory work: CUDA-registered host memory is now Rust/FFI-ACCESSIBLE (`codec::encrypt_registered`), fully correctness/concurrency-verified at BOTH the C++ and Rust layers, and the prewarming anomaly is RESOLVED (it was a benchmark confound) -- STILL RETAINED AS OPT-IN; the only remaining question is magnitude-vs-complexity, not safety, gpufhe_ks_fixture_create(), gpufhe_ks_fixture_destroy(), gpufhe_ks_fixture_run_reference(), gpufhe_ks_fixture_run_resident(), gpufhe_ks_fixture_secret_key() (+11 more)

### Community 16 - "CaseResult"
Cohesion: 0.08
Nodes (23): 1. Environment actually discovered, 2. The two checkouts, and which one is authoritative, 3. Partial Phase 25 work recovered from the lost session, 4. Toolchain and hardware, as measured, 5. Memory gate at recovery — NO-GO, 6. Protective constraints honored, 7. Recovery status, Authoritative (used for all Git operations) (+15 more)

### Community 17 - "KeySwitchConstants"
Cohesion: 0.06
Nodes (33): KeySwitchConstants, alpha, av, bv, evalKeyTowers, fullQ, mdBMuHi, mdBMuLo (+25 more)

### Community 18 - "encrypt_host_shoup.cpp"
Cohesion: 0.21
Nodes (12): am(), build_enc_tab_shoup(), EncDevTab, fp, fr, encrypt_host_shoup(), mulmod_cold(), shoup_mulmod() (+4 more)

### Community 19 - "bench_encrypt_registered_lifecycle.cpp"
Cohesion: 0.09
Nodes (29): Clock, encrypt_registered_fresh_per_call(), Fixture, m, mod, n, pkA, pkB (+21 more)

### Community 20 - "Round 2 — post-master-directive, in progress"
Cohesion: 0.12
Nodes (29): LaunchAutomorphPermute(), LaunchAddInto(), 3.3 What is already clean, O3 — Hoist `pt_to_eval_host`'s transform onto the device, Experiment ledger, Round 2 — post-master-directive, in progress, Standing discipline (unchanged from round 1, still enforced), Literature review (+21 more)

### Community 21 - "test_registered_host_buffer.cpp"
Cohesion: 0.23
Nodes (16): RegisteredHostBuffer, lastError_, ptr_, GOD PROMPT "CKKS PHASE 20": CUDA-registered host memory PROTOTYPE built, correctness-verified (0 mismatches across 400+ checks), measured against the real encrypt/EvalMult pipeline — RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE, NOT promoted to production, Ownership model, designed before implementation (directive Section 2), main(), scenario10_shutdown_while_draining(), scenario12_cleanup_after_failed_async_op() (+8 more)

### Community 22 - "EvalMultFixtureCpp"
Cohesion: 0.07
Nodes (26): EvalMultFixtureCpp, ba0, ba1, C, c0a, c0b, c1a, c1b (+18 more)

### Community 23 - "encrypt_host_shoup_pooled_barrett_prewarm.cpp"
Cohesion: 0.05
Nodes (56): GOD PROMPT "CKKS PHASE 16": built, proved correct, and concurrency-hardened an explicit pinned-memory prewarming API — found the allocation cost itself is unpredictable in this environment, so retained it as an opt-in experimental feature rather than promoting it to production, READ THIS SIXTH if resuming into performance work: a pinned-memory PREWARMING API exists, fully correct and concurrency-safe, but was NOT promoted because its own allocation cost is unpredictable in this environment -- Phase 20 (top of this file) built a registered-memory alternative that resolves this exact weakness, Benchmark artifact check (Hypothesis E), Decision (directive Step 8's three categories), Decision (directive Step 9), Engineering response (directive Step 6), Experiment 1: fresh-allocate-then-free, 10 repeats per size, one process, Experiment 2: never-free, matching production's actual behavior (+48 more)

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
Cohesion: 0.08
Nodes (28): Clock, main(), meminfo_field(), print_sample(), report(), Sample, allocIndex, anonHuge_kB (+20 more)

### Community 28 - "encrypt_host_shoup_pooled_barrett_batched_registered.cpp"
Cohesion: 0.05
Nodes (47): Concurrency audit (directive Section 4/8), Implementation (directive Section 4), Investigating the prewarming anomaly (directive Section 6) -- RESOLVED: a benchmark confound, not a real effect, Production impact assessment and decision (directive Section 10), Reconstructing Phase 20 (directive Section 1) -- confirmed, not assumed, gpufhe_prewarm_registered_buffer(), am(), barrett_reduce_u64() (+39 more)

### Community 29 - "decrypt_host"
Cohesion: 0.27
Nodes (11): am(), build_enc_tab(), decrypt_host(), EncDevTab, fp, fr, invmod(), mm() (+3 more)

### Community 30 - "bench_thp_diagnostic.cpp"
Cohesion: 0.12
Nodes (13): Clock, main(), test_size(), us_between(), Clock, main(), one_call(), us() (+5 more)

### Community 31 - "Fixture"
Cohesion: 0.20
Nodes (11): checkOne(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 32 - "KeySwitchContext"
Cohesion: 0.25
Nodes (10): err_from_buf(), gpufhe_ks_fixture_run_resident(), invalid_input_length_is_rejected(), keyswitch_matches_host_reference_random_data(), keyswitch_matches_reference_on_boundary_inputs(), keyswitch_matches_reference_reused_context_repeated_calls(), KeySwitchContext, new_ctx() (+2 more)

### Community 33 - "main"
Cohesion: 0.18
Nodes (14): GOD PROMPT "CKKS PHASE 10": repository-wide parameter audit + centralized, exact enforcement of the Hybrid key-switch's P>=Q_part_max requirement — four more invalid fixtures found and fixed, keyswitch.rs's own fixture corrected with a new decrypt-based test, invalid configurations now REJECTED at construction time, compute_keyswitch_constants(), invmod(), mulmod(), powmod(), precon(), prod_mod(), addmod() (+6 more)

### Community 34 - "encrypt_host_shoup_pooled_barrett_batched"
Cohesion: 0.16
Nodes (18): GOD PROMPT "CKKS PHASE 15": merged encrypt_host's 8 separate transfer calls into 2 over one packed buffer (1.07x-1.17x faster) — then measured a SECOND pinned-memory experiment that looked even better in isolation, caught it was actually a net loss for real usage via the full-pipeline benchmark, and correctly did NOT ship it, READ THIS SEVENTH if resuming into performance work on the OLDER (Phase 15) material: transfer batching is DONE and shipped; a pinned-memory follow-up was built, verified, measured FASTER in isolation, then found to be a net loss for real usage and correctly NOT shipped, am(), barrett_reduce_u64(), build_enc_tab_batched(), emb_barrett(), EncDevTab, fp (+10 more)

### Community 35 - "encrypt_host_shoup_pooled_barrett_batched_pinned.cpp"
Cohesion: 0.06
Nodes (45): am(), barrett_reduce_u64(), build_enc_tab_pinned(), emb_barrett(), EncDevTab, fp, fr, encrypt_host_shoup_pooled_barrett_batched_pinned() (+37 more)

### Community 36 - "ks_context_create"
Cohesion: 0.14
Nodes (31): The bigger, still-open finding: GPU clock behavior, GOD PROMPT round: large-n gate re-checked (still NO-GO), bench_evalmult_resident root-caused and fixed, test_polyeval investigated (inconclusive, reverted), Later round still: GPU-memory lifecycle isolated with real production APIs; external-kill correlation directly demonstrated; NO-GO decision for the next full attempt under current host conditions, A real, tiny, confirmed leak found and fixed (not the explanation for the MB-scale growth above), Experiment 2: enhanced instrumentation, 5th real n=131072 attempt, Go/no-go decision for the next full n=131072 attempt, Run 1: 4 stages (tw=30,29,28,27) x 20 diagonals — 68 diagonals completed (far more than the 6th real attempt's 8) before an external kill, Run 2: 8 stages (tw=30..23) x 3 diagonals — killed after only 2 diagonals, with an IDENTICAL internal trajectory to Run 1's first 2 diagonals (+23 more)

### Community 37 - "codec.rs"
Cohesion: 0.18
Nodes (26): concurrent_rust_callers_same_and_distinct_keys(), decode(), decrypt(), encrypt(), encrypt_registered(), EncryptRegisteredResult, Fixture, gpufhe_decode() (+18 more)

### Community 38 - "Sample"
Cohesion: 0.12
Nodes (16): Sample, alloc_us, bytes, cudaPtrType, errorCode, free_us, memAvail_kB, memFree_kB (+8 more)

### Community 39 - "ckks-cuda/src/lib.rs"
Cohesion: 0.19
Nodes (4): budget_exhaustion_falls_back_safely_and_stays_correct(), Fixture, make_fixture(), test_message()

### Community 40 - "bench_embed_barrett_vs_reference.cpp"
Cohesion: 0.07
Nodes (40): emb_barrett(), emb_old(), main(), make_tb(), report(), Stats, max_, mean (+32 more)

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
Cohesion: 0.39
Nodes (6): bitrev(), check(), forward_ntt(), main(), make_case(), upload()

### Community 45 - "bench_alloc_path_comparison.cpp"
Cohesion: 0.24
Nodes (11): Clock, main(), meminfo_field(), path_name(), query_pointer_type(), release_retained(), report(), run_one() (+3 more)

### Community 46 - "bench_encrypt_prewarm_workloads.cpp"
Cohesion: 0.19
Nodes (11): main(), median_of(), report(), Stats, max_, mean, min_, n (+3 more)

### Community 47 - "Phase 22: Rust/FFI integration of CUDA-registered host memory -- correctness/concurrency PROVEN at the FFI boundary for the first time; Phase 20's prewarming anomaly RESOLVED (a benchmark confound, not a real effect); registered path confirmed faster through the real Rust API too -- RETAINED AS OPT-IN, promotion criteria now fully satisfied except magnitude"
Cohesion: 0.12
Nodes (20): GOD PROMPT "CKKS PHASE 14": audited Phase 13's device-buffer pool (found it already sound, hardened with 4 new targeted concurrency tests) and optimized `embed` — the remaining largest stage — with a per-tower Barrett reduction, derived for the FULL int64_t range including LONG_MIN, validated 22M+ times with zero mismatches — 1.44x-1.73x faster `encrypt_host`, 1.17x-1.36x faster full pipeline, zero correctness regression, Concurrency and lifecycle (directive Step 4), Correctness (directive Step 3/7), FFI interface design (directive Section 3): Approach B, smallest safe interface, Git, Graphify refresh and verification (directive Section 9/11), Graphify used, not just read (directive Section 0/9), Lifecycle design (directive Step 2, decided BEFORE implementation) (+12 more)

### Community 48 - "Fixture"
Cohesion: 0.20
Nodes (11): checkOne(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 49 - "KeySwitchFixtureCpp"
Cohesion: 0.18
Nodes (8): KeySwitchFixtureCpp, C, K, W, main(), median(), mulmod_b(), report()

### Community 50 - "rotation.rs"
Cohesion: 0.09
Nodes (30): gpufhe_ks_fixture_run_reference(), cold_start_vs_steady_state_timing(), ERR_BUF_LEN, err_from_buf(), gpufhe_rot_fixture_create(), gpufhe_rot_fixture_destroy(), gpufhe_rot_fixture_galois_element(), gpufhe_rot_fixture_modulus_q() (+22 more)

### Community 51 - "HoistedDigits"
Cohesion: 0.10
Nodes (23): HoistedDigits, d_compl, n, numPart, sizeQlP, Priority queue (highest value first), 0. The answer, up front, 1. What hoisting actually changes (+15 more)

### Community 52 - "ks_fastcore.cu"
Cohesion: 0.38
Nodes (4): ks_macc_kernel(), LaunchKSMultAcc(), LaunchKSMultAccBatched(), mulmod64()

### Community 53 - "build_enc_tab_pooled"
Cohesion: 0.20
Nodes (12): 11. What remains the dominant cost, post-Phase-13, GOD PROMPT "CKKS PHASE 13": deep-profiled `encrypt_host` — NOT the already-batched NTT, but two host-side loops (embed + combine) dominate; combine's modular multiply was calling libgcc's slow `__umodti3`; fixed with Shoup modmul + a device-buffer pool, catching and fixing a real concurrency bug along the way — 1.10x-1.84x faster, zero correctness regression, am(), build_enc_tab_pooled(), EncDevTab, fp, fr, encrypt_host_shoup_pooled() (+4 more)

### Community 54 - "Fixture"
Cohesion: 0.18
Nodes (11): checkOne(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 55 - "Fixture"
Cohesion: 0.22
Nodes (11): check(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 56 - "Tab"
Cohesion: 0.20
Nodes (10): build_tab(), invmod(), powmod(), Tab, fp, fr, ip, ir (+2 more)

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

### Community 62 - "Graphify setup, usage, and validation (Phase 21)"
Cohesion: 0.17
Nodes (11): Generating and refreshing the graph, Graphify setup, usage, and validation (Phase 21), How Claude Code should use this (Layer D, kept short deliberately), How to disable or remove the integration, If Graphify is missing or broken in a future session, Installation and integration performed this phase, Known extraction gaps (honestly documented, not hidden), Phase 22 field validation: used for real engineering work, not just the validation suite (+3 more)

### Community 63 - "ParamRecord"
Cohesion: 0.09
Nodes (22): ParamRecord, Delta_in, Delta_pt, driver, encryptSeedBase, hammingWeight, krotCap, kscCap (+14 more)

### Community 64 - "bootstrap"
Cohesion: 0.53
Nodes (4): bootstrap(), main(), median(), quantile()

### Community 65 - "Fixture"
Cohesion: 0.21
Nodes (11): check(), Fixture, m, mod, n, pkA, pkB, root (+3 more)

### Community 66 - "Fixture"
Cohesion: 0.09
Nodes (22): buildFixture(), diffLimbs(), Fixture, c0, c1, got0, got1, h0 (+14 more)

### Community 67 - ".multiply_relin_rescale_breakdown"
Cohesion: 0.29
Nodes (3): EventTimer, a, b

### Community 68 - "encrypt_host_shoup_pooled"
Cohesion: 0.10
Nodes (26): 1. Buffer-pool safety audit (directive Section 3), 3. Independent differential-test results (directive Section 6: built BEFORE touching production), 5. Benchmarks, 6. Security review, 7. What remains the dominant cost, post-Phase-14, Embed loop alone (`bench/bench_embed_barrett_vs_reference.cpp`, no GPU, interleaved, 9 trials x 20 reps), Embed modular arithmetic derivation and buffer-pool safety audit (Phase 14), Full `encrypt_host` (unified 3-way interleaved, `bench/bench_encrypt_phase14_all_variants.cpp`, 9 trials x 20 reps, run twice independently) (+18 more)

### Community 69 - "test_automorph_eval_fast.cpp"
Cohesion: 0.24
Nodes (7): gcd64(), main(), run_case(), am(), cd, main(), mm()

### Community 70 - "FFT-based CKKS canonical-embedding encode/decode — derivation, implementation, and verification"
Cohesion: 0.10
Nodes (31): 1. The exact convention already in this repository (authoritative), 2. Derivation: why a single size-`M=2N` FFT suffices, 3. Why this is NOT the same permutation as a textbook FFT's bit-reversal, 4. Independent verification BEFORE writing any C++ (Python/numpy), 5. C++ implementation (`src/encode_fft.cpp`), 6. Numerical correctness results (`tests/test_encode_fft_matches_reference.cpp`), 7. Real cryptographic correctness, 8. Benchmark results (+23 more)

### Community 71 - "cuda_runtime"
Cohesion: 0.11
Nodes (6): clk, clk, main(), up(), check_switch(), main()

### Community 72 - "Memory feasibility — measured, not estimated"
Cohesion: 0.07
Nodes (30): GOD PROMPT "CKKS PHASE 8": entry #40's `evalkeygen_host` diagnosis DISPROVEN (a sign bug in its own test) — the real EvalMult blocker is pinpointed to a `sizeP=2`-specific defect in the native key-switch reconstruction (`compute_keyswitch_constants`+`keyswitch_core_resident`), the exact P-tower count the real Rust fixture uses, Caught in the act: an ordinary, previously-reliable `test_boot` (n=8192) run was itself OOM-killed minutes later, with WSL reporting "10Gi free" throughout, Experiment 1: `tests/test_n131072_cache_churn.cpp` (new, committed), GPU VRAM at n=131072 (for completeness, now secondary), growth source exists BEYOND KROT/KSC; Objective C is NOT complete, Memory feasibility — measured, not estimated, not WSL2's self-reported 10GB ceiling, Priority 2 result: what is PROVEN, what is HYPOTHESIS, what remains UNKNOWN about the kill mechanism (+22 more)

### Community 73 - "Stats"
Cohesion: 0.22
Nodes (8): Stats, max_, mean, min_, n, stats_of(), p50, p95

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

### Community 81 - "DeviceBuf"
Cohesion: 0.16
Nodes (4): control_a_relin_combine_matches_raw_tensor_num_part_1_test(), former_invalid_num_part_2_fixture_is_now_rejected(), new_ctx_num_part(), DeviceBuf

### Community 82 - "Stats"
Cohesion: 0.20
Nodes (10): main(), report(), Stats, max_, mean, min_, n, stats_of() (+2 more)

### Community 83 - "Stats"
Cohesion: 0.20
Nodes (10): main(), report(), Stats, max_, mean, min_, n, stats_of() (+2 more)

### Community 84 - "test_encode_decode_parallel_matches_sequential.cpp"
Cohesion: 0.67
Nodes (3): main(), oracle_decode(), oracle_encode()

### Community 85 - "CaseRecord"
Cohesion: 0.11
Nodes (19): CaseRecord, abortReason, caseIdx, eFin, encryptSeed, firstFailedStage, messageSeed, nominalAmplitude (+11 more)

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
Nodes (20): Before trusting any new benchmark number, Blocked in Phase 23, Blocked in Phase 24 (carried forward from Phase 23, now with a sharper gate), Blocked this round (Phase 25) — and what is NOT blocked, Earlier commit (Phase 21), Earlier commit (Phase 24), Earlier commit still (Phase 19), Environment facts worth not rediscovering (+12 more)

### Community 90 - "EncryptBufPool"
Cohesion: 0.22
Nodes (9): EncryptBufPool, d_e0, d_e1, d_fp, d_fr, d_m, d_mod, d_v (+1 more)

### Community 91 - "test_evalkeygen.cpp"
Cohesion: 0.39
Nodes (7): addmod_local(), invmod(), main(), mm(), powmod(), sub(), up()

### Community 92 - "Sample"
Cohesion: 0.14
Nodes (13): Path, A_Production, B_Pinned, C_RegisteredFresh, D_RegisteredReused, E_RegisteredPrewarmed, Sample, n (+5 more)

### Community 93 - "test_boot_multi_input.cpp"
Cohesion: 0.43
Nodes (6): cd, main(), makeDriverMessage(), mmu(), rssMB(), smu()

### Community 94 - "rotation_bridge.cpp"
Cohesion: 0.07
Nodes (20): galois_element_for_rotation(), gpufhe_rot_fixture_create(), gpufhe_rot_fixture_run_reference(), gpufhe_rot_fixture_run_resident(), KeySwitchConstants, mulmod_bridge(), RotFixture, C (+12 more)

### Community 95 - "test_moddown_crt_basis_isolated.cpp"
Cohesion: 0.35
Nodes (7): addmod(), main(), mulmod(), P_mod(), run_conversion(), submod(), up()

### Community 96 - "test_decompose_crt_basis_isolated.cpp"
Cohesion: 0.40
Nodes (7): addmod(), main(), mulmod(), Q_mod(), run_conversion(), submod(), up()

### Community 97 - "run_config"
Cohesion: 0.29
Nodes (6): gpu_used_mb(), host_rss_mb(), main(), mulmod_b(), report(), run_config()

### Community 98 - "Phase 18: controlled THP experiment -- the system-wide toggle is blocked by permissions (correctly not worked around); a real but small THP-hint effect found at exactly 2MiB; Phase 17's "not CUDA-specific" conclusion is REVISED based on same-session interleaved re-testing"
Cohesion: 0.18
Nodes (11): Decision, Phase 18: controlled THP experiment -- the system-wide toggle is blocked by permissions (correctly not worked around); a real but small THP-hint effect found at exactly 2MiB; Phase 17's "not CUDA-specific" conclusion is REVISED based on same-session interleaved re-testing, Step 1: baseline environment (read-only), Step 2/3: the system-wide THP toggle is blocked by permissions -- correctly NOT worked around, Step 4/5: `bench_pinned_alloc_diagnostic3.cpp` -- per-sample methodology under the unchanged THP=madvise setting, Step 4: `bench_thp_hint_by_size.cpp` -- the decisive, demonstrated THP-hint effect, Step 6/self-correction: Phase 17's "not CUDA-specific" conclusion does NOT hold up under same-session interleaved re-testing -- REVISED, Step 7: production implications (+3 more)

### Community 99 - "test_encrypt_host_fast_matches_sequential.cpp"
Cohesion: 0.39
Nodes (7): am(), invmod(), main(), mm(), oracle_encrypt_host(), powmod(), up()

### Community 100 - "boot"
Cohesion: 0.43
Nodes (5): boot(), main(), median(), mm(), quantile()

### Community 101 - "Tables"
Cohesion: 0.20
Nodes (8): build_tables(), Tables, fp, fr, ip, ir, ninv, ninv_p

### Community 102 - "ScopedKSWork"
Cohesion: 0.16
Nodes (7): ScopedKSContext, ctx_, ScopedKSWork, work_, gpuMem(), main(), mm()

### Community 103 - "bench_evalmult.cpp"
Cohesion: 0.43
Nodes (5): am(), clk, main(), median(), mm()

### Community 104 - "cmath"
Cohesion: 0.06
Nodes (33): cd, am(), cd, main(), mm(), sm(), cd, am() (+25 more)

### Community 105 - "DevTab"
Cohesion: 0.29
Nodes (7): DevTab, fp, fr, ip, ir, ninv, ninv_p

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

### Community 110 - "LaunchScaledAddInto"
Cohesion: 0.29
Nodes (3): LaunchPtMulAcc(), LaunchScaledAddInto(), main()

### Community 111 - "test_keygen.cpp"
Cohesion: 0.48
Nodes (5): invmod(), main(), mm(), powmod(), up()

### Community 112 - "RootTab"
Cohesion: 0.20
Nodes (10): build_tab(), invmod(), powmod(), RootTab, fp, fr, ip, ir (+2 more)

### Community 113 - "Performance baseline — full bootstrap, this session"
Cohesion: 0.40
Nodes (4): Performance baseline — full bootstrap, this session, Security reality check (n=8192, log2(QP)=1775, sparse secret h=64), What the stage breakdown looks like at the most recent checkpoint, Where the time actually goes (profiled, not guessed)

### Community 114 - "boot"
Cohesion: 0.53
Nodes (4): boot(), main(), median(), quantile()

### Community 115 - "LaunchNTT_CT"
Cohesion: 0.08
Nodes (30): LaunchApproxSwitchCRTBasis(), ilog2_u(), LaunchINTT_GS(), LaunchINTT_GS_Batched(), LaunchModDownCombine(), ct_shared_tail(), ct_shared_tail_b(), ct_stage() (+22 more)

### Community 116 - "2. Main experiment: the embed loop"
Cohesion: 0.40
Nodes (5): 2. Main experiment: the embed loop, Assembly inspection (directive: "inspect optimized assembly rather than assuming"), Derivation: per-tower Barrett reduction, Operand ranges (determined, not assumed), Signedness and overflow analysis

### Community 118 - "test_boot_multi_input_small.cpp"
Cohesion: 0.15
Nodes (15): cd, endStage(), finalizeCase(), makeDriverMessage(), maxAbsDiff(), nowSecs(), printHumanSummary(), printStages() (+7 more)

### Community 119 - "atLevel"
Cohesion: 0.09
Nodes (24): 0. Method, and the honesty boundary, 1. Static call-site inventory, 2. Bootstrap structure, for scale, 3.1 The dominant transfer is the key, not the ciphertext, 3.2 The ciphertext makes a full round trip per diagonal, 3. The per-diagonal hot loop, line by line, 4. Opportunities, ranked, each with its risk and its test, 5. Before implementing O2/O3: the user's uncommitted work already prototypes them (+16 more)

### Community 120 - "test_chain.cpp"
Cohesion: 0.60
Nodes (4): am(), cd, main(), mm()

### Community 121 - "boot_driver_harness.h"
Cohesion: 0.43
Nodes (10): caseJsonLine(), jbool(), jnum(), jsonEscape(), jsonlPath(), jstr(), jU64Array(), paramsJsonLine() (+2 more)

### Community 122 - "Ct"
Cohesion: 0.33
Nodes (5): Ct, c0, c1, scale, tw

### Community 123 - "Path"
Cohesion: 0.29
Nodes (7): Path, A_Pageable, B_MlockPure, C_MallocHost, D2_HostAllocWC, D_HostAllocDefault, E_HostRegister

### Community 124 - "bench_evalmult.rs"
Cohesion: 0.11
Nodes (23): deterministic_input(), main(), median(), N, NUM_PART, report(), SIZE_P, SIZE_Q (+15 more)

### Community 125 - "test_boot_a.cpp"
Cohesion: 0.60
Nodes (4): cd, main(), mmu(), smu()

### Community 126 - "Ct"
Cohesion: 0.33
Nodes (5): Ct, c0, c1, scale, tw

### Community 127 - "test_n131072_cache_churn.cpp"
Cohesion: 0.39
Nodes (6): clk, EL(), gpuMemMB(), main(), mm(), rssMB()

### Community 128 - "MessageProfile"
Cohesion: 0.25
Nodes (8): MessageProfile, AlternatingSign, Constant, LargeMagnitude, Reference, SmallMagnitude, Zero, profileName()

### Community 129 - "basis_convert.cu"
Cohesion: 0.83
Nodes (3): approx_switch_kernel(), barrett_u128_mod_u64(), shoup_mulmod()

### Community 130 - "Batched CUDA transfer optimization for encrypt_host (Phase 15)"
Cohesion: 0.09
Nodes (21): 1. Existing transfer data flow (traced before changing anything), 2. Experiment 1 (KEPT): batched transfers, pageable memory, 4. Security review, 5. What remains the dominant cost, post-Phase-15, Batched CUDA transfer optimization for encrypt_host (Phase 15), CMake/CTest integration (directive Section 9), Correctness, independently verified (directive Section 5), Environment re-verified (not assumed) (+13 more)

### Community 132 - "StageRecord"
Cohesion: 0.25
Nodes (8): StageRecord, err, name, note, ok, ran, secs, tol

### Community 134 - "interleaved_ab.sh"
Cohesion: 0.67
Nodes (3): PATH, run_one(), interleaved_ab.sh script

### Community 135 - "randomized_ab.sh"
Cohesion: 0.67
Nodes (3): PATH, run_one(), randomized_ab.sh script

### Community 136 - "run_trials.sh"
Cohesion: 0.67
Nodes (3): gpu_snapshot(), PATH, run_trials.sh script

### Community 138 - "GPU-Resident-Library"
Cohesion: 0.20
Nodes (9): Benchmark methodology, Building, Design discipline, GPU-Resident-Library, Performance (RTX 2060 Max-Q, ring 32768), Proven components, Status: complete, self-contained, numerically-proven CKKS multiply, What's next (optional) (+1 more)

### Community 139 - "Post-quantum instant messaging — architecture proposal"
Cohesion: 0.20
Nodes (9): 0. The distinction that governs everything, 1. Key establishment: hybrid X25519 + ML-KEM-768, 2. Payload encryption: ChaCha20-Poly1305 or AES-256-GCM via HKDF, 3. Session design: Double Ratchet + KEM ratchet, 4. Authentication, 5. Operational concerns, 6. Latency: the quantitative case against CKKS, 7. Required before production (+1 more)

### Community 140 - "native_primes"
Cohesion: 0.11
Nodes (25): 4. A test bug caught and fixed (reported honestly, not hidden), GOD PROMPT "CKKS PHASE 9": the real encrypted EvalMult round trip PASSES end-to-end for the first time (since Phase 5) — entry #41's "sizeP=2-specific defect" framing is SUPERSEDED; the real cause was a test-fixture parameter choice (NUM_PART=2 violating the Hybrid key-switch's P>=Q_part_max requirement), not a code defect, and it is fixed, GOD PROMPT "CKKS RUST + CUDA PHASE 3": third operation (full Hybrid key switching) ported, bit-exact vs. the established reference, a real latent C++ bug found and guarded against, memory measured not guessed, GOD PROMPT "RESOLVE POLYEVAL CORRECTNESS" round: every library-defect hypothesis for test_polyeval refuted by direct evidence; one formula proven wrong, not the library, GOD PROMPT round: test_polyeval's real root cause found — a genuine library bug in the native rescale pathway at small n, never a test-logic issue, Bootstrap: build-verified, not runtime-verified (entry #56, Phase 23), Memory and benchmarks: deferred this round for the real-encryption primitives (entry #38), Rust/CUDA migration matrix (+17 more)

### Community 141 - "EncryptBufPool"
Cohesion: 0.22
Nodes (9): EncryptBufPool, d_e0, d_e1, d_fp, d_fr, d_m, d_mod, d_v (+1 more)

### Community 142 - "Master state"
Cohesion: 0.06
Nodes (38): combine_kernel(), LaunchCombine(), LaunchTensor(), mulmod64(), tensor_kernel(), Measured, not estimated: token-reduction and timing, Validation: 12 questions, tested against real evidence, 1. Hoisting wired into the live production bootstrap, behind a flag (+30 more)

### Community 144 - "test_keyswitch_stage_trace.cpp"
Cohesion: 0.17
Nodes (15): invmod(), keygen_host(), mk_tab(), mulmod(), ntt_tower(), powmod(), submod(), Tab (+7 more)

### Community 145 - "Retained"
Cohesion: 0.33
Nodes (6): Retained, bytes, cudaFreeHostNotPlainFree, path, ptr, wasRegistered

### Community 147 - "bench_hoist_ab.cpp"
Cohesion: 0.24
Nodes (7): cd, main(), median(), mm(), modinv(), rssMB(), cd

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

### Community 152 - "test_polyeval.cpp"
Cohesion: 0.53
Nodes (5): am(), cd, main(), mm(), sm()

### Community 153 - "bootstrap_preflight.ps1"
Cohesion: 0.33
Nodes (4): Get-BootstrapGoNoGoDecision(), Get-BootstrapWorkloadRequirement(), Get-GpuFreeVramMB(), Test-WslAvailable()

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

### Community 164 - "test_evalmult.cpp"
Cohesion: 0.83
Nodes (3): am(), main(), mm()

### Community 165 - "test_keyswitch_core_native_diff.cpp"
Cohesion: 0.83
Nodes (3): am(), main(), mm()

### Community 166 - "test_keyreuse.cpp"
Cohesion: 0.60
Nodes (4): am(), cd, main(), mm()

## Knowledge Gaps
- **718 isolated node(s):** `fp`, `fr`, `alloc_us`, `combine_us`, `download_us` (+713 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1233 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Round 2 — post-master-directive, in progress` connect `Round 2 — post-master-directive, in progress` to `StageRecord`, `bench_rescale_cpp_vs_rust.cpp`, `DeviceCiphertext`, `encrypt_host_shoup_pooled_barrett.cpp`, `native_primes`, `Master state`, `keyswitch_bridge.cpp`, `test_keyswitch_stage_trace.cpp`, `Security assessment — GPU-Resident-Library`, `decrypt_host`, `main`, `ks_context_create`, `bench_pinned_alloc_diagnostic.cpp`, `bench_alloc_path_comparison.cpp`, `HoistedDigits`, `ks_fastcore.cu`, `encrypt_host_shoup_pooled`, `FFT-based CKKS canonical-embedding encode/decode — derivation, implementation, and verification`, `Memory feasibility — measured, not estimated`, `DeviceBuf`, `CaseRecord`, `test_evalkeygen.cpp`, `LaunchScaledAddInto`, `LaunchNTT_CT`, `atLevel`?**
  _High betweenness centrality (0.145) - this node is a cross-community bridge._
- **Are the 70 inferred relationships involving `Round 2 — post-master-directive, in progress` (e.g. with `run_one()` and `forward_ntt_host()`) actually correct?**
  _`Round 2 — post-master-directive, in progress` has 70 INFERRED edges - model-reasoned connections that need verification._
- **What connects `fp`, `fr`, `alloc_us` to the rest of the system?**
  _718 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `keyswitch_resident.h` be split into smaller, more focused modules?**
  _Cohesion score 0.05727644652250146 - nodes in this community are weakly interconnected._
- **Why does `former_invalid_num_part_2_fixture_is_now_rejected()` connect `DeviceBuf` to `evalmult.rs`, `Round 2 — post-master-directive, in progress`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Should `keyswitch.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.09879032258064516 - nodes in this community are weakly interconnected._
- **Why does `CkksError` connect `RingDimension` to `KeySwitchContext`, `keyswitch.rs`, `evalmult.rs`, `codec.rs`, `rescale.rs`, `DeviceBuf`, `rotation.rs`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._