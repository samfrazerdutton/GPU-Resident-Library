cmake_minimum_required(VERSION 3.18)
project(GPUResidentFHE LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)   # RTX 2060 (Turing)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# GOD PROMPT "CKKS RUST + CUDA EVOLUTION"/"FAIR BENCHMARKS" rounds: a
# `-DCMAKE_CUDA_FLAGS="-g -G"` set on SOME earlier `cmake` invocation (not
# in this file, so not reviewable in version control until now) got
# cached in CMakeCache.txt and has been SILENTLY applied to every build
# since, Release included -- CMake always appends CMAKE_CUDA_FLAGS on top
# of the build-type-specific flags (CMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG
# in this project), it never replaces them. `-G` is CUDA's DEVICE DEBUG
# flag: it disables nearly all device-code optimizations regardless of
# `-O3` also being present. Discovered this round while investigating why
# a Rust-host NTT benchmark looked ~2.3x faster than the equivalent C++
# path -- isolated via a from-scratch nvcc recompile of the exact same
# kernel source (ruling out language, FFI, and even -rdc=true/false as
# explanations) down to this ONE flag, confirmed via `VERBOSE=1 cmake
# --build` showing `-g -G -O3 -DNDEBUG` on every kernel compile. This has
# likely affected EVERY prior CUDA kernel performance measurement in this
# project's history, not just this round's work -- see
# research/EXPERIMENT_LEDGER.md for the full isolation experiment and
# research/MASTER_STATE.md for which specific benchmark numbers predate
# this fix and should be treated as device-debug-mode measurements, not
# true Release performance, until re-run.
set(CMAKE_CUDA_FLAGS "" CACHE STRING "Extra CUDA flags applied to every build type; must NOT contain -G (device debug) for performance measurements to be meaningful. See comment above." FORCE)

# Core library: pure device-resident FHE, NO OpenFHE dependency.
add_library(gpufhe STATIC
  src/device_ciphertext.cpp
  src/ckks_ops.cpp
  src/keyswitch.cpp
  src/ks_constants.cpp
  src/ks_params_validate.cpp
  src/keygen.cpp
  src/evalkeygen.cpp
  src/encdec.cpp
  src/encode.cpp
  src/encode_fft.cpp
  src/encrypt_host_shoup.cpp
  src/encrypt_host_shoup_pooled.cpp
  src/encrypt_host_shoup_pooled_barrett.cpp
  src/encrypt_host_shoup_pooled_barrett_batched.cpp
  src/encrypt_host_shoup_pooled_barrett_batched_pinned.cpp
  src/rotate.cpp
  src/ptops.cpp
  src/params.cpp
  src/keyswitch_resident.cpp
  src/keyswitch_hoist.cpp
  kernels/ntt_ct.cu
  kernels/rns_arith.cu
  kernels/intt_ct.cu
  kernels/switch_modulus.cu
  kernels/rescale.cu
  kernels/basis_convert.cu
  kernels/ks_fastcore.cu
  kernels/moddown.cu
  kernels/tensor.cu
  kernels/automorph.cu
  kernels/automorph_eval.cu
  kernels/stage_acc.cu
)
find_package(CUDAToolkit REQUIRED)
# GOD PROMPT "CKKS PHASE 11": src/encode.cpp's encode_host/decode_host now
# use std::thread (parallelizing their O(n^2) host loops across hardware
# threads -- see research/EXPERIMENT_LEDGER.md entry #44). Modern glibc
# (>=2.34) folds pthread into libc, so this links fine without it on this
# machine, but find_package(Threads) + Threads::Threads is the portable,
# correct way to request it regardless of libc version.
find_package(Threads REQUIRED)
target_include_directories(gpufhe PUBLIC include)
target_link_libraries(gpufhe PUBLIC CUDA::cudart Threads::Threads)
target_include_directories(gpufhe PUBLIC ${CUDAToolkit_INCLUDE_DIRS} /usr/local/cuda-13.2/include)
set_target_properties(gpufhe PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# OpenFHE is a TEST-ONLY dependency: ground-truth oracle, never linked into
# the core library. Tests encrypt/decrypt in OpenFHE and compare bit-exact.
find_package(OpenFHE QUIET)

enable_testing()
function(gpufhe_test name src)
  add_executable(${name} ${src})
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${name} PRIVATE gpufhe CUDA::cudart)
  target_include_directories(${name} PRIVATE
    /usr/local/include/openfhe
    /usr/local/include/openfhe/pke
    /usr/local/include/openfhe/core
    /usr/local/include/openfhe/binfhe)
  target_link_libraries(${name} PRIVATE OPENFHEpke OPENFHEcore)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

gpufhe_test(test_ntt_resident tests/test_ntt_resident.cpp)
gpufhe_test(test_ntt_towers tests/test_ntt_towers.cpp)
gpufhe_test(test_intt_towers tests/test_intt_towers.cpp)
gpufhe_test(test_mult_resident tests/test_mult_resident.cpp)
gpufhe_test(test_intt_resident tests/test_intt_resident.cpp)
gpufhe_test(test_switchmod_resident tests/test_switchmod_resident.cpp)
gpufhe_test(test_rescale_resident tests/test_rescale_resident.cpp)
gpufhe_test(test_basis_convert tests/test_basis_convert.cpp)
gpufhe_test(test_decompose tests/test_decompose.cpp)
gpufhe_test(test_ks_fastcore tests/test_ks_fastcore.cpp)
gpufhe_test(test_moddown tests/test_moddown.cpp)
gpufhe_test(test_keyswitch tests/test_keyswitch.cpp)
# GOD PROMPT "CKKS PHASE 6": root-causing the relin algebraic identity
# (entries #30/#38) with OpenFHE as the operational reference -- a real
# cc->EvalMult()+cc->Decrypt() round trip, plus the SAME algebraic check
# re-done using OpenFHE's own DCRTPoly operators directly (not hand-
# extracted raw towers) to rule out a test-side extraction bug.
gpufhe_test(test_evalmult_relin_openfhe_reference tests/test_evalmult_relin_openfhe_reference.cpp)
gpufhe_test(test_ks_constants tests/test_ks_constants.cpp)
# GOD PROMPT "CKKS PHASE 10": standalone, no-GPU correctness test for
# check_hybrid_keyswitch_params (include/ks_params_validate.h), the
# single authoritative Hybrid key-switch P>=Q_part_max validator enforced
# at ks_context_create (src/keyswitch_resident.cpp, entry #43) -- cross-
# checked independently via tools/verify_ks_params_validate.py (Python's
# native arbitrary-precision arithmetic, a different implementation
# stack).
gpufhe_test(test_ks_params_validate tests/test_ks_params_validate.cpp)
# GOD PROMPT "CKKS PHASE 11": correctness tests for the two kept
# performance optimizations (parallelized encode_host/decode_host,
# batched encrypt_host) -- each checked bit-exact against an
# INDEPENDENT, hand-written reimplementation of the exact pre-Phase-11
# algorithm, never the production code as its own oracle. See
# research/EXPERIMENT_LEDGER.md entry #44.
gpufhe_test(test_encode_decode_parallel_matches_sequential tests/test_encode_decode_parallel_matches_sequential.cpp)
gpufhe_test(test_encrypt_host_fast_matches_sequential tests/test_encrypt_host_fast_matches_sequential.cpp)
# GOD PROMPT "CKKS PHASE 12": correctness test for the new O(N log N)
# FFT-based encode_host_fft/decode_host_fft (src/encode_fft.cpp) against
# the EXISTING, untouched, algorithmically DIFFERENT O(N^2) encode_host/
# decode_host -- see research/FFT_ENCODING_DERIVATION.md and research/
# EXPERIMENT_LEDGER.md entry #45.
gpufhe_test(test_encode_fft_matches_reference tests/test_encode_fft_matches_reference.cpp)
# GOD PROMPT "CKKS PHASE 13": correctness tests for the Shoup-modmul
# (encrypt_host_shoup) and Shoup+pooled-buffer (encrypt_host_shoup_
# pooled) encrypt_host variants against the EXISTING, untouched
# encrypt_host -- see research/ENCRYPT_PROFILING.md and research/
# EXPERIMENT_LEDGER.md entry #46. The pooled thread-safety test directly
# reproduces, via real std::threads (not relying on cargo's own
# scheduling), the concurrency bug found and fixed this round.
gpufhe_test(test_encrypt_host_shoup_matches_reference tests/test_encrypt_host_shoup_matches_reference.cpp)
gpufhe_test(test_encrypt_host_shoup_pooled_matches_reference tests/test_encrypt_host_shoup_pooled_matches_reference.cpp)
gpufhe_test(test_encrypt_host_shoup_pooled_thread_safety tests/test_encrypt_host_shoup_pooled_thread_safety.cpp)
# GOD PROMPT "CKKS PHASE 14": independent differential test for the
# Barrett-based embed() replacement (22.4M+ cases incl. LONG_MIN/
# LONG_MAX, against this project's own real native_primes() moduli),
# the full production-integration bit-exact test against the untouched
# encrypt_host, and a dedicated buffer-pool concurrency audit (4
# scenarios: same-key reuse, concurrent pool growth, longevity,
# mixed-key race) re-validating Phase 13's mutex design under direct
# stress. See research/EMBED_AND_POOL_HARDENING.md and research/
# EXPERIMENT_LEDGER.md entry #47.
gpufhe_test(test_embed_barrett_matches_reference tests/test_embed_barrett_matches_reference.cpp)
gpufhe_test(test_encrypt_host_shoup_pooled_barrett_matches_reference tests/test_encrypt_host_shoup_pooled_barrett_matches_reference.cpp)
gpufhe_test(test_encrypt_host_shoup_pooled_barrett_concurrency_audit tests/test_encrypt_host_shoup_pooled_barrett_concurrency_audit.cpp)
# GOD PROMPT "CKKS PHASE 15": correctness + concurrency tests for the
# batched-transfer production candidate (encrypt_host_shoup_pooled_
# barrett_batched, bit-exact vs. both encrypt_host and Phase 14, incl.
# a packed-buffer-boundary stress case) and for the pinned-memory
# experiment (encrypt_host_shoup_pooled_barrett_batched_pinned, built,
# verified, and deliberately NOT promoted to production -- see
# research/TRANSFER_PROFILING.md). Each new pooled resource gets its
# own dedicated 4-scenario concurrency audit rather than assuming
# Phase 14's audit generalizes. See research/EXPERIMENT_LEDGER.md
# entry #48.
gpufhe_test(test_encrypt_host_shoup_pooled_barrett_batched_matches_reference tests/test_encrypt_host_shoup_pooled_barrett_batched_matches_reference.cpp)
gpufhe_test(test_encrypt_host_shoup_pooled_barrett_batched_concurrency_audit tests/test_encrypt_host_shoup_pooled_barrett_batched_concurrency_audit.cpp)
gpufhe_test(test_encrypt_host_shoup_pooled_barrett_batched_pinned_matches_reference tests/test_encrypt_host_shoup_pooled_barrett_batched_pinned_matches_reference.cpp)
gpufhe_test(test_encrypt_host_shoup_pooled_barrett_batched_pinned_concurrency_audit tests/test_encrypt_host_shoup_pooled_barrett_batched_pinned_concurrency_audit.cpp)
gpufhe_test(test_keygen tests/test_keygen.cpp)
gpufhe_test(test_evalkeygen tests/test_evalkeygen.cpp)
# GOD PROMPT "CKKS PHASE 8": fully native (OpenFHE-free) isolation of
# evalkeygen_host/evalkeygen_host_sold -- proved correct (entry #41).
# NOTE: this test's own header comment describes an earlier, SUPERSEDED
# "sizeP=2 is the trigger" framing -- entry #42 found the real cause was
# NUM_PART=2 violating P>=Q_part_max, not sizeP's raw count. See entries
# #41/#42 and the tests registered below.
gpufhe_test(test_evalkeygen_isolated tests/test_evalkeygen_isolated.cpp)
# GOD PROMPT "CKKS PHASE 9": isolate ApproxSwitchCRTBasis (both directions:
# ModDown's P->Q and decompose's Q->complement) and the full ModDown
# combine chain, each against an independent CPU reference -- all PASS
# cleanly for sizeP in {1,2,3}, ruling out the basis-conversion kernel and
# both its constant families as the real defect (entry #42).
gpufhe_test(test_moddown_crt_basis_isolated tests/test_moddown_crt_basis_isolated.cpp)
gpufhe_test(test_decompose_crt_basis_isolated tests/test_decompose_crt_basis_isolated.cpp)
gpufhe_test(test_moddown_combine_isolated tests/test_moddown_combine_isolated.cpp)
# Instruments the REAL end-to-end pipeline (real secret/eval key, real
# decompose+accumulate+moddown) stage by stage across the full sizeQl x
# sizeP x numPart discriminating matrix -- isolates the failure to the
# relin-identity check specifically, never decompose or accumulate
# (both exact, zero-tolerance, always pass), pointing at the P-vs-Q_part
# ratio rather than a code defect (entry #42).
gpufhe_test(test_keyswitch_stage_trace tests/test_keyswitch_stage_trace.cpp)
# Decisive discriminator: holds sizeP=2 FIXED and sweeps ONLY numPart,
# showing the relin-identity residual flips from ~50% of q to ~1e-11% of q
# exactly when P - Q_part_max crosses from negative to positive -- proving
# sizeP=2 itself was never the defect (entry #42).
gpufhe_test(test_ks_pq_ratio_sweep tests/test_ks_pq_ratio_sweep.cpp)
gpufhe_test(test_encdec tests/test_encdec.cpp)
gpufhe_test(test_evalmult tests/test_evalmult.cpp)
gpufhe_test(test_ks_resident tests/test_ks_resident.cpp)
gpufhe_test(test_evalmult_rescale tests/test_evalmult_rescale.cpp)
gpufhe_test(test_ks_graph tests/test_ks_graph.cpp)
gpufhe_test(test_encode tests/test_encode.cpp)
gpufhe_test(test_packed_evalmult tests/test_packed_evalmult.cpp)
gpufhe_test(test_rotate tests/test_rotate.cpp)
gpufhe_test(test_ptops tests/test_ptops.cpp)
gpufhe_test(test_lintrans tests/test_lintrans.cpp)
gpufhe_test(test_native_params tests/test_native_params.cpp)
gpufhe_test(test_native_e2e tests/test_native_e2e.cpp)
gpufhe_test(test_c2s_engine tests/test_c2s_engine.cpp)
gpufhe_test(test_c2s tests/test_c2s.cpp)
gpufhe_test(test_s2c tests/test_s2c.cpp)
gpufhe_test(test_polyeval tests/test_polyeval.cpp)
gpufhe_test(test_poly2 tests/test_poly2.cpp)
gpufhe_test(test_cheb tests/test_cheb.cpp)
gpufhe_test(test_twsweep tests/test_twsweep.cpp)
gpufhe_test(test_evalmod tests/test_evalmod.cpp)
gpufhe_test(test_stagediag tests/test_stagediag.cpp)
gpufhe_test(test_stage2 tests/test_stage2.cpp)
gpufhe_test(test_modraise tests/test_modraise.cpp)
gpufhe_test(test_c2s_bsgs tests/test_c2s_bsgs.cpp)
gpufhe_test(test_boot_a tests/test_boot_a.cpp)
gpufhe_test(test_boot tests/test_boot.cpp)
# Exercises the feature-flagged hoisted rotation path end-to-end, now
# covering BOTH C2S and S2C (shares one KROT_hoist cache across both
# phases). NOT added to the default ctest run yet -- correctness is
# proven (byte-identical max slot err to the trusted path across 8
# interleaved A/B trials) but the measured speedup is small and noisy
# once run-order bias is corrected for; performance acceptance criteria
# have not cleanly passed (see research/NEXT_SESSION.md). Buildable
# explicitly.
add_executable(test_boot_hoist_enabled tests/test_boot_hoist_enabled.cpp)
target_link_libraries(test_boot_hoist_enabled PRIVATE gpufhe CUDA::cudart)
gpufhe_test(test_c2s_stages tests/test_c2s_stages.cpp)
gpufhe_test(test_alpha10 tests/test_alpha10.cpp)
gpufhe_test(test_chain tests/test_chain.cpp)
gpufhe_test(test_rot_resident tests/test_rot_resident.cpp)
gpufhe_test(test_stage_acc tests/test_stage_acc.cpp)
gpufhe_test(test_ntt_batched tests/test_ntt_batched.cpp)
gpufhe_test(test_intt_batched tests/test_intt_batched.cpp)
gpufhe_test(test_automorph_eval_fast tests/test_automorph_eval_fast.cpp)
# test_hoist_single.cpp: correct-order rotation-hoisting single-rotation
# correctness test (see research/NEXT_SESSION.md). Previously gated out as
# a known-failing reference while the key-generation convention was
# unresolved; now passes 27/27 cases (3 ciphertexts x 9 rotation indices,
# decrypted-plaintext comparison) and is promoted to the default suite.
gpufhe_test(test_hoist_single tests/test_hoist_single.cpp)
gpufhe_test(test_hoist_multi tests/test_hoist_multi.cpp)
gpufhe_test(test_hoist_levelreduced tests/test_hoist_levelreduced.cpp)
gpufhe_test(test_n131072_keysize tests/test_n131072_keysize.cpp)
# Isolates the n=131072 KROT-cache churn mechanism (bounded LRU eviction +
# evalkeygen_host_sold) from the rest of the bootstrap, to distinguish two
# competing explanations for why n=131072 OOMs far sooner than the
# KROT_CAP/KSC_CAP bound predicts: (a) the real rotFull()/keyAt() code
# generates the new key BEFORE evicting the old one, so the transient peak
# is (cap+1) keys, not cap; (b) allocator retention across repeated
# evict+regenerate cycles. See research/MEMORY_FEASIBILITY.md and
# research/NEXT_SESSION.md for context. Fast (~12 keygens, ~90s), no
# bootstrap/ciphertext machinery involved.
gpufhe_test(test_n131072_cache_churn tests/test_n131072_cache_churn.cpp)
gpufhe_test(test_boot_n131072 tests/test_boot_n131072.cpp)
gpufhe_test(test_hoist_stage0 tests/test_hoist_stage0.cpp)
# Validates rotation hoisting across ALL 3 real C2S stages (not just
# stage 0). Found and fixed two real bugs this round (segfault in
# ks_context_create from an empty-av/bv shape-only key; a correctness
# bug from generating a hoisted key directly at a reduced tower count
# instead of via the trusted path's own build-full-then-atLevel-view
# pattern) -- see research/NEXT_SESSION.md for the full story. Now
# passes. Heavy like test_boot (runs a full trusted C2S pass plus a
# full hoisted-only C2S pass, ~8GB peak RSS, ~2 min) -- same ctest
# memory caution as test_boot applies, nothing new.
gpufhe_test(test_hoist_allstages tests/test_hoist_allstages.cpp)
gpufhe_test(test_keyreuse tests/test_keyreuse.cpp)
gpufhe_test(bench_resident_chain bench/bench_resident_chain.cpp)
gpufhe_test(bench_batch bench/bench_batch.cpp)
gpufhe_test(bench_evalmult bench/bench_evalmult.cpp)
gpufhe_test(bench_evalmult_resident bench/bench_evalmult_resident.cpp)
gpufhe_test(bench_hoist_multi bench/bench_hoist_multi.cpp)
# GOD PROMPT "CKKS RUST + CUDA EVOLUTION"/"FAIR BENCHMARKS" rounds: Path A
# (existing C++ host + existing CUDA C++ kernel) of a three-way comparison
# against rust/crates/ckks-cuda's Path B (Rust host + the SAME kernel).
# Same n=1024 workload, same tower-0 modulus, same deterministic input,
# same four-tier (cold-start/context-init/steady-state/end-to-end)
# measurement structure -- see that file's own header comment for the
# full methodology and research/EXPERIMENT_LEDGER.md entry #32 for the
# result (equivalent within noise) and the build-flag bug found along the
# way.
gpufhe_test(bench_ntt_roundtrip_cpp_vs_rust bench/bench_ntt_roundtrip_cpp_vs_rust.cpp)
# GOD PROMPT "CKKS RUST + CUDA PHASE 2" round: the SECOND ported operation
# (single-modulus RNS rescale, rust::RescaleContext) gets the SAME fair
# four-tier C++-vs-Rust-host comparison as the first (NTT). This
# benchmark's own RescaleContextCpp deliberately does NOT use
# DeviceKSContext/ks_context_create (the production rescale_resident_raw's
# own dependency) -- it rebuilds root tables independently, mirroring
# rust::RescaleContext exactly, so both languages are measured doing the
# SAME minimal setup work. See that file's header comment and
# research/EXPERIMENT_LEDGER.md for the full methodology and results.
gpufhe_test(bench_rescale_cpp_vs_rust bench/bench_rescale_cpp_vs_rust.cpp)
# GOD PROMPT "CKKS RUST + CUDA PHASE 3" round: third operation's fair
# comparison (key switching) against rust/crates/ckks-cuda::KeySwitchContext.
# Unlike bench_rescale_cpp_vs_rust, this one is NOT a deliberately-narrowed
# reimplementation -- key switching genuinely needs the full eval key +
# DeviceKSContext/DeviceKSWork, so KeySwitchFixtureCpp calls the exact same
# existing native key-generation pipeline (native_primes/native_root/
# keygen_host/compute_keyswitch_constants/evalkeygen_host/
# ks_context_create/ks_work_create) as rust/crates/ckks-cuda/cpp_bridge/
# keyswitch_bridge.cpp wraps, so both languages measure the SAME setup.
gpufhe_test(bench_keyswitch_cpp_vs_rust bench/bench_keyswitch_cpp_vs_rust.cpp)
# GOD PROMPT "CKKS RUST + CUDA PHASE 4" round: fourth operation's fair
# comparison (EvalMult: tensor + relin + combine + rescale x2) against
# rust/crates/ckks-cuda::EvalMultContext. Reuses the SAME native key-switch
# fixture recipe as bench_keyswitch_cpp_vs_rust, extended with
# LaunchTensor/LaunchCombine and two rescale_resident_raw calls on the SAME
# DeviceKSContext (it only needs plain moduli + cached root tables, never
# the eval key -- entry #35), so both languages measure the SAME setup.
gpufhe_test(bench_evalmult_cpp_vs_rust bench/bench_evalmult_cpp_vs_rust.cpp)
# GOD PROMPT "CKKS PHASE 11": the real-world per-stage performance
# baseline -- encode/encrypt/tensor/relin/rescale/decrypt/decode, each
# independently timed, across n=1024/2048/4096/8192, with mandatory
# P>=Q_part_max validation and a correctness gate (RNS-level +  real
# encrypted round trip) before any timing is trusted. See research/
# EXPERIMENT_LEDGER.md entry #44.
gpufhe_test(bench_evalmult_e2e_breakdown bench/bench_evalmult_e2e_breakdown.cpp)
# GOD PROMPT "CKKS PHASE 12": standalone (no GPU, no encryption) A/B
# benchmark isolating exactly the OLD (parallel O(N^2)) vs NEW (FFT,
# O(N log N)) encode_host/decode_host algorithmic change.
gpufhe_test(bench_encode_fft_vs_parallel bench/bench_encode_fft_vs_parallel.cpp)
# GOD PROMPT "CKKS PHASE 13": encrypt_host internal stage-level profiler
# (RNG/embed/alloc/upload/NTT-kernel/download/free/combine breakdown)
# plus three A/B benchmarks isolating, in turn, the Shoup-modmul change
# alone, the buffer-pooling change alone (on top of Shoup), and all
# three variants (old/Shoup/Shoup+pooled) together in one interleaved
# run. See research/ENCRYPT_PROFILING.md and research/
# EXPERIMENT_LEDGER.md entry #46.
gpufhe_test(bench_encrypt_profile bench/bench_encrypt_profile.cpp)
gpufhe_test(bench_encrypt_shoup_vs_reference bench/bench_encrypt_shoup_vs_reference.cpp)
gpufhe_test(bench_encrypt_shoup_pooled_vs_reference bench/bench_encrypt_shoup_pooled_vs_reference.cpp)
gpufhe_test(bench_encrypt_all_variants bench/bench_encrypt_all_variants.cpp)
# GOD PROMPT "CKKS PHASE 14": isolated embed-loop-only A/B benchmark
# (OLD signed-% vs NEW per-tower Barrett) plus a unified three-way
# interleaved benchmark (OLD encrypt_host / Phase 13 Shoup+pooled /
# Phase 14 +Barrett-embed) in one run. See research/
# EMBED_AND_POOL_HARDENING.md and research/EXPERIMENT_LEDGER.md entry
# #47.
gpufhe_test(bench_embed_barrett_vs_reference bench/bench_embed_barrett_vs_reference.cpp)
gpufhe_test(bench_encrypt_phase14_all_variants bench/bench_encrypt_phase14_all_variants.cpp)
# GOD PROMPT "CKKS PHASE 15": unified 4-way interleaved benchmark
# (OLD/Phase13/Phase14/Phase15-Experiment1) plus a dedicated isolated
# A/B isolating Experiment 2's (pinned memory) incremental contribution
# on top of Experiment 1. See research/TRANSFER_PROFILING.md and
# research/EXPERIMENT_LEDGER.md entry #48.
gpufhe_test(bench_encrypt_phase15_all_variants bench/bench_encrypt_phase15_all_variants.cpp)
gpufhe_test(bench_encrypt_phase15_pinned_vs_batched bench/bench_encrypt_phase15_pinned_vs_batched.cpp)
gpufhe_test(test_mult_barrett tests/test_mult_barrett.cpp)
# Directive "VERIFY OOM DIAGNOSIS AND REMOVE THE LARGE-N DIAGONAL
# BOTTLENECK", Priority 3: proves an O(S*2^RST) analytic replacement for
# stageDiags()/stageDiagsFwd()'s O(S^2*RST) brute-force probing (measured
# 268s/call at n=131072, research/MEMORY_FEASIBILITY.md) is exactly
# equivalent to the existing brute-force oracle across many (n,LST,g,
# direction) combinations, then benchmarks both at scales where the oracle
# is still tractable (n<=32768) plus the fast path alone at n=131072. Pure
# CPU, no CUDA/GPU/bootstrap involved -- cheap and safe to run in the
# default suite. Does not itself change tests/test_boot.cpp or
# tests/test_boot_n131072.cpp; see research/NEXT_SESSION.md for the wiring
# decision once this proves out.
add_executable(test_c2s_diag_fast tests/test_c2s_diag_fast.cpp)
add_test(NAME test_c2s_diag_fast COMMAND test_c2s_diag_fast)
# Directive "ISOLATE GPU MEMORY GROWTH AND PROVE LARGE-N FEASIBILITY":
# isolates the real ks_context_create/keyswitch_resident/ks_context_destroy
# per-diagonal device-memory lifecycle (plus ks_work_create/destroy per
# stage) from the rest of the n=131072 bootstrap, using the REAL production
# APIs and the same bounded-LRU KROT cache, across more diagonals and stage
# boundaries than the 6th real attempt survived to reach. See
# research/MEMORY_FEASIBILITY.md / NEXT_SESSION.md for why. Registered via
# gpufhe_test like every other test here, but HEAVY (~80 real n=131072
# keygens, several minutes) -- exclude explicitly with
# `ctest -E test_n131072_gpu_lifecycle` in any full-suite run, same
# convention already used for test_boot_n131072.
gpufhe_test(test_n131072_gpu_lifecycle tests/test_n131072_gpu_lifecycle.cpp)
# Directive "GOD PROMPT -- CKKS ENGINE": isolates whether
# rescale_resident_raw + native_rescale_consts work correctly at n=1024
# (test_polyeval's scale) -- neither has been independently validated at
# this n anywhere else in the repo (test_rescale_resident.cpp uses n=32768
# sourced from OpenFHE's own constants; test_boot.cpp uses n=8192 with
# native_rescale_consts). See research/EXPERIMENT_LEDGER.md for the
# instrumented trace that localized test_polyeval's bug to exactly this
# combination. Fast, no bootstrap, no OpenFHE dependency.
add_executable(test_rescale_n1024 tests/test_rescale_n1024.cpp)
target_link_libraries(test_rescale_n1024 PRIVATE gpufhe CUDA::cudart)
add_test(NAME test_rescale_n1024 COMMAND test_rescale_n1024)
# GOD PROMPT "CKKS CORRECTNESS BREAKTHROUGH" round, Priority 1 Step 2: pure
# NTT/INTT round-trip at n=1024 across test_rescale_n1024's exact modulus
# chain, independent of rescale/keyswitch/SwitchModulus entirely -- isolates
# whether the transform kernels themselves are correct at this n.
add_executable(test_ntt_roundtrip_n1024 tests/test_ntt_roundtrip_n1024.cpp)
target_link_libraries(test_ntt_roundtrip_n1024 PRIVATE gpufhe CUDA::cudart)
add_test(NAME test_ntt_roundtrip_n1024 COMMAND test_ntt_roundtrip_n1024)
# GOD PROMPT "CKKS CORRECTNESS BREAKTHROUGH" round, Priority 1 Step 3:
# scalar differential test driving rescale_resident_raw's REAL orchestration
# (SwitchModulus + NTT + RescaleFuse, real row_of/root-table indexing) with
# constant-polynomial inputs against an independent Python/arbitrary-
# precision CRT reference, after Step 2 cleared the NTT kernels and a
# by-hand check cleared the s1/s2 precomputed-constant formula.
add_executable(test_rescale_scalar_diff_n1024 tests/test_rescale_scalar_diff_n1024.cpp)
target_link_libraries(test_rescale_scalar_diff_n1024 PRIVATE gpufhe CUDA::cudart)
add_test(NAME test_rescale_scalar_diff_n1024 COMMAND test_rescale_scalar_diff_n1024)
# GOD PROMPT "CKKS CORRECTNESS BREAKTHROUGH" round, Priority 1 Step 3
# (continued): the scalar-diff test above used CONSTANT polynomials, a
# degenerate case where coefficient-domain and eval-domain data coincide
# (NTT(constant)=constant). This vector-diff test uses INDEPENDENT
# per-coefficient random residues, forward-NTT'ing both the real input and
# an RNS-only reference output, to check rescale_resident_raw end-to-end on
# genuinely non-constant data.
add_executable(test_rescale_vector_diff_n1024 tests/test_rescale_vector_diff_n1024.cpp)
target_link_libraries(test_rescale_vector_diff_n1024 PRIVATE gpufhe CUDA::cudart)
add_test(NAME test_rescale_vector_diff_n1024 COMMAND test_rescale_vector_diff_n1024)
# GOD PROMPT "CKKS CORRECTNESS BREAKTHROUGH" round, Priority 1 Step 3
# (targeted hypothesis): the vector-diff test above calls
# rescale_resident_raw exactly ONCE per trial, but test_rescale_n1024.cpp
# (the original failing reproduction) calls it TWICE back-to-back sharing
# the same scr/drp scratch buffers on the same stream (once for c0, once
# for c1). This test isolates exactly that call pattern.
add_executable(test_rescale_double_call_n1024 tests/test_rescale_double_call_n1024.cpp)
target_link_libraries(test_rescale_double_call_n1024 PRIVATE gpufhe CUDA::cudart)
add_test(NAME test_rescale_double_call_n1024 COMMAND test_rescale_double_call_n1024)
# GOD PROMPT "CKKS CORRECTNESS BREAKTHROUGH" round, Priority 1 Step 3
# (confirming experiment): byte-for-byte copy of test_rescale_n1024.cpp
# except Delta=2^55 instead of 2^50, matching qLast's bit-size as sound
# CKKS parameter selection requires. Tests the hypothesis that
# test_rescale_n1024's maxerr=24602 failure is a scale/modulus-size
# parameter mismatch in that test's OWN design, not a rescale_resident_raw
# defect (which the three differential tests above already cleared).
add_executable(test_rescale_n1024_matched_scale tests/test_rescale_n1024_matched_scale.cpp)
target_link_libraries(test_rescale_n1024_matched_scale PRIVATE gpufhe CUDA::cudart)
add_test(NAME test_rescale_n1024_matched_scale COMMAND test_rescale_n1024_matched_scale)
# GOD PROMPT "CKKS CORRECTNESS BREAKTHROUGH" round, Priority 6: validates
# ks_context_create's new K.av/K.bv size check (src/keyswitch_resident.cpp)
# -- confirms the historical empty-av/bv footgun now throws a clear
# std::invalid_argument instead of segfaulting, and that a correctly-
# populated K still succeeds with zero behavior change.
add_executable(test_ks_context_create_validation tests/test_ks_context_create_validation.cpp)
target_link_libraries(test_ks_context_create_validation PRIVATE gpufhe CUDA::cudart)
add_test(NAME test_ks_context_create_validation COMMAND test_ks_context_create_validation)
# GOD PROMPT "CKKS ENGINE: RESOLVE POLYEVAL CORRECTNESS" round -- scratch
# diagnostic verifying encode_host's actual output for a constant input,
# used to build an independent reference for test_polyeval's raw
# tensor-product residue. Not wired into ctest (diagnostic only).
add_executable(test_encode_impulse_check tests/test_encode_impulse_check.cpp)
target_link_libraries(test_encode_impulse_check PRIVATE gpufhe CUDA::cudart)
# GOD PROMPT "CKKS ENGINE: RESOLVE POLYEVAL CORRECTNESS" round: direct,
# unambiguous eval-domain differential check of keyswitch_core_resident's
# core algebraic property (ba0+ba1*s =?= a*s mod q, per tower, per
# coefficient) using NATIVELY-generated KeySwitchConstants exactly as
# test_polyeval builds them (compute_keyswitch_constants+evalkeygen_host,
# numPart=2, n=1024) -- a combination test_keyswitch.cpp never covers
# (it only validates against OpenFHE-sourced constants at n=32768).
add_executable(test_keyswitch_core_native_diff tests/test_keyswitch_core_native_diff.cpp)
target_link_libraries(test_keyswitch_core_native_diff PRIVATE gpufhe CUDA::cudart)
# NOT registered with add_test: this checks an algebraic formula
# (ba0+ba1*s =?= a*s^2) that FAILED even when applied to OpenFHE's own
# bit-exact-proven-correct output (see test_keyswitch.cpp's own appended
# check) -- the formula itself is unconfirmed, so this failing is expected
# and NOT a regression signal. Kept as a documented, reproducible research
# artifact only.
# GOD PROMPT "CKKS ENGINE: RESOLVE POLYEVAL CORRECTNESS" round: same
# eval-domain algebraic check as test_keyswitch_core_native_diff, but with
# a=t2 from a REAL tensor product of two fresh valid encryptions (exactly
# what test_polyeval's mulCt feeds keyswitch_core_resident), to test
# whether the ~50%-of-q failure seen with purely random "a" is specific to
# out-of-domain random input, or also present for realistic input.
add_executable(test_keyswitch_core_realct_diff tests/test_keyswitch_core_realct_diff.cpp)
target_link_libraries(test_keyswitch_core_realct_diff PRIVATE gpufhe CUDA::cudart)
# NOT registered with add_test, same reason as test_keyswitch_core_native_
# diff above -- the underlying formula is unconfirmed, not the library.
# GOD PROMPT "CKKS ENGINE: RESOLVE POLYEVAL CORRECTNESS" round: checks
# test_polyeval.cpp's per-mulCt-call rebuilt Kl against a top-level K for
# the FIRST mulCt call (tw=sizeQ, no slicing effect) -- bit-exact ba0/ba1
# comparison, independent of the still-unresolved algebraic-formula
# question above.
add_executable(test_polyeval_kl_rebuild_diff tests/test_polyeval_kl_rebuild_diff.cpp)
target_link_libraries(test_polyeval_kl_rebuild_diff PRIVATE gpufhe CUDA::cudart)
add_test(NAME test_polyeval_kl_rebuild_diff COMMAND test_polyeval_kl_rebuild_diff)
# Directive "GOD PROMPT -- CKKS ENGINE": Priority 2 (RAII refactor), Phase C.
# Validates include/ks_raii.h's ScopedKSContext/ScopedKSWork wrappers in
# ISOLATION (construction/destruction, repeated cycles with stable GPU
# memory, move semantics, double-reset safety, C-API interop) before any
# production call site is touched. Uses n=8192 (proven scale), deliberately
# avoiding the unrelated small-n rescale bug found this same round.
gpufhe_test(test_ks_raii tests/test_ks_raii.cpp)
