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
  src/encrypt_host_shoup_pooled_barrett_prewarm.cpp
  src/encrypt_host_shoup_pooled_barrett_batched_registered.cpp
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
  kernels/finance_resident.cu
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

# GOD PROMPT "CKKS PHASE 19": `gpufhe_test`'s `add_test(...)` call means
# EVERY target registered with it -- including pure timing/benchmark
# diagnostics, not just pass/fail correctness tests -- is automatically
# picked up by a plain `ctest` run. In practice this project's actual
# regression gate has been the Rust suite (`cargo test`), not a full
# `ctest` run (gated out by this machine's persistent host-memory
# constraint; see research/TRANSFER_PROFILING.md), so this has not yet
# caused a real "ctest reports noise as a failure" incident. But it is
# a real latent landmine: a future `ctest` run would silently execute
# every pinned-memory/THP timing diagnostic (several seconds each,
# none of them a correctness check) as if it were an ordinary test.
# `gpufhe_benchmark` is the SAME build-target setup as `gpufhe_test`,
# minus the `add_test(...)` call -- the binary still builds normally
# and can still be run directly or via `cmake --build . --target
# <name>`, it simply is not registered with CTest at all, so it can
# never be silently absorbed into a `ctest` regression run. This is
# deliberately NOT applied retroactively to every other `bench_*`
# target in this file (several pre-date this phase and are out of this
# investigation's scope); see research/TRANSFER_PROFILING.md's Phase
# 19 section for the full audit and why the fix is scoped this way.
function(gpufhe_benchmark name src)
  add_executable(${name} ${src})
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${name} PRIVATE gpufhe CUDA::cudart)
  target_include_directories(${name} PRIVATE
    /usr/local/include/openfhe
    /usr/local/include/openfhe/pke
    /usr/local/include/openfhe/core
    /usr/local/include/openfhe/binfhe)
  target_link_libraries(${name} PRIVATE OPENFHEpke OPENFHEcore)
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
# GOD PROMPT "CKKS PHASE 28": concurrency stress test for encode_host_fft/
# decode_host_fft, found necessary before any production-integration
# decision -- these functions lazily populate an UNSYNCHRONIZED
# static std::map cache (fft_twiddles, slot_idx in src/encode_fft.cpp),
# the same pattern class as keyswitch_resident.cpp's g_rtab/upc. Uses REAL
# std::thread, never fork() -- Phase 27 found fork() after CUDA init unsafe;
# this file touches no CUDA, but the discipline is followed regardless.
gpufhe_test(test_encode_fft_concurrency tests/test_encode_fft_concurrency.cpp)
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
gpufhe_test(test_encrypt_host_shoup_concurrency tests/test_encrypt_host_shoup_concurrency.cpp)
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
# GOD PROMPT "CKKS PHASE 16": correctness + lifecycle/concurrency tests
# for the explicit pinned-memory PREWARMING API (encrypt_prewarm_
# pinned_buffer/encrypt_pinned_buffer_is_ready/encrypt_host_shoup_
# pooled_barrett_prewarmed). RETAINED AS AN OPT-IN EXPERIMENTAL
# FEATURE, NOT promoted to production and NOT wired into the Rust FFI
# bridge this round -- see research/TRANSFER_PROFILING.md's Phase 16
# section and research/EXPERIMENT_LEDGER.md entry #49 for the full
# workload-level decision.
gpufhe_test(test_encrypt_host_prewarm_matches_reference tests/test_encrypt_host_prewarm_matches_reference.cpp)
gpufhe_test(test_encrypt_host_prewarm_concurrency_audit tests/test_encrypt_host_prewarm_concurrency_audit.cpp)
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
# GOD PROMPT "CKKS PHASE 30": root-cause investigation and fix for
# test_ks_graph's CUDA-graph-capture defect in keyswitch_resident (see
# research/KS_GRAPH_CAPTURE_INVESTIGATION.md for the full story -- a
# dangling-host-pointer bug AND a separate value-aliasing bug, both in the
# per-call host staging buffers ks_accumulate_qp/ks_moddown upload via
# cudaMemcpyAsync, now fixed with per-(part,role) dedicated buffers owned
# by DeviceKSWork). Native (no OpenFHE), fast. Full lifecycle coverage:
# capture/instantiate/replay/updated-input-replay/multi-graph/destruction
# order/memory-leak check -- all deterministic, 14/14 passing across many
# repeated runs. The thing found NOT to be fully resolved at the time (a
# believed separate concurrency race in the key-generation/context-setup
# pipeline, triggered only under many SIMULTANEOUS threads building fresh
# contexts) was root-caused and fixed in Phase 31 -- see test_ks_graph_setup_
# concurrency_diagnostic.cpp below, now a real ctest-registered regression
# test (it turned out to be CUDA stream-capture API misuse in that test
# file, not a defect in keygen_host/evalkeygen_host/ks_context_create).
add_executable(test_ks_graph_lifecycle tests/test_ks_graph_lifecycle.cpp)
target_link_libraries(test_ks_graph_lifecycle PRIVATE gpufhe CUDA::cudart)
add_test(NAME test_ks_graph_lifecycle COMMAND test_ks_graph_lifecycle)
# GOD PROMPT "CKKS PHASE 31": root-caused and fixed. What Phase 30 left as
# a non-ctest, nondeterministic diagnostic for a believed setup/keygen-path
# race turned out to be CUDA stream-capture API misuse in the TEST itself
# (wrong capture mode; allocate/free racing with another thread's active
# capture) -- see this file's own header comment and research/
# SETUP_PATH_CONCURRENCY_INVESTIGATION.md for the full evidence trail. No
# changes were needed in keygen_host/evalkeygen_host/ks_context_create/
# ks_work_create/keyswitch_resident. Now deterministic (25/25 plain + 3/3
# compute-sanitizer runs clean) and registered as a real regression test.
add_executable(test_ks_graph_setup_concurrency_diagnostic tests/test_ks_graph_setup_concurrency_diagnostic.cpp)
target_link_libraries(test_ks_graph_setup_concurrency_diagnostic PRIVATE gpufhe CUDA::cudart)
add_test(NAME test_ks_graph_setup_concurrency_diagnostic COMMAND test_ks_graph_setup_concurrency_diagnostic)
# GOD PROMPT "CKKS PHASE 30", Objective G: direct vs CUDA-graph performance
# for keyswitch_resident at a realistic (n=8192/tw=30, bootstrap-scale)
# configuration. Honest measurement -- reports whether replay is faster,
# does not force a favorable number. Native, no OpenFHE.
add_executable(bench_ks_graph_perf bench/bench_ks_graph_perf.cpp)
target_link_libraries(bench_ks_graph_perf PRIVATE gpufhe CUDA::cudart)
add_test(NAME bench_ks_graph_perf COMMAND bench_ks_graph_perf)
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
# GOD PROMPT "CKKS PHASE 25": multi-input bootstrap repeatability driver.
# Deliberately NOT registered with `gpufhe_test` / `add_test`, for the same
# reason as test_boot_hoist_enabled above: this runs the FULL n=8192
# bootstrap pipeline NUM_CASES times over, so its peak host RSS is in
# test_boot's own documented 8.12-8.32GB band. Letting a plain `ctest`
# absorb it would hand a routine regression run an 8GB+ memory
# requirement, which this machine has never been able to satisfy (see
# research/PHASE25_RECOVERY.md section 5). Build it explicitly, gate it on
# research/benchmarks/bootstrap_preflight.ps1, and run it by hand:
#   cmake --build build2 --target test_boot_multi_input -j4
add_executable(test_boot_multi_input tests/test_boot_multi_input.cpp)
target_link_libraries(test_boot_multi_input PRIVATE gpufhe CUDA::cudart)
# GOD PROMPT "CKKS PHASE 25": the RUNNABLE sibling of the driver above. Unlike
# test_boot_multi_input it is cheap (n=1024, no bootstrap stages, no key
# switching), so it IS registered with ctest -- it is the regression gate on
# tests/boot_driver_harness.h, the shared harness BOTH drivers include. It
# validates the case loop, the deterministic-message profiles, the
# test-seed/crypto-seed separation, the per-stage failure attribution (with
# deliberate fault injection) and the JSONL emitter. It executes NO bootstrap
# stage and must never be cited as bootstrap evidence; see the file's own
# header for that boundary stated in full.
add_executable(test_boot_multi_input_small tests/test_boot_multi_input_small.cpp)
target_link_libraries(test_boot_multi_input_small PRIVATE gpufhe CUDA::cudart)
add_test(NAME test_boot_multi_input_small COMMAND test_boot_multi_input_small)
# GOD PROMPT "CKKS PHASE 25": standalone reproducer for the concurrency defect
# in the unsynchronized process-global device-table caches on the resident
# rotation path (src/keyswitch_resident.cpp's upc()/g_rtab and src/rotate.cpp's
# dev_tab()/autoPermMap()). Required by the governing directive before any
# change to src/keyswitch_resident.cpp internals. Cheap (n<=4096, tw<=6, ~9
# small keygens), so it IS registered with ctest. Its race window is the first
# concurrent round in a fresh process, while the caches are still cold --
# see the file's own header for why a single pass is evidence, not proof.
add_executable(test_rotate_concurrent_cache tests/test_rotate_concurrent_cache.cpp)
target_link_libraries(test_rotate_concurrent_cache PRIVATE gpufhe CUDA::cudart)
add_test(NAME test_rotate_concurrent_cache COMMAND test_rotate_concurrent_cache)
# GOD PROMPT "CKKS PHASE 25": controlled A/B for rotation hoisting
# (decompose-sharing) at a runnable scale -- ~600MB, unlike the full-bootstrap
# A/B (research/benchmarks/randomized_ab.sh) which needs ~8GB x 20 runs and has
# been memory-blocked since Phase 23. Registered with `gpufhe_benchmark`, NOT
# `gpufhe_test`: it is a timing diagnostic, and this file's own Phase 19 comment
# explains why timing targets must never be absorbed into a `ctest` run.
# It self-verifies bit-exactness before reporting any timing.
gpufhe_benchmark(bench_hoist_ab bench/bench_hoist_ab.cpp)
# GOD PROMPT "CKKS PHASE 26", Mission 4: standalone experiment for a resident
# eval-key DeviceKSContext cache, keyed (k, tw) -- never k alone (see
# EXPERIMENT_LEDGER.md entry #20). Verifies the per-diagonal eval-key upload
# size independently (measured from a live KeySwitchConstants, not re-derived
# from source) and measures the actual transfer-volume/time reduction from
# reusing resident contexts. Does NOT modify src/keyswitch_resident.cpp and
# does NOT wire anything into production -- registered as a benchmark, not a
# test, for the same reason as bench_hoist_ab.cpp above.
gpufhe_benchmark(bench_ks_context_resident_ab bench/bench_ks_context_resident_ab.cpp)
# GOD PROMPT "CKKS PHASE 27", Mission 1: diagnostic probe quantifying the
# ks_context_create vs ks_context_destroy timing asymmetry identified while
# reconstructing bench_ks_context_resident_ab.cpp's A/B methodology (Arm A's
# measured interval includes 48 destroys; Arm B's includes 0, deferred past
# its timer). Not a correctness test -- a timing diagnostic, hence
# gpufhe_benchmark not gpufhe_test.
gpufhe_benchmark(probe_create_destroy_cost bench/probe_create_destroy_cost.cpp)
# GOD PROMPT "CKKS PHASE 27", Missions 2+6: realistic A/B against the actual
# bootstrap access pattern (30-key S2C double real/imaginary pass, atLevel-
# style key construction), replacing Phase 26's toy round-robin benchmark's
# memory model and access pattern with faithful ones extracted from
# tests/test_boot.cpp's own source.
gpufhe_benchmark(bench_realistic_ks_cache bench/bench_realistic_ks_cache.cpp)
# GOD PROMPT "CKKS PHASE 27", Mission 5: full test suite for
# include/ks_resident_cache.h -- correctness, cache behavior, concurrency
# (including a system()-invoked-helper-process test that the enforced
# single-thread abort actually fires -- a bare fork() was tried first and
# found unsafe once CUDA is active, see tests/ks_cache_thread_violation_helper.cpp),
# and memory. Cheap (n=1024), registered with gpufhe_test.
gpufhe_test(test_ks_resident_cache tests/test_ks_resident_cache.cpp)
# GOD PROMPT "CKKS PHASE 27", Mission 5 (C3): standalone helper executable
# that deliberately violates KsResidentCache's single-thread contract.
# Invoked by test_ks_resident_cache.cpp via system() (fork+exec, CUDA-safe)
# rather than a bare fork() from an already-CUDA-active process -- see that
# test's own comment for why a bare fork() corrupted CUDA state instead.
# Registered WITHOUT add_test (gpufhe_benchmark, not gpufhe_test): this
# binary is EXPECTED to abort every run, so ctest must never pick it up as
# an independent pass/fail target -- it exists only to be invoked by name
# from test_ks_resident_cache.cpp's own working directory.
gpufhe_benchmark(ks_cache_thread_violation_helper tests/ks_cache_thread_violation_helper.cpp)
gpufhe_test(test_c2s_stages tests/test_c2s_stages.cpp)
gpufhe_test(test_alpha10 tests/test_alpha10.cpp)
gpufhe_test(test_chain tests/test_chain.cpp)
gpufhe_test(test_rot_resident tests/test_rot_resident.cpp)
gpufhe_test(test_stage_acc tests/test_stage_acc.cpp)
# GOD PROMPT "CKKS PHASE 26", Mission 2: extended differential test for the
# fused pt-mul-accumulate kernel (LaunchPtMulAcc) -- multiple sizes, boundary
# values, multi-diagonal accumulation, repeated executions, and concurrent
# independent chains. Does NOT replace tests/test_stage_acc.cpp's existing
# single-shot gate; both stay in the default suite. Cheap (largest config is
# the real n=8192/tw=30 bootstrap case, single accumulate), so registered with
# gpufhe_test like its sibling.
gpufhe_test(test_stage_acc_differential tests/test_stage_acc_differential.cpp)
# GOD PROMPT "CKKS PHASE 27", Mission 8: end-to-end STAGE-level A/B for
# wiring LaunchPtMulAcc into the bootstrap's accumulate loop, including
# transfers -- not just the isolated kernel (already proven bit-exact,
# tests/test_stage_acc_differential.cpp). pt_to_eval_host's own round trip
# is deliberately unchanged in both arms; rotation is simulated identically
# in both arms. See the file's own header for the full scope statement.
gpufhe_benchmark(bench_stage_acc_integration bench/bench_stage_acc_integration.cpp)
# GOD PROMPT "CKKS PHASE 28", Objectives A+B: full-statistics re-verification
# of Phase 27's stage-level fused-kernel A/B (mean/median/stddev/min/max/all
# samples, not just median+min/max), an explicit measurement of the
# Arm-A-vs-Arm-B allocation/deallocation asymmetry, and a CUDA-event-based
# decomposition of the stage into its cost components (alloc, encode,
# pt_to_eval_host, H2D, kernel execution, D2H, free).
gpufhe_benchmark(bench_stage_timeline bench/bench_stage_timeline.cpp)
# GOD PROMPT "CKKS PHASE 28": isolated stage-level A/B for swapping
# encode_host for encode_host_fft at tests/test_boot.cpp's two per-diagonal
# hot-loop call sites. A SEPARATE experiment from the fused
# pt-mul-accumulate kernel -- only the encode call differs between arms.
gpufhe_benchmark(bench_encode_fft_stage bench/bench_encode_fft_stage.cpp)
# GOD PROMPT "CKKS PHASE 28", Objective C: standalone, additive-only probe for
# the "make pt_to_eval_host's output device-resident, skip its D2H and the
# fused kernel's subsequent re-upload of that plaintext" hypothesis. Touches
# NO production file (not even src/rotate.cpp) -- times only a raw
# cudaMemcpy of the exact buffer size pt_to_eval_host moves, to put a ceiling
# on the hypothesis before risking any change. Result: ~2.2% of stage time
# at most, an order of magnitude below the encode_host_fft swap's measured
# 2.5x -- NOT pursued further this round; see
# research/BOOTSTRAP_GPU_RESIDENCY_AUDIT.md's Phase 28 addendum.
gpufhe_benchmark(bench_residency_hypothesis bench/bench_residency_hypothesis.cpp)
# GOD PROMPT "CKKS PHASE 28", Objective E: the first real consumer-GPU CKKS
# workload through the actual library API only (keygen_host/encode_host_fft/
# encrypt_host/ct_add_ct_host/decrypt_host/decode_host_fft -- no private
# kernels). Privacy-preserving aggregation of K client-held values, needing
# no relin/rotation/rescale (a pure ciphertext sum consumes no depth) --
# see the file's own header for why this workload was chosen over a
# weighted-sum/EvalMult variant. Registered with ctest: cheap (n=8192,
# sizeQ=4, no bootstrap machinery, ~seconds).
gpufhe_test(bench_private_aggregation_workload bench/bench_private_aggregation_workload.cpp)
# GOD PROMPT "CKKS PHASE 29", Objectives C+D+E: decomposes the aggregation
# workload into per-stage wall-clock times (encode/encrypt/aggregate/
# decrypt/decode), full statistics (mean/median/stddev/min/max), and
# compares Phase 28's actual encrypt_host call against the real production
# path the Rust bridge uses, encrypt_host_shoup_pooled_barrett_batched --
# already bit-exact verified, already measured faster in isolation; this
# checks whether that holds inside the real K-client application.
gpufhe_test(bench_aggregation_decomposed bench/bench_aggregation_decomposed.cpp)
# GOD PROMPT "CKKS PHASE 29", Objective D: independent CPU-only baseline
# (OpenFHE, OPENFHE_GPU=0) for the aggregation workload -- gpufhe itself has
# no CPU-only encrypt path (every encrypt_host variant uses CUDA for its
# NTT step), so this is a DIFFERENT library's implementation, not this
# project's own CPU mode. See the file's own header for the full caveat.
gpufhe_test(bench_aggregation_openfhe_cpu_baseline bench/bench_aggregation_openfhe_cpu_baseline.cpp)
# GOD PROMPT "CKKS PHASE 29", Objective F: VRAM/RSS/runtime scaling across
# client counts (1,2,4,8,16,32,64), production encrypt path.
gpufhe_test(bench_aggregation_client_scaling bench/bench_aggregation_client_scaling.cpp)
# GOD PROMPT "CKKS PHASE 29", Objective A: extends the Phase 12/28 FFT
# encode/decode differential test with different scales (Delta) and
# near-numerical-boundary values -- the two dimensions that test did not
# cover. See the file's own header for why "different modulus/tower
# configurations" does not apply to these particular functions.
gpufhe_test(test_encode_fft_scale_and_boundary tests/test_encode_fft_scale_and_boundary.cpp)
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
# GOD PROMPT "CKKS PHASE 16": workload-level benchmark for the
# prewarming API -- context-creation/prewarm cost, first-operation and
# steady-state latency, and the three workload-profile comparisons
# (cold-start/typical/long-lived) the decision in research/
# TRANSFER_PROFILING.md's Phase 16 section is based on.
# GOD PROMPT "CKKS PHASE 19" audited this whole pinned-memory/THP
# diagnostic cluster (Phases 16-19) and found every one of them was
# registered via `gpufhe_test`, which calls `add_test(...)` -- meaning
# a plain `ctest` run would silently execute these multi-second timing
# benchmarks as if they were pass/fail correctness tests. Switched
# the whole cluster to `gpufhe_benchmark` (same build target, no
# `add_test`): still buildable and runnable directly or via `cmake
# --build . --target <name>`, but never picked up by `ctest`. See
# research/TRANSFER_PROFILING.md's Phase 19 section for the full audit
# (this fix is deliberately scoped to this cluster only -- other
# pre-existing `bench_*` targets elsewhere in this file are left
# unchanged, out of this investigation's scope).
gpufhe_benchmark(bench_encrypt_prewarm_workloads bench/bench_encrypt_prewarm_workloads.cpp)
# GOD PROMPT "CKKS PHASE 17": pure diagnostic instruments for the
# allocation-cost variance Phase 16 found but could not explain. None
# of these call into gpufhe:: code at all (pure CUDA/Linux allocator
# timing). See research/
# TRANSFER_PROFILING.md's Phase 17 section and research/
# EXPERIMENT_LEDGER.md entry #50 for the full investigation and
# ranked root-cause assessment.
gpufhe_benchmark(bench_pinned_alloc_diagnostic bench/bench_pinned_alloc_diagnostic.cpp)
gpufhe_benchmark(bench_pinned_alloc_diagnostic2 bench/bench_pinned_alloc_diagnostic2.cpp)
gpufhe_benchmark(bench_thp_diagnostic bench/bench_thp_diagnostic.cpp)
# GOD PROMPT "CKKS PHASE 18": controlled THP experiment diagnostics.
# The system-wide THP toggle was blocked by permissions (documented,
# not worked around); these two standalone files instead investigate
# the per-sample allocation-cost variables under the unchanged THP
# setting, and the one THP-related variable controllable without
# root (per-call madvise(MADV_HUGEPAGE/NOHUGEPAGE) hints). Neither
# calls into gpufhe:: code. See research/
# TRANSFER_PROFILING.md's Phase 18 section and research/
# EXPERIMENT_LEDGER.md entry #51.
gpufhe_benchmark(bench_pinned_alloc_diagnostic3 bench/bench_pinned_alloc_diagnostic3.cpp)
gpufhe_benchmark(bench_thp_hint_by_size bench/bench_thp_hint_by_size.cpp)
# GOD PROMPT "CKKS PHASE 19": controlled A-E host-allocation-path
# comparison (pageable malloc, pure-Linux mlock, cudaMallocHost,
# cudaHostAlloc with two documented flag variants, cudaHostRegister),
# with CUDA context initialization isolated as its own stage. Does
# not call into gpufhe:: code. See research/TRANSFER_PROFILING.md's
# Phase 19 section and research/EXPERIMENT_LEDGER.md entry #52.
gpufhe_benchmark(bench_alloc_path_comparison bench/bench_alloc_path_comparison.cpp)
# GOD PROMPT "CKKS PHASE 20": CUDA-registered host memory prototype.
# RegisteredHostBuffer (include/registered_host_buffer.h) is the new
# move-only RAII ownership model; its own standalone lifecycle/
# concurrency audit and the real encrypt_host_shoup_pooled_barrett_
# batched_registered correctness/concurrency tests are genuine
# correctness/lifecycle tests -- registered via gpufhe_test (CTest
# SHOULD run these). The dedicated full-lifecycle timing comparison
# (bench_encrypt_registered_lifecycle) is a pure timing diagnostic --
# registered via gpufhe_benchmark (buildable/runnable directly, never
# auto-picked-up by ctest), per the SAME discipline Phase 19
# established. See research/TRANSFER_PROFILING.md's Phase 20 section
# and research/EXPERIMENT_LEDGER.md entry #53.
gpufhe_test(test_registered_host_buffer tests/test_registered_host_buffer.cpp)
gpufhe_test(test_encrypt_host_shoup_pooled_barrett_batched_registered_matches_reference tests/test_encrypt_host_shoup_pooled_barrett_batched_registered_matches_reference.cpp)
gpufhe_test(test_encrypt_host_shoup_pooled_barrett_batched_registered_concurrency_audit tests/test_encrypt_host_shoup_pooled_barrett_batched_registered_concurrency_audit.cpp)
gpufhe_benchmark(bench_encrypt_registered_lifecycle bench/bench_encrypt_registered_lifecycle.cpp)
# GOD PROMPT "CKKS PHASE 22": controlled re-test of Phase 20's
# explicit-prewarming anomaly, using the IDENTICAL (n,sizeQ) key for
# "lazy" and "prewarmed" conditions (via separate fresh-process
# invocations) to remove the sizeQ-vs-sizeQ+1 confound Phase 20's own
# benchmark had. A pure timing diagnostic -- gpufhe_benchmark, same
# discipline as above. See research/TRANSFER_PROFILING.md's Phase 22
# section and research/EXPERIMENT_LEDGER.md entry #55.
gpufhe_benchmark(bench_registered_prewarm_anomaly bench/bench_registered_prewarm_anomaly.cpp)
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

# GOD PROMPT "CKKS PHASE 32", Work Package B: a real, non-FHE GPU-resident
# streaming simulation (EWMA volatility + vectorized Black-Scholes pricing)
# built on the two new general-purpose primitives this phase proposed and
# implemented (include/device_buffer.h's DeviceBuffer<T>, include/
# cuda_check.h's cuda_check) instead of CKKS-specific types -- proof the
# architecture work generalizes beyond FHE. No OpenFHE dependency needed
# (manual registration, not the gpufhe_test macro, mirroring tests/
# test_ks_graph_setup_concurrency_diagnostic.cpp's own OpenFHE-free pattern).
# Self-checks its own GPU-vs-CPU-reference correctness and exits non-zero on
# mismatch, so add_test doubles as both the demo and a regression test.
add_executable(resident_finance examples/resident_finance/main.cpp)
target_link_libraries(resident_finance PRIVATE gpufhe CUDA::cudart)
add_test(NAME resident_finance COMMAND resident_finance)
