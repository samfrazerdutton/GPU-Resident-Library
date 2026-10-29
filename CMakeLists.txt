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

# Core library: pure device-resident FHE, NO OpenFHE dependency.
add_library(gpufhe STATIC
  src/device_ciphertext.cpp
  src/ckks_ops.cpp
  src/keyswitch.cpp
  src/ks_constants.cpp
  src/keygen.cpp
  src/evalkeygen.cpp
  src/encdec.cpp
  src/encode.cpp
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
target_include_directories(gpufhe PUBLIC include)
target_link_libraries(gpufhe PUBLIC CUDA::cudart)
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
gpufhe_test(test_ks_constants tests/test_ks_constants.cpp)
gpufhe_test(test_keygen tests/test_keygen.cpp)
gpufhe_test(test_evalkeygen tests/test_evalkeygen.cpp)
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
# Directive "GOD PROMPT -- CKKS ENGINE": Priority 2 (RAII refactor), Phase C.
# Validates include/ks_raii.h's ScopedKSContext/ScopedKSWork wrappers in
# ISOLATION (construction/destruction, repeated cycles with stable GPU
# memory, move semantics, double-reset safety, C-API interop) before any
# production call site is touched. Uses n=8192 (proven scale), deliberately
# avoiding the unrelated small-n rescale bug found this same round.
gpufhe_test(test_ks_raii tests/test_ks_raii.cpp)
