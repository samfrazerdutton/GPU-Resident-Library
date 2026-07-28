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
  kernels/ntt_ct.cu
  kernels/rns_arith.cu
  kernels/intt_ct.cu
  kernels/switch_modulus.cu
  kernels/rescale.cu
  kernels/basis_convert.cu
  kernels/ks_fastcore.cu
  kernels/moddown.cu
  kernels/tensor.cu
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
gpufhe_test(bench_resident_chain bench/bench_resident_chain.cpp)
gpufhe_test(bench_batch bench/bench_batch.cpp)
gpufhe_test(bench_evalmult bench/bench_evalmult.cpp)
gpufhe_test(bench_evalmult_resident bench/bench_evalmult_resident.cpp)
gpufhe_test(test_mult_barrett tests/test_mult_barrett.cpp)
