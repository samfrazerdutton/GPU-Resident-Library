// Eval-domain Galois automorphism: a PURE PERMUTATION of the n evaluation
// values, no sign flip, no modular arithmetic. Replaces the existing
// INTT -> coefficient-domain-permute -> NTT round trip (automorphism_eval_device
// in rotate.cpp) with a single gather kernel, for the SAME mathematical
// operation on the SAME eval-domain (bit-reversed-output) representation this
// codebase's NTT already uses.
//
// WHY this is valid (independently verified via a from-scratch, GPU-free
// Python reference against 20 randomized trials before writing this file --
// see research/EXPERIMENT_LEDGER.md): for a negacyclic NTT with standard-
// order coefficients in / bit-reversed evaluation out (this codebase's exact
// convention, matching OpenFHE bit-for-bit per kernels/ntt_ct.cu's own
// comment), applying sigma_k: X -> X^k to a polynomial f gives
// g(zeta^(2j+1)) = f(zeta^((2j+1)*k mod 2n)) -- and since k and (2j+1) are
// both odd, (2j+1)*k mod 2n is ALWAYS odd too, i.e. always some OTHER valid
// evaluation point, never requiring the X^n=-1 sign-flip that the
// COEFFICIENT-domain version needs when an exponent wraps past n. So in eval
// domain, automorphism is just "read from a different (precomputable) index."
//
// Permutation formula (array position p -> source array position):
//   j      = bitrev(p, logn)              // array index -> logical (natural) index
//   e      = (2*j+1)*k mod 2n              // apply the automorphism exponent
//   j'     = (e-1)/2                       // logical index of the new odd power
//   source = bitrev(j', logn)              // logical index -> array index
//   out[p] = in[source]

#include <cuda_runtime.h>
#include <cstdint>

__global__ void auto_eval_permute_kernel(
    uint64_t* __restrict__ out, const uint64_t* __restrict__ in,
    const uint32_t* __restrict__ map, uint32_t n)
{
    uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    out[p] = in[map[p]];
}

// Tower-batched: grid.y = tower. Same map reused across every tower (the
// permutation only depends on (n, k), not on which modulus q a tower uses --
// it's a pure index operation, no arithmetic mod q at all).
__global__ void auto_eval_permute_kernel_b(
    uint64_t* __restrict__ out, const uint64_t* __restrict__ in,
    const uint32_t* __restrict__ map, uint32_t n)
{
    uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    uint32_t tw = blockIdx.y;
    out[(size_t)tw*n + p] = in[(size_t)tw*n + map[p]];
}

extern "C" void LaunchAutoEvalPermute(uint64_t* d_out, const uint64_t* d_in,
                                      const uint32_t* d_map, uint32_t n,
                                      cudaStream_t s)
{
    uint32_t blocks = (n + 255) / 256;
    auto_eval_permute_kernel<<<blocks, 256, 0, s>>>(d_out, d_in, d_map, n);
}

extern "C" void LaunchAutoEvalPermuteBatched(uint64_t* d_out, const uint64_t* d_in,
                                             const uint32_t* d_map,
                                             uint32_t towers, uint32_t n,
                                             cudaStream_t s)
{
    uint32_t blocksX = (n + 255) / 256;
    dim3 g(blocksX, towers);
    auto_eval_permute_kernel_b<<<g, 256, 0, s>>>(d_out, d_in, d_map, n);
}
