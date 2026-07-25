#include <cuda_runtime.h>
#include <cstdint>

// Base conversion matching OpenFHE's NativeVectorT::SwitchModulus exactly.
// For each coefficient v under old modulus q_old, switching to q_new:
//   halfQ = q_old >> 1;  diff = |q_old - q_new|
//   if q_new > q_old:  v += (v > halfQ) ? diff : 0        (plain add, no reduce)
//   else:              v -= (v > halfQ) ? diff : 0  (mod q_new)
// Threshold is strict >. This preserves the signed representative across the
// modulus change; a >= or a swapped branch drifts on the upper-half coeffs.
__global__ void switch_modulus_kernel(
    uint64_t* __restrict__ v,
    uint64_t q_old, uint64_t q_new, uint64_t half_q, uint64_t diff,
    int newLarger, uint32_t n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint64_t x = v[idx];
    uint64_t d = (x > half_q) ? diff : 0;
    if (newLarger) {
        v[idx] = x + d;                 // sum < q_new by construction
    } else {
        // ModSubEq(d, q_new): x is < q_old; bring into [0,q_new) then subtract.
        uint64_t xm = (x >= q_new) ? (x % q_new) : x;
        v[idx] = (xm >= d) ? (xm - d) : (xm + q_new - d);
    }
}

// Switch one tower (length n) in place from q_old to q_new.
extern "C" void LaunchSwitchModulus(
    uint64_t* d_v, uint64_t q_old, uint64_t q_new, uint32_t n, cudaStream_t s)
{
    uint64_t half_q = q_old >> 1;
    uint64_t diff   = (q_old > q_new) ? (q_old - q_new) : (q_new - q_old);
    int newLarger   = (q_new > q_old) ? 1 : 0;
    // Smaller-modulus branch subtracts diff mod q_new (OpenFHE's ModSubEq
    // reduces both operands); diff can exceed q_new when q_old >> q_new.
    if (!newLarger) diff %= q_new;
    uint32_t blocks = (n + 255) / 256;
    switch_modulus_kernel<<<blocks, 256, 0, s>>>(
        d_v, q_old, q_new, half_q, diff, newLarger, n);
}
