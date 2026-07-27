#include <cuda_runtime.h>
#include <cstdint>

// ApproxModDown combine step: per Q tower, ans[k] = (a[k] - b[k]) * s mod q,
// where a = the Q-part input coefficient, b = the P-part switched back to Q,
// s = PInvModq[i] (scalar per tower). Matches OpenFHE's ApproxModDown final
// loop (CKKS path, t=0 so the BGV t-multiplies are skipped). Plain modmul to
// match OpenFHE's DCRTPoly arithmetic bit-for-bit.

__global__ void moddown_combine_kernel(
    uint64_t* __restrict__ ans,        // output, one tower
    const uint64_t* __restrict__ a,    // Q-part input tower (eval)
    const uint64_t* __restrict__ b,    // P-switched-to-Q tower (eval)
    uint64_t s, uint64_t q, uint32_t n)
{
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    uint64_t diff = a[k];
    if (diff < b[k]) diff += q;        // (a - b) mod q
    diff -= b[k];
    ans[k] = (uint64_t)(((unsigned __int128)diff * s) % q);
}

extern "C" void LaunchModDownCombine(
    uint64_t* d_ans, const uint64_t* d_a, const uint64_t* d_b,
    uint64_t s, uint64_t q, uint32_t n, cudaStream_t st)
{
    uint32_t blocks = (n + 255) / 256;
    moddown_combine_kernel<<<blocks, 256, 0, st>>>(d_ans, d_a, d_b, s, q, n);
}
