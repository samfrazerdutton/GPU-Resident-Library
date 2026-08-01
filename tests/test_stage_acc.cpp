// Gate: LaunchPtMulAcc must be BIT-EXACT vs the host ct_mul_pt + ct_add_ct pair
// it replaces. This is the inner op of a staged transform (acc += rot_ct * diag),
// so any drift here corrupts every diagonal.
#include "keyswitch.h"
#include "stage_acc.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <random>
namespace gpufhe {
void ct_mul_pt_host(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<uint64_t>&,
                    uint32_t, uint32_t, const std::vector<uint64_t>&);
void ct_add_ct_host(std::vector<uint64_t>&, std::vector<uint64_t>&,
                    const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                    uint32_t, uint32_t, const std::vector<uint64_t>&);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
}
int main(){
    const uint32_t n=8192, tw=30;
    std::vector<uint64_t> mod;
    gpufhe::native_primes(mod,1,60,n,{});
    { std::vector<uint64_t> md; gpufhe::native_primes(md,tw-1,55,n,mod); for(auto m:md)mod.push_back(m); }
    const size_t T=(size_t)tw*n;
    std::mt19937_64 rng(12345);
    std::vector<uint64_t> a0(T),a1(T),b0(T),b1(T),d(T);
    for(uint32_t t=0;t<tw;++t){ uint64_t q=mod[t];
        for(uint32_t k=0;k<n;++k){ size_t x=(size_t)t*n+k;
            a0[x]=rng()%q; a1[x]=rng()%q; b0[x]=rng()%q; b1[x]=rng()%q; d[x]=rng()%q; } }

    // host reference: tmp = b * d ; acc += tmp
    std::vector<uint64_t> h0=a0,h1=a1,t0=b0,t1=b1;
    gpufhe::ct_mul_pt_host(t0,t1,d,tw,n,mod);
    gpufhe::ct_add_ct_host(h0,h1,t0,t1,tw,n,mod);

    // device: fused, one launch per tower
    uint64_t *dA0,*dA1,*dB0,*dB1,*dD;
    cudaMalloc(&dA0,T*8);cudaMalloc(&dA1,T*8);cudaMalloc(&dB0,T*8);cudaMalloc(&dB1,T*8);cudaMalloc(&dD,T*8);
    cudaMemcpy(dA0,a0.data(),T*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dA1,a1.data(),T*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dB0,b0.data(),T*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dB1,b1.data(),T*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dD ,d.data() ,T*8,cudaMemcpyHostToDevice);
    for(uint32_t t=0;t<tw;++t)
        LaunchPtMulAcc(dA0+(size_t)t*n, dA1+(size_t)t*n,
                       dB0+(size_t)t*n, dB1+(size_t)t*n,
                       dD +(size_t)t*n, mod[t], n, 0);
    cudaDeviceSynchronize();
    std::vector<uint64_t> g0(T),g1(T);
    cudaMemcpy(g0.data(),dA0,T*8,cudaMemcpyDeviceToHost);
    cudaMemcpy(g1.data(),dA1,T*8,cudaMemcpyDeviceToHost);

    size_t bad=0;
    for(size_t i=0;i<T;++i){ if(g0[i]!=h0[i])++bad; if(g1[i]!=h1[i])++bad; }
    std::cout<<"n="<<n<<" tw="<<tw<<"  mismatches = "<<bad<<" / "<<2*T<<"\n";
    if(bad==0){std::cout<<"[PASS] fused pt-mul-accumulate is bit-exact\n";return 0;}
    std::cout<<"[FAIL]\n"; return 1;
}
