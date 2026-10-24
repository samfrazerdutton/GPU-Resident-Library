// Gate: LaunchINTT_GS_Batched (grid.y = tower) must be BIT-IDENTICAL to looping
// LaunchINTT_GS per tower. The batched shared head / stage kernels read
// invroots[gi]/invprecon[gi] where gi depends only on blockIdx.x and the stage,
// so each tower indexes its OWN table at the same position -- a mispaired
// pointer array yields plausible garbage, not an obvious failure. Hence a
// dedicated bit-exact gate, plus an in-run A/B timing. Mirrors
// test_ntt_batched.cpp 1:1 for the inverse transform.
#include "intt.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
}
static uint64_t mulmod(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
static uint64_t powmod(uint64_t b,uint64_t e,uint64_t q){unsigned __int128 r=1,bb=b%q;
    while(e){if(e&1)r=(r*bb)%q;bb=(bb*bb)%q;e>>=1;}return(uint64_t)r;}

int main(int argc,char**argv){
    const uint32_t n=(argc>1)?std::stoul(argv[1]):8192;
    const uint32_t tw=(argc>2)?std::stoul(argv[2]):30;
    std::vector<uint64_t> mod;
    gpufhe::native_primes(mod,1,60,n,{});
    { std::vector<uint64_t> md; gpufhe::native_primes(md,tw-1,55,n,mod); for(auto m:md)mod.push_back(m); }
    uint32_t logn=0; while((1u<<logn)<n)++logn;
    auto br=[&](uint32_t v){uint32_t r=0;for(uint32_t k=0;k<logn;k++){r=(r<<1)|(v&1);v>>=1;}return r;};

    // per-tower inverse-root tables on device + pointer arrays, plus n^{-1}/precon
    std::vector<uint64_t*> hr(tw),hp(tw);
    std::vector<uint64_t> n_inv(tw),n_inv_precon(tw);
    for(uint32_t t=0;t<tw;++t){
        uint64_t q=mod[t], rt=gpufhe::native_root(n,q);
        uint64_t rtInv=powmod(rt,q-2,q);
        std::vector<uint64_t> fr(n),fp(n),pw(n);
        pw[0]=1; for(uint32_t i=1;i<n;++i)pw[i]=mulmod(pw[i-1],rtInv,q);
        for(uint32_t i=0;i<n;++i){uint64_t v=pw[br(i)];fr[i]=v;fp[i]=(uint64_t)(((__uint128_t)v<<64)/q);}
        cudaMalloc(&hr[t],n*8); cudaMemcpy(hr[t],fr.data(),n*8,cudaMemcpyHostToDevice);
        cudaMalloc(&hp[t],n*8); cudaMemcpy(hp[t],fp.data(),n*8,cudaMemcpyHostToDevice);
        uint64_t ninv=powmod(n,q-2,q);
        n_inv[t]=ninv;
        n_inv_precon[t]=(uint64_t)(((__uint128_t)ninv<<64)/q);
    }
    uint64_t **dR,**dP; uint64_t *dM,*dNI,*dNIP;
    cudaMalloc(&dR,tw*sizeof(uint64_t*)); cudaMemcpy(dR,hr.data(),tw*sizeof(uint64_t*),cudaMemcpyHostToDevice);
    cudaMalloc(&dP,tw*sizeof(uint64_t*)); cudaMemcpy(dP,hp.data(),tw*sizeof(uint64_t*),cudaMemcpyHostToDevice);
    cudaMalloc(&dM,tw*8); cudaMemcpy(dM,mod.data(),tw*8,cudaMemcpyHostToDevice);
    cudaMalloc(&dNI,tw*8); cudaMemcpy(dNI,n_inv.data(),tw*8,cudaMemcpyHostToDevice);
    cudaMalloc(&dNIP,tw*8); cudaMemcpy(dNIP,n_inv_precon.data(),tw*8,cudaMemcpyHostToDevice);

    const size_t T=(size_t)tw*n;
    std::mt19937_64 rng(999);
    std::vector<uint64_t> in(T);
    for(uint32_t t=0;t<tw;++t) for(uint32_t k=0;k<n;++k) in[(size_t)t*n+k]=rng()%mod[t];
    uint64_t *dA,*dB;
    cudaMalloc(&dA,T*8); cudaMalloc(&dB,T*8);

    auto runPer=[&](uint64_t* d){ cudaMemcpy(d,in.data(),T*8,cudaMemcpyHostToDevice);
        for(uint32_t t=0;t<tw;++t) LaunchINTT_GS(d+(size_t)t*n,hr[t],hp[t],n,mod[t],n_inv[t],n_inv_precon[t],0); };
    auto runBat=[&](uint64_t* d){ cudaMemcpy(d,in.data(),T*8,cudaMemcpyHostToDevice);
        LaunchINTT_GS_Batched(d,(const uint64_t* const*)dR,(const uint64_t* const*)dP,dM,dNI,dNIP,tw,n,0); };

    runPer(dA); runBat(dB); cudaDeviceSynchronize();
    std::vector<uint64_t> a(T),b(T);
    cudaMemcpy(a.data(),dA,T*8,cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(),dB,T*8,cudaMemcpyDeviceToHost);
    size_t bad=0; for(size_t i=0;i<T;++i) if(a[i]!=b[i])++bad;

    auto bench=[&](auto f,uint64_t* d){ f(d); cudaDeviceSynchronize();
        std::vector<double> v;
        for(int r=0;r<15;++r){ auto p=std::chrono::steady_clock::now();
            f(d); cudaDeviceSynchronize();
            v.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-p).count()); }
        std::sort(v.begin(),v.end()); return v[7]; };
    double tp=bench(runPer,dA), tb=bench(runBat,dB);

    std::cout<<"n="<<n<<" towers="<<tw<<"  mismatches="<<bad<<" / "<<T<<"\n";
    std::cout<<"  per-tower med "<<tp<<" s   batched med "<<tb<<" s   ("<<tp/tb<<"x)\n";
    if(bad==0){std::cout<<"[PASS] batched INTT bit-exact\n";return 0;}
    std::cout<<"[FAIL]\n"; return 1;
}
