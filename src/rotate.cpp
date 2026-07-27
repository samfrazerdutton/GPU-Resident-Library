// Galois automorphism sigma_k on an eval-form RNS poly: x -> x^k.
// Coefficient-domain (correct-first): per tower INTT -> permute coeff j to
// index (j*k mod 2n), negating when it lands in [n,2n) (x^n = -1) -> NTT.
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "ntt.h"
#include "intt.h"
namespace gpufhe {
namespace {
uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
uint64_t powmod(uint64_t b,uint64_t e,uint64_t q){unsigned __int128 r=1,bb=b%q;while(e){if(e&1)r=(r*bb)%q;bb=(bb*bb)%q;e>>=1;}return(uint64_t)r;}
uint64_t invmod(uint64_t a,uint64_t q){return powmod(a%q,q-2,q);}
uint64_t* up(const std::vector<uint64_t>&h){uint64_t*d;cudaMalloc(&d,h.size()*8);cudaMemcpy(d,h.data(),h.size()*8,cudaMemcpyHostToDevice);return d;}
void xf(std::vector<uint64_t>&h,uint32_t n,uint64_t q,uint64_t root,bool inv){
    auto br=[](uint32_t v,uint32_t b){uint32_t r=0;for(uint32_t k=0;k<b;k++){r=(r<<1)|(v&1);v>>=1;}return r;};
    uint32_t logn=0;while((1u<<logn)<n)++logn;
    uint64_t base=inv?invmod(root,q):root;
    std::vector<uint64_t> tr(n),tp(n),pw(n);pw[0]=1;
    for(uint32_t i=1;i<n;++i)pw[i]=mm(pw[i-1],base,q);
    for(uint32_t i=0;i<n;++i){uint64_t r=pw[br(i,logn)];tr[i]=r;tp[i]=(uint64_t)(((__uint128_t)r<<64)/q);}
    uint64_t*dx=up(h),*dr=up(tr),*dp=up(tp);
    if(inv){uint64_t ni=invmod(n,q),nip=(uint64_t)(((__uint128_t)ni<<64)/q);
        LaunchINTT_GS(dx,dr,dp,n,q,ni,nip,0);} else LaunchNTT_CT(dx,dr,dp,n,q,0);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(),dx,(size_t)n*8,cudaMemcpyDeviceToHost);
    cudaFree(dx);cudaFree(dr);cudaFree(dp);
}
} // anon

// In-place automorphism on a towers*n eval-form poly.
void automorphism_eval_host(std::vector<uint64_t>& v, uint32_t towers, uint32_t n,
                            uint32_t k, const std::vector<uint64_t>& mod,
                            const std::vector<uint64_t>& root)
{
    const uint32_t M=2*n;
    for(uint32_t t=0;t<towers;++t){ uint64_t q=mod[t];
        std::vector<uint64_t> c(v.begin()+(size_t)t*n, v.begin()+(size_t)(t+1)*n);
        xf(c,n,q,root[t],true);                        // -> coeff
        std::vector<uint64_t> o(n,0);
        for(uint32_t j=0;j<n;++j){ uint64_t idx=((uint64_t)j*k)%M;
            uint64_t cv=c[j];
            if(idx<n) o[idx]=cv;
            else      o[idx-n]=cv? q-cv : 0;           // x^n = -1
        }
        xf(o,n,q,root[t],false);                       // -> eval
        std::copy(o.begin(),o.end(),v.begin()+(size_t)t*n);
    }
}
} // namespace gpufhe
