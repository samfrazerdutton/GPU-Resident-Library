// Stage 6b: native keygen validity. Generate (s, pk=(b,a)); verify b + a*s is a
// SMALL poly (the RLWE key-validity criterion) -- i.e. b + a*s = ns*e, so its
// coefficients are ~ns*sigma, NOT uniform mod q. Compute b+a*s eval, INTT to
// coeff, check |centered coeff| is small for the vast majority.
#include "openfhe.h"
#include "keygen.h"
#include "intt.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace lbcrypto;

static uint64_t powmod(uint64_t b,uint64_t e,uint64_t q){unsigned __int128 r=1,bb=b%q;while(e){if(e&1)r=(r*bb)%q;bb=(bb*bb)%q;e>>=1;}return(uint64_t)r;}
static uint64_t invmod(uint64_t a,uint64_t q){return powmod(a%q,q-2,q);}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
static uint64_t* up(const std::vector<uint64_t>&h){uint64_t*d;cudaMalloc(&d,h.size()*8);cudaMemcpy(d,h.data(),h.size()*8,cudaMemcpyHostToDevice);return d;}

int main(){
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(3); params.SetScalingModSize(50);
    params.SetRingDim(32768); params.SetBatchSize(16384);
    params.SetKeySwitchTechnique(HYBRID);
    auto cc=GenCryptoContext(params);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    auto cp=std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto ep=cp->GetElementParams();
    uint32_t n=ep->GetRingDimension(), sizeQ=ep->GetParams().size();
    uint64_t ns=cp->GetNoiseScale();

    std::vector<uint64_t> mod(sizeQ), root(sizeQ);
    for(uint32_t i=0;i<sizeQ;++i){ mod[i]=ep->GetParams()[i]->GetModulus().ConvertToInt();
        root[i]=ep->GetParams()[i]->GetRootOfUnity().ConvertToInt(); }

    auto K=gpufhe::keygen_host(n,mod,root,ns,3.19,12345);

    // check b + a*s small, per tower, in coeff form
    // build inv tables (reuse mk_tab logic inline via powmod)
    auto br=[](uint32_t v,uint32_t b){uint32_t r=0;for(uint32_t k=0;k<b;k++){r=(r<<1)|(v&1);v>>=1;}return r;};
    uint32_t logn=0; while((1u<<logn)<n)++logn;

    uint32_t worst_big=0; double worst_frac=0;
    for(uint32_t t=0;t<sizeQ;++t){
        uint64_t q=mod[t], rt=root[t], rti=invmod(rt,q);
        // inv tables
        std::vector<uint64_t> ir(n),ip(n); std::vector<uint64_t> pw(n); pw[0]=1;
        for(uint32_t i=1;i<n;++i)pw[i]=mm(pw[i-1],rti,q);
        for(uint32_t i=0;i<n;++i){uint64_t r=pw[br(i,logn)];ir[i]=r;ip[i]=(uint64_t)(((__uint128_t)r<<64)/q);}
        uint64_t ninv=invmod(n,q), ninvp=(uint64_t)(((__uint128_t)ninv<<64)/q);
        // v = b + a*s eval
        std::vector<uint64_t> v(n);
        for(uint32_t k=0;k<n;++k){ uint64_t as=mm(K.pkA[(size_t)t*n+k],K.s[(size_t)t*n+k],q);
            uint64_t b=K.pkB[(size_t)t*n+k]; uint64_t s=b+as; if(s>=q)s-=q; v[k]=s; }
        uint64_t*dx=up(v),*dir=up(ir),*dip=up(ip);
        LaunchINTT_GS(dx,dir,dip,n,q,ninv,ninvp,0); cudaDeviceSynchronize();
        cudaMemcpy(v.data(),dx,(size_t)n*8,cudaMemcpyDeviceToHost);
        cudaFree(dx);cudaFree(dir);cudaFree(dip);
        // count coefficients that are "large" (centered magnitude > q/1000)
        uint64_t thresh=q/1000; uint32_t big=0;
        for(uint32_t k=0;k<n;++k){ uint64_t c=v[k]; uint64_t mag=(c>q/2)?(q-c):c; if(mag>thresh)++big; }
        double frac=(double)big/n;
        if(frac>worst_frac){worst_frac=frac;worst_big=big;}
    }
    std::cout<<"sizeQ="<<sizeQ<<" ns="<<ns<<" worst tower: "<<worst_big<<" large coeffs ("
             <<(worst_frac*100)<<"%)\n";
    // A valid key: b+a*s = ns*e is small, so essentially ALL coeffs small.
    if(worst_frac < 0.01){ std::cout<<"[PASS] native keygen produces valid RLWE key (b+a*s is small)\n"; return 0; }
    std::cout<<"[FAIL] b+a*s is not small -- key invalid\n"; return 1;
}
