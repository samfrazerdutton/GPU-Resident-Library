// Stage 6c: native eval (relin) key, validated by the eval-key decryption
// identity: bv[p] - av[p]*s == P*s2 in part p's Q-tower range + small noise.
// Generate keypair over QP, s2=s*s, assemble eval key, check the identity poly
// (bv - av*s - P*s2[in range]) is SMALL in coeff form for every part.
#include "openfhe.h"
#include "keyswitch.h"
#include "keygen.h"
#include "intt.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace lbcrypto;

static uint64_t powmod(uint64_t b,uint64_t e,uint64_t q){unsigned __int128 r=1,bb=b%q;while(e){if(e&1)r=(r*bb)%q;bb=(bb*bb)%q;e>>=1;}return(uint64_t)r;}
static uint64_t invmod(uint64_t a,uint64_t q){return powmod(a%q,q-2,q);}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
static uint64_t sub(uint64_t a,uint64_t b,uint64_t q){return a>=b?a-b:a+q-b;}
static uint64_t addmod_local(uint64_t a,uint64_t b,uint64_t q){uint64_t s=a+b;return s>=q?s-q:s;}
static uint64_t* up(const std::vector<uint64_t>&h){uint64_t*d;cudaMalloc(&d,h.size()*8);cudaMemcpy(d,h.data(),h.size()*8,cudaMemcpyHostToDevice);return d;}

int main(){
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(3); params.SetScalingModSize(50);
    params.SetRingDim(32768); params.SetBatchSize(16384);
    params.SetKeySwitchTechnique(HYBRID);
    auto cc=GenCryptoContext(params);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    auto cp=std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto ep=cp->GetElementParams(); auto paramsP=cp->GetParamsP();
    uint32_t n=ep->GetRingDimension(), sizeQ=ep->GetParams().size(), sizeP=paramsP->GetParams().size();
    uint32_t sizeQlP=sizeQ+sizeP, ns=cp->GetNoiseScale();

    // QP moduli + roots
    std::vector<uint64_t> modQP(sizeQlP), rootQP(sizeQlP);
    for(uint32_t i=0;i<sizeQ;++i){ modQP[i]=ep->GetParams()[i]->GetModulus().ConvertToInt(); rootQP[i]=ep->GetParams()[i]->GetRootOfUnity().ConvertToInt(); }
    for(uint32_t j=0;j<sizeP;++j){ modQP[sizeQ+j]=paramsP->GetParams()[j]->GetModulus().ConvertToInt(); rootQP[sizeQ+j]=paramsP->GetParams()[j]->GetRootOfUnity().ConvertToInt(); }

    // keypair over QP
    auto KP=gpufhe::keygen_host(n,modQP,rootQP,ns,3.19,999);

    // P mod each QP modulus (P = product of P primes)
    std::vector<uint64_t> PModq_QP(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){ uint64_t q=modQP[t], P=1%q; for(uint32_t j=0;j<sizeP;++j)P=mm(P,modQP[sizeQ+j]%q,q); PModq_QP[t]=P; }

    // build a KeySwitchConstants shell (just what evalkeygen needs)
    gpufhe::KeySwitchConstants K; K.n=n; K.sizeQl=sizeQ; K.sizeP=sizeP;
    K.alpha=(uint32_t)std::ceil((double)sizeQ/cp->GetNumPartQ()); K.numPart=cp->GetNumPartQ();

    gpufhe::evalkeygen_host(K, KP.s, KP.pkA, KP.pkB, PModq_QP, modQP, rootQP, ns, 3.19, 777);

    // s2 = s*s over QP
    std::vector<uint64_t> s2((size_t)sizeQlP*n);
    for(uint32_t t=0;t<sizeQlP;++t){ uint64_t q=modQP[t]; for(uint32_t k=0;k<n;++k) s2[(size_t)t*n+k]=mm(KP.s[(size_t)t*n+k],KP.s[(size_t)t*n+k],q); }

    auto br=[](uint32_t v,uint32_t b){uint32_t r=0;for(uint32_t k=0;k<b;k++){r=(r<<1)|(v&1);v>>=1;}return r;};
    uint32_t logn=0; while((1u<<logn)<n)++logn;

    double worst=0;
    for(uint32_t part=0;part<K.numPart;++part){
        uint32_t startIdx=K.alpha*part;
        uint32_t sizePart=(sizeQ>startIdx+K.alpha)?K.alpha:(sizeQ-startIdx);
        uint32_t endIdx=startIdx+sizePart;
        for(uint32_t t=0;t<sizeQlP;++t){ uint64_t q=modQP[t], rt=rootQP[t], rti=invmod(rt,q);
            // identity poly (eval): bv - av*s - (P*s2 if in range)
            std::vector<uint64_t> v(n);
            for(uint32_t k=0;k<n;++k){ uint64_t avs=mm(K.av[part][(size_t)t*n+k],KP.s[(size_t)t*n+k],q);
                uint64_t val=addmod_local(K.bv[part][(size_t)t*n+k],avs,q);
                if(t>=startIdx&&t<endIdx){ uint64_t Ps2=mm(PModq_QP[t]%q,s2[(size_t)t*n+k],q); val=sub(val,Ps2,q); }
                v[k]=val; }
            // INTT to coeff, check small
            std::vector<uint64_t> ir(n),ip(n),pw(n);pw[0]=1;
            for(uint32_t i=1;i<n;++i)pw[i]=mm(pw[i-1],rti,q);
            for(uint32_t i=0;i<n;++i){uint64_t r=pw[br(i,logn)];ir[i]=r;ip[i]=(uint64_t)(((__uint128_t)r<<64)/q);}
            uint64_t ninv=invmod(n,q),ninvp=(uint64_t)(((__uint128_t)ninv<<64)/q);
            uint64_t*dx=up(v),*dir=up(ir),*dip=up(ip);
            LaunchINTT_GS(dx,dir,dip,n,q,ninv,ninvp,0);cudaDeviceSynchronize();
            cudaMemcpy(v.data(),dx,(size_t)n*8,cudaMemcpyDeviceToHost);
            cudaFree(dx);cudaFree(dir);cudaFree(dip);
            uint64_t thresh=100000; uint32_t big=0; uint64_t maxmag=0;
            for(uint32_t k=0;k<n;++k){uint64_t c=v[k],mag=(c>q/2)?(q-c):c;if(mag>thresh)++big;if(mag>maxmag)maxmag=mag;}
            if(part==0)std::cout<<"      tower "<<t<<" q="<<q<<" maxmag="<<maxmag<<" (q/maxmag="<<(maxmag?q/maxmag:0)<<")\n";
            double frac=(double)big/n; if(frac>worst)worst=frac;
            if(part==0) std::cout<<"    part0 tower "<<t<<" (inrange="<<(t>=startIdx&&t<endIdx)<<") large-frac="<<(frac*100)<<"%\n";
        }
    }
    std::cout<<"sizeQ="<<sizeQ<<" sizeP="<<sizeP<<" numPart="<<K.numPart<<" worst large-frac="<<(worst*100)<<"%\n";
    if(worst<0.01){ std::cout<<"[PASS] native eval key valid (bv - av*s == P*s2 + small noise)\n"; return 0; }
    std::cout<<"[FAIL] eval-key identity not small\n"; return 1;
}
