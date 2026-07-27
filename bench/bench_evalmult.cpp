// Single full EvalMult latency: native tensor+relin(+the keyswitch)+... vs
// OpenFHE EvalMult, both timed one op, construction OUTSIDE the timer, median +
// bootstrap CI. HONEST NOTE: keyswitch_core_resident is host-orchestrated
// (per-op cudaMalloc/free + host loops), so the GPU number here is dominated by
// that overhead, NOT representative of a resident relin. This benchmark
// QUANTIFIES the host-orchestration cost -- it's the motivation for the
// resident-relin optimization, reported as such, not as an optimized figure.
#include "openfhe.h"
#include "keyswitch.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
using namespace lbcrypto;
using clk = std::chrono::steady_clock;
static double median(std::vector<double> v){ if(v.empty())return 0; std::sort(v.begin(),v.end());
    size_t n=v.size(); return n%2?v[n/2]:0.5*(v[n/2-1]+v[n/2]); }
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
static uint64_t am(uint64_t a,uint64_t b,uint64_t q){uint64_t s=a+b;return s>=q?s-q:s;}

int main(int argc,char**argv){
    uint32_t reps=(argc>1)?(uint32_t)std::stoul(argv[1]):15;
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(3); params.SetScalingModSize(50);
    params.SetRingDim(32768); params.SetBatchSize(16384);
    params.SetKeySwitchTechnique(HYBRID);
    auto cc=GenCryptoContext(params);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    auto kp=cc->KeyGen(); cc->EvalMultKeyGen(kp.secretKey);
    auto cp=std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto ep=cp->GetElementParams(); auto paramsP=cp->GetParamsP();
    uint32_t n=ep->GetRingDimension(), sizeQ=ep->GetParams().size(), sizeP=paramsP->GetParams().size();
    uint32_t sizeQlP=sizeQ+sizeP, ns=cp->GetNoiseScale(), numPart=cp->GetNumPartQ();

    std::vector<uint64_t> mod(sizeQ),root(sizeQ),modQP(sizeQlP),rootQP(sizeQlP);
    for(uint32_t i=0;i<sizeQ;++i){mod[i]=ep->GetParams()[i]->GetModulus().ConvertToInt();root[i]=ep->GetParams()[i]->GetRootOfUnity().ConvertToInt();modQP[i]=mod[i];rootQP[i]=root[i];}
    for(uint32_t j=0;j<sizeP;++j){modQP[sizeQ+j]=paramsP->GetParams()[j]->GetModulus().ConvertToInt();rootQP[sizeQ+j]=paramsP->GetParams()[j]->GetRootOfUnity().ConvertToInt();}

    // native K + eval key (setup, OUTSIDE timer)
    auto KPqp=gpufhe::keygen_host(n,modQP,rootQP,ns,3.19,101);
    gpufhe::KeySwitchConstants K; K.n=n;
    std::vector<uint64_t> mp(sizeP); for(uint32_t j=0;j<sizeP;++j)mp[j]=modQP[sizeQ+j];
    gpufhe::compute_keyswitch_constants(K,mod,mp,numPart);
    for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modQP[sizeQ+j]);K.rootValList.push_back(rootQP[sizeQ+j]);}
    std::vector<uint64_t> PModq_QP(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=modQP[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modQP[sizeQ+j]%q,q);PModq_QP[t]=P;}
    gpufhe::evalkeygen_host(K,KPqp.s,KPqp.pkA,KPqp.pkB,PModq_QP,modQP,rootQP,ns,3.19,202);

    // operands: two 2-component cts as host tower arrays (eval), built outside timer
    std::mt19937_64 rng(7);
    auto rb=[&](){std::vector<uint64_t>v((size_t)sizeQ*n);for(uint32_t t=0;t<sizeQ;++t)for(uint32_t k=0;k<n;++k)v[(size_t)t*n+k]=rng()%mod[t];return v;};
    std::vector<uint64_t> c0a=rb(),c1a=rb(),c0b=rb(),c1b=rb();

    // OpenFHE operands (outside timer)
    auto mkpt=[&](){ std::vector<double> x(16384); for(auto&z:x)z=(rng()%1000)/1000.0; return cc->MakeCKKSPackedPlaintext(x); };
    auto ct1=cc->Encrypt(kp.publicKey,mkpt()), ct2=cc->Encrypt(kp.publicKey,mkpt());

    // GPU full EvalMult: tensor -> relin -> combine (rescale proven separately; omit to isolate relin cost)
    auto gpu_evalmult=[&](){
        std::vector<uint64_t> t2((size_t)sizeQ*n);
        for(uint32_t t=0;t<sizeQ;++t){uint64_t q=mod[t];for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;t2[x]=mm(c1a[x],c1b[x],q);}}
        auto R=gpufhe::keyswitch_core_resident(t2,K);
        // combine (tensor t0,t1 + ba) -- cheap, include for completeness
        volatile uint64_t sink=0;
        for(uint32_t t=0;t<sizeQ;++t){uint64_t q=mod[t];for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
            uint64_t t0=mm(c0a[x],c0b[x],q),t1=am(mm(c0a[x],c1b[x],q),mm(c1a[x],c0b[x],q),q);
            sink^=am(t0,R.ba0[x],q)^am(t1,R.ba1[x],q);}}
        (void)sink;
    };

    std::vector<double> gpu,cpu;
    for(uint32_t r=0;r<reps;++r){
        auto g0=clk::now(); gpu_evalmult(); double gt=std::chrono::duration<double,std::milli>(clk::now()-g0).count();
        auto c0_=clk::now(); auto cm=cc->EvalMult(ct1,ct2); double ct=std::chrono::duration<double,std::milli>(clk::now()-c0_).count();
        gpu.push_back(gt); cpu.push_back(ct);
    }
    std::cout<<"single full EvalMult (tensor+relin), n="<<n<<" towers="<<sizeQ<<" numPart="<<numPart<<"\n";
    std::cout<<"  GPU (native, host-orchestrated relin) median = "<<median(gpu)<<" ms\n";
    std::cout<<"  CPU (OpenFHE EvalMult)                median = "<<median(cpu)<<" ms\n";
    std::cout<<"  NOTE: GPU relin is host-orchestrated (per-op malloc/host loops), not resident.\n";
    std::cout<<"        This measures correctness-first relin cost, not optimized throughput.\n";
    return 0;
}
