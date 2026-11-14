// GOD PROMPT "CKKS PHASE 42": isolated, order-alternated, pre-warmed A/B
// measurement of rescale_resident_raw's new batched path (g_rescale_batched)
// vs the preserved reference serial-per-tower path -- same in-process
// toggle convention this project already uses for g_ks_batched (see tests/
// test_rot_resident.cpp), avoiding a git-stash round trip. Sweeps
// MultiplicativeDepth to get a range of `towers` (last) values, since the
// batching win is expected to scale with how many towers the serial loop
// would otherwise iterate over.
#include "openfhe.h"
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
using namespace lbcrypto;
namespace gpufhe { extern bool g_rescale_batched; }
static uint64_t mm_(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}

int main() {
    using namespace gpufhe;
    for (uint32_t depth : {3u, 6u, 10u}) {
        CCParams<CryptoContextCKKSRNS> params;
        params.SetMultiplicativeDepth(depth); params.SetScalingModSize(50);
        params.SetRingDim(32768); params.SetBatchSize(16384);
        params.SetKeySwitchTechnique(HYBRID); params.SetScalingTechnique(FIXEDMANUAL);
        auto cc=GenCryptoContext(params);
        auto cp=std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
        auto ep=cp->GetElementParams(); auto paramsP=cp->GetParamsP();
        const uint32_t n=ep->GetRingDimension(), sizeQ=(uint32_t)ep->GetParams().size(),
                       sizeP=(uint32_t)paramsP->GetParams().size();
        const uint32_t ns=cp->GetNoiseScale(), numPart=cp->GetNumPartQ();
        std::vector<uint64_t> mod(sizeQ),root(sizeQ),modQP(sizeQ+sizeP),rootQP(sizeQ+sizeP);
        for (uint32_t i=0;i<sizeQ;++i){mod[i]=ep->GetParams()[i]->GetModulus().ConvertToInt();root[i]=ep->GetParams()[i]->GetRootOfUnity().ConvertToInt();modQP[i]=mod[i];rootQP[i]=root[i];}
        for (uint32_t j=0;j<sizeP;++j){modQP[sizeQ+j]=paramsP->GetParams()[j]->GetModulus().ConvertToInt();rootQP[sizeQ+j]=paramsP->GetParams()[j]->GetRootOfUnity().ConvertToInt();}
        auto KPqp = keygen_host(n, modQP, rootQP, ns, 3.19, 101);
        KeySwitchConstants K; K.n=n;
        std::vector<uint64_t> mp(sizeP); for (uint32_t j=0;j<sizeP;++j) mp[j]=modQP[sizeQ+j];
        compute_keyswitch_constants(K, mod, mp, numPart);
        for (uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
        for (uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modQP[sizeQ+j]);K.rootValList.push_back(rootQP[sizeQ+j]);}
        std::vector<uint64_t> PModq_QP(sizeQ+sizeP);
        for (uint32_t t=0;t<sizeQ+sizeP;++t){uint64_t q=modQP[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm_(P,modQP[sizeQ+j]%q,q);PModq_QP[t]=P;}
        evalkeygen_host(K, KPqp.s, KPqp.pkA, KPqp.pkB, PModq_QP, modQP, rootQP, ns, 3.19, 202);
        auto C = ks_context_create(K);
        std::vector<uint64_t> s1(sizeQ-1), s2(sizeQ-1);
        { const auto&a=cp->GetqlInvModq(0); const auto&b=cp->GetQlQlInvModqlDivqlModq(0);
          for (uint32_t t=0;t<sizeQ-1;++t){ s1[t]=a[t].ConvertToInt(); s2[t]=b[t].ConvertToInt(); } }

        std::mt19937_64 rng(7);
        size_t T=(size_t)sizeQ*n;
        std::vector<uint64_t> h0(T);
        for (uint32_t t=0;t<sizeQ;++t) for (uint32_t k=0;k<n;++k) h0[(size_t)t*n+k]=rng()%mod[t];
        uint64_t *d0; cudaMalloc(&d0,T*8);
        uint64_t *scr,*drp; cudaMalloc(&scr,(size_t)n*8); cudaMalloc(&drp,(size_t)n*8);

        cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        auto run_once=[&](bool batched){
            g_rescale_batched = batched;
            cudaMemcpy(d0,h0.data(),T*8,cudaMemcpyHostToDevice);
            cudaEventRecord(e0,0);
            rescale_resident_raw(d0, sizeQ, C, s1, s2, scr, drp, 0);
            cudaEventRecord(e1,0); cudaEventSynchronize(e1);
            float t; cudaEventElapsedTime(&t,e0,e1); return (double)t;
        };
        for (int i=0;i<50;++i){ run_once(false); run_once(true); } // warm-up

        const int reps=500;
        std::vector<double> oldT, newT;
        for (int rep=0; rep<reps; ++rep) {
            if (rep%2==0){ oldT.push_back(run_once(false)); newT.push_back(run_once(true)); }
            else { newT.push_back(run_once(true)); oldT.push_back(run_once(false)); }
        }
        std::sort(oldT.begin(),oldT.end()); std::sort(newT.begin(),newT.end());
        double om=oldT[oldT.size()/2], nm=newT[newT.size()/2];
        std::cout << "depth=" << depth << " towers=" << sizeQ << " (last=" << sizeQ-1 << ") n=" << n
                  << "  OLD(serial) median=" << om << "ms  NEW(batched) median=" << nm
                  << "ms  speedup=" << (om/nm) << "x\n";
        g_rescale_batched = true; // restore default
    }
    return 0;
}
