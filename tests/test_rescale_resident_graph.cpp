// GOD PROMPT "CKKS PHASE 42": gate test for rescale_resident_raw's new
// batched path (kernels/switch_modulus.cu's LaunchSwitchModulusBatched,
// the already-proven LaunchNTT_CT_Batched, kernels/rescale.cu's
// LaunchRescaleFuseBatched) under CUDA graph capture -- the exact risk
// class this project's own history (research/KS_GRAPH_CAPTURE_INVESTIGATION.md,
// and DeviceKSWork's own long comment in include/keyswitch_resident.h) warns
// about: a captured cudaMemcpyAsync node stores the HOST POINTER, re-read on
// every future replay, so getRescaleBatchWork's persistent host-staging
// vectors must hold the correct value at capture time for every later replay.
//
// Two scenarios, matching this project's actual usage
// (bench/bench_evalmult_resident.cpp is the only real caller that captures
// rescale_resident_raw):
//   1. ONE call captured and replayed vs the same call made directly --
//      must be bit-exact (mirrors tests/test_ks_graph.cpp's own pattern for
//      keyswitch_resident).
//   2. TWO calls (same towers, same s1/s2 -- the c0/c1 pattern every real
//      EvalMult rescale actually uses, since CKKS's rescale constants
//      depend only on the level, never on which ciphertext component)
//      inside ONE capture, replayed, vs the same two calls made directly --
//      the specific case getRescaleBatchWork's own comment documents as
//      safe FOR THIS REASON, verified here rather than just argued.
#include "openfhe.h"
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
using namespace lbcrypto;
static uint64_t mm_(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}

int main() {
    using namespace gpufhe;
    // Same OpenFHE-derived parameter set as tests/test_ks_graph.cpp (already
    // proven to satisfy P>=Q_part_max) -- this test's own purpose is the
    // CUDA-graph-capture gate, not an independent parameter search.
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(3); params.SetScalingModSize(50);
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

    std::mt19937_64 rng(2026);
    size_t T=(size_t)sizeQ*n;
    std::vector<uint64_t> h0(T), h1(T);
    for (uint32_t t=0;t<sizeQ;++t) for (uint32_t k=0;k<n;++k) {
        h0[(size_t)t*n+k]=rng()%mod[t]; h1[(size_t)t*n+k]=rng()%mod[t];
    }

    uint64_t *scr, *drp; cudaMalloc(&scr,(size_t)n*8); cudaMalloc(&drp,(size_t)n*8);
    bool allPass = true;

    // --- Scenario 1: one call, captured vs direct ---
    {
        uint64_t *d0; cudaMalloc(&d0,T*8); cudaMemcpy(d0,h0.data(),T*8,cudaMemcpyHostToDevice);
        rescale_resident_raw(d0, sizeQ, C, s1, s2, scr, drp, 0);
        cudaDeviceSynchronize();
        size_t T2=(size_t)(sizeQ-1)*n;
        std::vector<uint64_t> ref(T2); cudaMemcpy(ref.data(),d0,T2*8,cudaMemcpyDeviceToHost);

        uint64_t *d0b; cudaMalloc(&d0b,T*8); cudaMemcpy(d0b,h0.data(),T*8,cudaMemcpyHostToDevice);
        cudaStream_t s; cudaStreamCreate(&s);
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal);
        rescale_resident_raw(d0b, sizeQ, C, s1, s2, scr, drp, s);
        cudaStreamEndCapture(s,&g);
        cudaError_t ie = cudaGraphInstantiate(&ge,g,nullptr,nullptr,0);
        if (ie!=cudaSuccess){ std::cout<<"[FAIL] scenario1 graph instantiate: "<<cudaGetErrorString(ie)<<"\n"; allPass=false; }
        else {
            cudaGraphLaunch(ge,s); cudaStreamSynchronize(s);
            std::vector<uint64_t> got(T2); cudaMemcpy(got.data(),d0b,T2*8,cudaMemcpyDeviceToHost);
            uint32_t bad=0; for (size_t x=0;x<T2;++x) if (got[x]!=ref[x]) ++bad;
            std::cout<<"scenario1 (1 call, captured vs direct): mismatches="<<bad<<"/"<<T2<<"\n";
            if (bad) allPass=false;
            cudaGraphExecDestroy(ge);
        }
        cudaGraphDestroy(g); cudaStreamDestroy(s);
        cudaFree(d0); cudaFree(d0b);
    }

    // --- Scenario 2: two calls (same towers, same s1/s2 -- the real c0/c1
    // EvalMult pattern), captured together vs direct ---
    {
        uint64_t *d0,*d1; cudaMalloc(&d0,T*8); cudaMalloc(&d1,T*8);
        cudaMemcpy(d0,h0.data(),T*8,cudaMemcpyHostToDevice);
        cudaMemcpy(d1,h1.data(),T*8,cudaMemcpyHostToDevice);
        rescale_resident_raw(d0, sizeQ, C, s1, s2, scr, drp, 0);
        rescale_resident_raw(d1, sizeQ, C, s1, s2, scr, drp, 0);
        cudaDeviceSynchronize();
        size_t T2=(size_t)(sizeQ-1)*n;
        std::vector<uint64_t> ref0(T2), ref1(T2);
        cudaMemcpy(ref0.data(),d0,T2*8,cudaMemcpyDeviceToHost);
        cudaMemcpy(ref1.data(),d1,T2*8,cudaMemcpyDeviceToHost);

        uint64_t *d0b,*d1b; cudaMalloc(&d0b,T*8); cudaMalloc(&d1b,T*8);
        cudaMemcpy(d0b,h0.data(),T*8,cudaMemcpyHostToDevice);
        cudaMemcpy(d1b,h1.data(),T*8,cudaMemcpyHostToDevice);
        cudaStream_t s; cudaStreamCreate(&s);
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal);
        rescale_resident_raw(d0b, sizeQ, C, s1, s2, scr, drp, s);
        rescale_resident_raw(d1b, sizeQ, C, s1, s2, scr, drp, s);
        cudaStreamEndCapture(s,&g);
        cudaError_t ie = cudaGraphInstantiate(&ge,g,nullptr,nullptr,0);
        if (ie!=cudaSuccess){ std::cout<<"[FAIL] scenario2 graph instantiate: "<<cudaGetErrorString(ie)<<"\n"; allPass=false; }
        else {
            cudaGraphLaunch(ge,s); cudaStreamSynchronize(s);
            std::vector<uint64_t> got0(T2), got1(T2);
            cudaMemcpy(got0.data(),d0b,T2*8,cudaMemcpyDeviceToHost);
            cudaMemcpy(got1.data(),d1b,T2*8,cudaMemcpyDeviceToHost);
            uint32_t bad0=0,bad1=0;
            for (size_t x=0;x<T2;++x){ if (got0[x]!=ref0[x]) ++bad0; if (got1[x]!=ref1[x]) ++bad1; }
            std::cout<<"scenario2 (2 calls same towers/s1/s2, captured vs direct): "
                     <<"mismatches d0="<<bad0<<"/"<<T2<<" d1="<<bad1<<"/"<<T2<<"\n";
            if (bad0||bad1) allPass=false;
            cudaGraphExecDestroy(ge);
        }
        cudaGraphDestroy(g); cudaStreamDestroy(s);
        cudaFree(d0); cudaFree(d1); cudaFree(d0b); cudaFree(d1b);
    }

    std::cout << (allPass ? "[OVERALL PASS]\n" : "[OVERALL FAIL]\n");
    return allPass ? 0 : 1;
}
