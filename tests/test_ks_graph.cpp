// Gate: keyswitch_resident captured into a CUDA graph and replayed must match
// direct launch bit-exact (same input, same context/work buffers).
#include "openfhe.h"
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
using namespace lbcrypto;
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}

int main(){
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(3); params.SetScalingModSize(50);
    params.SetRingDim(32768); params.SetBatchSize(16384);
    params.SetKeySwitchTechnique(HYBRID); params.SetScalingTechnique(FIXEDMANUAL);
    auto cc=GenCryptoContext(params);
    auto cp=std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto ep=cp->GetElementParams(); auto paramsP=cp->GetParamsP();
    uint32_t n=ep->GetRingDimension(), sizeQ=ep->GetParams().size(), sizeP=paramsP->GetParams().size();
    uint32_t sizeQlP=sizeQ+sizeP, ns=cp->GetNoiseScale(), numPart=cp->GetNumPartQ();
    std::vector<uint64_t> mod(sizeQ),root(sizeQ),modQP(sizeQlP),rootQP(sizeQlP);
    for(uint32_t i=0;i<sizeQ;++i){mod[i]=ep->GetParams()[i]->GetModulus().ConvertToInt();root[i]=ep->GetParams()[i]->GetRootOfUnity().ConvertToInt();modQP[i]=mod[i];rootQP[i]=root[i];}
    for(uint32_t j=0;j<sizeP;++j){modQP[sizeQ+j]=paramsP->GetParams()[j]->GetModulus().ConvertToInt();rootQP[sizeQ+j]=paramsP->GetParams()[j]->GetRootOfUnity().ConvertToInt();}
    auto KPqp=gpufhe::keygen_host(n,modQP,rootQP,ns,3.19,101);
    gpufhe::KeySwitchConstants K; K.n=n;
    std::vector<uint64_t> mp(sizeP); for(uint32_t j=0;j<sizeP;++j)mp[j]=modQP[sizeQ+j];
    gpufhe::compute_keyswitch_constants(K,mod,mp,numPart);
    for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modQP[sizeQ+j]);K.rootValList.push_back(rootQP[sizeQ+j]);}
    std::vector<uint64_t> PModq_QP(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=modQP[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modQP[sizeQ+j]%q,q);PModq_QP[t]=P;}
    gpufhe::evalkeygen_host(K,KPqp.s,KPqp.pkA,KPqp.pkB,PModq_QP,modQP,rootQP,ns,3.19,202);

    auto C=gpufhe::ks_context_create(K); auto W=gpufhe::ks_work_create(C);
    std::mt19937_64 rng(88);
    std::vector<uint64_t> a((size_t)sizeQ*n);
    for(uint32_t t=0;t<sizeQ;++t)for(uint32_t k=0;k<n;++k)a[(size_t)t*n+k]=rng()%mod[t];
    uint64_t *d_a,*d_b0,*d_b1;
    cudaMalloc(&d_a,a.size()*8); cudaMemcpy(d_a,a.data(),a.size()*8,cudaMemcpyHostToDevice);
    cudaMalloc(&d_b0,(size_t)sizeQ*n*8); cudaMalloc(&d_b1,(size_t)sizeQ*n*8);

    // direct
    gpufhe::keyswitch_resident(d_a,d_b0,d_b1,C,W,0);
    cudaDeviceSynchronize();
    std::vector<uint64_t> ref0((size_t)sizeQ*n),ref1((size_t)sizeQ*n);
    cudaMemcpy(ref0.data(),d_b0,ref0.size()*8,cudaMemcpyDeviceToHost);
    cudaMemcpy(ref1.data(),d_b1,ref1.size()*8,cudaMemcpyDeviceToHost);

    // graph capture + replay (zero outputs first so a no-op replay can't pass)
    cudaMemset(d_b0,0,(size_t)sizeQ*n*8); cudaMemset(d_b1,0,(size_t)sizeQ*n*8);
    cudaStream_t s; cudaStreamCreate(&s);
    cudaGraph_t g; cudaGraphExec_t ge;
    cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal);
    gpufhe::keyswitch_resident(d_a,d_b0,d_b1,C,W,s);
    cudaStreamEndCapture(s,&g);
    cudaError_t ie=cudaGraphInstantiate(&ge,g,nullptr,nullptr,0);
    if(ie!=cudaSuccess){ std::cout<<"[FAIL] graph instantiate: "<<cudaGetErrorString(ie)<<"\n"; return 1; }
    cudaGraphLaunch(ge,s); cudaStreamSynchronize(s);

    std::vector<uint64_t> g0((size_t)sizeQ*n),g1((size_t)sizeQ*n);
    cudaMemcpy(g0.data(),d_b0,g0.size()*8,cudaMemcpyDeviceToHost);
    cudaMemcpy(g1.data(),d_b1,g1.size()*8,cudaMemcpyDeviceToHost);
    uint32_t bad=0;
    for(size_t x=0;x<g0.size();++x){ if(g0[x]!=ref0[x])++bad; if(g1[x]!=ref1[x])++bad; }
    std::cout<<"graph-vs-direct mismatches="<<bad<<"\n";
    if(!bad){std::cout<<"[PASS] graph-replayed keyswitch bit-exact vs direct\n";return 0;}
    std::cout<<"[FAIL]\n"; return 1;
}
