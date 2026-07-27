// Full pipeline gate: encrypt -> tensor -> resident relin -> combine ->
// RESIDENT RESCALE -> decrypt. ScalingModSize=35 so post-rescale scale
// Delta^2/qLast = 2^15 >> rescale noise. Recover round(dec*qLast/Delta^2).
#include "openfhe.h"
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
using namespace lbcrypto;
extern "C" void LaunchTensor(const uint64_t*,const uint64_t*,const uint64_t*,const uint64_t*,
    uint64_t*,uint64_t*,uint64_t*,const uint64_t*,uint32_t,uint32_t,cudaStream_t);
extern "C" void LaunchCombine(uint64_t*,uint64_t*,const uint64_t*,const uint64_t*,
    const uint64_t*,const uint64_t*,const uint64_t*,uint32_t,uint32_t,cudaStream_t);
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}

int main(){
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(3); params.SetScalingModSize(35);
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

    // rescale constants at level 0
    std::vector<uint64_t> s1(sizeQ-1),s2(sizeQ-1);
    { const auto&a=cp->GetqlInvModq(0); const auto&b=cp->GetQlQlInvModqlDivqlModq(0);
      for(uint32_t t=0;t<sizeQ-1;++t){ s1[t]=a[t].ConvertToInt(); s2[t]=b[t].ConvertToInt(); } }

    // secret over Q for enc/dec = KPqp's low towers (same seed poly)
    auto KPq=gpufhe::keygen_host(n,mod,root,ns,3.19,101);

    const int64_t D=(int64_t)1<<25;
    std::vector<int64_t> a1(n,0),a2(n,0); a1[0]=3;a1[1]=-2; a2[0]=4;a2[2]=5;
    std::vector<int64_t> m1(n),m2(n);
    for(uint32_t k=0;k<n;++k){m1[k]=a1[k]*D;m2[k]=a2[k]*D;}
    std::vector<uint64_t> c0a,c1a,c0b,c1b;
    gpufhe::encrypt_host(c0a,c1a,m1,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,303);
    gpufhe::encrypt_host(c0b,c1b,m2,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,304);

    auto C=gpufhe::ks_context_create(K); auto W=gpufhe::ks_work_create(C);
    const size_t T=(size_t)sizeQ*n;
    auto up=[&](const std::vector<uint64_t>&h){uint64_t*d;cudaMalloc(&d,T*8);cudaMemcpy(d,h.data(),T*8,cudaMemcpyHostToDevice);return d;};
    uint64_t *dc0a=up(c0a),*dc1a=up(c1a),*dc0b=up(c0b),*dc1b=up(c1b);
    uint64_t *t0,*t1,*t2,*ba0,*ba1,*r0,*r1,*scr,*drp,*dmods;
    for(uint64_t** p:{&t0,&t1,&t2,&ba0,&ba1,&r0,&r1})cudaMalloc(p,T*8);
    cudaMalloc(&scr,(size_t)n*8);cudaMalloc(&drp,(size_t)n*8);
    cudaMalloc(&dmods,sizeQ*8);cudaMemcpy(dmods,mod.data(),sizeQ*8,cudaMemcpyHostToDevice);

    LaunchTensor(dc0a,dc1a,dc0b,dc1b,t0,t1,t2,dmods,sizeQ,n,0);
    gpufhe::keyswitch_resident(t2,ba0,ba1,C,W,0);
    LaunchCombine(r0,r1,t0,t1,ba0,ba1,dmods,sizeQ,n,0);
    gpufhe::rescale_resident_raw(r0,sizeQ,C,s1,s2,scr,drp,0);
    gpufhe::rescale_resident_raw(r1,sizeQ,C,s1,s2,scr,drp,0);
    cudaDeviceSynchronize();

    std::vector<uint64_t> h0(T),h1(T);
    cudaMemcpy(h0.data(),r0,T*8,cudaMemcpyDeviceToHost);
    cudaMemcpy(h1.data(),r1,T*8,cudaMemcpyDeviceToHost);
    std::vector<int64_t> dec;
    gpufhe::decrypt_host(dec,h0,h1,KPq.s,n,mod,root);

    std::vector<int64_t> expect(n,0);
    for(uint32_t i=0;i<n;++i)if(a1[i])for(uint32_t j=0;j<n;++j)if(a2[j]){
        uint32_t idx=(i+j)%n; int64_t sg=((i+j)>=n)?-1:1; expect[idx]+=sg*a1[i]*a2[j]; }

    long double qLast=(long double)mod[sizeQ-1], D2=(long double)D*(long double)D;
    int bad=0; int64_t maxerr=0;
    for(uint32_t k=0;k<n;++k){ int64_t r=(int64_t)llroundl((long double)dec[k]*qLast/D2);
        int64_t e=r-expect[k]; if(e<0)e=-e; if(e>maxerr)maxerr=e; if(r!=expect[k])++bad; }
    std::cout<<"sizeQ="<<sizeQ<<" post-rescale scale=2^"<<(int)std::log2((double)(D2/qLast))<<" mismatches="<<bad<<" maxerr="<<maxerr<<"\n";
    std::cout<<"  expect[0..4]="; for(int k=0;k<5;++k)std::cout<<expect[k]<<" "; std::cout<<"\n";
    std::cout<<"  got[0..4]   ="; for(int k=0;k<5;++k)std::cout<<llroundl((long double)dec[k]*qLast/D2)<<" "; std::cout<<"\n";
    if(bad==0){std::cout<<"[PASS] full pipeline incl RESIDENT RESCALE numerically correct\n";return 0;}
    std::cout<<"[FAIL]\n"; return 1;
}
