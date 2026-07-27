// REAL CKKS end-to-end: encode two real vectors -> encrypt -> tensor ->
// resident relin -> combine -> resident rescale -> decrypt -> decode ->
// verify slotwise products. Post-rescale scale = Delta^2/qLast.
#include "openfhe.h"
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
using namespace lbcrypto;
namespace gpufhe {
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void decode_host(std::vector<std::complex<double>>&, const std::vector<int64_t>&, uint32_t, double);
}
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
    auto KPq =gpufhe::keygen_host(n,mod,root,ns,3.19,101);
    gpufhe::KeySwitchConstants K; K.n=n;
    std::vector<uint64_t> mp(sizeP); for(uint32_t j=0;j<sizeP;++j)mp[j]=modQP[sizeQ+j];
    gpufhe::compute_keyswitch_constants(K,mod,mp,numPart);
    for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modQP[sizeQ+j]);K.rootValList.push_back(rootQP[sizeQ+j]);}
    std::vector<uint64_t> PModq_QP(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=modQP[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modQP[sizeQ+j]%q,q);PModq_QP[t]=P;}
    gpufhe::evalkeygen_host(K,KPqp.s,KPqp.pkA,KPqp.pkB,PModq_QP,modQP,rootQP,ns,3.19,202);
    std::vector<uint64_t> rs1(sizeQ-1),rs2(sizeQ-1);
    { const auto&a=cp->GetqlInvModq(0); const auto&b=cp->GetQlQlInvModqlDivqlModq(0);
      for(uint32_t t=0;t<sizeQ-1;++t){rs1[t]=a[t].ConvertToInt();rs2[t]=b[t].ConvertToInt();} }

    // real vectors in slots
    const uint32_t S=n/2; const double Delta=std::pow(2.0,35);
    std::vector<std::complex<double>> z1(S),z2(S);
    for(uint32_t k=0;k<S;++k){ z1[k]={0.5*std::sin(0.001*k),0}; z2[k]={0.5*std::cos(0.0007*k),0}; }
    std::vector<int64_t> m1,m2;
    gpufhe::encode_host(m1,z1,n,Delta); gpufhe::encode_host(m2,z2,n,Delta);

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
    gpufhe::rescale_resident_raw(r0,sizeQ,C,rs1,rs2,scr,drp,0);
    gpufhe::rescale_resident_raw(r1,sizeQ,C,rs1,rs2,scr,drp,0);
    cudaDeviceSynchronize();

    std::vector<uint64_t> h0(T),h1(T);
    cudaMemcpy(h0.data(),r0,T*8,cudaMemcpyDeviceToHost);
    cudaMemcpy(h1.data(),r1,T*8,cudaMemcpyDeviceToHost);
    std::vector<int64_t> dec;
    gpufhe::decrypt_host(dec,h0,h1,KPq.s,n,mod,root);

    // decode at the post-rescale scale Delta^2/qLast
    double DeltaOut=Delta*Delta/(double)mod[sizeQ-1];
    std::vector<std::complex<double>> zout;
    gpufhe::decode_host(zout,dec,n,DeltaOut);

    double maxerr=0;
    for(uint32_t k=0;k<S;++k){ double exp=z1[k].real()*z2[k].real();
        maxerr=std::max(maxerr,std::abs(zout[k].real()-exp)); }
    std::cout<<"slots="<<S<<" DeltaOut=2^"<<std::log2(DeltaOut)<<" max slot err="<<maxerr<<"\n";
    std::cout<<"  z1*z2[0..3]="; for(int k=0;k<4;++k)std::cout<<z1[k].real()*z2[k].real()<<" "; std::cout<<"\n";
    std::cout<<"  got  [0..3]="; for(int k=0;k<4;++k)std::cout<<zout[k].real()<<" "; std::cout<<"\n";
    if(maxerr<1e-3){ std::cout<<"[PASS] REAL packed CKKS multiply end-to-end (encode->encrypt->mult->rescale->decrypt->decode)\n"; return 0; }
    std::cout<<"[FAIL]\n"; return 1;
}
