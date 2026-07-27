// BOOTSTRAP CORE MACHINE: homomorphic linear transform via rotations+diagonals:
//   y = d0 (x) x  +  d1 (x) rot_1(x)  +  d2 (x) rot_2(x)
// (the BSGS building block of CoeffsToSlots). One rotation key per index,
// plaintext diagonals at scale Delta, one rescale at the end, decode, compare.
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
void automorphism_eval_host(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t,
                            const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void rotate_ct_host(std::vector<uint64_t>&, std::vector<uint64_t>&, uint32_t,
                    const KeySwitchConstants&, uint32_t, uint32_t,
                    const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void pt_to_eval_host(std::vector<uint64_t>&, const std::vector<int64_t>&, uint32_t, uint32_t,
                     const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void ct_mul_pt_host(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<uint64_t>&,
                    uint32_t, uint32_t, const std::vector<uint64_t>&);
void ct_add_ct_host(std::vector<uint64_t>&, std::vector<uint64_t>&,
                    const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                    uint32_t, uint32_t, const std::vector<uint64_t>&);
}
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
    const uint32_t M=2*n;
    std::vector<uint64_t> mod(sizeQ),root(sizeQ),modQP(sizeQlP),rootQP(sizeQlP);
    for(uint32_t i=0;i<sizeQ;++i){mod[i]=ep->GetParams()[i]->GetModulus().ConvertToInt();root[i]=ep->GetParams()[i]->GetRootOfUnity().ConvertToInt();modQP[i]=mod[i];rootQP[i]=root[i];}
    for(uint32_t j=0;j<sizeP;++j){modQP[sizeQ+j]=paramsP->GetParams()[j]->GetModulus().ConvertToInt();rootQP[sizeQ+j]=paramsP->GetParams()[j]->GetRootOfUnity().ConvertToInt();}
    auto KPqp=gpufhe::keygen_host(n,modQP,rootQP,ns,3.19,101);
    auto KPq =gpufhe::keygen_host(n,mod,root,ns,3.19,101);
    std::vector<uint64_t> PModq_QP(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=modQP[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modQP[sizeQ+j]%q,q);PModq_QP[t]=P;}

    // rotation keys for r=1,2 (k=5^r mod 2n)
    auto mkK=[&](uint32_t k)->gpufhe::KeySwitchConstants{
        std::vector<uint64_t> sA=KPqp.s;
        gpufhe::automorphism_eval_host(sA,sizeQlP,n,k,modQP,rootQP);
        gpufhe::KeySwitchConstants K; K.n=n;
        std::vector<uint64_t> mp(sizeP); for(uint32_t j=0;j<sizeP;++j)mp[j]=modQP[sizeQ+j];
        gpufhe::compute_keyswitch_constants(K,mod,mp,numPart);
        for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
        for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modQP[sizeQ+j]);K.rootValList.push_back(rootQP[sizeQ+j]);}
        gpufhe::evalkeygen_host_sold(K,KPqp.s,sA,KPqp.pkA,KPqp.pkB,PModq_QP,modQP,rootQP,ns,3.19,800+k);
        return K; };
    uint64_t k1=5%M, k2=(5*5)%M;
    auto K1=mkK((uint32_t)k1), K2=mkK((uint32_t)k2);

    // rescale-table context + constants
    gpufhe::KeySwitchConstants Kr; Kr.n=n;
    { std::vector<uint64_t> mp(sizeP); for(uint32_t j=0;j<sizeP;++j)mp[j]=modQP[sizeQ+j];
      gpufhe::compute_keyswitch_constants(Kr,mod,mp,numPart);
      for(uint32_t i=0;i<sizeQ;++i){Kr.rootModList.push_back(mod[i]);Kr.rootValList.push_back(root[i]);}
      for(uint32_t j=0;j<sizeP;++j){Kr.rootModList.push_back(modQP[sizeQ+j]);Kr.rootValList.push_back(rootQP[sizeQ+j]);}
      Kr.av.assign(Kr.numPart,{}); Kr.bv.assign(Kr.numPart,{}); Kr.evalKeyTowers=sizeQlP;
      for(uint32_t p=0;p<Kr.numPart;++p){Kr.av[p].assign((size_t)sizeQlP*n,0);Kr.bv[p].assign((size_t)sizeQlP*n,0);} }
    auto C=gpufhe::ks_context_create(Kr);
    std::vector<uint64_t> rs1(sizeQ-1),rs2(sizeQ-1);
    { const auto&a=cp->GetqlInvModq(0); const auto&b=cp->GetQlQlInvModqlDivqlModq(0);
      for(uint32_t t=0;t<sizeQ-1;++t){rs1[t]=a[t].ConvertToInt();rs2[t]=b[t].ConvertToInt();} }

    // input + diagonals
    const uint32_t S=n/2; const double Delta=std::pow(2.0,35);
    std::vector<std::complex<double>> x(S),d0(S),d1(S),d2(S);
    for(uint32_t i=0;i<S;++i){ x[i]={0.3*std::sin(0.002*i)+0.05,0};
        d0[i]={0.4*std::cos(0.001*i),0}; d1[i]={0.3*std::sin(0.0015*i)+0.1,0}; d2[i]={0.25*std::cos(0.0008*i)-0.05,0}; }
    std::vector<int64_t> mx,md0,md1,md2;
    gpufhe::encode_host(mx,x,n,Delta);
    gpufhe::encode_host(md0,d0,n,Delta); gpufhe::encode_host(md1,d1,n,Delta); gpufhe::encode_host(md2,d2,n,Delta);
    std::vector<uint64_t> e0,e1,e2;
    gpufhe::pt_to_eval_host(e0,md0,sizeQ,n,mod,root);
    gpufhe::pt_to_eval_host(e1,md1,sizeQ,n,mod,root);
    gpufhe::pt_to_eval_host(e2,md2,sizeQ,n,mod,root);

    std::vector<uint64_t> c0,c1;
    gpufhe::encrypt_host(c0,c1,mx,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,909);

    // term 0: d0*x
    std::vector<uint64_t> a0=c0,a1=c1;
    gpufhe::ct_mul_pt_host(a0,a1,e0,sizeQ,n,mod);
    // term 1: d1*rot1(x)
    std::vector<uint64_t> b0=c0,b1=c1;
    gpufhe::rotate_ct_host(b0,b1,(uint32_t)k1,K1,sizeQ,n,mod,root);
    gpufhe::ct_mul_pt_host(b0,b1,e1,sizeQ,n,mod);
    gpufhe::ct_add_ct_host(a0,a1,b0,b1,sizeQ,n,mod);
    // term 2: d2*rot2(x)
    std::vector<uint64_t> g0=c0,g1=c1;
    gpufhe::rotate_ct_host(g0,g1,(uint32_t)k2,K2,sizeQ,n,mod,root);
    gpufhe::ct_mul_pt_host(g0,g1,e2,sizeQ,n,mod);
    gpufhe::ct_add_ct_host(a0,a1,g0,g1,sizeQ,n,mod);

    // rescale, decrypt at reduced level, decode
    const size_t T=(size_t)sizeQ*n;
    uint64_t *dr0,*dr1,*scr,*drp;
    cudaMalloc(&dr0,T*8);cudaMalloc(&dr1,T*8);cudaMalloc(&scr,(size_t)n*8);cudaMalloc(&drp,(size_t)n*8);
    cudaMemcpy(dr0,a0.data(),T*8,cudaMemcpyHostToDevice); cudaMemcpy(dr1,a1.data(),T*8,cudaMemcpyHostToDevice);
    gpufhe::rescale_resident_raw(dr0,sizeQ,C,rs1,rs2,scr,drp,0);
    gpufhe::rescale_resident_raw(dr1,sizeQ,C,rs1,rs2,scr,drp,0);
    cudaDeviceSynchronize();
    cudaMemcpy(a0.data(),dr0,T*8,cudaMemcpyDeviceToHost); cudaMemcpy(a1.data(),dr1,T*8,cudaMemcpyDeviceToHost);

    std::vector<uint64_t> modR(mod.begin(),mod.end()-1), rootR(root.begin(),root.end()-1);
    auto KPr=gpufhe::keygen_host(n,modR,rootR,ns,3.19,101);
    std::vector<int64_t> dec; gpufhe::decrypt_host(dec,a0,a1,KPr.s,n,modR,rootR);
    double DeltaOut=Delta*Delta/(double)mod[sizeQ-1];
    std::vector<std::complex<double>> y; gpufhe::decode_host(y,dec,n,DeltaOut);

    double maxerr=0;
    for(uint32_t i=0;i<S;++i){
        double exp=d0[i].real()*x[i].real()
                  +d1[i].real()*x[(i+1)%S].real()
                  +d2[i].real()*x[(i+2)%S].real();
        maxerr=std::max(maxerr,std::abs(y[i].real()-exp)); }
    std::cout<<"linear transform (3 diagonals, rot 0/1/2) max slot err = "<<maxerr<<"\n";
    if(maxerr<1e-3){std::cout<<"[PASS] homomorphic linear transform (rotations + diagonals + rescale)\n";return 0;}
    std::cout<<"[FAIL]\n"; return 1;
}
