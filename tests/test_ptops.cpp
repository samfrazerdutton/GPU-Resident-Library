// BOOTSTRAP PREREQ 2: plaintext-ciphertext ops. Enc(z) * pt(w) + Enc(z2):
// mul by encoded plaintext (scale->Delta^2, rescale), add ciphertexts, decode.
// Verify slots = z*w + z2.
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
    auto ep=cp->GetElementParams();
    uint32_t n=ep->GetRingDimension(), sizeQ=ep->GetParams().size();
    uint32_t ns=cp->GetNoiseScale(), sizeP=cp->GetParamsP()->GetParams().size();
    std::vector<uint64_t> mod(sizeQ),root(sizeQ);
    for(uint32_t i=0;i<sizeQ;++i){mod[i]=ep->GetParams()[i]->GetModulus().ConvertToInt();root[i]=ep->GetParams()[i]->GetRootOfUnity().ConvertToInt();}
    auto KPq=gpufhe::keygen_host(n,mod,root,ns,3.19,101);

    // rescale constants + a KS context purely for rescale tables
    std::vector<uint64_t> modQP, rootQP; modQP=mod; rootQP=root;
    auto paramsP=cp->GetParamsP();
    for(uint32_t j=0;j<sizeP;++j){modQP.push_back(paramsP->GetParams()[j]->GetModulus().ConvertToInt());rootQP.push_back(paramsP->GetParams()[j]->GetRootOfUnity().ConvertToInt());}
    gpufhe::KeySwitchConstants K; K.n=n;
    std::vector<uint64_t> mp(sizeP); for(uint32_t j=0;j<sizeP;++j)mp[j]=modQP[sizeQ+j];
    gpufhe::compute_keyswitch_constants(K,mod,mp,cp->GetNumPartQ());
    for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modQP[sizeQ+j]);K.rootValList.push_back(rootQP[sizeQ+j]);}
    K.av.assign(K.numPart,{}); K.bv.assign(K.numPart,{}); K.evalKeyTowers=sizeQ+sizeP; // no key needed
    for(uint32_t p=0;p<K.numPart;++p){K.av[p].assign((size_t)(sizeQ+sizeP)*n,0);K.bv[p].assign((size_t)(sizeQ+sizeP)*n,0);}
    auto C=gpufhe::ks_context_create(K);
    std::vector<uint64_t> rs1(sizeQ-1),rs2(sizeQ-1);
    { const auto&a=cp->GetqlInvModq(0); const auto&b=cp->GetQlQlInvModqlDivqlModq(0);
      for(uint32_t t=0;t<sizeQ-1;++t){rs1[t]=a[t].ConvertToInt();rs2[t]=b[t].ConvertToInt();} }

    const uint32_t S=n/2; const double Delta=std::pow(2.0,35);
    std::vector<std::complex<double>> z(S),w(S),z2(S);
    for(uint32_t i=0;i<S;++i){ z[i]={0.3*std::sin(0.002*i),0}; w[i]={0.5*std::cos(0.0011*i)+0.2,0}; z2[i]={0.1*std::sin(0.0007*i),0}; }

    std::vector<int64_t> mz,mw,mz2;
    gpufhe::encode_host(mz,z,n,Delta); gpufhe::encode_host(mw,w,n,Delta); gpufhe::encode_host(mz2,z2,n,Delta);
    std::vector<uint64_t> c0,c1,d0,d1;
    gpufhe::encrypt_host(c0,c1,mz,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,606);

    std::vector<uint64_t> wEval;
    gpufhe::pt_to_eval_host(wEval,mw,sizeQ,n,mod,root);
    gpufhe::ct_mul_pt_host(c0,c1,wEval,sizeQ,n,mod);   // scale Delta^2

    // rescale both components (device path)
    const size_t T=(size_t)sizeQ*n;
    uint64_t *dr0,*dr1,*scr,*drp;
    cudaMalloc(&dr0,T*8);cudaMalloc(&dr1,T*8);cudaMalloc(&scr,(size_t)n*8);cudaMalloc(&drp,(size_t)n*8);
    cudaMemcpy(dr0,c0.data(),T*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dr1,c1.data(),T*8,cudaMemcpyHostToDevice);
    gpufhe::rescale_resident_raw(dr0,sizeQ,C,rs1,rs2,scr,drp,0);
    gpufhe::rescale_resident_raw(dr1,sizeQ,C,rs1,rs2,scr,drp,0);
    cudaDeviceSynchronize();
    cudaMemcpy(c0.data(),dr0,T*8,cudaMemcpyDeviceToHost);
    cudaMemcpy(c1.data(),dr1,T*8,cudaMemcpyDeviceToHost);
    // now scale = Delta^2/qLast, level dropped to sizeQ-1

    // encrypt z2 AT THE POST-RESCALE SCALE and reduced level, then add
    double DeltaOut=Delta*Delta/(double)mod[sizeQ-1];
    std::vector<int64_t> mz2s; gpufhe::encode_host(mz2s,z2,n,DeltaOut);
    std::vector<uint64_t> modR(mod.begin(),mod.end()-1), rootR(root.begin(),root.end()-1);
    auto KPr=gpufhe::keygen_host(n,modR,rootR,ns,3.19,101);   // same seed => same s poly
    gpufhe::encrypt_host(d0,d1,mz2s,KPr.pkA,KPr.pkB,n,modR,rootR,ns,3.19,707);
    gpufhe::ct_add_ct_host(c0,c1,d0,d1,sizeQ-1,n,modR);

    std::vector<int64_t> dec; gpufhe::decrypt_host(dec,c0,c1,KPr.s,n,modR,rootR);
    std::vector<std::complex<double>> zout; gpufhe::decode_host(zout,dec,n,DeltaOut);

    double maxerr=0;
    for(uint32_t i=0;i<S;++i){ double exp=z[i].real()*w[i].real()+z2[i].real();
        maxerr=std::max(maxerr,std::abs(zout[i].real()-exp)); }
    std::cout<<"max slot err (z*w+z2) = "<<maxerr<<"\n";
    if(maxerr<1e-3){std::cout<<"[PASS] plaintext mult + rescale + ct add correct\n";return 0;}
    std::cout<<"[FAIL]\n"; return 1;
}
