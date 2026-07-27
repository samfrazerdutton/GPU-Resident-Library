// BOOTSTRAP PREREQ 1: homomorphic rotation. sigma_k applied to (c0,c1), then
// keyswitch sigma(s)->s using a rotation key (evalkeygen with sOld=sigma(s)).
// New ct = (sigma(c0)+ba0, ba1). Decode: slots rotated. k = 5^r mod 2n.
#include "openfhe.h"
#include "keyswitch.h"
#include "keygen.h"
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
}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
static uint64_t am(uint64_t a,uint64_t b,uint64_t q){uint64_t s=a+b;return s>=q?s-q:s;}

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

    const uint32_t r=1, M=2*n;
    uint64_t k=1; for(uint32_t i=0;i<r;++i) k=(k*5)%M;   // 5^r mod 2n

    // rotation key: sOld = sigma_k(s) over QP
    std::vector<uint64_t> sAuto=KPqp.s;
    gpufhe::automorphism_eval_host(sAuto,sizeQlP,n,(uint32_t)k,modQP,rootQP);
    gpufhe::KeySwitchConstants K; K.n=n;
    std::vector<uint64_t> mp(sizeP); for(uint32_t j=0;j<sizeP;++j)mp[j]=modQP[sizeQ+j];
    gpufhe::compute_keyswitch_constants(K,mod,mp,numPart);
    for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modQP[sizeQ+j]);K.rootValList.push_back(rootQP[sizeQ+j]);}
    std::vector<uint64_t> PModq_QP(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=modQP[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modQP[sizeQ+j]%q,q);PModq_QP[t]=P;}
    gpufhe::evalkeygen_host_sold(K,KPqp.s,sAuto,KPqp.pkA,KPqp.pkB,PModq_QP,modQP,rootQP,ns,3.19,404);

    // encode + encrypt a recognizable vector
    const uint32_t S=n/2; const double Delta=std::pow(2.0,35);
    std::vector<std::complex<double>> z(S);
    for(uint32_t i=0;i<S;++i) z[i]={0.4*std::sin(0.003*i)+0.1,0};
    std::vector<int64_t> m; gpufhe::encode_host(m,z,n,Delta);
    std::vector<uint64_t> c0,c1;
    gpufhe::encrypt_host(c0,c1,m,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,505);

    // automorph both components, keyswitch sigma(c1)
    gpufhe::automorphism_eval_host(c0,sizeQ,n,(uint32_t)k,mod,root);
    gpufhe::automorphism_eval_host(c1,sizeQ,n,(uint32_t)k,mod,root);
    auto R=gpufhe::keyswitch_core_resident(c1,K);
    std::vector<uint64_t> r0((size_t)sizeQ*n),r1=R.ba1;
    for(uint32_t t=0;t<sizeQ;++t){uint64_t q=mod[t];
        for(uint32_t kk=0;kk<n;++kk){size_t x=(size_t)t*n+kk; r0[x]=am(c0[x],R.ba0[x],q);}}

    std::vector<int64_t> dec; gpufhe::decrypt_host(dec,r0,r1,KPq.s,n,mod,root);
    std::vector<std::complex<double>> zout; gpufhe::decode_host(zout,dec,n,Delta);

    double eL=0,eR=0;
    for(uint32_t i=0;i<S;++i){
        eL=std::max(eL,std::abs(zout[i].real()-z[(i+r)%S].real()));
        eR=std::max(eR,std::abs(zout[i].real()-z[(i+S-r)%S].real())); }
    double best=std::min(eL,eR);
    std::cout<<"rot r="<<r<<" k=5^r="<<k<<"  err(left)="<<eL<<" err(right)="<<eR
             <<"  direction="<<(eL<eR?"LEFT (slot i <- i+1)":"RIGHT (slot i <- i-1)")<<"\n";
    if(best<1e-3){std::cout<<"[PASS] homomorphic rotation works (automorphism + rotation-key keyswitch)\n";return 0;}
    std::cout<<"[FAIL] best err="<<best<<"\n"; return 1;
}
