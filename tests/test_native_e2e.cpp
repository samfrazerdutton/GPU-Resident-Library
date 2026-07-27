// MILESTONE: fully OpenFHE-free packed CKKS multiply at n=1024. Native primes,
// roots, keys, keyswitch constants, rescale constants — NO openfhe include.
// encode -> encrypt -> tensor -> relin -> combine -> resident rescale ->
// decrypt -> decode -> verify slotwise products.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
namespace gpufhe {
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void decode_host(std::vector<std::complex<double>>&, const std::vector<int64_t>&, uint32_t, double);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void native_rescale_consts(std::vector<uint64_t>&, std::vector<uint64_t>&,
                           const std::vector<uint64_t>&, uint32_t);
}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
static uint64_t am(uint64_t a,uint64_t b,uint64_t q){uint64_t s=a+b;return s>=q?s-q:s;}

int main(){
    const uint32_t n=1024, S=n/2, sizeQ=4, sizeP=2, numPart=2;
    const uint32_t sizeQlP=sizeQ+sizeP; const uint64_t ns=1;

    std::vector<uint64_t> mod, modP;
    gpufhe::native_primes(mod,sizeQ,35,n,{});
    gpufhe::native_primes(modP,sizeP,36,n,mod);
    std::vector<uint64_t> modQP=mod; for(auto p:modP) modQP.push_back(p);
    std::vector<uint64_t> root(sizeQ), rootQP;
    for(uint32_t i=0;i<sizeQ;++i) root[i]=gpufhe::native_root(n,mod[i]);
    rootQP=root; for(auto p:modP) rootQP.push_back(gpufhe::native_root(n,p));

    auto KPqp=gpufhe::keygen_host(n,modQP,rootQP,ns,3.19,101);
    auto KPq =gpufhe::keygen_host(n,mod,root,ns,3.19,101);

    gpufhe::KeySwitchConstants K; K.n=n;
    gpufhe::compute_keyswitch_constants(K,mod,modP,numPart);
    for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modQP[sizeQ+j]);K.rootValList.push_back(rootQP[sizeQ+j]);}
    std::vector<uint64_t> PModq_QP(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=modQP[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modQP[sizeQ+j]%q,q);PModq_QP[t]=P;}
    gpufhe::evalkeygen_host(K,KPqp.s,KPqp.pkA,KPqp.pkB,PModq_QP,modQP,rootQP,ns,3.19,202);

    std::vector<uint64_t> rs1,rs2;
    gpufhe::native_rescale_consts(rs1,rs2,mod,sizeQ-1);

    const double Delta=std::pow(2.0,28);
    std::vector<std::complex<double>> z1(S),z2(S);
    for(uint32_t i=0;i<S;++i){ z1[i]={0.4*std::sin(0.05*i),0}; z2[i]={0.4*std::cos(0.03*i)+0.1,0}; }
    std::vector<int64_t> m1,m2;
    gpufhe::encode_host(m1,z1,n,Delta); gpufhe::encode_host(m2,z2,n,Delta);
    std::vector<uint64_t> c0a,c1a,c0b,c1b;
    gpufhe::encrypt_host(c0a,c1a,m1,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,303);
    gpufhe::encrypt_host(c0b,c1b,m2,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,304);

    const size_t T=(size_t)sizeQ*n;
    std::vector<uint64_t> t0(T),t1(T),t2(T);
    for(uint32_t t=0;t<sizeQ;++t){uint64_t q=mod[t];
        for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
            t0[x]=mm(c0a[x],c0b[x],q);
            t1[x]=am(mm(c0a[x],c1b[x],q),mm(c1a[x],c0b[x],q),q);
            t2[x]=mm(c1a[x],c1b[x],q);}}

    auto R=gpufhe::keyswitch_core_resident(t2,K);
    std::vector<uint64_t> r0(T),r1(T);
    for(uint32_t t=0;t<sizeQ;++t){uint64_t q=mod[t];
        for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
            r0[x]=am(t0[x],R.ba0[x],q); r1[x]=am(t1[x],R.ba1[x],q);}}

    // resident rescale (context from native K — still zero OpenFHE)
    auto C=gpufhe::ks_context_create(K);
    uint64_t *dr0,*dr1,*scr,*drp;
    cudaMalloc(&dr0,T*8);cudaMalloc(&dr1,T*8);cudaMalloc(&scr,(size_t)n*8);cudaMalloc(&drp,(size_t)n*8);
    cudaMemcpy(dr0,r0.data(),T*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dr1,r1.data(),T*8,cudaMemcpyHostToDevice);
    gpufhe::rescale_resident_raw(dr0,sizeQ,C,rs1,rs2,scr,drp,0);
    gpufhe::rescale_resident_raw(dr1,sizeQ,C,rs1,rs2,scr,drp,0);
    cudaDeviceSynchronize();
    cudaMemcpy(r0.data(),dr0,T*8,cudaMemcpyDeviceToHost);
    cudaMemcpy(r1.data(),dr1,T*8,cudaMemcpyDeviceToHost);

    std::vector<uint64_t> modR(mod.begin(),mod.end()-1), rootR(root.begin(),root.end()-1);
    auto KPr=gpufhe::keygen_host(n,modR,rootR,ns,3.19,101);
    std::vector<int64_t> dec;
    gpufhe::decrypt_host(dec,r0,r1,KPr.s,n,modR,rootR);
    double DeltaOut=Delta*Delta/(double)mod[sizeQ-1];
    std::vector<std::complex<double>> zout;
    gpufhe::decode_host(zout,dec,n,DeltaOut);

    double maxerr=0;
    for(uint32_t i=0;i<S;++i){ double exp=z1[i].real()*z2[i].real();
        maxerr=std::max(maxerr,std::abs(zout[i].real()-exp)); }
    std::cout<<"n=1024 slots="<<S<<" DeltaOut=2^"<<std::log2(DeltaOut)<<" max slot err="<<maxerr<<"\n";
    std::cout<<"  expect[0..3]="; for(int k=0;k<4;++k)std::cout<<z1[k].real()*z2[k].real()<<" "; std::cout<<"\n";
    std::cout<<"  got   [0..3]="; for(int k=0;k<4;++k)std::cout<<zout[k].real()<<" "; std::cout<<"\n";
    if(maxerr<5e-3){std::cout<<"[PASS] FULLY OPENFHE-FREE packed CKKS multiply (native primes/roots/keys/constants/rescale)\n";return 0;}
    std::cout<<"[FAIL]\n"; return 1;
}
