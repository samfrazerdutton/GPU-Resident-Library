// CoeffsToSlots ENGINE gate: apply an ARBITRARY complex linear map U (S x S)
// homomorphically via full diagonal decomposition, verify against plaintext U*x.
//   ct_out = sum_{j=0}^{S-1} diag_j(U) (x) rot_j(ct),  diag_j(U)[i] = U[i][(i+j)%S]
// Fully OpenFHE-free (n=256 native params). This IS the machine CoeffsToSlots
// runs on; the real encode matrix is a specific U (built next).
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
void native_rescale_consts(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<uint64_t>&, uint32_t);
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
    const uint32_t n=1024, S=n/2, sizeQ=4, sizeP=2, numPart=2;  // proven ring floor; S=512 diagonals
    const uint32_t sizeQlP=sizeQ+sizeP, M=2*n; const uint64_t ns=1;

    std::vector<uint64_t> mod, modP;
    gpufhe::native_primes(mod,sizeQ,50,n,{});
    gpufhe::native_primes(modP,sizeP,51,n,mod);
    std::vector<uint64_t> modQP=mod; for(auto p:modP)modQP.push_back(p);
    std::vector<uint64_t> root(sizeQ),rootQP;
    for(uint32_t i=0;i<sizeQ;++i)root[i]=gpufhe::native_root(n,mod[i]);
    rootQP=root; for(auto p:modP)rootQP.push_back(gpufhe::native_root(n,p));
    auto KPqp=gpufhe::keygen_host(n,modQP,rootQP,ns,3.19,101);
    auto KPq =gpufhe::keygen_host(n,mod,root,ns,3.19,101);
    std::vector<uint64_t> PModq_QP(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=modQP[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modQP[sizeQ+j]%q,q);PModq_QP[t]=P;}

    // rotation key builder for shift r (k = 5^r mod 2n)
    auto mkK=[&](uint32_t r)->gpufhe::KeySwitchConstants{
        uint64_t k=1; for(uint32_t i=0;i<r;++i)k=(k*5)%M;
        std::vector<uint64_t> sA=KPqp.s;
        gpufhe::automorphism_eval_host(sA,sizeQlP,n,(uint32_t)k,modQP,rootQP);
        gpufhe::KeySwitchConstants K; K.n=n;
        gpufhe::compute_keyswitch_constants(K,mod,modP,numPart);
        for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
        for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modQP[sizeQ+j]);K.rootValList.push_back(rootQP[sizeQ+j]);}
        gpufhe::evalkeygen_host_sold(K,KPqp.s,sA,KPqp.pkA,KPqp.pkB,PModq_QP,modQP,rootQP,ns,3.19,900+r);
        return K; };

    // rescale-table context
    gpufhe::KeySwitchConstants Kr; Kr.n=n;
    gpufhe::compute_keyswitch_constants(Kr,mod,modP,numPart);
    for(uint32_t i=0;i<sizeQ;++i){Kr.rootModList.push_back(mod[i]);Kr.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){Kr.rootModList.push_back(modQP[sizeQ+j]);Kr.rootValList.push_back(rootQP[sizeQ+j]);}
    Kr.av.assign(Kr.numPart,{});Kr.bv.assign(Kr.numPart,{});Kr.evalKeyTowers=sizeQlP;
    for(uint32_t p=0;p<Kr.numPart;++p){Kr.av[p].assign((size_t)sizeQlP*n,0);Kr.bv[p].assign((size_t)sizeQlP*n,0);}
    auto C=gpufhe::ks_context_create(Kr);
    std::vector<uint64_t> rs1,rs2; gpufhe::native_rescale_consts(rs1,rs2,mod,sizeQ-1);

    // arbitrary target matrix U (S x S, complex) and input x
    std::vector<std::complex<double>> x(S);
    for(uint32_t i=0;i<S;++i) x[i]={0.3*std::sin(0.1*i),0};
    auto Uentry=[&](uint32_t i,uint32_t j)->std::complex<double>{
        return {0.02*std::cos(0.03*i*j+0.1)+ (i==j?0.5:0.0), 0.0}; };  // real, diag-dominant

    const double Delta=std::pow(2.0,35);

    // precompute rotation keys for all needed shifts (build lazily: only nonzero diagonals)
    // encrypt x
    std::vector<int64_t> mx; gpufhe::encode_host(mx,x,n,Delta);
    std::vector<uint64_t> c0,c1;
    gpufhe::encrypt_host(c0,c1,mx,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,303);

    // accumulate sum_j diag_j (x) rot_j(ct)
    const size_t T=(size_t)sizeQ*n;
    std::vector<uint64_t> acc0(T,0),acc1(T,0); bool first=true;
    for(uint32_t j=0;j<S;++j){
        // diagonal j of U: d[i] = U[i][(i+j)%S]
        std::vector<std::complex<double>> d(S);
        double nz=0; for(uint32_t i=0;i<S;++i){ d[i]=Uentry(i,(i+j)%S); nz+=std::abs(d[i]); }
        if(nz<1e-9) continue;
        std::vector<int64_t> md; gpufhe::encode_host(md,d,n,Delta);
        std::vector<uint64_t> dEval; gpufhe::pt_to_eval_host(dEval,md,sizeQ,n,mod,root);
        // rot_j(ct)
        std::vector<uint64_t> b0=c0,b1=c1;
        if(j>0){ auto Kj=mkK(j);
            uint64_t kk=1; for(uint32_t t=0;t<j;++t)kk=(kk*5)%M;
            gpufhe::rotate_ct_host(b0,b1,(uint32_t)kk,Kj,sizeQ,n,mod,root); }
        gpufhe::ct_mul_pt_host(b0,b1,dEval,sizeQ,n,mod);
        if(first){acc0=b0;acc1=b1;first=false;}
        else gpufhe::ct_add_ct_host(acc0,acc1,b0,b1,sizeQ,n,mod);
    }

    // rescale, decrypt at reduced level, decode
    uint64_t *dr0,*dr1,*scr,*drp;
    cudaMalloc(&dr0,T*8);cudaMalloc(&dr1,T*8);cudaMalloc(&scr,(size_t)n*8);cudaMalloc(&drp,(size_t)n*8);
    cudaMemcpy(dr0,acc0.data(),T*8,cudaMemcpyHostToDevice);cudaMemcpy(dr1,acc1.data(),T*8,cudaMemcpyHostToDevice);
    gpufhe::rescale_resident_raw(dr0,sizeQ,C,rs1,rs2,scr,drp,0);
    gpufhe::rescale_resident_raw(dr1,sizeQ,C,rs1,rs2,scr,drp,0);
    cudaDeviceSynchronize();
    cudaMemcpy(acc0.data(),dr0,T*8,cudaMemcpyDeviceToHost);cudaMemcpy(acc1.data(),dr1,T*8,cudaMemcpyDeviceToHost);

    std::vector<uint64_t> modR(mod.begin(),mod.end()-1),rootR(root.begin(),root.end()-1);
    auto KPr=gpufhe::keygen_host(n,modR,rootR,ns,3.19,101);
    std::vector<int64_t> dec; gpufhe::decrypt_host(dec,acc0,acc1,KPr.s,n,modR,rootR);
    double DeltaOut=Delta*Delta/(double)mod[sizeQ-1];
    std::vector<std::complex<double>> y; gpufhe::decode_host(y,dec,n,DeltaOut);

    // plaintext reference: (U x)[i] = sum_j U[i][j] x[j]  == sum_j diag_j[i] x[(i+j)%S]
    double maxerr=0;
    for(uint32_t i=0;i<S;++i){ std::complex<double> e{0,0};
        for(uint32_t j=0;j<S;++j) e+=Uentry(i,j)*x[j];
        maxerr=std::max(maxerr,std::abs(y[i].real()-e.real())); }
    std::cout<<"n=1024 S="<<S<<" arbitrary linear map, DeltaOut=2^"<<std::log2(DeltaOut)<<" max err="<<maxerr<<"\n";
    std::cout<<"  y[0..3]="; for(int k=0;k<4;++k)std::cout<<y[k].real()<<" "; std::cout<<"\n";
    if(maxerr<5e-3){std::cout<<"[PASS] arbitrary homomorphic linear transform (CoeffsToSlots engine)\n";return 0;}
    std::cout<<"[FAIL]\n"; return 1;
}
