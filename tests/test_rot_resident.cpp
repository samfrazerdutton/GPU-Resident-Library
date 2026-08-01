// Gate + benchmark: rotate_ct_resident vs rotate_ct_host. Must be BIT-EXACT
// (same algorithm; only the keyswitch implementation differs). The bootstrap
// currently spends nearly all its time in rotate_ct_host, which calls the
// host-orchestrated keyswitch_core_resident; this measures what the truly
// resident path buys per rotation.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <algorithm>
namespace gpufhe {
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void automorphism_eval_host(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t,
                            const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void rotate_ct_host(std::vector<uint64_t>&, std::vector<uint64_t>&, uint32_t,
                    const KeySwitchConstants&, uint32_t, uint32_t,
                    const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void rotate_ct_resident(std::vector<uint64_t>&, std::vector<uint64_t>&, uint32_t,
                        const KeySwitchConstants&, uint32_t, uint32_t,
                        const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void set_secret_hamming_weight(uint32_t);
}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
using cd=std::complex<double>;

int main(int argc,char**argv){
    const uint32_t n=(argc>1)?std::stoul(argv[1]):8192;
    const uint32_t tw=(argc>2)?std::stoul(argv[2]):30;
    const uint32_t S=n/2, sizeP=2, numPart=(tw+1)/2; const uint64_t ns=1;
    gpufhe::set_secret_hamming_weight(64);
    std::vector<uint64_t> mod,modP;
    gpufhe::native_primes(mod,1,60,n,{});
    { std::vector<uint64_t> md; gpufhe::native_primes(md,tw-1,55,n,mod); for(auto m:md)mod.push_back(m); }
    gpufhe::native_primes(modP,sizeP,60,n,mod);
    std::vector<uint64_t> mq(mod),rq; for(auto p:modP)mq.push_back(p);
    std::vector<uint64_t> root(tw); for(uint32_t i=0;i<tw;++i)root[i]=gpufhe::native_root(n,mod[i]);
    rq=root; for(auto p:modP)rq.push_back(gpufhe::native_root(n,p));
    const uint32_t M=2*n, sizeQlP=tw+sizeP;

    auto KPqp=gpufhe::keygen_host(n,mq,rq,ns,3.19,101);
    auto KPq =gpufhe::keygen_host(n,mod,root,ns,3.19,101);
    std::vector<uint64_t> PM(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=mq[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modP[j]%q,q);PM[t]=P;}
    uint64_t k=5; std::vector<uint64_t> sA=KPqp.s;
    gpufhe::automorphism_eval_host(sA,sizeQlP,n,(uint32_t)k,mq,rq);
    gpufhe::KeySwitchConstants K; K.n=n;
    gpufhe::compute_keyswitch_constants(K,mod,modP,numPart);
    for(uint32_t i=0;i<tw;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rq[tw+j]);}
    gpufhe::evalkeygen_host_sold(K,KPqp.s,sA,KPqp.pkA,KPqp.pkB,PM,mq,rq,ns,3.19,900);

    std::vector<cd> z(S); for(uint32_t i=0;i<S;++i) z[i]={0.3*std::sin(0.01*i),0};
    std::vector<int64_t> mz; gpufhe::encode_host(mz,z,n,std::pow(2.0,55));
    std::vector<uint64_t> c0,c1;
    gpufhe::encrypt_host(c0,c1,mz,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,303);

    // This machine shows ~30% run-to-run spread (WSL + thermals), which is
    // wider than most effects being measured. Interleave both paths REP times
    // and report medians, so a difference has to survive the noise.
    const uint32_t REP=(argc>3)?std::stoul(argv[3]):7;
    std::vector<double> vh,vr;
    std::vector<uint64_t> a0,a1,b0,b1;
    for(uint32_t rep=0;rep<REP;++rep){
        a0=c0;a1=c1;b0=c0;b1=c1;
        auto p=std::chrono::steady_clock::now();
        gpufhe::rotate_ct_host(a0,a1,(uint32_t)k,K,tw,n,mod,root);
        vh.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-p).count());
        auto q2=std::chrono::steady_clock::now();
        gpufhe::rotate_ct_resident(b0,b1,(uint32_t)k,K,tw,n,mod,root);
        vr.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-q2).count());
    }
    auto med=[](std::vector<double> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; };
    auto lo=[](std::vector<double> v){ std::sort(v.begin(),v.end()); return v.front(); };
    auto hi=[](std::vector<double> v){ std::sort(v.begin(),v.end()); return v.back(); };
    double th=med(vh);
    std::cout<<"  reps="<<REP<<"  host  min/med/max = "<<lo(vh)<<" / "<<th<<" / "<<hi(vh)<<"\n";
    std::cout<<"            resident min/med/max = "<<lo(vr)<<" / "<<med(vr)<<" / "<<hi(vr)<<"\n";
    double tr=med(vr);

    // PHASE BREAKDOWN of the resident path: which part of the 158 ms is real
    // keyswitch work and which is per-call setup that could be hoisted/cached?
    {   auto NOW=[](){return std::chrono::steady_clock::now();};
        auto EL=[](auto a){return std::chrono::duration<double>(
                    std::chrono::steady_clock::now()-a).count();};
        auto p0=NOW(); auto C2=gpufhe::ks_context_create(K);      double tctx=EL(p0);
        auto p1=NOW(); auto W2=gpufhe::ks_work_create(C2);        double twrk=EL(p1);
        const size_t T2=(size_t)tw*n, B2=T2*8;
        uint64_t *da,*db0,*db1;
        auto p2=NOW(); cudaMalloc(&da,B2);cudaMalloc(&db0,B2);cudaMalloc(&db1,B2);
                       double tmal=EL(p2);
        auto p3=NOW(); cudaMemcpy(da,c1.data(),B2,cudaMemcpyHostToDevice);
                       double tup=EL(p3);
        auto p4=NOW(); gpufhe::keyswitch_resident(da,db0,db1,C2,W2,0);
                       cudaDeviceSynchronize();  double tks=EL(p4);
        std::vector<uint64_t> o0(T2),o1(T2);
        auto p5=NOW(); cudaMemcpy(o0.data(),db0,B2,cudaMemcpyDeviceToHost);
                       cudaMemcpy(o1.data(),db1,B2,cudaMemcpyDeviceToHost);
                       double tdn=EL(p5);
        auto p6=NOW(); cudaFree(da);cudaFree(db0);cudaFree(db1);
                       gpufhe::ks_work_destroy(W2); gpufhe::ks_context_destroy(C2);
                       double tfree=EL(p6);
        std::cout<<"  --- resident phase breakdown ---\n"
                 <<"    ks_context_create : "<<tctx <<" s\n"
                 <<"    ks_work_create    : "<<twrk <<" s\n"
                 <<"    cudaMalloc x3     : "<<tmal <<" s\n"
                 <<"    upload ct         : "<<tup  <<" s\n"
                 <<"    KEYSWITCH KERNELS : "<<tks  <<" s\n"
                 <<"    download ct       : "<<tdn  <<" s\n"
                 <<"    free/destroy      : "<<tfree<<" s\n";
        // automorphism cost, measured separately (it is host-side per tower)
        auto v=c0; auto p7=NOW();
        gpufhe::automorphism_eval_host(v,tw,n,(uint32_t)k,mod,root);
        std::cout<<"    automorphism x1   : "<<EL(p7)<<" s  (x2 per rotation)\n"; }

    // GATE: device automorphism must be BIT-IDENTICAL to the host one.
    {   auto hv=c1; 
        gpufhe::automorphism_eval_host(hv,tw,n,(uint32_t)k,mod,root);
        std::vector<uint64_t> dv=c1;
        const size_t TT=(size_t)tw*n;
        uint64_t *d_v,*d_sc;
        cudaMalloc(&d_v,TT*8); cudaMalloc(&d_sc,(size_t)n*8);
        cudaMemcpy(d_v,dv.data(),TT*8,cudaMemcpyHostToDevice);
        auto pa=std::chrono::steady_clock::now();
        gpufhe::automorphism_eval_device(d_v,tw,n,(uint32_t)k,mod,root,d_sc,0);
        cudaDeviceSynchronize();
        double tdev=std::chrono::duration<double>(std::chrono::steady_clock::now()-pa).count();
        cudaMemcpy(dv.data(),d_v,TT*8,cudaMemcpyDeviceToHost);
        cudaFree(d_v); cudaFree(d_sc);
        size_t bad=0; for(size_t i=0;i<hv.size();++i) if(hv[i]!=dv[i]) ++bad;
        std::cout<<"  automorphism device vs host: "<<bad<<" mismatches, device "
                 <<tdev<<" s\n"; }

    size_t diff=0;
    for(size_t i=0;i<a0.size();++i){ if(a0[i]!=b0[i])++diff; if(a1[i]!=b1[i])++diff; }
    std::cout<<"n="<<n<<" tw="<<tw<<" parts="<<numPart<<"\n";
    std::cout<<"  host-orchestrated : "<<th<<" s\n";
    std::cout<<"  fully resident    : "<<tr<<" s   ("<<th/tr<<"x)\n";
    std::cout<<"  mismatching coeffs: "<<diff<<" / "<<2*a0.size()<<"\n";
    if(diff==0){std::cout<<"[PASS] resident rotation is bit-exact\n";return 0;}
    std::cout<<"[FAIL]\n"; return 1;
}
