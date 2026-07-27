// Batched FULL EvalMult (tensor + RESIDENT relin + combine) on N streams vs
// OpenFHE EvalMult x N. All setup (context, work buffers, operand upload)
// OUTSIDE the timer. Interleaved GPU/CPU per rep, median + bootstrap CI.
// This is the honest with-relin throughput number the resident relin unlocks.
#include "openfhe.h"
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
using namespace lbcrypto;
using clk = std::chrono::steady_clock;
extern "C" void LaunchTensor(const uint64_t*,const uint64_t*,const uint64_t*,const uint64_t*,
    uint64_t*,uint64_t*,uint64_t*,const uint64_t*,uint32_t,uint32_t,cudaStream_t);
extern "C" void LaunchCombine(uint64_t*,uint64_t*,const uint64_t*,const uint64_t*,
    const uint64_t*,const uint64_t*,const uint64_t*,uint32_t,uint32_t,cudaStream_t);

static double median(std::vector<double> v){ if(v.empty())return 0; std::sort(v.begin(),v.end());
    size_t n=v.size(); return n%2?v[n/2]:0.5*(v[n/2-1]+v[n/2]); }
static double quantile(std::vector<double> v,double q){ if(v.empty())return 0;
    std::sort(v.begin(),v.end()); double pos=q*(v.size()-1);
    size_t lo=(size_t)pos,hi=std::min(lo+1,v.size()-1); return v[lo]+(pos-lo)*(v[hi]-v[lo]); }
static std::pair<double,double> boot(const std::vector<double>&d,int it=4000){
    if(d.size()<3)return{0,0}; std::mt19937 rng(7);
    std::uniform_int_distribution<size_t> pick(0,d.size()-1);
    std::vector<double> m; m.reserve(it); std::vector<double> s(d.size());
    for(int k=0;k<it;++k){ for(size_t i=0;i<d.size();++i)s[i]=d[pick(rng)];
        m.push_back(median(s)); } return {quantile(m,0.025),quantile(m,0.975)}; }
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}

int main(int argc,char**argv){
    uint32_t N=(argc>1)?(uint32_t)std::stoul(argv[1]):16;
    uint32_t reps=(argc>2)?(uint32_t)std::stoul(argv[2]):9;
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(3); params.SetScalingModSize(50);
    params.SetRingDim(32768); params.SetBatchSize(16384);
    params.SetKeySwitchTechnique(HYBRID);
    params.SetScalingTechnique(FIXEDMANUAL);
    auto cc=GenCryptoContext(params);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    auto kp=cc->KeyGen(); cc->EvalMultKeyGen(kp.secretKey);
    auto cp=std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto ep=cp->GetElementParams(); auto paramsP=cp->GetParamsP();
    uint32_t n=ep->GetRingDimension(), sizeQ=ep->GetParams().size(), sizeP=paramsP->GetParams().size();
    uint32_t sizeQlP=sizeQ+sizeP, ns=cp->GetNoiseScale(), numPart=cp->GetNumPartQ();

    std::vector<uint64_t> mod(sizeQ),root(sizeQ),modQP(sizeQlP),rootQP(sizeQlP);
    for(uint32_t i=0;i<sizeQ;++i){mod[i]=ep->GetParams()[i]->GetModulus().ConvertToInt();root[i]=ep->GetParams()[i]->GetRootOfUnity().ConvertToInt();modQP[i]=mod[i];rootQP[i]=root[i];}
    for(uint32_t j=0;j<sizeP;++j){modQP[sizeQ+j]=paramsP->GetParams()[j]->GetModulus().ConvertToInt();rootQP[sizeQ+j]=paramsP->GetParams()[j]->GetRootOfUnity().ConvertToInt();}

    // native K + eval key + resident context (ALL outside timer)
    auto KPqp=gpufhe::keygen_host(n,modQP,rootQP,ns,3.19,101);
    gpufhe::KeySwitchConstants K; K.n=n;
    std::vector<uint64_t> mp(sizeP); for(uint32_t j=0;j<sizeP;++j)mp[j]=modQP[sizeQ+j];
    gpufhe::compute_keyswitch_constants(K,mod,mp,numPart);
    for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modQP[sizeQ+j]);K.rootValList.push_back(rootQP[sizeQ+j]);}
    std::vector<uint64_t> PModq_QP(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=modQP[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modQP[sizeQ+j]%q,q);PModq_QP[t]=P;}
    gpufhe::evalkeygen_host(K,KPqp.s,KPqp.pkA,KPqp.pkB,PModq_QP,modQP,rootQP,ns,3.19,202);

    auto C=gpufhe::ks_context_create(K);
    std::vector<gpufhe::DeviceKSWork> W(N);
    std::vector<cudaStream_t> streams(N);
    for(uint32_t c=0;c<N;++c){ W[c]=gpufhe::ks_work_create(C); cudaStreamCreate(&streams[c]); }

    // per-chain device operands + outputs (outside timer)
    std::mt19937_64 rng(7);
    const size_t T=(size_t)sizeQ*n;
    auto rb=[&](){std::vector<uint64_t>v(T);for(uint32_t t=0;t<sizeQ;++t)for(uint32_t k=0;k<n;++k)v[(size_t)t*n+k]=rng()%mod[t];return v;};
    uint64_t* d_mods; cudaMalloc(&d_mods,sizeQ*8); cudaMemcpy(d_mods,mod.data(),sizeQ*8,cudaMemcpyHostToDevice);
    struct Chain{uint64_t *c0a,*c1a,*c0b,*c1b,*t0,*t1,*t2,*ba0,*ba1,*r0,*r1;};
    std::vector<Chain> ch(N);
    for(uint32_t c=0;c<N;++c){ auto&x=ch[c];
        for(uint64_t** p : {&x.c0a,&x.c1a,&x.c0b,&x.c1b,&x.t0,&x.t1,&x.t2,&x.ba0,&x.ba1,&x.r0,&x.r1})
            cudaMalloc(p,T*8);
        auto h=rb(); cudaMemcpy(x.c0a,h.data(),T*8,cudaMemcpyHostToDevice);
        h=rb(); cudaMemcpy(x.c1a,h.data(),T*8,cudaMemcpyHostToDevice);
        h=rb(); cudaMemcpy(x.c0b,h.data(),T*8,cudaMemcpyHostToDevice);
        h=rb(); cudaMemcpy(x.c1b,h.data(),T*8,cudaMemcpyHostToDevice); }
    cudaDeviceSynchronize();

    auto run_gpu=[&](){
        auto t0=clk::now();
        for(uint32_t c=0;c<N;++c){ auto&x=ch[c];
            LaunchTensor(x.c0a,x.c1a,x.c0b,x.c1b,x.t0,x.t1,x.t2,d_mods,sizeQ,n,streams[c]);
            gpufhe::keyswitch_resident(x.t2,x.ba0,x.ba1,C,W[c],streams[c]);
            LaunchCombine(x.r0,x.r1,x.t0,x.t1,x.ba0,x.ba1,d_mods,sizeQ,n,streams[c]); }
        cudaDeviceSynchronize();
        return std::chrono::duration<double,std::milli>(clk::now()-t0).count(); };

    // CPU: N OpenFHE EvalMults (operands outside timer)
    auto mkpt=[&](){ std::vector<double> x(16384); for(auto&z:x)z=(rng()%1000)/1000.0; return cc->MakeCKKSPackedPlaintext(x); };
    std::vector<Ciphertext<DCRTPoly>> ca(N),cb(N);
    for(uint32_t c=0;c<N;++c){ ca[c]=cc->Encrypt(kp.publicKey,mkpt()); cb[c]=cc->Encrypt(kp.publicKey,mkpt()); }
    auto run_cpu=[&](){ auto t0=clk::now();
        for(uint32_t c=0;c<N;++c){ auto r=cc->EvalMult(ca[c],cb[c]); (void)r; }
        return std::chrono::duration<double,std::milli>(clk::now()-t0).count(); };

    run_gpu(); run_cpu();  // warmup
    std::vector<double> g,cv,d;
    for(uint32_t r=0;r<reps;++r){ double gt=run_gpu(), ct=run_cpu();
        g.push_back(gt); cv.push_back(ct); d.push_back(gt-ct); }
    auto ci=boot(d);
    std::cout<<"BATCH FULL EvalMult (tensor+RESIDENT relin+combine) N="<<N<<" ring="<<n<<" towers="<<sizeQ<<"\n";
    std::cout<<"  gpu batch = "<<median(g)<<" ms  ("<<median(g)/N<<" ms/op)\n";
    std::cout<<"  cpu batch = "<<median(cv)<<" ms  ("<<median(cv)/N<<" ms/op)\n";
    std::cout<<"  gpu-cpu median = "<<median(d)<<" ms  95% CI ["<<ci.first<<", "<<ci.second<<"]\n";
    std::cout<<"  => "<<((ci.second<0)?"GPU FASTER":(ci.first>0)?"GPU SLOWER":"inconclusive")<<" ("
             <<median(cv)/median(g)<<"x)\n";
    for(uint32_t c=0;c<N;++c){ gpufhe::ks_work_destroy(W[c]); cudaStreamDestroy(streams[c]);
        auto&x=ch[c]; for(uint64_t* p:{x.c0a,x.c1a,x.c0b,x.c1b,x.t0,x.t1,x.t2,x.ba0,x.ba1,x.r0,x.r1})cudaFree(p); }
    gpufhe::ks_context_destroy(C); cudaFree(d_mods);
    return 0;
}
