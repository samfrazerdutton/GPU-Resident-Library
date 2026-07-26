// THE THESIS TEST. Does a fully VRAM-resident multiply-rescale chain beat
// OpenFHE CPU once the PCIe boundary is gone? Pure throughput: the GPU chain is
// internally self-consistent (correctness proven separately in the bit-exact
// tests); this measures speed only.
//
// Chain: upload once -> mul, rescale, mul, rescale (all resident, no host
// transfer between ops) -> download once. CPU reference: OpenFHE DCRTPoly
// multiply + DropLastElementAndScale, the same op sequence.
//
// WSL can't lock GPU clocks (~30% IQR), so modes are interleaved per rep and
// the reported statistic is median(gpu-cpu) with a bootstrap CI -- common-mode
// noise cancels in the pairing.

#include "openfhe.h"
#include "device_ciphertext.h"
#include "ckks_ops.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace lbcrypto;
using clk = std::chrono::steady_clock;

static double median(std::vector<double> v){
    if(v.empty())return 0; std::sort(v.begin(),v.end());
    size_t n=v.size(); return n%2? v[n/2] : 0.5*(v[n/2-1]+v[n/2]);
}
static double quantile(std::vector<double> v,double q){
    if(v.empty())return 0; std::sort(v.begin(),v.end());
    double pos=q*(v.size()-1); size_t lo=(size_t)pos, hi=std::min(lo+1,v.size()-1);
    return v[lo]+(pos-lo)*(v[hi]-v[lo]);
}
static std::pair<double,double> bootstrap(const std::vector<double>&d,int it=4000){
    if(d.size()<3)return{0,0}; std::mt19937 rng(7);
    std::uniform_int_distribution<size_t> pick(0,d.size()-1);
    std::vector<double> meds; meds.reserve(it); std::vector<double> s(d.size());
    for(int k=0;k<it;++k){ for(size_t i=0;i<d.size();++i)s[i]=d[pick(rng)];
        meds.push_back(median(s)); }
    return {quantile(meds,0.025),quantile(meds,0.975)};
}

int main() {
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(4);
    params.SetScalingModSize(50);
    params.SetRingDim(32768);
    params.SetBatchSize(16384);
    auto cc = GenCryptoContext(params);
    cc->Enable(PKE); cc->Enable(LEVELEDSHE);
    auto kp = cc->KeyGen();

    auto ep = cc->GetCryptoParameters()->GetElementParams();
    const uint32_t n = ep->GetRingDimension();
    const uint32_t towers0 = (uint32_t)ep->GetParams().size();

    std::vector<uint64_t> moduli(towers0);
    for(uint32_t t=0;t<towers0;++t) moduli[t]=ep->GetParams()[t]->GetModulus().ConvertToInt();

    auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(
        cc->GetCryptoParameters());

    // Build device tables ONCE (this is setup, not timed).
    gpufhe::DeviceTables T;
    gpufhe::build_device_tables(T, n, moduli);

    // Two random eval-form ciphertexts as flat tower-major host buffers.
    std::mt19937_64 rng(2026);
    auto randbuf=[&](uint32_t tw){ std::vector<uint64_t> v((size_t)n*tw);
        for(uint32_t t=0;t<tw;++t) for(uint32_t i=0;i<n;++i) v[t*n+i]=rng()%moduli[t];
        return v; };

    const int reps=50, warmup=10;
    std::vector<double> tg, tc;

    // Constants for the two rescales in the chain (levels 0 and 1).
    std::vector<std::vector<uint64_t>> S1(2), S2(2);
    for(int lv=0;lv<2;++lv){
        const auto& q1=cryptoParams->GetqlInvModq(lv);
        const auto& q2=cryptoParams->GetQlQlInvModqlDivqlModq(lv);
        uint32_t sv=towers0-1-lv;
        S1[lv].resize(sv); S2[lv].resize(sv);
        for(uint32_t t=0;t<sv;++t){ S1[lv][t]=q1[t].ConvertToInt(); S2[lv][t]=q2[t].ConvertToInt(); }
    }

    auto run_gpu=[&](){
        auto ha=randbuf(towers0), hb=randbuf(towers0);
        gpufhe::DeviceCiphertext A(gpufhe::Scheme::CKKS,n,towers0), B(gpufhe::Scheme::CKKS,n,towers0);
        std::vector<uint64_t> z((size_t)n*towers0,0);
        A.upload(ha,z); B.upload(hb,z);
        static cudaStream_t stream = nullptr;
        static uint64_t *scratch=nullptr,*drop=nullptr;
        if(!stream){ cudaStreamCreate(&stream);
            cudaMalloc(&scratch,(size_t)n*sizeof(uint64_t));
            cudaMalloc(&drop,(size_t)n*sizeof(uint64_t)); }
        cudaDeviceSynchronize();
        auto t0=clk::now();
        gpufhe::mul_resident(A,B,T,stream);
        gpufhe::rescale_resident(A,T,S1[0],S2[0],stream,scratch,drop);
        gpufhe::mul_resident(A,B,T,stream);
        gpufhe::rescale_resident(A,T,S1[1],S2[1],stream,scratch,drop);
        cudaStreamSynchronize(stream);
        std::vector<uint64_t> o0,o1; A.to_host(o0,o1);
        double dt=std::chrono::duration<double,std::milli>(clk::now()-t0).count();
        return dt;
    };

    auto run_cpu=[&](){
        // OpenFHE CPU equivalent: two eval-form multiplies + two drops.
        using Poly=DCRTPoly::PolyType;
        auto mk=[&](){ std::vector<Poly> tv; tv.reserve(towers0);
            for(uint32_t t=0;t<towers0;++t){ auto pp=ep->GetParams()[t];
                NativeVector v(n,pp->GetModulus());
                for(uint32_t i=0;i<n;++i) v[i]=NativeInteger(rng()%moduli[t]);
                Poly p(pp,Format::EVALUATION,true); p.SetValues(std::move(v),Format::EVALUATION);
                tv.push_back(std::move(p)); }
            return DCRTPoly(tv); };
        // Drop trailing towers of a fresh operand to match A's current level.
        auto drop_to=[&](DCRTPoly p, uint32_t target){
            while(p.GetNumOfElements()>target) p.DropLastElement();
            return p; };
        // Build operands BEFORE timing (GPU path uploads before t0 too).
        DCRTPoly A=mk(), B=mk(), B2pre=mk();
        auto t0=clk::now();
        A=A*B;                                   // full towers
        A.DropLastElementAndScale(cryptoParams->GetQlQlInvModqlDivqlModq(0),
                                  cryptoParams->GetqlInvModq(0));
        DCRTPoly B2=drop_to(B2pre, A.GetNumOfElements());  // match rescaled A
        A=A*B2;
        A.DropLastElementAndScale(cryptoParams->GetQlQlInvModqlDivqlModq(1),
                                  cryptoParams->GetqlInvModq(1));
        double dt=std::chrono::duration<double,std::milli>(clk::now()-t0).count();
        return dt;
    };

    for(int i=0;i<warmup;++i){ run_gpu(); run_cpu(); }
    for(int i=0;i<reps;++i){ tg.push_back(run_gpu()); tc.push_back(run_cpu()); }

    std::vector<double> d(reps);
    for(int i=0;i<reps;++i) d[i]=tg[i]-tc[i];
    double md=median(d); auto [lo,hi]=bootstrap(d);

    std::cout<<"resident multiply-rescale chain, n="<<n<<" towers="<<towers0<<"\n";
    std::cout<<"  gpu  median = "<<median(tg)<<" ms\n";
    std::cout<<"  cpu  median = "<<median(tc)<<" ms\n";
    std::cout<<"  gpu-cpu median = "<<md<<" ms  95% CI ["<<lo<<", "<<hi<<"]\n";
    if(hi<0)      std::cout<<"  => GPU FASTER (CI below zero)\n";
    else if(lo>0) std::cout<<"  => GPU SLOWER (CI above zero)\n";
    else          std::cout<<"  => no significant difference (CI brackets zero)\n";
    return 0;
}
