// Batched throughput: does resident FHE win on THROUGHPUT where it lost on
// latency? N independent chains, GPU runs them concurrently on N streams (its
// natural advantage: one chain at large n may underfill the SMs, N chains
// don't), CPU runs N chains through OpenFHE's own threading. Construction is
// OUTSIDE the timer on both sides (the bias that faked the earlier win).
//
// Also reports the overlap ratio: batched-GPU-time vs N * single-chain-time.
// Near-linear speedup => streams overlapped. batched ~= N*single => the device
// was already saturated by one chain and batching added no concurrency.
//
// Args: [N] [ring]   defaults N=16 ring=32768.

#include "openfhe.h"
#include "device_ciphertext.h"
#include "ckks_ops.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

using namespace lbcrypto;
using clk = std::chrono::steady_clock;

static double median(std::vector<double> v){
    if(v.empty())return 0; std::sort(v.begin(),v.end());
    size_t n=v.size(); return n%2? v[n/2] : 0.5*(v[n/2-1]+v[n/2]); }
static double quantile(std::vector<double> v,double q){
    if(v.empty())return 0; std::sort(v.begin(),v.end());
    double pos=q*(v.size()-1); size_t lo=(size_t)pos,hi=std::min(lo+1,v.size()-1);
    return v[lo]+(pos-lo)*(v[hi]-v[lo]); }
static std::pair<double,double> boot(const std::vector<double>&d,int it=4000){
    if(d.size()<3)return{0,0}; std::mt19937 rng(7);
    std::uniform_int_distribution<size_t> pick(0,d.size()-1);
    std::vector<double> m; m.reserve(it); std::vector<double> s(d.size());
    for(int k=0;k<it;++k){ for(size_t i=0;i<d.size();++i)s[i]=d[pick(rng)];
        m.push_back(median(s)); } return {quantile(m,0.025),quantile(m,0.975)}; }

int main(int argc, char** argv) {
    uint32_t N    = (argc>1)? (uint32_t)std::stoul(argv[1]) : 16;
    uint32_t ring = (argc>2)? (uint32_t)std::stoul(argv[2]) : 32768;

    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(4);
    params.SetScalingModSize(50);
    params.SetRingDim(ring);
    params.SetBatchSize(ring/2);
    CryptoContext<DCRTPoly> cc;
    try { cc = GenCryptoContext(params); }
    catch(const std::exception& e){ std::cout<<"skip ring "<<ring<<": "<<e.what()<<"\n"; return 0; }
    cc->Enable(PKE); cc->Enable(LEVELEDSHE);
    auto kp = cc->KeyGen();

    auto ep = cc->GetCryptoParameters()->GetElementParams();
    const uint32_t n = ep->GetRingDimension();
    const uint32_t tw0 = (uint32_t)ep->GetParams().size();
    std::vector<uint64_t> moduli(tw0);
    for(uint32_t t=0;t<tw0;++t) moduli[t]=ep->GetParams()[t]->GetModulus().ConvertToInt();
    auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());

    gpufhe::DeviceTables T;
    gpufhe::build_device_tables(T, n, moduli);

    std::vector<std::vector<uint64_t>> S1(2),S2(2);
    for(int lv=0;lv<2;++lv){ const auto&q1=cp->GetqlInvModq(lv); const auto&q2=cp->GetQlQlInvModqlDivqlModq(lv);
        uint32_t sv=tw0-1-lv; S1[lv].resize(sv); S2[lv].resize(sv);
        for(uint32_t t=0;t<sv;++t){ S1[lv][t]=q1[t].ConvertToInt(); S2[lv][t]=q2[t].ConvertToInt(); } }

    std::mt19937_64 rng(2026);
    auto randbuf=[&](){ std::vector<uint64_t> v((size_t)n*tw0);
        for(uint32_t t=0;t<tw0;++t) for(uint32_t i=0;i<n;++i) v[t*n+i]=rng()%moduli[t]; return v; };

    // Per-chain GPU resources (all allocated OUTSIDE the timer).
    std::vector<cudaStream_t> streams(N);
    std::vector<uint64_t*> scratch(N), drop(N);
    std::vector<gpufhe::DeviceCiphertext> A, B;
    A.reserve(N); B.reserve(N);
    std::vector<uint64_t> z((size_t)n*tw0,0);
    for(uint32_t c=0;c<N;++c){
        cudaStreamCreate(&streams[c]);
        cudaMalloc(&scratch[c],(size_t)n*sizeof(uint64_t));
        cudaMalloc(&drop[c],(size_t)n*sizeof(uint64_t));
        A.emplace_back(gpufhe::Scheme::CKKS,n,tw0);
        B.emplace_back(gpufhe::Scheme::CKKS,n,tw0);
    }

    auto reset_upload=[&](){ for(uint32_t c=0;c<N;++c){ A[c].~DeviceCiphertext();
        new (&A[c]) gpufhe::DeviceCiphertext(gpufhe::Scheme::CKKS,n,tw0);
        B[c].~DeviceCiphertext();
        new (&B[c]) gpufhe::DeviceCiphertext(gpufhe::Scheme::CKKS,n,tw0);
        auto ha=randbuf(), hb=randbuf(); A[c].upload(ha,z); B[c].upload(hb,z); }
        cudaDeviceSynchronize(); };

    auto run_gpu_batch=[&](){
        auto t0=clk::now();
        for(uint32_t c=0;c<N;++c){
            gpufhe::mul_resident(A[c],B[c],T,streams[c]);
            gpufhe::rescale_resident(A[c],T,S1[0],S2[0],streams[c],scratch[c],drop[c]);
            gpufhe::mul_resident(A[c],B[c],T,streams[c]);
            gpufhe::rescale_resident(A[c],T,S1[1],S2[1],streams[c],scratch[c],drop[c]);
        }
        cudaDeviceSynchronize();
        return std::chrono::duration<double,std::milli>(clk::now()-t0).count();
    };

    // CPU batch: N chains through OpenFHE (its own OpenMP within each op).
    using Poly=DCRTPoly::PolyType;
    auto mk=[&](){ std::vector<Poly> tv; tv.reserve(tw0);
        for(uint32_t t=0;t<tw0;++t){ auto pp=ep->GetParams()[t];
            NativeVector v(n,pp->GetModulus());
            for(uint32_t i=0;i<n;++i) v[i]=NativeInteger(rng()%moduli[t]);
            Poly p(pp,Format::EVALUATION,true); p.SetValues(std::move(v),Format::EVALUATION);
            tv.push_back(std::move(p)); } return DCRTPoly(tv); };
    auto drop_to=[&](DCRTPoly p,uint32_t tgt){ while(p.GetNumOfElements()>tgt) p.DropLastElement(); return p; };

    auto run_cpu_batch=[&](std::vector<std::array<DCRTPoly,3>>& ops){
        auto t0=clk::now();
        for(uint32_t c=0;c<N;++c){
            DCRTPoly A=ops[c][0];
            A=A*ops[c][1];
            A.DropLastElementAndScale(cp->GetQlQlInvModqlDivqlModq(0),cp->GetqlInvModq(0));
            DCRTPoly B2=drop_to(ops[c][2],A.GetNumOfElements());
            A=A*B2;
            A.DropLastElementAndScale(cp->GetQlQlInvModqlDivqlModq(1),cp->GetqlInvModq(1));
        }
        return std::chrono::duration<double,std::milli>(clk::now()-t0).count();
    };

    const int reps=30, warmup=5;
    // Single-chain GPU time for the overlap ratio (N=1 path).
    reset_upload();
    for(int i=0;i<warmup;++i) run_gpu_batch();
    std::vector<double> tg, tc;

    for(int i=0;i<reps;++i){
        reset_upload();
        tg.push_back(run_gpu_batch());
        std::vector<std::array<DCRTPoly,3>> ops(N);
        for(uint32_t c=0;c<N;++c) ops[c]={mk(),mk(),mk()};
        tc.push_back(run_cpu_batch(ops));
    }

    std::vector<double> d(reps);
    for(int i=0;i<reps;++i) d[i]=tg[i]-tc[i];
    double md=median(d); auto [lo,hi]=boot(d);
    double g=median(tg), c=median(tc);

    std::cout<<"BATCH N="<<N<<" ring="<<n<<" towers="<<tw0<<"\n";
    std::cout<<"  gpu batch  = "<<g<<" ms  ("<<g/N<<" ms/chain)\n";
    std::cout<<"  cpu batch  = "<<c<<" ms  ("<<c/N<<" ms/chain)\n";
    std::cout<<"  gpu-cpu median = "<<md<<" ms  95% CI ["<<lo<<", "<<hi<<"]\n";
    if(hi<0)      std::cout<<"  => GPU FASTER on throughput\n";
    else if(lo>0) std::cout<<"  => GPU SLOWER on throughput\n";
    else          std::cout<<"  => no significant difference\n";

    for(uint32_t c2=0;c2<N;++c2){ cudaStreamDestroy(streams[c2]);
        cudaFree(scratch[c2]); cudaFree(drop[c2]); }
    return 0;
}
