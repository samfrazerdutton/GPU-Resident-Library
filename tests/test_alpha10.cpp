// Isolate the alpha=10 failure: plaintext-constant multiply + rescale ONLY.
// No keyswitch anywhere. EvalMod's "affine" step is exactly this, and it blows
// up at alpha=10 while INPUT is clean -- so either this path breaks, or the
// fault is upstream in the chain rather than in the operation.
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
void pt_to_eval_host(std::vector<uint64_t>&, const std::vector<int64_t>&, uint32_t, uint32_t,
                     const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void ct_mul_pt_host(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<uint64_t>&,
                    uint32_t, uint32_t, const std::vector<uint64_t>&);
void set_secret_hamming_weight(uint32_t);
}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
using cd=std::complex<double>;

int main(int argc,char**argv){
    const uint32_t ALPHA = (argc>1)? std::stoul(argv[1]) : 10;
    const uint32_t SZP   = (argc>2)? std::stoul(argv[2]) : 9;
    const uint32_t n=1024,S=n/2,sizeQ=32; const uint64_t ns=1;
    auto npFor=[&](uint32_t tw){ return (tw+ALPHA-1)/ALPHA; };
    gpufhe::set_secret_hamming_weight(64);
    std::vector<uint64_t> mod,modP;
    gpufhe::native_primes(mod,1,60,n,{});
    { std::vector<uint64_t> mid; gpufhe::native_primes(mid,sizeQ-1,50,n,mod); for(auto m:mid)mod.push_back(m); }
    gpufhe::native_primes(modP,SZP,60,n,mod);
    std::vector<uint64_t> root(sizeQ);
    for(uint32_t i=0;i<sizeQ;++i) root[i]=gpufhe::native_root(n,mod[i]);
    std::vector<uint64_t> rootP(SZP);
    for(uint32_t j=0;j<SZP;++j) rootP[j]=gpufhe::native_root(n,modP[j]);

    const double D=std::pow(2.0,50), C=0.0245;   // C = EvalMod's affine coefficient
    std::vector<cd> z(S);
    for(uint32_t i=0;i<S;++i) z[i]={8.0*std::sin(0.01*i),0};   // |slot| ~ 16 like the real input

    std::cout<<"alpha="<<ALPHA<<" sizeP="<<SZP<<": ct_mul_pt(const) + rescale, NO keyswitch\n";
    uint32_t bad=0;
    for(uint32_t tw : {30u,29u,28u,27u,21u,20u,19u,11u,10u}){
        std::vector<uint64_t> ml(mod.begin(),mod.begin()+tw),rl(root.begin(),root.begin()+tw);
        auto KP=gpufhe::keygen_host(n,ml,rl,ns,3.19,101);
        std::vector<int64_t> mz; gpufhe::encode_host(mz,z,n,D);
        std::vector<uint64_t> c0,c1;
        gpufhe::encrypt_host(c0,c1,mz,KP.pkA,KP.pkB,n,ml,rl,ns,3.19,303);
        std::vector<cd> cc(S,cd{C,0}); std::vector<int64_t> mc;
        gpufhe::encode_host(mc,cc,n,D);
        std::vector<uint64_t> ce; gpufhe::pt_to_eval_host(ce,mc,tw,n,ml,rl);
        gpufhe::ct_mul_pt_host(c0,c1,ce,tw,n,ml);
        // rescale with a context built at this level and this alpha
        gpufhe::KeySwitchConstants K; K.n=n;
        gpufhe::compute_keyswitch_constants(K,ml,modP,npFor(tw));
        for(uint32_t i=0;i<tw;++i){K.rootModList.push_back(ml[i]);K.rootValList.push_back(rl[i]);}
        for(uint32_t j=0;j<SZP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rootP[j]);}
        K.av.assign(K.numPart,{});K.bv.assign(K.numPart,{});K.evalKeyTowers=tw+SZP;
        for(uint32_t p=0;p<K.numPart;++p){K.av[p].assign((size_t)(tw+SZP)*n,0);K.bv[p].assign((size_t)(tw+SZP)*n,0);}
        auto Cx=gpufhe::ks_context_create(K);
        std::vector<uint64_t> s1,s2; gpufhe::native_rescale_consts(s1,s2,ml,tw-1);
        size_t T=(size_t)tw*n; uint64_t *d0,*d1,*sc,*dp;
        cudaMalloc(&d0,T*8);cudaMalloc(&d1,T*8);cudaMalloc(&sc,(size_t)n*8);cudaMalloc(&dp,(size_t)n*8);
        cudaMemcpy(d0,c0.data(),T*8,cudaMemcpyHostToDevice);cudaMemcpy(d1,c1.data(),T*8,cudaMemcpyHostToDevice);
        gpufhe::rescale_resident_raw(d0,tw,Cx,s1,s2,sc,dp,0);
        gpufhe::rescale_resident_raw(d1,tw,Cx,s1,s2,sc,dp,0);
        cudaDeviceSynchronize();
        std::vector<uint64_t> h0(T),h1(T);
        cudaMemcpy(h0.data(),d0,T*8,cudaMemcpyDeviceToHost);cudaMemcpy(h1.data(),d1,T*8,cudaMemcpyDeviceToHost);
        cudaFree(d0);cudaFree(d1);cudaFree(sc);cudaFree(dp);gpufhe::ks_context_destroy(Cx);
        uint32_t nt=tw-1; size_t T2=(size_t)nt*n;
        std::vector<uint64_t> o0(h0.begin(),h0.begin()+T2),o1(h1.begin(),h1.begin()+T2);
        std::vector<uint64_t> mf(ml.begin(),ml.begin()+nt),rf(rl.begin(),rl.begin()+nt);
        auto KR=gpufhe::keygen_host(n,mf,rf,ns,3.19,101);
        std::vector<int64_t> dec; gpufhe::decrypt_host(dec,o0,o1,KR.s,n,mf,rf);
        std::vector<cd> y; gpufhe::decode_host(y,dec,n,D*D/(double)ml[tw-1]);
        double e=0; for(uint32_t i=0;i<S;++i) e=std::max(e,std::abs(y[i].real()-C*z[i].real()));
        bool ok=e<1e-6; if(!ok)++bad;
        std::cout<<"  tw="<<tw<<" parts="<<npFor(tw)<<"  err="<<e<<(ok?"   OK":"   <-- BAD")<<"\n";
    }
    std::cout<<(bad?"[FAIL] ":"[PASS] ")<<bad<<" bad levels\n";
    return bad?1:0;
}
