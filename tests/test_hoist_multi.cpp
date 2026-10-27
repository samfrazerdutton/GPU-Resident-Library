// Standalone correctness test for the multi-rotation decompose-sharing
// mechanism (research/NEXT_SESSION.md priority queue item 1, "next step").
// Two checks:
//
//  1. Mechanical equivalence: ks_decompose_only + ks_accumulate_from_digits
//     (the new, split path) must produce BIT-EXACT the same res0/res1 as
//     the existing, untouched ks_accumulate_qp (the fused path), for the
//     SAME key -- proving the split introduces zero change by itself.
//
//  2. Multi-rotation sharing: decompose c1 ONCE, then accumulate against
//     SEVERAL different rotation-index keys (same hoisted key-generation
//     convention proven in test_hoist_single.cpp: permute pkA_QP/pkB_QP by
//     k^-1, keep payload = unpermuted base secret), fold c0*P, apply the
//     per-index automorphism, moddown -- compare the DECRYPTED plaintext
//     against the existing, proven rotate_ct_resident for each index (same
//     noise-tolerant comparison as test_hoist_single.cpp, for the same
//     reason: decompose/automorphism route data through different
//     rounding paths than the textbook ground truth, so only the
//     decrypted message, not the raw ciphertext, is the right invariant).
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keyswitch_hoist.h"
#include "keygen.h"
#include "stage_acc.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
namespace gpufhe {
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void automorphism_eval_host(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t,
                            const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void automorphism_eval_fast_device(uint64_t*, uint32_t, uint32_t, uint32_t, uint64_t*, cudaStream_t);
void rotate_ct_resident(std::vector<uint64_t>&, std::vector<uint64_t>&, uint32_t,
                        const KeySwitchConstants&, uint32_t, uint32_t,
                        const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void set_secret_hamming_weight(uint32_t);
void decrypt_host(std::vector<int64_t>&, const std::vector<uint64_t>&,
                  const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                  uint32_t, const std::vector<uint64_t>&, const std::vector<uint64_t>&);
}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
using cd=std::complex<double>;

static uint64_t modinv(uint64_t a, uint64_t m){
    int64_t old_r=(int64_t)a, r=(int64_t)m, old_s=1, s=0;
    while(r!=0){ int64_t q=old_r/r;
        int64_t tr=old_r-q*r; old_r=r; r=tr;
        int64_t ts=old_s-q*s; old_s=s; s=ts; }
    int64_t res=old_s % (int64_t)m; if(res<0) res+=(int64_t)m;
    return (uint64_t)res;
}

int main(){
    const uint32_t n=8192, tw=30, S=n/2, sizeP=2, numPart=(tw+1)/2; const uint64_t ns=1;
    gpufhe::set_secret_hamming_weight(64);
    std::vector<uint64_t> mod,modP;
    gpufhe::native_primes(mod,1,60,n,{});
    { std::vector<uint64_t> md; gpufhe::native_primes(md,tw-1,55,n,mod); for(auto m:md)mod.push_back(m); }
    gpufhe::native_primes(modP,sizeP,60,n,mod);
    std::vector<uint64_t> mq(mod),rq;
    std::vector<uint64_t> root(tw); for(uint32_t i=0;i<tw;++i)root[i]=gpufhe::native_root(n,mod[i]);
    rq=root; for(auto p:modP){ mq.push_back(p); rq.push_back(gpufhe::native_root(n,p)); }
    const uint32_t sizeQlP=tw+sizeP;

    auto KPqp=gpufhe::keygen_host(n,mq,rq,ns,3.19,101);
    auto KPq =gpufhe::keygen_host(n,mod,root,ns,3.19,101);
    std::vector<uint64_t> PM(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=mq[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modP[j]%q,q);PM[t]=P;}

    std::vector<cd> z(S); for(uint32_t i=0;i<S;++i) z[i]={0.3*std::sin(0.01*i),0.1*std::cos(0.02*i)};
    std::vector<int64_t> mz; gpufhe::encode_host(mz,z,n,std::pow(2.0,55));
    std::vector<uint64_t> c0,c1;
    gpufhe::encrypt_host(c0,c1,mz,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,303);

    // ---- Check 1: mechanical equivalence vs. the existing, untouched
    // ks_accumulate_qp, using a SINGLE arbitrary key (k=7's hoisted key,
    // chosen arbitrarily -- any key works, this isn't testing the key
    // convention, just that decompose+accumulate split == fused). ----
    {
        uint32_t k=7; uint64_t kinv=modinv(k,2*n);
        std::vector<uint64_t> pkA_h=KPqp.pkA, pkB_h=KPqp.pkB;
        gpufhe::automorphism_eval_host(pkA_h,sizeQlP,n,(uint32_t)kinv,mq,rq);
        gpufhe::automorphism_eval_host(pkB_h,sizeQlP,n,(uint32_t)kinv,mq,rq);
        gpufhe::KeySwitchConstants Kh; Kh.n=n;
        gpufhe::compute_keyswitch_constants(Kh,mod,modP,numPart);
        for(uint32_t i=0;i<tw;++i){Kh.rootModList.push_back(mod[i]);Kh.rootValList.push_back(root[i]);}
        for(uint32_t j=0;j<sizeP;++j){Kh.rootModList.push_back(modP[j]);Kh.rootValList.push_back(rq[tw+j]);}
        gpufhe::evalkeygen_host_sold(Kh,KPqp.s,KPqp.s,pkA_h,pkB_h,PM,mq,rq,1,3.19,777);

        auto C=gpufhe::ks_context_create(Kh);
        auto W=gpufhe::ks_work_create(C);
        size_t T=(size_t)tw*n, B=T*8, TQP=(size_t)sizeQlP*n, BQP=TQP*8;
        uint64_t *d_c1; cudaMalloc(&d_c1,B); cudaMemcpy(d_c1,c1.data(),B,cudaMemcpyHostToDevice);

        gpufhe::ks_accumulate_qp(d_c1,C,W,0);
        cudaDeviceSynchronize();
        std::vector<uint64_t> fused_res0(TQP), fused_res1(TQP);
        cudaMemcpy(fused_res0.data(),W.d_res0,BQP,cudaMemcpyDeviceToHost);
        cudaMemcpy(fused_res1.data(),W.d_res1,BQP,cudaMemcpyDeviceToHost);

        auto D=gpufhe::hoist_digits_create(C);
        gpufhe::ks_decompose_only(d_c1,C,D,0);
        gpufhe::ks_accumulate_from_digits(d_c1,D,C,W,0);
        cudaDeviceSynchronize();
        std::vector<uint64_t> split_res0(TQP), split_res1(TQP);
        cudaMemcpy(split_res0.data(),W.d_res0,BQP,cudaMemcpyDeviceToHost);
        cudaMemcpy(split_res1.data(),W.d_res1,BQP,cudaMemcpyDeviceToHost);

        size_t bad=0;
        for(size_t i=0;i<TQP;++i){ if(fused_res0[i]!=split_res0[i]) ++bad; if(fused_res1[i]!=split_res1[i]) ++bad; }
        std::cout<<"[check 1] decompose+accumulate split vs fused ks_accumulate_qp: "
                  <<bad<<" / "<<2*TQP<<" mismatches "<<(bad==0?"[PASS bit-exact]":"[FAIL]")<<"\n";
        if(bad){ std::cout<<"[FAIL] check 1 must be bit-exact -- aborting\n"; return 1; }

        gpufhe::hoist_digits_destroy(D);
        cudaFree(d_c1);
        gpufhe::ks_work_destroy(W); gpufhe::ks_context_destroy(C);
    }

    // ---- Check 2: decompose ONCE, accumulate against several different
    // rotation indices' keys, compare decrypted plaintext against the
    // proven rotate_ct_resident for each. ----
    const uint32_t M=2*n;
    std::vector<uint32_t> ks = {5u,7u,11u,13u,99u,M-1u};
    bool allOk=true; size_t ncases=0, nfail=0;

    // Build all N contexts up front (mirrors how a real bootstrap stage
    // would already have every rotation key cached in KROT), then
    // decompose c1 against the FIRST one and reuse that HoistedDigits
    // for every other context -- this is the actual claim under test:
    // decompose's result doesn't depend on which key you decomposed it
    // "against" (ks_decompose_only ignores a context's av/bv entirely),
    // only on the shared chain shape.
    std::vector<gpufhe::KeySwitchConstants> Khoist(ks.size());
    for(size_t ki=0; ki<ks.size(); ++ki){
        uint32_t k=ks[ki]; uint64_t kinv=modinv(k,M);
        std::vector<uint64_t> pkA_h=KPqp.pkA, pkB_h=KPqp.pkB;
        gpufhe::automorphism_eval_host(pkA_h,sizeQlP,n,(uint32_t)kinv,mq,rq);
        gpufhe::automorphism_eval_host(pkB_h,sizeQlP,n,(uint32_t)kinv,mq,rq);
        Khoist[ki].n=n;
        gpufhe::compute_keyswitch_constants(Khoist[ki],mod,modP,numPart);
        for(uint32_t i=0;i<tw;++i){Khoist[ki].rootModList.push_back(mod[i]);Khoist[ki].rootValList.push_back(root[i]);}
        for(uint32_t j=0;j<sizeP;++j){Khoist[ki].rootModList.push_back(modP[j]);Khoist[ki].rootValList.push_back(rq[tw+j]);}
        gpufhe::evalkeygen_host_sold(Khoist[ki],KPqp.s,KPqp.s,pkA_h,pkB_h,PM,mq,rq,1,3.19,900+(uint64_t)k);
    }

    size_t T=(size_t)tw*n, B=T*8, TQP=(size_t)sizeQlP*n, BQP=TQP*8;
    uint64_t *d_c0,*d_c1,*d_scratch_qp;
    cudaMalloc(&d_c0,B); cudaMalloc(&d_c1,B); cudaMalloc(&d_scratch_qp,BQP);
    cudaMemcpy(d_c0,c0.data(),B,cudaMemcpyHostToDevice);
    cudaMemcpy(d_c1,c1.data(),B,cudaMemcpyHostToDevice);

    auto C0=gpufhe::ks_context_create(Khoist[0]);
    auto D=gpufhe::hoist_digits_create(C0);
    gpufhe::ks_decompose_only(d_c1,C0,D,0);   // *** decompose exactly ONCE ***
    cudaDeviceSynchronize();
    gpufhe::ks_context_destroy(C0);   // only needed D's shape constants; av/bv unused for decompose

    for(size_t ki=0; ki<ks.size(); ++ki){
        uint32_t k=ks[ki];
        // Ground truth must use a TEXTBOOK key (psi_k(s)), not the hoisted
        // one -- rebuild it exactly like test_hoist_single.cpp.
        std::vector<uint64_t> sA_truth=KPqp.s;
        gpufhe::automorphism_eval_host(sA_truth,sizeQlP,n,k,mq,rq);
        gpufhe::KeySwitchConstants Ktruth; Ktruth.n=n;
        gpufhe::compute_keyswitch_constants(Ktruth,mod,modP,numPart);
        for(uint32_t i=0;i<tw;++i){Ktruth.rootModList.push_back(mod[i]);Ktruth.rootValList.push_back(root[i]);}
        for(uint32_t j=0;j<sizeP;++j){Ktruth.rootModList.push_back(modP[j]);Ktruth.rootValList.push_back(rq[tw+j]);}
        gpufhe::evalkeygen_host_sold(Ktruth,KPqp.s,sA_truth,KPqp.pkA,KPqp.pkB,PM,mq,rq,1,3.19,900+(uint64_t)k);
        std::vector<uint64_t> gc0=c0, gc1=c1;
        gpufhe::rotate_ct_resident(gc0,gc1,k,Ktruth,tw,n,mod,root);

        auto Ck=gpufhe::ks_context_create(Khoist[ki]);
        auto Wk=gpufhe::ks_work_create(Ck);
        gpufhe::ks_accumulate_from_digits(d_c1,D,Ck,Wk,0);   // *** cheap, reused digits ***
        for(uint32_t t=0;t<tw;++t)
            LaunchScaledAddInto(Wk.d_res0+(size_t)t*n, d_c0+(size_t)t*n, PM[t], mod[t], n, 0);
        gpufhe::automorphism_eval_fast_device(Wk.d_res0,sizeQlP,n,k,d_scratch_qp,0);
        gpufhe::automorphism_eval_fast_device(Wk.d_res1,sizeQlP,n,k,d_scratch_qp,0);
        uint64_t *d_ba0,*d_ba1; cudaMalloc(&d_ba0,B); cudaMalloc(&d_ba1,B);
        gpufhe::ks_moddown(Ck,Wk,Wk.d_res0,d_ba0,0);
        gpufhe::ks_moddown(Ck,Wk,Wk.d_res1,d_ba1,0);
        cudaDeviceSynchronize();
        std::vector<uint64_t> hc0(T),hc1(T);
        cudaMemcpy(hc0.data(),d_ba0,B,cudaMemcpyDeviceToHost);
        cudaMemcpy(hc1.data(),d_ba1,B,cudaMemcpyDeviceToHost);
        cudaFree(d_ba0); cudaFree(d_ba1);
        gpufhe::ks_work_destroy(Wk); gpufhe::ks_context_destroy(Ck);

        std::vector<int64_t> mg, mh;
        gpufhe::decrypt_host(mg, gc0, gc1, KPq.s, n, mod, root);
        gpufhe::decrypt_host(mh, hc0, hc1, KPq.s, n, mod, root);
        int64_t maxAbs=0, maxDiff=0;
        for(size_t i=0;i<mg.size();++i){
            int64_t a=(mg[i]<0?-mg[i]:mg[i]); if(a>maxAbs) maxAbs=a;
            int64_t d=mg[i]-mh[i]; if(d<0) d=-d; if(d>maxDiff) maxDiff=d;
        }
        bool ok = maxDiff < (maxAbs>>20) + (1LL<<30);
        std::cout<<"[check 2] k="<<k<<" (shared-decompose) maxAbs="<<maxAbs
                  <<" maxDiff="<<maxDiff<<(ok?" [close enough]":" [FAIL]")<<"\n";
        ++ncases; if(!ok) ++nfail;
        allOk = allOk && ok;
    }
    gpufhe::hoist_digits_destroy(D);
    cudaFree(d_c0); cudaFree(d_c1); cudaFree(d_scratch_qp);

    std::cout<<(allOk?"[PASS] ":"[FAIL] ")<<"multi-rotation decompose-sharing, "
              <<(ncases-nfail)<<"/"<<ncases<<" indices off ONE shared decompose\n";
    return allOk?0:1;
}
