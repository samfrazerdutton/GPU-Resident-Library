// GOD PROMPT "CKKS CORRECTNESS BREAKTHROUGH" round, Priority 1 Step 3
// (confirming experiment). test_rescale_scalar_diff_n1024 and
// test_rescale_vector_diff_n1024 and test_rescale_double_call_n1024 (this
// round, all PASSED with 0 mismatches) jointly prove rescale_resident_raw's
// RNS arithmetic is bit-exact at n=1024 with this exact 8-tower chain, for
// constant data, independent random per-coefficient data, and the
// production back-to-back-scratch-buffer-reuse call pattern alike.
//
// That leaves test_rescale_n1024.cpp's ORIGINAL failure (maxerr=24602,
// using Delta=2^50) unexplained by a library defect. Hypothesis: that
// test's OWN choice of Delta=2^50 while the rescaled-away modulus qLast is
// a ~55-bit prime (~2^55) is not a sound CKKS parameter choice for
// observing a SINGLE rescale in isolation -- real CKKS rescale is designed
// to divide the scale by approximately the modulus being dropped, so a
// modulus roughly 32x (2^5) LARGER than Delta crushes the ~0.3*Delta signal
// down to ~0.3*Delta/qLast =~ 0.009, an integer that rounds to 0, while the
// rescale's own +-0.5-unit rounding noise (inherent to ANY correct
// rescale, not a bug) gets amplified by 1/outScale = qLast/Delta =~32x on
// decode -- easily producing exactly the kind of "garbage-looking" result
// observed, with NO arithmetic defect required.
//
// This test is a byte-for-byte copy of test_rescale_n1024.cpp except for
// ONE line: Delta=2^55 instead of 2^50, matching qLast's bit-size the way
// real CKKS parameter selection requires (rescale modulus approx. equal to
// the scale being removed). If the hypothesis above is correct, this
// should PASS (maxerr small) using the exact same rescale_resident_raw +
// native_rescale_consts call sequence that test_rescale_n1024.cpp uses.
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
}
using cd=std::complex<double>;

int main(){
    const uint32_t n=1024, S=n/2, sizeQ=8, sizeP=2; const uint64_t ns=1;
    std::vector<uint64_t> mod;
    gpufhe::native_primes(mod,1,60,n,{});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids,sizeQ-1,55,n,mod); for(auto m:mids) mod.push_back(m); }
    std::vector<uint64_t> modP; gpufhe::native_primes(modP,sizeP,60,n,mod);
    std::vector<uint64_t> root(sizeQ); for(uint32_t i=0;i<sizeQ;++i) root[i]=gpufhe::native_root(n,mod[i]);
    std::vector<uint64_t> rootP(sizeP); for(uint32_t j=0;j<sizeP;++j) rootP[j]=gpufhe::native_root(n,modP[j]);

    const double Delta=std::pow(2.0,55);   // ONLY DIFFERENCE from test_rescale_n1024.cpp: matched to qLast's ~55-bit size.
    std::vector<cd> z(S, cd{0.3,0});
    std::vector<int64_t> m; gpufhe::encode_host(m,z,n,Delta);
    auto KP=gpufhe::keygen_host(n,mod,root,ns,3.19,101);
    std::vector<uint64_t> c0,c1;
    gpufhe::encrypt_host(c0,c1,m,KP.pkA,KP.pkB,n,mod,root,ns,3.19,303);

    std::vector<int64_t> dec_pre;
    {
        gpufhe::decrypt_host(dec_pre,c0,c1,KP.s,n,mod,root);
        std::vector<cd> y; gpufhe::decode_host(y,dec_pre,n,Delta);
        std::cout<<"before rescale: y[0]="<<y[0].real()<<" (expect ~0.3)\n"<<std::flush;
    }

    std::vector<uint64_t> modQP(mod); for(auto p:modP) modQP.push_back(p);
    std::vector<uint64_t> rootQP(root); for(auto rp:rootP) rootQP.push_back(rp);
    auto KPqp=gpufhe::keygen_host(n,modQP,rootQP,ns,3.19,101);
    std::vector<uint64_t> PModq_QP(sizeQ+sizeP);
    for(uint32_t t=0;t<sizeQ+sizeP;++t){uint64_t q=modQP[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=(uint64_t)(((unsigned __int128)P*(modQP[sizeQ+j]%q))%q);PModq_QP[t]=P;}
    gpufhe::KeySwitchConstants K; K.n=n;
    gpufhe::compute_keyswitch_constants(K,mod,modP,2);
    for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rootP[j]);}
    gpufhe::evalkeygen_host(K,KPqp.s,KPqp.pkA,KPqp.pkB,PModq_QP,modQP,rootQP,ns,3.19,202);
    auto C=gpufhe::ks_context_create(K);
    std::vector<uint64_t> s1,s2; gpufhe::native_rescale_consts(s1,s2,mod,sizeQ-1);

    // Diagnostic: decrypt tower 7 (qLast) ALONE, pre-rescale, to check it
    // agrees with dec_pre (recovered via tower 0 alone) -- i.e. that the
    // real encrypted ciphertext's RNS representation is internally
    // consistent across towers for these small-magnitude coefficients,
    // before blaming rescale_resident_raw for any post-rescale mismatch.
    {
        std::vector<uint64_t> c0_t7(c0.begin()+(size_t)(sizeQ-1)*n, c0.begin()+(size_t)sizeQ*n);
        std::vector<uint64_t> c1_t7(c1.begin()+(size_t)(sizeQ-1)*n, c1.begin()+(size_t)sizeQ*n);
        std::vector<uint64_t> s_t7(KP.s.begin()+(size_t)(sizeQ-1)*n, KP.s.begin()+(size_t)sizeQ*n);
        std::vector<uint64_t> mod_t7={mod[sizeQ-1]}, root_t7={root[sizeQ-1]};
        std::vector<int64_t> dec_pre_t7;
        gpufhe::decrypt_host(dec_pre_t7,c0_t7,c1_t7,s_t7,n,mod_t7,root_t7);
        int ndiag=0;
        for(uint32_t k=0;k<n && ndiag<10;++k){
            if(dec_pre_t7[k]!=dec_pre[k]){
                std::cout<<"[diag-t7] k="<<k<<" dec_pre(tower0)="<<dec_pre[k]<<" dec_pre_t7(tower7)="<<dec_pre_t7[k]<<" diff="<<(dec_pre_t7[k]-dec_pre[k])<<"\n";
                ++ndiag;
            }
        }
        if(ndiag==0) std::cout<<"[diag-t7] first 10 positions: tower-0 and tower-7 pre-rescale recovery AGREE exactly\n";
    }

    size_t T=(size_t)sizeQ*n;
    uint64_t *d0,*d1,*scr,*drp;
    cudaMalloc(&d0,T*8); cudaMalloc(&d1,T*8); cudaMalloc(&scr,(size_t)n*8); cudaMalloc(&drp,(size_t)n*8);
    cudaMemcpy(d0,c0.data(),T*8,cudaMemcpyHostToDevice); cudaMemcpy(d1,c1.data(),T*8,cudaMemcpyHostToDevice);
    gpufhe::rescale_resident_raw(d0,sizeQ,C,s1,s2,scr,drp,0);
    gpufhe::rescale_resident_raw(d1,sizeQ,C,s1,s2,scr,drp,0);
    cudaDeviceSynchronize();

    uint32_t nt=sizeQ-1; size_t T2=(size_t)nt*n;
    std::vector<uint64_t> h0(T),h1(T);
    cudaMemcpy(h0.data(),d0,T*8,cudaMemcpyDeviceToHost); cudaMemcpy(h1.data(),d1,T*8,cudaMemcpyDeviceToHost);
    std::vector<uint64_t> oc0(h0.begin(),h0.begin()+T2), oc1(h1.begin(),h1.begin()+T2);
    double outScale = Delta/(double)mod[sizeQ-1];

    std::vector<uint64_t> modl(mod.begin(),mod.begin()+nt), rootl(root.begin(),root.begin()+nt);
    auto KPr=gpufhe::keygen_host(n,modl,rootl,ns,3.19,101);
    std::vector<int64_t> dec2; gpufhe::decrypt_host(dec2,oc0,oc1,KPr.s,n,modl,rootl);

    // Diagnostic: manually compute expected post-rescale tower-0 coefficient
    // (round(dec_pre[k]/qLast), centered convention) directly from the REAL
    // pre-rescale decrypted coefficients, and compare to decrypt_host's
    // ACTUAL post-rescale tower-0 result, at several positions -- this is
    // the same reference method as test_rescale_vector_diff_n1024 but
    // applied to REAL encrypted data instead of synthetic random residues.
    {
        int64_t qLastI=(int64_t)mod[sizeQ-1];
        int ndiag=0;
        for(uint32_t k=0;k<n && ndiag<10;++k){
            double ratio=(double)dec_pre[k]/(double)qLastI;
            int64_t expected=(int64_t)llround(ratio);
            if(expected!=dec2[k]){
                std::cout<<"[diag] k="<<k<<" dec_pre="<<dec_pre[k]<<" qLast="<<qLastI
                          <<" expected_round="<<expected<<" actual_dec2="<<dec2[k]
                          <<" diff="<<(dec2[k]-expected)<<"\n";
                ++ndiag;
            }
        }
        if(ndiag==0) std::cout<<"[diag] first 10 positions: dec2[k] matches round(dec_pre[k]/qLast) exactly\n";
        std::cout<<"[diag] dec_pre[0]="<<dec_pre[0]<<" dec2[0]="<<dec2[0]<<" round(dec_pre[0]/qLast)="<<(int64_t)llround((double)dec_pre[0]/(double)qLastI)<<"\n";
    }

    std::vector<cd> y2; gpufhe::decode_host(y2,dec2,n,outScale);
    double maxerr=0; for(uint32_t i=0;i<S;++i) maxerr=std::max(maxerr,std::abs(y2[i].real()-0.3));
    std::cout<<"after ONE rescale (n=1024, Delta=2^55 matched to qLast): y[0]="<<y2[0].real()
              <<" maxerr_vs_0.3="<<maxerr<<" outScale="<<outScale<<"\n";
    std::cout<<(maxerr<1e-2?"[PASS]":"[FAIL]")<<" test_rescale_n1024_matched_scale\n";

    cudaFree(d0);cudaFree(d1);cudaFree(scr);cudaFree(drp);
    gpufhe::ks_context_destroy(C);
    return maxerr<1e-2?0:1;
}
