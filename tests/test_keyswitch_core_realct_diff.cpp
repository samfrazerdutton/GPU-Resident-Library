// GOD PROMPT "CKKS ENGINE: RESOLVE POLYEVAL CORRECTNESS" round, Priority A.
//
// test_keyswitch_core_native_diff.cpp (this round) found that
// ba0+ba1*s =?= a*s^2 fails by ~50% of q for a PURELY RANDOM "a" (uniform
// mod q) -- but then an algebraic check appended to test_keyswitch.cpp
// showed the SAME ~50%-of-q failure even using OpenFHE's own bit-exact-
// proven-correct ba0/ba1 output for the SAME kind of random "a". That
// rules out a native-vs-OpenFHE discrepancy, but leaves open whether
// "ba0+ba1*s ~ a*s^2" is simply the wrong property to expect for
// ARBITRARY full-range random "a" in the first place (RNS hybrid
// keyswitch's approximate CRT basis extension may only be accurate for
// inputs with realistic/bounded magnitude, like an actual ciphertext
// component -- not uniform random data spanning the full modulus).
//
// This test redoes the SAME eval-domain check, but with a=t2 taken from a
// REAL tensor product of two FRESH, valid encryptions (exactly what
// test_polyeval's mulCt actually feeds keyswitch_core_resident), checked
// directly in eval domain against a*s^2 -- no decode/INTT/centering
// ambiguity at all, closing the gap left by the random-input tests.
#include "keyswitch.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <complex>
#include <cmath>

namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
static uint64_t am(uint64_t a,uint64_t b,uint64_t q){uint64_t s=a+b;return s>=q?s-q:s;}

int main(){
    const uint32_t n=1024, sizeQ=8, sizeP=2, numPart=2; const uint64_t ns=1;
    std::vector<uint64_t> mod,modP;
    gpufhe::native_primes(mod,1,60,n,{});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids,sizeQ-1,50,n,mod); for(auto m:mids) mod.push_back(m); }
    gpufhe::native_primes(modP,sizeP,60,n,mod);
    std::vector<uint64_t> modQP=mod; for(auto p:modP) modQP.push_back(p);
    std::vector<uint64_t> root(sizeQ),rootQP;
    for(uint32_t i=0;i<sizeQ;++i) root[i]=gpufhe::native_root(n,mod[i]);
    rootQP=root; for(auto p:modP) rootQP.push_back(gpufhe::native_root(n,p));

    auto KPqp=gpufhe::keygen_host(n,modQP,rootQP,ns,3.19,101);
    std::vector<uint64_t> PModq_QP(sizeQ+sizeP);
    for(uint32_t t=0;t<sizeQ+sizeP;++t){uint64_t q=modQP[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modQP[sizeQ+j]%q,q);PModq_QP[t]=P;}
    gpufhe::KeySwitchConstants K; K.n=n;
    gpufhe::compute_keyswitch_constants(K,mod,modP,numPart);
    for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modQP[sizeQ+j]);K.rootValList.push_back(rootQP[sizeQ+j]);}
    gpufhe::evalkeygen_host(K,KPqp.s,KPqp.pkA,KPqp.pkB,PModq_QP,modQP,rootQP,ns,3.19,202);

    // Build TWO real, fresh, valid encryptions (Q-only, sizeQ towers),
    // exactly mirroring test_polyeval's encAt, and compute their REAL
    // tensor-product t2 = A.c1*B.c1 (eval domain, elementwise).
    const uint32_t S=n/2; const double Delta=std::pow(2.0,50);
    std::vector<std::complex<double>> xa(S,{0.6,0}), xb(S,{1.0,0});
    std::vector<int64_t> ma,mb;
    gpufhe::encode_host(ma,xa,n,Delta); gpufhe::encode_host(mb,xb,n,Delta);
    std::vector<uint64_t> A0,A1,B0,B1;
    gpufhe::encrypt_host(A0,A1,ma,KPqp.pkA,KPqp.pkB,n,mod,root,ns,3.19,11);
    gpufhe::encrypt_host(B0,B1,mb,KPqp.pkA,KPqp.pkB,n,mod,root,ns,3.19,12);
    size_t T=(size_t)sizeQ*n;
    std::vector<uint64_t> t2(T);
    for(uint32_t t=0;t<sizeQ;++t){uint64_t q=mod[t];
        for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k; t2[x]=mm(A1[x],B1[x],q);}}

    auto R=gpufhe::keyswitch_core_resident(t2,K);

    int total_bad=0; double worst_frac_of_q=0;
    for(uint32_t t=0;t<sizeQ;++t){
        uint64_t q=mod[t];
        uint64_t worst_diff=0; uint32_t worst_idx=0;
        for(uint32_t k=0;k<n;++k){
            size_t x=(size_t)t*n+k;
            uint64_t sv=KPqp.s[x];
            uint64_t s2=mm(sv,sv,q);
            uint64_t lhs=am(R.ba0[x], mm(R.ba1[x],sv,q), q);
            uint64_t rhs=mm(t2[x], s2, q);
            uint64_t diff=(lhs>=rhs)?(lhs-rhs):(rhs-lhs);
            uint64_t cdiff=(diff>q-diff)?(q-diff):diff;
            if(cdiff>worst_diff){ worst_diff=cdiff; worst_idx=k; }
        }
        double frac=(double)worst_diff/(double)q;
        std::cout<<"tower="<<t<<" q="<<q<<" worst|ba0+ba1*s - t2*s^2| (centered)="<<worst_diff
                  <<" ("<<(frac*100.0)<<"% of q) at k="<<worst_idx<<"\n";
        if(frac>worst_frac_of_q) worst_frac_of_q=frac;
        if(frac>0.01) ++total_bad;
    }
    std::cout<<"---\nworst fraction of q across all towers: "<<(worst_frac_of_q*100.0)<<"%\n";
    std::cout<<(total_bad==0?"[PASS]":"[FAIL]")<<" test_keyswitch_core_realct_diff (real tensor-product t2, native K, numPart=2, n=1024)\n";
    return total_bad==0?0:1;
}
