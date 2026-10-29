// GOD PROMPT "CKKS ENGINE: RESOLVE POLYEVAL CORRECTNESS" round, Priority A.
//
// test_keyswitch.cpp validates keyswitch_core_resident bit-exact against
// OpenFHE -- but ONLY using OpenFHE-SOURCED KeySwitchConstants at n=32768.
// test_polyeval.cpp builds its own K NATIVELY (compute_keyswitch_constants +
// evalkeygen_host, numPart=2, n=1024) -- that combination has never been
// independently validated against keyswitch_core_resident, matching the
// EXACT same "OpenFHE-sourced vs native" gap pattern that caused the
// rescale confusion in an earlier round (resolved there by direct
// differential testing, not by assuming either path is correct).
//
// A diagnostic inside test_polyeval.cpp this round found what LOOKED like a
// huge (~26% of q0) discrepancy between the raw tensor-product residue and
// the post-keyswitch residue -- but that comparison went through
// decrypt_host's INTT+embedding-sum machinery, which has its own scale-
// dependent centering assumptions that are hard to rule out as a confound.
// This test sidesteps all of that: it checks keyswitch_core_resident's core
// algebraic property DIRECTLY in eval domain, per tower, per coefficient,
// with no INTT/decode/encode/centering involved at all. evalkeygen_host's
// own comment confirms it builds a RELIN eval key, i.e. sOld=s^2 encrypted
// under s (src/evalkeygen.cpp: "relin wrapper: sOld = s^2") -- so the
// correct identity to check is
//   ba0[x] + ba1[x]*s[x]  =?=  aTowers[x]*s[x]*s[x]   (mod q[tower])
// (an EARLIER version of this test incorrectly compared against
// aTowers[x]*s[x], one power of s short -- that comparison showed ~50% of
// q everywhere, i.e. pure noise, which in hindsight was the expected
// signature of comparing against the WRONG target, not a defect) --
// up to keyswitch's own (expected-small) noise. Centering the per-position
// difference mod q[t] and reporting its magnitude directly answers whether
// that noise is in a sane range or wildly large, with no ambiguity from
// values exceeding a single tower's representable range.
#include "keyswitch.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <string>

namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
static uint64_t am(uint64_t a,uint64_t b,uint64_t q){uint64_t s=a+b;return s>=q?s-q:s;}

int main(int argc, char** argv){
    // Mirror test_polyeval's exact first-level parameters: n=1024, tw=8
    // (1x60-bit + 7x50-bit), sizeP=2, numPart=2 by default; numPart
    // overridable via argv[1] to isolate whether a failure is specific to
    // multi-part digit decomposition (numPart>1) or present even with a
    // single part (numPart=1, no decomposition at all).
    const uint32_t n=1024, sizeQ=8, sizeP=2; const uint64_t ns=1;
    const uint32_t numPart = (argc>1) ? (uint32_t)std::stoul(argv[1]) : 2;
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

    std::mt19937_64 rng(555);
    size_t T=(size_t)sizeQ*n;
    std::vector<uint64_t> aTowers(T);
    for(uint32_t t=0;t<sizeQ;++t){
        std::uniform_int_distribution<uint64_t> d(0,mod[t]-1);
        for(uint32_t k=0;k<n;++k) aTowers[(size_t)t*n+k]=d(rng);
    }

    auto R=gpufhe::keyswitch_core_resident(aTowers,K);

    int total_bad=0; double worst_frac_of_q=0;
    for(uint32_t t=0;t<sizeQ;++t){
        uint64_t q=mod[t];
        uint64_t worst_diff=0; uint32_t worst_idx=0;
        for(uint32_t k=0;k<n;++k){
            size_t x=(size_t)t*n+k;
            uint64_t lhs=am(R.ba0[x], mm(R.ba1[x],KPqp.s[x],q), q);
            uint64_t s2=mm(KPqp.s[x],KPqp.s[x],q);
            uint64_t rhs=mm(aTowers[x], s2, q);
            uint64_t diff = (lhs>=rhs) ? (lhs-rhs) : (rhs-lhs);
            uint64_t centered_diff = (diff > q-diff) ? (q-diff) : diff;   // distance mod q
            if(centered_diff>worst_diff){ worst_diff=centered_diff; worst_idx=k; }
        }
        double frac=(double)worst_diff/(double)q;
        std::cout<<"tower="<<t<<" q="<<q<<" worst|ba0+ba1*s - a*s| (centered, mod q)="<<worst_diff
                  <<" ("<<(frac*100.0)<<"% of q) at k="<<worst_idx<<"\n";
        if(frac>worst_frac_of_q) worst_frac_of_q=frac;
        if(frac>0.01) ++total_bad;   // >1% of q is NOT plausible keyswitch noise for sane params
    }
    std::cout<<"---\nworst fraction of q across all towers: "<<(worst_frac_of_q*100.0)<<"%\n";
    std::cout<<(total_bad==0?"[PASS]":"[FAIL]")<<" test_keyswitch_core_native_diff (ba0+ba1*s vs a*s, native K, numPart=2, n=1024)\n";
    return total_bad==0?0:1;
}
