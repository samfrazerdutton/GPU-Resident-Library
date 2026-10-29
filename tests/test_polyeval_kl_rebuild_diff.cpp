// GOD PROMPT "CKKS ENGINE: RESOLVE POLYEVAL CORRECTNESS" round, Priority A.
//
// Multiple attempts to independently verify keyswitch_core_resident's
// algebraic contract (ba0+ba1*s =?= a*s^2) failed by ~50% of q -- but that
// SAME failure reproduces even on OpenFHE's own bit-exact-proven-correct
// ba0/ba1 output (checked directly in test_keyswitch.cpp, with the
// external GPU-keyswitch-shadow path also ruled out via OPENFHE_GPU/
// OPENFHE_GPU_KS=0). That means the formula itself is not understood yet,
// NOT that keyswitch_core_resident is broken -- test_keyswitch.cpp's
// existing bit-exact-vs-OpenFHE comparison remains the trusted ground
// truth for its correctness.
//
// This test asks a narrower, independently answerable question instead:
// does test_polyeval.cpp's per-mulCt-call REBUILT Kl (compute_keyswitch_
// constants + evalkeygen_host, re-run inline every call, with sQPl/pkAl/
// pkBl SLICED out of the top-level KPqp data) produce IDENTICAL ba0/ba1 to
// a top-level K built the SAME way, for the FIRST mulCt call specifically
// (tw=sizeQ, where the slicing has no effect since there's nothing to
// slice out)? If yes, Kl's rebuild is a faithful reconstruction for this
// call and is not the source of any discrepancy there. (Later calls, where
// tw<sizeQ and slicing is non-trivial, are a separate question for a
// follow-up session if this one passes clean.)
#include "keyswitch.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>

namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}

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

    // K: built exactly like test_polyeval's top-level K.
    gpufhe::KeySwitchConstants K; K.n=n;
    gpufhe::compute_keyswitch_constants(K,mod,modP,numPart);
    for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modQP[sizeQ+j]);K.rootValList.push_back(rootQP[sizeQ+j]);}
    gpufhe::evalkeygen_host(K,KPqp.s,KPqp.pkA,KPqp.pkB,PModq_QP,modQP,rootQP,ns,3.19,202);

    // Kl: built EXACTLY like test_polyeval's mulCt rebuilds it, at tw=sizeQ
    // (the FIRST mulCt call's own tw, before any level has dropped).
    const uint32_t tw=sizeQ;
    gpufhe::KeySwitchConstants Kl; Kl.n=n;
    std::vector<uint64_t> modl(mod.begin(),mod.begin()+tw);
    std::vector<uint64_t> rootl(root.begin(),root.begin()+tw);
    gpufhe::compute_keyswitch_constants(Kl,modl,modP,numPart);
    for(uint32_t i=0;i<tw;++i){Kl.rootModList.push_back(modl[i]);Kl.rootValList.push_back(rootl[i]);}
    for(uint32_t j=0;j<sizeP;++j){Kl.rootModList.push_back(modQP[sizeQ+j]);Kl.rootValList.push_back(rootQP[sizeQ+j]);}
    std::vector<uint64_t> sQPl((size_t)(tw+sizeP)*n);
    for(uint32_t t=0;t<tw;++t) std::copy(KPqp.s.begin()+(size_t)t*n, KPqp.s.begin()+(size_t)(t+1)*n, sQPl.begin()+(size_t)t*n);
    for(uint32_t j=0;j<sizeP;++j) std::copy(KPqp.s.begin()+(size_t)(sizeQ+j)*n, KPqp.s.begin()+(size_t)(sizeQ+j+1)*n, sQPl.begin()+(size_t)(tw+j)*n);
    std::vector<uint64_t> pkAl((size_t)(tw+sizeP)*n),pkBl((size_t)(tw+sizeP)*n);
    for(uint32_t t=0;t<tw;++t){std::copy(KPqp.pkA.begin()+(size_t)t*n,KPqp.pkA.begin()+(size_t)(t+1)*n,pkAl.begin()+(size_t)t*n);
        std::copy(KPqp.pkB.begin()+(size_t)t*n,KPqp.pkB.begin()+(size_t)(t+1)*n,pkBl.begin()+(size_t)t*n);}
    for(uint32_t j=0;j<sizeP;++j){std::copy(KPqp.pkA.begin()+(size_t)(sizeQ+j)*n,KPqp.pkA.begin()+(size_t)(sizeQ+j+1)*n,pkAl.begin()+(size_t)(tw+j)*n);
        std::copy(KPqp.pkB.begin()+(size_t)(sizeQ+j)*n,KPqp.pkB.begin()+(size_t)(sizeQ+j+1)*n,pkBl.begin()+(size_t)(tw+j)*n);}
    std::vector<uint64_t> modQPl(modl); for(auto p:modP)modQPl.push_back(p);
    std::vector<uint64_t> rootQPl(rootl); for(uint32_t j=0;j<sizeP;++j)rootQPl.push_back(rootQP[sizeQ+j]);
    std::vector<uint64_t> PModq_l(tw+sizeP);
    for(uint32_t t=0;t<tw+sizeP;++t){uint64_t q=modQPl[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modP[j]%q,q);PModq_l[t]=P;}
    gpufhe::evalkeygen_host(Kl,sQPl,pkAl,pkBl,PModq_l,modQPl,rootQPl,ns,3.19,202);

    // Same random input a, through BOTH K and Kl.
    std::vector<uint64_t> a((size_t)sizeQ*n);
    for(uint32_t t=0;t<sizeQ;++t)for(uint32_t k=0;k<n;++k) a[(size_t)t*n+k]=(uint64_t)(t*1000003u+k*7u+12345u)%mod[t];

    auto R_K=gpufhe::keyswitch_core_resident(a,K);
    auto R_Kl=gpufhe::keyswitch_core_resident(a,Kl);

    int bad0=0,bad1=0;
    for(size_t x=0;x<R_K.ba0.size();++x){ if(R_K.ba0[x]!=R_Kl.ba0[x]) ++bad0; if(R_K.ba1[x]!=R_Kl.ba1[x]) ++bad1; }
    std::cout<<"ba0 mismatches="<<bad0<<"/"<<R_K.ba0.size()<<" ba1 mismatches="<<bad1<<"/"<<R_K.ba1.size()<<"\n";
    std::cout<<(bad0==0&&bad1==0?"[PASS]":"[FAIL]")<<" test_polyeval_kl_rebuild_diff (K vs Kl at tw=sizeQ, same input)\n";
    return (bad0==0&&bad1==0)?0:1;
}
