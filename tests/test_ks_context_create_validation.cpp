// GOD PROMPT "CKKS CORRECTNESS BREAKTHROUGH" round, Priority 6
// (library-finishing work). `ks_context_create` unconditionally reads
// K.av[p]/K.bv[p] for every part -- `compute_keyswitch_constants` alone
// does NOT populate them (only `evalkeygen_host`/`evalkeygen_host_sold`
// does). Passing a K built without that call previously segfaulted with
// no diagnostic (confirmed via gdb at least twice this project, including
// in a brand-new test written this same round). `src/keyswitch_resident.cpp`
// now validates K.av.size()==K.numPart && K.bv.size()==K.numPart at the top
// of ks_context_create and throws std::invalid_argument instead. This test
// confirms: (1) the invalid pattern now throws a catchable, clear exception
// rather than crashing, and (2) a correctly-populated K still succeeds,
// proving zero behavior change for every valid caller.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>

namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
}

int main(){
    const uint32_t n=1024, sizeQ=4, sizeP=2; const uint64_t ns=1;
    std::vector<uint64_t> mod; gpufhe::native_primes(mod,sizeQ,55,n,{});
    std::vector<uint64_t> modP; gpufhe::native_primes(modP,sizeP,60,n,mod);
    std::vector<uint64_t> root(sizeQ); for(uint32_t i=0;i<sizeQ;++i) root[i]=gpufhe::native_root(n,mod[i]);
    std::vector<uint64_t> rootP(sizeP); for(uint32_t j=0;j<sizeP;++j) rootP[j]=gpufhe::native_root(n,modP[j]);

    int failures=0;

    // Case 1: K.av/K.bv left empty (the exact historical footgun) must
    // throw, not segfault.
    {
        gpufhe::KeySwitchConstants K; K.n=n;
        gpufhe::compute_keyswitch_constants(K,mod,modP,2);
        for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
        for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rootP[j]);}
        bool threw=false;
        try{ gpufhe::ks_context_create(K); }
        catch(const std::invalid_argument& e){ threw=true; std::cout<<"[info] caught expected exception: "<<e.what()<<"\n"; }
        if(!threw){ std::cout<<"[FAIL] empty av/bv did not throw -- validation missing or broken\n"; ++failures; }
        else std::cout<<"[PASS] empty av/bv correctly throws std::invalid_argument\n";
    }

    // Case 2: a correctly-populated K (via evalkeygen_host) must still
    // succeed exactly as before -- zero behavior change for valid callers.
    {
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
        bool threw=false;
        try{
            auto C=gpufhe::ks_context_create(K);
            if(C.sizeQl!=sizeQ){ std::cout<<"[FAIL] valid K produced unexpected context shape\n"; ++failures; }
            gpufhe::ks_context_destroy(C);
        }
        catch(const std::exception& e){ threw=true; std::cout<<"[FAIL] valid, correctly-populated K unexpectedly threw: "<<e.what()<<"\n"; }
        if(!threw) std::cout<<"[PASS] correctly-populated K still succeeds (zero behavior change)\n";
        else ++failures;
    }

    std::cout<<"---\n"<<(failures==0?"[PASS]":"[FAIL]")<<" test_ks_context_create_validation\n";
    return failures==0?0:1;
}
