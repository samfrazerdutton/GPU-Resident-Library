// Regression test for a real segfault found via gdb while extending
// rotation hoisting to multiple C2S stages (research/NEXT_SESSION.md):
// ks_context_create(K) unconditionally uploads EVERY part's K.av[part]/
// K.bv[part] to device, regardless of what the caller intends to do with
// the resulting DeviceKSContext. A hand-built "shape-only" KeySwitchConstants
// (correct sizeQl/numPart/root-tables, but av/bv left as default-constructed
// EMPTY vectors, because the caller only needed it for ks_decompose_only's
// shape fields) crashes ks_context_create with a SIGSEGV when it indexes
// those empty vectors -- confirmed via `gdb --batch -ex run -ex "bt full"`,
// crash squarely inside gpufhe::ks_context_create.
//
// The fix is procedural, not a code change to ks_context_create (which is
// existing, proven, untouched): never pass a shape-only key to it. Use the
// SAME level-view construction the trusted rotation-key path already uses
// (copy a FULL key's real, non-empty av/bv across while re-deriving the
// shape fields for a smaller tower count -- see test_boot.cpp's atLevel()).
// This test exercises exactly that pattern directly, at a REDUCED tower
// count (sizeQ-2, simulating two prior-stage rescales), which is the
// specific scenario that crashed.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keyswitch_hoist.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void set_secret_hamming_weight(uint32_t);
void decrypt_host(std::vector<int64_t>&, const std::vector<uint64_t>&,
                  const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                  uint32_t, const std::vector<uint64_t>&, const std::vector<uint64_t>&);
}

// Mirrors test_boot.cpp's atLevel(): re-shape a FULL key down to `tw`
// towers, copying its REAL (non-empty) av/bv across -- never leave av/bv
// empty on anything handed to ks_context_create.
static gpufhe::KeySwitchConstants atLevel(const gpufhe::KeySwitchConstants& KF,
    uint32_t tw, uint32_t n, const std::vector<uint64_t>& mod, const std::vector<uint64_t>& root,
    const std::vector<uint64_t>& modP, const std::vector<uint64_t>& rootP,
    uint32_t sizeQ)
{
    auto npFor=[](uint32_t t)->uint32_t{ return t<=2?1u:(t+1)/2; };
    gpufhe::KeySwitchConstants K; K.n=n;
    gpufhe::compute_keyswitch_constants(K,std::vector<uint64_t>(mod.begin(),mod.begin()+tw),modP,npFor(tw));
    for(uint32_t i=0;i<tw;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<modP.size();++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rootP[j]);}
    K.fullQ=sizeQ; K.evalKeyTowers=KF.evalKeyTowers;
    K.av.assign(K.numPart,{}); K.bv.assign(K.numPart,{});
    for(uint32_t p=0;p<K.numPart;++p){K.av[p]=KF.av[p]; K.bv[p]=KF.bv[p];}
    return K;
}

int main(){
    const uint32_t n=8192, sizeQ=30, sizeP=2, reducedTw=sizeQ-2; const uint64_t ns=1;
    gpufhe::set_secret_hamming_weight(64);
    std::vector<uint64_t> mod,modP;
    gpufhe::native_primes(mod,1,60,n,{});
    { std::vector<uint64_t> md; gpufhe::native_primes(md,sizeQ-1,55,n,mod); for(auto m:md)mod.push_back(m); }
    gpufhe::native_primes(modP,sizeP,60,n,mod);
    std::vector<uint64_t> modQP=mod; for(auto p:modP)modQP.push_back(p);
    std::vector<uint64_t> root(sizeQ),rootQP;
    for(uint32_t i=0;i<sizeQ;++i)root[i]=gpufhe::native_root(n,mod[i]);
    rootQP=root; for(auto p:modP)rootQP.push_back(gpufhe::native_root(n,p));
    std::vector<uint64_t> rootP(rootQP.begin()+sizeQ,rootQP.end());

    auto KPqp=gpufhe::keygen_host(n,modQP,rootQP,ns,3.19,101);
    auto KPq =gpufhe::keygen_host(n,mod,root,ns,3.19,101);
    std::vector<uint64_t> PModq(sizeQ+sizeP);
    auto mmu=[](uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);};
    for(uint32_t t=0;t<sizeQ+sizeP;++t){uint64_t q=modQP[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mmu(P,modQP[sizeQ+j]%q,q);PModq[t]=P;}

    // FULL relin-style key, real non-empty av/bv (same role as test_boot.cpp's KREL).
    auto npFor=[](uint32_t t)->uint32_t{ return t<=2?1u:(t+1)/2; };
    gpufhe::KeySwitchConstants KREL; KREL.n=n;
    gpufhe::compute_keyswitch_constants(KREL,mod,modP,npFor(sizeQ));
    for(uint32_t i=0;i<sizeQ;++i){KREL.rootModList.push_back(mod[i]);KREL.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){KREL.rootModList.push_back(modP[j]);KREL.rootValList.push_back(rootQP[sizeQ+j]);}
    gpufhe::evalkeygen_host(KREL,KPqp.s,KPqp.pkA,KPqp.pkB,PModq,modQP,rootQP,ns,3.19,202);

    // THE REGRESSION: build a level-reduced view at reducedTw towers (NOT
    // the full sizeQ) and hand it to ks_context_create + ks_decompose_only.
    // Before the fix, doing this with a hand-rolled shape-only key (no
    // av/bv) segfaulted inside ks_context_create; atLevel's real-av/bv
    // copy must not.
    auto Klevel = atLevel(KREL, reducedTw, n, mod, root, modP, rootP, sizeQ);
    auto C = gpufhe::ks_context_create(Klevel);
    auto D = gpufhe::hoist_digits_create(C);

    std::vector<uint64_t> c0,c1;
    std::vector<int64_t> mz(n,0);
    std::vector<uint64_t> redMod(mod.begin(),mod.begin()+reducedTw), redRoot(root.begin(),root.begin()+reducedTw);
    gpufhe::encrypt_host(c0,c1,mz,KPq.pkA,KPq.pkB,n,redMod,redRoot,ns,3.19,303);
    // encrypt_host above used KPq (sizeQ-sized pk) at reducedTw moduli slice --
    // fine since KPq.s/pkA/pkB are defined per-tower and slicing the first
    // reducedTw towers is a valid sub-ciphertext for this decompose-only check.
    size_t T=(size_t)reducedTw*n, B=T*8;
    uint64_t *d_c1; cudaMalloc(&d_c1,B); cudaMemcpy(d_c1,c1.data(),B,cudaMemcpyHostToDevice);

    gpufhe::ks_decompose_only(d_c1,C,D,0);   // this line crashed before the fix
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();

    gpufhe::hoist_digits_destroy(D);
    gpufhe::ks_context_destroy(C);
    cudaFree(d_c1);

    bool ok = (err == cudaSuccess);
    std::cout<<(ok?"[PASS]":"[FAIL]")<<" level-reduced ks_context_create/ks_decompose_only did not crash "
              <<"(reducedTw="<<reducedTw<<" vs full sizeQ="<<sizeQ<<"), cuda err="<<cudaGetErrorString(err)<<"\n";
    return ok?0:1;
}
