// Directive "GOD PROMPT -- CKKS ENGINE": Priority 1 investigation of
// test_polyeval's maxerr=2376.98 failure. Instrumented tracing (see
// tests/test_polyeval.cpp's temporary [chk] diagnostics, and
// research/EXPERIMENT_LEDGER.md) isolated the corruption to SPECIFICALLY
// the rescale_resident_raw step: the raw tensor product and the
// post-keyswitch (pre-rescale) result both decrypt correctly (~0, matching
// the expected value), but the result AFTER rescale_resident_raw is wildly
// wrong (maxerr in the thousands). Both native_rescale_consts (proven only
// at n=8192, tests/test_boot.cpp) and rescale_resident_raw (proven only at
// n=32768, tests/test_rescale_resident.cpp, which sources its s1/s2
// constants from OpenFHE, NOT native_rescale_consts) have NEVER been
// cross-validated together at test_polyeval's n=1024. This isolates
// EXACTLY that combination, independent of all Chebyshev/relin complexity:
// encrypt a known constant at n=1024 with test_polyeval's exact modulus
// chain (one 60-bit + seven 50-bit towers), rescale ONCE, check the
// result.
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
    std::cerr<<"[step] q0 done, mod.size()="<<mod.size()<<std::flush<<"\n";
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids,sizeQ-1,55,n,mod); for(auto m:mids) mod.push_back(m); }   // 55-bit (test_boot.cpp's proven bit size) instead of 50-bit, SAME n=1024 -- isolates n vs. tower-bit-size as the variable
    std::cerr<<"[step] mids done, mod.size()="<<mod.size()<<std::flush<<"\n";
    std::vector<uint64_t> modP; gpufhe::native_primes(modP,sizeP,60,n,mod);
    std::cerr<<"[step] modP done, modP.size()="<<modP.size()<<std::flush<<"\n";
    std::vector<uint64_t> root(sizeQ); for(uint32_t i=0;i<sizeQ;++i) root[i]=gpufhe::native_root(n,mod[i]);
    std::cerr<<"[step] root done"<<std::flush<<"\n";
    std::vector<uint64_t> rootP(sizeP); for(uint32_t j=0;j<sizeP;++j) rootP[j]=gpufhe::native_root(n,modP[j]);
    std::cerr<<"[step] rootP done"<<std::flush<<"\n";

    const double Delta=std::pow(2.0,50);
    std::vector<cd> z(S, cd{0.3,0});   // known constant, same order of magnitude as test_polyeval's x values
    std::vector<int64_t> m; gpufhe::encode_host(m,z,n,Delta);
    std::cerr<<"[step] encode done, m.size()="<<m.size()<<std::flush<<"\n";
    auto KP=gpufhe::keygen_host(n,mod,root,ns,3.19,101);
    std::cerr<<"[step] keygen done, KP.s.size()="<<KP.s.size()<<std::flush<<"\n";
    std::vector<uint64_t> c0,c1;
    gpufhe::encrypt_host(c0,c1,m,KP.pkA,KP.pkB,n,mod,root,ns,3.19,303);
    std::cerr<<"[step] encrypt done, c0.size()="<<c0.size()<<std::flush<<"\n";

    // Sanity: fresh ciphertext decrypts correctly BEFORE any rescale.
    {
        std::vector<int64_t> dec;
        std::cerr<<"[step] about to decrypt_host, KP.s.size()="<<KP.s.size()<<" mod.size()="<<mod.size()<<" root.size()="<<root.size()<<std::flush<<"\n";
        gpufhe::decrypt_host(dec,c0,c1,KP.s,n,mod,root);
        std::cerr<<"[step] decrypt_host done, dec.size()="<<dec.size()<<std::flush<<"\n";
        std::vector<cd> y; gpufhe::decode_host(y,dec,n,Delta);
        std::cerr<<"[step] decode_host done, y.size()="<<y.size()<<std::flush<<"\n";
        std::cout<<"before rescale: y[0]="<<y[0].real()<<" (expect ~0.3)\n"<<std::flush;
    }

    // Rescale ONCE: drop tower sizeQ-1=7, survivors [0,7).
    // ks_context_create unconditionally reads K.av[p]/K.bv[p] for every part
    // (established, documented pattern from an earlier round this session,
    // the exact cause of a prior ks_context_create segfault) -- so K needs a
    // REAL eval key populated via evalkeygen_host, even though this test only
    // exercises rescale_resident_raw (which never reads av/bv itself). Build
    // a full QP keypair and a trivial relin key, matching test_polyeval's
    // OWN Kl construction exactly, to isolate ONLY the n=1024 question.
    std::vector<uint64_t> modQP(mod); for(auto p:modP) modQP.push_back(p);
    std::vector<uint64_t> rootQP(root); for(auto rp:rootP) rootQP.push_back(rp);
    auto KPqp=gpufhe::keygen_host(n,modQP,rootQP,ns,3.19,101);
    std::vector<uint64_t> PModq_QP(sizeQ+sizeP);
    for(uint32_t t=0;t<sizeQ+sizeP;++t){uint64_t q=modQP[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=(uint64_t)(((unsigned __int128)P*(modQP[sizeQ+j]%q))%q);PModq_QP[t]=P;}
    gpufhe::KeySwitchConstants K; K.n=n;
    gpufhe::compute_keyswitch_constants(K,mod,modP,2);   // numPart=2, matching test_polyeval's own setting
    std::cerr<<"[step] compute_keyswitch_constants done, K.sizeQl="<<K.sizeQl<<" K.fullQ="<<K.fullQ<<" K.numPart="<<K.numPart<<std::flush<<"\n";
    for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rootP[j]);}
    std::cerr<<"[step] rootModList built, size="<<K.rootModList.size()<<std::flush<<"\n";
    gpufhe::evalkeygen_host(K,KPqp.s,KPqp.pkA,KPqp.pkB,PModq_QP,modQP,rootQP,ns,3.19,202);
    std::cerr<<"[step] evalkeygen_host done, K.av.size()="<<K.av.size()<<" K.av[0].size()="<<(K.av.empty()?0:K.av[0].size())<<std::flush<<"\n";
    auto C=gpufhe::ks_context_create(K);
    std::cerr<<"[step] ks_context_create done, C.sizeQl="<<C.sizeQl<<" cudaGetLastError="<<cudaGetLastError()<<std::flush<<"\n";
    std::vector<uint64_t> s1,s2; gpufhe::native_rescale_consts(s1,s2,mod,sizeQ-1);
    std::cerr<<"[step] native_rescale_consts done, s1.size()="<<s1.size()<<" s2.size()="<<s2.size()<<std::flush<<"\n";

    size_t T=(size_t)sizeQ*n;
    uint64_t *d0,*d1,*scr,*drp;
    cudaMalloc(&d0,T*8); cudaMalloc(&d1,T*8); cudaMalloc(&scr,(size_t)n*8); cudaMalloc(&drp,(size_t)n*8);
    std::cerr<<"[step] device buffers allocated, cudaGetLastError="<<cudaGetLastError()<<std::flush<<"\n";
    cudaMemcpy(d0,c0.data(),T*8,cudaMemcpyHostToDevice); cudaMemcpy(d1,c1.data(),T*8,cudaMemcpyHostToDevice);
    std::cerr<<"[step] device upload done, cudaGetLastError="<<cudaGetLastError()<<std::flush<<"\n";
    gpufhe::rescale_resident_raw(d0,sizeQ,C,s1,s2,scr,drp,0);
    std::cerr<<"[step] rescale d0 done, cudaGetLastError="<<cudaGetLastError()<<std::flush<<"\n";
    gpufhe::rescale_resident_raw(d1,sizeQ,C,s1,s2,scr,drp,0);
    std::cerr<<"[step] rescale d1 done, cudaGetLastError="<<cudaGetLastError()<<std::flush<<"\n";
    cudaDeviceSynchronize();
    std::cerr<<"[step] sync done, cudaGetLastError="<<cudaGetLastError()<<std::flush<<"\n";

    uint32_t nt=sizeQ-1; size_t T2=(size_t)nt*n;
    std::vector<uint64_t> h0(T),h1(T);
    cudaMemcpy(h0.data(),d0,T*8,cudaMemcpyDeviceToHost); cudaMemcpy(h1.data(),d1,T*8,cudaMemcpyDeviceToHost);
    std::vector<uint64_t> oc0(h0.begin(),h0.begin()+T2), oc1(h1.begin(),h1.begin()+T2);
    double outScale = Delta/(double)mod[sizeQ-1];

    std::vector<uint64_t> modl(mod.begin(),mod.begin()+nt), rootl(root.begin(),root.begin()+nt);
    auto KPr=gpufhe::keygen_host(n,modl,rootl,ns,3.19,101);
    std::vector<int64_t> dec2; gpufhe::decrypt_host(dec2,oc0,oc1,KPr.s,n,modl,rootl);
    std::vector<cd> y2; gpufhe::decode_host(y2,dec2,n,outScale);
    double maxerr=0; for(uint32_t i=0;i<S;++i) maxerr=std::max(maxerr,std::abs(y2[i].real()-0.3));
    std::cout<<"after ONE rescale (n=1024, 60+50-bit chain): y[0]="<<y2[0].real()
              <<" maxerr_vs_0.3="<<maxerr<<"\n";
    std::cout<<(maxerr<1e-2?"[PASS]":"[FAIL]")<<" rescale_resident_raw + native_rescale_consts at n=1024\n";

    cudaFree(d0);cudaFree(d1);cudaFree(scr);cudaFree(drp);
    gpufhe::ks_context_destroy(C);
    return maxerr<1e-2?0:1;
}
