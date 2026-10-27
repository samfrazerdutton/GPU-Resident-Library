// Objective C precursor: measure the REAL (not extrapolated) per-key host
// memory footprint at n=131072 before attempting anything expensive, per
// the governing directive's "estimate footprint before allocating" step.
// Does NOT run a bootstrap -- builds the actual n=131072 modulus chain
// (same shape as the validated n=8192 config: sizeQ=30, sizeP=2, same
// per-tower bit sizes, log2(QP)=1775 unchanged -- see
// research/SECURITY_ASSESSMENT.md) and generates ONE real rotation key,
// measuring getrusage RSS before/after.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <sys/resource.h>
namespace gpufhe {
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
uint64_t native_root(uint32_t, uint64_t);
void automorphism_eval_host(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t,
                            const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void set_secret_hamming_weight(uint32_t);
}
static long rssMB(){ struct rusage ru; getrusage(RUSAGE_SELF,&ru); return ru.ru_maxrss/1024; }
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
using clk=std::chrono::steady_clock;
static double EL(clk::time_point a){return std::chrono::duration<double>(clk::now()-a).count();}

int main(){
    const uint32_t n=131072, tw=30, sizeP=2, numPart=(tw+1)/2; const uint64_t ns=1;
    gpufhe::set_secret_hamming_weight(64);
    std::cout<<"RSS before any allocation: "<<rssMB()<<" MB\n";

    auto t0=clk::now();
    std::vector<uint64_t> mod,modP;
    gpufhe::native_primes(mod,1,60,n,{});
    { std::vector<uint64_t> md; gpufhe::native_primes(md,tw-1,55,n,mod); for(auto m:md)mod.push_back(m); }
    gpufhe::native_primes(modP,sizeP,60,n,mod);
    std::vector<uint64_t> mq(mod),rq;
    std::vector<uint64_t> root(tw); for(uint32_t i=0;i<tw;++i)root[i]=gpufhe::native_root(n,mod[i]);
    rq=root; for(auto p:modP){ mq.push_back(p); rq.push_back(gpufhe::native_root(n,p)); }
    std::cout<<"Modulus chain + NTT roots generated in "<<EL(t0)<<"s ("<<mod.size()<<" Q towers, "
              <<modP.size()<<" P towers)\n";
    std::cout<<"RSS after modulus/root generation: "<<rssMB()<<" MB\n";

    const uint32_t sizeQlP=tw+sizeP;
    auto t1=clk::now();
    auto KPqp=gpufhe::keygen_host(n,mq,rq,ns,3.19,101);
    std::cout<<"Base QP keypair generated in "<<EL(t1)<<"s\n";
    long rssBeforeKey=rssMB();
    std::cout<<"RSS before any rotation key: "<<rssBeforeKey<<" MB\n";

    std::vector<uint64_t> PM(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=mq[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modP[j]%q,q);PM[t]=P;}

    auto t2=clk::now();
    std::vector<uint64_t> sA=KPqp.s;
    gpufhe::automorphism_eval_host(sA,sizeQlP,n,5,mq,rq);
    gpufhe::KeySwitchConstants K; K.n=n;
    gpufhe::compute_keyswitch_constants(K,mod,modP,numPart);
    for(uint32_t i=0;i<tw;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rq[tw+j]);}
    gpufhe::evalkeygen_host_sold(K,KPqp.s,sA,KPqp.pkA,KPqp.pkB,PM,mq,rq,1,3.19,900);
    double keygenTime=EL(t2);
    long rssAfterKey=rssMB();
    std::cout<<"ONE rotation key (evalkeygen_host_sold) generated in "<<keygenTime<<"s\n";
    std::cout<<"RSS after one rotation key: "<<rssAfterKey<<" MB\n";
    std::cout<<"REAL measured per-key delta: "<<(rssAfterKey-rssBeforeKey)<<" MB (vs the "
              <<"extrapolated ~1000MB figure used throughout research/MEMORY_FEASIBILITY.md)\n";

    // Exact structural size (numPart * sizeQlP * n * 8 bytes * 2 for av+bv)
    size_t exact = (size_t)2 * K.numPart * sizeQlP * n * 8;
    std::cout<<"Exact structural size (2*numPart*sizeQlP*n*8B): "<<(exact/1024/1024)<<" MB "
              <<"(numPart="<<K.numPart<<", sizeQlP="<<sizeQlP<<")\n";
    std::cout<<"Extrapolated total for 76 such keys: "<<(exact*76/1024/1024/1024)<<" GB (structural only, "
              <<"excludes allocator/measurement overhead -- compare to the real RSS delta above)\n";
    std::cout<<"Single-key keygen time: "<<keygenTime<<"s (compare to n=8192's measured ~0.64s/key)\n";
    return 0;
}
