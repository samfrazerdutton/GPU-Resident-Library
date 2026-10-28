// Objective C diagnosis, round 2: isolate WHY n=131072 OOMs far sooner than
// the KROT_CAP/KSC_CAP bound predicts, without paying for a full bootstrap
// (ModRaise, ciphertext buffers, O(S^2) stageDiags -- all excluded here on
// purpose). research/MEMORY_FEASIBILITY.md's "per-diagonal instrumentation"
// update got ONE real data point (RSS=3506MB at KROT=1/KSC=1, OOM within the
// next 1-2 diagonals) but could not distinguish two live hypotheses:
//   (a) rotFull()/keyAt() in tests/test_boot_n131072.cpp generate the NEW
//       key BEFORE evicting the old one (see the real code: K is built and
//       evalkeygen_host_sold() called, THEN "if(KROT.size()>=KROT_CAP)"
//       evicts) -- so the TRANSIENT peak is (cap+1) keys resident, not cap.
//       At this scale (~1085MB/key) that is a real, code-verified, concrete
//       extra ~1GB the original cap-based estimate never counted.
//   (b) glibc's allocator does not promptly return freed multi-megabyte
//       blocks to the OS, so repeated evict+regenerate cycles make RSS climb
//       roughly with the TOTAL number of distinct keys ever generated, not
//       the current cache size -- meaning no fixed cap would ever fully
//       bound RSS, only slow its growth.
// This test drives the EXACT SAME bounded-LRU eviction logic as
// tests/test_boot_n131072.cpp's rotFull (same code shape, same order of
// operations, cap=2) through N=12 DISTINCT rotation indices (forcing 10
// real evictions), printing RSS and GPU free/used memory at four points
// per call: on entry, right after evalkeygen_host_sold (before any
// eviction check -- isolates (a)), right after eviction (before the new
// entry is inserted -- isolates (b): does RSS actually drop?), and after
// the insert completes. If (a) is the whole story, RSS should plateau at
// roughly (cap+1)*~1085MB after the first few calls. If (b) also
// contributes, RSS should keep climbing roughly linearly with call index
// instead of plateauing.
#include "keyswitch.h"
#include "keyswitch_resident.h"
#include "keygen.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <map>
#include <list>
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
static void gpuMemMB(long& usedMB, long& freeMB){
    size_t freeB=0,totalB=0; cudaMemGetInfo(&freeB,&totalB);
    freeMB=(long)(freeB/1024/1024); usedMB=(long)((totalB-freeB)/1024/1024);
}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
using clk=std::chrono::steady_clock;
static double EL(clk::time_point a){return std::chrono::duration<double>(clk::now()-a).count();}

int main(){
    const uint32_t n=131072, tw=30, sizeP=2, numPart=(tw+1)/2; const uint64_t ns=1;
    constexpr size_t KROT_CAP=2;   // identical to tests/test_boot_n131072.cpp's current setting
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
    std::vector<uint64_t> PM(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=mq[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modP[j]%q,q);PM[t]=P;}

    long baseRss=rssMB(); long gUsed0,gFree0; gpuMemMB(gUsed0,gFree0);
    std::cout<<"=== baseline before any rotation key ===\n";
    std::cout<<"RSS="<<baseRss<<"MB  GPU used="<<gUsed0<<"MB free="<<gFree0<<"MB\n\n";

    std::map<uint32_t,gpufhe::KeySwitchConstants> KROT;
    std::list<uint32_t> KROT_lru;
    std::map<uint32_t,std::list<uint32_t>::iterator> KROT_lru_pos;
    uint32_t nKeygen=0, nEvictions=0;

    // Same order of operations as tests/test_boot_n131072.cpp's rotFull:
    // build+keygen the NEW key fully BEFORE checking/evicting the cap.
    auto rotFull=[&](uint32_t k)->void{
        auto t0=clk::now();
        std::vector<uint64_t> sA=KPqp.s;
        gpufhe::automorphism_eval_host(sA,sizeQlP,n,k,mq,rq);
        gpufhe::KeySwitchConstants K; K.n=n;
        gpufhe::compute_keyswitch_constants(K,mod,modP,numPart);
        for(uint32_t i=0;i<tw;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
        for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rq[tw+j]);}
        gpufhe::evalkeygen_host_sold(K,KPqp.s,sA,KPqp.pkA,KPqp.pkB,PM,mq,rq,1,3.19,900+k);
        double kgTime=EL(t0);
        ++nKeygen;

        long rssAfterKeygen=rssMB(); long gu1,gf1; gpuMemMB(gu1,gf1);
        std::cout<<"k="<<k<<" (call #"<<nKeygen<<", keygen "<<kgTime<<"s): "
                  <<"after evalkeygen_host_sold (BEFORE eviction check): RSS="<<rssAfterKeygen
                  <<"MB  GPU used="<<gu1<<"MB  [live KROT entries="<<KROT.size()<<", +1 new not yet inserted]\n";

        bool evicted=false; uint32_t victim=0;
        if(KROT.size()>=KROT_CAP){
            victim=KROT_lru.back(); KROT_lru.pop_back();
            KROT_lru_pos.erase(victim); KROT.erase(victim);
            ++nEvictions; evicted=true;
        }
        if(evicted){
            long rssAfterEvict=rssMB(); long gu2,gf2; gpuMemMB(gu2,gf2);
            std::cout<<"  after evicting k="<<victim<<" (BEFORE insert): RSS="<<rssAfterEvict
                      <<"MB  GPU used="<<gu2<<"MB  (delta vs pre-evict: "<<(rssAfterEvict-rssAfterKeygen)<<"MB)\n";
        }
        KROT.emplace(k,std::move(K));
        KROT_lru.push_front(k); KROT_lru_pos[k]=KROT_lru.begin();
        long rssAfterInsert=rssMB(); long gu3,gf3; gpuMemMB(gu3,gf3);
        std::cout<<"  after insert (steady state for this call): RSS="<<rssAfterInsert
                  <<"MB  GPU used="<<gu3<<"MB  KROT.size()="<<KROT.size()<<" nEvictions="<<nEvictions<<"\n\n";
    };

    // 12 distinct rotation indices, forcing 10 real evictions at KROT_CAP=2.
    // Indices chosen arbitrarily distinct and coprime-ish to M=2n; the actual
    // automorphism validity isn't being checked here (no bootstrap/decrypt),
    // only memory behavior of the cache+keygen mechanism itself.
    const std::vector<uint32_t> ks={3,5,7,9,11,13,15,17,19,21,23,25};
    for(uint32_t k:ks) rotFull(k);

    std::cout<<"=== phase 1 summary (KROT alone) ===\n";
    std::cout<<"nKeygen="<<nKeygen<<" nEvictions="<<nEvictions<<" final RSS="<<rssMB()<<"MB\n";
    std::cout<<"If RSS plateaus near (cap+1)*~1085MB after the first few calls, hypothesis (a) "
              <<"(keygen-before-evict transient) explains the behavior and eviction IS reclaiming "
              <<"memory. If RSS instead keeps climbing roughly with call index, allocator retention "
              <<"(hypothesis (b)) is also a real, compounding contributor.\n\n";

    // ---- Phase 2: add the KSC view-cache layer, mirroring keyAt()'s EXACT
    // logic in tests/test_boot_n131072.cpp (evict the old KSC victim FIRST,
    // THEN build the new view via atLevel(rotFull(...))). Unlike rotFull's
    // own cap-violating order, keyAt() itself evicts-before-building for
    // KSC -- but it calls rotFull() internally, which still carries
    // rotFull's own (cap+1) transient. This phase tests whether stacking
    // KROT+KSC churn together still plateaus, or climbs further.
    constexpr size_t KSC_CAP=1;
    auto atLevelCopy=[&](const gpufhe::KeySwitchConstants& KF)->gpufhe::KeySwitchConstants{
        gpufhe::KeySwitchConstants K; K.n=n;
        gpufhe::compute_keyswitch_constants(K,mod,modP,numPart);
        for(uint32_t i=0;i<tw;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
        for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rq[tw+j]);}
        K.fullQ=tw; K.evalKeyTowers=KF.evalKeyTowers;
        K.av.assign(K.numPart,{}); K.bv.assign(K.numPart,{});
        for(uint32_t p=0;p<K.numPart;++p){K.av[p]=KF.av[p];K.bv[p]=KF.bv[p];}
        return K; };
    std::map<uint32_t,gpufhe::KeySwitchConstants> KSC;
    std::list<uint32_t> KSC_lru;
    std::map<uint32_t,std::list<uint32_t>::iterator> KSC_lru_pos;
    uint32_t nKscBuilds=0, nKscEvictions=0;
    auto rotFullCached=[&](uint32_t k)->const gpufhe::KeySwitchConstants&{
        auto it=KROT.find(k);
        if(it!=KROT.end()){
            KROT_lru.erase(KROT_lru_pos[k]); KROT_lru.push_front(k); KROT_lru_pos[k]=KROT_lru.begin();
            return it->second;
        }
        std::vector<uint64_t> sA=KPqp.s;
        gpufhe::automorphism_eval_host(sA,sizeQlP,n,k,mq,rq);
        gpufhe::KeySwitchConstants K; K.n=n;
        gpufhe::compute_keyswitch_constants(K,mod,modP,numPart);
        for(uint32_t i=0;i<tw;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
        for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modP[j]);K.rootValList.push_back(rq[tw+j]);}
        gpufhe::evalkeygen_host_sold(K,KPqp.s,sA,KPqp.pkA,KPqp.pkB,PM,mq,rq,1,3.19,900+k);
        ++nKeygen;
        if(KROT.size()>=KROT_CAP){
            uint32_t victim=KROT_lru.back(); KROT_lru.pop_back();
            KROT_lru_pos.erase(victim); KROT.erase(victim); ++nEvictions;
        }
        KROT.emplace(k,std::move(K));
        KROT_lru.push_front(k); KROT_lru_pos[k]=KROT_lru.begin();
        return KROT[k]; };
    auto keyAt=[&](uint32_t k)->void{
        if(KSC.size()>=KSC_CAP && KSC.find(k)==KSC.end()){
            uint32_t victim=KSC_lru.back(); KSC_lru.pop_back();
            KSC_lru_pos.erase(victim); KSC.erase(victim); ++nKscEvictions;
        }
        if(KSC.find(k)==KSC.end()){
            KSC.emplace(k,atLevelCopy(rotFullCached(k)));
            KSC_lru.push_front(k); KSC_lru_pos[k]=KSC_lru.begin();
            ++nKscBuilds;
        }
        long rss=rssMB(); long gu,gf; gpuMemMB(gu,gf);
        std::cout<<"k="<<k<<": RSS="<<rss<<"MB  GPU used="<<gu<<"MB  "
                  <<"KROT.size()="<<KROT.size()<<" KSC.size()="<<KSC.size()
                  <<" nEvictions(KROT)="<<nEvictions<<" nKscEvictions="<<nKscEvictions<<"\n"; };

    std::cout<<"=== phase 2: KROT+KSC combined churn (mirrors real keyAt()) ===\n";
    const std::vector<uint32_t> ks2={27,29,31,33,35,37,39,41,43,45,47,49};
    for(uint32_t k:ks2) keyAt(k);

    std::cout<<"\n=== phase 2 summary ===\n";
    std::cout<<"nKeygen(total)="<<nKeygen<<" KROT evictions="<<nEvictions
              <<" KSC builds="<<nKscBuilds<<" KSC evictions="<<nKscEvictions
              <<" final RSS="<<rssMB()<<"MB\n";
    std::cout<<"If this plateaus near the phase-1 level (no large additional climb), the KROT+KSC "
              <<"cache mechanism itself is NOT the runaway cause and the real bottleneck must be "
              <<"elsewhere (ciphertext/ModRaise/stageDiags/ks_context_create host-side buffers -- "
              <<"none of which this test touches). If it keeps growing well past phase 1's plateau, "
              <<"the KSC layer itself is the additional, previously-unquantified contributor.\n";
    return 0;
}
