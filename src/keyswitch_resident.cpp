#include "keyswitch_resident.h"
#include "ntt.h"
#include "intt.h"
#include "basis_convert.h"
#include "ks_fastcore.h"
#include "moddown.h"
#include "switch_modulus.h"
#include "rescale.h"
#include "ks_params_validate.h"
#include <cstring>
#include <stdexcept>
#include <map>
#include <mutex>
#include <utility>

namespace gpufhe {
bool g_ks_batched=true;   // A/B switch for the batched fastcore
namespace {
uint64_t mulmod(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
uint64_t powmod(uint64_t b,uint64_t e,uint64_t q){unsigned __int128 r=1,bb=b%q;while(e){if(e&1)r=(r*bb)%q;bb=(bb*bb)%q;e>>=1;}return(uint64_t)r;}
uint64_t invmod(uint64_t a,uint64_t q){return powmod(a%q,q-2,q);}
// GOD PROMPT "CKKS PHASE 39": for LaunchModDownCombine's new sPrecon
// parameter -- duplicated per this project's established convention.
uint64_t shoup_precon(uint64_t c, uint64_t q) { return (uint64_t)(((unsigned __int128)c << 64) / q); }
uint64_t* upv(const std::vector<uint64_t>& h, cudaStream_t s=0){ uint64_t* d;
    cudaMalloc(&d,h.size()*8);
    cudaMemcpyAsync(d,h.data(),h.size()*8,cudaMemcpyHostToDevice,s);
    return d; }
// GOD PROMPT "CKKS PHASE 30": g_ks_cache_mutex() guards BOTH this function's
// `m` cache and g_rtab below (ks_context_create). Both are plain
// process-lifetime `static std::map`s populated via an unsynchronized
// find()-then-emplace() -- the same latent-race pattern class Phase 25
// found and FIXED as a genuinely active bug in src/rotate.cpp's dev_tab(),
// and separately found HERE (this exact file) via test_rotate_concurrent_
// cache.cpp but characterized as latent (that test's access pattern never
// triggered it). Phase 30's own concurrency testing of the CUDA-graph fix
// below (tests/test_ks_graph_lifecycle.cpp: 8 threads, each building its
// OWN KeySwitchConstants from the SAME native_primes-derived moduli, so
// all 8 race g_rtab/upc's cold-start insert for the SAME (n,q) keys
// simultaneously) DID trigger it for real: only 4-1 of 8 threads produced
// correct output, run to run, the classic signature of concurrent
// std::map corruption. Fixed with the SAME minimal, already-established
// pattern this project used for src/encode_fft.cpp's analogous cache
// (GOD PROMPT "CKKS PHASE 28"): lookups/inserts are cheap relative to the
// actual keyswitch work, so a lock here costs nothing that matters.
std::mutex& g_ks_cache_mutex(){ static std::mutex m; return m; }
// These constants depend only on (n,sizeQl,numPart) -- NOT on the key -- yet
// every rotation rebuilt ~135 identical allocations. Cache by that tuple; only
// av/bv stay per-key. Contexts do not own these (destroy must not free them).
// Guarded by g_ks_cache_mutex() -- see that function's comment for why.
uint64_t* upc(uint64_t key, const std::vector<uint64_t>& h){
    std::lock_guard<std::mutex> lock(g_ks_cache_mutex());
    static std::map<uint64_t,uint64_t*> m;
    auto it=m.find(key); if(it!=m.end()) return it->second;
    uint64_t* d=upv(h); m.emplace(key,d); return d; }
} // anon

DeviceKSContext ks_context_create(const KeySwitchConstants& K, cudaStream_t s){
    // K.av/K.bv must hold K.numPart entries each -- this function
    // unconditionally reads K.av[p]/K.bv[p] for every p below. Populating
    // K.rootModList/rootValList and calling compute_keyswitch_constants
    // alone does NOT fill av/bv; only evalkeygen_host (or
    // evalkeygen_host_sold) does. Passing a K built without one of those
    // calls previously segfaulted with no diagnostic (confirmed via gdb
    // across multiple occurrences this project) -- this check turns that
    // into a clear, immediate error instead.
    if(K.av.size()!=K.numPart || K.bv.size()!=K.numPart){
        throw std::invalid_argument(
            "ks_context_create: K.av.size()="+std::to_string(K.av.size())+
            " K.bv.size()="+std::to_string(K.bv.size())+
            " but K.numPart="+std::to_string(K.numPart)+
            " -- call evalkeygen_host (or evalkeygen_host_sold) to populate "
            "K.av/K.bv before ks_context_create; compute_keyswitch_constants "
            "alone does not set them.");
    }
    // GOD PROMPT "CKKS PHASE 10": the Hybrid key-switch's required
    // correctness condition, P >= Q_part_max (research/EXPERIMENT_LEDGER.md
    // entry #42/#43; full derivation in include/ks_params_validate.h),
    // enforced HERE -- the single, construction-time choke point every
    // real caller passes through (the Rust FFI bridge's
    // gpufhe_ks_fixture_create calls this; every benchmark/example that
    // builds a usable key-switch context calls this). Deliberately NOT
    // enforced inside compute_keyswitch_constants itself, which several
    // permanent diagnostic/regression tests (tests/test_ks_pq_ratio_sweep.cpp,
    // tests/test_keyswitch_stage_trace.cpp, tests/test_evalkeygen_isolated.cpp)
    // intentionally drive with INVALID configurations on purpose, to
    // measure and document the failure -- those tests call
    // keyswitch_core_resident directly and never reach ks_context_create,
    // so they are unaffected by this check. Uses K.qMod/K.pMod/K.startIdx/
    // K.sizePart -- compute_keyswitch_constants's OWN partition output --
    // so this check can never silently drift out of sync with the actual
    // partitioning logic. Zero runtime cost in the per-call hot path
    // (keyswitch_resident/keyswitch_core_resident never call this).
    {
        HybridKsValidation pv = check_hybrid_keyswitch_params(K.qMod, K.pMod, K.startIdx, K.sizePart);
        if(!pv.ok){
            throw std::invalid_argument(
                "ks_context_create: invalid Hybrid key-switch parameters -- "
                "P (product of the "+std::to_string(K.sizeP)+" auxiliary moduli) = "+pv.p_decimal+
                " is SMALLER than Q_part_max (the largest key-switch digit's own "
                "modulus product, partition "+std::to_string(pv.worst_part)+" of "+
                std::to_string(K.numPart)+") = "+pv.q_part_max_decimal+
                ". The Hybrid key-switch requires P >= Q_part_max; otherwise the "
                "decompose approximation's bounded error is not fully absorbed by "
                "ModDown's division by P, and the resulting key-switched ciphertext "
                "is cryptographically meaningless (measured ~50% of q relin-identity "
                "error in research/EXPERIMENT_LEDGER.md entry #42). Fix by increasing "
                "numPart (shrinks each digit's own modulus product, no effect on "
                "security parameters) or by increasing sizeP (changes log2(QP), a "
                "security-relevant parameter -- re-validate research/"
                "SECURITY_ASSESSMENT.md if you do this).");
        }
    }
    DeviceKSContext C;
    C.n=K.n; C.sizeQl=K.sizeQl; C.sizeP=K.sizeP; C.sizeQlP=K.sizeQl+K.sizeP;
    C.numPart=K.numPart; C.fullQ=K.fullQ;
    const uint32_t n=K.n;

    // root tables for every modulus in the root list (built like keyswitch.cpp)
    auto br=[](uint32_t v,uint32_t b){uint32_t r=0;for(uint32_t k=0;k<b;k++){r=(r<<1)|(v&1);v>>=1;}return r;};
    uint32_t logn=0; while((1u<<logn)<n)++logn;
    // Root tables depend only on (n,q) and are IDENTICAL across every context,
    // but each context was rebuilding and re-uploading all of them: 4 cudaMallocs
    // per modulus x 32 moduli = 128 allocations per rotation, and the measured
    // ks_context_create cost was 54 ms of a 170 ms rotation. Cache them for the
    // process lifetime and share the pointers; contexts no longer own them
    // (ks_context_destroy must not free these).
    struct RTab { uint64_t *fr,*fp,*ir,*ip, ninv, ninv_p; };
    static std::map<std::pair<uint32_t,uint64_t>,RTab> g_rtab;
    for(size_t r=0;r<K.rootModList.size();++r){
        uint64_t q=K.rootModList[r], root=K.rootValList[r];
        auto key=std::make_pair(n,q);
        // GOD PROMPT "CKKS PHASE 30": guarded by g_ks_cache_mutex() -- see
        // that function's comment (near upc(), above) for why this cache
        // needed it. Held across both the lookup/insert AND the read of
        // it->second into C's vectors, for simplicity; the work inside is
        // small (one size-n fill on a true cache miss, a few pointer reads
        // otherwise) relative to real keyswitch work.
        std::lock_guard<std::mutex> lock(g_ks_cache_mutex());
        auto it=g_rtab.find(key);
        if(it==g_rtab.end()){
            uint64_t rootInv=invmod(root,q);
            std::vector<uint64_t> fr(n),fp(n),ir(n),ip(n),pw(n);
            auto fill=[&](uint64_t base,std::vector<uint64_t>&tr,std::vector<uint64_t>&tp){
                pw[0]=1; for(uint32_t i=1;i<n;++i)pw[i]=mulmod(pw[i-1],base,q);
                for(uint32_t i=0;i<n;++i){uint64_t ri=pw[br(i,logn)];tr[i]=ri;tp[i]=(uint64_t)(((__uint128_t)ri<<64)/q);}};
            fill(root,fr,fp); fill(rootInv,ir,ip);
            uint64_t ni=invmod(n,q);
            RTab t{upv(fr),upv(fp),upv(ir),upv(ip),ni,(uint64_t)(((__uint128_t)ni<<64)/q)};
            it=g_rtab.emplace(key,t).first;
        }
        C.modList.push_back(q);
        C.d_fr.push_back(it->second.fr); C.d_fp.push_back(it->second.fp);
        C.d_ir.push_back(it->second.ir); C.d_ip.push_back(it->second.ip);
        C.ninv.push_back(it->second.ninv); C.ninv_p.push_back(it->second.ninv_p);
    }

    // per-part constants
    const uint64_t KB=((uint64_t)n<<40)|((uint64_t)K.sizeQl<<32)|((uint64_t)K.numPart<<24);
    C.sizePart=K.sizePart; C.sizeCompl=K.sizeCompl; C.startIdx=K.startIdx;
    for(uint32_t p=0;p<K.numPart;++p){
        C.complModHost.push_back(K.partComplMod[p]);
        C.d_qhi.push_back(upc(KB|((uint64_t)p<<8)|0,K.partQHatInv[p]));
        C.d_qhip.push_back(upc(KB|((uint64_t)p<<8)|1,K.partQHatInvPrec[p]));
        C.d_srcMod.push_back(upc(KB|((uint64_t)p<<8)|2,K.partSrcMod[p]));
        C.d_qmp.push_back(upc(KB|((uint64_t)p<<8)|3,K.partQHatModp[p]));
        C.d_complMod.push_back(upc(KB|((uint64_t)p<<8)|4,K.partComplMod[p]));
        C.d_mlo.push_back(upc(KB|((uint64_t)p<<8)|5,K.partBMuLo[p]));
        C.d_mhi.push_back(upc(KB|((uint64_t)p<<8)|6,K.partBMuHi[p]));
        C.d_av.push_back(upv(K.av[p], s));
        C.d_bv.push_back(upv(K.bv[p], s));
    }

    // moddown constants
    C.d_pHatInv=upc(KB|100,K.pHatInv); C.d_pHatInvPrec=upc(KB|101,K.pHatInvPrec);
    C.d_pMod=upc(KB|102,K.pMod); C.d_pHatModq=upc(KB|103,K.pHatModq);
    C.d_qMod=upc(KB|104,K.qMod); C.d_mdMuLo=upc(KB|105,K.mdBMuLo); C.d_mdMuHi=upc(KB|106,K.mdBMuHi);
    C.pInvModqHost=K.pInvModq; C.qModHost=K.qMod; C.pModHost=K.pMod;
    {   // constants for the batched fastcore launch (same every call)
        std::vector<uint64_t> mq(C.sizeQlP); std::vector<uint32_t> kr(C.sizeQlP);
        const uint32_t dl=C.fullQ-C.sizeQl;
        for(uint32_t i=0;i<C.sizeQlP;++i){
            mq[i]=(i<C.sizeQl)? K.qMod[i] : K.pMod[i-C.sizeQl];
            kr[i]=(i>=C.sizeQl)? i+dl : i; }
        C.d_modsQlP=upc(KB|200,mq);
        cudaMalloc(&C.d_keyRow,C.sizeQlP*4);
        // kr is pageable; cudaMemcpyAsync falls back to sync here -- do not pin kr without adding an explicit cudaStreamSynchronize first.
        cudaMemcpyAsync(C.d_keyRow,kr.data(),C.sizeQlP*4,cudaMemcpyHostToDevice,s); }
    return C;
}

DeviceKSWork ks_work_create(const DeviceKSContext& C){
    DeviceKSWork W; const size_t n=C.n;
    uint32_t maxPart=0; for(auto s:C.sizePart) maxPart=std::max(maxPart,s);
    cudaMalloc(&W.d_part,(size_t)maxPart*n*8);
    cudaMalloc(&W.d_compl,(size_t)C.sizeQlP*n*8);
    cudaMalloc(&W.d_res0,(size_t)C.sizeQlP*n*8);
    cudaMalloc(&W.d_res1,(size_t)C.sizeQlP*n*8);
    cudaMalloc(&W.d_pwork,(size_t)C.sizeP*n*8);
    cudaMalloc(&W.d_qsw,(size_t)C.sizeQl*n*8);
    cudaMalloc(&W.d_digPtrs,(size_t)C.sizeQlP*sizeof(uint64_t*));
    cudaMalloc(&W.d_rootsA,(size_t)C.sizeQlP*sizeof(uint64_t*));
    cudaMalloc(&W.d_rootsB,(size_t)C.sizeQlP*sizeof(uint64_t*));
    cudaMalloc(&W.d_towerMods,(size_t)C.sizeQlP*8);
    cudaMalloc(&W.d_nInv,(size_t)C.sizeQlP*8);
    cudaMalloc(&W.d_nInvPrecon,(size_t)C.sizeQlP*8);
    cudaMalloc(&W.d_mdS,(size_t)C.sizeQl*8);
    cudaMalloc(&W.d_mdSPrecon,(size_t)C.sizeQl*8);
    cudaMalloc(&W.d_mdQ,(size_t)C.sizeQl*8);
    // GOD PROMPT "CKKS PHASE 30": one dedicated, persistent host buffer per
    // (part, role) -- see keyswitch_resident.h's own comment on these
    // fields for the full story (a CUDA-graph-capture dangling-pointer bug
    // AND a separate value-aliasing bug, both proven via standalone
    // reproducers, research/KS_GRAPH_CAPTURE_INVESTIGATION.md).
    //
    // Sized DEFENSIVELY, not from C.numPart/C.sizeCompl directly: some
    // existing callers (tests/test_hoist_stage0.cpp's stageRotBegin) build
    // a deliberate "shape-only" placeholder DeviceKSContext to size a
    // long-lived W once, setting only n/sizeQl/sizeP/sizeQlP/sizePart --
    // numPart and sizeCompl are left at their default (0 / empty) on that
    // placeholder, since the ORIGINAL flat, sizeQlP-worst-case-sized
    // scratch buffers never needed them. Indexing by C.numPart (0 on that
    // placeholder) or C.sizeCompl[part] (empty) segfaulted the very first
    // time ks_accumulate_qp ran against the REAL per-diagonal context --
    // caught by the full regression suite, not by this phase's own new
    // tests (which always build a complete DeviceKSContext). Using
    // C.sizePart.size() for the part count (every known caller, including
    // this placeholder, DOES populate sizePart with one real entry per
    // part) and C.sizeQlP as a uniform worst-case inner size (every part's
    // slice or complement is, by construction, never larger than the
    // whole) gives buffers that are always large enough and never depend
    // on a field a shape-only context doesn't set.
    const uint32_t numPartSafe = (uint32_t)C.sizePart.size();
    W.h_rootsA_part.resize(numPartSafe); W.h_rootsB_part.resize(numPartSafe);
    W.h_towerMods_part.resize(numPartSafe); W.h_nInv_part.resize(numPartSafe); W.h_nInvPrecon_part.resize(numPartSafe);
    W.h_rootsA_compl.resize(numPartSafe); W.h_rootsB_compl.resize(numPartSafe); W.h_towerMods_compl.resize(numPartSafe);
    W.h_digPtrs_part.resize(numPartSafe);
    for (uint32_t part = 0; part < numPartSafe; ++part) {
        W.h_rootsA_part[part].resize(C.sizeQlP); W.h_rootsB_part[part].resize(C.sizeQlP);
        W.h_towerMods_part[part].resize(C.sizeQlP); W.h_nInv_part[part].resize(C.sizeQlP); W.h_nInvPrecon_part[part].resize(C.sizeQlP);
        W.h_rootsA_compl[part].resize(C.sizeQlP); W.h_rootsB_compl[part].resize(C.sizeQlP); W.h_towerMods_compl[part].resize(C.sizeQlP);
        W.h_digPtrs_part[part].resize(C.sizeQlP);
    }
    W.h_rootsA_pwork.resize(C.sizeP); W.h_rootsB_pwork.resize(C.sizeP);
    W.h_towerMods_pwork.resize(C.sizeP); W.h_nInv_pwork.resize(C.sizeP); W.h_nInvPrecon_pwork.resize(C.sizeP);
    W.h_rootsA_qsw.resize(C.sizeQl); W.h_rootsB_qsw.resize(C.sizeQl); W.h_towerMods_qsw.resize(C.sizeQl);
    W.h_mdS.resize(C.sizeQl); W.h_mdSPrecon.resize(C.sizeQl); W.h_mdQ.resize(C.sizeQl);
    return W;
}

void ks_context_destroy(DeviceKSContext& C){
    // NOTE: d_fr/d_fp/d_ir/d_ip point into the process-lifetime root-table
    // cache and are SHARED across contexts -- freeing them here would corrupt
    // every context created later. Only the per-context arrays are freed.
    // d_keyRow IS per-context (fresh cudaMalloc in ks_context_create, not
    // routed through the upc()/g_rtab shared caches) -- found via direct
    // code reading while investigating per-diagonal GPU-memory growth at
    // n=131072 (research/MEMORY_FEASIBILITY.md); confirmed too small
    // (sizeQlP*4 bytes, ~128 bytes at n=131072) to explain that growth by
    // itself, but a real, genuine per-call leak regardless -- fixed here.
    for(auto p:C.d_av)cudaFree(p);
    for(auto p:C.d_bv)cudaFree(p);
    cudaFree(C.d_keyRow); C.d_keyRow=nullptr;
}
void ks_work_destroy(DeviceKSWork& W){
    cudaFree(W.d_part);cudaFree(W.d_compl);cudaFree(W.d_res0);cudaFree(W.d_res1);
    cudaFree(W.d_pwork);cudaFree(W.d_qsw);cudaFree(W.d_digPtrs);
    cudaFree(W.d_rootsA);cudaFree(W.d_rootsB);cudaFree(W.d_towerMods);cudaFree(W.d_nInv);cudaFree(W.d_nInvPrecon);
    cudaFree(W.d_mdS);cudaFree(W.d_mdSPrecon);cudaFree(W.d_mdQ);
}

// modulus -> root-table row
static int row_of(const DeviceKSContext& C, uint64_t q){
    for(size_t r=0;r<C.modList.size();++r) if(C.modList[r]==q) return (int)r;
    throw std::runtime_error("no root table for modulus");
}

// Mechanical extraction from the original monolithic keyswitch_resident
// (same body, zero logic change) -- creates a callable seam between the
// QP-basis accumulate step and moddown, so a caller can insert work (e.g.
// applying an automorphism to the QP-basis accumulator) in between. Added
// to support correct-order rotation hoisting (see research/NEXT_SESSION.md);
// keyswitch_resident below is now a thin wrapper and is UNCHANGED in
// behavior -- verified via the existing test_ks_resident/test_rot_resident/
// test_boot suite, not just asserted.
void ks_accumulate_qp(const uint64_t* d_a, const DeviceKSContext& C, DeviceKSWork& W, cudaStream_t s)
{
    const uint32_t n=C.n, sizeQl=C.sizeQl, sizeQlP=C.sizeQlP;
    const uint32_t delta=C.fullQ - sizeQl;

    cudaMemsetAsync(W.d_res0,0,(size_t)sizeQlP*n*8,s);
    cudaMemsetAsync(W.d_res1,0,(size_t)sizeQlP*n*8,s);

    for(uint32_t part=0;part<C.numPart;++part){
        const uint32_t sp=C.sizePart[part], sc=C.sizeCompl[part], st=C.startIdx[part];
        const uint32_t en=st+sp;

        // 1. copy part slice (keep d_a intact), INTT each tower in place
        cudaMemcpyAsync(W.d_part, d_a+(size_t)st*n, (size_t)sp*n*8,
                        cudaMemcpyDeviceToDevice, s);
        {
            // GOD PROMPT "CKKS PHASE 30": writes into a buffer DEDICATED to
            // this (part, role) combination -- see keyswitch_resident.h's
            // long comment on these fields. A single buffer shared across
            // every part's INTT-the-slice role (the first fix attempt) is
            // NOT enough: during graph replay, every one of this loop's
            // numPart memcpy nodes would read whatever was written LAST
            // during capture, not its own part's value.
            auto& hrA=W.h_rootsA_part[part]; auto& hrB=W.h_rootsB_part[part];
            auto& htm=W.h_towerMods_part[part]; auto& hni=W.h_nInv_part[part]; auto& hnip=W.h_nInvPrecon_part[part];
            for(uint32_t i=0;i<sp;++i){
                uint64_t q=C.qModHost[st+i]; int r=row_of(C,q);
                hrA[i]=C.d_ir[r]; hrB[i]=C.d_ip[r];
                htm[i]=q; hni[i]=C.ninv[r]; hnip[i]=C.ninv_p[r];
            }
            cudaMemcpyAsync(W.d_rootsA,hrA.data(),sp*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(W.d_rootsB,hrB.data(),sp*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(W.d_towerMods,htm.data(),sp*8,cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(W.d_nInv,hni.data(),sp*8,cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(W.d_nInvPrecon,hnip.data(),sp*8,cudaMemcpyHostToDevice,s);
            LaunchINTT_GS_Batched(W.d_part, W.d_rootsA, W.d_rootsB, W.d_towerMods,
                                  W.d_nInv, W.d_nInvPrecon, sp, n, s);
        }
        // 2. basis convert partQ -> complement
        LaunchApproxSwitchCRTBasis(W.d_part, W.d_compl,
            C.d_qhi[part], C.d_qhip[part], C.d_srcMod[part],
            C.d_qmp[part], C.d_complMod[part], C.d_mlo[part], C.d_mhi[part],
            sp, sc, n, s);
        // 3. NTT each complement tower back to eval
        {
            // GOD PROMPT "CKKS PHASE 30": see the long comment above -- a
            // buffer dedicated to (part, "complement" role), distinct from
            // this same part's "part slice" role buffer above.
            auto& hrA=W.h_rootsA_compl[part]; auto& hrB=W.h_rootsB_compl[part]; auto& htm=W.h_towerMods_compl[part];
            for(uint32_t j=0;j<sc;++j){
                uint64_t q=C.complModHost[part][j]; int r=row_of(C,q);
                hrA[j]=C.d_fr[r]; hrB[j]=C.d_fp[r]; htm[j]=q;
            }
            cudaMemcpyAsync(W.d_rootsA,hrA.data(),sc*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(W.d_rootsB,hrB.data(),sc*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
            cudaMemcpyAsync(W.d_towerMods,htm.data(),sc*8,cudaMemcpyHostToDevice,s);
            LaunchNTT_CT_Batched(W.d_compl, W.d_rootsA, W.d_rootsB, W.d_towerMods, sc, n, s);
        }
        // 4. accumulate: digit tower i = d_a (in range) or d_compl (else).
        // BATCHED: was sizeQlP launches per part (32*15 = 480 per keyswitch at
        // n=8192/tw=30, each on a single 8192-element tower => launch-bound).
        // Now one launch per part with grid.y = tower, as tensor.cu does.
        // The digit source differs per tower (inside the part's range it is d_a,
        // otherwise d_compl with a shifted index), so pass a pointer array
        // rather than branching in the kernel.
        if(!g_ks_batched){
            for(uint32_t i=0;i<sizeQlP;++i){
                uint64_t q=(i<sizeQl)? C.qModHost[i] : C.pModHost[i-sizeQl];
                uint32_t idx=(i>=sizeQl)? i+delta : i;
                const uint64_t* dig = (i>=st && i<en) ? d_a+(size_t)i*n
                                    : (i<st) ? W.d_compl+(size_t)i*n
                                             : W.d_compl+(size_t)(i-sp)*n;
                LaunchKSMultAcc(W.d_res0+(size_t)i*n, W.d_res1+(size_t)i*n,
                                dig, C.d_bv[part]+(size_t)idx*n, C.d_av[part]+(size_t)idx*n,
                                q, n, s);
            }
        } else
        {   // GOD PROMPT "CKKS PHASE 30": per-part dedicated buffer, same
            // reasoning as the two blocks above -- this one, uniquely, also
            // depends on the caller's `d_a` pointer VALUE, not just
            // context-fixed data; still safe to stage through a W-owned
            // buffer, since a graph capturing this call is bound to that
            // exact d_a pointer identity for the life of the graph
            // regardless (see research/KS_GRAPH_CAPTURE_INVESTIGATION.md's
            // execution-model section) -- but MUST be indexed by part, or
            // replay would reuse only the LAST part's digit-pointer array
            // for every part's kernel launch.
            auto& hp=W.h_digPtrs_part[part];
            for(uint32_t i=0;i<sizeQlP;++i)
                hp[i] = (i>=st && i<en) ? d_a+(size_t)i*n
                      : (i<st)          ? W.d_compl+(size_t)i*n
                                        : W.d_compl+(size_t)(i-sp)*n;
            cudaMemcpyAsync(W.d_digPtrs,hp.data(),sizeQlP*sizeof(uint64_t*),
                            cudaMemcpyHostToDevice,s);
            LaunchKSMultAccBatched(W.d_res0, W.d_res1, W.d_digPtrs,
                                   C.d_bv[part], C.d_av[part],
                                   C.d_keyRow, C.d_modsQlP, sizeQlP, n, s);
        }
    }
}

// Mechanical extraction (same body as the original inline moddown lambda,
// zero logic change) -- the other half of the seam described above.
void ks_moddown(const DeviceKSContext& C, DeviceKSWork& W,
                uint64_t* d_res, uint64_t* d_out, cudaStream_t s)
{
    const uint32_t n=C.n, sizeQl=C.sizeQl, sizeP=C.sizeP;
    cudaMemcpyAsync(W.d_pwork, d_res+(size_t)sizeQl*n, (size_t)sizeP*n*8,
                    cudaMemcpyDeviceToDevice, s);
    {
        // GOD PROMPT "CKKS PHASE 30": dedicated "pwork" (P-part INTT) role
        // buffer -- see keyswitch_resident.h's long comment. Distinct from
        // the "qsw" (Ql NTT) role buffer below: this function's TWO roles
        // must not share one buffer, for the same reason ks_accumulate_qp's
        // per-part roles must not. Safe to reuse unchanged across this
        // function's two call sites per keyswitch_resident (d_ba0, d_ba1):
        // the computed value depends only on the fixed context C, never on
        // d_res/d_out, so both calls always (re)write the IDENTICAL value.
        for(uint32_t j=0;j<sizeP;++j){
            uint64_t q=C.pModHost[j]; int r=row_of(C,q);
            W.h_rootsA_pwork[j]=C.d_ir[r]; W.h_rootsB_pwork[j]=C.d_ip[r];
            W.h_towerMods_pwork[j]=q; W.h_nInv_pwork[j]=C.ninv[r]; W.h_nInvPrecon_pwork[j]=C.ninv_p[r];
        }
        cudaMemcpyAsync(W.d_rootsA,W.h_rootsA_pwork.data(),sizeP*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
        cudaMemcpyAsync(W.d_rootsB,W.h_rootsB_pwork.data(),sizeP*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
        cudaMemcpyAsync(W.d_towerMods,W.h_towerMods_pwork.data(),sizeP*8,cudaMemcpyHostToDevice,s);
        cudaMemcpyAsync(W.d_nInv,W.h_nInv_pwork.data(),sizeP*8,cudaMemcpyHostToDevice,s);
        cudaMemcpyAsync(W.d_nInvPrecon,W.h_nInvPrecon_pwork.data(),sizeP*8,cudaMemcpyHostToDevice,s);
        LaunchINTT_GS_Batched(W.d_pwork, W.d_rootsA, W.d_rootsB, W.d_towerMods,
                              W.d_nInv, W.d_nInvPrecon, sizeP, n, s);
    }
    LaunchApproxSwitchCRTBasis(W.d_pwork, W.d_qsw,
        C.d_pHatInv, C.d_pHatInvPrec, C.d_pMod,
        C.d_pHatModq, C.d_qMod, C.d_mdMuLo, C.d_mdMuHi,
        sizeP, sizeQl, n, s);
    {
        // GOD PROMPT "CKKS PHASE 30": dedicated "qsw" (Ql NTT) role buffer.
        for(uint32_t i=0;i<sizeQl;++i){
            uint64_t q=C.qModHost[i]; int r=row_of(C,q);
            W.h_rootsA_qsw[i]=C.d_fr[r]; W.h_rootsB_qsw[i]=C.d_fp[r]; W.h_towerMods_qsw[i]=q;
        }
        cudaMemcpyAsync(W.d_rootsA,W.h_rootsA_qsw.data(),sizeQl*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
        cudaMemcpyAsync(W.d_rootsB,W.h_rootsB_qsw.data(),sizeQl*sizeof(uint64_t*),cudaMemcpyHostToDevice,s);
        cudaMemcpyAsync(W.d_towerMods,W.h_towerMods_qsw.data(),sizeQl*8,cudaMemcpyHostToDevice,s);
        LaunchNTT_CT_Batched(W.d_qsw, W.d_rootsA, W.d_rootsB, W.d_towerMods, sizeQl, n, s);
    }
    // GOD PROMPT "CKKS PHASE 40" Track E: ONE batched launch across all
    // sizeQl towers instead of sizeQl separate LaunchModDownCombine calls --
    // the one remaining per-tower serial loop at this call site (the INTT/
    // NTT steps just above were already batched this way). Justified by a
    // standalone isolated reproducer (not committed) measuring this exact
    // serial-loop-vs-one-launch shape at this project's own n=8192 scale:
    // median 3.88x (sizeQl=5) / 5.93x (sizeQl=8) speedup, launch-overhead-
    // dominated exactly like every other kernel-merging win this project has
    // shipped (Phases 9, 30, 34, 38). h_mdS/h_mdSPrecon/h_mdQ are a pure
    // function of the fixed DeviceKSContext C (never of d_res/d_out), so
    // re-filling and re-uploading them identically on every call -- even
    // under CUDA graph capture, where only the LAST host-side write before
    // capture-end is ever read on replay -- is safe, matching the existing
    // h_towerMods_qsw/h_rootsA_qsw role's own established rule above.
    for(uint32_t i=0;i<sizeQl;++i){
        uint64_t q=C.qModHost[i];
        W.h_mdQ[i]=q; W.h_mdS[i]=C.pInvModqHost[i];
        W.h_mdSPrecon[i]=shoup_precon(C.pInvModqHost[i], q);
    }
    cudaMemcpyAsync(W.d_mdS,W.h_mdS.data(),sizeQl*8,cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(W.d_mdSPrecon,W.h_mdSPrecon.data(),sizeQl*8,cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(W.d_mdQ,W.h_mdQ.data(),sizeQl*8,cudaMemcpyHostToDevice,s);
    LaunchModDownCombineBatched(d_out, d_res, W.d_qsw, W.d_mdS, W.d_mdSPrecon, W.d_mdQ, sizeQl, n, s);
}

// keyswitch_resident: unchanged behavior, now a thin wrapper over the two
// pieces above (verified via test_ks_resident/test_rot_resident/test_boot,
// not just asserted).
void keyswitch_resident(const uint64_t* d_a, uint64_t* d_ba0, uint64_t* d_ba1,
                        const DeviceKSContext& C, DeviceKSWork& W, cudaStream_t s)
{
    ks_accumulate_qp(d_a, C, W, s);
    ks_moddown(C, W, W.d_res0, d_ba0, s);
    ks_moddown(C, W, W.d_res1, d_ba1, s);
}

void rescale_resident_raw(uint64_t* d_c, uint32_t towers,
                          const DeviceKSContext& C,
                          const std::vector<uint64_t>& s1, const std::vector<uint64_t>& s2,
                          uint64_t* d_scratch, uint64_t* d_drop, cudaStream_t s)
{
    const uint32_t n=C.n; if(towers<2) return;
    const uint32_t last=towers-1;
    uint64_t qLast=C.qModHost[last]; int rl=row_of(C,qLast);
    cudaMemcpyAsync(d_drop, d_c+(size_t)last*n, (size_t)n*8, cudaMemcpyDeviceToDevice, s);
    LaunchINTT_GS(d_drop, C.d_ir[rl], C.d_ip[rl], n, qLast, C.ninv[rl], C.ninv_p[rl], s);
    for(uint32_t t=0;t<last;++t){
        uint64_t qi=C.qModHost[t]; int r=row_of(C,qi);
        cudaMemcpyAsync(d_scratch, d_drop, (size_t)n*8, cudaMemcpyDeviceToDevice, s);
        LaunchSwitchModulus(d_scratch, qLast, qi, n, s);
        LaunchNTT_CT(d_scratch, C.d_fr[r], C.d_fp[r], n, qi, s);
        LaunchRescaleFuse(d_c+(size_t)t*n, d_scratch, s1[t], shoup_precon(s1[t],qi),
                          s2[t], shoup_precon(s2[t],qi), qi, n, s);
    }
}

} // namespace gpufhe
