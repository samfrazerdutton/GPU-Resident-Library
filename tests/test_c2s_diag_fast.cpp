// Directive Priority 3: stageDiags()/stageDiagsFwd() (tests/test_boot.cpp,
// tests/test_boot_n131072.cpp) compute the sparse "generalized diagonal"
// decomposition of one merged-FFT-stage group (applyMerged/applyMergedFwd,
// RST consecutive radix-2 butterfly stages) by probing with S=n/2 unit
// basis vectors, each probe costing O(S*RST) inside applyMerged -- O(S^2*RST)
// total per call, measured at 268.047s for ONE call at n=131072
// (research/MEMORY_FEASIBILITY.md). This is pure brute-force enumeration of
// an operator whose structure is already fully known in closed form from
// its own butterfly recursion -- it does not need to be probed at all.
//
// This file implements and PROVES EQUIVALENT (before anything is wired into
// a production test) an analytic replacement: instead of running the whole
// S-point transform once per basis vector, track, for every output index,
// the SET of (source index, coefficient) pairs it depends on via a single
// forward symbolic dynamic-programming pass through the RST butterfly
// stages (each stage merges two DISJOINT index sets, by the standard
// radix-2 decimation invariant -- the same reason FFT works at all -- so
// each output index's list grows by exactly 2x per stage, capping at
// 2^RST entries, never more). Complexity: O(S*2^RST) total, vs. the
// original O(S^2*RST) -- a reduction of S/(2^RST*RST) at fixed RST,
// independent of n. This matches two things already true at n=8192/
// n=131072: RST=4 (fixed by LST=4), and the diagonal COUNT (not derived
// here, taken as given) the original code's own comment already states in
// closed form (2^RST for the first merged stage, 2^(RST+1)-1 after) --
// confirming the operator's sparsity really is a known, fixed combinatorial
// fact, not something that needs empirical discovery.
//
// Correctness strategy (per the governing directive -- test BEFORE trusting,
// old implementation kept as oracle, no change wired into production until
// equivalence is proven):
//   1. Reproduce the EXACT brute-force stageDiags/stageDiagsFwd +
//      applyMerged/applyMergedFwd/twd from tests/test_boot_n131072.cpp,
//      unchanged, as the oracle.
//   2. Implement the analytic DP version independently (fastStageDiags/
//      fastStageDiagsFwd).
//   3. Compare BOTH the set of nonzero diagonal offsets AND every
//      coefficient, exactly (bit-for-bit where the arithmetic order
//      matches, else within a tight numerical tolerance), across multiple
//      n (512..32768, kept small enough that the O(S^2) oracle is still
//      tractable to run directly -- no n=131072 brute-force run is
//      attempted, by design) and every stage g of every LST that evenly
//      divides log2(S) at that n.
//   4. Only after (3) passes everywhere tested, benchmark both
//      implementations' wall time at each n (including n=131072, where
//      ONLY the fast path is run -- the brute-force oracle is skipped
//      there on purpose, consistent with "no large bootstrap launched
//      merely to see what happens").
#include <complex>
#include <vector>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <chrono>
#include <algorithm>
using cd = std::complex<double>;
using clk = std::chrono::steady_clock;
static double EL(clk::time_point a){ return std::chrono::duration<double>(clk::now()-a).count(); }

// ---- Shared problem setup, identical to tests/test_boot_n131072.cpp's own
// C2S/S2C section (rk[], twd, applyMerged/applyMergedFwd, brute-force
// stageDiags/stageDiagsFwd) ----
struct Problem {
    uint32_t n, S, M, lgS;
    std::vector<uint64_t> rk;
    Problem(uint32_t n_): n(n_), S(n_/2), M(2*n_) {
        rk.resize(S);
        uint64_t r=1; for(uint32_t k=0;k<S;++k){ rk[k]=r; r=(r*5)%M; }
        lgS=0; while((1u<<lgS)<S) ++lgS;
    }
    cd twd(uint32_t k, uint32_t st) const {
        uint64_t e=((uint64_t)rk[k]<<st)%M;
        return std::polar(1.0, 2*M_PI*(double)e/(double)M);
    }
    // C2S direction: stage st applied in increasing order within a group.
    void applyMerged(std::vector<cd>& v, uint32_t g, uint32_t RST) const {
        for(uint32_t st=g*RST; st<(g+1)*RST; ++st){
            uint32_t h=S>>(st+1); std::vector<cd> o(S);
            for(uint32_t base=0;base<S;base+=2*h)
                for(uint32_t k=0;k<h;++k){
                    cd a=v[base+k], b=v[base+k+h], t=twd(k,st);
                    o[base+k]=(a+b)*0.5; o[base+k+h]=(a-b)/(2.0*t);
                }
            v.swap(o);
        }
    }
    // S2C direction: stage st applied in DECREASING order within a group,
    // inverse butterfly (no 0.5 scale, multiply instead of divide by t).
    void applyMergedFwd(std::vector<cd>& v, uint32_t g, uint32_t RST) const {
        for(int st=(int)((g+1)*RST)-1; st>=(int)(g*RST); --st){
            uint32_t h=S>>((uint32_t)st+1); std::vector<cd> o(S);
            for(uint32_t base=0;base<S;base+=2*h)
                for(uint32_t k=0;k<h;++k){
                    cd a=v[base+k], b=v[base+k+h], t=twd(k,(uint32_t)st);
                    o[base+k]=a+t*b; o[base+k+h]=a-t*b;
                }
            v.swap(o);
        }
    }
    // ---- Brute-force oracle (unchanged algorithm from production) ----
    void bruteDiags(uint32_t g, uint32_t RST, bool fwd,
                     std::vector<uint32_t>& offs, std::vector<std::vector<cd>>& dg) const {
        std::vector<std::vector<cd>> acc(S); std::vector<uint8_t> used(S,0);
        for(uint32_t jj=0;jj<S;++jj){
            std::vector<cd> u(S,cd{0,0}); u[jj]=1;
            if(fwd) applyMergedFwd(u,g,RST); else applyMerged(u,g,RST);
            for(uint32_t ii=0;ii<S;++ii) if(std::abs(u[ii])>1e-12){
                uint32_t d=(jj+S-ii)%S;
                if(!used[d]){ used[d]=1; acc[d].assign(S,cd{0,0}); }
                acc[d][ii]=u[ii];
            }
        }
        offs.clear(); dg.clear();
        for(uint32_t d=0;d<S;++d) if(used[d]){ offs.push_back(d); dg.push_back(acc[d]); }
    }

    // ---- Analytic DP replacement ----
    // map[i] holds the (source_index, coefficient) list such that, after
    // the stages processed so far, current_value[i] = sum coeff*original[src].
    using Lin = std::vector<std::pair<uint32_t,cd>>;
    static void scaleInto(const Lin& src, cd s, Lin& dst){
        dst.reserve(dst.size()+src.size());
        for(auto& p: src) dst.push_back({p.first, p.second*s});
    }
    void fastDiags(uint32_t g, uint32_t RST, bool fwd,
                    std::vector<uint32_t>& offs, std::vector<std::vector<cd>>& dg) const {
        std::vector<Lin> map(S);
        for(uint32_t i=0;i<S;++i) map[i]={{i, cd{1.0,0.0}}};
        if(!fwd){
            for(uint32_t st=g*RST; st<(g+1)*RST; ++st){
                uint32_t h=S>>(st+1); std::vector<Lin> nm(S);
                for(uint32_t base=0;base<S;base+=2*h)
                    for(uint32_t k=0;k<h;++k){
                        uint32_t ia=base+k, ib=base+k+h; cd t=twd(k,st);
                        Lin& A=nm[ia]; Lin& B=nm[ib];
                        scaleInto(map[ia], 0.5, A); scaleInto(map[ib], 0.5, A);
                        scaleInto(map[ia], 1.0/(2.0*t), B); scaleInto(map[ib], -1.0/(2.0*t), B);
                    }
                map.swap(nm);
            }
        } else {
            for(int st=(int)((g+1)*RST)-1; st>=(int)(g*RST); --st){
                uint32_t h=S>>((uint32_t)st+1); std::vector<Lin> nm(S);
                for(uint32_t base=0;base<S;base+=2*h)
                    for(uint32_t k=0;k<h;++k){
                        uint32_t ia=base+k, ib=base+k+h; cd t=twd(k,(uint32_t)st);
                        Lin& A=nm[ia]; Lin& B=nm[ib];
                        scaleInto(map[ia], 1.0, A); scaleInto(map[ib], t, A);
                        scaleInto(map[ia], 1.0, B); scaleInto(map[ib], -t, B);
                    }
                map.swap(nm);
            }
        }
        std::vector<std::vector<cd>> acc(S); std::vector<uint8_t> used(S,0);
        for(uint32_t ii=0;ii<S;++ii){
            for(auto& p: map[ii]){
                if(std::abs(p.second)<=1e-12) continue;
                uint32_t jj=p.first;
                uint32_t d=(jj+S-ii)%S;
                if(!used[d]){ used[d]=1; acc[d].assign(S,cd{0,0}); }
                acc[d][ii]=p.second;
            }
        }
        offs.clear(); dg.clear();
        for(uint32_t d=0;d<S;++d) if(used[d]){ offs.push_back(d); dg.push_back(acc[d]); }
    }
};

static bool compareResult(const std::vector<uint32_t>& o1, const std::vector<std::vector<cd>>& d1,
                           const std::vector<uint32_t>& o2, const std::vector<std::vector<cd>>& d2,
                           double& maxErr, size_t& offsMismatch){
    maxErr=0; offsMismatch=0;
    // Build offset->index maps and compare as sets first.
    std::vector<uint32_t> s1=o1, s2=o2;
    std::sort(s1.begin(),s1.end()); std::sort(s2.begin(),s2.end());
    if(s1!=s2){ offsMismatch = 1; return false; }
    for(size_t a=0;a<o1.size();++a){
        // find matching offset in o2
        auto it=std::find(o2.begin(),o2.end(),o1[a]);
        size_t b=(size_t)std::distance(o2.begin(),it);
        for(size_t i=0;i<d1[a].size();++i){
            double e=std::abs(d1[a][i]-d2[b][i]);
            if(e>maxErr) maxErr=e;
        }
    }
    return true;
}

int main(){
    bool allPass=true;
    // std::cerr, unbuffered, per this codebase's own established convention
    // (tests/test_boot_n131072.cpp's [mem] probes) -- guarantees progress
    // is visible even if the process is killed mid-run, unlike std::cout
    // when redirected to a file (fully block-buffered, nothing flushed
    // until either the buffer fills or the process exits normally).
    std::cerr<<"=== Phase 1: correctness, brute-force oracle vs analytic DP ===\n";
    // n small enough that the O(S^2*RST) oracle is tractable directly.
    const std::vector<uint32_t> ns={512,1024,2048,4096,8192,16384};
    for(uint32_t n: ns){
        Problem P(n);
        // Enumerate every LST that evenly divides lgS (same rule production
        // enforces), EXCEPT skip RST so large that 2^RST approaches S itself
        // (e.g. LST=1) -- that regime makes applyMerged's diagonal
        // decomposition genuinely DENSE (2^RST ~= S distinct S-sized
        // vectors, multi-GB at these n, by design of the math, not a bug),
        // which is not representative of any production LST (always 4) and
        // would blow up memory for a reason unrelated to the algorithm
        // being tested. RST<=10 (<=1024 diagonals, each <=S/2 complex
        // doubles) keeps every case here comfortably bounded while still
        // covering RST far larger than production's RST=4.
        for(uint32_t LST=1; LST<=P.lgS; ++LST){
            if(P.lgS % LST) continue;
            uint32_t RST=P.lgS/LST;
            if(RST>10) continue;
            for(uint32_t g=0; g<LST; ++g){
                for(int fwd=0; fwd<2; ++fwd){
                    std::vector<uint32_t> o1,o2; std::vector<std::vector<cd>> d1,d2;
                    P.bruteDiags(g,RST,(bool)fwd,o1,d1);
                    P.fastDiags(g,RST,(bool)fwd,o2,d2);
                    double maxErr; size_t offsMismatch;
                    bool ok=compareResult(o1,d1,o2,d2,maxErr,offsMismatch);
                    bool countOk = (o1.size()==o2.size());
                    bool tolOk = ok && maxErr<1e-9;
                    if(!tolOk || !countOk){
                        allPass=false;
                        std::cerr<<"[FAIL] n="<<n<<" LST="<<LST<<" RST="<<RST<<" g="<<g
                                  <<" fwd="<<fwd<<" offsMatch="<<ok<<" maxErr="<<maxErr
                                  <<" ndiag(brute)="<<o1.size()<<" ndiag(fast)="<<o2.size()<<"\n";
                    }
                }
            }
            std::cerr<<"  n="<<n<<" LST="<<LST<<" RST="<<RST<<": checked ("<<P.S<<"-point, "<<LST
                      <<" group(s), both directions)\n";
        }
    }
    std::cerr<<(allPass?"[PASS] analytic DP matches brute-force oracle exactly (within 1e-9) "
                         "across every tested (n,LST,g,direction) combination\n"
                       :"[FAIL] see mismatches above\n");

    std::cerr<<"\n=== Phase 2: benchmark, both implementations, multiple n (scaling, not extrapolation) ===\n";
    // Same production rule as tests/test_boot_n131072.cpp: use LST=4 when it
    // divides lgS, else fall back to the smallest valid divisor -- but at
    // n=16384 (lgS=14=2*7) 4 does not divide 14, so this reports whichever
    // divisor is actually used at that n, same as production would.
    const std::vector<uint32_t> benchNs={2048,4096,8192,16384};
    for(uint32_t n: benchNs){
        Problem P(n);
        uint32_t LST=4; if(P.lgS%LST){ LST=1; while(P.lgS%LST && LST<=P.lgS) ++LST; }
        uint32_t RST=P.lgS/LST;
        std::vector<uint32_t> o1,o2; std::vector<std::vector<cd>> d1,d2;
        auto t0=clk::now(); P.bruteDiags(0,RST,false,o1,d1); double tBrute=EL(t0);
        auto t1=clk::now(); P.fastDiags(0,RST,false,o2,d2); double tFast=EL(t1);
        // This n/RST combination may fall outside Phase 1's RST<=10 cap
        // (e.g. n=16384's lgS=13 is prime, so production's LST=4 fallback
        // lands on RST=13 here) -- verify equivalence directly at the
        // ACTUAL benchmarked configuration too, rather than relying on
        // Phase 1 having covered a nearby but not identical case.
        double maxErr; size_t offsMismatch;
        bool ok=compareResult(o1,d1,o2,d2,maxErr,offsMismatch) && o1.size()==o2.size() && maxErr<1e-9;
        if(!ok){ allPass=false; std::cerr<<"  [FAIL] n="<<n<<" RST="<<RST<<" benchmark config itself mismatched! maxErr="<<maxErr<<"\n"; }
        std::cerr<<"  n="<<n<<" S="<<P.S<<" LST="<<LST<<" RST="<<RST<<" (equivalence at this exact config: "
                  <<(ok?"VERIFIED":"FAILED")<<"): brute="<<tBrute<<"s fast="<<tFast<<"s speedup="<<(tBrute/std::max(tFast,1e-9))<<"x\n";
    }

    std::cerr<<"\n=== Phase 3: n=131072, FAST PATH ONLY (brute-force oracle skipped by design -- "
              <<"268s/call already measured in research/MEMORY_FEASIBILITY.md; re-running it here "
              <<"would not add new information and the directive asks not to repeat costly runs) ===\n";
    {
        uint32_t n=131072; Problem P(n);
        uint32_t LST=4, RST=P.lgS/LST;
        for(uint32_t g=0;g<LST;++g){
            std::vector<uint32_t> offs; std::vector<std::vector<cd>> dg;
            auto t0=clk::now(); P.fastDiags(g,RST,false,offs,dg); double t=EL(t0);
            std::cerr<<"  C2S stage g="<<g<<": fast="<<t<<"s diags="<<offs.size()<<"\n";
        }
        for(uint32_t g=0;g<LST;++g){
            std::vector<uint32_t> offs; std::vector<std::vector<cd>> dg;
            auto t0=clk::now(); P.fastDiags(g,RST,true,offs,dg); double t=EL(t0);
            std::cerr<<"  S2C stage g="<<g<<": fast="<<t<<"s diags="<<offs.size()<<"\n";
        }
    }
    return allPass?0:1;
}
