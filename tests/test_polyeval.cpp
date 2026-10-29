// EvalMod CORE: homomorphic polynomial evaluation via Chebyshev recurrence on
// encrypted data. T_0=1, T_1=x, T_{k+1}=2*x*T_k - T_{k-1}. Each T-step is a
// ct*ct multiply (relin) + rescale, consuming a level -> a DEEP chain.
// Gate: p(x)=sum c_k T_k(x) under encryption vs plaintext p(x), degree 7.
// Deep native context: sizeQ towers so depth ~ log2(deg)+slack. n=1024 sandbox.
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
void pt_to_eval_host(std::vector<uint64_t>&, const std::vector<int64_t>&, uint32_t, uint32_t,
                     const std::vector<uint64_t>&, const std::vector<uint64_t>&);
void ct_mul_pt_host(std::vector<uint64_t>&, std::vector<uint64_t>&, const std::vector<uint64_t>&,
                    uint32_t, uint32_t, const std::vector<uint64_t>&);
void ct_add_ct_host(std::vector<uint64_t>&, std::vector<uint64_t>&,
                    const std::vector<uint64_t>&, const std::vector<uint64_t>&,
                    uint32_t, uint32_t, const std::vector<uint64_t>&);
}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
static uint64_t am(uint64_t a,uint64_t b,uint64_t q){uint64_t s=a+b;return s>=q?s-q:s;}
static uint64_t sm(uint64_t a,uint64_t b,uint64_t q){return a>=b?a-b:a+q-b;}
using cd=std::complex<double>;

// A leveled ciphertext: 2 host components + current tower count + current scale.
struct Ct { std::vector<uint64_t> c0,c1; uint32_t tw; double scale; };

int main(){
    const uint32_t n=1024,S=n/2,numPart=2; const uint64_t ns=1;
    const uint32_t sizeQ=8, sizeP=2;   // 8 towers: q0 big + 6 rescale levels + tail
    std::vector<uint64_t> mod,modP;
    // q0 large (60-bit) for final decrypt headroom; mid towers 50-bit ~ Delta
    gpufhe::native_primes(mod,1,60,n,{});
    { std::vector<uint64_t> mids; gpufhe::native_primes(mids,sizeQ-1,50,n,mod);
      for(auto m:mids) mod.push_back(m); }
    gpufhe::native_primes(modP,sizeP,60,n,mod);
    std::vector<uint64_t> modQP=mod; for(auto p:modP)modQP.push_back(p);
    std::vector<uint64_t> root(sizeQ),rootQP;
    for(uint32_t i=0;i<sizeQ;++i)root[i]=gpufhe::native_root(n,mod[i]);
    rootQP=root; for(auto p:modP)rootQP.push_back(gpufhe::native_root(n,p));
    const uint32_t sizeQlP=sizeQ+sizeP;

    auto KPqp=gpufhe::keygen_host(n,modQP,rootQP,ns,3.19,101);
    std::vector<uint64_t> PModq_QP(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){uint64_t q=modQP[t],P=1%q;for(uint32_t j=0;j<sizeP;++j)P=mm(P,modQP[sizeQ+j]%q,q);PModq_QP[t]=P;}
    // relin key over QP
    gpufhe::KeySwitchConstants K; K.n=n;
    gpufhe::compute_keyswitch_constants(K,mod,modP,numPart);
    for(uint32_t i=0;i<sizeQ;++i){K.rootModList.push_back(mod[i]);K.rootValList.push_back(root[i]);}
    for(uint32_t j=0;j<sizeP;++j){K.rootModList.push_back(modQP[sizeQ+j]);K.rootValList.push_back(rootQP[sizeQ+j]);}
    gpufhe::evalkeygen_host(K,KPqp.s,KPqp.pkA,KPqp.pkB,PModq_QP,modQP,rootQP,ns,3.19,202);
    auto C=gpufhe::ks_context_create(K);

    const double Delta=std::pow(2.0,50);   // mid-tower scale

    // rescale constants per drop level: dropping tower (curTw-1) with survivors [0,curTw-1)
    auto rescaleConsts=[&](uint32_t curTw, std::vector<uint64_t>&s1, std::vector<uint64_t>&s2){
        std::vector<uint64_t> sub(mod.begin(), mod.begin()+curTw);
        gpufhe::native_rescale_consts(s1,s2,sub,curTw-1); };

    // ---- helpers over the leveled Ct ----
    // mul two Ct at same tower count -> tensor+relin+combine+rescale (drops 1 tower)
    auto mulCt=[&](const Ct&A,const Ct&B)->Ct{
        uint32_t tw=A.tw; size_t T=(size_t)tw*n;
        std::vector<uint64_t> t0(T),t1(T),t2(T);
        for(uint32_t t=0;t<tw;++t){uint64_t q=mod[t];
            for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
                t0[x]=mm(A.c0[x],B.c0[x],q);
                t1[x]=am(mm(A.c0[x],B.c1[x],q),mm(A.c1[x],B.c0[x],q),q);
                t2[x]=mm(A.c1[x],B.c1[x],q);}}
        static int mulCtCallCount=0;
        int callIdx=mulCtCallCount++;   // 0-indexed BEFORE increment: call 0 = twoxT=mulCt(xl,Tk[1]) (x*x, trivially 0 at slot 0); call 1 = tkm2=mulCt(Tk[0],one) (1.0*1.0, NOT trivially 0 anywhere -- a much more informative known-answer case). Extended this round from "callIdx==0 only" to cover both, with full-array max|err| instead of slot-0-only printouts, since slot 0's x=0.6*sin(0)=0 made the original call-0 check uninformative for distinguishing "correct" from "silently wrong".
        if(callIdx<=1){
            // Diagnostic: decrypt the RAW tensor (t0+t1*s+t2*s^2, no keyswitch)
            // under a freshly-derived secret at this level, to see whether the
            // tensor-product step itself (before any relin/keyswitch) is
            // already correct. Uses the SAME "slice modulus list, same seed"
            // re-derivation pattern the ORIGINAL (unmodified) code already
            // uses for its own final decrypt of `acc` -- confirmed consistent
            // by direct reading of src/keygen.cpp (ternary secret + Gaussian
            // error are sampled ONCE before the per-tower loop, independent of
            // how many towers are requested; each tower's "a" draws exactly n
            // RNG values in a fixed per-tower-index sequence -- so re-deriving
            // with a shorter, same-prefix modulus list and the same seed
            // reproduces the IDENTICAL secret for the shared tower indices).
            std::vector<uint64_t> modlX(mod.begin(),mod.begin()+tw),rootlX(root.begin(),root.begin()+tw);
            auto KPt=gpufhe::keygen_host(n,modlX,rootlX,ns,3.19,101);
            std::vector<uint64_t> s2t(T);
            for(uint32_t t=0;t<tw;++t){uint64_t q=mod[t];
                for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
                    s2t[x]=mm(KPt.s[x],KPt.s[x],q);}}
            std::vector<uint64_t> raw0(T);
            for(uint32_t t=0;t<tw;++t){uint64_t q=mod[t];
                for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k;
                    raw0[x]=am(am(t0[x],mm(t1[x],KPt.s[x],q),q),mm(t2[x],s2t[x],q),q);}}
            std::vector<uint64_t> zero(T,0);
            std::vector<int64_t> dect; gpufhe::decrypt_host(dect,raw0,zero,KPt.s,n,modlX,rootlX);
            std::vector<cd> yt; gpufhe::decode_host(yt,dect,n,A.scale*B.scale);
            double maxabs=0; for(uint32_t i=0;i<S;++i) maxabs=std::max(maxabs,std::abs(yt[i].real()));
            std::cerr<<"[chk] RAW TENSOR (pre-keyswitch) call="<<callIdx<<" tw="<<tw<<": y[0]="<<yt[0].real()
                      <<" y[1]="<<yt[1].real()<<" max|y| over all "<<S<<" slots="<<maxabs
                      <<" (expect A.msg*B.msg per-slot; call 0 is x*x [0 at slot0], call 1 is 1.0*1.0 [1.0 everywhere])\n";
            // IMPORTANT CAVEAT just discovered: decrypt_host's "tower 0 only"
            // recovery is valid ONLY when the true coefficient's magnitude
            // fits within q0 (centered, |v|<=q0/2). The raw tensor product's
            // true magnitude is ~A.scale*B.scale (~Delta^2 ~ 2^100 for
            // call=1), vastly exceeding q0 (~2^60) -- so dect[]'s "centered"
            // value above is an ALIASED residue mod q0, NOT the true value;
            // the y[0]/y[1]/max|y| printed above are NOT valid evidence of a
            // tensor-product defect by themselves. Print the raw, uncentered
            // residue mod q0 (dect[0] after undoing the centering) and q0
            // itself so this can be checked against an independent
            // Delta^2 mod q0 computation instead of trusting the (invalid)
            // centered interpretation.
            {
                uint64_t q0=mod[0];
                int64_t v=dect[0];
                uint64_t resid0 = (v<0) ? (uint64_t)(v+(int64_t)q0) : (uint64_t)v;
                std::cerr<<"[chk]   raw residue check: q0="<<q0<<" dect[0](centered)="<<dect[0]
                          <<" dect[0] mod q0 (uncentered)="<<resid0<<" A.scale*B.scale="<<(A.scale*B.scale)<<"\n";
            }
        }
        // relin uses the full-QP key; keyswitch_core_resident works at full tw? It
        // assumes sizeQl=K.sizeQl. For leveled we keyswitch at tw: rebuild a K at tw.
        // Simpler: keyswitch operates on the CURRENT towers -> need K sized to tw.
        // Build a per-level relin constant set once per tw (cache).
        // (For the gate: rebuild inline.)
        gpufhe::KeySwitchConstants Kl; Kl.n=n;
        std::vector<uint64_t> modl(mod.begin(),mod.begin()+tw);
        std::vector<uint64_t> rootl(root.begin(),root.begin()+tw);
        gpufhe::compute_keyswitch_constants(Kl,modl,modP,numPart);
        for(uint32_t i=0;i<tw;++i){Kl.rootModList.push_back(modl[i]);Kl.rootValList.push_back(rootl[i]);}
        for(uint32_t j=0;j<sizeP;++j){Kl.rootModList.push_back(modQP[sizeQ+j]);Kl.rootValList.push_back(rootQP[sizeQ+j]);}
        // eval key over (modl + P): re-run evalkeygen with sQP restricted to these towers+P
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
        auto R=gpufhe::keyswitch_core_resident(t2,Kl);
        std::vector<uint64_t> r0(T),r1(T);
        for(uint32_t t=0;t<tw;++t){uint64_t q=mod[t];
            for(uint32_t k=0;k<n;++k){size_t x=(size_t)t*n+k; r0[x]=am(t0[x],R.ba0[x],q); r1[x]=am(t1[x],R.ba1[x],q);}}
        if(callIdx<=1){
            // Isolate the keyswitch step: decrypt (r0+r1*s) = t0+t1*s+ba0+ba1*s
            // under the first-degree secret and compare against the raw-tensor
            // check above (t0+t1*s+t2*s^2, already proven ~0/correct). If this
            // ALSO comes out ~0 (full-array, not just slot 0), keyswitch_core_
            // resident's output correctly re-expresses the t2*s^2 term as a
            // linear-in-s term and the bug is elsewhere (the rescale step, or
            // something after it). If this diverges, the bug is isolated to
            // keyswitch_core_resident's consumption of Kl at this reduced
            // tw+sizeP basis. callIdx==1 (Tk[0]*one, both constant 1.0) is the
            // first mulCt call that later shows large post-rescale error
            // (6629.5/1437.4 at slots 0/1) -- checking it HERE, pre-rescale,
            // decides whether that error originates in tensor/keyswitch or is
            // introduced specifically by the rescale step.
            std::vector<uint64_t> modlX2(mod.begin(),mod.begin()+tw),rootlX2(root.begin(),root.begin()+tw);
            auto KPt2=gpufhe::keygen_host(n,modlX2,rootlX2,ns,3.19,101);
            std::vector<int64_t> decR; gpufhe::decrypt_host(decR,r0,r1,KPt2.s,n,modlX2,rootlX2);
            std::vector<cd> yR; gpufhe::decode_host(yR,decR,n,A.scale*B.scale);
            double maxabs=0; for(uint32_t i=0;i<S;++i) maxabs=std::max(maxabs,std::abs(yR[i].real()));
            std::cerr<<"[chk] POST-KEYSWITCH r0+r1*s (pre-rescale) call="<<callIdx<<" tw="<<tw<<": y[0]="<<yR[0].real()
                      <<" y[1]="<<yR[1].real()<<" max|y| over all "<<S<<" slots="<<maxabs
                      <<" (expect same as the RAW TENSOR check above for this same call)\n";
            // Same caveat as the RAW TENSOR check: decR here is ALSO a
            // "centered mod q0" value of a quantity whose true magnitude is
            // ~A.scale*B.scale (~Delta^2), so the centered/decoded numbers
            // above are similarly unreliable. Print the raw uncentered
            // residue mod q0 so it can be diffed directly against the RAW
            // TENSOR check's own residue (printed just above, same callIdx)
            // -- if keyswitch_core_resident correctly preserves the
            // decrypted value, these two residues should match almost
            // exactly (differing only by keyswitch's own small noise
            // contribution), the same way the tensor's residue matched its
            // independently-computed expectation.
            {
                uint64_t q0=mod[0];
                int64_t v=decR[0];
                uint64_t resid0=(v<0)?(uint64_t)(v+(int64_t)q0):(uint64_t)v;
                std::cerr<<"[chk]   raw residue check: decR[0](centered)="<<decR[0]
                          <<" decR[0] mod q0 (uncentered)="<<resid0<<"\n";
            }
        }
        // rescale via resident path with a context at these towers
        auto Cl=gpufhe::ks_context_create(Kl);
        std::vector<uint64_t> s1,s2; rescaleConsts(tw,s1,s2);
        uint64_t *dr0,*dr1,*scr,*drp;
        cudaMalloc(&dr0,T*8);cudaMalloc(&dr1,T*8);cudaMalloc(&scr,(size_t)n*8);cudaMalloc(&drp,(size_t)n*8);
        cudaMemcpy(dr0,r0.data(),T*8,cudaMemcpyHostToDevice);cudaMemcpy(dr1,r1.data(),T*8,cudaMemcpyHostToDevice);
        gpufhe::rescale_resident_raw(dr0,tw,Cl,s1,s2,scr,drp,0);
        gpufhe::rescale_resident_raw(dr1,tw,Cl,s1,s2,scr,drp,0);
        cudaDeviceSynchronize();
        Ct out; out.tw=tw-1; out.scale=A.scale*B.scale/(double)mod[tw-1];
        size_t T2=(size_t)out.tw*n; out.c0.resize(T2);out.c1.resize(T2);
        std::vector<uint64_t> h0(T),h1(T);
        cudaMemcpy(h0.data(),dr0,T*8,cudaMemcpyDeviceToHost);cudaMemcpy(h1.data(),dr1,T*8,cudaMemcpyDeviceToHost);
        std::copy(h0.begin(),h0.begin()+T2,out.c0.begin());
        std::copy(h1.begin(),h1.begin()+T2,out.c1.begin());
        cudaFree(dr0);cudaFree(dr1);cudaFree(scr);cudaFree(drp);
        gpufhe::ks_context_destroy(Cl);
        return out; };

    // drop a Ct's level WITHOUT multiply (rescale by a plaintext 1) to match levels:
    // simpler: multiply by encoded 1.0 then rescale — but that changes scale by Delta.
    // For Chebyshev we keep all terms at the SAME level by tracking and only
    // combining same-level cts. We handle level alignment by re-scaling constants.

    // ---- test: p(x) = sum_{k=0}^{7} c_k T_k(x), x scalar in [-0.8,0.8] per slot
    const uint32_t deg=7;
    std::vector<double> c={0.1,-0.3,0.2,0.15,-0.1,0.05,0.08,-0.04};
    std::vector<cd> xin(S);
    for(uint32_t i=0;i<S;++i) xin[i]={0.6*std::sin(0.02*i),0};

    // encrypt x at full towers
    auto encFull=[&](const std::vector<cd>&z,double scale)->Ct{
        std::vector<int64_t> m; gpufhe::encode_host(m,z,n,scale);
        Ct ct; ct.tw=sizeQ; ct.scale=scale;
        // fresh keypair at full towers for enc (same seed => matches KPqp low towers)
        static auto KPq=gpufhe::keygen_host(n,mod,root,ns,3.19,101);
        gpufhe::encrypt_host(ct.c0,ct.c1,m,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,303);
        return ct; };

    // We evaluate p via Chebyshev recurrence but must keep terms level-aligned.
    // Strategy (baby approach for the gate): compute T_k as PLAINTEXT-scaled cts
    // is not possible; instead evaluate the polynomial by Horner in the MONOMIAL
    // basis with ct*ct mults, which is the same depth and simpler to level-track.
    // p(x) = ((...((c7 x + c6) x + c5) x + ...) x + c0). Each step: ctacc = ctacc*x
    // (mul, drops level) then + c_k (plaintext add at that level's scale).
    // Monomial coeffs from Chebyshev:
    // Evaluate in the CHEBYSHEV basis (stable): result = sum_k c_k T_k(x).
    // T_0=1, T_1=x, T_{k+1}=2 x T_k - T_{k-1}. Each T is in [-1,1] (no coeff blowup).
    // Keep a running weighted sum "acc" = sum c_k T_k, aligned in level to the
    // freshest T. All cts carry scale ~Delta. ct*ct drops one level; to add cts
    // at different levels we drop the higher one to match (mul by encoded 1).
    auto encAt=[&](const std::vector<cd>&z,uint32_t tw,double scale,uint32_t seed)->Ct{
        std::vector<int64_t> m; gpufhe::encode_host(m,z,n,scale);
        Ct ct; ct.tw=tw; ct.scale=scale;
        static auto KPe=gpufhe::keygen_host(n,mod,root,ns,3.19,101);
        std::vector<uint64_t> modl(mod.begin(),mod.begin()+tw),rootl(root.begin(),root.begin()+tw);
        gpufhe::encrypt_host(ct.c0,ct.c1,m,KPe.pkA,KPe.pkB,n,modl,rootl,ns,3.19,seed);
        return ct; };
    // add plaintext-weighted ct into acc (acc += w * ctT), both at same tw/scale.
    // w is a real constant -> encode w as a constant plaintext, ct_mul_pt (scale*Delta),
    // rescale back. Simpler: fold w into a plaintext-diagonal multiply then align.
    // For the gate we sum c_k T_k by: scale each T_k by plaintext c_k (mul_pt+rescale)
    // then add all at the common lowest level.
    // Build T_0..T_deg as cts, tracking level. T_{k} lives at tw = sizeQ - (depth to make it).
    // Diagnostic: decrypt+decode Tk[k] right after construction and compare
    // against the independently-computed Chebyshev value, to localize EXACTLY
    // where (which k, which tower count) the divergence begins.
    auto checkTk=[&](uint32_t k, const Ct& t){
        std::vector<uint64_t> modl(mod.begin(),mod.begin()+t.tw),rootl(root.begin(),root.begin()+t.tw);
        auto KPchk=gpufhe::keygen_host(n,modl,rootl,ns,3.19,101);
        std::vector<int64_t> dec; gpufhe::decrypt_host(dec,t.c0,t.c1,KPchk.s,n,modl,rootl);
        std::vector<cd> yy; gpufhe::decode_host(yy,dec,n,t.scale);
        double maxe=0;
        for(uint32_t i=0;i<S;++i){
            double x=xin[i].real(); double Tprev=1,Tcur=x,Tval;
            if(k==0) Tval=1; else if(k==1) Tval=x;
            else { for(uint32_t j=2;j<=k;++j){ double Tn=2*x*Tcur-Tprev; Tprev=Tcur; Tcur=Tn; } Tval=Tcur; }
            maxe=std::max(maxe,std::abs(yy[i].real()-Tval));
        }
        std::cerr<<"[chk] Tk["<<k<<"] tw="<<t.tw<<" scale=2^"<<std::log2(t.scale)
                  <<" maxerr_vs_T_"<<k<<"(x)="<<maxe<<" y[0]="<<yy[0].real()<<"\n";
    };
    std::vector<Ct> Tk(deg+1);
    Tk[0]=encAt(std::vector<cd>(S,cd{1.0,0}), sizeQ, Delta, 600);
    Tk[1]=encAt(xin, sizeQ, Delta, 601);
    checkTk(0,Tk[0]); checkTk(1,Tk[1]);
    for(uint32_t k=2;k<=deg;++k){
        // 2*x*T_{k-1}: need x and T_{k-1} at same level = Tk[k-1].tw
        Ct xl=encAt(xin, Tk[k-1].tw, Delta, 700+k);
        Ct twoxT=mulCt(xl,Tk[k-1]);              // level tw-1, scale Delta
        // 2*: scale ciphertext by 2 (plaintext const 2.0 via mul_pt would drop a level;
        // instead double c0,c1 directly mod q — exact, no level cost)
        for(uint32_t t=0;t<twoxT.tw;++t){uint64_t q=mod[t];
            for(uint32_t kk=0;kk<n;++kk){size_t x=(size_t)t*n+kk;
                twoxT.c0[x]=(twoxT.c0[x]*2)%q; twoxT.c1[x]=(twoxT.c1[x]*2)%q;}}
        // subtract T_{k-2}: must be at twoxT.tw. Re-encrypt T_{k-2} value? We don't have
        // its plaintext. Instead drop T_{k-2} to twoxT.tw by mul-by-1 rescales.
        Ct tkm2=Tk[k-2];
        while(tkm2.tw>twoxT.tw){
            Ct one=encAt(std::vector<cd>(S,cd{1.0,0}), tkm2.tw, Delta, 800+k*10+tkm2.tw);
            if(k==2){
                // Both operands of the upcoming mulCt(tkm2,one) call, checked
                // INDEPENDENTLY before the multiply -- tkm2 is Tk[0], already
                // proven correct by checkTk(0,...) above; "one" is a FRESH
                // encAt call with the same structure. If both check out here
                // but the product (checked inside mulCt as call=1) is still
                // wrong, the defect is specifically in the multiply/tensor
                // step for this operand pair, not in either input.
                std::vector<uint64_t> moX(mod.begin(),mod.begin()+tkm2.tw),roX(root.begin(),root.begin()+tkm2.tw);
                auto KPX=gpufhe::keygen_host(n,moX,roX,ns,3.19,101);
                std::vector<int64_t> decTkm2; gpufhe::decrypt_host(decTkm2,tkm2.c0,tkm2.c1,KPX.s,n,moX,roX);
                std::vector<cd> yTkm2; gpufhe::decode_host(yTkm2,decTkm2,n,tkm2.scale);
                double eTkm2=0; for(uint32_t i=0;i<S;++i) eTkm2=std::max(eTkm2,std::abs(yTkm2[i].real()-1.0));
                std::vector<int64_t> decOne; gpufhe::decrypt_host(decOne,one.c0,one.c1,KPX.s,n,moX,roX);
                std::vector<cd> yOne; gpufhe::decode_host(yOne,decOne,n,one.scale);
                double eOne=0; for(uint32_t i=0;i<S;++i) eOne=std::max(eOne,std::abs(yOne[i].real()-1.0));
                std::cerr<<"[chk] pre-mulCt operands for call=1: tkm2(=Tk[0]) maxerr_vs_1.0="<<eTkm2
                          <<" one maxerr_vs_1.0="<<eOne<<" (both should be ~0 if both operands are valid 1.0-encryptions)\n";
                std::cerr<<"[chk]   raw coefficient[0] (integer, before decode): decTkm2[0]="<<decTkm2[0]
                          <<" decOne[0]="<<decOne[0]<<" (Delta="<<(int64_t)tkm2.scale
                          <<"; noise_A[0]=decTkm2[0]-Delta="<<(decTkm2[0]-(int64_t)tkm2.scale)
                          <<" noise_B[0]=decOne[0]-Delta="<<(decOne[0]-(int64_t)one.scale)<<")\n";
            }
            tkm2=mulCt(tkm2,one); }   // each drops a level, scale stays ~Delta
        if(k==2) checkTk(2,twoxT);   // expect 2*x*T_1(x) = 2x^2 (not T_2 itself -- just a label for the print)
        if(k==2){
            std::vector<uint64_t> modl2(mod.begin(),mod.begin()+tkm2.tw),rootl2(root.begin(),root.begin()+tkm2.tw);
            auto KPchk2=gpufhe::keygen_host(n,modl2,rootl2,ns,3.19,101);
            std::vector<int64_t> dec2; gpufhe::decrypt_host(dec2,tkm2.c0,tkm2.c1,KPchk2.s,n,modl2,rootl2);
            std::vector<cd> yy2; gpufhe::decode_host(yy2,dec2,n,tkm2.scale);
            double maxerr2=0; for(uint32_t i=0;i<S;++i) maxerr2=std::max(maxerr2,std::abs(yy2[i].real()-1.0));
            std::cerr<<"[chk] k=2 tkm2 (should be ~1.0 everywhere): tw="<<tkm2.tw<<" scale=2^"<<std::log2(tkm2.scale)<<" y[0]="<<yy2[0].real()<<" y[1]="<<yy2[1].real()<<" maxerr_vs_1.0 over all "<<S<<" slots="<<maxerr2<<"\n";
        }
        // now subtract
        Ct res; res.tw=twoxT.tw; res.scale=Delta; size_t T=(size_t)res.tw*n;
        res.c0.resize(T);res.c1.resize(T);
        for(uint32_t t=0;t<res.tw;++t){uint64_t q=mod[t];
            for(uint32_t kk=0;kk<n;++kk){size_t x=(size_t)t*n+kk;
                res.c0[x]=sm(twoxT.c0[x],tkm2.c0[x],q); res.c1[x]=sm(twoxT.c1[x],tkm2.c1[x],q);}}
        Tk[k]=res;
        checkTk(k,Tk[k]);
    }
    // find common lowest level
    uint32_t low=sizeQ; for(uint32_t k=0;k<=deg;++k) low=std::min(low,Tk[k].tw);
    // drop each T_k to `low`, scale by c_k (plaintext), sum
    Ct acc; acc.tw=low; acc.scale=Delta; size_t TT=(size_t)low*n;
    acc.c0.assign(TT,0); acc.c1.assign(TT,0); bool first=true;
    for(uint32_t k=0;k<=deg;++k){
        Ct t=Tk[k];
        while(t.tw>low){ Ct one=encAt(std::vector<cd>(S,cd{1.0,0}), t.tw, Delta, 850+k*7+t.tw); t=mulCt(t,one); }
        // multiply by plaintext c_k -> scale Delta^2, then we'll rescale the SUM once.
        std::vector<cd> zc(S, cd{c[k],0});
        std::vector<int64_t> mc; gpufhe::encode_host(mc,zc,n,Delta);
        std::vector<uint64_t> modl(mod.begin(),mod.begin()+low),rootl(root.begin(),root.begin()+low);
        std::vector<uint64_t> cE; gpufhe::pt_to_eval_host(cE,mc,low,n,modl,rootl);
        std::vector<uint64_t> b0=t.c0,b1=t.c1;
        gpufhe::ct_mul_pt_host(b0,b1,cE,low,n,modl);   // scale Delta^2
        if(first){acc.c0=b0;acc.c1=b1;acc.scale=Delta*Delta;first=false;}
        else gpufhe::ct_add_ct_host(acc.c0,acc.c1,b0,b1,low,n,modl);
    }
    // rescale acc once: Delta^2 -> Delta^2/qLast
    {
        gpufhe::KeySwitchConstants Kl; Kl.n=n;
        std::vector<uint64_t> modl(mod.begin(),mod.begin()+low),rootl(root.begin(),root.begin()+low);
        gpufhe::compute_keyswitch_constants(Kl,modl,modP,numPart);
        for(uint32_t i=0;i<low;++i){Kl.rootModList.push_back(modl[i]);Kl.rootValList.push_back(rootl[i]);}
        for(uint32_t j=0;j<sizeP;++j){Kl.rootModList.push_back(modQP[sizeQ+j]);Kl.rootValList.push_back(rootQP[sizeQ+j]);}
        Kl.av.assign(Kl.numPart,{});Kl.bv.assign(Kl.numPart,{});Kl.evalKeyTowers=low+sizeP;
        for(uint32_t p=0;p<Kl.numPart;++p){Kl.av[p].assign((size_t)(low+sizeP)*n,0);Kl.bv[p].assign((size_t)(low+sizeP)*n,0);}
        auto Cl=gpufhe::ks_context_create(Kl);
        std::vector<uint64_t> s1,s2; { std::vector<uint64_t> sub(mod.begin(),mod.begin()+low); gpufhe::native_rescale_consts(s1,s2,sub,low-1); }
        size_t T=(size_t)low*n; uint64_t *dr0,*dr1,*scr,*drp;
        cudaMalloc(&dr0,T*8);cudaMalloc(&dr1,T*8);cudaMalloc(&scr,(size_t)n*8);cudaMalloc(&drp,(size_t)n*8);
        cudaMemcpy(dr0,acc.c0.data(),T*8,cudaMemcpyHostToDevice);cudaMemcpy(dr1,acc.c1.data(),T*8,cudaMemcpyHostToDevice);
        gpufhe::rescale_resident_raw(dr0,low,Cl,s1,s2,scr,drp,0);
        gpufhe::rescale_resident_raw(dr1,low,Cl,s1,s2,scr,drp,0);
        cudaDeviceSynchronize();
        uint32_t nt=low-1; size_t T2=(size_t)nt*n;
        std::vector<uint64_t> h0(T),h1(T); cudaMemcpy(h0.data(),dr0,T*8,cudaMemcpyDeviceToHost);cudaMemcpy(h1.data(),dr1,T*8,cudaMemcpyDeviceToHost);
        acc.c0.assign(h0.begin(),h0.begin()+T2); acc.c1.assign(h1.begin(),h1.begin()+T2);
        acc.tw=nt; acc.scale=Delta*Delta/(double)mod[low-1];
        cudaFree(dr0);cudaFree(dr1);cudaFree(scr);cudaFree(drp); gpufhe::ks_context_destroy(Cl);
    }

    // decrypt acc at its level
    std::vector<uint64_t> modl(mod.begin(),mod.begin()+acc.tw),rootl(root.begin(),root.begin()+acc.tw);
    auto KPr=gpufhe::keygen_host(n,modl,rootl,ns,3.19,101);
    std::vector<int64_t> dec; gpufhe::decrypt_host(dec,acc.c0,acc.c1,KPr.s,n,modl,rootl);
    std::vector<cd> yout; gpufhe::decode_host(yout,dec,n,acc.scale);

    double maxerr=0;
    for(uint32_t i=0;i<S;++i){
        double x=xin[i].real();
        // p = sum c_k T_k(x) via Chebyshev recurrence
        double Tprev=1, Tcur=x, p=c[0]*1.0 + (deg>=1? c[1]*x : 0.0);
        for(uint32_t k=2;k<=deg;++k){ double Tn=2*x*Tcur-Tprev; p+=c[k]*Tn; Tprev=Tcur; Tcur=Tn; }
        maxerr=std::max(maxerr,std::abs(yout[i].real()-p)); }
    std::cout<<"deg-7 poly eval, final tw="<<acc.tw<<" scale=2^"<<std::log2(acc.scale)<<" maxerr="<<maxerr<<"\n";
    std::cout<<"  y[0..3]="; for(int k=0;k<4;++k)std::cout<<yout[k].real()<<" "; std::cout<<"\n";
    if(maxerr<1e-2){std::cout<<"[PASS] homomorphic polynomial evaluation (EvalMod core: deep Chebyshev/Horner chain)\n";return 0;}
    std::cout<<"[FAIL]\n"; return 1;
}
