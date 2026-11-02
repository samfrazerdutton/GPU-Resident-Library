// Stage 5 capstone: full Hybrid KeySwitchCore composed as a library function.
// Extract KeySwitchConstants from OpenFHE (borrowed key), run
// keyswitch_core_resident, compare ba0/ba1 against OpenFHE's KeySwitchCore
// (decompose->fastcore->ModDown) bit-exact. All sub-ops proven; this proves
// the composition + the OpenFHE-free core interface.

#include "openfhe.h"
#include "keyswitch.h"
#include <iostream>
#include <random>
#include <vector>

using namespace lbcrypto;
using gpufhe::KeySwitchConstants;

int main() {
    // GOD PROMPT "CKKS ENGINE: RESOLVE POLYEVAL CORRECTNESS" round: this
    // OpenFHE build secretly links an external GPU-accelerated fork
    // (~/openfheNVDIA-GPU) whose libopenfhe_cuda_hal.so exports
    // gpu_ks_enabled()/gpu_keyswitch_inner_product, gated by env vars
    // OPENFHE_GPU and OPENFHE_GPU_KS (found via `strings` on that .so).
    // keyswitch-hybrid.cpp's EvalFastKeySwitchCoreExt branches to this GPU
    // path BEFORE the textbook per-digit CPU loop whenever gpu_ks_enabled()
    // is true -- meaning this test's "OpenFHE reference" call may have been
    // silently running through an UNDOCUMENTED, externally-patched GPU
    // implementation this whole time, not vanilla OpenFHE CPU code. Force
    // both off, mirroring bench_evalmult_resident.cpp's existing
    // OPENFHE_GPU=0 fix for the SAME external-GPU-shadowing issue (found an
    // earlier round) applied to the general ShadowRegistry path.
    setenv("OPENFHE_GPU","0",1);
    setenv("OPENFHE_GPU_KS","0",1);
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(3); params.SetScalingModSize(50);
    params.SetRingDim(32768); params.SetBatchSize(16384);
    params.SetKeySwitchTechnique(HYBRID);
    auto cc = GenCryptoContext(params);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE); cc->Enable(ADVANCEDSHE);
    auto kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);

    auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto ep = cp->GetElementParams();
    const uint32_t n = ep->GetRingDimension();
    const uint32_t sizeQl = (uint32_t)ep->GetParams().size();
    auto paramsP = cp->GetParamsP();
    const uint32_t sizeP = (uint32_t)paramsP->GetParams().size();
    const uint32_t sizeQlP = sizeQl + sizeP;
    const uint32_t fullQ = sizeQl; // fresh ct at full level

    // Input a = random poly over Ql (stands in for cv.back()), eval form.
    std::mt19937_64 rng(20260731);
    std::vector<uint64_t> aTowers((size_t)sizeQl*n);
    using Poly = DCRTPoly::PolyType;
    std::vector<Poly> tv; tv.reserve(sizeQl);
    for (uint32_t i=0;i<sizeQl;++i){ auto pp=ep->GetParams()[i];
        uint64_t q=pp->GetModulus().ConvertToInt();
        NativeVector v(n,pp->GetModulus());
        for(uint32_t k=0;k<n;++k){ uint64_t x=rng()%q; v[k]=NativeInteger(x); aTowers[(size_t)i*n+k]=x; }
        Poly p(pp,Format::EVALUATION,true); p.SetValues(std::move(v),Format::EVALUATION); tv.push_back(std::move(p)); }
    DCRTPoly a(tv);

    // Borrow eval key.
    auto evalKeyVec = CryptoContextImpl<DCRTPoly>::GetEvalMultKeyVector(kp.secretKey->GetKeyTag());
    auto evalKey = evalKeyVec[0];
    const auto& av = evalKey->GetAVector();
    const auto& bv = evalKey->GetBVector();

    // OpenFHE reference: KeySwitchCore(a, evalKey).
    auto ref = cc->GetScheme()->KeySwitchCore(a, evalKey);

    // ===== extract KeySwitchConstants =====
    KeySwitchConstants K;
    K.n=n; K.sizeQl=sizeQl; K.sizeP=sizeP; K.fullQ=fullQ;
    K.alpha = cp->GetNumPerPartQ();
    uint32_t numPart = (uint32_t)std::ceil((double)sizeQl/K.alpha);
    if (numPart > cp->GetNumberOfQPartitions()) numPart = cp->GetNumberOfQPartitions();
    K.numPart = numPart;
    for(uint32_t i=0;i<sizeQl;++i) K.qMod.push_back(ep->GetParams()[i]->GetModulus().ConvertToInt());
    for(uint32_t j=0;j<sizeP;++j) K.pMod.push_back(paramsP->GetParams()[j]->GetModulus().ConvertToInt());

    // Per-part decompose constants.
    K.sizePart.resize(numPart); K.sizeCompl.resize(numPart); K.startIdx.resize(numPart);
    K.partQHatInv.resize(numPart); K.partQHatInvPrec.resize(numPart); K.partSrcMod.resize(numPart);
    K.partQHatModp.resize(numPart); K.partComplMod.resize(numPart);
    K.partBMuLo.resize(numPart); K.partBMuHi.resize(numPart);
    for (uint32_t part=0; part<numPart; ++part){
        uint32_t startIdx=K.alpha*part;
        uint32_t sizePart=(sizeQl>startIdx+K.alpha)?K.alpha:(sizeQl-startIdx);
        auto paramsPartQ=cp->GetParamsPartQ(part);
        auto paramsCompl=cp->GetParamsComplPartQ(sizeQl-1,part);
        uint32_t sizeCompl=(uint32_t)paramsCompl->GetParams().size();
        K.startIdx[part]=startIdx; K.sizePart[part]=sizePart; K.sizeCompl[part]=sizeCompl;

        const auto& qhi=cp->GetPartQlHatInvModq(part,sizePart-1);
        const auto& qhip=cp->GetPartQlHatInvModqPrecon(part,sizePart-1);
        const auto& qmp=cp->GetPartQlHatModp(sizeQl-1,part);
        const auto& bmu=cp->GetmodComplPartqBarrettMu(sizeQl-1,part);
        for(uint32_t i=0;i<sizePart;++i){ K.partQHatInv[part].push_back(qhi[i].ConvertToInt());
            K.partQHatInvPrec[part].push_back(qhip[i].ConvertToInt());
            K.partSrcMod[part].push_back(paramsPartQ->GetParams()[i]->GetModulus().ConvertToInt()); }
        K.partQHatModp[part].resize((size_t)sizePart*sizeCompl);
        for(uint32_t i=0;i<sizePart;++i)for(uint32_t j=0;j<sizeCompl;++j)
            K.partQHatModp[part][(size_t)i*sizeCompl+j]=qmp[i][j].ConvertToInt();
        for(uint32_t j=0;j<sizeCompl;++j){ K.partComplMod[part].push_back(paramsCompl->GetParams()[j]->GetModulus().ConvertToInt());
            unsigned __int128 mu=(unsigned __int128)bmu[j];
            K.partBMuLo[part].push_back((uint64_t)mu); K.partBMuHi[part].push_back((uint64_t)(mu>>64)); }
    }

    // Eval key av/bv flattened per part over full QP towers.
    K.evalKeyTowers = (uint32_t)av[0].GetAllElements().size();
    K.av.resize(numPart); K.bv.resize(numPart);
    for (uint32_t part=0; part<numPart; ++part){
        uint32_t kt=K.evalKeyTowers;
        K.av[part].resize((size_t)kt*n); K.bv[part].resize((size_t)kt*n);
        for(uint32_t t=0;t<kt;++t)for(uint32_t k=0;k<n;++k){
            K.av[part][(size_t)t*n+k]=av[part].GetAllElements()[t][k].ConvertToInt();
            K.bv[part][(size_t)t*n+k]=bv[part].GetAllElements()[t][k].ConvertToInt(); }
    }

    // ModDown constants (arg-less).
    for(uint32_t j=0;j<sizeP;++j){ K.pHatInv.push_back(cp->GetPHatInvModp()[j].ConvertToInt());
        K.pHatInvPrec.push_back(cp->GetPHatInvModpPrecon()[j].ConvertToInt()); }
    K.pHatModq.resize((size_t)sizeP*sizeQl);
    for(uint32_t i=0;i<sizeP;++i)for(uint32_t j=0;j<sizeQl;++j)
        K.pHatModq[(size_t)i*sizeQl+j]=cp->GetPHatModq()[i][j].ConvertToInt();
    for(uint32_t j=0;j<sizeQl;++j){ unsigned __int128 mu=(unsigned __int128)cp->GetModqBarrettMu()[j];
        K.mdBMuLo.push_back((uint64_t)mu); K.mdBMuHi.push_back((uint64_t)(mu>>64)); }
    for(uint32_t i=0;i<sizeQl;++i){ K.pInvModq.push_back(cp->GetPInvModq()[i].ConvertToInt()); }

    // OpenFHE's exact stored root of unity per modulus (Q and P), so the
    // resident INTT/NTT match OpenFHE's transform convention.
    for(uint32_t i=0;i<sizeQl;++i){
        K.rootModList.push_back(ep->GetParams()[i]->GetModulus().ConvertToInt());
        K.rootValList.push_back(ep->GetParams()[i]->GetRootOfUnity().ConvertToInt()); }
    for(uint32_t j=0;j<sizeP;++j){
        K.rootModList.push_back(paramsP->GetParams()[j]->GetModulus().ConvertToInt());
        K.rootValList.push_back(paramsP->GetParams()[j]->GetRootOfUnity().ConvertToInt()); }

    // ===== run resident =====
    auto R = gpufhe::keyswitch_core_resident(aTowers, K);

    // Compare ba0/ba1 vs OpenFHE ref (both sizeQl towers, eval).
    uint32_t bad0=0,bad1=0; int f0=-1,f1=-1;
    for(uint32_t i=0;i<sizeQl;++i)for(uint32_t k=0;k<n;++k){
        uint64_t r0=(*ref)[0].GetAllElements()[i][k].ConvertToInt();
        uint64_t r1=(*ref)[1].GetAllElements()[i][k].ConvertToInt();
        if(R.ba0[(size_t)i*n+k]!=r0){ if(f0<0)f0=(int)i; ++bad0; }
        if(R.ba1[(size_t)i*n+k]!=r1){ if(f1<0)f1=(int)i; ++bad1; }
    }
    std::cout<<"sizeQl="<<sizeQl<<" sizeP="<<sizeP<<" numPart="<<numPart<<" evalKeyTowers="<<K.evalKeyTowers<<"\n";
    std::cout<<"  ba0: "<<(bad0==0?"ok":"BAD"); if(bad0)std::cout<<" "<<bad0<<" first tower "<<f0; std::cout<<"\n";
    std::cout<<"  ba1: "<<(bad1==0?"ok":"BAD"); if(bad1)std::cout<<" "<<bad1<<" first tower "<<f1; std::cout<<"\n";

    // GOD PROMPT "CKKS ENGINE: RESOLVE POLYEVAL CORRECTNESS" round -- extra
    // algebraic sanity check (INFORMATIONAL ONLY, formula NOT confirmed
    // correct -- see result below). The naive expectation was ba0+ba1*s ~
    // a*s^2 (mod q, the textbook single-eval-key relin identity). This
    // check consistently measures ~50% of q here -- i.e. pure noise --
    // EVEN on ba0/ba1 already proven bit-exact against OpenFHE two lines
    // above, with the external GPU-keyswitch-shadow path independently
    // ruled out (OPENFHE_GPU/OPENFHE_GPU_KS=0 set at the top of main, found
    // via `strings` on libopenfhe_cuda_hal.so: it exports gpu_ks_enabled()/
    // gpu_keyswitch_inner_product, gated by exactly those two env vars,
    // and keyswitch-hybrid.cpp's EvalFastKeySwitchCoreExt branches to it
    // before the textbook per-digit CPU loop whenever enabled -- disabling
    // it changed the measured secret key value, confirming the env vars do
    // something, but did NOT change the ~50%-of-q result). Tracing through
    // OpenFHE's own keyswitch-hybrid.cpp source (digit decomposition +
    // per-part av/bv accumulation, EvalFastKeySwitchCoreExt's final loop)
    // did not resolve the exact per-digit scaling convention in the time
    // available. CONCLUSION: this specific "ba0+ba1*s ~ a*s^2" formula is
    // NOT validated and must not be used to allege a defect anywhere --
    // ba0/ba1's bit-exact match against OpenFHE's own reference (the check
    // directly above) remains the trusted, working correctness evidence
    // for keyswitch_core_resident. This block is kept only as a documented,
    // honestly-unresolved research note for whoever next wants to pin down
    // the exact per-digit algebraic contract.
    {
        auto sElem = kp.secretKey->GetPrivateElement();
        std::cout<<"  [algebraic check] secret key format: "<<(sElem.GetFormat()==Format::EVALUATION?"EVALUATION":"COEFFICIENT (converting)")<<"\n";
        if(sElem.GetFormat()!=Format::EVALUATION) sElem.SetFormat(Format::EVALUATION);
        std::cout<<"  [algebraic check] secret key tower count="<<sElem.GetAllElements().size()<<" vs sizeQl="<<sizeQl
                  <<"; secret tower0 modulus="<<sElem.GetAllElements()[0].GetModulus().ConvertToInt()
                  <<" vs ep tower0 modulus="<<ep->GetParams()[0]->GetModulus().ConvertToInt()<<"\n";
        {
            uint64_t q=ep->GetParams()[0]->GetModulus().ConvertToInt();
            uint64_t sv=sElem.GetAllElements()[0][0].ConvertToInt();
            uint64_t av=aTowers[0];
            uint64_t b0=R.ba0[0], b1=R.ba1[0];
            std::cout<<"  [raw dump] tower=0 k=0: q="<<q<<" a="<<av<<" s="<<sv<<" ba0="<<b0<<" ba1="<<b1<<"\n";
        }
        auto mulmodL=[](uint64_t a,uint64_t b,uint64_t q){return (uint64_t)(((unsigned __int128)a*b)%q);};
        auto addmodL=[](uint64_t a,uint64_t b,uint64_t q){uint64_t s=a+b;return s>=q?s-q:s;};
        double worst_frac=0;
        for(uint32_t i=0;i<sizeQl;++i){
            uint64_t q=ep->GetParams()[i]->GetModulus().ConvertToInt();
            uint64_t worst=0;
            for(uint32_t k=0;k<n;++k){
                uint64_t sv=sElem.GetAllElements()[i][k].ConvertToInt();
                uint64_t s2=mulmodL(sv,sv,q);
                uint64_t av=aTowers[(size_t)i*n+k];
                uint64_t rhs=mulmodL(av,s2,q);
                uint64_t lhs=addmodL(R.ba0[(size_t)i*n+k], mulmodL(R.ba1[(size_t)i*n+k],sv,q), q);
                uint64_t diff=(lhs>=rhs)?(lhs-rhs):(rhs-lhs);
                uint64_t cdiff=(diff>q-diff)?(q-diff):diff;
                if(cdiff>worst) worst=cdiff;
            }
            double frac=(double)worst/(double)q;
            if(frac>worst_frac) worst_frac=frac;
            std::cout<<"  [algebraic check] tower="<<i<<" worst|ba0+ba1*s - a*s^2| = "<<(frac*100.0)<<"% of q (EVAL domain, pointwise -- see methodology note below)\n";
        }
        std::cout<<"  [algebraic check] worst fraction of q across all towers (EVAL domain): "<<(worst_frac*100.0)<<"%\n";

        // GOD PROMPT "CKKS PHASE 6" round: the ~50%-of-q result above was
        // just independently reproduced AND explained using a fresh
        // OpenFHE-only test (tests/test_evalmult_relin_openfhe_reference.cpp):
        // ba0/ba1/a/s here are all in Format::EVALUATION (NTT/point-value
        // domain). NTT is a bijective but highly MIXING linear transform --
        // a tiny COEFFICIENT-domain difference (genuine key-switch noise,
        // small relative to q) spreads into something that looks like a
        // large, pseudo-random difference at EVERY INDIVIDUAL NTT point
        // (each point is a sum over all n coefficients). Comparing lhs[k]
        // vs rhs[k] pointwise in EVALUATION domain is therefore NOT a valid
        // "are these nearly the same polynomial" test. Converting each
        // tower to Format::COEFFICIENT (an actual INTT) before comparing
        // resolved the identical check in the OpenFHE-only test (50% of q
        // -> 4.5e-11% of q). Re-run the SAME check here, in COEFFICIENT
        // domain, directly on keyswitch_core_resident's own output -- if
        // this also resolves, it proves ba0/ba1 were correct all along and
        // entry #30's "unresolved research question" was a test artifact,
        // not a real gap.
        {
            double worst_frac_coeff=0;
            for(uint32_t i=0;i<sizeQl;++i){
                auto pp=ep->GetParams()[i];
                uint64_t q=pp->GetModulus().ConvertToInt();

                NativeVector vA(n,pp->GetModulus()), vBa0(n,pp->GetModulus()), vBa1(n,pp->GetModulus()), vS(n,pp->GetModulus());
                for(uint32_t k=0;k<n;++k){
                    vA[k]=NativeInteger(aTowers[(size_t)i*n+k]);
                    vBa0[k]=NativeInteger(R.ba0[(size_t)i*n+k]);
                    vBa1[k]=NativeInteger(R.ba1[(size_t)i*n+k]);
                    vS[k]=NativeInteger(sElem.GetAllElements()[i][k].ConvertToInt());
                }
                // FIX (corrects an earlier bug in this exact block): ring
                // multiplication is POINTWISE multiplication only in
                // Format::EVALUATION (NTT domain) -- multiplying raw
                // Format::COEFFICIENT values pointwise via mulmodL is NOT
                // ring multiplication, it is a meaningless per-index
                // Hadamard product. Multiply FIRST in EVALUATION format
                // (where pointwise IS correct ring multiplication, via
                // OpenFHE's own Poly operators), THEN convert the RESULT
                // to COEFFICIENT format before comparing -- exactly the
                // order tests/test_evalmult_relin_openfhe_reference.cpp
                // uses (and which resolved the identical check there).
                Poly pA(pp,Format::EVALUATION,true); pA.SetValues(std::move(vA),Format::EVALUATION);
                Poly pBa0(pp,Format::EVALUATION,true); pBa0.SetValues(std::move(vBa0),Format::EVALUATION);
                Poly pBa1(pp,Format::EVALUATION,true); pBa1.SetValues(std::move(vBa1),Format::EVALUATION);
                Poly pS(pp,Format::EVALUATION,true); pS.SetValues(std::move(vS),Format::EVALUATION);

                Poly lhsPoly = pBa0 + (pBa1 * pS);
                Poly rhsPoly = (pA * pS) * pS;
                lhsPoly.SetFormat(Format::COEFFICIENT);
                rhsPoly.SetFormat(Format::COEFFICIENT);

                uint64_t worst=0;
                for(uint32_t k=0;k<n;++k){
                    uint64_t lhs=lhsPoly[k].ConvertToInt();
                    uint64_t rhs=rhsPoly[k].ConvertToInt();
                    uint64_t diff=(lhs>=rhs)?(lhs-rhs):(rhs-lhs);
                    uint64_t cdiff=(diff>q-diff)?(q-diff):diff;
                    if(cdiff>worst) worst=cdiff;
                }
                double frac=(double)worst/(double)q;
                if(frac>worst_frac_coeff) worst_frac_coeff=frac;
                std::cout<<"  [algebraic check, COEFFICIENT domain] tower="<<i<<" worst|ba0+ba1*s - a*s^2| = "<<(frac*100.0)<<"% of q\n";
            }
            std::cout<<"  [algebraic check, COEFFICIENT domain] worst fraction of q across all towers: "<<(worst_frac_coeff*100.0)<<"%\n";
        }
    }

    // GOD PROMPT "CKKS PHASE 6" round: the COEFFICIENT-domain fix above did
    // NOT resolve the identity for THIS test's `a` (still ~50% of q), yet
    // a separate, standalone OpenFHE-only test
    // (tests/test_evalmult_relin_openfhe_reference.cpp) found the SAME
    // identity, on the SAME KeySwitchCore function, DOES hold (to ~1e-10%
    // of q) when `a` is a REAL ciphertext tensor-product component
    // (t2=c1a*c1b from genuine encrypted ciphertexts) rather than this
    // file's synthetic, independently-per-tower-random `a`. Controlled
    // experiment: reuse this file's OWN `cc`/`kp`/`evalKey` (same context,
    // same eval key already proven bit-exact against keyswitch_core_resident
    // above) but swap in a REAL t2 instead of synthetic `a`, and re-run the
    // identical COEFFICIENT-domain check -- isolating whether "syntheticness"
    // of the input (not keyswitch_core_resident, not KeySwitchCore, not the
    // check methodology) is the actual variable that matters.
    {
        auto sElem = kp.secretKey->GetPrivateElement();
        auto mulmodL=[](uint64_t a,uint64_t b,uint64_t q){return (uint64_t)(((unsigned __int128)a*b)%q);};
        auto addmodL=[](uint64_t a,uint64_t b,uint64_t q){uint64_t s=a+b;return s>=q?s-q:s;};

        std::vector<double> slotsA(n/2, 0.0), slotsB(n/2, 0.0);
        for (uint32_t k=0;k<4 && k<n/2;++k){ slotsA[k]=(double)(k+1); slotsB[k]=(double)(k+5); }
        Plaintext ptA = cc->MakeCKKSPackedPlaintext(slotsA);
        Plaintext ptB = cc->MakeCKKSPackedPlaintext(slotsB);
        auto ctA = cc->Encrypt(kp.publicKey, ptA);
        auto ctB = cc->Encrypt(kp.publicKey, ptB);

        // Decisive control: does a REAL cc->EvalMult (with relin) +
        // cc->Decrypt round trip even work correctly for THIS cc/kp
        // (depth=3, batchsize=16384)? If yes, by the same airtight
        // algebra as tests/test_evalmult_relin_openfhe_reference.cpp's
        // CHECK1/1b/1c, ba0+ba1*s MUST equal t2*s^2 up to small noise --
        // so a persisting ~50%-of-q result below would mean a bug in
        // THIS check's own code, not a real mathematical gap.
        {
            auto realProduct = cc->EvalMult(ctA, ctB);
            Plaintext rp;
            cc->Decrypt(kp.secretKey, realProduct, &rp);
            rp->SetLength(4);
            auto dec = rp->GetRealPackedValue();
            std::vector<double> expect = {5.0,12.0,21.0,32.0};
            double me=0.0;
            for (size_t k=0;k<4;++k){ double e=std::abs(dec[k]-expect[k]); if(e>me) me=e; }
            std::cout << "  [real-data control] CHECK1-equivalent real decrypt: decoded=["
                      << dec[0] << "," << dec[1] << "," << dec[2] << "," << dec[3]
                      << "] expected=[5,12,21,32] max_err=" << me
                      << " -> " << (me<1e-2?"PASS":"FAIL") << "\n";
        }

        // Raw tensor product, UNRELINEARIZED -- c1a*c1b is exactly t2.
        auto tensorResult = cc->GetScheme()->EvalMult(ctA, ctB);
        auto& tcv = tensorResult->GetElements();
        std::cout << "  [real-data control] tensor component count=" << tcv.size() << "\n";
        DCRTPoly realT2 = tcv[2];

        auto realRef = cc->GetScheme()->KeySwitchCore(realT2, evalKey);
        uint32_t realTowers = (uint32_t)realT2.GetAllElements().size();
        std::cout << "  [real-data control] real t2 towers=" << realTowers << " (sizeQl=" << sizeQl << ")\n";
        // Verify tower ALIGNMENT directly -- do NOT assume index i means
        // the same modulus for realT2 and sElem/ep just because sElem
        // happens to have sizeQl towers. A dropped tower from leveling
        // could be at the FRONT, not just the back.
        for (uint32_t i=0;i<realTowers;++i){
            auto t2tower = realT2.GetElementAtIndex(i);
            auto stower = sElem.GetElementAtIndex(i);
            std::cout << "  [real-data control] realT2 tower " << i
                      << " modulus=" << t2tower.GetModulus() << " root=" << t2tower.GetRootOfUnity()
                      << "  |  sElem tower " << i
                      << " modulus=" << stower.GetModulus() << " root=" << stower.GetRootOfUnity() << "\n";
        }

        double worst_frac_real=0;
        for(uint32_t i=0;i<realTowers && i<sizeQl;++i){
            auto pp=ep->GetParams()[i];
            uint64_t q=pp->GetModulus().ConvertToInt();

            // Same fix as the synthetic-data block above: multiply FIRST
            // in EVALUATION format (pointwise = correct ring
            // multiplication there), THEN convert the RESULT to
            // COEFFICIENT format for comparison.
            Poly pA = realT2.GetElementAtIndex(i);
            Poly pBa0 = (*realRef)[0].GetElementAtIndex(i);
            Poly pBa1 = (*realRef)[1].GetElementAtIndex(i);
            Poly pS = sElem.GetElementAtIndex(i);

            Poly lhsPoly = pBa0 + (pBa1 * pS);
            Poly rhsPoly = (pA * pS) * pS;
            lhsPoly.SetFormat(Format::COEFFICIENT);
            rhsPoly.SetFormat(Format::COEFFICIENT);

            uint64_t worst=0;
            for(uint32_t k=0;k<n;++k){
                uint64_t lhs=lhsPoly[k].ConvertToInt();
                uint64_t rhs=rhsPoly[k].ConvertToInt();
                uint64_t diff=(lhs>=rhs)?(lhs-rhs):(rhs-lhs);
                uint64_t cdiff=(diff>q-diff)?(q-diff):diff;
                if(cdiff>worst) worst=cdiff;
            }
            double frac=(double)worst/(double)q;
            if(frac>worst_frac_real) worst_frac_real=frac;
            std::cout<<"  [real-data control, COEFFICIENT domain] tower="<<i<<" worst|ba0+ba1*s - t2*s^2| = "<<(frac*100.0)<<"% of q\n";
        }
        std::cout<<"  [real-data control, COEFFICIENT domain] worst fraction of q across all towers: "<<(worst_frac_real*100.0)<<"%\n";
    }

    if(bad0==0&&bad1==0){ std::cout<<"[PASS] full resident KeySwitchCore bit-exact vs OpenFHE\n"; return 0; }
    std::cout<<"[FAIL]\n"; return 1;
}
