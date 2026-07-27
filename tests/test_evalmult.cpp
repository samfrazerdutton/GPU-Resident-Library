// CAPSTONE: full native CKKS EvalMult, numerically proven end-to-end.
// encrypt(m1*D), encrypt(m2*D) -> tensor (c0,c1,c2) -> relin via
// keyswitch_core_resident(c2, native K) -> (c0+ba0, c1+ba1) -> decrypt ->
// recover m1*m2 (negacyclic poly product) after /D^2. Everything native:
// keygen, eval key, constants. OpenFHE only supplies the moduli/roots numbers.
#include "openfhe.h"
#include "keyswitch.h"
#include "keygen.h"
#include <iostream>
#include <vector>
#include <cmath>
using namespace lbcrypto;

static uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
static uint64_t am(uint64_t a,uint64_t b,uint64_t q){uint64_t s=a+b;return s>=q?s-q:s;}

int main(){
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(3); params.SetScalingModSize(50);
    params.SetRingDim(32768); params.SetBatchSize(16384);
    params.SetKeySwitchTechnique(HYBRID);
    auto cc=GenCryptoContext(params);
    auto cp=std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto ep=cp->GetElementParams(); auto paramsP=cp->GetParamsP();
    uint32_t n=ep->GetRingDimension(), sizeQ=ep->GetParams().size(), sizeP=paramsP->GetParams().size();
    uint32_t sizeQlP=sizeQ+sizeP, ns=cp->GetNoiseScale(), numPart=cp->GetNumPartQ();

    std::vector<uint64_t> mod(sizeQ),root(sizeQ),modQP(sizeQlP),rootQP(sizeQlP);
    for(uint32_t i=0;i<sizeQ;++i){ mod[i]=ep->GetParams()[i]->GetModulus().ConvertToInt(); root[i]=ep->GetParams()[i]->GetRootOfUnity().ConvertToInt(); modQP[i]=mod[i]; rootQP[i]=root[i]; }
    for(uint32_t j=0;j<sizeP;++j){ modQP[sizeQ+j]=paramsP->GetParams()[j]->GetModulus().ConvertToInt(); rootQP[sizeQ+j]=paramsP->GetParams()[j]->GetRootOfUnity().ConvertToInt(); }

    // ---- native keygen: keypair over Q (for enc/dec) and over QP (for eval key)
    auto KPq =gpufhe::keygen_host(n,mod,root,ns,3.19,101);
    // eval key needs s,pubkey over QP; extend the SAME secret coeff poly. Simplest:
    // regenerate a QP keypair with the same seed so s's low towers match KPq.
    // (Our keygen samples s from seed; same seed => same coeff poly across the
    // shared Q towers, plus P towers. Pubkey differs but that's fine for the key.)
    auto KPqp=gpufhe::keygen_host(n,modQP,rootQP,ns,3.19,101);

    // native constants
    gpufhe::KeySwitchConstants K; K.n=n;
    std::vector<uint64_t> mp(sizeP); for(uint32_t j=0;j<sizeP;++j) mp[j]=modQP[sizeQ+j];
    gpufhe::compute_keyswitch_constants(K, mod, mp, numPart);
    // root list (Q then P)
    for(uint32_t i=0;i<sizeQ;++i){ K.rootModList.push_back(mod[i]); K.rootValList.push_back(root[i]); }
    for(uint32_t j=0;j<sizeP;++j){ K.rootModList.push_back(modQP[sizeQ+j]); K.rootValList.push_back(rootQP[sizeQ+j]); }
    // P mod each QP tower (for eval key P*s2)
    std::vector<uint64_t> PModq_QP(sizeQlP);
    for(uint32_t t=0;t<sizeQlP;++t){ uint64_t q=modQP[t],P=1%q; for(uint32_t j=0;j<sizeP;++j)P=mm(P,modQP[sizeQ+j]%q,q); PModq_QP[t]=P; }
    // native eval key (uses KPqp: s,pubkey over QP)
    gpufhe::evalkeygen_host(K, KPqp.s, KPqp.pkA, KPqp.pkB, PModq_QP, modQP, rootQP, ns, 3.19, 202);

    // ---- messages
    const int64_t D=(int64_t)1<<25;   // scale (D^2 must stay < q0)
    std::vector<int64_t> a1(n,0), a2(n,0);
    a1[0]=3; a1[1]=-2; a2[0]=4; a2[2]=5;   // small sparse polys
    std::vector<int64_t> m1(n),m2(n);
    for(uint32_t k=0;k<n;++k){ m1[k]=a1[k]*D; m2[k]=a2[k]*D; }

    // encrypt under KPq (Q basis)
    std::vector<uint64_t> c0a,c1a,c0b,c1b;
    gpufhe::encrypt_host(c0a,c1a,m1,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,303);
    gpufhe::encrypt_host(c0b,c1b,m2,KPq.pkA,KPq.pkB,n,mod,root,ns,3.19,304);

    // ---- tensor (eval, per tower): t0=c0a*c0b, t1=c0a*c1b+c1a*c0b, t2=c1a*c1b
    std::vector<uint64_t> t0((size_t)sizeQ*n),t1((size_t)sizeQ*n),t2((size_t)sizeQ*n);
    for(uint32_t t=0;t<sizeQ;++t){ uint64_t q=mod[t];
        for(uint32_t k=0;k<n;++k){ size_t x=(size_t)t*n+k;
            t0[x]=mm(c0a[x],c0b[x],q);
            t1[x]=am(mm(c0a[x],c1b[x],q),mm(c1a[x],c0b[x],q),q);
            t2[x]=mm(c1a[x],c1b[x],q); } }

    // ---- relin: keyswitch c2 (=t2) over Q, get (ba0,ba1), add to (t0,t1)
    auto R=gpufhe::keyswitch_core_resident(t2, K);
    std::vector<uint64_t> r0((size_t)sizeQ*n),r1((size_t)sizeQ*n);
    for(uint32_t t=0;t<sizeQ;++t){ uint64_t q=mod[t];
        for(uint32_t k=0;k<n;++k){ size_t x=(size_t)t*n+k;
            r0[x]=am(t0[x],R.ba0[x],q); r1[x]=am(t1[x],R.ba1[x],q); } }

    // ---- decrypt (r0,r1) under KPq.s
    std::vector<int64_t> dec;
    gpufhe::decrypt_host(dec,r0,r1,KPq.s,n,mod,root);

    // expected: negacyclic product a1*a2 (mod X^n+1), scaled by D^2
    // dec ~ (a1*a2)*D^2 ; recover round(dec/D^2)
    std::vector<int64_t> expect(n,0);
    for(uint32_t i=0;i<n;++i)if(a1[i])for(uint32_t j=0;j<n;++j)if(a2[j]){
        uint32_t idx=(i+j)%n; int64_t sign=((i+j)>=n)?-1:1;
        expect[idx]+=sign*a1[i]*a2[j]; }

    int bad=0; int64_t maxerr=0;
    long double D2=(long double)D*(long double)D;
    for(uint32_t k=0;k<n;++k){ int64_t r=(int64_t)llroundl((long double)dec[k]/D2);
        int64_t e=r-expect[k]; if(e<0)e=-e; if(e>maxerr)maxerr=e; if(r!=expect[k])++bad; }
    std::cout<<"n="<<n<<" D=2^25 mismatches="<<bad<<" maxerr(after /D^2)="<<maxerr<<"\n";
    std::cout<<"  expect[0..4] = "; for(int k=0;k<5;++k)std::cout<<expect[k]<<" "; std::cout<<"\n";
    std::cout<<"  got[0..4]    = "; for(int k=0;k<5;++k)std::cout<<llroundl((long double)dec[k]/D2)<<" "; std::cout<<"\n";
    if(maxerr==0){ std::cout<<"[PASS] full native CKKS EvalMult numerically correct end-to-end\n"; return 0; }
    std::cout<<"[FAIL]\n"; return 1;
}
