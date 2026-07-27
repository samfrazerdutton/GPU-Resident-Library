// Encrypt/decrypt round-trip: encrypt a known integer-coeff message under the
// native pubkey, decrypt via c0+c1*s, verify recovery (small noise). Proves the
// encryption layer end-to-end with native keygen.
#include "keygen.h"
#include <iostream>
#include <vector>
#include <random>
// borrow OpenFHE just for the moduli/roots (a real param set), not for crypto.
#include "openfhe.h"
using namespace lbcrypto;

int main(){
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(3); params.SetScalingModSize(50);
    params.SetRingDim(32768); params.SetBatchSize(16384);
    params.SetKeySwitchTechnique(HYBRID);
    auto cc=GenCryptoContext(params);
    auto cp=std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto ep=cp->GetElementParams();
    uint32_t n=ep->GetRingDimension(), sizeQ=ep->GetParams().size();
    uint64_t ns=cp->GetNoiseScale();
    std::vector<uint64_t> mod(sizeQ),root(sizeQ);
    for(uint32_t i=0;i<sizeQ;++i){ mod[i]=ep->GetParams()[i]->GetModulus().ConvertToInt();
        root[i]=ep->GetParams()[i]->GetRootOfUnity().ConvertToInt(); }

    auto KP=gpufhe::keygen_host(n,mod,root,ns,3.19,555);

    // known message: small ints in first 8 coeffs, zeros elsewhere
    std::vector<int64_t> m(n,0);
    const int64_t DELTA=(int64_t)1<<40;
    std::vector<int64_t> mval(n,0);
    for(int k=0;k<8;++k){ mval[k]=(k*7)%11 - 5; m[k]=mval[k]*DELTA; }

    std::vector<uint64_t> c0,c1;
    gpufhe::encrypt_host(c0,c1,m,KP.pkA,KP.pkB,n,mod,root,ns,3.19,556);
    std::vector<int64_t> mout;
    gpufhe::decrypt_host(mout,c0,c1,KP.s,n,mod,root);

    // recover: round(dec/DELTA) should equal mval
    int bad=0; int64_t maxerr=0;
    for(uint32_t k=0;k<n;++k){ double d=(double)mout[k]/(double)DELTA; int64_t r=(int64_t)llround(d);
        int64_t e=r-mval[k]; if(e<0)e=-e; if(e>maxerr)maxerr=e; if(r!=mval[k])++bad; }
    std::cout<<"n="<<n<<" recovered-mismatches="<<bad<<" maxerr(after /DELTA)="<<maxerr<<"\n";
    std::cout<<"  mval[0..7] = "; for(int k=0;k<8;++k)std::cout<<mval[k]<<" "; std::cout<<"\n";
    std::cout<<"  rec[0..7]  = "; for(int k=0;k<8;++k)std::cout<<llround((double)mout[k]/(double)DELTA)<<" "; std::cout<<"\n";
    if(maxerr==0){ std::cout<<"[PASS] encrypt/decrypt round-trip exact after rescaling\n"; return 0; }
    std::cout<<"[FAIL]\n"; return 1;
}
