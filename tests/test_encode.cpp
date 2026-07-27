// Gate 1: round-trip decode(encode(z)) == z (~1e-6).
// Gate 2: encode matches OpenFHE MakeCKKSPackedPlaintext coefficients (+-1
// rounding tolerance; both are double-precision rounded to int).
#include "openfhe.h"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
using namespace lbcrypto;
namespace gpufhe {
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void decode_host(std::vector<std::complex<double>>&, const std::vector<int64_t>&, uint32_t, double);
}
int main(){
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(3); params.SetScalingModSize(35);
    params.SetRingDim(32768); params.SetBatchSize(16384);
    params.SetKeySwitchTechnique(HYBRID); params.SetScalingTechnique(FIXEDMANUAL);
    auto cc=GenCryptoContext(params); cc->Enable(PKE);
    uint32_t N=32768, S=N/2; double Delta=std::pow(2.0,35);

    std::vector<double> xr(S); std::vector<std::complex<double>> z(S);
    for(uint32_t k=0;k<S;++k){ xr[k]=std::sin(0.001*k)*0.5; z[k]={xr[k],0.0}; }

    std::vector<int64_t> m; gpufhe::encode_host(m,z,N,Delta);
    std::vector<std::complex<double>> zz; gpufhe::decode_host(zz,m,N,Delta);
    double rerr=0; for(uint32_t k=0;k<S;++k) rerr=std::max(rerr,std::abs(zz[k].real()-xr[k]));
    std::cout<<"round-trip max err = "<<rerr<<"\n";

    auto pt=cc->MakeCKKSPackedPlaintext(xr);
    auto el=pt->GetElement<DCRTPoly>(); el.SetFormat(Format::COEFFICIENT);
    const auto& p0=el.GetAllElements()[0]; uint64_t q0=p0.GetModulus().ConvertToInt();
    int64_t maxd=0; uint32_t big=0;
    for(uint32_t j=0;j<N;++j){ uint64_t c=p0[j].ConvertToInt();
        int64_t ref=(c>q0/2)?((int64_t)c-(int64_t)q0):(int64_t)c;
        int64_t d=m[j]-ref; if(d<0)d=-d; if(d>maxd)maxd=d; if(d>1)++big; }
    std::cout<<"vs OpenFHE encoder: maxdiff="<<maxd<<" coeffs>1 off: "<<big<<"\n";
    if(rerr<1e-5 && big==0){ std::cout<<"[PASS] canonical encode/decode correct (round-trip + matches OpenFHE +-1)\n"; return 0; }
    std::cout<<"[FAIL]\n"; return 1;
}
