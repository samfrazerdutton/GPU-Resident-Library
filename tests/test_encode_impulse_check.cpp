// GOD PROMPT "CKKS ENGINE: RESOLVE POLYEVAL CORRECTNESS" round -- scratch
// diagnostic (Priority A, Step C: independent quantitative reference).
// Verifies by direct inspection, not hand-derivation, what encode_host
// actually produces for a constant-1.0 input at n=1024, Delta=2^50: is the
// plaintext coefficient array really a pure impulse (m[0]=Delta, all other
// positions exactly 0), as assumed when predicting test_polyeval's raw
// tensor-product residue mod q0?
#include <vector>
#include <complex>
#include <cstdint>
#include <iostream>
#include <cmath>
namespace gpufhe {
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
}
using cd=std::complex<double>;
int main(){
    const uint32_t n=1024,S=n/2;
    const double Delta=std::pow(2.0,50);
    std::vector<cd> z(S, cd{1.0,0});
    std::vector<int64_t> m; gpufhe::encode_host(m,z,n,Delta);
    std::cout<<"Delta="<<(int64_t)Delta<<"\n";
    std::cout<<"m[0]="<<m[0]<<" (expect Delta="<<(int64_t)Delta<<")\n";
    int64_t maxabs_rest=0; uint32_t argmax=0;
    for(uint32_t j=1;j<n;++j) if(std::abs(m[j])>maxabs_rest){maxabs_rest=std::abs(m[j]);argmax=j;}
    std::cout<<"max|m[j]| for j=1..n-1: "<<maxabs_rest<<" at j="<<argmax<<" (expect 0 if pure impulse)\n";
    int64_t sum_rest=0; for(uint32_t j=1;j<n;++j) sum_rest+=m[j];
    std::cout<<"sum of m[j] for j=1..n-1: "<<sum_rest<<"\n";
    return 0;
}
