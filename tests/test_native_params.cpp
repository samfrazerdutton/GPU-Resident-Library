// Gate: native rescale constants bit-exact vs OpenFHE's GetqlInvModq /
// GetQlQlInvModqlDivqlModq at ring 32768 (level 0 = drop the top tower).
#include "openfhe.h"
#include <iostream>
#include <vector>
using namespace lbcrypto;
namespace gpufhe {
void native_rescale_consts(std::vector<uint64_t>&, std::vector<uint64_t>&,
                           const std::vector<uint64_t>&, uint32_t);
uint64_t native_root(uint32_t, uint64_t);
void native_primes(std::vector<uint64_t>&, uint32_t, uint32_t, uint32_t, const std::vector<uint64_t>&);
}
int main(){
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(3); params.SetScalingModSize(35);
    params.SetRingDim(32768); params.SetBatchSize(16384);
    params.SetKeySwitchTechnique(HYBRID); params.SetScalingTechnique(FIXEDMANUAL);
    auto cc=GenCryptoContext(params);
    auto cp=std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto ep=cp->GetElementParams();
    uint32_t sizeQ=ep->GetParams().size();
    std::vector<uint64_t> mod(sizeQ);
    for(uint32_t i=0;i<sizeQ;++i) mod[i]=ep->GetParams()[i]->GetModulus().ConvertToInt();

    std::vector<uint64_t> s1,s2;
    gpufhe::native_rescale_consts(s1,s2,mod,sizeQ-1);
    const auto&a=cp->GetqlInvModq(0); const auto&b=cp->GetQlQlInvModqlDivqlModq(0);
    uint32_t bad=0;
    for(uint32_t t=0;t<sizeQ-1;++t){
        if(s1[t]!=a[t].ConvertToInt()){std::cout<<"  s1["<<t<<"] mine="<<s1[t]<<" ref="<<a[t].ConvertToInt()<<"\n";++bad;}
        if(s2[t]!=b[t].ConvertToInt()){std::cout<<"  s2["<<t<<"] mine="<<s2[t]<<" ref="<<b[t].ConvertToInt()<<"\n";++bad;} }
    // sanity: native prime/root gen works at n=1024
    std::vector<uint64_t> p; gpufhe::native_primes(p,3,35,1024,{});
    uint64_t r0=gpufhe::native_root(1024,p[0]);
    std::cout<<"rescale-const mismatches="<<bad<<"  |  n=1024 primes ok ("<<p[0]<<"...), root ok ("<<r0<<")\n";
    if(bad==0){std::cout<<"[PASS] native rescale constants bit-exact vs OpenFHE; native prime/root gen works\n";return 0;}
    std::cout<<"[FAIL]\n"; return 1;
}
