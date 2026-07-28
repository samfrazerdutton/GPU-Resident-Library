// Native parameter generation: NTT-friendly primes (q = 1 mod 2n), primitive
// 2n-th roots, and the rescale constant pair (qlInvModq, QlQlInvModqlDivqlModq)
// computed WITHOUT bignum. Makes the library fully OpenFHE-free and unlocks
// small rings (n=1024) for fast bootstrap iteration.
#include <cstdint>
#include <vector>
#include <stdexcept>
namespace gpufhe {
namespace {
uint64_t mm(uint64_t a,uint64_t b,uint64_t q){return(uint64_t)(((unsigned __int128)a*b)%q);}
// overflow-safe mulmod for up-to-120-bit moduli (double-and-add; intermediates < 2*m)
unsigned __int128 mulmod128(unsigned __int128 a, unsigned __int128 b, unsigned __int128 m){
    a%=m; unsigned __int128 r=0;
    while(b){ if(b&1){ r+=a; if(r>=m)r-=m; } a<<=1; if(a>=m)a-=m; b>>=1; }
    return r; }
uint64_t powmod(uint64_t b,uint64_t e,uint64_t q){unsigned __int128 r=1,bb=b%q;while(e){if(e&1)r=(r*bb)%q;bb=(bb*bb)%q;e>>=1;}return(uint64_t)r;}
bool is_prime(uint64_t x){ if(x<2)return false;
    for(uint64_t a:{2ull,3ull,5ull,7ull,11ull,13ull,17ull,19ull,23ull,29ull,31ull,37ull}){
        if(x%a==0) return x==a; }
    uint64_t d=x-1; int s=0; while(!(d&1)){d>>=1;++s;}
    for(uint64_t a:{2ull,3ull,5ull,7ull,11ull,13ull,17ull,19ull,23ull,29ull,31ull,37ull}){
        uint64_t v=powmod(a,d,x); if(v==1||v==x-1)continue;
        bool ok=false; for(int i=1;i<s;++i){ v=mm(v,v,x); if(v==x-1){ok=true;break;} }
        if(!ok)return false; }
    return true; }
} // anon

// first `count` primes of ~`bits` bits with q = 1 mod 2n, descending from 2^bits
void native_primes(std::vector<uint64_t>& out, uint32_t count, uint32_t bits, uint32_t n,
                   const std::vector<uint64_t>& avoid)
{
    const uint64_t M=2ull*n;
    uint64_t q=((1ull<<bits)/M)*M+1;
    while(out.size()<count){
        if(q<(1ull<<(bits-1))) throw std::runtime_error("prime search exhausted");
        bool skip=false; for(uint64_t a:avoid) if(a==q) skip=true;
        if(!skip && is_prime(q)) out.push_back(q);
        q-=M; }
}

// primitive 2n-th root of unity mod q (self-consistent native world)
uint64_t native_root(uint32_t n, uint64_t q){
    const uint64_t M=2ull*n;
    for(uint64_t g=2;g<q;++g){
        uint64_t c=powmod(g,(q-1)/M,q);
        if(powmod(c,n,q)==q-1) return c; }
    throw std::runtime_error("no 2n-th root");
}

// rescale constants for dropping tower `last` onto survivors [0,last):
//   s1[i] = qLast^{-1} mod q_i
//   s2[i] = floor(Ql * (Ql^{-1} mod qLast) / qLast) mod q_i,  Ql = prod q_0..q_{last-1}
// floor via exactness: Ql*u = qLast*D + 1  =>  D=(Ql*u-1)/qLast, tracked mod (qLast*q_i).
void native_rescale_consts(std::vector<uint64_t>& s1, std::vector<uint64_t>& s2,
                           const std::vector<uint64_t>& mod, uint32_t last)
{
    uint64_t ql=mod[last];
    s1.resize(last); s2.resize(last);
    // u = Ql^{-1} mod ql: fold Ql mod ql, invert
    uint64_t QlModql=1%ql;
    for(uint32_t t=0;t<last;++t) QlModql=mm(QlModql,mod[t]%ql,ql);
    uint64_t u=powmod(QlModql,ql-2,ql);
    for(uint32_t i=0;i<last;++i){ uint64_t qi=mod[i];
        s1[i]=powmod(ql%qi,qi-2,qi);
        unsigned __int128 mod2=(unsigned __int128)ql*qi;   // < 2^120 for 60-bit primes
        unsigned __int128 Ql2=1%mod2;
        for(uint32_t t=0;t<last;++t) Ql2=mulmod128(Ql2,mod[t],mod2);   // overflow-safe
        unsigned __int128 prod=mulmod128(Ql2,u,mod2);       // Ql*u mod (ql*qi)
        // (Ql*u - 1) is divisible by ql; D mod qi = ((prod - 1)/ql) mod qi needs
        // the true quotient mod qi: D = (Ql*u-1)/ql. Note Ql*u-1 = ql*D, so
        // (prod - 1) mod (ql*qi) = ql*D mod (ql*qi) = ql*(D mod qi).
        unsigned __int128 pm1=(prod==0)? mod2-1 : prod-1;
        s2[i]=(uint64_t)((pm1/ql)%qi);
    }
}
} // namespace gpufhe
