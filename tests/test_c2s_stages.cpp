// Gate for the multi-stage FFT factorisation of CoeffsToSlots, in plain host
// arithmetic (no encryption). Proves the C++ diagonal generation matches the
// dense matrix A[i][c] = (2/N)*zeta^(-i*5^c) up to the bit-reversal permutation.
//
// Stage s: h = S>>(s+1), blocks of 2h, twiddle at position k is pts[k]^(2^s),
// pts[k] = zeta^(5^k).  Stage s flips index bit (lg-s-1), so merging r
// consecutive stages flips r consecutive bits => block-diagonal over blocks of
// 2^r indices.  Diagonals per merged stage: 2^r for the first, 2^(r+1)-1 after.
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cstdint>
using cd=std::complex<double>;

int main(int argc,char**argv){
    const uint32_t n = (argc>1)? std::stoul(argv[1]) : 1024;
    const uint32_t S=n/2, M=2*n;
    uint32_t lg=0; while((1u<<lg)<S) ++lg;
    const uint32_t L = (argc>2)? std::stoul(argv[2]) : 3;
    if(lg%L){ std::cout<<"L must divide log2(S)="<<lg<<"\n"; return 1; }
    const uint32_t R=lg/L;

    std::vector<uint64_t> rk(S); { uint64_t r=1; for(uint32_t k=0;k<S;++k){rk[k]=r;r=(r*5)%M;} }
    auto tw=[&](uint32_t k,uint32_t s)->cd{                    // pts[k]^(2^s), exact
        uint64_t e=(rk[k]<<s)%M; return std::polar(1.0, 2*M_PI*(double)e/(double)M); };

    // apply merged stage g to a host vector (the r butterflies, in order)
    auto applyMerged=[&](std::vector<cd>& v,uint32_t g){
        for(uint32_t s=g*R;s<(g+1)*R;++s){
            uint32_t h=S>>(s+1); std::vector<cd> o(S);
            for(uint32_t base=0;base<S;base+=2*h)
                for(uint32_t k=0;k<h;++k){
                    cd a=v[base+k], b=v[base+k+h], t=tw(k,s);
                    o[base+k]     = (a+b)*0.5;
                    o[base+k+h]   = (a-b)/(2.0*t);
                }
            v.swap(o);
        } };

    // reference: dense A, then bit-reverse the output
    auto Aent=[&](uint32_t i,uint32_t c)->cd{
        return (2.0/(double)n)*std::polar(1.0, -M_PI*(double)((i*rk[c])%M)/(double)n); };
    auto brev=[&](uint32_t i)->uint32_t{ uint32_t r=0;
        for(uint32_t b=0;b<lg;++b) if(i&(1u<<b)) r|=1u<<(lg-1-b); return r; };

    // random probe vector
    std::vector<cd> z(S);
    for(uint32_t k=0;k<S;++k) z[k]={std::sin(0.7*k+0.2), std::cos(0.31*k)};

    std::vector<cd> ref(S,cd{0,0});
    for(uint32_t i=0;i<S;++i){ cd a{0,0};
        for(uint32_t c=0;c<S;++c) a+=Aent(i,c)*z[c];
        ref[brev(i)]=a; }                                   // stages produce bitrev order

    std::vector<cd> got=z;
    for(uint32_t g=0;g<L;++g) applyMerged(got,g);

    double e=0; for(uint32_t i=0;i<S;++i) e=std::max(e,std::abs(got[i]-ref[i]));
    std::cout<<"n="<<n<<" S="<<S<<" L="<<L<<" radix=2^"<<R<<"\n";
    std::cout<<"  || stages(z) - bitrev(A z) || = "<<e<<"\n";

    // ---- FORWARD stages (= SlotsToCoeffs): reverse stage order, inverted
    // butterfly. Consumes the bit-reversed vector the inverse stages produce,
    // so applying inverse then forward must be the identity.
    auto applyMergedFwd=[&](std::vector<cd>& v,uint32_t g){
        for(int st=(int)((g+1)*R)-1; st>=(int)(g*R); --st){
            uint32_t h=S>>(st+1); std::vector<cd> o(S);
            for(uint32_t base=0;base<S;base+=2*h)
                for(uint32_t k=0;k<h;++k){
                    cd a=v[base+k], b=v[base+k+h], t=tw(k,(uint32_t)st);
                    o[base+k]   = a + t*b;
                    o[base+k+h] = a - t*b;
                }
            v.swap(o);
        } };
    {   std::vector<cd> rt=z;
        for(uint32_t g=0;g<L;++g) applyMerged(rt,g);          // C2S
        for(int g=(int)L-1; g>=0; --g) applyMergedFwd(rt,(uint32_t)g);  // S2C
        double e2=0; for(uint32_t i=0;i<S;++i) e2=std::max(e2,std::abs(rt[i]-z[i]));
        std::cout<<"  || S2C(C2S(z)) - z ||        = "<<e2<<"   (round trip)\n";
        if(e2>1e-9){ std::cout<<"[FAIL] forward stages are not the inverse\n"; return 1; } }

    // diagonal census per merged stage (via unit vectors; fine at sandbox sizes)
    uint32_t total=0;
    for(uint32_t g=0;g<L;++g){
        std::vector<uint8_t> nz(S,0);
        for(uint32_t j=0;j<S;++j){
            std::vector<cd> u(S,cd{0,0}); u[j]=1;
            applyMerged(u,g);
            for(uint32_t i=0;i<S;++i) if(std::abs(u[i])>1e-12) nz[(j+S-i)%S]=1;
        }
        uint32_t d=0; for(uint32_t i=0;i<S;++i) d+=nz[i];
        std::cout<<"  merged stage "<<g<<": "<<d<<" diagonals\n"; total+=d;
    }
    std::cout<<"  TOTAL "<<total<<" diagonals vs "<<S<<" dense  ("<<(double)S/total<<"x fewer)\n";
    if(e<1e-9){ std::cout<<"[PASS] multi-stage factorisation matches the dense transform\n"; return 0; }
    std::cout<<"[FAIL]\n"; return 1;
}
