// CKKS canonical encode/decode (host, double precision). Slots z_0..z_{N/2-1}
// (complex) <-> integer coefficient poly, scale Delta.
//   encode: m_j = round( (2*Delta/N) * sum_k Re( z_k * e^{-i*pi*j*r_k/N} ) )
//   decode: z_k = (1/Delta) * sum_j m_j * e^{+i*pi*j*r_k/N}
// with r_k = 5^k mod 2N (the CKKS rotation-group ordering, conjugate-symmetric
// half). Twiddles from a precomputed 2N table; exponents indexed mod 2N.
#include <cstdint>
#include <vector>
#include <complex>
#include <cmath>
namespace gpufhe {

void encode_host(std::vector<int64_t>& m, const std::vector<std::complex<double>>& z,
                 uint32_t N, double Delta)
{
    const uint32_t M=2*N, S=N/2;
    std::vector<std::complex<double>> tw(M);
    for(uint32_t t=0;t<M;++t) tw[t]=std::polar(1.0, -M_PI*(double)t/(double)N);
    std::vector<uint64_t> r(S); uint64_t rk=1;
    for(uint32_t k=0;k<S;++k){ r[k]=rk; rk=(rk*5)%M; }
    m.assign(N,0);
    const double sc=2.0*Delta/(double)N;
    for(uint32_t j=0;j<N;++j){ double acc=0;
        for(uint32_t k=0;k<S;++k){
            uint64_t t=((uint64_t)j*r[k])%M;
            acc += z[k].real()*tw[t].real() - z[k].imag()*tw[t].imag(); }
        m[j]=(int64_t)llround(sc*acc); }
}

void decode_host(std::vector<std::complex<double>>& z, const std::vector<int64_t>& m,
                 uint32_t N, double Delta)
{
    const uint32_t M=2*N, S=N/2;
    std::vector<std::complex<double>> tw(M);
    for(uint32_t t=0;t<M;++t) tw[t]=std::polar(1.0, +M_PI*(double)t/(double)N);
    std::vector<uint64_t> r(S); uint64_t rk=1;
    for(uint32_t k=0;k<S;++k){ r[k]=rk; rk=(rk*5)%M; }
    z.assign(S,{0,0});
    for(uint32_t k=0;k<S;++k){ std::complex<double> acc{0,0};
        for(uint32_t j=0;j<N;++j){
            uint64_t t=((uint64_t)j*r[k])%M;
            acc += (double)m[j]*tw[t]; }
        z[k]=acc/Delta; }
}
} // namespace gpufhe
