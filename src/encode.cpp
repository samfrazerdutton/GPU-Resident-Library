// CKKS canonical encode/decode (host, double precision). Slots z_0..z_{N/2-1}
// (complex) <-> integer coefficient poly, scale Delta.
//   encode: m_j = round( (2*Delta/N) * sum_k Re( z_k * e^{-i*pi*j*r_k/N} ) )
//   decode: z_k = (1/Delta) * sum_j m_j * e^{+i*pi*j*r_k/N}
// with r_k = 5^k mod 2N (the CKKS rotation-group ordering, conjugate-symmetric
// half). Twiddles from a precomputed 2N table; exponents indexed mod 2N.
//
// GOD PROMPT "CKKS PHASE 11": measured the dominant end-to-end bottleneck
// for a real encode->encrypt->EvalMult->decrypt->decode round trip --
// NOT any CUDA kernel, but these two O(N^2) host loops (bench/
// bench_evalmult_e2e_breakdown.cpp: at n=8192, encode+decode together took
// ~464ms of a ~523ms total round trip, vs ~6.4ms for the ENTIRE GPU-resident
// tensor+keyswitch+combine+rescale chain -- see research/EXPERIMENT_LEDGER.md
// entry #44). Parallelized across hardware threads: each output element
// (m[j] for encode, z[k] for decode) is an independent reduction over
// READ-ONLY inputs (z/tw/r for encode; m/tw/r for decode) with no
// cross-element dependency, so splitting the outer loop into disjoint,
// contiguous index ranges computed by separate threads changes NOTHING
// about the arithmetic any single output element performs -- same
// operations, same summation order, same rounding, per element. This is
// NOT an algorithmic change (still O(N^2) total work) and is NOT a change
// to the function's signature or semantics -- every caller gets
// BIT-IDENTICAL results, verified via tests/test_encode_decode_parallel_
// matches_sequential.cpp (an independent, OLD single-threaded
// reimplementation of this exact file's pre-Phase-11 algorithm, kept as
// a permanent oracle distinct from the actual (now-parallel) production
// code, per the governing directive's "do not make the production
// implementation its own oracle" requirement).
#include <cstdint>
#include <vector>
#include <complex>
#include <cmath>
#include <thread>
#include <algorithm>
namespace gpufhe {
namespace {
// Below this N, thread-pool spawn/join overhead is not worth it -- matches
// this project's established convention of small toy configurations in
// unit tests (e.g. n=4/n=16) staying on the cheap, simple path.
constexpr uint32_t kParallelThreshold = 256;

unsigned worker_count(uint32_t units) {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    return std::min<unsigned>(hw, units);
}

// Runs `body(i0,i1)` across disjoint, contiguous [i0,i1) ranges partitioning
// [0,total) over worker_count(total) threads (or inline, sequentially, if
// total is below the parallel threshold or only one worker is available).
template <typename Body>
void parallel_for(uint32_t total, Body&& body) {
    if (total < kParallelThreshold) { body(0u, total); return; }
    unsigned nt = worker_count(total);
    if (nt <= 1) { body(0u, total); return; }
    uint32_t chunk = (total + nt - 1) / nt;
    std::vector<std::thread> pool;
    pool.reserve(nt);
    for (unsigned ti = 0; ti < nt; ++ti) {
        uint32_t i0 = ti * chunk;
        uint32_t i1 = std::min(total, i0 + chunk);
        if (i0 >= i1) break;
        pool.emplace_back(body, i0, i1);
    }
    for (auto& th : pool) th.join();
}
} // namespace

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
    parallel_for(N, [&](uint32_t j0, uint32_t j1){
        for(uint32_t j=j0;j<j1;++j){ double acc=0;
            for(uint32_t k=0;k<S;++k){
                uint64_t t=((uint64_t)j*r[k])%M;
                acc += z[k].real()*tw[t].real() - z[k].imag()*tw[t].imag(); }
            m[j]=(int64_t)llround(sc*acc); }
    });
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
    parallel_for(S, [&](uint32_t k0, uint32_t k1){
        for(uint32_t k=k0;k<k1;++k){ std::complex<double> acc{0,0};
            for(uint32_t j=0;j<N;++j){
                uint64_t t=((uint64_t)j*r[k])%M;
                acc += (double)m[j]*tw[t]; }
            z[k]=acc/Delta; }
    });
}
} // namespace gpufhe
