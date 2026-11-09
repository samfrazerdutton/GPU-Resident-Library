// GOD PROMPT "CKKS PHASE 12": O(N log N) FFT-based replacement for
// src/encode.cpp's O(N^2) encode_host/decode_host, derived and
// independently verified against numpy's FFT (a completely separate
// implementation) BEFORE any C++ was written -- see research/
// FFT_ENCODING_DERIVATION.md for the full derivation. src/encode.cpp's
// own encode_host/decode_host are DELIBERATELY left untouched, both as
// the correctness oracle these new functions are tested against and as
// a permanent fallback.
//
// === The exact transform (restated from src/encode.cpp's own header
// comment, which remains authoritative) ===
//   M = 2N, S = N/2
//   r_k = 5^k mod M                              (k = 0..S-1)
//   encode: m[j] = round( (2*Delta/N) * Re( sum_k z_k * e^{-i*pi*j*r_k/N} ) )
//   decode: z[k] = (1/Delta) * sum_j m[j] * e^{+i*pi*j*r_k/N}
//
// === Why a single size-M=2N FFT suffices (the derivation) ===
// Let zeta = e^{i*pi/N} = e^{2*pi*i/M}, a primitive M-th root of unity.
// decode's sum is exactly m(zeta^{r_k}) (m(X) = sum_j m_j X^j), so DECODE
// is "evaluate m at zeta^{r_k} for k=0..S-1". A single size-M DFT of the
// zero-padded array (m_0,...,m_{N-1},0,...,0) gives, using the standard
// forward-DFT convention X[t] = sum_j x[j] * e^{-2*pi*i*j*t/M}:
//   X[t] = sum_j m_j * e^{-i*pi*j*t/N} = m(zeta^{-t})
// so X[t] evaluates m at zeta^{-t} -- the FULL set of M evaluations, at
// every power of zeta, is obtained from ONE size-M FFT; decode simply
// GATHERS the S needed bins at index t = (M - r_k) % M (since we want
// zeta^{r_k} = zeta^{-t} => t = -r_k mod M), then divides by Delta.
//
// ENCODE is the corresponding inverse: scatter z_k and its conjugate
// into the two bins of a size-M array Z that a decode of some m would
// read FROM (Z[(M-r_k)%M] = z_k, Z[r_k] = conj(z_k), all other M-N
// bins zero -- these are forced by m's realness, matching decode's own
// GATHER exactly in reverse), then take the UNNORMALIZED inverse DFT
// x[j] = sum_t Z[t] * e^{+2*pi*i*j*t/M}. Because each of the S pairs
// contributes z_k*e^{-i*pi*j*r_k/N} + conj(z_k)*e^{+i*pi*j*r_k/N} =
// 2*Re(z_k*e^{-i*pi*j*r_k/N}), x[j] = 2 * Re(sum_k z_k*e^{-i*pi*j*r_k/N})
// -- i.e. EXACTLY TWICE the quantity encode's own Re(...) term needs, so
// m_j = round( (Delta/N) * Re(x[j]) ), NOT (2*Delta/N) -- the factor of
// 2 is already absorbed by the scatter. This exact relationship (and
// the resulting Delta/N, not 2*Delta/N, scale) was confirmed empirically
// against numpy to ZERO error (not just small) across every tested N,
// not assumed from the algebra alone.
//
// Only the first N of the M=2N inverse-DFT outputs are used for encode
// (x[N..M-1] are discarded); only S=N/2 of the M forward-DFT outputs are
// used for decode (gathered, not all M). This "zero-pad to 2N, run one
// standard FFT, use only half the results" approach has a built-in 2x
// constant-factor overhead relative to a tighter N-point negacyclic-
// twisted FFT, but requires NO additional derivation beyond a textbook
// radix-2 FFT and the SAME r_k array src/encode.cpp already computes --
// deliberately the SIMPLEST correct approach, chosen over a tighter
// N-point construction per the governing directive's own "evaluate the
// simplest correct approach first" instruction. The asymptotic win
// (O(N log N) vs O(N^2)) dominates this constant factor at every N this
// project targets.
#include <cstdint>
#include <vector>
#include <complex>
#include <cmath>
#include <map>
#include <utility>
#include <mutex>

namespace gpufhe {
namespace {

using cd = std::complex<double>;

// GOD PROMPT "CKKS PHASE 28": this function's cache (and slot_idx's below)
// originally used a plain `static std::map` with no synchronization --
// the SAME pattern class as src/keyswitch_resident.cpp's g_rtab/upc, which
// Phase 25's own concurrency audit found to be a LATENT (tested, never
// triggered) risk. A real std::thread-based stress test (tests/
// test_encode_fft_concurrency.cpp) initially appeared to show this cache
// was WORSE than latent -- reproducing heap corruption 5/5 runs. root-
// caused with AddressSanitizer to a bug in the TEST's own inputs (it
// called encode_host_fft with non-power-of-2 N, violating this file's
// documented M=2N-must-be-a-power-of-2 precondition and overrunning the
// FFT buffers -- see that test's own correctness-note comment and
// research/ENCODE_FFT_PROMOTION.md for the full isolation). Once the test
// was fixed to use only valid power-of-2 N, this cache's race is LATENT,
// exactly like g_rtab/upc -- not observed to misbehave, but still
// concurrent std::map mutation, which is undefined behavior per the C++
// standard regardless of whether any given run happens to observe a
// visible failure. Fixed with a single mutex guarding both this function
// and slot_idx() below as cheap, unconditional hardening: lookups/inserts
// are trivial relative to the FFT itself (a handful of map operations vs
// O(M log M) arithmetic), so a lock here costs nothing that matters while
// closing a real (if latent) hazard. Re-verified after this fix:
// tests/test_encode_fft_concurrency.cpp, 10 consecutive runs, 0 crashes.
std::mutex& fft_cache_mutex() { static std::mutex m; return m; }

// Cached per size-M forward-FFT twiddle table: root[t] = e^{-2*pi*i*t/M}
// for t = 0..M/2-1 (enough to index every butterfly stage via a stride),
// precomputed ONCE via direct trig calls (not iterative multiplication,
// avoiding any compounding rounding drift across O(log M) stages) and
// cached for the process lifetime, matching this codebase's established
// pattern (src/evalkeygen.cpp's ek_dev_tab, src/encdec.cpp's enc_dev_tab).
// Guarded by fft_cache_mutex() -- see that function's own comment for why.
struct FftTwiddles { std::vector<cd> root; };
FftTwiddles& fft_twiddles(uint32_t M) {
    std::lock_guard<std::mutex> lock(fft_cache_mutex());
    static std::map<uint32_t, FftTwiddles> cache;
    auto it = cache.find(M);
    if (it != cache.end()) return it->second;
    FftTwiddles T;
    T.root.resize(M / 2);
    for (uint32_t t = 0; t < M / 2; ++t) {
        double ang = -2.0 * M_PI * (double)t / (double)M;
        T.root[t] = cd(std::cos(ang), std::sin(ang));
    }
    return cache.emplace(M, std::move(T)).first->second;
}

// Standard iterative radix-2 Cooley-Tukey FFT, decimation-in-time,
// forward convention X[t] = sum_j x[j] * e^{-2*pi*i*j*t/M} (matches
// numpy.fft.fft exactly -- verified in this round's Python derivation,
// research/FFT_ENCODING_DERIVATION.md). `a.size()` must be a power of 2
// (M = 2N; N is a power of 2 everywhere in this project already).
void fft_forward_inplace(std::vector<cd>& a) {
    const size_t M = a.size();
    // Bit-reversal permutation.
    for (size_t i = 1, j = 0; i < M; ++i) {
        size_t bit = M >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    const FftTwiddles& T = fft_twiddles((uint32_t)M);
    for (size_t len = 2; len <= M; len <<= 1) {
        size_t half = len / 2;
        size_t stride = M / len; // index into T.root for this stage's twiddles
        for (size_t i = 0; i < M; i += len) {
            for (size_t k = 0; k < half; ++k) {
                cd w = T.root[k * stride];
                cd u = a[i + k];
                cd v = a[i + k + half] * w;
                a[i + k] = u + v;
                a[i + k + half] = u - v;
            }
        }
    }
}

// Unnormalized inverse: x[j] = sum_t Z[t] * e^{+2*pi*i*j*t/M}, via the
// standard conjugate identity (avoids a second FFT kernel): conj(FFT(conj(Z)))
// equals exactly this unnormalized inverse sum (no 1/M factor) -- verified
// against numpy's `ifft(Z)*M` in the Python derivation.
std::vector<cd> fft_unnormalized_inverse(std::vector<cd> Z) {
    for (auto& v : Z) v = std::conj(v);
    fft_forward_inplace(Z);
    for (auto& v : Z) v = std::conj(v);
    return Z;
}

// Cached per-N: the r_k=5^k mod M array and the gather index array
// gidx_k=(M-r_k)%M decode/encode both need -- identical derivation to
// src/encode.cpp's own r[] array, cached here since it is pure
// read-only data depending only on N.
struct SlotIdx { std::vector<uint64_t> r, gidx; };
SlotIdx& slot_idx(uint32_t N) {
    // Guarded by fft_cache_mutex() -- same confirmed-active race as
    // fft_twiddles() above (see that function's comment); this cache has
    // the identical find()-then-emplace() pattern and is exercised by the
    // same concurrency reproducer (tests/test_encode_fft_concurrency.cpp).
    std::lock_guard<std::mutex> lock(fft_cache_mutex());
    static std::map<uint32_t, SlotIdx> cache;
    auto it = cache.find(N);
    if (it != cache.end()) return it->second;
    uint32_t M = 2 * N, S = N / 2;
    SlotIdx idx;
    idx.r.resize(S); idx.gidx.resize(S);
    uint64_t rk = 1;
    for (uint32_t k = 0; k < S; ++k) {
        idx.r[k] = rk;
        idx.gidx[k] = (M - rk) % M;
        rk = (rk * 5) % M;
    }
    return cache.emplace(N, std::move(idx)).first->second;
}

} // namespace

void encode_host_fft(std::vector<int64_t>& m, const std::vector<cd>& z,
                      uint32_t N, double Delta)
{
    uint32_t M = 2 * N, S = N / 2;
    const SlotIdx& idx = slot_idx(N);
    std::vector<cd> Z(M, cd(0.0, 0.0));
    for (uint32_t k = 0; k < S; ++k) {
        Z[idx.gidx[k]] = z[k];
        Z[idx.r[k]] = std::conj(z[k]);
    }
    std::vector<cd> x = fft_unnormalized_inverse(Z);
    m.assign(N, 0);
    const double sc = Delta / (double)N;
    for (uint32_t j = 0; j < N; ++j) m[j] = (int64_t)std::llround(sc * x[j].real());
}

void decode_host_fft(std::vector<cd>& z, const std::vector<int64_t>& m,
                      uint32_t N, double Delta)
{
    uint32_t M = 2 * N, S = N / 2;
    const SlotIdx& idx = slot_idx(N);
    std::vector<cd> padded(M, cd(0.0, 0.0));
    for (uint32_t j = 0; j < N; ++j) padded[j] = cd((double)m[j], 0.0);
    fft_forward_inplace(padded);
    z.assign(S, cd(0.0, 0.0));
    for (uint32_t k = 0; k < S; ++k) z[k] = padded[idx.gidx[k]] / Delta;
}

} // namespace gpufhe
