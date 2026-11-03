// GOD PROMPT "CKKS PHASE 11": correctness test for the parallelized
// encode_host/decode_host (src/encode.cpp). The production code's own
// per-element arithmetic is UNCHANGED by parallelization (same summation
// order, same operations, per output element -- only WHICH thread
// computes a given element differs), so this test's oracle is an
// INDEPENDENT, hand-written, single-threaded reimplementation of the
// exact pre-Phase-11 algorithm, written fresh here -- NOT a call into
// src/encode.cpp, and NOT merely diffing against git history. The
// production implementation is never used as its own oracle.
//
// Since both sides perform IDENTICAL IEEE-754 double-precision operations
// in the IDENTICAL order for any given output index (parallelization only
// changes which thread performs that work, never the work itself), the
// expected result is EXACT bit-identical equality, not an approximate
// tolerance -- any difference at all would indicate a genuine correctness
// regression, not floating-point noise.
#include <cstdint>
#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <random>

namespace gpufhe {
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void decode_host(std::vector<std::complex<double>>&, const std::vector<int64_t>&, uint32_t, double);
}

namespace {
// Independent, single-threaded oracle -- a fresh transcription of the
// exact mathematical definition (the same formula this file's own header
// comment states), not a copy-paste of src/encode.cpp's current code.
void oracle_encode(std::vector<int64_t>& m, const std::vector<std::complex<double>>& z,
                    uint32_t N, double Delta)
{
    const uint32_t M = 2 * N, S = N / 2;
    std::vector<std::complex<double>> tw(M);
    for (uint32_t t = 0; t < M; ++t) tw[t] = std::polar(1.0, -M_PI * (double)t / (double)N);
    std::vector<uint64_t> r(S);
    uint64_t rk = 1;
    for (uint32_t k = 0; k < S; ++k) { r[k] = rk; rk = (rk * 5) % M; }
    m.assign(N, 0);
    const double sc = 2.0 * Delta / (double)N;
    for (uint32_t j = 0; j < N; ++j) {
        double acc = 0;
        for (uint32_t k = 0; k < S; ++k) {
            uint64_t t = ((uint64_t)j * r[k]) % M;
            acc += z[k].real() * tw[t].real() - z[k].imag() * tw[t].imag();
        }
        m[j] = (int64_t)llround(sc * acc);
    }
}

void oracle_decode(std::vector<std::complex<double>>& z, const std::vector<int64_t>& m,
                    uint32_t N, double Delta)
{
    const uint32_t M = 2 * N, S = N / 2;
    std::vector<std::complex<double>> tw(M);
    for (uint32_t t = 0; t < M; ++t) tw[t] = std::polar(1.0, +M_PI * (double)t / (double)N);
    std::vector<uint64_t> r(S);
    uint64_t rk = 1;
    for (uint32_t k = 0; k < S; ++k) { r[k] = rk; rk = (rk * 5) % M; }
    z.assign(S, {0, 0});
    for (uint32_t k = 0; k < S; ++k) {
        std::complex<double> acc{0, 0};
        for (uint32_t j = 0; j < N; ++j) {
            uint64_t t = ((uint64_t)j * r[k]) % M;
            acc += (double)m[j] * tw[t];
        }
        z[k] = acc / Delta;
    }
}
} // namespace

int main() {
    bool allPass = true;
    std::mt19937_64 rng(2024);
    std::uniform_real_distribution<double> val(-10.0, 10.0);

    // Sweep N across the exact sizes bench_evalmult_e2e_breakdown.cpp's
    // own baseline sweep uses, plus below/above the kParallelThreshold=256
    // boundary to exercise both the sequential-fallback path and the
    // actual multi-threaded path.
    for (uint32_t N : {4u, 128u, 256u, 512u, 1024u, 2048u, 4096u, 8192u}) {
        uint32_t S = N / 2;
        double Delta = (double)(1ull << 50);

        // Trial 1: random real-only slots (matches this project's own
        // established real-slot convention elsewhere).
        std::vector<std::complex<double>> z(S);
        for (uint32_t k = 0; k < S; ++k) z[k] = {val(rng), 0.0};

        std::vector<int64_t> m_oracle, m_actual;
        oracle_encode(m_oracle, z, N, Delta);
        gpufhe::encode_host(m_actual, z, N, Delta);

        bool encodeMatch = (m_oracle.size() == m_actual.size());
        for (size_t i = 0; encodeMatch && i < m_oracle.size(); ++i)
            if (m_oracle[i] != m_actual[i]) encodeMatch = false;

        std::vector<std::complex<double>> z_oracle, z_actual;
        oracle_decode(z_oracle, m_oracle, N, Delta);
        gpufhe::decode_host(z_actual, m_actual, N, Delta);

        bool decodeMatch = (z_oracle.size() == z_actual.size());
        for (size_t i = 0; decodeMatch && i < z_oracle.size(); ++i)
            if (z_oracle[i].real() != z_actual[i].real() || z_oracle[i].imag() != z_actual[i].imag())
                decodeMatch = false;

        std::cout << "N=" << N << " encode=" << (encodeMatch ? "[EXACT MATCH]" : "[MISMATCH]")
                  << " decode=" << (decodeMatch ? "[EXACT MATCH]" : "[MISMATCH]") << "\n";
        if (!encodeMatch || !decodeMatch) allPass = false;

        // Trial 2: complex slots too (full generality, not just real-only).
        std::vector<std::complex<double>> zc(S);
        for (uint32_t k = 0; k < S; ++k) zc[k] = {val(rng), val(rng)};
        std::vector<int64_t> mc_oracle, mc_actual;
        oracle_encode(mc_oracle, zc, N, Delta);
        gpufhe::encode_host(mc_actual, zc, N, Delta);
        bool cMatch = (mc_oracle.size() == mc_actual.size());
        for (size_t i = 0; cMatch && i < mc_oracle.size(); ++i)
            if (mc_oracle[i] != mc_actual[i]) cMatch = false;
        std::cout << "N=" << N << " (complex slots) encode=" << (cMatch ? "[EXACT MATCH]" : "[MISMATCH]") << "\n";
        if (!cMatch) allPass = false;
    }

    std::cout << (allPass ? "[OVERALL PASS]\n" : "[OVERALL FAIL]\n");
    return allPass ? 0 : 1;
}
