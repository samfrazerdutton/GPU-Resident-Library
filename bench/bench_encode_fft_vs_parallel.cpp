// GOD PROMPT "CKKS PHASE 12": standalone A/B benchmark, OLD (Phase 11's
// std::thread-parallelized O(N^2), src/encode.cpp) vs NEW (O(N log N)
// FFT, src/encode_fft.cpp) encode_host/decode_host, with NO GPU and NO
// encryption involved -- isolates exactly the algorithmic change this
// round makes. Interleaved A/B (old, new, old, new, ... per trial) to
// control for thermal/scheduling drift, matching this project's
// established methodology. Reports median/p95/min/max/sample count via
// a monotonic host clock (std::chrono::steady_clock) -- both
// implementations are pure host functions, no GPU timing needed here.
#include <cstdint>
#include <vector>
#include <complex>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <random>

namespace gpufhe {
void encode_host(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void decode_host(std::vector<std::complex<double>>&, const std::vector<int64_t>&, uint32_t, double);
void encode_host_fft(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
void decode_host_fft(std::vector<std::complex<double>>&, const std::vector<int64_t>&, uint32_t, double);
}

namespace {
struct Stats { double min_, p50, p95, max_, mean; size_t n; };
Stats stats_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    Stats s; s.n = v.size(); s.min_ = v.front(); s.max_ = v.back();
    s.p50 = v[v.size() / 2]; s.p95 = v[(size_t)(0.95 * (v.size() - 1))];
    double sum = 0; for (double x : v) sum += x; s.mean = sum / v.size();
    return s;
}
void report(const char* label, const Stats& s) {
    std::cout << "    " << label << ": median=" << s.p50 << "us p95=" << s.p95
              << "us min=" << s.min_ << "us max=" << s.max_ << "us mean=" << s.mean
              << "us (n=" << s.n << ")\n";
}
using cd = std::complex<double>;
} // namespace

int main(int argc, char** argv) {
    size_t trials = argc > 1 ? (size_t)std::atoi(argv[1]) : 15;
    std::mt19937_64 rng(2024);

    for (uint32_t N : {1024u, 2048u, 4096u, 8192u}) {
        uint32_t S = N / 2;
        double Delta = std::pow(2.0, 50);
        std::uniform_real_distribution<double> d(-8.0, 8.0);
        std::vector<cd> z(S);
        for (uint32_t k = 0; k < S; ++k) z[k] = cd(d(rng), 0);
        std::vector<int64_t> m_for_decode;
        gpufhe::encode_host(m_for_decode, z, N, Delta); // seed decode input with a realistic message

        // Warm up both paths once (first-touch allocation / cache effects).
        std::vector<int64_t> m_tmp;
        std::vector<cd> z_tmp;
        gpufhe::encode_host(m_tmp, z, N, Delta);
        gpufhe::encode_host_fft(m_tmp, z, N, Delta);
        gpufhe::decode_host(z_tmp, m_for_decode, N, Delta);
        gpufhe::decode_host_fft(z_tmp, m_for_decode, N, Delta);

        std::vector<double> encOld, encNew, decOld, decNew;
        for (size_t t = 0; t < trials; ++t) {
            // Interleaved order: old, new, old, new per trial.
            {
                auto t0 = std::chrono::steady_clock::now();
                gpufhe::encode_host(m_tmp, z, N, Delta);
                encOld.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
            }
            {
                auto t0 = std::chrono::steady_clock::now();
                gpufhe::encode_host_fft(m_tmp, z, N, Delta);
                encNew.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
            }
            {
                auto t0 = std::chrono::steady_clock::now();
                gpufhe::decode_host(z_tmp, m_for_decode, N, Delta);
                decOld.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
            }
            {
                auto t0 = std::chrono::steady_clock::now();
                gpufhe::decode_host_fft(z_tmp, m_for_decode, N, Delta);
                decNew.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
            }
        }

        std::cout << "=== N=" << N << " (" << trials << " interleaved trials) ===\n";
        report("encode OLD (parallel O(N^2))", stats_of(encOld));
        report("encode NEW (FFT O(N log N))", stats_of(encNew));
        report("decode OLD (parallel O(N^2))", stats_of(decOld));
        report("decode NEW (FFT O(N log N))", stats_of(decNew));
        Stats eo = stats_of(encOld), en = stats_of(encNew), doo = stats_of(decOld), dn = stats_of(decNew);
        std::cout << "    encode speedup (median): " << (eo.p50 / en.p50) << "x\n";
        std::cout << "    decode speedup (median): " << (doo.p50 / dn.p50) << "x\n";
    }
    return 0;
}
