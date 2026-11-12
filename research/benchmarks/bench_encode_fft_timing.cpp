// Phase 35, Mission 6: isolates encode_host_fft's own per-call cost,
// independent of the aggregation workload's other stages. Calls the real,
// unmodified production function as a black box.
#include <cstdio>
#include <complex>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>

namespace gpufhe {
void encode_host_fft(std::vector<int64_t>&, const std::vector<std::complex<double>>&, uint32_t, double);
}
using cd = std::complex<double>;
using clk = std::chrono::steady_clock;

int main() {
    const uint32_t n = 8192, S = n / 2;
    const double Delta = 1ull << 45;
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> d(-50.0, 50.0);
    std::vector<cd> z(S);
    for (auto& v : z) v = cd(d(rng), 0.0);

    const int R = 400;
    std::vector<double> ms(R);
    for (int rep = 0; rep < R; ++rep) {
        std::vector<int64_t> m;
        auto t0 = clk::now();
        gpufhe::encode_host_fft(m, z, n, Delta);
        ms[rep] = std::chrono::duration<double, std::micro>(clk::now() - t0).count();
    }
    std::sort(ms.begin(), ms.end());
    double median = ms[R/2];
    double mean = 0; for (double x : ms) mean += x; mean /= R;
    std::printf("=== Phase 35: encode_host_fft isolated timing (n=%u, R=%d) ===\n", n, R);
    std::printf("  median=%.3f us  mean=%.3f us  min=%.3f us  max=%.3f us\n", median, mean, ms.front(), ms.back());
    return 0;
}
