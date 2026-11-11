// Phase 32, Work Package B: a real, non-FHE, GPU-resident streaming
// simulation demonstrating the project's architecture outside CKKS.
//
// Model: K instruments each hold GPU-resident state (price, EWMA variance
// of log returns). A stream of incoming price ticks arrives; each tick
// updates the resident EWMA volatility estimate and immediately re-prices
// a European call (Black-Scholes, vectorized across all K instruments) on
// the SAME stream with no intervening host round-trip -- this is the
// chained-resident-execution property the phase is meant to demonstrate,
// built using gpufhe::DeviceBuffer<T> (research/GPU_RESIDENT_ARCHITECTURE.md
// section 9.1), not raw hand-rolled cudaMalloc.
//
// This is an illustrative numerical simulation, not a production pricing or
// risk engine: the EWMA volatility model (RiskMetrics-style, zero-mean log
// returns) and the fixed shared contract terms are simplifications stated
// explicitly here, not hidden assumptions.
#include "device_buffer.h"
#include "finance_resident.h"
#include "cuda_check.h"
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <numeric>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct SimParams {
    double lambda = 0.94;          // RiskMetrics-style EWMA decay
    double annualization = 252.0;  // trading days/year (illustrative)
    double strike = 100.0;
    double rate = 0.01;
    double T = 0.5;                // years to expiry
    double trueVol = 0.20;         // synthetic-tick generator's true vol
    double dt = 1.0 / 252.0;       // one simulated trading day per tick
};

// Generates the full tick stream up front, shared verbatim by both the GPU
// and CPU paths -- so correctness comparison is never confounded by two
// independently-seeded random streams.
static void generate_ticks(uint32_t K, uint32_t nTicks, uint64_t seed,
                            std::vector<double>& initialPrice,
                            std::vector<std::vector<double>>& ticks,
                            const SimParams& P)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> priceDist(80.0, 120.0);
    std::normal_distribution<double> z(0.0, 1.0);
    initialPrice.resize(K);
    for (uint32_t i = 0; i < K; ++i) initialPrice[i] = priceDist(rng);
    ticks.assign(nTicks, std::vector<double>(K));
    std::vector<double> cur = initialPrice;
    double drift = (0.0 - 0.5 * P.trueVol * P.trueVol) * P.dt;
    double diffScale = P.trueVol * std::sqrt(P.dt);
    for (uint32_t t = 0; t < nTicks; ++t) {
        for (uint32_t i = 0; i < K; ++i) {
            double zi = z(rng);
            cur[i] = cur[i] * std::exp(drift + diffScale * zi);
            ticks[t][i] = cur[i];
        }
    }
}

// CPU reference: identical recurrence, no GPU at all.
struct CpuResult {
    std::vector<double> price, ewmaVar, callPrice, vol;
    double wall_ms;
};
static CpuResult run_cpu_reference(uint32_t K, const std::vector<double>& initialPrice,
                                    const std::vector<std::vector<double>>& ticks,
                                    const SimParams& P)
{
    CpuResult R;
    R.price = initialPrice;
    R.ewmaVar.assign(K, P.trueVol * P.trueVol * P.dt); // same prior as GPU path
    R.callPrice.assign(K, 0.0);
    R.vol.assign(K, 0.0);
    auto t0 = Clock::now();
    for (const auto& tick : ticks) {
        for (uint32_t i = 0; i < K; ++i) {
            double logret = std::log(tick[i] / R.price[i]);
            R.ewmaVar[i] = P.lambda * R.ewmaVar[i] + (1.0 - P.lambda) * logret * logret;
            R.price[i] = tick[i];
        }
        for (uint32_t i = 0; i < K; ++i) {
            double S = R.price[i];
            double sigma = std::sqrt(std::max(R.ewmaVar[i], 1e-12) * P.annualization);
            double sqrtT = std::sqrt(P.T);
            double d1 = (std::log(S / P.strike) + (P.rate + 0.5 * sigma * sigma) * P.T) / (sigma * sqrtT);
            double d2 = d1 - sigma * sqrtT;
            auto N = [](double x){ return 0.5 * std::erfc(-x / std::sqrt(2.0)); };
            R.callPrice[i] = S * N(d1) - P.strike * std::exp(-P.rate * P.T) * N(d2);
            R.vol[i] = sigma;
        }
    }
    R.wall_ms = ms_since(t0);
    return R;
}

struct GpuResult {
    std::vector<double> price, ewmaVar, callPrice, vol;
    double total_wall_ms;
    double warm_median_tick_issue_us; // steady-state host-side issue latency per tick
    double first_tick_issue_us;       // cold first-tick issue latency
};
static GpuResult run_gpu_resident(uint32_t K, const std::vector<double>& initialPrice,
                                   const std::vector<std::vector<double>>& ticks,
                                   const SimParams& P, uint32_t readbackEvery)
{
    gpufhe::DeviceBuffer<double> price(K), ewmaVar(K), newPrice(K), callPrice(K), vol(K);
    cudaStream_t s; gpufhe::cuda_check(cudaStreamCreate(&s), "cudaStreamCreate");

    price.upload(initialPrice.data(), K);
    std::vector<double> priorVar(K, P.trueVol * P.trueVol * P.dt);
    ewmaVar.upload(priorVar.data(), K);

    std::vector<double> stagingNewPrice(K); // persistent host staging buffer,
                                             // reused every tick -- never a
                                             // function-local temporary handed
                                             // to an async copy (Phase 30's
                                             // lesson), and upload() here is
                                             // the SYNCHRONOUS DeviceBuffer
                                             // variant specifically so this
                                             // staging buffer's lifetime is
                                             // never in question.
    std::vector<double> dlPrice(K), dlVar(K), dlCall(K), dlVol(K);

    GpuResult R;
    std::vector<double> perTickUs;
    perTickUs.reserve(ticks.size());

    auto t0 = Clock::now();
    for (size_t t = 0; t < ticks.size(); ++t) {
        auto tTickStart = Clock::now();
        stagingNewPrice = ticks[t];
        newPrice.upload(stagingNewPrice.data(), K);  // H2D into the DEDICATED
                                                      // newPrice buffer -- price
                                                      // itself still holds the
                                                      // prior resident value
                                                      // until the kernel below
                                                      // overwrites it in place.
        gpufhe::LaunchUpdateRollingStats(price.data(), ewmaVar.data(), newPrice.data(), P.lambda, K, s);
        gpufhe::LaunchBlackScholes(price.data(), ewmaVar.data(), callPrice.data(), vol.data(),
                                    P.strike, P.rate, P.T, P.annualization, K, s);
        if ((t + 1) % readbackEvery == 0 || t + 1 == ticks.size()) {
            price.download(dlPrice.data(), K);
            ewmaVar.download(dlVar.data(), K);
            callPrice.download(dlCall.data(), K);
            vol.download(dlVol.data(), K);
        }
        perTickUs.push_back(std::chrono::duration<double, std::micro>(Clock::now() - tTickStart).count());
    }
    R.total_wall_ms = ms_since(t0);
    cudaStreamDestroy(s);

    R.price = dlPrice; R.ewmaVar = dlVar; R.callPrice = dlCall; R.vol = dlVol;
    R.first_tick_issue_us = perTickUs.empty() ? 0.0 : perTickUs.front();
    std::vector<double> warm(perTickUs.begin() + std::min<size_t>(10, perTickUs.size()), perTickUs.end());
    std::sort(warm.begin(), warm.end());
    R.warm_median_tick_issue_us = warm.empty() ? 0.0 : warm[warm.size() / 2];
    return R;
}

int main(int argc, char** argv) {
    uint32_t K = argc > 1 ? (uint32_t)std::stoul(argv[1]) : 4096;
    uint32_t nTicks = argc > 2 ? (uint32_t)std::stoul(argv[2]) : 2000;
    uint32_t readbackEvery = argc > 3 ? (uint32_t)std::stoul(argv[3]) : 100;

    SimParams P;
    std::printf("=== Phase 32: resident_finance -- GPU-resident streaming EWMA vol + Black-Scholes ===\n");
    std::printf("K=%u instruments, nTicks=%u, readback every %u ticks\n", K, nTicks, readbackEvery);

    std::vector<double> initialPrice;
    std::vector<std::vector<double>> ticks;
    generate_ticks(K, nTicks, /*seed=*/12345ULL, initialPrice, ticks, P);

    auto cpu = run_cpu_reference(K, initialPrice, ticks, P);
    auto gpu = run_gpu_resident(K, initialPrice, ticks, P, readbackEvery);

    double maxErrPrice = 0, maxErrVar = 0, maxErrCall = 0, maxErrVol = 0;
    for (uint32_t i = 0; i < K; ++i) {
        maxErrPrice = std::max(maxErrPrice, std::fabs(cpu.price[i] - gpu.price[i]));
        maxErrVar   = std::max(maxErrVar,   std::fabs(cpu.ewmaVar[i] - gpu.ewmaVar[i]));
        maxErrCall  = std::max(maxErrCall,  std::fabs(cpu.callPrice[i] - gpu.callPrice[i]));
        maxErrVol   = std::max(maxErrVol,   std::fabs(cpu.vol[i] - gpu.vol[i]));
    }
    bool correct = maxErrPrice < 1e-6 && maxErrVar < 1e-9 && maxErrCall < 1e-6 && maxErrVol < 1e-9;

    std::printf("\n-- Correctness (GPU vs CPU reference, same tick stream) --\n");
    std::printf("  max abs err: price=%.3e ewmaVar=%.3e callPrice=%.3e vol=%.3e\n",
                maxErrPrice, maxErrVar, maxErrCall, maxErrVol);
    std::printf("  %s\n", correct ? "[PASS] GPU resident path matches CPU reference" : "[FAIL] correctness mismatch");

    std::printf("\n-- Latency --\n");
    std::printf("  GPU cold first-tick issue latency: %.1f us\n", gpu.first_tick_issue_us);
    std::printf("  GPU warm median per-tick issue latency: %.1f us\n", gpu.warm_median_tick_issue_us);
    std::printf("  GPU total wall time (%u ticks incl. periodic readback): %.2f ms (%.1f ticks/sec)\n",
                nTicks, gpu.total_wall_ms, 1000.0 * nTicks / gpu.total_wall_ms);
    std::printf("  CPU total wall time (%u ticks, pure host): %.2f ms (%.1f ticks/sec)\n",
                nTicks, cpu.wall_ms, 1000.0 * nTicks / cpu.wall_ms);
    std::printf("  end-to-end speedup (GPU vs CPU, this K): %.3fx %s\n",
                cpu.wall_ms / gpu.total_wall_ms,
                (cpu.wall_ms > gpu.total_wall_ms ? "(GPU faster)" : "(CPU faster at this K)"));

    size_t h2dBytes = (size_t)nTicks * K * sizeof(double);
    size_t d2hBytes = (size_t)(nTicks / readbackEvery + 1) * K * 4 * sizeof(double);
    std::printf("\n-- Transfer volume --\n");
    std::printf("  H2D: %.2f MiB (%u ticks x %u instruments x 8 bytes)\n", h2dBytes / 1048576.0, nTicks, K);
    std::printf("  D2H: %.2f MiB (%u readbacks x %u instruments x 4 arrays x 8 bytes)\n",
                d2hBytes / 1048576.0, nTicks / readbackEvery + 1, K);

    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    std::printf("\n-- Memory --\n");
    std::printf("  VRAM free: %.0f/%.0f MiB\n", freeB / 1048576.0, totalB / 1048576.0);

    std::printf("\n=== TOTAL: %s ===\n", correct ? "1 passed, 0 failed" : "0 passed, 1 failed");
    return correct ? 0 : 1;
}
