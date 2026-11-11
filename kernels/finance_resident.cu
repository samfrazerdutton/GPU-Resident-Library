#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// GPU-resident streaming finance simulation kernels (Phase 32, Work Package
// B). Everything here operates in place on device-resident per-instrument
// state arrays -- no kernel reads back to host, and the two kernels below
// are launched back-to-back on the same stream with no intervening sync, so
// a tick's volatility update and its resulting option-price recompute are a
// single resident chain: upload the new tick -> update stats -> price ->
// (periodically) download. This is the numerical-simulation analog of
// ckks_ops.cpp's mul_resident/rescale_resident chain, generalized away from
// CKKS. Illustrative EWMA risk model, not a production pricing/risk engine.

// Updates each instrument's resident state from one new tick:
//   logret   = ln(newPrice / price)
//   ewmaVar  = lambda*ewmaVar + (1-lambda)*logret^2      (RiskMetrics-style
//              EWMA variance of log returns, zero-mean approximation --
//              standard for short-horizon returns)
//   price    = newPrice
// All four arrays are device-resident for the whole simulation; only
// newPrice is uploaded per tick.
__global__ void update_rolling_stats_kernel(double* __restrict__ price,
                                             double* __restrict__ ewmaVar,
                                             const double* __restrict__ newPrice,
                                             double lambda, uint32_t k)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    double p0 = price[i];
    double p1 = newPrice[i];
    double logret = log(p1 / p0);
    double v = ewmaVar[i];
    ewmaVar[i] = lambda * v + (1.0 - lambda) * logret * logret;
    price[i] = p1;
}

// Vectorized Black-Scholes European call, reading the SAME resident price/
// ewmaVar arrays update_rolling_stats_kernel just wrote (no round trip):
//   sigma = sqrt(ewmaVar * annualizationFactor)
//   d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
//   d2 = d1 - sigma*sqrt(T)
//   call = S*N(d1) - K*exp(-r*T)*N(d2)
// using CUDA's built-in normcdf() device intrinsic for N(.). strike/rate/T
// are shared scalars (uniform contract terms across instruments), matching
// a simple vectorized pricing sweep.
__global__ void black_scholes_kernel(const double* __restrict__ price,
                                      const double* __restrict__ ewmaVar,
                                      double* __restrict__ callPrice,
                                      double* __restrict__ volOut,
                                      double strike, double rate, double T,
                                      double annualizationFactor, uint32_t k)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    double S = price[i];
    double sigma = sqrt(fmax(ewmaVar[i], 1e-12) * annualizationFactor);
    double sqrtT = sqrt(T);
    double d1 = (log(S / strike) + (rate + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;
    double call = S * normcdf(d1) - strike * exp(-rate * T) * normcdf(d2);
    callPrice[i] = call;
    volOut[i] = sigma;
}

extern "C" void LaunchUpdateRollingStats(double* d_price, double* d_ewmaVar,
                                          const double* d_newPrice,
                                          double lambda, uint32_t k, cudaStream_t s)
{
    uint32_t threads = 256;
    uint32_t blocks = (k + threads - 1) / threads;
    update_rolling_stats_kernel<<<blocks, threads, 0, s>>>(d_price, d_ewmaVar, d_newPrice, lambda, k);
}

extern "C" void LaunchBlackScholes(const double* d_price, const double* d_ewmaVar,
                                    double* d_callPrice, double* d_volOut,
                                    double strike, double rate, double T,
                                    double annualizationFactor, uint32_t k, cudaStream_t s)
{
    uint32_t threads = 256;
    uint32_t blocks = (k + threads - 1) / threads;
    black_scholes_kernel<<<blocks, threads, 0, s>>>(d_price, d_ewmaVar, d_callPrice, d_volOut,
                                                      strike, rate, T, annualizationFactor, k);
}
