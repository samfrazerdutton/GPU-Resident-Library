#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace gpufhe {

// Resident EWMA rolling-volatility update: mutates d_price/d_ewmaVar in
// place from one tick's d_newPrice. See kernels/finance_resident.cu for the
// exact recurrence and the chained-execution rationale.
extern "C" void LaunchUpdateRollingStats(double* d_price, double* d_ewmaVar,
                                          const double* d_newPrice,
                                          double lambda, uint32_t k, cudaStream_t s);

// Vectorized Black-Scholes European call over k instruments' resident
// price/ewmaVar state. strike/rate/T are shared scalars across all k
// instruments (a simple vectorized pricing sweep, not per-instrument terms).
extern "C" void LaunchBlackScholes(const double* d_price, const double* d_ewmaVar,
                                    double* d_callPrice, double* d_volOut,
                                    double strike, double rate, double T,
                                    double annualizationFactor, uint32_t k, cudaStream_t s);

} // namespace gpufhe
