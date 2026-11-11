# resident_finance

Phase 32, Work Package B: a real, non-FHE GPU-resident streaming
simulation, built to prove GPU-Resident-Library's architecture has value
outside CKKS/FHE.

## What it does

`K` instruments each hold GPU-resident state: last price and an EWMA
estimate of log-return variance (RiskMetrics-style, decay `lambda=0.94`).
A stream of incoming price ticks arrives; for each tick:

1. The new price is uploaded (one small H2D transfer, `K` doubles).
2. `LaunchUpdateRollingStats` updates the resident EWMA variance and price
   in place on the GPU.
3. `LaunchBlackScholes` immediately re-prices a European call for every
   instrument, **on the same CUDA stream, with no intervening host
   round-trip** — reading the state the previous kernel just wrote. This
   chained, no-readback-between-kernels execution is the property being
   demonstrated.
4. Every `readbackEvery` ticks (and at the end), the resident price,
   volatility, and option-price arrays are downloaded in one batch.

Both new kernels operate on `gpufhe::DeviceBuffer<double>`
(`include/device_buffer.h`) rather than hand-rolled `cudaMalloc`/`cudaFree`
— the smallest new primitive this phase's architecture audit
(`research/GPU_RESIDENT_ARCHITECTURE.md`) justified, proven here by a
genuinely new, non-FHE caller.

This is an **illustrative numerical simulation**, not a production pricing
or risk engine: the EWMA volatility model and the shared, fixed contract
terms (strike/rate/expiry, identical across all instruments) are stated
simplifications, not hidden assumptions. It is not a competitive
exchange-facing HFT system — see the latency numbers below, which are
GPU-kernel-launch-and-PCIe-transfer latency on a consumer laptop GPU, not
microsecond colocated-exchange latency.

## Build

```bash
cd ~/GPU-Resident-Library
export PATH=/usr/local/cuda-13.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cmake -S . -B build2
cmake --build build2 --target resident_finance -j4
```

## Run

```bash
./build2/resident_finance [K] [nTicks] [readbackEvery]
# defaults: K=4096, nTicks=2000, readbackEvery=100
```

It is also registered with `ctest` (`ctest -R resident_finance`) — its own
GPU-vs-CPU-reference correctness check is the pass/fail condition, so it
doubles as a regression test.

## Correctness

Both the GPU path and a CPU-only reference implementation run the
**identical formulas against the identical, pre-generated tick stream**
(generated once, shared by both — never two independently-seeded random
streams), so any divergence is a real bug, not sampling noise. Measured
max absolute error at the default configuration (K=4096, 2000 ticks):
`price=0`, `ewmaVar=1.6e-19`, `callPrice=3.6e-14`, `vol=1.1e-16` — all at
double-precision transcendental-function-level noise (GPU `normcdf`/`exp`/
`log` vs host `libm` can legitimately differ in the last few bits), not a
real discrepancy.

## Measured performance (this hardware: RTX 2060 Max-Q, 6144 MiB, CUDA 13.2)

At the default K=4096, nTicks=2000:

| Metric | Value |
|---|---|
| GPU cold first-tick issue latency | ~1970 us |
| GPU warm median per-tick issue latency | ~135 us |
| GPU total wall time, 2000 ticks | ~280 ms (~7150 ticks/sec) |
| CPU total wall time, 2000 ticks | ~1577 ms (~1268 ticks/sec) |
| End-to-end speedup | ~5.6x (GPU faster) |
| H2D transfer volume | 62.5 MiB (2000 ticks x 4096 instruments x 8 bytes) |
| D2H transfer volume | 2.6 MiB (21 readbacks) |

### Honest crossover sweep (`./resident_finance <K> 500 50` for each K)

| K (instruments) | GPU wins? | Speedup (GPU/CPU) |
|---|---|---|
| 8 | No | 0.004x — CPU ~230x faster (fixed launch overhead dominates) |
| 32 | No | 0.023x |
| 128 | No | 0.149x |
| 512 | No | 0.636x |
| 2048 | **Yes** | 2.231x |
| 8192 | **Yes** | 5.588x |
| 32768 | **Yes** | 7.964x |
| 131072 | **Yes** | 18.952x |

**The crossover is around K≈1000-2000 instruments** on this hardware and
kernel shape. Below that, per-tick launch and transfer overhead exceeds
the GPU's compute advantage; the CPU reference wins, and this is reported
as-is rather than hidden or only benchmarked at a favorable K. This is
the single most important number this workload produces: not "GPU is
faster," but "GPU is faster once the resident working set crosses roughly
this size, and slower below it."

## Files

- `examples/resident_finance/main.cpp` — the application.
- `include/device_buffer.h`, `include/cuda_check.h` — the two new
  reusable primitives it's built on.
- `include/finance_resident.h`, `kernels/finance_resident.cu` — the two
  CUDA kernels (EWMA update, vectorized Black-Scholes) and their launch
  wrappers.
