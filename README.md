# GPU-Resident-Library

A from-scratch, GPU-resident CKKS implementation that **owns its ciphertext
representation end-to-end** — the transform, arithmetic, keyswitch, and key
generation all run natively on the GPU with **zero OpenFHE dependency at
runtime**. OpenFHE is linked in tests only, as a bit-exact oracle.

## What this is (and why)

An earlier approach tried to accelerate OpenFHE via a CUDA HAL that dispatched
individual operations to the GPU. A decisive paired sweep (depth 5/10/15 ×
ring 32k/64k) showed **no CPU/GPU crossover anywhere**: operation-level dispatch
across the host boundary can't win, because per-op PCIe transfer scales with
data volume. Worse, cross-op residency is structurally impossible inside
OpenFHE (its polynomial storage reallocates on every operation).

The conclusion: to win, the library must **own the ciphertext** and keep it
resident on the device across an entire circuit. That's what this is.

## Status: complete, self-contained, numerically-proven CKKS multiply

The full leveled-CKKS multiply pipeline works end-to-end, entirely native:

    keygen → encrypt → tensor → relinearize (Hybrid keyswitch) → rescale → decrypt

Every arithmetic primitive is validated **bit-exact** against OpenFHE in
isolation, and the composed multiply is validated **numerically correct
end-to-end** (a homomorphic product of two encrypted polynomials decrypts to
exactly the expected result).

### Proven components

| Primitive | Validation |
|---|---|
| Forward NTT (Cooley-Tukey, Shoup) | bit-exact, all 12 tower moduli, n=32768 |
| Inverse NTT (Gentleman-Sande) | bit-exact + round-trip INTT(NTT(x))==x |
| RNS multiply (Montgomery) | bit-exact, all tower moduli |
| SwitchModulus / Rescale | bit-exact vs DropLastElementAndScale |
| ApproxSwitchCRTBasis | bit-exact (Hybrid keyswitch feasibility gate) |
| Digit decompose | bit-exact vs EvalKeySwitchPrecomputeCore |
| Fast-keyswitch inner product | bit-exact vs EvalFastKeySwitchCoreExt |
| ApproxModDown (QP→Q) | bit-exact |
| Full KeySwitchCore | bit-exact vs OpenFHE KeySwitchCore |
| Native keygen (constants, RLWE key, relin key) | decryption-identity validated |
| Encrypt / Decrypt | round-trip exact (after Δ scaling) |
| **Full EvalMult (composed)** | **numerically exact end-to-end** |

## Performance (RTX 2060 Max-Q, ring 32768)

The honest headline: **resident CKKS loses single-ciphertext latency but wins
batched throughput.** The GPU's advantage is throughput — one ciphertext at
n=32768 underfills the SMs; many concurrent chains fill them.

**Resident multiply-rescale chain, batch of 16 (bit-exact, relin-free):**

| Metric | GPU | CPU | Result |
|---|---|---|---|
| Single-chain latency | 7.65 ms | ~6.4 ms | GPU slower |
| Batch throughput | 3.49 ms/chain | 6.67 ms/chain | **GPU 1.91× faster** |

The batch win was built up across three bit-exact optimizations (Montgomery
multiply, shared-memory forward NTT, shared-memory INTT), 1.24× → 1.91×. Each
improved throughput (bandwidth-bound, concurrent) while barely moving latency
(dependency-bound, serial) — exactly as the theory predicts.

**Batched full EvalMult (tensor + resident relinearization), N=16:**

| | GPU | CPU (OpenFHE EvalMult) | Result |
|---|---|---|---|
| Per-op cost | 5.59 ms | 62.9 ms | **GPU ~11× faster** |

This is the headline. Relinearization is the most expensive CKKS operation on
CPU (~60 of the 62.9 ms), and it's dense, parallel modular arithmetic — exactly
where batched GPU execution wins most. The keyswitch runs **fully resident**:
all constants, eval-key material, and root tables are uploaded once at setup;
each operation is pure kernel launches and device-to-device copies on a stream,
with zero mallocs or host transfers. (An earlier host-orchestrated version of
the same keyswitch cost 124.7 ms/op — residency recovered all of it.)

The comparison is verified under FIXEDMANUAL scaling so OpenFHE's EvalMult does
exactly tensor + relin, matching the GPU pipeline op-for-op.

### Benchmark methodology

All GPU/CPU comparisons construct operands **outside** the timed region (an
earlier version constructed CPU operands inside the timer and faked a GPU win —
corrected). Benchmarks interleave GPU/CPU per rep and report median with a
bootstrap CI, since WSL2 blocks GPU clock locking and common-mode noise cancels
in the paired difference.

## Building

Requires CUDA 13.2 (arch 75), and OpenFHE installed under `/usr/local` (tests
only). `cmake .. -DCMAKE_BUILD_TYPE=Release && make`. Tests validate each
primitive against OpenFHE; benchmarks under `bench/`.

## What's next (optional)

- Fold rescale into the composed EvalMult benchmark.
- Sweep batch size (N=32/64/128) and ring dimension to find device saturation —
  the 11× at N=16 is likely not the peak.
- Batch per-tower kernel launches into single grid-spanning launches (the next
  launch-overhead lever).
- Complex canonical encode/decode for real message packing.
- Bootstrapping for unbounded depth.

## Design discipline

Every primitive is validated bit-exact against OpenFHE's *low-level* routine (not
scale-aware wrappers) before it's composed or benchmarked. Hard modular
arithmetic is **ported** from OpenFHE exactly, never reconstructed — a
hand-rolled Barrett reduction and a reconstructed root of unity each caused
failures that the bit-exact gates caught immediately. Correctness is proven
before speed, every time.
