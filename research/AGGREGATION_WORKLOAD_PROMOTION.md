# Private aggregation workload: decomposed benchmark, CPU baseline, memory scaling, and promotion decision (Phase 29)

GOD PROMPT "CKKS PHASE 29", Objectives C+D+E+F+G. Builds on Phase 28's
`bench/bench_private_aggregation_workload.cpp` (the first real
consumer-GPU CKKS workload this project reported): K client-held values,
each independently encoded+encrypted, homomorphically summed via
`ct_add_ct_host`, decrypted once. Phase 28 reported 156ms/aggregation at
K=32; this round independently re-verified that number (reproduced at
141.86ms on a fresh run, same machine, real run-to-run variance; worst
error 3.347054e-09, identical to Phase 28's reported 3.3e-9) before
investigating further.

## Objective C: what the workload actually does, decomposed

`bench/bench_aggregation_decomposed.cpp` (new) separates every stage via
wall-clock timing, full statistics (mean/median/stddev/min/max, n=10,
interleaved A/B to control for thermal/clock bias), for K=32 clients at
n=8192, S=4096, sizeQ=4, h=64, sigma=3.19, Delta=2^45:

- **encode** (K calls, pure CPU, `encode_host_fft`): ~10.4ms total across
  32 calls -- NOT GPU-accelerated at all; confirmed by grep that `src/
  encode_fft.cpp` includes no `cuda_runtime.h` and calls no CUDA API.
- **encrypt** (K calls, GPU NTT): 83-132ms depending on which variant is
  used (see Objective E below) -- the dominant cost, 86-91% of
  end-to-end time.
- **aggregate** (K-1 calls, pure CPU, `ct_add_ct_host`): ~1.4ms total --
  confirmed pure CPU, no CUDA in `src/ptops.cpp`.
- **decrypt** (1 call, GPU NTT): ~0.7-0.8ms -- cheap because it is only
  ONE call, unlike encrypt's K calls.
- **decode** (1 call, pure CPU): ~0.3ms.
- Key generation (~325-350ms) is reported SEPARATELY and amortized
  across every aggregation in a real deployment (one key, many
  aggregations) -- never folded into the per-aggregation numbers above,
  per the governing directive's explicit "never hide setup time" / "do
  not omit expensive setup operations without explicitly reporting them
  separately" instructions.

**What the Phase 28 "156ms" and "124MB RSS" numbers actually measured**:
156ms was the full per-aggregation wall-clock (encode+encrypt+aggregate+
decrypt+decode for all 32 clients, EXCLUDING the one-time keygen); 124MB
RSS was the process's peak resident set size at that point, dominated by
host-side vectors (ciphertext/plaintext buffers), not a GPU-memory
figure (that is the separately reported VRAM-free number).

## Objective E: the real bottleneck is NOT encoding -- it is per-client encrypt, and Phase 28 used the wrong path

Phase 28's workload called plain `gpufhe::encrypt_host` -- the Phase-11
REFERENCE implementation. This project's own Rust bridge (`rust/crates/
ckks-cuda/cpp_bridge/encdec_bridge.cpp`, confirmed by direct inspection)
calls a DIFFERENT function for real encryption:
`encrypt_host_shoup_pooled_barrett_batched` -- already bit-exact verified
against `encrypt_host` across every supported dimension (`tests/
test_encrypt_host_shoup_pooled_barrett_batched_matches_reference.cpp`),
already concurrency-audited (`tests/test_encrypt_host_shoup_pooled_
barrett_batched_concurrency_audit.cpp`, re-run this round: 3/3 clean,
0/396 mismatches), and already measured 1.57x-2.78x faster than
`encrypt_host` in ISOLATION (`research/ENCRYPT_PROFILING.md`, `research/
EMBED_AND_POOL_HARDENING.md`). Phase 28's workload never used it.

`bench_aggregation_decomposed.cpp` checked whether that isolated gain
survives inside the real K-client application (interleaved, 10 reps
each):

```
Arm A (OLD encrypt_host):                      encrypt mean=131.5ms  END-TO-END median=144.3ms
Arm B (PRODUCTION encrypt_host_shoup_pooled_barrett_batched): encrypt mean=83.2ms  END-TO-END median=95.1ms
END-TO-END speedup (median): 1.518x
encrypt-stage-only speedup (median): 1.596x
encrypt stage as % of end-to-end: OLD=90.7%  PRODUCTION=86.3%
worst abs error: OLD=3.346969e-09  PRODUCTION=3.347068e-09 (same correctness)
```

**A real, substantial, measured end-to-end gain (1.52x) from using code
that already existed and was already verified** -- not a new
optimization, a mis-wired benchmark. Encrypt remains the dominant cost
even after the swap (86.3% of end-to-end), confirming it as the right
target and ruling out encoding (which Phase 28 found dominant in a
DIFFERENT context -- the bootstrap's per-diagonal hot loop, where
`encode_host`'s O(N²) reference was still wired in; this workload already
used the fast FFT path from the start).

## Objective D: fair CPU baseline -- and the limitation that forced a workaround

**gpufhe has no pure-CPU-only encrypt/decrypt path.** Every
`encrypt_host*` variant in `src/encrypt_host_shoup*.cpp` and `src/
encdec.cpp` calls `cudaMalloc`/`cudaMemcpy`/an NTT kernel internally
(confirmed by grep: all include `cuda_runtime.h`, all call
`cudaDeviceSynchronize`). `encode_host_fft`/`decode_host_fft` and
`ct_add_ct_host`, by contrast, are ALREADY pure CPU -- there is no
"GPU-accelerated encode" in this library at all, so no GPU-vs-CPU
question exists for those stages. The only real GPU-vs-CPU question in
this whole workload is encrypt/decrypt's NTT step.

Since gpufhe cannot supply its own CPU-only comparison, `bench/
bench_aggregation_openfhe_cpu_baseline.cpp` (new) uses OpenFHE -- already
present in this repo as its own CPU ground-truth oracle elsewhere
(`bench/bench_evalmult.cpp`) -- run with `OPENFHE_GPU=0` (Phase 25/27's
own finding: this specific OpenFHE build is secretly GPU-shadow-
accelerated by an external dependency unless disabled) to guarantee
genuine CPU-only execution, at matching n=8192, for the same K=32-client
sum workload:

```
OpenFHE CPU-only:  encode+encrypt mean=383.4ms  END-TO-END median=379.7ms  worst err=7.24e-11
gpufhe GPU (production path):                   END-TO-END median=95.1ms
```

**~4x faster on GPU for this workload** -- a genuine, honest data point,
with the explicit caveat (stated in the benchmark's own header and
repeated here) that this is a DIFFERENT library with its own secret
distribution, noise sampling, and internal optimizations, not this
project's own CPU mode. This is the smallest valid comparison available
given gpufhe's architecture, not an invented one.

## Objective F: memory scaling across client count

`bench/bench_aggregation_client_scaling.cpp` (new), K ∈
{1,2,4,8,16,32,64}, production encrypt path:

```
K    runtime(ms)  VRAM_free_before  VRAM_free_after  delta_MiB
1    7.87         5105              5103             2
2    7.69         5103              5103             0
4    11.98         5103              5103             0
8    23.68         5103              5103             0
16   49.12         5103              5103             0
32   92.99         5103              5103             0
64   185.21        5103              5103             0
```

**VRAM stays flat regardless of K** (confirming the hypothesis: each
client's ciphertext is encrypted, folded into the accumulator, and its
buffer is returned to `encrypt_host_shoup_pooled_barrett_batched`'s
internal pool rather than held concurrently -- no per-client VRAM growth).
**Runtime scales linearly in K** from K=4 onward (K=8 through K=64 each
roughly double the previous K's runtime); K=1/K=2 are anomalously flat,
consistent with one-time pool-warmup cost dominating at that scale and
being amortized away as K grows. This is a genuinely good consumer-GPU
property: 64 clients cost no more VRAM than 1, on a 6GB card with 5.1GB
free throughout.

## Objective G: production promotion decision

**Promote.** `bench/bench_private_aggregation_workload.cpp` now calls
`encrypt_host_shoup_pooled_barrett_batched` as its default (matching
what this project's own Rust-facing API actually uses), keeping plain
`encrypt_host` reachable and unchanged for any future comparison.
Re-verified after the swap: worst error 3.347054e-09 (unchanged),
runtime dropped from ~142-156ms to ~93-112ms (median 100.27ms) for the
same 10-repetition, fresh-randomized-input protocol. No cryptographic
parameter changed; the swap is between two already-bit-exact-verified
implementations of the identical operation.

This was not a new optimization experiment -- it was finding and fixing
a benchmark that measured the wrong code path. The actual production
path was already built, tested, and proven in Phases 13-15; Phase 29's
contribution is discovering that the newest, most application-facing
benchmark in the project had not been wired to use it.
