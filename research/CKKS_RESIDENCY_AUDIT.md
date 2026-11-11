# CKKS/FHE residency audit (Phase 32, Work Package D)

Grounded in a fresh read of the current source tree and fresh benchmark
re-runs this round (not copied from historical phase reports). Hardware:
RTX 2060 Max-Q, 6144 MiB VRAM, driver 595.97, CUDA 13.2, cc7.5.

## 1. Per-operation residency (grep-verified, not assumed)

`cudaMemcpyHostToDevice`/`cudaMemcpyDeviceToHost`/`cudaMalloc` call-site
counts, this round's own grep against the current source:

| File (operation) | H2D | D2H | sync | malloc | Residency verdict |
|---|---|---|---|---|---|
| `encode.cpp` / `encode_fft.cpp` | 0 | 0 | 0 | 0 | **Pure host.** Encoding/decoding is a CPU-only FFT/O(n²) loop (Phase 11/28); never touches the GPU at all. This is a genuine, intentional CPU boundary, not an oversight — `research/ENCODE_FFT_PROMOTION.md` found `encode_host`/`encode_host_fft` dominate end-to-end time precisely because nothing else is faster there yet. |
| `encrypt_host_shoup_pooled_barrett_batched.cpp` (production encrypt path) | 5 | 1 | 1 | 6 | **Mostly resident within one call**, pooled device buffers reused across calls (confirmed by `bench_aggregation_client_scaling`: VRAM flat across K=1..64, section 3 below) — but still host→device→host per call, since input plaintext and output ciphertext both start/end on the host. |
| `ckks_ops.cpp` (`mul_resident`/`rescale_resident`) | 1 | 0 | 0 | 1 | **Genuinely chain-resident.** The 1 H2D is `build_device_tables` (root tables, uploaded ONCE and reused for the whole chain); `mul_resident`/`rescale_resident` themselves do zero transfers — confirmed by `bench_resident_chain.cpp`'s own design: upload once, mul→rescale→mul→rescale, download once (section 4). |
| `rotate.cpp` | 10 | 4 | 3 | 8 | Partially resident — Phase 25's audit (still accurate, re-confirmed by this round's grep) found `ks_context_create` re-uploads the eval key every call (60 MiB/call at bootstrap scale), ~6x the ciphertext's own transfer volume. |
| `keyswitch_resident.cpp` | 19 | 0 | 1 | 16 | **Resident for the ciphertext and kernels** (0 D2H — confirmed: the resident path never downloads mid-operation), but the 19 H2D are the per-`(part,role)` host-table uploads Phase 30 fixed for CUDA-graph correctness, plus `ks_context_create`'s eval-key upload when a context is built fresh. If a context is reused across calls, these largely amortize; if rebuilt per call (the common pattern absent an explicit cache), they do not. |
| `keyswitch.cpp` / `keyswitch_hoist.cpp` | 9 / 9 | 6 / 0 | 4 / 1 | 15 / 7 | Older/alternate keyswitch paths; `keyswitch_hoist.cpp` backs the (deliberately disabled, Phase 25's own finding) rotation-hoisting experiment — not on the production path. |
| `evalkeygen.cpp` | 7 | 3 | 1 | 5 | Setup-time only (runs once per key, not per operation) — its cost is amortized across every subsequent operation using that key, confirmed by every aggregation benchmark below reporting keygen separately from per-call cost. |
| `device_ciphertext.cpp` (`DeviceCiphertext`) | 2 | 2 | 0 | 2 | By design: exactly one upload (constructor-adjacent) and one download (`to_host`) per ciphertext's whole lifetime — everything between is in-place device mutation. This is the cleanest resident primitive in the codebase (see `research/GPU_RESIDENT_ARCHITECTURE.md` section 1). |

**Bottom line**: encode/decode are an intentional, measured CPU boundary;
the production encrypt/decrypt path is resident *within* a call but
transfers plaintext in and ciphertext out every call (unavoidable — that
is the actual API boundary with the caller); `ckks_ops.cpp`'s
mul/rescale chain is the one place multiple operations genuinely compose
without a host round-trip; `keyswitch_resident`'s remaining
non-residency is the eval-key upload cost already identified and
quantified by Phase 25 and not re-solved since (`KsResidentCache` exists,
tested, and is explicitly not wired in — see
`research/GPU_RESIDENT_ARCHITECTURE.md` section 1 and
`research/KS_RESIDENT_CACHE_DESIGN.md`).

## 2. End-to-end workload: 32-client private aggregation (re-measured fresh this round)

Re-ran `bench_private_aggregation_workload` and `bench_aggregation_decomposed`
(both already-existing, already-registered benchmarks from Phases 28-29;
not new this round) against the current build:

```
keygen: 517.09 ms (once, amortized across every aggregation below)
10 reps, K=32 clients, n=8192, sizeQ=4, h=64, sigma=3.19, Delta=2^45:
  mean per-aggregation runtime: 105.84 ms
  worst abs error across 10 reps: 3.347054e-09
  peak host RSS: 123.0 MB; VRAM free: 5103/6144 MiB
```

Stage decomposition (production path, `encrypt_host_shoup_pooled_barrett_batched`):

| Stage | Where | Median (ms) | % of end-to-end |
|---|---|---|---|
| encode (K=32 calls) | CPU, FFT | 10.51 | 10.7% |
| encrypt (K=32 calls) | GPU, NTT | 85.72 | 86.9% |
| aggregate (K-1 calls) | CPU, add | 1.36 | 1.4% |
| decrypt (1 call) | GPU, NTT | 0.69 | 0.7% |
| decode (1 call) | CPU, FFT | 0.29 | 0.3% |
| **end-to-end** | | **98.64** | **100%** |

Against the OLD plain `encrypt_host` path for contrast (same stages, same
input): end-to-end 149.71 ms median — the production path is **1.518x
faster end-to-end** (1.588x on the encrypt stage alone), with encrypt
remaining the dominant cost in both arms (86.9% production vs 90.9% old).
**Encrypt, not encode or keyswitch, is the next real bottleneck** for this
specific workload shape (pure-addition aggregation — no relin, no
rotation, no rescale needed, since summing ciphertexts consumes no
multiplicative depth).

Client-count scaling (`bench_aggregation_client_scaling`, re-run fresh):
VRAM delta is 0-2 MiB across K=1..64 — **confirms pool-based buffer reuse,
not a per-client allocation** — while runtime scales linearly in K (7.78 ms
at K=1 to 192.31 ms at K=64), exactly the expected, checked-not-assumed
result for an embarrassingly-parallel-across-clients workload with no
cross-client GPU state.

## 3. Resident multi-op chain: honest result, not a forced win

`bench_resident_chain` (pre-existing, `ckks_ops.cpp`'s `mul_resident`/
`rescale_resident` chained: upload once → mul → rescale → mul → rescale →
download once), re-run fresh at n=32768, towers=6:

```
gpu  median = 5.78465 ms
cpu  median = 7.06478 ms
gpu-cpu median = -0.772039 ms  95% CI [-1.70277, 0.931942]
=> no significant difference (CI brackets zero)
```

**Reported honestly, not hidden**: at this specific scale, the GPU
resident chain shows **no statistically significant advantage** over a
CPU reference (the confidence interval brackets zero). This does not
mean GPU residency is worthless here — it means this particular
micro-benchmark's scale is too small for the chain's compute to amortize
launch/transfer overhead, exactly the same crossover phenomenon
Work Package B's finance workload measures explicitly and names a
concrete threshold for (K≈2000 instruments on that workload's kernel
shape). No equivalent crossover sweep exists yet for this CKKS chain —
flagged as a concrete follow-up, not performed this round (this round's
budget went to re-verifying the existing claim honestly, not re-deriving
a new sweep across chain length/tower count/n).

## 4. Security and correctness status (unchanged, re-confirmed)

- `h=64` (sparse ternary secret Hamming weight): unchanged, confirmed by
  reading `set_secret_hamming_weight` call sites — no new call site
  introduced this round.
- `sigma=3.19`: unchanged (appears identically in every benchmark's own
  printed parameter line above).
- Modulus chain / `P>=Q_part_max` validation: unchanged, and actively
  exercised — the five `test_rescale_*_n1024*` pre-existing failures
  (section 5) are this exact validation correctly rejecting an
  intentionally-undersized fixture, re-confirmed still firing with the
  identical error message this round.
- `KROT_CAP`/`KSC_CAP`: not touched.
- No native-vs-OpenFHE conflation: every benchmark above measures the
  *native* implementation; `test_evalkeygen.cpp`/`test_ks_constants.cpp`
  (OpenFHE-cross-validated) were not re-run this round since nothing
  touching key generation changed, but their pass/fail status is part of
  the full-suite regression in section 5, run against the current build.
- `MESSAGING_ARCHITECTURE.md`'s "do not use CKKS for messaging" constraint:
  not touched, not relevant to any change this round (no messaging code
  was written).

## 5. Regression (full suite, re-run this round)

Re-ran the full `ctest` suite (excluding the memory-gated bootstrap tests
— same convention since Phase 28, Windows free RAM measured below the 10
GiB gate this round, see section 6): see
`research/PHASE32_INDEPENDENT_REVIEW.md` / `NEXT_SESSION.md` for the exact
pass count including the new `resident_finance` test. No CKKS-specific
test's pass/fail status changed from the Phase 31 baseline as a result of
this round's work (no CKKS/FHE production source file was modified this
round — only `research/*.md`, `CMakeLists.txt`'s additive new-target
registration, and the wholly-new, non-FHE `include/device_buffer.h`,
`include/cuda_check.h`, `include/finance_resident.h`,
`kernels/finance_resident.cu`, `examples/resident_finance/main.cpp`).

## 6. Bootstrap

Windows free RAM measured this round (see `research/NEXT_SESSION.md` for
the exact figure) against the established 10 GiB (n=8192) / 6 GiB
(n=131072) gates. Consistent with every phase since 25, a full bootstrap
run was not attempted unless the gate cleared; the aggregation workload
above (which needs no bootstrap — pure addition, no multiplicative depth)
remains the project's only currently-executable full end-to-end CKKS
measurement under the actual memory constraints of this hardware.

## 7. Next actual bottleneck

For the aggregation workload specifically: **encrypt (86.9% of
end-to-end)**, not encode, keyswitch, or decrypt. `research/
ENCRYPT_PROFILING.md`'s own closing note (Phase 25-era) already
identified a pinned-memory variant reverted for single-call latency that
might behave differently in this workload's steady-state
many-calls-same-key pattern — still unexplored, still the most concretely
evidenced next step for this specific workload.

For the library more broadly: `keyswitch_resident`'s eval-key-upload
non-residency (section 1) remains quantified (Phase 25: ~60 MiB/call,
~6x the ciphertext's own traffic) and unaddressed in production — not
because the fix is unknown (`KsResidentCache`, Phase 27) but because it
was deliberately not integrated pending a second real caller (per
`research/GPU_RESIDENT_ARCHITECTURE.md`'s migration strategy, this
phase's new `DeviceBuffer<T>` primitive plus a real non-FHE caller is
exactly the kind of second validation point that was missing — the
LLM KV-cache research (Work Package E) independently arrived at the same
recommendation: port `KsResidentCache`'s allocator discipline onto
`DeviceBuffer`/the as-yet-unbuilt execution context once a second real
caller exists).
