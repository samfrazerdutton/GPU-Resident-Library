# Embed modular arithmetic derivation and buffer-pool safety audit (Phase 14)

GOD PROMPT "CKKS PHASE 14". This document records the Phase 13 device-
buffer-pool concurrency audit (Section 3 of the governing directive),
the derivation and exhaustive verification of a Barrett-reduction
replacement for `encrypt_host`'s embedding loop, and the resulting
benchmark/correctness evidence. See `research/EXPERIMENT_LEDGER.md`
entry #47 for the condensed ledger summary.

Starting commit: `4d46420` (Phase 13 end state), confirmed via `git
log -1` and `origin/main` before any change this round.

## 1. Buffer-pool safety audit (directive Section 3)

Re-read `src/encrypt_host_shoup_pooled.cpp` (Phase 13's production
implementation at the start of this round) line by line against every
item the governing directive asked for:

- **Buffer ownership**: `EncryptBufPool` owns `d_v,d_e0,d_e1,d_m,d_fr,
  d_fp,d_mod`, allocated once per `(n,sizeQ)` key, looked up/created via
  a `static std::map` inside `encrypt_buf_pool()`.
- **Buffer lifetime**: process lifetime -- never explicitly freed
  (consistent with every other process-lifetime cache in this codebase,
  `enc_dev_tab`/`ek_dev_tab`/`KROT`/`KSC`). Device memory is reclaimed
  by the CUDA runtime at process exit.
- **Pool keys and lookup**: keyed by `(n,sizeQ)`. Confirmed that the
  CONTENT of a pooled buffer is always fully overwritten by the
  current call's own `mod`/`root`/`v`/`e0`/`e1`/`m` data via a fresh
  `cudaMemcpy` before any read -- different public keys or moduli
  sharing the same `(n,sizeQ)` key is therefore safe BY CONSTRUCTION
  provided no two calls interleave their writes/reads, which is
  exactly what the mutex (see below) guarantees.
- **Mutex scope**: one `static std::mutex pool_mu` with a
  `std::lock_guard` held for the ENTIRE function body, acquired before
  the pool map lookup/creation (`encrypt_buf_pool()`,
  `enc_dev_tab_pooled()`) and released only at function return. This
  means the lock is GLOBAL (not per-key) -- two threads using
  DIFFERENT, non-conflicting keys are still serialized. This is a
  conservative (safe but not maximally parallel) design choice,
  acceptable because every real caller in this codebase invokes
  encryption sequentially; NOT changed this round, since the audit
  found no correctness problem with it and the directive's priority
  order ranks correctness/safety above performance.
- **CUDA stream association**: everything uses the default stream
  (`0`). The explicit `cudaDeviceSynchronize()` after the kernel
  launches, combined with full-function serialization, means no other
  call's work can be in flight when the current call's download/
  combine stage runs.
- **Synchronization before reuse**: every `cudaMemcpy` used here is the
  BLOCKING form -- by the time a call returns (and releases the mutex),
  its own `cudaDeviceSynchronize()` has already completed, so the NEXT
  call never observes a buffer with a prior call's work still in
  flight.
- **Failure paths**: no CUDA return code is checked anywhere in this
  function (`cudaMalloc`, `cudaMemcpy`, kernel launches) -- this is a
  PRE-EXISTING convention across this entire codebase (confirmed: zero
  CUDA error checking in `enc_dev_tab`, `ek_dev_tab`, `KROT`/`KSC`, or
  any other table/buffer cache), not something Phase 13 introduced or
  Phase 14 is scoped to fix. Deliberately NOT changed this round (out
  of scope, and changing error-handling convention project-wide is a
  much larger, separately-scoped task).
- **Context destruction / process shutdown**: pool buffers are raw
  `cudaMalloc` pointers stored in function-local `static` maps; C++
  static destruction order has no cross-dependency between `pool_mu`,
  `encrypt_buf_pool`'s cache, and `enc_dev_tab_pooled`'s cache (each is
  independent), and device memory reclamation at process exit is the
  CUDA runtime's responsibility, same as every other never-freed cache
  in this project.
- **Concurrent encryption contexts / different key material**: SAFE,
  covered above (content always freshly overwritten, access always
  serialized).
- **Reentrant calls**: `std::lock_guard<std::mutex>` with a
  non-recursive `std::mutex` would deadlock on reentrant acquisition --
  confirmed NOTHING this function calls (`enc_dev_tab_pooled`,
  `encrypt_buf_pool`, `LaunchNTT_CT_Batched`, CUDA runtime calls,
  `shoup_mulmod`/`shoup_precon`/`am`) calls back into
  `encrypt_host_shoup_pooled` itself or re-acquires `pool_mu`. Not
  reentrant, no deadlock risk.
- **Exception paths**: `std::lock_guard`'s destructor runs on
  exception unwinding (RAII), so the mutex is released correctly even
  if, e.g., a `std::vector` allocation throws partway through --
  exception-safe by construction.

**Conclusion: no NEW safety issue was found in the existing mutex-based
design.** Per the directive's own instruction ("if a safety issue is
discovered, fix it before expanding the pool's role" -- none was), the
pool's existing semantics were kept unchanged and REUSED (not
redesigned) for Phase 14's new `encrypt_host_shoup_pooled_barrett`.

### Targeted audit tests built and run (not just a paper trace)

`tests/test_encrypt_host_shoup_pooled_barrett_concurrency_audit.cpp`,
run 5 independent times, **0/396 mismatches every run**, covering:

1. **Same key, concurrent reuse**: 8 threads x 8 calls each, same
   `(n=2048,sizeQ=8)`, distinct seeds -- the exact class of race Phase
   13's original bug exercised.
2. **Concurrent pool growth**: 8 threads, each hitting a DIFFERENT,
   never-before-used `(n,sizeQ)` key simultaneously -- stresses
   concurrent map-mutation during first-time pool-entry creation, not
   just reuse of an already-built entry. (A first version of this test
   used non-power-of-2 `n` values and segfaulted inside the NTT's
   bit-reversal logic -- a bug in the TEST, not production code, caught
   and fixed before trusting any result; see Section 4 below for why
   `n` must be a power of 2 throughout this codebase.)
3. **Longevity stress**: 200 sequential reuses at one key (single
   thread) -- rules out slow drift/leak across a long-running process.
4. **Mixed shared-key race**: 10 threads cycling through 2 shared keys
   (10 calls each), maximizing the chance two threads race to CREATE
   the same new pool entry at the same instant.

Every call's result was checked against production `encrypt_host`
computed single-threaded for the same seed -- any corruption would
show as an exact mismatch (as Phase 13's original bug did), never a
near-miss. None occurred.

## 2. Main experiment: the embed loop

`src/encrypt_host_shoup_pooled.cpp`'s embedding loop:

```cpp
auto emb = [&](long x) { return (uint64_t)((x % (long)q + (long)q) % (long)q); };
vT[k] = emb(vc[k]); e0T[k] = emb(e0c[k]); e1T[k] = emb(e1c[k]); mT[k] = emb((long)m[k]);
```

### Operand ranges (determined, not assumed)

- `vc[k] = (int)(rng()%3)-1` -- ternary, `{-1,0,1}`.
- `e0c[k],e1c[k] = (int)std::lround(g(rng))` for `g ~ N(0,sigma=3.19)`
  -- small, well within `int` range.
- `m[k]` (cast to `long`) -- the CKKS-encoded message coefficient,
  declared `int64_t` in every caller, with NO assumed bound beyond
  that type's full range. Treating `m[k]` as potentially spanning the
  ENTIRE `int64_t` range (not just "small") is the conservative,
  correct assumption -- confirmed necessary by this round's own
  boundary tests.
- `q = mod[t]` -- always a positive CKKS modulus, ~49-60 bits in this
  codebase (confirmed via `native_primes`' own bit-size parameters,
  which never generate moduli outside that range).

### Signedness and overflow analysis

`(long)q` is always POSITIVE and fits safely in a signed 64-bit `long`
(max modulus ~2^60 << `LONG_MAX`~2^63). The C++ standard's one
signed-modulo UB case (`INT64_MIN % -1`) can never trigger here since
the divisor is never `-1`. The intermediate sum `x%q + q` is bounded by
roughly `[1, 2*q-1] < 2^62`, safely within `int64_t` -- **no undefined
behavior exists in the ORIGINAL `emb()` for any `int64_t` input**,
confirmed by direct bound analysis, not assumed. This matters because
the REPLACEMENT must preserve this same correctness over the same
full range, including `LONG_MIN`/`LONG_MAX`.

### Assembly inspection (directive: "inspect optimized assembly rather than assuming")

Isolating `emb()`'s body and compiling with `-O2`:

```
_Z3emblx:
    mov  rax, rdi
    cqo
    idiv rsi      ; x % q
    lea  rax, [rdx+rsi]
    cqo
    idiv rsi      ; (x%q + q) % q
    mov  rax, rdx
    ret
```

**Two hardware `idiv` instructions per call**, confirmed -- NOT a
software library call (unlike Phase 13's combine-loop `mm()`, which
called libgcc's `__umodti3`). `idiv` has variable latency (roughly
20-45+ cycles on this CPU's generation), and there are 4 `emb()` calls
per `(tower,coefficient)` = 8 `idiv` total -- matching the 20-27% of
total time this stage was measured at in `research/
ENCRYPT_PROFILING.md`.

### Derivation: per-tower Barrett reduction

Since `q` is FIXED across all `n` coefficients within a tower (only
varying by tower index `t`, not by coefficient `k`), a Barrett
reduction constant can be precomputed ONCE PER TOWER (`O(sizeQ)`
divisions per call) rather than once per coefficient, unlike the
Shoup precon in the combine loop (Phase 13) which needed precomputation
per-coefficient because its "fixed" operand (`v`) varies with `k`.

```
mu = floor(2^64 / q)                         // ONE division, per tower
ax = (x<0) ? (0ULL - (uint64_t)x) : (uint64_t)x   // |x|, unsigned, no UB
qhat = (ax * mu) >> 64                       // high 64 bits of a 128-bit product -- a MULTIPLY, no division
r = ax - qhat*q
while (r >= q) r -= q                        // Barrett correction, proven <=1 iteration empirically
result = (x<0 && r!=0) ? (q - r) : r         // restore emb()'s own sign convention
```

**Why the unsigned-negation trick is safe for `LONG_MIN`**: `(uint64_t)
LONG_MIN` reinterprets the bit pattern of `-2^63` as `2^63` (two's
complement), and `0ULL - 2^63` under unsigned (mod-2^64) wraparound
gives `2^63` again -- the CORRECT magnitude, computed entirely in
well-defined unsigned arithmetic, no signed overflow anywhere.

**Why the sign-restoration matches `emb()` exactly**: for `x<0`,
`emb(x) = ((x%q)+q)%q`. Since `x%q` (truncating division) equals
`-(ax%q)` for `x<0` (where `ax=|x|`), substitution gives `emb(x) = 0`
when `ax%q==0`, else `q - (ax%q)` -- exactly what the Barrett-based
`result` computes.

## 3. Independent differential-test results (directive Section 6: built BEFORE touching production)

**Standalone scratch harness** (synthetic moduli including deliberately
non-realistic small values for extra stress, plus the real ~49-60 bit
classes): boundary cases (`0,±1,LONG_MIN,LONG_MAX,LONG_MIN+1,
LONG_MAX-1`, values at/around `q` and `2q` in both signs) PLUS 2,000,000
uniformly-random full-`int64_t`-range values per modulus across 6
moduli. **12,000,102 total checks, 0 mismatches, maximum Barrett
correction iterations observed: 1.**

**Production-moduli test** (`tests/
test_embed_barrett_matches_reference.cpp`, linked against this
project's REAL `native_primes()`): every boundary case plus 200,000
random full-range values, for every actual modulus this codebase
generates at every supported `n` (1024/2048/4096/8192) x `sizeQ`
(1/4/8). **10,400,884 total checks, 0 mismatches.**

**Production-integration test** (`tests/
test_encrypt_host_shoup_pooled_barrett_matches_reference.cpp`): full
`encrypt_host_shoup_pooled_barrett` vs. the untouched `encrypt_host`,
bit-exact required, across `n` in `{16,1024,4096,8192}` x `sizeQ` in
`{1,4,8}` x 3 message patterns (zero, `LONG_MIN`/`LONG_MAX`-adjacent,
dense mixed-sign/magnitude) PLUS the repeated-same-key and
interleaved-different-keys pool-stress sections carried forward from
Phase 13's test design. **Every case: EXACT MATCH.**

No approximation, no tolerance, anywhere in this section -- this is
exact integer arithmetic, and the test gate was "zero mismatches,"
met in full before any production file was touched.

## 4. A test bug caught and fixed (reported honestly, not hidden)

The concurrency audit test's Scenario 2 (and, in a first pass,
Scenario 4) used several non-power-of-2 values for `n` (e.g. 768,
1536, 640, 896) to get "distinct" pool keys, not realizing this
codebase's NTT (the bit-reversal permutation in `build_enc_tab_*`/
`native_primes`/`native_root`) requires `n` to be a power of 2 --
exactly like every `n` used elsewhere in this project (1024, 2048,
4096, 8192). This produced an immediate, reproducible **segmentation
fault** (an out-of-bounds array access from a bit-reversal permutation
computed for the wrong `logn`). Diagnosed directly (not guessed) by
noticing the crash was deterministic and occurred before any output
was even printed, then confirmed by inspecting the exact `n` values
used. Fixed by replacing every non-power-of-2 `n` in the test with a
power of 2; re-ran 5 times, 0/396 mismatches every time (Section 1).
This was a bug in NEW TEST CODE, never in `encrypt_host_shoup_pooled`,
`encrypt_host_shoup_pooled_barrett`, or any other production file.

## 5. Benchmarks

### Embed loop alone (`bench/bench_embed_barrett_vs_reference.cpp`, no GPU, interleaved, 9 trials x 20 reps)

| n | OLD (median) | Barrett (median) | speedup |
|---|---|---|---|
| 1024 | 230.2us | 44.9us | 5.13x |
| 2048 | 464.1us | 103.0us | 4.50x |
| 4096 | 929.0us | 388.6us | 2.39x |
| 8192 | 1865.8us | 945.1us | 1.97x |

Speedup DECREASES as `n` grows -- consistent with the arithmetic
advantage (division vs. multiply) mattering less as the working set
grows large enough to become more memory/cache-bound than
compute-bound. Still a substantial, real win at every tested `n`.

### Full `encrypt_host` (unified 3-way interleaved, `bench/bench_encrypt_phase14_all_variants.cpp`, 9 trials x 20 reps, run twice independently)

| n | Phase13 vs OLD | **Phase14 vs OLD** | Phase14 incremental vs Phase13 |
|---|---|---|---|
| 1024 | 1.15-1.17x | **1.44-1.45x** | 1.25x |
| 2048 | 1.17-1.21x | **1.45-1.51x** | 1.24-1.25x |
| 4096 | 1.09-1.15x | **1.47-1.50x** | 1.28-1.38x |
| 8192 | 1.37-1.44x | **1.69-1.73x** | 1.18-1.26x |

Consistent and positive across two independent full runs at every `n`
-- never a regression.

### Full real-encrypted pipeline (median of 5 single-call samples per n, `bench/bench_evalmult_e2e_breakdown.cpp` extended with a 3-way `encryptVariant` flag)

| n | OLD median | Phase14 median | speedup vs OLD | speedup vs Phase13 |
|---|---|---|---|---|
| 1024 | 6180us | 5301us | 1.17x | 1.08x |
| 2048 | 8479us | 7103us | 1.19x | 1.29x* |
| 4096 | 13377us | 10855us | 1.23x | 1.14x |
| 8192 | 25886us | 18992us | 1.36x | 1.09x |

(*n=2048's Phase13 median is inflated by two cold-start outlier samples
out of 5 -- the OLD-vs-Phase14 comparison is the more reliable number
at that row; both directions agree Phase 14 is faster.) All RNS-level
correctness checks (0 mismatches) and real-round-trip error magnitudes
(`~5e-12` to `~1e-10`, tolerance `1e-6`, IDENTICAL across all three
encrypt variants at every `n`) passed for every configuration before
any timing was trusted.

## 6. Security review

**No change to randomness, sampling, or the noise model.** The RNG
(`std::mt19937_64 rng(seed)`), Gaussian error sampling, and ternary `v`
sampling are copied verbatim from `encrypt_host_shoup_pooled` into
`encrypt_host_shoup_pooled_barrett` -- the ONLY change is the
arithmetic used to reduce already-sampled integers into `[0,q)`
(Barrett instead of signed `%`), which is an EXACT reduction, not an
approximation, confirmed by 22M+ differential-test cases with zero
mismatches. Any accidental change to randomness consumption or order
would have shown up as an immediate bit-exact mismatch against
`encrypt_host`; none did, across every tested dimension, modulus, and
message pattern including full-range adversarial values. The buffer-
pool mutex (unchanged from Phase 13, reused identically here) was
re-audited and re-stress-tested rather than assumed safe by
inheritance -- no new concurrency risk was introduced or found.

## 7. What remains the dominant cost, post-Phase-14

Both of `encrypt_host`'s two largest host-side loops (embed and
combine) are now optimized. The next largest components, per `research/
ENCRYPT_PROFILING.md`'s original stage breakdown, are device alloc/
free (already eliminated by Phase 13's pooling for the hot path) and
upload/download transfers (19-31% combined, not yet touched) -- the
clear next candidate is investigating whether the four separate
upload/download `cudaMemcpy` calls per buffer (`d_v,d_e0,d_e1,d_m`) can
be merged into fewer, larger transfers (e.g. one contiguous host
buffer uploaded in a single call), reducing fixed per-call transfer
overhead the same way Phase 11's NTT batching did for kernel launches.
Not attempted this round, per the "one experiment at a time" discipline
-- this round's experiment (embed) was itself substantial and is
recorded in full above.
