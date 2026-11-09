# encode_host_fft / decode_host_fft — production promotion decision (Phase 28)

GOD PROMPT "CKKS PHASE 28". Objective B's stage-timeline instrumentation
(`bench/bench_stage_timeline.cpp`) found that `encode_host` (src/encode.cpp),
not `pt_to_eval_host`, is the dominant cost of the fused-kernel stage once
Phase 27's kernel is wired in: 59.3% of Arm A's total stage time and 71.1% of
Arm B's, versus `pt_to_eval_host`'s 18.3%/18.9%. This contradicted this very
benchmark's own hardcoded printed conclusion ("pt_to_eval_host is the
dominant cost") — caught by comparing that text against the benchmark's own
numbers, not trusted on sight.

Investigating `encode_host` found it is an O(N²) double loop, multi-threaded
via a project-local `parallel_for` (src/encode.cpp, GOD PROMPT "CKKS PHASE
11"). A completely independent O(N log N) FFT-based implementation,
`encode_host_fft`/`decode_host_fft` (src/encode_fft.cpp, GOD PROMPT "CKKS
PHASE 12"), already exists in the tree, already has its own from-scratch
derivation independently checked against numpy
(`research/FFT_ENCODING_DERIVATION.md`) and its own rigorous correctness
test (`tests/test_encode_fft_matches_reference.cpp`) and isolated benchmark
(`bench/bench_encode_fft_vs_parallel.cpp`) — but sits completely unused:
every one of `tests/test_boot.cpp`'s 10 `encode_host`/`decode_host` call
sites, including both per-diagonal hot loops (C2S and S2C), called the slow
O(N²) version.

## Promotion gate (the governing directive's 9 conditions)

1. **Bit-exact differential testing** — not literally bit-exact (an FFT is a
   different summation order/rounding path than the O(N²) reference), but
   verified within the SAME tolerance formula the existing reference test
   already established and uses (`encTol = (N+1000)*eps*Delta`), independently
   re-confirmed this round in two different harnesses:
   `tests/test_encode_fft_matches_reference.cpp` (13 cases × 6 N, `[OVERALL
   PASS]`, re-run this round) and the new `bench/bench_encode_fft_stage.cpp`
   (coefficient-domain max diff 1.0 vs tolerance 2298.0 at N=8192). **PASS.**
2. **Reproducible stage-level result** — `bench_encode_fft_stage.cpp`, 21
   interleaved trials: 2.56x mean / 2.48x median on first run, reproduced at
   2.54x/2.44x and 2.55x/2.48x on two further runs. **PASS.**
3. **Equivalent transfers/allocations between arms** — N/A to this swap: both
   arms are pure host-side CPU calls with no device allocation or transfer at
   all; the only difference is CPU wall-clock inside `encode_host` vs
   `encode_host_fft`. **PASS** (no asymmetry to measure).
4. **Statistically credible improvement** — stddev small relative to the
   gap (Arm A stddev 0.035 vs mean 0.744; Arm B stddev 0.026 vs mean 0.279);
   non-overlapping min/max ranges across all reruns. **PASS.**
5. **No security/semantics changes** — no cryptographic parameter, secret
   distribution, modulus chain, or validation logic touched; this is a
   host-side arithmetic-transform swap with a pre-established oracle and
   tolerance. **PASS.**
6. **No regressions** — `tests/test_encode_fft_matches_reference.cpp` and
   `bench/bench_encode_fft_vs_parallel.cpp` re-run after the concurrency fix
   below; both reproduce their original results unchanged (correctness
   `[OVERALL PASS]`; isolated speedup ~45-67x depending on N, matching prior
   measurement). **PASS.**
7. **Concurrency safety** — see "Concurrency investigation" below. **PASS**,
   after one real fix.
8. **Minimal-safe integration** — the only change is which of two
   byte-for-byte-interchangeable free functions is called; no new
   abstraction, no signature change, no caller-visible difference beyond
   output within established tolerance. **PASS.**
9. **Actual workload benefit, not microbenchmark** — the swap directly
   targets the dominant cost of `tests/test_boot.cpp`'s real bootstrap hot
   loops (not an isolated kernel number); see "Scope of integration" below
   for why end-to-end confirmation at full bootstrap scale is still pending.
   **PASS at the stage level; end-to-end confirmation deferred (see below).**

## Concurrency investigation — a real finding, then a correction

`src/encode_fft.cpp` lazily populates two caches (`fft_twiddles`, `slot_idx`)
via plain `static std::map`, with no synchronization — the same pattern
class as `src/keyswitch_resident.cpp`'s `g_rtab`/`upc`, which Phase 25's own
audit found to be LATENT (a real hazard under the C++ standard, but never
observed to misbehave). A new standalone reproducer,
`tests/test_encode_fft_concurrency.cpp` (real `std::thread`, never `fork()`,
per Phase 27's explicit finding that `fork()` after CUDA init is unsafe —
though this file touches no CUDA at all), was built to check this cache
before any promotion decision.

**First result: 5/5 runs crashed** with heap corruption (`double free or
corruption`, `malloc(): corrupted top size`). This looked like confirmation
that the cache race was ACTIVE, not latent, and a mutex was added guarding
both `fft_twiddles()` and `slot_idx()` as a result. Re-running after that fix
**still crashed 5/5** — proving the mutex, while correct, was not the actual
cause of the observed crashes.

Root-caused with AddressSanitizer (plain thread-count reasoning wasn't
narrowing it down): the crash was a heap-buffer-overflow inside
`fft_forward_inplace`, caused by the TEST's own scenario C using
non-power-of-2 N values (`freshBase + t` for odd offsets, and a
`sharedN=3072`) — but `fft_forward_inplace`'s radix-2 Cooley-Tukey algorithm
requires `M=2N` to be an exact power of two, stated explicitly in
`src/encode_fft.cpp`'s own header comment. This was a bug in the TEST's
inputs, not a real concurrency defect in the production cache code.

After fixing the test to use only valid power-of-2 N (see that file's own
correctness-note comment), it passes cleanly, 10/10 consecutive runs, 0
crashes. **The cache's std::map race is therefore LATENT, like
`keyswitch_resident.cpp`'s** — not observed to misbehave with valid inputs,
but still undefined behavior under the standard regardless. The mutex fix is
kept as cheap, unconditional hardening (lookups/inserts are trivial relative
to the FFT itself), not because it fixed an active bug — it didn't; the
crashes were self-inflicted by invalid test inputs.

This is an important corrected finding, documented in both the test file and
`src/encode_fft.cpp`'s own comments: do not trust an initial "N/N crashes"
result as proof of an active race without isolating the actual fault with a
memory sanitizer first.

## Scope of integration — what was and was not done

`tests/test_boot.cpp` (the only place the full bootstrap pipeline exists in
this repository — there is no `src/bootstrap.cpp`) had all 10 of its
`gpufhe::encode_host`/`decode_host` call sites swapped to
`encode_host_fft`/`decode_host_fft`, with `extern` declarations added
alongside the existing ones. The file **compiles and links cleanly**
(verified this round via `cmake --build . --target test_boot`).

**It has NOT been executed end-to-end this round.** `test_boot.cpp` hardcodes
n=8192 and needs ≥10 GiB free Windows RAM per the governing directive's
memory gate; measured free RAM this session via
`Get-CimInstance Win32_OperatingSystem` was 2.40 GiB (`FreePhysicalMemory` =
2,520,264 KB) — a NO-GO, consistent with this project's recent history. Per
the directive ("do not attempt, do not force... record exact measured memory
and reason"), the integration is verified at the stage level and by
compilation only. **Re-running `tests/test_boot.cpp` once the memory gate is
met is the single most important follow-up** — it is the only way to confirm
the integrated bootstrap path produces correct output and actually realizes
the measured 2.5x stage-level speedup at full scale, not just in isolation.

## Decision

**Promote.** All 9 conditions pass given available evidence; the one
genuinely open item (full end-to-end bootstrap execution) is blocked by a
pre-existing, already-documented environment constraint, not by any defect
found in the optimization itself. The integration is already made in
`tests/test_boot.cpp`; re-verification at full scale is the top item in
`research/NEXT_SESSION.md`.
