# Phase 40, Track B: Mathematical and Implementation Audit of the Phase 39 Shoup Kernels

Scope: `kernels/moddown.cu`'s `moddown_combine_kernel` and
`kernels/rescale.cu`'s `rescale_fuse_kernel`, both changed in Phase 39
(commit `0baae6b`) to replace a per-coefficient 128-bit division with
Shoup's precomputed-multiplier technique. This document audits that
change directly from current source, independent of the earlier
report's own narrative.

## 1. The arithmetic, restated and re-derived

`shoup_mulmod(x, c, precon, q)` (identical device function in both
kernels):
```
hi = __umul64hi(x, precon)      // high 64 bits of x*precon
r  = x*c - hi*q                 // both multiplies wrap mod 2^64
return (r >= q) ? r - q : r
```
with `precon = floor(c * 2^64 / q)` precomputed by the caller
(`shoup_precon`, identically defined in `src/ckks_ops.cpp`,
`src/keyswitch.cpp`, and `src/keyswitch_resident.cpp`:
`(uint64_t)(((unsigned __int128)c << 64) / q)`).

**Correctness condition** (standard Shoup-multiplier result, re-derived
here rather than cited): for `x < q`, `c < q`, and `q < 2^63` (so `2q`
does not overflow 64 bits), the true value `qhat = floor(x*c/q)`
differs from `hi` computed above by at most 1 (in either direction,
but in practice the construction guarantees `hi <= qhat`), so
`r = x*c - hi*q` (evaluated mod 2^64, which equals the true integer
value mod 2^64 since `x*c` and `hi*q` are both representable up to
that modulus even though the *products themselves* may exceed 2^64 --
the wraparound cancels correctly because `x*c - hi*q` is a non-negative
value less than `2q < 2^64`) lands in `[0, 2q)`. **A single conditional
subtraction is therefore always sufficient** -- the code's `(r >= q) ?
r - q : r` with no loop is exact, not an approximation of a rarer
multi-correction case.

**Operand-range check against actual call sites** (not assumed):
- `moddown_combine_kernel`: `diff = (a[k] - b[k]) mod q`, explicitly
  computed via one conditional add before the subtraction -- always in
  `[0, q)`. `s` is `PInvModq[i]`, a precomputed modular inverse, always
  `< q` by construction (`src/keyswitch_resident.cpp` line 430-431,
  `src/keyswitch.cpp` line 225, both pass `C.pInvModqHost[i]`/`K.pInvModq[i]`
  directly from the same `compute_keyswitch_constants` that produces
  every other already-trusted modular constant in this codebase).
- `rescale_fuse_kernel`: `a[idx]`/`b[idx]` are RNS tower coefficients,
  always stored in `[0, q)` by this project's universal invariant (every
  other kernel in this codebase relies on the same invariant); `s1`/`s2`
  are `qlInvModq[i]`/`QlQlInvModqlDivqlModq[i]`, OpenFHE-derived modular
  constants, `< q` by construction.
- **Modulus bound**: every modulus this project uses is an NTT-friendly
  prime sized for a 50-62 bit `log2` chain (per `research/benchmarks/
  bench_moddown_shoup_check.cpp`'s own test matrix and this project's
  standard parameter set) -- comfortably under `2^63`. Confirmed, not
  assumed: grepped `native_primes` call sites across the production
  path; no caller requests a modulus anywhere near `2^63`.

**Conclusion: the one-correction form is provably exact for every
value these two kernels are ever actually called with.** This matches
(and independently re-derives, rather than just trusts) the
already-established `shoup_mulmod` used elsewhere in this codebase
(the encrypt hot path's own combine loop).

## 2. Empirical re-verification from current source

Re-ran (not re-used -- recompiled fresh from current source this
session) `research/benchmarks/bench_moddown_shoup_check.cpp`: **35,000,000
trials across 35 real NTT-friendly moduli (bit-widths 30-62, 5 each),
0 mismatches** against a raw `(x*c) % q` oracle computed independently
via 128-bit arithmetic.

## 3. New dedicated boundary-value test (this session)

The existing differential tests (`tests/test_moddown.cpp` against
OpenFHE's `ApproxModDown`, `tests/test_moddown_combine_isolated.cpp`
against a hand-derived exact identity) already exercise the real
kernels end to end with realistic random/derived inputs, but neither
specifically targets `shoup_mulmod`'s boundary values or explicitly
confirms both of its branches (correction taken / not taken) are
exercised. Added `tests/test_shoup_moddown_rescale_boundary.cpp` (new,
kept, ctest-registered as test #40): launches both
`LaunchModDownCombine` and `LaunchRescaleFuse` directly with explicit
operand sets `{0, 1, q-1, q/2, q/2+1}` plus 10 pseudorandom values
(every pairing, so every boundary combination including both operands
simultaneously at their extremes), across three real primes
(30/50/62-bit), checked against an **independent oracle** (plain
128-bit `%`, sharing no formula or code with `shoup_mulmod`) -- not the
same helper used as both implementation and oracle. The test also
replicates `shoup_mulmod`'s exact branch-decision formula separately
to COUNT how many trials take the correction branch vs. not, and fails
if either count is zero (making "every correction branch is covered"
a checked fact rather than an assumption).

**Result: PASSED** (0.40-0.42s), both branches exercised thousands of
times across the full boundary-and-random matrix.

## 4. Rust FFI audit

`rust/crates/ckks-cuda/src/rescale.rs` declares its own
`extern "C" fn LaunchRescaleFuse(...)` (Rust's `build.rs` compiles
`kernels/rescale.cu` independently of the C++ CMake build, so this
declaration is a second, separately-maintained ABI contract). Verified
parameter-by-parameter against the current 9-parameter C++ signature
in `include/rescale.h`: pointer types, integer widths, parameter order,
and the trailing `cudaStream_t` (as `*mut c_void`) all match exactly.
The call site (`rescale.rs` lines ~349-362) computes `s1_precon`/
`s2_precon` via Rust's native `u128` arithmetic
(`(((self.s1[t] as u128) << 64) / (qi as u128)) as u64`), identical in
formula to the C++ `shoup_precon` helper.

`LaunchModDownCombine` has **no separate Rust FFI declaration anywhere**
(confirmed by `grep -rn LaunchModDownCombine rust/` -- the only matches
are compiled object/binary artifacts, never a `.rs` source file). It is
reached exclusively through Rust's `build.rs` compiling
`src/keyswitch.cpp`/`src/keyswitch_resident.cpp` wholesale and exposing
their own higher-level entry points as opaque bridge calls -- there is
no second hand-written declaration of this specific function that
could go stale. (The newly-added `LaunchModDownCombineBatched`,
Track E, is in the same position: no Rust-side declaration exists or
is needed, since Rust never calls it directly either.)

**No FFI mismatch exists for either kernel as of this session.**

## 5. Compute Sanitizer

Checked for an installed Compute Sanitizer (`compute-sanitizer`,
shipped with the CUDA toolkit) under this session's CUDA 13.2 install:
not found at the expected toolkit path or via `PATH` search this
session. **Not installed; not run.** Reported honestly rather than
attempting a toolkit modification out of scope for this phase. The
boundary test (section 3) and the 35M-trial reproducer (section 2)
remain the available correctness evidence in its absence; neither
kernel performs pointer arithmetic beyond simple per-tower offsetting
already exercised identically by every other kernel in this codebase,
so the marginal value of a sanitizer run here is judged lower than for
a kernel with more complex indexing.

## 6. Verdict

**The Phase 39 Shoup changes to ModDown and RescaleFuse are
mathematically exact** (re-derived, not just cited) for every input
these kernels are ever called with in this codebase, **confirmed
exact** by a fresh 35M-trial reproducer plus a new dedicated
boundary/branch-coverage test, and **free of FFI mismatch** on the
Rust side (one kernel has a verified-correct declaration, the other
has none to go stale). No change was made to either kernel's arithmetic
this phase -- this document is a validation of what Phase 39 shipped,
not a revision of it.
