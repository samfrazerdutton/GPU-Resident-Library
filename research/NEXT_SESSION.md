# Next session — exact continuation

## Current commit

Check `git log -3` on resume. As of this writing, HEAD is a new commit
on top of entry #43's commit `653cf6c` (the GOD PROMPT "CKKS PHASE 11"
round: the first real-world per-stage performance baseline for
encode->encrypt->EvalMult->decrypt->decode, plus two verified, kept
optimizations -- parallelized encode/decode and batched encrypt_host
NTTs -- giving a 2.6x-5.3x total pipeline speedup with zero
correctness regression. See `EXPERIMENT_LEDGER.md` entry #44 for full
detail). Confirm push status -- push again if `git status` shows ahead
of origin.

**Reminder, confirmed the hard way in an earlier round**: this machine
has (at least) two SEPARATE git clones of this GitHub repo -- the WSL
one (`~/GPU-Resident-Library`) is authoritative for all Rust/relin/CKKS
work and should always be checked via a WSL shell, not a Windows one;
a Windows-filesystem clone (`C:\Users\samca\GPU-Resident-Library`) has
an entirely unrelated commit history (raw CUDA-kernel-optimization
work) and untracked, stale copies of `rust/`/`research/` -- never run
`git status`/`log`/commit decisions against it for this work.

## READ THIS FIRST if resuming into performance work: host-side encode/encrypt/decode dominate, NOT the GPU kernels -- the clear next experiment is an O(n log n) FFT-based encode/decode

**Phase 11's core finding**: for a real encode->encrypt->EvalMult->
decrypt->decode round trip at n=8192, the ENTIRE GPU-resident
tensor+keyswitch+combine+rescale chain is only ~1% of total latency.
The dominant cost is three pure-host (or host-orchestrated) stages:
`encode_host`/`decode_host` (O(n^2) naive canonical-embedding DFT) and
`encrypt_host` (originally 4*sizeQ unbatched per-tower NTT round
trips). Two fixes were made this round, BOTH kept, BOTH bit-exact
verified against independent oracles, BOTH measured via interleaved
A/B:
1. `src/encode.cpp`: `encode_host`/`decode_host` parallelized via
   `std::thread` across the O(n^2) outer loop (each output element is
   an independent read-only reduction -- zero change to the
   arithmetic, bit-identical results guaranteed and confirmed).
   ~5.7-6.8x speedup on encode/decode specifically.
2. `src/encdec.cpp`: `encrypt_host` batched (one upload + one
   `LaunchNTT_CT_Batched` call + one download per buffer, across all
   `sizeQ` towers, with root tables cached by `(n,q)` -- the SAME
   pattern `src/evalkeygen.cpp` already uses successfully) instead of
   32 separate per-tower-per-buffer GPU round trips at `sizeQ=8`.
   ~2.2-2.8x speedup on encrypt specifically.

**Combined result**: total end-to-end latency dropped from
30.5/48/154.7/522.9ms to 11.8/16.6/38.2/98.9ms at
n=1024/2048/4096/8192 respectively (2.6x/2.9x/4.1x/5.3x -- the speedup
GROWS with n). Every real encrypted round trip still passes, with the
SAME error magnitude as before the optimization at every n (bit-exact
identical cryptographic output, not merely "close"). Full Rust suite
unaffected: 38 passed, 0 failed, 1 ignored.

**POST-optimization ranked bottleneck, n=8192** (the next thing to fix
if performance work continues): Encode 53.0ms (54%); Decode 23.2ms
(23%); Encrypt 17.4ms (18%); Relin/keyswitch (GPU) 1.96ms (2%); Rescale
(GPU) 1.63ms (1.6%); Decrypt 0.54ms; Tensor+Combine (GPU) ~0.07ms.
Host-side encode+decode+encrypt STILL account for ~95% of total
latency -- the GPU kernels are not, and were never, the bottleneck at
any tested n.

**The clear next highest-value experiment, NOT attempted this round**:
replace `encode_host`/`decode_host`'s O(n^2) algorithm with a true
O(n log n) FFT-based canonical embedding (the standard technique used
by OpenFHE/SEAL/HElib) -- this would address the now-dominant ~77% of
remaining latency, far more than parallelizing the O(n^2) loop already
did. NOT pursued this round because it requires correctly re-deriving
the existing `r_k=5^k mod 2N` permuted-slot-ordering convention against
a standard radix-2 FFT -- meaningfully higher correctness risk than
either of this round's two kept optimizations, and the governing
directive's own experiment discipline ("do not change several
independent components... unless the combined effect is the explicit
subject") argues for tackling it as its own dedicated, carefully-tested
round rather than folding it into this one. A secondary, lower-value
candidate also NOT pursued: a persistent (reused-across-calls) thread
pool for encode/decode to avoid per-call `pthread_create`/`join`
overhead (estimated ~1-3% of encode's own cost -- not worth the added
lifecycle complexity given the ALREADY-achieved 6x win from the simple
per-call thread-spawn approach).

**Do not re-measure the OLD (sequential/unbatched) baseline numbers as
if they still apply** -- `bench/bench_evalmult_e2e_breakdown.cpp` is
the current, correct per-stage benchmark; re-run it fresh rather than
citing this entry's own baseline table, which predates both
optimizations.

## Phase 10 recap (for key-switch/EvalMult correctness context, superseded as the primary focus by Phase 11's performance work above): the real encrypted EvalMult AND key-switch round trips both PASS. Invalid configurations are REJECTED at construction time, not just avoided by convention.

**The multi-phase EvalMult/key-switch investigation (Phases 5-10) is
RESOLVED.** Every prior round's "defect" hypothesis (entry #30/#38's
relin identity, entry #40's `evalkeygen_host`, entry #41's
`sizeP=2`-specific key-switch reconstruction) is understood as either a
test methodology bug or an incomplete parameter sweep -- NOT a defect
in `src/` or `kernels/`. The real, final cause (entry #42): a key-switch
digit's own modulus product (`Q_part`) must not exceed the auxiliary
modulus `P` (`P >= Q_part_max`, the standard RNS-CKKS Hybrid
key-switch requirement). This round (entry #43) went further:

1. **Implemented the EXACT check as code**, not just a derived fact in
   a research doc: `include/ks_params_validate.h` + `src/
   ks_params_validate.cpp`'s `check_hybrid_keyswitch_params`, using
   true arbitrary-precision integer products (a little-endian
   base-2^64 BigUint), never a bit-length approximation. Cross-checked
   independently against Python's native bigints
   (`tools/verify_ks_params_validate.py`) -- byte-for-byte decimal
   agreement, a completely different implementation stack.
2. **Enforced it at `ks_context_create`** (`src/keyswitch_resident.cpp`),
   the single construction-time choke point every real caller passes
   through (the Rust FFI bridge, hence both `KeySwitchContext::new`
   and `EvalMultContext::new`; every C++ benchmark/example that builds
   a usable context directly). Invalid configurations now throw
   `std::invalid_argument` with the actual `P`/`Q_part_max` decimal
   values in the message, BEFORE any eval-key generation or GPU
   context work would otherwise proceed.
3. **Audited every hardcoded fixture in the repository** and found
   FOUR MORE invalid ones entry #42's audit missed (it only checked
   `rust/crates/ckks-cuda/src/`): `examples/bench_evalmult.rs`,
   `examples/bench_keyswitch.rs` (both `NUM_PART=2`), and `bench/
   bench_evalmult_cpp_vs_rust.cpp`, `bench/bench_keyswitch_cpp_vs_rust.cpp`
   (both `numPart=2`, direct C++, registered as auto-run `ctest`s --
   these would have broken the test suite outright once validation
   landed, not just produced a misleading benchmark number). All four
   fixed to `NUM_PART=4`/`numPart=4`, matching `evalmult.rs`'s own
   Phase 9 fix.
4. **Fixed `keyswitch.rs`'s own fixture** (the GOD PROMPT's explicitly
   named target): `NUM_PART` 2->4, plus a NEW decrypt-based correctness
   test (`real_decrypt_based_relin_round_trip`) that builds a real
   ciphertext, relinearizes it through this module's real `keyswitch`
   call, rescales, decrypts, decodes, and compares against the
   plaintext product computed in the clear -- max abs error `2.23e-11`
   (tolerance `1e-6`). This module's tests had NEVER done this before;
   every prior test checked only `keyswitch` against `keyswitch_
   reference` (self-consistency between two implementations of the
   SAME formula), which is blind to whether the formula's own
   precondition holds.

Full Rust suite: **38 passed, 0 failed, 1 ignored** (up from 34/1 at
entry #42), identical in `--release` and debug.

**The general lesson (restate this if the pattern recurs)**: a passing
RNS-level or self-consistency test does NOT prove a parameter
configuration satisfies the real ciphertext decryption invariant.
`keyswitch.rs`'s invalid fixture passed every one of its own tests for
multiple rounds (entry #36 onward) precisely because none of them
checked a real decryption invariant.

**Audited and confirmed VALID (no fix needed)**: `tests/
test_ks_raii.cpp`, `tests/test_ks_context_create_validation.cpp`
(`n=8192,sizeQ=4,sizeP=2,numPart=2`, alpha=2); `tests/test_hoist_*.cpp`,
`tests/test_rot_resident.cpp`, `bench/bench_hoist_multi.cpp` (all
`n=8192,sizeQ=30,sizeP=2,numPart=15`, alpha=2 -- confirmed via the
exact validator, not assumed); `tests/test_ks_resident.cpp` (OpenFHE's
own `HYBRID`-technique parameter choice at `n=32768`, re-built fresh
with validation active, still bit-exact). **Trusted, not
independently re-derived**: `bench/bench_evalmult.cpp`/`bench_evalmult_
resident.cpp` (OpenFHE-sourced `GetNumPartQ`/`GetParamsP`). **Note**:
`tests/test_hoist_stage0.cpp`/`test_hoist_allstages.cpp` carry a STALE
comment ("P must cover a part (10x50=500 bits), hence sizeP=9x60=540
bits") describing a different, hypothetical alpha=10 configuration the
code does NOT actually use (the real `npFor` lambda gives alpha=2) --
not fixed this round, flagged as a readability issue only, no
correctness impact.

**Deliberately exempt, never call `ks_context_create`, unaffected by
this round's enforcement**: `tests/test_evalkeygen_isolated.cpp`,
`tests/test_ks_pq_ratio_sweep.cpp`, `tests/test_keyswitch_stage_trace.cpp`,
`tests/test_moddown_crt_basis_isolated.cpp`, `tests/
test_decompose_crt_basis_isolated.cpp`, `tests/
test_moddown_combine_isolated.cpp` -- all intentionally drive invalid
configurations to document the phenomenon, calling `keyswitch_core_
resident`/lower-level functions directly. Re-ran `test_ks_pq_ratio_
sweep.cpp` this round: byte-identical to entry #42's own figures.

**If a FUTURE round picks a new `SIZE_Q`/`SIZE_P`/`NUM_PART`
combination and sees the "~50% of q" signature again**: call
`gpufhe::check_hybrid_keyswitch_params(moduliQ, moduliP, numPart)`
(`include/ks_params_validate.h`) directly rather than re-deriving a
bit-length estimate by hand -- it is now the single authoritative,
exact implementation, independently cross-checked against Python.

**Not addressed, deliberately out of scope this round**:
- `control_a_relin_combine_matches_raw_tensor_num_part_1_test` and
  `keyswitch.rs`'s analogous negative case for `numPart=1` correctly
  remain `#[ignore]`d / untested -- `numPart=1` at `SIZE_P=2`
  genuinely violates `P>=Q_part_max` (`Q_part=410b` vs `P=120b`) and
  would need a much larger `SIZE_P` to fix, a security-relevant change.
- Exact equality `P==Q_part_max` was not independently tested (see
  `include/ks_params_validate.h`'s own honesty note: `P` and
  `Q_part_max` are products of disjoint-search-space primes, making
  literal equality a measure-zero coincidence `native_primes` does not
  naturally produce).
- Whether `NUM_PART=4` is the BEST (fastest) choice for any of the
  fixtures touched this round was not measured -- only that each is
  correct. Performance work can resume now that correctness and
  configuration safety are both established.
- `test_polyeval`'s own, separate multi-level noise question (entry
  #26/#29/#30, never resolved) is untouched by this round.

## `-G` build-flag bug: CLOSED for the four originally-flagged benchmarks; one remaining sweep still open

Entry #32 found `CMakeCache.txt` had cached `CMAKE_CUDA_FLAGS=-g -G`
since some earlier, unrecorded `cmake` invocation, and that CMake
always appends this ON TOP OF the build-type flags rather than
replacing them — so `-G` (CUDA device debug, disables nearly all
device-code optimization) had been silently active in every build,
Release included, for this ENTIRE project's history until fixed in
`CMakeLists.txt` (`set(CMAKE_CUDA_FLAGS "" CACHE STRING ... FORCE)`).

**All four benchmarks flagged after that discovery are now re-verified
with a genuine before/after (old binary, then rebuilt binary, same
machine) — do NOT re-run these again, read the recorded numbers
instead:**
- `bench_resident_chain`: the fix REVERSED the verdict (GPU was
  measured slower than CPU before, faster after — a 27% GPU-side
  kernel time drop).
- `bench_evalmult_resident`: the CUDA-graph-vs-direct speedup ratio
  was essentially UNCHANGED (186.7x -> 192.4x) — robust to the bug.
- `bench_batch`: also robust (~2% shift, verdict unchanged).
- `bench_hoist_multi`: qualitative verdict ("hoisting wins") survives,
  but the exact magnitude does NOT — both paths sped up
  non-proportionally once fixed, so the relative speedup ratio
  actually GREW (2.42x->3.59x) while the absolute ms-saved-per-call
  SHRANK (178.85ms->118.80ms). Do not cite the old absolute figure.

**Still open, lower priority**: a final sweep of `MASTER_STATE.md`/
`EXPERIMENT_LEDGER.md` for any OTHER standalone "ms"/"us"/"speedup"
performance claim NOT from one of these four benchmarks (e.g. anything
from `bench_evalmult.cpp`, `bench_ks_fastcore`, or narrative mentions
of "CUDA graph" speedups outside the four already covered) and mark it
provisionally suspect if it predates entry #32's commit and hasn't
been independently re-verified, so a future reader doesn't have to
re-derive which claims are settled vs. still open. Not done this round
(time budget) — the four benchmarks `NEXT_SESSION.md` explicitly named
were the priority, and that list is now complete.

## Rust evolution: FOUR operations ported, real encode/encrypt/decrypt proven, the real EvalMult round trip NOW PASSES (entry #42, see above)

`rust/` workspace (`ckks-core`, `ckks-cuda`) builds and all 20 tests
pass (4 `ckks-core` + 3 NTT + 6 rescale [4 original + 2 new device-
resident regression tests] + 6 key-switch + 3 EvalMult differential +
2 EvalMult construction/input-validation). `ckks_cuda::NttContext`
(entry #32) and `ckks_cuda::RescaleContext` (entry #35) both rebuild
their own input-independent state in Rust; `ckks_cuda::KeySwitchContext`
(entry #36) does NOT -- its setup state is a genuine eval key, so it
makes ONE opaque FFI call into the existing native key-generation
pipeline instead. `ckks_cuda::EvalMultContext` (entry #37, this round)
does neither on its own: it COMPOSES one `KeySwitchContext` (relin) and
one `RescaleContext` (the two post-relin rescale steps), via two new
`pub(crate)` device-pointer adaptation methods
(`KeySwitchContext::keyswitch_device_resident`, `RescaleContext::
rescale_device_resident`) that let the composition stay fully
device-resident instead of paying host round trips between stages. See
`research/MIGRATION_MATRIX.md` for the exact per-operation status table
— do not re-derive which operations are done from narrative text alone,
read that table. **Key switching's correctness evidence remains
explicitly weaker than rescale's/NTT's/EvalMult's own** (unchanged from
entry #36) -- EvalMult's OWN differential test, by contrast, IS a
genuinely independent from-scratch reference (schoolbook tensor/combine
math + the established keyswitch oracle + rescale's own from-scratch
RNS identity), stronger evidence than key-switching's own test, achieved
by composing already-independently-tested pieces rather than inheriting
key-switching's specific gap.

**A domain-confusion bug class is now confirmed to recur across
rounds** (entry #35's rescale benchmark, independently rediscovered in
entry #37's EvalMult test): applying an RNS-rescale "center the dropped
tower" identity to data that is NOT actually in coefficient domain
produces a systematic, 100%-of-elements mismatch, not a subtle
numerical error. **Any future test or benchmark that uses this specific
identity (`rescale`'s own, or anything built on top of it) MUST
forward-NTT its coefficient-domain inputs before feeding the real
kernel path, and inverse-NTT the kernel's eval-domain output before
applying the identity for comparison** -- do not re-learn this a third
time; `rescale::tests` and `evalmult::tests` both now contain worked
examples of the correct pattern.

**Not yet done, in priority order**:
1. **RESOLVED (entry #42)**: the relin algebraic-identity blocker is
   fixed (`NUM_PART` 2->4 in `evalmult.rs`'s `new_ctx`); all 4
   `real_encrypted_evalmult_*` tests are un-ignored and passing. The
   NEW highest-value task is item 2 below (a fifth operation), OR
   revisiting `keyswitch.rs`'s own dormant `NUM_PART=2` fixture (not
   currently broken, but one decrypt-based test away from reproducing
   this exact pattern) if a stronger key-switching correctness claim
   is ever needed there.
2. Port a FIFTH operation. The natural candidates, in rough order of
   how much of the key-switch/rescale/evalmult machinery they could
   reuse vs. how much is genuinely new: rotation (`rotate_ct_resident`
   -- an automorphism permutation, likely a SMALL new kernel, composed
   with the SAME key-switch backend this round proved reusable), or
   bootstrap's C2S/S2C/EvalMod stages (each substantially larger; do
   NOT attempt the full bootstrap pipeline in one step, per every prior
   round's own explicit warning). Inspect the real call graph first,
   same discipline as every round so far. Note: rotation ALSO uses
   `keyswitch_resident` internally (`rotate_ct_resident` ->
   keyswitch), so it may be worth checking whether rotation's OWN
   decrypt-based correctness is ALSO blocked by the same relin-identity
   gap before investing in porting it -- if so, resolving item 1 first
   is doubly justified.
3. Evaluate whether a native-CUDA-Rust-kernel toolchain (e.g. the
   `rust-cuda`/`rustc_codegen_nvvm` project) actually supports sm_75 +
   CUDA 13.2 in THIS environment before attempting Path C of the
   three-way comparison — do not assume compatibility; check the
   toolchain's own documented support matrix and attempt a trivial
   build first. If incompatible, document why and stay on Path B
   (already proven to work, and proven performance-equivalent to pure
   C++ for all FOUR ported operations so far).
4. Memory efficiency: entries #36/#37 MEASURED this (not guessed) and
   found no current benefit to cross-context table sharing in this
   crate — all four operations peak at a GPU floor dominated by fixed
   CUDA driver/context overhead (85-108MiB depending on operation, NOT
   operation data) at n=1024, so there is nothing to optimize yet.
   Revisit ONLY if/when a genuine multi-tower or bootstrap-scale Rust
   workload emerges where the per-tower table cost would actually be
   visible against that floor (the C++ engine's own `g_rtab`/`upc`
   process-lifetime caches are the pattern to port then, already proven
   at larger scale). One open, not-yet-investigated detail from entry
   #37: EvalMult's floor (108MiB) is ~23MiB higher than the other three
   operations' (85MiB) even accounting for a session-level baseline
   shift -- plausibly CUDA module-loading overhead for the two new
   kernel symbols, not independently confirmed; worth a quick check if
   convenient, not worth a dedicated round on its own.
5. `ckks-ffi` (a stable C ABI surface for the Rust layer, so existing
   C++ consumers could call into it) — not yet justified; revisit once
   there are more ported operations and a concrete consumer need.

## test_polyeval multi-level noise quantification: still the open correctness question, unchanged from entry #30

Not touched this round. See the previous section of this file
(preserved below) for the exact next experiment: instrument
`test_polyeval.cpp`'s Chebyshev loop to check whether its maxerr growth
across levels is quantitatively consistent with inherent
independent-component rescale noise (entry #29's ~769-unit single-
rescale figure) rather than a remaining defect.

## Randomized hoisting A/B: script ready, exact command to run once the memory gate clears

`research/benchmarks/randomized_ab.sh` is built and dry-run validated
(trivial dummy binaries, confirmed genuine non-alternating shuffle +
well-formed JSONL output) but has NOT been run with the real binaries
yet — blocked by the same host-memory gate as the large-n bootstrap.
**Exact next command, once `Get-CimInstance Win32_OperatingSystem`
shows >=6GB real free**:
```
cd ~/GPU-Resident-Library && ./research/benchmarks/randomized_ab.sh 10 ./test_boot ./test_boot_hoist_enabled
```
(10 trials per arm = 20 total `test_boot`-class runs, ~110s each ≈
37 minutes total — budget accordingly, and re-check the memory gate
partway through if the session is long, since host conditions have
fluctuated significantly within single sessions this whole project).
Analyze the result as two independent samples (median + bootstrap 95%
CI per arm, check for CI overlap) — see the script's own header
comment for why NOT to pair by position.

## RAII wrappers: isolated validation complete, production integration is the next concrete step (once memory allows)

`include/ks_raii.h` (`ScopedKSContext`/`ScopedKSWork`) is built and
fully validated in isolation (`tests/test_ks_raii.cpp`, 7/7 checks
pass, including 20 create/destroy cycles with zero GPU-memory drift).
**Do not re-audit or re-design from scratch.** The next concrete step,
per the directive's own "small, reviewable changes with tests at each
stage" instruction, is a CAREFUL integration pass into ONE production
call site at a time (start with `tests/test_boot.cpp`'s simplest
single-context use, e.g. the relin key's `ks_context_create(K)` at the
top of `main`, NOT the per-diagonal `stageRotDiag` hot loop first),
rebuilding and running the FULL existing regression (`test_boot`'s own
`max slot err` must stay bit-exact at `0.000547871`) after each single
call site is converted, before moving to the next. This needs real
host memory headroom (`test_boot` peaks at ~8GB RSS) — check
`Get-CimInstance Win32_OperatingSystem` first; do not attempt this
integration pass while free memory is under ~6GB, same threshold as
the large-n bootstrap gate.

## `test_polyeval` / native rescale pathway: entry #26's "library bug" conclusion REVERSED by entry #29 — do not re-chase a rescale defect

**Do not re-investigate `rescale_resident_raw`/`native_rescale_consts`/
the NTT/SwitchModulus/RescaleFuse kernels for a correctness bug —
entry #29 (this round) proved all of them correct via four independent
differential experiments** (`test_ntt_roundtrip_n1024.cpp` 56/56,
`test_rescale_scalar_diff_n1024.cpp` 10/10, `test_rescale_vector_diff_
n1024.cpp` 0/35840 mismatches, `test_rescale_double_call_n1024.cpp`
0/43008 mismatches — the last one exercising the exact production
back-to-back-scratch-buffer-reuse call pattern). `native_rescale_consts`'s
`s1`/`s2` were also hand-verified (Python) to be mathematically
equivalent to the textbook single-modulus RNS rescale formula.

`test_rescale_n1024.cpp`'s maxerr=24602.7 is explained, not mysterious:
its Delta=2^50 vs. its own qLast≈2^55 is a ~32x scale/modulus-size
mismatch (an artifact of an EARLIER round's "isolate bit-size as a
variable" experiment, which changed the tower bit-size to 55 without
re-matching Delta). `tests/test_rescale_n1024_matched_scale.cpp`
(Delta=2^55, otherwise identical) cut the error to 769.125 — almost
exactly the predicted 32x reduction — and instrumented diagnostics in
that test traced the REMAINING ~769 to inherent, expected CKKS noise:
rescaling ciphertext components c0/c1 independently (as ALL CKKS
implementations do) means `round(c0/qL)+round(c1/qL)*s !=
round((c0+c1*s)/qL)` in general, an O(1)-per-coefficient "combining"
discrepancy that `decode_host`'s n=1024-term inverse embedding sum
amplifies into a few-hundred-unit decode deviation. This is NOT a bug.

**Entry #30 (this round) spent a full round trying to find a library
defect behind test_polyeval's error and found NONE — do not re-chase
`rescale_resident_raw`, the NTT kernels, or `keyswitch_core_resident`.**
Three escalating eval-domain differential tests
(`test_keyswitch_core_native_diff.cpp`, `test_keyswitch_core_realct_
diff.cpp`) tried to directly verify keyswitch's algebraic contract
(`ba0+ba1*s =?= a*s^2`) and all showed ~50% of q (pure noise) — but
this was PROVEN to be a wrong formula, not a defect: the identical
~50%-of-q failure reproduces even applying the same check to OpenFHE's
own bit-exact-proven-correct `ba0`/`ba1` (see the appended check in
`test_keyswitch.cpp`). An external GPU-patched OpenFHE fork's
keyswitch shadow path was also found and ruled out as a confound
(`OPENFHE_GPU`/`OPENFHE_GPU_KS` env vars, now set to `0` in
`test_keyswitch.cpp`, same class of fix as `bench_evalmult_resident`'s
pre-existing one) — disabling it changed the secret key value but not
the 50% result. **If anyone wants to resolve the exact per-digit/
per-part algebraic contract**: read OpenFHE's
`src/pke/lib/keyswitch/keyswitch-hybrid.cpp` (`EvalKeySwitchPrecompute
Core`'s digit decomposition + `ApproxSwitchCRTBasis`, and
`EvalFastKeySwitchCoreExt`'s final per-digit `cji*bji`/`cji*aji`
accumulation loop around line 470) — the high-level "add ba0/ba1 into
c0/c1" convention is confirmed correct, but the per-digit CRT-weighted
scaling that makes the elementwise identity hold was not derived in
the time available. This is NOT blocking anything — `test_keyswitch.
cpp`'s existing bit-exact-vs-OpenFHE comparison remains the trusted
correctness evidence for `keyswitch_core_resident`, independent of
this unresolved formula.

**What's genuinely still open, and the single most concrete next
step**: `test_polyeval.cpp` itself uses 50-bit mid-towers matched to
its own Delta=2^50 (no Delta/qLast mismatch, confirmed), so its
original maxerr=2376.98 is *consistent with* the same inherent
independent-component rescale noise (entry #29) accumulating across
MULTIPLE chained Chebyshev-level mulCt calls — but this has not yet
been separately, quantitatively confirmed for the multi-level case.
Next experiment: instrument `test_polyeval.cpp`'s own Chebyshev loop
to track `maxerr` level-by-level (already partially visible in the
existing `[chk] Tk[k]` printouts — this round's run recorded the full
sequence: Tk[0]=9.97e-12, Tk[1]=9.04e-12, Tk[2]=6640.36, Tk[3]=910.19,
Tk[4]=13188.1, Tk[5]=1706.43, Tk[6]=13188.1, Tk[7]=2468.38, final=
2376.98 — note this OSCILLATES rather than growing monotonically,
which is itself suggestive of noise rather than a systematic bug) and
check whether the GROWTH/magnitude is quantitatively consistent with
accumulating O(1)-per-coefficient combining noise through
`decode_host`'s embedding sum at each level (a back-of-envelope
estimate, informed by this round's single-rescale ~769 figure from
entry #29, should be computable and compared against these actual
numbers). If the magnitude roughly matches, `test_polyeval`'s `[FAIL]`
is ALSO inherent CKKS noise, not a defect, and the real fix (if any is
wanted) is parameter selection (larger Delta, bigger n, or noise-aware
scale management across levels) — not a code change anywhere in the
rescale/keyswitch/NTT kernels, ALL of which are now independently
verified correct by entries #29 and #30.

`tests/test_rescale_n1024.cpp` remains committed with its original
(mismatched) parameters as a permanent artifact of an earlier
isolation experiment — its `[FAIL]` is UNDERSTOOD and EXPECTED, not a
red flag requiring a fix. Do not weaken its tolerance or delete it;
do not "fix" it by changing library code. `tests/test_polyeval.cpp`'s
`[chk]` diagnostic checkpoints (inside `mulCt`, extended this round to
cover the second mulCt call with full-array stats, not just slot 0)
are kept, additive-only — directly useful for the level-by-level
noise-growth experiment above. `tests/test_keyswitch_core_native_diff.
cpp` and `tests/test_keyswitch_core_realct_diff.cpp` are built but
deliberately NOT registered with `add_test` (their failure is expected
given the unconfirmed formula) — do not "fix" them by changing the
formula without first deriving the correct one from OpenFHE's source,
and do not re-register them with `add_test` until the formula is
actually confirmed correct.

## Read first

`research/MEMORY_FEASIBILITY.md`'s final `## UPDATE (next round): GPU-
memory lifecycle isolated with real production APIs...` section. It
does three new things beyond the prior round: (1) isolates the real
`ks_context_create`/`keyswitch_resident`/`ks_context_destroy` +
`ks_work_create`/`destroy` lifecycle using production APIs (not a
synthetic loop), finding host RSS flat across 68 diagonals (far more
robust than before) and a real, still-not-root-caused ~200MB/stage-
transition GPU-memory growth (separate from a tiny, linear, already-
understood ~0.5MB/diagonal drift); (2) DIRECTLY demonstrates (not just
infers from absent kernel logs) that external host memory, not this
process's own behavior, determines survival time — the same
deterministic internal trajectory survived 68 diagonals in one run and
only 2 in another, purely as a function of host free memory at launch;
(3) fixes one real, confirmed, but numerically tiny leak
(`ks_context_destroy` never freed `C.d_keyRow`). Then
`research/MASTER_STATE.md`'s most recent section, and
`research/EXPERIMENT_LEDGER.md` entries #22-#24.

## Immediate next action (the one concrete, executable item left open)

**Current decision: NO-GO for a 7th full n=131072 attempt** — real
Windows host free memory was only 2.49GB when this was last checked
(`Get-CimInstance Win32_OperatingSystem`), well below the ~6-7GB window
that let this workload get furthest in every attempt that made real
progress (the 6th real bootstrap attempt reached diagonal 8 at 7.43GB
free; this round's isolated Run 1 reached 68 diagonals at a favorable
moment too). **Before attempting a 7th full run, check real host free
memory first and require >=6GB** — do not launch below that threshold,
and do not retry repeatedly hoping conditions improve on their own.

**The two genuinely open technical threads, in priority order**:

1. **The ~200MB/stage-transition GPU-memory growth is real, reproduced
   twice, but not root-caused to a specific line.** Ruled out:
   `ks_work_create`/`destroy` (pairs cleanly), every per-stage-keyed
   host-side cache (`g_rtab`, `upc` — all KB-scale), and
   `evalkeygen_host_sold`'s per-part transients (identically sized
   every call, not transition-specific). Best-supported hypothesis:
   CUDA device-allocator pool retention across differently-sized
   per-stage allocations (sizeQl shrinks 30→29→...→23, never returning
   to a previously-seen size) — NOT independently confirmed, since
   `compute-sanitizer` is not installed and `nvidia-smi
   --query-compute-apps` returns no process rows in this WSL2
   CUDA-passthrough setup (both checked directly, no path forward with
   currently-available tooling). **If a future session has access to
   deeper CUDA profiling tools (a native Linux machine with
   `compute-sanitizer`/`cuda-memcheck`, or Nsight Systems) outside this
   WSL2 passthrough limitation, that would be the direct way to close
   this.** Extrapolating linearly across a full 8-stage run projects to
   only ~3-3.5GB GPU usage (well under the 6GB card), so this is a
   secondary concern, not the primary blocker.
2. **The exact external kill trigger remains only partially
   characterized**: proven NOT a Linux OOM-kill or process crash
   (checked twice via `dmesg`/`journalctl`), and now directly
   demonstrated to correlate with fluctuating real host memory rather
   than this process's own (bounded, well-characterized) behavior — but
   the PRECISE mechanism (most likely this session's own tool harness)
   cannot be confirmed with more certainty from inside this
   environment. A future session with visibility into the harness's
   own monitoring logic, or one running outside it entirely, could
   close this gap; this session's tooling cannot.

**Do NOT further lower `KROT_CAP`/`KSC_CAP` expecting a different
result** — proven (twice now, including 68 diagonals and 22+ evictions
in isolation) not the limiting factor at the caps already in use.
**`stageDiags`/`stageDiagsFwd`'s O(S²) cost is FIXED** (268s/call ->
0.11s/call, verified in production) — do not re-flag as open. **The
`d_keyRow` leak is FIXED** (tiny, ~128 bytes/call, verified via
regression) — do not re-flag as open, and do not expect it to explain
the ~200MB/transition growth (it is 3-4 orders of magnitude too small).
Full detail: `MEMORY_FEASIBILITY.md`.

Secondary open items from this round (lower priority than the above):
- A larger, truly-randomized (not block-alternated) A/B trial series to
  settle whether rotation hoisting's end-to-end effect on the full
  bootstrap is real-but-small or statistically zero — the current
  8-trial, order-balanced series could not distinguish the two (explicit
  order-effect decomposition: median=-1.826s, mean=-2.564s, stdev=6.680s
  across the 4 pairs — a direction consistent with GPU warm-up but not
  statistically crisp at this sample size; see `MASTER_STATE.md`).
- The SageMath lattice-estimator cross-check and the direct
  eprint-2021/152 / eprint-2026/1060 parameter check named in
  `SECURITY_ASSESSMENT.md`'s updated `## Disposition`.

## Two pre-existing bugs newly discovered this round (NOT caused by
## this round's changes, NOT fixed — out of scope for this directive)

A full `ctest` run (excluding `test_boot_n131072`, the known-blocked
n=131072 feasibility target) was completed for the first time all
session — every prior attempt had been OOM-killed partway through, so
these two failures had never actually been observed before now:

1. **`test_polyeval` fails deterministically**: `deg-7 poly eval, final
   tw=1 scale=2^50 maxerr=2376.98` (`[FAIL]`), reproduced identically
   twice (same exact error values both times, in and out of the full
   ctest run). `git log -- tests/test_polyeval.cpp` shows it was last
   touched in `78d7407`, long before this session's hoisting work —
   this round touched none of the C2S/polynomial-evaluation code paths.
   Not investigated further (out of scope for the current directive);
   whoever picks this up next should treat it as a real, pre-existing,
   previously-undiscovered correctness bug, not a flake.
2. **`bench_evalmult_resident` aborts**: `terminate called after
   throwing an instance of 'std::runtime_error' what(): ShadowRegistry
   cudaMalloc failed (VRAM exhausted?)`, reproduced standalone with GPU
   VRAM confirmed otherwise idle (`nvidia-smi`: 10MiB/6144MiB used
   immediately before the retry). `git log -- bench/
   bench_evalmult_resident.cpp` shows it was last touched in `6efc812`
   (CUDA-graphs-era commit, well before this session). "ShadowRegistry"
   does not appear anywhere in this repo's own source — it is coming
   from CUDA's own graph-capture machinery, suggesting a driver/CUDA-
   graph resource limit specific to this old benchmark on this GPU,
   not a host-RAM or this-round issue. Not investigated further (out of
   scope); this is a benchmark, not a correctness test.

**Everything else in the full suite passed**, including all of this
round's own correctness evidence: `test_boot` (113.71s, matching
established timing), `test_hoist_single`/`test_hoist_multi`/
`test_hoist_levelreduced`/`test_hoist_stage0`/`test_hoist_allstages`
(the last one needed to be run standalone after the first full-suite
attempt OOM'd entering it — passed cleanly alone: `[PASS] all-stage C2S
hoisting validation`, maxDiff=35 vs maxAbs=5.26e15, speedup=1.297x in
that one specific sequential-pass comparison — consistent with entry
#20's prior result, not a new number to generalize from).

## Before trusting any new benchmark number

GPU thermal state matters on this machine (Max-Q mobile part):
```bash
wsl -d Ubuntu -- bash -lc 'export PATH=/usr/bin:/bin:/usr/sbin:/sbin; /usr/lib/wsl/lib/nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,power.draw --format=csv'
```
Use `research/benchmarks/run_trials.sh <label> <N>` or `interleaved_ab.sh`
for repeated/interleaved measurement — never trust a single sequential run.

## This round's real, delivered results (commits `93f2231`, `0043af3`)

1. **KSC bounded cache**: peak RSS 10.78GB → 8.12-8.32GB (~25%), bit-exact.
2. **Eval-domain automorphism** (ranked experiment #2): replaced
   `automorphism_eval_device`'s INTT→permute→NTT with a single precomputed
   gather kernel (`automorphism_eval_fast_device`, `kernels/automorph_eval.cu`).
   Derived AND independently validated (GPU-free Python reference, 20
   trials) before writing CUDA. Caught a real use-after-free bug (async
   copy from a stack-local vector) via the standalone test
   (`tests/test_automorph_eval_fast.cpp`) before it ever reached production.
   40/40 bit-exact at real scale (n=8192). **Measured: rotate bucket down
   to 55-60s (was 62-82s depending on thermal state), total bootstrap down
   to 108-114s — the lowest this entire session.**

## Priority queue (highest value first)

1. **Correct-order rotation hoisting (ranked experiment #3) — single-
   rotation correctness now RESOLVED (commit `46998b7`). The "100%
   mismatch" recorded in earlier revisions of this file was a TEST BUG
   (bit-exact ciphertext comparison, the wrong invariant), not an algebra
   bug — see `EXPERIMENT_LEDGER.md` entry #14 for the full explanation.
   Key-generation convention attempt 2 (permute `pkA_QP`/`pkB_QP` by
   `k^-1`, keep payload = unpermuted base secret) is CORRECT. Do not
   re-derive this from scratch — it is proven, move on to the next step.**

   **What's PROVEN correct now** (`tests/test_hoist_single.cpp`, promoted
   into the default test suite, `GPUFHE_BUILD_EXPERIMENTAL_TESTS` gate
   removed from `CMakeLists.txt` since it's unused): the full pipeline —
   `ks_accumulate_qp` on the unpermuted c1, folding `c0*(P mod q)` into
   the Q-tower range of the QP-basis accumulator via `LaunchScaledAddInto`
   (`kernels/stage_acc.cu`), applying `automorphism_eval_fast_device` to
   the QP-basis accumulator (sizeQlP towers), then `ks_moddown`, using a
   key generated by permuting `pkA_QP`/`pkB_QP` by `k^-1` while keeping
   the payload as the unpermuted base secret — matches the existing,
   proven `rotate_ct_resident` **after decryption, within CKKS's own
   noise budget, across 27/27 cases** (3 independently-encrypted random
   ciphertexts x 9 rotation indices including k=M-1, k=M/2-1 edge cases).
   Decrypted-coefficient error is a stable ~2000-3300 against a
   ~1e14-1e15 signal — consistent with ordinary keyswitch rounding noise.

   **Multi-rotation decompose-sharing mechanism: DONE, real measured win
   (commit pending at time of writing — see git log for the actual hash).**
   New, separate files (do NOT touch `src/keyswitch_resident.cpp`'s logic,
   per the hard constraint): `include/keyswitch_hoist.h` +
   `src/keyswitch_hoist.cpp` provide `hoist_digits_create`/
   `hoist_digits_destroy`, `ks_decompose_only` (steps 1-3 of
   `ks_accumulate_qp`, factored apart, storing per-part `d_compl` results
   in a `HoistedDigits` struct instead of a single shared scratch buffer)
   and `ks_accumulate_from_digits` (step 4 only, against a SPECIFIC
   context's av/bv, reusing a previously-computed `HoistedDigits`).
   `tests/test_hoist_multi.cpp` proves TWO things: (1) the decompose/
   accumulate split is BIT-EXACT identical to the existing fused
   `ks_accumulate_qp`, 0/524288 mismatches (the refactor itself changes
   nothing); (2) decomposing c1 ONCE and reusing it across 6 different
   rotation indices matches the proven ground truth within noise, same
   magnitude (~2000-3300) as the single-rotation test. `bench/
   bench_hoist_multi.cpp` measured the actual performance claim required
   before trusting any speedup: interleaved A/B (baseline and hoisted
   methods alternated every rep, 30 reps + 5 warmup, within one process
   so GPU thermal drift is common-mode and cancels in the paired
   difference, same statistical approach as `bench_resident_chain.cpp`) —
   **N=5 rotation indices: baseline (5 independent full keyswitches)
   median 94.49ms vs. hoisted (1 decompose + 5 cheap accumulates) median
   46.32ms — a ~2.04x speedup, bootstrap 95% CI on the paired difference
   [47.86, 48.60]ms, entirely excluding zero.** This is a real, clean win
   at the micro-benchmark level, not noise.

   **Re-measured at a REALISTIC bucket size (N=16, matching C2S stage
   0's actual measured diagonal count "diags=16" exactly, not an
   arbitrary N=5): baseline median 374.76ms vs hoisted median 153.62ms —
   ~2.44x, bootstrap 95% CI on the paired difference [220.0, 297.0]ms,
   entirely excludes zero.** (Raw per-rep times drifted upward 320ms→
   492ms across the run from GPU thermal throttling — expected and
   harmless here specifically because baseline/hoisted are measured
   back-to-back within each rep, so the ratio stays valid even as
   absolute numbers drift; this is exactly why the paired difference,
   not either raw mean, is the number to trust.) The relative win is
   LARGER at N=16 than at N=5 (2.44x vs 2.04x), as expected since more
   rotations amortize one shared decompose further.

   **Real end-to-end validation against an ACTUAL bootstrap stage: DONE,
   PASSED.** `tests/test_hoist_stage0.cpp` (a modified copy of
   `test_boot.cpp`, identical through ModRaise and C2S stage 0's own
   trusted computation, then ALSO computes a hoisted-mechanism version
   of that SAME real stage — own fresh hoisted-convention keys, no
   cache advantage over the trusted path, which is also generating
   fresh keys here on its first use via `keyAt`): **maxDiff=613 against
   maxAbs=5.76e17 (negligible, ~1e-15 relative — [MATCH within noise]).
   Hoisted total 7.53s (decompose-once=0.073s + 16 fresh keygens=6.71s +
   compute=0.75s) vs trusted 25.89s — a real ~3.4x speedup on the
   actual first C2S stage of a live bootstrap, both sides paying for
   fresh key generation.** This is a substantially larger win than the
   N=16 synthetic micro-benchmark's ~2.44x, plausibly because the real
   stage's trusted per-diagonal compute (full `ks_accumulate_qp` via
   `keyswitch_resident`, 16 times) carries more overhead than the
   synthetic benchmark isolated.

   Debugging note worth preserving: the first attempt at this
   validation produced a dramatic, nonsensical mismatch (maxDiff twice
   maxAbs) that turned out to be a TEST bug, not an algorithm bug —
   decrypting the PRE-`rescaleAt` accumulated sum (16 ciphertext×
   plaintext products summed, at a combined scale that exceeds what
   `decrypt_host`'s `int64_t` return represents) overflows silently.
   Isolating each diagonal's raw rotation (pre-multiply, still at the
   original encryption scale) showed perfect agreement throughout,
   which is what correctly pointed at the comparison itself rather than
   the mechanism. Always rescale before decrypting an intermediate
   stage's accumulator.

   **Reproduced**: re-ran stage 0's validation a second time independently
   — [PASS] again, maxDiff=613 (bit-identical to the first run, since
   decrypted noise from a deterministic pipeline is itself deterministic),
   ~3.0x-3.4x speedup across the two runs (normal thermal/timing variance,
   not a correctness concern). Extended `tests/test_hoist_stage0.cpp` to
   attempt ALL 3 C2S stages in one run (not just stage 0) — it OOM'd
   entering stage 1 (31 diagonals vs stage 0's 16) on BOTH attempts, after
   stage 0 itself passed cleanly both times. This is consistent with
   this session's established ~10GB ceiling being tight even at n=8192
   when stacking: the trusted path's own `KROT`/`KSC` caches accumulating
   across multiple stages' worth of keys, PLUS the hoisted path's fresh,
   uncached per-diagonal key generation (by design, for a fair
   no-cache-advantage comparison) adds real additional pressure on top.
   Not chased further via that design — the single-stage result (now
   reproduced twice) is strong enough evidence for this round.

   **Follow-up: `tests/test_hoist_allstages.cpp` — NOW PASSING, all 3
   real C2S stages, after fixing two bugs found in sequence** (new file,
   sequential-pass design: run the trusted path's existing, UNMODIFIED
   C2S loop to completion first, THEN reset to the same ModRaise output
   and run a SEPARATE hoisted-only pass over all 3 stages — never
   holding both paths' keys simultaneously, avoiding the memory
   coexistence problem above). Pass 1 (trusted) completed cleanly every
   time (matches known timings). Pass 2 (hoisted) surfaced TWO distinct
   bugs, in sequence (both now fixed, see below for the resolution and
   final result):
   1. The per-stage hoisted key construction (`Kh`) and the decompose
      shape reference (`Cshape`) both passed the FULL, un-sliced `mod`
      array (30 entries) to `compute_keyswitch_constants` even after
      `hctw` shrank to 29, 28 via rescale — a genuine bug (not just a
      stage-0 coincidence masking it): the trusted path's own
      `atLevel()` always slices `mod` to the current tower count first,
      which this code failed to do. Threw `std::runtime_error("no root
      table for modulus")` entering stage 1. **Fixed** by slicing to
      `hml` (already computed per-stage) before calling
      `compute_keyswitch_constants`, matching `atLevel`'s pattern.
   2. After that fix, pass 2 instead SEGFAULTS. **RESOLVED (later
      continuation round)**: ran under `gdb --batch -ex run -ex "bt full"`
      on a fast, memory-light repro (a scratch copy with pass 1 skipped
      entirely, `g<0` instead of `g<LST`, to avoid pass 1's own ~7.5GB
      RSS competing with gdb's overhead) — crash squarely inside
      `gpufhe::ks_context_create` (existing, proven, untouched). Root
      cause: a hand-built "shape-only" `KeySwitchConstants` (correct
      `sizeQl`/`numPart`/root-tables, but default-constructed EMPTY
      `av`/`bv`, since decompose doesn't read them) — but
      `ks_context_create` unconditionally uploads every part's `av`/`bv`
      regardless of caller intent, indexing the empty vectors. Fixed by
      never hand-rolling a shape-only key for anything passed to
      `ks_context_create`: use `atLevel(KREL,hctw)` instead (the trusted
      path's own existing mechanism for exactly this — re-shape a FULL
      key down to `hctw` towers while copying its REAL, non-empty
      `av`/`bv` across). Added `tests/test_hoist_levelreduced.cpp` as a
      permanent regression test for this exact defect class (builds a
      real level-reduced context via the correct pattern, asserts no
      CUDA error) — now in the default test suite.
   3. **A second, more subtle bug surfaced immediately after fixing #2**:
      with the segfault gone, pass 2 completed all 3 stages but produced
      a real correctness MISMATCH (`maxDiff` exceeding `maxAbs` — not
      noise, wrong). Cause (found by reasoning through
      `ks_accumulate_qp`'s own `fullQ`/`delta` indexing, not another
      gdb session): `evalkeygen_host_sold` always writes `av`/`bv` using
      the FULL, fixed 32-tower `modQP` layout (real Q at indices 0-29,
      real P at 30-31) — generating the hoisted rotation key directly at
      a REDUCED `sizeQl=hctw` (as the code did) does not shrink that
      layout, it only changes which towers get the "+P*s2" payload. The
      later accumulate step locates the P-range WITHIN `av`/`bv` using
      `sizeQl` as an offset into that fixed layout — correct only when
      `hctw==sizeQ` (stage 0 exclusively, which is exactly why ONLY
      stage 0 ever passed in every earlier single-stage test). Fixed by
      using the same two-step pattern the trusted path already uses for
      every rotation key at every level (`rotFull`+`atLevel`): build the
      hoisted key ONCE at full `sizeQl=sizeQ`, then take an
      `atLevel`-style view for the current `hctw` (`atLevel` already
      sets `K.fullQ=sizeQ`, which is what makes `ks_accumulate_qp`'s
      `delta=fullQ-sizeQl` correctly skip the now-irrelevant dropped
      towers and land on the real P-range).

   **Result after both fixes: `[PASS]` across ALL 3 real C2S stages**
   (not just stage 0) — `maxDiff=35` against `maxAbs=5.26e15` (negligible
   noise, ~6.7e-15 relative). This is the strongest correctness evidence
   yet for the hoisting mechanism: proven right across every stage of
   a real bootstrap's C2S phase, not just one stage or synthetic data.

   **Timing from this specific run: 0.93x (pass 2 slightly SLOWER than
   pass 1) — do NOT read this as "hoisting doesn't help."** This
   benchmark's sequential-pass design (deliberately, for a fair
   no-cache-advantage comparison) makes the hoisted path pay full fresh
   `evalkeygen_host_sold` cost on EVERY diagonal, while the trusted
   path's `keyAt` benefits from its own `KSC` view-cache (cheap
   re-copies, not fresh keygen, for most diagonals after the first few).
   This conflates "decompose-sharing vs. not" with "no cache vs. has a
   cache" — two different variables — unlike the EARLIER, still-valid
   `bench/bench_hoist_multi.cpp` (N=5/N=16, ~2.04x/~2.44x) and
   `tests/test_hoist_stage0.cpp` (stage 0 alone, ~3.4x) results, where
   BOTH sides paid fresh keygen fairly. A clean multi-stage speedup
   measurement would need the hoisted path to ALSO use a bounded view
   cache analogous to `KSC`, not fresh-every-diagonal — not yet built.

   **DONE, in a later round (see `MASTER_STATE.md`'s `## This round`
   section and `EXPERIMENT_LEDGER.md` entry #21 — this paragraph is
   preserved for history, do not re-plan it from scratch).** Extended
   from "validated on stage 0 in isolation" to "wired into the live
   full bootstrap, both C2S and S2C," via exactly the separate
   `KROT_hoist` cache design this paragraph called for, feature-flagged
   behind `HOIST_ROTATION_ENABLED` (off by default) with per-diagonal/
   per-stage CUDA-error fallback to the trusted path. Correctness is
   proven (byte-identical flag-off behavior; byte-identical `max slot
   err` with the flag on across 4 trials). **What that later round did
   NOT achieve: a clean, statistically solid performance win** — an
   8-trial order-balanced A/B series found the effect small and within
   noise once the warm-up order confound (the original design below
   didn't anticipate it) is controlled for. `HOIST_ROTATION_ENABLED`
   remains `false` by default and `test_boot_hoist_enabled` remains
   un-promoted (not in the default ctest run) pending a larger,
   fully-randomized trial series. The paragraph below (describing the
   design BEFORE it was built) stays for history:

   needs a real architectural decision, not a quick swap — `test_boot.cpp`'s
   `rotFull`/`keyAt`/`KROT`/`KSC` build and cache TEXTBOOK-convention
   keys (permute the payload secret by `k`); the hoisted mechanism needs
   the OTHER convention (permute `pkA_QP`/`pkB_QP` by `k^-1`, payload
   unpermuted) — a structurally different key, not a drop-in replacement
   for what's cached today. Correct, safe approach: add a SEPARATE,
   parallel `KROT_hoist`/`KSC_hoist` cache (same bounded-LRU pattern)
   feeding a `stageRotHoistDiag`, wired into ALL stages (not just stage
   0), with the existing trusted path kept available for the interleaved
   comparison until the full-bootstrap numbers are in — do not simply
   replace it. `tests/test_hoist_stage0.cpp` is the proven template for
   each stage's validation; extending it to loop over every `g` instead
   of returning after `g==0` is the mechanical part of this next step.

2. **`KROT`'s own memory scaling — lever analysis DONE, safety-net
   implemented, actual eviction tuning for n=131072 NOT yet done.** A
   delegated investigation ruled out zero-copy host/device sharing with
   direct code evidence (`ks_context_create` already re-uploads+frees
   av/bv to device on EVERY call — there is no persistent device copy to
   deduplicate; also VRAM 6GB < host RAM 10GB on this machine, so it
   wouldn't help anyway), and rejected both BSGS/LST retuning (collides
   with the fixed 1775-bit security budget; flagged as the highest
   long-term-value lever, needs a dedicated session to re-derive the
   scale-management chain safely) and numPart/alpha retuning (same
   budget collision, PLUS surfaced a real pre-existing latent correctness
   bug at alpha=10, `tests/test_alpha10.cpp`, previously undocumented).
   Full analysis: `MEMORY_FEASIBILITY.md`'s "KROT lever analysis" section.
   **Implemented**: `KROT_CAP` bounded LRU cache in `tests/test_boot.cpp`
   (same pattern as the already-merged `KSC_CAP`), defaulted to 96 —
   comfortably above the measured 77-entry n=8192 working set, so this
   causes ZERO evictions and ZERO behavior change today (verified:
   `test_boot` reproduces the exact same `max slot err = 5.47871e-4` and
   peak RSS ~8.11GB as before this change). This is the SAFETY NET, not
   yet a fix — at n=131072 the cap will need to be lowered well below the
   full working set (infeasible at ~1GB/key × 77 keys), and doing so has
   a real, measured regen cost (≈0.64s/key at n=8192, extrapolated
   ≈10.9s/key at n=131072 — NOT measured at that scale) that must be
   re-measured, not assumed free, once n=131072 is actually attempted
   (item 3 below). Do not claim n=131072 is memory-feasible until that
   measurement is done with a real cap chosen for that scale.

3. **Run n=131072 end to end — ATTEMPTED THREE TIMES this round, all
   OOM'd, NOT complete. Security target independently reconfirmed; real
   per-key cost measured; a genuine, unidentified memory-growth source
   beyond KROT/KSC still blocks it.** Keep the sparse secret (h=64) — no
   `keygen.cpp` change needed, re-confirmed fresh this round:
   `python3 tools/hybrid_security.py 131072 1775 64` → 134 classical
   bits (output preserved in `SECURITY_ASSESSMENT.md`).

   **What's real from this round**: `tests/test_n131072_keysize.cpp`
   measured the ACTUAL per-key cost at n=131072 (not extrapolated):
   1085MB RSS, 7.5s keygen time per rotation key (vs. 62.9MB/0.64s at
   n=8192). `tests/test_boot_n131072.cpp` is a parameterized copy of
   `test_boot.cpp` (default n=8192 test untouched) with `n=131072`,
   `LST=4` (lgS=16 needs a divisor other than n=8192's LST=3; LST=4
   keeps the same RST=4 per-stage radix the original choice was tuned
   around — NOT independently re-validated for numerical accuracy at
   this scale).

   **What's NOT done**: three full-run attempts, with `KROT_CAP`/
   `KSC_CAP` tightened each time (96/48 → 4/3 → 2/1), ALL OOM'd before
   completing even C2S stage 0 (confirmed: zero stdout/stderr ever
   printed — not even the first stage-completion line). The first
   failure is explained (left `KSC_CAP` unbounded, same ~1GB-per-entry
   magnitude as `KROT` at this scale — see `MEMORY_FEASIBILITY.md`).
   The second and third are NOT explained: attempt 3's observed RSS
   (up to 4.5GB) exceeded its own theoretical two-cache ceiling (~3.3GB)
   substantially, meaning something ELSE grows with n that neither cap
   bounds. **Do not guess at cache constants again as the next step** —
   add per-diagonal `[mem]` instrumentation inside the C2S/S2C stage
   loops first (the existing checkpoints only print at stage
   boundaries, never reached in any of the three attempts) to actually
   localize the growth before touching `KROT_CAP`/`KSC_CAP` further.
   Full detail, including the three-attempt table: `MEMORY_FEASIBILITY.md`.

## What NOT to do

- Do not re-attempt the OLD hoisting approach (permuting decomposed
  digits directly) — falsified via direct debug test AND via literature
  confirming the correct order is different (permute the QP-basis OUTPUT,
  not the digits).
- For experiment #3's key-generation convention: permuting the payload
  secret `s2` by `k^-1` while keeping `pkA_QP`/`pkB_QP` unpermuted is
  WRONG (100% mismatch, confirmed real, not a test artifact — this one
  really does encrypt the wrong operand). Permuting `pkA_QP`/`pkB_QP` by
  `k^-1` while keeping the payload as the unpermuted base secret is
  CORRECT (see priority queue item 1 above) — do not re-litigate this,
  move on to building the multi-rotation decompose-sharing mechanism.
- Do not switch to a uniform ternary secret — corrected security analysis
  shows it doesn't help for this codebase's circuit. Keep h=64.
- Do not re-attempt experiment #1 (resident-rotation context/stream
  hoisting) expecting a different result — properly validated as a null
  result via interleaved A/B post-thermal-equilibrium.
- Do not modify `keyswitch_resident.cpp`'s actual logic for experiment #3
  — the safe seam (`ks_accumulate_qp`/`ks_moddown`) already exists and is
  verified zero-behavior-change; keep building on top of it, don't touch
  its internals.
- Do not use CKKS for messaging — `MESSAGING_ARCHITECTURE.md` stands.

## Standing environment facts

RTX 2060 Max-Q, 6144MB VRAM, CUDA 13.2, driver 595.97, WSL2 Ubuntu, 10GB
host RAM visible to WSL2, 16 CPU threads. GPU clock throttling under WSL2
and thermal drift over long sessions are both real, measured, not fixed.
