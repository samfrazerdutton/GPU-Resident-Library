# Rust/CUDA migration matrix

Tracks, per CKKS primitive operation, exactly how far the "CKKS RUST +
CUDA EVOLUTION" effort (`rust/` workspace, Path B: Rust host +
EXISTING, unmodified CUDA C++ kernels via FFI) has gotten. Updated each
round a new operation is touched. See `EXPERIMENT_LEDGER.md` entries
#32 (NTT), #35 (rescale), #36 (key switching), #37 (EvalMult), and #38
(real encode/encrypt/decrypt, and the BLOCKED real-encrypted-EvalMult
round trip) for full narrative detail behind each row.

Columns, left to right, are cumulative prerequisites (a later column
implies the earlier ones hold):

- **C++-only**: the operation exists in the main `gpufhe` C++/CUDA
  engine (`src/`, `kernels/`) and nowhere else. This is the default/
  starting state for everything in the library.
- **FFI-exposed**: a narrow `cpp_bridge/*.cpp` file forwards an
  existing C++ function through `extern "C"` so Rust can call it
  without reimplementing its arithmetic.
- **Rust-abstracted**: a Rust struct/function in `ckks-cuda` wraps the
  FFI calls + existing CUDA kernels behind a safe API, following the
  setup/hot-path split (`NttContext`/`RescaleContext` pattern).
- **Independently tested**: the Rust wrapper is checked against a
  reference computed FRESH in Rust (not imported from the C++ test),
  not merely "two implementations agree." This column is about ISOLATED
  RNS-level arithmetic correctness ONLY -- it says nothing about whether
  a REAL ciphertext (real plaintext, real encryption noise, real scale)
  decrypts correctly through the operation. See the new **Real
  encrypted round trip** column for that, added this round after
  finding the two are NOT interchangeable (entry #38).
- **Benchmarked fairly**: a same-setup-scope 4-tier (cold-start /
  context-init / steady-state / end-to-end) comparison exists in BOTH
  languages, with results reported as medians across multiple trials,
  run in both orders to rule out a GPU-state confound.
- **Real encrypted round trip** (new this round): encode -> encrypt ->
  [this operation] -> decrypt -> decode, compared against the
  mathematically expected result within a derived (not fitted)
  tolerance, using REAL ciphertext data and REAL encryption noise --
  not synthetic RNS residues. Strictly stronger evidence than
  "independently tested" alone; an operation can pass that column and
  still fail this one (confirmed: EvalMult does, see below).
- **Production-integrated**: something in the actual engine or a real
  application calls the Rust path instead of (or alongside) the C++
  path. **Nothing in this table has reached this column yet.**

| Operation | C++-only | FFI-exposed | Rust-abstracted | Independently tested (RNS-level) | Real encrypted round trip | Benchmarked fairly | Production-integrated |
|---|---|---|---|---|---|---|---|
| Primitive-root search (`native_root`) | done | done (`native_root_bridge.cpp`) | done (`ntt::native_root`) | done (order-exactly-2n check) | n/a (not a ciphertext operation) | n/a (sub-microsecond host call, folded into NTT's cold-start tier) | no |
| Coefficient-domain NTT/INTT round trip | done | n/a (kernel called directly, no bridge needed) | done (`NttContext`) | done (56/56 exact cases, `test_ntt_roundtrip_n1024.cpp`-equivalent, both one-shot and reused-context) | n/a (not a ciphertext operation) | **done** (entry #32: C++ 119.6us vs Rust 108.7us median, equivalent within noise) | no |
| Fused-rescale constants (`native_rescale_consts`) | done | done (`rescale_consts_bridge.cpp`, this round) | done (`rescale::native_rescale_consts`) | done (folded into rescale's own differential test) | n/a (not a ciphertext operation) | n/a (sub-microsecond host call, folded into rescale's cold-start tier) | no |
| Single-modulus RNS rescale (`rescale_resident_raw`'s algorithm) | done | n/a (kernels called directly; constants via the bridge above) | done (`RescaleContext`, this round) | **done** (entry #35: 4 new tests, 0 mismatches vs. a fresh independent RNS reference, including a real domain-mismatch bug CAUGHT in the first draft of the C++ comparison and fixed before any timing was trusted) | not attempted standalone (exercised only as part of EvalMult's blocked round trip below) | **done** (entry #35: C++ 535-573us vs Rust 534-542us steady-state median, equivalent within noise, reproduced in both run orders) | no |
| Full Hybrid key switching (`keyswitch_resident`, `DeviceKSContext`/`DeviceKSWork`) | done | done (`keyswitch_bridge.cpp`, this round -- a single fixture-lifecycle bridge, not one function per primitive, since the setup state is a genuine eval key, not bookkeeping) | done (`KeySwitchContext`, this round) | **done** (entry #36: 6 tests bit-exact vs. `keyswitch_core_resident`; entry #42 (Phase 9) additionally proved the decompose and accumulate stages EXACT, zero-tolerance, via independent CPU references across the full `sizeQl x sizeP x numPart` matrix -- see `tests/test_keyswitch_stage_trace.cpp`, `tests/test_moddown_crt_basis_isolated.cpp`, `tests/test_decompose_crt_basis_isolated.cpp`, `tests/test_moddown_combine_isolated.cpp`) | **done, and now construction-time ENFORCED** (entry #43, Phase 10: the relin-identity invariant `ba0+ba1*s~=a*s_old` holds to small tolerance PROVIDED the auxiliary modulus `P`'s exact product is at least as large as the largest key-switch digit's own exact modulus product `Q_part_max` (entry #42's finding, now implemented as code -- `include/ks_params_validate.h`'s `check_hybrid_keyswitch_params`, true arbitrary-precision arithmetic, cross-checked against Python's native bigints). This crate's OWN dedicated `keyswitch.rs` test fixture used `SIZE_Q=8,SIZE_P=2,NUM_PART=2` (`Q_part_max>P`, violating the requirement) for every round since entry #36 -- FIXED this round (`NUM_PART` 2->4), with a NEW decrypt-based test (`real_decrypt_based_relin_round_trip`: real ciphertext -> real `keyswitch` -> rescale -> decrypt -> decode -> compare to the clear-text product) passing at max abs error `2.23e-11`. The old, invalid configuration is now explicitly REJECTED at `KeySwitchContext::new` construction time (`invalid_num_part_at_old_value_is_now_rejected`), not merely avoided by convention -- enforced centrally at `ks_context_create` (`src/keyswitch_resident.cpp`), so every construction path (Rust FFI, C++ benchmarks/examples) gets this for free) | **done** (entry #36: fixture creation ~6.5-6.8ms, steady-state ~1.06-1.09ms, end-to-end ~8.4-9.2ms median, both languages, equivalent within noise, reproduced in both run orders) | no |
| EvalMult (tensor product + relin + combine + rescale x2) | done | n/a (kernels called directly; relin via the key-switch bridge, rescale via the already-exposed rescale kernels) | done (`EvalMultContext`, this round -- composes `KeySwitchContext` + `RescaleContext` via two new device-pointer adaptation methods, `keyswitch_device_resident`/`rescale_device_resident`) | **done, genuinely independent** (entry #37: 3 new tests, 0 mismatches vs. a FRESH from-scratch reference -- schoolbook tensor/combine arithmetic + the established keyswitch host oracle + rescale's own from-scratch RNS identity, correctly domain-converted around the rescale step; a real domain-confusion bug was caught and fixed in THIS test before it was trusted, see the entry) | **DONE -- PASSES end-to-end for the first time (entry #42, Phase 9)**. Entries #40/#41's "evalkeygen_host defect" / "sizeP=2-specific reconstruction defect" diagnoses are BOTH superseded: every constant family and kernel in `ks_constants.cpp`/`keyswitch.cpp`/`basis_convert.cu`/`moddown.cu` is independently proven correct at every `sizeP` in {1,2,3} (entry #42's Stage B/C/D isolated tests). The real cause was `evalmult.rs`'s test fixture choosing `NUM_PART=2` (alpha=4), giving each key-switch digit a modulus product (`Q_part~210 bits`) LARGER than the auxiliary modulus `P` (2x60-bit=~120 bits) -- violating the Hybrid key-switch's standard `P>=Q_part_max` correctness requirement (confirmed decisively: holding `sizeP=2` fixed and sweeping only `numPart`, the relin-identity residual flips from ~50% of q to ~6e-12% of q exactly when `P-Q_part_max` crosses from negative to positive -- `tests/test_ks_pq_ratio_sweep.cpp`). **Fix**: raised `NUM_PART` from 2 to 4 in `new_ctx` (alpha 4->2, `Q_part_max` ~210->~110 bits, +10-bit margin over `P`) -- `SIZE_Q`/`SIZE_P`/`n`/`h` all unchanged, zero security-parameter impact. `real_encrypted_evalmult_matches_expected_product` now passes: max abs error `2.5896e-11` (tolerance `1e-6`); all 3 siblings pass too. Full Rust suite: 34 passed, 0 failed, 1 ignored (`control_a_relin_combine_matches_raw_tensor_num_part_1_test`, which forces `numPart=1` -- `Q_part=410b` vs `P=120b`, a deficit `NUM_PART=4` does not fix; correctly documented and kept ignored). **Phase 10 (entry #43) found this fix was incomplete**: a repository-wide audit turned up FOUR more fixtures still using the invalid `NUM_PART=2`/`numPart=2` combo that entry #42's audit missed -- `examples/bench_evalmult.rs`, `examples/bench_keyswitch.rs`, and `bench/bench_evalmult_cpp_vs_rust.cpp`, `bench/bench_keyswitch_cpp_vs_rust.cpp` (the latter two registered as auto-run `ctest`s, so they would have broken the suite outright once construction-time validation landed). All four fixed to `NUM_PART=4`. The requirement is now ALSO enforced in code (`include/ks_params_validate.h`, exact arbitrary-precision arithmetic, not a bit-length approximation) at `ks_context_create`, so any future fixture using an invalid combination is rejected at construction time with a descriptive error rather than silently producing cryptographically meaningless output | **done** (entry #37: fixture creation ~10-20ms, steady-state ~2.4-5.1ms, end-to-end ~14-23ms median across 3 repeated A/B runs in alternating order; wider run-to-run variance than the simpler single-kernel-chain operations, overlapping ranges, no consistent winner either language) | no |
| CKKS encode/decode (`encode_host_fft`/`decode_host_fft`, production; `encode_host`/`decode_host` retained as tested reference/fallback) | done | done (`encdec_bridge.cpp`, re-pointed to the FFT functions this round) | done (`codec::encode_real`/`codec::decode`, unchanged -- the swap is entirely inside the C++ bridge) | n/a (these ARE the real-data check; no separate RNS-level form) | **done** (entry #38's `encode_decode_round_trip_matches_input` unaffected by the swap; entry #45, Phase 12 ADDS `tests/test_encode_fft_matches_reference.cpp` -- the new FFT vs. the untouched O(n²) implementation as an independent oracle, across n in {16,...,8192} x 9 input categories, principled tolerance `(N+1000)*eps*Delta`, all pass) | **done (entry #44 Phase 11, superseded by entry #45 Phase 12)** -- Phase 11 found this O(n²) pure-host pair was the LARGEST single component of real end-to-end EvalMult latency and parallelized it (~5.7-6.8x). **Phase 12 replaced the algorithm itself** with a derived, independently-verified (numpy + mpmath + the untouched O(n²) oracle) O(n log n) FFT (`src/encode_fft.cpp`, see `research/FFT_ENCODING_DERIVATION.md`): measured ~31-61x (encode) / ~30-81x (decode) speedup over the Phase-11 parallelized baseline, interleaved A/B. No longer the dominant cost: at n=8192 encode+decode dropped from ~77% to ~1.5% of total pipeline latency; `encrypt_host` is now the single largest stage (see its own row) | no |
| CKKS encrypt/decrypt, single ciphertext (`encrypt_host_shoup_pooled_barrett_batched`, production; `encrypt_host_shoup_pooled_barrett`/`encrypt_host_shoup_pooled`/`encrypt_host`/`decrypt_host` retained as tested reference/fallback; `encrypt_host_shoup_pooled_barrett_batched_pinned` AND `encrypt_host_shoup_pooled_barrett_prewarmed`/`encrypt_prewarm_pinned_buffer` built, verified, measured, and deliberately NOT promoted; `encrypt_host_shoup_pooled_barrett_batched_registered`/`encrypt_prewarm_registered_buffer` (Phase 20, CUDA-registered host memory) ALSO built, verified, measured, and deliberately NOT promoted -- all three retained as opt-in experimental features) | done | done (`encdec_bridge.cpp`, re-pointed to `encrypt_host_shoup_pooled_barrett_batched` this round) | done (`codec::encrypt`/`codec::decrypt`, unchanged -- the swap is entirely inside the C++ bridge) | n/a (same reasoning as encode/decode) | **done** (entry #38's and entry #47 Phase 14's tests unaffected by the swap; entry #48, Phase 15 ADDS `tests/test_encrypt_host_shoup_pooled_barrett_batched_matches_reference.cpp` (bit-exact vs. both `encrypt_host` and Phase 14, incl. a packed-buffer-boundary stress case) and `tests/test_encrypt_host_shoup_pooled_barrett_batched_pinned_matches_reference.cpp` (bit-exact vs. both `encrypt_host` and Experiment 1) -- plus TWO new concurrency audits (one per new pooled resource), 4 scenarios each, 5 runs x 0/396 mismatches each; entry #49, Phase 16 ADDS `tests/test_encrypt_host_prewarm_matches_reference.cpp` (bit-exact across BOTH the fallback and prewarmed code paths, incl. idempotency) and `tests/test_encrypt_host_prewarm_concurrency_audit.cpp` (6 NEW lifecycle scenarios incl. prewarm/encrypt race interleaving, 5 runs x 0/372 mismatches, 0 prewarm-consistency failures) -- re-validating the mutex design under direct stress for each NEW pooled buffer rather than assuming it generalizes from any earlier phase's audit) | **done (entry #47 Phase 14, superseded by entry #48 Phase 15)** -- Phase 14 optimized `embed`. **Phase 15 traced the four separate per-buffer H2D/D2H transfers** (`d_v/d_e0/d_e1/d_m`, confirmed identical size/lifetime/pattern) **and merged 8 transfer calls into 2** over one packed pooled buffer (`src/encrypt_host_shoup_pooled_barrett_batched.cpp`), kernel launches unchanged -- measured 1.57x-2.78x faster than the ORIGINAL Phase-11 `encrypt_host`, 1.07x-1.17x faster than Phase 14 alone (interleaved 4-way, run twice independently). **A second experiment (pinned host staging memory, `src/encrypt_host_shoup_pooled_barrett_batched_pinned.cpp`) was built, bit-exact verified, concurrency-audited, and measured a genuine further 1.06x-1.17x steady-state speedup in isolation -- but was found, via the full-pipeline benchmark's cold single-call measurement, to be a NET LOSS for low-repetition usage** (`cudaMallocHost` costs ~3-8ms per distinct `(n,sizeQ)` key, more than `encrypt_host`'s entire typical latency) and was correctly NOT integrated into production, per the governing directive's own "if pinned memory introduces more overhead than it saves, revert it" instruction. **Phase 16 built an explicit PREWARMING API (`encrypt_prewarm_pinned_buffer`/`encrypt_pinned_buffer_is_ready`/`encrypt_host_shoup_pooled_barrett_prewarmed`) to decouple that cost from the first real encryption call, fully correct and concurrency-safe (0/372 mismatches, 5 runs, 6 scenarios) -- but found the allocation cost ITSELF is unpredictable in this environment (16us-5.9ms depending on n, reproducibly, not simply size-dependent), making only the long-lived-service workload profile an unconditional win. RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE, NOT wired into the Rust API or FFI bridge. Phase 17 (pure diagnostic, NO code changed) then DEMONSTRATED the root mechanism: the SAME variance reproduces via a pure Linux `mmap`+`madvise`+`mlock` test with ZERO CUDA code -- the Linux kernel's own memory-locking path, not a CUDA allocator quirk (HIGH confidence); host/WSL2 memory pressure as a contributing amplifier remains plausible but unisolated (MEDIUM confidence); a WSL2-Hyper-V-ballooning mechanism DISTINCT from ordinary kernel pressure could NOT be confirmed with available tooling, downgrading Phase 16's own hedged guess to an explicitly open question. Decision unchanged: category 3 (externally controlled, now demonstrated rather than merely suspected). **Phase 18 (controlled THP experiment, NO code changed) found the system-wide THP toggle (`/sys/kernel/mm/transparent_hugepage/enabled`) BLOCKED BY PERMISSIONS (no passwordless sudo; per the user's explicit choice, documented rather than worked around). Using the one THP variable controllable without root (per-call `madvise(MADV_HUGEPAGE/NOHUGEPAGE)` hints), found a real, demonstrated, but narrow effect: a consistent ~5x `mlock()` slowdown specific to EXACTLY 2MiB allocations with the hint disabled (absent at 256KB/512KB/1MB) -- too small to explain the dominant variance. Critically, a same-session INTERLEAVED re-test REVISES Phase 17's HIGH-confidence "not CUDA-specific" finding: pure Linux `mlock()` (174-1613us) and `cudaMallocHost` (2984-5079us, same size, same short window) are now shown to diverge by an order of magnitude, promoting CUDA/WSL2-driver-passthrough overhead to the leading hypothesis (MEDIUM-HIGH confidence) and downgrading the pure-kernel-`mlock()` explanation to MEDIUM. Decision still unchanged: category 3, opt-in design stands, now on a partially revised rationale; no production or config change made.** **Phase 19 (investigation phase, NO code changed) mechanistically PROVED the dominant mechanism using `nsys` (NVIDIA Nsight Systems, usable without root): 87-97% of every `cudaHostAlloc`/`cudaMallocHost` call's wall-clock time is spent inside `ioctl()` syscalls into the NVIDIA driver, on the calling thread, every call -- not a one-time cost, confirmed by querying the tracer's own exported trace directly rather than inferred from comparative timing. `cudaMallocHost` and `cudaHostAlloc(cudaHostAllocDefault)` are confirmed the SAME operation (statistically indistinguishable). This retroactively explains Phase 17/18's `mlock()` confusion: `cudaPointerGetAttributes` confirms plain `mlock()`'d memory is never recognized by CUDA as pinned, so it was never a valid comparison. The practically useful new finding: `cudaHostRegister` on pre-touched memory is 1.5x-4x CHEAPER than `cudaMallocHost`/`cudaHostAlloc` for the full lifecycle (reproduced across 4 independent fresh-process runs; `nsys`-confirmed ~3x cheaper per call, same `ioctl`-bound mechanism, fewer/faster calls) -- a credible, well-evidenced alternative, but NOT YET implemented or correctness-verified inside the real pooled/prewarmed production paths. A separate new finding: plain pageable `malloc`+`memset` (zero CUDA) also jumps sharply at exactly 2MiB, generalizing Phase 18's THP-hint effect beyond pinning entirely. Decision: still no production code change -- the recommended next step is to prototype `cudaHostRegister` inside `encrypt_host_shoup_pooled_barrett_batched_pinned`/the prewarming API, re-verify correctness/concurrency from scratch, and re-measure the full pipeline before deciding whether to ship it.** **Phase 20 did exactly that: built `encrypt_host_shoup_pooled_barrett_batched_registered` (mirrors the pinned variant exactly except the host staging buffer's allocation mechanism -- `RegisteredHostBuffer`, a new move-only RAII class, instead of raw `cudaMallocHost`; registers ONCE per key; self-imposed 16MiB cumulative budget with a safe fallback to the unmodified production pageable path if exceeded or if registration fails), bit-exact verified against `encrypt_host`/batched/pinned (0 mismatches) and against the REAL encrypted encode/encrypt/EvalMult/decrypt/decode pipeline (identical max error across all three implementations), and concurrency-audited (6 scenarios, 436 checks, 0 mismatches, including prewarm-racing-with-encryption and fallback-under-genuine-budget-pressure). Measured: cold-start ~2.1-2.4x CHEAPER than the pinned variant's own cold-start (directly resolving the exact weakness that blocked pinned's promotion), and a real, reproducible ~7-14% steady-state win over production across 3 independent fresh-process runs, confirmed via `nsys` in the real production-shaped code (~4.8x cheaper per `cudaHostRegister` call vs. `cudaHostAlloc`, matching Phase 19's isolated finding). One open anomaly (explicitly-prewarmed was consistently slightly slower than lazily-created) and zero Rust/FFI-layer validation remain. Decision: Category B -- RETAINED AS AN OPT-IN EXPERIMENTAL FEATURE, NOT promoted to production, NOT wired into the Rust API this round.** **Phase 22 closed both open items. Rust/FFI integration added (`codec::encrypt_registered`/`prewarm_registered_buffer`/`registered_buffer_is_ready`, backed by 3 new `extern "C"` functions and the 1 missing `build.rs` compile-list entry that was the sole reason this path was unreachable from Rust before) -- Approach B (a distinct sibling function, never a hidden flag inside the existing bridge), `codec::encrypt`'s own behavior completely unchanged. 8 new Rust tests, 0 mismatches, including the real encrypted encode/encrypt_registered/EvalMult/decrypt/decode round trip through the ACTUAL Rust API (not RNS-only) -- PASSES; a Rust-layer concurrency audit (84 operations) complements the re-confirmed C++ one (436 checks). The prewarming anomaly is RESOLVED, not left open: Phase 20's own benchmark compared DIFFERENT `(n,sizeQ)` keys between its lazy/prewarmed configs (a confound); a controlled re-test at the IDENTICAL key found no real difference, independently confirmed by a structural code argument (the pool cannot distinguish how a key became ready). Performance reconfirmed through Rust: `encrypt_registered` at or below `encrypt` in 3/3 runs (ratios 0.873-0.970), the win surviving the FFI boundary. A real, separate, narrow bug was found and documented (not fixed, out of scope): `gpufhe_ks_fixture_create` crashes with `std::bad_alloc` at `size_q=5, size_p=2, num_part=4` specifically. Decision still Category B -- STILL RETAINED AS OPT-IN, NOT promoted -- every safety/correctness criterion is now satisfied; the only remaining question is whether the modest win justifies the added complexity for the default path, a judgment call, not a safety one.** Full real-encrypted pipeline (production path): 1.14x-1.33x faster than OLD, 1.04x-1.05x faster than Phase 14, on top of Phase 12's gains. `decrypt_host` left unchanged (already cheap, not a bottleneck). See `research/TRANSFER_PROFILING.md` (Phase 15, 16, 17, 18, 19, 20, and 22 sections) | no |
| Bootstrap pipeline (C2S/S2C/EvalMod) | done | not started | not started | not started | not started | not started | no |

**Reminder for future rounds**: do not infer from this table that the
whole engine is "being ported" or that any security property has been
re-certified for the Rust path -- four operations out of a large engine
are abstracted and benchmarked, nothing is production-integrated, and
each new row requires its own independent correctness reference, not
an assumption that the established pattern transfers automatically.
**"Independently tested" (RNS-level) and "real encrypted round trip"
are NOT the same claim and must never be conflated** (entry #38's
original lesson) -- a RNS-level check, by design, never exercises the
actual decryption invariant a real ciphertext depends on. EvalMult
now passes BOTH columns (entry #42, Phase 9) but only after fixing a
PARAMETER choice (`NUM_PART`), not the algorithm -- the lesson
generalizes: **a passing RNS-level/self-consistency check does not by
itself prove `P>=Q_part_max` holds for whatever `SIZE_Q`/`SIZE_P`/
`NUM_PART` combination a given fixture uses.** Key switching's own
dedicated test fixture (`keyswitch.rs`) still uses a combination that
VIOLATES this (see its row above) and would reproduce the same
~50%-of-q failure if a decrypt-based check were ever added there --
do not assume it is safe to copy that fixture's constants elsewhere
without first checking `Q_part_max <= P`.

## Memory: what this round measured (entry #36)

Peak GPU device memory was measured (sampled every 50ms via
`nvidia-smi` across an extended run of each operation's own Rust
benchmark) for all three Rust-abstracted operations at their respective
n=1024-scale configurations: NTT, rescale, and key switching ALL peak
at **85 MiB** -- identical across all three, with no growth across
500,000 (NTT/rescale) or 200,000 (key switching) repeated steady-state
calls on one reused context (confirmed leak-free, not merely
"eventually stable"). This 85 MiB is the fixed CUDA runtime/driver
context floor for this GPU/driver combination, not operation-specific
data: analytically, NTT's own tables are ~40KB, rescale's ~200KB, and
even the FULL key-switch eval key + device context + scratch is on the
order of a few hundred KB at this n/tower count -- all negligible next
to the fixed floor. **Conclusion, per the governing directive's own
"measure before changing the architecture" instruction: there is
currently no measurable benefit to sharing root tables or constants
across Rust contexts at this scale** -- the C++ engine's OWN
process-lifetime static caches for key-switching's root tables and
per-part constants (`g_rtab`/`upc` in `src/keyswitch_resident.cpp`,
confirmed via source reading this round) already solve this problem at
the ONE place it has ever mattered in this codebase's history (dozens
of `DeviceKSContext`s rebuilt across many rotations in the bootstrap
hot loop) -- and Rust's `KeySwitchContext` inherits that sharing for
free simply by calling into the existing `ks_context_create`, without
needing its own cache. Introducing a similar shared, mutable cache
PURELY WITHIN the Rust crate (across separate `NttContext`/
`RescaleContext` instances at the same modulus) is deliberately NOT
done this round: no real multi-context-same-modulus Rust workload
exists yet to justify the added complexity (thread-safety, invalidation,
shared-ownership risk) against a savings of tens of KB. Revisit this
decision if/when a genuine multi-tower or bootstrap-scale Rust workload
emerges -- at n=131072 the per-tower table sizes (and tower counts)
could make the trade-off real, unlike at n=1024.

## Memory: EvalMult's own measurement (entry #37)

Same methodology as entry #36 (50ms-sampled `nvidia-smi` across an
extended run, this time 50,000 repeated `multiply_relin_rescale` calls
on one reused `EvalMultContext`). Peak: **108 MiB**, with zero growth
across the full run (confirmed leak-free). The baseline floor observed
THIS round (idle, no process attached) was 22 MiB, not the 0 MiB entry
#36 recorded -- a session-level WSL2/driver-reporting difference, not a
regression attributable to any specific process (confirmed via
`nvidia-smi --query-compute-apps`: no process was listed as holding
it). Measuring EvalMult's delta against ITS OWN session's floor (108 -
22 = 86 MiB) matches keyswitch's historical delta (85 - 0 = 85 MiB)
closely, consistent with entry #36's finding that this gap is fixed
CUDA context/driver overhead, not operation data -- `EvalMultContext`'s
OWN new allocations (11 buffers of `size_ql * n * 8` bytes each ~64KB,
plus one embedded `RescaleContext`'s ~200KB, both already accounted for
analytically in entry #36) remain on the order of 1MB, nowhere near
explaining an 86MiB floor by themselves. The most likely remaining
explanation -- not independently confirmed this round, flagged
honestly rather than asserted -- is CUDA driver/module-loading overhead
for the two NEW kernel symbols this operation introduces
(`tensor_kernel`/`combine_kernel`), loaded into the context for the
first time. **Conclusion, same as entry #36: no cross-context
table-sharing work is justified by this measurement** -- EvalMult's
own new buffers are trivial next to the fixed floor, and the floor
itself is not attributable to anything `EvalMultContext` allocates.

## Memory and benchmarks: deferred this round for the real-encryption primitives (entry #38)

`codec::encode_real`/`decode`/`encrypt`/`decrypt` are PURE HOST
functions (confirmed by reading `src/encode.cpp`/`src/encdec.cpp`: no
device allocation, no kernel launch retained across calls -- `encrypt`/
`decrypt` each do one NTT/INTT internally, but that's a single kernel
launch on a scratch buffer freed before the function returns, not a
persistent GPU context). There is therefore no GPU-memory floor to
measure for them specifically (any GPU footprint observed is entirely
from whatever `KeySwitchContext`/`EvalMultContext` is already alive in
the same process) and no "setup vs. steady-state" GPU-context
distinction for the fair 4-tier benchmark methodology to apply to.
Deliberately NOT building a benchmark for them this round -- consistent
with the governing directive's own "benchmark only after correctness
passes" ordering, and correctness of the ONE operation that would make
a `codec`-based benchmark meaningful (the full encrypted EvalMult round
trip) is exactly what entry #38 found BLOCKED. Revisit once that is
resolved, not before.
