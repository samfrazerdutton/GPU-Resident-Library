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
| CKKS encode/decode (`encode_host`/`decode_host`) | done | done (`encdec_bridge.cpp`, this round) | done (`codec::encode_real`/`codec::decode`) | n/a (these ARE the real-data check; no separate RNS-level form) | **done** (entry #38: `encode_decode_round_trip_matches_input`, 3 input cases incl. zero/negative/fractional, error ~1e-14, far below a 1e-9 tolerance) | **done (entry #44, Phase 11)** -- `bench/bench_evalmult_e2e_breakdown.cpp` found this O(n²) pure-host pair is the LARGEST single component of real end-to-end EvalMult latency (59%+29%=~88% of total at n=8192, vs ~1% for the entire GPU-resident tensor+keyswitch+combine+rescale chain -- do not assume the GPU kernels dominate). Parallelized across hardware threads (`std::thread`, zero change to the per-element arithmetic, bit-identical output verified via `tests/test_encode_decode_parallel_matches_sequential.cpp` against an independent sequential oracle): measured ~5.7-6.4x (encode) / ~6.1-6.8x (decode) speedup, interleaved A/B, n=8192. STILL the dominant cost post-optimization (~54%+23%=77% of total) -- the clear next experiment is an O(n log n) FFT-based replacement, not yet attempted (see `NEXT_SESSION.md`) | no |
| CKKS encrypt/decrypt, single ciphertext (`encrypt_host`/`decrypt_host`) | done | done (`encdec_bridge.cpp`, this round) | done (`codec::encrypt`/`codec::decrypt`, plus `KeySwitchContext::secret_key`/`public_key_a`/`public_key_b`, TEST-ONLY accessors) | n/a (same reasoning as encode/decode) | **done** (entry #38: `encrypt_decrypt_round_trip_matches_input`, 3 input cases, error ~1e-12, consistent with the expected Gaussian-noise budget, well below a 1e-6 tolerance) | **done (entry #44, Phase 11)** -- `encrypt_host` found to issue 4*sizeQ (32 at sizeQ=8) separate, unbatched per-tower-per-buffer GPU round trips, each also rebuilding its own root table from scratch; `decrypt_host` left unchanged (already cheap, ~0.5-0.6ms at n=8192, not a bottleneck). Batched into one upload+one `LaunchNTT_CT_Batched`+one download per buffer with cached `(n,q)` root tables (same pattern `src/evalkeygen.cpp` already used), bit-identical output verified via `tests/test_encrypt_host_fast_matches_sequential.cpp` against an independent unbatched oracle: measured ~2.2-2.8x speedup, interleaved A/B, n=8192; real-round-trip error magnitude identical before/after (bit-exact same ciphertext) | no |
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
