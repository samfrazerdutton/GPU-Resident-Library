# Next session — exact continuation

## Current commit

Check `git log -8 --oneline` on resume. As of this writing HEAD should be
`275c598` (on top of `f155710`, `80545f8`, and Phase 39's `05ada33`,
verified matching `origin/main` exactly after push). Hardware re-confirmed unchanged:
RTX 2060 Max-Q, 6144 MiB, CUDA 13.2, cmake 3.28.3, Rust/cargo 1.94.0,
16 logical cores, Ubuntu 24.04.4. Windows free RAM fluctuated roughly
1.2-7.1 GiB through this session (other desktop applications, not this
project's own work, were the main driver).

**CRITICAL environment finding, read before running any `cmake --build`
or full reconfigure**: this session's build stalled for 19+ minutes due
to WSL's default Windows-PATH import (54 `/mnt/c/...` entries) combined
with a full CMake reconfigure -- each toolchain-detection `find_program`
lookup had to make a slow 9P round-trip for every Windows PATH entry.
Root-caused via the kernel's own `wchan=p9_client_rpc`; fixed by
exporting the project's own already-established clean, hardcoded `PATH`
(`/usr/local/cuda-13.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`)
before the build command -- not just before ad hoc `nvcc`/`g++` calls,
where this project already did so. Turned the same operation from
19+ minutes (still climbing) into a consistent ~3.2-3.4 seconds. Full
diagnosis: `research/PHASE40_WSL_PERFORMANCE_DIAGNOSIS.md`. **Always
export that PATH before any toolchain-invoking command in this
environment, every session, not just when something seems slow.**

**Full detail for this phase**: `research/PHASE40_REGRESSION_BASELINE_RECONCILIATION.md`,
`research/PHASE40_SHOUP_RNS_VALIDATION.md`,
`research/PHASE40_PERFORMANCE_AND_NEXT_BOTTLENECK.md`,
`research/PHASE40_WSL_PERFORMANCE_DIAGNOSIS.md`.

**Phase 38/39's own "130/134 passing" headline sentences were
internally inconsistent** (130 was the count of tests RUN in the 4
batches, not the count that PASSED -- 120 passed, 10 failed, within
that batch of 130; 134 is a THIRD number, the full discovery count
including 4 large-n tests handled separately). Not a data error --
the underlying batch tables were always arithmetically self-consistent
-- just a mislabeled summary sentence, now corrected (original
phrasing left intact at its source, not erased) in
`research/PHASE40_REGRESSION_BASELINE_RECONCILIATION.md`. **If writing
a future phase's own regression summary sentence, state the PASSED
count as the numerator against the DISCOVERED count as the
denominator, with run/skipped/resource-exhausted counts separate --
not a "tests run" count dressed up as "passing."**

**One new kernel-launch-batching optimization shipped this phase**
(Track E): `src/keyswitch_resident.cpp`'s `ks_moddown` had one
remaining serial per-tower loop (`LaunchModDownCombine` called once
per Q-tower) sitting right next to two ALREADY-batched steps
(`LaunchINTT_GS_Batched`/`LaunchNTT_CT_Batched`) in the same function --
an oversight, not a new idea. Added `LaunchModDownCombineBatched`
(`kernels/moddown.cu`/`include/moddown.h`), one launch across all
towers via `blockIdx.y`, bit-exact with the per-tower kernel (both
still exist; the original is still used by every other call site).
Isolated reproducer: 3.88x (sizeQl=5) to 5.93x (sizeQl=8) median
speedup at n=8192, launch-overhead-dominated. Full correctness
re-verified (every keyswitch/rotation/hoist test, both C++ and Rust
suites, bit-exact). **End-to-end effect NOT visible** in
`test_hoist_allstages`/`bench_hoist_multi` (within those tests' own
pre-existing noise band) -- honestly reported as a real, verified
kernel-level win with unmeasured end-to-end impact, exactly like the
Phase 39 Shoup kernels' own finding. **`rescale_resident_raw`'s
analogous per-tower loop (3 kernel types per tower, not 1) is a
second, likely comparably-sized opportunity, deliberately deferred to
respect this phase's "at most one" scope -- a strong Phase 41
candidate**, but structurally harder (needs its own reproducer-first
investigation, not a rushed extension of this phase's `ks_moddown`
change).

**A near-miss, caught and stopped, not hidden**: a `ctest -R` pattern
meant to match `test_boot_multi_input_small` also matched
`test_boot_n131072` via the substring `boot` -- a test that has never
completed in this project's history and has unbounded memory growth.
Caught via a free-RAM check (1.23 GiB) within about a minute and
stopped (only the one background task this session itself started).
**Never use a `ctest -R`/`-E` substring pattern containing `boot`
without also excluding `test_boot_n131072` explicitly** (e.g.
`-E test_boot_n131072`) -- this is now the second phase-level incident
near this specific test name; worth hard-coding the exclusion into a
project convention rather than relying on memory each time.

**Regression**: 124/135 verified passing this session (121 across the
4 standard batches + `test_n131072_keysize` + `test_n131072_cache_churn`
+ `test_n131072_gpu_lifecycle`, the last two both unexpectedly healthy
this session, unlike Phase 37/39), same 10 named pre-existing failures,
zero new regressions, `test_boot_n131072` deliberately not run.
Rust 62/0/1, both modes, unchanged, run both before and after the
Track E change.

## Earlier commit (Phase 39)

Check `git log -8 --oneline` on resume. As of this writing HEAD should be
the Phase 39 commit(s), on top of Phase 38's `d1dc2be` (verified matching
`origin/main` exactly, 0 ahead/0 behind, at Phase 39 start). Hardware
re-confirmed unchanged: RTX 2060 Max-Q, 6144 MiB, driver 595.97, CUDA
13.2, cmake 3.28.3, Rust/cargo 1.94.0, 16 logical cores, gcc 13.3.0,
Ubuntu 24.04.4. Windows free RAM started very low (~1.25 GiB) and
fluctuated ~1.9-6.7 GiB through the session -- `test_n131072_cache_churn`
was killed by the host twice and not verified passing this session.

**Full detail for this phase**:
`research/PHASE39_FHE_TEXTBOOK_RNS_AND_BATCHED_ENCRYPTION.md` and its
companion `research/PHASE39_FHE_TEXTBOOK_IMPLEMENTATION_MAP.md`.

**Reference PDFs** (the FHE textbook, Gentry's dissertation, TLShare)
are at `~/fhe-reference-material/` -- found in `~/Downloads/`, not
pre-placed; NOT committed to the repo (copyrighted, no redistribution
license). If this directory is missing next session, re-locate the
files or ask the user to re-supply them before assuming the textbook
mapping needs to be redone from scratch -- it doesn't; the mapping
document already captures the findings.

**K=32 is NOT a batched-kernel defect -- do not re-investigate the
batching implementation for it.** Root-caused this phase: the
UNCHANGED sequential arm shows the identical per-client-cost jump at
the identical K, ruling out anything about Phase 38's own grid/
occupancy/register design. Most likely a GPU power-management state
transition on this specific Max-Q laptop, tied to cumulative per-round
GPU-busy duration, not literally "32" as a number. A continuous
`nvidia-smi` clock-logging attempt across one full sweep did NOT
succeed this session (tooling limitation, not disproof) -- if revisiting,
fix that logging first rather than re-deriving the grid-geometry
argument again.

**Two RNS kernels were Shoup-ified this phase** (`kernels/moddown.cu`'s
`ApproxModDown` combine, `kernels/rescale.cu`'s `rescale_fuse_kernel`) --
bit-exact verified (35M-trial standalone reproducer + full existing
differential suite), isolated speedup near-invisible at n=8192 (launch-
overhead-dominated, ~1.03x) but real at n=131072 (1.44x-1.88x). **A
genuine near-miss, caught by the Rust suite, not the C++ one**: Rust's
own `rescale.rs` has a SEPARATE FFI declaration for `LaunchRescaleFuse`
(its build.rs compiles the kernel independently) that still had the old
signature after the C++ side was fixed -- caused a real SIGSEGV until
fixed. **If touching any kernel with a Rust FFI declaration again, grep
`rust/crates/*/src/*.rs` for its name FIRST**, before assuming the C++
call sites are the only ones.

**The textbook mapping identified `P >= Q_part_max`'s own textbook
justification** (`ApproxSwitchCRTBasis` = `FastBConv`, genuinely
approximate, `P` must absorb its noise) and found this project has NO
separate `ModDrop` operation at all (only full rescale) -- not a bug,
just a gap this project's shallow workloads have never needed to close.
`SmallMont`/`FastBConvEx` remain deliberately unimplemented (would touch
security-relevant `P`, or aren't needed by this project's CKKS
multiplication structure respectively).

**Regression**: C++ 130/134 (same 10 pre-existing failures, zero new
regressions) + `test_n131072_keysize` passing (twice); `cache_churn` NOT
verified this session. Rust 62/0/1, both modes, AFTER the FFI fix above.

## Immediate next actions, in order (Phase 39 handoff)

1. **Re-run `test_n131072_cache_churn` first** once free RAM is
   confirmed stable above ~3.2 GiB, not just momentarily.
2. **If revisiting the K=32 anomaly**, get continuous `nvidia-smi`
   clock/throttle-reason logging working across one full sweep before
   anything else -- this session's attempt failed for tooling reasons,
   not because the hypothesis is wrong.
3. **Do not add `SmallMont` or `FastBConvEx`** without a dedicated
   security review of `P >= Q_part_max`'s margin -- this phase
   deliberately left both unimplemented.
4. **If adding an n=131072 rescale/key-switch differential test**
   (deferred this phase, section 11 of the main report), that would
   close a real coverage gap -- this project's existing `test_rescale_*`
   suite only covers n=1024/n=8192.
5. **`test_boot_n131072` still needs fine-grained per-allocation memory
   instrumentation** before another attempt -- unchanged from every
   prior phase's handoff on this exact point.
6. Re-check Windows free RAM before any memory-sensitive step.

## Earlier commit (Phase 38)

Check `git log -8 --oneline` on resume. As of this writing HEAD should be
the Phase 38 commit(s), on top of Phase 37's `5061631` (verified matching
`origin/main` exactly, 0 ahead/0 behind, at Phase 38 start). Hardware
re-confirmed unchanged: RTX 2060 Max-Q, 6144 MiB, driver 595.97, CUDA
13.2, cmake 3.28.3, Rust/cargo 1.94.0, 16 logical cores, gcc 13.3.0,
Ubuntu 24.04.4. Windows free RAM fluctuated ~2.6-6.9 GiB this session --
this time nothing was killed; both n=131072 light tests passed cleanly
(including `test_n131072_cache_churn`, which Phase 37 could not verify
due to host memory pressure).

**Full detail for this phase**:
`research/PHASE38_MULTI_CLIENT_THROUGHPUT.md`.

**The headline result**: the existing sequential multi-client path has
NO economy of scale (confirmed by re-running Phase 29's own
`bench_aggregation_client_scaling` unmodified -- runtime scales
linearly with K, VRAM stays flat). Built a genuine BATCHED multi-client
encrypt function, `encrypt_host_shoup_pooled_barrett_batched_multiclient`
(new file -- the existing single-client function is completely
untouched), by generalizing Phase 34's own already-proven kernel-merge
technique one level further: `kernels/ntt_ct.cu` indexes towers purely
via `blockIdx.y`, so repeating the per-tower tables M times (once per
client, since mod/root are shared across clients in one aggregation
round) and widening `towers` to `4*sizeQ*M` is the identical
mathematical argument Phase 34 already used for v/e0/e1/m, applied
once more for M clients. Verified bit-exact against M sequential calls
to the existing function across a wide (n,sizeQ,M) matrix (including
M=1 and non-power-of-2 M) before trusting any performance number.

**Throughput result, reported honestly, not oversold**: real,
reproducible speedup for K<=16 (1.07x-1.38x, peaking ~1.3-1.4x at
K=16, confirmed across 3 independent runs), but a consistently
reproduced, NOT root-caused dip to ~1.02-1.04x at K=32 specifically
(bracketed by stronger results at K=16 and K=64, so it doesn't fit a
simple "overhead amortizes with scale" story on its own). **Do not
assume this dip is noise without checking** -- it reproduced across 3
separate runs. If investigating, start with Phase 37's own `[P37]`-style
internal CPU/GPU split instrumentation technique, applied to the
batched function specifically at K=32, before guessing.

**CPU/GPU overlap (Track C) was investigated, not implemented**: the
batched function front-loads ALL CPU work before its one GPU call, so
there's no within-call CPU work to overlap with that call's own GPU
phase. Real overlap needs a CROSS-BATCH double-buffered pipeline
(two pool sets, async transfers, a redesigned mutex model) -- a
separate, larger-scope change. **Do not attempt this with threads** --
Phase 36 already showed thread-spawn overhead can exceed small per-call
CPU work; any overlap attempt must use streams/async transfers, not
`std::thread`.

**A real, documented design characteristic, not a bug**: the new
batched pool is keyed by `(n,sizeQ,M)`, so each DISTINCT M used creates
its own permanent pool entry (unlike the single-client pool, keyed by
`(n,sizeQ)` only). Fine for a fixed-batch-size deployment (the realistic
case); would need an eviction policy if a future caller varies M across
many distinct values repeatedly. Not given one this phase since the
real workload doesn't need it.

**Regression**: C++ 130/134 (same 10 pre-existing failures, zero new
regressions) + both n=131072 tests passing (both, this time). Rust
62/0/1, both modes, unchanged.

## Immediate next actions, in order (Phase 38 handoff)

1. **Root-cause the K=32 throughput dip** before assuming it's noise --
   it reproduced across 3 independent runs this phase. Use internal
   CPU/GPU-split instrumentation (Phase 37's own technique) on the
   batched function specifically at K=32, not just external timing.
2. **If pursuing CPU/GPU overlap**, design a genuine cross-batch
   double-buffered pipeline (two pool sets, `cudaMemcpyAsync` + a
   dedicated stream + explicit `cudaEvent` ordering) -- do not spawn
   threads, and do not attempt it without a workload that actually has
   back-to-back batches to overlap (this project's current benchmarks
   don't model a sustained multi-round server; that may need building
   first).
3. **Do not assume the batched function's `(n,sizeQ,M)` pool needs an
   eviction policy** unless a real caller actually varies M across many
   distinct values repeatedly -- the realistic workload (fixed batch
   size) doesn't need one; don't add complexity speculatively.
4. **`test_boot_n131072` still needs fine-grained per-allocation memory
   instrumentation before another attempt** -- unchanged from every
   prior phase's handoff on this exact point.
5. Re-check Windows free RAM before any memory-sensitive step.

## Earlier commit (Phase 37)

Check `git log -8 --oneline` on resume. As of this writing HEAD should be
the Phase 37 commit(s), on top of Phase 36's `69abd0d` (verified matching
`origin/main` exactly, 0 ahead/0 behind, at Phase 37 start). Hardware
re-confirmed unchanged: RTX 2060 Max-Q, 6144 MiB, driver 595.97, CUDA
13.2, cmake 3.28.3, Rust/cargo 1.94.0, 16 logical cores, gcc 13.3.0,
Ubuntu 24.04.4. Windows free RAM was EVEN MORE VOLATILE this session
than Phase 36's own (observed range ~1.1-7.3 GiB, vs Phase 36's
1.98-5.9 GiB) — a base-suite `ctest` batch was killed by the host for
low memory even AFTER excluding the 4 heaviest n=131072-class tests;
re-run successfully in 4 smaller sub-batches by test-ID range instead.
`test_n131072_cache_churn` specifically was killed by the host TWICE in
a row despite both attempts starting with free RAM well above its own
~3.2 GiB documented requirement — check free RAM is not just above the
threshold but STABLE before re-attempting it.

**Full detail for this phase**:
`research/PHASE37_PRODUCTION_ENCRYPTION_CPU_OPTIMIZATION.md`.

**Measurement correction (not new science, but overdue bookkeeping)**:
added an explicit, non-destructive correction note directly under
Phase 34's original "CPU-side work is only 1.2-2.5%" claim in
`research/PHASE34_PERFORMANCE_INVESTIGATION.md` itself (not just in a
later phase's report) — confirmed no OTHER document's numeric claim
depends on the same flawed external-cudaEvent-bracketing technique.

**Track A — finer CPU profile (refines, not contradicts, Phase 36)**:
direct internal instrumentation (applied, measured across 320 real
production calls, reverted) cleanly separates RNG/alloc/embed/
table-build (CPU) from the GPU section from the combine loop (CPU) for
the first time — Phase 36's own 40-48% CPU-side estimate had lumped
the GPU section and the combine loop together, biasing it down; the
real split is **~53% CPU / ~43% GPU**. **RNG sampling is the single
largest CPU component (19.6% of a call, 97.8% of which is the actual
Gaussian/ternary draws, not setup overhead)** — explicitly NOT a safe
optimization target given this project's randomness/security
constraints; do not revisit this without dedicated, independently
reviewed cryptographic analysis of any proposed alternative.

**Track B/C — Phase 36's own speculative ranking was wrong, and the
actual safe candidate was smaller than hoped, but real**: Phase 36
ranked "replace the combine loop's per-element Shoup-reduction division
with a Barrett reduction" as its best-supported remaining candidate
WITHOUT working out the actual arithmetic. This phase did that analysis
and found a correct double-wide Barrett reduction needs genuine
multi-precision arithmetic (the 128-bit product times a ~63-bit mu
exceeds what `unsigned __int128` holds directly) and is NOT clearly
cheaper than Shoup's existing approach (which already reuses its one
division across 2 multiplies) — **do not re-attempt this exact
candidate without first costing out the real multi-precision
arithmetic**, which this phase's analysis already did once; repeating
that work from scratch would be wasted effort unless genuinely new
information changes the picture.

Instead implemented the EMBED loop's `barrett_reduce_u64`: replaced an
unbounded `while (r>=T.q) r-=T.q;` with 2 bounded `if` corrections,
justified by classical Barrett theory AND a 70-million-trial standalone
reproducer (kept: `research/benchmarks/bench_barrett_reduce_bound_check.cpp`)
plus a new PERMANENT regression test
(`tests/test_barrett_reduce_bounded_correction.cpp`). Isolated speedup:
a clean, reproducible **1.033x (3.3%)**. End-to-end
(`bench_aggregation_decomposed`, 7 interleaved A/B pairs): essentially
a wash (NEW median 81.82ms vs OLD 81.76ms; NEW mean 82.06ms vs OLD
82.44ms, ~0.46% favoring NEW) — small and noisy, but exactly consistent
with the isolated measurement's own back-of-envelope prediction (a 3.3%
win on a 16.4%-weighted component predicts ~0.5% end-to-end). KEPT: not
a fabricated win, but a real, rigorously-verified, zero-risk
improvement whose end-to-end contribution is honestly small. **If
re-measuring this change's end-to-end effect under calmer system
conditions, expect it to still be small — do not expect a large
end-to-end number just because the host is quieter.**

**Track D**: with this change in place, the dominant remaining costs
are GPU-side transfer (already at/near the measured PCIe floor,
pinned memory already explored and rejected) and CPU-side RNG sampling
(off-limits per above). The combine loop (14.6%) remains the only
other large CPU bucket and does NOT have an obviously-safe cheaper
algorithm per this phase's analysis — any future attempt needs its own
fresh standalone reproducer, not a repeat of Phase 36's speculation.

**Track E**: reviewed (not expanded) the `authenticated_financial_input`
boundary against the full mission checklist — confirmed already
adequate (metadata binding real, shape validation present, fail-closed,
deterministic serialization, mock labeling intact, core-library
isolation intact). Zero changes made.

**Regression**: C++ 118/128 (same 10 pre-existing failures, zero new
regressions) + `test_n131072_keysize` passing; `test_n131072_cache_churn`
NOT verified this session (resource exhaustion, see above — re-run
first next session once memory is confirmed stable, not just
momentarily sufficient); `test_boot_n131072`/`test_n131072_gpu_lifecycle`
deferred as usual. Rust 62/0/1, both modes, unchanged.

## Immediate next actions, in order (Phase 37 handoff)

1. **Re-run `test_n131072_cache_churn` first** — it was not verified
   this session due to resource exhaustion (not a regression). Confirm
   free RAM is stable above ~3.2 GiB for several checks in a row, not
   just momentarily, before attempting it.
2. **Do not re-rank the combine-loop Barrett-reduction candidate from
   scratch** — this phase already costed out the real multi-precision
   arithmetic required and found it unpromising as currently
   understood. If revisiting, start from that analysis
   (`research/PHASE37_PRODUCTION_ENCRYPTION_CPU_OPTIMIZATION.md`
   section 5/6.1), not a fresh guess.
3. **Do not attempt an RNG algorithm/distribution change** without
   dedicated, independently reviewed cryptographic analysis — it is
   the largest CPU cost found this phase, but explicitly off-limits
   without that review.
4. **`test_boot_n131072` still needs fine-grained per-allocation memory
   instrumentation before another attempt** — unchanged from every
   prior phase's handoff on this exact point.
5. Re-check Windows free RAM before any memory-sensitive step — this
   session's volatility (1.1-7.3 GiB) may or may not persist.

## Earlier commit (Phase 36)

Check `git log -8 --oneline` on resume. As of this writing HEAD should be
the Phase 36 commit(s), on top of Phase 35's `a07ecf8` (verified matching
`origin/main` exactly, 0 ahead/0 behind, at Phase 36 start). Hardware
re-confirmed unchanged: RTX 2060 Max-Q, 6144 MiB, driver 595.97, CUDA
13.2, cmake 3.28.3, Rust/cargo 1.94.0, 16 logical cores. Windows free RAM
was UNUSUALLY VOLATILE this session (observed 1.98-5.9 GiB) — an initial
full-suite `ctest` run was killed outright by the host for running the
system low on memory; subsequent work used memory-aware batches with an
explicit `Get-CimInstance Win32_OperatingSystem` check before each
memory-sensitive step. Check free RAM again at the start of the next
session before assuming the same batching plan still applies.

**Full detail for this phase**:
`research/PHASE36_END_TO_END_OPTIMIZATION.md`.

**Track A/B — two documented null/negative performance results, no
change shipped**: re-confirmed encrypt is still 85.6% of end-to-end
(`bench_aggregation_decomposed`). Direct internal instrumentation (fully
reverted) corrected Phase 34's own prior claim that CPU-side work is
only ~1-2% of one encrypt call — the real split is ~40-48% CPU-side
(RNG+embed+combine) vs ~52-60% GPU-side; Phase 34's external-cudaEvent-
bracketing methodology was flawed (the closing event can only be
enqueued after all host-only work already ran). Implemented and
rigorously measured the top 2 of 5 ranked candidates: (1) thread-
parallelizing the embed/combine loops (duplicated `parallel_for` from
`src/encode.cpp`) was bit-exact-correct but a CONFIRMED REGRESSION
(encrypt-stage median 75ms -> ~155-157ms — `std::thread` spawn/join
overhead per call swamps the sub-millisecond work being parallelized,
since this function calls `parallel_for` twice per encrypt call and
encrypt is called 32x per aggregation round); (2) pooling the per-call
host scratch buffers was bit-exact-correct but a CONFIRMED NULL RESULT
via interleaved `git stash` A/B (74-75ms both before and after, within
noise — glibc already reuses freed blocks cheaply at this size). BOTH
reverted, confirmed zero diff. **Next session should NOT re-attempt
either of these exact changes** — the evidence against both is now
direct and specific (thread-spawn-overhead granularity mismatch;
allocator-already-cheap), not merely "didn't help."

**The best-supported remaining candidate, not yet attempted**:
replacing the combine loop's per-element 128-bit `shoup_precon` division
with a Barrett-style reduction reusing this file's own `TowerBarrett`
machinery (already used for the embed step). This touches core modular
arithmetic used by every encrypt call — if pursued, build a STANDALONE
reproducer first (a tiny isolated test comparing Shoup-reduction output
vs. a proposed Barrett-reduction-of-a-128-bit-product output across many
random inputs, bit-for-bit) before touching
`encrypt_host_shoup_pooled_barrett_batched.cpp` itself, per this
project's rule against directly modifying core numeric kernels without
one.

**Track C — authenticated-input hardening (shipped)**: found and fixed
two real gaps in the Phase 35 prototype — `CkksEncodingMetadata` was
never checked against anything (now bound into the attestation AND
checked against the session's expected values, two distinct new
rejection reasons: `RejectedEncodingMismatch` for a self-consistent-but-
wrong-session envelope, `RejectedBadAttestation` for a post-attestation
tamper of `enc_meta`), and ciphertext shape was never validated beyond
non-emptiness (now checked via `RejectedCiphertextShapeMismatch`). 14/14
checks passing (up from 11/11). `libgpufhe` untouched.

**Track D — CKKS provenance research (not implemented, as intended)**:
separated the provenance problem into six distinct sub-claims (TLS
provenance / commitment / parsing / CKKS encoding correctness /
encryption correctness / ciphertext-matches-committed-value) and
identified that CKKS's floating-point canonical-embedding encoding step
— which BGV/BFV do not have — is the one sub-claim with no identified
proof technique at all (floating-point rounding inside a ZK proof is a
known-hard, unsolved-in-general problem). Named candidate lattice-native
proof-system families (LaBRADOR/Greyhound/LatticeFold-class) as a
plausible-but-unsized direction for the encryption-correctness
sub-claim specifically — not verified to fit this project's actual
parameters, not implemented.

**Regression**: C++ 117/127 (same 10 pre-existing failures) plus both
`test_n131072_keysize`/`test_n131072_cache_churn` passing (measured
against their OWN actual memory requirements, ~1.3GB/~3.2GB, not a
blanket threshold). `test_boot_n131072`/`test_n131072_gpu_lifecycle`
deferred — memory gate (6-10 GiB class) never cleared this session, and
`test_boot_n131072` specifically has never completed a single bootstrap
stage in this project's history regardless of memory available (see
`research/MEMORY_FEASIBILITY.md`'s multiple OOM attempts) — do not
re-attempt it without first adding the fine-grained per-allocation
memory instrumentation that document's own "correct next step" section
calls for. Rust: 62/0/1, both modes, unchanged.

## Immediate next actions, in order (Phase 36 handoff)

1. **Re-check Windows free RAM before anything memory-sensitive** — this
   session's volatility (1.98-5.9 GiB) may or may not persist.
2. **If pursuing more encrypt-stage CPU optimization, start with a
   standalone Shoup-vs-Barrett reduction reproducer** (see above) before
   touching production code — do not skip straight to the production
   file given this project's now-two-for-two record of plausible-seeming
   CPU optimizations failing to transfer to the real workload once
   actually measured.
3. **Do not re-attempt thread-based parallelization of small per-call
   CPU loops in this file** — the evidence (section 4.1 of the Phase 36
   report) is specific and direct, not just "didn't help."
4. **`test_boot_n131072` needs fine-grained per-allocation memory
   instrumentation BEFORE another attempt** — three-plus prior attempts
   across multiple phases have all OOM'd without ever reaching a full
   bootstrap stage; guessing at `KROT_CAP`/`KSC_CAP` again is not a sound
   next step (see `research/MEMORY_FEASIBILITY.md`).
5. **If pursuing CKKS provenance research further**, the first bounded
   step is sizing a lattice-native proof system (LaBRADOR/Greyhound/
   LatticeFold-family or a successor) against this project's actual ring
   dimensions for the encryption-correctness sub-claim ALONE — treat the
   CKKS-encoding-correctness sub-claim (floating-point rounding) as open
   research, not an engineering task.

## Earlier commit (Phase 35)

Check `git log -8 --oneline` on resume. As of this writing HEAD should be
the Phase 35 commit(s), on top of Phase 34's `2012092` (verified matching
`origin/main` exactly, 0 ahead/0 behind, at Phase 35 start). Hardware
re-confirmed unchanged: RTX 2060 Max-Q, 6144 MiB, driver 595.97, CUDA
13.2, cmake 3.28.3, Rust/cargo 1.94.0. Windows free RAM fluctuated
1.97-5.11 GiB across this session.

**Read the uploaded paper in full** (TLShare, IACR CiC Vol 3 No 3, all
33 technical pages) before any implementation. **Conclusion, stated
directly**: none of TLShare's three protocols (MPC/Pedersen-VSS, FHE
with NIZK/CGGI, FHE with IZK/prime-field LWE) are compatible with this
project's ring-based CKKS as published — two use a different FHE scheme
entirely (CGGI/TFHE) or no FHE at all (MPC mode), and the third is
explicitly limited by the paper's OWN authors to prime-field LWE,
stated as lacking ring-structure support. A genuine CKKS-compatible
provenance proof would need to handle ring (polynomial) arithmetic, a
second encoding-approximation source CKKS has that neither paper
construction needs to prove anything about, and circuits plausibly
orders of magnitude larger than the paper's own already-substantial
148-million-gate/1.6-million-edaBit n=2048 scalar LWE circuit — none of
this was attempted, invented, or claimed solved this round. Full
protocol-by-protocol comparison, the exact compatibility reasoning, and
the paper's own threat-model assumptions:
`research/PHASE35_TLSHARE_CKKS_INTEGRATION.md`.

**Confirmed, from direct source reading, the actual input-authenticity
gaps in the existing financial aggregation application**: client values
in `bench/bench_private_aggregation_workload.cpp` are 100% synthetic
(`std::uniform_real_distribution` fed by a fixed-seed `mt19937_64`), with
no source authentication, no session/client-identity concept, and no
replay protection anywhere in the codebase — none of these are
confirmed vulnerabilities (there is no live service or network boundary
to exploit), all are architectural gaps relative to a hypothetical real
deployment, exactly the gap TLShare's paper targets. Full findings
table in the Phase 35 report section 5.

**Built a scoped, explicitly-mock authenticated-input prototype**
(`include/authenticated_financial_input.h`,
`src/authenticated_financial_input.cpp`,
`examples/authenticated_financial_input/`), kept OUT of `libgpufhe`
(compiled only into its own executable) so this research module cannot
affect any core CKKS build target. Demonstrates session binding, source
identity, and rejection of tampered records/changed sessions/altered
metadata/duplicate submissions/invalid attestations/missing
ciphertexts (11/11 checks, `ctest -R authenticated_financial_input`,
re-run 3x stable) — using the REAL, unmodified
`encode_host_fft`/`encrypt_host_shoup_pooled_barrett_batched`/
`ct_add_ct_host`/`decrypt_host`/`decode_host_fft` pipeline for the actual
cryptography, with the "attestation" itself explicitly, repeatedly
labeled as a non-cryptographic mock (NOT HMAC, NOT a signature scheme).
A real bug was caught in this round's own test harness before trusting
any result: the test's commitment-digest helper didn't incorporate the
session, causing two of its own test cases to collide and fail for the
wrong reason — fixed, then re-verified all 11 checks pass for the right
reasons.

**Encode/FFT investigation — a documented null result, not a shipped
change**: re-verified encode is still the next-largest relative stage
(~11.7-12% of end-to-end). Found, by direct reading of
`src/encode_fft.cpp`, that `fft_unnormalized_inverse` takes its
parameter by value and its one call site passes an lvalue, silently
copying a 256 KB buffer every encode call that could safely be moved
instead (confirmed unused afterward). Isolated measurement showed a
real, reproducible ~1.82-1.87x improvement — but a proper `git
stash`-based A/B directly on the real `bench_aggregation_decomposed`
workload showed **no measurable difference** (9.98-10.05ms before vs.
10.00-10.07ms after, within noise). Investigated why rather than just
reporting the discrepancy: the isolated probe runs encode in a tight,
continuously-CPU-busy loop; the real workload interleaves each encode
call with a multi-millisecond GPU-bound phase, plausibly letting the CPU
downclock between calls — the direct CPU-frequency-scaling analog of
Phase 34's own GPU-clock-drift lesson. Per this phase's explicit "if
the result is null, revert your own experiment" instruction, the
`src/encode_fft.cpp` change was reverted (confirmed zero diff via
`git diff` afterward) despite being correctness-neutral — the standalone
timing probe is kept as a reusable tool, not shipped as a production
change.

**Regression (re-run fresh this round)**: C++ `ctest`: **117/127
passing**, the exact same 10 pre-existing documented failures as every
prior phase, zero new regressions — this count includes both
n=131072 tests (both re-confirmed passing this round, `test_n131072_cache_churn`
run once memory recovered to 5.11 GiB) and the new
`authenticated_financial_input` test. Rust: 62 passed/0 failed/1
ignored, both debug-serial and release-parallel, unchanged.

**Left genuinely uncommitted and out of scope** (carried forward
unchanged): `graphify-out/GRAPH_REPORT.md`/`graph.json`, `.claude/agents/`,
`.claude/skills/scrum-orchestrator/`, `.scrum-orchestrator/`,
`tools/__pycache__/`.

## Immediate next actions, in order (Phase 35 handoff)

1. **Do not re-attempt a CKKS provenance proof by transplanting CGGI or
   prime-field LWE** — read `research/PHASE35_TLSHARE_CKKS_INTEGRATION.md`
   section 3 first. If pursuing this at all, the first real step is
   sizing an RLWE-adapted proof circuit's complexity at this project's
   actual ring dimensions (8192+), not implementing one blind.
2. **The mock attestor's `attest()` currently receives the plaintext
   value indirectly via the commitment** — a real design should make it
   operate on commitments only. Fix this before adding any more features
   to the authenticated-input prototype.
3. **The CPU-power-state hypothesis for the encode null result is not
   independently confirmed** — only inferred from the measurement
   pattern. If revisiting encode performance, confirm it directly (e.g.
   sampling `/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`
   during the real workload) before trying another isolated
   micro-optimization that may not transfer either.
4. **Run `tests/test_boot.cpp` the moment the 10 GiB memory gate
   clears** — now EIGHT phases running without this verification.
5. Re-run the Windows free-RAM gate check before any of the above.

## Earlier commit (Phase 34)

Check `git log -8 --oneline` on resume. As of this writing HEAD should be
the Phase 34 commit(s), on top of Phase 33's `f95eb74` (verified matching
`origin/main` exactly, 0 ahead/0 behind, at Phase 34 start). Hardware
re-confirmed unchanged: RTX 2060 Max-Q, 6144 MiB, driver 595.97, CUDA
13.2, cmake 3.28.3, Rust/cargo 1.94.0. Windows free RAM fluctuated
2.08-5.14 GiB across this session (competing desktop load, not this
phase's own commands).

**Both n=131072 tests Phase 33 deferred are now closed, genuinely
passing, not just attempted**: `test_n131072_keysize` (peak RSS 1314 MB,
matches history exactly) and `test_n131072_cache_churn` (peak RSS
3232-3233 MB, matches history's documented plateau exactly). Phase 33's
own "same memory-gate class as `test_boot_n131072` (6 GiB)" characterization
was corrected this round: each test has its own much smaller,
already-measured requirement (~1.3 GB and ~3.2 GB respectively,
documented in `research/MEMORY_FEASIBILITY.md` and
`research/EXPERIMENT_LEDGER.md` entry #22) — both ran successfully well
below the blanket 6 GiB figure that actually applies only to the full
bootstrap scenario.

**Regression, re-run before and after this round's production change**:
C++ `ctest` (excluding the 5 always-excluded full-bootstrap tests):
**115/125 passing both times**, the exact same 10 pre-existing
documented failures as every prior phase, zero new regressions — this
count now includes BOTH n=131072 tests as passing members, not
exclusions. Rust: 62 passed/0 failed/1 ignored, both debug-serial and
release-parallel, unchanged before and after.

**Performance: the historical bottleneck (encrypt, ~86% of end-to-end)
was re-verified still current (86.3-86.4% fresh, vs. 86.9% historical —
within noise), then genuinely reduced by ~1.1x end-to-end with a real,
evidence-backed, bit-exact-preserving experiment.** Three new standalone,
non-invasive probes (zero production code touched for the measurement
itself) decomposed WHY encrypt dominates, fresh: CPU-side work
(RNG+Barrett-embed+Shoup-combine) is now only 1.2-2.5% of one encrypt
call (Phases 13/14's optimizations fully succeeded); the remaining
97.5-98.8% is GPU-side transfer+kernel. Reading `kernels/ntt_ct.cu`
directly found `LaunchNTT_CT_Batched` issues ~5 internal kernel launches
per call, and production calls it 4 separate times (v/e0/e1/m) = 20
total launches/encrypt-call. Verified (bit-exact, 0/131072 limbs differ)
that merging those 4 calls into 1 (towers=4*sizeQ, exact reorganization
of identical work, not an approximation) is valid, then measured its
real impact with a properly INTERLEAVED A/B (a non-interleaved first
attempt gave wildly inconsistent 2.3x-5.5x readings across repeated
runs — tracked to this laptop GPU's clock ramp-up/down behavior over
millisecond-scale kernels, a real methodology lesson, not a flaw in the
underlying effect) — stable, reproducible **~1.2x** on the transfer+kernel
portion (1.206x, 1.240x across two runs).

**Implemented in production** (`src/encrypt_host_shoup_pooled_barrett_batched.cpp`):
the pool's per-tower tables are now 4*sizeQ-length (repeating the real
sizeQ tables 4 times, built ONCE per (n,sizeQ) pool entry — zero new
per-call host work), and the hot path issues 1 merged
`LaunchNTT_CT_Batched` call instead of 4. All 4 directly-relevant
existing tests (2 correctness, 2 concurrency-audit) re-run immediately
after and PASSED completely (832 total checks, 0 mismatches) before
anything else proceeded. Real end-to-end re-measurement (not an
extrapolation): `bench_aggregation_decomposed` encrypt-stage median
83.76-84.61ms → 75.05-75.64ms (~1.11-1.13x), end-to-end median
97.05-97.91ms → 87.84-88.82ms (~1.09-1.12x); `bench_private_aggregation_workload`
mean 98.97ms → 91.13ms (~1.086x), correctness (`worst abs err`)
identical. Full regression re-confirmed clean after the change (same
115/125, same Rust 62/0/1).

**`DeviceBuffer<T>` (Part 6)**: not reopened — no new genuine second
caller was found or sought; the kernel-merge change touches only table
shape/launch-count inside `EncryptBufPoolBatched`, a type Phase 33
already found unsuitable for migration without a redesign.

**Left genuinely uncommitted and out of scope** (carried forward
unchanged): `graphify-out/GRAPH_REPORT.md`/`graph.json`, `.claude/agents/`,
`.claude/skills/scrum-orchestrator/`, `.scrum-orchestrator/`,
`tools/__pycache__/`.

## Immediate next actions, in order (Phase 34 handoff)

1. **Encode is now, in relative terms, a larger share of end-to-end**
   (~12% of the new, smaller total, up from ~11% before this round's
   encrypt win) — purely because encrypt shrank, not because encode
   itself changed. It remains a pure-CPU FFT stage
   (`encode_host_fft`/`decode_host_fft`, confirmed zero CUDA calls) —
   if performance work continues on this exact workload, encode is the
   next largest single stage after encrypt, and has not been
   re-profiled since Phase 11/28's own work.
2. **The non-interleaved-vs-interleaved GPU clock-drift lesson** (this
   round's section 6) should be written up as a reusable methodology
   note if another sub-millisecond GPU micro-benchmark is ever built on
   this hardware — right now it exists only inside
   `research/PHASE34_PERFORMANCE_INVESTIGATION.md`, not anywhere more
   general.
3. **Run `tests/test_boot.cpp` the moment the 10 GiB memory gate
   clears** — now SEVEN phases running without this verification; this
   round's own gate checks never saw free RAM above 5.14 GiB.
4. **Re-run the 4 standalone probes this round left in
   `research/benchmarks/`** (`bench_encrypt_cpu_gpu_split.cpp`,
   `bench_transfer_only.cpp`, `bench_ntt_merge_experiment.cpp`,
   `bench_full_sequence_v2.cpp` — all genuinely standalone, not wired
   into `CMakeLists.txt`, zero risk to the main build) if revisiting the
   encrypt path again, rather than re-deriving the CPU/GPU split or
   transfer-floor numbers from scratch.
5. Re-run the Windows free-RAM gate check before any of the above.

## Earlier commit (Phase 33)

Check `git log -8 --oneline` on resume. As of this writing HEAD should be
the Phase 33 commit(s), on top of Phase 32's `f6ab541` (verified matching
`origin/main` exactly, 0 ahead/0 behind, at Phase 33 start). Hardware
re-confirmed unchanged: RTX 2060 Max-Q, 6144 MiB, driver 595.97, CUDA
13.2, Rust/cargo 1.94.0.

**All four caches Phase 32 flagged as unguarded are now resolved** — not
with a blind mutex sweep, but with real reachability tracing, real
mechanism-level proof via ThreadSanitizer, and a real before/after
performance measurement. `research/PHASE33_CACHE_CONCURRENCY_AUDIT.md`
has the full evidence trail; summary:

**Inventory**: Phase 32's "4 of 13 unguarded" was 4 FILES, 5 cache
INSTANCES (`src/rotate.cpp` has two). All 5 traced to their real callers
this round, not assumed: `rotate.cpp`'s `dev_tab()` (host-orchestrated
automorphism path) and `autoPermMap()` (resident-fast automorphism path)
are both reachable via `tests/test_rotate_concurrent_cache.cpp`'s
existing Phases D and B respectively (9 real concurrent threads each,
with genuine identical-key collisions for `autoPermMap` since the
fixture's rotation amount is fixed across each 3-thread group).
`keyswitch.cpp`'s `dev_tab()` (inside `keyswitch_core_resident` — a
DIFFERENT function from the already-fixed `keyswitch_resident.cpp`;
an initial hypothesis that `rotate_ct_resident` reaches it was checked
directly against the function body and found FALSE) is reachable via
the same test's Phase D AND via the Rust FFI bridge's
`gpufhe_ks_fixture_run_reference` oracle. `encdec.cpp`'s `enc_dev_tab()`
is reachable via `tests/test_encrypt_host_shoup_pooled_thread_safety.cpp`'s
own concurrent reference computation (8 threads, same key, incidental
but real). `encrypt_host_shoup.cpp`'s `enc_dev_tab_shoup()` had NO
existing concurrent reproducer at all — the one genuine gap, closed with
a new test this round.

**Mechanism proof**: ThreadSanitizer cannot run directly against this
project's CUDA-linked binaries in this WSL2 environment (`FATAL:
ThreadSanitizer: unexpected memory mapping`, reproduced even against a
trivial CUDA-free program under default ASLR — resolved with `setarch
$(uname -m) -R` for the one invocation, a documented TSan/high-entropy-
ASLR interaction on modern kernels, not a system config change). A
faithful, CUDA-free extraction of the exact `find()`-then-`emplace()`
pattern, stressed with the identical 9-thread/3-colliding-key-groups
shape as the real test, produced **confirmed ThreadSanitizer data
races** in `std::map`'s internal `_Rb_tree` (concurrent `find()` vs.
`emplace()`-driven tree rebalancing, and a heap-allocation race). The
identical reproducer with only a mutex added: zero races. This is real
proof the pattern is unsafe and real proof the fix resolves it — not a
theoretical C++-memory-model argument.

**Fix**: the established, already-proven-low-risk pattern from Phases
28/30 (one `std::mutex` per file, `std::lock_guard` around the whole
find+build+emplace+return sequence) applied to all 5 instances across
`src/rotate.cpp`, `src/keyswitch.cpp`, `src/encrypt_host_shoup.cpp`,
`src/encdec.cpp`. Zero lines of existing cache-building logic altered.
**13/13 process-lifetime caches in this codebase are now mutex-guarded**
(up from 9/13).

**New regression test**: `tests/test_encrypt_host_shoup_concurrency.cpp`
(8 threads x 8 iters, same-seed same-key double-call agreement check) —
closes `encrypt_host_shoup.cpp`'s previously-nonexistent coverage.
Registered with `ctest`.

**Performance**: real `git stash`-based before/after A/B (not assumed) —
`bench_evalmult`, `bench_hoist_ab`, `bench_encrypt_shoup_vs_reference`,
3 runs each, before and after. No systematic difference in either
direction; both sit inside this machine's own already-documented ~30%
run-to-run spread.

**`DeviceBuffer<T>` second-caller evaluation**: audited for a genuine
production migration candidate. None of the five just-fixed caches'
value types (`DevTab`, `EncDevTab` — each holding MULTIPLE distinct
device pointers per entry) fit `DeviceBuffer<T>`'s current
single-pointer/single-type-array shape without a non-trivial redesign;
`DeviceCiphertext` remains narrower-contracted (synchronous-only,
fixed two-array role) and migrating it would risk a proven-correct type
for no measurable benefit. **Deferred, by evidence, not oversight** —
the missing capability (a documented multi-buffer-per-cache-entry
convention) is named in
`research/PHASE33_CACHE_CONCURRENCY_AUDIT.md` section 7.

**Regression (re-run fresh this round)**: C++ `ctest` (same memory-gate
exclusions since Phase 28): **115/125 passing**, the exact same 10
pre-existing documented failures as Phase 32's baseline
(`test_evalkeygen_isolated`, `test_keyswitch_stage_trace`,
`test_polyeval`, `test_stage2`, `test_boot_multi_input_small`, and the
five `test_rescale_*_n1024*` family), zero new regressions, plus the new
`test_encrypt_host_shoup_concurrency` passing. Run in 5 smaller batches
rather than one monolithic invocation: the Windows host's free RAM was
unusually tight this round (2.2-5.1 GiB across the session, likely
competing desktop load on the user's machine, not caused by this
phase's own commands — confirmed via `Get-Process`: Opera, Spotify, and
Memory Compression were the largest consumers alongside `vmmemWSL`), and
three single-invocation full-suite attempts were killed by the harness's
own low-memory guard before completing. Splitting into ~20-45-test
batches let each individual `ctest` invocation finish before the next
spike. **Two tests were honestly deferred, not run and not claimed
passing or failing**: `test_n131072_keysize` and
`test_n131072_cache_churn` are n=131072-scale tests that fall under this
project's own established 6 GiB memory gate (same class as
`test_boot_n131072`); measured free RAM when attempting them was 4.84-5.1
GiB, below that gate. Both passed cleanly in Phase 31/32's runs when
memory was sufficient, and neither is touched by this phase's changes
(no cache fixed this round sits on their call path) — re-run them the
moment the gate clears, before claiming the suite complete. Rust
(debug-serial, release-parallel): **62 passed, 0 failed, 1 ignored**,
both modes, identical to the established baseline since Phase 25.

**Left genuinely uncommitted and out of this round's scope** (carried
forward unchanged): `graphify-out/GRAPH_REPORT.md`/`graph.json`,
`.claude/agents/`, `.claude/skills/scrum-orchestrator/`,
`.scrum-orchestrator/`, `tools/__pycache__/`. This round's `build_tsan/`
directory (the ThreadSanitizer build tree used for the standalone
reproducer in section 3) was removed after its evidence was captured —
a local build artifact, never staged, not needed going forward.

## Immediate next actions, in order (Phase 33 handoff)

1. **Re-run `test_n131072_keysize` and `test_n131072_cache_churn`** the
   moment Windows free RAM clears the established 6 GiB gate — deferred
   this round due to a live, momentary host memory constraint (section
   above), not run, not claimed passing or failing. Both passed cleanly
   last time memory was sufficient (Phase 31/32) and neither is on this
   round's changed call path, but re-verify rather than assume.
2. **`DeviceBuffer<T>`'s multi-buffer-per-entry gap** (section 7 of
   `research/PHASE33_CACHE_CONCURRENCY_AUDIT.md`): design a documented
   convention (small fixed-size array of `DeviceBuffer<T>` members, or a
   `DeviceBufferGroup` type) before attempting any of these 5 caches'
   `DevTab`/`EncDevTab` value types as a migration target.
3. **Re-run `tests/test_boot.cpp` the moment the memory gate clears** —
   now SIX phases running without this verification.
4. **The generic synchronized-cache-template follow-up** (de-prioritized
   this round, since the actual safety gap is closed): revisit only if a
   14th cache of this exact shape gets added and the duplication becomes
   a real maintenance cost.
5. Re-run the Windows free-RAM gate check before any of the above.

## Earlier commit (Phase 32)

Check `git log -8 --oneline` on resume. As of this writing HEAD should be
the Phase 32 commit(s), on top of Phase 31's `42944c7` (verified matching
`origin/main` exactly, 0 ahead/0 behind, at Phase 32 start). Hardware
re-confirmed unchanged: RTX 2060 Max-Q, 6144 MiB, driver 595.97, CUDA
13.2 (`nvcc` 13.2.51), Rust/cargo 1.94.0, compute-sanitizer 2026.1.0.0.
Windows free RAM measured 2.23-4.60 GiB across this session — still
below both the 10 GiB (n=8192) and 6 GiB (n=131072) bootstrap gates;
the full bootstrap was NOT run this round (consistent with every phase
since 25).

**Phase 32 delivered the five mandated work packages**, organized as:
architecture audit → two minimal new primitives → a real non-FHE
application proving them → a fresh CKKS end-to-end re-measurement →
bounded edge-AI/LLM-KV-cache research prototypes → a self-conducted
independent review. Used three parallel research-only forks (never
touching production code concurrently with the integration owner) for
the architecture audit, edge-AI prototype, and LLM-KV-cache prototype;
the application workload, CKKS audit, and all integration/commits were
done by one owner.

**Work Package A (architecture)**: `research/GPU_RESIDENT_ARCHITECTURE.md`.
Finding: no general device-buffer or execution-context abstraction
existed; every feature (keyswitch, encrypt, encode, rotate) independently
reinvented a similar RAII-wrapper-plus-process-lifetime-cache shape.
Independently re-verified this round's most actionable claim (grep
confirms 4 of 13 `static std::map` device-resource caches —
`rotate.cpp` x2, `keyswitch.cpp`, `encrypt_host_shoup.cpp`,
`encdec.cpp` — have zero `std::mutex`, unlike the 9 already fixed in
Phases 28/30; `rotate.cpp`'s case is Phase 25's own documented
"latent, not active" decision, the other three were never flagged
either way). Recommended exactly two additive primitives (not a
framework): `DeviceBuffer<T>` and `cuda_check()`. Explicitly recommended
AGAINST a generic execution-context or generalized CUDA-graph facility
this round — no second real caller had asked for either yet.

**Implemented both recommended primitives**: `include/device_buffer.h`
(`gpufhe::DeviceBuffer<T>`, generalizing `DeviceCiphertext`'s proven
RAII-move-only-owned-pointer pattern to an arbitrary type/count, sync by
default with explicit `_async` variants requiring the caller to own
ordering) and `include/cuda_check.h` (promoting `device_ciphertext.cpp`'s
one-off `ck()` helper to a shared header — directly closing the "nothing
checked return codes" gap Phase 31 found at the root of its own bug).

**Work Package B (real non-FHE application)**: `examples/resident_finance/`
— a GPU-resident streaming EWMA-volatility + vectorized Black-Scholes
simulation, built on `DeviceBuffer<double>` (verified by reading the
file, not by trusting its own comments — see
`research/PHASE32_INDEPENDENT_REVIEW.md` section 3). Caught and fixed a
real bug in this round's OWN draft before any build: an early version
uploaded a tick directly into the resident `price` buffer and then
passed that same buffer as both the "old price" and "new price" argument
to the EWMA kernel, which would have made every log-return exactly zero
and produced a silently-wrong, constant-zero-volatility simulation that
still "passed" its own self-consistency check (GPU and CPU would have
agreed on the same wrong answer) — fixed by uploading into a dedicated
`newPrice` buffer instead. Correctness: GPU vs. CPU-reference (identical
tick stream, never two independently-seeded RNGs) max abs error at
K=4096/2000 ticks: price=0, ewmaVar=1.6e-19, callPrice=3.6e-14,
vol=1.1e-16 — all double-precision-transcendental-noise scale. **Honest
crossover sweep, not a single cherry-picked K**: GPU LOSES below ~K=1000
(0.004x-0.636x at K=8/32/128/512 — CPU wins, launch/transfer overhead
dominates) and WINS from ~K=2000 up (2.231x at K=2048, rising to 18.952x
at K=131072). Registered with `ctest` (`resident_finance`, self-checking,
exit-code pass/fail).

**Work Package D (CKKS)**: `research/CKKS_RESIDENCY_AUDIT.md`.
Grep-verified per-operation residency table (encode/decode: pure host,
intentional; `ckks_ops.cpp`'s mul/rescale: genuinely chain-resident, one
upload + zero mid-chain transfers; `keyswitch_resident`: resident for
ciphertext/kernels, not yet for the eval key — Phase 25's finding,
unchanged, `KsResidentCache` still built-tested-unintegrated). Re-ran
(not copied) the existing 32-client aggregation benchmark fresh:
end-to-end mean 105.84 ms, production-path speedup 1.518x vs. the old
plain-encrypt arm, encrypt remaining 86.9% of end-to-end (the confirmed
next bottleneck for this workload). Re-ran `bench_resident_chain` fresh
and reported its result exactly as measured: **no statistically
significant GPU advantage at n=32768/towers=6** (95% CI brackets zero) —
stated honestly rather than re-tuned into a better-looking number. No
cryptographic parameter touched; h=64/sigma=3.19/P>=Q_part_max all
re-confirmed unchanged and, for the latter, actively still rejecting an
intentionally-undersized fixture with the identical error message
(the five `test_rescale_*_n1024*` pre-existing failures, section 5).

**Work Package C (edge AI) and E (LLM KV-cache)**: both delivered their
required research documents (`research/EDGE_SENSOR_RESIDENCY_AUDIT.md`,
`research/LLM_KV_CACHE_RESEARCH.md`) PLUS a working prototype each
(neither deferred), independently re-built and re-run by the integration
owner this round rather than trusted on the forks' own say-so:
- Edge: `research/edge_prototype/edge_sensor_residency_prototype.cu`
  (standalone, not wired into CMakeLists.txt) — 1M-point synthetic cloud,
  chained pose-update→transform→EMA-filter with zero host round-trips
  between stages, periodic centroid readback. Re-run this round: 15.54x
  warm per-step speedup vs. CPU reference (fork reported ~13x — both
  real measurements, the difference being this project's own
  already-documented run-to-run GPU clock/thermal variance, not a
  discrepancy), correctness 5.96e-8 (float32-epsilon scale), reproduced
  exactly. No crossover sweep across point count was done (flagged
  follow-up).
- LLM KV-cache: `research/kv_cache_prototype/kv_block_cache_sim.cpp`
  (standalone, CPU-only, zero CUDA, zero GPU contention risk) — a block
  pool modeled directly on this project's own `KsResidentCache` design
  (composite-key-only, bound-by-bytes-and-count, fail-closed,
  synchronize-before-free lessons). Re-run this round: 768/768 block
  reads bit-exact under a deliberate 6x-oversubscribed adversarial access
  pattern (704 evictions, 384 host-spill-and-restores), reproduced
  exactly. Explicit recommendation: do NOT build a production KV-cache
  yet; port this allocator logic onto `DeviceBuffer`/a real
  `cudaMallocAsync` path only once a second real GPU caller (this
  prototype, or something like it) actually needs it.

**Independent review**: `research/PHASE32_INDEPENDENT_REVIEW.md`. Honest
about its own limitation — the architecture/edge/LLM-KV-cache work had
genuine cross-agent independence (three parallel forks with no
visibility into each other), but the application workload and CKKS audit
were self-reviewed by the same reasoning process that built them, stated
plainly rather than dressed up as a second opinion. Re-verified (not
re-trusted) the architecture audit's unguarded-cache claim and both
prototypes' numbers by independently re-running everything.

**Regression (re-run fresh this round)**: C++ `ctest` (excluding the
memory-gated bootstrap tests, same convention since Phase 28):
**114/124 passing** (up from 113/123 — `resident_finance` newly
registered and passing), the exact same 10 pre-existing documented
failures as Phase 31's own baseline (name-for-name cross-checked, not
just counted): `test_evalkeygen_isolated`, `test_keyswitch_stage_trace`,
`test_polyeval`, `test_stage2`, `test_boot_multi_input_small`, and the
five `test_rescale_*_n1024*` family. Zero new regressions. Rust
(debug-serial, release-parallel): 62 passed/0 failed/1 ignored, both
modes, identical to the established Phase 25-31 baseline.

**Mistake made and corrected this round, recorded rather than hidden**:
an early `CMakeLists.txt` edit was accidentally applied to the
NON-AUTHORITATIVE Windows-filesystem clone
(`C:\Users\samca\GPU-Resident-Library`) instead of the authoritative WSL
checkout. Caught immediately via `git diff --stat` showing far more
insertions than the intended edit; reverted with the exact inverse Edit
(restoring that clone's pre-existing, unrelated, already-dirty working
tree to byte-identical with its state before this round touched it,
confirmed via `grep` finding zero trace of the mistaken addition
afterward); the correct edit was then redone in the authoritative WSL
checkout. No data was lost; the Windows clone's own pre-existing
divergence (documented since Phase 25/26, never this project's concern
to reconcile) was never evaluated or touched beyond this single
caught-and-reverted accident.

**Left genuinely uncommitted and out of this round's scope** (carried
forward unchanged from Phase 31's own note): `graphify-out/GRAPH_REPORT.md`
and `graphify-out/graph.json` (pre-existing, unrelated `/graphify` run
output), `.claude/agents/`, `.claude/skills/scrum-orchestrator/`,
`.scrum-orchestrator/`, `tools/__pycache__/` (pre-existing, unrelated
tooling). Not touched, not staged, not committed this round either.

## Immediate next actions, in order (Phase 32 handoff)

1. **The single highest-value next engineering milestone, per this
   round's own evidence**: decide on the 4 unguarded caches
   (`rotate.cpp` x2, `keyswitch.cpp`, `encrypt_host_shoup.cpp`,
   `encdec.cpp` — section "Work Package A" above). Before fixing: first
   independently verify whether `keyswitch.cpp`/`encrypt_host_shoup.cpp`/
   `encdec.cpp` are genuinely cold (not reachable from the production
   Rust-bridge call path) — this round did NOT write a test proving that,
   it only cited the prior phases' own characterization. If cold,
   document that explicitly (closing the open question, matching
   `rotate.cpp`'s already-documented status); if any is reachable,
   mutex-guard it following the established Phase 28/30 pattern.
2. **Migrate a second real caller onto `DeviceBuffer`/`cuda_check`** to
   validate the primitives generalize beyond their first use
   (`resident_finance`) — the LLM KV-cache prototype's own recommendation
   (port its allocator onto `DeviceBuffer` + real `cudaMallocAsync`) is
   the most concretely evidenced candidate; the edge-AI prototype's three
   raw `cudaMalloc` buffers are the other.
3. **Sweep `resident_finance`'s crossover more finely** (between K=512
   and K=2048, where this round's coarse sweep bracketed but did not
   pinpoint the actual crossover) and consider wiring a CUDA-graph
   capture of its 2-kernel chain (same shape as `keyswitch_resident`'s
   already-proven Phase 30/31 graph path) for a warm-replay speedup
   measurement.
4. **Re-run `tests/test_boot.cpp` the moment the memory gate clears**
   (`Get-CimInstance Win32_OperatingSystem`, need >=10 GiB free) — now
   FIVE phases running without this verification.
5. **Encrypt-stage optimization** (`research/CKKS_RESIDENCY_AUDIT.md`
   section 7): the aggregation workload's own 86.9%-encrypt-dominance is
   unchanged since Phase 25; the pinned-memory variant reverted for
   single-call latency might behave differently in this workload's
   steady-state many-calls-same-key pattern — still unexplored.
6. Re-run the Windows free-RAM gate check before any of the above.

## Earlier commit (Phase 31)

Check `git log -8 --oneline` on resume. As of this writing HEAD should be
the Phase 31 commit(s), on top of Phase 30's `c43b721` (verified matching
`origin/main` exactly, 0 ahead/0 behind, at Phase 31 start). Hardware
re-confirmed unchanged: RTX 2060 Max-Q, 6144 MiB, cc7.5, driver 595.97,
CUDA 13.2, Rust/cargo 1.94.0.

**The one item every Phase 30 handoff flagged as the top carry-over —
the residual key-generation/setup-path concurrency race — is RESOLVED.**
Resumed with the fix already substantially drafted, uncommitted, by an
interrupted prior session (`tests/test_ks_graph_setup_concurrency_diagnostic.cpp`
rewritten, `CMakeLists.txt` updated to register it with `ctest`). This
round's actual contribution was **independently re-verifying every claim
before trusting it** (never assume a prior session's uncommitted work is
correct just because it reads well), completing the regression matrix,
and writing it up. Re-extracted and re-compiled the original committed
(`c43b721`) version of the diagnostic standalone and re-ran it 10 times:
confirmed genuinely, severely non-deterministic (1–6/8 threads
self-consistent, varying run to run, full-tower-scale diffs) — the
baseline really was broken, independent of any claim in the carried-over
comments. **Root cause: two CUDA stream-capture API misuses in the
REPRODUCER ITSELF, not a gpufhe defect** — (1) `cudaStreamCaptureModeGlobal`
makes ALL process-wide stream activity (including other threads'
concurrent use of the shared legacy default stream, which
`evalkeygen_host`/`ks_context_create` both default to when callers omit
a stream, as the reproducer's own `buildFixture`/`runDirect` did) subject
to a capture's exclusion rules, and the committed file checked zero CUDA
return codes, so the resulting failures were silent; (2) `cudaMalloc`/
`cudaFree` restrictions during an active capture are not fully scoped
away by `cudaStreamCaptureModeThreadLocal` alone, so unordered teardown
could invalidate another thread's in-flight capture, also silently. This
confirms, but relocates, Phase 30's own "legacy default stream 0...
plausible but unconfirmed" hypothesis: real, but in the test's own
scheduling, not inside `keygen_host`/`evalkeygen_host_sold`. **Fix,
confined entirely to the test file**: two spin-barriers excluding
alloc/free from any other thread's active capture window,
`cudaStreamCaptureModeThreadLocal` + a non-blocking stream, and explicit
`cudaError_t` checking throughout. No mutex was added to any production
function — there was no shared host state there to protect. Full
evidence trail, ownership map, and the resulting supported/unsupported
concurrency contract: `research/SETUP_PATH_CONCURRENCY_INVESTIGATION.md`.

**Verified, not assumed**: fixed reproducer 25/25 plain + 3/3
compute-sanitizer memcheck clean; old version re-run 10x for contrast,
0/10 clean. Full `ctest` suite (same memory-gate exclusions as every
phase since 28): **113/123 passing** (up from 112/122 — the newly
promoted test), the exact same 10 pre-existing documented failures as
every prior baseline, zero new regressions. Rust (debug-serial,
release-parallel): 62 passed/0 failed/1 ignored both modes, matching
Phase 25–30's end state exactly (this fix does not cross the Rust FFI
boundary). No production source file modified. No cryptographic
parameter touched. Phase 30's CUDA-graph fix and its two cache mutexes
(`g_rtab`/`upc`, `ek_dev_tab`) all confirmed intact and unchanged.

**Left genuinely uncommitted and out of this round's scope**:
`graphify-out/GRAPH_REPORT.md` and `graphify-out/graph.json` carried
pre-existing local modifications (from an earlier, unrelated `/graphify`
run) when this round started. Not touched, not committed — unrelated to
Phase 31; the next session should decide whether to commit or discard
them on their own merits.

## Immediate next actions, in order (Phase 31 handoff)

1. **Re-run `tests/test_boot.cpp` the moment the memory gate clears**
   (`Get-CimInstance Win32_OperatingSystem`, need >=10 GiB free) — this
   is now FOUR phases running without this verification (Phase 28's
   `encode_host_fft` bootstrap integration remains unexecuted at full
   scale). Windows free RAM measured 2.23 GiB this round — still NO-GO.
2. **Only after (1) passes**, wire the fused kernel into `tests/
   test_boot.cpp`'s real accumulate loop (Phase 27/28's "promote"
   verdict, never yet executed in the real file) as a SEPARATE,
   separately-measured change.
3. Consider whether `keyswitch_resident`'s measured 1.49x graph-replay
   speedup (Phase 30) is worth wiring into a real multi-call site (e.g.
   a bootstrap stage's rotation-key-switch loop) — now that the
   multi-threaded capture contract is also proven (section 14 of
   `research/SETUP_PATH_CONCURRENCY_INVESTIGATION.md`), a genuine
   optimization candidate, not attempted this round (this phase's
   mandate was the one documented correctness/concurrency carry-over,
   not a new performance initiative).
4. **Decide on `graphify-out/` artifacts** left modified by an earlier,
   unrelated `/graphify` run (see "Current commit" above) — commit,
   regenerate fresh, or discard; not evaluated this round.
5. Re-run the Windows free-RAM gate check before any of the above.

## Earlier commit (Phase 30)

Check `git log -8 --oneline` on resume. As of this writing HEAD is the
Phase 30 commit. Verified: `b09ba1b` matched `origin/main` exactly at
Phase 30 start (confirmed via `git fetch` + `git rev-parse HEAD
origin/main`). Hardware re-confirmed unchanged: RTX 2060 Max-Q, 6144 MiB,
cc7.5, driver 595.97, CUDA 13.2.

**`test_ks_graph`'s CUDA-graph-capture defect: FIXED, rigorously
validated.** Root cause: `ks_accumulate_qp`/`ks_moddown` built small
host-side root-table/modulus-selector arrays in function-local
`std::vector`s, then `cudaMemcpyAsync(..., HostToDevice, s)`'d them --
safe for direct calls (pageable H2D is synchronously staged) but a
captured graph node stores the host *pointer*, re-read at every future
replay, long after the local vector was destroyed (proven with a
minimal, gpufhe-independent reproducer: a captured graph read back the
literal bytes `0xDEADBEEF` of an unrelated vector allocated AFTER
capture ended). A first fix (move the buffers into `DeviceKSWork`,
giving them a stable address) was NECESSARY but NOT SUFFICIENT: it
exposed a second, distinct bug -- multiple (part, role) combinations
within one call shared ONE such buffer, and since replay never re-runs
the host C++ loops that fill it with each role's own value, every shared
memcpy node read whatever was written LAST during capture. Fixed with
one dedicated, persistent host buffer per (part, role), sized in
`ks_work_create`. `tests/test_ks_graph.cpp` now passes bit-exact, 5/5
runs, 0 compute-sanitizer memcheck errors. Full story, including the two
reproducers, the exact mechanism, and all validation (lifecycle, memory,
concurrency, performance): `research/KS_GRAPH_CAPTURE_INVESTIGATION.md`.

**Found and fixed two REAL, separate, previously-undocumented
concurrency bugs** while stress-testing the fix: `src/
keyswitch_resident.cpp`'s `g_rtab`/`upc` caches and `src/evalkeygen.cpp`'s
`ek_dev_tab` cache were plain unsynchronized `static std::map`s, actively
(not just latently) racing when multiple threads build independent
`KeySwitchConstants` from identical moduli. Fixed with a mutex each,
matching this project's own established pattern (Phase 28's `src/
encode_fft.cpp` fix).

**A THIRD, deeper concurrency issue remains, found but deliberately NOT
fixed this round**: even after both cache fixes, many threads
simultaneously building FRESH contexts from scratch can still show a
direct (non-graph) execution disagreeing with itself across two
back-to-back calls on the same thread -- the residual race is in the
key-generation/context-setup pipeline, not in `keyswitch_resident` or its
now-fixed graph-capture path. Per the governing directive's explicit
instruction not to let this phase become an open-ended concurrency audit
of the whole keygen pipeline, this is documented as a known limitation
with a precise, committed, standalone reproducer
(`tests/test_ks_graph_setup_concurrency_diagnostic.cpp`, deliberately NOT
registered with ctest since it is non-deterministic) and handed off.

**Performance (honest, not forced)**: at n=8192/tw=30 (bootstrap scale),
warm graph replay measured 1.49x faster than direct execution (median
4.32ms vs 6.43ms), one-time capture+instantiate+first-launch overhead
breaking even after ~4 replays. `bench/bench_ks_graph_perf.cpp`.

**New tests registered with ctest**: `test_ks_graph_lifecycle` (14
checks: capture/instantiate/first-launch/10x-repeated-replay/updated-
input-replay/multi-graph/multi-context/destruction-order/leak-check, all
100% reliable across 8+ repeated runs), `bench_ks_graph_perf`.
**Deliberately NOT registered** (non-deterministic by design):
`test_ks_graph_setup_concurrency_diagnostic`.

## Immediate next actions, in order (Phase 30 handoff)

1. **Investigate the residual key-generation/context-setup concurrency
   race** -- the single most important carry-over. Start with
   `tests/test_ks_graph_setup_concurrency_diagnostic.cpp` (already
   reproduces it) and the hypothesis noted in `research/
   KS_GRAPH_CAPTURE_INVESTIGATION.md` section 7 (legacy default stream
   `0` usage in `keygen_host`/`evalkeygen_host_sold` across concurrent
   threads is plausible but UNCONFIRMED -- verify before fixing, per the
   governing directive's explicit "do not add a mutex without proof"
   instruction).
2. **Re-run `tests/test_boot.cpp` the moment the memory gate clears**
   (`Get-CimInstance Win32_OperatingSystem`, need >=10 GiB free) -- this
   is now THREE phases running without this verification (Phase 28's
   `encode_host_fft` bootstrap integration remains unexecuted at full
   scale).
3. **Only after (2) passes**, wire the fused kernel into `tests/
   test_boot.cpp`'s real accumulate loop (Phase 27/28's "promote"
   verdict, never yet executed in the real file) as a SEPARATE,
   separately-measured change.
4. Consider whether `keyswitch_resident`'s now-measured 1.49x graph-replay
   speedup is worth wiring into a real multi-call site (e.g. a bootstrap
   stage's rotation-key-switch loop, which calls it many times per stage
   with the same context) -- a genuine, newly-available optimization
   candidate, not attempted this round (this phase's mandate was the
   correctness defect, not a new performance initiative).
5. Re-run the Windows free-RAM gate check before any of the above.

## Earlier commit (Phase 29)

Check `git log -5` on resume. As of this writing HEAD is the Phase 29
commit (verified: `ff6a253` matched `origin/main` exactly at Phase 29
start, confirmed via `git fetch` + `git rev-parse HEAD origin/main`).

**Repository/hardware verification**: all of Phase 28's reported claims
(3 commits, final `ff6a253`, clean fast-forward push) independently
re-verified correct. Hardware re-confirmed unchanged: RTX 2060 Max-Q,
6144 MiB, cc7.5, driver 595.97, CUDA 13.2, cargo/rustc 1.94.0.

**Memory gate: still NO-GO.** Windows free RAM measured 3.54 GiB then
4.41 GiB across this session (`Get-CimInstance Win32_OperatingSystem`,
`FreePhysicalMemory`) -- both below the 10 GiB gate for `tests/
test_boot.cpp`'s n=8192 bootstrap. WSL's own `free -h` showed "10Gi
available" the whole time -- exactly the trap the governing directive
warns about (WSL2's VM allocation, not true host pressure); GPU VRAM
was never the constraint (5942-5103/6144 MiB free throughout). **The
encode_host_fft bootstrap integration from Phase 28 remains UNEXECUTED
end-to-end** -- still only compile/link-verified. This is the single
most important remaining proof obligation, unchanged from Phase 28's own
handoff.

**Objective A (encoding promotion, extended verification)**: `tests/
test_encode_fft_matches_reference.cpp` and `tests/test_encode_fft_
concurrency.cpp` re-run fresh, both clean. New `tests/
test_encode_fft_scale_and_boundary.cpp` extends coverage to different
Delta scales (2^20 to 2^60) and near-numerical-boundary values the
original test didn't cover -- found and fixed TWO tolerance-design bugs
in this new test itself along the way (an over-aggressive Delta-scaled
decode tolerance, and an under-provisioned encode tolerance for a
sparse/alternating large-magnitude pattern the original test's own
formula was never exercised against), landing at 32/32 passing with a
properly-derived, not-weakened bound. Call-graph verified: all 10 of
`tests/test_boot.cpp`'s call sites genuinely call `encode_host_fft`/
`decode_host_fft` (not merely linked), confirmed by direct grep.

**Objectives C+D+E+F+G (the aggregation workload)**: found that Phase
28's `bench_private_aggregation_workload.cpp` called plain `encrypt_host`
(the Phase-11 reference), NOT this project's own production path
(`encrypt_host_shoup_pooled_barrett_batched`, what the Rust bridge
actually calls, already bit-exact verified, already measured 1.57x-2.78x
faster in isolation). New `bench/bench_aggregation_decomposed.cpp`
decomposed the workload stage-by-stage and confirmed the gain survives
end-to-end: 1.52x (144.3ms -> 95.1ms median, K=32, 10 interleaved reps),
encrypt remaining 86-91% of total in both arms. Promoted this path into
`bench_private_aggregation_workload.cpp` itself (100.27ms mean, same
error profile). New `bench/bench_aggregation_openfhe_cpu_baseline.cpp`
established the CPU comparison point gpufhe's architecture cannot supply
natively (~4x GPU speedup vs OpenFHE CPU-only, explicitly caveated as a
different implementation). New `bench/bench_aggregation_client_scaling.cpp`
confirmed VRAM stays flat across K=1..64 (pool-based reuse, no per-client
growth) while runtime scales linearly. Full writeup: `research/
AGGREGATION_WORKLOAD_PROMOTION.md`.

**Regression (Section 12, all 11 failures individually investigated,
not just re-cited)**: 109/120 passing (5 new Phase 29 tests all pass).
Every one of the 11 failures reproduced identically to Phase 28 and was
individually classified: 3 expected-negative/parameter-sweep-boundary
tests, 6 known pre-existing defects (`test_polyeval` and the
`test_rescale_n1024` family -- documented root cause, `EXPERIMENT_LEDGER.md`
#25-26), 1 environment-dependent failure (relative-path JSONL write),
and 1 **genuinely new finding**: `test_ks_graph` (CUDA-graph capture of
`keyswitch_resident` produces a fully-wrong result, not a small
numerical error) was never previously documented anywhere and had never
been revisited since its single creation commit -- confirmed NOT a
Phase 28/29 regression (neither touched anything CUDA-graph or
keyswitch-related) and confirmed NOT a general CUDA-graph environment
problem (`bench_evalmult_resident.cpp`'s own graph path works and is
measured 172x-22634x faster). Root cause not yet isolated -- a real
candidate for a dedicated future investigation. Full classification:
`research/PHASE29_REGRESSION_CLASSIFICATION.md`. Rust `cargo test` (both
debug and release): 57 passed, 0 failed, 1 ignored (documented negative
case) in each.

## Immediate next actions, in order (Phase 29 handoff)

1. **Re-run `tests/test_boot.cpp` the moment the memory gate clears**
   (`Get-CimInstance Win32_OperatingSystem`, need >=10 GiB free) -- this
   is now TWO phases running without this verification. Confirm the
   integrated `encode_host_fft` bootstrap path is correct and realizes
   its measured speedup at full n=8192 scale.
2. **Investigate `test_ks_graph`'s CUDA-graph-capture defect** -- a
   genuinely new finding this round, not yet root-caused. Requires
   careful, isolated work inside `src/keyswitch_resident.cpp`'s
   internals (this test IS the reproducer the governing directive
   requires first); do not combine with any other change.
3. **Only after (1) passes**, wire the fused kernel into `tests/
   test_boot.cpp`'s real accumulate loop (Phase 27/28's "promote"
   verdict, never yet executed in the real file) as a SEPARATE,
   separately-measured change.
4. Consider whether the aggregation workload's own encrypt-dominance
   (86% of end-to-end even on the production path) is worth a further
   pass -- `research/ENCRYPT_PROFILING.md`'s own closing note mentions a
   pinned-memory variant that was reverted for single-call latency but
   might behave differently in a steady-state many-calls-same-key pattern
   like this one; not attempted this round, a legitimate next candidate.
5. Re-run the Windows free-RAM gate check before any of the above.

## Earlier commit (Phase 28)

Check `git log -5` on resume. As of this writing HEAD is the Phase 28 commit
(on top of Phase 27's final commit, see "Earlier commit (Phase 27)" below).

**Objective A (re-verify Phase 27's fused-kernel 1.2x), re-confirmed with
full statistics, no cherry-picking**: 5 fresh interleaved runs of `bench/
bench_stage_acc_integration.cpp`, speedups 1.18x-1.22x every time, bit-exact
every time. New `bench/bench_stage_timeline.cpp` (CUDA-event-based
decomposition, mean/median/stddev/min/max) found Arm B pays a real
`cudaMalloc`/`cudaFree` cost (0.86-0.98% of its total) that Arm A structurally
cannot — meaning the reported ~1.2x is a slight UNDERSTATEMENT, not
overstatement, of the achievable production benefit.

**Objective B (the real bottleneck) corrected Phase 27's own hypothesis**:
Phase 27 expected `pt_to_eval_host` to dominate once the fused kernel is
wired in. The stage timeline's own numbers show otherwise: `encode_host`
(src/encode.cpp's O(N²) parallel host loop) is 62-75% of stage time;
`pt_to_eval_host` is only 16%. (This was caught as a self-inflicted
mistake first — the benchmark's own hardcoded printed conclusion claimed
`pt_to_eval_host` dominated, contradicting its own numbers; fixed in the
benchmark's source, not just noted.) A completely independent,
already-derived-and-tested O(N log N) FFT replacement,
`encode_host_fft`/`decode_host_fft` (`src/encode_fft.cpp`), already existed
in the tree but was unused by `tests/test_boot.cpp`. Full writeup, promotion
decision, and the concurrency-bug story (an apparent active race that
AddressSanitizer proved was actually invalid test inputs, not a real
cache defect) are in `research/ENCODE_FFT_PROMOTION.md`.

**Objective C (safe residency experiment for `pt_to_eval_host`)**:
quantified, not implemented. `bench/bench_residency_hypothesis.cpp`
(standalone, touches zero production files) measured a hard ceiling of
~2.3% of stage time for fully eliminating `pt_to_eval_host`'s own D2H plus
the stage's redundant plaintext re-upload — an order of magnitude below the
encode swap's 2.5x, and not worth the implementation risk this round. Full
writeup: `research/BOOTSTRAP_GPU_RESIDENCY_AUDIT.md` section 8.

**Objective D (promotion decisions)**: the fused kernel's Phase 27 "promote"
verdict is reconfirmed, but deliberately NOT wired into `tests/
test_boot.cpp`'s accumulate loop this round (see item 1 below for why).
`encode_host_fft`/`decode_host_fft` ARE promoted and wired into all 10 of
`tests/test_boot.cpp`'s call sites — compiles and links cleanly, but **has
NOT been executed end-to-end**: the memory gate for running this file
(n=8192, needs ≥10 GiB free Windows RAM) was measured at 2.40 GiB free
(`Get-CimInstance Win32_OperatingSystem`, `FreePhysicalMemory`=2,520,264 KB)
— a NO-GO, consistent with this project's recent history. **This is the
single most important follow-up** (item 1 below).

**Objective E (first real consumer-GPU workload)**: `bench/
bench_private_aggregation_workload.cpp` — privacy-preserving aggregation of
32 client-held values via `keygen_host`/`encode_host_fft`/`encrypt_host`/
`ct_add_ct_host`/`decrypt_host`/`decode_host_fft` only (no private kernels,
no relin/rotation/rescale needed since a pure ciphertext sum consumes no
depth — a weighted-sum/EvalMult variant was deliberately avoided because it
would need the rescale pathway `tests/test_rescale_n1024.cpp` documents a
real, unresolved bug in). 10 repetitions, fresh randomized client values and
fresh per-client encryption randomness each time. Registered with ctest.

**Regression**: ran the full suite (115 registered tests, excluding
`test_boot`/`test_boot_a`/`test_boot_n131072`/`test_hoist_allstages`/
`test_n131072_gpu_lifecycle` per the memory gate) — 104/115 passed. The 11
failures were individually checked against `NEXT_SESSION.md`/
`MASTER_STATE.md`/`EXPERIMENT_LEDGER.md` and confirmed PRE-EXISTING and
unrelated to this round's changes (none of the failing tests reference
`encode_fft`/`encode_host_fft`; the `test_rescale_n1024*` family and
`test_polyeval` are the long-documented native-rescale-at-small-n bug;
`test_evalkeygen_isolated`/`test_keyswitch_stage_trace` are deliberately
invalid-configuration tests; `test_boot_multi_input_small`'s one failing
check is a relative-path JSONL write that resolves against ctest's `build/`
cwd, where `research/benchmarks/` doesn't exist — pre-existing, unrelated).

## Immediate next actions, in order

1. **Run `tests/test_boot.cpp` (with the `encode_host_fft` swap already
   applied) the moment the memory gate clears** (`Get-CimInstance
   Win32_OperatingSystem`, need ≥10 GiB free). This is the only way to
   confirm the integrated bootstrap path is correct and actually realizes
   the measured 2.5x stage-level speedup at full scale, not just in
   isolation. Do this BEFORE anything else below — it is this round's one
   unverified production change.
2. **Only after (1) passes**, wire the fused kernel into `tests/
   test_boot.cpp`'s real accumulate loop (Phase 27's and Phase 28's own
   verdict: promote). Keep it a SEPARATE, separately-measured change from
   (1) — do not land both in the same untested commit.
3. **The `pt_to_eval_host` residency idea is now a quantified, low-priority
   backlog item** (~2.3% ceiling) — worth a proper design (an ownership
   contract for exposing `src/rotate.cpp`'s `thread_local` scratch pointer
   across a function boundary) only after (1) and (2) land, not before.
4. **Extend Objective E's workload** once rescale's n=1024 bug
   (`research/EXPERIMENT_LEDGER.md` entries #25-26, `test_rescale_n1024.cpp`)
   is actually root-caused — a weighted-sum variant (EvalMult by a plaintext
   weight vector, not just addition) is the natural next workload and only
   avoided this round for that reason.
5. Re-run the Windows free-RAM gate check before any of the above — it
   changes session to session and was NOT re-checked at the moment this
   file was written versus when work actually resumes.

## Earlier commit (Phase 27)

Check `git log -5` on resume. As of this writing HEAD is the final Phase 27
commit on top of Phase 26's `8cc1e12`.

**Verification (Mission 1): Phase 26's experiment reproduced, with one real
discrepancy found and quantified.** `bench/bench_ks_context_resident_ab.cpp`
rebuilt and rerun: 60.000 MiB/context, bit-exact, 1.6871x speedup (vs Phase
26's 1.6621x — same order, expected machine spread) all reproduced exactly.
**But**: Arm A's measured interval charges for 48 `ks_context_create` AND 48
matching `ks_context_destroy`; Arm B's charges for only 8 creates — the 8
matching destroys happen in the cache's destructor, AFTER the timer stops.
New `bench/probe_create_destroy_cost.cpp` measured `ks_context_destroy`
directly: **36% of create's cost** (0.0080s vs 0.0219s). Corrected, the honest
speedup is closer to **~1.6x**, not 1.69x — real, just not quite as large as
reported.

**THE CENTRAL FINDING (Mission 2): Phase 26's 1.66x does not generalize to
the real bootstrap.** New `research/benchmarks/extract_real_key_sequence.cpp`
replicates `tests/test_boot.cpp`'s own diagonal-finding logic (pure CPU
arithmetic, no CUDA, no bootstrap execution) to produce the ACTUAL ordered
sequence of rotation-key requests across a full bootstrap:

```
Total diagonal rotation accesses: 181
Distinct (k, tw) pairs [the REAL device-context working set]: 151
Distinct k alone [KROT's host-side working set]: 76
```

**76 independently cross-validates against a previously INSTRUMENTED
measurement already in `EXPERIMENT_LEDGER.md`** ("evalkeygen calls total=77 =
76+1 relin key") — strong evidence the extraction is faithful. **The "~77
entries" Phase 26's limitations section cited for sizing a device cache was
the WRONG quantity** (KROT's k-only count) — the real device-context working
set is 151, not 76.

**Of 181 total accesses, only 30 are repeats, and they ALL come from exactly
one place**: S2C's stage at `g==LST-1` processes the same 30 rotation offsets
TWICE (real branch, then imaginary branch). Every other stage (C2S's 3; S2C's
other 2) has **zero internal reuse** — each diagonal's key is used exactly
once, never needed again at that tower count anywhere else in the bootstrap.
An LRU hit-rate simulation against this REAL sequence confirms: **0% hit rate
for any cap < 30; a FLAT 16.6% ceiling for any cap ≥ 32** (confirmed again by
an actual GPU capacity sweep, `bench/bench_realistic_ks_cache.cpp`'s Mission 6
section: cap=4/8/16 all measure 0 hits; cap=30 measures 50% hit rate on that
one stage's own 60 accesses). Scaling a cache past ~32 entries buys nothing.

Also found: `atLevel` (the real mechanism behind every reduced-tw key) copies
`av`/`bv` **unchanged, full-size** from the full key — only the number of
parts shrinks with `tw`, not each array. So the full 151-entry working set's
memory cost, **calculated (never allocated, per the directive's explicit
instruction)**, is **~5.62 GiB** — would not safely coexist with everything
else this project allocates on the 6144 MiB card, and would be pointless
regardless (the hit-rate ceiling is identical at cap=32 and cap=151).

**Result: a correctly-scoped cache (~30-32 entries, ~600 MiB, matching the
ONE stage that benefits) gives a real, bit-exact 1.5-1.8x on that stage
alone** (`bench/bench_realistic_ks_cache.cpp`, using the REAL atLevel-style
key construction and the REAL double-pass access pattern, not a synthetic
round-robin). **Full detail, the complete memory model, and the architecture
built on this evidence: `research/KS_RESIDENT_CACHE_DESIGN.md`.**

**Mission 3/4: `include/ks_resident_cache.h` built.** `KsResidentCache`,
on top of the project's proven `ScopedKSContext` RAII primitive
(`include/ks_raii.h`). Keyed `{k, tw}` (never `k` alone — entry #20's bug
class). Bounded by BOTH entry count (default 32) and resident bytes (default
768 MiB). Fails CLOSED on VRAM pressure (`KsResidentCacheOOM`, thrown BEFORE
any allocation attempt, closing a gap `ks_raii.h`'s own docs flag in the
underlying C API: neither `ks_context_create` nor `ks_work_create` reports
allocation failure today). Enforces single-threaded access by **aborting the
process** on a cross-thread call (not just documenting the limitation).
Closes an in-flight safety gap found during this round's own design review:
evicting a context while a kernel against its buffers is still in flight
would race `cudaFree` against that kernel — closed by recording a stream per
entry and `cudaStreamSynchronize`-ing it before any destroy. `src/
keyswitch_resident.cpp` was NOT touched — the whole cache lives in this one
new header.

**Mission 5: `tests/test_ks_resident_cache.cpp` — 34/34 checks, confirmed
stable across 5 repeated runs from 2 different working directories.** Three
real methodology bugs were found and fixed WHILE WRITING these tests (kept
documented in both the header and the ledger, not papered over):
1. A bare `fork()` from an already-CUDA-active process is unsafe (CUDA
   explicitly does not support it) — corrupted the child's CUDA state,
   produced a `SIGSEGV`. Fixed with a separate helper EXECUTABLE (`tests/
   ks_cache_thread_violation_helper.cpp`) invoked via `system()`
   (fork+exec, safe).
2. `system()`'s signal encoding: a child that dies by signal is reported by
   the wrapping shell as a NORMAL exit with status 128+signal, not
   `WIFSIGNALED` — an earlier check misread a confirmed-working abort as a
   failure.
3. `stdout` is fully buffered when piped — a crash discarded every
   already-printed `[PASS]` line, making a working test look like it
   produced no output at all. Fixed with `setvbuf(..., _IOLBF, ...)`.
4. **A genuine intermittent failure**, caught exactly as Mission 5
   instructs (running tests from a different working directory): the
   thread-violation helper was invoked via a bare relative path, silently
   resolving to nothing when run from the repo root instead of `build2/`.
   Fixed by resolving the helper's path via `/proc/self/exe`.

**Mission 6: realistic capacity sweep.** `bench/bench_realistic_ks_cache.cpp`
tests cap=4/8/16/30 against the REAL access pattern (not Phase 26's
round-robin): 0% hit rate below the real stage width, 50% (of that one
stage) at cap=30, flat beyond. Confirms section 2.3's CPU-only simulation
with an actual GPU run.

**Mission 7: integration investigated, NOT performed.** Given the evidence
above, general-purpose production integration is **not recommended** — see
`research/KS_RESIDENT_CACHE_DESIGN.md` section 6 for the full reasoning and
the narrowly-scoped alternative (a cache built at the start of the one S2C
stage that benefits, torn down at its end) that WOULD be justified if ever
pursued.

**Mission 8: fused pt-mul-accumulate kernel, end-to-end STAGE-level A/B.**
New `bench/bench_stage_acc_integration.cpp` measures the ACTUAL stage-level
effect of wiring `LaunchPtMulAcc` in (Phase 26 only measured the isolated
kernel, ~7.6x). Simulates one real 16-diagonal stage: current CPU
multiply+add+per-diagonal-nothing-extra vs fused-kernel+ONE-final-download
(pt_to_eval_host's own round trip deliberately unchanged in both arms —
separate, larger optimization target, out of scope here). **Result:
bit-exact (0/491520 limbs), real end-to-end stage speedup of 1.20x** — much
smaller than the isolated kernel's 7.6x once the unchanged round trip
dominates, exactly the correction Mission 8 warns to make. Both of Mission
8's promotion conditions are met (correctness preserved, end-to-end latency
improves) — this is a small, low-risk, evidence-backed candidate, UNLIKE the
cache. Not wired into `tests/test_boot.cpp` this round (kept as a separate,
independently-verifiable change from the cache work, per the directive's own
instruction not to combine the two experiments). Full detail: `research/
FUSED_KERNEL_STAGE_INTEGRATION.md`.

**Mission 9: bootstrap gate.** Windows free RAM measured 1.07-2.81 GiB this
session against the 10/6 GiB floors. Both preflight checks NO-GO, logged.
Full bootstrap NOT executed; no claim of correctness made; gate not weakened.

**Mission 10: regression.** All four Rust combinations (debug/release x
serial/parallel): 62 passed/0 failed/1 ignored, identical to Phase 25/26's
end state. C++ targeted: 14/14 pass (122-569 MB peak RSS), confirmed stable
from both the repo root and `build2/`, 5 repeated runs each.

## Immediate next actions, in order

1. **If ever pursuing the cache**: scope it to exactly the ONE S2C stage
   that benefits (built at that stage's start, torn down at its end, sized
   ~30-32 entries / ~768 MiB), using `include/ks_resident_cache.h` as-is.
   Do NOT build a general-purpose always-on cache — the evidence this round
   built says plainly that it would not help the other five passes and
   would add real memory/complexity risk for a narrow gain.
2. **The fused-kernel wiring (Mission 8) is the better next move** if
   pursuing EITHER optimization: smaller, lower-risk, already end-to-end
   verified at the stage level. The actual edit would be in `tests/
   test_boot.cpp`'s accumulate call sites (not `src/keyswitch_resident.cpp`,
   so the "no internals without a reproducer" rule does not block it).
3. **Reconsider `pt_to_eval_host`'s own round trip** (the residency audit's
   O3) — Mission 8's stage-level result shows THIS, not the accumulate step,
   now dominates a stage's cost once the fused kernel is in place. This is
   the next real bottleneck to investigate, not previously quantified this
   precisely.
4. **Run the bootstrap the moment the gate clears** — same sequence as
   Phase 25/26's handoff (see `research/benchmarks/bootstrap_preflight.ps1`).

## Earlier commit (Phase 26)


Check `git log -5` on resume. As of this writing HEAD is the final Phase 26
commit on top of Phase 25's `1d73518`. Phase 26's commits, in order:

- Windows-clone reconciliation report (no code changes)
- Mission 2: extended differential test for the fused pt-mul-accumulate kernel
- Mission 4: resident eval-key context cache A/B + Mission 3 writeup
- Documentation (this file, the experiment ledger)

**Recovery.** Session resumed in Git Bash on Windows, NOT inside WSL. WSL
`Ubuntu` was Stopped and started on the first ordinary attempt. Both
checkouts verified unchanged since Phase 25: WSL HEAD `1d73518` =
`origin/main` exactly; Windows clone HEAD `f425cdd`, same 65 dirty entries.

**THE CENTRAL RECONCILIATION FINDING: the Windows clone contains ZERO
genuinely novel, unported code.** `git merge-base --is-ancestor f425cdd HEAD`
returns true — the clone's committed history is 82 commits BEHIND main, not
a divergent branch. Both commits the Phase 26 directive named as prototypes
("Fused pt-mul-accumulate kernel", "Device automorphism kernel") are
confirmed ancestors of main's current HEAD — already fully merged, long ago.
Every one of the clone's 65 dirty working-tree entries was then checked
content-by-content against current main (controlling for CRLF-vs-LF noise,
which inflated several apparent diffs from 0 real lines to hundreds): 29 of
33 comparable untracked files are byte-identical; the other 4 and every
tracked-modified file differ for exactly one of two reasons — predating the
`P>=Q_part_max` hybrid key-switch validation fix (Phase 10, entries #42/#43),
or predating Phase 25's own `thread_local` concurrency fix. Full detail and
the reconciliation matrix: `research/PHASE26_RECONCILIATION.md`. **Nothing
was ported** — there was nothing to port. No recovery branch was needed. The
Windows clone itself was not modified.

**This reframed Missions 2-3 from "port a prototype" to "validate
already-merged code that may not be wired into the bootstrap's hot path"** —
exactly Phase 25's own residency-audit finding about `LaunchPtMulAcc`
(`kernels/stage_acc.cu`): the fused kernel exists, has its own bit-exact
test, but `tests/test_boot.cpp`'s actual per-diagonal loop still calls
`ct_mul_pt_host`/`ct_add_ct_host` on the CPU.

**Mission 2 — extended differential test for `LaunchPtMulAcc`.** New
`tests/test_stage_acc_differential.cpp`, 26/26 checks pass (142 MB peak RSS,
~1s wall): boundary coefficient values (0, 1, q-1, q-2, (q-1)/2, (q+1)/2,
q/2) mixed with random; 5 (n, towers) pairs including the real n=8192/tw=30
bootstrap config; 16-sequential-diagonal accumulation on one stream with one
sync at the end (mirrors the real BSGS stage structure); 3 full repeated
sweeps with distinct seeds; 6 independent concurrent accumulation chains on
disjoint streams/buffers with **zero cross-contamination confirmed**. Noted
explicitly: `LaunchPtMulAcc` holds no static/global device state at all
(confirmed by reading `kernels/stage_acc.cu`) — a materially safer design
than `xf_multi`'s `scratch()` buffer, the actual root cause of Phase 25's
concurrency bug. Isolated-op benchmark: fused kernel-only 0.69ms vs CPU host
pair 5.26ms (~7.6x on the compute alone) at n=8192/tw=30 — explicitly scoped
as an isolated-op figure, not a bootstrap number.

**Mission 4 — eval-key transfer estimate independently verified, and a
bit-exact resident-context experiment.** New `bench/bench_ks_context_resident_ab.cpp`
measures (not re-derives) the per-`ks_context_create`-call eval-key upload
from a LIVE `KeySwitchConstants` object: **60.000 MiB exactly**, matching
Phase 25's static estimate. Confirmed by full code reading that
`ks_context_destroy` frees ONLY `d_av`/`d_bv`/`d_keyRow` — everything else a
context holds is already process-lifetime shared state keyed on chain
SHAPE, not on the rotation key — so the eval key is the ENTIRE marginal cost
of reusing an already-built key. A/B experiment: 48 simulated diagonal calls
drawing from 8 distinct rotation keys (mirroring KROT's real measured reuse
pattern — Phase 24's ledger: ~77 entries reused with zero evictions), against
a bounded LRU cache of `DeviceKSContext` keyed **(k, tw)** — never `k` alone,
per `EXPERIMENT_LEDGER.md` entry #20's documented bug class. Bit-exact
checked PER CALL before any timing (0/48 differ). Result: **1.66x measured
speedup**, **83.3% measured H2D traffic reduction (2400 MiB avoided)** — not
estimated: byte counts come from actual cache-miss counts times the measured
60 MiB figure. `src/keyswitch_resident.cpp` was NOT touched — this is a
standalone experiment offered as evidence for a future integration decision,
not an integration. Explicit limitations stated in the file and the report:
single-threaded only (no lock; untested under concurrency), cache sized to
exactly fit this run's working set (a production cap needs its own study
against KROT's real ~77-entry reuse and the 6144 MiB card), key-switch-adjacent
scope only (not end-to-end). Full detail:
`research/EVAL_KEY_TRANSFER_VERIFICATION.md`.

**Mission 3 — device automorphism / stage residency, sharpened beyond
Phase 25's file-level audit.** `tests/test_rot_resident` re-run fresh this
round (0/491520 mismatches, bit-exact, 4.00x vs host-orchestrated, 253 MB
peak RSS) — the existing standalone correctness test already meets the
directive's bar; it was not rebuilt. New: a FUNCTION-level call-site tally
(not file-level) comparing `rotate_ct_resident` against `rotate_ct_host`
directly, by reading both bodies and `keyswitch_core_resident`'s full
166-line body. **The "resident" path has nearly 3x the H2D call sites (33 vs
12) and more than 3x the malloc/free call sites (47 vs 14) of the "host"
path it replaces** — every one of them from `ks_context_create` re-uploading
the eval key fresh every call. What it genuinely wins: D2H count (3 vs 10)
and sync count (2 vs 8), because the ciphertext and the keyswitch kernels
themselves ARE genuinely resident (confirmed zero transfers/allocations/syncs
inside `automorphism_eval_fast_device` and `keyswitch_resident` by direct
reading). The phase breakdown confirms this isn't academic:
**`ks_context_create` + its matching `destroy` are 63% of `rotate_ct_resident`'s
own total time** at the bootstrap's parameters — more than the automorphism
and actual keyswitch kernels combined. So "GPU-resident" is accurate for the
ciphertext and kernel execution, and false for the eval key — stated
precisely rather than left implicit. Full detail:
`research/MISSION3_DEVICE_AUTOMORPHISM_RESIDENCY.md`.

**Mission 5 — bootstrap memory gate.** Windows free RAM measured 1.75-1.94
GiB for the entire session against the 10 GiB (n=8192) and 6 GiB (n=131072)
floors. Both preflight checks returned NO-GO, logged to
`research/benchmarks/bootstrap_preflight_log.jsonl`. The full bootstrap was
NOT executed. No claim of bootstrap runtime correctness is made. The gate
was not weakened in any way.

**Mission 6 — security/correctness guardrails.** No cryptographic parameter
touched: h=64, sigma=3.19, the modulus chain, `KROT_CAP`, `KSC_CAP`, and the
centralized `P>=Q_part_max` validation are all untouched (and, per the
reconciliation finding, the Windows clone's stale `numPart=2` default was
explicitly REJECTED as a port target precisely because adopting it would
have reintroduced a known-invalid parameter). No production source file was
modified this round — `src/keyswitch_resident.cpp` and `src/rotate.cpp`
were read extensively but not edited.

**Mission 7 — regression.** C++ targeted suite: 13/13 pass (123-569 MB peak
RSS each) — `test_stage_acc`, `test_stage_acc_differential` (new),
`test_rotate_concurrent_cache`, `test_rot_resident`, `test_ks_resident`,
`test_hoist_single`, `test_hoist_multi`, `test_hoist_levelreduced`,
`test_boot_multi_input_small`, `test_encdec`, `test_encode`, `test_evalmod`,
`test_cheb`. Rust: **all four combinations** (debug/release x
serial/parallel) give **62 passed / 0 failed / 1 ignored**, identical to
Phase 25's end state — confirming Phase 25's `thread_local` concurrency fix
remains solid under this round's work. Preflight mocked unit tests: 20/20
pass.

## Immediate next actions, in order

1. **Decide on the O1 integration** (cache `DeviceKSContext` per `(k, tw)`
   inside `src/keyswitch_resident.cpp`, or inside `tests/test_boot.cpp`'s own
   `stageRotDiag`). Mission 4's experiment is the evidence (1.66x,
   83.3% traffic reduction, bit-exact), but it is single-threaded and sized
   to a toy working set. Before integrating: (a) design the cache's
   concurrency story if `test_boot.cpp`'s single-threaded bootstrap is truly
   the only caller (likely fine as-is) or add a lock/test if not; (b) size
   the production cap against KROT's real ~77-entry reuse pattern and the
   6144 MiB card (77 x 60 MiB = ~4.6 GiB would NOT fit alongside everything
   else — a much smaller cap, LRU-evicted like KROT_CAP itself, is needed in
   practice); (c) build the integration as a small, reviewable change with a
   bit-exact test at each step, per the directive's own standing instruction.

2. **Decide on wiring `LaunchPtMulAcc` into `test_boot.cpp`'s hot loop.**
   Mission 2 proved it bit-exact across sizes/boundaries/multi-diagonal/
   concurrency and ~7.6x faster in isolation. It is NOT currently used there.
   Wiring it in would need: the ciphertext accumulator kept device-resident
   across the diagonal loop (it currently round-trips via `pt_to_eval_host`
   and `ct_mul_pt_host`/`ct_add_ct_host`), which is a larger structural change
   than swapping one call — see `research/BOOTSTRAP_GPU_RESIDENCY_AUDIT.md`
   section 4's O2/O3 for the shape of that work.

3. **Run the bootstrap the moment the gate clears.** Same exact sequence as
   Phase 25's handoff:
   ```
   powershell.exe -File research/benchmarks/bootstrap_preflight.ps1
   # only on GO (exit 0):
   wsl -d Ubuntu -- bash -lc 'cd ~/GPU-Resident-Library && \
     cmake --build build2 --target test_boot_multi_input -j4 && \
     /usr/bin/time -v ./build2/test_boot_multi_input'
   ```

4. **The latent `std::map` race (Phase 25's falsified-but-recorded
   hypothesis) remains unfixed, deliberately** — `keyswitch_resident.cpp:30`/
   `:106`, `rotate.cpp:29`/`:130`. No change this round.

## Environment facts (unchanged from Phase 25, re-confirmed)

- Build with a Linux-only `PATH`
  (`export PATH=/usr/local/cuda-13.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`)
  — `cmake` configure under the full WSL login `PATH` is pathologically slow
  (9p filesystem walk over ~50 `/mnt/c/...` entries).
- WSL's own `free -h` is NOT the memory gate — it reports a ceiling
  (`.wslconfig`'s `memory=11GB`), not actual Windows host availability. Only
  `Get-CimInstance Win32_OperatingSystem` is trustworthy.
- RTX 2060 Max-Q, 6144 MiB VRAM, driver 595.97, CUDA 13.2, 16 CPU threads,
  Windows 11 host with 15.42 GiB total RAM. ~30% run-to-run timing spread on
  this machine — always use median-of-many-trials with alternating arm order
  for any A/B.

## Earlier commit (Phase 25)


Check `git log -5` on resume. As of this writing HEAD is `1eae71a`, the third
of three Phase 25 commits on top of Phase 24's `68e9f65`:

- `6bbb8b6` Phase 25 (1/n) — recovery + bootstrap validation harness
- `e2697e9` Phase 25 (2/n) — **concurrency defect found and fixed**
- `1eae71a` Phase 25 (3/n) — hoisting A/B + residency audit

**Recovery.** The session resumed in Git Bash on Windows, NOT inside WSL. WSL
`Ubuntu` was Stopped and started on the first ordinary attempt — no
`HCS_E_CONNECTION_TIMEOUT` recurrence, no diagnostics, no config change. The
authoritative checkout `/home/samfrazerdutton/GPU-Resident-Library` was intact
at `68e9f65` = `origin/main`, 0 ahead / 0 behind, zero tracked or staged
modifications. Full detail: `research/PHASE25_RECOVERY.md`.

**The shutdown had caught the previous session mid-Phase-25.** Three preflight
PowerShell files and a 646-line `tests/test_boot_multi_input.cpp` survived as
untracked work (Nov 5 21:23–21:29). All recovered, plus five gaps closed: the
driver had never actually been compiled (not registered in CMakeLists despite
its header claiming "BUILD-VERIFIED ONLY"); no machine-readable output; no
per-stage fields; the promised `test_boot_multi_input_small.cpp` did not exist;
and the preflight took `-BootstrapConfig` as free text it never parsed, so it
could only apply one fixed threshold.

**The headline result is a production bug nobody was looking for.**
`src/rotate.cpp`'s `scratch()` handed THE SAME process-global device buffer to
every caller (as did the five `xf_multi` metadata arrays), so two threads in
`xf_multi` uploaded into the same device memory and read back each other's
data — **silently wrong ciphertexts, no crash, no diagnostic.** Reached by
`pt_to_eval_host` and `automorphism_eval_host` (hence `rotate_ct_host`); NOT
reached by `rotate_ct_resident`, which allocates its own per-call scratch. Fixed
with `thread_local`, which preserves the file's allocate-once/grow/never-free
doctrine per thread and is byte-identical single-threaded.

Measured with the new `tests/test_rotate_concurrent_cache.cpp` (9 concurrent
fixtures, distinct (n, towers)):

```
BEFORE  phase B control (rotate_ct_resident)  PASS 6/6 runs
        phase C (pt_to_eval_host)             FAIL 9/9, 72,704 wrong limbs
        phase D (rotate_ct_host)              FAIL 9/9, 182,272 wrong limbs
AFTER   all three phases                      PASS 6/6 runs
```

**Phase 24's diagnosis was half right, and the correction matters.** Phase 24
said both Rust failures were test-order artifacts of
`budget_exhaustion_falls_back_safely_and_stays_correct` and that "skipping that
one test makes everything else pass unconditionally". True only in SERIAL mode.
In the default parallel mode the failures were the `rotation::` tests, the
failing SET varied run to run (3, then 2, then 4 — the signature of a race, not
of ordering), and skipping the budget test did nothing for them. There were two
independent defects.

**Result — both modes now fully green, nothing skipped:**

| | before | after |
|---|---|---|
| `cargo test --release -- --test-threads=1` | 55 passed; 2 failed; 1 ignored | **62 passed; 0 failed; 1 ignored** |
| `cargo test --release` (parallel) | 55 passed; 2–3 failed (varying) | **62 passed; 0 failed; 1 ignored** |

Phase 24's best was 56 passed/0 failed *with the budget test skipped*, in one
mode only.

**Defect 2 (test isolation), fixed properly.** The 16 MiB
`kRegisteredBudgetBytes` is tracked by `g_registeredBytesUsed`, which only
increases and has no release path (by design), and the pool never evicts. So the
budget-exhaustion test permanently consumed the budget for everything sharing
its process. Moved to
`rust/crates/ckks-cuda/tests/registered_budget_exhaustion.rs`, where Cargo gives
it its own process. NOT skipped, NOT weakened — every original assertion intact,
and in a fresh process its ~23 MiB of requests still exceed the 16 MiB cap, so
the budget still genuinely binds. No production change: in particular **no
test-only budget-reset hook was added**, which would have put a deregistration
path into production purely to serve a test. Regression test
`no_unit_test_may_exhaust_the_shared_registered_budget` encodes the invariant
(a 48 KiB fresh key must still register inside the unit-test process) and its
failure message names the cause.

**Rotation hoisting: the "why is it disabled" question is answered.** It is an
amortisation effect, and at one rotation per decompose it is a net loss. New
`bench/bench_hoist_ab.cpp`, at n=8192/sizeQ=30/sizeP=2/numPart=15:

```
rotations sharing one decompose:   1       2       3       4       6
speedup upper bound, 9 trials:     0.59x   1.18x   1.43x   1.69x   1.90x
speedup upper bound, 21 trials:    0.76x   1.18x   1.43x   1.57x   1.89x
```

Bit-exact both runs (0 of 2,949,120 limbs differ across 6 keys; the benchmark
refuses to print timings unless that passes first). Correctness was never the
problem. **Keep `HOIST_ROTATION_ENABLED = false`**; if ever enabled, scope it to
a BSGS stage loop (~16 diagonals share one input half — the N≥4 regime), not
globally. Scope caveat stated everywhere it appears: decompose+accumulate+moddown
only, so it is an UPPER BOUND on rotation speedup and **NOT** a bootstrap
speedup. Full writeup: `research/ROTATION_HOISTING_INVESTIGATION.md`.

**Residency audit: the dominant transfer is the KEY, not the ciphertext.**
`ks_context_create` runs INSIDE the per-diagonal loop and re-uploads the entire
eval key every time — `upv(K.av[p])`/`upv(K.bv[p])` per part, each `sizeQlP*n`.
At n=8192/sizeQlP=32/numPart=15 that is **60 MiB H2D per diagonal** plus 31
cudaMalloc/cudaFree pairs, against only ~9.4 MiB of ciphertext traffic for the
same diagonal — about **6×**. `KROT`/`KSC` do not help; they are host-side
caches (`KSC` is explicitly a *view* cache). `src/rotate.cpp`'s own comment
anticipated it ("if that dominates, cache contexts across calls next") and by
static byte count it does. Six ranked opportunities with risk/test/fallback
each: `research/BOOTSTRAP_GPU_RESIDENCY_AUDIT.md`.

## Blocked this round (Phase 25) — and what is NOT blocked

**The full n=8192 bootstrap was NOT executed.** Windows free RAM measured
2.35–3.19 GiB for the entire session against the 10 GiB floor;
`research/benchmarks/bootstrap_preflight.ps1` returned NO-GO for both n=8192 and
n=131072, logged to `research/benchmarks/bootstrap_preflight_log.jsonl`. **No
claim of bootstrap runtime correctness, precision or repeatability is made.**
`tests/test_boot_multi_input.cpp` is build-verified only (it now genuinely
compiles and links, for the first time). Also blocked by the same gate:
`test_hoist_allstages` (~8 GB), `test_boot_n131072`, and
`research/benchmarks/randomized_ab.sh` (~8 GB × 20 runs ≈ 37 min).

**What turned out NOT to be blocked**, and this is worth internalising for next
time: `test_hoist_single` (275 MB), `test_hoist_multi` (569 MB),
`test_hoist_levelreduced` (260 MB) and `test_rot_resident` at full n=8192/tw=30
(248 MB) all run comfortably. Several rounds treated "n=8192" as a proxy for
"8 GB"; it is not. Only the full bootstrap and the all-stages C2S pass are in
that band. Check peak RSS with `/usr/bin/time -v` before assuming a test is
gated.

## Immediate next actions, in order

1. **Reconcile the Windows clone before any new residency work.**
   `C:\Users\samca\GPU-Resident-Library` (HEAD `f425cdd`, 65 dirty entries)
   carries commit subjects including "Fused pt-mul-accumulate kernel (acc +=
   rot_ct * diag): bit-exact vs host ct_mul_pt + ct_add_ct" — which **is** the
   residency audit's O2 — and "Device automorphism kernel ... Unblocks
   stage-resident transforms", which is O3's enabler. Nothing was read out of
   that clone this round beyond `git log`/`git status`, and nothing was copied in
   either direction. Diff `f425cdd` + its working tree against `origin/main` and
   decide what is genuinely new **before** re-implementing a fused accumulate
   kernel that may already exist and be verified. Owner's call.

2. **Residency audit O1: cache `DeviceKSContext` per `(k, tw)`.** Largest static
   target (~60 MiB H2D per diagonal, ~6 GiB per bootstrap, ~3,100 malloc/free
   pairs). **Key it on `(k, tw)`, never `k` alone** — `EXPERIMENT_LEDGER.md`
   entry #20 records a real correctness bug from building a key at a reduced
   tower count, and a `k`-only cache reproduces it exactly. Entries must come
   from the trusted `rotFull` + `atLevel` path. Needs a bounded LRU (60 MiB per
   live entry against a 6144 MiB card). Isolated test: extend
   `tests/test_rot_resident.cpp` with a repeated-rotation loop asserting
   bit-exactness and measuring VRAM high-water; `bench/bench_hoist_ab.cpp` is
   the ready-made template.

3. **Run the bootstrap the moment the gate clears.** Exact sequence:
   ```
   powershell.exe -File research/benchmarks/bootstrap_preflight.ps1
   # only on GO (exit 0):
   wsl -d Ubuntu -- bash -lc 'cd ~/GPU-Resident-Library && \
     cmake --build build2 --target test_boot_multi_input -j4 && \
     /usr/bin/time -v ./build2/test_boot_multi_input'
   ```
   The driver emits JSONL (params + per-case + per-stage + summary) to
   `research/benchmarks/`, so the run is comparable afterwards without trusting
   a commit message. Do NOT relaunch after a NO-GO without a fresh preflight.

4. **Optionally fix the latent map race.** `keyswitch_resident.cpp:30`/`:106`,
   `rotate.cpp:29`/`:130` are `find()`-then-`emplace()` on `static std::map`
   with **no mutex in either file**. Phase B of
   `tests/test_rotate_concurrent_cache.cpp` drives exactly that path and passed
   6/6, so it is latent, not active — left unfixed deliberately because there is
   no failing test and the directive forbids changing
   `keyswitch_resident.cpp` internals without one. Phase B stays as the control:
   **if it ever fails, reopen this.**

## Environment facts worth not rediscovering

- **`cmake -S . -B build2` under the WSL login shell took over 13 minutes with
  no progress.** `find_package` walks the ~50 `/mnt/c/...` entries that shell
  puts on `PATH`, over the 9p filesystem. With a Linux-only PATH the identical
  configure finishes in **2.1 seconds**. Always build with:
  ```
  export PATH=/usr/local/cuda-13.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  ```
  (add `~/.cargo/bin` for Rust). A bare `wsl -d Ubuntu -- bash script.sh` has
  neither `nvcc` nor `cargo` on `PATH` at all — the login shell is needed for
  those, or set PATH explicitly as above.
- WSL reports **10 GB total / 10 GB free** at the very moment the Windows host
  has 2.35 GiB actually free. `memory=11GB` in `.wslconfig` is a ceiling, not a
  reservation. This is why the gate reads `Get-CimInstance
  Win32_OperatingSystem` and nothing else.
- The preflight's `*RamGB` values are really **GiB** (`FreePhysicalMemory` is in
  KB and the harness divides by 1048576). Documented and relabelled rather than
  retuned — GiB makes the gate slightly stricter, the safe direction.
- RTX 2060 Max-Q, 6144 MiB VRAM, driver 595.97, CUDA 13.2 (five toolkits
  installed; `/usr/local/cuda` → 13.2), 16 CPU threads, Windows 11 host with
  15.42 GiB total RAM. GPU clock throttling under WSL2 and thermal drift over
  long sessions are real and measured — hence median-of-many-trials with
  **alternating arm order** for any A/B on this machine.

## Earlier commit (Phase 24)


Check `git log -3` on resume. As of this writing, HEAD is a new commit
on top of entry #56's commit `6ca185d` (the GOD PROMPT "CKKS PHASE 24"
round: "CKKS BOOTSTRAP RUNTIME VALIDATION, PRECISION RECOVERY &
REGRESSION STABILIZATION," plus an unplanned WSL/Windows recovery
mission that opened the round). **WSL recovery**: a `HCS_E_CONNECTION_
TIMEOUT` from the prior session traced to Windows host free RAM
bottoming at ~0.49GB (96.8% used by ordinary desktop apps -- none
touched). No destructive action needed or taken: free RAM rose on its
own and a plain `wsl -d Ubuntu` retry succeeded once it did. Repo
verified intact (HEAD=origin/main=`6ca185d`, untracked user work
preserved). **Resource gate established and enforced**: `tests/
test_boot.cpp`'s own source comment records a historical 8.12-8.32GB
peak host RSS at n=8192/sizeQ=30/sizeP=2 -- not the "~2-3GB" Phase 23
speculated. Windows free RAM measured 1.8-3.6GB throughout this round,
nowhere near a safe ~10GB floor, so **the bootstrap was correctly NOT
executed** -- no claim of runtime success or failure is made.
**Architecture fully audited instead** (the complete 960-line file read
end to end): rotation-hoisting is disabled by default in the current
file (the trusted `KROT`/`KSC`-cached path is what actually runs); the
pipeline deliberately bounces CPU/GPU per diagonal, not fully
GPU-resident; EvalMod's double-angle depth (`rDbl=7` at `HW=64`) has a
real, quantified safety margin over the file's own measured actual
input range (`K=9.49`); ModRaise/C2S/EvalMod(R) each already have a
genuinely independent numerical check wired in (exact Garner-CRT, not
circular); and the test as written is single-shot/fully-deterministic
(hardcoded seeds, one fixed message) -- cannot demonstrate
repeatability even once resources allow running it, without a new
small driver. Rust/FFI exposure reconfirmed at zero (unchanged from
Phase 23). **Phase 23's two "registered-memory" test failures,
re-investigated and the diagnosis CORRECTED**: proved (not inferred)
that both are deterministic test-order artifacts from `budget_
exhaustion_falls_back_safely_and_stays_correct`'s own deliberate
16MiB-budget-exceeding design, NOT Windows-RAM-pressure sensitivity as
Phase 23 claimed -- skipping that one test makes everything else pass
unconditionally; not skipping it reproduces the identical two failures
regardless of current free RAM. Not a production defect; not fixed
(test-hygiene, out of this phase's scope) -- documented precisely
instead. Full regression (debug, no skip): 55 passed/2 failed
(explained)/1 ignored, identical to Phase 23's counts. Release (skip
the ordering-sensitive test): 56 passed/0 failed/1 ignored. No source
file changed this round -- investigation and documentation only. See
`EXPERIMENT_LEDGER.md` entry #57, `research/MIGRATION_MATRIX.md`'s
bootstrap milestone tracking, and `research/TRANSFER_PROFILING.md`'s
Phase 24 correction section for full detail.

## Blocked in Phase 24 (carried forward from Phase 23, now with a sharper gate)

Bootstrap runtime re-verification remains blocked -- same test, same
dependency set, build-ready. The threshold is now explicitly ~10GB
Windows free RAM (not the "~2-3GB" earlier speculation), justified by
the test file's own documented historical peak RSS. Re-attempt once
free RAM is reliably at or above that level for the session's
duration. Additionally, per this round's own audit: the test as
written is single-shot/deterministic -- a genuine "independent
repeatability across fresh encryption randomness" check (required by
the governing directive) will need a small new driver varying seeds
and messages, not just re-running the existing binary multiple times.

## Earlier commit (Phase 23)

HEAD was previously a commit on top of entry #55's commit `43d541f` (the GOD PROMPT "CKKS PHASE 23"
round: "FULL CKKS CAPABILITY COMPLETION, APPLICATION READINESS &
PERFORMANCE AUDIT" -- the first round of a new mission, explicitly NOT
another isolated microbenchmark. **Section 1-2 (capability audit)**:
read `research/MIGRATION_MATRIX.md` plus direct source, confirming the
"Bootstrap pipeline: C++-only done, everything else not started" row
was accurate. **Section 3 (bootstrap investigation)**: found the
COMPLETE bootstrap (ModRaise/C2S/EvalMod x2/S2C/recovery, real keys,
sparse secret h=64, analytic-inverse-DFT C2S with rotation hoisting,
degree-10-Chebyshev double-angle EvalMod) lives entirely inside
`tests/test_boot.cpp` (960 lines) -- no dedicated `src/bootstrap.cpp`
exists. Confirmed, via a memory-safe BUILD-ONLY verification (6 missing
object files chased down and compiled: `src/ptops.cpp`, `src/rotate.cpp`,
`kernels/rns_arith.cu`, `kernels/automorph.cu`, `kernels/automorph_eval.cu`,
`kernels/stage_acc.cu`; final link against `test_boot.cpp` had ZERO
undefined references), that this pipeline is NOT bit-rotted -- but it
was **NOT re-run at runtime**: Windows host free RAM measured
repeatedly at 0.34-0.92GB throughout this phase, the lowest sustained
range this project has recorded, below any level this project's own
discipline would permit for an n=8192 run generating dozens of real
rotation keys. **Explicitly NOT claimed to numerically pass this
round** -- re-run deferred to a safer-memory session (see `## Blocked
this round` below). **Section 4 (target selection)**: compared
rotation, plaintext ops, conjugation, polynomial evaluation, and
serialization via direct grep/read before choosing. Conjugation
inherits rotation's status (it IS rotation, by `k=M-1`). Polynomial
evaluation was rejected -- grepping it surfaces only `test_polyeval`'s
own still-UNRESOLVED multi-level noise question, which this phase does
not resolve and the governing directive forbids claiming fixed.
Serialization does not exist at the C++ level AT ALL (a far larger,
greenfield lift). **ROTATION selected**: fully implemented in C++,
already used correctly inside the bootstrap itself (`test_boot.cpp`'s
own `KROT` rotation-key cache IS this operation), zero Rust/FFI
exposure, and needed no new cryptographic primitive -- reading
`src/evalkeygen.cpp` found `evalkeygen_host_sold` already takes an
explicit, general `sOld` parameter ("generalized: caller supplies the
switched key" per its own comment), so a rotation eval key
(`sOld = sigma_k(s)`, not `s^2`) needed only a different argument to an
already-general function. **Section 5 (implementation)**: new
`rust/crates/ckks-cuda/cpp_bridge/rotation_bridge.cpp` + new
`rust/crates/ckks-cuda/src/rotation.rs` (`RotationContext`, same
setup/hot-path split as `KeySwitchContext`) -- composes ONLY existing
`gpufhe::` functions, zero production C++ file changed. Caches
`DeviceKSContext`/`DeviceKSWork` ONCE at construction, unlike
`rotate_ct_resident`'s own per-call rebuild (its own source comment
flags this as a known opportunity -- taken here). `build.rs` gained
`src/rotate.cpp` + 3 kernel files. **9 new Rust tests, 0 mismatches**:
`rotate` bit-exact vs. the host-orchestrated `rotate_ct_host` oracle
across random/boundary/repeated-call cases, PLUS a genuinely
independent real-ciphertext check (encrypt -> rotate -> decrypt ->
decode vs. an expected slot permutation computed without touching the
crypto pipeline) that caught a real bug on its first run -- in the
TEST's own expectation (it assumed a narrow 4-element wraparound
instead of the TRUE full n/2=512-slot cyclic shift), not the
implementation; fixed the test, confirmed `rot_amount=r` is a LEFT
shift (`k=5^r mod 2n`) empirically. **Performance** (n=1024, sizeQ=8,
sizeP=2, numPart=4, release, cold-start/steady-state kept separate):
`RotationContext::new` ~524.5ms cold-start; `rotate` steady-state
median 1.84ms (20 reused-context calls, range 1.64-1.95ms); ~285x
cold-to-steady ratio. **Full regression**: `cargo test` (debug, 58
tests): 56 passed, 2 FAILED, but both failures are in
`codec::registered_tests` (Phase 20/22's unrelated registered-memory
feature, zero files touched this round), reproduce ONLY alongside 2
specific earlier tests in that module and PASS in total isolation --
diagnosed as this session's severe Windows-RAM pressure hitting
`cudaHostRegister`'s own documented capacity sensitivity, NOT a
rotation regression; left as found per the governing directive's own
"do not reopen Phase 22 without a reproducible defect" instruction.
`cargo test --release rotation::`: 10/10 passed. Graphify refreshed and
verified (2601->2683 nodes, 5734->5915 edges, 154->159 communities; new
`RotationContext`/bridge symbols confirmed correctly attributed). See
`EXPERIMENT_LEDGER.md` entry #56 and `research/MIGRATION_MATRIX.md`'s
new rotation row for full detail.

## Blocked in Phase 23

**Bootstrap runtime re-verification** -- `/tmp/test_boot_build` (or an
equivalent fresh build) is ready to run; it was NOT executed this round
because Windows host free RAM was measured at 0.34-0.92GB throughout
the session (`(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB`
from PowerShell -- WSL `free -h`/`nvidia-smi` both reported abundant
headroom the whole time, re-confirming yet again that Windows host RAM,
not WSL or GPU memory, is this project's real constraint). Re-attempt
`tests/test_boot.cpp` (n=8192, sizeQ=30, sizeP=2) the NEXT time Windows
free RAM is reliably above ~2-3GB for the session's duration, to get a
genuine runtime confirmation of `eFin<2e-2` and correct level
restoration -- per the governing directive's own "never claim
bootstrapping works until an encrypted ciphertext has passed through
the complete bootstrap and its restored precision/levels are
independently validated" requirement. This is the natural next step
once memory allows; rotation (this round's actual deliverable) is
already a prerequisite the bootstrap itself depends on, so nothing
about this round's work needs to be redone first.

## Earlier commit (Phase 22)

HEAD was previously a commit on top of entry #54's commit `a22ff5f` (the GOD PROMPT "CKKS PHASE 22"
round: completed the missing Rust/FFI validation Phase 20 left open,
RESOLVED Phase 20's unexplained prewarming anomaly, and reassessed the
promotion decision with full evidence. Used Graphify BEFORE writing
code -- `graphify explain` confirmed in one command that no Rust/FFI
caller existed yet for the registered path; refreshed the graph 3
times during the work and verified the new symbols appear correctly
connected afterward. Added `codec::encrypt_registered`/
`prewarm_registered_buffer`/`registered_buffer_is_ready` to Rust (3
new `extern "C"` functions in `cpp_bridge/encdec_bridge.cpp`, 1
missing `build.rs` compile-list entry that was the SOLE reason this
path was unreachable from Rust before, a new diagnostic C++ entry
point the ORIGINAL Phase 20 function now delegates to unchanged).
`codec::encrypt`'s own behavior is completely untouched. **8 new Rust
tests, 0 mismatches**: correctness vs. production, pool reuse, prewarm
idempotency, budget exhaustion with real fallback (14 keys, some
registered, some correctly falling back), validation-error parity, 2
concurrency scenarios (84 operations), and the critical real encrypted
encode→encrypt_registered→EvalMult→decrypt→decode round trip through
the actual Rust API (not RNS-only) -- PASSES. **A real bug found, NOT
fixed (out of scope), documented**: `gpufhe_ks_fixture_create` crashes
with `std::bad_alloc` at `size_q=5, size_p=2, num_part=4` specifically
-- a pre-existing eval-keygen defect, unrelated to registered memory,
this phase's own tests simply avoid. **Prewarming anomaly RESOLVED,
not left inconclusive**: Phase 20's own benchmark compared DIFFERENT
`(n,sizeQ)` keys between its lazy/prewarmed configs -- a confound. A
controlled re-test at the IDENTICAL key (separate fresh processes)
found NO real difference (confirmed by a structural source-code
argument too: the C++ pool cannot distinguish how a key became
ready). **Full pipeline re-measured**: `encrypt_registered` at or
below `encrypt` in 3/3 runs through the ACTUAL Rust API (ratios
0.873-0.970), confirming the win survives the FFI boundary. **Decision:
Category B -- STILL RETAINED AS OPT-IN, not promoted** -- every
safety/correctness criterion Phase 20 left open is now resolved; the
ONLY remaining question is whether a modest, reproducible win
justifies the added complexity for the DEFAULT path, a judgment call,
not a safety one. See `EXPERIMENT_LEDGER.md` entry #55 and `research/
TRANSFER_PROFILING.md`'s Phase 22 section for full detail.

## Earlier commit (Phase 21)

HEAD was previously a commit on top of entry #53's commit `f20fa88`
(the GOD PROMPT "CKKS PHASE 21"
round: NOT a CKKS research round -- a tooling/workflow integration.
Installed Graphify (`pipx install graphifyy`, version `0.9.102`), a
local, zero-LLM-backend code knowledge graph (tree-sitter AST,
`GRAPHIFY_BACKEND=none` explicitly set throughout -- no project
material was sent to any external service). Project-scoped Claude Code
integration (`graphify install --project --platform claude`,
deliberately WITHOUT `--strict` -- verified non-blocking) appended one
`## graphify` section to the end of the pre-existing `CLAUDE.md`
(the Scrum-Orchestrator block, diffed byte-for-byte against a backup,
is untouched) and added `.claude/skills/graphify/` +
`.claude/settings.json` alongside, never inside, the existing
`.claude/agents/`/`scrum-orchestrator` skill. Four git hooks
(`post-commit`/`post-checkout`/`post-merge`/`post-rewrite`, all
verified non-blocking -- detached background rebuild, 600s timeout,
never delays `git commit`) plus a union-merge driver for
`graphify-out/graph.json` were installed for auto-refresh. The graph
itself (2539 nodes, 5563 edges, 162 communities at the time) was
committed (`graphify-out/graph.json`, `GRAPH_REPORT.md`,
`.graphify_analysis.json`); `graph.html`/`cache/`/`manifest.json`/
`.graphify_root` are gitignored as machine-specific or disposable.
**A documented, honest 12-question validation suite** (`research/
GRAPHIFY_GUIDE.md`) found the graph genuinely useful for code
navigation (god nodes, call paths, which tests reference a function)
but confirmed real gaps: `CMakeLists.txt`/CTest registration is NOT
indexed at all (CMake isn't a parsed language here); research `*.md`
docs appear only as lightweight reference nodes, not semantically
searchable; free-text `graphify query` can mis-seed on confusingly-
named local symbols (`graphify explain "<exact name>"` is more
reliable). No CKKS code, cryptographic behavior, or research
conclusion was touched that round. See `EXPERIMENT_LEDGER.md` entry
#54 and `research/GRAPHIFY_GUIDE.md` for full detail.

HEAD was previously a commit on top of entry #52's commit `8d6dbe4`
(the GOD PROMPT "CKKS PHASE 20"
round: the first phase in the pinned-memory investigation to move from
diagnosis to a real implementation PROTOTYPE. Built `include/
registered_host_buffer.h`'s `RegisteredHostBuffer` (move-only RAII,
4-state machine) owning a page-aligned host allocation registered via
`cudaHostRegister`, and `src/encrypt_host_shoup_pooled_barrett_
batched_registered.cpp` (mirrors Phase 15's pinned variant exactly,
except the host staging buffer's allocation mechanism; registers ONCE
per key; self-imposed 16MiB cumulative budget with a safe, leak-free
fallback to the unmodified production pageable path if exceeded or if
registration fails). **Correctness**: bit-exact vs. `encrypt_host`,
Phase 15 batched, AND Phase 15 pinned across every size/boundary/
repeat/interleave this project tests -- 0 mismatches. Real encrypted
pipeline (encode/encrypt/EvalMult/decrypt/decode): identical max error
across production/pinned/registered -- [PASS] for all three.
**Concurrency**: 6 scenarios, 436 checks, 0 mismatches, including
prewarm-racing-with-encryption and fallback-under-genuine-budget-
pressure (12 registered + 4 fell back, both correct). **Performance**:
the registered path's cold-start (~9.0-9.4ms) is ~2.1-2.4x CHEAPER
than the pinned path's cold-start (~18.9-21.6ms) -- directly resolving
the exact cold-start weakness that blocked Phase 15/16's pinned
variant from production promotion. Steady-state: registered
consistently beat production by ~7-14% across 3 independent runs, and
was competitive with pinned. Confirmed via `nsys` in the REAL
production-shaped code (not just Phase 19's isolated diagnostic):
`cudaHostRegister` ~4.8x cheaper per call than `cudaHostAlloc`.
**Unresolved, honestly flagged**: explicitly-prewarmed was
consistently slightly SLOWER than lazily-created-on-first-use in every
run -- a genuine open question, not investigated further. **Decision:
Category B -- RETAIN AS AN OPT-IN EXPERIMENTAL FEATURE, NOT promoted**
(a real, modest win, but not yet validated at the Rust/FFI layer, and
one unexplained anomaly). See `EXPERIMENT_LEDGER.md` entry #53 and
`research/TRANSFER_PROFILING.md`'s Phase 20 section for full detail.
Confirm push status -- push again if `git status` shows ahead of
origin.

## Earlier commit still (Phase 19)

HEAD was previously a commit on top of entry #51's commit `91ae35f`
(the GOD PROMPT "CKKS PHASE 19"
round: an INVESTIGATION phase, no production or experimental code
changed. Mechanistically ISOLATED the cause of Phases 16-18's
allocation-cost variance using a new comparison diagnostic
(`bench/bench_alloc_path_comparison.cpp`, 5 allocation paths x 4
sizes, CUDA context init isolated as its own stage, randomized
fixed-seed interleaved steady-state rounds, 425 samples/run, raw CSV)
plus `nsys` (NVIDIA Nsight Systems, installed and usable WITHOUT root
on this system -- not previously known/used). **Decisive finding**:
87-97% of every `cudaHostAlloc`/`cudaMallocHost` call's wall-clock
time is spent inside `ioctl()` syscalls into the NVIDIA driver, on
the SAME calling thread, confirmed by querying `nsys`'s own exported
SQLite trace directly -- not inferred from comparative timing, as
every prior phase's conclusions were. `cudaMallocHost` and
`cudaHostAlloc(cudaHostAllocDefault)` are statistically
indistinguishable (confirms CUDA's documentation empirically).
`cudaHostRegister` on pre-touched memory is 1.5x-4x CHEAPER than
`cudaMallocHost`/`cudaHostAlloc` for the full lifecycle, reproduced
across 4 independent fresh-process runs, and shows the SAME `ioctl`-
bound profile but ~3x faster per call. A separate, new finding: plain
pageable `malloc`+`memset` (zero pinning, zero CUDA) ALSO jumps
sharply at exactly 2MiB -- Phase 18's THP-hint effect generalizes
beyond `mlock()`/pinning entirely, to ordinary anonymous-mapping
first-touch at the huge-page size boundary (a separate, additive
mechanism from the CUDA/`ioctl` cost). Also audited the CMake
`gpufhe_test` macro (it auto-registers every target with `ctest`,
including pure timing diagnostics) and added a `gpufhe_benchmark`
function (same build target, no `add_test`) for the whole Phase
16-19 pinned-memory/THP diagnostic cluster -- **this discipline was
then reused unchanged in Phase 20, see "Current commit" above**.
No production or experimental code was changed this round. See
`EXPERIMENT_LEDGER.md` entry #52 and `research/TRANSFER_PROFILING.md`'s
Phase 19 section for full detail.
Confirm push status -- push again if `git status` shows ahead of
origin.

**Reminder, confirmed the hard way in an earlier round**: this machine
has (at least) two SEPARATE git clones of this GitHub repo -- the WSL
one (`~/GPU-Resident-Library`) is authoritative for all Rust/relin/CKKS
work and should always be checked via a WSL shell, not a Windows one;
a Windows-filesystem clone (`C:\Users\samca\GPU-Resident-Library`) has
an entirely unrelated commit history (raw CUDA-kernel-optimization
work) and untracked, stale copies of `rust/`/`research/` -- never run
`git status`/`log`/commit decisions against it for this work.

## READ THIS FIRST, ALWAYS, before any substantial task: Graphify (a local code knowledge graph) is installed and usable -- `graphify query`/`explain`/`path` for navigation, but it is NEVER the authoritative source for a decision, experiment result, or CKKS fact

Phase 21 installed [Graphify](https://github.com/Graphify-Labs/graphify)
(`pipx install graphifyy`, v0.9.102) as a project-scoped Claude Code
skill + non-blocking advisory hooks (verified: NOT `--strict`, confirmed
via direct testing that `graphify hook-guard search`/`read` exit 0 with
no output). A committed graph (`graphify-out/graph.json`/
`GRAPH_REPORT.md`/`.graphify_analysis.json`) covers this repo's C++/
CUDA/Rust source (2539 nodes, 5563 edges, 162 communities as of this
phase) with ZERO LLM/cloud calls (`GRAPHIFY_BACKEND=none` set
explicitly throughout -- nothing left this machine).

**Use it for**: `graphify explain "<exact symbol name>"` to find a
function/class, its callers, and which tests reference it;
`graphify path "A" "B"` for a dependency relationship; `graphify
god-nodes` for the architectural hubs. All far cheaper than a broad
`Grep`/`Read` (the tool's own `graphify benchmark` measured 91.2x
fewer tokens per query on this actual repo).

**Do NOT use it for**: anything this project's `research/*.md` docs
already own -- what Phase 19/20 (or any phase) concluded, which
implementation is production vs. experimental and WHY, what must
never be weakened. The graph has research docs only as lightweight
reference pointers, not semantic content (confirmed via testing, see
below) -- read the actual markdown for that. `CMakeLists.txt`/CTest
registration is NOT indexed at all (confirmed: zero nodes) -- grep
`CMakeLists.txt` directly for the `gpufhe_test`/`gpufhe_benchmark`
split. Free-text `graphify query` can mis-seed on a confusingly-named
local variable (tested: it seeded on an enum value named
`A_Production` instead of the real `encrypt_host_shoup_pooled_
barrett_batched` function) -- prefer `explain`/`path` with exact
names for anything you intend to rely on.

**Keep it fresh**: run `graphify update .` after editing code (fast,
local, no API cost) -- also auto-runs via a local-only post-commit/
checkout/merge/rewrite git hook (never blocks `git commit`; logs to
`~/.cache/graphify-rebuild.log`). `GRAPH_REPORT.md`'s own "Built from
commit" line vs. `git rev-parse HEAD` tells you if it's stale.

Full detail, the complete 12-question validation suite with evidence,
every known extraction gap, and the four-layer memory-architecture
rationale: `research/GRAPHIFY_GUIDE.md`. `EXPERIMENT_LEDGER.md` entry
#54.

## READ THIS SECOND if resuming into pinned-memory work: CUDA-registered host memory is now Rust/FFI-ACCESSIBLE (`codec::encrypt_registered`), fully correctness/concurrency-verified at BOTH the C++ and Rust layers, and the prewarming anomaly is RESOLVED (it was a benchmark confound) -- STILL RETAINED AS OPT-IN; the only remaining question is magnitude-vs-complexity, not safety

**Phase 20 built the C++ prototype; Phase 22 completed the missing
Rust/FFI validation and resolved both open items Phase 20 left
behind.** `RegisteredHostBuffer` (move-only RAII) + `encrypt_host_
shoup_pooled_barrett_batched_registered` (C++, Phase 20) are now
reachable from Rust via `codec::encrypt_registered`/
`prewarm_registered_buffer`/`registered_buffer_is_ready` (Phase 22) --
`codec::encrypt`'s own behavior is completely unchanged either way.

**Both of Phase 20's own "unresolved follow-ups" are now resolved**:
(1) the explicit-prewarming anomaly (prewarmed buffers measured
slightly slower, every run) was found to be a BENCHMARK CONFOUND --
Phase 20's own lazy/prewarmed comparison used DIFFERENT `(n,sizeQ)`
keys (`sizeQ` vs. `sizeQ+1`). A controlled re-test at the IDENTICAL
key, via separate fresh processes, found NO real difference
(confirmed independently by a structural code argument: the C++ pool
cannot tell how a key became ready). (2) Rust/FFI validation now
exists: 8 new tests, 0 mismatches, including the real encrypted
encode→encrypt_registered→EvalMult→decrypt→decode round trip through
the ACTUAL Rust API (not RNS-only) -- PASSES. A concurrency audit at
the Rust layer (84 operations, 0 mismatches) complements the
re-confirmed C++ one (436 checks, 0 mismatches).

**Performance confirmed again, now through Rust too**: `encrypt_
registered` was at or below `encrypt` in 3/3 runs through the actual
Rust API (ratios 0.873-0.970) -- the modest, reproducible win Phase
20 found at the C++ level survives the FFI boundary crossing.

**A real, separate, narrow bug was found and documented (NOT fixed,
out of scope)**: `gpufhe_ks_fixture_create` crashes with `std::bad_
alloc` at `size_q=5, size_p=2, num_part=4` specifically, regardless of
`n` -- a pre-existing eval-key-generation defect, unrelated to
registered memory itself. Avoid this exact combination in any new
test; a future phase may want to investigate it directly.

**Decision: Category B -- STILL RETAINED AS AN OPT-IN EXPERIMENTAL
FEATURE, NOT promoted to production.** Every safety/correctness
criterion is now satisfied; the ONLY remaining question is whether a
modest (single-digit-to-low-double-digit percent) win justifies the
registered pool's own machinery becoming part of the DEFAULT
production path -- a complexity judgment call, not something further
testing resolves. `encrypt_host_shoup_pooled_barrett_batched`/
`codec::encrypt` (production) remain completely unchanged. Full
detail: `research/TRANSFER_PROFILING.md`'s Phase 20 AND Phase 22
sections.

## READ THIS THIRD if resuming into pinned-memory work: the root mechanism is now PROVEN (`ioctl()`-bound, every call, via NVIDIA's own tracer) -- this is the finding Phase 20 (above) turned into a working prototype

**Phase 19's core result**: built a new controlled comparison
(`bench/bench_alloc_path_comparison.cpp`: pageable malloc, pure-Linux
`mlock`, `cudaMallocHost`, `cudaHostAlloc` with two documented flags,
`cudaHostRegister` -- 5 paths x 4 sizes, CUDA context init isolated
as its own stage, fixed-seed randomized interleaving, 425 samples/run
in raw CSV, reproduced across 4 independent fresh-process runs) and
profiled it with `nsys` (NVIDIA Nsight Systems -- installed and
usable WITHOUT root on this system; not previously known/used in
Phases 16-18).

**Mechanistic proof, not comparative inference**: querying `nsys`'s
own exported SQLite trace directly (restricted to the single
CUDA-calling thread) shows 87-97% of every `cudaHostAlloc`/
`cudaMallocHost` call's wall-clock time is spent inside `ioctl()`
syscalls into the NVIDIA driver -- 4 `ioctl` calls per call, every
time, not a one-time cost. The very first `cudaMalloc` (CUDA context
init, 371-845ms, now cleanly isolated as its own stage so it never
contaminates any other measurement) is 86% `ioctl`-bound too, via
many calls in a retry/poll-like pattern.

**`cudaMallocHost` and `cudaHostAlloc(cudaHostAllocDefault)` are
confirmed the SAME operation** (statistically indistinguishable at
every size, across 4 runs) -- CUDA's documentation claim, now
verified empirically rather than assumed.

**This also retroactively explains Phase 17/18's `mlock()` confusion**:
`cudaPointerGetAttributes` confirms plain `malloc`+`mlock()` memory
is `cudaMemoryTypeUnregistered` -- CUDA genuinely does not consider it
pinned. `mlock()` and `cudaMallocHost` were never comparable
operations; Phase 17's "pure Linux reproduces the variance" claim was
unreliable independent of any timing confound.

**The practically useful finding**: `cudaHostRegister` on
already-allocated, already-touched memory is 1.5x-4x CHEAPER than
`cudaMallocHost`/`cudaHostAlloc` for the FULL lifecycle (alloc+touch+
pin+unpin+free), reproduced across all 4 fresh-process runs, and
shows the identical `ioctl`-bound mechanism but ~3x faster per call
(`nsys`-confirmed: ~1.03ms/call vs. ~3.2ms/call). **No production
code was changed this phase, by design** -- this is a credible,
well-evidenced DIAGNOSTIC finding, not yet implemented or
correctness-verified inside the real pooled/prewarmed production
paths. **The clearest next step, if this is revisited, is to
prototype substituting `cudaHostRegister` for `cudaHostAlloc`/
`cudaMallocHost` inside `src/encrypt_host_shoup_pooled_barrett_
batched_pinned.cpp` and the Phase 16 prewarming API, re-verify
bit-exact correctness and concurrency safety from scratch, and
re-run the full-pipeline benchmark before deciding whether to ship
it.** Also found, separately: plain pageable `malloc`+`memset` (zero
CUDA, zero pinning) jumps sharply at exactly 2MiB too -- the THP
effect Phase 18 found is not pinning-specific, it is a property of
ordinary anonymous-mapping first-touch at the huge-page boundary (an
ADDITIVE, separate mechanism from the CUDA/`ioctl` cost). Full
detail: `research/TRANSFER_PROFILING.md`'s Phase 19 section.

## READ THIS FOURTH if resuming into pinned-memory/THP work: the system-wide THP toggle is BLOCKED BY PERMISSIONS (documented, not worked around); a real but narrow ~5x THP-hint effect exists at exactly 2MiB; Phase 17's "not CUDA-specific" conclusion was REVISED -- see Phase 19/20 above for the mechanistic proof of why

**Phase 18's core result**: attempted the originally-planned
controlled THP experiment (THP=madvise vs. THP=disabled vs. restored).
The system-wide toggle (`/sys/kernel/mm/transparent_hugepage/enabled`)
requires root; a same-value no-op write returned "Permission denied"
and no passwordless sudo is available. **Per the user's explicit
choice ("Skip the THP toggle, document the blocker (recommended)"),
no workaround was attempted and the setting was never changed** --
verified identical to baseline (`madvise`) both immediately after the
denied attempt and at phase end.

Substituted the one THP variable controllable without root: per-call
`madvise(MADV_HUGEPAGE/NOHUGEPAGE)` hints
(`bench/bench_thp_hint_by_size.cpp`). Result: NO meaningful effect at
256KB/512KB/1MB, but a consistent, reproducible **~5x slowdown at
EXACTLY 2MiB** when the hint is disabled (≈230-247us →
≈1217-1270us, reproduced across 2 independent runs) -- the first
DEMONSTRATED, mechanistic THP effect this project has found. Its
magnitude is still far too small to explain Phase 16/17's multi-
millisecond `cudaMallocHost` variance.

**More importantly, this phase REVISES Phase 17's own conclusion.**
A same-session, INTERLEAVED re-test of "does pure Linux `mlock()`
reproduce the same variance as `cudaMallocHost`" directly contradicts
Phase 17's HIGH-confidence "yes, so it's not CUDA-specific" finding:
pure Linux `mlock()` was clean and fast (174-1613us) while
`cudaMallocHost`, interleaved in the same short window for the
identical size, remained consistently 2984-5079us -- an order of
magnitude slower. Phase 17's finding was most likely a coincidental
same-window timing artifact, not a robust equivalence. The updated
ranked assessment PROMOTES CUDA/WSL2-driver-passthrough overhead to
the leading hypothesis (MEDIUM-HIGH confidence) and DOWNGRADES the
pure-kernel-`mlock()` explanation to MEDIUM.

**No code was changed.** Phase 16's decision (opt-in experimental
feature, not promoted, pageable path remains production) still
stands -- category 3 of the decision gate still applies, just with a
more precisely (and partially differently) understood external
mechanism. **Do not re-attempt a code-level fix.** If this is
revisited, the clearest next step is obtaining passwordless sudo (or
running the toggle manually) to finally exercise the originally-
planned system-wide A/B/C THP comparison, or a GPU-passthrough-layer
trace to test the CUDA/driver hypothesis directly. Full detail:
`research/TRANSFER_PROFILING.md`'s Phase 18 section.

## READ THIS FIFTH if resuming into pinned-memory work: the allocation-cost variance was EXPLORED (Linux kernel mlock() path vs. CUDA/driver overhead -- see Phase 19/20 above for the mechanistic proof), the Phase 16 decision stands, and a WSL2-ballooning claim from Phase 16 has been downgraded to "not confirmed"

**Phase 17's core result**: Phase 16 found `cudaMallocHost`'s cost
varies wildly (16us-5.9ms) and guessed, without evidence, that WSL2
memory ballooning was responsible. Phase 17 investigated this
directly and found: (a) Phase 16's OWN measurement bundled 5 separate
CUDA calls into one timed number -- a real methodology flaw, now fixed
with 3 new properly-separated diagnostics
(`bench/bench_pinned_alloc_diagnostic.cpp`,
`bench/bench_pinned_alloc_diagnostic2.cpp`,
`bench/bench_thp_diagnostic.cpp`); (b) the SAME qualitative variance
reproduces using a PURE Linux `mmap`+`madvise`+`mlock` test with ZERO
CUDA code -- proving the mechanism is the Linux kernel's own memory-
locking path, not a CUDA allocator quirk; (c) host memory pressure
(Windows free RAM measured at 1.2-2.4GB throughout) is a plausible
contributing amplifier but only correlationally supported, not
isolated; (d) a WSL2-Hyper-V-ballooning mechanism DISTINCT from
ordinary kernel memory pressure could NOT be confirmed or refuted with
available tooling -- it is no longer asserted as even a hedged guess,
only as an open, undecidable question given current tools.

**No code was changed.** Phase 16's decision (opt-in experimental
feature, not promoted, pageable path remains production) stands,
now on stronger evidence: the variance is DEMONSTRATED to be
externally controlled (category 3 of the decision gate), not merely
suspected. **Do not re-attempt a code-level fix for this variance** --
it is outside this project's control (kernel/hypervisor-level). If
pinned prewarming is revisited, the next concrete experiment (not yet
done) is a controlled sweep of many different sizes in a fixed,
non-repeating order with memory tracked at each step, to better
characterize the "size + cumulative live memory" interaction (ranked
hypothesis 3 in `research/TRANSFER_PROFILING.md`'s Phase 17 section),
or a controlled, reversible test of disabling Transparent Huge Pages
to see if that is a necessary ingredient. Full detail: `research/
TRANSFER_PROFILING.md`'s Phase 17 section.

## READ THIS SIXTH if resuming into performance work: a pinned-memory PREWARMING API exists, fully correct and concurrency-safe, but was NOT promoted because its own allocation cost is unpredictable in this environment -- Phase 20 (top of this file) built a registered-memory alternative that resolves this exact weakness

**Phase 16's core result**: `encrypt_prewarm_pinned_buffer(n,sizeQ)` /
`encrypt_pinned_buffer_is_ready(n,sizeQ)` / `encrypt_host_shoup_
pooled_barrett_prewarmed(...)` (`src/encrypt_host_shoup_pooled_
barrett_prewarm.cpp`) exist, are bit-exact correct, and are
concurrency-safe (0/372 mismatches across 5 independent runs covering
6 lifecycle scenarios: same-key/different-key concurrent prewarm,
simultaneous encrypt at a prewarmed key, prewarm/encrypt interleaved on
a fresh key, 200-cycle longevity, mixed shared-key race). An
application that never calls `encrypt_prewarm_pinned_buffer` pays
NOTHING extra and behaves exactly like today's production path
(`encrypt_host_shoup_pooled_barrett_batched`, unchanged).

**Why it was NOT promoted**: a workload-level benchmark
(`bench/bench_encrypt_prewarm_workloads.cpp`) found the one-time
`cudaMallocHost` cost ITSELF is unpredictable -- n=1024/8192
consistently cost several milliseconds while n=2048/4096 consistently
cost under 20 microseconds, reproduced across two independent full
runs in the same order, plausibly tied to this session's own host-
memory pressure interacting with WSL2's dynamic-memory ballooning
(NOT independently confirmed -- no WSL2 VM memory-ballooning trace
tool was available to verify the mechanism directly). This makes the
prewarm/no-prewarm break-even point itself unpredictable for short
workloads (Profile A: cold-start, 1-3 ops; Profile B: typical, 20 ops
-- both MIXED, winning or losing depending on which allocation-cost
regime a given key happens to land in). Only Profile C (long-lived,
500-5000 ops) wins unconditionally at every tested n. **Do not
re-promote this to production without either (a) getting real
visibility into the allocation-cost variance's root cause, or (b)
accepting the risk explicitly for a specific, named long-lived-service
deployment where Profile C's unconditional win is what matters.** Full
detail: `research/TRANSFER_PROFILING.md`'s Phase 16 section.

## READ THIS SEVENTH if resuming into performance work on the OLDER (Phase 15) material: transfer batching is DONE and shipped; a pinned-memory follow-up was built, verified, measured FASTER in isolation, then found to be a net loss for real usage and correctly NOT shipped

**Phase 15's core result**: `encrypt_host_shoup_pooled_barrett_batched`
(`src/encrypt_host_shoup_pooled_barrett_batched.cpp`) is now the actual
production path (`cpp_bridge/encdec_bridge.cpp`'s `gpufhe_encrypt`
calls it) -- Phase 14's own "clear next experiment" (batch the
upload/download transfers) is DONE. Traced the four separate
`d_v/d_e0/d_e1/d_m` transfers, confirmed they share identical size/
lifetime/pattern, and merged the 8 separate `cudaMemcpy` calls into 2
(one packed H2D, one packed D2H) over ONE pooled device buffer -- the
four kernel launches are UNCHANGED, just receiving pointer offsets
into the packed region. Bit-exact verified against `encrypt_host`
including a packed-buffer-boundary stress case; the new pooled
resource's concurrency safety re-audited (not assumed inherited) with
4 scenarios, 5 runs, 0/396 mismatches. Measured 1.57x-2.78x faster than
the original `encrypt_host`, 1.07x-1.17x faster than Phase 14 alone.

**A second experiment (pinned host staging memory) was ALSO built,
fully verified, and measured FASTER in an isolated warmed-up
benchmark (1.06x-1.17x further speedup) -- but is NOT in production.**
Wiring it into the full-pipeline benchmark's cold single-call
measurement showed the OPPOSITE result (slower than Phase 14),
investigated directly: `cudaMallocHost` costs ~3-8ms per distinct
`(n,sizeQ)` key on first use (vs. ~0.4-0.5ms for ordinary `cudaMalloc`)
-- more than `encrypt_host`'s entire typical latency, so any usage
pattern that doesn't repeat enough calls at one key to amortize this
cost would see it as a net loss despite the real steady-state benefit.
**Do not re-attempt wiring the pinned variant (`src/
encrypt_host_shoup_pooled_barrett_batched_pinned.cpp`, already fully
built and tested) into production without first solving this cold-
start problem** -- e.g. an explicit pre-warming step at context-
creation time, decoupling the one-time `cudaMallocHost` cost from the
first real encryption call. Full investigation: `research/
TRANSFER_PROFILING.md`.

The old `encrypt_host_shoup_pooled_barrett` (Phase 14) and
`encrypt_host` (Phase 11) both remain fully present, tested, and
directly callable as reference/fallback. Full Rust suite unaffected:
38 passed, 0 failed, 1 ignored, both build profiles, re-confirmed
across 4 separate release runs under cargo's DEFAULT PARALLEL
execution with zero flakiness.

**POST-Phase-15 ranked state inside encrypt_host**: embed, combine, and
transfer/allocation are all now optimized. The clear next candidate
(if performance work continues) is the pinned-memory pre-warming idea
above -- a real, already-measured 1.06x-1.17x win is sitting there,
blocked only by the cold-start problem, not by any correctness or
concurrency issue (both are already proven). The GPU-resident
tensor+keyswitch+combine+rescale chain is still only ~1% of total
latency, same conclusion as every prior phase.

**Do not re-measure the Phase-14-era baseline numbers as if they still
apply** -- `bench/bench_evalmult_e2e_breakdown.cpp` (now extended with
a 5-way `encryptVariant` int flag: 0=OLD, 1=Phase13, 2=Phase14,
3=Phase15 Experiment1/production, 4=Phase15 Experiment2/pinned-
not-shipped, alongside Phase 12's `useFft`) is the current, correct
per-stage benchmark; re-run it fresh rather than citing this entry's or
any earlier entry's own tables. The single-config mode's own default
is `encryptVariant=3` (the actual production path), NOT 4.

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
