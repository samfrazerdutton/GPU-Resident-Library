# Phase 32 independent review

Per the Phase 32 god-prompt's Work Package 6 / "Agent 5": a skeptical,
verification-focused pass over this phase's own claims, conducted by
re-running things rather than re-reading my own prose and agreeing with
it. Where a claim could not be independently re-verified this round, that
is stated rather than silently assumed.

## 1. How this review was actually conducted

Not via a separate fresh-context reviewer persona — the three research
deliverables (architecture audit, edge AI prototype, LLM KV-cache
prototype) WERE produced by independent parallel agents with no visibility
into each other's work or into the implementation that followed, which is
real independence for those three. The application workload
(`resident_finance`), the CKKS audit, and the integration/commit decisions
were done by the same session that is now reviewing them, so that part of
this review is self-verification (re-running, re-deriving, checking
against ground truth) rather than a second opinion from a different
reasoning process. This limitation is stated plainly rather than
disguised as independent review.

## 2. Claims independently re-checked this round, and what was found

- **Architecture audit's "4 of 13 caches unguarded" claim**: re-verified
  directly by grepping `src/rotate.cpp`, `src/keyswitch.cpp`,
  `src/encrypt_host_shoup.cpp`, `src/encdec.cpp` for `static std::map`
  and `std::mutex`. **Confirmed accurate**: all four files contain the
  static-map cache pattern with zero `std::mutex` occurrences. Not acted
  on this phase (see section 4) — this is a verification, not a fix.
- **Both prototype forks' reported numbers**: re-compiled and re-ran both
  standalone (`kv_block_cache_sim.cpp`, `edge_sensor_residency_prototype.cu`)
  independently rather than trusting the agents' self-reported output.
  KV-cache: reproduced 768/768 bit-exact, matching evictions/spills/
  restores counts exactly. Edge sensor: reproduced correctness (5.96e-8,
  identical) and a comparable speedup (15.5x on this re-run vs. ~13x
  reported — both real measurements, the difference being normal run-to-
  run GPU clock/thermal variance this project has documented since Phase
  25, not a discrepancy requiring investigation).
- **`resident_finance`'s own correctness and crossover claims**: this is
  the one deliverable with no second reviewer at all. Re-read against
  the actual kernel math in `kernels/finance_resident.cu` and the CPU
  reference in `examples/resident_finance/main.cpp` line by line this
  round, specifically hunting for the exact bug class this project keeps
  finding (a buffer aliasing issue): confirmed `update_rolling_stats_kernel`
  reads old price from `price[]` and new price from a **separate**
  `newPrice[]` array, and the host loop uploads into the dedicated
  `newPrice` `DeviceBuffer`, not into `price` — this was caught and fixed
  during this round's own implementation (an earlier draft uploaded into
  `price` directly and passed it as both arguments, which would have made
  every log-return exactly zero and silently produced a constant,
  zero-volatility simulation that still "passed" its own internal
  self-consistency check, since the GPU and CPU would have agreed on the
  wrong answer identically). That the bug was self-caught before any
  build, not found by a second reviewer, is the honest limitation here.

## 3. Methodology check: did the application genuinely use the library?

Per the god-prompt's explicit "do not mark an application workload
complete if it bypasses the library's core architecture" instruction:
`examples/resident_finance/main.cpp` includes `device_buffer.h` and
`cuda_check.h` and uses `gpufhe::DeviceBuffer<double>` for all five
resident arrays (`price`, `ewmaVar`, `newPrice`, `callPrice`, `vol`) —
verified by reading the file, not by trusting its own comments. It does
NOT use a `GpuExecutionContext` because the architecture audit explicitly
recommended against building one this phase (no second justified caller
yet) — `resident_finance` manages its own raw `cudaStream_t` directly,
which is consistent with, not a violation of, that recommendation.

## 4. What was found but deliberately not fixed, and why that is defensible

- **4 unguarded caches** (section 2): not fixed. Three of four
  (`keyswitch.cpp`, `encrypt_host_shoup.cpp`, `encdec.cpp`) are, per the
  architecture audit, likely-superseded code paths not on the production
  hot path (production uses `keyswitch_resident.cpp` and
  `encrypt_host_shoup_pooled_barrett_batched.cpp`, both already
  mutex-guarded) — but this was **not independently re-verified this
  round** (no new test was written to prove they are truly cold). The
  fourth (`rotate.cpp`) is a Phase 25 **documented, deliberate** decision
  ("latent, not active," with a standing control test). Fixing all four
  without a demonstrated active failure would violate this project's own
  repeatedly-stated rule (Phase 31: "do not add a mutex without proof it
  protects identified shared state actively racing"). **This is flagged
  as the single highest-value next-phase candidate**, not silently
  dropped — see `research/NEXT_SESSION.md`.
- **`bench_resident_chain`'s "no significant difference" result**: not
  investigated further or re-tuned to produce a more favorable number.
  Reported as-is (section 3 of `research/CKKS_RESIDENCY_AUDIT.md`),
  because re-tuning a benchmark until it shows a win would be exactly the
  cherry-picking the god-prompt explicitly forbids.
- **Edge AI and LLM KV-cache prototypes**: correctly scoped as
  prototypes, not production paths, and not wired into the main build
  (`CMakeLists.txt`) — verified by `git status` showing them as untracked
  files outside any build target, not merely by their own documentation
  saying so.

## 5. Regression results (re-run this round, see `research/NEXT_SESSION.md` for exact figures)

C++ `ctest` (excluding memory-gated bootstrap tests, same convention
since Phase 28): pass count and the exact 10 pre-existing failures are
recorded in `research/NEXT_SESSION.md`'s Phase 32 entry — cross-checked
against Phase 31's own baseline list name-for-name; zero new failures
introduced by this phase's additions (`resident_finance` itself passed).
Rust debug/release: recorded identically. No CKKS/FHE production source
file was modified this phase (`git diff --stat` against the Phase 31
commit, limited to additive new files plus `CMakeLists.txt`'s
additive-only registration block, confirms this directly rather than by
assertion).

## 6. What this review did NOT do

- Did not have a second, independently-reasoning agent review
  `resident_finance`'s implementation or the CKKS audit — both were
  authored and reviewed by the same reasoning process. A future phase
  with more budget should have a genuinely separate reviewer re-read
  `examples/resident_finance/main.cpp` and `kernels/finance_resident.cu`
  cold.
- Did not stress-test `resident_finance` under concurrency (multiple
  threads each running an independent simulation) — only the
  architecture-level thread-safety contract from Phase 31 was cited, not
  re-proven against this specific new workload.
- Did not attempt the bootstrap-scale CKKS measurement (memory-gated,
  see `research/CKKS_RESIDENCY_AUDIT.md` section 6) — this is a stated
  external constraint, not an oversight.
