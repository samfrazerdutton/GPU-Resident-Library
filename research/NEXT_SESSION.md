# Next session — exact continuation

## Current commit

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
