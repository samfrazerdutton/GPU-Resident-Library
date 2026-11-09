# Phase 26 — Windows clone reconciliation report

GOD PROMPT "CKKS PHASE 26", Missions 0-1. Written before any code change.

## 1. Environment recovered

| Checkout | Location | Branch | HEAD | vs origin/main |
|---|---|---|---|---|
| Authoritative (WSL) | `/home/samfrazerdutton/GPU-Resident-Library` | main | `1d73518` | 0 ahead / 0 behind |
| Windows clone | `C:\Users\samca\GPU-Resident-Library` | main | `f425cdd` | behind (see Section 2) |

Session resumed in Git Bash on Windows (`MINGW64_NT-10.0-26200`), not inside
WSL. WSL `Ubuntu` was `Stopped`; it started on the first ordinary
`wsl -d Ubuntu` attempt — no diagnostics, no retry, no configuration change.

WSL checkout: HEAD `1d73518` = `origin/main` exactly, 0 ahead/0 behind, working
tree contains only the four pre-existing untracked entries Phase 25 also found
(`.claude/agents/`, `.claude/skills/scrum-orchestrator/`,
`.scrum-orchestrator/`, `tools/__pycache__/`). Neither checkout has advanced
since Phase 25 — the Windows clone's 65 dirty entries are byte-for-byte the
same set Phase 25 reported.

Windows host free RAM measured 1.75-1.94 GiB throughout this round
(`Get-CimInstance Win32_OperatingSystem`). Both bootstrap preflight gates
(n=8192, n=131072) returned NO-GO, logged to
`research/benchmarks/bootstrap_preflight_log.jsonl`. GPU: RTX 2060 Max-Q,
6144 MiB, 5942 MiB free, driver 595.97, CUDA 13.2.

## 2. The central finding: f425cdd is 82 commits BEHIND main, not ahead

`git merge-base --is-ancestor f425cdd HEAD` returns true inside the WSL
checkout. Concretely:

```
git log --oneline f425cdd..HEAD | wc -l
82
```

The Windows clone's committed history is a strict prefix of main's history,
not a divergent branch. It is simply old — pinned at a commit from well
before Phase 15's NTT/INTT batching work, Phase 20-22's registered-memory
work, Phases 23-25's bootstrap/hoisting/concurrency work, and dozens of other
commits.

This directly affects how to read the two commits the Phase 26 directive
names as "prototypes":

```
git merge-base --is-ancestor a8ce687 HEAD   # Fused pt-mul-accumulate kernel
-> YES, already an ancestor of main
git merge-base --is-ancestor 6357991 HEAD   # Device automorphism kernel
-> YES, already an ancestor of main
```

Both named prototypes are already fully merged into authoritative main's
commit history. They are not uncommitted, not divergent, and not waiting to
be ported — their commits are 82+ generations back in main's own ancestry.

## 3. The 65 dirty entries: content-level verdict

Given the above, the real reconciliation question is not "are these two
historical commits in main" (yes) but "does the Windows clone's uncommitted
working tree (on top of its stale f425cdd checkout) contain any edit that is
NOT also present in current main." This was checked file by file, with two
confounds controlled for:

- CRLF noise. The Windows clone checks out with CRLF line endings; the WSL
  checkout uses LF. A naive `diff` reports every line of an untouched file as
  "different." All comparisons below use `diff --strip-trailing-cr`.
- Staleness vs novelty. A file differing because the Windows clone predates a
  later fix is not novel work; a file differing because the Windows clone has
  content main lacks IS novel work. Each diff was inspected for which
  direction the difference runs (`<` = only in clone, `>` = only in main).

### 3.1 Untracked files/dirs (41 entries)

Every untracked entry's path already exists by name in main. Content compare:

| Result | Count | Explanation |
|---|---|---|
| Byte-identical to main | 29 of 33 comparable files | Already fully incorporated |
| Differs — clone is the OLD, pre-fix version | 4 (`CLAUDE.md`, `bench_evalmult_cpp_vs_rust.cpp`, `bench_keyswitch_cpp_vs_rust.cpp`, `test_keyswitch_core_native_diff.cpp`) | All differ for exactly one reason: `numPart=2` (invalid, violates P>=Q_part_max) vs main's corrected `numPart=4` — Phase 10, EXPERIMENT_LEDGER.md entries #42/#43. CLAUDE.md differs only by missing later tooling instructions (graphify, session-start checklist), not CKKS code. |
| `docs/` | identical | recursive diff: no output |
| `research/` | main strictly adds more | every delta is "only in main" (newer docs: PHASE25_RECOVERY.md, BOOTSTRAP_GPU_RESIDENCY_AUDIT.md, etc., and 4 living docs that are newer in main as expected) |
| `rust/` | main strictly adds more | rotation.rs, rotation_bridge.cpp, tests/, Cargo.lock, target/ exist only in main; files present in both (codec.rs, evalmult.rs, keyswitch.rs, etc.) differ by 569, 870, 223 lines respectively — ALL in the "main-only" direction except a handful of clone-only lines that are the SAME pre-Phase-10 numPart=2 issue, now resolved, plus a #[ignore]d test block documenting an EvalMult relin investigation (entry #30/#38) that was long since resolved (Phase 25's own report: "the real encrypted EvalMult AND key-switch round trips both PASS"). |

### 3.2 Tracked-modified files (21 entries), diffed against CURRENT main

| File | Real diff (CRLF-stripped) | Verdict |
|---|---|---|
| bench/bench_evalmult_resident.cpp | 0 | identical |
| include/intt.h, include/ntt.h, include/rns_arith.h | 0 | identical |
| include/automorph.h, include/keygen.h, include/keyswitch_resident.h, include/stage_acc.h | 0 | identical |
| kernels/intt_ct.cu, kernels/ntt_ct.cu, kernels/rns_arith.cu, kernels/stage_acc.cu | 0 | identical |
| src/evalkeygen.cpp, src/keyswitch.cpp | 0 | identical |
| tests/test_keyswitch.cpp, tests/test_boot.cpp, tests/test_polyeval.cpp | 0 | identical |
| tools/hybrid_security.py | 0 | identical |
| src/keyswitch_resident.cpp | 41 lines, 100% main-only | main added the centralized P>=Q_part_max validation (ks_params_validate.h, Phase 10, entry #42/#43) that the clone's stale edit lacks |
| src/rotate.cpp | 46 lines, 100% main-only | main added Phase 25's thread_local concurrency fix (this session's own prior work), which postdates the clone |
| CMakeLists.txt | 305 lines, essentially all main-only (1 trivial clone-only line: `target_link_libraries(gpufhe PUBLIC CUDA::cudart)` vs main's differently-scoped linkage — a build-config triviality, not a feature) | main has accumulated ~82 commits of new targets |

Every tracked-modified file is either byte-identical to main or a pure subset
of main's current content. Zero files contain clone-only content that isn't
explained by the two already-tracked historical fixes (P>=Q_part_max
validation, Phase 25 thread_local).

## 4. Verdict: nothing to port

There is no uncommitted, novel, divergent source code anywhere in the Windows
clone's 65 dirty entries. The clone is a stale snapshot of a point in history
that:

1. Predates the P>=Q_part_max hybrid key-switch validation fix (Phase 10).
2. Predates the EvalMult relin resolution (entries #30/#38 resolved by Phase
   10's fix — the same root cause).
3. Predates 82 commits of subsequent work: NTT/INTT batching refinements,
   registered-memory (Phase 20-22), Graphify integration (Phase 21), bootstrap
   architecture (Phase 23-24), and Phase 25's recovery/concurrency/hoisting work.
4. Already contains, in its own COMMITTED history (not its dirty tree), the
   two commits the Phase 26 directive specifically named as prototypes — both
   of which are ancestors of main's current HEAD and have been for most of
   that 82-commit span.

### Reconciliation matrix

| Prototype | File/symbol | Already in main? | Depends on other WC changes? | Compiles? | Test exists? | Trustworthy? | Changes numerics/security? | Safe to port independently? | Disposition |
|---|---|---|---|---|---|---|---|---|---|
| Fused pt-mul-accumulate kernel | kernels/stage_acc.cu LaunchPtMulAcc | YES (commit a8ce687, ancestor of HEAD) | No | Yes (builds clean) | Yes, tests/test_stage_acc.cpp | Partially — single random invocation only, no multi-diagonal/boundary/concurrency coverage | No | N/A — already present | REJECTED as a port target (redundant); ACCEPTED as a validation target — see Mission 2 |
| Device automorphism kernel | kernels/automorph_eval.cu automorphism_eval_fast_device | YES (commit 6357991 and successors, ancestors of HEAD) | No | Yes | Yes, tests/test_automorph_eval_fast.cpp, tests/test_rot_resident.cpp | Yes — bit-exact gated, already used in Phase 25's residency audit | No | N/A — already present | REJECTED as a port target (redundant); ACCEPTED as a validation target — see Mission 3 |
| numPart=2 benchmark/test defaults | bench_evalmult_cpp_vs_rust.cpp, bench_keyswitch_cpp_vs_rust.cpp, test_keyswitch_core_native_diff.cpp | N/A — superseded | — | — | — | — | Would REINTRODUCE a known-invalid parameter (P < Q_part_max) | No — would be a regression | REJECTED |
| CMake gpufhe linkage scope | CMakeLists.txt | main has a different (newer) scope | — | — | — | — | No | Trivial, not worth porting | REJECTED — not a feature, no behavior difference identified |
| Everything else (docs, Rust bridge files, kernels) | — | YES, byte-identical or main strictly newer | — | — | — | — | — | — | NO ACTION |

No recovery branch or patch archive was needed: there is nothing to preserve
that main does not already have. The Windows clone itself was not modified —
all investigation was read-only diffing from Git Bash and the WSL UNC path.

## 5. What this changes about Missions 2-4

Since both named "prototypes" are already in main, Missions 2 and 3 become
validation of existing, already-merged code (is it correct? is it actually
wired into the bootstrap's hot path? does it genuinely remove host round
trips?) rather than porting exercises. This matches Phase 25's own residency
audit, which found:

- The fused kernel (LaunchPtMulAcc) exists and passes its own test, but
  tests/test_boot.cpp's actual per-diagonal hot loop still calls
  ct_mul_pt_host/ct_add_ct_host on the CPU (lines 599-600, 915-916, etc.) —
  the kernel is validated but not wired into the bootstrap.
- The device automorphism kernel (automorphism_eval_fast_device) IS wired
  into rotate_ct_resident, which is bit-exact gated and already measured.

This reframes the next-highest-value work precisely: Mission 2 should extend
test_stage_acc.cpp's coverage to the directive's explicit bar (multiple
sizes/towers/diagonals, boundary values, repeated/concurrent execution), and
separately investigate whether wiring LaunchPtMulAcc into the bootstrap's
host loop is safe and valuable. Mission 4 (per-diagonal eval-key transfer) is
the one area Phase 25 only estimated statically from source — this round
should verify it with an actual instrumented measurement.
