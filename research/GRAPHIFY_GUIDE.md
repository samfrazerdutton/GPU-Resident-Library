# Graphify setup, usage, and validation (Phase 21)

GOD PROMPT "CKKS PHASE 21". This is the dedicated setup/usage guide for
the Graphify code-intelligence integration, installed this phase.
**This document explains the tool and how to use it. It is not an
authoritative record of CKKS project state** — that remains
`research/NEXT_SESSION.md`, `research/MASTER_STATE.md`,
`research/MIGRATION_MATRIX.md`, and `research/EXPERIMENT_LEDGER.md`,
exactly as before this phase. See "What the graph is and is not" below
for the precise division of responsibility.

## What Graphify is

[Graphify](https://github.com/Graphify-Labs/graphify) (PyPI package
`graphifyy` — note the double-y; unrelated `graphify*` packages exist)
turns a codebase into a queryable knowledge graph via deterministic
tree-sitter AST parsing. It ships a `/graphify` skill for Claude Code
and 20+ other AI assistants, plus a standalone `graphify` CLI. It is
explicitly NOT a vector store or a semantic-search replacement — every
edge is either `EXTRACTED` (read directly from source) or `INFERRED`
(derived, with a confidence score), and the tool keeps that
distinction visible in every answer.

**Installed version**: `graphify 0.9.102` (the project's README calls
this release line "v8"; trust the CLI's own `--version` output, not a
README's possibly-stale label, per this phase's own `graphify
install --help` vs. top-level `--help` discrepancy below).

**Installation method**: `pipx install graphifyy` (isolated install;
`uv` was not present on this machine, so `pipx`, the documented
alternative, was used instead).

## Installation and integration performed this phase

```bash
pipx install graphifyy                       # isolated install
graphify install --project --platform claude # project-scoped, NON-strict
graphify hook install                        # post-commit/checkout/merge/rewrite auto-rebuild
```

**Deliberately NOT used**: `--strict`. `graphify install --help`
documents `--strict` as blocking the first raw `Read`/`Grep` per
session unless recently oriented with `graphify query`. Per this
phase's own directive ("start with non-blocking guidance... a stale
graph must never... prevent access to the authoritative source
code"), strict mode was never enabled. Verified empirically: `echo
"{}" | graphify hook-guard search` / `... read` both exit 0 with no
output in this installation — the PreToolUse hooks Graphify
registered in `.claude/settings.json` are advisory-only as installed.
If strict mode is ever wanted, `graphify install --project --strict`
can be re-run, but test it on a throwaway branch first — it changes
Claude Code's own tool-call behavior, not just this project's docs.

**What got installed, and what did not get touched**: this repo
already had a `CLAUDE.md` with a Scrum-Orchestrator integration
(`<!-- scrum-orchestrator:begin/end -->` markers) and a `.claude/`
tree with 8 agent files and a `scrum-orchestrator` skill. The install
was verified, by diffing a backup taken immediately before running
`graphify install`, to be purely additive: it appended one `## graphify`
section to the end of the existing `CLAUDE.md` (the Scrum-Orchestrator
block is untouched, byte-for-byte) and added only new files
(`.claude/skills/graphify/`, `.claude/CLAUDE.md`, `.claude/settings.json`)
alongside, never inside, the existing agent/skill directories.

## Generating and refreshing the graph

```bash
# Full extraction (first time, or after --purge). Local-only, zero LLM
# calls, zero external network use for code -- GRAPHIFY_BACKEND=none
# is explicit and deliberate (see "Privacy decision" below).
GRAPHIFY_BACKEND=none graphify extract .

# Regenerate GRAPH_REPORT.md/graph.html after extraction (or after
# changing the clustering): --no-label avoids any LLM community-naming
# call (communities render as "Community N" placeholders instead of
# semantic names -- see "Known gaps" below).
graphify cluster-only . --no-label

# Incremental refresh after code changes (no LLM, fast: ~6s for a
# 26-file delta on this repo; re-extracts only changed files, not the
# whole corpus).
GRAPHIFY_BACKEND=none graphify update .
```

**Auto-refresh**: `graphify hook install` registered FOUR git hooks
(`post-commit`, `post-checkout`, `post-merge`, `post-rewrite`) plus a
union-merge driver for `graphify-out/graph.json` itself (so two
branches that both touched the graph don't produce an unreadable merge
conflict). Read before trusting: the installed `post-commit` hook
launches the rebuild as a **detached background process** — `git
commit` returns immediately, never blocked by a graph rebuild; it
skips entirely during an in-progress rebase/merge/cherry-pick; it has
a 600s timeout with proper subprocess cleanup; and it logs to
`~/.cache/graphify-rebuild.log`, never to the commit's own stdout.
This hook is **local to this checkout** (`.git/hooks/` is never
pushed) — another clone must run `graphify hook install` itself to
get the same behavior; the committed `graphify-out/graph.json` means
they are not blocked from having *a* graph, just from having it
auto-refresh without that one command.

**Freshness verification**: `GRAPH_REPORT.md`'s own "Graph Freshness"
section records the commit the graph was built from (`Built from
commit: <hash>`) — compare against `git rev-parse HEAD` to check
staleness by hand. The project's own `CLAUDE.md` section (installed by
`graphify claude install`, see above) already instructs Claude to run
`graphify update .` after modifying code, which is the primary
freshness mechanism in practice; the git hooks are a safety net for
changes made outside a Claude Code editing session (a human edit, a
pull, a branch switch). **Known limitation, stated plainly**: nothing
here guarantees the graph is current if someone edits code without
committing and without asking Claude to update it — there is no
file-watch daemon running by default (`graphify watch <path>` exists
for that, but was not enabled this phase, to avoid adding a
long-running background process nobody asked for). If in doubt, run
`graphify update .` — it is fast and idempotent.

## Privacy decision: code-only, zero LLM backend

Graphify's code extraction (tree-sitter AST) is fully local and free.
Its documentation/PDF/image/video processing is NOT — it requires an
API key for Gemini, Claude, OpenAI, Kimi, DeepSeek, Azure, Bedrock, or
a local Ollama server. This phase's governing directive explicitly
required avoiding any cloud subscription or external paid service and
avoiding unintentional transmission of project materials to external
services. Accordingly, every extraction this phase ran set
`GRAPHIFY_BACKEND=none` explicitly (not merely relying on the absence
of an API key in the environment), and community labeling used
`--no-label` for the same reason. A local Ollama instance was checked
for and found NOT running (`curl localhost:11434` refused the
connection) — starting one up was judged more new infrastructure than
this task's own "prefer local-first, avoid heavyweight infrastructure
unless justified" instruction supports, given the fallback path
(below) already satisfies the project's actual needs.

**What this costs**: with no LLM backend, documentation/PDF/image
content does not get deep semantic extraction. See "Known extraction
gaps" below for exactly what this means in practice for this repo (it
is less total than the first scan message suggested — see below).

## What the graph is and is not (the four-layer memory architecture)

This project now has four distinct, deliberately non-overlapping
memory layers. Do not duplicate the same fact across more than one of
them — when they disagree, the resolution order is always: current
source/Git state > a re-run test or measurement > the original
experiment record in Layer B > update Layer A > refresh Layer C.

- **Layer A — authoritative project state** (unchanged by this
  phase): `research/NEXT_SESSION.md`, `research/MASTER_STATE.md`,
  `research/MIGRATION_MATRIX.md`. What commit/branch to start from,
  what is production vs. experimental, what must not change, the next
  recommended action. Concise by design — never copy the full
  experiment ledger in here.
- **Layer B — experimental evidence and decisions** (unchanged by
  this phase): `research/EXPERIMENT_LEDGER.md` and the per-phase
  sections of `research/TRANSFER_PROFILING.md`. Hypothesis, exact
  configuration tested, raw data location, results, confounders,
  decision, confidence, and whether a later phase superseded it (never
  silently collapsed into a false consensus — Phase 17's reversed
  conclusion about `mlock()` is still readable as its own entry,
  marked superseded, not deleted).
- **Layer C — navigable code graph** (new this phase):
  `graphify-out/graph.json` + the `graphify query`/`explain`/`path`/
  `affected`/`god-nodes` CLI. Answers "which functions implement this,"
  "what calls what," "what would this change affect" — structural
  navigation, never the authoritative record of WHY a decision was
  made or WHAT an experiment concluded. An `INFERRED` edge is a
  tool-computed guess with a confidence score, not a second source of
  truth for runtime behavior.
- **Layer D — concise session-start guidance** (new this phase): the
  `## graphify` section `graphify install` appended to `CLAUDE.md`,
  plus this guide. Tells a fresh session HOW to use layers A-C
  together, in under 10 lines, with links rather than copied content.

## Validation: 12 questions, tested against real evidence

Each question below was tested against the actual installed graph on
this repository (commit `f20fa882`, later re-run after the Phase 21
commit) and cross-checked against source. "Graph-first" means the
answer came primarily from `graphify query`/`explain`/`path`;
"source-verified" means the specific file/line was then opened to
confirm the graph's claim.

1. **Current production encryption implementation?**
   Graph-first via `graphify explain "encrypt_host_shoup_pooled_barrett_batched"`:
   correctly surfaces the function (`src/encrypt_host_shoup_pooled_barrett_batched.cpp:L146`),
   that it is called (INFERRED, 0.85 confidence) by both
   `encrypt_host_shoup_pooled_barrett_prewarmed()` (Phase 16's
   fallback) and `encrypt_host_shoup_pooled_barrett_batched_registered()`
   (Phase 20's fallback), and lists every correctness test that calls
   it by name. Source-verified: matches `cpp_bridge/encdec_bridge.cpp`'s
   actual call (not itself graphed — see gap below).
2. **How does the Rust API reach the C++ encryption implementation?**
   Graph-first via `graphify explain "gpufhe_encrypt"`: ambiguous
   between `rust/crates/ckks-cuda/src/codec.rs` (the `extern "C"`
   declaration, called from `codec::encrypt()`) and
   `rust/crates/ckks-cuda/cpp_bridge/encdec_bridge.cpp` (the C++
   definition) — correctly identifies BOTH sides of the FFI boundary
   as two same-named nodes in different files/languages. **Gap,
   honestly flagged**: the graph does not draw an edge BETWEEN those
   two nodes (cross-language FFI linkage by matching symbol name isn't
   resolved) — a reader must notice the matching names themselves.
3. **Which CUDA operations are involved in the real EvalMult pipeline?**
   Graph-first via `graphify explain "EvalMultFixtureCpp"` (a listed
   god node, 34 edges): surfaces `LaunchTensor`/`LaunchCombine`/
   `keyswitch_resident`/`rescale_resident_raw` as directly-called
   operations. Source-verified against `bench/bench_evalmult_e2e_breakdown.cpp`'s
   own `multiply_relin_rescale_breakdown()`.
4. **Which host-memory implementations are production vs. experimental?**
   Graph-first via `graphify god-nodes` + the community list in
   `GRAPH_REPORT.md`: the clustering ALREADY separates
   `encrypt_host_shoup_pooled_barrett_batched` (production),
   `..._batched_pinned` (Phase 15, unshipped), `..._prewarmed` (Phase
   16, opt-in), and `..._batched_registered` (Phase 20, opt-in) into
   four distinct communities, purely from code structure — it never
   labels which ONE is production, that fact lives only in Layer A/B.
5/6/7. **Phase 19/20 findings and the registered-path promotion
   decision?** **Not answerable from the graph at all** — these are
   narrative conclusions inside `research/TRANSFER_PROFILING.md`,
   which (see gap below) is present in the graph only as a
   lightweight reference node, not semantically indexed. This is the
   expected, correct division of labor: Layer B is authoritative here,
   not Layer C. Verified by reading `research/TRANSFER_PROFILING.md`'s
   Phase 19/20 sections directly.
8. **Which correctness tests establish real ciphertext behavior
   (not RNS-only)?** Graph-first via `graphify explain` on
   `encrypt_host_shoup_pooled_barrett_batched_registered` surfaces
   `tests/test_encrypt_host_shoup_pooled_barrett_batched_registered_matches_reference.cpp`
   by name; distinguishing "RNS-only" from "real ciphertext" tests
   by INTENT still requires reading the test (the graph has no "real
   vs. RNS-only" tag — a genuine gap, noted below).
9. **Key-switch constraints that must never be weakened?**
   Graph-first via `graphify explain "check_hybrid_keyswitch_params"`:
   finds the function, `ks_context_create()`'s dependency on it, and
   (via the lightweight doc-reference nodes) that `research/EXPERIMENT_LEDGER.md`
   and `research/MIGRATION_MATRIX.md` reference it near specific
   lines — enough to know WHERE to look, not WHAT the constraint says
   (that requires reading `src/ks_params_validate.cpp` directly).
10. **Which experiments were superseded?** Not graph-answerable
    (Layer B's own job — `research/EXPERIMENT_LEDGER.md` already
    marks superseded entries explicitly, e.g. entry #40/#41 vs. #42).
11. **Which test/benchmark targets run under CTest automatically?**
    **Confirmed gap**: `CMakeLists.txt` does not appear in the graph
    at all (`grep -c CMakeLists graphify-out/graph.json` → 0) — CMake
    is not one of Graphify's parsed languages/extensions on this
    install. The `gpufhe_test` (auto-runs under `ctest`) vs.
    `gpufhe_benchmark` (Phase 19's own addition, does NOT) distinction
    must be read directly from `CMakeLists.txt` — grep for
    `gpufhe_test(`/`gpufhe_benchmark(` calls; there is no graph
    shortcut for this question.
12. **What should the next engineering session do?** Correctly NOT a
    graph question — `research/NEXT_SESSION.md`'s "READ THIS FIRST"
    section, unchanged in authority by this phase.

### Measured, not estimated: token-reduction and timing

`graphify benchmark` (the tool's own built-in measurement, run on this
actual repo, not a synthetic one) reported, for 3 representative
questions against the full corpus (~110,550 words / ~147,400 tokens
naive) vs. the graph (2,211 nodes / 4,892 edges at the time of that
run): **91.2x fewer tokens per query on average** (per-question:
91.8x / 90.2x / 91.6x). A single `graphify query` against this repo's
graph took ~0.5s wall-clock. An incremental `graphify update .` after
26 changed files took well under a minute (re-extracted only the
changed files via its own AST cache, not the full 234-file corpus).
These are the tool's own measured numbers on this repository, not a
self-reported estimate.

## Known extraction gaps (honestly documented, not hidden)

- **No LLM backend configured** (deliberate, see "Privacy decision"
  above): community names are placeholders ("Community 37," not a
  semantic name); no deep semantic cross-file "relates to" inference
  beyond what local AST calls/imports/inherits already provide.
- **CMake is not indexed.** `CMakeLists.txt` produces zero graph
  nodes. Questions about CTest registration, build targets, or the
  `gpufhe_test`/`gpufhe_benchmark` split must be answered by reading
  `CMakeLists.txt` directly.
- **6 CUDA kernel files show partial AST extraction**
  (`kernels/basis_convert.cu`, `intt_ct.cu`, `ks_fastcore.cu`,
  `ntt_ct.cu`, `rns_arith.cu`, +1 more): tree-sitter's C++ grammar hit
  a syntax error region (CUDA-specific syntax it does not fully
  parse, e.g. kernel launch syntax or `__global__`/`__device__`
  qualifiers in some position) and extracted a partial symbol set
  from the surrounding, parseable code. Do not assume these files'
  graph entries are complete — read the source for anything
  kernel-launch-configuration-related in these specific files.
- **13 header files yield zero symbols** (e.g. `include/automorph.h`,
  `include/basis_convert.h`, `include/intt.h` — mostly thin
  declaration-only headers where the extractor's own model has
  nothing to extract). Expected, not a parsing failure — thin headers
  genuinely have little to extract.
- **Markdown research docs are present only as lightweight reference
  nodes**, not deeply indexed. `research/*.md` files DO appear as
  graph nodes (confirmed: `EXPERIMENT_LEDGER.md`, `MASTER_STATE.md`,
  `MIGRATION_MATRIX.md`, `NEXT_SESSION.md`, `TRANSFER_PROFILING.md`,
  and 10 other research docs all have at least one node), reached
  from code comments that cite them by path, and some have a nearby
  text snippet as a node label (zero LLM cost, confirmed via
  `GRAPH_REPORT.md`'s own "Token cost: 0 input · 0 output" line — this
  is local extraction, not semantic understanding). They are NOT
  semantically searchable for their actual findings/conclusions —
  that is Layer B's job, not the graph's.
- **Cross-language FFI edges are not resolved** (see validation
  question 2 above): a Rust `extern "C"` declaration and its C++
  definition of the same symbol appear as two separate, same-named
  nodes in different files, never linked by an edge.
- **Free-text `graphify query` can mis-seed on confusingly-named local
  symbols.** `graphify query "what is the current production
  encryption implementation"` seeded on `A_Production` — an enum-like
  value inside `bench/bench_encrypt_registered_lifecycle.cpp` that
  merely contains the word "production" — rather than the actual
  `encrypt_host_shoup_pooled_barrett_batched` function. `graphify
  explain "<exact symbol name>"` is far more reliable for "what is
  the real implementation of X" questions; reserve free-text `query`
  for genuinely relational questions ("what connects X to Y") and
  verify its seed nodes before trusting the answer.

## How Claude Code should use this (Layer D, kept short deliberately)

The `## graphify` section already in `CLAUDE.md` (installed by the
tool itself) covers the moment-to-moment instruction. The fuller
procedure, for a substantial engineering task:

1. Verify Git state (`git log -3`, `git status`, compare to
   `research/NEXT_SESSION.md`'s claimed starting commit).
2. Read `research/NEXT_SESSION.md` → `MASTER_STATE.md` →
   `MIGRATION_MATRIX.md` (Layer A) for current state, not the full
   ledger.
3. Check graph freshness: `GRAPH_REPORT.md`'s "Built from commit" line
   vs. `git rev-parse HEAD`; run `graphify update .` if stale or
   absent.
4. `graphify query`/`explain`/`path` for the relevant subsystem BEFORE
   broad `Grep`/`Read` — but verify a graph claim against the actual
   source before relying on it for a code change, every time.
5. Check `research/EXPERIMENT_LEDGER.md` (Layer B) for prior attempts
   at this exact change — do not repeat a disproven experiment.
6. Form a plan; implement; test.
7. Update Layer A/B as appropriate; run `graphify update .`
   (immediate, local, no LLM cost) so the graph does not silently
   drift from the code it describes.

## How to disable or remove the integration

```bash
graphify claude uninstall   # removes the CLAUDE.md section + PreToolUse hooks
graphify hook uninstall     # removes the 4 git hooks + merge driver
graphify uninstall --purge  # removes the skill AND graphify-out/ entirely
pipx uninstall graphifyy    # removes the tool itself
```

None of these touch `research/*.md`, `CLAUDE.md`'s pre-existing
Scrum-Orchestrator section, or `.claude/agents/`/`.claude/skills/scrum-orchestrator/`.

## If Graphify is missing or broken in a future session

The skill/hooks degrade to no-ops if `graphify` is not on `PATH` (the
hook scripts probe for it and print a one-line warning to their own
log, never fail the commit). If `graphify query` etc. are unavailable:
fall back to `Grep`/`Read` as before this phase — nothing about this
integration is load-bearing for actually doing CKKS engineering work.
Re-install with `pipx install graphifyy` (or `pipx reinstall graphifyy`
if already present but broken) and re-run the `install`/`extract`
commands at the top of this guide.

## Phase 22 field validation: used for real engineering work, not just the validation suite

Phase 21's validation suite tested Graphify against synthetic
questions. Phase 22 is the first phase to actually USE it to do real
engineering work (wiring Phase 20's CUDA-registered host memory into
Rust), and the results reconfirm this guide's own earlier findings
under real conditions rather than a scripted test:

- **Genuinely saved effort**: `graphify explain
  "encrypt_host_shoup_pooled_barrett_batched_registered" --all` (one
  command) confirmed immediately that no Rust/FFI caller existed yet
  for the registered path -- exactly the fact needed before writing
  any FFI code, and faster than a multi-file `grep` across `rust/`.
- **The documented cross-file-call-edge gap recurred, for real work
  this time**: `graphify explain "gpufhe_encrypt" --all` on the
  EXISTING FFI function returned mostly documentation
  cross-references, not its actual call target
  (`encrypt_host_shoup_pooled_barrett_batched`) -- the real answer
  still had to come from reading `cpp_bridge/encdec_bridge.cpp`
  directly. This is the SAME limitation this guide's own validation
  question 2 already flagged; it recurring on a real task (not a
  scripted question) is useful independent confirmation that the gap
  is real and worth remembering, not a validation-suite artifact.
- **The graph was refreshed 3 times mid-phase** (after the C++/Rust
  source changes, after an unrelated small Rust API addition, after
  a `CMakeLists.txt`-only change) and verified correct each time --
  new symbols appeared, connected to their real callers, exactly as
  expected; the `CMakeLists.txt` change correctly produced NO graph
  change (confirming the CMake gap is still accurately reported, not
  silently "fixed" by coincidence).
- **Not overused**: `graphify query` (free-text) was not reached for
  at all this phase -- every question was specific enough for
  `explain`/`path` directly, consistent with this guide's own
  recommendation to prefer exact-name lookups. Reading the actual
  source (`registered_host_buffer.h`, the registered-encrypt `.cpp`,
  `encdec_bridge.cpp`) remained necessary before writing any code, as
  this guide always said it would be -- the graph oriented the
  investigation, it did not replace reading the code.
