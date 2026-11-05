## Session start (GOD PROMPT CKKS phases)

Before a substantial engineering task: verify Git state (`git log -3`,
`git status`, compare HEAD to what `research/NEXT_SESSION.md` claims),
then read `research/NEXT_SESSION.md` → `research/MASTER_STATE.md` →
`research/MIGRATION_MATRIX.md` (current state — NOT the full
`research/EXPERIMENT_LEDGER.md`, which is the detailed history behind
them). See `research/GRAPHIFY_GUIDE.md` for how the knowledge graph
below fits into this (navigation only — it is never the authoritative
record of a decision or experiment result).

<!-- scrum-orchestrator:begin -->
## Scrum-Orchestrator

This project has [Scrum-Orchestrator](https://github.com/samfrazerdutton/Scrum-Orchestrator)
installed: a Scrum-style AI engineering team with a real local task board.

- `/scrum-orchestrator` — start or continue an orchestration session for a feature/bug/refactor.
- Specialist subagents live in `.claude/agents/`: scrum-master, architect, frontend-developer,
  backend-developer, qa-engineer, security-engineer, code-reviewer, release-engineer.
- The task board is a local file at `.scrum-orchestrator/board.json`, mutated only through the
  `scrum-orchestrator` CLI. Treat it as the source of truth for task status — never describe a
  task as DONE/BLOCKED/etc. without having read or written that state through the CLI.
- Run `npx scrum-orchestrator serve` to open the pixel-art dashboard at http://localhost:4317
  (reads the same local files; nothing is sent off this machine).
<!-- scrum-orchestrator:end -->

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.
