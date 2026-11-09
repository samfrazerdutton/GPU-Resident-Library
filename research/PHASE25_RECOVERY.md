# Phase 25 — Recovery report (post-unexpected-shutdown)

Written at the start of the GOD PROMPT "CKKS PHASE 25" round, whose
opening mission was to recover the project after an unexpected computer
shutdown destroyed the previous terminal session. Everything below is a
measured observation from this session, not a carry-over claim.

## 1. Environment actually discovered

The agent session did **not** resume inside WSL. It started in **Git Bash
(MINGW64_NT-10.0-26200)** on the Windows host, working directory
`C:\Users\samca`. Both a Bash tool (MSYS) and a Windows PowerShell 5.1
tool were available. No assumption was made about which environment held
the repo; both filesystems were searched.

| Probe | Result |
|---|---|
| Host OS | Windows 11 Home 10.0.26200, `LAPTOP-TJUPVUQK` |
| Agent shell | Git Bash / MSYS (**not** WSL) |
| WSL `Ubuntu` | existed, state **Stopped** at session start |
| WSL `kali-linux` | Running (unrelated, **not touched**) |
| WSL `docker-desktop`, `docker-desktop-data` | Stopped (unrelated, **not touched**) |

`wsl -d Ubuntu -- echo` succeeded on the **first ordinary attempt**
(exit 0). No diagnostics, no retry loop, no service restart, no memory
reconfiguration, and no `.wslconfig` edit were needed or performed.
Phase 24's `HCS_E_CONNECTION_TIMEOUT` did **not** recur.

## 2. The two checkouts, and which one is authoritative

Two independent working copies exist. They were both inspected
**read-only** before anything was changed.

### Authoritative (used for all Git operations)

`/home/samfrazerdutton/GPU-Resident-Library` inside WSL `Ubuntu`.

| Property | Value |
|---|---|
| Branch | `main` |
| HEAD | `68e9f6571428644066307b19d02f3c308aa43f77` (`68e9f65`) |
| Tracking | `## main...origin/main` |
| Ahead / behind `origin/main` | **0 / 0** |
| Remote | `git@github.com:samfrazerdutton/GPU-Resident-Library.git` |
| Untracked entries | 9 |
| Tracked modifications | **0** |
| Staged modifications | **0** |

HEAD matches the Phase 24 handoff's reported `68e9f65` **exactly**. The
shutdown therefore did **not** corrupt, roll back, or partially apply any
commit: the repository is exactly where Phase 24 left it, and is fully
in sync with the remote.

### Non-authoritative (deliberately left untouched)

`C:\Users\samca\GPU-Resident-Library` on the Windows filesystem.

| Property | Value |
|---|---|
| Branch | `main` |
| HEAD | `f425cdd` |
| Working tree | **65** entries (20 tracked modifications + 45 untracked) |

`f425cdd` **does exist** in the authoritative repo's object database
(`git cat-file -t f425cdd` returns `commit`). So this clone is not on a
divergent *branch*; it is parked on an **older commit** carrying a large
volume of uncommitted experimental work (tower-batched fastcore,
`rotate_ct_device`, fused pt-mul-accumulate, and more).

**No Git operation of any kind was run against this clone beyond
read-only inspection, and none of its 65 working-tree entries were
read into, merged into, or copied out of the authoritative checkout.**
Per the governing directive it is not authoritative for Git, and its
uncommitted work is the user's to reconcile deliberately — not
something this session is entitled to absorb or discard.

## 3. Partial Phase 25 work recovered from the lost session

This is the most consequential recovery finding. The destroyed session
had already begun Phase 25, and its work survived on disk as untracked
files. Timestamps cluster tightly at **Nov 5 21:23-21:29**, i.e.
minutes before the shutdown.

| File | Lines | Timestamp | State as recovered |
|---|---|---|---|
| `research/benchmarks/bootstrap_preflight_lib.ps1` | 106 | Nov 5 21:23 | complete, coherent |
| `research/benchmarks/bootstrap_preflight.ps1` | 147 | Nov 5 21:23 | complete, coherent |
| `research/benchmarks/bootstrap_preflight.tests.ps1` | 104 | Nov 5 21:24 | complete; **verified passing this session** |
| `tests/test_boot_multi_input.cpp` | 646 | Nov 5 21:29 | substantially complete, **with real gaps (below)** |

Other untracked entries (`.claude/agents/`, `.claude/skills/scrum-orchestrator/`,
`.scrum-orchestrator/`, `tools/__pycache__/`, `build2/`) are pre-existing
user/tooling state and were preserved as-is.

Nothing was lost to the shutdown that was not recoverable from disk: no
tracked file was mid-write, and no commit was mid-creation.

### Verified immediately on recovery

`bootstrap_preflight.tests.ps1` was executed against the recovered
library as the first act after recovery — **12 passed, 0 failed, exit 0**.
These are pure mocked-input tests; no real system memory was touched.

### Gaps found in the recovered work (the shutdown caught it mid-task)

1. `tests/test_boot_multi_input.cpp` is **not registered in
   `CMakeLists.txt`**, so despite its header's claim of being
   "BUILD-VERIFIED ONLY", it had in fact **never been compiled**.
2. The driver emits **human-readable stdout only** — no machine-readable
   output, which the directive explicitly requires.
3. `CaseResult` records only a final error (`eFin`) and tower counts; it
   has **no per-stage fields**, so it cannot "identify exactly which
   bootstrap stage fails".
4. Its header references `tests/test_boot_multi_input_small.cpp` as the
   file that validates the loop/comparison logic at a runnable scale.
   **That file does not exist.**
5. `bootstrap_preflight.ps1` takes `BootstrapConfig` as free text that is
   never parsed, so it cannot apply the directive's workload-specific
   floor (n=131072 requires at least 6 GiB) — it only ever applies one
   fixed threshold.

These five gaps are the concrete starting point for this round's work.

## 4. Toolchain and hardware, as measured

| Component | Status |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 with Max-Q Design |
| Driver | 595.97 |
| VRAM | 6144 MiB total, **~5948-5955 MiB free** |
| CUDA | **nvcc 13.2, V13.2.51** |
| CMake | 3.28.3 |
| Rust | cargo 1.94.0, rustc present |
| CPU | 16 threads (`nproc`) |

Caveat worth recording: `nvcc` and `cargo` are **not** on the
non-login-shell `PATH`. A bare `wsl -d Ubuntu -- bash script.sh` reports
`nvcc: command not found`. Builds must use a **login shell**
(`bash -l` / `bash -lc`), which puts `/usr/local/cuda-13.2/bin` and
`~/.cargo/bin` on `PATH`. Five CUDA toolkits are installed
(12.8, 12.9, 13, 13.2, and the `cuda` symlink); `/usr/local/cuda`
resolving to 13.2 is what the login shell selects.

`build2/` is a pre-existing configured build tree:
`CMAKE_BUILD_TYPE=Release`, `CMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG`,
and **no `-G` anywhere in the cache** — the optimized build is clean of
device-debug flags, as the directive requires. Its newest artifacts
(`test_boot`, `test_ks_resident`, `test_rot_resident`, `libgpufhe.a`)
date to Oct 26, so the build tree is reusable and no full rebuild is
warranted.

## 5. Memory gate at recovery — NO-GO

Measured on the Windows host with `Get-CimInstance Win32_OperatingSystem`
(never WSL `free -h`, per the directive):

| Metric | Value |
|---|---|
| `TotalVisibleMemorySize` | 16,168,544 KB, about **15.42 GiB** |
| `FreePhysicalMemory` | 2,460,336 KB, about **2.35 GiB** |

The n=8192 bootstrap gate is **~10 GB free**, justified by
`tests/test_boot.cpp`'s own documented 8.12-8.32 GB historical peak RSS.
At 2.35 GiB free the decision is an unambiguous **NO-GO**, consistent
with Phase 24's own 1.8-3.6 GB observations — host memory pressure on
this machine is a persistent condition, not a transient one.

Note the contrast that makes WSL `free -h` unusable as the gate: inside
WSL, `free -g` reports **10 GB total / 10 GB free / 10 GB available** at
the very moment the Windows host has only 2.35 GiB actually free. WSL2's
`memory=11GB` allocation in `C:\Users\samca\.wslconfig` is a *ceiling*,
not a reservation; the guest cheerfully reports headroom the host cannot
honor. This is precisely why the preflight harness reads
`Win32_OperatingSystem` and nothing else.

**Consequence for this round:** the full n=8192 bootstrap is NOT executed
and no claim of its runtime success or failure is made. Work proceeds on
everything that does not require it — exactly as the directive instructs.

## 6. Protective constraints honored

Not run, not once, in any form: `git reset --hard`, `git clean`, any
deletion of untracked files, any force-push, any WSL distribution or
VHDX deletion, any `wsl --shutdown`, any Windows restart, any
termination of a user application to free memory, and any Git write
against the Windows clone. No cryptographic parameter (`h=64`, sigma=3.19,
the modulus chain, `KROT_CAP`, `KSC_CAP`, `P >= Q_part_max`) was read for
modification. Unrelated projects and all `.claude` directories are
untouched.

## 7. Recovery status

**Recovery is complete.** The authoritative checkout is confirmed intact
at `68e9f65` = `origin/main`, every pre-existing user change is
preserved, the development environment is usable (CUDA 13.2 + GPU
present and queryable, Rust present), and the lost session's partial
Phase 25 work has been found, read, and in the preflight harness's case
already re-verified. The single hard blocker is host memory, which gates
one specific deliverable (full-bootstrap execution) and nothing else.
