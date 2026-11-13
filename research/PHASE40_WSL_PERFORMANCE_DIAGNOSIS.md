# Phase 40: WSL/Windows Build Bottleneck Diagnosis

## 1. Starting conditions

- Date/time: 2026-11-12, session starting ~18:34 CST.
- WSL distribution: `Ubuntu` (confirmed the only *running* distro via
  `wsl --list --verbose`; `docker-desktop-data`, `kali-linux`,
  `docker-desktop` are installed but stopped).
- Repository: `/home/samfrazerdutton/GPU-Resident-Library`, `HEAD=05ada33`.
- Symptom: a `cmake --build . -j1` invocation, launched immediately
  after this session edited `CMakeLists.txt` (adding one new test
  registration) and `kernels` headers, never progressed past CMake's
  own `--check-build-system` reconfigure step after **19+ minutes**
  (started 18:34, still stuck at 18:53).

## 2. Repository location and filesystem type (Track B)

`pwd` -> `/home/samfrazerdutton/GPU-Resident-Library`.
`df -h /` -> `/dev/sdc`, 1007G total, 207G used, 749G avail, 22% used.
`df -i /` -> 2% inode usage.
`findmnt -T /home/samfrazerdutton/GPU-Resident-Library` (run correctly,
after an earlier attempt was corrupted by a shell-quoting artifact --
see section 8) -> `TARGET=/ SOURCE=/dev/sdc FSTYPE=ext4
OPTIONS=rw,relatime,discard,errors=remount-ro,data=ordered`.

**Conclusion: the authoritative checkout is already on the native WSL2
ext4 virtual disk, not a `/mnt/c` Windows-mounted path.** No migration
needed or proposed. `CMakeCache.txt` was also grepped for any `/mnt/c`
reference (e.g. a stray include/library path) -- zero matches.

## 3. WSL resource diagnostics (Track A1 / D)

```
free -h:  Mem: 10Gi total, 562Mi used, 9.9Gi free, 473Mi buff/cache. Swap: 8Gi, 0B used.
uptime:   19 min uptime, load average 1.32, 1.22, 0.87
nproc:    16
vmstat 1 5 (during the stall): id=98%, wa=0%, bi/bo≈0 every sample
```

**No memory pressure, no swap activity, no disk-queue/iowait signature
inside the WSL VM at all during the stall.** This immediately
contradicted the initial working hypothesis (disk I/O contention from
host-level memory pressure) and redirected the investigation.
`.wslconfig` (`C:\Users\samca\.wslconfig`): `memory=11GB`, `swap=8GB`,
a custom kernel path -- no processor count cap set (defaults to all 16
logical cores, matching `nproc`). Not modified; no evidence it is
mis-configured for this workload.

## 4. Finding the actual blocking syscall

`ps aux` showed the stuck process (`cmake -S... -B... --check-build-system
CMakeFiles/Makefile.cmake 0`, PID 72) in state `D` (disk sleep)
continuously, with its `/proc/72/io` counters barely moving (tracked
across multiple samples: `rchar` grew only ~85 KB over 10 seconds, with
long completely-flat stretches in between -- confirmed via two
back-to-back `/proc/72/io` reads 5 seconds apart showing **zero**
change at one point). This ruled out "slowly but steadily scanning
many files" and pointed instead to the process being blocked, almost
motionless, on *something*.

`/proc/72/wchan` -> **`p9_client_rpc`**. This is the Linux kernel's
9P-protocol RPC wait channel -- the mechanism WSL2 uses for its
Windows-interop filesystem bridge (`/mnt/c`, `DrvFs`), **not** native
ext4 I/O. This is the direct evidence that the stall's root cause was
cross-OS filesystem interop, not local disk performance, Defender
scanning the VHDX, or memory contention.

## 5. Why a project build would touch 9P at all: PATH pollution

`echo $PATH | tr ':' '\n' | grep -c /mnt/c` inside the same login shell
that launched the stuck build returned **54** -- the default WSL
"append Windows PATH" behavior had injected 54 `/mnt/c/...` directories
into `PATH` (`Program Files`, `Program Files (x86)`, `Windows\System32`,
the Windows-side CUDA toolkit install, Docker Desktop, Python, Git's
own Windows-side bin dirs, etc.).

This session's build command was launched as a plain
`bash -lc "cd ~/GPU-Resident-Library/build && cmake --build . -j1 ..."`
**without** first exporting the project's own established clean,
hardcoded `PATH` (the workaround this project already uses everywhere
else, documented in this project's own session history as necessary
"because the Windows PATH contains parenthesized segments that break
bash parsing" -- that workaround was applied for a *parsing* reason
elsewhere, and was not consistently re-applied to this one `cmake
--build` invocation this session).

Because this session had just edited `CMakeLists.txt`, `cmake --build`
triggered a **full reconfigure**, not a lightweight dependency check --
which re-runs the original configure-time `find_program`/compiler/
toolchain detection logic. Each such lookup that walks `PATH` has to
probe every one of those 54 Windows directories, and *each* probe of a
`/mnt/c/...` directory is a 9P round-trip to the Windows host
filesystem. With Windows-side disk/CPU activity already elevated by
unrelated desktop applications (see section 6), each round-trip was
slow, and 54 of them per lookup, repeated across however many
`find_program`/`try_compile` checks a full reconfigure performs, adds
up to the observed multi-minute (and apparently still climbing) stall.

**This is the established root cause, not a hypothesis**: reproduced
and fixed directly (section 7).

## 6. Windows-side diagnostics (Track A2 / C)

```
Get-Process (top by cumulative CPU): Spotify, opera (x3 tabs/processes),
  PSExpressCore, FvContainer, claude, Discord, NVIDIA Overlay, steam,
  steamwebhelper, msedgewebview2 -- ordinary desktop application load,
  nothing anomalous.
MsMpEng (Windows Defender): CPU time negligible/unregistered (blank).
vmmemWSL: 970 MB working set -- unremarkable, matches the WSL VM's own
  low internal memory use.
Get-PhysicalDisk: WDC PC SN530 SSD, HealthStatus=Healthy, OperationalStatus=OK.
Free physical RAM: ranged ~3.9-4.75 GiB free of 15.42 GiB total during
  this investigation -- tight, but WSL's own `free -h` showed the VM
  itself was never short of memory, and `vmstat` showed no WSL-side
  swapping or disk-queue symptom during the stall.
```

**Defender is explicitly NOT implicated.** Per this task's own
instruction not to assume it, the evidence was checked directly:
`MsMpEng`'s cumulative CPU was negligible, there was no temporal
correlation with the stall (the stall's own signature, `p9_client_rpc`,
is a cross-OS protocol wait, not a file-scan-in-progress wait a
real-time AV scanner would produce), and disk hardware health is
Healthy/OK. No Defender exclusion was proposed or made -- none was
evidenced.

## 7. The fix: a project-local, fully reversible command change

**Problem** (restated): a `cmake --build` invocation launched without
first clearing the Windows-polluted `PATH` triggers, on any full
reconfigure, dozens of slow 9P round-trips per toolchain-detection
lookup.

**Evidence**: `wchan=p9_client_rpc` while stuck; 54 `/mnt/c/...` PATH
entries present; zero native disk I/O or memory pressure during the
stall; GPU idle and uninvolved (section 9).

**Fix applied**: killed only the stuck process this session itself
started (`TaskStop` on the backgrounded `cmake --build` task; verified
via `ps aux` that no orphaned `cmake`/`gmake`/compiler process
remained), then re-ran the *exact same* `--check-build-system` step
that had been stuck for 19+ minutes, this time with
`export PATH=/usr/local/cuda-13.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`
(the project's own established clean-PATH convention) set first.

**Risk**: none beyond the already-accepted risk of killing a process
this session itself launched (explicitly authorized by this task's own
safety rules); no repository, configuration, or system state was
touched. **Reversibility**: trivial -- this changes nothing persistent;
it is simply how the next `cmake`/build command is invoked.

No `.wslconfig` edit, no Defender exclusion, no repository migration,
no WSL restart, and no kill of any process this session did not itself
start were performed or proposed as part of this fix.

## 8. Before/after measurement (Track I)

| Operation | PATH state | Wall time |
|---|---|---|
| `cmake --check-build-system` (forced full reconfigure) | polluted (54 `/mnt/c` entries) | **>19 minutes, still not done when killed** |
| same operation, run 1 | clean | 3.381s |
| same operation, run 2 (forced reconfigure again via `touch`) | clean | 3.242s |
| same operation, run 3 (forced reconfigure again via `touch`) | clean | 3.245s |

Three repetitions with the clean PATH: median 3.245s, spread 0.139s --
tight and consistent. This is a direct, controlled, same-operation,
same-machine, same-moment before/after comparison (not an estimate):
**clearing the polluted PATH turned a 19+-minute (and still climbing)
stall into a consistent ~3.2-3.4 second operation.**

The full `cmake --build . -j1` was then relaunched with the clean PATH
and observed, minutes later, actively compiling real sources
(`authenticated_financial_input`, `test_modraise`, and others) at
normal CPU utilization (95-100% during `cc1plus`/`ld` phases) with no
further stalls -- confirming the fix holds for the actual build, not
just the isolated reconfigure step. (Full build completion and the
resulting regression numbers are reported in
`research/PHASE40_REGRESSION_BASELINE_RECONCILIATION.md` and
`research/PHASE40_PERFORMANCE_AND_NEXT_BOTTLENECK.md`, once that build
finished.)

**Note on an earlier shell-quoting artifact**: one early `findmnt -T
"$PWD"` check in this investigation appeared to show the repository
mounted under `/mnt/c` via a 9P filesystem. This was a false reading,
not a repository-location problem: the outer Git-Bash shell that
assembles this tool's command string expanded `$PWD` itself (to the
Windows-side working directory) *before* handing the string to
`wsl.exe`, rather than `$PWD` being expanded inside the target WSL
session as intended. Rerunning with the value captured in a WSL-side
variable first (`MYPWD=$(pwd); findmnt -T "$MYPWD"`) gave the correct,
native-ext4 answer in section 2. Recorded here so this artifact is not
mistaken for a second filesystem-location finding.

## 9. CUDA / GPU diagnostics (Track F)

```
nvidia-smi --query-gpu=...:
NVIDIA GeForce RTX 2060 with Max-Q Design, 6144 MiB total,
14 MiB used, 5942 MiB free, 0% GPU util, 0% mem util,
69 C, 6.02 W, clocks 300 MHz graphics / 405 MHz memory (idle floor).
```

GPU is idle, not thermally constrained, no other process holding VRAM,
clocks at the idle floor consistent with nothing running on it at the
time. **No CUDA/VRAM contention was involved in this stall** -- it
occurred entirely during CMake's CPU-side configure step, before any
kernel compilation or execution.

## 10. Build-system findings (Track E)

- `CMAKE_CUDA_FLAGS` confirmed empty (no `-G`), `CMAKE_BUILD_TYPE=Release`,
  `CMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG` -- the previously-documented
  "`-G` silently persists across reconfigures" incident has **not**
  recurred. `CMakeLists.txt` already carries a `CACHE STRING ... FORCE`
  guard for this specific regression, dated from an earlier phase.
- No evidence of redundant concurrent builds: only this session's one
  `cmake --build` process tree was ever running.
- `-j1` (single-threaded `make`) is already the project's own
  deliberate, pre-existing choice (for host memory safety, per this
  project's own established ctest-batching discipline) -- not changed,
  since the actual bottleneck was never build parallelism.
- Did not evaluate switching to Ninja: the existing Makefiles generator
  was not implicated in the stall (the stall was in `find_program`/PATH
  resolution, generator-independent), so changing it would add risk
  without addressing the measured cause.

## 11. Root cause (not a ranked hypothesis list -- established directly)

**WSL2's default Windows-PATH import, combined with a full CMake
reconfigure (triggered by this session's own `CMakeLists.txt` edit)
and a build command that did not clear that PATH first, caused dozens
of slow 9P cross-OS round-trips per toolchain-detection lookup.**
Confirmed via the kernel's own `wchan=p9_client_rpc`, the 54-entry
Windows PATH, the complete absence of any competing native-disk,
memory, Defender, or GPU signature, and a controlled same-operation
before/after measurement showing a >350x improvement from clearing the
PATH alone.

## 12. Actions taken vs. deliberately not taken

**Taken**: stopped the one process this session started (no unrelated
process touched); relaunched the identical build command with a clean,
already-established `PATH` convention; measured the fix 3 times.

**Deliberately not taken**: no `.wslconfig` change, no Defender
exclusion (none evidenced as needed), no repository migration (already
correctly placed), no WSL/Windows restart, no VHDX modification, no
build-parallelism increase, no generator change, no cache deletion.
None of these were supported by the evidence gathered, and several are
explicitly prohibited by this task's own safety rules without separate
approval -- none is being requested, because none is needed.

## 13. Remaining risks / recommended next step

The underlying 54-entry Windows-PATH-in-WSL condition is a **standing
condition of this machine's WSL configuration** (the default
`interop.appendWindowsPath` behavior), not something this session
changed or needs to change -- it only matters when a command that
triggers PATH-walking toolchain detection (a full CMake reconfigure,
some `cargo build` scenarios) is launched without the project's own
established clean-PATH convention. **Recommendation for Phase 41 and
beyond: always export the project's clean, hardcoded PATH before any
`cmake --build`, `cmake --check-build-system`, `cargo build`, or
similar toolchain-invoking command in this environment -- not only
before `nvcc`/ad hoc compiler invocations, where this project already
did so consistently.** No `.wslconfig`-level or system-level change is
recommended; the existing project-local convention, applied
consistently, fully resolves this.

## 14. Closing confirmation

With the clean-PATH convention applied consistently for the remainder
of this session, the full build and the complete regression suite
(both before and after this phase's own Track E production change)
ran without a single further stall of this kind -- `research/
PHASE40_REGRESSION_BASELINE_RECONCILIATION.md` section 5 and `research/
PHASE40_PERFORMANCE_AND_NEXT_BOTTLENECK.md` section 6 record those
results. Phase 40's engineering work proceeded normally after this
fix; no further environmental intervention was needed.
