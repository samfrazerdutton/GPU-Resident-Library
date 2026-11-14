# Phase 42: Performance Audit

## 1. Recovering actual Phase 41 state (mandatory, done first)

This god-prompt's own framing assumed a completed "Phase 41" with
specific reported findings (ModDown ~1.46-1.48x, RescaleFuse
~1.92-1.95x at n=131072, both a wash at n=8192; a WSL/PATH build-
bottleneck fix). Verified directly, not assumed:

```
git rev-parse HEAD          -> 1357624... (== origin/main)
git log -12 --oneline        -> newest commit is "Phase 40: documentation,
                                 ledger entry #70..."; no Phase 41 commit
                                 anywhere in history
git reflog -20                -> no Phase 41 entry, no dangling commit
git stash list                -> empty
git branch -a                 -> only main / origin/main
find research -iname '*PHASE41*' -> no matches
```

**Conclusion: Phase 41 never happened.** The repository's actual state
at the start of this session was exactly Phase 40's own final commit.
The specific numbers this god-prompt attributes to "Phase 41" are
Phase 40's own real, already-documented Track C findings
(`research/PHASE40_PERFORMANCE_AND_NEXT_BOTTLENECK.md` section 2) --
re-checked against that document and confirmed to match verbatim. The
WSL/PATH fix is also Phase 40's own work
(`research/PHASE40_WSL_PERFORMANCE_DIAGNOSIS.md`), reused (not
re-discovered) this session.

**Classification of the one piece of carried-forward unfinished work**
(`rescale_resident_raw`'s analogous per-tower launch loop, which Phase
40 explicitly identified and explicitly deferred): **(4) Deferred**,
with a clear, evidence-backed recommendation already on record. This
phase completes it (section 3 below, full detail in
`research/PHASE42_IMPLEMENTATION.md`).

## 2. Environment (redetected, not assumed)

```
GPU: NVIDIA GeForce RTX 2060 with Max-Q Design, driver 595.97
VRAM: 6144 MiB total, 5942 MiB free, 0% util, 53C (idle, cold boot)
CUDA: 13.2.51 (nvcc), CMAKE_CUDA_FLAGS empty (no -G), CMAKE_BUILD_TYPE=Release,
      CMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG (re-verified from CMakeCache.txt)
cmake: 3.28.3
gcc: 13.3.0
Rust/cargo: 1.94.0
CPU: 16 logical cores, load average ~0.07 (idle) at session start
WSL: fresh VM boot (~2 min uptime at session start -- 2 days had passed
     since the prior session), /dev/sdc ext4, 10GiB allotted, 0 swap used
Windows host free RAM: fluctuated ~1.2-7.1 GiB through the session
```

**Clean PATH, reused from Phase 40's own established procedure**
(`research/PHASE40_WSL_PERFORMANCE_DIAGNOSIS.md` section 7), applied to
every `cmake`/`cargo`/`nvcc` invocation this session from the start:
```
export PATH=/usr/local/cuda-13.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```
(with `:/usr/lib/wsl/lib` appended specifically for `nvidia-smi`, and
`:/home/samfrazerdutton/.cargo/bin` for `cargo`). No build stalled this
session -- the one `cmake --build` performed with a stale CMakeCache
(before any source edit) took 8.4s; every subsequent reconfigure after
editing `CMakeLists.txt` took 3-10s. The WSL/PATH pathology Phase 40
diagnosed did not recur, confirming that fix's own recommendation
("always export this PATH before any toolchain command") is sufficient
on its own, invented nothing new.

## 3. Current bottleneck: measured, not assumed

Static call-graph review of `src/keyswitch_resident.cpp` (the
production resident key-switch/rescale path, already extensively
optimized across Phases 9, 30, 34, 38, 40) found exactly ONE remaining
structural launch-overhead pattern that Phase 40 itself had already
identified and flagged, but not implemented:
`rescale_resident_raw`'s per-tower loop issuing `LaunchSwitchModulus` +
`LaunchNTT_CT` (which itself issues ~5 internal launches at n=8192-32768,
per its own code comment: "collapsing 5 launches per tower into 5
total") + `LaunchRescaleFuse` -- three kernel TYPES, the middle one
multi-launch, repeated `towers-1` times, serially, reusing one shared
scratch buffer.

**Quantified before implementing** (isolated reproducer, not assumed):
at n=8192, Phase 40 already measured that collapsing ONE such
per-tower launch (ModDownCombine, a single-kernel-type analogue) into
one batched launch gives 3.88x-5.93x at sizeQl=5-8. Since
`rescale_resident_raw`'s loop multiplies THREE kernel types per
iteration (one of which is itself 5 launches), the expected
launch-count reduction here is larger in absolute terms, confirmed
directly in section 5 below once implemented (1.18x-3.89x isolated,
growing with `towers`, at the real production function -- not a proxy).

**Why this and not something else**: every other previously-optimized
area (NTT/INTT, encryption batching, ModDown, key switching, rotation
infrastructure, CUDA graph capture, cache synchronization, host/device
transfer) has already had at least one dedicated phase of attention,
confirmed by reading the relevant reports
(`research/PHASE34_PERFORMANCE_INVESTIGATION.md`,
`research/PHASE38_MULTI_CLIENT_THROUGHPUT.md`,
`research/PHASE39_FHE_TEXTBOOK_RNS_AND_BATCHED_ENCRYPTION.md`,
`research/PHASE40_PERFORMANCE_AND_NEXT_BOTTLENECK.md`). This specific
loop was the one Phase 40 itself named, with evidence, as the clear
next candidate, and no new evidence surfaced during this session's own
call-graph review to override that ranking.

## 4. Ranked candidates considered

1. **`rescale_resident_raw`'s per-tower loop (selected)** -- highest
   measured-expected value (direct extension of an already-proven
   technique, three kernel types collapsed at once, used by this
   project's entire rotation/bootstrapping/polynomial-evaluation/EvalMult
   call graph -- ~30 call sites), moderate implementation risk (no
   public signature change; internal state addition follows an
   already-established caching convention in the same file), low
   correctness risk given the enormous existing differential-test
   coverage across that call graph (see Track I below).
2. CPU/GPU overlap for independent host work -- explicitly NOT
   pursued: Phase 36's own thread-parallel-host-loop experiment is a
   confirmed regression at this project's granularity, and no new
   evidence this session suggests revisiting it.
3. Further RNG/arithmetic changes -- explicitly out of scope (RNG is
   security-sensitive; the Shoup kernels were already re-verified exact
   in Phase 40 with no further arithmetic change indicated).
4. A second new abstraction layer (e.g. generalizing the new
   `RescaleBatchWork` cache into a shared utility) -- explicitly NOT
   done: the project's own convention is to duplicate small per-file
   helpers rather than centralize them, and introducing a new shared
   abstraction for a single call site would be scope creep without
   measured benefit.

**Selected: #1.** Full detail in `research/PHASE42_IMPLEMENTATION.md`;
full results in `research/PHASE42_PERFORMANCE_RESULTS.md`.
