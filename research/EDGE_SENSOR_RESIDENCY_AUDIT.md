# Edge AI / sensor-fusion GPU-residency audit (Phase 32, Work Package C)

**Scope note**: this is a numerical/memory-residency demonstration only.
Nothing here is a robotics, autonomy, or safety system, makes any
safety claim, or should be read as production sensor-fusion code. The
goal is narrow: show that intermediate state in a chained,
sensor-fusion-*shaped* numerical pipeline can stay GPU-resident across
many update steps instead of round-tripping through host memory each
step, and measure what that actually costs and saves on this project's
reference hardware (RTX 2060 Max-Q, 6 GiB).

## 1. Why a prototype was built (not deferred)

The god-prompt allows deferring the prototype if no clean library
primitive exists to anchor it to. At the time this was written, Phase
32's architecture audit (Work Package A) was running in parallel and
had not yet landed a stable buffer/context abstraction. Rather than
guess at an API that might not match what lands, this prototype is
written directly against raw CUDA, fully standalone (its own `.cu`
file, compiled directly with `nvcc`, **not** wired into `CMakeLists.txt`
or the main test suite) so it cannot collide with or affect the main
project build, and so it carries zero risk of GPU contention with any
other benchmark running concurrently in this phase. Building it (rather
than only writing research) was justified because: it was cheap (one
file, ~2.5s build+run), it directly answers the god-prompt's required
measurements (upload/per-update/chained-op/readback latency, transfer
volume, sync count, peak VRAM, CPU-reference correctness), and a real
number beats a projected one.

**Migration note**: once Work Package A's buffer/context primitive
lands, this prototype's three raw `cudaMalloc` buffers and one
`cudaStream_t` are exactly the shape that primitive should replace —
see section 6.

## 2. The pipeline

File: `research/edge_prototype/edge_sensor_residency_prototype.cu`
(self-contained, no project dependency).

- A resident synthetic point cloud, **1,000,000 points** (xyz, `float`,
  12 MB), uploaded to the GPU **once**.
- A resident 3×4 pose transform matrix (12 floats) living in `__device__`
  global memory, updated **in place on the GPU** each step from a small
  6-float delta (`dx,dy,dz,drx,dry,drz`) uploaded that step. The update
  is a first-order small-angle incremental rotation — **explicitly a
  simplified numerical approximation, not a validated SO(3)/quaternion
  integrator**; adequate for a residency demo at the small per-step
  angles used here (±0.002 rad), not a general pose-estimation
  algorithm.
- Each step, chained on the GPU with **no host round-trip between
  stages**: `pose_update` → `transform` (apply the resident pose to all
  1M points) → `ema_filter` (exponential moving average into a
  **resident** filtered buffer, so the filter's own state persists
  across steps without ever leaving the GPU).
- Every 50 steps (10 times over 500 steps), the filtered cloud's
  centroid (mean, not sum) is computed with a GPU reduction and the
  resulting 3 floats are read back to the host — the only data to
  leave the GPU after the initial upload.
- A single-threaded CPU reference repeats the identical sequence (same
  seed, same deltas) for correctness comparison.

Three device buffers (points, transformed, filtered) plus two tiny ones
(delta, centroid) are allocated once at startup and reused for all 500
steps — no per-step allocation.

## 3. Build and run

```bash
export PATH=/usr/local/cuda-13.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd ~/GPU-Resident-Library/research/edge_prototype
nvcc -O3 -std=c++17 -arch=sm_75 -o /tmp/edge_proto edge_sensor_residency_prototype.cu -lcudart
/tmp/edge_proto
```

Compiles and runs in ~2.6s total (`sm_75`, matching the RTX 2060's
compute capability 7.5). Exit code 0 iff GPU/CPU centroid agreement is
within 1e-3 (see section 5 for why that threshold, and the actual
measured error, which is far tighter).

## 4. Measured results (3 runs, RTX 2060 Max-Q, 6144 MiB, driver 595.97, CUDA 13.2, GPU otherwise idle)

| Metric | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Initial upload (12.01 MB) | 2.90 ms | — | — |
| Per-step latency, cold (step 0) | 0.600 ms | 0.646 ms | 0.533 ms |
| Per-step latency, warm median (steps 50–499) | 0.305 ms | 0.306 ms | 0.324 ms |
| Total GPU wall time, 500 steps | 162.4 ms | — | — |
| CPU reference, 500 steps | 2029.9 ms (4.06 ms/step) | — | — |
| GPU/CPU per-step speedup (warm) | 13.31x | 13.20x | 12.84x |
| H2D transfer volume | 12.01 MB (upload) + 500×24 B (deltas) ≈ 12.02 MB | | |
| D2H transfer volume | 10×12 B = 120 B | | |
| Sync calls | 501 (1 upload + 500 per-step, see note) | | |
| Peak net device allocation (`cudaMemGetInfo` delta) | 41.94 MB | | |
| Max abs error, GPU vs CPU centroid (mean, O(1–10) magnitude) | 5.96×10⁻⁸ (identical across all 3 runs) | | |

**Correctness**: the 5.96×10⁻⁸ figure is at float32 machine-epsilon
scale for values of this magnitude — the GPU's tree+atomic reduction
and the CPU's sequential summation accumulate in different orders, so
a tiny float32 rounding difference is expected and this is not evidence
of any defect. It was identical across all 3 runs (the reduction order
for a fixed grid/block configuration is itself deterministic on this
hardware).

**Honest synchronization note**: this run syncs the stream after
*every* step specifically so that **per-step latency could be measured
honestly** (per the god-prompt's "do not hide launch overhead"
instruction) — 501 explicit syncs for 500 steps. A real low-latency
residency pipeline would NOT do this: with everything already enqueued
to one stream in FIFO order, a caller only needs to synchronize when it
actually wants the next periodic readback (10 times here, not 500),
letting the GPU run ahead of the host the rest of the time. This
prototype deliberately over-synchronizes to produce a transparent
per-step number; it is not the recommended usage pattern. (Workload
size at which GPU wins at all is already apparent even with this
pessimistic sync-every-step accounting — see section 5.)

## 5. Performance crossover

Even with a synchronization point after *every* step (the worst case
for exposing launch/sync overhead), warm per-step GPU latency (~0.31 ms
for 1M points, 3 chained kernels) beats the single-threaded CPU
reference (~4.06 ms/step) by ~13x. The crossover point — where the GPU
stops winning — was not swept in this round (would require re-running
at several `N_POINTS` values, e.g. 10k/100k/1M/10M, to find where
per-step launch/sync overhead dominates the actual compute); at
1,000,000 points the GPU clearly wins even under pessimistic
per-step-sync accounting. This is a legitimate, moderate residency
demo, not a claim that any point-cloud size benefits from the GPU —
flagged as a concrete follow-up (section 7), not asserted here.

## 6. Limitations and what the real library would need

- **Raw CUDA, not the library**: this prototype intentionally does not
  use any GPU-Resident-Library primitive, because none existed yet at
  the point this was written (see section 1). If Work Package A lands
  a `DeviceBuffer`/`GpuExecutionContext`-shaped primitive, this
  prototype's three raw `cudaMalloc` buffers (points, transformed,
  filtered — each allocated once, reused every step, sized once at
  startup) and its one `cudaStream_t` are exactly the shape such a
  primitive should wrap. Porting it would be a small, mechanical
  follow-up, not a redesign.
- **Simplified pose math**: the first-order small-angle rotation update
  is adequate for a residency/throughput demo at small per-step angles
  and is explicitly NOT a validated attitude estimator — a real
  sensor-fusion system would need a proper quaternion or rotation-matrix
  renormalization step, which this deliberately omits to stay small.
- **No crossover sweep**: see section 5 — the point count at which GPU
  residency stops winning was not measured this round.
- **Single-stream, single-thread**: no concurrency story was tested
  here (Phase 31's established contract — independent contexts per
  thread, no shared mutable state without synchronization — would apply
  identically if this were extended to multiple concurrent sensor
  streams, but that extension was not built or tested).
- **No CUDA graph capture**: at 500 steps of identical kernel shape,
  this is a natural capture/replay candidate (same shape as the
  already-proven `keyswitch_resident` graph path from Phase 30/31) —
  not attempted this round, a concrete, well-scoped follow-up.

## 7. Recommended next step

If this workload is pursued further: (a) sweep `N_POINTS` to find the
real GPU/CPU crossover point on this hardware, (b) once Work Package
A's buffer/context primitive exists, port this prototype onto it as the
primitive's first non-FHE validation case, (c) capture the 3-kernel
per-step chain into a CUDA graph and measure replay speedup the same
way Phase 30 did for `keyswitch_resident`. None of these were necessary
to answer this round's actual question (can intermediate state stay
GPU-resident across a chained update sequence, measurably faster than
round-tripping) — that question is answered, with real numbers, above.
