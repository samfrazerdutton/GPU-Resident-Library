# Bootstrap GPU-residency audit

GOD PROMPT "CKKS PHASE 25", Mission 4C. An inventory of what actually crosses
the PCIe boundary in the bootstrap, where the CPU/GPU round trips are, and which
of them are worth removing — with the evidence status of every claim stated.

## 0. Method, and the honesty boundary

This audit is **static**: every number below is derived by reading
`tests/test_boot.cpp`, `src/rotate.cpp`, `src/keyswitch_resident.cpp`,
`src/keyswitch.cpp` and `kernels/*.cu`, and by counting CUDA API calls per file.

It is **not** a profiler trace. The bootstrap itself could not be executed this
round: `tests/test_boot.cpp` has a documented 8.12–8.32 GB peak host RSS and
Windows free RAM measured 2.35–3.19 GiB all session, so
`research/benchmarks/bootstrap_preflight.ps1` returned NO-GO (logged in
`research/benchmarks/bootstrap_preflight_log.jsonl`). Nothing here is a measured
wall-clock attribution of the full bootstrap, and nothing here should be quoted
as one. Where a figure is a byte count derived from array dimensions, it says so;
where a figure was actually measured this round, it says that too and names the
command.

Counts below come from `grep` over each file for `cudaMemcpy*` direction flags,
`cudaDeviceSynchronize`/`cudaStreamSynchronize`, `cudaMalloc`, `cudaFree`, and
`<<<` kernel launches. They count **call sites**, not dynamic invocations — a
single call site inside the per-diagonal loop executes once per diagonal.

## 1. Static call-site inventory

| File | H2D | D2H | D2D | sync | cudaMalloc | cudaFree | kernel launches |
|---|---|---|---|---|---|---|---|
| `tests/test_boot.cpp` | 10 | 10 | 2 | 6 | 13 | 10 | 0 |
| `src/keyswitch_resident.cpp` | 19 | 0 | 4 | 1 | 16 | 6 | 0 |
| `src/rotate.cpp` | 10 | 4 | 3 | 3 | 8 | 5 | 0 |
| `src/keyswitch.cpp` | 9 | 6 | 0 | 4 | 15 | 9 | 0 |
| `src/keyswitch_hoist.cpp` | 9 | 0 | 1 | 1 | 7 | 3 | 0 |
| `src/encdec.cpp` | 7 | 5 | 0 | 2 | 6 | 2 | 0 |
| `src/evalkeygen.cpp` | 7 | 3 | 0 | 1 | 5 | 2 | 0 |
| `kernels/intt_ct.cu` | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| `kernels/ntt_ct.cu` | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| `kernels/stage_acc.cu` | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| `kernels/rns_arith.cu` | 3 | 0 | 0 | 1 | 3 | 1 | 2 |
| `kernels/automorph_eval.cu` | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| `kernels/ks_fastcore.cu`, `tensor.cu` | 0 | 0 | 0 | 0 | 0 | 0 | 2 each |
| `kernels/moddown.cu`, `rescale.cu`, `basis_convert.cu`, `switch_modulus.cu`, `automorph.cu` | 0 | 0 | 0 | 0 | 0 | 0 | 1 each |

Two things stand out. The kernels are clean — no transfers, no syncs, no
allocation, exactly as GPU-resident kernels should be. All the traffic lives in
the host-orchestration layer, and the single densest concentration is
`src/keyswitch_resident.cpp` with 19 H2D call sites and 16 `cudaMalloc`s.

## 2. Bootstrap structure, for scale

`tests/test_boot.cpp` at n=8192: `S = n/2 = 4096`, `lgS = 12`, `LST = 3`
linear-transform stages, `RST = lgS/LST = 4` butterfly stages merged per
linear-transform stage. A merged stage therefore has at most `2^RST = 16`
non-zero diagonals, which matches the n=131072 run logs in
`research/MEMORY_FEASIBILITY.md` reporting "diagonal 8 of 16".

So roughly **16 diagonals × 3 stages = ~48 diagonals per linear transform**, and
the bootstrap runs CoeffsToSlots and SlotsToCoeffs, with the S2C side's final
stage processing two sources (the real and imaginary accumulators) — call it
**~96–112 diagonal iterations per bootstrap**. `test_boot.cpp` prints the exact
`ndiagTot` at runtime; that value has **not** been observed this round, because
the test cannot run. Treat ~100 as an order-of-magnitude anchor, not a measured
count.

## 3. The per-diagonal hot loop, line by line

This is the structure that matters. From `tests/test_boot.cpp`'s stage loop
(`stageRotBegin` / `stageRotDiag` / `stageRotEnd` at lines 475–535, and the
per-diagonal body that follows):

```
stageRotBegin(R, cs0, cs1, ctw, keyAt(offs[0], ctw))     // ONCE per stage
    cudaMalloc x7 (grow-on-demand, towers*n*8 each)
    cudaStreamCreate
    ks_work_create
    cudaMemcpyAsync H2D x2                                // the ciphertext, once per stage

for each diagonal x:                                      // ~16 per stage
    encode_host(md, dd, n, dpt)                           // CPU: encode the diagonal
    pt_to_eval_host(dE, md, stw, n, ml, rl)               // -> xf_multi:
                                                          //    H2D 1.875 MiB
                                                          //    batched NTT launch
                                                          //    cudaDeviceSynchronize
                                                          //    D2H 1.875 MiB
    stageRotDiag(R, t0, t1, rotAmt, keyAt(...), stw, ...)  //
        cudaMemcpyAsync D2D x2                            //    2 x 1.875 MiB on device
        automorphism_eval_fast_device x2                   //    2 kernel launches
        ks_context_create(Krot, s)                        //    *** 60 MiB H2D, 30 cudaMalloc ***
        keyswitch_resident(...)                           //    kernels only (good)
        ks_context_destroy(C)                             //    30 cudaFree
        cudaMemcpyAsync D2H x3                            //    3 x 1.875 MiB back to host
        cudaStreamSynchronize                             //    full pipeline stall
        host loop over towers*n                           //    CPU modular add
    ct_mul_pt_host(t0, t1, dE, stw, n, ml)                 // CPU
    ct_add_ct_host(a0b, a1b, t0, t1, stw, n, ml)           // CPU accumulate

rescaleAt(a0b, a1b, stw)
stageRotEnd(R)
```

### 3.1 The dominant transfer is the key, not the ciphertext

`ks_context_create` is called **inside the per-diagonal loop** and it
unconditionally uploads the whole eval key. From
`src/keyswitch_resident.cpp`'s loop over parts:

```
C.d_av.push_back(upv(K.av[p], s));   // fresh cudaMalloc + H2D
C.d_bv.push_back(upv(K.bv[p], s));   // fresh cudaMalloc + H2D
```

and `src/evalkeygen.cpp` sizes each one at `sizeQlP * n`:

```
K.av[part].resize((size_t)sizeQlP*n);
K.bv[part].resize((size_t)sizeQlP*n);
```

At the bootstrap's own parameters (n=8192, sizeQ=30, sizeP=2 so sizeQlP=32,
numPart=(30+1)/2=15):

| Quantity | Derivation | Value |
|---|---|---|
| one `av[p]` or `bv[p]` | 32 × 8192 × 8 B | **2 MiB** |
| per `ks_context_create` | 15 parts × 2 arrays × 2 MiB | **60 MiB H2D** |
| allocations per call | 15 × 2, plus `d_keyRow` | **31 cudaMalloc / 31 cudaFree** |
| ciphertext traffic per diagonal | (1 H2D + 1 D2H) for `pt_to_eval_host` + 3 D2H in `stageRotDiag` | **~9.4 MiB** |

So the key re-upload is roughly **6× the entire ciphertext traffic of the
diagonal**, and it is pure waste: the same rotation key is re-uploaded every
time its index recurs, and `KROT`/`KSC` in `test_boot.cpp` are **host-side**
caches (`KSC` is explicitly a *view* cache — an `atLevel()` copy of a `KROT`
full key), so neither prevents the device-side re-upload.

Across ~100 diagonals this is on the order of **6 GiB of H2D key traffic per
bootstrap** and **~3,100 cudaMalloc/cudaFree pairs**, derived from the
dimensions above. Not profiled — see §0.

`src/rotate.cpp`'s own comment already anticipated exactly this:

> "The context (eval key + constants + root tables) is uploaded once per call
> here; if that dominates, cache contexts across calls next."

This audit's answer to that conditional: **by static byte count, it dominates.**

### 3.2 The ciphertext makes a full round trip per diagonal

`stageRotDiag` ends by pulling `out0`/`ba0`/`out1` back to the host, syncing,
and doing the modular add on the CPU. Then `ct_mul_pt_host` and
`ct_add_ct_host` do the plaintext multiply and the accumulation on the CPU too.
Meanwhile `pt_to_eval_host` has *already* done its own independent H2D → NTT →
D2H round trip for the diagonal's plaintext.

The data is on the GPU, gets copied to the host, is operated on by the CPU, and
the next diagonal copies it straight back. Every diagonal pays a full
`cudaStreamSynchronize`, so there is no overlap between diagonals at all.

### 3.3 What is already clean

Worth stating explicitly so nobody "optimises" it twice:

* Every `kernels/*.cu` file is transfer-free and sync-free.
* `keyswitch_resident` itself is pure kernel launches — its header promises
  "no malloc, no host transfer" and the inventory in §1 confirms zero D2H.
* Root tables and the decompose shape constants are already cached for the
  process lifetime (`g_rtab`, `upc`), which removed a measured 54 ms of a
  170 ms rotation. Those caches are *not* the problem; `av`/`bv` are.
* `rotate_ct_device` is already a pure device-pointer rotation.
* `automorphism_eval_fast_device` already does all towers in one batched gather.

## 4. Opportunities, ranked, each with its risk and its test

Per the directive, no speculative changes are made here. For each item: the
exact location, the data dependency, the correctness risk, the isolated test
that would prove it, and the fallback.

### O1 — Cache `DeviceKSContext` across diagonals (highest value by a wide margin)

* **Location** `src/keyswitch_resident.cpp` `ks_context_create`/`ks_context_destroy`;
  call site `tests/test_boot.cpp` `stageRotDiag`.
* **Dependency** `av`/`bv` depend only on (rotation index k, level/tower count).
  Nothing in a diagonal's data flows into them. Within a stage the tower count
  is fixed, so a per-(k, tw) device-context cache is well defined.
* **Correctness risk** *Medium-high, and specific.* Entry #20 in
  `research/EXPERIMENT_LEDGER.md` already records two real bugs in this exact
  neighbourhood: a segfault from a shape-only `KeySwitchConstants` with empty
  `av`/`bv`, and a genuine correctness bug from building a key directly at a
  reduced tower count instead of build-full-then-`atLevel`. A context cache
  keyed carelessly on `k` alone — ignoring `tw` — would reproduce the second bug
  exactly. The cache key must be `(k, tw)`, and entries must come from the
  trusted `rotFull` + `atLevel` path.
* **Also** it interacts with the `~200 MB per stage transition` device-memory
  growth in entry #24, whose best-supported (unconfirmed) explanation is CUDA
  allocator retention of differently-sized freed blocks. Removing ~3,100
  malloc/free pairs per bootstrap plausibly *reduces* that pressure, but a
  context cache also holds 60 MiB per live entry — so it needs a bounded LRU,
  exactly like `KROT_CAP`/`KSC_CAP`, and the cap must be chosen against the
  6144 MiB card.
* **Isolated test** Extend `tests/test_rot_resident.cpp` (already bit-exact
  gated, runs in ~275 MB) with a repeated-rotation loop over a fixed index set,
  asserting bit-exactness against the current per-call path and measuring both
  VRAM high-water and wall time. `bench/bench_hoist_ab.cpp` added this round is
  the ready-made template: it already builds N contexts up front and verifies
  bit-exactness before reporting any timing.
* **Fallback** Feature-flag it off by default, as `HOIST_ROTATION_ENABLED`
  already is.
* **Status** NOT implemented this round. Static evidence only.

### O2 — Keep the diagonal accumulation on the device

* **Location** `tests/test_boot.cpp` per-diagonal body: the 3 D2H in
  `stageRotDiag`, the host modular-add loop, `ct_mul_pt_host`, `ct_add_ct_host`.
* **Dependency** The accumulator is a pure sum over diagonals; each term is
  `rotate(ct, k) * diag`. Nothing needs the host.
* **Correctness risk** Medium. Needs a fused device kernel for
  `acc += rot_ct * diag` that is bit-exact against the host pair, and the
  plaintext diagonal still has to reach the device.
* **Isolated test** Bit-exactness of the fused kernel against
  `ct_mul_pt_host` + `ct_add_ct_host` on identical inputs.
* **IMPORTANT — this may already be done.** See §5.
* **Status** NOT implemented this round.

### O3 — Hoist `pt_to_eval_host`'s transform onto the device

* **Location** `src/rotate.cpp` `pt_to_eval_host` → `xf_multi`.
* **Dependency** The diagonal plaintext is produced on the CPU by
  `encode_host`; only its NTT needs the GPU. The round trip exists solely
  because the result is then consumed by host-side `ct_mul_pt_host`. So **O3 is
  largely subsumed by O2** — fix the accumulation and the plaintext never needs
  to come back.
* **Correctness risk** Low in isolation.
* **Note** `xf_multi` is the function whose shared global scratch buffer was the
  concurrency defect fixed this round (now `thread_local`). Any change here must
  preserve that.
* **Status** NOT implemented this round.

### O4 — Remove the per-diagonal `cudaStreamSynchronize`

* **Location** `stageRotDiag`'s terminating sync.
* **Dependency** Only needed because the host immediately reads the result.
  Falls out of O2 for free; pointless to attempt before it.
* **Status** NOT implemented. Do not attempt independently.

### O5 — Rotation hoisting (decompose-sharing)

Measured this round. See `research/ROTATION_HOISTING_INVESTIGATION.md` — a real
1.9× on the key-switch portion at 6 shared rotations, but a **loss** at 1, and
not an end-to-end figure. Orthogonal to O1 (O1 removes key *uploads*; O5 removes
redundant *decomposes*) and they compose.

### O6 — Latent: unsynchronized process-global caches

`src/keyswitch_resident.cpp:30` (`upc`), `:106` (`g_rtab`),
`src/rotate.cpp:29` (`dev_tab`), `:130` (`autoPermMap`) are all
`find()`-then-`emplace()` on a `static std::map` with **no mutex anywhere in
either file**. C++11 guarantees thread-safe *initialization* of a function-local
static, not concurrent *mutation*.

Tested this round and **found NOT to be the active defect**: phase B of
`tests/test_rotate_concurrent_cache.cpp` drives exactly that path from 9 threads
on cold caches with distinct moduli and passed 6/6 runs bit-exact. Left unfixed
deliberately — there is no failing test for it, and the directive forbids
changing `keyswitch_resident.cpp` internals without one. Phase B remains in the
suite as the control; if it ever fails, reopen this.

## 5. Before implementing O2/O3: the user's uncommitted work already prototypes them

This is the most practically important finding in this audit, and it is a
process finding rather than a technical one.

The non-authoritative Windows clone at `C:\Users\samca\GPU-Resident-Library`
(HEAD `f425cdd`, 65 dirty working-tree entries — see
`research/PHASE25_RECOVERY.md` §2) carries commits whose subjects are:

* `Fused pt-mul-accumulate kernel (acc += rot_ct * diag): bit-exact vs host ct_mul_pt + ct_add_ct` — **this is O2.**
* `rotate_ct_device: pure device-pointer rotation, bit-exact, 71.6ms kernels-only. Adds LaunchAddInto`
* `rotate_ct_resident fully device-side: both automorphisms as kernels on uploaded data, 1.80x over host path, bit-exact`
* `Device automorphism kernel (permute in coeff domain): bit-exact vs host, 6.4ms vs 9ms. Unblocks stage-resident transforms` — **this is O3's enabler.**
* `Tower-batched fastcore (grid.y=tower): 57.2ms -> 48.6ms keyswitch, A/B measured in-run, bit-exact`

Some of these subjects correspond to work already present at `origin/main`;
others look like they go further. **Nothing was read out of that clone beyond
`git log` and `git status`**, and no code was copied in either direction, per the
directive's rule that it is not authoritative for Git.

**Recommendation: reconcile that clone before writing any new O2/O3 code.** The
next session's first move on residency should be to diff `f425cdd` + its
working tree against `origin/main` and determine what is genuinely new, rather
than re-implementing a fused accumulate kernel that may already exist and be
bit-exact-verified. That is a decision for the repository owner, not something
this session should resolve by absorbing or discarding the work.

## 6. What this audit does NOT establish

* No wall-clock attribution. The byte counts are derived from array
  dimensions; they are not a profiler trace, and the bootstrap was never run.
* `ndiagTot` is derived from `LST=3`/`RST=4`, not observed.
* No claim that removing the key re-upload yields any particular end-to-end
  speedup. Host-side diagonal processing is also a major cost (entry #23
  measured `stageDiags` for one stage at 268 s before its O(S²) fix), so
  transfer elimination is necessary but demonstrably not sufficient.
* The ~200 MB/stage-transition device-memory growth from entry #24 remains
  un-root-caused; `compute-sanitizer` is not installed and
  `nvidia-smi --query-compute-apps` returns no process rows under this WSL2
  CUDA passthrough, both re-confirmed as unavailable.

## 7. Commands used

```
# per-file CUDA call-site inventory (counts call sites, not invocations)
for f in src/*.cpp kernels/*.cu tests/test_boot.cpp include/*.h; do
  grep -c 'cudaMemcpyHostToDevice' "$f"; grep -c 'cudaMemcpyDeviceToHost' "$f"
  grep -cE 'cudaDeviceSynchronize|cudaStreamSynchronize' "$f"
  grep -cE 'cudaMalloc' "$f"; grep -c '<<<' "$f"
done

# key-size derivation
grep -n 'K.av\[part\].resize' src/evalkeygen.cpp
grep -n 'C.d_av.push_back' src/keyswitch_resident.cpp

# resource gate (NO-GO all session)
powershell.exe -File research/benchmarks/bootstrap_preflight.ps1
```

---

## 8. Phase 28 addendum: the `pt_to_eval_host` residency hypothesis, quantified

GOD PROMPT "CKKS PHASE 28", Objective C. This round's stage-timeline
instrumentation (`bench/bench_stage_timeline.cpp`) measured `pt_to_eval_host`
at 16.0-16.3% of the fused-kernel stage's total time (see
`research/ENCODE_FFT_PROMOTION.md` for why it is NOT the dominant cost —
`encode_host` is, at 62-75%). `pt_to_eval_host` (`src/rotate.cpp`'s
`xf_multi`) does its own H2D, a batched NTT kernel launch, and its own D2H,
into a `thread_local` scratch device buffer that already persists across
calls (never freed, grown on demand) but is still round-tripped to host
every call. The fused-kernel stage then separately re-uploads that same
plaintext (as `d_d`, alongside the simulated ciphertext's `d_b0`/`d_b1`) for
`LaunchPtMulAcc`. **Hypothesis**: if `pt_to_eval_host` exposed its already-
resident device buffer directly instead of copying it back to host, both
that D2H and the stage's subsequent re-upload of the same data could be
skipped.

### The experiment (standalone, zero production changes)

`bench/bench_residency_hypothesis.cpp` — touches no production file, not
even `src/rotate.cpp`. It times a raw `cudaMemcpy` (CUDA-event-timed, 200
reps) of exactly the buffer size `pt_to_eval_host`'s `xf_multi` moves
(towers×n×8 bytes = 1.875 MiB at n=8192, towers=30), to put a hard ceiling on
the hypothesis before risking any change to crypto-adjacent code.

### Result

```
H2D (this buffer size): 0.4384 ms
D2H (this buffer size): 0.4467 ms
pt_to_eval_host per diagonal (measured, Arm B, this round): 6.475 ms
  -> this D2H alone is 6.9% of pt_to_eval_host's own time
  -> the remaining 93.1% is H2D-in + the batched NTT kernel itself, which a
     residency change CANNOT remove (the transform still has to run)
plaintext's share of the stage's separate H2D (1 of 3 equal-size buffers
per diagonal: b0, b1, d): 0.438 ms, 1.11% of the stage's per-diagonal total
UPPER BOUND if BOTH were eliminated: ~2.24% of stage time
```

### Decision: not pursued further this round

~2.2% is real but an order of magnitude below the encode swap's measured
2.5x stage-level speedup (`research/ENCODE_FFT_PROMOTION.md`), and actually
implementing it is not a trivial change: it means exposing
`src/rotate.cpp`'s `thread_local` scratch pointer across a function
boundary (a new `pt_to_eval_device` variant returning a raw device pointer
instead of copying to host), which raises real ownership/lifetime questions
for a buffer this file currently guarantees is only ever touched by its own
internal functions. Given the governing directive's explicit "do not stack
several unverified optimizations" instruction, and that this phase already
has one compiles-but-unverified change in `tests/test_boot.cpp` (the encode
swap) pending the memory-gate re-verification, spending the remaining
accuracy budget on a ~2% ceiling is not a good trade. **Recorded as a
future candidate, not implemented**: `research/NEXT_SESSION.md` carries this
forward, explicitly gated on the encode swap's own end-to-end verification
landing first (per the "do not change multiple unrelated bottlenecks at
once" instruction) and on a deliberate design for the new function's
ownership contract, not a quick patch.

This updates section 6's "no claim that removing the key re-upload yields
any particular end-to-end speedup" — it is no longer unquantified: ~2.2% of
stage time is the measured ceiling, not a guess.
