# Benchmark / driver baselines

Committed reference outputs, kept so a future run has something concrete to
diff against rather than a remembered number.

## `boot_multi_input_small.baseline.jsonl`

First recorded run of `tests/test_boot_multi_input_small.cpp`, captured during
the GOD PROMPT "CKKS PHASE 25" round.

- **Produced by:** `./build2/test_boot_multi_input_small`
- **Machine:** RTX 2060 Max-Q (6144 MiB), driver 595.97, CUDA 13.2, WSL2
  Ubuntu, 16 CPU threads, Windows 11 host with 15.42 GiB total RAM
- **Build:** `CMAKE_BUILD_TYPE=Release`, `-O3 -DNDEBUG`, no `-G`
- **Result:** 12/12 cases passed, 56/56 checks passed
- **Peak host RSS:** 121.8 MiB (cheap by design -- this driver is deliberately
  runnable under host-memory pressure)
- **Max final error:** 3.59e-10 across all 12 cases

**What this baseline is NOT.** It contains no bootstrap measurement. The small
driver runs no ModRaise, C2S, EvalMod or S2C stage; it validates the shared
harness (`tests/boot_driver_harness.h`) and real CKKS
encode/encrypt/decrypt/decode at n=1024. Bootstrap numbers can only come from
`tests/test_boot_multi_input.cpp`, which has never been executed on this
machine -- see `research/PHASE25_RECOVERY.md` section 5 for why.

The live (uncommitted, gitignored) output path is
`research/benchmarks/boot_multi_input_small.jsonl`; override it with the
`BOOT_DRIVER_JSONL` environment variable. To refresh this baseline
deliberately, copy the live file over it and say so in the commit message.
