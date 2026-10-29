#!/bin/bash
# Genuinely randomized A/B trial runner for rotation hoisting's end-to-end
# effect (GOD PROMPT "CORRECTNESS FIRST, RESOURCE-SAFE ARCHITECTURE" round,
# Priority 4). Supersedes interleaved_ab.sh for THIS specific comparison:
# that script runs baseline-then-experiment in a FIXED order every round --
# exactly the order confound this repo's own prior work was burned by twice
# (the resident-rotation null-result experiment, and the first hoisting A/B
# series -- see research/MASTER_STATE.md/EXPERIMENT_LEDGER.md for both).
# Fixing the order bias requires randomizing which condition runs first on
# EVERY trial, not just alternating a fixed pattern.
#
# Usage: randomized_ab.sh [N_TRIALS_PER_ARM] [BASELINE_BIN] [EXPERIMENT_BIN]
#   Defaults: 10 trials per arm (20 total runs), ./test_boot (flag off) vs.
#   ./test_boot_hoist_enabled (flag on) -- this repo's current hoisting
#   feature-flag pair (see tests/test_boot.cpp / test_boot_hoist_enabled.cpp).
#
# Methodology (document BEFORE running, so results can't be reinterpreted
# after the fact to favor either arm):
#   1. Build a sequence of exactly N_TRIALS_PER_ARM "baseline" labels and
#      N_TRIALS_PER_ARM "experiment" labels, then SHUFFLE the whole sequence
#      with `shuf` (a real permutation, not a fixed alternation) -- this is
#      the actual fix for the order confound: which arm runs 1st, 2nd, 3rd...
#      is now determined by the shuffle, not by a repeating A-then-B pattern.
#   2. Run the sequence in that randomized order, one process at a time
#      (never concurrently -- this machine has one GPU and tight host RAM;
#      running two heavy test_boot-class processes at once would make an
#      already-tight memory situation worse, which the governing directive
#      explicitly warns against).
#   3. Record wall time, GPU temp/clock/power before and after, and the
#      reported max slot err for EVERY run (not just a summary) -- the full
#      per-trial JSONL is the source of truth, kept alongside this script's
#      output exactly like the existing interleaved_*.jsonl files.
#   4. ANALYSIS: because order is now randomized (not a fixed A-then-B
#      pairing), do NOT compute a "paired difference" by matching trial i's
#      baseline to trial i's experiment the way the old block-alternated
#      series did -- there is no natural pairing once order is randomized.
#      Instead treat baseline's N times and experiment's N times as TWO
#      INDEPENDENT SAMPLES: report each arm's median and a bootstrap 95% CI
#      on the median (reusing this repo's own bootstrap-CI pattern from
#      bench/bench_hoist_multi.cpp and bench_evalmult_resident.cpp), and
#      report whether the CIs overlap. Do not claim a speedup unless the
#      CIs are clearly separated; if they overlap, report "inconclusive at
#      this N," not a manufactured winner.
#   5. Require the host-memory gate (>=6GB real Windows free,
#      Get-CimInstance Win32_OperatingSystem, not WSL's own free -h) before
#      starting -- this script does NOT check that itself (it runs inside
#      WSL and cannot query the Windows host directly); the CALLER must
#      verify the gate before invoking this script, per
#      research/MEMORY_FEASIBILITY.md's established convention.
set -u
cd "$(dirname "$0")/../../build"
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
NVSMI=/usr/lib/wsl/lib/nvidia-smi
N="${1:-10}"
BASE_BIN="${2:-./test_boot}"
EXP_BIN="${3:-./test_boot_hoist_enabled}"
OUTFILE="../research/benchmarks/randomized_ab_$(date -u +%Y%m%dT%H%M%SZ).jsonl"
echo "output: $OUTFILE"
echo "N per arm: $N   baseline: $BASE_BIN   experiment: $EXP_BIN"

if [ ! -x "$BASE_BIN" ] || [ ! -x "$EXP_BIN" ]; then
  echo "ERROR: one or both binaries not found/executable: $BASE_BIN $EXP_BIN" >&2
  exit 1
fi

run_one() {
  local label="$1" bin="$2" trial_idx="$3"
  local gb ga t0 t1 out wall err pass
  gb=$("$NVSMI" --query-gpu=temperature.gpu,clocks.current.sm,power.draw --format=csv,noheader,nounits 2>/dev/null)
  t0=$(date +%s.%N)
  out=$("$bin" 2>&1)
  t1=$(date +%s.%N)
  ga=$("$NVSMI" --query-gpu=temperature.gpu,clocks.current.sm,power.draw --format=csv,noheader,nounits 2>/dev/null)
  wall=$(echo "$t1 - $t0" | bc)
  err=$(echo "$out" | grep -oP 'max slot err \|boot\(z\) - z\| = \K[0-9.e-]+')
  pass=false
  echo "$out" | grep -q '\[PASS\]' && pass=true
  err="${err:-NA}"
  printf '{"sequence_pos":%s,"label":"%s","wall_s":%s,"err":"%s","pass":%s,"gpu_before":"%s","gpu_after":"%s"}\n' \
    "$trial_idx" "$label" "$wall" "$err" "$pass" "$gb" "$ga" | tee -a "$OUTFILE"
}

# Build the randomized sequence: N "baseline" + N "experiment" labels,
# shuffled into a genuine random permutation (not alternation).
seq_file=$(mktemp)
for i in $(seq 1 "$N"); do echo baseline; done >> "$seq_file"
for i in $(seq 1 "$N"); do echo experiment; done >> "$seq_file"
shuffled=$(shuf "$seq_file")
rm -f "$seq_file"

echo "Randomized sequence (position: label):"
echo "$shuffled" | nl -ba
echo "---"

pos=0
while IFS= read -r label; do
  pos=$((pos+1))
  if [ "$label" = "baseline" ]; then
    run_one baseline "$BASE_BIN" "$pos"
  else
    run_one experiment "$EXP_BIN" "$pos"
  fi
done <<< "$shuffled"

echo "---"
echo "Done. $((2*N)) total runs in randomized order, written to $OUTFILE"
echo "Next step: compute each arm's median + bootstrap 95% CI as two INDEPENDENT samples (see this script's header comment for why NOT to pair them), not a per-position paired difference."
