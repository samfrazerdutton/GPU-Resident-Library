#!/bin/bash
# Reliable benchmark harness: N trials of test_boot, with GPU environment
# metadata captured before/after each trial, median+spread computed,
# everything logged to a timestamped JSON-lines file for later comparison.
# Usage: run_trials.sh <label> <N trials> [binary path, default ./test_boot]
set -u
LABEL="${1:-run}"
NTRIALS="${2:-3}"
BIN="${3:-./test_boot}"
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:/usr/sbin:/sbin
NVSMI=/usr/lib/wsl/lib/nvidia-smi
OUTDIR="$(dirname "$0")"
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUTFILE="${OUTDIR}/${LABEL}_${TS}.jsonl"
COMMIT=$(git -C "${OUTDIR}/../.." rev-parse HEAD 2>/dev/null || echo "unknown")
DIRTY=$(git -C "${OUTDIR}/../.." status --short 2>/dev/null | wc -l)

gpu_snapshot() {
  "$NVSMI" --query-gpu=temperature.gpu,clocks.current.sm,power.draw,memory.used --format=csv,noheader,nounits 2>/dev/null
}

echo "label=$LABEL trials=$NTRIALS commit=$COMMIT dirty_files=$DIRTY bin=$BIN" >&2
echo "output: $OUTFILE" >&2

declare -a WALLTIMES
for i in $(seq 1 "$NTRIALS"); do
  GPU_BEFORE=$(gpu_snapshot)
  T0=$(date +%s.%N)
  OUT=$("$BIN" 2>&1)
  T1=$(date +%s.%N)
  GPU_AFTER=$(gpu_snapshot)
  WALL=$(echo "$T1 - $T0" | bc)
  BOOTREPORTED=$(echo "$OUT" | grep -oP 'bootstrap wall time = \K[0-9.]+' || echo "NA")
  ROTBUCKET=$(echo "$OUT" | grep -oP 'rotate_ct_resident=\K[0-9.]+' || echo "NA")
  MAXERR=$(echo "$OUT" | grep -oP 'max slot err \|boot\(z\) - z\| = \K[0-9.e-]+' || echo "NA")
  PASS=$(echo "$OUT" | grep -q "\[PASS\]" && echo "true" || echo "false")
  printf '{"label":"%s","trial":%d,"commit":"%s","dirty_files":%d,"wall_s":%s,"reported_bootstrap_s":%s,"rotate_bucket_s":%s,"max_slot_err":"%s","pass":%s,"gpu_before":"%s","gpu_after":"%s","ts":"%s"}\n' \
    "$LABEL" "$i" "$COMMIT" "$DIRTY" "$WALL" "$BOOTREPORTED" "$ROTBUCKET" "$MAXERR" "$PASS" "$GPU_BEFORE" "$GPU_AFTER" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUTFILE"
  WALLTIMES+=("$WALL")
  echo "  trial $i: wall=${WALL}s reported=${BOOTREPORTED}s rot=${ROTBUCKET}s pass=${PASS} gpu_after=${GPU_AFTER}" >&2
done

# median + spread
SORTED=($(printf '%s\n' "${WALLTIMES[@]}" | sort -n))
N=${#SORTED[@]}
MID=$((N/2))
if (( N % 2 == 1 )); then MEDIAN=${SORTED[$MID]}; else MEDIAN=$(echo "(${SORTED[$MID-1]} + ${SORTED[$MID]}) / 2" | bc -l); fi
MIN=${SORTED[0]}; MAX=${SORTED[$((N-1))]}
echo "=== $LABEL: median=${MEDIAN}s min=${MIN}s max=${MAX}s (n=$N) ===" >&2
echo "$OUTFILE"
