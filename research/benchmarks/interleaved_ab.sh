#!/bin/bash
set -u
cd "$(dirname "$0")/../../build"
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
NVSMI=/usr/lib/wsl/lib/nvidia-smi
OUTFILE="../research/benchmarks/interleaved_$(date -u +%Y%m%dT%H%M%SZ).jsonl"
echo "output: $OUTFILE"
ROUNDS="${1:-3}"

run_one() {
  local label="$1" bin="$2" round="$3"
  local gb ga t0 t1 out wall rep rot err pass
  gb=$("$NVSMI" --query-gpu=temperature.gpu,clocks.current.sm,power.draw --format=csv,noheader,nounits)
  t0=$(date +%s.%N)
  out=$("$bin" 2>&1)
  t1=$(date +%s.%N)
  ga=$("$NVSMI" --query-gpu=temperature.gpu,clocks.current.sm,power.draw --format=csv,noheader,nounits)
  wall=$(echo "$t1 - $t0" | bc)
  rep=$(echo "$out" | grep -oP 'bootstrap wall time = \K[0-9.]+')
  rot=$(echo "$out" | grep -oP 'rotate_ct_resident=\K[0-9.]+')
  err=$(echo "$out" | grep -oP 'max slot err \|boot\(z\) - z\| = \K[0-9.e-]+')
  pass=false
  echo "$out" | grep -q '\[PASS\]' && pass=true
  rep="${rep:-NA}"; rot="${rot:-NA}"; err="${err:-NA}"
  printf '{"round":%s,"label":"%s","wall_s":%s,"reported_s":"%s","rot_s":"%s","err":"%s","pass":%s,"gpu_before":"%s","gpu_after":"%s"}\n' \
    "$round" "$label" "$wall" "$rep" "$rot" "$err" "$pass" "$gb" "$ga" | tee -a "$OUTFILE"
}

for i in $(seq 1 "$ROUNDS"); do
  run_one baseline ./test_boot "$i"
  run_one experiment ./test_boot_exp "$i"
done
