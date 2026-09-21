#!/bin/zsh
# Seeds and folds that turn single runs into distributions. Twelve fine-tunes, about 28 hours on
# an M4 Pro, ordered so the most informative land first and any stopping point leaves a complete
# comparison:
#   1. harmonised band, seeds 1 and 2, both cross-lab directions   (spread on the main result)
#   2. full band, seeds 1 and 2, both directions                    (is the collapse robust?)
#   3. the four within-lab IBL folds not yet run                    (seven-fold paired test)
# Serial by construction: each run is the next line, so nothing polls for a process and nothing
# can wait on itself. The throughput gate's budget is raised to 4 h because other analysis shares
# the CPU while these train, and a gate measured during a busy minute would refuse a healthy run.
setopt PIPE_FAIL
cd "$(dirname "$0")/.."
LOG=/tmp/lfpscratch
mkdir -p $LOG

run() {  # run <tag> <args...>
  local tag=$1; shift
  echo "=== $tag start $(date '+%m-%d %H:%M') ==="
  .venv/bin/lfpaudit finetune run "$@" --budget-hours 4 --out results/finetune_v2 \
    > "$LOG/eb_${tag}.log" 2>&1 || echo "FAILED: $tag"
  grep -E "4class:|lab identity" "$LOG/eb_${tag}.log" | tail -2
  echo "=== $tag end $(date '+%m-%d %H:%M') ==="
}

for seed in 1 2; do
  for scheme in cross_lab_ibl_to_allen cross_lab_allen_to_ibl; do
    run "H_${scheme}_s${seed}" --scheme $scheme --seed $seed --lowpass 100
  done
done
for seed in 1 2; do
  for scheme in cross_lab_ibl_to_allen cross_lab_allen_to_ibl; do
    run "F_${scheme}_s${seed}" --scheme $scheme --seed $seed
  done
done
for fold in 3638d102_probe01 54238fd6_probe00 5dcee0eb_probe00 d2832a38_probe00; do
  run "L_${fold}" --scheme loso_ibl --fold $fold --seed 0
done
echo "=== all error-bar runs complete $(date '+%m-%d %H:%M') ==="
