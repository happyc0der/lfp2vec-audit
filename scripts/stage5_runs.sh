#!/bin/zsh
# Stage 5, lever H: fine-tune with both datasets low-passed at 100 Hz, the corner chosen in
# Stage 2 before any cross-lab number existed. Serial, no process polling: the second run cannot
# start before the first returns because it is the next line.
setopt PIPE_FAIL
cd "$(dirname "$0")/.."
for scheme in cross_lab_ibl_to_allen cross_lab_allen_to_ibl; do
  echo "=== H: $scheme at $(date +%H:%M) ==="
  .venv/bin/lfpaudit finetune run --scheme "$scheme" --seed 0 --lowpass 100 --save-model \
    --out results/finetune_v2 > "/tmp/lfpscratch/s5_${scheme}.log" 2>&1 || echo "FAILED: $scheme"
  grep -E "measured|4class:|lab identity|saved weights" "/tmp/lfpscratch/s5_${scheme}.log" | tail -4
done
echo "=== stage 5 H runs complete ==="
