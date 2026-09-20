#!/bin/zsh
# Stage 5, lever W: per-probe spectral whitening at train and test. Launched only if lever H
# leaves the cross-lab margin short of the paper's +0.11; the decision is made from H's numbers,
# not in advance. Serial, no process polling.
setopt PIPE_FAIL
cd "$(dirname "$0")/.."
for scheme in cross_lab_ibl_to_allen cross_lab_allen_to_ibl; do
  echo "=== W: $scheme at $(date +%H:%M) ==="
  # Whitening runs per chunk on the CPU and halves throughput to ~14 chunks/s, so ten epochs
  # estimate at 4.3 h and the default 3 h gate refuses. The budget is raised here, explicitly,
  # for this lever only: it is the principled form of the lever that already closed the gap,
  # and early stopping has cut every run so far to six to nine epochs.
  .venv/bin/lfpaudit finetune run --scheme "$scheme" --seed 0 --whiten --save-model \
    --budget-hours 5 --out results/finetune_v2 \
    > "/tmp/lfpscratch/s5w_${scheme}.log" 2>&1 || echo "FAILED: $scheme"
  grep -E "measured|4class:|lab identity|saved weights" "/tmp/lfpscratch/s5w_${scheme}.log" | tail -4
done
echo "=== stage 5 W runs complete ==="
