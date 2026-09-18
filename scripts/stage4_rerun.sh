#!/bin/zsh
# Re-run both cross-lab directions saving everything a post-hoc stage needs: weights, validation
# logits, and embeddings for the validation and test sets. Stage 3 saved only test logits, which
# left temperature scaling, abstention and ablation unanswerable without retraining.
#
# Runs serially in one process on purpose. An earlier version of this chain polled
# `pgrep -f "lfpaudit finetune run"` to wait its turn, which matched the polling shell's own
# command line and waited on itself for a day. The second run here cannot start before the first
# returns, because it is the next line.
set -e
cd "$(dirname "$0")/.."

for scheme in cross_lab_ibl_to_allen cross_lab_allen_to_ibl; do
  echo "=== $scheme at $(date +%H:%M) ==="
  .venv/bin/lfpaudit finetune run \
    --scheme "$scheme" --seed 0 --save-model \
    --out results/finetune_v2 \
    > "/tmp/lfpscratch/s4_$scheme.log" 2>&1
  echo "=== $scheme done ==="
  grep -E "4class:|lab identity|saved weights" "/tmp/lfpscratch/s4_$scheme.log" | tail -3
done
echo "=== stage 4 re-run complete ==="
