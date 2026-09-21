#!/bin/zsh
# Post-hoc steps for the runs scripts/error_bar_runs.sh trains. Safe to re-run at any time:
# it processes whichever runs have finished and regenerates every table and figure from files.
set -u
cd "$(dirname "$0")/.."
CLI=.venv/bin/lfpaudit

for seed in 1 2; do
  for scheme in cross_lab_ibl_to_allen cross_lab_allen_to_ibl; do
    for variant in "" "__lp100"; do
      tag="${scheme}__all_target_groups__seed${seed}${variant}"
      if ls results/finetune_v2/${tag}/*/metrics.json >/dev/null 2>&1; then
        echo "=== ${tag}"
        $CLI postprocess --run "${tag}" --view 4class || echo "postprocess FAILED for ${tag}"
        $CLI calibrate --run "${tag}" || echo "calibrate FAILED for ${tag}"
      else
        echo "--- ${tag}: not finished, skipped"
      fi
    done
  done
done

$CLI fixes-table
$CLI finetune report --out docs/RESULTS_FINETUNE.md
.venv/bin/python scripts/make_figures.py
