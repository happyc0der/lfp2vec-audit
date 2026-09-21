"""Score a linear head on frozen, harmonised wav2vec2 embeddings on every evaluation scheme.

Nothing in the 95M-parameter encoder is trained. Predictions are written in the same format as
the Stage 2 baselines, so ``lfpaudit postprocess --baseline w2v2_frozen_lp100`` applies the
paper's post-processing with the code every other row of the fixes table went through.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from lfpaudit import REGIONS
from lfpaudit.data.corpus import Corpus
from lfpaudit.data.splits import verify_no_leakage
from lfpaudit.eval.folds import build_schemes
from lfpaudit.eval.metrics import evaluate
from lfpaudit.models.baselines import fit_predict

NAME = "w2v2_frozen_lp100"
SOURCE = Path("data/features/w2v2_frozen_lp100_sub")
PREDICTIONS = Path("results/baselines/predictions")
OUT = Path("results/efficient/folds.csv")
CA2 = REGIONS.index("CA2")


def main() -> None:
    corpus = Corpus.load({"ibl": "data/stores/ibl", "allen": "data/stores/allen"})
    index = corpus.index.set_index("chunk_id")
    ids = np.load(SOURCE / "chunk_ids.npy")
    features = np.load(SOURCE / "embeddings.npy").astype(np.float32)
    row_of = pd.Series(np.arange(len(ids)), index=ids)
    labels = index.loc[ids, "region"].map({r: i for i, r in enumerate(REGIONS)}).to_numpy()
    usable = set(ids[labels != CA2].tolist())  # the 4-class view

    rng = np.random.default_rng(0)
    records = []
    for scheme, folds in build_schemes(corpus.index, seed=0).items():
        if scheme.startswith("in_session"):
            continue
        cache: dict[tuple, tuple] = {}
        for fold in folds:
            verify_no_leakage(fold.split, corpus.index)
            train_ids = [i for i in fold.split.train + fold.split.val if i in usable]
            test_ids = [i for i in fold.split.test if i in usable]
            train_rows, test_rows = row_of[train_ids].to_numpy(), row_of[test_ids].to_numpy()
            truth = labels[test_rows]
            key = (len(train_rows), int(train_rows.sum()))
            for control in ("model", "permuted"):
                train_y = labels[train_rows]
                if control == "permuted":
                    train_y = rng.permutation(train_y)
                start = time.time()
                # Cross-lab folds share one training set; fit it once per control.
                if (key, control) not in cache or scheme.startswith("loso"):
                    all_test = row_of[[i for f in folds for i in f.split.test if i in usable]]
                    target = test_rows if scheme.startswith("loso") else all_test.to_numpy()
                    probs = fit_predict("logreg", features[train_rows], train_y, features[target])
                    cache[(key, control)] = (target, probs, time.time() - start)
                target, probs, seconds = cache[(key, control)]
                where = pd.Series(np.arange(len(target)), index=target)
                mine = probs[where[test_rows].to_numpy()]
                report = evaluate(mine, truth)
                records.append(
                    {
                        "scheme": scheme,
                        "fold": fold.name,
                        "features": NAME,
                        "control": control,
                        "balanced_accuracy": report.balanced_accuracy,
                        "chance": report.chance,
                        "ece": report.ece,
                        "n_train": len(train_rows),
                        "n_test": len(test_rows),
                        "fit_seconds": round(seconds, 1),
                    }
                )
                if control == "model":
                    frame = pd.DataFrame({"chunk_id": ids[test_rows], "label": truth})
                    for i, region in enumerate(REGIONS):
                        frame[f"p_{region}"] = mine[:, i].astype(np.float16)
                    frame.to_parquet(
                        PREDICTIONS / f"{scheme}__4class__{NAME}__logreg__{fold.name}.parquet"
                    )
            print(scheme, fold.name, f"{records[-2]['balanced_accuracy']:.3f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(OUT, index=False)
    print(
        table.groupby(["scheme", "control"])[["balanced_accuracy", "chance", "ece"]].mean().round(3)
    )


if __name__ == "__main__":
    main()
