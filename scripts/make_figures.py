"""Figures for the baseline stage, drawn only from the committed results tables.

Nothing here recomputes anything. If a number appears in a figure it appears in
``results/baselines/folds.csv`` first, so a reader can check any bar against the table and a
figure can never drift from the experiment that produced it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Order feature sets from least to most information about the signal, so the reader's eye moves
#: from "what you get for free" to "what the neural data adds".
FEATURE_ORDER = ["amplitude", "geometry", "bandpower_clean", "bandpower_full", "w2v2_frozen"]

FEATURE_LABELS = {
    "amplitude": "amplitude only",
    "geometry": "electrode position only",
    "bandpower_clean": "band power ≤100 Hz",
    "bandpower_full": "band power, full",
    "w2v2_frozen": "frozen audio model",
}

SCHEME_LABELS = {
    "loso_ibl": "within lab (IBL)",
    "loso_allen": "within lab (Allen)",
    "cross_lab_ibl_to_allen": "cross lab: IBL → Allen",
    "cross_lab_allen_to_ibl": "cross lab: Allen → IBL",
}

COLOURS = {
    "amplitude": "#b0b0b0",
    "geometry": "#c1440e",
    "bandpower_clean": "#7fa8c9",
    "bandpower_full": "#1f4e79",
    "w2v2_frozen": "#4a7c59",
}


def baseline_figure(folds: pd.DataFrame, out_path: Path, view: str = "4class") -> Path:
    """One panel per scheme: mean balanced accuracy per feature set, with each fold as a point.

    Every panel carries its own chance line, because a held-out probe need not contain every
    region and chance moves with the classes present.
    """
    data = folds[
        (folds["view"] == view) & (folds["model"] == "logreg") & (folds["control"] == "model")
    ]
    schemes = [s for s in SCHEME_LABELS if s in set(data["scheme"])]
    present = [f for f in FEATURE_ORDER if f in set(data["features"])]

    fig, axes = plt.subplots(1, len(schemes), figsize=(3.6 * len(schemes), 4.2), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, scheme in zip(axes, schemes, strict=False):
        part = data[data["scheme"] == scheme]
        chance = float(part["chance"].mean())

        for position, feature in enumerate(present):
            rows = part[part["features"] == feature]
            if rows.empty:
                continue
            mean = float(rows["balanced_accuracy"].mean())
            ax.bar(position, mean, color=COLOURS.get(feature, "#888888"), width=0.68, zorder=2)
            # Individual folds, jittered, so the spread is visible rather than summarised away.
            jitter = np.random.default_rng(0).uniform(-0.16, 0.16, len(rows))
            ax.scatter(
                position + jitter,
                rows["balanced_accuracy"],
                s=13,
                color="#222222",
                alpha=0.75,
                zorder=3,
                linewidths=0,
            )

        ax.axhline(chance, color="#d62728", linestyle="--", linewidth=1.2, zorder=1)
        # Anchored below the line at the left margin: above it or on the right the label lands on
        # top of a bar in the panels where chance is high.
        ax.text(
            -0.45,
            chance - 0.02,
            f"chance {chance:.2f}",
            color="#d62728",
            fontsize=7,
            va="top",
            ha="left",
        )
        ax.set_title(SCHEME_LABELS.get(scheme, scheme), fontsize=9)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(
            [FEATURE_LABELS.get(f, f) for f in present], rotation=35, ha="right", fontsize=7.5
        )
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25, zorder=0)

    axes[0].set_ylabel("balanced accuracy", fontsize=9)
    fig.suptitle(
        "What predicts brain region, before any model is trained on it "
        f"({view}, one point per held-out session)",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def discriminator_figure(folds: pd.DataFrame, out_path: Path) -> Path:
    """How well the source dataset can be identified, with and without the contaminated band."""
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    order = ["bandpower_full", "bandpower_clean"]
    labels = ["band power, full", "band power ≤100 Hz"]

    for position, feature in enumerate(order):
        rows = folds[folds["features"] == feature]
        ax.bar(position, rows["auc"].mean(), color=COLOURS[feature], width=0.6, zorder=2)
        jitter = np.random.default_rng(0).uniform(-0.14, 0.14, len(rows))
        ax.scatter(position + jitter, rows["auc"], s=16, color="#222222", zorder=3, linewidths=0)

    ax.axhline(0.5, color="#d62728", linestyle="--", linewidth=1.2, zorder=1)
    ax.text(1.45, 0.5, " chance", color="#d62728", fontsize=7, va="bottom", ha="right")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("area under the curve", fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_title("Can a linear model tell which lab a recording came from?", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/baselines"))
    parser.add_argument("--discriminator", type=Path, default=Path("results/lab_discriminator"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    parser.add_argument("--view", default="4class")
    args = parser.parse_args()

    folds = pd.read_csv(args.results / "folds.csv")
    print("wrote", baseline_figure(folds, args.out / "baselines.png", view=args.view))

    discriminator_path = args.discriminator / "folds.csv"
    if discriminator_path.exists():
        print(
            "wrote",
            discriminator_figure(
                pd.read_csv(discriminator_path), args.out / "lab_discriminator.png"
            ),
        )


if __name__ == "__main__":
    main()
