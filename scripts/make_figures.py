"""Figures for the baseline stage, drawn only from the committed results tables.

Nothing here recomputes anything. If a number appears in a figure it appears in
``results/baselines/folds.csv`` first, so a reader can check any bar against the table and a
figure can never drift from the experiment that produced it.
"""

from __future__ import annotations

import argparse
import json
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


def finetune_figure(
    folds: pd.DataFrame, finetune: pd.DataFrame, out_path: Path, view: str = "4class"
) -> Path:
    """The fine-tune placed against every baseline that needed less.

    The comparison the paper does not make: a 95-million-parameter model beside four numbers
    describing where the electrode sits, and beside the same checkpoint with nothing trained at
    all, on the same folds with the same metric.
    """
    baselines = folds[
        (folds["view"] == view) & (folds["model"] == "logreg") & (folds["control"] == "model")
    ]
    tuned = finetune[(finetune["view"] == view) & (~finetune["permuted"])]
    schemes = [s for s in SCHEME_LABELS if s in set(tuned["scheme"])]
    if not schemes:
        raise ValueError("no fine-tune runs to plot")

    order = [f for f in FEATURE_ORDER if f in set(baselines["features"])] + ["finetuned"]
    labels = {**FEATURE_LABELS, "finetuned": "fine-tuned wav2vec2"}
    colours = {**COLOURS, "finetuned": "#7b3294"}

    fig, axes = plt.subplots(
        1, len(schemes), figsize=(3.7 * len(schemes), 4.4), sharey=True, squeeze=False
    )
    axes = axes[0]

    for ax, scheme in zip(axes, schemes, strict=False):
        base = baselines[baselines["scheme"] == scheme]
        runs = tuned[tuned["scheme"] == scheme]
        chance = float(runs["chance"].mean())

        for position, feature in enumerate(order):
            rows = (
                runs["balanced_accuracy"]
                if feature == "finetuned"
                else base[base["features"] == feature]["balanced_accuracy"]
            )
            if not len(rows):
                continue
            ax.bar(position, rows.mean(), color=colours.get(feature, "#888"), width=0.68, zorder=2)
            jitter = np.random.default_rng(0).uniform(-0.16, 0.16, len(rows))
            ax.scatter(
                position + jitter, rows, s=13, color="#222222", alpha=0.75, zorder=3, linewidths=0
            )

        ax.axhline(chance, color="#d62728", linestyle="--", linewidth=1.2, zorder=1)
        ax.text(-0.45, chance - 0.02, f"chance {chance:.2f}", color="#d62728", fontsize=7, va="top")
        ax.set_title(SCHEME_LABELS.get(scheme, scheme), fontsize=9)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([labels.get(f, f) for f in order], rotation=35, ha="right", fontsize=7.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25, zorder=0)

    axes[0].set_ylabel("balanced accuracy", fontsize=9)
    fig.suptitle(
        f"Fine-tuned wav2vec2 against what needed less ({view}, one point per held-out session)",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def load_finetune(root: Path) -> pd.DataFrame | None:
    """Every fine-tune run's per-group scores, or None when none have been run."""
    paths = sorted(root.glob("*/*/per_group.csv"))
    if not paths:
        return None
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["run"] = path.parent.parent.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/baselines"))
    parser.add_argument("--discriminator", type=Path, default=Path("results/lab_discriminator"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    parser.add_argument("--finetune", type=Path, default=Path("results/finetune"))
    parser.add_argument("--ablations", type=Path, default=Path("results/ablations"))
    parser.add_argument("--calibration", type=Path, default=Path("results/calibration"))
    parser.add_argument("--abstention", type=Path, default=Path("results/abstention"))
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

    tuned = load_finetune(args.finetune)
    if tuned is None:
        print(f"no fine-tune runs under {args.finetune}; skipping that panel")
    else:
        print("wrote", finetune_figure(folds, tuned, args.out / "finetune.png", view=args.view))

    # Stage 4 panels. Each says explicitly when it has nothing to draw, rather than leaving the
    # script to report success while quietly producing one fewer figure.
    ablation_files = sorted(args.ablations.glob("*.csv"))
    if not ablation_files:
        print(f"no ablation tables under {args.ablations}; skipping that panel")
    else:
        tables = pd.concat([pd.read_csv(f) for f in ablation_files], ignore_index=True)
        print("wrote", ablation_figure(tables, args.out / "ablations.png"))

    reports = {f.stem: json.loads(f.read_text()) for f in sorted(args.calibration.glob("*.json"))}
    if not reports:
        print(f"no calibration reports under {args.calibration}; skipping that panel")
    else:
        print("wrote", calibration_figure(reports, args.out / "calibration.png"))

    curve_files = sorted(args.abstention.glob("*_curves.json"))
    summary_files = sorted(f for f in args.abstention.glob("*.csv"))
    if not curve_files or not summary_files:
        print(f"no abstention results under {args.abstention}; skipping that panel")
    else:
        curves = {f.stem.replace("_curves", ""): json.loads(f.read_text()) for f in curve_files}
        summary = pd.concat([pd.read_csv(f) for f in summary_files], ignore_index=True)
        print("wrote", abstention_figure(curves, summary, args.out / "abstention.png"))


def ablation_figure(tables: pd.DataFrame, out_path: Path) -> Path:
    """Accuracy cost of each transform, one panel per model.

    Costs are plotted relative to the unmodified reference, so a bar reaching zero means the model
    lost everything that transform took away. The amplitude bars are controls and must sit at
    exactly zero; they are drawn rather than hidden so a reader can see the check passing.
    """
    models = [m for m in ("bandpower", "frozen", "finetuned") if m in set(tables["model"])]
    labels = {
        "bandpower": "band power",
        "frozen": "frozen audio model",
        "finetuned": "fine-tuned wav2vec2",
    }
    order = [a for a in tables["ablation"].unique() if a != "none"]
    # A table can hold both cross-lab directions; average the cost over them so each bar is one
    # number. Indexing without this returns several rows per ablation.
    tables = tables.groupby(["model", "ablation"], as_index=False).agg(
        delta=("delta", "mean"), is_control=("is_control", "first")
    )

    fig, axes = plt.subplots(
        1, len(models), figsize=(4.2 * len(models), 4.6), sharey=True, squeeze=False
    )
    for ax, model in zip(axes[0], models, strict=False):
        part = tables[tables["model"] == model].set_index("ablation")
        for position, name in enumerate(order):
            if name not in part.index:
                continue
            row = part.loc[name]
            colour = (
                "#999999"
                if row["is_control"]
                else ("#c1440e" if row["delta"] < -0.02 else "#1f4e79")
            )
            ax.barh(position, row["delta"], color=colour, height=0.68, zorder=2)
        ax.axvline(0, color="#222222", linewidth=0.9, zorder=3)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([n.replace("_", " ") for n in order], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_title(labels.get(model, model), fontsize=9)
        ax.set_xlabel("change in balanced accuracy", fontsize=8)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.grid(axis="x", alpha=0.25, zorder=0)

    fig.suptitle("What each model loses when part of the signal is removed", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def calibration_figure(reports: dict[str, dict], out_path: Path) -> Path:
    """Reliability diagrams in-lab and cross-lab, before and after one fitted temperature.

    Perfect calibration is the diagonal. A curve below it is overconfidence: the model claims more
    than it delivers. The question is whether a temperature fitted on the left panel moves the
    right one.
    """
    fig, axes = plt.subplots(
        1, 2 * len(reports), figsize=(3.4 * 2 * len(reports), 3.6), squeeze=False
    )
    column = 0
    for run, payload in reports.items():
        direction = run.split("__")[0].replace("cross_lab_", "").replace("_", " ")
        for split in ("in_lab", "cross_lab"):
            ax = axes[0][column]
            column += 1
            for key, colour, style in (
                ("reliability_before", "#c1440e", "-"),
                ("reliability_after", "#1f4e79", "-"),
            ):
                curve = payload[split][key]
                confidence = np.array(curve["confidence"], dtype=float)
                accuracy = np.array(curve["accuracy"], dtype=float)
                keep = np.isfinite(confidence) & np.isfinite(accuracy)
                ax.plot(
                    confidence[keep],
                    accuracy[keep],
                    style,
                    color=colour,
                    marker="o",
                    markersize=3,
                    linewidth=1.3,
                    label="before" if "before" in key else "after temperature",
                )
            ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1.0, label="perfect")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{direction}\n{split.replace('_', ' ')}", fontsize=8)
            ax.set_xlabel("confidence", fontsize=8)
            if column == 1:
                ax.set_ylabel("accuracy", fontsize=8)
                ax.legend(fontsize=6.5, loc="upper left")
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)

    fig.suptitle("Does a temperature fitted in one lab reach the other?", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def abstention_figure(curves: dict[str, dict], summary: pd.DataFrame, out_path: Path) -> Path:
    """Risk against coverage for each uncertainty score, one panel per direction.

    Dropping the most uncertain predictions first should lower the error rate of what is kept. A
    flat curve is a score that carries no information about its own mistakes.
    """
    colours = {"max_softmax": "#c1440e", "entropy": "#e08214", "mahalanobis": "#1f4e79"}
    names = {
        "max_softmax": "confidence",
        "entropy": "entropy",
        "mahalanobis": "distance from training data",
    }
    fig, axes = plt.subplots(1, len(curves), figsize=(4.4 * len(curves), 4.0), squeeze=False)
    for ax, (run, payload) in zip(axes[0], sorted(curves.items()), strict=False):
        for score, curve in payload.items():
            ax.plot(
                curve["coverage"],
                curve["risk"],
                color=colours.get(score, "#666666"),
                linewidth=1.6,
                label=names.get(score, score),
            )
        part = summary[summary["run"] == run]
        if len(part):
            base = float(part["risk_full"].iloc[0])
            ax.axhline(base, color="#888888", linestyle="--", linewidth=1.0)
            ax.text(0.02, base + 0.01, "error at full coverage", fontsize=6.5, color="#666666")
        direction = run.split("__")[0].replace("cross_lab_", "").replace("_", " ")
        ax.set_title(direction, fontsize=9)
        ax.set_xlabel("coverage (fraction kept)", fontsize=8)
        ax.set_ylabel("error rate of what is kept", fontsize=8)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    fig.suptitle("Can the model tell which of its own predictions to discard?", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    main()
