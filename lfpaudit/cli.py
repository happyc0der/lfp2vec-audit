"""Command line entry point: ``lfpaudit <command>``.

Stage 0 exposes three commands. ``info`` reports the resolved device and versions, ``smoke``
runs the whole pipeline end to end on synthetic data, and ``verify-split`` re-checks a saved
split for leakage. Later stages add data, baseline, training and figure commands; the Makefile
already lists them so the intended surface is visible.
"""

from __future__ import annotations

import json
import tempfile
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import typer
from sklearn.linear_model import LogisticRegression

from lfpaudit import REGION_TO_INDEX, __version__
from lfpaudit.config import set_seed
from lfpaudit.data.manifest import RunManifest
from lfpaudit.data.splits import Split, make_split, verify_no_leakage
from lfpaudit.data.synthetic import SyntheticSpec, build_synthetic_store
from lfpaudit.device import environment_summary, pick_device
from lfpaudit.eval.metrics import evaluate
from lfpaudit.features.bandpower import band_power

app = typer.Typer(add_completion=False, help="Audit tooling for LFP2Vec-style region decoding.")

# Apple's Accelerate BLAS sets IEEE exception flags during ordinary finite matrix products on
# arm64, which NumPy reports as RuntimeWarnings. A plain `A @ B` on finite random matrices
# triggers all four, so they carry no information about our data. Filtered here, at the
# application boundary, rather than in library code.
warnings.filterwarnings(
    "ignore",
    message=r"(divide by zero|overflow|underflow|invalid value) encountered in matmul",
    category=RuntimeWarning,
)


@app.command()
def info() -> None:
    """Print the resolved device, library versions and available accelerators."""
    policy = pick_device()
    typer.echo(f"lfpaudit {__version__}")
    typer.echo(f"selected device: {policy.describe()}")
    for key, value in environment_summary().items():
        typer.echo(f"{key}: {value}")


@app.command("verify-split")
def verify_split(
    split_path: Path = typer.Argument(..., help="Path to a split JSON."),
    index_path: Path = typer.Argument(..., help="Path to the chunk index parquet."),
) -> None:
    """Re-check a saved split for chunk, group and dataset leakage."""
    import pandas as pd

    split = Split.from_json(split_path)
    sizes = verify_no_leakage(split, pd.read_parquet(index_path))
    typer.echo(f"split {split.kind!r} (seed {split.seed}) is clean: {sizes}")


@app.command()
def smoke(
    device: str | None = typer.Option(None, help="Force a device: cuda, mps or cpu."),
    seed: int = typer.Option(0, help="Random seed."),
    out: Path = typer.Option(Path("results/smoke"), help="Where to write the run directory."),
    keep_data: bool = typer.Option(False, help="Keep the generated synthetic cache."),
) -> None:
    """Run the full pipeline on synthetic data and assert it recovers the planted signal.

    This is the gate every real run must pass first. It exercises chunking, the chunk store,
    group-aware splitting, the leakage verifier, band-power features, the metric suite and the
    manifest writer, and it checks two things that a broken pipeline would fail: a real model
    beats chance, and the same model trained on permuted labels does not.
    """
    set_seed(seed)
    policy = pick_device(device)
    workdir = (
        Path(tempfile.mkdtemp(prefix="lfpaudit-smoke-"))
        if not keep_data
        else Path("data/synthetic")
    )

    store_path = workdir / "store"
    spec = SyntheticSpec(datasets=("synthA", "synthB"), sessions_per_dataset=3, seed=seed)
    store = build_synthetic_store(str(store_path), spec)
    typer.echo(f"synthetic store: {len(store.index)} chunks at {store.fs:g} Hz -> {store_path}")

    split = make_split(store.index, kind="cross_session", seed=seed)
    sizes = verify_no_leakage(split, store.index)
    typer.echo(f"cross_session split verified clean: {sizes}")

    manifest = RunManifest.create(
        experiment="smoke",
        seed=seed,
        device=policy.device,
        config={"spec": asdict(spec), "split": split.kind},
        data_dir=store_path,
    )
    run_dir = Path(out) / manifest.run_id
    manifest.write(run_dir)

    features = band_power(store.take(store.index["chunk_id"].to_numpy()), fs=store.fs)
    labels = store.index["region"].map(REGION_TO_INDEX).to_numpy()
    position = {int(cid): i for i, cid in enumerate(store.index["chunk_id"])}
    rows = lambda ids: np.array([position[i] for i in ids], dtype=np.int64)  # noqa: E731

    train_rows, test_rows = rows(split.train), rows(split.test)
    model = LogisticRegression(max_iter=2000)
    model.fit(features[train_rows], labels[train_rows])
    report = evaluate(model.predict_proba(features[test_rows]), labels[test_rows])

    rng = np.random.default_rng(seed)
    permuted = LogisticRegression(max_iter=2000)
    permuted.fit(features[train_rows], rng.permutation(labels[train_rows]))
    control = evaluate(permuted.predict_proba(features[test_rows]), labels[test_rows])

    typer.echo(f"band-power LR : {report.summary_line()}")
    typer.echo(f"permuted label: {control.summary_line()}")

    chance = 1.0 / len(report.class_names)
    failures = []
    if report.balanced_accuracy <= chance + 0.15:
        failures.append(
            f"model balanced accuracy {report.balanced_accuracy:.3f} is not above chance"
        )
    if control.balanced_accuracy > chance + 0.15:
        failures.append(f"permutation control {control.balanced_accuracy:.3f} is above chance")

    (run_dir / "metrics.json").write_text(
        json.dumps(
            {"model": report.to_dict(), "permutation_control": control.to_dict(), "chance": chance},
            indent=2,
        )
    )
    split.to_json(run_dir / "split.json")
    manifest.finish(run_dir, status="failed" if failures else "ok")

    if failures:
        for message in failures:
            typer.secho(f"GATE FAILED: {message}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho(f"smoke passed on {policy.device}; wrote {run_dir}", fg=typer.colors.GREEN)


if __name__ == "__main__":  # pragma: no cover
    app()
