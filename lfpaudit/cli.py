"""Command line entry point: ``lfpaudit <command>``.

Commands fall into three groups. ``info``, ``smoke`` and ``verify-split`` check that the machine
and the pipeline work. ``data``, ``card`` and ``inspect`` build and describe chunk stores from
the public datasets. ``make-splits`` and ``real-smoke`` turn those stores into the verified
partitions every later experiment runs on. The Makefile lists the not-yet-implemented commands
too, so the intended surface stays visible.
"""

from __future__ import annotations

import json
import tempfile
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.linear_model import LogisticRegression

from lfpaudit import REGION_TO_INDEX, REGIONS, __version__
from lfpaudit.config import set_seed
from lfpaudit.data.card import DatasetCard
from lfpaudit.data.chunk import ChunkStore
from lfpaudit.data.corpus import Corpus
from lfpaudit.data.inspect import inspection_figure
from lfpaudit.data.manifest import RunManifest
from lfpaudit.data.splits import Split, make_split, verify_no_leakage
from lfpaudit.data.synthetic import SyntheticSpec, build_synthetic_store
from lfpaudit.device import environment_summary, pick_device
from lfpaudit.eval.metrics import chance_level_band, evaluate, expand_probabilities
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


data_app = typer.Typer(help="Build chunk stores from the public datasets.")
app.add_typer(data_app, name="data")


@data_app.command("ibl")
def data_ibl(
    out: Path = typer.Option(Path("data/stores/ibl"), help="Where to write the store."),
    insertions: int = typer.Option(0, help="How many insertions to build; 0 means all."),
    start_s: float = typer.Option(200.0, help="Seconds into the recording the first chunk starts."),
    window_s: float = typer.Option(3.0, help="Chunk length in seconds."),
    chunks: int = typer.Option(100, help="Chunks per channel."),
    cache: Path = typer.Option(Path("data/ibl-cache"), help="Download cache directory."),
) -> None:
    """Fetch, preprocess and chunk IBL insertions."""
    from lfpaudit.data.ibl import PAPER_INSERTIONS, build_ibl_store

    chosen = PAPER_INSERTIONS[:insertions] if insertions else PAPER_INSERTIONS
    store, card = build_ibl_store(
        out,
        insertions=chosen,
        cache_dir=cache,
        window_s=window_s,
        start_s=start_s,
        chunks_per_channel=chunks,
    )
    typer.echo(card.to_markdown())
    typer.secho(f"wrote {len(store.index)} chunks to {out}", fg=typer.colors.GREEN)


@data_app.command("allen")
def data_allen(
    out: Path = typer.Option(Path("data/stores/allen"), help="Where to write the store."),
    sessions: str = typer.Option("719161530,798911424", help="Comma-separated session ids."),
    start_s: float = typer.Option(200.0, help="Seconds into the recording the first chunk starts."),
    window_s: float = typer.Option(3.0, help="Chunk length in seconds."),
    chunks: int = typer.Option(100, help="Chunks per channel."),
    cache: Path = typer.Option(Path("data/allen-cache"), help="Local copies, if any."),
) -> None:
    """Read Allen probe files and chunk them."""
    from lfpaudit.data.allen import build_allen_store

    store, card = build_allen_store(
        out,
        sessions=[int(s) for s in sessions.split(",") if s.strip()],
        cache_dir=cache,
        window_s=window_s,
        start_s=start_s,
        chunks_per_channel=chunks,
    )
    typer.echo(card.to_markdown())
    typer.secho(f"wrote {len(store.index)} chunks to {out}", fg=typer.colors.GREEN)


@app.command()
def card(
    store_paths: list[Path] = typer.Argument(..., help="One or more chunk stores."),
    out: Path = typer.Option(None, help="Also write a combined markdown document here."),
) -> None:
    """Print each store's dataset card, and optionally write them to one document."""
    sections = []
    for store_path in store_paths:
        path = Path(store_path) / "card.json"
        record = (
            DatasetCard.read(path)
            if path.exists()
            else DatasetCard.from_store(ChunkStore.open(store_path))
        )
        sections.append(record.to_markdown())
        typer.echo(sections[-1])

    if out is not None:
        header = (
            "# Dataset cards\n\n"
            "What each chunk store contains, per probe: how many channels survived labelling "
            "and quality screening, how they divide between regions, and which sources were "
            "skipped and why.\n\nGenerated by `lfpaudit card`; do not edit by hand.\n\n"
        )
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(header + "\n".join(sections))
        typer.secho(f"wrote {out}", fg=typer.colors.GREEN)


@app.command()
def inspect(
    store_path: Path = typer.Argument(..., help="Path to a chunk store."),
    out: Path = typer.Option(None, help="Output image path."),
    per_region: int = typer.Option(4, help="Example chunks per region."),
    seed: int = typer.Option(0, help="Sampling seed."),
) -> None:
    """Plot sampled chunks and per-region spectra for a human to look at."""
    store = ChunkStore.open(store_path)
    target = out or Path("docs/figures") / f"inspect_{Path(store_path).name}.png"
    written = inspection_figure(store, target, n_per_region=per_region, seed=seed)
    typer.secho(f"wrote {written}", fg=typer.colors.GREEN)


@app.command("make-splits")
def make_splits(
    ibl: Path = typer.Option(Path("data/stores/ibl"), help="IBL store."),
    allen: Path = typer.Option(Path("data/stores/allen"), help="Allen store."),
    out: Path = typer.Option(Path("experiments/splits"), help="Where to write split JSONs."),
    seed: int = typer.Option(0, help="Split seed."),
) -> None:
    """Write and verify every split the later experiments use."""
    out.mkdir(parents=True, exist_ok=True)
    corpus = Corpus.load({"ibl": ibl, "allen": allen})
    index = corpus.index

    wanted = {
        "in_session_ibl": dict(kind="in_session", subset="ibl"),
        "cross_session_ibl": dict(kind="cross_session", subset="ibl"),
        "cross_session_allen": dict(kind="cross_session", subset="allen"),
        "cross_lab_ibl_to_allen": dict(
            kind="cross_lab", train_datasets=["ibl"], test_datasets=["allen"]
        ),
        "cross_lab_allen_to_ibl": dict(
            kind="cross_lab", train_datasets=["allen"], test_datasets=["ibl"]
        ),
    }

    for name, spec in wanted.items():
        spec = dict(spec)
        subset = spec.pop("subset", None)
        frame = index[index["dataset"] == subset] if subset else index
        split = make_split(frame, seed=seed, **spec)
        sizes = verify_no_leakage(split, index)
        split.to_json(out / f"{name}.json")
        typer.echo(f"{name}: {sizes} groups test={split.test_groups}")
    typer.secho(f"wrote {len(wanted)} verified splits to {out}", fg=typer.colors.GREEN)


@app.command("real-smoke")
def real_smoke(
    split_path: Path = typer.Argument(..., help="A split JSON from make-splits."),
    ibl: Path = typer.Option(Path("data/stores/ibl"), help="IBL store."),
    allen: Path = typer.Option(Path("data/stores/allen"), help="Allen store."),
    out: Path = typer.Option(Path("results/real_smoke"), help="Run directory root."),
    seed: int = typer.Option(0, help="Random seed."),
) -> None:
    """Fit the interpretable baseline on real data, with the same gates as the synthetic smoke.

    This is the last check before any model training: it proves the labels in a real store carry
    recoverable signal, and that the signal disappears when the labels are shuffled.
    """
    set_seed(seed)
    corpus = Corpus.load({"ibl": ibl, "allen": allen})
    split = Split.from_json(split_path)
    sizes = verify_no_leakage(split, corpus.index)
    typer.echo(f"{split.kind} split verified clean: {sizes}")

    manifest = RunManifest.create(
        experiment=f"real_smoke_{Path(split_path).stem}",
        seed=seed,
        device="cpu",
        config={"split": str(split_path), "kind": split.kind},
        data_dir=ibl,
    )
    run_dir = Path(out) / manifest.run_id
    manifest.write(run_dir)

    def features_and_labels(ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        waveforms = corpus.take(ids)
        rows = corpus.positions(ids)
        labels = corpus.index.iloc[rows]["region"].map(REGION_TO_INDEX).to_numpy()
        return band_power(waveforms, fs=corpus.fs), labels

    train_x, train_y = features_and_labels(split.train)
    test_x, test_y = features_and_labels(split.test)

    def fit_and_score(labels: np.ndarray):
        model = LogisticRegression(max_iter=2000)
        model.fit(train_x, labels)
        # A region absent from training gets a zero column rather than a shifted one.
        probs = expand_probabilities(model.predict_proba(test_x), model.classes_)
        return evaluate(probs, test_y)

    report = fit_and_score(train_y)
    rng = np.random.default_rng(seed)
    control = fit_and_score(rng.permutation(train_y))

    trained_on = sorted(set(train_y.tolist()))
    unseen = [REGIONS[i] for i in range(len(REGIONS)) if i not in trained_on]
    if unseen:
        typer.secho(
            f"note: {', '.join(unseen)} absent from the training split; "
            "the model cannot predict it and its recall is zero by construction",
            fg=typer.colors.YELLOW,
        )

    typer.echo(f"band-power LR : {report.summary_line()}")
    typer.echo(f"permuted label: {control.summary_line()}")
    typer.echo(f"per-class recall: {report.per_class_recall}")

    # Both the chance level and the noise band come from the test labels themselves: a held-out
    # insertion need not contain every region, and one that contains three has a chance level of
    # 0.33 rather than 0.20.
    chance = report.chance
    band = chance_level_band(test_y)
    typer.echo(
        f"test classes: {', '.join(report.classes_present)} "
        f"-> chance {chance:.3f}, noise band +/-{band:.3f}"
    )

    failures = []
    if report.balanced_accuracy <= chance + band:
        failures.append(
            f"balanced accuracy {report.balanced_accuracy:.3f} is within noise of chance"
        )
    if control.balanced_accuracy > chance + band:
        failures.append(
            f"permutation control {control.balanced_accuracy:.3f} is above chance + {band:.3f}"
        )

    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model": report.to_dict(),
                "permutation_control": control.to_dict(),
                "chance": chance,
                "noise_band": band,
            },
            indent=2,
        )
    )
    manifest.finish(run_dir, status="failed" if failures else "ok")
    if failures:
        for message in failures:
            typer.secho(f"GATE FAILED: {message}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho(f"real smoke passed; wrote {run_dir}", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------------------------
# Stage 2: feature extraction, baselines and the lab discriminator.
# ---------------------------------------------------------------------------------------------

CHEAP_FEATURES = ("bandpower_full", "bandpower_clean", "geometry", "amplitude")


def _load_corpus(ibl: Path, allen: Path) -> Corpus:
    return Corpus.load({"ibl": ibl, "allen": allen})


def _build_feature_table(
    corpus: Corpus, name: str, cache_root: Path, device: str | None, rebuild: bool
) -> np.ndarray:
    """Return one feature table for the whole corpus, building and caching per store.

    Features are cached per store rather than per corpus so that adding a dataset later does not
    invalidate the work already done on the others.
    """
    from lfpaudit.config import BANDS
    from lfpaudit.features.amplitude import amplitude_features
    from lfpaudit.features.bandpower import band_power
    from lfpaudit.features.cache import FeatureCache
    from lfpaudit.features.geometry import geometry_features

    parts = []
    for store_name in sorted(corpus.stores):
        store = corpus.stores[store_name]
        cache = FeatureCache(Path(cache_root) / store_name, store.index)
        rows = np.arange(len(store.index))

        def builder(store=store, rows=rows, name=name, store_name=store_name):
            if name == "geometry":
                return geometry_features(store.index)
            if name == "amplitude":
                return amplitude_features(store.index)
            if name.startswith("bandpower"):
                # The clean variant drops the ripple band, where the two datasets' preprocessing
                # diverges by orders of magnitude (see DEVIATIONS D11), and renormalises so the
                # remaining bands still sum to one.
                bands = dict(BANDS)
                if name == "bandpower_clean":
                    bands.pop("ripple")
                typer.echo(f"  building {name} for {store_name} ({len(rows)} chunks)")
                return band_power(store.take(rows), fs=store.fs, bands=bands)
            if name == "w2v2_frozen":
                from lfpaudit.features.embed import embed_chunks

                typer.echo(f"  embedding {store_name} ({len(rows)} chunks) on {device}")
                return embed_chunks(
                    lambda r: store.take(r), rows, fs=store.fs, device=device or "cpu"
                )
            raise ValueError(f"unknown feature set {name!r}")

        parts.append(cache.get_or_build(name, builder, rebuild=rebuild))

    widths = {p.shape[1] for p in parts}
    if len(widths) != 1:
        raise ValueError(f"{name}: stores produced different feature widths {widths}")
    return np.concatenate(parts, axis=0).astype(np.float64)


features_app = typer.Typer(help="Build and cache feature tables.")
app.add_typer(features_app, name="features")


@features_app.command("build")
def features_build(
    which: str = typer.Option(",".join(CHEAP_FEATURES), help="Comma-separated feature sets."),
    ibl: Path = typer.Option(Path("data/stores/ibl")),
    allen: Path = typer.Option(Path("data/stores/allen")),
    cache: Path = typer.Option(Path("data/features")),
    device: str = typer.Option(None, help="Device for wav2vec2 embeddings."),
    rebuild: bool = typer.Option(False, help="Ignore any cached table."),
) -> None:
    """Compute feature tables and cache them against the stores they came from."""
    corpus = _load_corpus(ibl, allen)
    policy = pick_device(device)
    for name in [w.strip() for w in which.split(",") if w.strip()]:
        table = _build_feature_table(corpus, name, cache, policy.device, rebuild)
        typer.echo(f"{name}: {table.shape}")
    typer.secho("features ready", fg=typer.colors.GREEN)


baselines_app = typer.Typer(help="Run and report the baseline sweep.")
app.add_typer(baselines_app, name="baselines")


@baselines_app.command("run")
def baselines_run(
    which: str = typer.Option(",".join(CHEAP_FEATURES), help="Comma-separated feature sets."),
    models: str = typer.Option("constant,logreg", help="Comma-separated model names."),
    ibl: Path = typer.Option(Path("data/stores/ibl")),
    allen: Path = typer.Option(Path("data/stores/allen")),
    cache: Path = typer.Option(Path("data/features")),
    out: Path = typer.Option(Path("results/baselines")),
    device: str = typer.Option(None, help="Device for wav2vec2 embeddings."),
    seed: int = typer.Option(0),
) -> None:
    """Every feature set against every model, across every fold of every scheme."""
    from lfpaudit.eval.folds import build_schemes
    from lfpaudit.eval.runner import ExperimentSpec, run_sweep, write_results

    set_seed(seed)
    corpus = _load_corpus(ibl, allen)
    policy = pick_device(device)
    feature_names = [w.strip() for w in which.split(",") if w.strip()]

    tables = {
        name: _build_feature_table(corpus, name, cache, policy.device, rebuild=False)
        for name in feature_names
    }
    labels = corpus.index["region"].map(REGION_TO_INDEX).to_numpy()
    positions = {int(cid): i for i, cid in enumerate(corpus.index["chunk_id"])}

    schemes = build_schemes(corpus.index, seed=seed)
    typer.echo("schemes: " + ", ".join(f"{k} ({len(v)} folds)" for k, v in sorted(schemes.items())))

    manifest = RunManifest.create(
        experiment="baselines",
        seed=seed,
        device=policy.device,
        config={"features": feature_names, "models": models, "schemes": sorted(schemes)},
        data_dir=ibl,
    )
    manifest.write(out)

    spec = ExperimentSpec(
        feature_sets=feature_names,
        models=[m.strip() for m in models.split(",") if m.strip()],
        seed=seed,
        save_predictions_for=[k for k in schemes if k.startswith("cross_lab")] + ["loso_ibl"],
    )
    table = run_sweep(
        schemes, tables, labels, positions, spec, predictions_dir=Path(out) / "predictions"
    )
    paths = write_results(table, out)
    manifest.finish(out, status="ok")
    typer.secho(f"wrote {len(table)} rows to {paths['folds']}", fg=typer.colors.GREEN)


@baselines_app.command("table")
def baselines_table(
    results: Path = typer.Option(Path("results/baselines")),
    view: str = typer.Option("4class", help="Class view to report."),
    out: Path = typer.Option(None, help="Write a markdown document here."),
) -> None:
    """Render the baseline results as a readable table."""
    from lfpaudit.eval.runner import paired_comparison, summarise

    folds = pd.read_csv(results / "folds.csv")
    summary = summarise(folds)
    summary = summary[(summary["view"] == view) & (summary["control"] == "model")]

    lines = [f"# Baseline results ({view})", ""]
    for scheme in sorted(summary["scheme"].unique()):
        part = summary[summary["scheme"] == scheme].sort_values(
            "balanced_accuracy", ascending=False
        )
        control = folds[
            (folds["scheme"] == scheme) & (folds["view"] == view) & (folds["control"] == "permuted")
        ]
        lines += [
            f"## `{scheme}`",
            "",
            "| features | model | folds | balanced accuracy | chance | above chance | ECE |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in part.itertuples():
            spread = (
                "" if pd.isna(row.balanced_accuracy_sd) else f" ± {row.balanced_accuracy_sd:.3f}"
            )
            lines.append(
                f"| {row.features} | {row.model} | {row.folds} | "
                f"{row.balanced_accuracy:.3f}{spread} | {row.chance:.3f} | "
                f"{row.above_chance:+.3f} | {row.ece:.3f} |"
            )
        if len(control):
            lines += [
                "",
                f"Permutation controls on the same folds: mean balanced accuracy "
                f"{control['balanced_accuracy'].mean():.3f} against a mean chance of "
                f"{control['chance'].mean():.3f}.",
            ]
        comparison = paired_comparison(folds, scheme, view)
        if len(comparison):
            lines += ["", "Paired across folds (Wilcoxon signed-rank):", ""]
            lines += [
                "| a | b | folds | median difference | a wins | p |",
                "|---|---|---:|---:|---:|---:|",
            ]
            for row in comparison.itertuples():
                lines.append(
                    f"| {row.a} | {row.b} | {row.folds} | {row.median_difference:+.3f} | "
                    f"{row.a_wins}/{row.folds} | {row.p_value:.3f} |"
                )
        lines.append("")

    text = "\n".join(lines)
    typer.echo(text)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text)
        typer.secho(f"wrote {out}", fg=typer.colors.GREEN)


@app.command("lab-discriminator")
def lab_discriminator(
    ibl: Path = typer.Option(Path("data/stores/ibl")),
    allen: Path = typer.Option(Path("data/stores/allen")),
    cache: Path = typer.Option(Path("data/features")),
    out: Path = typer.Option(Path("results/lab_discriminator")),
    which: str = typer.Option(
        "bandpower_full,bandpower_clean,w2v2_frozen", help="Comma-separated feature sets."
    ),
    seed: int = typer.Option(0),
) -> None:
    """Ask how easily a classifier can tell which lab a chunk came from.

    This turns DEVIATIONS D11 from an observation into a number. If the full band separates the
    two datasets almost perfectly and a band restricted below 100 Hz does not, the difference
    bounds how much of any cross-lab result could have been filtering rather than anatomy.

    Each fold holds out one probe from *each* dataset. Holding out a single group would leave a
    test set containing only one label, where accuracy and area under the curve are both
    meaningless, and the question is precisely whether the separation generalises to probes the
    classifier has not seen.
    """
    from sklearn.metrics import roc_auc_score

    from lfpaudit.data.splits import Split, verify_no_leakage
    from lfpaudit.models.baselines import fit_predict

    set_seed(seed)
    corpus = _load_corpus(ibl, allen)
    index = corpus.index
    positions = {int(cid): i for i, cid in enumerate(index["chunk_id"])}
    dataset_label = (index["dataset"] == "allen").astype(int).to_numpy()

    by_dataset = {
        name: sorted(index[index["dataset"] == name]["group"].astype(str).unique())
        for name in ("ibl", "allen")
    }
    n_folds = max(len(v) for v in by_dataset.values())
    all_groups = index["group"].astype(str)

    def select(names: list[str]) -> list[int]:
        return sorted(int(i) for i in index.loc[all_groups.isin(names), "chunk_id"])

    def rotate(names: list[str], offset: int) -> tuple[str, str]:
        """The test group for this fold, and a different one held out for validation."""
        return names[offset % len(names)], names[(offset + 1) % len(names)]

    rows = []
    for name in [w.strip() for w in which.split(",") if w.strip()]:
        table = _build_feature_table(corpus, name, cache, None, rebuild=False)
        for fold_index in range(n_folds):
            test_ibl, val_ibl = rotate(by_dataset["ibl"], fold_index)
            test_allen, val_allen = rotate(by_dataset["allen"], fold_index)
            test_groups = sorted({test_ibl, test_allen})
            val_groups = sorted({val_ibl, val_allen} - set(test_groups))

            train_groups = sorted(set(all_groups) - set(test_groups) - set(val_groups))
            split = Split(
                kind="cross_session",
                seed=seed,
                train=select(train_groups),
                val=select(val_groups),
                test=select(test_groups),
                train_groups=train_groups,
                val_groups=val_groups,
                test_groups=test_groups,
            )
            verify_no_leakage(split, index)

            train = np.array([positions[i] for i in split.train], dtype=np.int64)
            test = np.array([positions[i] for i in split.test], dtype=np.int64)
            probs = fit_predict(
                "logreg", table[train], dataset_label[train], table[test], seed=seed, n_classes=2
            )
            predicted = probs.argmax(axis=1)
            truth = dataset_label[test]
            per_class = [float((predicted[truth == c] == c).mean()) for c in (0, 1)]
            rows.append(
                {
                    "features": name,
                    "held_out_ibl": test_ibl,
                    "held_out_allen": test_allen,
                    "n_test": len(test),
                    "accuracy": float((predicted == truth).mean()),
                    "balanced_accuracy": float(np.mean(per_class)),
                    "auc": float(roc_auc_score(truth, probs[:, 1])),
                }
            )

    frame = pd.DataFrame(rows)
    Path(out).mkdir(parents=True, exist_ok=True)
    frame.to_csv(Path(out) / "folds.csv", index=False)
    summary = frame.groupby("features")[["balanced_accuracy", "auc"]].agg(["mean", "std"]).round(4)
    typer.echo(summary.to_string())
    summary.to_csv(Path(out) / "summary.csv")

    means = frame.groupby("features")["auc"].mean()
    if {"bandpower_full", "bandpower_clean"} <= set(means.index):
        full, clean = means["bandpower_full"], means["bandpower_clean"]
        typer.echo(
            f"\nArea under the curve falls from {full:.3f} to {clean:.3f} when the ripple band "
            f"is removed: a drop of {full - clean:.3f}."
        )
    if "w2v2_frozen" in means.index:
        typer.echo(
            f"Frozen audio embeddings identify the source lab at {means['w2v2_frozen']:.3f}, "
            "which is what a representation dominated by acquisition looks like."
        )
    typer.secho(f"wrote {out}", fg=typer.colors.GREEN)


if __name__ == "__main__":  # pragma: no cover
    app()
