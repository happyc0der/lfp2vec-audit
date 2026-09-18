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
from dataclasses import asdict, replace
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
from lfpaudit.eval.metrics import chance_level_band, evaluate, expand_probabilities, softmax
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


# ---------------------------------------------------------------------------------------------
# Stage 3: the wav2vec2 fine-tune.
# ---------------------------------------------------------------------------------------------

finetune_app = typer.Typer(help="Fine-tune the audio checkpoint for region decoding.")
app.add_typer(finetune_app, name="finetune")


@finetune_app.command("smoke")
def finetune_smoke(
    device: str = typer.Option(None, help="Force a device."),
    steps: int = typer.Option(15, help="Optimiser steps."),
) -> None:
    """Assert the training loop learns, before any real run is launched."""
    from lfpaudit.models.train import smoke as run_smoke

    policy = pick_device(device)
    result = run_smoke(device=policy.device, steps=steps)
    typer.echo(result["model"])
    typer.echo(
        f"loss {result['first_loss']:.4f} -> {result['last_loss']:.4f} "
        f"over {result['steps']} steps on {result['device']}"
    )
    typer.secho("finetune smoke passed", fg=typer.colors.GREEN)


def _resolve_fold(index, scheme: str, fold_name: str | None, seed: int):
    from lfpaudit.eval.folds import build_schemes

    schemes = build_schemes(index, seed=seed)
    if scheme not in schemes:
        raise typer.BadParameter(f"unknown scheme {scheme!r}; have {sorted(schemes)}")
    folds = schemes[scheme]
    if fold_name is None:
        return folds[0]
    for fold in folds:
        if fold.name == fold_name:
            return fold
    raise typer.BadParameter(f"unknown fold {fold_name!r}; have {[f.name for f in folds]}")


@finetune_app.command("run")
def finetune_run(
    scheme: str = typer.Option("cross_lab_ibl_to_allen", help="Evaluation scheme."),
    fold: str = typer.Option(None, help="Which fold; defaults to the first."),
    ibl: Path = typer.Option(Path("data/stores/ibl")),
    allen: Path = typer.Option(Path("data/stores/allen")),
    out: Path = typer.Option(Path("results/finetune")),
    device: str = typer.Option(None),
    seed: int = typer.Option(0),
    epochs: int = typer.Option(10),
    max_per_class: int = typer.Option(6000),
    batch_size: int = typer.Option(8),
    lowpass: float = typer.Option(None, help="Low-pass corner in Hz for the harmonised band."),
    permute_labels: bool = typer.Option(False, help="Shuffle training labels: a leakage check."),
    save_model: bool = typer.Option(
        False, help="Persist weights so later stages can ablate without retraining."
    ),
    max_rate_check: bool = typer.Option(True, help="Measure throughput before committing."),
    budget_hours: float = typer.Option(3.0, help="Refuse to start if the estimate exceeds this."),
) -> None:
    """Fine-tune on one fold, scoring every target group and saving everything Stage 4 needs."""
    from lfpaudit.eval.folds import Fold
    from lfpaudit.eval.probe import lab_identity_auc
    from lfpaudit.eval.runner import CLASS_VIEWS
    from lfpaudit.models.dataset import ChunkDataset, balanced_sample, class_counts
    from lfpaudit.models.lfp2vec_lite import build_model
    from lfpaudit.models.train import (
        FineTuneConfig,
        embed,
        measure_throughput,
        predict_with_embeddings,
        train,
    )

    set_seed(seed)
    policy = pick_device(device)
    corpus = _load_corpus(ibl, allen)
    index = corpus.index
    labels_all = index["region"].map(REGION_TO_INDEX).to_numpy()
    positions = {int(c): i for i, c in enumerate(index["chunk_id"])}

    chosen = _resolve_fold(index, scheme, fold, seed)
    sizes = verify_no_leakage(chosen.split, index)

    # Every fold of a cross-lab scheme shares one training set and differs only in which target
    # group it scores. Training once per target probe would repeat the same two hours ten times,
    # so the test set here is the union of all of them and per-probe scores are recovered from
    # the saved predictions afterwards.
    if scheme.startswith("cross_lab") and fold is None:
        from lfpaudit.eval.folds import build_schemes

        siblings = build_schemes(index, seed=seed)[scheme]
        pooled = sorted({i for sibling in siblings for i in sibling.split.test})
        chosen = Fold(
            name="all_target_groups",
            scheme=scheme,
            split=replace(
                chosen.split,
                test=pooled,
                test_groups=sorted({g for s in siblings for g in s.split.test_groups}),
            ),
        )
        sizes = verify_no_leakage(chosen.split, index)
    typer.echo(f"{scheme} / fold {chosen.name}: {sizes}")

    config = FineTuneConfig(
        epochs=epochs,
        max_per_class=max_per_class,
        batch_size=batch_size,
        lowpass_hz=lowpass,
        seed=seed,
    )

    def rows_for(ids):
        return np.array([positions[i] for i in ids], dtype=np.int64)

    train_rows = rows_for(chosen.split.train)
    keep = balanced_sample(labels_all[train_rows], train_rows, config.max_per_class, seed=seed)
    train_rows = train_rows[keep]
    val_rows = rows_for(chosen.split.val)
    if len(val_rows) > config.max_val_chunks:
        val_rows = np.random.default_rng(seed).choice(
            val_rows, config.max_val_chunks, replace=False
        )
    test_rows = rows_for(chosen.split.test)

    train_y = labels_all[train_rows]
    if permute_labels:
        train_y = np.random.default_rng(seed).permutation(train_y)
    typer.echo(f"train classes: {class_counts(train_y, list(REGIONS))}")

    ids = index["chunk_id"].to_numpy()
    make = lambda rows, y: ChunkDataset(  # noqa: E731
        corpus.take, ids[rows], y, fs=corpus.fs, lowpass_hz=config.lowpass_hz
    )
    train_set = make(train_rows, train_y)
    val_set = make(val_rows, labels_all[val_rows])
    test_set = make(test_rows, labels_all[test_rows])

    model, info = build_model(
        num_labels=len(REGIONS), freeze_feature_encoder=config.freeze_feature_encoder
    )
    typer.echo(info.describe())

    # Gate: size the run from measured throughput rather than an estimate, and refuse to start a
    # run that will not fit the budget. Discovering that at hour four is the failure this avoids.
    if max_rate_check:
        measured = measure_throughput(model, train_set, policy.device, config, steps=12)
        rate = measured["chunks_per_second"]
        hours = (len(train_rows) * config.epochs) / max(rate, 1e-9) / 3600
        typer.echo(f"measured {rate:.1f} chunks/s -> {hours:.2f} h for {config.epochs} epochs")
        if hours > budget_hours:
            typer.secho(
                f"GATE FAILED: estimated {hours:.2f} h exceeds the {budget_hours:.1f} h budget. "
                "Reduce --epochs or --max-per-class.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

    tag = f"{scheme}__{chosen.name}__seed{seed}"
    if config.lowpass_hz:
        tag += f"__lp{int(config.lowpass_hz)}"
    if permute_labels:
        tag += "__permuted"

    manifest = RunManifest.create(
        experiment=f"finetune_{tag}",
        seed=seed,
        device=policy.device,
        config={
            "scheme": scheme,
            "fold": chosen.name,
            "permuted": permute_labels,
            "model": asdict(info) if hasattr(info, "__dataclass_fields__") else str(info),
            "train": asdict(config),
            "n_train": int(len(train_rows)),
            "n_val": int(len(val_rows)),
            "n_test": int(len(test_rows)),
        },
        data_dir=ibl,
    )
    run_dir = Path(out) / tag / manifest.run_id
    manifest.write(run_dir)

    history = train(model, train_set, val_set, config, policy.device, run_dir)

    logits, test_y, test_embeddings = predict_with_embeddings(
        model, test_set, policy.device, config.batch_size
    )
    probs = softmax(logits)
    metrics = {}
    for view, allowed in CLASS_VIEWS.items():
        keep_view = np.isin(test_y, [REGION_TO_INDEX[n] for n in allowed])
        if not keep_view.any():
            continue
        report = evaluate(probs[keep_view], test_y[keep_view])
        metrics[view] = report.to_dict()
        typer.echo(f"{view}: {report.summary_line()}")

    frame = pd.DataFrame(
        {
            "chunk_id": ids[test_rows],
            "label": test_y,
            "group": index.iloc[test_rows]["group"].to_numpy(),
        }
    )
    for i, region in enumerate(REGIONS):
        frame[f"logit_{region}"] = logits[:, i].astype(np.float32)
    frame.to_parquet(run_dir / "predictions.parquet", index=False)

    # Per-group scores, so a cross-lab run reports a spread over target probes rather than one
    # pooled number, and can be compared against the Stage 2 baselines fold for fold.
    per_group = []
    for name, part in frame.groupby("group"):
        rows = part.index.to_numpy()
        for view, allowed in CLASS_VIEWS.items():
            allowed_idx = [REGION_TO_INDEX[n] for n in allowed]
            keep_view = np.isin(test_y[rows], allowed_idx)
            if keep_view.sum() < 2 or len(np.unique(test_y[rows][keep_view])) < 2:
                continue
            report = evaluate(probs[rows][keep_view], test_y[rows][keep_view])
            per_group.append(
                {
                    "scheme": scheme,
                    "group": name,
                    "view": view,
                    "seed": seed,
                    "lowpass_hz": config.lowpass_hz or 0.0,
                    "permuted": permute_labels,
                    "n_test": int(keep_view.sum()),
                    "chance": report.chance,
                    "balanced_accuracy": report.balanced_accuracy,
                    "macro_f1": report.macro_f1,
                    "ece": report.ece,
                    "nll": report.nll,
                    "classes_present": "|".join(report.classes_present),
                }
            )
    pd.DataFrame(per_group).to_csv(run_dir / "per_group.csv", index=False)

    # Everything a post-hoc stage needs, so calibration, abstention and ablation never require
    # retraining. Stage 3 saved only test logits and had to be re-run to answer those questions.
    np.save(run_dir / "test_embeddings.npy", test_embeddings.astype(np.float32))

    val_logits, val_y, val_embeddings = predict_with_embeddings(
        model, val_set, policy.device, config.batch_size
    )
    val_frame = pd.DataFrame(
        {
            "chunk_id": ids[val_rows],
            "label": val_y,
            "group": index.iloc[val_rows]["group"].to_numpy(),
        }
    )
    for i, region in enumerate(REGIONS):
        val_frame[f"logit_{region}"] = val_logits[:, i].astype(np.float32)
    val_frame.to_parquet(run_dir / "val_predictions.parquet", index=False)
    np.save(run_dir / "val_embeddings.npy", val_embeddings.astype(np.float32))

    if save_model:
        model.save_pretrained(run_dir / "model")
        typer.echo(f"saved weights to {run_dir / 'model'}")

    # The other half of the audit: does fine-tuning remove the acquisition structure that made
    # the frozen representation useless across labs?
    rng = np.random.default_rng(seed)
    probe_rows = rng.choice(len(index), size=min(6000, len(index)), replace=False)
    probe_set = ChunkDataset(
        corpus.take,
        ids[probe_rows],
        labels_all[probe_rows],
        fs=corpus.fs,
        lowpass_hz=config.lowpass_hz,
    )
    embeddings = embed(model, probe_set, policy.device, config.batch_size)
    probe = lab_identity_auc(
        embeddings,
        index.iloc[probe_rows]["group"],
        index.iloc[probe_rows]["dataset"],
        seed=seed,
    )
    typer.echo(probe.summary_line("lab identity after fine-tuning"))
    probe.folds.to_csv(run_dir / "lab_identity.csv", index=False)
    np.save(run_dir / "probe_embeddings.npy", embeddings.astype(np.float32))

    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": metrics,
                "lab_identity_auc": probe.auc,
                "epochs_run": len(history),
                "best_val_balanced_accuracy": max(h.val_balanced_accuracy for h in history),
            },
            indent=2,
        )
    )
    manifest.finish(run_dir, status="ok")
    typer.secho(f"wrote {run_dir}", fg=typer.colors.GREEN)


@finetune_app.command("report")
def finetune_report(
    results: Path = typer.Option(Path("results/finetune")),
    baselines: Path = typer.Option(Path("results/baselines")),
    view: str = typer.Option("4class"),
    out: Path = typer.Option(None, help="Write a markdown document here."),
) -> None:
    """Collect every fine-tune run and place it beside the Stage 2 baselines.

    A fine-tune number on its own says nothing. The comparisons that matter are against electrode
    position, which needs no signal at all, and against the frozen checkpoint, which needed no
    training. Both are read from the committed baseline table so the two stages cannot drift.
    """
    runs = sorted(Path(results).glob("*/*/per_group.csv"))
    if not runs:
        raise typer.BadParameter(f"no runs found under {results}")

    frames = []
    for path in runs:
        frame = pd.read_csv(path)
        frame["run"] = path.parent.parent.name
        metrics_path = path.parent / "metrics.json"
        if metrics_path.exists():
            frame["lab_identity_auc"] = json.loads(metrics_path.read_text()).get(
                "lab_identity_auc", float("nan")
            )
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table = table[table["view"] == view]

    lines = [f"# Fine-tune results ({view})", ""]
    for scheme in sorted(table["scheme"].unique()):
        part = table[(table["scheme"] == scheme) & (~table["permuted"])]
        lines += [f"## `{scheme}`", ""]
        lines += [
            "| run | band | groups | balanced accuracy | chance | ECE | lab identity |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for run, group in part.groupby("run"):
            band = "<=100 Hz" if group["lowpass_hz"].iloc[0] else "full"
            spread = group["balanced_accuracy"].std()
            spread_text = "" if pd.isna(spread) else f" +/- {spread:.3f}"
            auc = group["lab_identity_auc"] if "lab_identity_auc" in group else None
            auc_text = "n/a" if auc is None or pd.isna(auc.iloc[0]) else f"{auc.iloc[0]:.3f}"
            lines.append(
                f"| {run} | {band} | {len(group)} | "
                f"{group['balanced_accuracy'].mean():.3f}{spread_text} | "
                f"{group['chance'].mean():.3f} | {group['ece'].mean():.3f} | {auc_text} |"
            )
        lines.append("")

        baseline_path = Path(baselines) / "folds.csv"
        if baseline_path.exists():
            folds = pd.read_csv(baseline_path)
            reference = folds[
                (folds["scheme"] == scheme)
                & (folds["view"] == view)
                & (folds["model"] == "logreg")
                & (folds["control"] == "model")
            ]
            if len(reference):
                lines += ["Stage 2 baselines on the same scheme, for comparison:", ""]
                lines += ["| features | balanced accuracy | ECE |", "|---|---:|---:|"]
                summary = reference.groupby("features")[["balanced_accuracy", "ece"]].mean()
                ordered = summary.sort_values("balanced_accuracy", ascending=False)
                for name, row in ordered.iterrows():
                    lines.append(f"| {name} | {row['balanced_accuracy']:.3f} | {row['ece']:.3f} |")
                lines.append("")

    # Within-lab schemes produce one run per held-out session, so the comparison against the
    # baselines has to be paired on those sessions. Comparing a few folds against a mean over all
    # of them measures which sessions happened to be run, not which method is better.
    baseline_path = Path(baselines) / "folds.csv"
    for scheme in sorted(table["scheme"].unique()):
        if not scheme.startswith("loso") or not baseline_path.exists():
            continue
        tuned = table[(table["scheme"] == scheme) & (~table["permuted"])].set_index("group")[
            "balanced_accuracy"
        ]
        if len(tuned) < 2:
            continue
        folds = pd.read_csv(baseline_path)
        reference = folds[
            (folds["scheme"] == scheme)
            & (folds["view"] == view)
            & (folds["model"] == "logreg")
            & (folds["control"] == "model")
        ]
        lines += [
            f"### `{scheme}`: paired against the baselines on the same {len(tuned)} sessions",
            "",
            "| features | baseline | fine-tune | difference | fine-tune wins |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, part in reference.groupby("features"):
            shared = part.set_index("fold")["balanced_accuracy"].reindex(tuned.index).dropna()
            if len(shared) < 2:
                continue
            difference = tuned.reindex(shared.index) - shared
            lines.append(
                f"| {name} | {shared.mean():.3f} | {tuned.reindex(shared.index).mean():.3f} | "
                f"{difference.mean():+.3f} | {int((difference > 0).sum())}/{len(difference)} |"
            )
        lines.append("")

    permuted = table[table["permuted"]]
    if len(permuted):
        lines += [
            "## Leakage check",
            "",
            f"Permuted-label fine-tune: balanced accuracy "
            f"{permuted['balanced_accuracy'].mean():.3f} against a chance of "
            f"{permuted['chance'].mean():.3f}.",
            "",
        ]

    text = "\n".join(lines)
    typer.echo(text)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text)
        typer.secho(f"wrote {out}", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------------------------
# Stage 4: calibration, abstention and ablations. All post-hoc except the ablation forward passes.
# ---------------------------------------------------------------------------------------------


def _latest_run(root: Path, tag: str) -> Path:
    """The most recent run directory for a tag, or a clear error naming what is missing."""
    candidates = sorted(Path(root).glob(f"{tag}/*/manifest.json"))
    if not candidates:
        raise typer.BadParameter(f"no run found under {root}/{tag}")
    return candidates[-1].parent


def _require(path: Path, why: str) -> Path:
    if not path.exists():
        raise typer.BadParameter(
            f"{path} is missing, so {why} cannot be computed. Runs made before Stage 4 saved only "
            "test logits; re-run with --save-model."
        )
    return path


@app.command()
def calibrate(
    run: str = typer.Option("cross_lab_ibl_to_allen__all_target_groups__seed0"),
    results: Path = typer.Option(Path("results/finetune_v2")),
    out: Path = typer.Option(Path("results/calibration")),
) -> None:
    """Fit one temperature in-lab, apply it cross-lab, and report whether it reaches.

    The paper's Broader Impact section asks for calibrated uncertainty. This measures whether the
    standard way of providing it survives the shift that breaks the model.
    """
    from lfpaudit.eval.calibration import calibration_report, fit_temperature

    run_dir = _latest_run(results, run)
    val = pd.read_parquet(_require(run_dir / "val_predictions.parquet", "temperature fitting"))
    test = pd.read_parquet(run_dir / "predictions.parquet")
    columns = [f"logit_{r}" for r in REGIONS]

    val_logits, val_y = val[columns].to_numpy(), val["label"].to_numpy()
    test_logits, test_y = test[columns].to_numpy(), test["label"].to_numpy()

    temperature = fit_temperature(val_logits, val_y)
    in_lab = calibration_report(val_logits, val_y, temperature)
    cross_lab = calibration_report(test_logits, test_y, temperature)
    # What the target lab would have needed, which is the size of the gap the fix does not close.
    oracle = fit_temperature(test_logits, test_y)
    oracle_report = calibration_report(test_logits, test_y, oracle)

    typer.echo(f"temperature fitted in-lab: {temperature:.3f}")
    typer.echo(in_lab.summary_line("  in-lab  "))
    typer.echo(cross_lab.summary_line("  cross-lab"))
    typer.echo(f"temperature the target lab would have needed: {oracle:.3f}")
    typer.echo(oracle_report.summary_line("  cross-lab, oracle T"))

    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run}.json").write_text(
        json.dumps(
            {
                "run": run,
                "temperature_in_lab": temperature,
                "temperature_oracle": oracle,
                "in_lab": in_lab.to_dict(),
                "cross_lab": cross_lab.to_dict(),
                "cross_lab_oracle": oracle_report.to_dict(),
            },
            indent=2,
        )
    )
    typer.secho(f"wrote {out / f'{run}.json'}", fg=typer.colors.GREEN)


@app.command()
def abstain(
    run: str = typer.Option("cross_lab_ibl_to_allen__all_target_groups__seed0"),
    results: Path = typer.Option(Path("results/finetune_v2")),
    out: Path = typer.Option(Path("results/abstention")),
) -> None:
    """Can the model tell when not to trust itself, from its output or its representation?

    Stage 3 showed confidence cannot: 0.98 while at chance. The representation identifies the
    source lab at 0.997, so distance from the training distribution should catch what confidence
    misses. Both are scored here the same way.
    """
    from lfpaudit.eval.selective import (
        MahalanobisScorer,
        confidence_scores,
        risk_coverage,
        separation_auc,
    )

    run_dir = _latest_run(results, run)
    columns = [f"logit_{r}" for r in REGIONS]
    val = pd.read_parquet(_require(run_dir / "val_predictions.parquet", "abstention"))
    test = pd.read_parquet(run_dir / "predictions.parquet")
    val_embeddings = np.load(_require(run_dir / "val_embeddings.npy", "the distance score"))
    test_embeddings = np.load(_require(run_dir / "test_embeddings.npy", "the distance score"))

    test_logits, test_y = test[columns].to_numpy(), test["label"].to_numpy()
    correct = test_logits.argmax(axis=1) == test_y

    scores = confidence_scores(test_logits)
    val_scores = confidence_scores(val[columns].to_numpy())
    scorer = MahalanobisScorer().fit(val_embeddings)
    scores["mahalanobis"] = scorer.score(test_embeddings)
    val_scores["mahalanobis"] = scorer.score(val_embeddings)

    rows, curves = [], {}
    for name, values in scores.items():
        curve = risk_coverage(values, correct, name)
        curves[name] = {"coverage": curve.coverage, "risk": curve.risk}
        separation = separation_auc(val_scores[name], values)
        rows.append(
            {
                "run": run,
                "score": name,
                "aurc": curve.area,
                "risk_full": curve.risk_at_full,
                "risk_half": curve.risk_at_half,
                "risk_fifth": curve.risk_at_fifth,
                "in_vs_out_auc": separation,
            }
        )
        typer.echo(f"{curve.summary_line()}; separates in-lab from cross-lab at {separation:.3f}")

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{run}.csv", index=False)
    (out / f"{run}_curves.json").write_text(json.dumps(curves))
    typer.secho(f"wrote {out / f'{run}.csv'}", fg=typer.colors.GREEN)


@app.command()
def ablate(
    model_kind: str = typer.Option("bandpower", help="bandpower, frozen or finetuned."),
    scheme: str = typer.Option("cross_lab_ibl_to_allen"),
    run: str = typer.Option("cross_lab_ibl_to_allen__all_target_groups__seed0"),
    results: Path = typer.Option(Path("results/finetune_v2")),
    ibl: Path = typer.Option(Path("data/stores/ibl")),
    allen: Path = typer.Option(Path("data/stores/allen")),
    cache: Path = typer.Option(Path("data/features")),
    out: Path = typer.Option(Path("results/ablations")),
    n_test: int = typer.Option(10000, help="Test chunks to subsample; fixed by seed."),
    device: str = typer.Option(None),
    seed: int = typer.Option(0),
) -> None:
    """What does each model lose when a band, the waveform, or a quarter of the window goes?

    The amplitude entries are controls: per-chunk normalisation must cancel them exactly. If they
    move, the pipeline is not normalising where it claims to and nothing else here is trustworthy.
    """
    from lfpaudit.eval.ablations import ABLATIONS
    from lfpaudit.eval.folds import build_schemes
    from lfpaudit.features.bandpower import band_power
    from lfpaudit.features.embed import load_encoder, prepare_waveforms
    from lfpaudit.models.baselines import fit_predict

    set_seed(seed)
    policy = pick_device(device)
    corpus = _load_corpus(ibl, allen)
    index = corpus.index
    labels_all = index["region"].map(REGION_TO_INDEX).to_numpy()
    positions = {int(c): i for i, c in enumerate(index["chunk_id"])}
    ids = index["chunk_id"].to_numpy()

    folds = build_schemes(index, seed=seed)[scheme]
    train_rows = np.array([positions[i] for i in folds[0].split.train], dtype=np.int64)
    test_rows = np.array(
        sorted({positions[i] for fold in folds for i in fold.split.test}), dtype=np.int64
    )
    rng = np.random.default_rng(seed)
    if len(test_rows) > n_test:
        test_rows = np.sort(rng.choice(test_rows, n_test, replace=False))
    typer.echo(f"{model_kind}: {len(train_rows)} train, {len(test_rows)} test chunks")

    encoder, tuned = None, None
    if model_kind == "frozen":
        encoder = load_encoder(device=policy.device)
    elif model_kind == "finetuned":
        from lfpaudit.models.train import load_model

        tuned = load_model(_latest_run(results, run), device=policy.device)
    elif model_kind != "bandpower":
        raise typer.BadParameter("model_kind must be bandpower, frozen or finetuned")

    def score(rows: np.ndarray, transform) -> np.ndarray:
        """Feature or logit matrix for rows, with the transform applied to the raw chunks."""
        out_chunks = []
        for start in range(0, len(rows), 512):
            block = corpus.take(ids[rows[start : start + 512]])
            out_chunks.append(transform(block, corpus.fs))
        chunks = np.concatenate(out_chunks)
        if model_kind == "bandpower":
            return band_power(chunks, fs=corpus.fs)
        import torch

        waveforms = prepare_waveforms(chunks, fs=corpus.fs)
        outputs = []
        model = encoder if encoder is not None else tuned
        with torch.no_grad():
            for start in range(0, len(waveforms), 32):
                batch = torch.from_numpy(waveforms[start : start + 32]).to(policy.device)
                if encoder is not None:
                    outputs.append(model(batch).last_hidden_state.mean(1).float().cpu().numpy())
                else:
                    outputs.append(model(batch).logits.float().cpu().numpy())
        return np.concatenate(outputs)

    identity = ABLATIONS["none"].apply
    train_features = None
    if model_kind != "finetuned":
        train_features = score(train_rows, identity)

    rows_out = []
    for name, ablation in ABLATIONS.items():
        test_features = score(test_rows, ablation.apply)
        if model_kind == "finetuned":
            probs = softmax(test_features)
        else:
            probs = fit_predict(
                "logreg", train_features, labels_all[train_rows], test_features, seed=seed
            )
        report = evaluate(probs, labels_all[test_rows])
        rows_out.append(
            {
                "model": model_kind,
                "scheme": scheme,
                "ablation": name,
                "question": ablation.question,
                "is_control": ablation.is_control,
                "n_test": len(test_rows),
                "chance": report.chance,
                "balanced_accuracy": report.balanced_accuracy,
                "ece": report.ece,
            }
        )
        typer.echo(f"  {name:18s} bal_acc {report.balanced_accuracy:.3f}  ece {report.ece:.3f}")

    frame = pd.DataFrame(rows_out)
    reference = float(frame.loc[frame["ablation"] == "none", "balanced_accuracy"].iloc[0])
    frame["delta"] = frame["balanced_accuracy"] - reference

    controls = frame[frame["is_control"]]["delta"].abs().max()
    if controls > 0.005:
        typer.secho(
            f"GATE FAILED: amplitude control moved balanced accuracy by {controls:.4f}. "
            "Per-chunk normalisation should cancel it exactly; nothing else here is trustworthy "
            "until that is explained.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(f"amplitude controls null to {controls:.4f}, as they must be")

    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / f"{model_kind}__{scheme}.csv", index=False)
    typer.secho(f"wrote {out / f'{model_kind}__{scheme}.csv'}", fg=typer.colors.GREEN)


if __name__ == "__main__":  # pragma: no cover
    app()
