import json

from typer.testing import CliRunner

from lfpaudit.cli import app

runner = CliRunner()


def test_info_runs():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, result.output
    assert "selected device" in result.output


def test_smoke_passes_its_own_gates(tmp_path):
    result = runner.invoke(app, ["smoke", "--device", "cpu", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "smoke passed" in result.output

    runs = list(tmp_path.iterdir())
    assert len(runs) == 1
    metrics = json.loads((runs[0] / "metrics.json").read_text())
    manifest = json.loads((runs[0] / "manifest.json").read_text())

    assert manifest["status"] == "ok"
    assert metrics["model"]["balanced_accuracy"] > metrics["chance"] + 0.15
    assert metrics["permutation_control"]["balanced_accuracy"] <= metrics["chance"] + 0.15
    assert (runs[0] / "split.json").exists()


def test_verify_split_command(tmp_path, synthetic_store):
    from lfpaudit.data.splits import make_split

    split_path = tmp_path / "split.json"
    make_split(synthetic_store.index, kind="cross_session", seed=0).to_json(split_path)
    index_path = synthetic_store.path / "index.parquet"

    result = runner.invoke(app, ["verify-split", str(split_path), str(index_path)])
    assert result.exit_code == 0, result.output
    assert "is clean" in result.output
