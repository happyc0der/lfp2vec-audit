import json

from lfpaudit.data.manifest import RunManifest, git_state, sha256_dir, sha256_file


def test_sha256_file_is_stable(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"lfp")
    first = sha256_file(path)
    assert first == sha256_file(path)
    path.write_bytes(b"lfp2")
    assert sha256_file(path) != first


def test_sha256_dir_covers_expected_patterns(tmp_path):
    (tmp_path / "index.parquet").write_bytes(b"x")
    (tmp_path / "store.json").write_text("{}")
    (tmp_path / "ignored.txt").write_text("no")
    hashes = sha256_dir(tmp_path)
    assert set(hashes) == {"index.parquet", "store.json"}


def test_manifest_is_written_before_and_after_a_run(tmp_path):
    manifest = RunManifest.create("unit", seed=3, device="cpu", config={"k": 1})
    manifest.write(tmp_path)
    started = json.loads((tmp_path / "manifest.json").read_text())
    assert started["status"] == "started"
    assert started["seed"] == 3
    assert started["config"] == {"k": 1}
    assert started["finished_at"] == ""

    manifest.finish(tmp_path, status="ok")
    done = json.loads((tmp_path / "manifest.json").read_text())
    assert done["status"] == "ok"
    assert done["finished_at"]
    assert done["run_id"] == started["run_id"]


def test_git_state_reports_a_commit_or_unknown():
    state = git_state()
    assert set(state) == {"commit", "dirty", "branch"}
    assert isinstance(state["commit"], str)
