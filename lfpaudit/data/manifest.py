"""Run manifests: the record that makes a number in this repository re-derivable.

Every run writes ``manifest.json`` *before* it does any work, so an interrupted or failed run
still leaves evidence of what was attempted. The manifest pins the git commit, whether the tree
was dirty, the config, the seed, the resolved device and the hashes of the data it consumed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lfpaudit import __version__
from lfpaudit.device import environment_summary


def sha256_file(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file, so multi-gigabyte caches can be hashed without loading them."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_dir(
    path: str | Path, patterns: tuple[str, ...] = ("*.parquet", "*.f16", "*.json")
) -> dict[str, str]:
    """Hash the interesting files in a cache directory, keyed by name, sorted for determinism."""
    path = Path(path)
    hashes: dict[str, str] = {}
    for pattern in patterns:
        for file in sorted(path.glob(pattern)):
            hashes[file.name] = sha256_file(file)
    return hashes


def git_state(repo: str | Path = ".") -> dict[str, str]:
    """Current commit and dirty flag; degrades gracefully outside a git checkout."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit or "unknown",
        "dirty": "unknown" if status is None else str(bool(status.strip())).lower(),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
    }


@dataclass
class RunManifest:
    """Provenance for one run. Written to ``results/<experiment>/<run_id>/manifest.json``."""

    experiment: str
    run_id: str
    seed: int
    device: str
    config: dict[str, Any] = field(default_factory=dict)
    data_hashes: dict[str, str] = field(default_factory=dict)
    git: dict[str, str] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    status: str = "started"
    lfpaudit_version: str = __version__

    @classmethod
    def create(
        cls,
        experiment: str,
        seed: int,
        device: str,
        config: dict[str, Any] | None = None,
        data_dir: str | Path | None = None,
        run_id: str | None = None,
    ) -> RunManifest:
        return cls(
            experiment=experiment,
            run_id=run_id or f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}",
            seed=seed,
            device=device,
            config=config or {},
            data_hashes=sha256_dir(data_dir) if data_dir else {},
            git=git_state(),
            environment=environment_summary(),
            started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    def write(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "manifest.json"
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        return path

    def finish(self, directory: str | Path, status: str = "ok") -> Path:
        self.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
        self.status = status
        return self.write(directory)
