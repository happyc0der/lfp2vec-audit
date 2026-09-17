"""Guards against the package being incomplete in a way tests would otherwise not notice.

Both checks here exist because of a real failure: a bare ``data/`` line in ``.gitignore``
matched the source package ``lfpaudit/data/`` as well as the data cache, so an entire
subpackage was absent from the first commit. Every test still passed locally, because pytest
imports from the working tree, and CI failed with a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
from pathlib import Path

import pytest

import lfpaudit

PACKAGE_ROOT = Path(lfpaudit.__file__).parent
REPO_ROOT = PACKAGE_ROOT.parent


def _tracked_files() -> set[Path] | None:
    """Files known to git, or None when this is not a git checkout."""
    result = subprocess.run(
        ["git", "ls-files", "lfpaudit"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return {REPO_ROOT / line for line in result.stdout.split() if line}


def test_every_source_file_is_tracked_by_git():
    tracked = _tracked_files()
    if tracked is None:
        pytest.skip("not a git checkout")
    on_disk = {p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts}
    untracked = sorted(str(p.relative_to(REPO_ROOT)) for p in on_disk - tracked)
    assert not untracked, f"source files exist but are not committed: {untracked}"


def test_every_submodule_imports():
    """Walk the package and import everything, so a broken module cannot hide behind stubs."""
    failures = []
    for info in pkgutil.walk_packages(lfpaudit.__path__, prefix="lfpaudit."):
        try:
            importlib.import_module(info.name)
        except Exception as error:  # noqa: BLE001 - the point is to report any failure
            failures.append(f"{info.name}: {error!r}")
    assert not failures, f"modules failed to import: {failures}"


def test_expected_subpackages_are_present():
    for name in ("data", "features", "models", "eval"):
        assert (PACKAGE_ROOT / name / "__init__.py").exists(), f"missing subpackage {name}"
