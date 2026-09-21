"""Guards against the package being incomplete in a way tests would otherwise not notice.

Both checks here exist because of a real failure: a bare ``data/`` line in ``.gitignore``
matched the source package ``lfpaudit/data/`` as well as the data cache, so an entire
subpackage was absent from the first commit. Every test still passed locally, because pytest
imports from the working tree, and CI failed with a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
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


def test_readme_status_matches_the_stage_table():
    """The status banner must name the highest stage the table marks done.

    This line has gone stale twice, both times because a string replacement silently matched
    nothing while the surrounding commit reported success. A banner that claims an earlier stage
    than the repository has reached is the single most misleading thing a reader can meet first.
    """
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()

    status = [line for line in readme.splitlines() if line.startswith("> **Status:")]
    if not status:
        # A finished README carries no stage banner, and then there is nothing to go stale.
        assert "| planned |" not in readme, "README lists planned work but has no status banner"
        return
    assert len(status) == 1, f"expected at most one status banner, found {len(status)}"

    rows = dict(
        (int(number), "**done**" in state)
        for number, state in re.findall(r"^\| (\d) \|[^|]*\|([^|]*)\|", readme, re.MULTILINE)
    )
    assert rows, "no stage rows found in the experiments table"

    claimed = re.search(r"Status: Stage (\d)", status[0])
    assert claimed, f"status banner does not name a stage: {status[0]}"
    stage = int(claimed.group(1))

    # Every stage up to and including the one claimed must be done. Checking only the maximum
    # would let an unfinished earlier stage hide behind a later one that happens to be complete.
    unfinished = sorted(n for n, is_done in rows.items() if n <= stage and not is_done)
    assert not unfinished, f"banner claims Stage {stage} but stages {unfinished} are not done"
    assert rows.get(stage), f"banner claims Stage {stage}, which the table does not mark done"

    # And nothing past it may be done either, or the banner understates the repository, which is
    # how this line went stale the first time. A row explicitly folded into an earlier stage is
    # the one exception, since it describes work the claimed stage already covers.
    folded = {
        int(number)
        for number, state in re.findall(r"^\| (\d) \|[^|]*\|([^|]*)\|", readme, re.MULTILINE)
        if "folded" in state
    }
    ahead = sorted(n for n, is_done in rows.items() if n > stage and is_done and n not in folded)
    assert not ahead, f"banner claims Stage {stage} but stages {ahead} are already done"
