import pytest

from lfpaudit import REGIONS, UNKNOWN
from lfpaudit.data.labels import VISUAL_CORTEX_ACRONYMS, label_counts, map_acronym


@pytest.mark.parametrize("acronym", sorted(VISUAL_CORTEX_ACRONYMS))
def test_visual_family_collapses_to_vis(acronym):
    assert map_acronym(acronym) == "VIS"


@pytest.mark.parametrize(
    ("acronym", "expected"),
    [
        ("CA1", "CA1"),
        ("CA1sp", "CA1"),
        ("CA1slm", "CA1"),
        ("CA2", "CA2"),
        ("CA3so", "CA3"),
        ("DG", "DG"),
        ("DG-sg", "DG"),
        ("DGmo", "DG"),
        ("VISp2/3", "VIS"),
        ("VISpm6a", "VIS"),
    ],
)
def test_known_acronyms(acronym, expected):
    assert map_acronym(acronym) == expected


@pytest.mark.parametrize("acronym", ["root", "void", "", None, "LP", "PO", "TH", "SUB", "MB"])
def test_unknown_acronyms(acronym):
    assert map_acronym(acronym) == UNKNOWN


def test_label_counts_has_every_region_and_unknown():
    counts = label_counts(["CA1", "CA1sp", "VISp", "LP", None])
    assert set(counts) == set(REGIONS) | {UNKNOWN}
    assert counts["CA1"] == 2
    assert counts["VIS"] == 1
    assert counts[UNKNOWN] == 2
    assert counts["CA2"] == 0
