"""Mapping CCF acronyms to the five-way region task.

Both IBL and Allen annotate each channel with an Allen Common Coordinate Framework acronym.
The LFP2Vec code collapses the visual-cortex family into a single ``VIS`` class and keeps the
four hippocampal subfields separate. Anything else is labelled :data:`~lfpaudit.UNKNOWN` and
dropped from supervised experiments, but is still counted in the dataset card so the reader can
see how much of each probe was discarded.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from lfpaudit import REGIONS, UNKNOWN

#: Visual-cortex acronyms aggregated into ``VIS``, matching upstream ``allen_preprocessing``.
VISUAL_CORTEX_ACRONYMS: frozenset[str] = frozenset(
    {
        "VIS",
        "VISa",
        "VISal",
        "VISam",
        "VISl",
        "VISli",
        "VISmma",
        "VISmmp",
        "VISp",
        "VISpl",
        "VISpm",
        "VISpor",
        "VISrl",
    }
)

#: Hippocampal subfields kept as their own classes.
HIPPOCAMPAL_ACRONYMS: frozenset[str] = frozenset({"CA1", "CA2", "CA3", "DG"})


#: A trailing cortical-layer designation: ``2``, ``2/3``, ``6a``, ``6b``.
_LAYER_SUFFIX = re.compile(r"(?<=[A-Za-z])[0-9]+(?:/[0-9]+)?[ab]?$")


def _strip_layer_suffix(acronym: str) -> str:
    """Remove a trailing cortical-layer designation, e.g. ``VISp2/3`` -> ``VISp``.

    CCF acronyms append layer numbers to cortical areas, sometimes with an ``a``/``b``
    sublamina (``VISpm6a``). Hippocampal acronyms instead use letter suffixes for sublayers
    (``CA1slm``, ``DGsg``); those are handled separately in :func:`map_acronym`, which matches
    on the unstripped name, so stripping ``CA1`` down to ``CA`` here is harmless.
    """
    return _LAYER_SUFFIX.sub("", acronym)


def map_acronym(acronym: str | None) -> str:
    """Map one CCF acronym to a region in :data:`~lfpaudit.REGIONS` or to ``UNK``.

    The mapping is intentionally conservative: an acronym only becomes a hippocampal class when
    it starts with that subfield's name, so ``CA1sp`` and ``CA1slm`` both become ``CA1`` while
    ``CA1`` appearing inside an unrelated string cannot.
    """
    if acronym is None:
        return UNKNOWN
    name = str(acronym).strip()
    if not name or name in {"root", "void", "nan"}:
        return UNKNOWN

    base = _strip_layer_suffix(name)
    if base in VISUAL_CORTEX_ACRONYMS or name in VISUAL_CORTEX_ACRONYMS:
        return "VIS"

    # Hippocampal sublayers: CA1sp, CA1slm, CA3so, DGsg, DG-sg, DG-mo ...
    for subfield in ("CA1", "CA2", "CA3"):
        if name.startswith(subfield):
            return subfield
    if name.startswith("DG"):
        return "DG"

    return UNKNOWN


def map_acronyms(acronyms: Iterable[str | None]) -> list[str]:
    """Vectorised :func:`map_acronym`."""
    return [map_acronym(a) for a in acronyms]


def label_counts(acronyms: Iterable[str | None]) -> dict[str, int]:
    """Count region labels including ``UNK``, with every known region present as a key."""
    counts = Counter(map_acronyms(acronyms))
    ordered = {region: counts.get(region, 0) for region in REGIONS}
    ordered[UNKNOWN] = counts.get(UNKNOWN, 0)
    return ordered
