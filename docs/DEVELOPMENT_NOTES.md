# Development notes

## Claude Code was used to write much of this repository

The scaffolding, module implementations, tests and documentation in this repository were
largely generated with Claude Code (Anthropic) under my direction. I am stating this plainly
because a reader evaluating the work deserves to know how it was produced.

What that does and does not mean:

**What the model did.** Wrote the package skeleton, the chunk store, the split logic, the
metric implementations, the synthetic generator and the test suite; drafted the documentation;
proposed the experimental design's implementation details.

**What I did, and what no coding agent can do for this project.** Chose the research question
and the controls. Decided which baselines make the comparison honest and which would be
flattering but empty. Read the paper, the supplement and the upstream code, and checked the
generated pipeline against them line by line where it mattered: the label aggregation, the
window length and sampling rate, the split grouping unit, the direction of every calibration
metric. Verified that split leakage is genuinely impossible rather than merely untested.
Decided what counts as a result worth reporting and what is an artefact.

**Specific things verified by hand, not taken on trust.**

- Band-power features were checked against a manual `scipy.signal.welch` integration, and a
  pure tone at each band's centre frequency was confirmed to land in that band.
- The expected calibration error implementation was checked against a case with a known answer
  by construction (ten predictions at confidence 0.9, half correct, ECE exactly 0.4).
- The leakage verifier was tested by deliberately corrupting a clean split in three different
  ways and confirming each is caught.
- The smoke run asserts both that a real classifier beats chance on planted signal and that the
  same classifier on permuted labels does not. A pipeline that silently leaks would pass the
  first check and fail the second.
- A batch of spurious `RuntimeWarning`s traced to Apple's Accelerate BLAS, not to our data, by
  reproducing them with a plain finite matrix product. Filtered with that reasoning recorded
  rather than suppressed blindly.

**Stage 1 additions.** The data layer is where a coding agent is least trustworthy, because
almost every mistake it can make produces numbers rather than errors. What I checked myself:

- That the byte prefix of an IBL recording really is a valid recording, by reading mtscomp's own
  chop routine and confirming which header fields it rewrites and why, rather than assuming a
  truncated file would simply work.
- That the electrode table is in raw channel order, since anatomy is joined to data by position
  and a reordered table would mislabel every channel while looking healthy. The first version of
  that check was wrong, and I kept it rather than deleting it once I understood the real
  geometry.
- That the Allen reader's output orientation, units and window offsets match the file, by
  comparing against the raw dataset directly in a test rather than trusting the shapes.
- That two probe files advertising LFP data contain only zeros, by counting non-zero samples
  across the whole file, before writing a size-based screen that skips them.
- That reading one channel at a time was slow for a structural reason, by measuring both access
  patterns against the real remote file instead of reasoning from the chunk shape.

**Standing rule for this project.** No number enters the README or the note without a results
file behind it, and no experiment runs without passing the gates in `make gate`. Generated code
is treated as a draft by a fast, careless collaborator: useful, and not to be trusted about
anything that determines a scientific conclusion.
