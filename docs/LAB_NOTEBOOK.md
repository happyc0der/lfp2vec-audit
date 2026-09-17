# Lab notebook

Running log of decisions, commands, dead ends and unresolved questions. Newest entry last.
Negative results and mistakes stay in this file; they are the part of a research log worth
having.

---

## 2026-09-17 — Stage 0: scaffold

**Machine.** Apple M4 Pro, 20 cores, 24 GB unified memory, macOS 26.5.2. Python 3.11.16 via uv,
torch 2.14.0, MPS available. Secondary compute: Colab Pro on an NYU account (unused so far).

### Background established before writing code

Read the NeurIPS 2025 paper, its supplement, the upstream repository and the LFP-LOC comparison
paper. Facts that shaped the design:

- **No pretrained LFP2Vec weights are public.** No weight files in the tree, no GitHub releases,
  nothing on the HuggingFace Hub. Any reproduction means training. Recorded as D3.
- **The upstream snapshot is thin**: 5 commits, last pushed to `main` in May 2025, no `LICENSE`
  despite the README claiming MIT, README pointing at two paths that do not exist, conflicting
  `spikeinterface` pins across the conda and pip files, and an unconditional `wandb.init`.
- **An apparent bug in the SSL loop**: the checkpoint save is guarded by
  `if max_probe_acc > probe_val_acc:` with `max_probe_acc = 0`, so it can never fire on a fresh
  run, yet the next line loads that checkpoint. Noted in `DEVIATIONS.md`, not acted on.
- **The paper reports no calibration metric anywhere**, while its own Broader Impact section
  says clinical use "should include calibrated uncertainty estimates". That gap is the research
  question.
- **LFP-LOC's critique is rhetorical, not empirical**: one paragraph, no head-to-head comparison
  on shared data. Their band-power features are a fair interpretable baseline and cost nothing
  to implement, so they become the floor this audit measures against.
- **Compute reported by the authors**: the checklist declines to state GPU hours. The only
  indirect evidence is the upstream SLURM script requesting 1×A100 for 24 h.

### Decisions

1. **Skip the self-supervised stage** (D1). The paper's own ablation says audio initialisation
   plus ~6k trials matches random initialisation plus 400k, so the audio prior is doing most of
   the work. Fine-tuning from the audio checkpoint is a defensible reduced reproduction on
   laptop compute; claiming a full reproduction would not be.
2. **Cache at 1250 Hz, upsample at load** (D6). Storing 48 000 float32 samples per 3-second
   chunk would inflate the cache roughly twentyfold for no added information.
3. **Synthetic data first.** Every module is exercised end to end on a generator with planted
   per-region spectral signatures before any real byte is downloaded. This makes CI free, makes
   the smoke gate meaningful, and means a pipeline bug shows up as chance-level accuracy rather
   than as a plausible-looking number on real data.
4. **Three split kinds, one verifier.** `in_session` is deliberately leaky and labelled as an
   upper bound; `cross_session` is the honest within-lab number; `cross_lab` is the claim under
   audit. The verifier checks chunk-id disjointness, group disjointness and, for `cross_lab`,
   dataset disjointness.

### Work done

Package skeleton, config and device selection, label mapping, chunk store, synthetic generator,
splits with verifier, band-power features, metric suite, run manifests, CLI with `info`,
`smoke` and `verify-split`. 90 unit tests, all synthetic, no network.

### Things that went wrong

- **Layer-suffix parsing bug.** The first `_strip_layer_suffix` stripped only trailing digits
  and slashes, so `VISpm6a` fell through to `UNK` — a real Allen acronym that would have
  silently discarded visual-cortex channels. Caught by a parametrised test written before the
  implementation was trusted. Replaced with a regex handling `2`, `2/3`, `6a`, `6b`.
- **Spurious floating-point warnings.** `LogisticRegression.fit` emitted "divide by zero",
  "overflow" and "invalid value encountered in matmul" during the smoke run, which looked like
  a data problem. Investigated rather than suppressed: the features are all finite and in
  `[-1.8, -0.2]`, the fitted coefficients max out below 0.8, and a plain `A @ B` on finite
  random matrices reproduces all four warnings. It is Apple's Accelerate BLAS setting IEEE
  exception flags, surfaced by NumPy. Filtered at the CLI boundary and in the pytest config,
  with the reasoning recorded in both places.

  ```
  numpy: BLAS detection method "system" (Accelerate)
  A = rng.normal(size=(480,6)); B = rng.normal(size=(6,5)); A @ B
  -> divide by zero / overflow / underflow / invalid value encountered in matmul
  ```

- **CI lint passed locally but failed on the runner.** Same ruff version, same command,
  different verdict on import ordering. Ruff's automatic first-party detection depends on how
  the package is installed, so `lfpaudit` was classified as first-party locally and split
  across groups on CI. Reproduced with `ruff check --isolated`, then fixed by declaring
  `known-first-party` explicitly rather than by reformatting to whatever the runner wanted.
- **CI could not install into the runner's system Python** (PEP 668, externally managed).
  Switched to a uv venv on PATH, which also puts CI on the same Python 3.11 as local.

- **An entire subpackage was missing from the first commit.** CI failed with
  `ModuleNotFoundError: No module named 'lfpaudit.data'` while all 90 tests passed locally. The
  cause was a bare `data/` line in `.gitignore`, intended for the raw-data cache, which also
  matched the source package `lfpaudit/data/`. Nothing local noticed, because pytest imports
  from the working tree rather than from what git tracks. Anchored the rule to `/data/` and
  added `tests/test_packaging.py`, which asserts every source file is tracked by git and every
  submodule imports. The lesson worth keeping: a green local test run says nothing about
  whether the repository someone else clones is complete.

  A second consequence only became visible after the fix: ruff skips gitignored files by
  default, so the same bad rule had been silently excluding that subpackage from linting and
  formatting as well. Six style errors were sitting in it. Two failures, one root cause.

### First smoke result (synthetic data, so this is a pipeline check, not a finding)

```
cross_session split verified clean: {'train': 240, 'val': 160, 'test': 80}
band-power LR : n=80 bal_acc=0.875 macro_f1=0.865 ece=0.295 nll=0.640
permuted label: n=80 bal_acc=0.113 macro_f1=0.111 ece=0.124 nll=1.693
```

Chance is 0.200. The classifier recovers the planted signal and the permutation control sits
below chance, so both gates behave. Worth noting even here: the model's ECE is 0.295, meaning
it is already substantially overconfident on data it handles well. That is the phenomenon the
real experiments are built to measure, showing up in a toy.

### Stage 0 closed

Repository public at https://github.com/happyc0der/lfp2vec-audit, CI green on commit `4bad59c`
(https://github.com/happyc0der/lfp2vec-audit/actions/runs/35172908689). 93 tests, lint and the
end-to-end smoke run all pass on a clean ubuntu checkout as well as locally on MPS and CPU.

### Open questions carried into Stage 1

- Does direct NWB reading from the Allen S3 bucket work, or is AllenSDK unavoidable (D4)?
- How severe is the CA2 class imbalance on real probes? If CA2 is a handful of channels,
  macro-averaged metrics will be dominated by noise in that class and per-class results must be
  reported alongside.
- Does the IBL streaming interface make the 500 s window cheap enough to fetch all seven
  insertions, or is a whole-file download forced?
- What throughput does wav2vec2-base reach on MPS at 48 000 samples per item? This determines
  the Stage 3 subsample size, and must be measured before any long run is launched.

---

## 2026-09-17 (later) — Stage 1: the data layer

Goal: turn the two public datasets into chunk stores, with enough recorded about them that a
reader can tell what they contain and what was thrown away.

### Decisions made from reading, before writing code

Three research passes over the upstream repository, the IBL access stack and the Allen file
format settled the design. The findings that changed it:

- **The upstream window and the Allen chunk spec agree exactly.** IBL reads seconds 0 to 500;
  Allen takes 100 chunks of 3 seconds starting at 200 s, which ends at 500 s. One spec therefore
  covers both. Recorded as an inference, since the IBL chunk parameters are command-line
  arguments whose values appear nowhere in the repository.
- **`Streamer`, the documented way to read part of an IBL file, lives inside the full `ibllib`
  package**, which pulls in PyQt5, OpenCV, numba, phylib and more. Not worth it for one class:
  the same result comes from a byte-range fetch plus a truncated header, using only `ONE-api`,
  `ibl-neuropixel` and `iblatlas`.
- **Two Allen probe files in session 719161530 are entirely zeros** while reporting that they
  have LFP data, and they are the only probes in that session carrying CA2 and CA3. Session
  798911424 was added for that coverage; all six of its probes are live.

### What the prefix trick rests on

mtscomp stores fixed one-second chunks in order, with a sidecar listing each chunk's byte
offset. So the first N seconds are literally the first `chunk_offsets[N]` bytes, and a header
truncated to those chunks makes the prefix a complete recording. mtscomp implements exactly this
for local files as `Reader.chop`; reading its source fixed the details worth copying, in
particular that both whole-file SHA-1 digests must be nulled because they describe bytes that
are no longer present. The reader tolerates a null digest with a warning.

Result: 329 MB per insertion instead of 3.2 GB, about a tenth.

### Things that went wrong

- **A prefix request downloaded the whole file.** The first `download` only attached a range
  header when resuming a partial file; on a fresh download it fetched everything and compared
  sizes at the end. Caught by watching the cache directory grow past 1.3 GB during what should
  have been a 3 MB fetch. Every request now carries an explicit range whenever the caller wants
  less than the whole file, and a server that ignores the range is an error rather than a
  silent multi-gigabyte download. There is a regression test.
- **The geometry assertion fired on correct data.** It expected the axial coordinates of a
  Neuropixels 1.0 probe to start at zero; the released tables start at 20 micrometres. The check
  now tests the pattern that actually matters (two channels per row, exactly 20 micrometre
  pitch, ascending, four staggered lateral columns) instead of hard-coded values. Worth keeping
  rather than deleting: if this table were ever reordered, every label would attach to the wrong
  channel, and nothing downstream would notice.
- **Reading Allen one channel at a time was 230 times too slow.** The files are chunked as one
  channel by 37.8 seconds, which suggests reading per channel. Measured over a remote 30-second
  window: 12 s for a single channel, 5 s for all 94 together. Consecutive chunks on disk are
  neighbouring channels of the same time block, so a time slab is nearly contiguous while one
  channel is a scatter of distant reads. Reading slabs.

  ```
  data.chunks = (47193, 1)   # one channel x 37.8 s
  30 s, one channel      : 12.2 s
  30 s, all 94 channels  :  5.0 s
  ```

### Deliberate departures from upstream, all in DEVIATIONS.md

Label matching is stricter (prefix rather than substring, and no channel-id range slice, D7).
Channels the destriper flagged as dead or noisy are dropped rather than kept as interpolated
copies of their neighbours (D10). Empty probe files are detected and recorded rather than
silently contributing nothing (D8).

