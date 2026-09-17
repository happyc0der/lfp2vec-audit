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

### Open questions carried into Stage 1

- Does direct NWB reading from the Allen S3 bucket work, or is AllenSDK unavoidable (D4)?
- How severe is the CA2 class imbalance on real probes? If CA2 is a handful of channels,
  macro-averaged metrics will be dominated by noise in that class and per-class results must be
  reported alongside.
- Does the IBL streaming interface make the 500 s window cheap enough to fetch all seven
  insertions, or is a whole-file download forced?
- What throughput does wav2vec2-base reach on MPS at 48 000 samples per item? This determines
  the Stage 3 subsample size, and must be measured before any long run is launched.
