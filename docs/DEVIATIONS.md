# Deviations from the paper and the upstream repository

Every place this work differs from LFP2Vec (He et al., NeurIPS 2025) or from
[`tianxiao18/lfp2vec`](https://github.com/tianxiao18/lfp2vec), and why. A reader comparing
numbers should read this file first.

## D1 — The self-supervised stage is skipped

**Paper:** wav2vec2-base initialised from `facebook/wav2vec2-base`, then 50 epochs of
self-supervised continuation on unlabelled LFP, then supervised fine-tuning.

**Here:** the audio checkpoint is fine-tuned directly, with no LFP self-supervised stage.

**Why:** the available compute is an M4 Pro laptop and Colab Pro. The upstream cluster script
requests a single A100 for 24 hours. The paper's own ablation reports that audio initialisation
with ~6k LFP trials matches random initialisation with over 400k, which is evidence that the
audio prior carries a large part of the benefit and that the continuation stage is a refinement
rather than the whole method.

**Consequence:** absolute accuracies here are expected to sit below the published ones. Every
comparison in this repository is between models trained under the *same* budget, so the
relative claims (baseline floor, calibration under shift, band attribution) remain meaningful;
the absolute ones do not transfer to the published model.

## D2 — Reimplemented rather than vendored

**Upstream:** the repository's README states "MIT License" but the tree contains no `LICENSE`
file, and GitHub reports no licence for it.

**Here:** the pipeline is reimplemented from the paper and from reading the upstream code,
which is cited throughout. No upstream source is copied.

**Consequence:** implementation details not stated in the paper may differ. Where a choice was
inferred from the upstream code it is noted in the relevant module's docstring.

## D3 — No published weights exist to check against

The paper's checklist says weights will be released; as of 2026-09-17 there are no weight files
in the repository, no GitHub releases, and no matching model on the HuggingFace Hub. Exact
published numbers therefore cannot be reproduced, only re-derived under a stated budget.

## D4 — Allen data read directly from NWB, not through AllenSDK

**Upstream:** `data/Allen/data_download.py` uses `EcephysProjectCache.from_warehouse`.

**Here:** the plan is to read per-probe LFP NWB files from the public S3 bucket with `h5py`.

**Why:** AllenSDK pins an old NumPy/pandas stack that conflicts with a current torch install.
Keeping it out of the environment avoids a dependency resolution that would otherwise force
either two environments or downgraded core libraries.

**Status:** to be confirmed in Stage 1. If direct NWB reading proves unreliable, the fallback is
an isolated `uv run --with allensdk` preprocessing step, and this entry will be updated.

## D5 — Private datasets are out of scope

The paper uses four datasets; two (Neuronexus SiNAPS mouse recordings from the Buzsáki lab, and
macaque Neuropixels-NHP recordings) are private. Only IBL and Allen are evaluated here. The
paper's weakest transfer result involves Neuronexus, so this repository cannot speak to it.

## D6 — Chunks are cached at 1250 Hz, upsampled at load time

**Upstream:** `scipy.signal.resample` from 1250 Hz to 16 kHz during preprocessing, so a 3-second
trial becomes 48 000 samples on disk.

**Here:** chunks are cached as float16 at the native 1250 Hz (3750 samples) and upsampled to
16 kHz inside the dataset. This is numerically equivalent for the model input and roughly
twentyfold smaller on disk, which is what makes a laptop-plus-Colab workflow practical.

## D7 — Region label set

Both follow the upstream aggregation: the visual-cortex acronym family collapses to `VIS`, the
four hippocampal subfields stay separate, and everything else is dropped. The paper's text also
mentions LP and PO for IBL; the upstream code targets the five-way set used here. The dataset
card reports how many channels are discarded as `UNK` so the reader can see the cost.

## Known upstream issues observed while reading the code

Recorded for accuracy, not as criticism of the authors' results. These are observations about
the public snapshot (5 commits, last updated May 2025), which may not be the code that produced
the paper.

- In `script/wav2vec_random_init.py` the self-supervised checkpoint is saved under
  `if max_probe_acc > probe_val_acc:` with `max_probe_acc` initialised to 0, so on a fresh run
  the save never fires, while the following line loads that checkpoint unconditionally. The
  comparison appears inverted.
- `blind_localization.yml` and `requirements.txt` pin conflicting `spikeinterface` versions
  (0.98.2 and ~0.101.2).
- The README references `environment.yml` and `script/dataset_preprocessing/`, neither of which
  exists in the tree.
- `wandb.init` is called unconditionally, so the entry point requires a Weights & Biases login.
