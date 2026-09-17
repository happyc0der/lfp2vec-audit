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

**Here:** per-probe LFP files are read straight from the public S3 bucket with `h5py`.

**Why:** AllenSDK pins an old NumPy/pandas stack that conflicts with a current torch install.
Keeping it out of the environment avoids a dependency resolution that would otherwise force
either two environments or downgraded core libraries.

**Status:** confirmed working in Stage 1. The files are ordinary HDF5 and the two things needed
from them, an LFP dataset and an electrode table carrying CCF acronyms, are read directly. One
substantive difference follows: AllenSDK's `get_lfp` masks samples inside intervals the session
marks invalid, which this reader does not do. For the sessions used here the only such interval
is tagged as a stimulus event rather than a probe fault, so AllenSDK would not have masked it
either.

Two further details were established by reading the files rather than the documentation. The
sampling rate recorded in each probe group's own metadata attribute is wrong, reading half the
true value, so the rate is measured from the timestamps instead. And channels in the released
LFP are every fourth electrode, giving roughly 90 per probe at 40 micrometre spacing, against
384 at 20 micrometres for IBL.

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

## D7 — Region label set and how acronyms are matched

Both follow the upstream aggregation: the visual-cortex acronym family collapses to `VIS`, the
four hippocampal subfields stay separate, and everything else is dropped. The paper's text also
mentions LP and PO for IBL; the upstream code targets the five-way set used here. The dataset
card reports how many channels are discarded as `UNK` so the reader can see the cost.

The matching rule differs. Upstream tests `region in label`, a substring search, for IBL and an
exact match plus a channel-id range slice for Allen. Substring matching is loose enough to
mislabel: any acronym merely containing `DG` or `VIS` is swept in. The id-range slice is looser
still, selecting every channel whose id falls between a region's lowest and highest, so channels
belonging to other structures are pulled in wherever a region is anatomically interrupted.

Here both datasets use the same prefix rule: an acronym is assigned to a hippocampal subfield
when it begins with that subfield's name, which captures the sublayer notation (`CA1sp`,
`CA1slm`, `DG-mo`) without capturing unrelated structures, and to `VIS` when stripping any
cortical layer suffix leaves a known visual area. This is stricter than upstream and will
produce slightly different channel sets.

## D8 — Empty Allen probe files

Three probe files in the sessions used here advertise LFP data and contain nothing but zeros:
`729445654` and `729445656` in session 719161530. They are the only probes in that session
carrying CA2 and CA3, which is why a second session, 798911424, is included.

Upstream never confronts this: its chunking loop discards all-zero trials silently, so those
probes simply contributed nothing. Here they are detected at the boundary, by file size and then
verified by sampling the data, and recorded in the dataset card as skipped with the reason. A
store built without that check would be full of well-formed chunks carrying no signal.

## D9 — High-pass corner: 0.5 Hz in the code, 2 Hz in the paper

The paper describes destriping as applying a 2 Hz high-pass. The library function the upstream
code calls defaults to a 0.5 to 300 Hz band-pass, and upstream does not override it. The code is
followed here. The difference affects only the lowest part of the delta band.

## D10 — Channels excluded for signal quality

Upstream keeps every channel after destriping, including ones the destriper itself flagged as
dead, noisy or outside the brain and then reconstructed by interpolating from neighbours. Those
channels carry a spatially smoothed copy of their neighbours' signal, which for a region
classifier is close to duplicated data with a label attached.

Here they are dropped, and the count is reported per probe in the dataset card. For Allen the
equivalent is the `valid_data` column, also dropped.

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

## D11 — The two datasets are filtered differently, and it shows above 200 Hz

Measured after both stores were built, comparing mean power spectra normalised to each store's
own value at 10 Hz:

| frequency | Allen power relative to IBL |
|---:|---:|
| 100 Hz | 1.1x |
| 200 Hz | 3.3x |
| 300 Hz | 33x |
| 500 Hz | 768x |

Below 100 Hz the two agree closely. Above it they diverge by orders of magnitude, because the
IBL destriping applies a 0.5-300 Hz band-pass while the Allen cache is delivered already
downsampled with no comparable corner. Neither is wrong; they are different pipelines, and both
are used here exactly as their upstream code uses them.

The consequence for this audit is direct. A classifier could separate the two labs perfectly by
looking above 300 Hz alone, without learning anything anatomical. The contamination also reaches
into a band that matters: relative ripple-band power (100-250 Hz) is 2.8 times higher in Allen,
and ripple activity is the classic physiological marker distinguishing hippocampal CA1.

This is not corrected in the stored data, because correcting it would hide it. Instead, from
Stage 2 onward every cross-lab result is reported twice: once on the full band, and once with
both datasets restricted to a common band below the IBL corner. The difference between those two
numbers is the part of cross-lab transfer that was never physiological to begin with.
