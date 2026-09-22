# LFP2Vec on public data: a reproduction, a preprocessing fix, and two missing measurements

[![ci](https://github.com/happyc0der/lfp2vec-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/happyc0der/lfp2vec-audit/actions/workflows/ci.yml)

[LFP2Vec](https://papers.nips.cc/paper_files/paper/2025/hash/b408053531ce6fd66a96bc3a86527bb9-Abstract-Conference.html)
(He, Patel, Li, Maslarova, Vöröslakos, Ramanathan, Hung, Buzsáki & Varol, NeurIPS 2025) adapts the
audio model `wav2vec2` to raw local field potential and fine-tunes it to say which brain region an
electrode sits in, from three seconds of one channel. This repository reproduces its fine-tuning
stage on the two public datasets the paper uses, IBL and Allen Neuropixels, and reports three
things:

1. **Within a lab it reproduces.** 0.74 ± 0.07 balanced accuracy over all seven held-out IBL
   sessions against the paper's 0.68, from a laptop, without the self-supervised stage.
2. **Across labs the reproduction fails, and the cause is preprocessing, not representation.** The
   two datasets are filtered differently above 100 Hz. Low-passing every input at 100 Hz takes
   cross-lab transfer from chance to well above it in both directions, using no labels from the
   target lab. Whether it reaches the published margin depends on the seed: see result 3.
3. **Two measurements the paper does not make change how its results read**: calibration under
   lab shift, which its Broader Impact section calls for, and an electrode-position control.

![cross-lab margins](docs/figures/fixes.png)

*Cross-lab transfer on the paper's own metric: raw accuracy minus the target lab's majority-class
rate. Light bars before the paper's post-processing, solid after; the dashed line is the published
margin, read from the paper's Figure 2e. Bars are means over three seeds, dots are individual
seeds.*

*Corrected 21 September 2026: an earlier version's position baseline leaked test labels. See
[Corrections](#corrections).*

## Results

Every number is balanced accuracy on sessions or probes the model never saw, four-class view
(CA1, CA3, DG, visual cortex), unless stated. Chance differs by column because balanced accuracy
averages over the classes a test set actually contains. Every configuration has a permuted-label
control on the same fold, and all sit at chance.

### 1. Within a lab

| | within IBL | within Allen |
|---|---:|---:|
| *chance* | *0.30* | *0.36* |
| band power, six bands | 0.44 | 0.59 |
| electrode position | 0.77 | 0.49 |
| audio checkpoint, nothing trained, linear head | 0.70 | 0.70 |
| **fine-tuned (this reproduction)** | **0.74 ± 0.07** (7 sessions) | not run |
| fine-tuned + the paper's post-processing | 0.81 | |
| LFP2Vec as published *(read from Figure 2a)* | 0.68 | 0.65 |

- **The reduced method lands where the paper does**, on every one of the seven sessions (range
  0.63–0.83), and the paper's post-processing adds 0.07 where the paper reports about 0.05.
- **Fine-tuning adds 0.04 over the untrained audio checkpoint** on the same seven sessions
  (5 of 7 folds, signed-rank p = 0.22). Replacing
  every input with a surrogate that keeps its power spectrum and destroys its waveform costs the
  fine-tuned model 0.008. On this task the model reads spectral power and little else, which is
  consistent with the paper's own ablation, where audio initialisation carries most of the benefit.
- **The signal clearly beats six-band power** (+0.30 on IBL, all seven folds, p = 0.016), so it is not reducible to the
  hand-designed summary that [LFP-LOC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13199280/) proposes
  in its place.
- **Electrode position is a serious competitor only where insertions are stereotyped.** On IBL,
  where all seven insertions target the same structures, depth along the shank alone reaches 0.77
  and ties the fine-tune over the seven sessions (0.77 against 0.74, 3 of 7 folds to the
  fine-tune, p = 0.58). On Allen it reaches 0.49 and the
  signal is far ahead. See result 5.

Details: [`docs/RESULTS_BASELINES.md`](docs/RESULTS_BASELINES.md),
[`docs/RESULTS_FINETUNE.md`](docs/RESULTS_FINETUNE.md).

### 2. Across labs, full band

Train on one lab, test on every probe of the other:

| | IBL → Allen | Allen → IBL |
|---|---:|---:|
| balanced accuracy (chance 0.25), mean of 3 seeds | 0.24 ± 0.01 | 0.27 ± 0.04 |
| raw accuracy minus majority rate, 3 seeds | −0.01 ± 0.01 | −0.04 ± 0.04 |
| LFP2Vec as published *(Figure 2e)* | +0.11 | +0.12 |
| expected calibration error, in lab → across labs | 0.10 → 0.56 | 0.11 → 0.66 |
| source lab recoverable from the embeddings (AUC) | 0.997 | 0.994 |

The model predicts visual cortex for 94% of the other lab's chunks at a mean confidence of 0.98.
A linear probe recovers which lab a chunk came from almost perfectly from the fine-tuned
embeddings, on probes it never saw, exactly as it does from the untrained checkpoint's (1.000):
nine epochs of supervised training do not touch the acquisition signature.

The signature has a measurable source. The two datasets' mean spectra agree below 100 Hz and
diverge by up to 768-fold above 300 Hz, because the IBL pipeline band-passes at 0.5–300 Hz and
the Allen release does not ([`docs/DEVIATIONS.md`](docs/DEVIATIONS.md), D11). This was recorded
while building the datasets, before any model was trained.

### 3. Matching the filters restores transfer

Low-pass every input at 100 Hz, a corner chosen from the spectra before any cross-lab result
existed, and fine-tune again:

| | IBL → Allen | Allen → IBL |
|---|---:|---:|
| balanced accuracy, full band → harmonised | 0.24 → **0.46** | 0.26 → **0.37** |
| with the paper's post-processing | 0.56 | 0.36 |
| raw accuracy minus majority, harmonised + post-processing | **+0.15** | **+0.12** |
| LFP2Vec as published *(Figure 2e)* | +0.11 | +0.12 |
| expected calibration error across labs | 0.56 → 0.35 | 0.66 → 0.26 |
| source lab recoverable from the embeddings | 0.997 → 0.83 | 0.994 → 0.74 |
| band power + the paper's post-processing (margin) | +0.17 | +0.03 |
| electrode position (margin) | −0.24 | 0.00 |

**Seed spread (three seeds per direction, 21 September 2026).** The table above is seed 0. Over
three seeds the harmonised model's post-processed margin is **+0.13 ± 0.03 for Allen → IBL**
(seeds +0.12, +0.11, +0.17), level with the published +0.12, and **+0.08 ± 0.07 for
IBL → Allen** (seeds +0.15, +0.02, +0.08), below the published +0.11. Balanced accuracy is
0.44 ± 0.10 and 0.37 ± 0.03 against 0.25 for the full-band model, whose collapse repeats on a
every seed (margin −0.01 ± 0.01 and −0.04 ± 0.04, lab identity 0.998 and 0.997). So harmonising reliably restores transfer,
matches the published margin in one direction, and falls short of it on average in the other.
[`docs/RESULTS_FIXES.md`](docs/RESULTS_FIXES.md) has the current means and spreads.

**No training at all does as well.** The untrained audio checkpoint with a linear head, on the
same harmonised inputs, reaches margins of +0.09 and +0.13 with the paper's post-processing
(balanced 0.48 and 0.36; chance 0.25; permuted-label controls at chance). That equals the
fine-tuned mean across labs with no backpropagation through the encoder: one eight-minute forward
pass against two and a half hours per fine-tune, and no seed to be lucky with. It is scored on a
fixed 25-chunks-per-channel subsample. Within a lab the same model gives up accuracy (0.62 on IBL
against 0.70 full band), so the high band is useful inside a lab and harmful between labs. A
100-bin power spectrum below 100 Hz with the same linear head does not transfer (0.33 and 0.36
balanced), so what carries across labs is in the audio features, not in coarse spectral power.

On seed 0, one preprocessing choice moves the reproduction from below the majority rate to the
published margin in both directions, halves the calibration error, and removes much of the lab signature.
On balanced accuracy the seed-0 harmonised model is the best method tested in both directions. On the
paper's raw-margin metric, six-band power with the paper's smoothing is slightly ahead in one
direction (+0.17 against +0.15) and far behind in the other.

Four alternatives were tested and ruled out, each with its control:

- **The paper's post-processing alone** moves the collapsed full-band model by 0.004. Averaging
  chunks that all say one class has nothing to change.
- **The self-supervised stage** is run per dataset in the released code and never sees two labs,
  so it cannot be what aligns them; the paper's Figure 5 shows it adding nothing on IBL within lab.
- **Subtracting each probe's mean embedding** removes the lab signature (0.997 → 0.53) and leaves
  transfer at chance. The labs are not the same structure displaced.
- **Whitening each probe's whole spectrum**, the thorough form of the same idea, is worse than
  the low-pass in both directions. The spectral shape a probe shares is part of what says which
  structure it is in.

Details: [`docs/RESULTS_FIXES.md`](docs/RESULTS_FIXES.md).

### 4. Calibration under shift

The paper's Broader Impact section says clinical use "should include calibrated uncertainty
estimates". The standard remedy is a temperature fitted on held-out data from the training lab.

![calibration](docs/figures/calibration.png)

| | fitted in lab | ECE in lab | ECE across labs, same temperature | temperature the target needed |
|---|---:|---:|---:|---:|
| full band, IBL → Allen | 1.84 | 0.10 → 0.03 | 0.56 → 0.51 | 8.26 |
| full band, Allen → IBL | 1.72 | 0.11 → 0.05 | 0.66 → 0.58 | 11.79 |
| harmonised, IBL → Allen | 1.73 | 0.14 → 0.03 | 0.35 → 0.24 | 3.66 |

In lab the remedy works. Across labs the same scalar barely moves the full-band model, and the
temperature the target lab needed is four to seven times larger, which cannot be found without
the labelled target data that zero-shot transfer is meant to avoid. Harmonising the filters
halves that gap but does not close it.

Abstention follows the same pattern. With the full-band model no uncertainty score ranks its own
errors, and the model is *more* confident on the unseen lab than on held-out data from its own.
With the harmonised model, distance from the training distribution in embedding space ranks
errors in both directions: keeping the fifth of predictions it trusts most gives 0.31 and 0.36
error against 0.51 and 0.52 overall.

Details: [`docs/RESULTS_CALIBRATION.md`](docs/RESULTS_CALIBRATION.md).

### 5. What the signal adds to knowing where the electrode is

Position is only useful if it is known. Here it is absolute depth along the shank plus lateral
offset, both given by the probe's geometry.

![depth uncertainty](docs/figures/depth_uncertainty.png)

- **Signal and position are complementary.** Fused as a product of experts, with nothing fitted
  on top, the pair reaches 0.78 on IBL and 0.72 on Allen, at or above both parts. Where insertions
  are stereotyped position leads and the signal adds a little; where they are not, the reverse.
- **Depth uncertainty decides which to trust.** Shift the held-out probe along its shank by a
  random offset, with the position model trained under the same uncertainty. On IBL position falls
  from 0.75 to 0.63 at ±400 µm and 0.47 at ±800 µm while the signal stays at 0.68: the signal is
  the better single source beyond roughly ±280 µm, and the fusion is above both at every level.
- **Across labs position carries nothing** (0.23 and 0.30, at or below chance). Trajectories
  differ between labs, which is exactly the setting signal-based localisation is for.
- **A hypothesis that failed:** the signal does not rescue position near anatomical boundaries.
  Every source degrades together there, where histological labels are also least certain.

Details: [`docs/RESULTS_POSITION.md`](docs/RESULTS_POSITION.md).

## Scope and limits

- **This is the reduced method**: audio initialisation plus supervised fine-tuning, without the
  self-supervised continuation. No pretrained LFP2Vec weights have been released, so published
  numbers are read from the paper's figures and carry about ±0.01; any error in reading them is
  this repository's. A gap measured here is a gap against this reproduction.
- **Two of the paper's four datasets are private** (Neuronexus, macaque) and untested.
- **Statistics are thin where training is expensive.** Baselines use all 17 leave-one-session-out
  folds. Fine-tunes have all seven IBL sessions within lab (one seed each) and three seeds per
  cross-lab configuration; within-Allen fine-tunes were not run. Every table and figure
  regenerates from saved results.
- **Labels are histological estimates** taken as given, for this work as for the paper. Every
  score is conditional on a channel being labelled with one of the target regions.
- **Conventions matter when comparing to the paper.** Its chance level is the majority-class rate,
  its within-session figure is balanced accuracy, and its cross-lab matrix is raw accuracy. Rows
  that cite the paper use its convention ([`docs/DEVIATIONS.md`](docs/DEVIATIONS.md), D13).

[`docs/DEVIATIONS.md`](docs/DEVIATIONS.md) lists every place this pipeline differs from the paper
or its released code, and why.

## Corrections

**21 September 2026.** Until this date the electrode-position baseline included depth rescaled to
the span of channels kept on each probe. Kept channels are chosen by histology label, so that
feature encoded the test probe's own anatomy: alone it scores 0.81 and 0.76 across labs, where
honest position scores 0.23 and 0.30. Earlier versions of this README therefore claimed, wrongly,
that position beat the signal-based models across labs, and overstated it within lab. The feature
is fixed, a unit test pins the property that distinguishes the two, and every table and figure
was regenerated ([`docs/DEVIATIONS.md`](docs/DEVIATIONS.md), D15). The permutation control for
one- and two-number features was also changed from one draw to the mean of twenty, because a model
with so few possible decision functions can match the anatomy on a single unlucky relabelling.

## Reproducing

```bash
git clone https://github.com/happyc0der/lfp2vec-audit && cd lfp2vec-audit
make setup        # uv environment, Python 3.11
make test         # 300+ unit tests on synthetic data, no network
make smoke        # end-to-end synthetic run with its permutation gate

make data-ibl data-allen   # about 5 GB of public recordings; builds the chunk stores
make features baselines    # baselines across all 17 folds (CPU, under an hour)
lfpaudit finetune run --scheme cross_lab_ibl_to_allen --lowpass 100 --save-model   # ~2.5 h on an M-series GPU
make figures
```

Every training run passes a synthetic smoke test, a leakage check on its split, and a measured
throughput gate before it starts, and writes a manifest with the git commit, data hashes, seed
and configuration before its first step. The test-set logits of every run are saved, so all
calibration, abstention and post-processing analysis is reproducible without retraining.

## Where things are

| | |
|---|---|
| [`note/note.pdf`](note/note.pdf) | the two-page summary |
| `docs/RESULTS_*.md` | full tables, generated from `results/` by the CLI |
| [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md) | every difference from the paper and its code, and every correction |
| [`docs/LAB_NOTEBOOK.md`](docs/LAB_NOTEBOOK.md) | dated working notes, including what went wrong |
| [`docs/DATA_CARD.md`](docs/DATA_CARD.md) | channels and chunks per probe and region |
| [`docs/DEVELOPMENT_NOTES.md`](docs/DEVELOPMENT_NOTES.md) | how this was built, including the use of Claude Code |
| `lfpaudit/` | data loaders, features, models, evaluation, CLI |
| `results/` | per-fold metrics and manifests for every run |

## References

- He, Patel, Li, Maslarova, Vöröslakos, Ramanathan, Hung, Buzsáki, Varol. *Self supervised learning for in vivo localization of microelectrode arrays using raw local field potential.* NeurIPS 2025. Code: [`tianxiao18/lfp2vec`](https://github.com/tianxiao18/lfp2vec).
- Perna, Adamo, Vincenzi, Angotzi, Ribeiro, Berdondini. *LFP-LOC.* Frontiers in Neuroscience, 2026.
- International Brain Laboratory, Brain-wide map. Allen Institute, Visual Coding Neuropixels.

## Licence

MIT. See [`LICENSE`](LICENSE).
