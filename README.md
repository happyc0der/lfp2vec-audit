# lfp2vec-audit

[![ci](https://github.com/happyc0der/lfp2vec-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/happyc0der/lfp2vec-audit/actions/workflows/ci.yml)

**Are LFP2Vec-style anatomical predictions calibrated and interpretable under cross-lab shift, and how much of their accuracy is recoverable by trivial, interpretable baselines?**

> **Status: Stage 1 — data layer.** Both public datasets are read into a common chunk format, with group-aware splits and a dataset card. No model has been trained yet. Every claim below that is not yet measured is marked *planned*.

## What LFP2Vec is

[LFP2Vec](https://papers.nips.cc/paper_files/paper/2025/hash/b408053531ce6fd66a96bc3a86527bb9-Abstract-Conference.html) (He, Patel, Li, Maslarova, Vöröslakos, Ramanathan, Hung, Buzsáki & Varol, NeurIPS 2025) adapts the audio model `facebook/wav2vec2-base` to raw local field potential, continues self-supervised training on unlabelled LFP, and fine-tunes it to predict which brain region an electrode sits in from a 3-second single-channel recording. The paper reports zero-shot transfer across labs and probe geometries. Upstream code: [`tianxiao18/lfp2vec`](https://github.com/tianxiao18/lfp2vec).

## Why this repository exists

The LFP2Vec paper's own Broader Impact section states that clinical deployment "should include calibrated uncertainty estimates" — but no calibration metric is reported anywhere in the paper or supplement. Separately, [LFP-LOC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13199280/) (Perna et al., *Frontiers in Neuroscience*, 2026) criticises LFP2Vec as lacking interpretability and requiring training, and proposes training-free band-power features instead — but offers no head-to-head comparison on shared data.

This repository measures both gaps on public data:

1. **Calibration under shift.** Does confidence stay meaningful when the test recording comes from a different lab?
2. **The interpretable-baseline floor.** How much of the accuracy is recoverable from six numbers per chunk (δ θ α β γ ripple relative power), from electrode depth alone, or from a frozen audio encoder with no LFP training at all?
3. **What the model listens to.** Which frequency bands actually drive predictions, and is that answer stable across seeds and sessions?

## What this repository does *not* claim

- It is **not** a full reproduction. The self-supervised continuation stage is out of scope on laptop-class compute; experiments start from the audio checkpoint and fine-tune. See [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).
- It does **not** evaluate on the Neuronexus mouse or macaque datasets, which are private to the original authors.
- A difference measured here is a difference **against this reproduction**, not a refutation of the published model. No pretrained LFP2Vec weights have been released, so exact published numbers cannot be checked.
- It does not claim LFP-based anatomical localisation is solved, unsolved, or anything in between.

## Planned experiments

| Stage | Experiment | Status |
|---|---|---|
| 1 | IBL and Allen data layer, group-aware splits, leakage verifier | **done** |
| 2 | Baselines: constant, depth-only, band-power, frozen wav2vec2 | planned |
| 3 | LFP2Vec-lite fine-tune, cross-session and cross-lab | planned |
| 4 | Temperature scaling, band-stop and phase-randomisation ablations, band attribution | planned |
| 5 | Selective prediction: risk–coverage under shift | planned |
| 6 | Two-page note and figures | planned |

## Reproducing what exists today

```bash
make setup    # uv venv + editable install
make test     # full unit suite, synthetic data only, no network
make smoke    # end-to-end pipeline run with planted ground truth
```

`make smoke` generates a synthetic dataset with known spectral signatures per region, builds a group-aware split, verifies it for leakage, fits a band-power classifier, and asserts two gates: the classifier beats chance, and the same classifier trained on permuted labels does not. Every real experiment must pass this first.

Building the real datasets, which does touch the network:

```bash
make data-ibl      # seven insertions, ~2.3 GB fetched
make data-allen    # two sessions, read over HTTP ranges
make splits        # write and verify every split
make real-smoke    # the same gates, on real data
```

## Data

| Source | Access | Labels | Fetched |
|---|---|---|---|
| [IBL](https://docs.internationalbrainlab.org/) Neuropixels | byte prefix of the compressed LF band, via ONE-api | CCF acronym per channel, from histological alignment | ~330 MB per insertion, of 2.4–3.8 GB files |
| [Allen Visual Coding](https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html) Neuropixels | HTTP range reads of the public NWB files | `location` column of the electrode table | ~140 MB per probe, of 1.3–2.5 GB files |

Both are cut to the same 3-second, 1250 Hz chunks and stored as one normalised float16 array plus a parquet index. The mean and scale removed by normalisation are recorded per chunk, so the original microvolt waveform is recoverable.

Neither dataset is fetched whole. The IBL recordings are compressed in one-second chunks stored in order, so the 500-second window the analysis uses is a byte prefix of the file; a truncated header describing only those chunks makes that prefix a valid standalone recording. The Allen files are read with range requests, one time slab across all channels at a time, which matches how their chunks are laid out on disk.

Raw data is never committed. Only run manifests, metrics, dataset cards and inspection figures live in the repository.

## Repository layout

```
lfpaudit/data/      fetch, label, chunk, index, split
lfpaudit/features/  band power, geometry
lfpaudit/models/    baselines and the wav2vec2 reproduction
lfpaudit/eval/      metrics, calibration, ablations, attribution, selective prediction
experiments/        one YAML per run
results/            manifests and metrics, committed
docs/               lab notebook, deviations, development notes
```

## Development notes

Much of this code was written with Claude Code. See [`docs/DEVELOPMENT_NOTES.md`](docs/DEVELOPMENT_NOTES.md) for what that means in practice and which parts were verified by hand. The running log of commands, decisions and dead ends is in [`docs/LAB_NOTEBOOK.md`](docs/LAB_NOTEBOOK.md).

## References

1. He, T., Patel, S., Li, S., Maslarova, A., Vöröslakos, M., Ramanathan, D., Hung, C., Buzsáki, G., & Varol, E. (2025). Self-supervised learning for in vivo localization of microelectrode arrays using raw local field potential. *NeurIPS 2025*.
2. Perna, G., Adamo, S., Vincenzi, M., Angotzi, G. N., Ribeiro, J. F., & Berdondini, L. (2026). LFP-LOC: an LFP power-based method for validating electrode localization. *Frontiers in Neuroscience*.
3. Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: a framework for self-supervised learning of speech representations. *NeurIPS 2020*.
4. International Brain Laboratory (2023). A brain-wide map of neural activity during complex behaviour. *bioRxiv*.
5. Siegle, J. H. et al. (2021). Survey of spiking in the mouse visual system reveals functional hierarchy. *Nature*.

## Licence

MIT. See [`LICENSE`](LICENSE).
