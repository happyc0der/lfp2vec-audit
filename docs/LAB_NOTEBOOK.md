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

### Allen store built, and what it verifies

50 200 chunks from 502 channels across ten live probes in two sessions, 3.0 s at 1249.999 Hz.

The physiology check that matters: the mean spectrum peaks at **7.00 Hz**, measured directly as
the spectral maximum below 20 Hz rather than read off a plot. Mouse theta is 6-10 Hz, so that
single number simultaneously confirms the sampling rate, the unit conversion and the absence of
an off-by-a-factor error anywhere in the chain. A pipeline with the wrong sampling rate or a
misapplied filter would not land there by accident.

Region differences are real and in the right direction, but modest:

| region | chunks | theta share | gamma share | median amplitude |
|---|---:|---:|---:|---:|
| CA1 | 15 100 | 0.448 | 0.116 | 153 uV |
| CA2 | 400 | 0.389 | 0.105 | 141 uV |
| CA3 | 3 300 | 0.354 | 0.144 | 101 uV |
| DG | 8 700 | 0.374 | 0.169 | 136 uV |
| VIS | 22 700 | 0.369 | 0.118 | 69 uV |

Strongest theta in CA1, elevated gamma in dentate gyrus, cortex lowest in amplitude and highest
in delta. All three match textbook descriptions. That the separations are modest rather than
dramatic is worth noting now, before any baseline is fitted: a six-number spectral summary has a
real problem to solve here, not a trivial one.

**Adding session 798911424 was necessary, not precautionary.** It supplies every CA2 channel (4)
and every CA3 channel (33) in the store. Session 719161530 alone, the first one upstream lists,
contains neither, because its only CA2/CA3 probes are the two that hold nothing but zeros.

**CA2 is as thin as feared**: 4 channels, 400 chunks, against 227 channels for visual cortex.
Macro-averaged scores will be dominated by noise in that class, which is why the chance band in
the smoke gate is computed from the rarest class rather than the total.

### Both stores built; first real numbers, and a confound found before any training

| store | chunks | channels | sources | fs |
|---|---:|---:|---:|---|
| ibl | 132 600 | 1 326 | 7 insertions, 5 labs | 1250.012 Hz |
| allen | 50 200 | 502 | 10 probes, 2 sessions | 1249.999 Hz |

**IBL contains no CA2 at all.** Not one channel across seven insertions. Every CA2 chunk in the
corpus comes from Allen, and only four channels there. This surfaced as a crash rather than a
number, which was lucky: a classifier trained on IBL returns four probability columns while the
labels span five classes, so the columns silently misalign unless they are explicitly placed.
They are now, and an unseen class scores zero recall rather than shifting every other class's
score by one position.

**A second correctness bug, found the same way.** The gate compared balanced accuracy against a
fixed chance level of one over five. But balanced accuracy averages recall over the classes
present in the test labels, and insertions pass through different structures: the held-out IBL
insertion contains only CA1, DG and VIS, so its chance level is 0.333. The permutation control
scored exactly 0.333 and was flagged as a leak. Chance and the macro averages are now both
derived from the classes actually present, and reported alongside every result so a reader can
see what a number is being compared against.

#### First real results, interpretable baseline only

Band-power logistic regression, the LFP-LOC feature set, against its own permutation control:

| split | classes | chance | balanced accuracy | permuted | ECE |
|---|---:|---:|---:|---:|---:|
| cross-session, IBL | 3 | 0.333 | 0.628 | 0.333 | 0.111 |
| cross-session, Allen | 3 | 0.333 | 0.625 | 0.333 | 0.066 |
| cross-lab, IBL to Allen | 5 | 0.200 | 0.301 | 0.197 | 0.109 |

Both permutation controls land on chance to three decimal places, which is the strongest
evidence so far that the splits are clean. Within-lab performance is nearly identical across two
independently built datasets, at 1.9 times chance.

Cross-lab degradation is selective rather than uniform, which is the interesting part. CA1 recall
falls from 0.65 to 0.53 and visual cortex holds at 0.76, while CA3 collapses to 0.08 and dentate
gyrus to 0.14. A uniform drop would suggest added noise; this pattern suggests specific features
stop transferring.

#### The confound, found before training anything

Comparing the two stores' mean spectra directly: they agree below 100 Hz and diverge by up to
768-fold above 300 Hz, because IBL destriping band-passes at 0.5-300 Hz and the Allen cache does
not. Recorded as D11. Relative ripple-band power is 2.8 times higher in Allen, so the
contamination reaches the band carrying the clearest hippocampal marker.

This matters more than a preprocessing note. Any cross-lab result in this repository is now
suspect until it is shown not to rest on that difference, which is why D11 commits every
cross-lab number from Stage 2 onward to being reported on both the full band and a common band
below the IBL corner.

---

## 2026-09-17 (later still) — Stage 2: the baseline floor, and what it says about the task

Leave-one-session-out over 17 folds, four class views crossed with five feature sets, every
configuration carrying its own permutation control on the same fold. 1 360 result rows.

### Result 1: electrode position beats the neural signal

Mean balanced accuracy, 4-class view, logistic regression:

| features | within IBL | within Allen | IBL to Allen | Allen to IBL |
|---|---:|---:|---:|---:|
| chance | 0.298 | 0.358 | 0.358 | 0.298 |
| amplitude only | 0.560 | 0.526 | 0.509 | 0.401 |
| **electrode position only** | **0.817** | **0.822** | **0.628** | **0.521** |
| band power, ≤100 Hz | 0.399 | 0.480 | 0.366 | 0.364 |
| band power, full | 0.441 | 0.590 | 0.460 | 0.384 |
| frozen audio model | 0.697 | 0.701 | 0.377 | 0.304 |

Four numbers for depth, lateral offset, relative depth and channel index, with the voltage
discarded entirely, reach 0.82 where six band powers reach 0.44. Probes are lowered along
stereotyped trajectories and structures come in a predictable order along a shank, so a large
part of what "decoding brain region from LFP" measures on these datasets is available without
the LFP. Amplitude alone, which is only the scale that normalisation removed, also beats band
power within lab.

This is the control the original paper does not report, and it is the first thing any number from
Stage 3 will have to be placed against.

### Result 2: cross-lab band power was mostly the filtering artefact

Dropping the ripple band, the one D11 identified as contaminated, costs cross-lab band power
almost everything it had: 0.460 to 0.366 against a chance of 0.358. Within lab the same change
costs much less. Whatever cross-lab transfer band power appeared to have was substantially the
difference between the two labs' filters.

### Result 3: the frozen audio model is the best signal feature, and it does not transfer at all

`facebook/wav2vec2-base` run forward with nothing fine-tuned, mean-pooled, into a linear
classifier, reaches 0.70 within lab. That beats hand-designed band power by a wide margin and it
beats it on every fold of the IBL scheme (Wilcoxon, 7/7, p = 0.016). So the audio prior really
does carry region information that a spectral summary does not, which is a point in the paper's
favour and against the LFP-LOC interpretability argument.

Then it collapses. Cross-lab it scores 0.377 against a chance of 0.358, and 0.304 against 0.298.
Not degraded: gone.

### Result 4: why it collapses, and the calibration failure that comes with it

The lab discriminator, each fold holding out one probe from each dataset:

| features | area under the curve |
|---|---:|
| band power, ≤100 Hz | 0.729 |
| band power, full | 0.837 |
| **frozen audio embeddings** | **1.000** |

Perfect separation, on every fold, on probes the classifier had never seen. The representation
that makes the audio model the best within-lab feature is dominated by which rig the recording
came from. A decision boundary fitted in IBL's region of that space says nothing about where
Allen's chunks fall, which is exactly a collapse to chance.

And the confidence does not collapse with it. Expected calibration error for the frozen model
goes from 0.124 within lab to **0.478 and 0.631** across labs. A model performing at chance while
reporting high confidence is worse than a model that is merely wrong, and this is precisely the
failure the paper's own Broader Impact section anticipated without measuring.

### What this predicts for Stage 3, and what it does not establish

A frozen encoder with a linear head is not LFP2Vec. The published method adds self-supervised
continuation on unlabelled LFP and then fine-tunes, and either stage could plausibly suppress the
acquisition structure that dominates here. Nothing above shows that it does not.

What Stage 2 does is turn the question into a measurable one. The fine-tune has to clear 0.82
from electrode position to claim it is reading physiology, and it has to reduce a lab-identity
area under the curve of 1.000 to claim it transfers. Both are now numbers on the same folds with
the same metrics, so the comparison will be direct.

### Two bugs found

- The first lab discriminator held out a single group, which left a test set containing only one
  label, where accuracy and area under the curve are both undefined in any useful sense. Each
  fold now holds out one probe per dataset. When I tried to satisfy the `Split` structure with a
  stub validation set that overlapped test, the existing leakage verifier refused it, which is
  the second time that check has earned itself.
- The neural baseline scored 0.74 on separable synthetic data where it should have scored 1.00,
  because `early_stopping` holds out a tenth of the training set and a tenth of 400 rows is too
  noisy a stopping signal. Measured the threshold rather than guessing it: 0.74 at 400 rows, 1.00
  at 1600. Early stopping is now disabled below 2000 rows, since real folds do get that small once
  a class view is applied, and an optimisation failure would be indistinguishable in the results
  table from an absence of signal.

---

## 2026-09-17 (evening) — Stage 3: the fine-tune, first direction

Fine-tuned `facebook/wav2vec2-base` on IBL, 23 900 class-balanced chunks, and scored all ten
Allen probes. Nine epochs, stopped early, 16.5 minutes each on MPS at 25.5 chunks per second.
The gate measured that rate before the run started and passed it against a three-hour budget.

### The numbers

| | value | reference |
|---|---:|---|
| within-lab validation, best epoch | **0.801** | electrode position alone reaches 0.817 |
| cross-lab test, mean over 10 probes | **0.340** | chance on those probes is 0.358 |
| expected calibration error, cross-lab | **0.556** | the frozen checkpoint scored 0.478 |
| negative log-likelihood, cross-lab | **4.76** | a uniform predictor scores 1.39 |
| lab identity after fine-tuning | **0.997** | before fine-tuning it was 1.000 |

Not one of the ten target probes exceeded its own chance level.

### What the model actually does on the other lab

Presented with Allen recordings it predicts visual cortex for **93.9%** of chunks, at a mean
softmax confidence of **0.98**. The true share of visual cortex in that test set is 45%.

That is not degradation. It is a classifier whose decision boundary was fitted in one region of
representation space being handed inputs that all fall in another, and answering with whichever
label that region happens to carry. The behaviour follows directly from the probe result.

### The measurement that explains it

Nine epochs of supervised training on region labels moved the lab-identity score from 1.000 to
0.997. Essentially nothing. Whatever in the representation encodes which rig produced a
recording survives fine-tuning intact, and as long as it does, a boundary learned in one lab
cannot mean anything in the other.

This was the question Stage 2 raised and could not answer, and it now has a number.

### What this is and is not

It is a result for the reduced method: audio initialisation plus supervised fine-tuning, without
the self-supervised continuation stage the published work runs on unlabelled LFP (D1). That
stage is the most plausible candidate for removing acquisition structure, precisely because it
trains on pooled unlabelled data rather than on one lab's labels, and nothing here tests it.

It is one direction, one seed. The reverse direction is running.

What it does establish, on public data with every control on the same folds:

- The fine-tune reaches 0.801 within lab, which is level with what four numbers describing
  electrode position achieve, and does not exceed it.
- Cross-lab it is at chance while reporting 0.98 confidence, which is worse than being wrong.
- The reason is measurable rather than speculative, and it is not fixed by fine-tuning.

### Method note

The trainer's early stopping watches a held-out *group*, not a random slice, so the stopping
decision cannot leak across sessions. Validation balanced accuracy was not monotonic: 0.621,
0.694, 0.671, 0.772, 0.800, 0.760, 0.801, 0.801, 0.740. A patience of one would have stopped at
epoch 2 and reported 0.694 instead of 0.801, understating the model by a tenth for no reason
other than noise.

### Both directions, and they agree

Repeating the experiment with the datasets exchanged, training on Allen and scoring all seven IBL
insertions:

| | IBL to Allen | Allen to IBL |
|---|---:|---:|
| within-lab validation | 0.801 | 0.757 |
| cross-lab test | 0.340 | 0.302 |
| chance on those groups | 0.358 | 0.298 |
| calibration error | 0.535 | 0.655 |
| negative log-likelihood | 4.76 | 4.67 |
| lab identity after fine-tuning | 0.997 | 0.994 |

The picture is symmetric. Both directions train to a usable within-lab model, both land on their
own chance level when moved to the other dataset, both do so while badly miscalibrated, and in
both the representation still identifies the source lab almost perfectly after training.

Placed against Stage 2 on the same schemes, the fine-tune is the **worst** cross-lab method
tested in both directions, below band power, below amplitude, below electrode position, and below
the same checkpoint with nothing trained at all. It is also the most confident.

That combination is the finding. A model can be simultaneously the best within-lab option
available and the worst possible one across labs, and nothing in a within-lab evaluation reveals
it. The only measurement that anticipates the failure is the one nobody runs: asking what else
the representation knows.

### The leakage control, and what it accidentally shows

Two epochs of the same fine-tune with the training labels shuffled:

| | permuted labels | real labels |
|---|---:|---:|
| training loss | 1.376 | 0.165 |
| validation balanced accuracy | 0.250 | 0.801 |
| cross-lab balanced accuracy | 0.250 | 0.340 |
| **calibration error** | **0.118** | **0.535** |
| negative log-likelihood | 1.333 | 4.76 |
| lab identity | 1.000 | 0.997 |

The control behaves exactly as it should. Training loss sits at 1.376 against the 1.386 that is
the entropy of four equally likely classes, so nothing is being learned, and both validation and
test land precisely on chance. The splits are sound.

Two things fall out of it that are worth more than the control itself.

The permuted model is **four and a half times better calibrated than the real one**. It knows
nothing and reports that it knows nothing, so its confidence matches its accuracy almost exactly.
The trained model is equally wrong across labs and reports 0.98 confidence. The problem was never
the accuracy, which is the same in both; it is that only one of them is honest about it.

And lab identity stays at 1.000 under permuted labels. The acquisition structure is not something
the region task induced, and not something a different objective would have avoided. It is in the
representation before any supervision, and supervision does not disturb it.

### A caveat on the within-lab number, and what fixes it

The 0.801 above is the **validation** score on a held-out IBL group, and that group was used for
early stopping. It is a within-lab number, but an optimistic one, and it is not directly
comparable with the 0.817 that electrode position achieves on clean held-out test folds.

The honest comparison needs the within-lab leave-one-session-out scheme, where the scored group
is never touched by the stopping decision. That is the next tier, and until it runs the correct
statement is that the fine-tune reaches roughly 0.80 on a group it was allowed to stop on, which
is at best level with position and may be below it.

## 2026-09-18 — a day of compute lost to a self-matching process check

The two remaining within-lab folds were queued to run back to back with this guard:

```
while pgrep -f "lfpaudit finetune run" >/dev/null; do sleep 120; done
```

The intent was to wait for the previous run to finish. What it actually did was match the queuing
shell's own command line, which contains that string verbatim, so the loop waited on itself and
never exited. The job sat spinning for about twenty-five hours and launched nothing. Both the
chain log and the fold logs stayed empty, which is exactly what a healthy chain waiting its turn
also looks like, so nothing about it drew attention.

Two lessons, both about the same thing.

**A process check that can match itself is not a check.** Replaced by running the folds serially
in a single script, which needs no polling at all: the second cannot start before the first
returns because it is the next line.

**Silence is not evidence of progress.** The earlier version of this mistake, in Stage 1, was a
`.gitignore` rule that hid a source package while every local test passed. The version before
that, in Stage 3, was a figure call that silently matched nothing and let the script report
success while producing one fewer figure. Each time the failure mode is the same: a step that
does nothing and says nothing is indistinguishable from a step that worked. The remedy that keeps
working is to check the thing itself rather than a proxy for it, which here means confirming a
process is burning processor time, not that a log file is quiet.

Nothing scientific was lost. Tier 1 was already complete and committed, and the first within-lab
fold had finished. The cost was a day of an idle laptop.


