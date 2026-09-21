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

### Within-lab, paired on the sessions both actually covered

Three leave-one-session-out folds, each scoring an insertion that took no part in training or in
the stopping decision. Compared against the Stage 2 baselines on those same three sessions:

| method | balanced accuracy | difference from fine-tune | fine-tune wins |
|---|---:|---:|---:|
| **electrode position** | **0.856** | −0.140 | 1 of 3 |
| fine-tuned wav2vec2 | 0.716 | — | — |
| frozen audio model | 0.709 | +0.007 | 2 of 3 |
| amplitude only | 0.593 | +0.122 | 3 of 3 |
| band power, full | 0.479 | +0.236 | 3 of 3 |
| band power ≤100 Hz | 0.429 | +0.287 | 3 of 3 |

Per fold, to show how much sessions differ:

| session | chance | geometry | frozen | fine-tuned |
|---|---:|---:|---:|---:|
| 0802ced5_probe00 | 0.250 | 0.627 | 0.578 | 0.631 |
| 0802ced5_probe01 | 0.333 | 0.947 | 0.775 | 0.697 |
| 0a018f12_probe00 | 0.333 | 0.994 | 0.775 | 0.819 |

Two things here, and the second is the one that matters.

**Fine-tuning clearly beats hand-designed features.** It is ahead of band power by 0.24 and of
amplitude by 0.12, winning every fold against both. Whatever the audio prior plus supervision is
doing, it is not reducible to a six-number spectral summary, and the interpretability argument
LFP-LOC makes against this family of model does not get to claim otherwise.

**Fine-tuning barely beats doing nothing.** Against the same checkpoint with no training at all,
the difference is **+0.007**, on two folds of three. Nine epochs on twenty-four thousand chunks,
about two and a half hours per fold, buys effectively nothing over running the untouched audio
model forward into a linear classifier.

And electrode position is ahead of all of it, by 0.140, on two folds of three.

Three folds is too few for a signed-rank test to say anything, so the table reports differences
and win counts rather than a p-value that would be theatre at this sample size.

### Stage 3, assembled

| | within lab (3 folds) | cross lab (both directions) |
|---|---:|---:|
| electrode position | 0.856 | 0.628 / 0.521 |
| fine-tuned wav2vec2 | 0.716 | 0.340 / 0.302 |
| frozen audio model | 0.709 | 0.377 / 0.304 |
| chance | ~0.31 | 0.358 / 0.298 |
| fine-tune calibration error | 0.19 | 0.54 / 0.66 |
| lab identity after fine-tuning | 0.985–0.997 | 0.994–0.997 |

The reduced method is a real model within a lab: better than every hand-designed feature, though
not better than its own starting point by any margin worth two hours, and behind electrode
position throughout. Across labs it is at chance in both directions, the worst of everything
tested, and the most confident of everything tested. Its representation identifies the source lab
at essentially perfect discrimination after training, exactly as it did before.

---

## 2026-09-18 (evening) — Stage 4: the remedy, the refusal, and a reading I had to correct

### The re-run reproduces Stage 3 exactly

Both cross-lab directions were re-run to save what Stage 3 did not: weights, validation logits,
and embeddings for the validation and test sets. Same seed, same folds. Every epoch matched Stage
3 to four decimal places, loss and validation accuracy alike, through early stopping at the same
epoch, to the same test score of 0.242 and the same calibration error of 0.556.

Training on this hardware is therefore deterministic given a seed, which is not guaranteed on
Metal and is worth knowing before anyone reads a difference between two runs as a result.

### Temperature scaling repairs the source lab and does not reach the target

Fit one scalar on the in-lab validation logits, apply it to the cross-lab test logits:

| | calibration error | negative log-likelihood | mean confidence | accuracy |
|---|---:|---:|---:|---:|
| in-lab, before | 0.100 | 0.610 | 0.923 | 0.825 |
| in-lab, after T=1.84 | **0.025** | 0.475 | 0.838 | 0.825 |
| cross-lab, before | 0.560 | 4.786 | 0.980 | 0.420 |
| cross-lab, after T=1.84 | **0.510** | 2.670 | 0.929 | 0.420 |
| cross-lab, oracle T=8.26 | 0.041 | 1.499 | 0.399 | 0.420 |

In-lab the standard remedy works: calibration error falls fourfold and the model stops claiming
more than it delivers. Applied across labs the same scalar moves calibration error by 0.05, from
0.560 to 0.510, and leaves the model asserting 0.93 confidence at 42% accuracy.

The target lab would have needed a temperature of **8.26**, four and a half times what in-lab
fitting produces. With it, calibration error drops to 0.041. So the fix exists and is a single
number, and it cannot be found without labelled data from the target lab, which is exactly what
zero-shot transfer claims not to need.

That is the finding in one line: the remedy the paper's Broader Impact section asks for does not
reach the failure the paper's headline claim would produce.

### Abstention: the output cannot refuse, the representation can

| score | area under risk-coverage | error at full coverage | error at one-fifth | separates in-lab from cross-lab |
|---|---:|---:|---:|---:|
| confidence | 0.618 | 0.580 | 0.721 | 0.315 |
| entropy | 0.625 | 0.580 | 0.732 | 0.327 |
| distance from training data | 0.651 | 0.580 | 0.764 | **0.955** |

Two separate things, and I first ran them together and read them wrongly.

**Ranking.** All three areas sit above the base error rate of 0.580, so discarding the most
uncertain predictions leaves a worse set than keeping everything. Keeping only the most confident
fifth gives 72% error against 58% overall. Abstention by any of these scores makes things worse.

**Detection.** Confidence and entropy score 0.315 and 0.327 at telling in-lab from cross-lab
inputs, below the 0.5 of no information: the model is *more* confident on recordings from a lab
it has never seen than on held-out data from its own. Distance from the training distribution
scores 0.955.

So the representation knows the input is foreign while the output insists it is not. That is the
constructive result available here, and it is narrower than it first looks: the distance score is
a **gate**, not a ranker. It can refuse an entire target recording. It cannot pick which cross-lab
predictions to trust, because at chance accuracy none of them are.

### The reading I had to correct

Seeing every area above the base error rate, I wrote that the most confident predictions were the
most wrong, implying confidence is anti-correlated with correctness. Checking it directly:

```
correlation(confidence, correct) = +0.132
  confidence 0.00-0.50: n=    473  accuracy 0.108
  confidence 0.50-0.90: n=  2 123  accuracy 0.139
  confidence 0.90-0.99: n=  1 071  accuracy 0.470
  confidence 0.99-1.01: n= 46 533  accuracy 0.435
```

The correlation is positive. Coarsely, the model's confidence carries a little real information.
What actually happens is that 93% of cross-lab chunks land in the top confidence bin at once,
because the model predicts visual cortex for 95% of them at mean confidence 0.997, and within
that pile the very highest confidences are slightly worse than the rest. There is no inversion,
only a mass of predictions at one confidence level with no discrimination inside it.

Both the practical conclusions survive: confidence-based abstention does not help, and the model
is more confident out of distribution than in it. But "confident predictions are the most wrong"
overstates a positive correlation into a negative one, and it would have been an easy sentence to
leave in a write-up and be caught on later.

### Both directions, and an asymmetry that one direction would have hidden

The reverse direction agrees on calibration and disagrees on abstention. Running only the
headline direction would have produced a cleaner story and a wrong one.

**Calibration, consistent and if anything worse in reverse:**

| | in-lab T | in-lab ECE | cross-lab ECE with that T | oracle T | gap |
|---|---:|---:|---:|---:|---:|
| IBL to Allen | 1.84 | 0.100 → 0.025 | 0.560 → 0.510 | 8.26 | 4.5× |
| Allen to IBL | 1.72 | 0.114 → 0.054 | 0.656 → 0.580 | 11.79 | 6.8× |

In both directions the in-lab fit works, the same scalar barely moves the cross-lab number, and
the temperature the target actually needed is several times larger than anything the source could
have told you.

**Abstention, asymmetric:**

| score | IBL to Allen | Allen to IBL |
|---|---:|---:|
| confidence | 0.315 | 0.400 |
| entropy | 0.327 | 0.398 |
| distance from training data | **0.955** | **0.495** |

Distance detects the shift almost perfectly in one direction and not at all in the other. Since
a supervised probe separates the two labs at 0.994 in *both*, the structure is certainly there;
what changes is whether an unsupervised distance can reach it.

The geometry says why:

```
train on IBL    val median distance  6.9   test median 16.9   ratio 2.44
train on Allen  val median distance  9.1   test median  8.4   ratio 0.92
```

In both directions the model squeezes foreign recordings into a tight cluster: the test
embeddings have roughly a fifth the spread of the validation ones. Training on IBL, that cluster
lands well outside the reference shell and the detector fires. Training on Allen, the cluster is
displaced by *more* in raw terms, 1.02 reference spreads against 0.84, but along directions the
reference cloud is already wide in, so its Mahalanobis radius comes out ordinary.

Mahalanobis measures distance from a mean in units of covariance. A displacement aligned with a
high-variance direction of the reference is cheap by that metric however large it is. A
supervised probe gets to pick its own direction; an unsupervised distance does not.

**What this costs the earlier claim.** I wrote after the first direction that the representation
knows the input is foreign while the output insists otherwise, and offered the distance score as
the constructive result of the audit. Half of that survives. The structure is in the
representation, confirmed at 0.994 in both directions. But recovering it *without labelled target
data* worked in one direction and failed in the other, so a distance-based gate is not a
dependable safeguard here, and reporting it as one on the strength of a single direction would
have been exactly the kind of claim this project exists to check.

## 2026-09-19 — Stage 4 ablations: a power spectrum with extra steps

Ten transforms applied to test inputs, on ten thousand chunks per direction, for band power, the
frozen checkpoint and the fine-tuned model alike.

### The controls pass exactly

Amplitude scaling by half and by two moves balanced accuracy by **0.0000** in all six
model-by-direction cells. Per-chunk normalisation cancels it, as the pipeline claims. Had this
moved at all, nothing else in the stage would have been worth reading.

### Phase randomisation costs nothing

Replacing every chunk with a surrogate that has the same power spectrum and a destroyed waveform:

| model | IBL → Allen | Allen → IBL |
|---|---:|---:|
| band power | −0.002 | −0.001 |
| frozen audio model | −0.024 | +0.013 |
| fine-tuned wav2vec2 | −0.008 | +0.001 |

Band power must be unaffected, since it is computed from the spectrum the transform preserves,
and it is; that is the invariant holding on real data rather than in a unit test.

The two neural models are also unaffected. A ninety-five-million-parameter transformer over raw
voltage, given inputs whose temporal structure has been destroyed and whose spectrum has not,
performs the same. On this task it is reading spectral power and essentially nothing else.

That is the answer to the question Stage 3 raised about why fine-tuning adds 0.007 over an
untrained checkpoint: there is not much else in the representation for supervision to sharpen.

### Removing the contaminated bands *helps*, in one direction

| transform | band power | frozen | fine-tuned |
|---|---:|---:|---:|
| stop gamma, Allen → IBL | −0.020 | **+0.085** | **+0.153** |
| stop ripple, Allen → IBL | −0.130 | +0.025 | +0.047 |
| stop gamma, IBL → Allen | −0.104 | −0.057 | −0.012 |
| stop ripple, IBL → Allen | −0.104 | −0.010 | −0.007 |

Training on Allen and testing on IBL, deleting the gamma band raises the fine-tuned model from
0.256 to 0.409 against a chance of 0.250. Above-chance performance goes from 0.006 to 0.159,
which is about two fifths of the within-lab margin, recovered by throwing information away.

Training on IBL and testing on Allen, the same deletion does nothing.

The asymmetry is explained by D11, recorded in Stage 1 before any model existed. Allen retains
energy above 300 Hz where the IBL pipeline band-passes it away, by up to a factor of 768. A model
trained on Allen learns frequency content that does not exist in IBL, and is misled by its
absence at test time; removing that content from both sides removes the mistake. A model trained
on IBL never had the opportunity to learn it, so there is nothing to remove.

Band power behaves in the opposite way throughout, losing accuracy whenever a band is deleted,
because every one of its six features carries signal and none of them is a learned shortcut.

### What Stage 4 establishes, in order of how much it took to find out

1. **The remedy does not reach the failure.** Temperature scaling repairs in-lab calibration
   fourfold and moves the cross-lab number by 0.05. The target lab needed 8.26 and 11.79 where
   in-lab fitting produced 1.84 and 1.72.
2. **The model cannot refuse, and the unsupervised alternative is unreliable.** Every abstention
   score has risk-coverage area above the base error rate. Distance from the training
   distribution detects the shift at 0.955 in one direction and 0.495 in the other.
3. **It is a spectral classifier.** Destroying temporal structure while preserving the spectrum
   costs it nothing.
4. **Part of the cross-lab failure is a learned artefact, and is removable.** Deleting the bands
   where the two pipelines disagree recovers two fifths of the within-lab margin, in the one
   direction where the artefact could have been learned.

The fourth is the only constructive result in the stage, and it is worth being precise about what
it offers: not a method, but a demonstration that some of the gap is preprocessing rather than
physiology, and that harmonising the two pipelines before training would be a cheaper experiment
than any architectural change.

---

## 2026-09-19 (later) — Stage 5: closing the gap, on the paper's own terms

The reproduction agrees with the paper within lab and falls below the majority-class rate across
labs where the paper reports about eleven points above it. Stage 5 asks what the paper does that
this reproduction does not, lever by lever, with every fix applied to every model alike.

### First, what the paper's code actually does

Reading `script/post_processing.ipynb` and `script/wav2vec_random_init.py` line by line settled
three things (D1, D14):

- The self-supervised stage runs **per dataset**. It never sees two labs and cannot align them.
  It is not the missing piece, and it will not be run.
- The post-processing averages **logits** over every trial of a channel, multiplies by a hand-set
  class weight, and argmaxes to one label per channel; then takes the mode over ±2 index-neighbours
  on the same shank. On their own session this is worth 0.716 → 0.793 → 0.836.
- Whether the cross-lab matrix includes post-processing is stated nowhere.

### Lever P: their post-processing, applied to everything

Raw accuracy against the target lab's majority rate, the paper's Figure 2e convention:

| | IBL → Allen, raw → after | margin | Allen → IBL, raw → after | margin |
|---|---:|---:|---:|---:|
| paper, Figure 2e *(figure)* | 0.56 | **+0.11** | 0.49 | **+0.12** |
| reproduction | 0.420 → 0.424 | −0.03 | 0.309 → 0.307 | −0.06 |
| band power | 0.520 → 0.622 | +0.06 → **+0.17** | 0.412 → 0.398 | +0.04 → +0.03 |
| **electrode position** | 0.729 → 0.729 | **+0.27** | 0.577 → 0.577 | **+0.21** |

Three findings.

**Post-processing cannot rescue the reproduction**, in either direction, by more than 0.004.
This was the prediction: the model predicts one class for 95% of chunks at 0.997 confidence, and
averaging chunks that agree then voting over neighbours that agree has nothing to change. The
unit test built for exactly this case passes on real data.

**Electrode position beats the paper's published cross-lab margin by more than double**, in both
directions, with no signal at all. On the paper's own metric and chance definition, four numbers
describing where the contact sits transfer across labs better than the published model does.

**Band power plus the paper's own post-processing clears the paper's margin** from IBL to Allen,
+0.17 against +0.11. Post-processing helps band power because its errors are noise, which is the
case the pipeline was built for, rather than a systematic collapse.

Within lab, post-processing lifts the three fine-tune folds from 0.716 to 0.764, a gain of 0.05,
which matches what the paper reports for itself.

### Lever C: per-probe embedding centering, with controls

| | value |
|---|---:|
| control: uncentred head on re-embedded training set | raw 0.416, balanced 0.194 (run's own: 0.420, 0.193) |
| control: centred training vs centred in-lab validation | balanced 0.647 (chance 0.250) |
| centred cross-lab | raw 0.130, balanced 0.239 (chance 0.200) |
| lab identity after centering | **0.526** (was 0.997) |

The first control lands on the run's own numbers, so the saved embeddings are aligned and the
rest can be read. The second shows centering keeps region information within lab.

What it does is remove the lab signature almost completely and leave cross-lab decoding at chance
regardless. The two labs are not the same structure displaced: once the displacement is removed,
the within-probe geometry that separates regions in one lab still does not separate them in the
other. Consistent with Stage 4's spectral-classifier finding, the region-discriminating content is
itself lab-specific. The lever that attacks the embedding geometry directly fails, informatively.

### Levers H and W act on the input instead

H, fine-tuning with both datasets low-passed at 100 Hz, is running. W, per-probe spectral
whitening, is implemented and tested and launches only if H falls short. Both remove the lab's
spectral fingerprint before the model sees it, which after lever C is the only place left for the
difference to live.

### Lever H: harmonised-band training restores transfer

Both directions fine-tuned with every input low-passed at 100 Hz, the corner chosen in Stage 2
before any cross-lab number existed. Then their post-processing on top.

| | IBL → Allen | Allen → IBL |
|---|---:|---:|
| paper, Figure 2e margin *(figure)* | +0.11 | +0.12 |
| full band, Stage 3 | −0.03 | −0.06 |
| **H, harmonised** | +0.04 | **+0.12** |
| **H + their post-processing** | **+0.15** | **+0.12** |
| electrode position | +0.27 | +0.21 |

Balanced accuracy tells the same story more plainly: 0.455 and 0.370 against a chance of 0.250,
where the full-band model sat at 0.242 and 0.258. Calibration error fell from 0.556 to 0.348 and
from 0.656 to 0.258. Lab identity fell from 0.997 to 0.830 and from 0.994 to 0.735.

So harmonised training plus the paper's own post-processing **clears the published cross-lab
margin in one direction and ties it in the other**, within the ±0.01 that reading a bar chart
allows, using no labels from the target lab. The full-band model was below the majority-class rate
in both. The difference is one preprocessing choice the paper does not control: which frequencies
the two pipelines let through.

Two things keep this honest. Electrode position is still ahead by 0.09 to 0.13. And in the
direction where H alone ties the paper, their post-processing adds almost nothing (+0.116 →
+0.124), so the tie is the representation's, not the prior's.

### What harmonisation does to the two Stage 4 failures

**Calibration.** Temperature scaling from in-lab now reaches cross-lab, partly. IBL → Allen goes
0.352 → 0.238 with the in-lab temperature, against an oracle of 0.044; the gap between fitted and
needed temperature shrank from 4.5× to 2.1×. Allen → IBL is already well calibrated in-lab
(0.045), the fitted temperature is 0.93, and applying it cross-lab moves 0.258 to 0.272: nothing
to transfer. The remedy now does something, and still not enough.

**Abstention.** This reversed. With the full-band model every risk-coverage area sat above the
error at full coverage. With H, distance from the training distribution ranks errors in *both*
directions: keeping the most confident fifth by that score gives 0.305 and 0.360 error against
0.506 and 0.515 for everything. Confidence and entropy rank in one direction only. So the
representation's distance is now a working ranker where before it was at best a gate, and the
reason is that the model is no longer collapsed onto one class, so its embeddings vary with the
input again.

Per-probe centering on H is running. Whitening launches next, because a tie in one direction is
the case the plan reserved it for.

### Lever C on the harmonised model

| | IBL → Allen | Allen → IBL |
|---|---:|---:|
| H alone: raw margin / balanced | +0.041 / 0.364 | +0.116 / 0.370 |
| H + C: raw margin / balanced | +0.034 / 0.355 | +0.096 / 0.413 |
| H + C calibration error | 0.389 | **0.145** |
| lab identity after C | 0.494 | 0.463 |
| control: uncentred head | +0.007 / 0.334 | +0.127 / 0.412 |
| control: centred in-lab | 0.574 | 0.638 |

Centering removes what remained of the lab signature, to 0.49 and 0.46, and is roughly neutral
on transfer: slightly worse raw accuracy in both directions, and in Allen → IBL a better balanced
accuracy, meaning it trades majority-class hits for minority-class ones. It produces the best
cross-lab calibration error in the project, 0.145, in that direction.

The reading is the same as on the full-band model, at a smaller scale. The lab signature and the
region signal are not separable by a translation. What harmonisation did was shrink the
signature; what centering does is remove the remainder without touching what limits transfer.

### The same bug, a second time, with a comment saying it was avoided

The whitening runs were chained to start when the centering runs released the GPU:

```
while pgrep -f "bin/lfpaudit adapt" >/dev/null; do sleep 60; done
```

with a comment above it reading "by waiting on their own process rather than polling a pattern
that could match this shell." The pattern `bin/lfpaudit adapt` appears verbatim in that shell's
own command line. It matched itself and waited forever, exactly as the Stage 3 chain did with
`lfpaudit finetune run`. Caught within the hour this time, because after Stage 3 the habit is to
check that the launched thing is burning processor time rather than trusting that the launch
happened. The cost was under an hour of idle GPU.

The comment is the instructive part. I knew the failure mode, named it, and reproduced it in the
next line, because the check I wrote was still a pattern match against process names and any
such check can match the process doing the checking. The only version that cannot fail this way
is the one that does not poll: run the second job on the line after the first in one script, or
wait on a specific process id. Whitening now runs that way.

### Lever W: whitening is worse than harmonisation, decisively

Per-probe spectral whitening at train and test, both directions, on the paper's terms:

| | IBL → Allen | Allen → IBL |
|---|---:|---:|
| H, harmonised: raw margin / balanced | +0.041 / 0.455 | +0.116 / 0.370 |
| **W, whitened: raw margin / balanced** | **−0.162 / 0.321** | +0.118 / 0.368 |
| W calibration error | 0.534 | 0.333 |
| W lab identity | 0.912 | 0.805 |

In one direction whitening is worse than doing nothing at all: −0.162 against the full-band
model's −0.032. In the other it ties harmonisation on accuracy and loses on calibration and lab
identity. Their post-processing hurts it in both directions.

The reading: whitening removes too much. Dividing every channel by its probe's mean spectrum
erases the spectral shape the whole probe shares, and on a hippocampal probe that shared shape,
strong theta, elevated gamma, *is* part of what says hippocampus. Harmonisation clips only the
band where the labs' pipelines disagree and leaves the rest; whitening flattens everything and
throws the anatomy out with the acquisition.

One number worth keeping from W: in Allen → IBL its distance-based abstention is the strongest
in the project, 0.174 error on the fifth it trusts most against 0.512 overall. The representation
is well organised even where its classifier is not. Not pursued further; the lever is H.

---

## 2026-09-21 — I attacked my own headline and it fell over

Preparing to strengthen the repository, I wrote down how its intended reader would attack each
claim. The loudest one, that electrode position alone beats the signal-based models, had an
obvious line of attack: position is only useful if it is known, so which of the four position
features would a user actually have? Depth along the shank, yes. Lateral offset, yes. Channel
number, yes. Depth as a fraction of the span of *kept* channels: no. Kept channels are chosen by
histology label. That feature is computed from the answer.

On its own it scores 0.81 and 0.76 across labs. Without it, position scores 0.09 and 0.30, at or
below chance. The cross-lab "position beats everything" result was the leak and nothing else. Full
ablation and the list of withdrawn claims are in DEVIATIONS D15; a correction notice went to the
top of the public README within the hour, before the regenerated tables were ready, because a
wrong claim with a notice on it is better than a wrong claim without one.

Why no test or control caught it: the permutation controls shuffle training labels, which breaks
the link between features and labels in *training*, but the leak lives in how a test-time feature
is computed from test-time label scope. A permuted-label model cannot exploit it, so the control
passes. The leakage verifier checks that no session is on both sides of a split, which was true.
The feature was the fifth kind of mistake this project has made, and the first the existing guards
were structurally unable to see. The new unit test asks the right question directly: does the
feature change when a channel drops out of scope?

What the corrected picture says is more useful than what it replaces. Honest position is strong
exactly where insertions are stereotyped (IBL, 0.77), weak where they are not (Allen, 0.49), and
worthless across labs. The signal is the opposite: about 0.70 within either lab. So the two are
complementary, and the study built the same day measures how:

- fusing them as a product of experts never hurts within lab, adding 0.02 to position on IBL and
  0.20 to position on Allen;
- on IBL the signal overtakes position once insertion depth is uncertain by more than roughly
  ±280 µm, and the fusion stays above both at every level tested;
- my expectation that the signal would rescue position near anatomical boundaries was wrong. Every
  source degrades together near boundaries, where the labels themselves are least certain.


