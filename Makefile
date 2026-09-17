# Every target is reproducible from a clean checkout. Targets belonging to later stages
# announce themselves rather than failing silently, so the intended surface stays visible.
PY := .venv/bin/python
CLI := .venv/bin/lfpaudit

.PHONY: setup lint format test smoke gate data-ibl data-allen splits real-smoke cards inspect stage1 features features-w2v2 lab-discriminator results-table baselines finetune ablate figures clean

setup:                ## create the virtualenv and install the package
	uv venv --python 3.11
	uv pip install -e ".[dev]"

lint:                 ## ruff
	.venv/bin/ruff check .
	.venv/bin/ruff format --check lfpaudit tests

format:
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format lfpaudit tests

test:                 ## unit suite, synthetic data only, no network
	$(PY) -m pytest -q -m "not slow"

smoke:                ## end-to-end pipeline run against planted ground truth
	$(CLI) smoke

gate:                 ## everything that must pass before a real training run
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) smoke

data-ibl:             ## fetch and chunk the seven IBL insertions
	$(CLI) data ibl

data-allen:           ## read and chunk the Allen sessions
	$(CLI) data allen

splits:               ## write and verify every split
	$(CLI) make-splits

real-smoke:           ## the smoke gates, on real data
	$(CLI) real-smoke experiments/splits/cross_session_ibl.json

stage1: splits cards inspect real-smoke   ## every post-build step, in order
	@echo "Stage 1 complete: splits verified, cards and figures regenerated, real-data gate passed"

cards:                ## regenerate docs/DATA_CARD.md from the built stores
	$(CLI) card data/stores/ibl data/stores/allen --out docs/DATA_CARD.md

inspect:              ## regenerate the per-region inspection figures
	$(CLI) inspect data/stores/ibl
	$(CLI) inspect data/stores/allen

features:             ## build and cache the cheap feature tables
	$(CLI) features build

features-w2v2:        ## build frozen wav2vec2 embeddings (about 40 minutes on MPS)
	$(CLI) features build --which w2v2_frozen

baselines:            ## the full sweep across every fold of every scheme
	$(CLI) baselines run --which bandpower_full,bandpower_clean,geometry,amplitude,w2v2_frozen

lab-discriminator:    ## how separable are the two datasets
	$(CLI) lab-discriminator

results-table:        ## regenerate docs/RESULTS_BASELINES.md
	$(CLI) baselines table --out docs/RESULTS_BASELINES.md

finetune:
	@echo "Stage 3 - not implemented yet"

ablate:
	@echo "Stage 4 - not implemented yet"

figures:              ## regenerate the result figures from the committed tables
	$(PY) scripts/make_figures.py

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
