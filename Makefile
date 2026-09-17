# Every target is reproducible from a clean checkout. Targets belonging to later stages
# announce themselves rather than failing silently, so the intended surface stays visible.
PY := .venv/bin/python
CLI := .venv/bin/lfpaudit

.PHONY: setup lint format test smoke gate data-ibl data-allen baselines finetune ablate figures clean

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

data-ibl:
	@echo "Stage 1 - not implemented yet"

data-allen:
	@echo "Stage 1 - not implemented yet"

baselines:
	@echo "Stage 2 - not implemented yet"

finetune:
	@echo "Stage 3 - not implemented yet"

ablate:
	@echo "Stage 4 - not implemented yet"

figures:
	@echo "Stage 6 - not implemented yet"

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
