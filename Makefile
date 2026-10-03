# ODEForge developer entry points.
#
# `lint` and `test` are hard gates: neither appends `|| true`, and CI runs exactly these.

PY ?= python
RUFF ?= $(PY) -m ruff
PYTEST ?= $(PY) -m pytest
RESULTS ?= results

.PHONY: help lint fmt test test-gates coverage demo demo-twice benchmark invariants clean check

help:
	@echo "ODEForge targets:"
	@echo "  make install    install dev dependencies from the lock file"
	@echo "  make lint       ruff check (hard gate, no || true)"
	@echo "  make fmt        ruff format in place"
	@echo "  make test       pytest with coverage"
	@echo "  make check      lint + test, the full local gate"
	@echo "  make demo       end-to-end demo -> $(RESULTS)/benchmark.json"
	@echo "  make demo-twice demo twice and assert bit-for-bit determinism"
	@echo "  make benchmark  full acceptance gates -> $(RESULTS)/benchmark.json"
	@echo "  make clean      remove caches and results"

install:
	$(PY) -m pip install -r requirements.lock.txt

lint:
	$(RUFF) check .

fmt:
	$(RUFF) format .

# Fast subset: ~35 s, the inner-loop signal.
test:
	$(PYTEST) -W ignore::UserWarning -m "not slow"

# Acceptance gates: runs the gates for real, so it costs minutes.
test-gates:
	$(PYTEST) -W ignore::UserWarning -m slow

coverage:
	$(PYTEST) -W ignore::UserWarning --cov=core --cov=data --cov=integrators \
		--cov=routing --cov=autodiff --cov=training --cov=hpo --cov=eval --cov=pipeline \
		--cov-report=term-missing --cov-fail-under=80 --cov-report=xml

check: lint test

demo:
	$(PY) examples/run_demo.py

demo-twice:
	$(PY) examples/run_demo.py
	$(PY) tools/check_determinism.py --snapshot
	$(PY) examples/run_demo.py
	$(PY) tools/check_determinism.py

benchmark:
	$(PY) cli.py benchmark --out $(RESULTS)/benchmark.json

invariants:
	$(PY) cli.py invariants

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov coverage.xml $(RESULTS)
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
