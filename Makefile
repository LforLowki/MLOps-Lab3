.PHONY: install install-notorch install-withtorch lint format test refactor all run-api run-cli run-train

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn

install-notorch:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .[notorch]

install-withtorch:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .[withtorch]

# Default install: withtorch
install: install-withtorch

lint:
	$(PY) -m pylint lab1 || true

format:
	$(PY) -m black lab1 tests

test:
	$(PY) -m pytest

refactor: format lint

run-train:
	$(PY) lab1/models/train.py --epochs 1 --batch-size 16 --lr 0.001
	$(PY) lab1/models/select_export.py

all: install-withtorch format lint test run-train

run-api:
	$(UVICORN) lab1.api.api:app --reload

run-cli:
	$(PY) -m lab1.cli.cli
