.PHONY: install lint format test refactor all run-api run-cli

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn

install:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .

lint:
	$(PY) -m pylint lab1 || true

format:
	$(PY) -m black lab1 tests

test:
	$(PY) -m pytest

refactor: format lint

all: install format lint test

run-api:
	$(UVICORN) lab1.api.api:app --reload

run-cli:
	$(PY) -m lab1.cli.cli
