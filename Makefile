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
	$(PIP) install pylint || true
	$(VENV)/bin/pylint lab1 || true

format:
	$(VENV)/bin/black lab1 tests

test:
	$(VENV)/bin/pytest -v --cov=lab1

refactor: format lint

all: install format lint test

run-api:
	$(UVICORN) lab1.api.api:app --reload

run-cli:
	$(PY) -m lab1.cli.cli
