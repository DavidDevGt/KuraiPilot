# Carretera de desarrollo. `make check` es el gate pre-commit.

.PHONY: setup setup-gpu test lint type check doctor models bench

setup:            ## Entorno base (sin GPU extras)
	uv sync

setup-gpu:        ## + onnxruntime-gpu y cupy (pesado, Fase 1)
	uv sync --extra gpu

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .

type:
	uv run mypy

check: lint type test  ## Gate completo: lint + tipos + tests

doctor:
	uv run kurai doctor

models:           ## Descarga y verifica ONNX (hash-pinned)
	uv run python tools/fetch_models.py

bench:            ## docs/05 §6 — se implementa ANTES que el pipeline (Fase 0)
	uv run kurai bench
