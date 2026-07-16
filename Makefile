# Carretera de desarrollo. `make check` es el gate local (= lo que exige CI).

.PHONY: setup setup-gpu test test-cpu lint type check doctor models bench hooks

setup:            ## Entorno base (sin GPU extras)
	uv sync

hooks:            ## Activa los git hooks del repo (pre-push corre make check)
	git config core.hooksPath .githooks

setup-gpu:        ## + onnxruntime-gpu y cupy (pesado, Fase 1)
	uv sync --extra gpu

test:
	uv run pytest

test-cpu:         ## La suite exactamente como la corre CI (sin GPU)
	KURAI_DISABLE_GPU=1 uv run pytest -v

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
