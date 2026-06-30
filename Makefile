# Supply Chain Control Tower - developer entrypoints.
# Windows users without `make`: run the underlying commands shown below,
# or use Docker (`make up` -> `docker compose up`).

.DEFAULT_GOAL := help
PY ?= python
PKG := sctower

.PHONY: help install dev-install hooks data lint format type test cov \
        run build up down logs clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime package
	$(PY) -m pip install -e .

dev-install: ## Install with dev + data extras and pre-commit hooks
	$(PY) -m pip install -e ".[dev,kaggle,postgres]"
	pre-commit install

hooks: ## Run all pre-commit hooks
	pre-commit run --all-files

data: ## Download and curate the Rossmann dataset
	$(PY) -m scripts.download_data
	$(PY) -m sctower.cli curate

lint: ## Ruff lint
	ruff check src tests

format: ## Black + ruff format
	black src tests scripts
	ruff check --fix src tests

type: ## mypy strict type-check
	mypy src

test: ## Run tests with coverage gate (>=80%)
	pytest

cov: ## Open the HTML coverage report
	pytest --cov-report=html && $(PY) -c "import webbrowser,pathlib;webbrowser.open(pathlib.Path('htmlcov/index.html').resolve().as_uri())"

run: ## Run the Dash app locally
	$(PY) -m sctower.app.main

build: ## Build the Docker image
	docker build -t sctower:local .

up: ## Start the full stack (app + postgres)
	docker compose up --build

down: ## Stop the stack
	docker compose down -v

logs: ## Tail app logs
	docker compose logs -f app

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
