.PHONY: install format lint test train pipeline docker precommit validate

install:
	uv sync

format:
	uv run ruff format .

lint:
	uv run ruff check .

test:
	uv run pytest

train:
	uv run python -m src.training.train

pipeline:
	uv run dvc repro

docker:
	docker-compose up --build

precommit:
	uv run pre-commit run --all-files

validate:
	uv run python validate_env.py
