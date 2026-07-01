install:
<<<<<<< HEAD
	uv sync

lint:
	ruff check .

train:
	uv run python -m src.training.train

pipeline:
	dvc repro

docker:
	docker-compose up --build
=======
uv sync
format:
uv run ruff format .
lint:
uv run ruff check .
test:
uv run pytest
precommit:
uv run pre-commit run --all-files
validate:
uv run python scripts/validate_env.py
>>>>>>> bb88f6dca1e1bace9202d3afa24a5d18ee7dbf70
