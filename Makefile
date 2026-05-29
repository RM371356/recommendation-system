install:
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
