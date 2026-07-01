install:
	uv sync

lint:
	ruff check .

train:
	uv run python -m src.training.train

pipeline:
	dvc repro

docker:
	docker-compose up --build