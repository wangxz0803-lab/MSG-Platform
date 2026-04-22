.PHONY: install lint format typecheck test test-unit test-integration migrate serve worker docker-up docker-down clean

install:
	pip install -e ".[platform,dev]"

lint:
	ruff check src/ tests/ platform/ scripts/
	black --check src/ tests/ platform/ scripts/

format:
	ruff check --fix src/ tests/ platform/ scripts/
	black src/ tests/ platform/ scripts/

typecheck:
	mypy src/msg_embedding/

test-unit:
	pytest tests/unit/ -x -v

test-integration:
	pytest tests/integration/ -x -v --timeout=600

test: test-unit

migrate:
	alembic upgrade head

serve:
	uvicorn platform.backend.app:create_app --factory --reload --port 8000

worker:
	dramatiq platform.worker.actors --processes 1 --threads 4

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
