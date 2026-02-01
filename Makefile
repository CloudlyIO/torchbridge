.PHONY: test test-unit test-gpu lint format typecheck validate validate-full doctor benchmark docker-build docker-test clean install release

# ---- Installation ----
install:
	pip install -e ".[dev]"

# ---- Testing ----
test:
	PYTHONPATH=src pytest tests/ -v --tb=short \
		-m "not gpu and not slow and not tpu and not amd and not intel"

test-unit:
	PYTHONPATH=src pytest tests/ -v --tb=short -m "unit"

test-gpu:
	PYTHONPATH=src pytest tests/ -v --tb=short -m "gpu"

# ---- Linting & Formatting ----
lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

# ---- Type Checking ----
typecheck:
	mypy src/torchbridge --ignore-missing-imports

# ---- Validation ----
validate:
	PYTHONPATH=src python -m torchbridge.cli.validate --level quick

validate-full:
	PYTHONPATH=src python -m torchbridge.cli.validate --level full

# ---- Doctor ----
doctor:
	PYTHONPATH=src python -m torchbridge.cli.doctor

# ---- Benchmarking ----
benchmark:
	PYTHONPATH=src python -m torchbridge.cli.benchmark --predefined optimization --quick

# ---- Docker ----
docker-build:
	docker build -t torchbridge:latest -f docker/Dockerfile .

docker-test:
	docker run --rm torchbridge:latest python -m pytest tests/ -v --tb=short \
		-m "not gpu and not slow and not tpu and not amd and not intel"

# ---- Cleanup ----
clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf reports/ benchmark_results/

# ---- Release ----
release:
	python -m build
	@echo "Built distribution in dist/"
	@echo "To upload: twine upload dist/*"
