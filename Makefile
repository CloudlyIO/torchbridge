PYTHON ?= $(shell command -v python3 || command -v python)

.PHONY: test test-unit test-gpu test-amd test-intel lint format typecheck validate validate-full doctor benchmark docker-build docker-build-amd docker-build-intel docker-test clean install release

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

test-amd:
	PYTHONPATH=src pytest tests/ -v --tb=short -m "amd"

test-intel:
	PYTHONPATH=src pytest tests/ -v --tb=short -m "intel"

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
	PYTHONPATH=src $(PYTHON) -m torchbridge.cli.validate --level quick

validate-full:
	PYTHONPATH=src $(PYTHON) -m torchbridge.cli.validate --level full

# ---- Doctor ----
doctor:
	PYTHONPATH=src $(PYTHON) -m torchbridge.cli.doctor

# ---- Benchmarking ----
benchmark:
	PYTHONPATH=src $(PYTHON) -m torchbridge.cli.benchmark --predefined optimization --quick

# ---- Docker ----
docker-build:
	docker build -t torchbridge:cpu -f docker/Dockerfile.cpu .

docker-build-amd:
	docker build -t torchbridge:amd -f docker/Dockerfile.amd .

docker-build-intel:
	docker build -t torchbridge:intel -f docker/Dockerfile.intel .

docker-test:
	docker run --rm torchbridge:cpu python3 -m pytest tests/ -v --tb=short \
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
	$(PYTHON) -m build
	@echo "Built distribution in dist/"
	@echo "To upload: twine upload dist/*"
