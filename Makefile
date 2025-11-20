.PHONY: help install install-dev clean clean-all test test-unit test-integration test-e2e test-fast test-watch coverage coverage-html lint format format-check typecheck check validate experiments benchmark docs serve-docs server server-dev server-workflow mcp-memory mcp-search docker-up docker-down docker-logs docker-clean version

# =============================================================================
# HoloLoom Makefile - Centralized Development Tasks
# =============================================================================
# This Makefile provides convenient shortcuts for common development tasks.
# Run `make help` to see all available targets.
# =============================================================================

# Color output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
RESET := \033[0m

# Paths
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy
UVICORN := $(PYTHON) -m uvicorn

PROJECT_NAME := hololoom
PYTHON_VERSION := 3.10+
ROOT_DIR := $(CURDIR)
SRC_DIR := $(ROOT_DIR)/HoloLoom
TESTS_DIR := $(ROOT_DIR)/HoloLoom/tests
DEMOS_DIR := $(ROOT_DIR)/demos
DOCS_DIR := $(ROOT_DIR)/docs
BUILD_DIR := $(ROOT_DIR)/build
DIST_DIR := $(ROOT_DIR)/dist
VENV_DIR := $(ROOT_DIR)/.venv

# Test paths by tier
UNIT_TESTS := $(TESTS_DIR)/unit
INTEGRATION_TESTS := $(TESTS_DIR)/integration
E2E_TESTS := $(TESTS_DIR)/e2e

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# HELP TARGET
# =============================================================================

help:  ## Show this help message
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(BLUE)║         HoloLoom Development Commands (Makefile)              ║$(RESET)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@echo "$(GREEN)TESTING$(RESET)"
	@echo "  make test              Run all tests"
	@echo "  make test-unit         Run unit tests only (<500ms)"
	@echo "  make test-integration  Run integration tests only (<2s)"
	@echo "  make test-e2e          Run end-to-end tests (<30s)"
	@echo "  make test-fast         Run unit + integration tests"
	@echo "  make test-watch        Run tests in watch mode (requires pytest-watch)"
	@echo "  make coverage          Generate coverage report (text)"
	@echo "  make coverage-html     Generate HTML coverage report"
	@echo ""
	@echo "$(GREEN)CODE QUALITY$(RESET)"
	@echo "  make lint              Run ruff linter"
	@echo "  make format            Format code with black"
	@echo "  make format-check      Check formatting without changes"
	@echo "  make typecheck         Run mypy type checking"
	@echo "  make check             Run all quality checks (lint + format + typecheck)"
	@echo ""
	@echo "$(GREEN)DEVELOPMENT$(RESET)"
	@echo "  make install           Install package in development mode"
	@echo "  make install-dev       Install with all dev dependencies"
	@echo "  make install-all       Install with all optional dependencies"
	@echo "  make clean             Remove build artifacts and caches"
	@echo "  make clean-all         Deep clean including .venv"
	@echo "  make docs              Build documentation"
	@echo "  make serve-docs        Serve documentation locally"
	@echo ""
	@echo "$(GREEN)SERVERS$(RESET)"
	@echo "  make server            Start API server (port 8000)"
	@echo "  make server-dev        Start server with auto-reload"
	@echo "  make server-workflow   Start workflow executor (port 8001)"
	@echo "  make mcp-memory        Start memory MCP server"
	@echo "  make mcp-search        Start search MCP server"
	@echo ""
	@echo "$(GREEN)VALIDATION$(RESET)"
	@echo "  make validate          Run complete validation pipeline"
	@echo "  make experiments       Run automated experiments suite"
	@echo "  make benchmark         Run performance benchmarks"
	@echo ""
	@echo "$(GREEN)DOCKER$(RESET)"
	@echo "  make docker-up         Start Neo4j + Qdrant containers"
	@echo "  make docker-down       Stop containers"
	@echo "  make docker-logs       View container logs"
	@echo "  make docker-clean      Remove containers and volumes"
	@echo ""
	@echo "$(GREEN)UTILITIES$(RESET)"
	@echo "  make version           Show version information"
	@echo "  make help              Show this help message"
	@echo ""

# =============================================================================
# INSTALLATION & SETUP
# =============================================================================

install:  ## Install package in development mode
	@echo "$(BLUE)Installing HoloLoom in development mode...$(RESET)"
	$(PIP) install -e .
	@echo "$(GREEN)✓ Installation complete$(RESET)"

install-dev:  ## Install with all development dependencies
	@echo "$(BLUE)Installing HoloLoom with dev dependencies...$(RESET)"
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)✓ Installation complete$(RESET)"

install-all:  ## Install with all optional dependencies
	@echo "$(BLUE)Installing HoloLoom with ALL dependencies...$(RESET)"
	$(PIP) install -e ".[all,dev]"
	@echo "$(GREEN)✓ Installation complete$(RESET)"

# =============================================================================
# CLEANING
# =============================================================================

clean:  ## Remove build artifacts, caches, and __pycache__
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	rm -rf $(BUILD_DIR) $(DIST_DIR)
	rm -rf .eggs *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf .ruff_cache .mypy_cache
	@echo "$(GREEN)✓ Clean complete$(RESET)"

clean-all: clean  ## Deep clean including .venv
	@echo "$(BLUE)Deep cleaning (including .venv)...$(RESET)"
	rm -rf $(VENV_DIR)
	rm -rf .pytest_cache .coverage htmlcov
	@echo "$(GREEN)✓ Deep clean complete$(RESET)"

# =============================================================================
# TESTING
# =============================================================================

test:  ## Run all tests
	@echo "$(BLUE)Running all tests...$(RESET)"
	$(PYTEST) $(TESTS_DIR) -v --tb=short
	@echo "$(GREEN)✓ All tests completed$(RESET)"

test-unit:  ## Run unit tests only (<500ms)
	@echo "$(BLUE)Running unit tests...$(RESET)"
	$(PYTEST) $(UNIT_TESTS) -v --tb=short -m "not integration and not e2e"
	@echo "$(GREEN)✓ Unit tests completed$(RESET)"

test-integration:  ## Run integration tests only (<2s)
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(PYTEST) $(INTEGRATION_TESTS) -v --tb=short
	@echo "$(GREEN)✓ Integration tests completed$(RESET)"

test-e2e:  ## Run end-to-end tests (<30s)
	@echo "$(BLUE)Running end-to-end tests...$(RESET)"
	$(PYTEST) $(E2E_TESTS) -v --tb=short
	@echo "$(GREEN)✓ End-to-end tests completed$(RESET)"

test-fast: test-unit test-integration  ## Run unit + integration tests (fast)
	@echo "$(GREEN)✓ Fast test suite completed$(RESET)"

test-watch:  ## Run tests in watch mode (requires pytest-watch)
	@echo "$(BLUE)Starting test watch mode (watching for file changes)...$(RESET)"
	@command -v ptw >/dev/null 2>&1 || (echo "$(RED)Error: pytest-watch not installed. Install with: pip install pytest-watch$(RESET)" && exit 1)
	ptw $(TESTS_DIR) -- -v --tb=short
	@echo "$(GREEN)✓ Test watch mode stopped$(RESET)"

coverage:  ## Generate coverage report (text format)
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	$(PYTEST) $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=xml -v
	@echo "$(GREEN)✓ Coverage report generated$(RESET)"

coverage-html: coverage  ## Generate HTML coverage report
	@echo "$(BLUE)Generating HTML coverage report...$(RESET)"
	$(PYTEST) $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=html
	@echo "$(GREEN)✓ HTML coverage report generated in htmlcov/index.html$(RESET)"
	@echo "Open htmlcov/index.html in your browser to view the report"

# =============================================================================
# CODE QUALITY
# =============================================================================

lint:  ## Run ruff linter
	@echo "$(BLUE)Running ruff linter...$(RESET)"
	@command -v ruff >/dev/null 2>&1 || (echo "$(RED)Error: ruff not installed. Install with: pip install ruff$(RESET)" && exit 1)
	$(RUFF) check $(SRC_DIR) $(DEMOS_DIR) --show-source
	@echo "$(GREEN)✓ Lint check complete$(RESET)"

format:  ## Format code with black
	@echo "$(BLUE)Formatting code with black...$(RESET)"
	@command -v black >/dev/null 2>&1 || (echo "$(RED)Error: black not installed. Install with: pip install black$(RESET)" && exit 1)
	$(BLACK) $(SRC_DIR) $(DEMOS_DIR) --line-length=100
	@echo "$(GREEN)✓ Code formatted$(RESET)"

format-check:  ## Check formatting without changes
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	@command -v black >/dev/null 2>&1 || (echo "$(RED)Error: black not installed. Install with: pip install black$(RESET)" && exit 1)
	$(BLACK) $(SRC_DIR) $(DEMOS_DIR) --line-length=100 --check --diff
	@echo "$(GREEN)✓ Format check complete$(RESET)"

typecheck:  ## Run mypy type checking
	@echo "$(BLUE)Running type checking with mypy...$(RESET)"
	@command -v mypy >/dev/null 2>&1 || (echo "$(RED)Error: mypy not installed. Install with: pip install mypy$(RESET)" && exit 1)
	$(MYPY) $(SRC_DIR) --ignore-missing-imports --show-error-codes
	@echo "$(GREEN)✓ Type checking complete$(RESET)"

check: lint format-check typecheck  ## Run all quality checks (lint + format + typecheck)
	@echo "$(GREEN)✓ All quality checks passed$(RESET)"

# =============================================================================
# DOCUMENTATION
# =============================================================================

docs:  ## Build documentation
	@echo "$(BLUE)Building documentation...$(RESET)"
	@if [ -d "$(DOCS_DIR)" ]; then \
		cd $(DOCS_DIR) && make html; \
	else \
		echo "$(YELLOW)Warning: docs directory not found at $(DOCS_DIR)$(RESET)"; \
	fi
	@echo "$(GREEN)✓ Documentation build complete$(RESET)"

serve-docs:  ## Serve documentation locally
	@echo "$(BLUE)Starting documentation server...$(RESET)"
	@if [ -d "$(DOCS_DIR)/_build/html" ]; then \
		echo "$(GREEN)Opening docs at http://localhost:8888$(RESET)"; \
		cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8888; \
	else \
		echo "$(RED)Error: Documentation not built. Run 'make docs' first.$(RESET)" && exit 1; \
	fi

# =============================================================================
# SERVERS & SERVICES
# =============================================================================

server:  ## Start API server (port 8000)
	@echo "$(BLUE)Starting HoloLoom API server on http://localhost:8000$(RESET)"
	@echo "$(YELLOW)Documentation available at http://localhost:8000/docs$(RESET)"
	PYTHONPATH=. $(UVICORN) HoloLoom.server.agentic_api:app --host 127.0.0.1 --port 8000

server-dev:  ## Start server with auto-reload
	@echo "$(BLUE)Starting HoloLoom API server with auto-reload on http://localhost:8000$(RESET)"
	@echo "$(YELLOW)The server will restart when code changes$(RESET)"
	PYTHONPATH=. $(UVICORN) HoloLoom.server.agentic_api:app --host 127.0.0.1 --port 8000 --reload

server-workflow:  ## Start workflow executor (port 8001)
	@echo "$(BLUE)Starting workflow executor on http://localhost:8001$(RESET)"
	PYTHONPATH=. $(PYTHON) HoloLoom/web_dashboard/workflow_executor.py

mcp-memory:  ## Start memory MCP server
	@echo "$(BLUE)Starting memory MCP server...$(RESET)"
	PYTHONPATH=. $(PYTHON) HoloLoom/server/mcp_memory.py

mcp-search:  ## Start search MCP server
	@echo "$(BLUE)Starting search MCP server...$(RESET)"
	PYTHONPATH=. $(PYTHON) HoloLoom/server/mcp_search.py

# =============================================================================
# VALIDATION & BENCHMARKING
# =============================================================================

validate:  ## Run complete validation pipeline
	@echo "$(BLUE)Running validation pipeline...$(RESET)"
	@echo "$(BLUE)  1. Running tests...$(RESET)"
	$(PYTEST) $(TESTS_DIR) -v --tb=short -q
	@echo "$(BLUE)  2. Checking code quality...$(RESET)"
	$(RUFF) check $(SRC_DIR) --show-source -q
	$(BLACK) $(SRC_DIR) --line-length=100 --check -q
	@echo "$(BLUE)  3. Running type checking...$(RESET)"
	$(MYPY) $(SRC_DIR) --ignore-missing-imports -q
	@echo "$(GREEN)✓ Validation pipeline complete$(RESET)"

experiments:  ## Run automated experiments suite
	@echo "$(BLUE)Running experiments...$(RESET)"
	PYTHONPATH=. $(PYTHON) experiments/run_experiments.py
	@echo "$(GREEN)✓ Experiments complete. Results in experiments/results/$(RESET)"

benchmark:  ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(RESET)"
	PYTHONPATH=. $(PYTHON) -m pytest HoloLoom/tests/ -v --benchmark-only
	@echo "$(GREEN)✓ Benchmarks complete$(RESET)"

# =============================================================================
# DOCKER MANAGEMENT
# =============================================================================

docker-up:  ## Start Neo4j + Qdrant containers
	@echo "$(BLUE)Starting Docker containers (Neo4j + Qdrant)...$(RESET)"
	@command -v docker-compose >/dev/null 2>&1 || (echo "$(RED)Error: docker-compose not installed$(RESET)" && exit 1)
	docker-compose up -d
	@echo "$(GREEN)✓ Containers started$(RESET)"
	@echo "$(YELLOW)Neo4j available at http://localhost:7687$(RESET)"
	@echo "$(YELLOW)Qdrant available at http://localhost:6333$(RESET)"

docker-down:  ## Stop Docker containers
	@echo "$(BLUE)Stopping Docker containers...$(RESET)"
	@command -v docker-compose >/dev/null 2>&1 || (echo "$(RED)Error: docker-compose not installed$(RESET)" && exit 1)
	docker-compose down
	@echo "$(GREEN)✓ Containers stopped$(RESET)"

docker-logs:  ## View Docker container logs
	@echo "$(BLUE)Showing Docker container logs (Ctrl+C to exit)...$(RESET)"
	@command -v docker-compose >/dev/null 2>&1 || (echo "$(RED)Error: docker-compose not installed$(RESET)" && exit 1)
	docker-compose logs -f

docker-clean:  ## Remove Docker containers and volumes
	@echo "$(BLUE)Removing Docker containers and volumes...$(RESET)"
	@command -v docker-compose >/dev/null 2>&1 || (echo "$(RED)Error: docker-compose not installed$(RESET)" && exit 1)
	docker-compose down -v
	@echo "$(GREEN)✓ Containers and volumes removed$(RESET)"

# =============================================================================
# UTILITIES
# =============================================================================

version:  ## Show version information
	@echo "$(BLUE)HoloLoom Project Information$(RESET)"
	@echo "  Project: $(PROJECT_NAME)"
	@echo "  Python Version: $(PYTHON_VERSION)"
	@echo "  Root Directory: $(ROOT_DIR)"
	@echo ""
	@echo "$(BLUE)Package Version:$(RESET)"
	@$(PYTHON) -c "from HoloLoom import __version__" 2>/dev/null && \
		$(PYTHON) -c "from HoloLoom import __version__; print('  HoloLoom: ' + __version__)" || \
		echo "  HoloLoom: Unable to determine version"
	@echo ""
	@echo "$(BLUE)Python Information:$(RESET)"
	@$(PYTHON) --version
	@$(PYTHON) -c "import sys; print('  Executable: ' + sys.executable)"

# =============================================================================
# UTILITY RULES (Hidden from help)
# =============================================================================

.PHONY: _check-python _check-poetry

_check-python:
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "$(RED)Error: Python not found$(RESET)"; exit 1; }

_check-poetry:
	@command -v poetry >/dev/null 2>&1 || { echo "$(RED)Error: Poetry not found$(RESET)"; exit 1; }

# =============================================================================
# Development Tips
# =============================================================================
#
# Common workflows:
#
#   # Quick development loop
#   make install-dev && make test-fast && make check
#
#   # Watch tests while developing
#   make test-watch
#
#   # Full validation before commit
#   make validate
#
#   # Work on a feature with auto-reload server
#   make server-dev
#
#   # Setup Docker for persistence layer
#   make docker-up
#
# =============================================================================
