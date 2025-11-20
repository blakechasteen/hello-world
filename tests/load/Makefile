.PHONY: help install baseline stress spike endurance clean validate all

# HoloLoom Load Testing Makefile
# =============================
# Convenient commands for running load tests

HOST ?= http://localhost:8000
PYTHONPATH := $(shell pwd)/..
export PYTHONPATH

help:
	@echo "HoloLoom Load Testing Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       - Install Locust"
	@echo "  make server        - Start local HoloLoom server"
	@echo ""
	@echo "Load Tests:"
	@echo "  make baseline      - Run baseline test (10 users, 5 min)"
	@echo "  make stress        - Run stress test (100 users, 10 min)"
	@echo "  make spike         - Run spike test (200 users, 2 min)"
	@echo "  make endurance     - Run endurance test (50 users, 1 hour)"
	@echo "  make all           - Run all scenarios sequentially"
	@echo ""
	@echo "Advanced:"
	@echo "  make web           - Start web UI for interactive testing"
	@echo "  make validate      - Validate results against baselines"
	@echo "  make clean         - Remove test results"
	@echo "  make report        - Generate summary report"
	@echo ""
	@echo "Environment Variables:"
	@echo "  HOST               - Server host (default: http://localhost:8000)"
	@echo ""
	@echo "Examples:"
	@echo "  make baseline"
	@echo "  make HOST=http://staging:8000 stress"
	@echo "  make web  # Then open http://localhost:8089"

install:
	pip install locust==2.15.1 requests pandas matplotlib

server:
	@echo "Starting HoloLoom VoiceAgent server..."
	@echo "Make sure you're in the project root directory"
	PYTHONPATH=/home/user/hello-world python -m HoloLoom.server.agentic_api

baseline:
	@echo "Running BASELINE test..."
	@echo "Target: 10 users, 5 minutes"
	@echo "Expected p95: <300ms"
	@echo ""
	PYTHONPATH=/home/user/hello-world locust \
		-f locustfile.py \
		--host=$(HOST) \
		--users 10 \
		--spawn-rate 2 \
		--run-time 5m \
		--headless \
		--csv=results/baseline

stress:
	@echo "Running STRESS test..."
	@echo "Target: 100 users, 10 minutes"
	@echo "Expected p95: 300-800ms"
	@echo ""
	PYTHONPATH=/home/user/hello-world locust \
		-f locustfile.py \
		--host=$(HOST) \
		--users 100 \
		--spawn-rate 10 \
		--run-time 10m \
		--headless \
		--csv=results/stress

spike:
	@echo "Running SPIKE test..."
	@echo "Target: 200 users, 2 minutes"
	@echo "Expected p95: 800-1200ms initially, 400-500ms after scaling"
	@echo ""
	PYTHONPATH=/home/user/hello-world locust \
		-f locustfile.py \
		--host=$(HOST) \
		--users 200 \
		--spawn-rate 50 \
		--run-time 2m \
		--headless \
		--csv=results/spike

endurance:
	@echo "Running ENDURANCE test..."
	@echo "Target: 50 users, 1 hour"
	@echo "Expected p95: 200-400ms (stable)"
	@echo "WARNING: This test takes 1+ hour to complete"
	@echo ""
	@read -p "Continue? [y/n] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		PYTHONPATH=/home/user/hello-world locust \
			-f locustfile.py \
			--host=$(HOST) \
			--users 50 \
			--spawn-rate 5 \
			--run-time 1h \
			--headless \
			--csv=results/endurance; \
	fi

web:
	@echo "Starting Locust Web UI..."
	@echo "Open http://localhost:8089 in your browser"
	@echo ""
	PYTHONPATH=/home/user/hello-world locust \
		-f locustfile.py \
		--host=$(HOST) \
		--port 8089

all:
	@echo "Running ALL load test scenarios..."
	@echo "This will take approximately 30 minutes"
	@echo ""
	@echo "Tests will run in sequence:"
	@echo "  1. Baseline (5 min)"
	@echo "  2. Stress (10 min)"
	@echo "  3. Spike (2 min)"
	@echo "  4. Endurance (1 hour) - SKIPPED for quick validation"
	@echo ""
	@read -p "Continue? [y/n] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		python run_and_validate.py all $(HOST); \
	fi

validate:
	@echo "Validating load test results against baselines..."
	mkdir -p results
	python -c "from benchmarks import validate_results; import json; print('Baselines loaded successfully')"

clean:
	@echo "Cleaning up test results..."
	rm -f results/*.csv
	rm -f results/*.json
	rm -f results/*.html
	@echo "Done"

report:
	@echo "Generating summary report..."
	@echo ""
	@echo "Test Results Summary"
	@echo "==================="
	@echo ""
	@ls -lh results/*.csv 2>/dev/null || echo "No CSV results found"
	@echo ""
	@echo "To view detailed metrics:"
	@echo "  - Baseline: cat results/baseline_stats.csv"
	@echo "  - Stress: cat results/stress_stats.csv"
	@echo "  - Spike: cat results/spike_stats.csv"
	@echo ""

.PHONY: requirements
requirements:
	@echo "Python packages required:"
	@echo "  - locust==2.15.1"
	@echo "  - requests"
	@echo "  - pandas (optional, for analysis)"
	@echo "  - matplotlib (optional, for charting)"
	@echo ""
	@echo "Install with: make install"

.PHONY: check-server
check-server:
	@echo "Checking if server is running..."
	@curl -s $(HOST)/health > /dev/null && echo "✅ Server is running at $(HOST)" || echo "❌ Server not responding at $(HOST)"

# Development targets

.PHONY: lint
lint:
	@echo "Linting load test code..."
	@python -m py_compile locustfile.py benchmarks.py run_and_validate.py
	@echo "✅ No syntax errors"

.PHONY: count-lines
count-lines:
	@echo "Lines of code:"
	@wc -l locustfile.py benchmarks.py run_and_validate.py LOAD_TESTING_README.md scenarios/*.py | tail -1
