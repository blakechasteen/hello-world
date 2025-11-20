#!/bin/bash
# pytest-cov Quick Reference Commands
# Save as: COVERAGE_QUICK_REFERENCE.sh
# Usage: source COVERAGE_QUICK_REFERENCE.sh

echo "==================================================================="
echo "pytest-cov Quick Reference - HoloLoom"
echo "==================================================================="
echo ""

# Function definitions
coverage_install() {
    echo "[1/4] Installing dependencies with pytest-cov..."
    pip install -r requirements.txt
}

coverage_test() {
    echo "[2/4] Running tests with coverage..."
    pytest HoloLoom/tests/ -v
}

coverage_view() {
    echo "[3/4] Opening coverage HTML report..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open htmlcov/index.html
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        start htmlcov\index.html
    else
        xdg-open htmlcov/index.html
    fi
}

coverage_report() {
    echo "[4/4] Showing coverage summary..."
    coverage report --data-file=.coverage
}

# Display available commands
echo "Available commands:"
echo ""
echo "  coverage_install    - Install dependencies (includes pytest-cov)"
echo "  coverage_test       - Run tests with coverage"
echo "  coverage_view       - Open HTML coverage report in browser"
echo "  coverage_report     - Show coverage summary in terminal"
echo ""
echo "Or run the full workflow:"
echo "  $ coverage_install && coverage_test && coverage_view"
echo ""
echo "==================================================================="
echo ""

# Configuration info
echo "Configuration Files:"
echo "  .coveragerc              - Coverage settings"
echo "  pytest.ini               - Pytest configuration (coverage auto-enabled)"
echo "  .github/workflows/coverage.yml - GitHub Actions workflow"
echo ""

echo "Report Locations:"
echo "  htmlcov/index.html       - Interactive HTML report"
echo "  coverage.xml             - XML report (for CI/CD)"
echo "  .coverage                - Binary coverage database"
echo ""

echo "Coverage Targets:"
echo "  Core modules:    >85%  (memory, policy, orchestrator)"
echo "  Features:        >75%  (embeddings, backends)"
echo "  Utils:           >70%  (helpers, types)"
echo ""

# Note about auto-coverage
echo "NOTE: Coverage runs automatically with pytest!"
echo "      Just run: pytest HoloLoom/tests/ -v"
echo ""
