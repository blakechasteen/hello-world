#!/bin/bash
#
# HoloLoom Performance Report Generator
# ======================================
# Runs performance tests and generates comprehensive report.
#

set -e

echo "============================================================================"
echo "HoloLoom Phase 2 Performance Report"
echo "============================================================================"
echo ""
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Python: $(python --version)"
echo "CPU: $(nproc) cores"
echo "Memory: $(free -h | awk '/^Mem:/ {print $2}') total"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if databases are running
echo "Checking database connectivity..."
echo ""

NEO4J_STATUS="❌ Not available"
if nc -z localhost 7687 2>/dev/null; then
    NEO4J_STATUS="✅ Available (bolt://localhost:7687)"
fi
echo "Neo4j:  $NEO4J_STATUS"

QDRANT_STATUS="❌ Not available"
if nc -z localhost 6333 2>/dev/null; then
    QDRANT_STATUS="✅ Available (http://localhost:6333)"
fi
echo "Qdrant: $QDRANT_STATUS"

echo ""
echo "Note: Performance tests will use mock mode if databases unavailable."
echo ""

# Run performance tests
echo "============================================================================"
echo "Running Performance Tests"
echo "============================================================================"
echo ""

# Storage performance
echo "1. Storage Performance Tests"
echo "----------------------------"
pytest HoloLoom/tests/performance/test_phase2_performance.py::TestStoragePerformance \
    -v -s --tb=short 2>&1 | grep -E "(📊|PASSED|FAILED|ERROR)"
echo ""

# Clustering performance
echo "2. Clustering Performance Tests"
echo "-------------------------------"
pytest HoloLoom/tests/performance/test_phase2_performance.py::TestClusteringPerformance \
    -v -s --tb=short 2>&1 | grep -E "(📊|PASSED|FAILED|ERROR)"
echo ""

# MCP performance
echo "3. MCP Server Performance Tests"
echo "--------------------------------"
pytest HoloLoom/tests/performance/test_phase2_performance.py::TestMCPPerformance \
    -v -s --tb=short 2>&1 | grep -E "(📊|PASSED|FAILED|ERROR)"
echo ""

# File processing performance
echo "4. File Processing Performance Tests"
echo "-------------------------------------"
pytest HoloLoom/tests/performance/test_phase2_performance.py::TestFileProcessingPerformance \
    -v -s --tb=short 2>&1 | grep -E "(📊|PASSED|FAILED|ERROR)"
echo ""

# End-to-end performance
echo "5. End-to-End Pipeline Performance"
echo "-----------------------------------"
pytest HoloLoom/tests/performance/test_phase2_performance.py::TestEndToEndPerformance \
    -v -s --tb=short 2>&1 | grep -E "(📊|PASSED|FAILED|ERROR)"
echo ""

# Summary
echo "============================================================================"
echo "Performance Report Complete"
echo "============================================================================"
echo ""
echo "Summary:"
echo "  ✓ All performance tests executed"
echo "  ✓ Metrics captured and displayed"
echo ""
echo "Next Steps:"
echo "  • Compare against baseline targets (see PERFORMANCE_TESTING.md)"
echo "  • Investigate any failures or slow tests"
echo "  • Run with real databases for production-like results"
echo ""
echo "To run with real databases:"
echo "  docker-compose -f HoloLoom/tests/docker-compose.yml up -d"
echo "  ./HoloLoom/tests/run_performance_report.sh"
echo "  docker-compose -f HoloLoom/tests/docker-compose.yml down"
echo ""
echo "============================================================================"
