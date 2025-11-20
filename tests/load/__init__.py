"""Load testing package for HoloLoom VoiceAgent + Elle AR integration.

This package contains comprehensive load testing infrastructure using Locust.

Components:
    - locustfile.py: Main Locust user simulation
    - benchmarks.py: Performance baselines and validation
    - scenarios/: Pre-configured test scenarios
    - LOAD_TESTING_README.md: Complete documentation

Quick Start:
    python -m pytest tests/load/ -v
    locust -f tests/load/locustfile.py --host=http://localhost:8000
"""

__version__ = "1.0.0"
__author__ = "Wave 3 Production Hardening"
__date__ = "2025-11-16"
