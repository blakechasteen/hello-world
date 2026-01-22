"""
SpinningWheel Spinner Tests
===========================

Comprehensive test suite for the SpinningWheel data ingestion spinners.

Test Modules:
-------------
Core Spinner Tests:
- test_protocol.py - SpinnerProtocol interface and BaseSpinner tests (~500 lines)
- test_codebase_spinner.py - Codebase analysis and code parsing tests (~800 lines)
- test_pdf_spinner.py - PDF document processing tests (~400 lines)
- test_auto.py - Universal auto-router and format detection tests (~400 lines)

Media Spinner Tests:
- test_youtube_spinner.py - YouTube video transcription tests
- test_image_spinner.py - OCR image processing tests
- test_url_spinner.py - Web page parsing tests
- test_email_spinner.py - Email parsing tests

Test Organization:
------------------
tests/
├── __init__.py          # This file
├── conftest.py          # Shared fixtures and helpers
├── test_protocol.py     # Protocol and base class tests
├── test_codebase_spinner.py  # Codebase spinner tests
├── test_pdf_spinner.py  # PDF spinner tests
├── test_auto.py         # Auto-router tests
└── ...                  # Additional spinner tests

Running Tests:
--------------
# Run all SpinningWheel tests
pytest HoloLoom/spinningWheel/tests/ -v

# Run specific test module
pytest HoloLoom/spinningWheel/tests/test_protocol.py -v

# Run with markers
pytest HoloLoom/spinningWheel/tests/ -v -m "not slow"
pytest HoloLoom/spinningWheel/tests/ -v -m "not requires_pdf"

# Run with coverage
pytest HoloLoom/spinningWheel/tests/ -v --cov=HoloLoom.spinningWheel

Test Markers:
-------------
- slow: Tests that take >5 seconds
- integration: Tests requiring external services
- requires_pdf: Tests requiring PDF parsing libraries (PyPDF2/pdfplumber)
- requires_git: Tests requiring git repository

Fixtures (from conftest.py):
----------------------------
- temp_dir: Temporary directory for test files
- sample_python_file: Sample Python code file
- sample_python_simple_file: Simple Python file
- sample_python_invalid_file: File with syntax errors
- sample_python_complex_file: High complexity Python file
- sample_typescript_file: TypeScript code file
- sample_javascript_file: JavaScript code file
- sample_codebase: Multi-file project structure
- sample_codebase_with_imports: Codebase with cross-file imports
- mock_memory_backend: Mock memory backend for testing
- importance_signals: Sample importance scoring signals

Total Test Coverage:
--------------------
- Protocol Tests: ~500 lines (17 test classes)
- Codebase Spinner Tests: ~800 lines (20 test classes)
- PDF Spinner Tests: ~400 lines (14 test classes)
- Auto Router Tests: ~400 lines (14 test classes)

Total: ~2,100+ lines of comprehensive tests

Author: Claude Code
Date: January 2026
"""

# Re-export test utilities for convenience
from .conftest import (
    create_memory_shard,
    assert_shard_valid,
    assert_spin_result_valid,
    MockMemoryShard,
    MockSpinResult,
    SAMPLE_PYTHON_CODE,
    SAMPLE_PYTHON_SIMPLE,
    SAMPLE_PYTHON_COMPLEX,
    SAMPLE_TYPESCRIPT_CODE,
    SAMPLE_JAVASCRIPT_CODE,
)

__all__ = [
    'create_memory_shard',
    'assert_shard_valid',
    'assert_spin_result_valid',
    'MockMemoryShard',
    'MockSpinResult',
    'SAMPLE_PYTHON_CODE',
    'SAMPLE_PYTHON_SIMPLE',
    'SAMPLE_PYTHON_COMPLEX',
    'SAMPLE_TYPESCRIPT_CODE',
    'SAMPLE_JAVASCRIPT_CODE',
]
