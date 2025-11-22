"""
Pytest configuration and shared fixtures for Trough & BossPig testing.

Created: 2025-11-22
"""

import pytest
import os
import tempfile
import json
from pathlib import Path
from typing import Generator, Dict, List, Any


# ============================================================================
# PATHS AND DIRECTORIES
# ============================================================================

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
TROUGH_FIXTURES_DIR = FIXTURES_DIR / "trough"
BOSSPIG_FIXTURES_DIR = FIXTURES_DIR / "bosspig"
LOGS_DIR = TESTS_DIR / "logs"

# Create logs directory if needed
LOGS_DIR.mkdir(exist_ok=True)


# ============================================================================
# PYTEST HOOKS
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    LOGS_DIR.mkdir(exist_ok=True)


def pytest_collection_modifyitems(config, items):
    """Mark tests automatically based on path."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        if "benchmark" in str(item.fspath) or "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)


# ============================================================================
# FIXTURE: TEMPORARY DIRECTORY
# ============================================================================

@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# FIXTURE: TROUGH FIXTURES
# ============================================================================

@pytest.fixture
def good_code_file(tmp_dir: Path) -> Path:
    """Load the good code fixture."""
    source = TROUGH_FIXTURES_DIR / "good_code.py"
    target = tmp_dir / "good_code.py"
    target.write_text(source.read_text())
    return target


@pytest.fixture
def bad_code_slop_file(tmp_dir: Path) -> Path:
    """Load the bad code with AI slop fixture."""
    source = TROUGH_FIXTURES_DIR / "bad_code_slop.py"
    target = tmp_dir / "bad_code_slop.py"
    target.write_text(source.read_text())
    return target


@pytest.fixture
def bad_code_security_file(tmp_dir: Path) -> Path:
    """Load the bad code with security issues fixture."""
    source = TROUGH_FIXTURES_DIR / "bad_code_security.py"
    target = tmp_dir / "bad_code_security.py"
    target.write_text(source.read_text())
    return target


@pytest.fixture
def bad_code_logic_file(tmp_dir: Path) -> Path:
    """Load the bad code with logic errors fixture."""
    source = TROUGH_FIXTURES_DIR / "bad_code_logic.py"
    target = tmp_dir / "bad_code_logic.py"
    target.write_text(source.read_text())
    return target


# ============================================================================
# FIXTURE: BOSSPIG FIXTURES
# ============================================================================

@pytest.fixture
def good_document_file(tmp_dir: Path) -> Path:
    """Load the good business document fixture."""
    source = BOSSPIG_FIXTURES_DIR / "good_document.md"
    target = tmp_dir / "good_document.md"
    target.write_text(source.read_text())
    return target


@pytest.fixture
def bad_document_slop_file(tmp_dir: Path) -> Path:
    """Load the bad business document with AI slop fixture."""
    source = BOSSPIG_FIXTURES_DIR / "bad_document_slop.md"
    target = tmp_dir / "bad_document_slop.md"
    target.write_text(source.read_text())
    return target


# ============================================================================
# FIXTURE: SAMPLE DATA
# ============================================================================

@pytest.fixture
def sample_code_snippets() -> Dict[str, str]:
    """Sample code snippets for testing specific detectors."""
    return {
        "missing_error_handling": "def divide(a, b):\n    return a / b",
        "hardcoded_secret": 'API_KEY = "sk-1234567890abcdefghijklmnopqrstuvwxyz"',
        "sql_injection": 'query = "SELECT * FROM users WHERE id = " + user_id',
        "xss_vulnerability": 'html = f"<div>{user_input}</div>"',
        "resource_leak": "def read_file():\n    f = open('file.txt')\n    content = f.read()\n    return content",
    }


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Configuration for test runs."""
    return {
        "trough": {
            "enabled": True,
            "categories": [
                "missing_error_handling",
                "hardcoded_secrets",
                "resource_leaks",
                "security_issues",
            ]
        },
        "bosspig": {
            "enabled": True,
            "categories": [
                "jargon",
                "vague_commitments",
                "missing_dates",
                "ai_hallucinations",
            ]
        }
    }


@pytest.fixture(autouse=True)
def cleanup_after_test(tmp_dir: Path):
    """Automatically clean up after each test."""
    yield
