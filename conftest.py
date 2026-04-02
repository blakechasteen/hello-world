"""
Pytest configuration for HoloLoom v2 test suite.

Some test files use a standalone runner pattern (custom check() + asyncio.run(main()))
with functions that pass results between stages. These can't be auto-collected by
pytest because pytest tries to resolve their parameters as fixtures.

This conftest:
  1. Excludes standalone-runner files from pytest collection
  2. Provides a subprocess wrapper so `pytest` still runs them
  3. Restricts collection to project root (skips .venv, archive, etc.)
"""

import subprocess
import sys

import pytest

# Files that use standalone runner (functions take manual args, not fixtures)
# These were moved from repo root to tests/manual/
STANDALONE_SCRIPTS = [
    "tests/manual/test_v2_bridge.py",
    "tests/manual/test_v2_cognitive.py",
]

collect_ignore_glob = ["tests/manual/*"]


# ---------------------------------------------------------------------------
# Subprocess wrappers — one pytest test per standalone script
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script", STANDALONE_SCRIPTS)
def test_standalone_suite(script):
    """Run standalone test scripts as subprocesses."""
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        cwd=".",
        timeout=120,
    )
    # Print output for visibility in pytest -v
    if result.stdout:
        print(result.stdout[-2000:])
    assert result.returncode == 0, (
        f"{script} failed (exit {result.returncode}):\n"
        f"{result.stderr[-1000:]}"
    )
