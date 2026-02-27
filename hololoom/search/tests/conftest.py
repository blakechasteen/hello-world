"""
Pytest configuration for search tests.
"""
import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def disable_guardrails():
    """Disable safety guardrails for tests."""
    os.environ["DISABLE_GUARDRAILS"] = "1"
    yield
    if "DISABLE_GUARDRAILS" in os.environ:
        del os.environ["DISABLE_GUARDRAILS"]
