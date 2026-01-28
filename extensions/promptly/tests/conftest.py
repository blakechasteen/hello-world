"""Shared fixtures for Promptly tests."""
import os
import sys
from pathlib import Path
import pytest

# Ensure promptly package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from promptly.core.database import PromptlyDB
from promptly.core.promptly import Promptly


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary PromptlyDB instance."""
    db_path = tmp_path / "test.db"
    db = PromptlyDB(db_path)
    db.connect()
    yield db
    db.close()


@pytest.fixture
def tmp_promptly(tmp_path):
    """Create a Promptly instance with temporary directories."""
    global_dir = tmp_path / "global"
    local_dir = tmp_path / "local"

    p = Promptly(global_dir=global_dir, local_dir=local_dir, auto_detect_local=False)
    p.connect()
    yield p
    p.close()


@pytest.fixture
def global_only_promptly(tmp_path):
    """Create a Promptly instance with only global storage."""
    global_dir = tmp_path / "global"

    p = Promptly(global_dir=global_dir, local_dir=None, auto_detect_local=False)
    p.connect()
    yield p
    p.close()
