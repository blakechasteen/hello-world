"""
SpinningWheel Test Fixtures and Configuration
==============================================

Shared pytest fixtures for SpinningWheel spinner tests.

Author: Claude Code
Date: January 2026
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib


# =============================================================================
# Mock Classes
# =============================================================================

@dataclass
class MockMemoryShard:
    """Mock MemoryShard for testing without full HoloLoom dependency."""
    id: str
    text: str
    episode: str
    entities: List[str] = field(default_factory=list)
    motifs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class MockSpinResult:
    """Mock SpinResult for testing."""
    shards: List[MockMemoryShard] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    items_processed: int = 0
    items_filtered: int = 0
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Sample Code Fixtures
# =============================================================================

SAMPLE_PYTHON_CODE = '''#!/usr/bin/env python3
"""
Sample Python module for testing.

This module demonstrates various Python language features.
"""

import os
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass


# Constants
VERSION = "1.0.0"
MAX_RETRIES = 3


@dataclass
class Configuration:
    """Configuration container."""
    host: str = "localhost"
    port: int = 8080
    debug: bool = False

    def validate(self) -> bool:
        """Validate configuration values."""
        return self.port > 0 and self.host


class BaseProcessor:
    """Abstract base processor class."""

    def __init__(self, config: Configuration):
        self.config = config
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the processor."""
        self._initialized = True

    def process(self, data: str) -> str:
        """Process input data."""
        raise NotImplementedError("Subclasses must implement process()")


class DataProcessor(BaseProcessor):
    """Concrete data processor implementation."""

    def __init__(self, config: Configuration, batch_size: int = 100):
        super().__init__(config)
        self.batch_size = batch_size
        self._cache: Dict[str, str] = {}

    def process(self, data: str) -> str:
        """Process the input data with caching."""
        cache_key = hashlib.md5(data.encode()).hexdigest()

        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._transform(data)
        self._cache[cache_key] = result
        return result

    def _transform(self, data: str) -> str:
        """Transform the data."""
        return data.upper()

    async def process_async(self, data: str) -> str:
        """Async version of process."""
        return self.process(data)


def calculate_metrics(values: List[float]) -> Dict[str, float]:
    """
    Calculate statistical metrics from a list of values.

    Args:
        values: List of numeric values

    Returns:
        Dictionary with min, max, mean, and count
    """
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "count": 0}

    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "count": len(values)
    }


def main() -> int:
    """Main entry point."""
    config = Configuration(host="127.0.0.1", port=9000, debug=True)
    processor = DataProcessor(config)
    processor.initialize()

    result = processor.process("hello world")
    print(f"Result: {result}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

SAMPLE_PYTHON_SIMPLE = '''"""Simple module with one function."""

def greet(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''

SAMPLE_PYTHON_INVALID = '''def broken_function(
    """This has a syntax error - unclosed function definition."""
    return "oops"
'''

SAMPLE_PYTHON_COMPLEX = '''"""Complex module with high cyclomatic complexity."""

def complex_function(a, b, c, d):
    """Function with many branches."""
    result = 0

    if a > 0:
        if b > 0:
            if c > 0:
                result = a + b + c
            elif c == 0:
                result = a + b
            else:
                result = a - b
        elif b == 0:
            result = a
        else:
            if d > 0:
                result = a + d
            else:
                result = a - d
    elif a == 0:
        if b > 0:
            result = b
        else:
            result = -b
    else:
        for i in range(10):
            if i % 2 == 0:
                result += i
            else:
                result -= i

    while result > 100:
        result = result // 2
        if result < 50:
            break

    return result
'''

SAMPLE_TYPESCRIPT_CODE = '''/**
 * Sample TypeScript module for testing.
 */

import { EventEmitter } from 'events';

export interface Config {
    host: string;
    port: number;
}

export class Service extends EventEmitter {
    private config: Config;

    constructor(config: Config) {
        super();
        this.config = config;
    }

    async start(): Promise<void> {
        this.emit('start');
    }

    stop(): void {
        this.emit('stop');
    }
}

export function createService(config: Config): Service {
    return new Service(config);
}
'''

SAMPLE_JAVASCRIPT_CODE = '''/**
 * Sample JavaScript module.
 */

const http = require('http');

class Server {
    constructor(port) {
        this.port = port;
        this.server = null;
    }

    start() {
        this.server = http.createServer((req, res) => {
            res.writeHead(200);
            res.end('Hello');
        });
        this.server.listen(this.port);
    }

    stop() {
        if (this.server) {
            this.server.close();
        }
    }
}

module.exports = { Server };
'''


# =============================================================================
# Sample PDF Content
# =============================================================================

SAMPLE_PDF_TEXT = """Sample PDF Document

This is a sample PDF document for testing purposes.

Chapter 1: Introduction

This chapter introduces the main concepts covered in this document.
The document is structured to demonstrate PDF parsing capabilities.

Chapter 2: Methodology

The methodology section describes the approach taken in this work.
We follow a rigorous scientific process to ensure accuracy.

References

[1] Author A, "Sample Paper", Journal 2024
[2] Author B, "Another Paper", Conference 2023
"""


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    tmp = tempfile.mkdtemp(prefix="spinner_test_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_python_file(temp_dir):
    """Create a sample Python file."""
    file_path = temp_dir / "sample.py"
    file_path.write_text(SAMPLE_PYTHON_CODE)
    return file_path


@pytest.fixture
def sample_python_simple_file(temp_dir):
    """Create a simple Python file."""
    file_path = temp_dir / "simple.py"
    file_path.write_text(SAMPLE_PYTHON_SIMPLE)
    return file_path


@pytest.fixture
def sample_python_invalid_file(temp_dir):
    """Create an invalid Python file."""
    file_path = temp_dir / "invalid.py"
    file_path.write_text(SAMPLE_PYTHON_INVALID)
    return file_path


@pytest.fixture
def sample_python_complex_file(temp_dir):
    """Create a complex Python file."""
    file_path = temp_dir / "complex.py"
    file_path.write_text(SAMPLE_PYTHON_COMPLEX)
    return file_path


@pytest.fixture
def sample_typescript_file(temp_dir):
    """Create a sample TypeScript file."""
    file_path = temp_dir / "sample.ts"
    file_path.write_text(SAMPLE_TYPESCRIPT_CODE)
    return file_path


@pytest.fixture
def sample_javascript_file(temp_dir):
    """Create a sample JavaScript file."""
    file_path = temp_dir / "sample.js"
    file_path.write_text(SAMPLE_JAVASCRIPT_CODE)
    return file_path


@pytest.fixture
def sample_codebase(temp_dir):
    """Create a sample codebase directory structure."""
    # Create directory structure
    src_dir = temp_dir / "src"
    tests_dir = temp_dir / "tests"
    src_dir.mkdir()
    tests_dir.mkdir()

    # Main module
    (src_dir / "main.py").write_text(SAMPLE_PYTHON_CODE)

    # Utils module
    (src_dir / "utils.py").write_text(SAMPLE_PYTHON_SIMPLE)

    # Complex module
    (src_dir / "processor.py").write_text(SAMPLE_PYTHON_COMPLEX)

    # TypeScript file
    (src_dir / "service.ts").write_text(SAMPLE_TYPESCRIPT_CODE)

    # Test file
    (tests_dir / "test_main.py").write_text('''"""Tests for main module."""
import pytest
from src.main import DataProcessor

def test_processor():
    assert True
''')

    # Create __init__.py files
    (src_dir / "__init__.py").write_text('"""Source package."""')
    (tests_dir / "__init__.py").write_text('"""Tests package."""')

    # Create a README
    (temp_dir / "README.md").write_text("# Sample Project\n\nA sample project for testing.")

    return temp_dir


@pytest.fixture
def sample_codebase_with_imports(temp_dir):
    """Create a codebase with cross-file imports."""
    # Create package structure
    pkg_dir = temp_dir / "mypackage"
    pkg_dir.mkdir()

    # __init__.py
    (pkg_dir / "__init__.py").write_text('''"""My Package."""
from .core import CoreClass
from .utils import helper_function
''')

    # core.py
    (pkg_dir / "core.py").write_text('''"""Core module."""
from .utils import helper_function

class CoreClass:
    def __init__(self):
        self.value = helper_function()

    def run(self):
        return self.value * 2
''')

    # utils.py
    (pkg_dir / "utils.py").write_text('''"""Utility functions."""

def helper_function():
    """Return a constant value."""
    return 42

def format_string(s: str) -> str:
    """Format a string."""
    return s.strip().upper()
''')

    # models.py (imports from core)
    (pkg_dir / "models.py").write_text('''"""Data models."""
from .core import CoreClass
from dataclasses import dataclass

@dataclass
class Model:
    name: str
    core: CoreClass
''')

    return temp_dir


@pytest.fixture
def empty_codebase(temp_dir):
    """Create an empty directory."""
    return temp_dir


@pytest.fixture
def mock_pdf_path(temp_dir):
    """Create a mock PDF-like file (not actual PDF)."""
    pdf_path = temp_dir / "sample.pdf"
    # Write some text that would be extracted
    pdf_path.write_bytes(b"%PDF-1.4\nMock PDF content\n%%EOF")
    return pdf_path


@pytest.fixture
def mock_memory_backend():
    """Create a mock memory backend."""
    class MockBackend:
        def __init__(self):
            self.shards = []

        async def add_shards(self, shards):
            self.shards.extend(shards)

        async def add_shard(self, shard):
            self.shards.append(shard)

        def get_all_shards(self):
            return self.shards

    return MockBackend()


@pytest.fixture
def importance_signals():
    """Create sample importance signals."""
    return {
        'length_score': 0.7,
        'technical_score': 0.8,
        'structural_score': 0.6,
        'authority_score': 0.5,
        'recency_score': 0.9,
        'engagement_score': 0.5,
        'reference_score': 0.4,
        'noise_penalty': 0.1,
        'custom_signals': {'complexity': 0.75}
    }


# =============================================================================
# Helper Functions for Tests
# =============================================================================

def create_memory_shard(
    text: str,
    episode: str = "test",
    entities: Optional[List[str]] = None,
    motifs: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> MockMemoryShard:
    """Helper to create a MockMemoryShard."""
    shard_id = hashlib.sha256(text.encode()).hexdigest()[:12]
    return MockMemoryShard(
        id=f"test_{shard_id}",
        text=text,
        episode=episode,
        entities=entities or [],
        motifs=motifs or [],
        metadata=metadata or {}
    )


def assert_shard_valid(shard: Any) -> None:
    """Assert that a shard has valid structure."""
    assert hasattr(shard, 'id'), "Shard must have id"
    assert hasattr(shard, 'text'), "Shard must have text"
    assert hasattr(shard, 'episode'), "Shard must have episode"
    assert hasattr(shard, 'entities'), "Shard must have entities"
    assert hasattr(shard, 'motifs'), "Shard must have motifs"
    assert hasattr(shard, 'metadata'), "Shard must have metadata"

    assert isinstance(shard.id, str) and len(shard.id) > 0
    assert isinstance(shard.text, str)
    assert isinstance(shard.episode, str)
    assert isinstance(shard.entities, list)
    assert isinstance(shard.motifs, list)
    assert isinstance(shard.metadata, dict)


def assert_spin_result_valid(result: Any) -> None:
    """Assert that a SpinResult has valid structure."""
    assert hasattr(result, 'shards'), "Result must have shards"
    assert hasattr(result, 'success'), "Result must have success"
    assert isinstance(result.shards, list)
    assert isinstance(result.success, bool)


# =============================================================================
# Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_pdf: marks tests that require PDF parsing libraries"
    )
    config.addinivalue_line(
        "markers", "requires_git: marks tests that require git"
    )
