"""
Tests for CodebaseSpinner

Tests Python AST parsing, class/function extraction, and MemoryShard conversion.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import ast

from hololoom.spinningWheel.codebase_spinner import (
    CodebaseSpinner,
    PythonParser,
    CodeFunction,
    CodeClass,
    CodeFile,
    spin_codebase,
    create_codebase_scorer
)


# Fixtures

@pytest.fixture
def temp_code_dir():
    """Create temporary directory for test code files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_python_file(temp_code_dir):
    """Create sample Python file"""
    code = '''"""
    This is a module docstring.
    """

import numpy as np
from typing import List

class DataProcessor:
    """Process data using various algorithms."""

    def __init__(self, config):
        self.config = config

    def process(self, data: List[int]) -> List[int]:
        """Process the data."""
        return [x * 2 for x in data]

async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    return {}

def helper_function():
    """Helper function."""
    pass
'''

    file_path = temp_code_dir / "example.py"
    file_path.write_text(code)
    return file_path


@pytest.fixture
def sample_code_file(sample_python_file):
    """Create sample CodeFile object"""
    return PythonParser.parse_file(sample_python_file)


# CodeFunction Tests

def test_code_function_properties():
    """Test CodeFunction properties"""
    func = CodeFunction(
        name="test_func",
        signature="def test_func(arg1: int) -> str",
        docstring="Test function",
        line_start=10,
        line_end=15,
        is_async=False,
        is_method=False,
        calls=["print", "len"]
    )

    assert func.name == "test_func"
    assert func.is_async == False
    assert func.is_method == False
    assert len(func.calls) == 2


# CodeClass Tests

def test_code_class_properties():
    """Test CodeClass properties"""
    cls = CodeClass(
        name="TestClass",
        bases=["BaseClass"],
        docstring="Test class",
        line_start=5,
        line_end=20,
        methods=[
            CodeFunction(
                name="method1",
                signature="def method1(self)",
                docstring=None,
                line_start=10,
                line_end=12,
                is_method=True
            )
        ]
    )

    assert cls.name == "TestClass"
    assert len(cls.bases) == 1
    assert len(cls.methods) == 1


# CodeFile Tests

def test_code_file_complexity_score(sample_code_file):
    """Test CodeFile.complexity_score property"""
    complexity = sample_code_file.complexity_score

    assert complexity > 0
    # Should be based on classes + functions + lines


# PythonParser Tests

def test_python_parser_parse_file(sample_python_file):
    """Test parsing Python file"""
    code_file = PythonParser.parse_file(sample_python_file)

    assert code_file.language == 'python'
    assert code_file.docstring == "This is a module docstring."
    assert len(code_file.imports) > 0
    assert len(code_file.classes) == 1
    assert len(code_file.functions) >= 2  # fetch_data, helper_function


def test_python_parser_extract_imports(sample_python_file):
    """Test import extraction"""
    with open(sample_python_file) as f:
        source = f.read()

    tree = ast.parse(source)
    imports = PythonParser._extract_imports(tree)

    assert 'numpy' in str(imports) or 'np' in str(imports)
    assert 'typing.List' in imports or 'List' in imports


def test_python_parser_extract_classes(sample_python_file):
    """Test class extraction"""
    with open(sample_python_file) as f:
        source = f.read()

    tree = ast.parse(source)
    classes = PythonParser._extract_classes(tree, source)

    assert len(classes) == 1
    assert classes[0].name == "DataProcessor"
    assert classes[0].docstring == "Process data using various algorithms."
    assert len(classes[0].methods) >= 2  # __init__, process


def test_python_parser_extract_functions(sample_python_file):
    """Test function extraction"""
    with open(sample_python_file) as f:
        source = f.read()

    tree = ast.parse(source)
    functions = PythonParser._extract_functions(tree, source)

    assert len(functions) >= 2

    # Check async function
    async_funcs = [f for f in functions if f.is_async]
    assert len(async_funcs) == 1
    assert async_funcs[0].name == "fetch_data"


def test_python_parser_count_lines(sample_python_file):
    """Test line counting"""
    with open(sample_python_file) as f:
        source = f.read()

    code_lines = PythonParser._count_code_lines(source)
    comment_lines = PythonParser._count_comment_lines(source)

    assert code_lines > 0
    # Comment lines might be 0 in this sample


# CodebaseSpinner Tests

@pytest.mark.asyncio
async def test_codebase_spinner_initialization():
    """Test CodebaseSpinner initialization"""
    spinner = CodebaseSpinner(
        importance_threshold=0.3,
        languages=['python'],
        include_tests=False
    )

    assert spinner.get_name() == "codebase"
    assert spinner.importance_threshold == 0.3
    assert 'python' in spinner.languages


@pytest.mark.asyncio
async def test_codebase_spinner_capabilities():
    """Test spinner capabilities"""
    spinner = CodebaseSpinner()

    caps = spinner.get_capabilities()
    assert caps.basic_processing
    assert caps.entity_extraction
    assert caps.motif_extraction
    assert caps.importance_scoring
    assert caps.streaming
    assert 'python' in caps.supported_formats


@pytest.mark.asyncio
async def test_codebase_spinner_availability():
    """Test spinner availability check"""
    spinner = CodebaseSpinner()

    # Should always be available (ast is stdlib)
    assert spinner.is_available()


def test_codebase_spinner_get_code_files(temp_code_dir):
    """Test getting code files from directory"""
    # Create test files
    (temp_code_dir / "file1.py").write_text("# Python file")
    (temp_code_dir / "file2.py").write_text("# Another Python file")
    (temp_code_dir / "test_file.py").write_text("# Test file")
    (temp_code_dir / "readme.txt").write_text("# Not a code file")

    spinner = CodebaseSpinner(languages=['python'], include_tests=False)

    files = spinner._get_code_files(temp_code_dir, recursive=False)

    # Should get file1.py and file2.py, but not test_file.py or readme.txt
    assert len(files) == 2


def test_codebase_spinner_parse_file(sample_python_file):
    """Test parsing code file"""
    spinner = CodebaseSpinner()

    code_file = spinner._parse_file(sample_python_file)

    assert code_file.language == 'python'
    assert len(code_file.classes) == 1
    assert len(code_file.functions) >= 2


def test_codebase_spinner_format_file_text(sample_code_file):
    """Test code file text formatting"""
    spinner = CodebaseSpinner()

    formatted = spinner._format_file_text(sample_code_file)

    assert "File:" in formatted
    assert "Language: python" in formatted
    assert "Lines:" in formatted
    assert "Classes" in formatted
    assert "Functions" in formatted
    assert "DataProcessor" in formatted  # Class name


def test_codebase_spinner_extract_entities(sample_code_file):
    """Test entity extraction from code file"""
    spinner = CodebaseSpinner()

    entities = spinner._extract_entities(sample_code_file)

    assert "DataProcessor" in entities  # Class
    assert "fetch_data" in entities     # Function
    # Imports
    assert any('numpy' in e or 'np' in e or 'typing' in e for e in entities)


def test_codebase_spinner_extract_motifs(sample_code_file):
    """Test motif extraction from code file"""
    spinner = CodebaseSpinner()

    motifs = spinner._extract_motifs(sample_code_file)

    assert 'python' in motifs
    assert 'object_oriented' in motifs  # Has classes
    assert 'functional' in motifs       # Has functions
    assert 'async' in motifs            # Has async function


def test_codebase_spinner_score_importance(sample_code_file):
    """Test code file importance scoring"""
    spinner = CodebaseSpinner()

    score = spinner.score_importance(sample_code_file)

    assert 0.0 <= score.score <= 1.0
    assert score.signals is not None

    # Should score reasonably high (has classes, functions, docstrings)
    assert score.score > 0.3


def test_codebase_spinner_file_to_shards(sample_code_file):
    """Test converting code file to shards"""
    spinner = CodebaseSpinner(importance_threshold=0.0)  # Include all

    shards = spinner._file_to_shards(sample_code_file)

    assert len(shards) == 1

    # Check shard
    shard = shards[0]
    assert shard.metadata['language'] == 'python'
    assert shard.metadata['class_count'] == 1
    assert shard.metadata['function_count'] >= 2
    assert shard.metadata['complexity_score'] > 0


def test_codebase_spinner_file_to_shards_filtering():
    """Test importance threshold filtering"""
    spinner = CodebaseSpinner(importance_threshold=1.0)  # Filter all

    # Create minimal code file
    code_file = CodeFile(
        file_path=Path("test.py"),
        language='python',
        total_lines=5,
        code_lines=3
    )

    shards = spinner._file_to_shards(code_file)

    # Should filter
    assert len(shards) == 0


@pytest.mark.asyncio
async def test_codebase_spinner_spin_single_file(sample_python_file):
    """Test spinning a single file"""
    spinner = CodebaseSpinner(importance_threshold=0.0)

    result = await spinner.spin(sample_python_file)

    assert result.success is True
    assert result.shard_count == 1
    assert result.shards[0].metadata['language'] == 'python'


@pytest.mark.asyncio
async def test_codebase_spinner_spin_directory(temp_code_dir):
    """Test spinning entire directory"""
    # Create test files
    (temp_code_dir / "file1.py").write_text('"""Doc"""\ndef func1(): pass')
    (temp_code_dir / "file2.py").write_text('"""Doc"""\ndef func2(): pass')

    spinner = CodebaseSpinner(importance_threshold=0.0)

    shards = await spinner.spin_directory(temp_code_dir, recursive=False)

    # Should get shards from both files
    assert len(shards) == 2


@pytest.mark.asyncio
async def test_codebase_spinner_max_files_limit(temp_code_dir):
    """Test max_files limit"""
    # Create many test files
    for i in range(10):
        (temp_code_dir / f"file{i}.py").write_text(f'"""Doc"""\ndef func{i}(): pass')

    spinner = CodebaseSpinner(importance_threshold=0.0, max_files=3)

    shards = await spinner.spin_directory(temp_code_dir, recursive=False)

    # Should only process 3 files
    assert len(shards) == 3


@pytest.mark.asyncio
async def test_create_codebase_scorer():
    """Test codebase-specific importance scorer creation"""
    scorer = create_codebase_scorer()

    # Should have code-specific terms
    assert len(scorer.technical_scorer.technical_terms) > 0

    # Test scoring
    code_text = "This function implements an async method with decorators."
    tech_score = scorer.technical_scorer.score(code_text)

    assert tech_score > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
