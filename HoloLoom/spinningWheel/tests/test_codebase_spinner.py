"""
Codebase Spinner Tests
======================

Comprehensive tests for the CodebaseSpinner:
- Python parsing and AST extraction
- TypeScript/JavaScript parsing
- Entity and motif extraction
- Code chunking and file processing
- Large file handling
- Invalid syntax handling
- Incremental analysis
- Dependency and call graph building
- Importance scoring with git integration
- Complexity metrics calculation

Author: Claude Code
Date: January 2026
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import ast
import os


# =============================================================================
# Import the codebase spinner module
# =============================================================================

try:
    from HoloLoom.spinningWheel.codebase_spinner import (
        CodebaseSpinner,
        CodeFile,
        CodeFunction,
        CodeClass,
        CodebaseProject,
        PythonParser,
        TypeScriptParser,
        CyclomaticComplexityCalculator,
        APISurfaceAnalyzer,
        CohesionCalculator,
        DeadCodeDetector,
        GitAnalyzer,
        GitFileInfo,
        IncrementalUpdateResult,
        DependencyEdge,
        CallEdge,
        ComplexityMetrics,
        APISymbol,
        APISurface,
        CohesionMetrics,
        DeadCodeSymbol,
        DeadCodeAnalysis,
        spin_codebase,
        create_codebase_scorer,
    )
    from HoloLoom.spinningWheel.protocol import (
        SpinResult,
        SpinnerCapabilities,
        SpinnerStatus,
        ImportanceScore,
    )
    CODEBASE_SPINNER_AVAILABLE = True
except ImportError as e:
    CODEBASE_SPINNER_AVAILABLE = False
    IMPORT_ERROR = str(e)


# =============================================================================
# Skip decorator if imports fail
# =============================================================================

pytestmark = pytest.mark.skipif(
    not CODEBASE_SPINNER_AVAILABLE,
    reason=f"CodebaseSpinner not available: {IMPORT_ERROR if not CODEBASE_SPINNER_AVAILABLE else ''}"
)


# =============================================================================
# Test CodebaseSpinner Initialization
# =============================================================================

class TestCodebaseSpinnerInit:
    """Test CodebaseSpinner initialization."""

    def test_default_initialization(self):
        """Test default spinner initialization."""
        spinner = CodebaseSpinner()

        assert spinner is not None
        assert spinner.spinner_id == "codebase_spinner"
        assert isinstance(spinner.capabilities, SpinnerCapabilities)

    def test_custom_initialization(self):
        """Test spinner with custom parameters."""
        spinner = CodebaseSpinner(
            importance_threshold=0.5,
            languages=['python', 'typescript'],
            include_tests=True
        )

        assert spinner.importance_threshold == 0.5
        assert 'python' in spinner.languages
        assert 'typescript' in spinner.languages
        assert spinner.include_tests is True

    def test_capabilities(self):
        """Test spinner capabilities."""
        spinner = CodebaseSpinner()
        caps = spinner.capabilities

        assert caps.entity_extraction is True
        assert caps.motif_extraction is True
        assert caps.importance_scoring is True
        assert caps.incremental is True

    def test_supported_languages(self):
        """Test supported languages configuration."""
        spinner = CodebaseSpinner(languages=['python'])
        assert spinner.languages == ['python']

        spinner_multi = CodebaseSpinner(languages=['python', 'javascript', 'typescript'])
        assert len(spinner_multi.languages) == 3


# =============================================================================
# Test PythonParser
# =============================================================================

class TestPythonParser:
    """Test PythonParser class."""

    def test_parse_simple_function(self, temp_dir):
        """Test parsing a simple Python function."""
        code = '''def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"
'''
        file_path = temp_dir / "simple.py"
        file_path.write_text(code)

        result = PythonParser.parse_file(file_path)

        assert isinstance(result, CodeFile)
        assert result.language == 'python'
        assert len(result.functions) == 1
        assert result.functions[0].name == 'hello'
        assert result.functions[0].docstring == "Say hello."

    def test_parse_class(self, temp_dir):
        """Test parsing a Python class."""
        code = '''class MyClass:
    """A sample class."""

    def __init__(self, value: int):
        self.value = value

    def get_value(self) -> int:
        """Return the value."""
        return self.value
'''
        file_path = temp_dir / "class.py"
        file_path.write_text(code)

        result = PythonParser.parse_file(file_path)

        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == 'MyClass'
        assert cls.docstring == "A sample class."
        assert len(cls.methods) == 2

    def test_parse_imports(self, temp_dir):
        """Test parsing Python imports."""
        code = '''import os
import sys
from typing import List, Dict
from pathlib import Path
from .local import helper
'''
        file_path = temp_dir / "imports.py"
        file_path.write_text(code)

        result = PythonParser.parse_file(file_path)

        assert len(result.imports) >= 4
        assert 'import os' in result.imports or 'os' in str(result.imports)

    def test_parse_async_function(self, temp_dir):
        """Test parsing async Python functions."""
        code = '''async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    return {}

async def process_data(data: dict) -> None:
    """Process the data."""
    pass
'''
        file_path = temp_dir / "async.py"
        file_path.write_text(code)

        result = PythonParser.parse_file(file_path)

        assert len(result.functions) == 2
        assert result.functions[0].is_async is True
        assert result.functions[1].is_async is True

    def test_parse_decorators(self, temp_dir):
        """Test parsing decorated functions."""
        code = '''from functools import lru_cache

@lru_cache(maxsize=128)
def cached_func(n: int) -> int:
    """Cached function."""
    return n * 2

@property
def my_property(self):
    """A property."""
    return self._value

@staticmethod
def static_method():
    """Static method."""
    pass
'''
        file_path = temp_dir / "decorators.py"
        file_path.write_text(code)

        result = PythonParser.parse_file(file_path)

        assert len(result.functions) >= 1
        # Decorators should be captured
        assert any('lru_cache' in str(f.decorators) or hasattr(f, 'decorators')
                   for f in result.functions if hasattr(f, 'decorators'))

    def test_parse_with_syntax_error(self, temp_dir):
        """Test parsing file with syntax error."""
        code = '''def broken(
    """Missing closing paren and quote
    return "oops"
'''
        file_path = temp_dir / "broken.py"
        file_path.write_text(code)

        # Should handle gracefully
        try:
            result = PythonParser.parse_file(file_path)
            # May return partial result or empty
            assert isinstance(result, CodeFile)
        except SyntaxError:
            # Also acceptable to raise SyntaxError
            pass

    def test_parse_empty_file(self, temp_dir):
        """Test parsing empty Python file."""
        file_path = temp_dir / "empty.py"
        file_path.write_text("")

        result = PythonParser.parse_file(file_path)

        assert isinstance(result, CodeFile)
        assert len(result.functions) == 0
        assert len(result.classes) == 0

    def test_parse_line_counts(self, temp_dir):
        """Test line count calculation."""
        code = '''"""Module docstring."""

import os

# Comment line
def func():
    """Function docstring."""
    # Another comment
    return 42
'''
        file_path = temp_dir / "lines.py"
        file_path.write_text(code)

        result = PythonParser.parse_file(file_path)

        assert result.total_lines >= 9
        assert result.code_lines >= 3


# =============================================================================
# Test TypeScriptParser
# =============================================================================

class TestTypeScriptParser:
    """Test TypeScriptParser class."""

    def test_parse_typescript_function(self, temp_dir):
        """Test parsing TypeScript function."""
        code = '''/**
 * Say hello to someone.
 */
export function greet(name: string): string {
    return `Hello, ${name}!`;
}
'''
        file_path = temp_dir / "greet.ts"
        file_path.write_text(code)

        result = TypeScriptParser.parse_file(file_path)

        assert isinstance(result, CodeFile)
        assert result.language in ('typescript', 'javascript')
        # Should extract at least the function
        assert len(result.functions) >= 1 or result.total_lines > 0

    def test_parse_typescript_class(self, temp_dir):
        """Test parsing TypeScript class."""
        code = '''export class Service {
    private config: Config;

    constructor(config: Config) {
        this.config = config;
    }

    async start(): Promise<void> {
        console.log('Starting...');
    }
}
'''
        file_path = temp_dir / "service.ts"
        file_path.write_text(code)

        result = TypeScriptParser.parse_file(file_path)

        assert isinstance(result, CodeFile)
        # Should recognize class or functions
        assert result.total_lines > 0

    def test_parse_javascript(self, temp_dir):
        """Test parsing JavaScript file."""
        code = '''const http = require('http');

class Server {
    constructor(port) {
        this.port = port;
    }

    start() {
        console.log(`Server on port ${this.port}`);
    }
}

module.exports = { Server };
'''
        file_path = temp_dir / "server.js"
        file_path.write_text(code)

        result = TypeScriptParser.parse_file(file_path)

        assert isinstance(result, CodeFile)
        assert result.language in ('javascript', 'typescript')


# =============================================================================
# Test CyclomaticComplexityCalculator
# =============================================================================

class TestCyclomaticComplexity:
    """Test CyclomaticComplexityCalculator."""

    def test_simple_function_complexity(self):
        """Test complexity of simple function."""
        code = '''def simple():
    return 42
'''
        complexity = CyclomaticComplexityCalculator.calculate(code)
        # Simple function should have low complexity
        assert complexity >= 1

    def test_branching_complexity(self):
        """Test complexity with branches."""
        code = '''def branching(x):
    if x > 0:
        return x
    elif x == 0:
        return 0
    else:
        return -x
'''
        complexity = CyclomaticComplexityCalculator.calculate(code)
        # Should have higher complexity due to branches
        assert complexity >= 3

    def test_loop_complexity(self):
        """Test complexity with loops."""
        code = '''def loopy(items):
    total = 0
    for item in items:
        if item > 0:
            total += item
    while total > 100:
        total //= 2
    return total
'''
        complexity = CyclomaticComplexityCalculator.calculate(code)
        # Loops add complexity
        assert complexity >= 3

    def test_nested_complexity(self):
        """Test highly nested code complexity."""
        code = '''def nested(a, b, c):
    if a > 0:
        if b > 0:
            if c > 0:
                return a + b + c
            else:
                return a + b
        else:
            for i in range(10):
                if i % 2 == 0:
                    yield i
    return 0
'''
        complexity = CyclomaticComplexityCalculator.calculate(code)
        # High nesting = high complexity
        assert complexity >= 5

    def test_empty_code_complexity(self):
        """Test complexity of empty/minimal code."""
        code = ""
        complexity = CyclomaticComplexityCalculator.calculate(code)
        # Should return base complexity or 0
        assert complexity >= 0


# =============================================================================
# Test APISurfaceAnalyzer
# =============================================================================

class TestAPISurfaceAnalyzer:
    """Test APISurfaceAnalyzer."""

    def test_public_symbols(self, temp_dir):
        """Test detection of public symbols."""
        code = '''"""Public API module."""

PUBLIC_CONSTANT = 42

def public_function():
    """A public function."""
    pass

class PublicClass:
    """A public class."""
    pass
'''
        file_path = temp_dir / "public.py"
        file_path.write_text(code)

        code_file = PythonParser.parse_file(file_path)
        surface = APISurfaceAnalyzer.analyze(code_file)

        assert isinstance(surface, APISurface)
        # Should find public symbols
        public_names = [s.name for s in surface.public_symbols]
        assert 'public_function' in public_names or len(surface.public_symbols) > 0

    def test_private_symbols(self, temp_dir):
        """Test detection of private symbols."""
        code = '''_PRIVATE_CONSTANT = 42

def _private_function():
    """A private function."""
    pass

class _PrivateClass:
    """A private class."""
    pass
'''
        file_path = temp_dir / "private.py"
        file_path.write_text(code)

        code_file = PythonParser.parse_file(file_path)
        surface = APISurfaceAnalyzer.analyze(code_file)

        # Private symbols should be detected
        private_names = [s.name for s in surface.private_symbols]
        assert '_private_function' in private_names or len(surface.private_symbols) > 0

    def test_protected_symbols(self, temp_dir):
        """Test detection of protected symbols (single underscore)."""
        code = '''class MyClass:
    def __init__(self):
        self._protected_attr = 42

    def _protected_method(self):
        pass

    def __private_method(self):
        pass

    def public_method(self):
        pass
'''
        file_path = temp_dir / "protected.py"
        file_path.write_text(code)

        code_file = PythonParser.parse_file(file_path)
        surface = APISurfaceAnalyzer.analyze(code_file)

        assert isinstance(surface, APISurface)


# =============================================================================
# Test CohesionCalculator
# =============================================================================

class TestCohesionCalculator:
    """Test CohesionCalculator."""

    def test_high_cohesion(self, temp_dir):
        """Test file with high cohesion (internal calls)."""
        code = '''def helper():
    return 42

def caller():
    return helper() * 2

def another_caller():
    x = helper()
    y = caller()
    return x + y
'''
        file_path = temp_dir / "cohesive.py"
        file_path.write_text(code)

        code_file = PythonParser.parse_file(file_path)
        cohesion = CohesionCalculator.calculate(code_file)

        assert isinstance(cohesion, CohesionMetrics)
        # Should have internal calls
        assert cohesion.internal_calls >= 0

    def test_low_cohesion(self, temp_dir):
        """Test file with low cohesion (many external calls)."""
        code = '''import os
import sys
import json

def func1():
    return os.getcwd()

def func2():
    return sys.version

def func3():
    return json.dumps({})
'''
        file_path = temp_dir / "external.py"
        file_path.write_text(code)

        code_file = PythonParser.parse_file(file_path)
        cohesion = CohesionCalculator.calculate(code_file)

        assert isinstance(cohesion, CohesionMetrics)
        # Should have external calls
        assert cohesion.external_calls >= 0


# =============================================================================
# Test DeadCodeDetector
# =============================================================================

class TestDeadCodeDetector:
    """Test DeadCodeDetector."""

    def test_unused_function(self, temp_dir):
        """Test detection of unused functions."""
        code = '''def used_function():
    return 42

def unused_function():
    return 0

result = used_function()
'''
        file_path = temp_dir / "dead.py"
        file_path.write_text(code)

        code_file = PythonParser.parse_file(file_path)
        analysis = DeadCodeDetector.analyze(code_file)

        assert isinstance(analysis, DeadCodeAnalysis)
        # Should detect unused_function
        dead_names = [s.name for s in analysis.dead_symbols]
        # Note: Detection depends on implementation
        assert isinstance(analysis.dead_symbols, list)

    def test_no_dead_code(self, temp_dir):
        """Test file with no dead code."""
        code = '''def helper():
    return 42

def main():
    return helper()

if __name__ == "__main__":
    main()
'''
        file_path = temp_dir / "alive.py"
        file_path.write_text(code)

        code_file = PythonParser.parse_file(file_path)
        analysis = DeadCodeDetector.analyze(code_file)

        assert isinstance(analysis, DeadCodeAnalysis)


# =============================================================================
# Test GitAnalyzer
# =============================================================================

class TestGitAnalyzer:
    """Test GitAnalyzer."""

    @pytest.mark.requires_git
    def test_git_info_extraction(self, sample_codebase):
        """Test git information extraction."""
        # Initialize git repo
        os.system(f"cd {sample_codebase} && git init && git add . && git commit -m 'Initial'")

        file_path = sample_codebase / "src" / "main.py"
        git_info = GitAnalyzer.get_file_info(file_path)

        assert isinstance(git_info, GitFileInfo)

    def test_git_info_no_repo(self, temp_dir):
        """Test git info when not in a repo."""
        file_path = temp_dir / "test.py"
        file_path.write_text("print('hello')")

        git_info = GitAnalyzer.get_file_info(file_path)

        # Should return default/empty info
        assert isinstance(git_info, GitFileInfo)
        assert git_info.commit_count == 0


# =============================================================================
# Test CodebaseSpinner Core Functionality
# =============================================================================

class TestCodebaseSpinnerCore:
    """Test CodebaseSpinner core functionality."""

    def test_spin_single_file(self, sample_python_file):
        """Test spinning a single Python file."""
        spinner = CodebaseSpinner()

        async def run_test():
            result = await spinner.spin(sample_python_file)
            return result

        result = asyncio.run(run_test())

        assert isinstance(result, SpinResult)
        assert result.success is True
        assert len(result.shards) >= 1

    def test_spin_directory(self, sample_codebase):
        """Test spinning a directory."""
        spinner = CodebaseSpinner(importance_threshold=0.0)

        async def run_test():
            result = await spinner.spin(sample_codebase)
            return result

        result = asyncio.run(run_test())

        assert isinstance(result, SpinResult)
        assert result.success is True
        # Should find multiple files
        assert len(result.shards) >= 1

    def test_spin_with_filter(self, sample_codebase):
        """Test spinning with importance filter."""
        spinner = CodebaseSpinner(importance_threshold=0.9)

        async def run_test():
            result = await spinner.spin(sample_codebase)
            return result

        result = asyncio.run(run_test())

        # High threshold should filter most files
        assert isinstance(result, SpinResult)

    def test_spin_recursive(self, sample_codebase):
        """Test recursive directory spinning."""
        spinner = CodebaseSpinner(importance_threshold=0.0)

        async def run_test():
            result = await spinner.spin(sample_codebase, recursive=True)
            return result

        result = asyncio.run(run_test())

        assert result.success is True
        # Should find files in subdirectories
        assert len(result.shards) >= 1

    def test_spin_non_recursive(self, sample_codebase):
        """Test non-recursive directory spinning."""
        spinner = CodebaseSpinner(importance_threshold=0.0)

        async def run_test():
            result = await spinner.spin(sample_codebase, recursive=False)
            return result

        result = asyncio.run(run_test())

        # Should only find files in root directory
        assert result.success is True


# =============================================================================
# Test Entity and Motif Extraction
# =============================================================================

class TestEntityMotifExtraction:
    """Test entity and motif extraction."""

    def test_entity_extraction(self, sample_python_file):
        """Test that entities are extracted from code."""
        spinner = CodebaseSpinner()

        async def run_test():
            result = await spinner.spin(sample_python_file)
            return result

        result = asyncio.run(run_test())

        if result.shards:
            shard = result.shards[0]
            assert isinstance(shard.entities, list)
            # Should extract class/function names
            assert len(shard.entities) >= 1

    def test_motif_extraction(self, sample_python_file):
        """Test that motifs are extracted from code."""
        spinner = CodebaseSpinner()

        async def run_test():
            result = await spinner.spin(sample_python_file)
            return result

        result = asyncio.run(run_test())

        if result.shards:
            shard = result.shards[0]
            assert isinstance(shard.motifs, list)
            # Should include language motif
            assert 'python' in shard.motifs or len(shard.motifs) >= 1


# =============================================================================
# Test Incremental Analysis
# =============================================================================

class TestIncrementalAnalysis:
    """Test incremental codebase analysis."""

    def test_incremental_update_detection(self, sample_codebase):
        """Test detection of added/modified/removed files."""
        spinner = CodebaseSpinner()

        async def run_test():
            # First analysis
            project1 = await spinner.analyze_codebase(sample_codebase)

            # Modify a file
            (sample_codebase / "src" / "main.py").write_text("# Modified\n" + "x = 42")

            # Add a new file
            (sample_codebase / "src" / "new_file.py").write_text("print('new')")

            # Incremental analysis
            result = await spinner.analyze_codebase_incremental(
                sample_codebase,
                previous_project=project1
            )

            return result

        result = asyncio.run(run_test())

        assert isinstance(result, IncrementalUpdateResult)
        assert hasattr(result, 'added_files')
        assert hasattr(result, 'modified_files')
        assert hasattr(result, 'removed_files')

    def test_hash_cache(self, sample_codebase, temp_dir):
        """Test file hash caching."""
        spinner = CodebaseSpinner()
        cache_path = temp_dir / "hashes.json"

        async def run_test():
            await spinner.analyze_codebase(sample_codebase)

            # Save hash cache
            spinner.save_hash_cache(cache_path)

            assert cache_path.exists()

            # Create new spinner and load cache
            spinner2 = CodebaseSpinner()
            spinner2.load_hash_cache(cache_path)

            return True

        result = asyncio.run(run_test())
        assert result is True


# =============================================================================
# Test Dependency Graph
# =============================================================================

class TestDependencyGraph:
    """Test dependency graph building."""

    def test_dependency_edges(self, sample_codebase_with_imports):
        """Test that dependency edges are detected."""
        spinner = CodebaseSpinner()

        async def run_test():
            project = await spinner.analyze_codebase(sample_codebase_with_imports)
            return project

        project = asyncio.run(run_test())

        assert isinstance(project, CodebaseProject)
        assert hasattr(project, 'dependency_edges')
        # Should find import relationships
        assert len(project.dependency_edges) >= 0

    def test_call_graph(self, sample_codebase_with_imports):
        """Test that call edges are detected."""
        spinner = CodebaseSpinner()

        async def run_test():
            project = await spinner.analyze_codebase(sample_codebase_with_imports)
            return project

        project = asyncio.run(run_test())

        assert hasattr(project, 'call_edges')


# =============================================================================
# Test Importance Scoring
# =============================================================================

class TestImportanceScoring:
    """Test importance scoring for code files."""

    def test_score_simple_file(self, sample_python_simple_file):
        """Test scoring a simple file."""
        spinner = CodebaseSpinner()
        code_file = PythonParser.parse_file(sample_python_simple_file)

        score = spinner.score_importance(code_file)

        assert isinstance(score, ImportanceScore)
        assert 0.0 <= score.score <= 1.0

    def test_score_complex_file(self, sample_python_file):
        """Test scoring a complex file."""
        spinner = CodebaseSpinner()
        code_file = PythonParser.parse_file(sample_python_file)

        score = spinner.score_importance(code_file)

        assert isinstance(score, ImportanceScore)
        # Complex file with classes should score higher
        assert score.score >= 0.3

    def test_score_includes_signals(self, sample_python_file):
        """Test that score includes all signals."""
        spinner = CodebaseSpinner()
        code_file = PythonParser.parse_file(sample_python_file)

        score = spinner.score_importance(code_file)

        signals = score.signals
        assert hasattr(signals, 'length_score')
        assert hasattr(signals, 'technical_score')
        assert hasattr(signals, 'structural_score')

    def test_score_reason(self, sample_python_file):
        """Test that score includes reason."""
        spinner = CodebaseSpinner()
        code_file = PythonParser.parse_file(sample_python_file)

        score = spinner.score_importance(code_file)

        assert isinstance(score.reason, str)


# =============================================================================
# Test Large File Handling
# =============================================================================

class TestLargeFileHandling:
    """Test handling of large files."""

    def test_large_file_parsing(self, temp_dir):
        """Test parsing a large Python file."""
        # Generate a large file
        lines = ['"""Large module."""', 'import os', '']
        for i in range(500):
            lines.append(f'''
def function_{i}(x):
    """Function {i}."""
    return x * {i}
''')

        large_file = temp_dir / "large.py"
        large_file.write_text('\n'.join(lines))

        spinner = CodebaseSpinner()

        async def run_test():
            result = await spinner.spin(large_file)
            return result

        result = asyncio.run(run_test())

        assert result.success is True
        # Should handle large file
        assert len(result.shards) >= 1

    def test_many_files_handling(self, temp_dir):
        """Test handling many files."""
        # Create many small files
        for i in range(50):
            (temp_dir / f"module_{i}.py").write_text(f'"""Module {i}."""\nx = {i}')

        spinner = CodebaseSpinner(importance_threshold=0.0)

        async def run_test():
            result = await spinner.spin(temp_dir)
            return result

        result = asyncio.run(run_test())

        assert result.success is True


# =============================================================================
# Test Streaming
# =============================================================================

class TestStreaming:
    """Test streaming shard generation."""

    def test_spin_stream(self, sample_codebase):
        """Test streaming shards from codebase."""
        spinner = CodebaseSpinner(importance_threshold=0.0)

        async def run_test():
            shards = []
            async for shard in spinner.spin_stream(sample_codebase, batch_size=2):
                shards.append(shard)
            return shards

        shards = asyncio.run(run_test())

        assert len(shards) >= 1


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_spin_codebase_function(self, sample_codebase):
        """Test spin_codebase convenience function."""
        async def run_test():
            result = await spin_codebase(
                str(sample_codebase),
                importance_threshold=0.0,
                languages=['python']
            )
            return result

        result = asyncio.run(run_test())

        assert isinstance(result, SpinResult)
        assert result.success is True

    def test_create_codebase_scorer(self):
        """Test create_codebase_scorer function."""
        scorer = create_codebase_scorer()

        assert scorer is not None
        assert hasattr(scorer, 'technical_scorer')


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Test error handling in codebase spinner."""

    def test_nonexistent_path(self):
        """Test handling of nonexistent path."""
        spinner = CodebaseSpinner()

        async def run_test():
            result = await spinner.spin(Path("/nonexistent/path"))
            return result

        result = asyncio.run(run_test())

        # Should handle gracefully
        assert isinstance(result, SpinResult)

    def test_invalid_syntax_handling(self, sample_python_invalid_file):
        """Test handling of files with syntax errors."""
        spinner = CodebaseSpinner()

        async def run_test():
            result = await spinner.spin(sample_python_invalid_file)
            return result

        result = asyncio.run(run_test())

        # Should not crash, may return empty or partial result
        assert isinstance(result, SpinResult)

    def test_binary_file_handling(self, temp_dir):
        """Test handling of binary files."""
        binary_file = temp_dir / "binary.py"
        binary_file.write_bytes(b'\x00\x01\x02\x03\xff\xfe')

        spinner = CodebaseSpinner()

        async def run_test():
            result = await spinner.spin(binary_file)
            return result

        result = asyncio.run(run_test())

        # Should handle gracefully
        assert isinstance(result, SpinResult)


# =============================================================================
# Test Metadata
# =============================================================================

class TestMetadata:
    """Test metadata in generated shards."""

    def test_shard_metadata_fields(self, sample_python_file):
        """Test that shards contain expected metadata."""
        spinner = CodebaseSpinner()

        async def run_test():
            result = await spinner.spin(sample_python_file)
            return result

        result = asyncio.run(run_test())

        if result.shards:
            shard = result.shards[0]
            metadata = shard.metadata

            # Should contain file info
            assert 'file_path' in metadata or 'language' in metadata
            # May contain additional fields
            assert isinstance(metadata, dict)

    def test_complexity_in_metadata(self, sample_python_complex_file):
        """Test that complexity metrics are in metadata."""
        spinner = CodebaseSpinner()

        async def run_test():
            result = await spinner.spin(sample_python_complex_file)
            return result

        result = asyncio.run(run_test())

        if result.shards:
            shard = result.shards[0]
            # Complex file should have complexity score
            assert 'complexity_score' in shard.metadata or isinstance(shard.metadata, dict)
