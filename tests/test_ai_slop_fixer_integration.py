#!/usr/bin/env python3
"""
Integration Tests for AI Slop Fixer
====================================
Comprehensive test suite verifying all components work together.

Run:
    PYTHONPATH=. pytest tests/test_ai_slop_fixer_integration.py -v
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path

from hololoom.agentic.codebase_ingestion import (
    CodebaseIndexer,
    Language,
    PythonParser,
    TypeScriptParser
)
from hololoom.agentic.hallucination_detector import (
    HallucinationDetector,
    PythonReferenceExtractor,
    TypeScriptReferenceExtractor
)
from hololoom.agentic.code_verification import (
    CodeVerifier,
    PythonVerifier,
    TypeScriptVerifier
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_python_codebase():
    """Sample Python codebase for testing."""
    return """
def get_user_by_username(username: str) -> dict:
    '''Fetch user by username.'''
    return {"username": username, "id": 123}

def check_password(password: str, hash: str) -> bool:
    '''Verify password.'''
    return password == hash

class UserSession:
    '''User session.'''
    def __init__(self, user_id: int):
        self.user_id = user_id

    def create_session(self) -> str:
        '''Create session token.'''
        return f"token_{self.user_id}"
"""


@pytest.fixture
def sample_typescript_codebase():
    """Sample TypeScript codebase for testing."""
    return """
export function getUserById(id: number): User {
    return { id, name: "Test" };
}

export class AuthService {
    async login(username: string, password: string): Promise<string> {
        return "token";
    }
}

export interface User {
    id: number;
    name: string;
}
"""


@pytest.fixture
def broken_python_code():
    """AI-generated broken Python code."""
    return """
def authenticate(username, password):
    # Hallucinations - these don't exist
    user = fetch_user_from_db(username)
    if verify_hash(password, user.hash):
        return generate_token(user)
    return None
"""


@pytest.fixture
def broken_typescript_code():
    """AI-generated broken TypeScript code."""
    return """
function processUser(id: number): User {
    const user = await fetchUserData(id);  // Missing async
    return user.data;  // Wrong type
}
"""


@pytest.fixture
async def indexer_with_python_code(sample_python_codebase):
    """Indexer with sample Python code loaded."""
    indexer = CodebaseIndexer()

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(sample_python_codebase)
        temp_file = f.name

    # Ingest
    await indexer.ingest_file(temp_file, Language.PYTHON)

    # Cleanup
    os.unlink(temp_file)

    return indexer


# =============================================================================
# Test: Codebase Ingestion
# =============================================================================

class TestCodebaseIngestion:
    """Test codebase ingestion system."""

    @pytest.mark.asyncio
    async def test_python_parser_extracts_functions(self, sample_python_codebase):
        """Test Python parser extracts functions correctly."""
        parser = PythonParser()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(sample_python_codebase)
            temp_file = f.name

        code_file = parser.parse_file(temp_file, sample_python_codebase)

        # Should find 2 functions
        functions = [e for e in code_file.entities if e.entity_type.value == 'function']
        assert len(functions) == 2

        # Check function names
        func_names = {f.name for f in functions}
        assert 'get_user_by_username' in func_names
        assert 'check_password' in func_names

        # Cleanup
        os.unlink(temp_file)

    @pytest.mark.asyncio
    async def test_python_parser_extracts_classes(self, sample_python_codebase):
        """Test Python parser extracts classes correctly."""
        parser = PythonParser()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(sample_python_codebase)
            temp_file = f.name

        code_file = parser.parse_file(temp_file, sample_python_codebase)

        # Should find 1 class
        classes = [e for e in code_file.entities if e.entity_type.value == 'class']
        assert len(classes) == 1
        assert classes[0].name == 'UserSession'

        # Should find 1 method
        methods = [e for e in code_file.entities if e.entity_type.value == 'method']
        assert len(methods) == 1
        assert methods[0].name == 'create_session'

        # Cleanup
        os.unlink(temp_file)

    @pytest.mark.asyncio
    async def test_indexer_builds_knowledge_graph(self, sample_python_codebase):
        """Test indexer builds knowledge graph correctly."""
        indexer = CodebaseIndexer()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(sample_python_codebase)
            temp_file = f.name

        await indexer.ingest_file(temp_file, Language.PYTHON)

        # Check statistics
        stats = indexer.get_statistics()
        assert stats['total_entities'] > 0
        assert stats['total_files'] == 1

        # Cleanup
        os.unlink(temp_file)

    @pytest.mark.asyncio
    async def test_indexer_search_finds_entities(self, indexer_with_python_code):
        """Test search finds indexed entities."""
        results = await indexer_with_python_code.search_entity('get_user_by_username')

        assert len(results) > 0
        assert results[0]['name'] == 'get_user_by_username'
        assert results[0]['type'] == 'function'

    @pytest.mark.asyncio
    async def test_indexer_fuzzy_search(self, indexer_with_python_code):
        """Test fuzzy search works."""
        results = await indexer_with_python_code.search_entity('get_user', fuzzy=True)

        assert len(results) > 0
        assert any('get_user' in r['name'] for r in results)


# =============================================================================
# Test: Hallucination Detection
# =============================================================================

class TestHallucinationDetection:
    """Test hallucination detection system."""

    @pytest.mark.asyncio
    async def test_python_reference_extraction(self, broken_python_code):
        """Test Python reference extractor finds references."""
        extractor = PythonReferenceExtractor()
        refs = extractor.extract_references(broken_python_code)

        # Should find function calls
        func_calls = [r for r in refs if r.reference_type == 'function_call']
        assert len(func_calls) > 0

        # Check specific hallucinated functions
        func_names = {r.name for r in func_calls}
        assert 'fetch_user_from_db' in func_names
        assert 'verify_hash' in func_names
        assert 'generate_token' in func_names

    @pytest.mark.asyncio
    async def test_hallucination_detector_finds_fake_functions(
        self,
        indexer_with_python_code,
        broken_python_code
    ):
        """Test detector identifies hallucinated functions."""
        detector = HallucinationDetector(indexer_with_python_code)

        hallucinations = await detector.detect(
            broken_python_code,
            Language.PYTHON,
            'test.py'
        )

        # Should find hallucinations
        assert len(hallucinations) > 0

        # Check hallucination types
        for h in hallucinations:
            assert h.reference in ['fetch_user_from_db', 'verify_hash', 'generate_token']
            assert h.confidence > 0.5

    @pytest.mark.asyncio
    async def test_hallucination_detector_provides_suggestions(
        self,
        indexer_with_python_code,
        broken_python_code
    ):
        """Test detector provides similar entity suggestions."""
        detector = HallucinationDetector(indexer_with_python_code)

        result = await detector.detect_and_explain(
            broken_python_code,
            Language.PYTHON,
            'test.py'
        )

        # Should have suggestions
        assert 'hallucinations' in result
        if len(result['hallucinations']) > 0:
            halluc = result['hallucinations'][0]
            # May have suggestions
            assert 'suggestions' in halluc

    @pytest.mark.asyncio
    async def test_hallucination_detector_ignores_builtins(
        self,
        indexer_with_python_code
    ):
        """Test detector doesn't flag built-in functions."""
        detector = HallucinationDetector(indexer_with_python_code)

        code_with_builtins = """
def test():
    x = len([1, 2, 3])
    print(x)
    return str(x)
"""

        hallucinations = await detector.detect(
            code_with_builtins,
            Language.PYTHON,
            'test.py'
        )

        # Should NOT flag len, print, str
        halluc_names = [h.reference for h in hallucinations]
        assert 'len' not in halluc_names
        assert 'print' not in halluc_names
        assert 'str' not in halluc_names


# =============================================================================
# Test: Code Verification
# =============================================================================

class TestCodeVerification:
    """Test code verification system."""

    @pytest.mark.asyncio
    async def test_python_syntax_check_detects_errors(self):
        """Test Python syntax checker finds syntax errors."""
        verifier = PythonVerifier()

        broken_code = """
def foo():
    return bar(  # Missing paren
"""

        result = await verifier.verify_syntax(broken_code)

        assert result.success is False
        assert len(result.errors) > 0
        assert result.errors[0].severity.value == 'error'

    @pytest.mark.asyncio
    async def test_python_syntax_check_passes_valid_code(self):
        """Test Python syntax checker passes valid code."""
        verifier = PythonVerifier()

        valid_code = """
def foo():
    return bar()
"""

        result = await verifier.verify_syntax(valid_code)

        assert result.success is True
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_code_verifier_comprehensive(self):
        """Test comprehensive verification."""
        verifier = CodeVerifier()

        broken_code = """
def foo():
    return bar(  # Syntax error
"""

        result = await verifier.verify_comprehensive(broken_code, 'python', 'test.py')

        assert 'success' in result
        assert 'errors' in result
        assert 'summary' in result

        if result['errors']:
            error = result['errors'][0]
            assert 'line' in error
            assert 'message' in error


# =============================================================================
# Test: Integration
# =============================================================================

class TestIntegration:
    """Test full integration of all systems."""

    @pytest.mark.asyncio
    async def test_complete_workflow(
        self,
        sample_python_codebase,
        broken_python_code
    ):
        """Test complete workflow: index → detect → verify."""
        # Step 1: Index codebase
        indexer = CodebaseIndexer()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(sample_python_codebase)
            temp_file = f.name

        await indexer.ingest_file(temp_file, Language.PYTHON)
        os.unlink(temp_file)

        # Step 2: Detect hallucinations
        detector = HallucinationDetector(indexer)
        halluc_result = await detector.detect_and_explain(
            broken_python_code,
            Language.PYTHON,
            'test.py'
        )

        # Step 3: Verify code
        verifier = CodeVerifier()
        verify_result = await verifier.verify_comprehensive(
            broken_python_code,
            'python',
            'test.py'
        )

        # Assertions
        assert halluc_result['summary']['total_hallucinations'] > 0
        # Verification may or may not have syntax errors (depends on code)

    @pytest.mark.asyncio
    async def test_memory_shard_conversion(self, indexer_with_python_code):
        """Test conversion to memory shards."""
        shards = indexer_with_python_code.to_memory_shards()

        assert len(shards) > 0

        for shard in shards:
            assert hasattr(shard, 'id')
            assert hasattr(shard, 'text')
            assert hasattr(shard, 'entities')
            assert hasattr(shard, 'metadata')


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_indexing_performance(self, sample_python_codebase):
        """Test indexing completes in reasonable time."""
        import time

        indexer = CodebaseIndexer()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(sample_python_codebase)
            temp_file = f.name

        start = time.time()
        await indexer.ingest_file(temp_file, Language.PYTHON)
        duration = time.time() - start

        os.unlink(temp_file)

        # Should complete in <1 second for small file
        assert duration < 1.0

    @pytest.mark.asyncio
    async def test_hallucination_detection_performance(
        self,
        indexer_with_python_code,
        broken_python_code
    ):
        """Test hallucination detection completes in reasonable time."""
        import time

        detector = HallucinationDetector(indexer_with_python_code)

        start = time.time()
        await detector.detect(broken_python_code, Language.PYTHON, 'test.py')
        duration = time.time() - start

        # Should complete in <1 second
        assert duration < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
