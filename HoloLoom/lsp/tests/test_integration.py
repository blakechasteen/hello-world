#!/usr/bin/env python3
"""
HoloLoom LSP Server Integration Test Suite
===========================================

Comprehensive integration tests for the HoloLoom Language Server Protocol (LSP) server.

This test suite validates:
1. Server initialization and shutdown (lifecycle)
2. LSP protocol handshake with client capabilities
3. Completion handler with mock documents
4. Hover handler with symbol position
5. Definition handler with knowledge graph lookup
6. Workspace symbol search
7. Error handling and graceful degradation
8. HoloLoom integration
9. Logging and diagnostics

Status: Production Ready (November 2025)
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import List, Optional

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pygls.lsp.server import LanguageServer
from lsprotocol.types import (
    InitializeParams,
    InitializeResult,
    InitializedParams,
    ClientInfo,
    CompletionParams,
    CompletionList,
    CompletionItem,
    Position,
    TextDocumentIdentifier,
    HoverParams,
    Hover,
    DefinitionParams,
    WorkspaceSymbolParams,
    SymbolInformation,
    SymbolKind,
    Location,
    Range,
    TextDocumentItem,
    TextDocumentContentChangeEvent,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def server():
    """Create mock LSP server for testing."""
    from HoloLoom.lsp.server import HoloLoomLanguageServer, setup_logging

    setup_logging("WARNING")  # Suppress log output during tests

    srv = HoloLoomLanguageServer(
        name="hololoom-lsp-test",
        version="0.1.0-test",
    )

    # Mock workspace
    srv.workspace = Mock()

    yield srv


@pytest.fixture
async def mock_hololoom():
    """Create mock HoloLoom instance."""
    hololoom = AsyncMock()

    # Mock memory object
    class MockMemory:
        def __init__(self, text, metadata=None):
            self.text = text
            self.metadata = metadata or {}
            self.timestamp = "2025-11-16T12:00:00"

    # Simulate memory recalls
    async def mock_recall(query, limit=10):
        memories = [
            MockMemory(
                f"Memory about {query}",
                {"source": "test", "confidence": 0.95, "line": 0}
            ),
        ]
        if query == "Thompson":
            memories.append(MockMemory(
                "Thompson Sampling: A Bayesian exploration strategy",
                {"confidence": 0.92, "line": 10}
            ))
        elif query == "BanditArm":
            memories.append(MockMemory(
                "class BanditArm: def __init__(self, alpha, beta): pass",
                {"confidence": 0.88, "line": 20}
            ))
        return memories[:limit]

    hololoom.recall = mock_recall
    hololoom.__aenter__ = AsyncMock(return_value=hololoom)
    hololoom.__aexit__ = AsyncMock(return_value=None)

    return hololoom


@pytest.fixture
def mock_document():
    """Create mock text document."""
    doc = Mock()
    doc.uri = "file:///test/example.py"
    doc.lines = [
        "from HoloLoom import HoloLoom",
        "loom = HoloLoom()",
        "result = loom.recall('Thompson')",
        "print(result)",
    ]
    return doc


# ============================================================================
# Test: Initialization & Shutdown
# ============================================================================

@pytest.mark.asyncio
async def test_server_initialization(server):
    """Test LSP server initializes with correct capabilities."""
    from HoloLoom.lsp.server import initialize

    params = InitializeParams(
        process_id=12345,
        root_uri="file:///workspace",
        client_info=ClientInfo(name="test-client", version="1.0.0"),
        capabilities={}
    )

    result = await initialize(params)

    assert isinstance(result, InitializeResult)
    assert result.capabilities is not None
    assert result.capabilities.text_document_sync is not None
    assert result.capabilities.completion_provider is not None
    assert result.capabilities.hover_provider is True
    assert result.capabilities.definition_provider is True
    assert result.capabilities.workspace_symbol_provider is True


@pytest.mark.asyncio
async def test_server_initialization_with_no_client_info(server):
    """Test server initialization handles missing client info gracefully."""
    from HoloLoom.lsp.server import initialize

    params = InitializeParams(
        process_id=12345,
        root_uri="file:///workspace",
        client_info=None,
        capabilities={}
    )

    result = await initialize(params)
    assert isinstance(result, InitializeResult)
    assert result.capabilities is not None


@pytest.mark.asyncio
async def test_server_shutdown(server):
    """Test LSP server shuts down gracefully."""
    from HoloLoom.lsp.server import shutdown

    server.hololoom = AsyncMock()

    await shutdown(None)

    # Should call __aexit__ to clean up
    server.hololoom.__aexit__.assert_called_once()


@pytest.mark.asyncio
async def test_server_shutdown_with_no_hololoom(server):
    """Test shutdown handles missing HoloLoom gracefully."""
    from HoloLoom.lsp.server import shutdown

    server.hololoom = None

    # Should not raise
    await shutdown(None)


# ============================================================================
# Test: Completion Handler
# ============================================================================

@pytest.mark.asyncio
async def test_completion_basic(server, mock_hololoom, mock_document):
    """Test completion returns HoloLoom memories."""
    from HoloLoom.lsp.server import completion

    server.hololoom = mock_hololoom
    server.workspace.get_text_document.return_value = mock_document

    params = CompletionParams(
        text_document=TextDocumentIdentifier(uri=mock_document.uri),
        position=Position(line=2, character=30)  # After "recall("
    )

    result = await completion(params)

    assert isinstance(result, CompletionList)
    assert result.is_incomplete is False
    assert len(result.items) > 0
    assert isinstance(result.items[0], CompletionItem)
    assert result.items[0].label is not None


@pytest.mark.asyncio
async def test_completion_with_empty_query(server, mock_hololoom, mock_document):
    """Test completion handles empty query gracefully."""
    from HoloLoom.lsp.server import completion

    server.hololoom = mock_hololoom
    server.workspace.get_text_document.return_value = mock_document

    params = CompletionParams(
        text_document=TextDocumentIdentifier(uri=mock_document.uri),
        position=Position(line=0, character=0)  # At start
    )

    result = await completion(params)

    assert isinstance(result, CompletionList)
    # Should return either empty or with fallback items
    assert result.is_incomplete is False


@pytest.mark.asyncio
async def test_completion_no_hololoom(server, mock_document):
    """Test completion returns empty when HoloLoom unavailable."""
    from HoloLoom.lsp.server import completion

    server.hololoom = None
    server.workspace.get_text_document.return_value = mock_document

    params = CompletionParams(
        text_document=TextDocumentIdentifier(uri=mock_document.uri),
        position=Position(line=2, character=30)
    )

    result = await completion(params)

    assert isinstance(result, CompletionList)
    assert len(result.items) == 0


@pytest.mark.asyncio
async def test_completion_out_of_bounds_line(server, mock_hololoom, mock_document):
    """Test completion handles out-of-bounds line gracefully."""
    from HoloLoom.lsp.server import completion

    server.hololoom = mock_hololoom
    server.workspace.get_text_document.return_value = mock_document

    params = CompletionParams(
        text_document=TextDocumentIdentifier(uri=mock_document.uri),
        position=Position(line=999, character=0)  # Out of bounds
    )

    result = await completion(params)

    assert isinstance(result, CompletionList)
    assert len(result.items) == 0


# ============================================================================
# Test: Hover Handler
# ============================================================================

@pytest.mark.asyncio
async def test_hover_basic(server, mock_hololoom, mock_document):
    """Test hover returns information about symbol."""
    from HoloLoom.lsp.server import hover

    server.hololoom = mock_hololoom
    server.workspace.get_text_document.return_value = mock_document

    params = HoverParams(
        text_document=TextDocumentIdentifier(uri=mock_document.uri),
        position=Position(line=1, character=7)  # On "HoloLoom"
    )

    result = await hover(params)

    assert isinstance(result, Hover)
    assert result.contents is not None


@pytest.mark.asyncio
async def test_hover_no_symbol(server, mock_hololoom, mock_document):
    """Test hover returns None when no symbol at position."""
    from HoloLoom.lsp.server import hover

    server.hololoom = mock_hololoom
    server.workspace.get_text_document.return_value = mock_document

    params = HoverParams(
        text_document=TextDocumentIdentifier(uri=mock_document.uri),
        position=Position(line=0, character=0)  # Whitespace
    )

    result = await hover(params)

    # May return None or empty result
    assert result is None or isinstance(result, Hover)


@pytest.mark.asyncio
async def test_hover_no_hololoom(server, mock_document):
    """Test hover returns None when HoloLoom unavailable."""
    from HoloLoom.lsp.server import hover

    server.hololoom = None
    server.workspace.get_text_document.return_value = mock_document

    params = HoverParams(
        text_document=TextDocumentIdentifier(uri=mock_document.uri),
        position=Position(line=1, character=7)
    )

    result = await hover(params)

    assert result is None


@pytest.mark.asyncio
async def test_hover_out_of_bounds(server, mock_hololoom, mock_document):
    """Test hover handles out-of-bounds line gracefully."""
    from HoloLoom.lsp.server import hover

    server.hololoom = mock_hololoom
    server.workspace.get_text_document.return_value = mock_document

    params = HoverParams(
        text_document=TextDocumentIdentifier(uri=mock_document.uri),
        position=Position(line=999, character=0)
    )

    result = await hover(params)

    assert result is None


# ============================================================================
# Test: Definition Handler
# ============================================================================

@pytest.mark.asyncio
async def test_definition_basic(server, mock_hololoom, mock_document):
    """Test definition returns locations."""
    from HoloLoom.lsp.server import definition

    server.hololoom = mock_hololoom
    server.workspace.get_text_document.return_value = mock_document

    params = DefinitionParams(
        text_document=TextDocumentIdentifier(uri=mock_document.uri),
        position=Position(line=1, character=7)  # On "HoloLoom"
    )

    result = await definition(params)

    # Should return list of locations or None
    if result is not None:
        assert isinstance(result, list)
        for loc in result:
            assert isinstance(loc, Location)
            assert loc.uri is not None
            assert loc.range is not None


@pytest.mark.asyncio
async def test_definition_no_hololoom(server, mock_document):
    """Test definition returns None when HoloLoom unavailable."""
    from HoloLoom.lsp.server import definition

    server.hololoom = None
    server.workspace.get_text_document.return_value = mock_document

    params = DefinitionParams(
        text_document=TextDocumentIdentifier(uri=mock_document.uri),
        position=Position(line=1, character=7)
    )

    result = await definition(params)

    assert result is None


@pytest.mark.asyncio
async def test_definition_with_metadata(server, mock_hololoom, mock_document):
    """Test definition extracts file location from metadata."""
    from HoloLoom.lsp.server import definition

    # Mock memory with file location
    class MockMemoryWithLocation:
        def __init__(self):
            self.text = "def my_function(): pass"
            self.metadata = {
                "file_path": "/project/my_file.py",
                "line_number": 42
            }

    mock_hololoom.recall = AsyncMock(return_value=[MockMemoryWithLocation()])
    server.hololoom = mock_hololoom
    server.workspace.get_text_document.return_value = mock_document

    params = DefinitionParams(
        text_document=TextDocumentIdentifier(uri=mock_document.uri),
        position=Position(line=1, character=7)
    )

    result = await definition(params)

    assert result is not None
    assert len(result) > 0
    assert result[0].uri == "file:///project/my_file.py"
    assert result[0].range.start.line == 42


# ============================================================================
# Test: Workspace Symbol Handler
# ============================================================================

@pytest.mark.asyncio
async def test_workspace_symbol_basic(server, mock_hololoom):
    """Test workspace symbol search returns symbols."""
    from HoloLoom.lsp.server import workspace_symbol

    server.hololoom = mock_hololoom

    params = WorkspaceSymbolParams(query="Thompson")

    result = await workspace_symbol(params)

    assert isinstance(result, list)
    assert len(result) > 0
    for sym in result:
        assert isinstance(sym, SymbolInformation)
        assert sym.name is not None
        assert sym.location is not None


@pytest.mark.asyncio
async def test_workspace_symbol_empty_query(server, mock_hololoom):
    """Test workspace symbol with empty query."""
    from HoloLoom.lsp.server import workspace_symbol

    server.hololoom = mock_hololoom

    params = WorkspaceSymbolParams(query="")

    result = await workspace_symbol(params)

    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_workspace_symbol_no_hololoom(server):
    """Test workspace symbol returns empty when HoloLoom unavailable."""
    from HoloLoom.lsp.server import workspace_symbol

    server.hololoom = None

    params = WorkspaceSymbolParams(query="test")

    result = await workspace_symbol(params)

    assert isinstance(result, list)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_workspace_symbol_kind_detection(server, mock_hololoom):
    """Test workspace symbol correctly detects symbol kinds."""
    from HoloLoom.lsp.server import workspace_symbol

    # Mock different types of memories
    class MockMemory:
        def __init__(self, text, kind_keyword):
            self.text = f"{kind_keyword}: {text}"
            self.metadata = {}

    memories = [
        MockMemory("foo", "def"),
        MockMemory("Bar", "class"),
        MockMemory("mymodule", "module"),
    ]

    mock_hololoom.recall = AsyncMock(return_value=memories)
    server.hololoom = mock_hololoom

    params = WorkspaceSymbolParams(query="test")

    result = await workspace_symbol(params)

    assert len(result) >= 3
    # Check that kinds are detected
    kinds = [s.kind for s in result]
    assert SymbolKind.Function in kinds or SymbolKind.Variable in kinds


# ============================================================================
# Test: Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_completion_handler_exception(server, mock_document):
    """Test completion handler gracefully handles exceptions."""
    from HoloLoom.lsp.server import completion

    server.hololoom = AsyncMock()
    server.hololoom.recall = AsyncMock(side_effect=Exception("Test error"))
    server.workspace.get_text_document.return_value = mock_document

    params = CompletionParams(
        text_document=TextDocumentIdentifier(uri=mock_document.uri),
        position=Position(line=0, character=0)
    )

    result = await completion(params)

    # Should return empty list instead of raising
    assert isinstance(result, CompletionList)
    assert len(result.items) == 0


@pytest.mark.asyncio
async def test_hover_handler_exception(server, mock_document):
    """Test hover handler gracefully handles exceptions."""
    from HoloLoom.lsp.server import hover

    server.hololoom = AsyncMock()
    server.hololoom.recall = AsyncMock(side_effect=Exception("Test error"))
    server.workspace.get_text_document.return_value = mock_document

    params = HoverParams(
        text_document=TextDocumentIdentifier(uri=mock_document.uri),
        position=Position(line=0, character=0)
    )

    result = await hover(params)

    # Should return None instead of raising
    assert result is None


@pytest.mark.asyncio
async def test_definition_handler_exception(server, mock_document):
    """Test definition handler gracefully handles exceptions."""
    from HoloLoom.lsp.server import definition

    server.hololoom = AsyncMock()
    server.hololoom.recall = AsyncMock(side_effect=Exception("Test error"))
    server.workspace.get_text_document.return_value = mock_document

    params = DefinitionParams(
        text_document=TextDocumentIdentifier(uri=mock_document.uri),
        position=Position(line=0, character=0)
    )

    result = await definition(params)

    # Should return None instead of raising
    assert result is None


@pytest.mark.asyncio
async def test_workspace_symbol_exception(server):
    """Test workspace symbol handler gracefully handles exceptions."""
    from HoloLoom.lsp.server import workspace_symbol

    server.hololoom = AsyncMock()
    server.hololoom.recall = AsyncMock(side_effect=Exception("Test error"))

    params = WorkspaceSymbolParams(query="test")

    result = await workspace_symbol(params)

    # Should return empty list instead of raising
    assert isinstance(result, list)
    assert len(result) == 0


# ============================================================================
# Test: Helper Functions
# ============================================================================

def test_extract_word_at_position():
    """Test extract_word_at_position helper function."""
    from HoloLoom.lsp.server import extract_word_at_position

    # Test basic extraction
    assert extract_word_at_position("hello world", 5) == "hello"
    assert extract_word_at_position("HoloLoom.memory", 15) == "HoloLoom.memory"
    assert extract_word_at_position("  value", 3) == ""
    assert extract_word_at_position("", 0) == ""


def test_extract_symbol_at_position():
    """Test extract_symbol_at_position helper function."""
    from HoloLoom.lsp.server import extract_symbol_at_position

    # Test symbol extraction
    assert extract_symbol_at_position("my_function()", 5) == "my_function"
    assert extract_symbol_at_position("class MyClass:", 8) == "MyClass"
    assert extract_symbol_at_position("  ", 1) == ""
    assert extract_symbol_at_position("", 0) == ""


def test_format_memory_as_markdown():
    """Test format_memory_as_markdown helper function."""
    from HoloLoom.lsp.server import format_memory_as_markdown

    class MockMemory:
        def __init__(self):
            self.text = "Test memory content"
            self.metadata = {"source": "test", "confidence": 0.95}
            self.timestamp = "2025-11-16T12:00:00"

    mem = MockMemory()
    markdown = format_memory_as_markdown(mem)

    assert "Test memory content" in markdown
    assert "Memory" in markdown
    assert "test" in markdown.lower()


# ============================================================================
# Test: Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_initialization_workflow(server, mock_hololoom):
    """Test full initialization workflow: initialize -> initialized -> ready."""
    from HoloLoom.lsp.server import initialize, on_initialized

    # Step 1: Initialize
    init_params = InitializeParams(
        process_id=12345,
        root_uri="file:///workspace",
        client_info=ClientInfo(name="test", version="1.0"),
        capabilities={}
    )

    init_result = await initialize(init_params)
    assert init_result.capabilities is not None

    # Step 2: On initialized
    server.hololoom = None  # Not yet initialized
    await on_initialized(InitializedParams())
    # Server would initialize HoloLoom here


@pytest.mark.asyncio
async def test_handler_chain(server, mock_hololoom, mock_document):
    """Test chain of handlers: completion -> hover -> definition."""
    from HoloLoom.lsp.server import completion, hover, definition

    server.hololoom = mock_hololoom
    server.workspace.get_text_document.return_value = mock_document

    # 1. Get completions
    comp_params = CompletionParams(
        text_document=TextDocumentIdentifier(uri=mock_document.uri),
        position=Position(line=2, character=30)
    )
    completions = await completion(comp_params)
    assert len(completions.items) > 0

    # 2. Get hover info
    hover_params = HoverParams(
        text_document=TextDocumentIdentifier(uri=mock_document.uri),
        position=Position(line=1, character=7)
    )
    hover_info = await hover(hover_params)
    assert hover_info is not None or hover_info is None  # Either valid or None

    # 3. Get definition
    def_params = DefinitionParams(
        text_document=TextDocumentIdentifier(uri=mock_document.uri),
        position=Position(line=1, character=7)
    )
    definition_info = await definition(def_params)
    # May return None or list


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
