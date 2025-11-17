#!/usr/bin/env python3
"""
HoloLoom LSP Server
===================
Language Server Protocol (LSP) implementation for HoloLoom neural memory system.

This server provides:
- Code completion from HoloLoom knowledge graph
- Hover information with entity context
- Go-to-definition via knowledge graph relationships
- Workspace symbol search (semantic + graph-based)
- Diagnostic reports from alignment framework

The server uses the Language Server Protocol (LSP) to communicate with editors
(VSCode, Neovim, Emacs, Sublime, Vim, etc.) over stdio or TCP.

Architecture:
- server: Main pygls LanguageServer instance
- handlers: Decorated functions for LSP requests
- context: HoloLoom orchestrator and memory backends (lazy-loaded)
- logging: Structured logging with timestamp and level

Usage:
    python -m HoloLoom.lsp.server [--port 8080] [--log-level DEBUG]

Wave 2 Integration (Phase 5 Wave 2 - Nov 2025):
- Real knowledge graph queries (no more placeholder data)
- Entity metadata from workspace indexer
- Graph traversal (DEFINES, USES, IS_A edges)
- Proper LSP protocol compliance
"""

import asyncio
import logging
import sys
import re
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import argparse
from datetime import datetime
from urllib.parse import urlparse, unquote

from pygls.lsp.server import LanguageServer
from lsprotocol.types import (
    InitializeParams,
    InitializeResult,
    ServerCapabilities,
    TextDocumentSyncKind,
    CompletionItem,
    CompletionItemKind,
    CompletionList,
    CompletionParams,
    Hover,
    HoverParams,
    MarkupContent,
    MarkupKind,
    Location,
    DefinitionParams,
    WorkspaceSymbolParams,
    SymbolInformation,
    SymbolKind,
    Range,
    Position,
)

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure structured logging for LSP server.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("hololoom-lsp")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler with timestamp
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# ============================================================================
# SERVER INITIALIZATION
# ============================================================================

class HoloLoomLanguageServer(LanguageServer):
    """Extended LanguageServer with HoloLoom-specific context.

    Attributes:
        hololoom: Optional HoloLoom instance
        config: Server configuration (port, log level, etc.)
    """

    def __init__(self, name: str, version: str, **kwargs):
        super().__init__(name, version, **kwargs)
        self.hololoom = None
        self.config: Dict[str, Any] = {}
        self.logger = logging.getLogger("hololoom-lsp")


# Initialize server with LSP capabilities
server = HoloLoomLanguageServer(
    name="hololoom-lsp",
    version="0.2.0",  # Wave 2: Real KG integration
)

logger = logging.getLogger("hololoom-lsp")


# ============================================================================
# HELPER FUNCTIONS - Knowledge Graph Queries (Wave 2)
# ============================================================================

def _uri_to_file_path(uri: str) -> str:
    """
    Convert LSP file URI to filesystem path.

    Args:
        uri: File URI (e.g., "file:///home/user/project/file.py")

    Returns:
        Filesystem path (e.g., "/home/user/project/file.py")
    """
    parsed = urlparse(uri)
    path = unquote(parsed.path)

    # On Windows, remove leading slash (file:///C:/... -> C:/...)
    if sys.platform == 'win32' and path.startswith('/'):
        path = path[1:]

    return path


def _file_path_to_uri(file_path: str) -> str:
    """
    Convert filesystem path to LSP file URI.

    Args:
        file_path: Filesystem path

    Returns:
        File URI
    """
    # Convert to Path for normalization
    p = Path(file_path)

    # Get absolute path
    if not p.is_absolute():
        p = p.resolve()

    # Convert to URI
    uri = p.as_uri()
    return uri


def _find_entity_by_name(kg, name: str) -> Optional[Tuple[str, Dict]]:
    """
    Find entity node by name in knowledge graph.

    Searches all nodes for exact name match.

    Args:
        kg: NetworkX MultiDiGraph
        name: Entity name to search for

    Returns:
        Tuple of (node_id, node_data) or None if not found

    Example:
        >>> node_id, entity = _find_entity_by_name(kg, "MyClass")
        >>> print(entity['type'])  # 'class'
        >>> print(entity['file_path'])  # 'src/mymodule.py'
    """
    for node_id in kg.nodes():
        node_data = kg.nodes[node_id]
        if node_data.get('name') == name:
            return node_id, node_data

    return None


def _find_entities_fuzzy(kg, query: str, limit: int = 20) -> List[Tuple[str, Dict]]:
    """
    Find entities by fuzzy name matching.

    Args:
        kg: NetworkX MultiDiGraph
        query: Search query
        limit: Maximum results

    Returns:
        List of (node_id, node_data) tuples
    """
    query_lower = query.lower()
    matches = []

    for node_id in kg.nodes():
        node_data = kg.nodes[node_id]
        name = node_data.get('name', '')

        # Skip file nodes (we want code entities)
        if node_data.get('type') == 'file':
            continue

        # Fuzzy match
        if query_lower in name.lower():
            matches.append((node_id, node_data))

            if len(matches) >= limit:
                break

    return matches


def _get_file_entities(kg, file_path: str) -> List[Tuple[str, Dict]]:
    """
    Get all entities defined in a file.

    Traverses DEFINES edges from file node to entity nodes.

    Args:
        kg: NetworkX MultiDiGraph
        file_path: Relative file path (e.g., "src/mymodule.py")

    Returns:
        List of (node_id, entity_data) tuples

    Example:
        >>> entities = _get_file_entities(kg, "src/mymodule.py")
        >>> for node_id, entity in entities:
        ...     print(f"{entity['name']} ({entity['type']})")
    """
    file_id = f"file::{file_path}"
    entities = []

    if not kg.has_node(file_id):
        logger.debug(f"File node not found: {file_id}")
        return []

    # Find all neighbors connected by DEFINES edge
    for neighbor in kg.neighbors(file_id):
        # Get all edges between file and neighbor (MultiDiGraph can have multiple)
        edge_data = kg.get_edge_data(file_id, neighbor)

        if edge_data:
            # Check each edge key (MultiDiGraph stores edges as {key: attrs})
            for key, attrs in edge_data.items():
                if attrs.get('type') == 'DEFINES':
                    entity = kg.nodes[neighbor]
                    entities.append((neighbor, entity))
                    break  # Only need to add entity once

    return entities


def _entity_to_completion_item(entity: Dict, node_id: str, rank: int = 0) -> CompletionItem:
    """
    Convert KG entity to LSP CompletionItem.

    Args:
        entity: Entity node data
        node_id: Entity node ID
        rank: Rank/priority (lower = higher priority)

    Returns:
        CompletionItem for LSP client
    """
    entity_type = entity.get('type', 'entity')

    # Map entity type to CompletionItemKind
    kind_map = {
        'class': CompletionItemKind.Class,
        'function': CompletionItemKind.Function,
        'method': CompletionItemKind.Method,
        'import': CompletionItemKind.Module,
        'variable': CompletionItemKind.Variable,
        'entity': CompletionItemKind.Text,
        'file': CompletionItemKind.File,
    }
    kind = kind_map.get(entity_type, CompletionItemKind.Text)

    # Build detail string
    file_path = entity.get('file_path', 'unknown')
    detail = f"{entity_type} from {file_path}"

    # Build documentation
    doc_parts = []

    if entity.get('docstring'):
        doc_parts.append(entity['docstring'])
        doc_parts.append("")

    doc_parts.append(f"**Type**: {entity_type}")
    doc_parts.append(f"**File**: {file_path}")

    if entity.get('line_number'):
        doc_parts.append(f"**Line**: {entity['line_number']}")

    if entity.get('language'):
        doc_parts.append(f"**Language**: {entity['language']}")

    documentation = "\n".join(doc_parts)

    # Sort text (lower = higher priority)
    # Prefix with rank: "0_name" for high priority, "9_name" for low priority
    name = entity.get('name', node_id.split('::')[-1])
    sort_text = f"{rank}_{name}"

    return CompletionItem(
        label=name,
        kind=kind,
        detail=detail,
        documentation=MarkupContent(
            kind=MarkupKind.Markdown,
            value=documentation
        ),
        insert_text=name,
        sort_text=sort_text
    )


def _entity_to_symbol_info(entity: Dict, node_id: str) -> SymbolInformation:
    """
    Convert KG entity to LSP SymbolInformation.

    Args:
        entity: Entity node data
        node_id: Entity node ID

    Returns:
        SymbolInformation for workspace symbol search
    """
    entity_type = entity.get('type', 'entity')

    # Map entity type to SymbolKind
    kind_map = {
        'class': SymbolKind.Class,
        'function': SymbolKind.Function,
        'method': SymbolKind.Method,
        'import': SymbolKind.Module,
        'variable': SymbolKind.Variable,
        'entity': SymbolKind.Variable,
        'file': SymbolKind.File,
    }
    kind = kind_map.get(entity_type, SymbolKind.Variable)

    # Build location
    file_path = entity.get('file_path')
    line_number = entity.get('line_number', 0)

    if file_path:
        file_uri = _file_path_to_uri(file_path)
    else:
        file_uri = "file:///unknown"

    location = Location(
        uri=file_uri,
        range=Range(
            start=Position(line=int(line_number), character=0),
            end=Position(line=int(line_number), character=100)
        )
    )

    name = entity.get('name', node_id.split('::')[-1])

    return SymbolInformation(
        name=name,
        kind=kind,
        location=location
    )


def _get_related_entities(kg, entity_id: str, max_depth: int = 2) -> List[Tuple[str, Dict]]:
    """
    Get entities related to given entity via graph edges.

    Traverses edges (USES, IS_A, PART_OF, MENTIONS) to find related entities.

    Args:
        kg: NetworkX MultiDiGraph
        entity_id: Starting entity node ID
        max_depth: Maximum traversal depth

    Returns:
        List of (node_id, entity_data) tuples
    """
    if not kg.has_node(entity_id):
        return []

    # BFS traversal
    visited = set()
    queue = [(entity_id, 0)]
    related = []

    while queue:
        current_id, depth = queue.pop(0)

        if current_id in visited or depth >= max_depth:
            continue

        visited.add(current_id)

        # Add to results (skip starting entity)
        if current_id != entity_id:
            entity_data = kg.nodes[current_id]

            # Skip file nodes
            if entity_data.get('type') != 'file':
                related.append((current_id, entity_data))

        # Traverse neighbors
        for neighbor in kg.neighbors(current_id):
            if neighbor not in visited:
                queue.append((neighbor, depth + 1))

    return related


# ============================================================================
# TEXT POSITION HELPERS
# ============================================================================

def extract_word_at_position(line: str, character: int) -> str:
    """Extract the word at or before the given character position.

    Args:
        line: The text line
        character: Character position (0-indexed)

    Returns:
        The word at the position, or empty string
    """
    if not line or character <= 0:
        return ""

    # Extract text before cursor
    text_before = line[:character]

    # Find last word boundary (whitespace, operators, etc.)
    # Match word characters (alphanumeric + underscore + dot)
    match = re.search(r'([\w.]+)$', text_before)
    if match:
        return match.group(1)

    return ""


def extract_symbol_at_position(line: str, character: int) -> str:
    """Extract the symbol at the given position (word on both sides of cursor).

    Args:
        line: The text line
        character: Character position (0-indexed)

    Returns:
        The symbol at the position
    """
    if not line:
        return ""

    # Find word boundaries around cursor
    # Look for word characters (alphanumeric + underscore)
    start = character
    end = character

    # Expand left
    while start > 0 and (line[start - 1].isalnum() or line[start - 1] == '_'):
        start -= 1

    # Expand right
    while end < len(line) and (line[end].isalnum() or line[end] == '_'):
        end += 1

    return line[start:end]


def format_memory_as_markdown(memory) -> str:
    """Format a memory object as Markdown for hover/documentation.

    Args:
        memory: Memory object from HoloLoom

    Returns:
        Markdown-formatted string
    """
    lines = [
        f"**Memory**: {memory.text[:100]}{'...' if len(memory.text) > 100 else ''}",
        "",
    ]

    if hasattr(memory, 'metadata') and memory.metadata:
        lines.append("**Metadata**:")
        for key, value in memory.metadata.items():
            lines.append(f"- `{key}`: {value}")
        lines.append("")

    if hasattr(memory, 'timestamp'):
        lines.append(f"**Timestamp**: {memory.timestamp}")

    return "\n".join(lines)


# ============================================================================
# LSP HANDLERS
# ============================================================================

@server.feature("initialize")
async def initialize(params: InitializeParams) -> InitializeResult:
    """Handle the initialize request.

    This is the first request from the client. We declare our server
    capabilities here: what LSP features we support.

    Args:
        params: Client initialization parameters

    Returns:
        Server capabilities and initialization result
    """
    logger.info(f"Server initializing for root URI: {params.root_uri}")

    # Store client capabilities for later use
    server.client_id = params.client_info.name if params.client_info else "unknown"
    logger.info(f"Client: {server.client_id}")

    # Declare server capabilities
    server_capabilities = ServerCapabilities(
        # Text document synchronization (we support full document sync)
        text_document_sync=TextDocumentSyncKind.FULL,

        # Completion support
        completion_provider={"resolve_provider": False, "trigger_characters": ["."]},

        # Hover support
        hover_provider=True,

        # Go-to-definition support
        definition_provider=True,

        # Workspace symbol search
        workspace_symbol_provider=True,
    )

    logger.info("Server capabilities declared")
    logger.info("  - Text Document Sync: FULL")
    logger.info("  - Completion: Enabled (trigger: '.')")
    logger.info("  - Hover: Enabled")
    logger.info("  - Definition: Enabled")
    logger.info("  - Workspace Symbol: Enabled")

    return InitializeResult(capabilities=server_capabilities)


@server.feature("initialized")
async def on_initialized(params):
    """Handle the initialized notification.

    This is called after the client acknowledges initialization.
    Use this to start background tasks or lazy-load expensive resources.
    """
    logger.info("Client acknowledged initialization")

    # Initialize HoloLoom
    try:
        from HoloLoom import HoloLoom
        from HoloLoom.config import Config

        logger.info("Initializing HoloLoom instance...")
        config = Config.fast()  # Use FAST mode for balance
        server.hololoom = HoloLoom(config=config)

        # Enter async context manager
        await server.hololoom.__aenter__()

        logger.info("HoloLoom initialized successfully")
        logger.info("  - Config mode: FAST")
        logger.info("  - Memory backend: In-memory (NetworkX)")
        logger.info("  - Embedding scales: %s", config.scales)
        logger.info("  - Knowledge graph nodes: %d", server.hololoom.graph.number_of_nodes())
        logger.info("  - Knowledge graph edges: %d", server.hololoom.graph.number_of_edges())

    except Exception as e:
        logger.error(f"Failed to initialize HoloLoom: {e}", exc_info=True)
        logger.error("Server will run in degraded mode (no HoloLoom integration)")
        server.hololoom = None

    logger.info("Ready to handle LSP requests")


@server.feature("shutdown")
async def shutdown(params):
    """Handle the shutdown request.

    Gracefully shut down the server, cleaning up resources.
    The exit notification will follow.
    """
    logger.info("Server shutdown requested")

    # Clean up HoloLoom context if initialized
    if server.hololoom is not None:
        try:
            logger.info("Closing HoloLoom context...")
            await server.hololoom.__aexit__(None, None, None)
            logger.info("HoloLoom context closed")
        except Exception as e:
            logger.error(f"Error closing HoloLoom context: {e}")

    logger.info("Shutdown complete")


# ============================================================================
# TEXT DOCUMENT HANDLERS - Real KG Integration (Wave 2)
# ============================================================================

@server.feature("textDocument/completion")
async def completion(params: CompletionParams) -> CompletionList:
    """Provide code completion items from knowledge graph.

    Wave 2 Implementation:
    - Queries real KG for entities in current file
    - Ranks by relevance (same file > related > others)
    - Returns entities with metadata (type, docstring, location)

    Args:
        params: Completion parameters (position, document, trigger char)

    Returns:
        List of completion items from KG
    """
    document_uri = params.text_document.uri
    line_num = params.position.line
    character = params.position.character

    logger.debug(f"Completion requested at {document_uri}:{line_num}:{character}")

    # Check if HoloLoom is available
    if not server.hololoom:
        logger.debug("HoloLoom not available, returning empty list")
        return CompletionList(is_incomplete=False, items=[])

    try:
        # Get knowledge graph
        kg = server.hololoom.graph

        if kg.number_of_nodes() == 0:
            logger.debug("Knowledge graph is empty (no workspace indexed)")
            return CompletionList(is_incomplete=False, items=[])

        # Get document
        doc = server.workspace.get_text_document(document_uri)

        # Extract context around cursor
        if line_num >= len(doc.lines):
            logger.debug(f"Line {line_num} out of bounds")
            return CompletionList(is_incomplete=False, items=[])

        line = doc.lines[line_num]

        # Get word before cursor
        query = extract_word_at_position(line, character)

        logger.debug(f"Completion query: '{query}'")

        # Convert URI to file path
        file_path = _uri_to_file_path(document_uri)

        # Try to find relative path (assume workspace root is parent of several dirs)
        # This is a heuristic - in production, we'd get workspace root from client
        file_path_obj = Path(file_path)

        # Try common patterns for relative path
        relative_path = None
        for i in range(1, min(5, len(file_path_obj.parts))):
            candidate = str(Path(*file_path_obj.parts[-i:]))
            file_entity_id = f"file::{candidate}"

            if kg.has_node(file_entity_id):
                relative_path = candidate
                break

        # Get entities from current file
        items = []

        if relative_path:
            logger.debug(f"Found file node: {relative_path}")

            # Get all entities defined in current file (highest priority)
            file_entities = _get_file_entities(kg, relative_path)

            for node_id, entity in file_entities:
                # Filter by query if provided
                if query and query.lower() not in entity.get('name', '').lower():
                    continue

                item = _entity_to_completion_item(entity, node_id, rank=0)
                items.append(item)

            logger.debug(f"Found {len(items)} entities in current file")

        # Also search globally by fuzzy match (lower priority)
        if query:
            fuzzy_matches = _find_entities_fuzzy(kg, query, limit=10)

            for node_id, entity in fuzzy_matches:
                # Skip if already in items (from current file)
                if any(item.label == entity.get('name') for item in items):
                    continue

                item = _entity_to_completion_item(entity, node_id, rank=5)
                items.append(item)

            logger.debug(f"Found {len(fuzzy_matches)} fuzzy matches")

        # Fallback: Use semantic search via HoloLoom.recall() if no KG results
        if not items and query:
            logger.debug("No KG results, falling back to semantic search")
            memories = await server.hololoom.recall(query, limit=5)

            for i, mem in enumerate(memories):
                # Extract entity info from memory context
                if mem.context.get('entity_id'):
                    # This is a stored entity
                    entity_id = mem.context['entity_id']

                    if kg.has_node(entity_id):
                        entity = kg.nodes[entity_id]
                        item = _entity_to_completion_item(entity, entity_id, rank=8)
                        items.append(item)
                else:
                    # Generic memory (not from workspace indexer)
                    label = mem.text.split('\n')[0][:50]

                    items.append(CompletionItem(
                        label=label,
                        kind=CompletionItemKind.Text,
                        detail=f"HoloLoom memory (rank {i+1})",
                        documentation=MarkupContent(
                            kind=MarkupKind.Markdown,
                            value=format_memory_as_markdown(mem)
                        ),
                        insert_text=mem.text[:200],
                        sort_text=f"9_{i}"  # Lowest priority
                    ))

        logger.debug(f"Returning {len(items)} completion items")
        return CompletionList(is_incomplete=False, items=items)

    except Exception as e:
        logger.error(f"Error in completion handler: {e}", exc_info=True)
        return CompletionList(is_incomplete=False, items=[])


@server.feature("textDocument/hover")
async def hover(params: HoverParams) -> Optional[Hover]:
    """Provide hover information from knowledge graph.

    Wave 2 Implementation:
    - Finds entity in KG by name
    - Returns entity metadata (type, file, line, docstring)
    - Shows related entities via graph traversal

    Args:
        params: Hover parameters (position, document)

    Returns:
        Hover information or None
    """
    document_uri = params.text_document.uri
    line_num = params.position.line
    character = params.position.character

    logger.debug(f"Hover requested at {document_uri}:{line_num}:{character}")

    # Check if HoloLoom is available
    if not server.hololoom:
        logger.debug("HoloLoom not available")
        return None

    try:
        # Get knowledge graph
        kg = server.hololoom.graph

        if kg.number_of_nodes() == 0:
            logger.debug("Knowledge graph is empty (no workspace indexed)")
            return None

        # Get document
        doc = server.workspace.get_text_document(document_uri)

        # Extract symbol at position
        if line_num >= len(doc.lines):
            logger.debug(f"Line {line_num} out of bounds")
            return None

        line = doc.lines[line_num]
        symbol = extract_symbol_at_position(line, character)

        if not symbol:
            logger.debug("No symbol at position")
            return None

        logger.debug(f"Hover query for symbol: '{symbol}'")

        # Find entity in KG
        result = _find_entity_by_name(kg, symbol)

        if not result:
            logger.debug(f"Entity '{symbol}' not found in KG")

            # Fallback: Use semantic search
            memories = await server.hololoom.recall(symbol, limit=3)

            if not memories:
                return None

            # Format as Markdown
            lines = [
                f"# {symbol}",
                "",
                "## Related Information (from memories)",
                "",
            ]

            for i, mem in enumerate(memories):
                lines.append(f"### Memory {i+1}")
                lines.append("")
                lines.append(mem.text[:300])
                lines.append("")
                lines.append("---")
                lines.append("")

            content = MarkupContent(
                kind=MarkupKind.Markdown,
                value="\n".join(lines)
            )

            return Hover(contents=content)

        # Entity found in KG
        entity_id, entity = result

        # Build hover content
        lines = [
            f"# {entity.get('name', symbol)}",
            "",
        ]

        # Entity type
        entity_type = entity.get('type', 'entity')
        lines.append(f"**Type**: `{entity_type}`")
        lines.append("")

        # Docstring
        if entity.get('docstring'):
            lines.append("## Documentation")
            lines.append("")
            lines.append(entity['docstring'])
            lines.append("")

        # Location
        lines.append("## Location")
        lines.append("")
        lines.append(f"**File**: `{entity.get('file_path', 'unknown')}`")

        if entity.get('line_number'):
            lines.append(f"**Line**: {entity['line_number']}")

        if entity.get('language'):
            lines.append(f"**Language**: {entity['language']}")

        lines.append("")

        # Related entities (via graph traversal)
        related = _get_related_entities(kg, entity_id, max_depth=1)

        if related:
            lines.append("## Related Entities")
            lines.append("")

            for rel_id, rel_entity in related[:5]:  # Limit to 5
                rel_name = rel_entity.get('name', rel_id.split('::')[-1])
                rel_type = rel_entity.get('type', 'entity')
                lines.append(f"- `{rel_name}` ({rel_type})")

            lines.append("")

        # Metadata
        if entity.get('indexed_at'):
            lines.append(f"*Indexed at: {entity['indexed_at']}*")

        content = MarkupContent(
            kind=MarkupKind.Markdown,
            value="\n".join(lines)
        )

        logger.debug(f"Returning hover information for '{symbol}'")
        return Hover(contents=content)

    except Exception as e:
        logger.error(f"Error in hover handler: {e}", exc_info=True)
        return None


@server.feature("textDocument/definition")
async def definition(params: DefinitionParams) -> Optional[List[Location]]:
    """Provide go-to-definition via knowledge graph.

    Wave 2 Implementation:
    - Finds entity in KG by name
    - Returns file path and line number
    - Converts to LSP Location

    Args:
        params: Definition parameters (position, document)

    Returns:
        List of locations or None
    """
    document_uri = params.text_document.uri
    line_num = params.position.line
    character = params.position.character

    logger.debug(f"Definition requested at {document_uri}:{line_num}:{character}")

    # Check if HoloLoom is available
    if not server.hololoom:
        logger.debug("HoloLoom not available")
        return None

    try:
        # Get knowledge graph
        kg = server.hololoom.graph

        if kg.number_of_nodes() == 0:
            logger.debug("Knowledge graph is empty (no workspace indexed)")
            return None

        # Get document
        doc = server.workspace.get_text_document(document_uri)

        # Extract symbol at position
        if line_num >= len(doc.lines):
            logger.debug(f"Line {line_num} out of bounds")
            return None

        line = doc.lines[line_num]
        symbol = extract_symbol_at_position(line, character)

        if not symbol:
            logger.debug("No symbol at position")
            return None

        logger.debug(f"Definition query for symbol: '{symbol}'")

        # Find entity in KG
        result = _find_entity_by_name(kg, symbol)

        if not result:
            logger.debug(f"Entity '{symbol}' not found in KG")
            return None

        entity_id, entity = result

        # Extract location from entity metadata
        file_path = entity.get('file_path')
        line_number = entity.get('line_number', 0)

        if not file_path:
            logger.debug("Entity has no file_path metadata")
            return None

        # Convert to absolute path and URI
        file_uri = _file_path_to_uri(file_path)

        location = Location(
            uri=file_uri,
            range=Range(
                start=Position(line=int(line_number), character=0),
                end=Position(line=int(line_number), character=100)
            )
        )

        logger.debug(f"Found definition at {file_uri}:{line_number}")
        return [location]

    except Exception as e:
        logger.error(f"Error in definition handler: {e}", exc_info=True)
        return None


@server.feature("workspace/symbol")
async def workspace_symbol(params: WorkspaceSymbolParams) -> List[SymbolInformation]:
    """Search for symbols via knowledge graph + semantic search.

    Wave 2 Implementation:
    - Fuzzy search in KG by name
    - Falls back to semantic search via HoloLoom.recall()
    - Returns symbols with locations

    Args:
        params: Workspace symbol parameters (query)

    Returns:
        List of matching symbols
    """
    query = params.query
    logger.debug(f"Symbol search requested: '{query}'")

    # Check if HoloLoom is available
    if not server.hololoom:
        logger.debug("HoloLoom not available, returning empty list")
        return []

    try:
        # Get knowledge graph
        kg = server.hololoom.graph

        if kg.number_of_nodes() == 0:
            logger.debug("Knowledge graph is empty (no workspace indexed)")
            return []

        symbols = []

        # Option 1: Fuzzy search in KG
        if query:
            fuzzy_matches = _find_entities_fuzzy(kg, query, limit=20)

            for node_id, entity in fuzzy_matches:
                symbol = _entity_to_symbol_info(entity, node_id)
                symbols.append(symbol)

            logger.debug(f"Found {len(fuzzy_matches)} fuzzy matches in KG")

        # Option 2: Semantic search via HoloLoom.recall()
        if not symbols and query:
            logger.debug("No KG results, using semantic search")
            memories = await server.hololoom.recall(query, limit=20)

            for mem in memories:
                # Try to extract entity from memory context
                entity_id = mem.context.get('entity_id')

                if entity_id and kg.has_node(entity_id):
                    # This is a known entity
                    entity = kg.nodes[entity_id]
                    symbol = _entity_to_symbol_info(entity, entity_id)
                    symbols.append(symbol)
                else:
                    # Generic memory (extract from metadata)
                    name = mem.text.split()[0] if mem.text.split() else mem.text[:30]

                    # Determine kind
                    kind = SymbolKind.Variable
                    if any(keyword in mem.text.lower() for keyword in ['function', 'def', 'method']):
                        kind = SymbolKind.Function
                    elif any(keyword in mem.text.lower() for keyword in ['class', 'interface']):
                        kind = SymbolKind.Class

                    # Extract location
                    file_path = mem.context.get('file_path') or mem.metadata.get('file_path')
                    line_number = mem.context.get('line_number', 0) or mem.metadata.get('line_number', 0)

                    if file_path:
                        file_uri = _file_path_to_uri(file_path)
                    else:
                        file_uri = "file:///unknown"

                    symbols.append(SymbolInformation(
                        name=name,
                        kind=kind,
                        location=Location(
                            uri=file_uri,
                            range=Range(
                                start=Position(line=int(line_number), character=0),
                                end=Position(line=int(line_number), character=100)
                            )
                        )
                    ))

        logger.debug(f"Returning {len(symbols)} matching symbols")
        return symbols

    except Exception as e:
        logger.error(f"Error in workspace/symbol handler: {e}", exc_info=True)
        return []


# ============================================================================
# ERROR HANDLING
# ============================================================================

def log_server_exception(exc: Exception) -> None:
    """Log server exceptions with full context.

    Args:
        exc: Exception to log
    """
    logger.exception(f"Server error: {type(exc).__name__}: {exc}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the LSP server.

    Parses command-line arguments and starts the server.
    """
    parser = argparse.ArgumentParser(
        description="HoloLoom Language Server Protocol (LSP) server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m HoloLoom.lsp.server
    Start server on stdio (default for editors)

  python -m HoloLoom.lsp.server --port 8080
    Start on TCP port 8080 (for testing/debugging)

  python -m HoloLoom.lsp.server --log-level DEBUG
    Start with debug logging
        """
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="TCP port to listen on (default: stdio if not specified)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )

    args = parser.parse_args()

    # Setup logging
    global logger
    logger = setup_logging(args.log_level)
    logger.info("=" * 70)
    logger.info("HoloLoom Language Server (LSP) v0.2.0 (Wave 2: Real KG)")
    logger.info("=" * 70)
    logger.info(f"Starting server...")
    logger.info(f"Log level: {args.log_level}")

    # Store config in server
    server.config = {
        "port": args.port,
        "host": args.host,
        "log_level": args.log_level,
    }

    # Start server
    try:
        if args.port:
            logger.info(f"Starting on TCP {args.host}:{args.port}")
            server.start_tcp(args.host, args.port)
        else:
            logger.info("Starting on stdio (stdin/stdout)")
            logger.info("Ready to accept connections from LSP client")
            server.start_io()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
