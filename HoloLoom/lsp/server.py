#!/usr/bin/env python3
"""
HoloLoom LSP Server
===================
Language Server Protocol (LSP) implementation for HoloLoom neural memory system.

This server provides:
- Code completion from HoloLoom memories
- Hover information with entity context
- Go-to-definition via knowledge graph relationships
- Workspace symbol search (semantic)
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
"""

import asyncio
import logging
import sys
import re
from typing import Optional, List, Dict, Any
from pathlib import Path
import argparse
from datetime import datetime

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
    version="0.1.0",
)

logger = logging.getLogger("hololoom-lsp")


# ============================================================================
# HELPER FUNCTIONS
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
# TEXT DOCUMENT HANDLERS
# ============================================================================

@server.feature("textDocument/completion")
async def completion(params: CompletionParams) -> CompletionList:
    """Provide code completion items.

    Called when the user triggers completion (e.g., Ctrl+Space or typing '.').
    Returns a list of completion items from HoloLoom memories.

    Args:
        params: Completion parameters (position, document, trigger char)

    Returns:
        List of completion items
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
        # Get document
        doc = server.workspace.get_text_document(document_uri)

        # Extract context around cursor
        if line_num >= len(doc.lines):
            logger.debug(f"Line {line_num} out of bounds")
            return CompletionList(is_incomplete=False, items=[])

        line = doc.lines[line_num]

        # Get word before cursor
        query = extract_word_at_position(line, character)

        if not query:
            logger.debug("No query extracted, using current line as context")
            query = line.strip()[:50]  # Use line as fallback

        logger.debug(f"Completion query: '{query}'")

        # Query HoloLoom for relevant memories
        memories = await server.hololoom.recall(query, limit=10)

        logger.debug(f"Retrieved {len(memories)} memories from HoloLoom")

        # Convert memories to CompletionItems
        items = []
        for i, mem in enumerate(memories):
            # Extract a meaningful label (first line or first 50 chars)
            label = mem.text.split('\n')[0][:50]

            # Determine kind based on content
            kind = CompletionItemKind.Text
            if any(keyword in mem.text.lower() for keyword in ['function', 'def', 'method']):
                kind = CompletionItemKind.Function
            elif any(keyword in mem.text.lower() for keyword in ['class', 'interface']):
                kind = CompletionItemKind.Class
            elif any(keyword in mem.text.lower() for keyword in ['import', 'module', 'package']):
                kind = CompletionItemKind.Module

            # Create completion item
            items.append(CompletionItem(
                label=label,
                kind=kind,
                detail=f"HoloLoom memory (rank {i+1})",
                documentation=MarkupContent(
                    kind=MarkupKind.Markdown,
                    value=format_memory_as_markdown(mem)
                ),
                insert_text=mem.text[:200]  # Limit insertion length
            ))

        logger.debug(f"Returning {len(items)} completion items")
        return CompletionList(is_incomplete=False, items=items)

    except Exception as e:
        logger.error(f"Error in completion handler: {e}", exc_info=True)
        return CompletionList(is_incomplete=False, items=[])


@server.feature("textDocument/hover")
async def hover(params: HoverParams) -> Optional[Hover]:
    """Provide hover information for symbols.

    Called when the user hovers over a symbol.
    Returns documentation, type information, etc. from HoloLoom knowledge graph.

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

        # Query HoloLoom for information about this symbol
        memories = await server.hololoom.recall(symbol, limit=5)

        if not memories:
            logger.debug("No memories found for symbol")
            return None

        # Format as Markdown
        lines = [
            f"# {symbol}",
            "",
            "## Related Information",
            "",
        ]

        for i, mem in enumerate(memories[:3]):  # Show top 3
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

        logger.debug(f"Returning hover information for '{symbol}'")
        return Hover(contents=content)

    except Exception as e:
        logger.error(f"Error in hover handler: {e}", exc_info=True)
        return None


@server.feature("textDocument/definition")
async def definition(params: DefinitionParams) -> Optional[List[Location]]:
    """Provide go-to-definition functionality.

    Called when the user requests "Go to Definition" (Ctrl+Click, etc.).
    Returns the definition location(s) from the knowledge graph.

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

        # Query HoloLoom for definition
        # Look for memories that contain "def", "class", or "define" with the symbol
        query = f"definition of {symbol}"
        memories = await server.hololoom.recall(query, limit=5)

        if not memories:
            logger.debug("No definition found")
            return None

        # Try to extract location from memory metadata
        locations = []
        for mem in memories:
            # Check if memory has file location metadata
            if hasattr(mem, 'metadata') and mem.metadata:
                file_path = mem.metadata.get('file_path') or mem.metadata.get('source_file')
                line_number = mem.metadata.get('line_number') or mem.metadata.get('line', 0)

                if file_path:
                    # Convert to URI if needed
                    if not file_path.startswith('file://'):
                        file_path = f"file://{file_path}"

                    locations.append(Location(
                        uri=file_path,
                        range=Range(
                            start=Position(line=int(line_number), character=0),
                            end=Position(line=int(line_number), character=100)
                        )
                    ))

        if locations:
            logger.debug(f"Found {len(locations)} definition locations")
            return locations

        # Fallback: if no metadata, return current file location
        # (This is a placeholder - in production, we'd index the codebase)
        logger.debug("No location metadata found, returning placeholder")
        return [Location(
            uri=document_uri,
            range=Range(
                start=Position(line=0, character=0),
                end=Position(line=0, character=10)
            )
        )]

    except Exception as e:
        logger.error(f"Error in definition handler: {e}", exc_info=True)
        return None


@server.feature("workspace/symbol")
async def workspace_symbol(params: WorkspaceSymbolParams) -> List[SymbolInformation]:
    """Search for symbols in the workspace.

    Called when the user searches for symbols (Ctrl+T in VSCode, etc.).
    Returns matching symbols from HoloLoom knowledge graph.

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
        # Query HoloLoom for symbols
        memories = await server.hololoom.recall(query, limit=20)

        logger.debug(f"Retrieved {len(memories)} memories for symbol search")

        # Convert memories to SymbolInformation
        symbols = []
        for mem in memories:
            # Extract name (first word or first 30 chars)
            name = mem.text.split()[0] if mem.text.split() else mem.text[:30]

            # Determine kind based on content
            kind = SymbolKind.Variable
            if any(keyword in mem.text.lower() for keyword in ['function', 'def', 'method']):
                kind = SymbolKind.Function
            elif any(keyword in mem.text.lower() for keyword in ['class', 'interface']):
                kind = SymbolKind.Class
            elif any(keyword in mem.text.lower() for keyword in ['module', 'package']):
                kind = SymbolKind.Module

            # Extract location from metadata or use placeholder
            file_uri = "file:///unknown"
            line_number = 0

            if hasattr(mem, 'metadata') and mem.metadata:
                file_path = mem.metadata.get('file_path') or mem.metadata.get('source_file')
                if file_path:
                    if not file_path.startswith('file://'):
                        file_uri = f"file://{file_path}"
                    else:
                        file_uri = file_path

                line_number = int(mem.metadata.get('line_number', 0) or mem.metadata.get('line', 0))

            symbols.append(SymbolInformation(
                name=name,
                kind=kind,
                location=Location(
                    uri=file_uri,
                    range=Range(
                        start=Position(line=line_number, character=0),
                        end=Position(line=line_number, character=100)
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
    logger.info("HoloLoom Language Server (LSP) v0.1.0")
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
