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
from typing import Optional, List, Dict, Any
from pathlib import Path
import argparse
from datetime import datetime

from pygls.server import LanguageServer
from pygls.lsp import (
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
    LocationLink,
    DefinitionParams,
    WorkspaceSymbolParams,
    SymbolInformation,
    SymbolKind,
    PublishDiagnosticsParams,
    Diagnostic,
    DiagnosticSeverity,
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
        hololoom_context: Optional HoloLoom orchestrator instance
        config: Server configuration (port, log level, etc.)
    """

    def __init__(self, name: str, version: str, **kwargs):
        super().__init__(name, version, **kwargs)
        self.hololoom_context = None
        self.config: Dict[str, Any] = {}
        self.logger = logging.getLogger("hololoom-lsp")


# Initialize server with LSP capabilities
server = HoloLoomLanguageServer(
    name="hololoom-lsp",
    version="0.1.0",
)

logger = logging.getLogger("hololoom-lsp")


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
    logger.info("Ready to handle LSP requests")


@server.feature("shutdown")
async def shutdown(params):
    """Handle the shutdown request.

    Gracefully shut down the server, cleaning up resources.
    The exit notification will follow.
    """
    logger.info("Server shutdown requested")

    # Clean up HoloLoom context if initialized
    if server.hololoom_context is not None:
        try:
            logger.info("Closing HoloLoom context...")
            # await server.hololoom_context.close()
            # (Future: implement async cleanup)
        except Exception as e:
            logger.error(f"Error closing context: {e}")

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
    document_path = params.text_document.uri
    line = params.position.line
    character = params.position.character

    logger.debug(f"Completion requested at {document_path}:{line}:{character}")

    # TODO: Integrate with HoloLoom
    # - Query HoloLoom memories for relevant entities
    # - Extract completions based on context
    # - Return ranked results

    items = [
        CompletionItem(
            label="HoloLoom.memory",
            kind=CompletionItemKind.Module,
            detail="Memory system (placeholder)",
            documentation="Access HoloLoom's semantic memory",
        ),
        CompletionItem(
            label="HoloLoom.weave",
            kind=CompletionItemKind.Method,
            detail="Main orchestration method (placeholder)",
            documentation="Invoke the full weaving cycle",
        ),
        CompletionItem(
            label="HoloLoom.recall",
            kind=CompletionItemKind.Method,
            detail="Memory retrieval (placeholder)",
            documentation="Retrieve memories from knowledge graph",
        ),
    ]

    logger.debug(f"Returning {len(items)} completion items")
    return CompletionList(is_incomplete=False, items=items)


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
    document_path = params.text_document.uri
    line = params.position.line
    character = params.position.character

    logger.debug(f"Hover requested at {document_path}:{line}:{character}")

    # TODO: Integrate with HoloLoom
    # - Extract symbol at position
    # - Query knowledge graph for entity information
    # - Return rich documentation

    # Placeholder: return sample documentation
    content = MarkupContent(
        kind=MarkupKind.Markdown,
        value="**HoloLoom Entity** (placeholder)\n\n"
               "This is a sample hover response. When integrated with HoloLoom, "
               "this would show:\n\n"
               "- Entity definition from knowledge graph\n"
               "- Related entities and relationships\n"
               "- Usage examples from semantic memory\n"
               "- Confidence scores\n\n"
               "`Code blocks work too`"
    )

    return Hover(contents=content)


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
    document_path = params.text_document.uri
    line = params.position.line
    character = params.position.character

    logger.debug(f"Definition requested at {document_path}:{line}:{character}")

    # TODO: Integrate with HoloLoom
    # - Extract symbol at position
    # - Query knowledge graph for definition
    # - Return location(s) of definition
    # - If multiple definitions exist, return all

    # Placeholder: return sample location
    location = Location(
        uri=document_path,
        range=Range(
            start=Position(line=0, character=0),
            end=Position(line=0, character=10)
        )
    )

    logger.debug(f"Returning definition at {location.uri}")
    return [location]


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
    logger.debug(f"Symbol search requested: {query}")

    # TODO: Integrate with HoloLoom
    # - Query knowledge graph for entities matching query
    # - Return ranked results
    # - Include location information

    # Placeholder: return sample symbols
    symbols = [
        SymbolInformation(
            name="HoloLoom",
            kind=SymbolKind.Module,
            location=Location(
                uri="file:///HoloLoom/__init__.py",
                range=Range(
                    start=Position(line=0, character=0),
                    end=Position(line=10, character=0)
                )
            )
        ),
        SymbolInformation(
            name="WeavingOrchestrator",
            kind=SymbolKind.Class,
            location=Location(
                uri="file:///HoloLoom/weaving_orchestrator.py",
                range=Range(
                    start=Position(line=50, character=0),
                    end=Position(line=100, character=0)
                )
            )
        ),
    ]

    logger.debug(f"Returning {len(symbols)} matching symbols")
    return symbols


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
