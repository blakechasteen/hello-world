from __future__ import annotations
"""
RAG Handlers for Matrix ChatOps
================================

Matrix command handlers for HoloLoom RAG (Retrieval-Augmented Generation).

Commands:
- !rag query <question> - Semantic Q&A with sources
- !rag ingest <text> - Add to knowledge base
- !rag search <query> - Retrieval only (no LLM)
- !rag stats - Show cache hit rate, sources

Usage:
    from hololoom.apps.chatops.handlers.rag_handlers import register_rag_handlers

    # In run_chatops.py:
    register_rag_handlers(bot, rag_instance)

Created: 2025-12-05
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.rag.simple_rag import SimpleRAG

try:
    from nio import MatrixRoom, RoomMessageText
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False

# Handler registry
try:
    from hololoom.apps.chatops.handlers.handler_registry import (
        HandlerCategory,
        HandlerRegistry,
        chatops_handler,
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Command Handlers
# ============================================================================

async def handle_rag_query(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    rag: 'SimpleRAG'
) -> str:
    """
    Handle !rag query command - Semantic Q&A with sources.

    Args:
        room: Matrix room
        event: Message event
        args: Query text
        rag: SimpleRAG instance

    Returns:
        Response message with answer and sources
    """
    if not args.strip():
        return "❌ Usage: !rag query <question>\nExample: !rag query What is Thompson Sampling?"

    question = args.strip()

    # Parse optional mode from args (--mode=verify)
    mode = "verify"  # default
    if question.startswith("--mode="):
        parts = question.split(" ", 1)
        mode = parts[0].replace("--mode=", "")
        question = parts[1] if len(parts) > 1 else ""

    if not question:
        return "❌ Usage: !rag query [--mode=direct|verify|research] <question>"

    try:
        result = await rag.query(question, mode=mode)

        # Format response
        response = "🔍 **RAG Query**\n\n"
        response += f"**Answer:** {result.response}\n\n"
        response += f"**Confidence:** {result.confidence:.2f}"

        if result.epistemic_confidence is not None:
            response += f" (epistemic: {result.epistemic_confidence:.2f})"
        response += "\n"

        response += f"**Mode:** {result.reasoning_mode}\n"

        if result.sources:
            response += f"\n**Sources ({len(result.sources)}):**\n"
            for i, source in enumerate(result.sources[:3], 1):
                # Truncate long sources
                truncated = source[:150] + "..." if len(source) > 150 else source
                response += f"{i}. {truncated}\n"
            if len(result.sources) > 3:
                response += f"   _...and {len(result.sources) - 3} more_\n"

        # Add metadata
        if result.metadata.get('cache_hit'):
            response += "\n_⚡ Cache hit_"
        if result.metadata.get('reranking_enabled'):
            response += f"\n_Reranked in {result.metadata.get('rerank_latency_ms', 0):.0f}ms_"

        return response

    except Exception as e:
        logger.error(f"Error in rag query: {e}")
        return f"❌ Query failed: {str(e)}"


async def handle_rag_ingest(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    rag: 'SimpleRAG'
) -> str:
    """
    Handle !rag ingest command - Add content to knowledge base.

    Args:
        room: Matrix room
        event: Message event
        args: Text content to ingest
        rag: SimpleRAG instance

    Returns:
        Confirmation message
    """
    if not args.strip():
        return "❌ Usage: !rag ingest <text>\nExample: !rag ingest Thompson Sampling balances exploration and exploitation"

    content = args.strip()

    try:
        await rag.ingest(content)

        # Truncate for display
        preview = content[:100] + "..." if len(content) > 100 else content

        return f"✅ **Ingested**\n\n_{preview}_\n\nContent added to knowledge base."

    except Exception as e:
        logger.error(f"Error in rag ingest: {e}")
        return f"❌ Ingestion failed: {str(e)}"


async def handle_rag_search(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    rag: 'SimpleRAG'
) -> str:
    """
    Handle !rag search command - Retrieval only (no LLM).

    Args:
        room: Matrix room
        event: Message event
        args: Search query
        rag: SimpleRAG instance

    Returns:
        List of retrieved documents
    """
    if not args.strip():
        return "❌ Usage: !rag search <query>\nExample: !rag search Thompson Sampling"

    query = args.strip()

    # Parse optional limit (--limit=10)
    limit = 5
    if query.startswith("--limit="):
        parts = query.split(" ", 1)
        try:
            limit = int(parts[0].replace("--limit=", ""))
        except ValueError:
            pass
        query = parts[1] if len(parts) > 1 else ""

    if not query:
        return "❌ Usage: !rag search [--limit=N] <query>"

    try:
        # Use loom.recall() directly for retrieval only (no LLM)
        if rag.loom is None:
            return "❌ RAG not initialized"

        memories = await rag.loom.recall(query, limit=limit)

        if not memories:
            return f"🔍 **No results found for:** _{query}_"

        response = f"🔍 **Search Results** ({len(memories)} found)\n\n"
        for i, mem in enumerate(memories, 1):
            # Truncate long text
            text = mem.text[:200] + "..." if len(mem.text) > 200 else mem.text
            response += f"**{i}.** {text}\n"
            if hasattr(mem, 'relevance') and mem.relevance:
                response += f"   _Relevance: {mem.relevance:.2f}_\n"
            response += "\n"

        return response

    except Exception as e:
        logger.error(f"Error in rag search: {e}")
        return f"❌ Search failed: {str(e)}"


async def handle_rag_stats(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    rag: 'SimpleRAG'
) -> str:
    """
    Handle !rag stats command - Show cache hit rate, sources.

    Args:
        room: Matrix room
        event: Message event
        args: (ignored)
        rag: SimpleRAG instance

    Returns:
        Statistics message
    """
    try:
        metrics = rag.get_metrics()

        response = "📊 **RAG Statistics**\n\n"

        # Memory stats
        if 'activation' in metrics:
            response += "**Memory:**\n"
            response += f"  Active nodes: {metrics['activation'].get('active_nodes', 0)}\n"
            response += f"  Activation density: {metrics['activation'].get('density', 0):.2f}\n"

        # Cache stats
        response += "\n**Cache:**\n"
        response += f"  Size: {metrics.get('cache_size', 0)} entries\n"
        response += f"  Hit rate: {metrics.get('cache_hit_rate', 0):.1%}\n"

        # LLM stats
        response += "\n**LLM:**\n"
        response += f"  Provider: {metrics.get('llm_provider', 'N/A')}\n"
        response += f"  Available: {'✓' if metrics.get('llm_available') else '✗'}\n"

        # Reranking stats
        if metrics.get('reranking_enabled'):
            response += "\n**Reranking:**\n"
            response += "  Enabled: ✓\n"
            response += f"  Total reranks: {metrics.get('total_reranks', 0)}\n"
            response += f"  Avg latency: {metrics.get('avg_rerank_latency_ms', 0):.1f}ms\n"
        else:
            response += "\n**Reranking:** Disabled\n"

        # Embedding stats
        if metrics.get('embedding_provider'):
            response += "\n**Embeddings:**\n"
            response += f"  Provider: {metrics.get('embedding_provider')}\n"
            response += f"  Dimension: {metrics.get('embedding_dimension', 'N/A')}\n"

        return response

    except Exception as e:
        logger.error(f"Error in rag stats: {e}")
        return f"❌ Stats failed: {str(e)}"


async def handle_rag_help(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !rag help command.

    Returns:
        Help message
    """
    return """📚 **RAG Commands**

**Query Commands:**
• `!rag query <question>` - Semantic Q&A with sources
• `!rag query --mode=research <question>` - Multi-query exploration
• `!rag query --mode=direct <question>` - Quick answer (no verification)

**Knowledge Base:**
• `!rag ingest <text>` - Add content to knowledge base
• `!rag search <query>` - Retrieval only (no LLM)
• `!rag search --limit=10 <query>` - Limit search results

**Statistics:**
• `!rag stats` - Show cache hit rate, memory stats

**Query Modes:**
• `direct` - Single-pass answer (fastest)
• `verify` - Answer with verification (default)
• `research` - Multi-query exploration (most thorough)

**Examples:**
```
!rag ingest Thompson Sampling balances exploration and exploitation
!rag query What is Thompson Sampling?
!rag search --limit=3 Bayesian
!rag stats
```
"""


# ============================================================================
# Registration
# ============================================================================

def register_rag_handlers(
    bot,
    rag: 'SimpleRAG'
) -> None:
    """
    Register all RAG command handlers.

    Args:
        bot: Matrix bot instance
        rag: SimpleRAG instance
    """
    if not MATRIX_AVAILABLE:
        logger.warning("Matrix not available, RAG handlers not registered")
        return

    logger.info("Registering RAG command handlers")

    # Define command map
    command_map = {
        "query": lambda room, event, args: handle_rag_query(room, event, args, rag),
        "ingest": lambda room, event, args: handle_rag_ingest(room, event, args, rag),
        "search": lambda room, event, args: handle_rag_search(room, event, args, rag),
        "stats": lambda room, event, args: handle_rag_stats(room, event, args, rag),
        "help": handle_rag_help
    }

    # Register with bot
    for cmd, handler in command_map.items():
        bot.register_command(f"rag {cmd}", handler)

    # Also register !rag with no args as help
    bot.register_command("rag", handle_rag_help)

    logger.info(f"Registered {len(command_map)} RAG commands")


# ============================================================================
# Decorator-Based Handler Class
# ============================================================================

class RAGHandlers:
    """
    Decorator-based ChatOps handlers for RAG operations.

    Usage:
        from hololoom.apps.chatops.handlers.rag_handlers import RAGHandlers

        handlers = RAGHandlers(rag=rag_instance)
        registry.register_instance(handlers)
    """

    def __init__(self, rag: 'SimpleRAG'):
        """
        Initialize with SimpleRAG instance.

        Args:
            rag: SimpleRAG instance for RAG operations
        """
        self.rag = rag

    @HandlerRegistry.register(
        command="rag query",
        description="Semantic Q&A with sources",
        usage="!rag query [--mode=direct|verify|research] <question>",
        category=HandlerCategory.QUERY,
        aliases=["rq"]
    )
    async def handle_query(self, room, event, args: str) -> str:
        """Handle !rag query command."""
        return await handle_rag_query(room, event, args, self.rag)

    @HandlerRegistry.register(
        command="rag ingest",
        description="Add content to knowledge base",
        usage="!rag ingest <text>",
        category=HandlerCategory.LEARNING,
        aliases=["ri"]
    )
    async def handle_ingest(self, room, event, args: str) -> str:
        """Handle !rag ingest command."""
        return await handle_rag_ingest(room, event, args, self.rag)

    @HandlerRegistry.register(
        command="rag search",
        description="Retrieval only (no LLM)",
        usage="!rag search [--limit=N] <query>",
        category=HandlerCategory.QUERY,
        aliases=["rs"]
    )
    async def handle_search(self, room, event, args: str) -> str:
        """Handle !rag search command."""
        return await handle_rag_search(room, event, args, self.rag)

    @HandlerRegistry.register(
        command="rag stats",
        description="Show cache hit rate and memory stats",
        usage="!rag stats",
        category=HandlerCategory.SYSTEM,
        aliases=["rstat"]
    )
    async def handle_stats(self, room, event, args: str) -> str:
        """Handle !rag stats command."""
        return await handle_rag_stats(room, event, args, self.rag)

    @HandlerRegistry.register(
        command="rag help",
        description="Show RAG command help",
        usage="!rag help",
        category=HandlerCategory.SYSTEM,
        hidden=True
    )
    async def handle_help(self, room, event, args: str) -> str:
        """Handle !rag help command."""
        return await handle_rag_help(room, event, args)
