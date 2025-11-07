"""
Agentic Search MCP Server
==========================
Exposes agentic search suite via Model Context Protocol.

Integrates with:
- RAG system (mcp_rag_server.py)
- HoloLoom memory
- Specialized search agents

Usage:
    python -m HoloLoom.search.mcp_agentic_search
"""

import asyncio
import logging
from typing import Sequence, Dict, Any

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP not installed. Run: pip install mcp")

from .agentic_search_suite import (
    SearchOrchestrator,
    SearchStrategy,
    SearchResult
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global orchestrator
orchestrator: SearchOrchestrator = None

# Create MCP server
if MCP_AVAILABLE:
    server = Server("holoLoom-agentic-search")
else:
    server = None


# ============================================================================
# MCP Tools
# ============================================================================

if MCP_AVAILABLE:
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available agentic search tools."""
        return [
            Tool(
                name="agentic_search",
                description=(
                    "Intelligent multi-agent search with automatic strategy selection. "
                    "Routes queries to specialized agents (Factual, Analytical, Multi-Hop, Exploratory). "
                    "Use this for high-quality answers with transparent reasoning."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Your search query or question"
                        },
                        "strategy": {
                            "type": "string",
                            "enum": ["factual", "analytical", "multi_hop", "exploratory", "auto"],
                            "default": "auto",
                            "description": "Search strategy (auto-detected if 'auto')"
                        },
                        "max_documents": {
                            "type": "integer",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50,
                            "description": "Maximum documents to retrieve"
                        }
                    },
                    "required": ["query"]
                }
            ),

            Tool(
                name="parallel_search",
                description=(
                    "Execute multiple searches in parallel. "
                    "Faster than sequential searches. "
                    "Use for batch queries or multi-angle analysis."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of queries to search"
                        },
                        "strategies": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["factual", "analytical", "multi_hop", "exploratory"]
                            },
                            "description": "Optional strategies (one per query)"
                        }
                    },
                    "required": ["queries"]
                }
            ),

            Tool(
                name="decompose_query",
                description=(
                    "Decompose complex query into sub-queries and search in parallel. "
                    "Synthesizes results into coherent answer. "
                    "Use for multi-part or complex questions."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Complex multi-part query"
                        }
                    },
                    "required": ["query"]
                }
            ),

            Tool(
                name="search_stats",
                description=(
                    "Get statistics about search orchestrator and agents. "
                    "Shows query counts, performance metrics, agent usage."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),

            Tool(
                name="compare_entities",
                description=(
                    "Specialized comparison between two or more entities. "
                    "Uses AnalyticalAgent with multi-document synthesis. "
                    "Perfect for 'X vs Y' queries."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "description": "Entities to compare (e.g., ['bread', 'brewing'])"
                        },
                        "dimensions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional comparison dimensions (e.g., ['cost', 'ROI', 'time'])"
                        }
                    },
                    "required": ["entities"]
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
        """Execute agentic search tools."""
        try:
            if name == "agentic_search":
                query = arguments.get("query")
                strategy_name = arguments.get("strategy", "auto")
                max_docs = arguments.get("max_documents", 10)

                if not query:
                    return [TextContent(type="text", text="Error: 'query' is required")]

                # Parse strategy
                strategy = None
                if strategy_name != "auto":
                    try:
                        strategy = SearchStrategy[strategy_name.upper()]
                    except KeyError:
                        return [TextContent(
                            type="text",
                            text=f"Invalid strategy: {strategy_name}. Use: factual, analytical, multi_hop, exploratory, or auto"
                        )]

                # Execute search
                result = await orchestrator.search(
                    query=query,
                    strategy=strategy,
                    max_documents=max_docs
                )

                # Format response
                lines = [
                    f"🤖 Agentic Search: {result.query}",
                    f"📊 Strategy: {result.strategy_used.value}",
                    f"💡 Confidence: {result.confidence:.2%}",
                    f"⏱️  Time: {result.elapsed_ms:.1f}ms",
                    "",
                    "🧠 Agent Reasoning:",
                ]

                for step in result.agent_reasoning:
                    lines.append(f"  • {step}")

                lines.extend([
                    "",
                    "📝 Answer:",
                    "-" * 70,
                    result.answer,
                    "",
                    f"📚 Sources: {len(result.sources)} documents"
                ])

                if result.sources:
                    lines.append("")
                    lines.append("Top Sources:")
                    for i, source in enumerate(result.sources[:3], 1):
                        lines.append(f"  {i}. [{source.get('score', 0):.3f}] {source.get('text', '')[:100]}...")
                        if 'id' in source:
                            lines.append(f"     ID: {source['id']}")

                if result.sub_queries:
                    lines.extend([
                        "",
                        "🔍 Sub-queries:",
                        *[f"  • {sq}" for sq in result.sub_queries]
                    ])

                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "parallel_search":
                queries = arguments.get("queries", [])
                strategies_names = arguments.get("strategies", [])

                if not queries:
                    return [TextContent(type="text", text="Error: 'queries' is required")]

                # Parse strategies
                strategies = []
                for name in strategies_names:
                    try:
                        strategies.append(SearchStrategy[name.upper()])
                    except KeyError:
                        strategies.append(None)

                # Pad strategies if needed
                while len(strategies) < len(queries):
                    strategies.append(None)

                # Execute parallel searches
                results = await orchestrator.parallel_search(queries, strategies[:len(queries)])

                # Format response
                lines = [
                    f"🤖 Parallel Search: {len(queries)} queries",
                    f"⏱️  Total Time: {max(r.elapsed_ms for r in results):.1f}ms (parallel)",
                    "",
                    "Results:",
                    "=" * 70
                ]

                for i, result in enumerate(results, 1):
                    lines.extend([
                        "",
                        f"Query {i}: {result.query}",
                        f"Strategy: {result.strategy_used.value} | Confidence: {result.confidence:.2%}",
                        f"Answer: {result.answer[:200]}...",
                        f"Sources: {len(result.sources)}"
                    ])

                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "decompose_query":
                query = arguments.get("query")

                if not query:
                    return [TextContent(type="text", text="Error: 'query' is required")]

                # Execute decomposition
                result = await orchestrator.decompose_and_search(query)

                # Format response
                lines = [
                    f"🔬 Query Decomposition: {result.query}",
                    f"📊 Sub-queries: {len(result.sub_queries)}",
                    f"⏱️  Total Time: {result.elapsed_ms:.1f}ms",
                    "",
                    "🧩 Decomposition:",
                ]

                for i, sq in enumerate(result.sub_queries, 1):
                    lines.append(f"  {i}. {sq}")

                lines.extend([
                    "",
                    "🧠 Reasoning:",
                    *[f"  • {r}" for r in result.agent_reasoning[:10]],
                    "",
                    "📝 Synthesized Answer:",
                    "-" * 70,
                    result.answer,
                    "",
                    f"📚 Total Sources: {len(result.sources)} documents"
                ])

                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "search_stats":
                stats = orchestrator.get_stats()

                lines = [
                    "📊 Agentic Search Statistics",
                    "=" * 70,
                    f"Total Queries: {stats['total_queries']}",
                    "",
                    "Agent Performance:",
                ]

                for strategy, agent_stats in stats['agents'].items():
                    lines.extend([
                        f"\n{strategy.value.title()} Agent:",
                        f"  Queries: {agent_stats['queries']}",
                        f"  Avg Time: {agent_stats['avg_time_ms']:.1f}ms",
                        f"  Total Time: {agent_stats['total_time_ms']:.1f}ms"
                    ])

                if stats.get('recent_queries'):
                    lines.extend([
                        "",
                        "Recent Queries (last 10):",
                    ])
                    for i, q in enumerate(stats['recent_queries'], 1):
                        lines.append(
                            f"  {i}. [{q['strategy']}] {q['query'][:50]}... "
                            f"({q['confidence']:.2%}, {q['elapsed_ms']:.1f}ms)"
                        )

                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "compare_entities":
                entities = arguments.get("entities", [])
                dimensions = arguments.get("dimensions", [])

                if len(entities) < 2:
                    return [TextContent(type="text", text="Error: Need at least 2 entities to compare")]

                # Build comparison query
                query = f"Compare {' and '.join(entities)}"
                if dimensions:
                    query += f" in terms of {', '.join(dimensions)}"

                # Force analytical strategy
                result = await orchestrator.search(
                    query=query,
                    strategy=SearchStrategy.ANALYTICAL
                )

                # Format response
                lines = [
                    f"⚖️  Entity Comparison: {' vs '.join(entities)}",
                    f"📐 Dimensions: {', '.join(dimensions) if dimensions else 'All aspects'}",
                    f"💡 Confidence: {result.confidence:.2%}",
                    "",
                    "📊 Comparison Results:",
                    "-" * 70,
                    result.answer,
                    "",
                    f"📚 Sources: {len(result.sources)} documents"
                ]

                return [TextContent(type="text", text="\n".join(lines))]

            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            import traceback
            traceback.print_exc()
            return [TextContent(type="text", text=f"Error: {e}")]


# ============================================================================
# Server Initialization
# ============================================================================

async def init_orchestrator():
    """Initialize search orchestrator."""
    global orchestrator

    logger.info("Initializing Agentic Search Orchestrator...")
    orchestrator = SearchOrchestrator()
    logger.info("Orchestrator ready with 4 specialized agents")


async def main():
    """Run the MCP server."""
    if not MCP_AVAILABLE:
        print("Cannot start server: MCP not installed")
        print("Install with: pip install mcp")
        return

    await init_orchestrator()

    logger.info("Starting Agentic Search MCP server on stdio...")
    logger.info("Available tools:")
    logger.info("  • agentic_search - Intelligent multi-agent search")
    logger.info("  • parallel_search - Batch queries in parallel")
    logger.info("  • decompose_query - Complex query decomposition")
    logger.info("  • compare_entities - Specialized comparison")
    logger.info("  • search_stats - Performance statistics")

    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
