"""
HoloLoom Memory MCP Server
==========================
Exposes the unified memory system via Model Context Protocol.

This allows Claude Desktop, VS Code, and other MCP-compatible tools
to query and store memories in your HoloLoom system.

NEW: Conversational interface with auto-spin signal filtering!
- chat() - Conversational interface with importance scoring
- Auto-spins important turns to memory (filters noise)
- Tracks conversation stats and signal/noise ratio

Usage:
    python -m HoloLoom.memory.mcp_server

Configuration:
    Set user_id and backend flags in main() or via environment variables.
"""

import asyncio
import logging
import re
from typing import Any, Sequence, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

try:
    from mcp.server import Server
    from mcp.types import Resource, Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP not installed. Run: pip install mcp")

try:
    from .protocol import UnifiedMemoryInterface, Strategy, create_unified_memory, Memory
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from protocol import UnifiedMemoryInterface, Strategy, create_unified_memory, Memory

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global memory instance (initialized in main)
memory: UnifiedMemoryInterface = None

# Global conversation state (per-session)
conversation_history: List[Dict] = []
conversation_stats = {
    'total_turns': 0,
    'remembered_turns': 0,
    'forgotten_turns': 0,
    'avg_importance': 0.0
}

# Create MCP server
if MCP_AVAILABLE:
    server = Server("holoLoom-memory")
else:
    server = None


# ============================================================================
# Conversational Intelligence: Importance Scoring
# ============================================================================

def score_importance(user_input: str, system_output: str, metadata: Optional[Dict] = None) -> float:
    """
    Score conversation turn importance (0.0-1.0).

    Signal vs Noise filtering:
    - HIGH: Questions, facts, decisions, domain terms
    - LOW: Greetings, acknowledgments, trivial exchanges

    Args:
        user_input: User's message
        system_output: System response
        metadata: Optional metadata (tool, confidence, etc.)

    Returns:
        Importance score (0.0 = noise, 1.0 = critical signal)
    """
    score = 0.5  # Start neutral

    # NOISE indicators (reduce score)
    if len(user_input) < 10 and len(system_output) < 20:
        score -= 0.3

    greetings = r'\b(hi|hello|hey|thanks|thank you|ok|okay|bye|goodbye)\b'
    if re.search(greetings, user_input.lower()) and len(user_input) < 30:
        score -= 0.4

    acknowledgments = r'^(ok|okay|sure|yes|no|got it|i see|alright)[\.\!]?$'
    if re.match(acknowledgments, user_input.lower().strip()):
        score -= 0.4

    if 'error' in system_output.lower() or 'failed' in system_output.lower():
        score -= 0.2

    # SIGNAL indicators (increase score)
    if '?' in user_input:
        score += 0.2

    if len(user_input) > 50:
        score += 0.1
    if len(system_output) > 100:
        score += 0.1

    info_keywords = ['how', 'what', 'why', 'when', 'where', 'who',
                     'explain', 'describe', 'tell me', 'show me',
                     'define', 'meaning', 'example', 'difference']
    keyword_matches = sum(1 for kw in info_keywords if kw in user_input.lower())
    score += keyword_matches * 0.1

    domain_terms = ['memory', 'recall', 'store', 'query', 'search',
                    'knowledge', 'graph', 'vector', 'semantic', 'temporal']
    domain_matches = sum(1 for term in domain_terms
                         if term in user_input.lower() or term in system_output.lower())
    score += domain_matches * 0.05

    # Metadata signals
    if metadata:
        confidence = metadata.get('confidence', 0.5)
        score += (confidence - 0.5) * 0.2

        if metadata.get('tool') in ['store_memory', 'recall_memories']:
            score += 0.15

    # Entity references
    caps_words = re.findall(r'\b[A-Z][a-z]+\b', user_input + ' ' + system_output)
    score += min(len(set(caps_words)) * 0.02, 0.2)

    return max(0.0, min(1.0, score))


def should_remember(importance: float, threshold: float = 0.4) -> bool:
    """Check if turn should be remembered."""
    return importance >= threshold


# ============================================================================
# Resources - Expose memories as browsable resources
# ============================================================================

if MCP_AVAILABLE:
    @server.list_resources()
    async def list_memories() -> list[Resource]:
        """
        List available memories as MCP resources.
        
        Returns recent memories that can be browsed like files.
        Each memory is exposed as memory://<id>
        """
        try:
            # Get recent memories (temporal strategy, last 100)
            result = await memory.recall(
                query="",
                strategy=Strategy.TEMPORAL,
                limit=100
            )
            
            resources = []
            for mem in result.memories:
                # Truncate text for resource name
                name = mem.text[:60] + "..." if len(mem.text) > 60 else mem.text
                
                # Build description with metadata
                desc_parts = [f"Stored: {mem.timestamp.isoformat()}"]
                if mem.context:
                    desc_parts.append(f"Context: {mem.context}")
                if mem.tags:
                    desc_parts.append(f"Tags: {', '.join(mem.tags)}")
                
                resources.append(Resource(
                    uri=f"memory://{mem.id}",
                    name=name,
                    mimeType="text/plain",
                    description=" | ".join(desc_parts)
                ))
            
            logger.info(f"Listed {len(resources)} memory resources")
            return resources
            
        except Exception as e:
            logger.error(f"Error listing memories: {e}")
            return []

    @server.read_resource()
    async def read_memory(uri: str) -> str:
        """
        Read a specific memory by URI.
        
        Args:
            uri: Memory URI in format memory://<id>
            
        Returns:
            Full memory content with metadata
        """
        try:
            # Extract memory ID from URI
            mem_id = uri.replace("memory://", "")
            
            # Search for this specific memory
            result = await memory.recall(
                query=mem_id,
                strategy=Strategy.TEMPORAL,
                limit=100  # Search more to find exact ID
            )
            
            # Find exact match
            target_mem = None
            for mem in result.memories:
                if mem.id == mem_id:
                    target_mem = mem
                    break
            
            if target_mem:
                # Format full memory details
                lines = [
                    f"Memory ID: {target_mem.id}",
                    f"Timestamp: {target_mem.timestamp.isoformat()}",
                    "",
                    "Content:",
                    target_mem.text,
                    "",
                ]
                
                if target_mem.context:
                    lines.append(f"Context: {target_mem.context}")
                if target_mem.tags:
                    lines.append(f"Tags: {', '.join(target_mem.tags)}")
                if target_mem.user_id:
                    lines.append(f"User: {target_mem.user_id}")
                
                return "\n".join(lines)
            else:
                return f"Memory {mem_id} not found"
                
        except Exception as e:
            logger.error(f"Error reading memory {uri}: {e}")
            return f"Error: {e}"


# ============================================================================
# Tools - Expose memory operations as callable tools
# ============================================================================

if MCP_AVAILABLE:
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """
        List available memory tools.

        Returns tools for searching, storing, navigating memories,
        and conversational interface with auto-spin.
        """
        return [
            Tool(
                name="recall_memories",
                description=(
                    "Search memories using various strategies. "
                    "Strategies: temporal (recent first), semantic (meaning-based), "
                    "graph (relationship-based), pattern (recursive patterns), "
                    "fused (weighted combination of all strategies)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (keywords, concepts, or empty for all)"
                        },
                        "strategy": {
                            "type": "string",
                            "enum": ["temporal", "semantic", "graph", "pattern", "fused"],
                            "default": "fused",
                            "description": "Search strategy to use"
                        },
                        "limit": {
                            "type": "integer",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100,
                            "description": "Maximum number of results"
                        },
                        "user_id": {
                            "type": "string",
                            "description": "Filter by user ID (optional)"
                        }
                    },
                    "required": ["query"]
                }
            ),
            
            Tool(
                name="store_memory",
                description=(
                    "Store a new memory in the system. "
                    "The memory will be indexed across all configured backends "
                    "(Mem0, Neo4j, Qdrant) for multi-strategy retrieval."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The memory content to store"
                        },
                        "context": {
                            "type": "object",
                            "description": "Contextual metadata (e.g., place, time, actors, themes)"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization"
                        },
                        "user_id": {
                            "type": "string",
                            "description": "User identifier (optional)"
                        }
                    },
                    "required": ["text"]
                }
            ),
            
            Tool(
                name="memory_health",
                description=(
                    "Check health status of the memory system. "
                    "Returns statistics about available backends and stored memories."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),

            Tool(
                name="chat",
                description=(
                    "Conversational interface with automatic importance scoring. "
                    "Important exchanges are auto-spun into memory (signal). "
                    "Trivial exchanges are filtered (noise). "
                    "Returns response with importance metadata."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Your message/question"
                        },
                        "importance_threshold": {
                            "type": "number",
                            "default": 0.4,
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Minimum importance to remember (0.4 = balanced)"
                        }
                    },
                    "required": ["message"]
                }
            ),

            Tool(
                name="conversation_stats",
                description=(
                    "Get conversation statistics and signal/noise ratio. "
                    "Shows what's being remembered vs filtered."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),

            Tool(
                name="process_text",
                description=(
                    "Process text through the SpinningWheel text spinner. "
                    "Automatically chunks long text, extracts entities and motifs, "
                    "and stores each chunk as a separate memory. "
                    "Perfect for documents, articles, transcripts, or any long-form text."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to process (will be chunked automatically)"
                        },
                        "source": {
                            "type": "string",
                            "default": "document",
                            "description": "Source type (e.g., 'document', 'article', 'transcript')"
                        },
                        "chunk_by": {
                            "type": "string",
                            "enum": ["paragraph", "sentence", "fixed"],
                            "description": "How to chunk the text (optional, defaults to smart chunking)"
                        },
                        "chunk_size": {
                            "type": "integer",
                            "default": 500,
                            "minimum": 100,
                            "maximum": 2000,
                            "description": "Target chunk size in characters (for 'fixed' mode)"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags to add to all chunks"
                        },
                        "user_id": {
                            "type": "string",
                            "description": "User identifier (optional)"
                        }
                    },
                    "required": ["text"]
                }
            ),

            Tool(
                name="ingest_webpage",
                description=(
                    "Ingest a webpage into memory. Automatically scrapes content, "
                    "chunks it intelligently, extracts entities and motifs, "
                    "and stores with URL metadata. Perfect for bookmarks, articles, "
                    "and research pages."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL of webpage to ingest"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Additional tags to add"
                        },
                        "user_id": {
                            "type": "string",
                            "description": "User identifier (optional)"
                        }
                    },
                    "required": ["url"]
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
        """
        Execute memory tool operations.
        
        Args:
            name: Tool name (recall_memories, store_memory, memory_health)
            arguments: Tool-specific arguments
            
        Returns:
            Tool execution results
        """
        try:
            if name == "recall_memories":
                # Parse arguments
                query = arguments.get("query", "")
                strategy_name = arguments.get("strategy", "fused").upper()
                limit = arguments.get("limit", 10)
                user_id = arguments.get("user_id")
                
                # Validate strategy
                try:
                    strategy = Strategy[strategy_name]
                except KeyError:
                    return [TextContent(
                        type="text",
                        text=f"Invalid strategy: {strategy_name}. Use: temporal, semantic, graph, pattern, or fused"
                    )]
                
                # Execute recall
                result = await memory.recall(
                    query=query,
                    strategy=strategy,
                    limit=limit,
                    user_id=user_id
                )
                
                # Format results
                if not result.memories:
                    return [TextContent(
                        type="text",
                        text=f"No memories found for query: '{query}'"
                    )]
                
                lines = [
                    f"Found {len(result.memories)} memories using {strategy.name} strategy:",
                    ""
                ]
                
                for i, (mem, score) in enumerate(zip(result.memories, result.scores), 1):
                    lines.append(f"{i}. [{score:.3f}] {mem.text}")
                    if mem.context:
                        lines.append(f"   Context: {mem.context}")
                    if mem.tags:
                        lines.append(f"   Tags: {', '.join(mem.tags)}")
                    lines.append(f"   ID: {mem.id} | Time: {mem.timestamp.isoformat()}")
                    lines.append("")
                
                return [TextContent(type="text", text="\n".join(lines))]
            
            elif name == "store_memory":
                # Parse arguments
                text = arguments.get("text")
                context = arguments.get("context", {})
                tags = arguments.get("tags", [])
                user_id = arguments.get("user_id")
                
                if not text:
                    return [TextContent(type="text", text="Error: 'text' is required")]
                
                # Store memory
                mem_id = await memory.store(
                    text=text,
                    context=context,
                    tags=tags,
                    user_id=user_id
                )
                
                return [TextContent(
                    type="text",
                    text=f"✓ Memory stored successfully\nID: {mem_id}\nText: {text[:100]}..."
                )]
            
            elif name == "memory_health":
                # Get health status
                health = await memory.health_check()

                lines = [
                    "Memory System Health Check:",
                    f"Status: {health['status']}",
                    f"Backend: {health['backend']}",
                    f"Total Memories: {health['memory_count']}",
                    ""
                ]

                if 'backends' in health:
                    lines.append("Active Backends:")
                    for backend_name, backend_health in health['backends'].items():
                        lines.append(f"  • {backend_name}: {backend_health['status']}")

                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "chat":
                # Conversational interface with auto-spin
                user_message = arguments.get("message", "")
                threshold = arguments.get("importance_threshold", 0.4)

                if not user_message:
                    return [TextContent(type="text", text="Error: 'message' is required")]

                # Search memories for context (semantic recall)
                try:
                    recall_result = await memory.recall(
                        query=user_message,
                        strategy=Strategy.SEMANTIC,
                        limit=5
                    )

                    # Build response from context
                    if recall_result.memories:
                        context_texts = [m.text for m in recall_result.memories[:3]]
                        system_response = (
                            f"Based on your memories, here's what I found:\n\n"
                            f"{chr(10).join('• ' + t[:100] + '...' for t in context_texts)}\n\n"
                            f"Found {len(recall_result.memories)} related memories."
                        )
                    else:
                        system_response = (
                            "I don't have specific memories about that yet. "
                            "This conversation will help me learn!"
                        )

                except Exception as e:
                    system_response = f"I can help with that! (Note: {str(e)})"

                # Score importance
                importance = score_importance(
                    user_message,
                    system_response,
                    metadata={'tool': 'chat', 'confidence': 0.7}
                )

                # Update stats (use global)
                global conversation_stats, conversation_history
                conversation_stats['total_turns'] += 1
                conversation_stats['avg_importance'] = (
                    (conversation_stats['avg_importance'] * (conversation_stats['total_turns'] - 1) + importance) /
                    conversation_stats['total_turns']
                )

                # Auto-remember if important
                remembered = should_remember(importance, threshold)
                if remembered:
                    conversation_stats['remembered_turns'] += 1

                    # Spin to memory
                    turn_text = f"""Conversation Turn {conversation_stats['total_turns']} ({datetime.now().isoformat()})

User: {user_message}

System: {system_response}

Importance: {importance:.2f}"""

                    try:
                        await memory.store(
                            text=turn_text,
                            context={'type': 'conversation', 'importance': importance},
                            tags=['conversation', 'auto-spun'],
                            user_id="blake"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to auto-spin conversation: {e}")
                else:
                    conversation_stats['forgotten_turns'] += 1

                # Add to history
                conversation_history.append({
                    'turn': conversation_stats['total_turns'],
                    'user': user_message,
                    'system': system_response,
                    'importance': importance,
                    'remembered': remembered
                })

                # Keep history manageable
                if len(conversation_history) > 100:
                    conversation_history = conversation_history[-100:]

                # Format response
                status = "✓ REMEMBERED" if remembered else "✗ FILTERED"

                lines = [
                    system_response,
                    "",
                    f"─────────────────────────────────",
                    f"{status} (Importance: {importance:.2f}, Threshold: {threshold})",
                    f"Turn: {conversation_stats['total_turns']} | "
                    f"Signal: {conversation_stats['remembered_turns']} | "
                    f"Noise: {conversation_stats['forgotten_turns']}"
                ]

                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "conversation_stats":
                # Return conversation statistics
                total = conversation_stats['total_turns']
                remembered = conversation_stats['remembered_turns']
                forgotten = conversation_stats['forgotten_turns']
                avg_importance = conversation_stats['avg_importance']

                remember_rate = (remembered / total * 100) if total > 0 else 0

                lines = [
                    "Conversation Statistics:",
                    "=" * 50,
                    f"Total Turns: {total}",
                    f"Signal (Remembered): {remembered} ({remember_rate:.1f}%)",
                    f"Noise (Filtered): {forgotten} ({(100-remember_rate):.1f}%)",
                    f"Avg Importance: {avg_importance:.2f}",
                    "",
                    "Recent History (last 5 turns):"
                ]

                for turn in conversation_history[-5:]:
                    status = "✓" if turn['remembered'] else "✗"
                    lines.append(
                        f"  [{status}] Turn {turn['turn']} (Score: {turn['importance']:.2f})"
                    )
                    lines.append(f"      User: {turn['user'][:50]}...")

                lines.extend([
                    "",
                    "Signal vs Noise Filtering:",
                    "  • Threshold: 0.4 (default)",
                    "  • Signal: Important Q&A, facts, decisions",
                    "  • Noise: Greetings, acknowledgments, trivial chat"
                ])

                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "process_text":
                # Process text through SpinningWheel text spinner
                text_content = arguments.get("text")
                source = arguments.get("source", "document")
                chunk_by = arguments.get("chunk_by")
                chunk_size = arguments.get("chunk_size", 500)
                extra_tags = arguments.get("tags", [])
                user_id = arguments.get("user_id")

                if not text_content:
                    return [TextContent(type="text", text="Error: 'text' is required")]

                try:
                    # Import text spinner and protocol helpers
                    from HoloLoom.spinning_wheel.text import spin_text
                    from .protocol import shards_to_memories

                    # Spin text into shards
                    shards = await spin_text(
                        text=text_content,
                        source=source,
                        chunk_by=chunk_by,
                        chunk_size=chunk_size
                    )

                    if not shards:
                        return [TextContent(
                            type="text",
                            text="No content to store (text may be too short or empty)"
                        )]

                    # Convert shards → Memory objects
                    memories = shards_to_memories(shards)

                    # Add extra tags and user_id to all memories
                    for mem in memories:
                        if extra_tags:
                            mem.tags = list(set((mem.tags or []) + extra_tags))
                        if user_id:
                            mem.user_id = user_id

                    # Batch store
                    memory_ids = await memory.store_many(memories)

                    # Collect statistics
                    total_entities = sum(len(mem.context.get('entities', [])) for mem in memories)
                    total_motifs = sum(len(mem.context.get('motifs', [])) for mem in memories)
                    all_entities = set()
                    all_motifs = set()

                    for mem in memories:
                        all_entities.update(mem.context.get('entities', []))
                        all_motifs.update(mem.context.get('motifs', []))

                    # Format response
                    lines = [
                        "✓ Text processed and stored via SpinningWheel",
                        "",
                        f"📄 Source: {source}",
                        f"📊 Chunks Created: {len(shards)}",
                        f"💾 Memories Stored: {len(memory_ids)}",
                        "",
                        "🔍 Extracted Features:",
                        f"  • Entities: {total_entities} total ({len(all_entities)} unique)",
                        f"  • Motifs: {total_motifs} total ({len(all_motifs)} unique)",
                        ""
                    ]

                    # Show sample entities and motifs
                    if all_entities:
                        sample_entities = list(all_entities)[:5]
                        lines.append(f"  Sample Entities: {', '.join(sample_entities)}")
                    if all_motifs:
                        sample_motifs = list(all_motifs)[:5]
                        lines.append(f"  Sample Motifs: {', '.join(sample_motifs)}")

                    lines.extend([
                        "",
                        "📝 Memory IDs:",
                        *[f"  {i+1}. {mid}" for i, mid in enumerate(memory_ids[:5])],
                    ])

                    if len(memory_ids) > 5:
                        lines.append(f"  ... and {len(memory_ids) - 5} more")

                    return [TextContent(type="text", text="\n".join(lines))]

                except ImportError as e:
                    return [TextContent(
                        type="text",
                        text=f"Error: Text spinner not available. {e}\n"
                             f"Make sure HoloLoom.spinning_wheel.text is installed."
                    )]
                except Exception as e:
                    logger.error(f"Error processing text: {e}")
                    return [TextContent(type="text", text=f"Error processing text: {e}")]

            elif name == "ingest_webpage":
                # Ingest webpage through WebsiteSpinner
                url = arguments.get("url")
                extra_tags = arguments.get("tags", [])
                user_id = arguments.get("user_id")

                if not url:
                    return [TextContent(type="text", text="Error: 'url' is required")]

                try:
                    # Import website spinner and protocol helpers
                    from HoloLoom.spinning_wheel.website import spin_webpage
                    from .protocol import shards_to_memories

                    # Spin webpage into shards (will scrape content)
                    logger.info(f"Ingesting webpage: {url}")
                    shards = await spin_webpage(url=url, tags=extra_tags)

                    if not shards:
                        return [TextContent(
                            type="text",
                            text=f"Failed to ingest {url}. Check that the URL is accessible."
                        )]

                    # Convert shards → Memory objects
                    memories = shards_to_memories(shards)

                    # Add user_id to all memories
                    if user_id:
                        for mem in memories:
                            mem.user_id = user_id

                    # Batch store
                    memory_ids = await memory.store_many(memories)

                    # Extract metadata
                    first_shard = shards[0]
                    domain = first_shard.metadata.get('domain', 'unknown')
                    title = first_shard.metadata.get('title', 'Untitled')

                    # Collect statistics
                    total_entities = sum(len(mem.context.get('entities', [])) for mem in memories)
                    total_motifs = sum(len(mem.context.get('motifs', [])) for mem in memories)
                    all_entities = set()
                    all_motifs = set()

                    for mem in memories:
                        all_entities.update(mem.context.get('entities', []))
                        all_motifs.update(mem.context.get('motifs', []))

                    # Format response
                    lines = [
                        "✓ Webpage ingested successfully",
                        "",
                        f"🌐 URL: {url}",
                        f"📄 Title: {title}",
                        f"🏠 Domain: {domain}",
                        f"📊 Chunks Created: {len(shards)}",
                        f"💾 Memories Stored: {len(memory_ids)}",
                        "",
                        "🔍 Extracted Features:",
                        f"  • Entities: {total_entities} total ({len(all_entities)} unique)",
                        f"  • Motifs: {total_motifs} total ({len(all_motifs)} unique)",
                        ""
                    ]

                    # Show sample entities and motifs
                    if all_entities:
                        sample_entities = list(all_entities)[:5]
                        lines.append(f"  Sample Entities: {', '.join(sample_entities)}")
                    if all_motifs:
                        sample_motifs = list(all_motifs)[:5]
                        lines.append(f"  Sample Motifs: {', '.join(sample_motifs)}")

                    lines.extend([
                        "",
                        f"🔖 Tags: {', '.join(extra_tags + [f'web:{domain}'])}" if extra_tags else f"🔖 Tags: web:{domain}",
                        "",
                        "Now you can search for this content using semantic queries!"
                    ])

                    return [TextContent(type="text", text="\n".join(lines))]

                except ImportError as e:
                    return [TextContent(
                        type="text",
                        text=f"Error: Website spinner not available. {e}\n"
                             f"Install with: pip install requests beautifulsoup4"
                    )]
                except Exception as e:
                    logger.error(f"Error ingesting webpage: {e}")
                    return [TextContent(type="text", text=f"Error ingesting webpage: {e}")]

            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
                
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return [TextContent(type="text", text=f"Error: {e}")]


# ============================================================================
# Server Initialization
# ============================================================================

async def init_memory(
    user_id: str = "default",
    enable_mem0: bool = True,
    enable_neo4j: bool = True,
    enable_qdrant: bool = True
):
    """
    Initialize the memory system with configured backends.
    
    Args:
        user_id: Default user identifier
        enable_mem0: Enable Mem0 backend
        enable_neo4j: Enable Neo4j backend
        enable_qdrant: Enable Qdrant backend
    """
    global memory
    
    logger.info("Initializing HoloLoom Memory MCP Server...")
    
    memory = await create_unified_memory(
        user_id=user_id,
        enable_mem0=enable_mem0,
        enable_neo4j=enable_neo4j,
        enable_qdrant=enable_qdrant
    )
    
    # Check health
    health = await memory.health_check()
    status = health.get('status', 'unknown')
    backend = health.get('backend', 'unknown')
    logger.info(f"Memory system initialized: {status} ({backend})")


async def main():
    """Run the MCP server."""
    if not MCP_AVAILABLE:
        print("Cannot start server: MCP not installed")
        print("Install with: pip install mcp")
        return

    # Initialize memory system
    # NOTE: Mem0 disabled due to OpenAI API quota exceeded
    # Using Qdrant + Neo4j directly (fully local, free)
    await init_memory(
        user_id="blake",
        enable_mem0=False,  # DISABLED - requires OpenAI API key with quota
        enable_neo4j=True,
        enable_qdrant=True
    )
    
    # Run server
    logger.info("Starting MCP server on stdio...")
    logger.info("Available tools: recall_memories, store_memory, process_text, ingest_webpage, memory_health, chat, conversation_stats")
    logger.info("NEW: ingest_webpage - Automatically scrape and ingest web content!")
    logger.info("NEW: process_text - Integrate SpinningWheel text spinner for smart chunking & entity extraction!")
    logger.info("NEW: Conversational interface with auto-spin signal filtering!")
    logger.info("Resources: memory://<id>")

    # Run server with stdio transport
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
