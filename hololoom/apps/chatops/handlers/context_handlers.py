"""
Context Handlers for Matrix ChatOps
=====================================

Matrix command handlers for HoloLoom Context Packing System.

Commands:
- !context pack <query> - Pack context with Matryoshka compression
- !context budget <bits> - MI-aware information budget packing
- !context expand <query> - Adaptive budget-aware graph expansion
- !context stream <query> - Streaming progressive expansion
- !context spread <nodes> - Beta wave activation spreading
- !context importance - Show 7-signal importance breakdown
- !context stats - Show packer/spreader/scorer statistics
- !context help - Show context help

Usage:
    from hololoom.apps.chatops.handlers.context_handlers import register_context_handlers

    # In run_chatops.py:
    register_context_handlers(bot, context_packer)

Created: 2025-12-15
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.context_packing import ContextPacker
    from hololoom.context_packing.activation_spreader import ActivationSpreader
    from hololoom.context_packing.importance_scorer import ImportanceScorer

try:
    from nio import MatrixRoom, RoomMessageText
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False
    MatrixRoom = None
    RoomMessageText = None

# Context packing imports with graceful degradation
try:
    from hololoom.context_packing import (
        ContextPacker,
        ContextPackerConfig,
        pack_context,
        information_budget_pack
    )
    PACKER_AVAILABLE = True
except ImportError:
    PACKER_AVAILABLE = False
    ContextPacker = None
    ContextPackerConfig = None
    pack_context = None
    information_budget_pack = None

try:
    from hololoom.context_packing.activation_spreader import ActivationSpreader
    from hololoom.context_packing.config import BetaWaveConfig
    SPREADER_AVAILABLE = True
except ImportError:
    SPREADER_AVAILABLE = False
    ActivationSpreader = None
    BetaWaveConfig = None

try:
    from hololoom.context_packing.importance_scorer import ImportanceScorer
    from hololoom.context_packing.protocol import ImportanceSignal
    SCORER_AVAILABLE = True
except ImportError:
    SCORER_AVAILABLE = False
    ImportanceScorer = None
    ImportanceSignal = None

# Adaptive expansion imports
try:
    from hololoom.memory.adaptive_expansion import (
        AdaptiveExpander,
        expand_context_adaptive
    )
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False
    AdaptiveExpander = None
    expand_context_adaptive = None

# Streaming expansion imports
try:
    from hololoom.memory.streaming_expansion import (
        StreamingContextBuilder,
        stream_context_expansion,
        ChunkYieldStrategy
    )
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    StreamingContextBuilder = None
    stream_context_expansion = None
    ChunkYieldStrategy = None

# Handler registry
try:
    from hololoom.apps.chatops.handlers.handler_registry import (
        HandlerRegistry, HandlerCategory, chatops_handler
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    HandlerRegistry = None
    HandlerCategory = None

logger = logging.getLogger(__name__)

# ============================================================================
# Global Instance Management
# ============================================================================

_context_packer: Optional['ContextPacker'] = None
_activation_spreader: Optional['ActivationSpreader'] = None
_importance_scorer: Optional['ImportanceScorer'] = None
_knowledge_graph: Optional[Any] = None


def get_context_packer() -> Optional['ContextPacker']:
    """Get global context packer instance."""
    return _context_packer


def set_context_packer(packer: 'ContextPacker') -> None:
    """Set global context packer instance."""
    global _context_packer
    _context_packer = packer


def get_activation_spreader() -> Optional['ActivationSpreader']:
    """Get global activation spreader instance."""
    return _activation_spreader


def set_activation_spreader(spreader: 'ActivationSpreader') -> None:
    """Set global activation spreader instance."""
    global _activation_spreader
    _activation_spreader = spreader


def get_importance_scorer() -> Optional['ImportanceScorer']:
    """Get global importance scorer instance."""
    return _importance_scorer


def set_importance_scorer(scorer: 'ImportanceScorer') -> None:
    """Set global importance scorer instance."""
    global _importance_scorer
    _importance_scorer = scorer


def get_knowledge_graph() -> Optional[Any]:
    """Get global knowledge graph instance."""
    return _knowledge_graph


def set_knowledge_graph(kg: Any) -> None:
    """Set global knowledge graph instance."""
    global _knowledge_graph
    _knowledge_graph = kg


# ============================================================================
# Handler Functions
# ============================================================================

async def handle_context_pack(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: List[str],
    packer: Optional['ContextPacker'] = None
) -> str:
    """
    Pack context with Matryoshka compression.

    Usage: !context pack <query> [target_tokens]
    Example: !context pack "What is Thompson Sampling?" 2000
    """
    if not args:
        return "Usage: !context pack <query> [target_tokens]\nExample: !context pack \"What is Thompson Sampling?\" 2000"

    # Parse arguments
    query = args[0] if args else "default query"
    target_tokens = 2000
    if len(args) > 1:
        try:
            target_tokens = int(args[1])
        except ValueError:
            pass

    # Get packer instance
    packer = packer or get_context_packer()
    if packer is None:
        if not PACKER_AVAILABLE:
            return "Context packing not available (missing dependencies)"
        # Create default packer
        try:
            config = ContextPackerConfig.balanced()
            packer = ContextPacker(config)
        except Exception as e:
            return f"Failed to create packer: {e}"

    # Get knowledge graph
    kg = get_knowledge_graph()
    if kg is None:
        return "No knowledge graph available. Set with set_knowledge_graph()"

    try:
        # Get candidate nodes from graph
        import networkx as nx
        if hasattr(kg, '_graph'):
            graph = kg._graph
        elif hasattr(kg, 'graph'):
            graph = kg.graph
        elif isinstance(kg, nx.MultiDiGraph):
            graph = kg
        else:
            return "Knowledge graph format not recognized"

        candidate_nodes = list(graph.nodes())[:100]  # Limit to 100 candidates

        # Pack context
        result = packer.pack(
            query=query,
            candidate_nodes=candidate_nodes,
            graph=graph,
            target_tokens=target_tokens
        )

        # Format response
        lines = [
            "**Context Packing Result**",
            "",
            f"Query: {query}",
            f"Target Tokens: {target_tokens}",
            "",
            "**Compression Statistics:**",
            f"- Original: {result.original_count} nodes",
            f"- Compressed: {result.compressed_count} nodes",
            f"- Compression Ratio: {result.compression_ratio:.1%}",
            f"- Token Savings: {result.token_savings} tokens",
            "",
            "**Top 10 Kept Nodes:**"
        ]

        for i, node_id in enumerate(result.compressed_nodes[:10]):
            score = result.importance_scores.get(node_id, 0.0)
            scale = result.scale_assignments.get(node_id, "384D")
            lines.append(f"  {i+1}. {node_id} (score: {score:.3f}, scale: {scale})")

        return "\n".join(lines)

    except Exception as e:
        logger.exception("Error in context pack")
        return f"Error packing context: {e}"


async def handle_context_budget(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: List[str],
    packer: Optional['ContextPacker'] = None
) -> str:
    """
    MI-aware information budget packing.

    Usage: !context budget <bits> [query]
    Example: !context budget 5.0 "What is Thompson Sampling?"
    """
    if not args:
        return "Usage: !context budget <bits> [query]\nExample: !context budget 5.0 \"What is Thompson Sampling?\""

    # Parse arguments
    try:
        information_budget = float(args[0])
    except ValueError:
        return f"Invalid budget: {args[0]}. Expected number (bits)"

    query = args[1] if len(args) > 1 else "default query"

    if not PACKER_AVAILABLE:
        return "Context packing not available (missing dependencies)"

    # Get knowledge graph
    kg = get_knowledge_graph()
    if kg is None:
        return "No knowledge graph available. Set with set_knowledge_graph()"

    try:
        import networkx as nx
        if hasattr(kg, '_graph'):
            graph = kg._graph
        elif hasattr(kg, 'graph'):
            graph = kg.graph
        elif isinstance(kg, nx.MultiDiGraph):
            graph = kg
        else:
            return "Knowledge graph format not recognized"

        candidate_nodes = list(graph.nodes())[:100]

        # Get node contents (simplified)
        node_contents = {}
        for node in candidate_nodes:
            node_data = graph.nodes.get(node, {})
            content = node_data.get('content', node_data.get('text', str(node)))
            node_contents[node] = content

        # Information budget pack
        if information_budget_pack is None:
            return "information_budget_pack not available"

        nodes, scales, mi_scores = information_budget_pack(
            query=query,
            candidate_nodes=candidate_nodes,
            graph=graph,
            node_contents=node_contents,
            information_budget=information_budget
        )

        # Format response
        lines = [
            "**Information Budget Packing**",
            "",
            f"Query: {query}",
            f"Budget: {information_budget:.2f} bits",
            "",
            "**Results:**",
            f"- Nodes Selected: {len(nodes)}",
            f"- Total MI: {sum(mi_scores.values()):.3f} bits",
            "",
            "**Top 10 by MI Score:**"
        ]

        sorted_nodes = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (node_id, mi) in enumerate(sorted_nodes):
            scale = scales.get(node_id, "384D")
            lines.append(f"  {i+1}. {node_id}: MI={mi:.4f} bits ({scale})")

        return "\n".join(lines)

    except Exception as e:
        logger.exception("Error in budget packing")
        return f"Error in budget packing: {e}"


async def handle_context_expand(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: List[str],
    **kwargs
) -> str:
    """
    Adaptive budget-aware graph expansion.

    Usage: !context expand <query> [token_budget]
    Example: !context expand "Thompson Sampling" 1000
    """
    if not args:
        return "Usage: !context expand <query> [token_budget]\nExample: !context expand \"Thompson Sampling\" 1000"

    query = args[0]
    token_budget = 2000
    if len(args) > 1:
        try:
            token_budget = int(args[1])
        except ValueError:
            pass

    if not ADAPTIVE_AVAILABLE:
        return "Adaptive expansion not available (missing dependencies)"

    # Get knowledge graph
    kg = get_knowledge_graph()
    if kg is None:
        return "No knowledge graph available. Set with set_knowledge_graph()"

    try:
        import networkx as nx
        if hasattr(kg, '_graph'):
            graph = kg._graph
        elif hasattr(kg, 'graph'):
            graph = kg.graph
        elif isinstance(kg, nx.MultiDiGraph):
            graph = kg
        else:
            return "Knowledge graph format not recognized"

        # Find seed nodes (nodes matching query terms)
        query_terms = query.lower().split()
        seed_nodes = []
        for node in graph.nodes():
            node_lower = str(node).lower()
            if any(term in node_lower for term in query_terms):
                seed_nodes.append(node)

        if not seed_nodes:
            seed_nodes = list(graph.nodes())[:3]  # Default to first 3

        # Run adaptive expansion
        result = await expand_context_adaptive(
            query=query,
            seed_nodes=seed_nodes,
            graph=graph,
            token_budget=token_budget,
            min_relevance=0.3,
            max_hops=5
        )

        # Format response
        lines = [
            "**Adaptive Context Expansion**",
            "",
            f"Query: {query}",
            f"Token Budget: {token_budget}",
            f"Seed Nodes: {', '.join(seed_nodes[:5])}{'...' if len(seed_nodes) > 5 else ''}",
            "",
            "**Expansion Results:**",
            f"- Nodes Expanded: {len(result.nodes)}",
            f"- Tokens Used: {result.total_tokens}/{token_budget}",
            f"- Avg Relevance: {result.avg_relevance:.3f}",
            f"- Max Hop Reached: {result.max_hop_reached}",
            f"- Stopping Reason: {result.stopping_reason}",
            "",
            "**Top 10 Expanded Nodes:**"
        ]

        for i, (node_id, relevance) in enumerate(list(result.relevance_scores.items())[:10]):
            lines.append(f"  {i+1}. {node_id} (relevance: {relevance:.3f})")

        return "\n".join(lines)

    except Exception as e:
        logger.exception("Error in adaptive expansion")
        return f"Error in adaptive expansion: {e}"


async def handle_context_stream(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: List[str],
    **kwargs
) -> str:
    """
    Streaming progressive expansion summary.

    Usage: !context stream <query> [chunk_size]
    Example: !context stream "Thompson Sampling" 500
    """
    if not args:
        return "Usage: !context stream <query> [chunk_size]\nExample: !context stream \"Thompson Sampling\" 500"

    query = args[0]
    chunk_size = 500
    if len(args) > 1:
        try:
            chunk_size = int(args[1])
        except ValueError:
            pass

    if not STREAMING_AVAILABLE:
        return "Streaming expansion not available (missing dependencies)"

    # Get knowledge graph
    kg = get_knowledge_graph()
    if kg is None:
        return "No knowledge graph available. Set with set_knowledge_graph()"

    try:
        import networkx as nx
        if hasattr(kg, '_graph'):
            graph = kg._graph
        elif hasattr(kg, 'graph'):
            graph = kg.graph
        elif isinstance(kg, nx.MultiDiGraph):
            graph = kg
        else:
            return "Knowledge graph format not recognized"

        # Find seed nodes
        query_terms = query.lower().split()
        seed_nodes = []
        for node in graph.nodes():
            node_lower = str(node).lower()
            if any(term in node_lower for term in query_terms):
                seed_nodes.append(node)

        if not seed_nodes:
            seed_nodes = list(graph.nodes())[:3]

        # Collect streaming chunks
        chunks_info = []
        total_nodes = 0
        total_tokens = 0

        async for chunk in stream_context_expansion(
            query=query,
            seed_nodes=seed_nodes,
            graph=graph,
            token_budget=2000,
            chunk_size=chunk_size,
            yield_strategy=ChunkYieldStrategy.HYBRID
        ):
            chunks_info.append({
                'index': chunk.chunk_index,
                'nodes': chunk.node_count,
                'tokens': chunk.token_count,
                'hop': chunk.hop_distance,
                'relevance': chunk.avg_relevance
            })
            total_nodes += chunk.node_count
            total_tokens += chunk.token_count

            if chunk.is_final:
                break

        # Format response
        lines = [
            "**Streaming Context Expansion**",
            "",
            f"Query: {query}",
            f"Chunk Size: {chunk_size} tokens",
            "",
            "**Stream Summary:**",
            f"- Total Chunks: {len(chunks_info)}",
            f"- Total Nodes: {total_nodes}",
            f"- Total Tokens: {total_tokens}",
            "",
            "**Chunks:**"
        ]

        for chunk in chunks_info[:10]:
            lines.append(
                f"  Chunk {chunk['index']}: {chunk['nodes']} nodes, "
                f"{chunk['tokens']} tokens, hop {chunk['hop']}, "
                f"relevance {chunk['relevance']:.3f}"
            )

        if len(chunks_info) > 10:
            lines.append(f"  ... and {len(chunks_info) - 10} more chunks")

        return "\n".join(lines)

    except Exception as e:
        logger.exception("Error in streaming expansion")
        return f"Error in streaming expansion: {e}"


async def handle_context_spread(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: List[str],
    spreader: Optional['ActivationSpreader'] = None
) -> str:
    """
    Beta wave activation spreading.

    Usage: !context spread <nodes> [max_hops]
    Example: !context spread "thompson_sampling,bayesian" 3
    """
    if not args:
        return "Usage: !context spread <nodes> [max_hops]\nExample: !context spread \"thompson_sampling,bayesian\" 3"

    # Parse arguments
    source_nodes = [n.strip() for n in args[0].split(',')]
    max_hops = 3
    if len(args) > 1:
        try:
            max_hops = int(args[1])
        except ValueError:
            pass

    # Get spreader
    spreader = spreader or get_activation_spreader()
    if spreader is None:
        if not SPREADER_AVAILABLE:
            return "Activation spreading not available (missing dependencies)"
        try:
            config = BetaWaveConfig()
            spreader = ActivationSpreader(config)
        except Exception as e:
            return f"Failed to create spreader: {e}"

    # Get knowledge graph
    kg = get_knowledge_graph()
    if kg is None:
        return "No knowledge graph available. Set with set_knowledge_graph()"

    try:
        import networkx as nx
        if hasattr(kg, '_graph'):
            graph = kg._graph
        elif hasattr(kg, 'graph'):
            graph = kg.graph
        elif isinstance(kg, nx.MultiDiGraph):
            graph = kg
        else:
            return "Knowledge graph format not recognized"

        # Spread activation
        activation_map = spreader.spread_activation(
            source_nodes=source_nodes,
            graph=graph,
            initial_activation=1.0,
            max_hops=max_hops
        )

        # Format response
        lines = [
            "**Beta Wave Activation Spreading**",
            "",
            f"Source Nodes: {', '.join(source_nodes)}",
            f"Max Hops: {max_hops}",
            "",
            "**Spreading Statistics:**",
            f"- Activated Nodes: {len(activation_map)}",
            f"- Avg Activation: {sum(activation_map.values()) / len(activation_map):.3f}" if activation_map else "- Avg Activation: 0.0",
            "",
            "**Top 15 Activated Nodes:**"
        ]

        sorted_nodes = sorted(activation_map.items(), key=lambda x: x[1], reverse=True)[:15]
        for i, (node_id, activation) in enumerate(sorted_nodes):
            bar = "█" * int(activation * 10)
            lines.append(f"  {i+1}. {node_id}: {activation:.3f} {bar}")

        return "\n".join(lines)

    except Exception as e:
        logger.exception("Error in activation spreading")
        return f"Error in activation spreading: {e}"


async def handle_context_importance(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: List[str],
    scorer: Optional['ImportanceScorer'] = None
) -> str:
    """
    Show 7-signal importance breakdown.

    Usage: !context importance [node_id] [query]
    Example: !context importance "thompson_sampling" "What is exploration?"
    """
    node_id = args[0] if args else None
    query = args[1] if len(args) > 1 else "default query"

    # Get scorer
    scorer = scorer or get_importance_scorer()
    if scorer is None:
        if not SCORER_AVAILABLE:
            return "Importance scoring not available (missing dependencies)"
        try:
            scorer = ImportanceScorer()
        except Exception as e:
            return f"Failed to create scorer: {e}"

    # Get knowledge graph
    kg = get_knowledge_graph()
    if kg is None:
        return "No knowledge graph available. Set with set_knowledge_graph()"

    try:
        import networkx as nx
        if hasattr(kg, '_graph'):
            graph = kg._graph
        elif hasattr(kg, 'graph'):
            graph = kg.graph
        elif isinstance(kg, nx.MultiDiGraph):
            graph = kg
        else:
            return "Knowledge graph format not recognized"

        # If no node specified, use first node
        if node_id is None:
            nodes = list(graph.nodes())
            if not nodes:
                return "No nodes in knowledge graph"
            node_id = nodes[0]

        # Score importance
        scores = scorer.score_importance(
            node_id=node_id,
            query=query,
            graph=graph
        )

        # Format response
        lines = [
            "**7-Signal Importance Breakdown**",
            "",
            f"Node: {node_id}",
            f"Query: {query}",
            "",
            "**Signal Scores:**"
        ]

        total = 0.0
        for signal, score in scores.items():
            signal_name = signal.value if hasattr(signal, 'value') else str(signal)
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(f"  {signal_name:20s}: {score:.3f} [{bar}]")
            total += score

        lines.append("")
        lines.append(f"**Weighted Total:** {total / len(scores):.3f}")
        lines.append("")
        lines.append("**Signal Descriptions:**")
        lines.append("  - RECENCY: How recently accessed")
        lines.append("  - RELEVANCE: Semantic similarity to query")
        lines.append("  - CENTRALITY: Graph importance (PageRank)")
        lines.append("  - ACCESS_FREQUENCY: Historical access count")
        lines.append("  - CONFIDENCE: Historical confidence scores")
        lines.append("  - HEAT: Hot pattern feedback score")
        lines.append("  - INFORMATION_CONTENT: Mutual information (Phase 5)")

        return "\n".join(lines)

    except Exception as e:
        logger.exception("Error in importance scoring")
        return f"Error in importance scoring: {e}"


async def handle_context_stats(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: List[str],
    **kwargs
) -> str:
    """
    Show packer/spreader/scorer statistics.

    Usage: !context stats
    """
    lines = [
        "**Context Packing System Statistics**",
        "",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]

    # Packer stats
    lines.append("**Context Packer:**")
    packer = get_context_packer()
    if packer is not None and hasattr(packer, '_stats'):
        stats = packer._stats
        lines.append(f"  - Total Packs: {stats.get('total_packs', 0)}")
        lines.append(f"  - Avg Compression Ratio: {stats.get('avg_compression_ratio', 0.0):.1%}")
        lines.append(f"  - Avg Token Savings: {stats.get('avg_token_savings', 0):.0f}")
    else:
        lines.append("  - Not initialized or no stats available")

    lines.append("")

    # Spreader stats
    lines.append("**Activation Spreader:**")
    spreader = get_activation_spreader()
    if spreader is not None and hasattr(spreader, '_stats'):
        stats = spreader._stats
        lines.append(f"  - Total Spreads: {stats.get('total_spreads', 0)}")
        lines.append(f"  - Avg Hops: {stats.get('avg_hops', 0.0):.1f}")
        lines.append(f"  - Avg Activated Nodes: {stats.get('avg_activated_nodes', 0):.0f}")
    else:
        lines.append("  - Not initialized or no stats available")

    lines.append("")

    # Scorer stats
    lines.append("**Importance Scorer:**")
    scorer = get_importance_scorer()
    if scorer is not None and hasattr(scorer, '_stats'):
        stats = scorer._stats
        lines.append(f"  - Total Scored: {stats.get('total_scored', 0)}")
        lines.append(f"  - Centrality Cache Hits: {stats.get('cache_hits', 0)}")
        lines.append(f"  - Centrality Cache Misses: {stats.get('cache_misses', 0)}")
        lines.append(f"  - MI Cache Hits: {stats.get('mi_cache_hits', 0)}")
        lines.append(f"  - MI Cache Misses: {stats.get('mi_cache_misses', 0)}")
    else:
        lines.append("  - Not initialized or no stats available")

    lines.append("")

    # Availability
    lines.append("**Component Availability:**")
    lines.append(f"  - Context Packer: {'Available' if PACKER_AVAILABLE else 'Not Available'}")
    lines.append(f"  - Activation Spreader: {'Available' if SPREADER_AVAILABLE else 'Not Available'}")
    lines.append(f"  - Importance Scorer: {'Available' if SCORER_AVAILABLE else 'Not Available'}")
    lines.append(f"  - Adaptive Expansion: {'Available' if ADAPTIVE_AVAILABLE else 'Not Available'}")
    lines.append(f"  - Streaming Expansion: {'Available' if STREAMING_AVAILABLE else 'Not Available'}")

    return "\n".join(lines)


async def handle_context_help(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: List[str],
    **kwargs
) -> str:
    """
    Show context packing help.

    Usage: !context help
    """
    return """**Context Packing Commands**

**Compression & Packing:**
  !context pack <query> [tokens] - Pack context with Matryoshka compression
  !context budget <bits> [query] - MI-aware information budget packing

**Graph Expansion:**
  !context expand <query> [budget] - Adaptive budget-aware graph expansion
  !context stream <query> [chunk] - Streaming progressive expansion

**Activation & Scoring:**
  !context spread <nodes> [hops] - Beta wave activation spreading
  !context importance [node] [q] - Show 7-signal importance breakdown

**Statistics:**
  !context stats - Show packer/spreader/scorer statistics
  !context help - Show this help

**Examples:**
  !context pack "What is Thompson Sampling?" 2000
  !context budget 5.0 "Exploration strategies"
  !context expand "neural networks" 1000
  !context stream "machine learning" 500
  !context spread "attention,transformer" 3
  !context importance "bert" "NLP models"

**Key Features:**
  - 40-90% token savings with Matryoshka compression
  - Beta wave physics-based activation spreading
  - 7-signal importance scoring (including MI)
  - Adaptive graph traversal with budget awareness
  - Streaming progressive expansion for low latency"""


# ============================================================================
# Handler Class (for grouped handlers)
# ============================================================================

class ContextHandlers:
    """Grouped context packing handlers."""

    def __init__(
        self,
        packer: Optional['ContextPacker'] = None,
        spreader: Optional['ActivationSpreader'] = None,
        scorer: Optional['ImportanceScorer'] = None,
        knowledge_graph: Optional[Any] = None
    ):
        """
        Initialize context handlers.

        Args:
            packer: ContextPacker instance
            spreader: ActivationSpreader instance
            scorer: ImportanceScorer instance
            knowledge_graph: Knowledge graph for operations
        """
        self.packer = packer
        self.spreader = spreader
        self.scorer = scorer
        self.knowledge_graph = knowledge_graph

        # Set global instances
        if packer:
            set_context_packer(packer)
        if spreader:
            set_activation_spreader(spreader)
        if scorer:
            set_importance_scorer(scorer)
        if knowledge_graph:
            set_knowledge_graph(knowledge_graph)

    async def handle_pack(self, room, event, args: List[str]) -> str:
        """Handle !context pack command."""
        return await handle_context_pack(room, event, args, self.packer)

    async def handle_budget(self, room, event, args: List[str]) -> str:
        """Handle !context budget command."""
        return await handle_context_budget(room, event, args, self.packer)

    async def handle_expand(self, room, event, args: List[str]) -> str:
        """Handle !context expand command."""
        return await handle_context_expand(room, event, args)

    async def handle_stream(self, room, event, args: List[str]) -> str:
        """Handle !context stream command."""
        return await handle_context_stream(room, event, args)

    async def handle_spread(self, room, event, args: List[str]) -> str:
        """Handle !context spread command."""
        return await handle_context_spread(room, event, args, self.spreader)

    async def handle_importance(self, room, event, args: List[str]) -> str:
        """Handle !context importance command."""
        return await handle_context_importance(room, event, args, self.scorer)

    async def handle_stats(self, room, event, args: List[str]) -> str:
        """Handle !context stats command."""
        return await handle_context_stats(room, event, args)

    async def handle_help(self, room, event, args: List[str]) -> str:
        """Handle !context help command."""
        return await handle_context_help(room, event, args)


# ============================================================================
# Registration Function
# ============================================================================

def register_context_handlers(
    bot,
    context_packer: Optional['ContextPacker'] = None,
    activation_spreader: Optional['ActivationSpreader'] = None,
    importance_scorer: Optional['ImportanceScorer'] = None,
    knowledge_graph: Optional[Any] = None
) -> ContextHandlers:
    """
    Register context packing handlers with Matrix bot.

    Args:
        bot: Matrix bot instance
        context_packer: Optional ContextPacker instance
        activation_spreader: Optional ActivationSpreader instance
        importance_scorer: Optional ImportanceScorer instance
        knowledge_graph: Optional knowledge graph for operations

    Returns:
        ContextHandlers instance for grouped access

    Example:
        from hololoom.apps.chatops.handlers.context_handlers import register_context_handlers
        from hololoom.context_packing import ContextPacker

        packer = ContextPacker()
        handlers = register_context_handlers(bot, packer)
    """
    # Create handler group
    handlers = ContextHandlers(
        packer=context_packer,
        spreader=activation_spreader,
        scorer=importance_scorer,
        knowledge_graph=knowledge_graph
    )

    # Set global instances
    if context_packer:
        set_context_packer(context_packer)
    if activation_spreader:
        set_activation_spreader(activation_spreader)
    if importance_scorer:
        set_importance_scorer(importance_scorer)
    if knowledge_graph:
        set_knowledge_graph(knowledge_graph)

    # Register commands with bot
    if hasattr(bot, 'register_command'):
        # Modern bot API
        bot.register_command('context pack', handlers.handle_pack)
        bot.register_command('context budget', handlers.handle_budget)
        bot.register_command('context expand', handlers.handle_expand)
        bot.register_command('context stream', handlers.handle_stream)
        bot.register_command('context spread', handlers.handle_spread)
        bot.register_command('context importance', handlers.handle_importance)
        bot.register_command('context stats', handlers.handle_stats)
        bot.register_command('context help', handlers.handle_help)
    elif hasattr(bot, 'commands'):
        # Dict-based command registration
        bot.commands['context pack'] = handlers.handle_pack
        bot.commands['context budget'] = handlers.handle_budget
        bot.commands['context expand'] = handlers.handle_expand
        bot.commands['context stream'] = handlers.handle_stream
        bot.commands['context spread'] = handlers.handle_spread
        bot.commands['context importance'] = handlers.handle_importance
        bot.commands['context stats'] = handlers.handle_stats
        bot.commands['context help'] = handlers.handle_help
    else:
        logger.warning("Bot does not support command registration")

    logger.info("Registered 8 context packing commands")
    return handlers


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Handler functions
    'handle_context_pack',
    'handle_context_budget',
    'handle_context_expand',
    'handle_context_stream',
    'handle_context_spread',
    'handle_context_importance',
    'handle_context_stats',
    'handle_context_help',
    # Handler class
    'ContextHandlers',
    # Registration
    'register_context_handlers',
    # Instance management
    'get_context_packer',
    'set_context_packer',
    'get_activation_spreader',
    'set_activation_spreader',
    'get_importance_scorer',
    'set_importance_scorer',
    'get_knowledge_graph',
    'set_knowledge_graph',
    # Availability flags
    'PACKER_AVAILABLE',
    'SPREADER_AVAILABLE',
    'SCORER_AVAILABLE',
    'ADAPTIVE_AVAILABLE',
    'STREAMING_AVAILABLE',
]
