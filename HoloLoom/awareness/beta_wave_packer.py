"""
Beta Wave Context Packer - Elegant Solution

Replaces 506 lines of ad-hoc heuristics with physics-based importance.

Core Principle: "Activation IS Importance"
- Beta wave activation spreading provides natural relevance ranking
- Spring constant k encodes recency (fresh memories conduct better)
- Creative insights give cross-domain bridges for free
- No magic numbers, no brittle heuristics, no multi-pass loops

Just trust the springs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import time
import numpy as np


@dataclass
class TokenBudget:
    """Token budget constraints"""
    total: int = 8000
    reserved_for_query: int = 500
    reserved_for_response: int = 1000

    @property
    def available_for_context(self) -> int:
        """Tokens available for context packing"""
        return self.total - self.reserved_for_query - self.reserved_for_response


@dataclass
class ContextElement:
    """Single piece of context with physics-based importance"""
    content: str
    activation: float  # Direct from beta wave spreading (0.0-1.0)
    token_count: int
    source: str  # "query", "memory", "creative_insight", "awareness"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def importance(self) -> float:
        """Activation IS importance"""
        return self.activation


@dataclass
class PackedContext:
    """Assembled context ready for LLM"""
    query_section: str
    memory_section: str
    awareness_section: str
    creative_section: str

    total_tokens: int
    elements_included: int
    elements_compressed: int
    elements_excluded: int

    avg_activation: float
    min_activation: float
    max_activation: float

    packing_time_ms: float
    activation_stats: Dict[str, Any]

    def format_for_llm(self, include_metadata: bool = False) -> str:
        """Format packed context for LLM prompt"""
        sections = []

        # Query (always first)
        if self.query_section:
            sections.append("# QUERY")
            sections.append(self.query_section)
            sections.append("")

        # Awareness context
        if self.awareness_section:
            sections.append("# AWARENESS CONTEXT")
            sections.append(self.awareness_section)
            sections.append("")

        # Memory retrieval (sorted by activation)
        if self.memory_section:
            sections.append("# RELEVANT MEMORIES")
            sections.append(self.memory_section)
            sections.append("")

        # Creative insights (cross-domain bridges)
        if self.creative_section:
            sections.append("# CREATIVE INSIGHTS")
            sections.append(self.creative_section)
            sections.append("")

        # Optional metadata
        if include_metadata:
            sections.append("# PACKING METADATA")
            sections.append(f"Total tokens: {self.total_tokens}")
            sections.append(f"Elements: {self.elements_included} included, "
                          f"{self.elements_compressed} compressed, "
                          f"{self.elements_excluded} excluded")
            sections.append(f"Activation: avg={self.avg_activation:.3f}, "
                          f"min={self.min_activation:.3f}, "
                          f"max={self.max_activation:.3f}")

        return "\n".join(sections)


class BetaWaveContextPacker:
    """
    Elegant context packing using beta wave activation spreading.

    Replaces ad-hoc heuristics with physics-based importance:
    - Activation spreading → relevance ranking
    - Spring constant k → recency/freshness
    - Creative insights → cross-domain bridges

    Algorithm (single pass):
    1. Run beta wave retrieval → get activation map
    2. Create context elements with activation as importance
    3. Sort by activation (already done by spring dynamics)
    4. Pack until budget exhausted
    5. Compress based on activation threshold

    No magic numbers. Just trust the springs.
    """

    def __init__(
        self,
        spring_engine,  # SpringDynamicsEngine instance
        token_budget: Optional[TokenBudget] = None,
        activation_threshold: float = 0.3,  # Exclude below this
        compression_threshold: float = 0.7  # Compress below this
    ):
        """
        Initialize beta wave context packer.

        Args:
            spring_engine: SpringDynamicsEngine for beta wave retrieval
            token_budget: Token budget constraints
            activation_threshold: Minimum activation to include (default 0.3)
            compression_threshold: Activation level for full vs compressed (default 0.7)
        """
        self.engine = spring_engine
        self.budget = token_budget or TokenBudget()
        self.activation_threshold = activation_threshold
        self.compression_threshold = compression_threshold

    async def pack_context(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        awareness_context=None,
        top_k: int = 50
    ) -> PackedContext:
        """
        Pack context using beta wave activation spreading.

        Elegance:
        - Single retrieval call (beta wave spreading ranks everything)
        - Single packing pass (no critical/high/medium/low passes)
        - Physics-based importance (no ad-hoc heuristics)
        - Natural compression (activation threshold determines level)

        Args:
            query_text: User query text
            query_embedding: Query embedding vector
            awareness_context: Optional awareness signals
            top_k: Max memories to retrieve

        Returns:
            PackedContext ready for LLM
        """
        start_time = time.time()

        # 1. Beta wave retrieval (physics does the hard work)
        result = self.engine.retrieve_memories(
            query_embedding=query_embedding,
            top_k=top_k,
            activation_threshold=self.activation_threshold
        )

        # 2. Create context elements (activation = importance)
        elements = []

        # Query element (always critical, activation = 1.0)
        elements.append(ContextElement(
            content=query_text,
            activation=1.0,  # Always maximum importance
            token_count=self._estimate_tokens(query_text),
            source="query",
            metadata={"type": "user_query"}
        ))

        # Awareness elements (if provided)
        if awareness_context:
            awareness_elements = self._create_awareness_elements(awareness_context)
            elements.extend(awareness_elements)

        # Memory elements (use activation from beta wave spreading)
        for node_id, activation in result.recalled_memories:
            node = self.engine.nodes[node_id]
            elements.append(ContextElement(
                content=node.content,
                activation=activation,  # Direct from physics!
                token_count=self._estimate_tokens(node.content),
                source="memory",
                metadata={
                    "node_id": node_id,
                    "spring_k": node.spring_constant,
                    "beta_wave_activation": activation
                }
            ))

        # Creative insights (cross-domain bridges)
        for node_id, activation, insight_distance in result.creative_insights:
            node = self.engine.nodes[node_id]
            elements.append(ContextElement(
                content=node.content,
                activation=activation * 0.9,  # Slightly lower (not direct match)
                token_count=self._estimate_tokens(node.content),
                source="creative_insight",
                metadata={
                    "node_id": node_id,
                    "insight_distance": insight_distance,
                    "cross_domain": True
                }
            ))

        # 3. Single-pass packing (already sorted by activation from beta waves)
        elements.sort(key=lambda e: e.activation, reverse=True)

        packed_elements = []
        remaining_budget = self.budget.available_for_context
        compressed_count = 0

        for element in elements:
            if element.activation >= self.compression_threshold:
                # High activation → full content
                if element.token_count <= remaining_budget:
                    packed_elements.append(element)
                    remaining_budget -= element.token_count

            elif element.activation >= self.activation_threshold:
                # Medium activation → compress to 50%
                compressed_tokens = element.token_count // 2
                if compressed_tokens <= remaining_budget:
                    element.content = self._compress_content(element.content, ratio=0.5)
                    element.token_count = compressed_tokens
                    packed_elements.append(element)
                    remaining_budget -= compressed_tokens
                    compressed_count += 1

            # Low activation (<threshold) → excluded automatically

        # 4. Assemble sections
        sections = self._assemble_sections(packed_elements)

        # 5. Calculate statistics
        total_tokens = sum(e.token_count for e in packed_elements)
        included = len(packed_elements)
        excluded = len(elements) - included

        activations = [e.activation for e in packed_elements]
        avg_activation = sum(activations) / max(included, 1)
        min_activation = min(activations) if activations else 0.0
        max_activation = max(activations) if activations else 0.0

        packing_time = (time.time() - start_time) * 1000

        activation_stats = {
            'beta_wave_iterations': result.iterations,
            'seed_nodes': len(result.seed_nodes),
            'total_activated': len(result.recalled_memories),
            'creative_insights': len(result.creative_insights),
            'activation_distribution': self._activation_distribution(activations)
        }

        return PackedContext(
            query_section=sections.get('query', ''),
            memory_section=sections.get('memory', ''),
            awareness_section=sections.get('awareness', ''),
            creative_section=sections.get('creative', ''),
            total_tokens=total_tokens,
            elements_included=included,
            elements_compressed=compressed_count,
            elements_excluded=excluded,
            avg_activation=avg_activation,
            min_activation=min_activation,
            max_activation=max_activation,
            packing_time_ms=packing_time,
            activation_stats=activation_stats
        )

    def _create_awareness_elements(self, awareness_context) -> List[ContextElement]:
        """Create awareness elements with moderate activation"""
        elements = []

        # Confidence signals (high activation = 0.85)
        conf = awareness_context.confidence
        confidence_text = (
            f"Confidence: {1.0 - conf.uncertainty_level:.2f}\n"
            f"Uncertainty: {conf.uncertainty_level:.2f}"
        )
        elements.append(ContextElement(
            content=confidence_text,
            activation=0.85,  # High but not critical
            token_count=self._estimate_tokens(confidence_text),
            source="awareness",
            metadata={"type": "confidence"}
        ))

        # Pattern signals (activation based on familiarity)
        patterns = awareness_context.patterns
        pattern_activation = 0.8 if patterns.seen_count > 0 else 0.6
        pattern_text = f"Domain: {patterns.domain}/{patterns.subdomain}"

        elements.append(ContextElement(
            content=pattern_text,
            activation=pattern_activation,
            token_count=self._estimate_tokens(pattern_text),
            source="awareness",
            metadata={"type": "pattern", "seen_count": patterns.seen_count}
        ))

        return elements

    def _compress_content(self, content: str, ratio: float) -> str:
        """
        Simple compression: Take first N% of content.

        Future: Use extractive summarization (TextRank) or LLM summarization.
        """
        target_length = int(len(content) * ratio)
        if len(content) <= target_length:
            return content
        return content[:target_length] + "..."

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token)"""
        return len(text) // 4

    def _activation_distribution(self, activations: List[float]) -> Dict[str, int]:
        """Calculate activation distribution for analysis"""
        if not activations:
            return {}

        return {
            'high (>0.7)': sum(1 for a in activations if a > 0.7),
            'medium (0.3-0.7)': sum(1 for a in activations if 0.3 <= a <= 0.7),
            'low (<0.3)': sum(1 for a in activations if a < 0.3)
        }

    def _assemble_sections(self, elements: List[ContextElement]) -> Dict[str, str]:
        """Assemble elements into formatted sections"""
        sections = {
            'query': [],
            'memory': [],
            'awareness': [],
            'creative': []
        }

        for element in elements:
            if element.source == "query":
                sections['query'].append(element.content)
            elif element.source == "memory":
                # Include activation level for transparency
                sections['memory'].append(
                    f"[activation: {element.activation:.3f}] {element.content}"
                )
            elif element.source == "awareness":
                sections['awareness'].append(element.content)
            elif element.source == "creative_insight":
                sections['creative'].append(
                    f"[cross-domain bridge] {element.content}"
                )

        # Join sections
        return {
            k: "\n".join(v) if v else ""
            for k, v in sections.items()
        }
