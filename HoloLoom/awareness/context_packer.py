"""
Smart Context Assembly - The Intelligent Feed

Packs awareness + memory + query into optimal LLM prompts with:
- Importance-based token budgeting
- Hierarchical compression (summary → detail → full)
- Temporal weighting (recent + relevant + resonant)
- Adaptive depth based on confidence

This is the bridge between consciousness (awareness) and generation (LLM).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import time


class ContextImportance(Enum):
    """Importance levels for context elements"""
    CRITICAL = 1.0      # Must include (query, high-confidence patterns)
    HIGH = 0.8          # Should include (recent memories, relevant patterns)
    MEDIUM = 0.5        # Nice to have (related concepts, background)
    LOW = 0.2           # Optional (distant associations, metadata)


class CompressionLevel(Enum):
    """Compression strategies"""
    FULL = "full"           # Complete content
    DETAILED = "detailed"   # Key points + examples
    SUMMARY = "summary"     # One-sentence summary
    MINIMAL = "minimal"     # Just metadata


@dataclass
class ContextElement:
    """Single piece of context to pack"""
    content: str
    importance: float  # 0.0-1.0
    token_count: int
    source: str  # "awareness", "memory", "pattern", "query"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Compression alternatives
    summary: Optional[str] = None
    detailed: Optional[str] = None
    
    def compress(self, level: CompressionLevel) -> str:
        """Get compressed version of content"""
        if level == CompressionLevel.FULL:
            return self.content
        elif level == CompressionLevel.DETAILED and self.detailed:
            return self.detailed
        elif level == CompressionLevel.SUMMARY and self.summary:
            return self.summary
        elif level == CompressionLevel.MINIMAL:
            return f"[{self.source}: {len(self.content)} chars]"
        return self.content
    
    def estimate_tokens(self, level: CompressionLevel) -> int:
        """Estimate tokens for compressed version"""
        compressed = self.compress(level)
        # Rough estimate: 1 token ≈ 4 characters
        return len(compressed) // 4


@dataclass
class TokenBudget:
    """Token budget constraints"""
    total: int = 8000           # Total token budget
    reserved_for_query: int = 500  # Reserve for query/instructions
    reserved_for_response: int = 1000  # Reserve for LLM response
    
    @property
    def available_for_context(self) -> int:
        """Tokens available for context"""
        return self.total - self.reserved_for_query - self.reserved_for_response


@dataclass
class PackedContext:
    """Assembled context ready for LLM"""
    
    # Core sections
    awareness_section: str
    memory_section: str
    pattern_section: str
    query_section: str
    
    # Metadata
    total_tokens: int
    elements_included: int
    elements_compressed: int
    elements_excluded: int
    
    # Importance statistics
    avg_importance: float
    min_importance: float
    
    # Provenance
    packing_time_ms: float
    compression_stats: Dict[str, int]
    
    def format_for_llm(self, include_metadata: bool = False) -> str:
        """Format packed context for LLM prompt"""
        sections = []
        
        # Awareness context (confidence, structure, patterns)
        if self.awareness_section:
            sections.append("# AWARENESS CONTEXT")
            sections.append(self.awareness_section)
            sections.append("")
        
        # Memory retrieval (relevant past interactions)
        if self.memory_section:
            sections.append("# RELEVANT MEMORIES")
            sections.append(self.memory_section)
            sections.append("")
        
        # Pattern analysis (compositional patterns)
        if self.pattern_section:
            sections.append("# RECOGNIZED PATTERNS")
            sections.append(self.pattern_section)
            sections.append("")
        
        # Query
        sections.append("# QUERY")
        sections.append(self.query_section)
        
        # Optional metadata
        if include_metadata:
            sections.append("")
            sections.append("# PACKING METADATA")
            sections.append(f"Total tokens: {self.total_tokens}")
            sections.append(f"Elements: {self.elements_included} included, "
                          f"{self.elements_compressed} compressed, "
                          f"{self.elements_excluded} excluded")
            sections.append(f"Importance: avg={self.avg_importance:.2f}, min={self.min_importance:.2f}")
        
        return "\n".join(sections)


class SmartContextPacker:
    """
    Intelligent context assembly with token optimization.
    
    Strategy:
    1. Collect all potential context elements
    2. Score by importance (awareness signals + recency + relevance)
    3. Optimize packing with token budget
    4. Apply hierarchical compression
    5. Ensure critical elements always included
    """
    
    def __init__(
        self,
        token_budget: Optional[TokenBudget] = None,
        min_importance_threshold: float = 0.2
    ):
        """
        Initialize context packer.
        
        Args:
            token_budget: Token budget constraints
            min_importance_threshold: Minimum importance to include
        """
        self.budget = token_budget or TokenBudget()
        self.min_importance = min_importance_threshold
    
    async def pack_context(
        self,
        query: str,
        awareness_context,
        memory_results: Optional[List[Any]] = None,
        max_memories: int = 10
    ) -> PackedContext:
        """
        Pack context optimally for LLM generation.
        
        Args:
            query: User query
            awareness_context: UnifiedAwarenessContext from awareness layer
            memory_results: Optional memory retrieval results
            max_memories: Maximum memories to include
            
        Returns:
            PackedContext ready for LLM
        """
        start_time = time.time()
        
        # 1. Collect all context elements
        elements = []
        
        # Query element (always critical)
        elements.append(ContextElement(
            content=query,
            importance=ContextImportance.CRITICAL.value,
            token_count=len(query) // 4,
            source="query",
            metadata={"type": "user_query"}
        ))
        
        # Awareness elements
        awareness_elements = self._extract_awareness_elements(awareness_context)
        elements.extend(awareness_elements)
        
        # Memory elements
        if memory_results:
            memory_elements = self._extract_memory_elements(
                memory_results,
                max_count=max_memories,
                query=query
            )
            elements.extend(memory_elements)
        
        # 2. Score and sort by importance
        elements = self._score_elements(elements, awareness_context)
        elements.sort(key=lambda e: e.importance, reverse=True)
        
        # 3. Optimize packing with token budget
        packed_elements, compression_stats = self._optimize_packing(
            elements,
            self.budget.available_for_context
        )
        
        # 4. Assemble sections
        sections = self._assemble_sections(packed_elements)
        
        # 5. Calculate statistics
        total_tokens = sum(e.token_count for e in packed_elements)
        included = len(packed_elements)
        compressed = compression_stats.get('compressed', 0)
        excluded = len(elements) - included
        
        avg_importance = sum(e.importance for e in packed_elements) / max(included, 1)
        min_importance_included = min((e.importance for e in packed_elements), default=0.0)
        
        packing_time = (time.time() - start_time) * 1000
        
        return PackedContext(
            awareness_section=sections.get('awareness', ''),
            memory_section=sections.get('memory', ''),
            pattern_section=sections.get('pattern', ''),
            query_section=sections.get('query', ''),
            total_tokens=total_tokens,
            elements_included=included,
            elements_compressed=compressed,
            elements_excluded=excluded,
            avg_importance=avg_importance,
            min_importance=min_importance_included,
            packing_time_ms=packing_time,
            compression_stats=compression_stats
        )
    
    def _extract_awareness_elements(self, awareness_context) -> List[ContextElement]:
        """Extract context elements from awareness context"""
        elements = []
        
        # Confidence signals (CRITICAL)
        conf = awareness_context.confidence
        confidence_text = (
            f"Confidence: {1.0 - conf.uncertainty_level:.2f}\n"
            f"Cache Status: {conf.query_cache_status}\n"
            f"Uncertainty: {conf.uncertainty_level:.2f}\n"
            f"Knowledge Gap: {'Yes' if conf.knowledge_gap_detected else 'No'}"
        )
        
        elements.append(ContextElement(
            content=confidence_text,
            importance=ContextImportance.CRITICAL.value,
            token_count=len(confidence_text) // 4,
            source="awareness",
            metadata={"type": "confidence_signals"},
            summary=f"Confidence: {1.0 - conf.uncertainty_level:.2f}"
        ))
        
        # Structural analysis (HIGH)
        struct = awareness_context.structural
        structural_text = (
            f"Structure: {struct.phrase_type}\n"
            f"Is Question: {struct.is_question}\n"
            f"Expected Response: {struct.suggested_response_type}"
        )
        
        elements.append(ContextElement(
            content=structural_text,
            importance=ContextImportance.HIGH.value,
            token_count=len(structural_text) // 4,
            source="awareness",
            metadata={"type": "structural_analysis"},
            summary=f"Type: {struct.suggested_response_type}"
        ))
        
        # Pattern analysis (HIGH if seen before, MEDIUM otherwise)
        patterns = awareness_context.patterns
        pattern_importance = (
            ContextImportance.HIGH.value if patterns.seen_count > 0
            else ContextImportance.MEDIUM.value
        )
        
        pattern_text = (
            f"Domain: {patterns.domain}/{patterns.subdomain}\n"
            f"Familiarity: {patterns.seen_count}× seen\n"
            f"Confidence: {patterns.confidence:.2f}"
        )
        
        elements.append(ContextElement(
            content=pattern_text,
            importance=pattern_importance,
            token_count=len(pattern_text) // 4,
            source="awareness",
            metadata={"type": "pattern_analysis"},
            summary=f"Domain: {patterns.domain} ({patterns.seen_count}× seen)"
        ))
        
        return elements
    
    def _extract_memory_elements(
        self,
        memory_results: List[Any],
        max_count: int,
        query: str
    ) -> List[ContextElement]:
        """Extract context elements from memory retrieval"""
        elements = []
        
        for i, memory in enumerate(memory_results[:max_count]):
            # Calculate importance based on:
            # - Relevance score (if available)
            # - Recency (more recent = higher)
            # - Position in results (earlier = higher)
            
            base_importance = 0.8 - (i * 0.05)  # Decay by position
            base_importance = max(base_importance, 0.3)
            
            # Extract memory text
            memory_text = self._extract_memory_text(memory)
            
            elements.append(ContextElement(
                content=memory_text,
                importance=base_importance,
                token_count=len(memory_text) // 4,
                source="memory",
                metadata={
                    "type": "retrieved_memory",
                    "position": i,
                    "memory_id": getattr(memory, 'id', f'mem_{i}')
                },
                summary=memory_text[:100] + "..." if len(memory_text) > 100 else memory_text,
                detailed=memory_text[:300] + "..." if len(memory_text) > 300 else memory_text
            ))
        
        return elements
    
    def _extract_memory_text(self, memory: Any) -> str:
        """Extract text from memory object (handles different formats)"""
        if isinstance(memory, dict):
            return memory.get('text', memory.get('content', str(memory)))
        elif hasattr(memory, 'text'):
            return memory.text
        elif hasattr(memory, 'content'):
            return memory.content
        else:
            return str(memory)
    
    def _score_elements(
        self,
        elements: List[ContextElement],
        awareness_context
    ) -> List[ContextElement]:
        """
        Adjust importance scores based on awareness signals.
        
        Strategy:
        - Boost elements related to high-uncertainty areas
        - Boost elements from familiar domains
        - Boost recent and relevant memories
        """
        conf = awareness_context.confidence
        patterns = awareness_context.patterns
        
        for element in elements:
            # Boost awareness elements when uncertain
            if element.source == "awareness" and conf.uncertainty_level > 0.7:
                element.importance = min(1.0, element.importance * 1.2)
            
            # Boost pattern elements when familiar
            if element.source == "awareness" and element.metadata.get('type') == 'pattern_analysis':
                if patterns.seen_count > 10:
                    element.importance = min(1.0, element.importance * 1.1)
            
            # Boost memory elements from same domain
            if element.source == "memory":
                # Check if memory content relates to recognized domain
                content_lower = element.content.lower()
                if patterns.domain.lower() in content_lower:
                    element.importance = min(1.0, element.importance * 1.15)
        
        return elements
    
    def _optimize_packing(
        self,
        elements: List[ContextElement],
        token_budget: int
    ) -> Tuple[List[ContextElement], Dict[str, int]]:
        """
        Optimize element packing within token budget.
        
        Strategy (greedy with compression):
        1. Always include CRITICAL elements (full)
        2. Include HIGH elements (compressed if needed)
        3. Include MEDIUM/LOW elements if budget allows
        4. Apply hierarchical compression to fit budget
        """
        packed = []
        remaining_budget = token_budget
        compression_stats = {
            'full': 0,
            'detailed': 0,
            'summary': 0,
            'minimal': 0,
            'compressed': 0
        }
        
        # First pass: Include critical elements (always full)
        for element in elements:
            if element.importance >= ContextImportance.CRITICAL.value:
                if element.token_count <= remaining_budget:
                    packed.append(element)
                    remaining_budget -= element.token_count
                    compression_stats['full'] += 1
        
        # Second pass: Include high-importance elements (compress if needed)
        for element in elements:
            if (ContextImportance.HIGH.value <= element.importance < ContextImportance.CRITICAL.value
                and element not in packed):
                
                # Try full first
                if element.token_count <= remaining_budget:
                    packed.append(element)
                    remaining_budget -= element.token_count
                    compression_stats['full'] += 1
                # Try detailed
                elif element.detailed:
                    detailed_tokens = element.estimate_tokens(CompressionLevel.DETAILED)
                    if detailed_tokens <= remaining_budget:
                        element.content = element.compress(CompressionLevel.DETAILED)
                        element.token_count = detailed_tokens
                        packed.append(element)
                        remaining_budget -= detailed_tokens
                        compression_stats['detailed'] += 1
                        compression_stats['compressed'] += 1
                # Try summary
                elif element.summary:
                    summary_tokens = element.estimate_tokens(CompressionLevel.SUMMARY)
                    if summary_tokens <= remaining_budget:
                        element.content = element.compress(CompressionLevel.SUMMARY)
                        element.token_count = summary_tokens
                        packed.append(element)
                        remaining_budget -= summary_tokens
                        compression_stats['summary'] += 1
                        compression_stats['compressed'] += 1
        
        # Third pass: Include medium/low elements (aggressively compress)
        for element in elements:
            if element.importance < ContextImportance.HIGH.value and element not in packed:
                
                # Only include if summary fits
                if element.summary:
                    summary_tokens = element.estimate_tokens(CompressionLevel.SUMMARY)
                    if summary_tokens <= remaining_budget:
                        element.content = element.compress(CompressionLevel.SUMMARY)
                        element.token_count = summary_tokens
                        packed.append(element)
                        remaining_budget -= summary_tokens
                        compression_stats['summary'] += 1
                        compression_stats['compressed'] += 1
        
        return packed, compression_stats
    
    def _assemble_sections(self, elements: List[ContextElement]) -> Dict[str, str]:
        """Assemble elements into formatted sections"""
        sections = {
            'awareness': [],
            'memory': [],
            'pattern': [],
            'query': []
        }
        
        for element in elements:
            if element.source == "query":
                sections['query'].append(element.content)
            elif element.source == "awareness":
                if element.metadata.get('type') == 'pattern_analysis':
                    sections['pattern'].append(element.content)
                else:
                    sections['awareness'].append(element.content)
            elif element.source == "memory":
                sections['memory'].append(f"- {element.content}")
        
        # Join sections
        return {
            k: "\n".join(v) if v else ""
            for k, v in sections.items()
        }
