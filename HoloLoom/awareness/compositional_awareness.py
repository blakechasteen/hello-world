"""
Compositional Awareness Layer - Phase 5 Integration

Provides real-time linguistic intelligence by combining:
- X-bar syntactic analysis (Universal Grammar)
- Merge compositional patterns (cached semantics)
- Confidence signals (cache hit/miss rates)
- AwarenessGraph integration (semantic positioning)

This layer feeds BOTH internal reasoning and external response streams.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

try:
    from HoloLoom.motif.xbar_chunker import UniversalGrammarChunker, XBarPhrase
    from HoloLoom.warp.merge import MergeOperator
    from HoloLoom.performance.compositional_cache import CompositionalCache
    from HoloLoom.memory.awareness_graph import AwarenessGraph
    UG_AVAILABLE = True
except ImportError:
    UG_AVAILABLE = False
    XBarPhrase = None


@dataclass
class StructuralAwareness:
    """Syntactic structure analysis from X-bar theory"""
    
    # Parse structure
    phrase_type: str              # "NP", "VP", "WH_QUESTION", etc.
    head_word: str                # Syntactic head
    x_bar_tree: Optional[Dict] = None
    
    # Query classification
    is_question: bool = False
    question_type: Optional[str] = None  # "WHAT", "HOW", "WHY"
    expects_definition: bool = False
    expects_procedure: bool = False
    expects_explanation: bool = False
    expects_comparison: bool = False
    
    # Linguistic features
    has_uncertainty: bool = False    # "maybe", "possibly"
    has_negation: bool = False      # "not", "never"
    has_comparison: bool = False    # "better than", "vs"
    
    # Response recommendations
    suggested_response_type: str = "STATEMENT"  # or "DEFINITION", "LIST", "EXPLANATION"


@dataclass
class PatternInfo:
    """Information about a compositional pattern"""
    seen_count: int
    confidence: float
    typical_contexts: List[str] = field(default_factory=list)


@dataclass
class CompositionalPatterns:
    """Patterns from compositional cache"""
    
    # Primary phrase analysis
    phrase: str
    seen_count: int = 0
    confidence: float = 0.0
    
    # Constituent patterns
    constituent_patterns: Dict[str, PatternInfo] = field(default_factory=dict)
    
    # Context analysis
    domain: str = "UNKNOWN"
    subdomain: str = "UNKNOWN"
    
    # Success patterns (from cache history)
    successful_responses: List[str] = field(default_factory=list)
    unsuccessful_responses: List[str] = field(default_factory=list)
    
    # Typical compositional contexts
    typical_compositions: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class ConfidenceSignals:
    """Confidence metrics from cache behavior"""
    
    # Cache status
    overall_cache_hit_rate: float = 0.0
    query_cache_status: str = "COLD_MISS"  # "HOT_HIT", "WARM_HIT", "PARTIAL_HIT", "COLD_MISS"
    
    # Compositional analysis
    compositions_cached: int = 0
    compositions_novel: int = 0
    
    # Uncertainty quantification
    uncertainty_level: float = 1.0  # 0.0 (certain) to 1.0 (uncertain)
    knowledge_gap_detected: bool = False
    gap_location: Optional[str] = None
    
    # Nearest neighbors for novel concepts
    nearest_known_patterns: List[Tuple[str, float]] = field(default_factory=list)
    
    # Recommendations
    should_ask_clarification: bool = False
    suggested_clarification: Optional[str] = None


@dataclass
class InternalStreamGuidance:
    """Guidance for internal reasoning stream"""
    
    # Reasoning structure
    reasoning_structure: str = "ANALYZE → DECIDE → VERIFY"
    expected_steps: List[str] = field(default_factory=list)
    
    # Confidence-based strategies
    high_confidence_shortcuts: List[str] = field(default_factory=list)
    low_confidence_checks: List[str] = field(default_factory=list)
    
    # Pattern-based hints
    similar_past_reasoning: List[str] = field(default_factory=list)
    successful_reasoning_paths: List[str] = field(default_factory=list)
    failed_reasoning_paths: List[str] = field(default_factory=list)
    
    # Meta-reasoning flags
    should_use_analogy: bool = False
    should_break_down_problem: bool = False
    should_ask_clarification: bool = False


@dataclass
class ExternalStreamGuidance:
    """Guidance for external response stream"""
    
    # Tone & style
    confidence_tone: str = "neutral"  # "confident", "tentative", "clarifying"
    appropriate_hedging: List[str] = field(default_factory=list)
    should_acknowledge_uncertainty: bool = False
    
    # Structure & format
    response_structure: str = "STATEMENT"  # "DEFINITION", "EXPLANATION", "QUESTION", "LIST"
    expected_length: str = "medium"  # "short", "medium", "detailed"
    include_examples: bool = False
    include_caveats: bool = False
    
    # Pattern-based composition
    successful_phrasings: List[str] = field(default_factory=list)
    avoid_phrasings: List[str] = field(default_factory=list)
    
    # Uncertainty handling
    clarification_needed: bool = False
    clarification_question: Optional[str] = None
    fallback_response: Optional[str] = None


@dataclass
class UnifiedAwarenessContext:
    """Complete awareness context feeding both streams"""
    
    # Core awareness (shared by both streams)
    structural: StructuralAwareness
    patterns: CompositionalPatterns
    confidence: ConfidenceSignals
    
    # Stream-specific guidance
    internal_guidance: InternalStreamGuidance
    external_guidance: ExternalStreamGuidance
    
    # Integration with AwarenessGraph
    semantic_position: Optional[np.ndarray] = None
    dominant_dimensions: List[str] = field(default_factory=list)
    trajectory_shift: float = 0.0
    
    # Statistics
    cache_statistics: Dict[str, Any] = field(default_factory=dict)


class CompositionalAwarenessLayer:
    """
    Real-time linguistic awareness layer for LLM guidance.
    
    Combines Phase 5 compositional intelligence with existing AwarenessGraph
    to provide unified awareness context for both internal reasoning and
    external response streams.
    """
    
    def __init__(
        self,
        ug_chunker = None,
        merge_operator = None,
        compositional_cache = None,
        awareness_graph = None
    ):
        """
        Initialize compositional awareness layer.
        
        Args:
            ug_chunker: Universal Grammar chunker (X-bar theory)
            merge_operator: Merge operator (compositional semantics)
            compositional_cache: 3-tier compositional cache
            awareness_graph: Existing AwarenessGraph for semantic tracking
        """
        self.ug_chunker = ug_chunker
        self.merge_operator = merge_operator
        self.cache = compositional_cache
        self.awareness = awareness_graph
        
        # Pattern history (for learning)
        self.pattern_history: Dict[str, PatternInfo] = {}
        
    async def get_unified_context(
        self,
        query: str,
        full_analysis: bool = True
    ) -> UnifiedAwarenessContext:
        """
        Generate unified awareness context for both streams.
        
        This is the main entry point that feeds awareness to:
        - Internal reasoning stream
        - External response stream
        
        Args:
            query: User query to analyze
            full_analysis: Whether to do full compositional analysis
            
        Returns:
            UnifiedAwarenessContext with all awareness signals
        """
        
        # 1. Structural analysis (X-bar)
        structural = await self._analyze_structure(query)
        
        # 2. Compositional pattern recognition
        patterns = await self._recognize_patterns(query)
        
        # 3. Confidence computation (cache statistics)
        confidence = await self._compute_confidence(query, patterns)
        
        # 4. Generate stream-specific guidance
        internal_guidance = self._generate_internal_guidance(
            structural, patterns, confidence
        )
        external_guidance = self._generate_external_guidance(
            structural, patterns, confidence
        )
        
        # 5. Integrate with AwarenessGraph (if available)
        semantic_position = None
        dominant_dimensions = []
        trajectory_shift = 0.0
        
        if self.awareness is not None:
            perception = await self.awareness.perceive(query)
            semantic_position = perception.position
            dominant_dimensions = perception.dominant_dimensions
            trajectory_shift = perception.shift_magnitude
        
        # 6. Gather cache statistics
        cache_stats = self._gather_cache_statistics()
        
        return UnifiedAwarenessContext(
            structural=structural,
            patterns=patterns,
            confidence=confidence,
            internal_guidance=internal_guidance,
            external_guidance=external_guidance,
            semantic_position=semantic_position,
            dominant_dimensions=dominant_dimensions,
            trajectory_shift=trajectory_shift,
            cache_statistics=cache_stats
        )
    
    async def _analyze_structure(self, query: str) -> StructuralAwareness:
        """Analyze syntactic structure using X-bar theory"""
        
        # Parse with UG chunker if available
        if self.ug_chunker is not None and UG_AVAILABLE:
            phrases = self.ug_chunker.chunk(query)
            
            if phrases:
                main_phrase = phrases[0]
                
                # Detect question type
                is_question = query.strip().endswith('?')
                question_type = None
                
                if is_question:
                    lower_query = query.lower()
                    if lower_query.startswith('what'):
                        question_type = "WHAT"
                    elif lower_query.startswith('how'):
                        question_type = "HOW"
                    elif lower_query.startswith('why'):
                        question_type = "WHY"
                    elif lower_query.startswith('when'):
                        question_type = "WHEN"
                    elif lower_query.startswith('where'):
                        question_type = "WHERE"
                
                return StructuralAwareness(
                    phrase_type=main_phrase.category if hasattr(main_phrase, 'category') else "NP",
                    head_word=main_phrase.head if hasattr(main_phrase, 'head') else query.split()[0],
                    is_question=is_question,
                    question_type=question_type,
                    expects_definition=question_type == "WHAT" and "is" in query.lower(),
                    expects_procedure=question_type == "HOW",
                    expects_explanation=question_type == "WHY",
                    expects_comparison="difference" in query.lower() or "vs" in query.lower(),
                    suggested_response_type="DEFINITION" if question_type == "WHAT" else "EXPLANATION"
                )
        
        # Fallback: simple heuristic analysis
        lower_query = query.lower()
        is_question = query.strip().endswith('?')
        
        return StructuralAwareness(
            phrase_type="QUERY" if is_question else "STATEMENT",
            head_word=query.split()[0] if query.split() else "",
            is_question=is_question,
            expects_definition="what is" in lower_query,
            expects_explanation="why" in lower_query or "how" in lower_query,
            suggested_response_type="DEFINITION" if "what is" in lower_query else "EXPLANATION"
        )
    
    async def _recognize_patterns(self, query: str) -> CompositionalPatterns:
        """Recognize compositional patterns from cache"""
        
        # Extract key phrases from query
        words = query.lower().strip('?').split()
        
        # Check cache for main phrase
        seen_count = 0
        confidence = 0.0
        constituent_patterns = {}
        
        if self.cache is not None:
            # Check merge cache for compositional patterns
            cache_stats = self.cache.get_statistics()
            
            # Get hit rate as confidence proxy
            merge_stats = cache_stats.get('merge_cache', {})
            hit_rate = merge_stats.get('hit_rate', 0.0)
            confidence = hit_rate
            
            # Count appearances in pattern history
            if query in self.pattern_history:
                pattern_info = self.pattern_history[query]
                seen_count = pattern_info.seen_count
                confidence = pattern_info.confidence
        
        # Determine domain (simple heuristic for now)
        domain = "GENERAL"
        subdomain = "UNKNOWN"
        
        if any(word in words for word in ['ball', 'toy', 'game']):
            domain = "PHYSICAL_OBJECTS"
            subdomain = "SPORTS/TOYS"
        elif any(word in words for word in ['quantum', 'physics', 'science']):
            domain = "SCIENTIFIC"
            subdomain = "PHYSICS"
        elif any(word in words for word in ['program', 'code', 'function']):
            domain = "TECHNICAL"
            subdomain = "PROGRAMMING"
        
        return CompositionalPatterns(
            phrase=query,
            seen_count=seen_count,
            confidence=confidence,
            constituent_patterns=constituent_patterns,
            domain=domain,
            subdomain=subdomain
        )
    
    async def _compute_confidence(
        self,
        query: str,
        patterns: CompositionalPatterns
    ) -> ConfidenceSignals:
        """Compute confidence signals from cache behavior"""
        
        # Default: low confidence (cold miss)
        uncertainty = 1.0
        cache_status = "COLD_MISS"
        
        # Check pattern history for this specific query
        pattern_info = self.pattern_history.get(query)
        
        if self.cache is not None:
            cache_stats = self.cache.get_statistics()
            
            # Get overall hit rates
            parse_hit_rate = cache_stats.get('parse_cache', {}).get('hit_rate', 0.0)
            merge_hit_rate = cache_stats.get('merge_cache', {}).get('hit_rate', 0.0)
            overall_hit_rate = (parse_hit_rate + merge_hit_rate) / 2.0
            
            # Determine cache status
            if overall_hit_rate > 0.8:
                cache_status = "HOT_HIT"
                uncertainty = 0.1
            elif overall_hit_rate > 0.5:
                cache_status = "WARM_HIT"
                uncertainty = 0.3
            elif overall_hit_rate > 0.2:
                cache_status = "PARTIAL_HIT"
                uncertainty = 0.6
            else:
                cache_status = "COLD_MISS"
                uncertainty = 0.9
        elif pattern_info and hasattr(pattern_info, 'confidence'):
            # Use pattern history confidence (for demos without full cache)
            history_confidence = pattern_info.confidence
            
            # Map confidence to uncertainty and cache status
            if history_confidence > 0.8:
                cache_status = "HOT_HIT"
                uncertainty = 0.1
            elif history_confidence > 0.6:
                cache_status = "WARM_HIT"
                uncertainty = 0.3
            elif history_confidence > 0.3:
                cache_status = "PARTIAL_HIT"
                uncertainty = 0.6
            else:
                cache_status = "COLD_MISS"
                uncertainty = 0.9
        
        # Pattern-based confidence (from patterns object)
        if patterns.seen_count > 10:
            uncertainty = min(uncertainty, 0.2)  # High familiarity
        elif patterns.seen_count > 3:
            uncertainty = min(uncertainty, 0.5)  # Some familiarity
        
        # Detect knowledge gap
        knowledge_gap = (
            cache_status == "COLD_MISS" and
            pattern_info is None and
            patterns.seen_count == 0
        )
        
        # Should ask clarification if very uncertain
        should_clarify = uncertainty > 0.7 or knowledge_gap
        
        return ConfidenceSignals(
            overall_cache_hit_rate=patterns.confidence,
            query_cache_status=cache_status,
            uncertainty_level=uncertainty,
            knowledge_gap_detected=knowledge_gap,
            should_ask_clarification=should_clarify,
            suggested_clarification="Could you provide more context?" if should_clarify else None
        )
    
    def _generate_internal_guidance(
        self,
        structural: StructuralAwareness,
        patterns: CompositionalPatterns,
        confidence: ConfidenceSignals
    ) -> InternalStreamGuidance:
        """Generate guidance for internal reasoning stream"""
        
        # Base reasoning structure
        if structural.expects_definition:
            reasoning_structure = "DEFINITION → EXAMPLE → CONTEXT"
        elif structural.expects_explanation:
            reasoning_structure = "UNDERSTAND → REASON → EXPLAIN"
        elif structural.expects_comparison:
            reasoning_structure = "IDENTIFY → COMPARE → CONTRAST"
        else:
            reasoning_structure = "ANALYZE → DECIDE → RESPOND"
        
        # Confidence-based shortcuts
        shortcuts = []
        checks = []
        
        if confidence.uncertainty_level < 0.3:
            shortcuts = ["skip_verification", "direct_answer"]
        else:
            checks = ["verify_understanding", "check_alternatives"]
        
        if confidence.uncertainty_level > 0.7:
            checks.append("consider_clarification")
        
        return InternalStreamGuidance(
            reasoning_structure=reasoning_structure,
            expected_steps=["parse", "analyze", "decide"],
            high_confidence_shortcuts=shortcuts,
            low_confidence_checks=checks,
            should_ask_clarification=confidence.should_ask_clarification
        )
    
    def _generate_external_guidance(
        self,
        structural: StructuralAwareness,
        patterns: CompositionalPatterns,
        confidence: ConfidenceSignals
    ) -> ExternalStreamGuidance:
        """Generate guidance for external response stream"""
        
        # Tone based on confidence
        if confidence.uncertainty_level < 0.3:
            tone = "confident"
            hedging = []
        elif confidence.uncertainty_level < 0.6:
            tone = "neutral"
            hedging = ["generally", "typically"]
        else:
            tone = "tentative"
            hedging = ["perhaps", "possibly", "might"]
        
        # Override tone if clarification needed
        if confidence.should_ask_clarification:
            tone = "clarifying"
        
        # Response structure
        response_structure = structural.suggested_response_type
        
        # Length based on confidence and complexity
        if confidence.uncertainty_level < 0.3:
            length = "short"  # Direct answer
        elif patterns.domain == "SCIENTIFIC":
            length = "detailed"  # Complex topics need more
        else:
            length = "medium"
        
        return ExternalStreamGuidance(
            confidence_tone=tone,
            appropriate_hedging=hedging,
            should_acknowledge_uncertainty=confidence.uncertainty_level > 0.7,
            response_structure=response_structure,
            expected_length=length,
            include_examples=patterns.domain == "PHYSICAL_OBJECTS",
            clarification_needed=confidence.should_ask_clarification,
            clarification_question=confidence.suggested_clarification
        )
    
    def _gather_cache_statistics(self) -> Dict[str, Any]:
        """Gather cache statistics for context"""
        
        if self.cache is not None:
            return self.cache.get_statistics()
        
        return {}
    
    async def update_from_generation(
        self,
        query: str,
        internal_reasoning: str,
        external_response: str,
        user_feedback: Optional[Dict] = None
    ):
        """
        Update awareness from LLM generation (feedback loop).
        
        Args:
            query: Original query
            internal_reasoning: Internal reasoning stream output
            external_response: External response stream output
            user_feedback: Optional user feedback (thumbs up/down, rating)
        """
        
        # Update pattern history
        success = user_feedback.get('success', True) if user_feedback else True
        
        if query in self.pattern_history:
            pattern_info = self.pattern_history[query]
            pattern_info.seen_count += 1
            
            # Update confidence based on success
            if success:
                pattern_info.confidence = min(1.0, pattern_info.confidence + 0.1)
            else:
                pattern_info.confidence = max(0.0, pattern_info.confidence - 0.1)
        else:
            self.pattern_history[query] = PatternInfo(
                seen_count=1,
                confidence=0.8 if success else 0.3
            )
        
        # Update AwarenessGraph if available
        if self.awareness is not None:
            content = f"Q: {query}\nA: {external_response}"
            perception = await self.awareness.perceive(query)
            await self.awareness.remember(
                content=content,
                perception=perception,
                context={
                    'success': success,
                    'rating': user_feedback.get('rating') if user_feedback else None
                }
            )


def format_awareness_for_prompt(context: UnifiedAwarenessContext) -> str:
    """
    Format awareness context for inclusion in LLM prompt.
    
    Returns human-readable awareness context that can be injected
    into system prompts or used for tool calls.
    """
    
    lines = ["[AWARENESS CONTEXT]"]
    lines.append("")
    
    # Structural awareness
    lines.append("Structure:")
    lines.append(f"  Type: {context.structural.phrase_type}")
    if context.structural.is_question:
        lines.append(f"  Question type: {context.structural.question_type or 'unknown'}")
        lines.append(f"  Expected response: {context.structural.suggested_response_type}")
    lines.append("")
    
    # Patterns
    lines.append("Patterns:")
    lines.append(f"  Phrase: \"{context.patterns.phrase}\"")
    lines.append(f"  Familiarity: {context.patterns.seen_count}× seen")
    lines.append(f"  Confidence: {context.patterns.confidence:.2f}")
    lines.append(f"  Domain: {context.patterns.domain}/{context.patterns.subdomain}")
    lines.append("")
    
    # Confidence
    lines.append("Confidence:")
    lines.append(f"  Cache status: {context.confidence.query_cache_status}")
    lines.append(f"  Uncertainty: {context.confidence.uncertainty_level:.2f}")
    if context.confidence.knowledge_gap_detected:
        lines.append(f"  ⚠️ Knowledge gap detected")
    if context.confidence.should_ask_clarification:
        lines.append(f"  ⚠️ Recommend clarification: {context.confidence.suggested_clarification}")
    lines.append("")
    
    # Guidance summary
    lines.append("Guidance:")
    lines.append(f"  Tone: {context.external_guidance.confidence_tone}")
    lines.append(f"  Structure: {context.external_guidance.response_structure}")
    lines.append(f"  Length: {context.external_guidance.expected_length}")
    if context.external_guidance.appropriate_hedging:
        lines.append(f"  Hedging: {', '.join(context.external_guidance.appropriate_hedging)}")
    
    return "\n".join(lines)
