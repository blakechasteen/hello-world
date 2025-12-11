"""
Query Capability Analyzer - Phase 7.2
======================================

Automatically maps query text to required AgentCapabilities.
Enables automatic task delegation without explicit capability specification.

Key Features:
- Keyword-based capability detection (<1ms overhead)
- Multi-capability detection for complex queries
- Complexity-aware ensemble triggering
- Extensible keyword mappings

Author: Claude Code
Date: 2025-12-09
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
import re

from .protocol import AgentCapability, QueryComplexity


@dataclass
class CapabilityAnalysis:
    """
    Result of analyzing a query for required capabilities.

    Attributes:
        primary_capability: The main capability required
        secondary_capabilities: Additional capabilities that may help
        confidence: Confidence in the analysis (0.0-1.0)
        reasoning: Human-readable explanation of the analysis
        requires_ensemble: Whether multiple agents should be used
        complexity: Estimated query complexity
        detected_keywords: Keywords that triggered capability detection
    """
    primary_capability: AgentCapability
    secondary_capabilities: List[AgentCapability] = field(default_factory=list)
    confidence: float = 0.5
    reasoning: str = ""
    requires_ensemble: bool = False
    complexity: QueryComplexity = QueryComplexity.SIMPLE
    detected_keywords: Set[str] = field(default_factory=set)


class QueryCapabilityAnalyzer:
    """
    Analyzes queries to determine required agent capabilities.

    Uses keyword matching and pattern analysis to map queries to
    the AgentCapability enum. Fast path (<1ms) for common queries.

    Example:
        analyzer = QueryCapabilityAnalyzer()
        result = analyzer.analyze("Review this Python code for bugs")
        # result.primary_capability == AgentCapability.CODE_REVIEW
    """

    # Default keyword mappings for each capability
    # Format: capability -> list of keywords/patterns
    DEFAULT_KEYWORD_MAPPINGS: Dict[AgentCapability, List[str]] = {
        # Research & Analysis
        AgentCapability.RESEARCH: [
            "research", "explore", "investigate", "find out", "look up",
            "search for", "discover", "learn about", "study"
        ],
        AgentCapability.ANALYSIS: [
            "analyze", "analyse", "examine", "evaluate", "assess",
            "understand", "interpret", "breakdown", "break down"
        ],
        AgentCapability.VERIFICATION: [
            "verify", "validate", "check", "confirm", "fact-check",
            "fact check", "is it true", "is this correct", "accurate"
        ],

        # Generation & Synthesis
        AgentCapability.SUMMARIZATION: [
            "summarize", "summarise", "summary", "tldr", "tl;dr",
            "brief", "condense", "shorten", "key points"
        ],
        AgentCapability.GENERATION: [
            "generate", "create", "produce", "make", "build",
            "construct", "develop", "come up with"
        ],
        AgentCapability.SYNTHESIS: [
            "synthesize", "synthesise", "combine", "merge", "integrate",
            "bring together", "consolidate", "unify"
        ],

        # Code & Technical
        AgentCapability.CODE_REVIEW: [
            "review", "code review", "audit", "inspect", "look at this code",
            "check this code", "review this", "evaluate code"
        ],
        AgentCapability.CODE_GENERATION: [
            "write code", "code", "implement", "program", "script",
            "function", "class", "method", "algorithm", "snippet"
        ],
        AgentCapability.DEBUGGING: [
            "debug", "fix", "bug", "error", "issue", "problem",
            "not working", "broken", "wrong", "crash", "exception"
        ],

        # Planning & Coordination
        AgentCapability.PLANNING: [
            "plan", "strategy", "roadmap", "schedule", "outline",
            "steps", "approach", "how to", "procedure"
        ],
        AgentCapability.COORDINATION: [
            "coordinate", "orchestrate", "manage", "organize",
            "delegate", "distribute", "assign"
        ],
        AgentCapability.DECISION: [
            "decide", "choose", "select", "pick", "which",
            "should i", "recommend", "suggest", "best option"
        ],

        # Domain-Specific
        AgentCapability.MATH: [
            "calculate", "compute", "math", "equation", "formula",
            "solve", "prove", "theorem", "integral", "derivative"
        ],
        AgentCapability.WRITING: [
            "write", "draft", "compose", "author", "essay",
            "article", "blog", "story", "content", "copy"
        ],
        AgentCapability.QA: [
            "test", "testing", "quality", "qa", "assurance",
            "validation", "coverage", "regression"
        ],
    }

    # Complexity indicators
    COMPLEXITY_KEYWORDS = {
        # Research level
        "comprehensive": QueryComplexity.RESEARCH,
        "thorough": QueryComplexity.RESEARCH,
        "in-depth": QueryComplexity.RESEARCH,
        "detailed analysis": QueryComplexity.RESEARCH,
        "deep dive": QueryComplexity.RESEARCH,
        "tradeoffs": QueryComplexity.RESEARCH,
        "trade-offs": QueryComplexity.RESEARCH,
        "pros and cons": QueryComplexity.RESEARCH,

        # Complex level
        "compare": QueryComplexity.COMPLEX,
        "contrast": QueryComplexity.COMPLEX,
        "versus": QueryComplexity.COMPLEX,
        "vs": QueryComplexity.COMPLEX,
        "explain": QueryComplexity.COMPLEX,
        "elaborate": QueryComplexity.COMPLEX,
        "why": QueryComplexity.COMPLEX,
        "how does": QueryComplexity.COMPLEX,

        # Moderate level
        "describe": QueryComplexity.MODERATE,
        "what is": QueryComplexity.MODERATE,
        "tell me about": QueryComplexity.MODERATE,

        # Simple level (explicit)
        "define": QueryComplexity.SIMPLE,
        "meaning of": QueryComplexity.SIMPLE,
    }

    # Trivial patterns
    TRIVIAL_PATTERNS = [
        r'^(hi|hello|hey|yo)[\s!?]*$',
        r'^(thanks|thank you|thx)[\s!?.]*$',
        r'^(bye|goodbye|cya)[\s!?.]*$',
        r'^(ok|okay|sure|yes|no)[\s!?.]*$',
    ]

    # Multi-capability trigger words (suggests ensemble)
    MULTI_CAPABILITY_TRIGGERS = [
        "and then", "also", "as well as", "plus",
        "both", "multiple", "together"
    ]

    def __init__(self):
        """Initialize analyzer with default mappings."""
        self._keyword_mappings = dict(self.DEFAULT_KEYWORD_MAPPINGS)
        self._trivial_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.TRIVIAL_PATTERNS
        ]

        # Build reverse index: keyword -> capability for fast lookup
        self._keyword_to_capability: Dict[str, AgentCapability] = {}
        for capability, keywords in self._keyword_mappings.items():
            for keyword in keywords:
                self._keyword_to_capability[keyword.lower()] = capability

    def analyze(self, query: str) -> CapabilityAnalysis:
        """
        Analyze query to determine required capabilities.

        Args:
            query: The query text to analyze

        Returns:
            CapabilityAnalysis with detected capabilities and metadata
        """
        query_lower = query.lower().strip()

        # Check for trivial queries first (fast path)
        if self._is_trivial(query_lower):
            return CapabilityAnalysis(
                primary_capability=AgentCapability.GENERAL,
                confidence=0.95,
                reasoning="Trivial query (greeting/acknowledgment)",
                complexity=QueryComplexity.TRIVIAL
            )

        # Detect capabilities from keywords
        detected_capabilities: Dict[AgentCapability, float] = {}
        detected_keywords: Set[str] = set()

        for keyword, capability in self._keyword_to_capability.items():
            if keyword in query_lower:
                # Score based on keyword length (longer = more specific)
                score = len(keyword) / 20.0  # Normalize to ~0-1
                score = min(score + 0.3, 1.0)  # Base boost

                if capability not in detected_capabilities:
                    detected_capabilities[capability] = score
                else:
                    detected_capabilities[capability] = max(
                        detected_capabilities[capability],
                        score
                    )
                detected_keywords.add(keyword)

        # Detect complexity
        complexity = self._detect_complexity(query_lower)

        # Check for multi-capability triggers
        requires_ensemble = self._detect_ensemble_trigger(query_lower, detected_capabilities)

        # Sort capabilities by score
        sorted_capabilities = sorted(
            detected_capabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if not sorted_capabilities:
            # No specific capability detected, use GENERAL
            return CapabilityAnalysis(
                primary_capability=AgentCapability.GENERAL,
                secondary_capabilities=[],
                confidence=0.4,
                reasoning="No specific capability keywords detected, defaulting to general",
                requires_ensemble=False,
                complexity=complexity,
                detected_keywords=detected_keywords
            )

        # Primary capability is highest scored
        primary_cap, primary_score = sorted_capabilities[0]

        # Secondary capabilities (if score > 0.3 and not primary)
        secondary_caps = [
            cap for cap, score in sorted_capabilities[1:]
            if score > 0.3
        ]

        # Build reasoning
        reasoning = self._build_reasoning(
            primary_cap,
            secondary_caps,
            detected_keywords,
            complexity
        )

        # Determine if ensemble is needed
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.RESEARCH]:
            if len(sorted_capabilities) >= 2:
                requires_ensemble = True

        return CapabilityAnalysis(
            primary_capability=primary_cap,
            secondary_capabilities=secondary_caps[:3],  # Top 3 secondaries max
            confidence=min(primary_score, 0.95),
            reasoning=reasoning,
            requires_ensemble=requires_ensemble,
            complexity=complexity,
            detected_keywords=detected_keywords
        )

    def _is_trivial(self, query_lower: str) -> bool:
        """Check if query matches trivial patterns."""
        for pattern in self._trivial_patterns:
            if pattern.match(query_lower):
                return True
        return len(query_lower) <= 3 and query_lower.isalpha()

    def _detect_complexity(self, query_lower: str) -> QueryComplexity:
        """Detect query complexity from keywords and length."""
        # Check keyword indicators
        for keyword, complexity in self.COMPLEXITY_KEYWORDS.items():
            if keyword in query_lower:
                return complexity

        # Length-based heuristics
        word_count = len(query_lower.split())

        if word_count <= 3:
            return QueryComplexity.SIMPLE
        elif word_count <= 8:
            return QueryComplexity.MODERATE
        elif word_count <= 15:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.RESEARCH

    def _detect_ensemble_trigger(
        self,
        query_lower: str,
        capabilities: Dict[AgentCapability, float]
    ) -> bool:
        """Detect if query suggests multi-agent ensemble."""
        # Check for explicit multi-capability triggers
        for trigger in self.MULTI_CAPABILITY_TRIGGERS:
            if trigger in query_lower:
                return True

        # Multiple high-confidence capabilities suggest ensemble
        high_confidence = sum(1 for score in capabilities.values() if score > 0.5)
        return high_confidence >= 2

    def _build_reasoning(
        self,
        primary: AgentCapability,
        secondary: List[AgentCapability],
        keywords: Set[str],
        complexity: QueryComplexity
    ) -> str:
        """Build human-readable reasoning for the analysis."""
        parts = [f"Primary: {primary.value}"]

        if keywords:
            parts.append(f"Keywords: {', '.join(sorted(keywords)[:5])}")

        if secondary:
            sec_names = [c.value for c in secondary[:2]]
            parts.append(f"Secondary: {', '.join(sec_names)}")

        parts.append(f"Complexity: {complexity.value}")

        return " | ".join(parts)

    def get_capability_keywords(self) -> Dict[AgentCapability, List[str]]:
        """Get current keyword mappings for all capabilities."""
        return dict(self._keyword_mappings)

    def add_custom_mapping(
        self,
        keywords: List[str],
        capability: AgentCapability
    ) -> None:
        """
        Add custom keyword mappings for a capability.

        Args:
            keywords: List of keywords to map
            capability: Target capability
        """
        if capability not in self._keyword_mappings:
            self._keyword_mappings[capability] = []

        for keyword in keywords:
            keyword_lower = keyword.lower()
            self._keyword_mappings[capability].append(keyword_lower)
            self._keyword_to_capability[keyword_lower] = capability

    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "total_capabilities": len(AgentCapability),
            "mapped_capabilities": len(self._keyword_mappings),
            "total_keywords": sum(
                len(kws) for kws in self._keyword_mappings.values()
            ),
            "complexity_keywords": len(self.COMPLEXITY_KEYWORDS),
            "trivial_patterns": len(self.TRIVIAL_PATTERNS),
        }


# Convenience function
def analyze_query_capabilities(query: str) -> CapabilityAnalysis:
    """
    Convenience function to analyze query capabilities.

    Args:
        query: The query text to analyze

    Returns:
        CapabilityAnalysis with detected capabilities
    """
    analyzer = QueryCapabilityAnalyzer()
    return analyzer.analyze(query)
