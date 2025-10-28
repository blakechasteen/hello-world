"""
HoloLoom Narrative Analyzer Dashboard Integration API
======================================================
Standardized API for smart dashboard integration.

The dashboard uses this API to:
1. Analyze narrative structure (Campbell's Hero's Journey)
2. Detect characters and archetypes
3. Perform matryoshka depth analysis (5 levels)
4. Cross-domain narrative adaptation
5. Get system status and analytics
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from .intelligence import NarrativeIntelligence
    from .matryoshka_depth import MatryoshkaNarrativeDepth
    from .cross_domain_adapter import CrossDomainAdapter
    from .cache import NarrativeCache
except ImportError as e:
    # Graceful degradation
    print(f"Warning: Some narrative modules not available: {e}")
    NarrativeIntelligence = None
    MatryoshkaNarrativeDepth = None
    CrossDomainAdapter = None
    NarrativeCache = None


@dataclass
class NarrativeAnalysisResult:
    """Result of narrative structure analysis."""

    # Metadata
    timestamp: str
    duration_ms: float

    # Campbell's Hero's Journey
    primary_stage: str  # e.g., "ordeal", "return_with_elixir"
    stage_confidence: float
    all_stage_scores: Dict[str, float]

    # Characters
    detected_characters: List[Dict[str, Any]]  # name, confidence, mythology
    character_count: int

    # Archetypes
    primary_archetypes: List[str]  # Hero, Mentor, Shadow, etc.
    archetype_scores: Dict[str, float]

    # Narrative functions
    narrative_function: str  # exposition, rising_action, climax, etc.
    function_confidence: float

    # Themes
    themes: List[str]
    theme_scores: Dict[str, float]

    # Overall confidence
    bayesian_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class DepthAnalysisResult:
    """Result of matryoshka depth analysis."""

    # Metadata
    timestamp: str
    duration_ms: float

    # Depth levels achieved
    max_depth_achieved: str  # SURFACE, SYMBOLIC, ARCHETYPAL, MYTHIC, COSMIC
    gates_unlocked: List[str]
    total_gates: int

    # Interpretations by level
    surface_literal: str
    symbolic_metaphor: Optional[str] = None
    archetypal_pattern: Optional[str] = None
    mythic_resonance: Optional[str] = None
    cosmic_truth: Optional[str] = None

    # Complexity scores
    complexity_scores: Dict[str, float] = None

    # Gate conditions
    gate_conditions: Dict[str, bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class CrossDomainResult:
    """Result of cross-domain narrative analysis."""

    # Metadata
    timestamp: str
    duration_ms: float
    domain: str  # business, science, personal, product, history

    # Base narrative analysis
    base_analysis: Dict[str, Any]

    # Domain-specific mappings
    domain_mappings: Dict[str, str]  # e.g., {"ordeal": "pivot", "mentor": "advisor"}

    # Domain insights
    domain_insights: List[str]

    # Confidence
    domain_fit_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SystemStatus:
    """System status and metrics."""

    # Health
    status: str  # "healthy", "degraded", "error"
    uptime_seconds: float

    # Components
    intelligence_ready: bool
    depth_analyzer_ready: bool
    cross_domain_ready: bool
    cache_ready: bool

    # Performance
    avg_analysis_time_ms: float
    total_analyses: int
    cache_hit_rate: float

    # Cache metrics
    cache_size: int
    cache_capacity: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class NarrativeAnalyzerAPI:
    """
    Main API for dashboard integration.

    Usage:
        api = NarrativeAnalyzerAPI()
        result = await api.analyze_narrative("Once upon a time...")
    """

    def __init__(
        self,
        enable_cache: bool = True,
        cache_capacity: int = 1000,
    ):
        """
        Initialize API.

        Args:
            enable_cache: Enable performance caching
            cache_capacity: Maximum cache entries
        """
        self._intelligence = None
        self._depth_analyzer = None
        self._cross_domain = None
        self._cache = None

        self._enable_cache = enable_cache
        self._cache_capacity = cache_capacity

        self._start_time = datetime.now()
        self._metrics = {
            "total_analyses": 0,
            "total_duration_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    async def initialize(self) -> None:
        """
        Initialize components.

        Note:
            Must be called before using the API.
        """
        if NarrativeIntelligence is None:
            raise ImportError("hololoom_narrative modules not available")

        # Initialize core components
        self._intelligence = NarrativeIntelligence()
        self._depth_analyzer = MatryoshkaNarrativeDepth()
        self._cross_domain = CrossDomainAdapter()

        # Initialize cache if enabled
        if self._enable_cache and NarrativeCache:
            self._cache = NarrativeCache(capacity=self._cache_capacity)

    async def analyze_narrative(
        self,
        text: str,
        use_cache: bool = True,
    ) -> NarrativeAnalysisResult:
        """
        Analyze narrative structure using Campbell's Hero's Journey.

        Args:
            text: Text to analyze
            use_cache: Use cached results if available

        Returns:
            NarrativeAnalysisResult with Campbell stages, characters, archetypes
        """
        start_time = datetime.now()

        # Check cache
        if use_cache and self._cache:
            cached = self._cache.get(text, analysis_type="narrative")
            if cached:
                self._metrics["cache_hits"] += 1
                return cached

        self._metrics["cache_misses"] += 1

        # Perform analysis
        result_obj = await self._intelligence.analyze(text)

        # Convert to API result format
        result = NarrativeAnalysisResult(
            timestamp=start_time.isoformat(),
            duration_ms=0.0,  # Will update below
            primary_stage=result_obj.narrative_arc.primary_arc.value,
            stage_confidence=result_obj.narrative_arc.stage_confidence,
            all_stage_scores={
                stage.value: score
                for stage, score in result_obj.narrative_arc.all_scores.items()
            },
            detected_characters=[
                {
                    "name": char.name,
                    "confidence": char.confidence,
                    "mythology": char.mythology,
                    "description": char.description,
                }
                for char in result_obj.detected_characters
            ],
            character_count=len(result_obj.detected_characters),
            primary_archetypes=[a.value for a in result_obj.primary_archetypes],
            archetype_scores={
                archetype.value: score
                for archetype, score in result_obj.archetype_scores.items()
            },
            narrative_function=result_obj.narrative_function.value,
            function_confidence=result_obj.function_confidence,
            themes=result_obj.themes,
            theme_scores=result_obj.theme_scores,
            bayesian_confidence=result_obj.bayesian_confidence,
        )

        # Update metrics
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        result.duration_ms = duration_ms
        self._metrics["total_analyses"] += 1
        self._metrics["total_duration_ms"] += duration_ms

        # Cache result
        if use_cache and self._cache:
            self._cache.put(text, result, analysis_type="narrative")

        return result

    async def analyze_depth(
        self,
        text: str,
        use_cache: bool = True,
    ) -> DepthAnalysisResult:
        """
        Perform matryoshka depth analysis (5 levels).

        Args:
            text: Text to analyze
            use_cache: Use cached results if available

        Returns:
            DepthAnalysisResult with interpretations at each level
        """
        start_time = datetime.now()

        # Check cache
        if use_cache and self._cache:
            cached = self._cache.get(text, analysis_type="depth")
            if cached:
                self._metrics["cache_hits"] += 1
                return cached

        self._metrics["cache_misses"] += 1

        # Perform depth analysis
        result_obj = await self._depth_analyzer.analyze_depth(text)

        # Convert to API result format
        result = DepthAnalysisResult(
            timestamp=start_time.isoformat(),
            duration_ms=0.0,  # Will update below
            max_depth_achieved=result_obj.max_depth_achieved.name,
            gates_unlocked=[gate.name for gate in result_obj.gates_unlocked],
            total_gates=5,
            surface_literal=result_obj.surface_literal,
            symbolic_metaphor=result_obj.symbolic_metaphor,
            archetypal_pattern=result_obj.archetypal_pattern,
            mythic_resonance=result_obj.mythic_resonance,
            cosmic_truth=result_obj.cosmic_truth,
            complexity_scores=result_obj.complexity_scores,
            gate_conditions=result_obj.gate_conditions,
        )

        # Update metrics
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        result.duration_ms = duration_ms
        self._metrics["total_analyses"] += 1
        self._metrics["total_duration_ms"] += duration_ms

        # Cache result
        if use_cache and self._cache:
            self._cache.put(text, result, analysis_type="depth")

        return result

    async def analyze_cross_domain(
        self,
        text: str,
        domain: str,
        use_cache: bool = True,
    ) -> CrossDomainResult:
        """
        Analyze narrative in a specific domain context.

        Args:
            text: Text to analyze
            domain: Domain name (business, science, personal, product, history)
            use_cache: Use cached results if available

        Returns:
            CrossDomainResult with domain-adapted narrative analysis
        """
        start_time = datetime.now()

        # Check cache
        cache_key = f"{text}:{domain}"
        if use_cache and self._cache:
            cached = self._cache.get(cache_key, analysis_type="cross_domain")
            if cached:
                self._metrics["cache_hits"] += 1
                return cached

        self._metrics["cache_misses"] += 1

        # Perform cross-domain analysis
        result_dict = await self._cross_domain.analyze_with_domain(text, domain_name=domain)

        # Convert to API result format
        result = CrossDomainResult(
            timestamp=start_time.isoformat(),
            duration_ms=0.0,  # Will update below
            domain=result_dict["domain"],
            base_analysis=result_dict["base_analysis"],
            domain_mappings=result_dict.get("domain_mappings", {}),
            domain_insights=result_dict.get("domain_insights", []),
            domain_fit_score=result_dict.get("domain_fit_score", 0.0),
        )

        # Update metrics
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        result.duration_ms = duration_ms
        self._metrics["total_analyses"] += 1
        self._metrics["total_duration_ms"] += duration_ms

        # Cache result
        if use_cache and self._cache:
            self._cache.put(cache_key, result, analysis_type="cross_domain")

        return result

    async def get_status(self) -> SystemStatus:
        """
        Get system status and metrics.

        Returns:
            SystemStatus with health and performance metrics
        """
        uptime = (datetime.now() - self._start_time).total_seconds()
        avg_time = (
            self._metrics["total_duration_ms"] / self._metrics["total_analyses"]
            if self._metrics["total_analyses"] > 0
            else 0.0
        )
        total_cache_ops = self._metrics["cache_hits"] + self._metrics["cache_misses"]
        cache_hit_rate = (
            self._metrics["cache_hits"] / total_cache_ops
            if total_cache_ops > 0
            else 0.0
        )

        cache_size = 0
        if self._cache:
            cache_size = len(self._cache._cache) if hasattr(self._cache, '_cache') else 0

        return SystemStatus(
            status="healthy",
            uptime_seconds=uptime,
            intelligence_ready=self._intelligence is not None,
            depth_analyzer_ready=self._depth_analyzer is not None,
            cross_domain_ready=self._cross_domain is not None,
            cache_ready=self._cache is not None,
            avg_analysis_time_ms=avg_time,
            total_analyses=self._metrics["total_analyses"],
            cache_hit_rate=cache_hit_rate,
            cache_size=cache_size,
            cache_capacity=self._cache_capacity,
        )

    async def get_analytics(self) -> Dict[str, Any]:
        """
        Get detailed analytics.

        Returns:
            Dictionary with analytics data
        """
        status = await self.get_status()

        return {
            "status": status.to_dict(),
            "cache_metrics": {
                "hits": self._metrics["cache_hits"],
                "misses": self._metrics["cache_misses"],
                "hit_rate": status.cache_hit_rate,
                "size": status.cache_size,
                "capacity": status.cache_capacity,
            },
            "performance": {
                "avg_analysis_time_ms": status.avg_analysis_time_ms,
                "total_analyses": status.total_analyses,
            },
        }

    async def close(self) -> None:
        """Clean up resources."""
        # Clear cache if needed
        if self._cache and hasattr(self._cache, 'clear'):
            self._cache.clear()

    # Context manager support
    async def __aenter__(self):
        """Enter async context."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        await self.close()


# Factory function for dashboard
def create_api(
    enable_cache: bool = True,
    cache_capacity: int = 1000,
) -> NarrativeAnalyzerAPI:
    """
    Factory function for creating API instances.

    Args:
        enable_cache: Enable performance caching
        cache_capacity: Maximum cache entries

    Returns:
        NarrativeAnalyzerAPI instance

    Example:
        >>> api = create_api(enable_cache=True, cache_capacity=2000)
        >>> result = await api.analyze_narrative("Once upon a time...")
    """
    return NarrativeAnalyzerAPI(
        enable_cache=enable_cache,
        cache_capacity=cache_capacity,
    )
