"""
Jenny LLM Compiler - LLM-Enhanced Panel Compilation
====================================================
Replaces heuristic panel selection with LLM-based reasoning.

Date: December 2025
Status: Phase A Implementation

This module provides:
1. LLM-enhanced panel type selection
2. MRF VERIFY strategy for prompt enhancement
3. Semantic axes for query intent detection
4. Graceful fallback chain: LLM → MRF → Heuristics

Philosophy:
> "Let the LLM reason about the best visualization, not just pattern match."

Fallback Chain:
    1. LLM with MRF enhancement (if available, <500ms target)
    2. MRF compiler with Thompson learning (if LLM unavailable)
    3. Base heuristic detection (always available)

Usage:
    from hololoom.visualization.jenny_llm_compiler import (
        LLMJennyCompiler,
        create_llm_compiler,
    )

    # LLM-enhanced compiler
    compiler = create_llm_compiler(llm_provider="ollama", llm_model="llama3.2:3b")
    specs = await compiler.compile(spacetime)

References:
- jenny_mrf.py (JennyMRFCompiler base class)
- prompting/unified_mrf.py (MRF framework)
- semantic_calculus/axes.py (16 interpretable axes)
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

from .jenny_spec import (
    JennySpec,
    PanelTypeJenny,
    PanelSizeJenny,
    LifecycleStage,
    get_default_actions,
)
from .jenny_mrf import (
    JennyMRFCompiler,
    PanelTypeLearner,
    MRF_AVAILABLE,
)
from .jenny_compiler import (
    QueryAnalysis,
    analyze_query,
    generate_text_panel,
    generate_confidence_panel,
    generate_sources_panel,
    generate_graph_panel,
    generate_timeline_panel,
    generate_reasoning_panel,
    generate_metric_panel,
)
from hololoom.protocols.jenny import CompilationStrategy

# Try to import MRF for prompt enhancement
try:
    from hololoom.prompting.unified_mrf import (
        UnifiedMRF,
        RefinementStrategyType,
        ModelProvider,
    )
    MRF_PROMPT_AVAILABLE = True
except ImportError:
    MRF_PROMPT_AVAILABLE = False
    UnifiedMRF = None
    RefinementStrategyType = None
    ModelProvider = None

# Try to import Semantic Calculus for intent detection
try:
    from hololoom.semantic_calculus.matryoshka_semantic import MatryoshkaSemanticCalculus
    SEMANTIC_CALCULUS_AVAILABLE = True
except ImportError:
    SEMANTIC_CALCULUS_AVAILABLE = False
    MatryoshkaSemanticCalculus = None

# Try to import Spacetime
try:
    from hololoom.fabric.spacetime import Spacetime, WeavingTrace
except ImportError:
    Spacetime = Any  # type: ignore
    WeavingTrace = Any  # type: ignore

# Try to import async LLM client (Phase M2)
try:
    from .jenny_llm_client import (
        AsyncLLMClientBase,
        LLMClientConfig,
        LLMResponse,
        create_llm_client as create_async_llm_client,
        create_best_available_client,
        OllamaClient,
        AnthropicClient,
        OpenAIClient,
    )
    ASYNC_LLM_AVAILABLE = True
except ImportError:
    ASYNC_LLM_AVAILABLE = False
    AsyncLLMClientBase = None
    LLMClientConfig = None
    LLMResponse = None
    create_async_llm_client = None
    create_best_available_client = None

# Legacy fallback
try:
    from hololoom.weaving_orchestrator_llm import create_llm_client
    LLM_CLIENT_AVAILABLE = True
except ImportError:
    LLM_CLIENT_AVAILABLE = False
    create_llm_client = None

logger = logging.getLogger(__name__)


# ============================================================================
# LLM Output Schema
# ============================================================================

LLM_OUTPUT_SCHEMA = {
    "panel_type": "TEXT|CODE|GRAPH|CONFIDENCE|TIMELINE|METRIC|REASONING|SOURCES",
    "reason": "string explaining why this panel type was chosen",
    "content_structure": {
        "title": "string - panel title",
        "sections": ["list of section headers"],
        "emphasis": "string - what to emphasize"
    },
    "secondary_panels": ["optional additional panel types"],
    "confidence": "float 0.0-1.0 - confidence in selection"
}

LLM_PANEL_TYPES = ["text", "code", "graph", "confidence", "timeline", "metric", "reasoning", "sources"]


# ============================================================================
# LLM Prompt Template
# ============================================================================

BASE_PROMPT_TEMPLATE = """You are an expert UI designer selecting the optimal visualization for a user query.

## Query Information
Query: "{query}"
Response Preview: "{response_preview}"
Confidence: {confidence:.0%}
Query Type: {query_type}
Complexity: {complexity}

## Available Context
- Thread Count: {thread_count}
- Duration: {duration_ms}ms
- Has Sources: {has_sources}
- Has Graph Context: {has_graph_context}
- Has Metrics: {has_metrics}
- Has Reasoning: {has_reasoning}

## Available Panel Types
- TEXT: Standard text response (always included first)
- CONFIDENCE: Shows confidence gauge/meter
- SOURCES: Lists sources and citations
- GRAPH: Network visualization of related concepts
- TIMELINE: Shows processing timeline/stages
- METRIC: Key numeric metrics
- REASONING: Step-by-step reasoning breakdown

## Semantic Profile
{semantic_profile}

## Task
Select the BEST panel types to display alongside the TEXT panel.
Consider the query's intent, complexity, and what would help the user most.

Respond with valid JSON:
{{
    "panel_type": "<primary secondary panel>",
    "reason": "<brief explanation>",
    "secondary_panels": ["<list of additional panel types if needed>"],
    "confidence": <0.0-1.0>
}}

Only respond with the JSON object, no other text.
"""


# ============================================================================
# Semantic Axes Integration
# ============================================================================

@dataclass
class SemanticProfile:
    """Profile of query across semantic axes."""
    technicality: float = 0.5
    complexity: float = 0.5
    actionability: float = 0.5
    urgency: float = 0.5
    abstraction: float = 0.5
    specificity: float = 0.5
    objectivity: float = 0.5
    novelty: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "technicality": self.technicality,
            "complexity": self.complexity,
            "actionability": self.actionability,
            "urgency": self.urgency,
            "abstraction": self.abstraction,
            "specificity": self.specificity,
            "objectivity": self.objectivity,
            "novelty": self.novelty,
        }

    def to_prompt_string(self) -> str:
        """Format for LLM prompt."""
        lines = []
        for axis, value in self.to_dict().items():
            descriptor = "low" if value < 0.4 else "high" if value > 0.6 else "medium"
            lines.append(f"- {axis.title()}: {descriptor} ({value:.2f})")
        return "\n".join(lines)


def analyze_semantic_axes(query: str) -> SemanticProfile:
    """
    Analyze query across 8 key semantic axes.

    Uses MatryoshkaSemanticCalculus if available, falls back to heuristics.
    """
    if SEMANTIC_CALCULUS_AVAILABLE and MatryoshkaSemanticCalculus:
        try:
            calculus = MatryoshkaSemanticCalculus()
            projection = calculus.project(query)

            return SemanticProfile(
                technicality=projection.get("technicality", 0.5),
                complexity=projection.get("complexity", 0.5),
                actionability=projection.get("actionability", 0.5),
                urgency=projection.get("urgency", 0.5),
                abstraction=projection.get("abstraction", 0.5),
                specificity=projection.get("specificity", 0.5),
                objectivity=projection.get("objectivity", 0.5),
                novelty=projection.get("novelty", 0.5),
            )
        except Exception as e:
            logger.warning(f"Semantic calculus failed, using heuristics: {e}")

    # Heuristic fallback
    query_lower = query.lower()

    return SemanticProfile(
        technicality=0.7 if any(w in query_lower for w in ["algorithm", "implementation", "code", "api"]) else 0.3,
        complexity=0.7 if len(query.split()) > 15 else 0.3 if len(query.split()) < 5 else 0.5,
        actionability=0.7 if any(w in query_lower for w in ["how", "implement", "create", "build"]) else 0.3,
        urgency=0.7 if any(w in query_lower for w in ["urgent", "quickly", "asap", "now"]) else 0.3,
        abstraction=0.7 if any(w in query_lower for w in ["concept", "theory", "principle"]) else 0.3,
        specificity=0.7 if any(w in query_lower for w in ["specific", "exactly", "precisely"]) else 0.3,
        objectivity=0.7 if any(w in query_lower for w in ["compare", "difference", "versus", "vs"]) else 0.3,
        novelty=0.5,  # Hard to detect without context
    )


# ============================================================================
# LLM Response Parser
# ============================================================================

@dataclass
class LLMPanelSelection:
    """Parsed LLM response for panel selection."""
    panel_type: PanelTypeJenny
    reason: str
    secondary_panels: List[PanelTypeJenny]
    confidence: float
    raw_response: str
    parse_method: str  # "json" or "regex"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "panel_type": self.panel_type.value,
            "reason": self.reason,
            "secondary_panels": [p.value for p in self.secondary_panels],
            "confidence": self.confidence,
            "parse_method": self.parse_method,
        }


def parse_llm_response(response: str) -> Optional[LLMPanelSelection]:
    """
    Parse LLM response with fallback strategies.

    Attempts:
    1. JSON parsing (preferred)
    2. Regex extraction (fallback)
    3. Returns None if both fail
    """
    raw_response = response.strip()

    # Strategy 1: Try JSON parsing
    try:
        # Find JSON object in response (may have surrounding text)
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            data = json.loads(json_str)

            # Extract panel type
            panel_type_str = data.get("panel_type", "confidence").lower()
            panel_type = _map_panel_type(panel_type_str)

            # Extract secondary panels
            secondary = []
            for pt in data.get("secondary_panels", []):
                mapped = _map_panel_type(pt.lower())
                if mapped and mapped != panel_type:
                    secondary.append(mapped)

            return LLMPanelSelection(
                panel_type=panel_type,
                reason=data.get("reason", "LLM selection"),
                secondary_panels=secondary,
                confidence=float(data.get("confidence", 0.7)),
                raw_response=raw_response,
                parse_method="json",
            )
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.debug(f"JSON parsing failed: {e}")

    # Strategy 2: Regex extraction
    try:
        # Look for panel type mention
        panel_types = ["text", "code", "graph", "confidence", "timeline", "metric", "reasoning", "sources"]
        found_type = None

        for pt in panel_types:
            if re.search(rf'\b{pt}\b', response.lower()):
                found_type = pt
                break

        if found_type:
            panel_type = _map_panel_type(found_type)

            # Try to extract confidence
            conf_match = re.search(r'confidence[:\s]*([0-9.]+)', response.lower())
            confidence = float(conf_match.group(1)) if conf_match else 0.5

            return LLMPanelSelection(
                panel_type=panel_type,
                reason="Extracted from LLM response",
                secondary_panels=[],
                confidence=min(confidence, 1.0),
                raw_response=raw_response,
                parse_method="regex",
            )
    except Exception as e:
        logger.debug(f"Regex extraction failed: {e}")

    return None


def _map_panel_type(type_str: str) -> PanelTypeJenny:
    """Map string to PanelTypeJenny enum."""
    mapping = {
        "text": PanelTypeJenny.TEXT,
        "code": PanelTypeJenny.CODE,
        "graph": PanelTypeJenny.GRAPH,
        "confidence": PanelTypeJenny.CONFIDENCE,
        "timeline": PanelTypeJenny.TIMELINE,
        "metric": PanelTypeJenny.METRIC,
        "reasoning": PanelTypeJenny.REASONING,
        "sources": PanelTypeJenny.SOURCES,
    }
    return mapping.get(type_str.lower(), PanelTypeJenny.CONFIDENCE)


# ============================================================================
# LLM Client Wrapper
# ============================================================================

class LLMClient:
    """
    Wrapper for LLM client with provider abstraction.

    Supports:
    - Ollama (local, default)
    - Anthropic (Claude)
    - OpenAI (GPT)
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.2:3b",
        timeout_ms: int = 500,
    ):
        self.provider = provider
        self.model = model
        self.timeout_ms = timeout_ms
        self._client = None
        self._available = False

        self._initialize_client()

    def _initialize_client(self):
        """Initialize LLM client based on provider."""
        try:
            if self.provider == "ollama":
                import ollama
                # Test connection
                ollama.list()
                self._client = ollama
                self._available = True
                logger.info(f"Ollama client initialized with model {self.model}")

            elif self.provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic()
                self._available = True
                logger.info("Anthropic client initialized")

            elif self.provider == "openai":
                import openai
                self._client = openai.OpenAI()
                self._available = True
                logger.info("OpenAI client initialized")

            else:
                logger.warning(f"Unknown LLM provider: {self.provider}")

        except ImportError as e:
            logger.warning(f"LLM provider {self.provider} not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")

    @property
    def available(self) -> bool:
        return self._available

    async def generate(self, prompt: str) -> Optional[str]:
        """Generate response from LLM."""
        if not self._available:
            return None

        try:
            if self.provider == "ollama":
                import asyncio
                # Run sync ollama in executor
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self._client.generate(model=self.model, prompt=prompt)
                )
                return response.get("response", "")

            elif self.provider == "anthropic":
                message = self._client.messages.create(
                    model=self.model or "claude-3-5-sonnet-20241022",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text

            elif self.provider == "openai":
                response = self._client.chat.completions.create(
                    model=self.model or "gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return None


# ============================================================================
# LLM Jenny Compiler
# ============================================================================

class LLMJennyCompiler(JennyMRFCompiler):
    """
    LLM-enhanced panel compilation with MRF prompt refinement.

    Extends JennyMRFCompiler with LLM-based panel type selection.
    Uses MRF VERIFY strategy to enhance LLM prompts.

    Phase M2: Now uses production-grade async LLM clients with:
    - Connection pooling (httpx)
    - Exponential backoff with jitter
    - Multi-provider support (Ollama, Anthropic, OpenAI)
    - Graceful degradation

    Fallback Chain:
        1. LLM with MRF enhancement (if available)
        2. MRF compiler with Thompson learning
        3. Base heuristic detection

    Usage:
        async with LLMJennyCompiler(llm_provider="ollama") as compiler:
            specs = await compiler.compile(spacetime)
    """

    VERSION = "3.1.0-llm-async"

    def __init__(
        self,
        llm_provider: str = "ollama",
        llm_model: str = "llama3.2:3b",
        llm_timeout_ms: int = 500,
        enable_learning: bool = True,
        learner: Optional[PanelTypeLearner] = None,
        learning_persist_path: Optional[str] = None,
        enable_mrf_prompt_enhancement: bool = True,
        enable_semantic_axes: bool = True,
        llm_config: Optional['LLMClientConfig'] = None,
    ):
        """
        Initialize LLM-enhanced compiler.

        Args:
            llm_provider: LLM provider ("ollama", "anthropic", "openai")
            llm_model: Model name (e.g., "llama3.2:3b", "claude-3-5-sonnet")
            llm_timeout_ms: Timeout for LLM calls in milliseconds
            enable_learning: Enable Thompson Sampling learning
            learner: Optional pre-configured learner
            learning_persist_path: Path to persist learning state
            enable_mrf_prompt_enhancement: Use MRF VERIFY to enhance prompts
            enable_semantic_axes: Use semantic calculus for intent detection
            llm_config: Optional LLMClientConfig for advanced configuration
        """
        # Initialize base MRF compiler
        super().__init__(
            enable_learning=enable_learning,
            learning_persist_path=learning_persist_path,
        )

        # Override learner if provided
        if learner:
            self.learner = learner

        # LLM configuration
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_timeout_ms = llm_timeout_ms

        # Feature flags
        self.enable_mrf_prompt_enhancement = enable_mrf_prompt_enhancement and MRF_PROMPT_AVAILABLE
        self.enable_semantic_axes = enable_semantic_axes

        # Async LLM client configuration (Phase M2)
        self._llm_config = llm_config
        self._async_llm: Optional[AsyncLLMClientBase] = None
        self._llm_available = False
        self._use_async_client = ASYNC_LLM_AVAILABLE

        # Legacy sync client fallback
        self._llm_legacy: Optional[LLMClient] = None

        # Track LLM usage statistics
        self._llm_calls = 0
        self._llm_successes = 0
        self._llm_fallbacks = 0
        self._llm_retries = 0

        # Initialize synchronously if async not available
        if not self._use_async_client:
            self._llm_legacy = LLMClient(
                provider=llm_provider,
                model=llm_model,
                timeout_ms=llm_timeout_ms,
            )
            self._llm_available = self._llm_legacy.available

        logger.info(
            f"LLMJennyCompiler v{self.VERSION} initialized "
            f"(llm={llm_provider}/{llm_model}, async={self._use_async_client}, "
            f"mrf_enhance={self.enable_mrf_prompt_enhancement}, "
            f"semantic_axes={self.enable_semantic_axes})"
        )

    async def __aenter__(self):
        """Async context manager entry - connects LLM client."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - closes LLM client."""
        await self.close()

    async def connect(self) -> bool:
        """
        Connect async LLM client.

        Returns True if connected successfully.
        """
        if not self._use_async_client:
            return self._llm_available

        try:
            # Build config if not provided
            if self._llm_config is None and LLMClientConfig:
                self._llm_config = LLMClientConfig(
                    provider=self.llm_provider,
                    model=self.llm_model,
                    timeout_seconds=self.llm_timeout_ms / 1000.0,
                    max_retries=2,
                )

            # Create and connect client
            self._async_llm = await create_async_llm_client(
                provider=self.llm_provider,
                model=self.llm_model,
                config=self._llm_config,
            )
            self._llm_available = self._async_llm.available

            logger.info(
                f"Async LLM client connected: {self.llm_provider}/{self.llm_model} "
                f"(available={self._llm_available})"
            )
            return self._llm_available

        except Exception as e:
            logger.warning(f"Failed to connect async LLM client: {e}")
            # Fall back to legacy client
            if not self._llm_legacy:
                self._llm_legacy = LLMClient(
                    provider=self.llm_provider,
                    model=self.llm_model,
                    timeout_ms=self.llm_timeout_ms,
                )
                self._llm_available = self._llm_legacy.available
            self._use_async_client = False
            return self._llm_available

    async def close(self) -> None:
        """Close LLM client connections."""
        if self._async_llm:
            await self._async_llm.close()
            self._async_llm = None
        self._llm_available = False

    async def compile(
        self,
        spacetime: Spacetime,
        strategy: CompilationStrategy = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[JennySpec]:
        """
        Compile Spacetime into UI specifications using LLM.

        Fallback chain:
        1. LLM with MRF enhancement (if available)
        2. MRF compiler with Thompson learning
        3. Base heuristic detection

        Args:
            spacetime: Woven output from hololoom weaving cycle
            strategy: Compilation strategy (None = use default)
            context: Optional user/session context

        Returns:
            List of JennySpec panels to render
        """
        strategy = strategy or self.default_strategy

        # Analyze query
        analysis = analyze_query(spacetime)

        # Auto-connect if not connected
        if self._use_async_client and not self._async_llm:
            await self.connect()

        # Try LLM-based compilation first
        if self._llm_available:
            llm_result = await self._compile_with_llm(spacetime, analysis)
            if llm_result:
                return llm_result

        # Fallback to MRF compiler
        logger.debug("LLM unavailable or failed, falling back to MRF compiler")
        self._llm_fallbacks += 1
        return await super().compile(spacetime, strategy, context)

    async def _generate_llm_response(self, prompt: str) -> Optional[str]:
        """
        Generate LLM response using async or legacy client.

        Returns response string or None on failure.
        """
        # Try async client first (Phase M2)
        if self._async_llm and self._async_llm.available:
            response = await self._async_llm.generate(prompt)
            if response:
                self._llm_retries += response.retries
                return response.content
            return None

        # Fall back to legacy sync client
        if self._llm_legacy and self._llm_legacy.available:
            return await self._llm_legacy.generate(prompt)

        return None

    async def _compile_with_llm(
        self,
        spacetime: Spacetime,
        analysis: QueryAnalysis,
    ) -> Optional[List[JennySpec]]:
        """
        Compile using LLM-based panel selection.

        Returns None if LLM fails, triggering fallback.
        """
        self._llm_calls += 1

        try:
            # Build LLM prompt
            prompt = await self._build_llm_prompt(spacetime, analysis)

            # Generate LLM response (uses async or legacy client)
            response = await self._generate_llm_response(prompt)
            if not response:
                return None

            # Parse LLM response
            selection = parse_llm_response(response)
            if not selection:
                logger.warning("Failed to parse LLM response")
                return None

            # Generate panels based on LLM selection
            specs = []

            # Always include text panel first
            specs.append(generate_text_panel(spacetime, priority=0))

            # Primary panel from LLM
            primary = self._generate_panel(selection.panel_type, spacetime, priority=1)
            if primary:
                specs.append(primary)

            # Secondary panels from LLM
            for idx, panel_type in enumerate(selection.secondary_panels[:2], start=2):
                panel = self._generate_panel(panel_type, spacetime, priority=idx)
                if panel:
                    specs.append(panel)

            # Add WHY panel if enabled
            if self.include_why_panel:
                from .jenny_mrf import generate_why_panel_mrf
                why_panel = await generate_why_panel_mrf(
                    spacetime, specs, analysis, self.mrf, priority=99
                )
                specs.append(why_panel)

            # Store metadata for learning
            for spec in specs:
                if spec.metadata is None:
                    object.__setattr__(spec, 'metadata', {})
                spec.metadata['query_type'] = analysis.query_type
                spec.metadata['complexity'] = analysis.complexity
                spec.metadata['llm_selection'] = selection.to_dict()
                spec.metadata['compiler_version'] = self.VERSION

            self._llm_successes += 1
            logger.info(
                f"LLM compiled {len(specs)} panels "
                f"(primary={selection.panel_type.value}, "
                f"reason='{selection.reason[:50]}...', "
                f"parse={selection.parse_method})"
            )

            return specs

        except Exception as e:
            logger.warning(f"LLM compilation failed: {e}")
            return None

    async def _build_llm_prompt(
        self,
        spacetime: Spacetime,
        analysis: QueryAnalysis,
    ) -> str:
        """
        Build LLM prompt with optional MRF enhancement.

        Uses MRF VERIFY strategy if enabled.
        """
        # Analyze semantic axes if enabled
        semantic_profile = SemanticProfile()
        if self.enable_semantic_axes:
            semantic_profile = analyze_semantic_axes(spacetime.query_text)

        # Extract context from spacetime
        response_preview = ""
        if hasattr(spacetime, 'response') and spacetime.response:
            response_preview = spacetime.response[:500]
        elif hasattr(spacetime, 'text') and spacetime.text:
            response_preview = spacetime.text[:500]

        thread_count = 0
        duration_ms = 0
        if spacetime.trace:
            thread_count = len(getattr(spacetime.trace, 'threads', []))
            duration_ms = getattr(spacetime.trace, 'total_duration_ms', 0)

        # Build base prompt
        base_prompt = BASE_PROMPT_TEMPLATE.format(
            query=spacetime.query_text,
            response_preview=response_preview,
            confidence=spacetime.confidence,
            query_type=analysis.query_type,
            complexity=analysis.complexity,
            thread_count=thread_count,
            duration_ms=duration_ms,
            has_sources=analysis.has_sources,
            has_graph_context=analysis.has_graph_context,
            has_metrics=analysis.has_metrics,
            has_reasoning=analysis.has_reasoning,
            semantic_profile=semantic_profile.to_prompt_string(),
        )

        # Apply MRF VERIFY enhancement if enabled
        if self.enable_mrf_prompt_enhancement and self.mrf:
            try:
                enhanced = await self.mrf.refine_prompt(
                    original_prompt=base_prompt,
                    strategy=RefinementStrategyType.VERIFY,
                    context={
                        "task": "panel_type_selection",
                        "query_type": analysis.query_type,
                    },
                    epistemic_confidence=spacetime.confidence,
                )
                return enhanced.get("enhanced_prompt", base_prompt)
            except Exception as e:
                logger.warning(f"MRF prompt enhancement failed: {e}")

        return base_prompt

    def get_llm_statistics(self) -> Dict[str, Any]:
        """Get LLM usage statistics."""
        success_rate = self._llm_successes / max(self._llm_calls, 1)

        stats = {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_available": self._llm_available,
            "using_async_client": self._use_async_client,
            "total_calls": self._llm_calls,
            "successful_calls": self._llm_successes,
            "fallbacks": self._llm_fallbacks,
            "total_retries": self._llm_retries,
            "success_rate": success_rate,
            "mrf_prompt_enhancement": self.enable_mrf_prompt_enhancement,
            "semantic_axes_enabled": self.enable_semantic_axes,
        }

        # Add async client stats if available
        if self._async_llm:
            stats["async_client_stats"] = self._async_llm.stats

        return stats

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get combined learning and LLM statistics."""
        base_stats = super().get_learning_statistics()
        llm_stats = self.get_llm_statistics()

        return {
            **base_stats,
            "llm": llm_stats,
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_llm_compiler(
    llm_provider: str = "ollama",
    llm_model: str = "llama3.2:3b",
    enable_learning: bool = True,
    learning_persist_path: Optional[str] = None,
    enable_mrf_prompt_enhancement: bool = True,
    enable_semantic_axes: bool = True,
) -> LLMJennyCompiler:
    """
    Create an LLM-enhanced Jenny compiler.

    Args:
        llm_provider: LLM provider ("ollama", "anthropic", "openai")
        llm_model: Model name
        enable_learning: Enable Thompson Sampling learning
        learning_persist_path: Path to persist learning state
        enable_mrf_prompt_enhancement: Use MRF VERIFY strategy
        enable_semantic_axes: Use semantic calculus for intent

    Returns:
        Configured LLMJennyCompiler instance
    """
    return LLMJennyCompiler(
        llm_provider=llm_provider,
        llm_model=llm_model,
        enable_learning=enable_learning,
        learning_persist_path=learning_persist_path,
        enable_mrf_prompt_enhancement=enable_mrf_prompt_enhancement,
        enable_semantic_axes=enable_semantic_axes,
    )


def create_compiler_with_fallback(
    prefer_llm: bool = True,
    llm_provider: str = "ollama",
    llm_model: str = "llama3.2:3b",
    enable_learning: bool = True,
) -> Union[LLMJennyCompiler, JennyMRFCompiler]:
    """
    Create the best available compiler with automatic fallback.

    Tries:
    1. LLMJennyCompiler (if LLM available and prefer_llm=True)
    2. JennyMRFCompiler (MRF with Thompson learning)

    Args:
        prefer_llm: Try LLM compiler first
        llm_provider: LLM provider
        llm_model: Model name
        enable_learning: Enable Thompson Sampling

    Returns:
        Best available compiler
    """
    if prefer_llm:
        compiler = create_llm_compiler(
            llm_provider=llm_provider,
            llm_model=llm_model,
            enable_learning=enable_learning,
        )

        if compiler._llm_available:
            logger.info("Using LLMJennyCompiler")
            return compiler
        else:
            logger.info("LLM not available, falling back to JennyMRFCompiler")

    from .jenny_mrf import create_mrf_compiler
    return create_mrf_compiler(enable_learning=enable_learning)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Classes
    'LLMJennyCompiler',
    'LLMClient',  # Legacy sync client
    'LLMPanelSelection',
    'SemanticProfile',

    # Async LLM client (Phase M2) - re-exported from jenny_llm_client
    'AsyncLLMClientBase',
    'LLMClientConfig',
    'LLMResponse',
    'OllamaClient',
    'AnthropicClient',
    'OpenAIClient',

    # Functions
    'create_llm_compiler',
    'create_compiler_with_fallback',
    'analyze_semantic_axes',
    'parse_llm_response',

    # Constants
    'LLM_OUTPUT_SCHEMA',
    'LLM_PANEL_TYPES',
    'BASE_PROMPT_TEMPLATE',
]
