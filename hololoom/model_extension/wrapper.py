"""
Memory-Augmented LLM Wrapper

The core SDK for wrapping any LLM with HoloLoom's memory infrastructure.
Makes models smarter without retraining through:
- Persistent memory (knowledge graph + vector store)
- Continuous learning (Thompson Sampling adaptation)
- Domain adaptation (instant domain switching)
- Personalization (per-user memory graphs)

Quick Start:
    from hololoom.model_extension import MemoryAugmentedLLM

    async with MemoryAugmentedLLM(provider="anthropic") as llm:
        # Ingest domain knowledge
        await llm.learn("Thompson Sampling balances exploration and exploitation")

        # Query with memory-augmented context
        response = await llm.query("What is Thompson Sampling?")
        print(response.answer)
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Sources: {response.sources}")
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    List, Optional, Dict, Any, AsyncIterator,
    Union, Callable, Awaitable
)
import asyncio
import logging
import uuid

from .providers import (
    create_provider,
    get_best_available_provider,
    BaseLLMProvider,
    GenerationConfig,
    GenerationResult,
    Message,
    MessageRole,
)
from .uncertainty import (
    UncertaintyEnvelope,
    ConfidenceTier,
    UncertaintySource,
    compute_variance_aware_confidence,
    should_refuse_to_converge,
)
from .verification import (
    VerificationStatus,
    VerificationTier,
    ClaimType,
    VerificationEngine,
    verify_response,
    should_block_for_safety,
)
from .governance import (
    PolicyTier,
    PolicyDecision,
    GovernanceEngine,
    is_safe_for_execution,
    requires_human_approval,
)
from hololoom.protocols import Memory


logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Strategy for retrieving context from memory."""
    SEMANTIC = "semantic"      # Vector similarity only
    GRAPH = "graph"            # Knowledge graph traversal only
    HYBRID = "hybrid"          # Both semantic + graph (default)
    ADAPTIVE = "adaptive"      # Auto-select based on query type


class LearningMode(Enum):
    """Mode for continuous learning."""
    DISABLED = "disabled"      # No learning
    PASSIVE = "passive"        # Learn from interactions (default)
    ACTIVE = "active"          # Actively seek to improve
    RESEARCH = "research"      # Full exploration mode


@dataclass
class LLMConfig:
    """
    Configuration for MemoryAugmentedLLM.

    Attributes:
        provider: LLM provider name (anthropic, openai, ollama, google, cohere)
        model: Specific model to use (default: provider's default)
        api_key: API key (optional, uses env vars if not provided)

        # Memory settings
        retrieval_strategy: How to retrieve context (default: HYBRID)
        max_context_tokens: Maximum tokens for context (default: 4000)
        context_overlap: Overlap between context chunks (default: 100)

        # Learning settings
        learning_mode: Continuous learning mode (default: PASSIVE)
        learning_rate: Adaptation rate (default: 0.1)
        exploration_bonus: Bonus for exploring new patterns (default: 0.2)

        # Quality settings
        confidence_threshold: Minimum confidence to return (default: 0.6)
        require_verification: Whether to verify factual claims (default: True)
        verification_tier: Maximum verification tier (default: TIER_2)

        # Safety settings
        enable_governance: Whether to apply governance policies (default: True)
        human_approval_threshold: When to require human approval (default: 0.3)

        # Generation settings
        generation_config: LLM generation parameters
    """
    provider: str = "anthropic"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    # Memory settings
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    max_context_tokens: int = 4000
    context_overlap: int = 100
    max_retrieved_memories: int = 10

    # Learning settings
    learning_mode: LearningMode = LearningMode.PASSIVE
    learning_rate: float = 0.1
    exploration_bonus: float = 0.2

    # Quality settings
    confidence_threshold: float = 0.6
    require_verification: bool = True
    verification_tier: VerificationTier = VerificationTier.TIER_2_EXTERNAL
    multi_sample_count: int = 1  # For variance-aware confidence

    # Safety settings
    enable_governance: bool = True
    human_approval_threshold: float = 0.3

    # Generation settings
    generation_config: Optional[GenerationConfig] = None

    def __post_init__(self):
        if self.generation_config is None:
            self.generation_config = GenerationConfig()

        # Convert string values to enums if needed
        if isinstance(self.learning_mode, str):
            try:
                self.learning_mode = LearningMode(self.learning_mode)
            except ValueError:
                self.learning_mode = LearningMode.PASSIVE

        if isinstance(self.retrieval_strategy, str):
            try:
                self.retrieval_strategy = RetrievalStrategy(self.retrieval_strategy)
            except ValueError:
                self.retrieval_strategy = RetrievalStrategy.HYBRID

        if isinstance(self.verification_tier, str):
            try:
                self.verification_tier = VerificationTier(self.verification_tier)
            except ValueError:
                self.verification_tier = VerificationTier.TIER_2_EXTERNAL


@dataclass
class MemorySource:
    """A source of information from memory."""
    content: str
    node_id: str
    relevance: float  # 0.0-1.0
    source_type: str  # "semantic", "graph", "learned"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryAugmentedResponse:
    """
    Response from MemoryAugmentedLLM with rich metadata.

    Attributes:
        answer: The generated response text
        confidence: Overall confidence (0.0-1.0)
        uncertainty: Full uncertainty envelope
        sources: Memory sources used to generate response
        verification: Verification status (if enabled)
        governance: Governance decision (if enabled)
        raw_response: Raw LLM response
        metadata: Additional metadata
    """
    answer: str
    confidence: float
    uncertainty: UncertaintyEnvelope
    sources: List[MemorySource] = field(default_factory=list)
    verification: Optional[VerificationStatus] = None
    governance: Optional[PolicyDecision] = None
    raw_response: Optional[GenerationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_verified(self) -> bool:
        """Check if response passed verification."""
        if self.verification is None:
            return False
        return self.verification.all_required_passed

    @property
    def requires_human_review(self) -> bool:
        """Check if response requires human review."""
        if self.governance is not None:
            if self.governance.decision.value == "escalate":
                return True
        if self.uncertainty.requires_verification:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty.to_dict(),
            "sources": [
                {
                    "content": s.content[:200] + "..." if len(s.content) > 200 else s.content,
                    "node_id": s.node_id,
                    "relevance": s.relevance,
                    "source_type": s.source_type,
                }
                for s in self.sources
            ],
            "verification": self.verification.to_dict() if self.verification else None,
            "governance": self.governance.to_dict() if self.governance else None,
            "is_verified": self.is_verified,
            "requires_human_review": self.requires_human_review,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class MemoryAugmentedLLM:
    """
    Wrap any LLM with HoloLoom's memory and learning infrastructure.

    This is the main entry point for the Model Extension SDK.
    It provides:
    - Persistent memory (knowledge graph + vector store)
    - Continuous learning (Thompson Sampling adaptation)
    - Domain adaptation (instant domain switching)
    - Personalization (per-user memory graphs)
    - Safety guarantees (governance, verification, uncertainty)

    Usage:
        async with MemoryAugmentedLLM(provider="anthropic") as llm:
            # Learn domain knowledge
            await llm.learn("Thompson Sampling balances exploration and exploitation")

            # Query with memory-augmented context
            response = await llm.query("What is Thompson Sampling?")
            print(response.answer)
            print(f"Confidence: {response.confidence:.2f}")

            # Get similar memories
            memories = await llm.recall("bandit algorithms")

            # Reflect and improve
            await llm.reflect(response, feedback={"helpful": True})
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize MemoryAugmentedLLM.

        Args:
            config: Full configuration (takes precedence)
            provider: LLM provider shorthand
            model: Model name shorthand
            api_key: API key shorthand
            **kwargs: Additional config overrides
        """
        # Build config from various inputs
        if config is not None:
            self.config = config
        else:
            config_kwargs = {}
            if provider is not None:
                config_kwargs["provider"] = provider
            if model is not None:
                config_kwargs["model"] = model
            if api_key is not None:
                config_kwargs["api_key"] = api_key
            config_kwargs.update(kwargs)
            self.config = LLMConfig(**config_kwargs)

        # Components (initialized in __aenter__)
        self._provider: Optional[BaseLLMProvider] = None
        self._memory_backend = None  # HoloLoom memory system
        self._knowledge_graph = None  # Yarn Graph
        self._vector_store = None     # Vector memory
        self._governance_engine: Optional[GovernanceEngine] = None
        self._verification_engine: Optional[VerificationEngine] = None

        # Learning state
        self._learning_buffer: List[Dict[str, Any]] = []
        self._pattern_weights: Dict[str, float] = {}
        self._query_history: List[Dict[str, Any]] = []

        # State
        self._initialized = False
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    async def __aenter__(self) -> "MemoryAugmentedLLM":
        """Async context manager entry - initialize all components."""
        await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - cleanup resources."""
        await self._cleanup()

    async def _initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        # Initialize LLM provider
        try:
            self._provider = create_provider(
                provider=self.config.provider,
                model=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
            if not self._provider.is_available():
                # Try to find any available provider
                self._provider = get_best_available_provider()
                if self._provider is None:
                    logger.warning("No LLM provider available")
        except Exception as e:
            logger.warning(f"Failed to initialize provider: {e}")
            self._provider = get_best_available_provider()

        # Initialize memory backend (try to import HoloLoom components)
        await self._initialize_memory()

        # Initialize governance engine
        if self.config.enable_governance:
            self._governance_engine = GovernanceEngine(
                enable_audit=True,
                audit_callback=self._log_governance_decision,
            )

        # Initialize verification engine
        if self.config.require_verification:
            self._verification_engine = VerificationEngine(
                knowledge_graph=self._knowledge_graph,
                vector_store=self._vector_store,
            )

        self._initialized = True
        logger.info(f"MemoryAugmentedLLM initialized with provider: {self.config.provider}")

    async def _initialize_memory(self) -> None:
        """Initialize HoloLoom memory components."""
        try:
            # Try to import HoloLoom memory components
            from hololoom.memory.backend_factory import create_memory_backend
            from hololoom.config import Config

            # Create memory backend
            holo_config = Config.fast()
            self._memory_backend = await create_memory_backend(holo_config)

            # Extract knowledge graph and vector store
            if hasattr(self._memory_backend, 'graph'):
                self._knowledge_graph = self._memory_backend.graph
            if hasattr(self._memory_backend, 'vector_store'):
                self._vector_store = self._memory_backend.vector_store

            logger.info("HoloLoom memory backend initialized")

        except ImportError:
            logger.warning("HoloLoom memory not available, using in-memory fallback")
            self._memory_backend = InMemoryBackend()
            self._knowledge_graph = self._memory_backend
            self._vector_store = self._memory_backend

        except Exception as e:
            logger.warning(f"Failed to initialize HoloLoom memory: {e}, using fallback")
            self._memory_backend = InMemoryBackend()
            self._knowledge_graph = self._memory_backend
            self._vector_store = self._memory_backend

    async def _cleanup(self) -> None:
        """Cleanup resources."""
        # Flush learning buffer
        if self._learning_buffer and self.config.learning_mode != LearningMode.DISABLED:
            await self._flush_learning_buffer()

        # Close memory backend
        if self._memory_backend is not None:
            if hasattr(self._memory_backend, 'close'):
                await self._memory_backend.close()

        self._initialized = False
        logger.info("MemoryAugmentedLLM cleaned up")

    def _log_governance_decision(self, decision: PolicyDecision) -> None:
        """Log governance decision for audit."""
        logger.debug(f"Governance decision: {decision.decision.value} - {decision.reason}")

    # =========================================================================
    # Core API: learn(), query(), recall(), reflect()
    # =========================================================================

    async def learn(
        self,
        content: Union[str, List[str]],
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "user",
    ) -> List[str]:
        """
        Learn from content by storing it in memory.

        Args:
            content: Text content or list of texts to learn
            metadata: Optional metadata for the content
            source: Source identifier

        Returns:
            List of memory node IDs created
        """
        if isinstance(content, str):
            contents = [content]
        else:
            contents = content

        node_ids = []
        for text in contents:
            # Check governance
            if self.config.enable_governance and self._governance_engine:
                blocked, reason = self._governance_engine.should_block(text)
                if blocked:
                    logger.warning(f"Content blocked by governance: {reason}")
                    continue

            # Store in memory
            node_id = await self._store_memory(text, metadata, source)
            if node_id:
                node_ids.append(node_id)

            # Update learning state
            if self.config.learning_mode != LearningMode.DISABLED:
                self._learning_buffer.append({
                    "type": "learn",
                    "content": text,
                    "node_id": node_id,
                    "timestamp": datetime.now().isoformat(),
                })

        logger.info(f"Learned {len(node_ids)} memories")
        return node_ids

    async def query(
        self,
        question: str,
        context_override: Optional[str] = None,
        claim_type: ClaimType = ClaimType.FACTUAL,
        stakes: str = "normal",
    ) -> MemoryAugmentedResponse:
        """
        Query with memory-augmented context.

        Args:
            question: The question to answer
            context_override: Optional context to use instead of retrieval
            claim_type: Type of claims expected (for verification)
            stakes: Stakes level ("low", "normal", "high")

        Returns:
            MemoryAugmentedResponse with answer and metadata
        """
        # Check governance
        governance_decision = None
        if self.config.enable_governance and self._governance_engine:
            governance_decision = await self._governance_engine.evaluate(question)
            if governance_decision.decision.value == "block":
                return MemoryAugmentedResponse(
                    answer="I cannot answer this question due to policy constraints.",
                    confidence=0.0,
                    uncertainty=UncertaintyEnvelope.from_confidence(0.0),
                    governance=governance_decision,
                )

        # Retrieve context from memory
        if context_override is not None:
            context = context_override
            sources = []
        else:
            sources = await self._retrieve_context(question)
            context = self._format_context(sources)

        # Generate response
        if self._provider is None or not self._provider.is_available():
            # Fallback to memory-only response
            return self._memory_only_response(question, sources)

        # Build prompt with context
        prompt = self._build_query_prompt(question, context)

        # Multi-sample for variance-aware confidence
        samples = []
        confidences = []

        for _ in range(self.config.multi_sample_count):
            result = await self._provider.generate(
                prompt=prompt,
                config=self.config.generation_config,
                system_prompt=self._get_system_prompt(),
            )
            samples.append(result.text)
            # Extract confidence from response (simplified)
            conf = self._extract_confidence(result.text)
            confidences.append(conf)

        # Compute variance-aware confidence
        mean_conf, variance, is_uncertain = compute_variance_aware_confidence(
            confidences
        )

        # Build uncertainty envelope
        uncertainty_sources = []
        if len(sources) < 3:
            uncertainty_sources.append(
                UncertaintySource(
                    type="retrieval_sparsity",
                    contribution=0.2 - len(sources) * 0.05,
                    details=f"Only {len(sources)} sources found",
                )
            )
        if is_uncertain:
            uncertainty_sources.append(
                UncertaintySource(
                    type="sample_disagreement",
                    contribution=min(variance, 0.3),
                    details=f"Variance: {variance:.3f}",
                )
            )

        uncertainty = UncertaintyEnvelope.from_confidence(
            confidence=mean_conf,
            sources=uncertainty_sources,
            sample_variance=variance if len(samples) > 1 else None,
        )

        # Check if should refuse
        should_refuse, refuse_reason = should_refuse_to_converge(
            mean_conf, stakes, self.config.confidence_threshold
        )

        if should_refuse:
            answer = f"I'm not confident enough to answer definitively. {refuse_reason}"
            answer += f"\n\nPreliminary answer: {samples[0]}"
        else:
            answer = samples[0]

        # Verify response if enabled
        verification = None
        if self.config.require_verification:
            verification = await self._verify_response(
                answer, claim_type, mean_conf
            )

            # Check for safety block
            should_block, block_reason = should_block_for_safety(
                claim_type, mean_conf, verification
            )
            if should_block:
                return MemoryAugmentedResponse(
                    answer=f"Response blocked for safety: {block_reason}",
                    confidence=mean_conf,
                    uncertainty=uncertainty,
                    sources=sources,
                    verification=verification,
                    governance=governance_decision,
                )

        # Build response
        response = MemoryAugmentedResponse(
            answer=answer,
            confidence=mean_conf,
            uncertainty=uncertainty,
            sources=sources,
            verification=verification,
            governance=governance_decision,
            raw_response=GenerationResult(
                text=samples[0],
                model=self._provider.model if self._provider else "unknown",
                provider=self.config.provider,
                usage={},
                finish_reason=None,
                metadata={},
            ),
            metadata={
                "multi_sample": len(samples) > 1,
                "variance": variance,
                "stakes": stakes,
                "claim_type": claim_type.value,
            },
        )

        # Record for learning
        self._query_history.append({
            "question": question,
            "answer": answer,
            "confidence": mean_conf,
            "timestamp": datetime.now().isoformat(),
        })

        return response

    async def recall(
        self,
        query: str,
        k: int = 10,
        strategy: Optional[RetrievalStrategy] = None,
    ) -> List[MemorySource]:
        """
        Recall memories similar to query.

        Args:
            query: Query text
            k: Number of memories to retrieve
            strategy: Retrieval strategy (uses config default if not specified)

        Returns:
            List of MemorySource objects
        """
        strategy = strategy or self.config.retrieval_strategy
        return await self._retrieve_context(query, k=k, strategy=strategy)

    async def reflect(
        self,
        response: MemoryAugmentedResponse,
        feedback: Dict[str, Any],
    ) -> None:
        """
        Reflect on a response and learn from feedback.

        Args:
            response: The response to reflect on
            feedback: Feedback dictionary (e.g., {"helpful": True, "rating": 5})
        """
        if self.config.learning_mode == LearningMode.DISABLED:
            return

        # Record feedback
        self._learning_buffer.append({
            "type": "feedback",
            "question": self._query_history[-1]["question"] if self._query_history else None,
            "answer": response.answer,
            "confidence": response.confidence,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
        })

        # Update pattern weights based on feedback
        if feedback.get("helpful", False):
            # Strengthen patterns used in this response
            for source in response.sources:
                pattern_key = f"source:{source.source_type}"
                self._pattern_weights[pattern_key] = (
                    self._pattern_weights.get(pattern_key, 0.5) +
                    self.config.learning_rate
                )
        else:
            # Weaken patterns used in this response
            for source in response.sources:
                pattern_key = f"source:{source.source_type}"
                self._pattern_weights[pattern_key] = (
                    self._pattern_weights.get(pattern_key, 0.5) -
                    self.config.learning_rate * 0.5
                )

        # Clamp weights
        for key in self._pattern_weights:
            self._pattern_weights[key] = max(0.1, min(0.9, self._pattern_weights[key]))

        # Flush if buffer is large
        if len(self._learning_buffer) >= 100:
            await self._flush_learning_buffer()

    # =========================================================================
    # Memory Operations
    # =========================================================================

    async def _store_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "user",
    ) -> Optional[str]:
        """Store content in memory backend."""
        if self._memory_backend is None:
            return None

        try:
            if hasattr(self._memory_backend, 'store'):
                # Use HoloLoom memory backend - construct Memory object
                memory = Memory(
                    id=str(uuid.uuid4()),
                    text=content,
                    timestamp=datetime.now(),
                    context={"source": source},
                    metadata=metadata or {},
                )
                node_id = await self._memory_backend.store(memory)
                return node_id
            elif hasattr(self._memory_backend, 'add'):
                # Fallback in-memory backend
                node_id = self._memory_backend.add(content, metadata)
                return node_id
        except Exception as e:
            logger.warning(f"Failed to store memory: {e}")
            return None

        return None

    async def _retrieve_context(
        self,
        query: str,
        k: Optional[int] = None,
        strategy: Optional[RetrievalStrategy] = None,
    ) -> List[MemorySource]:
        """Retrieve context from memory."""
        k = k or self.config.max_retrieved_memories
        strategy = strategy or self.config.retrieval_strategy

        sources = []

        if self._memory_backend is None:
            return sources

        try:
            if strategy == RetrievalStrategy.SEMANTIC:
                sources = await self._semantic_retrieval(query, k)
            elif strategy == RetrievalStrategy.GRAPH:
                sources = await self._graph_retrieval(query, k)
            elif strategy == RetrievalStrategy.HYBRID:
                # Combine semantic and graph retrieval
                semantic = await self._semantic_retrieval(query, k // 2)
                graph = await self._graph_retrieval(query, k // 2)
                sources = self._merge_sources(semantic, graph)
            elif strategy == RetrievalStrategy.ADAPTIVE:
                # Choose based on query characteristics
                sources = await self._adaptive_retrieval(query, k)
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")

        return sources[:k]

    async def _semantic_retrieval(
        self,
        query: str,
        k: int,
    ) -> List[MemorySource]:
        """Semantic (vector) retrieval."""
        sources = []

        if hasattr(self._memory_backend, 'search') or hasattr(self._memory_backend, 'recall'):
            try:
                # Try HoloLoom recall
                if hasattr(self._memory_backend, 'recall'):
                    results = await self._memory_backend.recall(query, k=k)
                else:
                    results = self._memory_backend.search(query, k=k)

                for r in results:
                    sources.append(MemorySource(
                        content=getattr(r, 'content', str(r)),
                        node_id=getattr(r, 'id', str(id(r))),
                        relevance=getattr(r, 'score', 0.5),
                        source_type="semantic",
                        metadata=getattr(r, 'metadata', {}),
                    ))
            except Exception as e:
                logger.warning(f"Semantic retrieval failed: {e}")

        return sources

    async def _graph_retrieval(
        self,
        query: str,
        k: int,
    ) -> List[MemorySource]:
        """Graph-based retrieval."""
        sources = []

        if self._knowledge_graph is not None:
            try:
                # Try to get related nodes from knowledge graph
                if hasattr(self._knowledge_graph, 'get_related'):
                    results = self._knowledge_graph.get_related(query, k=k)
                elif hasattr(self._knowledge_graph, 'search'):
                    results = self._knowledge_graph.search(query, k=k)
                else:
                    return sources

                for r in results:
                    sources.append(MemorySource(
                        content=getattr(r, 'content', str(r)),
                        node_id=getattr(r, 'id', str(id(r))),
                        relevance=getattr(r, 'score', 0.5),
                        source_type="graph",
                        metadata=getattr(r, 'metadata', {}),
                    ))
            except Exception as e:
                logger.warning(f"Graph retrieval failed: {e}")

        return sources

    async def _adaptive_retrieval(
        self,
        query: str,
        k: int,
    ) -> List[MemorySource]:
        """Adaptively choose retrieval strategy."""
        # Simple heuristic: use pattern weights
        semantic_weight = self._pattern_weights.get("source:semantic", 0.5)
        graph_weight = self._pattern_weights.get("source:graph", 0.5)

        semantic_k = int(k * semantic_weight / (semantic_weight + graph_weight))
        graph_k = k - semantic_k

        semantic = await self._semantic_retrieval(query, max(1, semantic_k))
        graph = await self._graph_retrieval(query, max(1, graph_k))

        return self._merge_sources(semantic, graph)

    def _merge_sources(
        self,
        sources1: List[MemorySource],
        sources2: List[MemorySource],
    ) -> List[MemorySource]:
        """Merge two source lists, removing duplicates."""
        seen_ids = set()
        merged = []

        all_sources = sources1 + sources2
        all_sources.sort(key=lambda s: s.relevance, reverse=True)

        for source in all_sources:
            if source.node_id not in seen_ids:
                seen_ids.add(source.node_id)
                merged.append(source)

        return merged

    # =========================================================================
    # Prompt Building
    # =========================================================================

    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return """You are a helpful AI assistant with access to a knowledge base.
When answering questions:
1. Use the provided context when relevant
2. Be clear about uncertainty
3. Cite sources when possible
4. Prefer factual accuracy over speculation

If you're uncertain, say so explicitly."""

    def _build_query_prompt(self, question: str, context: str) -> str:
        """Build prompt with context."""
        if context:
            return f"""Context from knowledge base:
{context}

Question: {question}

Please answer based on the context above. If the context doesn't contain relevant information, say so."""
        else:
            return f"""Question: {question}

Please answer to the best of your knowledge."""

    def _format_context(self, sources: List[MemorySource]) -> str:
        """Format sources into context string."""
        if not sources:
            return ""

        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(f"[{i}] {source.content}")

        return "\n\n".join(context_parts)

    # =========================================================================
    # Response Processing
    # =========================================================================

    def _extract_confidence(self, response_text: str) -> float:
        """Extract confidence from response text."""
        # Simple heuristic: look for uncertainty markers
        uncertainty_markers = [
            "I'm not sure", "I'm uncertain", "I don't know",
            "might be", "could be", "possibly", "perhaps",
            "I think", "I believe", "probably",
        ]

        text_lower = response_text.lower()
        uncertainty_count = sum(
            1 for marker in uncertainty_markers
            if marker.lower() in text_lower
        )

        # Base confidence of 0.8, reduced by uncertainty markers
        confidence = 0.8 - (uncertainty_count * 0.1)
        return max(0.3, min(0.95, confidence))

    def _memory_only_response(
        self,
        question: str,
        sources: List[MemorySource],
    ) -> MemoryAugmentedResponse:
        """Generate response from memory only (no LLM)."""
        if not sources:
            answer = "I don't have any information about this topic in my memory."
            confidence = 0.1
        else:
            # Use most relevant source
            answer = f"Based on my memory:\n\n{sources[0].content}"
            if len(sources) > 1:
                answer += f"\n\nRelated: {sources[1].content[:200]}..."
            confidence = sources[0].relevance * 0.7  # Discount without LLM

        return MemoryAugmentedResponse(
            answer=answer,
            confidence=confidence,
            uncertainty=UncertaintyEnvelope.from_confidence(
                confidence,
                sources=[
                    UncertaintySource(
                        type="no_llm_available",
                        contribution=0.3,
                        details="Memory-only response without LLM",
                    )
                ],
            ),
            sources=sources,
            metadata={"memory_only": True},
        )

    async def _verify_response(
        self,
        response: str,
        claim_type: ClaimType,
        confidence: float,
    ) -> VerificationStatus:
        """Verify response using verification engine."""
        if self._verification_engine is None:
            return VerificationStatus(tier_1_passed=True)

        try:
            return await self._verification_engine.verify(
                response=response,
                claims=[response],  # Simplified: treat whole response as one claim
                claim_type=claim_type,
                confidence=confidence,
            )
        except Exception as e:
            logger.warning(f"Verification failed: {e}")
            return VerificationStatus(tier_1_passed=True)

    # =========================================================================
    # Learning
    # =========================================================================

    async def _flush_learning_buffer(self) -> None:
        """Flush learning buffer to persistent storage."""
        if not self._learning_buffer:
            return

        try:
            # TODO: Implement persistent learning storage
            # For now, just log
            logger.debug(f"Flushing {len(self._learning_buffer)} learning records")
            self._learning_buffer.clear()
        except Exception as e:
            logger.warning(f"Failed to flush learning buffer: {e}")

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "buffer_size": len(self._learning_buffer),
            "pattern_weights": dict(self._pattern_weights),
            "query_count": len(self._query_history),
            "session_id": self._session_id,
        }


# =============================================================================
# Fallback In-Memory Backend
# =============================================================================

class InMemoryBackend:
    """Simple in-memory backend for when HoloLoom is not available."""

    def __init__(self):
        self._memories: Dict[str, Dict[str, Any]] = {}
        self._counter = 0

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add memory."""
        self._counter += 1
        node_id = f"mem_{self._counter}"
        self._memories[node_id] = {
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        return node_id

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Simple substring search."""
        results = []
        query_lower = query.lower()

        for node_id, data in self._memories.items():
            content = data["content"]
            if query_lower in content.lower():
                results.append({
                    "id": node_id,
                    "content": content,
                    "score": 0.8,  # Exact match
                    "metadata": data["metadata"],
                })
            else:
                # Fuzzy match based on word overlap
                query_words = set(query_lower.split())
                content_words = set(content.lower().split())
                overlap = len(query_words & content_words)
                if overlap > 0:
                    score = overlap / max(len(query_words), 1) * 0.5
                    results.append({
                        "id": node_id,
                        "content": content,
                        "score": score,
                        "metadata": data["metadata"],
                    })

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def get_related(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Alias for search."""
        return self.search(query, k)

    async def recall(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Async recall wrapper."""
        return self.search(query, k)

    async def store(
        self,
        content: str,
        metadata: Dict[str, Any],
        source: str,
    ) -> str:
        """Async store wrapper."""
        metadata["source"] = source
        return self.add(content, metadata)

    async def close(self) -> None:
        """Cleanup."""
        pass


# =============================================================================
# Convenience Functions
# =============================================================================

async def create_memory_augmented_llm(
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> MemoryAugmentedLLM:
    """
    Create and initialize a MemoryAugmentedLLM instance.

    Convenience function for non-context-manager usage.

    Args:
        provider: LLM provider name
        model: Model name
        api_key: API key
        **kwargs: Additional config options

    Returns:
        Initialized MemoryAugmentedLLM

    Example:
        llm = await create_memory_augmented_llm("anthropic")
        try:
            response = await llm.query("What is Thompson Sampling?")
        finally:
            await llm._cleanup()
    """
    llm = MemoryAugmentedLLM(
        provider=provider,
        model=model,
        api_key=api_key,
        **kwargs,
    )
    await llm._initialize()
    return llm


def create_config(
    provider: str = "anthropic",
    learning_mode: str = "passive",
    enable_verification: bool = True,
    **kwargs,
) -> LLMConfig:
    """
    Create LLMConfig with sensible defaults.

    Args:
        provider: LLM provider
        learning_mode: Learning mode ("disabled", "passive", "active", "research")
        enable_verification: Whether to verify responses
        **kwargs: Additional config options

    Returns:
        LLMConfig instance
    """
    learning_mode_enum = LearningMode(learning_mode)

    return LLMConfig(
        provider=provider,
        learning_mode=learning_mode_enum,
        require_verification=enable_verification,
        **kwargs,
    )
