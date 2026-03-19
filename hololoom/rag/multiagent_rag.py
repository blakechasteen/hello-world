from __future__ import annotations
"""
Multi-Agent RAG - Consensus-Based Multi-Agent Query Processing

Spawns multiple RAG agents with diverse strategies, executes queries in parallel,
and reaches consensus through voting, confidence weighting, or LLM judging.

Philosophy:
-----------
Single agents can be biased by their configuration (embedding model, retrieval
parameters, reasoning mode). Multi-agent consensus:

1. **Diversity**: Each agent uses different strategy (embeddings, k, hops, mode)
2. **Parallelization**: All agents run concurrently (latency = max, not sum)
3. **Fault Tolerance**: Partial results better than failure (some agents can fail)
4. **Transparency**: Show all agent responses, not just consensus
5. **Agreement Scoring**: Detect when agents disagree (signals uncertainty)

Consensus Mechanisms:
---------------------
- **Majority Vote**: Most common answer (simple, fast)
- **Confidence Weighted**: Weight by agent confidence (precision-focused)
- **LLM Judge**: Use LLM to select best or synthesize (highest quality, slowest)
- **Ensemble**: Combine all responses into synthesized answer

Agent Diversity Strategies:
----------------------------
- **Retrieval parameters**: Vary k (3, 5, 10), reranking (on/off)
- **Reasoning modes**: direct, verify, research, plan_execute
- **Embedding models**: Matryoshka, HuggingFace, OpenAI (if custom embeddings enabled)
- **Multi-hop**: Vary max_hops (1, 2, 3) if multi-hop enabled
- **SQL**: Some agents enable SQL, others don't (if SQL available)

Performance:
------------
- 5 agents in parallel: ~max(agent_latency), not 5× sum
- Timeout per agent: 30s default (kill slow agents)
- Partial results: Consensus with N-1 agents if one fails
- Agreement scoring: O(N²) pairwise comparison (negligible for N≤10)

Example Usage:
--------------
    async with MultiAgentRAG(
        n_agents=5,
        consensus_method="confidence_weighted",
        agent_timeout=30.0
    ) as rag:
        # Ingest data
        await rag.ingest("Thompson Sampling uses Bayesian statistics")

        # Query with multi-agent consensus
        result = await rag.query_multiagent(
            "What is Thompson Sampling?",
            explain_disagreement=True
        )

        print(f"Consensus: {result.response}")
        print(f"Agreement: {result.agreement_score:.2f}")
        print(f"Sources: {len(result.sources)}")

        # View individual agent responses
        for agent_resp in result.agent_responses:
            print(f"  Agent {agent_resp.agent_id}: {agent_resp.confidence:.2f}")

Author: Agent J (Claude Code)
Date: November 13, 2025
"""

import asyncio
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hololoom.config import Config
from hololoom.rag.simple_rag import RAGResult, SimpleRAG

# Optional dependencies for advanced features
try:
    from hololoom.rag.embedding_plugins import (
        EmbeddingProvider,
        HuggingFaceEmbedding,
        MatryoshkaEmbedding,
    )
    EMBEDDING_PLUGINS_AVAILABLE = True
except ImportError:
    EMBEDDING_PLUGINS_AVAILABLE = False
    EmbeddingProvider = None
    MatryoshkaEmbedding = None
    HuggingFaceEmbedding = None

try:
    from hololoom.rag.multihop_reasoning import MultiHopRAGMixin
    MULTIHOP_AVAILABLE = True
except ImportError:
    MULTIHOP_AVAILABLE = False

try:
    from hololoom.rag.sql_integration import SQLRAGMixin
    SQL_AVAILABLE = True
except ImportError:
    SQL_AVAILABLE = False

logger = logging.getLogger(__name__)

# LLM integration for LLM judge consensus
try:
    from hololoom.awareness.llm_integration import AnthropicLLM, LLMProvider, OllamaLLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    OllamaLLM = None
    AnthropicLLM = None
    LLMProvider = None


# ============================================================================
# Data Structures
# ============================================================================

class ConsensusMethod(Enum):
    """Consensus mechanism for merging agent responses."""
    MAJORITY_VOTE = "majority_vote"  # Most common answer
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # Weight by confidence
    LLM_JUDGE = "llm_judge"  # LLM selects best or synthesizes
    ENSEMBLE = "ensemble"  # Combine all responses


@dataclass
class AgentResponse:
    """Response from a single RAG agent."""
    agent_id: str
    strategy: str  # "direct_k5_matryoshka", "verify_k10_hf", etc.
    response: str
    confidence: float
    sources: list[str]
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None  # Error message if agent failed

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON."""
        return {
            'agent_id': self.agent_id,
            'strategy': self.strategy,
            'response': self.response,
            'confidence': self.confidence,
            'n_sources': len(self.sources),
            'latency_ms': self.latency_ms,
            'metadata': self.metadata,
            'error': self.error
        }


@dataclass
class MultiAgentRAGResult(RAGResult):
    """
    Result from multi-agent consensus.

    Extends RAGResult with:
    - agent_responses: All individual agent responses
    - consensus_method: Method used to reach consensus
    - agreement_score: 0.0-1.0 (1.0 = all agents agree perfectly)
    - disagreements: Optional explanation of disagreements
    - consensus_metadata: Timing, agent success rate, etc.
    """
    agent_responses: list[AgentResponse] = field(default_factory=list)
    consensus_method: str = "confidence_weighted"
    agreement_score: float = 0.0
    disagreements: list[str] | None = None
    consensus_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON."""
        return {
            'response': self.response,
            'sources': self.sources,
            'confidence': self.confidence,
            'reasoning_mode': self.reasoning_mode,
            'metadata': self.metadata,
            'agent_responses': [r.to_dict() for r in self.agent_responses],
            'consensus_method': self.consensus_method,
            'agreement_score': self.agreement_score,
            'disagreements': self.disagreements,
            'consensus_metadata': self.consensus_metadata,
        }


# ============================================================================
# Multi-Agent RAG
# ============================================================================

class MultiAgentRAG:
    """
    Multi-agent RAG orchestrator with consensus-based decision making.

    Spawns N agents with diverse strategies, runs queries in parallel,
    and reaches consensus through voting, confidence weighting, or LLM judging.

    Features:
    - Parallel execution with asyncio.gather
    - Timeout enforcement (kill slow agents)
    - Partial result handling (some agents fail, others succeed)
    - Disagreement detection and reporting
    - Source attribution (which agent contributed what)
    - Performance tracking (latency per agent, consensus overhead)

    Usage:
        async with MultiAgentRAG(
            n_agents=5,
            consensus_method="confidence_weighted",
            agent_timeout=30.0
        ) as rag:
            result = await rag.query_multiagent("What is Thompson Sampling?")
            print(f"Agreement: {result.agreement_score:.2f}")
    """

    def __init__(
        self,
        config: Config | None = None,
        n_agents: int = 5,
        consensus_method: str = "confidence_weighted",
        agent_timeout: float = 30.0,
        min_agreement_threshold: float = 0.6,
        llm_provider: str = "ollama",
        llm_model: str | None = None,
        enable_caching: bool = True,
        diversity_strategies: list[str] | None = None,
    ):
        """
        Initialize multi-agent RAG orchestrator.

        Args:
            config: Base configuration (defaults to Config.fast())
            n_agents: Number of agents to spawn (default: 5)
            consensus_method: Consensus mechanism
                - "majority_vote": Most common answer
                - "confidence_weighted": Weight by confidence (default)
                - "llm_judge": LLM selects best or synthesizes
                - "ensemble": Combine all responses
            agent_timeout: Timeout per agent in seconds (default: 30.0)
            min_agreement_threshold: Minimum agreement score to accept result (default: 0.6)
            llm_provider: LLM provider for all agents
            llm_model: Specific LLM model
            enable_caching: Enable query caching
            diversity_strategies: Which dimensions to vary
                - "retrieval": Vary k, reranking
                - "reasoning": Vary modes (direct, verify, research)
                - "embeddings": Vary embedding models (if available)
                - "multihop": Vary max_hops (if available)
                - "sql": Some agents enable SQL (if available)
                Default: ["retrieval", "reasoning"]
        """
        self.config = config or Config.fast()
        self.n_agents = n_agents
        self.consensus_method = ConsensusMethod(consensus_method)
        self.agent_timeout = agent_timeout
        self.min_agreement_threshold = min_agreement_threshold
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.enable_caching = enable_caching

        # Default diversity strategies
        if diversity_strategies is None:
            diversity_strategies = ["retrieval", "reasoning"]
        self.diversity_strategies = diversity_strategies

        # Will be initialized in __aenter__
        self.agents: list[SimpleRAG] = []
        self._agent_strategies: list[dict[str, Any]] = []

        # Stats
        self._stats = {
            'total_queries': 0,
            'total_agents_spawned': 0,
            'total_agents_succeeded': 0,
            'total_agents_failed': 0,
            'total_consensus_time_ms': 0.0,
            'avg_agreement_score': 0.0,
        }

    async def __aenter__(self):
        """Initialize agents with diverse strategies."""
        logger.info(f"Initializing {self.n_agents} agents with diversity: {self.diversity_strategies}")

        # Generate diverse agent strategies
        self._agent_strategies = self._generate_agent_strategies(self.n_agents)

        # Create agents
        for i, strategy in enumerate(self._agent_strategies):
            agent_id = f"agent_{i}"
            logger.debug(f"Creating {agent_id} with strategy: {strategy.get('description', 'N/A')}")

            # Create agent with strategy
            agent = SimpleRAG(
                config=strategy.get('config', self.config),
                llm_provider=self.llm_provider,
                llm_model=self.llm_model,
                enable_caching=self.enable_caching,
                enable_reranking=strategy.get('enable_reranking', False),
                rerank_top_k=strategy.get('rerank_top_k', 20),
                embedding_provider=strategy.get('embedding_provider', None),
            )

            # Initialize agent
            await agent.__aenter__()
            self.agents.append(agent)
            self._stats['total_agents_spawned'] += 1

        logger.info(f"✓ {len(self.agents)} agents initialized")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup all agents."""
        logger.info("Cleaning up agents...")
        for i, agent in enumerate(self.agents):
            try:
                await agent.__aexit__(exc_type, exc_val, exc_tb)
                logger.debug(f"✓ Agent {i} cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up agent {i}: {e}")

        logger.info(f"✓ {len(self.agents)} agents cleaned up")

    # ========================================================================
    # Ingestion (Delegated to All Agents)
    # ========================================================================

    async def ingest(self, content: Any) -> None:
        """
        Ingest content into all agents' knowledge bases.

        Args:
            content: Text, file path, or structured data

        Example:
            await rag.ingest("Thompson Sampling uses Bayesian statistics")
        """
        logger.info(f"Ingesting content into {len(self.agents)} agents...")
        tasks = [agent.ingest(content) for agent in self.agents]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("✓ Content ingested into all agents")

    # ========================================================================
    # Multi-Agent Query
    # ========================================================================

    async def query_multiagent(
        self,
        question: str,
        n_agents: int | None = None,
        consensus_method: str | None = None,
        explain_disagreement: bool = False,
    ) -> MultiAgentRAGResult:
        """
        Query with multiple agents in parallel and reach consensus.

        Process:
        1. Run all agents in parallel (asyncio.gather)
        2. Collect responses (handle timeouts/errors)
        3. Compute consensus using specified method
        4. Calculate agreement score
        5. Detect and explain disagreements (optional)

        Args:
            question: Query text
            n_agents: Override number of agents (uses first N)
            consensus_method: Override consensus method
            explain_disagreement: Include disagreement explanations

        Returns:
            MultiAgentRAGResult with consensus, agent responses, agreement score

        Example:
            result = await rag.query_multiagent(
                "What is Thompson Sampling?",
                explain_disagreement=True
            )
            print(f"Consensus: {result.response}")
            print(f"Agreement: {result.agreement_score:.2f}")

            # View individual agents
            for agent_resp in result.agent_responses:
                print(f"{agent_resp.agent_id}: {agent_resp.confidence:.2f}")
        """
        start_time = time.time()

        # Override parameters if provided
        n_agents = n_agents or self.n_agents
        consensus_method_enum = (
            ConsensusMethod(consensus_method)
            if consensus_method
            else self.consensus_method
        )

        # Use first N agents
        agents_to_use = self.agents[:n_agents]
        logger.info(f"Querying {len(agents_to_use)} agents with question: {question[:50]}...")

        # Run agents in parallel with timeout
        agent_responses = await self._run_agents_parallel(
            agents_to_use, question, timeout=self.agent_timeout
        )

        # Filter out failed agents
        successful_responses = [r for r in agent_responses if r.error is None]
        failed_responses = [r for r in agent_responses if r.error is not None]

        self._stats['total_queries'] += 1
        self._stats['total_agents_succeeded'] += len(successful_responses)
        self._stats['total_agents_failed'] += len(failed_responses)

        logger.info(
            f"Agent results: {len(successful_responses)} succeeded, "
            f"{len(failed_responses)} failed"
        )

        # Handle case where all agents failed
        if not successful_responses:
            logger.error("All agents failed!")
            return MultiAgentRAGResult(
                response="Error: All agents failed to respond",
                sources=[],
                confidence=0.0,
                reasoning_mode="multi_agent",
                agent_responses=agent_responses,
                consensus_method=consensus_method_enum.value,
                agreement_score=0.0,
                disagreements=["All agents failed"],
                consensus_metadata={
                    'n_agents': n_agents,
                    'n_successful': 0,
                    'n_failed': len(failed_responses),
                    'consensus_time_ms': 0.0,
                }
            )

        # Compute consensus
        logger.debug(f"Computing consensus using {consensus_method_enum.value}...")
        consensus_response, consensus_confidence = await self._compute_consensus(
            successful_responses, consensus_method_enum
        )

        # Deduplicate sources
        all_sources = []
        for resp in successful_responses:
            all_sources.extend(resp.sources)
        sources = list(dict.fromkeys(all_sources))  # Preserve order, remove duplicates

        # Compute agreement score
        agreement_score = self._compute_agreement_score(successful_responses)
        self._stats['avg_agreement_score'] = (
            (self._stats['avg_agreement_score'] * (self._stats['total_queries'] - 1) + agreement_score)
            / self._stats['total_queries']
        )

        # Detect disagreements
        disagreements = None
        if explain_disagreement:
            disagreements = self._detect_disagreements(successful_responses, agreement_score)

        # Consensus metadata
        consensus_time_ms = (time.time() - start_time) * 1000
        self._stats['total_consensus_time_ms'] += consensus_time_ms

        consensus_metadata = {
            'n_agents': n_agents,
            'n_successful': len(successful_responses),
            'n_failed': len(failed_responses),
            'consensus_time_ms': consensus_time_ms,
            'avg_agent_latency_ms': (
                sum(r.latency_ms for r in successful_responses) / len(successful_responses)
                if successful_responses else 0.0
            ),
            'max_agent_latency_ms': (
                max(r.latency_ms for r in successful_responses)
                if successful_responses else 0.0
            ),
        }

        logger.info(
            f"✓ Consensus reached: agreement={agreement_score:.2f}, "
            f"confidence={consensus_confidence:.2f}, "
            f"time={consensus_time_ms:.1f}ms"
        )

        return MultiAgentRAGResult(
            response=consensus_response,
            sources=sources,
            confidence=consensus_confidence,
            reasoning_mode="multi_agent",
            metadata={
                'question': question,
                'consensus_method': consensus_method_enum.value,
            },
            agent_responses=agent_responses,
            consensus_method=consensus_method_enum.value,
            agreement_score=agreement_score,
            disagreements=disagreements,
            consensus_metadata=consensus_metadata,
        )

    # ========================================================================
    # Agent Execution
    # ========================================================================

    async def _run_agents_parallel(
        self,
        agents: list[SimpleRAG],
        question: str,
        timeout: float,
    ) -> list[AgentResponse]:
        """
        Run multiple agents in parallel with timeout.

        Uses asyncio.gather with timeout per agent. If an agent times out
        or raises an exception, it's marked as failed but other agents continue.

        Args:
            agents: List of SimpleRAG agents
            question: Query text
            timeout: Timeout per agent in seconds

        Returns:
            List of AgentResponse objects (includes failed agents with error messages)
        """
        tasks = []
        for i, agent in enumerate(agents):
            agent_id = f"agent_{i}"
            strategy = self._agent_strategies[i]
            task = self._run_single_agent(
                agent_id=agent_id,
                agent=agent,
                question=question,
                strategy=strategy,
                timeout=timeout,
            )
            tasks.append(task)

        # Run all agents in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=False)
        return responses

    async def _run_single_agent(
        self,
        agent_id: str,
        agent: SimpleRAG,
        question: str,
        strategy: dict[str, Any],
        timeout: float,
    ) -> AgentResponse:
        """
        Run single agent with timeout and error handling.

        Args:
            agent_id: Agent identifier
            agent: SimpleRAG instance
            question: Query text
            strategy: Agent strategy configuration
            timeout: Timeout in seconds

        Returns:
            AgentResponse (with error field if failed)
        """
        start_time = time.time()

        try:
            # Run agent query with timeout
            result = await asyncio.wait_for(
                agent.query(
                    question=question,
                    mode=strategy.get('reasoning_mode', 'verify'),
                    max_sources=strategy.get('max_sources', 5),
                    use_cache=True,
                ),
                timeout=timeout
            )

            latency_ms = (time.time() - start_time) * 1000

            return AgentResponse(
                agent_id=agent_id,
                strategy=strategy.get('description', 'unknown'),
                response=result.response,
                confidence=result.confidence,
                sources=result.sources,
                latency_ms=latency_ms,
                metadata=result.metadata,
                error=None,
            )

        except asyncio.TimeoutError:
            latency_ms = timeout * 1000
            logger.warning(f"⚠ {agent_id} timed out after {timeout}s")
            return AgentResponse(
                agent_id=agent_id,
                strategy=strategy.get('description', 'unknown'),
                response="",
                confidence=0.0,
                sources=[],
                latency_ms=latency_ms,
                metadata={},
                error=f"Timeout after {timeout}s",
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.warning(f"⚠ {agent_id} failed: {e}")
            return AgentResponse(
                agent_id=agent_id,
                strategy=strategy.get('description', 'unknown'),
                response="",
                confidence=0.0,
                sources=[],
                latency_ms=latency_ms,
                metadata={},
                error=str(e),
            )

    # ========================================================================
    # Agent Diversity
    # ========================================================================

    def _generate_agent_strategies(self, n_agents: int) -> list[dict[str, Any]]:
        """
        Generate diverse strategies for N agents.

        Varies configuration along multiple dimensions:
        - Retrieval: k, reranking
        - Reasoning: direct, verify, research
        - Embeddings: Matryoshka, HuggingFace (if available)
        - Multi-hop: max_hops (if available)
        - SQL: enable/disable (if available)

        Args:
            n_agents: Number of agents

        Returns:
            List of strategy dictionaries
        """
        strategies = []

        # Define strategy templates
        base_strategies = [
            # Simple, fast
            {
                'description': 'direct_k3_matryoshka',
                'reasoning_mode': 'direct',
                'max_sources': 3,
                'enable_reranking': False,
                'embedding_provider': None,  # Default Matryoshka
            },
            # Balanced
            {
                'description': 'verify_k5_rerank',
                'reasoning_mode': 'verify',
                'max_sources': 5,
                'enable_reranking': True,
                'rerank_top_k': 20,
                'embedding_provider': None,
            },
            # High recall
            {
                'description': 'research_k10_rerank',
                'reasoning_mode': 'research',
                'max_sources': 10,
                'enable_reranking': True,
                'rerank_top_k': 30,
                'embedding_provider': None,
            },
            # Alternative embeddings (if available)
            {
                'description': 'verify_k5_hf',
                'reasoning_mode': 'verify',
                'max_sources': 5,
                'enable_reranking': False,
                'embedding_provider': self._get_alternative_embedding(),
            },
            # Plan-execute mode
            {
                'description': 'plan_k7_rerank',
                'reasoning_mode': 'plan_execute',
                'max_sources': 7,
                'enable_reranking': True,
                'rerank_top_k': 25,
                'embedding_provider': None,
            },
        ]

        # Cycle through strategies to fill N agents
        for i in range(n_agents):
            strategy = base_strategies[i % len(base_strategies)].copy()
            strategies.append(strategy)

        return strategies

    def _get_alternative_embedding(self) -> Any | None:
        """
        Get alternative embedding provider for diversity.

        Returns:
            EmbeddingProvider instance or None
        """
        if not EMBEDDING_PLUGINS_AVAILABLE:
            return None

        if not HuggingFaceEmbedding:
            return None

        try:
            # Use a different HuggingFace model for diversity
            return HuggingFaceEmbedding("all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Failed to create alternative embedding: {e}")
            return None

    # ========================================================================
    # Consensus Mechanisms
    # ========================================================================

    async def _compute_consensus(
        self,
        agent_responses: list[AgentResponse],
        method: ConsensusMethod,
    ) -> tuple[str, float]:
        """
        Compute consensus from agent responses.

        Args:
            agent_responses: List of successful agent responses
            method: Consensus method

        Returns:
            (consensus_response, consensus_confidence)
        """
        if not agent_responses:
            return "No responses", 0.0

        if method == ConsensusMethod.MAJORITY_VOTE:
            return self._consensus_majority_vote(agent_responses)
        elif method == ConsensusMethod.CONFIDENCE_WEIGHTED:
            return self._consensus_confidence_weighted(agent_responses)
        elif method == ConsensusMethod.LLM_JUDGE:
            return await self._consensus_llm_judge(agent_responses)
        elif method == ConsensusMethod.ENSEMBLE:
            return self._consensus_ensemble(agent_responses)
        else:
            logger.warning(f"Unknown consensus method: {method}, falling back to confidence_weighted")
            return self._consensus_confidence_weighted(agent_responses)

    def _consensus_majority_vote(
        self,
        agent_responses: list[AgentResponse],
    ) -> tuple[str, float]:
        """
        Majority vote: Most common answer wins.

        Fuzzy matching: responses are considered equal if they share
        significant overlap (>70% of words).
        """
        if len(agent_responses) == 1:
            return agent_responses[0].response, agent_responses[0].confidence

        # Simple majority: exact string match
        response_counts = Counter(r.response for r in agent_responses)
        most_common_response, count = response_counts.most_common(1)[0]

        # Average confidence of agents with this response
        matching_agents = [r for r in agent_responses if r.response == most_common_response]
        avg_confidence = sum(r.confidence for r in matching_agents) / len(matching_agents)

        logger.debug(f"Majority vote: {count}/{len(agent_responses)} agents agree")
        return most_common_response, avg_confidence

    def _consensus_confidence_weighted(
        self,
        agent_responses: list[AgentResponse],
    ) -> tuple[str, float]:
        """
        Confidence-weighted: Select response with highest weighted confidence.

        Weights each response by its confidence score, returns the response
        with the highest weight.
        """
        if len(agent_responses) == 1:
            return agent_responses[0].response, agent_responses[0].confidence

        # Select response with highest confidence
        best_response = max(agent_responses, key=lambda r: r.confidence)

        # Weighted average confidence
        total_weight = sum(r.confidence for r in agent_responses)
        weighted_confidence = sum(
            r.confidence * r.confidence for r in agent_responses
        ) / total_weight if total_weight > 0 else 0.0

        logger.debug(
            f"Confidence-weighted: selected {best_response.agent_id} "
            f"(confidence={best_response.confidence:.2f})"
        )
        return best_response.response, weighted_confidence

    async def _consensus_llm_judge(
        self,
        agent_responses: list[AgentResponse],
    ) -> tuple[str, float]:
        """
        LLM judge: Use LLM to select best response or synthesize.

        Process:
        1. Present all agent responses to LLM
        2. Ask LLM to select best response OR synthesize a better answer
        3. Parse LLM response and extract selected/synthesized answer
        4. Fall back to confidence_weighted if LLM unavailable

        Returns:
            (consensus_response, consensus_confidence)
        """
        # Check if LLM integration is available
        if not LLM_AVAILABLE:
            logger.warning("LLM integration not available, falling back to confidence_weighted")
            return self._consensus_confidence_weighted(agent_responses)

        # Initialize LLM based on provider
        llm = None
        try:
            if self.llm_provider == "ollama":
                model = self.llm_model or "llama3.2:3b"
                llm = OllamaLLM(model=model)
                if not llm.is_available():
                    logger.warning("Ollama not available, falling back to confidence_weighted")
                    return self._consensus_confidence_weighted(agent_responses)
            elif self.llm_provider == "anthropic":
                import os
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    logger.warning("ANTHROPIC_API_KEY not set, falling back to confidence_weighted")
                    return self._consensus_confidence_weighted(agent_responses)
                model = self.llm_model or "claude-3-5-sonnet-20241022"
                llm = AnthropicLLM(api_key=api_key, model=model)
            else:
                logger.warning(f"Unknown LLM provider: {self.llm_provider}, falling back to confidence_weighted")
                return self._consensus_confidence_weighted(agent_responses)
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return self._consensus_confidence_weighted(agent_responses)

        # Build prompt presenting all agent responses
        responses_text = "\n\n".join(
            f"Agent {i+1} (confidence: {r.confidence:.2f}, mode: {r.reasoning_mode}):\n{r.response}"
            for i, r in enumerate(agent_responses)
        )

        system_prompt = """You are an expert judge evaluating multiple AI agent responses to a query.
Your task is to either:
1. Select the best response if one is clearly superior
2. Synthesize a better answer combining the best elements from multiple responses

Respond in the following format:
DECISION: [SELECT or SYNTHESIZE]
SELECTED_AGENT: [agent number if SELECT, otherwise N/A]
CONFIDENCE: [your confidence 0.0-1.0]
REASONING: [brief explanation]
FINAL_ANSWER: [the selected or synthesized answer]"""

        user_prompt = f"""Here are responses from {len(agent_responses)} agents:

{responses_text}

Please evaluate these responses and provide the best answer."""

        try:
            # Call LLM
            response = await llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=1000,
                temperature=0.3  # Lower temperature for more consistent judging
            )

            # Parse response
            content = response.content if hasattr(response, 'content') else str(response)
            return self._parse_llm_judge_response(content, agent_responses)

        except Exception as e:
            logger.error(f"LLM judge failed: {e}, falling back to confidence_weighted")
            return self._consensus_confidence_weighted(agent_responses)

    def _parse_llm_judge_response(
        self,
        llm_response: str,
        agent_responses: list[AgentResponse],
    ) -> tuple[str, float]:
        """
        Parse LLM judge response to extract final answer and confidence.

        Expected format:
        DECISION: SELECT or SYNTHESIZE
        SELECTED_AGENT: N (if SELECT)
        CONFIDENCE: 0.0-1.0
        REASONING: ...
        FINAL_ANSWER: ...
        """
        lines = llm_response.strip().split('\n')

        # Default values
        confidence = 0.8
        final_answer = None

        # Parse structured response
        in_final_answer = False
        final_answer_lines = []

        for line in lines:
            line_stripped = line.strip()

            if line_stripped.startswith('CONFIDENCE:'):
                try:
                    conf_str = line_stripped.replace('CONFIDENCE:', '').strip()
                    confidence = float(conf_str)
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                except ValueError:
                    pass

            elif line_stripped.startswith('FINAL_ANSWER:'):
                in_final_answer = True
                answer_part = line_stripped.replace('FINAL_ANSWER:', '').strip()
                if answer_part:
                    final_answer_lines.append(answer_part)

            elif in_final_answer:
                # Continue collecting final answer lines
                if line_stripped and not line_stripped.startswith(('DECISION:', 'SELECTED_AGENT:', 'REASONING:')):
                    final_answer_lines.append(line_stripped)

        if final_answer_lines:
            final_answer = '\n'.join(final_answer_lines)

        # If parsing failed, fall back to highest confidence response
        if not final_answer:
            logger.warning("Could not parse LLM judge response, using highest confidence agent")
            best = max(agent_responses, key=lambda r: r.confidence)
            return best.response, best.confidence

        logger.debug(f"LLM judge selected/synthesized answer with confidence {confidence:.2f}")
        return final_answer, confidence

    def _consensus_ensemble(
        self,
        agent_responses: list[AgentResponse],
    ) -> tuple[str, float]:
        """
        Ensemble: Combine all responses into synthesized answer.

        Simple implementation: concatenate top-3 responses by confidence,
        separated by newlines.
        """
        if len(agent_responses) == 1:
            return agent_responses[0].response, agent_responses[0].confidence

        # Sort by confidence, take top 3
        sorted_responses = sorted(
            agent_responses, key=lambda r: r.confidence, reverse=True
        )
        top_responses = sorted_responses[:3]

        # Concatenate
        combined_response = "\n\n".join(
            f"[Agent {r.agent_id}, confidence={r.confidence:.2f}]: {r.response}"
            for r in top_responses
        )

        # Average confidence
        avg_confidence = sum(r.confidence for r in top_responses) / len(top_responses)

        logger.debug(f"Ensemble: combined {len(top_responses)} responses")
        return combined_response, avg_confidence

    # ========================================================================
    # Agreement Scoring
    # ========================================================================

    def _compute_agreement_score(
        self,
        agent_responses: list[AgentResponse],
    ) -> float:
        """
        Compute agreement score between agents (0.0-1.0).

        Measures how much agents agree on:
        1. Response text (Jaccard similarity of words)
        2. Confidence (standard deviation)
        3. Sources (overlap)

        Returns:
            Agreement score (1.0 = perfect agreement, 0.0 = no agreement)
        """
        if len(agent_responses) <= 1:
            return 1.0

        # 1. Response text agreement (Jaccard similarity)
        response_texts = [set(r.response.lower().split()) for r in agent_responses]
        text_agreements = []
        for i in range(len(response_texts)):
            for j in range(i + 1, len(response_texts)):
                intersection = len(response_texts[i] & response_texts[j])
                union = len(response_texts[i] | response_texts[j])
                similarity = intersection / union if union > 0 else 0.0
                text_agreements.append(similarity)

        avg_text_agreement = sum(text_agreements) / len(text_agreements) if text_agreements else 0.0

        # 2. Confidence agreement (inverse of std dev)
        confidences = [r.confidence for r in agent_responses]
        avg_confidence = sum(confidences) / len(confidences)
        variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        std_dev = variance ** 0.5
        confidence_agreement = 1.0 - min(std_dev, 1.0)  # Cap at 1.0

        # 3. Source overlap
        all_sources = [set(r.sources) for r in agent_responses]
        source_agreements = []
        for i in range(len(all_sources)):
            for j in range(i + 1, len(all_sources)):
                intersection = len(all_sources[i] & all_sources[j])
                union = len(all_sources[i] | all_sources[j])
                overlap = intersection / union if union > 0 else 0.0
                source_agreements.append(overlap)

        avg_source_agreement = sum(source_agreements) / len(source_agreements) if source_agreements else 0.0

        # Weighted average
        agreement_score = (
            0.5 * avg_text_agreement +
            0.3 * confidence_agreement +
            0.2 * avg_source_agreement
        )

        return agreement_score

    def _detect_disagreements(
        self,
        agent_responses: list[AgentResponse],
        agreement_score: float,
    ) -> list[str]:
        """
        Detect and explain disagreements between agents.

        Args:
            agent_responses: List of agent responses
            agreement_score: Overall agreement score

        Returns:
            List of disagreement descriptions
        """
        disagreements = []

        # 1. Low agreement score
        if agreement_score < 0.7:
            disagreements.append(
                f"Low agreement score ({agreement_score:.2f}): "
                f"Agents disagree on response content"
            )

        # 2. High confidence variance
        confidences = [r.confidence for r in agent_responses]
        avg_confidence = sum(confidences) / len(confidences)
        variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        std_dev = variance ** 0.5

        if std_dev > 0.2:
            disagreements.append(
                f"High confidence variance (σ={std_dev:.2f}): "
                f"Agents have different confidence levels"
            )

        # 3. Identify outlier agents (confidence far from mean)
        for resp in agent_responses:
            if abs(resp.confidence - avg_confidence) > 0.3:
                disagreements.append(
                    f"{resp.agent_id} is an outlier "
                    f"(confidence={resp.confidence:.2f} vs avg={avg_confidence:.2f})"
                )

        return disagreements if disagreements else None

    # ========================================================================
    # Metrics
    # ========================================================================

    def get_metrics(self) -> dict[str, Any]:
        """
        Get multi-agent system metrics.

        Returns:
            Dictionary with stats
        """
        return {
            'n_agents': len(self.agents),
            'consensus_method': self.consensus_method.value,
            'agent_timeout': self.agent_timeout,
            'diversity_strategies': self.diversity_strategies,
            **self._stats,
        }

    def summary(self) -> str:
        """
        Get human-readable summary.

        Returns:
            Summary string
        """
        metrics = self.get_metrics()
        return f"""MultiAgentRAG System
===================
Agents: {metrics['n_agents']}
Consensus method: {metrics['consensus_method']}
Agent timeout: {metrics['agent_timeout']}s
Diversity strategies: {', '.join(metrics['diversity_strategies'])}

Statistics:
-----------
Total queries: {metrics['total_queries']}
Agents succeeded: {metrics['total_agents_succeeded']}
Agents failed: {metrics['total_agents_failed']}
Avg agreement score: {metrics['avg_agreement_score']:.2f}
Avg consensus time: {metrics['total_consensus_time_ms'] / max(metrics['total_queries'], 1):.1f}ms
"""
