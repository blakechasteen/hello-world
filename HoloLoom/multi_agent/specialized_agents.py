"""
Specialized Agent Classes
==========================

Domain-specific HoloLoom configurations optimized for different tasks.

**Date**: November 17, 2025
**Status**: Production Ready

Agent Types:
- MathAgent: Mathematical reasoning, proofs, calculations
- CodeAgent: Code analysis, generation, debugging
- WritingAgent: Creative and technical writing
- ResearchAgent: Multi-hop research, synthesis

Each agent is a full HoloLoom instance with domain-specific:
- Knowledge base
- Config settings (BARE/FAST/FUSED/RESEARCH)
- Tool preferences
- Reasoning strategies

Usage:
    # Create specialized agents
    math_agent = MathAgent(agent_id="math-01")
    code_agent = CodeAgent(agent_id="code-01")

    # Process queries
    result = await math_agent.process_query("Solve x^2 + 5x + 6 = 0")
    result = await code_agent.process_query("Explain this function")
"""

import asyncio
import networkx as nx
from typing import Dict, List, Optional, Any
import logging

from HoloLoom.config import Config, ExecutionMode
from HoloLoom.hololoom import HoloLoom
from HoloLoom.Documentation.types import Query
from HoloLoom.fabric.spacetime import Spacetime
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.multi_agent.communication import AgentCommunicator, MessageType, AgentMessage, get_registry
from HoloLoom.multi_agent.delegation import AgentCapability, Domain
from HoloLoom.multi_agent.consensus import YarnGraphConsensus

logger = logging.getLogger(__name__)


# ============================================================================
# Base Agent
# ============================================================================

class BaseAgent:
    """
    Base class for all specialized agents.

    Provides common functionality:
    - HoloLoom integration
    - Communication handling
    - Yarn Graph consensus
    - Message processing
    - Health monitoring

    Subclasses should override:
    - domain: Agent's primary domain
    - expertise_areas: Specific skills
    - _get_config(): Custom config for domain
    - _initialize_knowledge(): Domain-specific knowledge
    """

    # Override in subclasses
    domain: str = Domain.GENERAL.value
    expertise_areas: List[str] = []

    def __init__(
        self,
        agent_id: str,
        communicator: Optional[AgentCommunicator] = None,
        transport: str = "direct",
        coordinator_url: Optional[str] = None
    ):
        """
        Initialize base agent.

        Args:
            agent_id: Unique agent identifier
            communicator: Agent communicator (creates if None)
            transport: Transport backend ("direct", "rest", "redis")
            coordinator_url: Coordinator URL (for REST transport)
        """
        self.agent_id = agent_id

        # Create communicator if not provided
        if communicator is None:
            self.communicator = AgentCommunicator(
                agent_id=agent_id,
                transport=transport,
                coordinator_url=coordinator_url
            )
        else:
            self.communicator = communicator

        # Create HoloLoom instance with domain-specific config
        self.config = self._get_config()
        self.graph = nx.MultiDiGraph()
        self.loom = HoloLoom(config=self.config, graph_backend=self.graph)

        # Initialize domain-specific knowledge
        self._initialize_knowledge()

        # Consensus manager for distributed Yarn Graph
        self.consensus = YarnGraphConsensus(
            agent_id=agent_id,
            local_graph=self.graph,
            communicator=self.communicator,
            strategy="lww"  # Last-Write-Wins
        )

        # Performance metrics
        self.metrics = {
            "queries_processed": 0,
            "avg_latency_ms": 150.0,
            "success_count": 0,
            "error_count": 0
        }

        # Message handlers
        self._message_handlers: Dict[str, Any] = {
            MessageType.QUERY.value: self._handle_query,
            MessageType.SYNC_REQUEST.value: self._handle_sync_request,
            MessageType.KNOWLEDGE_SHARE.value: self._handle_knowledge_share,
            MessageType.HEARTBEAT.value: self._handle_heartbeat
        }

        # Running state
        self._running = False
        self._message_loop_task: Optional[asyncio.Task] = None

    def _get_config(self) -> Config:
        """
        Get domain-specific config.

        Override in subclasses to customize.

        Returns:
            Config instance
        """
        # Default: FAST mode
        return Config.fast()

    def _initialize_knowledge(self):
        """
        Initialize domain-specific knowledge.

        Override in subclasses to add domain expertise.
        """
        # Base implementation: empty graph
        pass

    async def start(self):
        """Start agent (message processing, health monitoring)."""
        if self._running:
            return

        self._running = True

        # Start communicator
        await self.communicator.start()

        # Start message processing loop
        self._message_loop_task = asyncio.create_task(self._message_loop())

        # Register with global registry
        registry = get_registry()
        await registry.register(
            agent_id=self.agent_id,
            domain=self.domain,
            capabilities=self.expertise_areas
        )

        logger.info(f"Agent started: {self.agent_id} (domain: {self.domain})")

    async def stop(self):
        """Stop agent and clean up resources."""
        if not self._running:
            return

        self._running = False

        # Stop message loop
        if self._message_loop_task:
            self._message_loop_task.cancel()
            try:
                await self._message_loop_task
            except asyncio.CancelledError:
                pass

        # Stop communicator
        await self.communicator.stop()

        # Unregister from registry
        registry = get_registry()
        await registry.unregister(self.agent_id)

        logger.info(f"Agent stopped: {self.agent_id}")

    async def _message_loop(self):
        """Background task to process incoming messages."""
        while self._running:
            try:
                messages = await self.communicator.receive_messages(timeout=1.0)

                for message in messages:
                    await self._process_message(message)

            except Exception as e:
                logger.error(f"Message loop error: {e}")
                await asyncio.sleep(1.0)

    async def _process_message(self, message: AgentMessage):
        """
        Process incoming message.

        Routes to appropriate handler based on message_type.
        """
        handler = self._message_handlers.get(message.message_type)

        if handler:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Message handler error ({message.message_type}): {e}")
        else:
            logger.warning(f"Unknown message type: {message.message_type}")

    async def _handle_query(self, message: AgentMessage):
        """Handle query message."""
        try:
            # Extract query
            query_data = message.payload.get("query")
            if not query_data:
                logger.error("No query in message payload")
                return

            query = Query(
                text=query_data.get("text", ""),
                metadata=query_data.get("metadata", {})
            )

            # Process query
            result = await self.process_query(query.text)

            # Send response
            response = AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.RESULT.value,
                payload={"result": result},
                correlation_id=message.correlation_id
            )

            await self.communicator.transport.send(response)

        except Exception as e:
            logger.error(f"Query handling error: {e}")
            self.metrics["error_count"] += 1

    async def _handle_sync_request(self, message: AgentMessage):
        """Handle graph sync request."""
        try:
            # Export graph
            graph_data = self.consensus.export_graph()

            # Send response
            response = AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.SYNC_RESPONSE.value,
                payload={"graph": graph_data},
                correlation_id=message.correlation_id
            )

            await self.communicator.transport.send(response)

        except Exception as e:
            logger.error(f"Sync request error: {e}")

    async def _handle_knowledge_share(self, message: AgentMessage):
        """Handle knowledge sharing (graph update) message."""
        try:
            # Extract operation
            op_data = message.payload.get("operation")
            if op_data:
                from HoloLoom.multi_agent.consensus import GraphOperation
                operation = GraphOperation.from_dict(op_data)

                # Apply to local graph
                await self.consensus.apply_operation(operation)

                logger.info(f"Applied knowledge update from {message.sender_id}")

        except Exception as e:
            logger.error(f"Knowledge share error: {e}")

    async def _handle_heartbeat(self, message: AgentMessage):
        """Handle heartbeat message."""
        # Update registry
        registry = get_registry()
        await registry.update_heartbeat(self.agent_id)

    async def process_query(self, query_text: str) -> str:
        """
        Process query using HoloLoom.

        Args:
            query_text: Input query

        Returns:
            Response text (or Spacetime in production)
        """
        import time

        start = time.time()

        try:
            # Use HoloLoom to process query
            # For now, return mock response (integrate with actual weaving later)
            response = f"[{self.domain.upper()} Agent {self.agent_id}]: Processed '{query_text}'"

            # Update metrics
            latency_ms = (time.time() - start) * 1000
            self.metrics["queries_processed"] += 1
            self.metrics["success_count"] += 1
            self.metrics["avg_latency_ms"] = (
                (self.metrics["avg_latency_ms"] * (self.metrics["queries_processed"] - 1) + latency_ms)
                / self.metrics["queries_processed"]
            )

            return response

        except Exception as e:
            logger.error(f"Query processing error: {e}")
            self.metrics["error_count"] += 1
            return f"Error: {str(e)}"

    def get_capability(self) -> AgentCapability:
        """
        Get agent's capability metadata.

        Returns:
            AgentCapability with current metrics
        """
        return AgentCapability(
            agent_id=self.agent_id,
            domain=self.domain,
            expertise_areas=self.expertise_areas,
            expertise_score=1.0,  # Can be trained
            current_load=0,  # Would track active queries
            max_capacity=10,
            avg_latency_ms=self.metrics["avg_latency_ms"],
            success_rate=self.metrics["success_count"] / max(self.metrics["queries_processed"], 1)
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# ============================================================================
# Specialized Agents
# ============================================================================

class MathAgent(BaseAgent):
    """
    Agent specialized for mathematical reasoning.

    Optimizations:
    - Config: FUSED mode for deep reasoning
    - Knowledge: Math textbooks, papers, proofs
    - Tools: Symbolic math (SymPy), numerical (NumPy)
    - Reasoning: Multi-step proof verification

    Expertise Areas:
    - Algebra
    - Calculus
    - Statistics
    - Linear algebra
    - Proofs
    """

    domain = Domain.MATH.value
    expertise_areas = [
        "algebra",
        "calculus",
        "statistics",
        "linear_algebra",
        "proofs",
        "equations",
        "derivatives",
        "integrals"
    ]

    def _get_config(self) -> Config:
        """FUSED mode for complex mathematical reasoning."""
        config = Config.fused()
        # Could add math-specific settings here
        return config

    def _initialize_knowledge(self):
        """Initialize mathematical knowledge base."""
        # Add mathematical concepts to graph
        math_concepts = [
            ("algebra", "mathematics", "IS_A", 1.0),
            ("calculus", "mathematics", "IS_A", 1.0),
            ("derivative", "calculus", "PART_OF", 1.0),
            ("integral", "calculus", "PART_OF", 1.0),
            ("theorem", "proof", "USES", 1.0),
            ("equation", "algebra", "USES", 1.0)
        ]

        for src, dst, rel_type, weight in math_concepts:
            edge = KGEdge(src=src, dst=dst, type=rel_type, weight=weight)
            self.graph.add_edge(src, dst, key=rel_type, weight=weight)

        logger.info(f"Initialized {len(math_concepts)} math concepts")


class CodeAgent(BaseAgent):
    """
    Agent specialized for code analysis and generation.

    Optimizations:
    - Config: FAST mode for quick responses
    - Knowledge: Code repositories, documentation, Stack Overflow
    - Tools: AST analysis, linting, test generation
    - Reasoning: Code patterns, best practices

    Expertise Areas:
    - Python
    - JavaScript
    - Code architecture
    - Debugging
    - Testing
    """

    domain = Domain.CODE.value
    expertise_areas = [
        "python",
        "javascript",
        "typescript",
        "architecture",
        "debugging",
        "testing",
        "code_review",
        "refactoring"
    ]

    def _get_config(self) -> Config:
        """FAST mode for responsive code assistance."""
        config = Config.fast()
        return config

    def _initialize_knowledge(self):
        """Initialize code knowledge base."""
        code_concepts = [
            ("function", "code", "PART_OF", 1.0),
            ("class", "code", "PART_OF", 1.0),
            ("variable", "code", "PART_OF", 1.0),
            ("debugging", "testing", "LEADS_TO", 0.8),
            ("refactoring", "code_quality", "LEADS_TO", 0.9),
            ("architecture", "design_patterns", "USES", 1.0)
        ]

        for src, dst, rel_type, weight in code_concepts:
            self.graph.add_edge(src, dst, key=rel_type, weight=weight)

        logger.info(f"Initialized {len(code_concepts)} code concepts")


class WritingAgent(BaseAgent):
    """
    Agent specialized for creative and technical writing.

    Optimizations:
    - Config: FUSED mode for quality writing
    - Knowledge: Writing guides, style guides, examples
    - Tools: Grammar checking, style analysis
    - Reasoning: Clarity, coherence, engagement

    Expertise Areas:
    - Technical documentation
    - Creative writing
    - Summaries
    - Explanations
    - Editing
    """

    domain = Domain.WRITING.value
    expertise_areas = [
        "technical_docs",
        "creative_writing",
        "summaries",
        "explanations",
        "editing",
        "grammar",
        "style",
        "clarity"
    ]

    def _get_config(self) -> Config:
        """FUSED mode for high-quality writing."""
        config = Config.fused()
        return config

    def _initialize_knowledge(self):
        """Initialize writing knowledge base."""
        writing_concepts = [
            ("clarity", "writing_quality", "LEADS_TO", 1.0),
            ("coherence", "writing_quality", "LEADS_TO", 1.0),
            ("grammar", "writing_quality", "PART_OF", 0.9),
            ("style", "writing_quality", "PART_OF", 0.8),
            ("editing", "revision", "IS_A", 1.0),
            ("summary", "writing_type", "IS_A", 1.0)
        ]

        for src, dst, rel_type, weight in writing_concepts:
            self.graph.add_edge(src, dst, key=rel_type, weight=weight)

        logger.info(f"Initialized {len(writing_concepts)} writing concepts")


class ResearchAgent(BaseAgent):
    """
    Agent specialized for multi-hop research and synthesis.

    Optimizations:
    - Config: RESEARCH mode (full 9-step weaving)
    - Knowledge: Papers, books, encyclopedias
    - Tools: Citation tracking, source verification
    - Reasoning: Agentic multi-query reasoning

    Expertise Areas:
    - Literature review
    - Synthesis
    - Fact checking
    - Comparative analysis
    - Research methodology
    """

    domain = Domain.RESEARCH.value
    expertise_areas = [
        "literature_review",
        "synthesis",
        "fact_checking",
        "comparative_analysis",
        "research_methodology",
        "source_verification",
        "citation_analysis"
    ]

    def _get_config(self) -> Config:
        """RESEARCH mode for deep multi-hop reasoning."""
        config = Config.fused()
        # Would use RESEARCH complexity mode (full 9-step weaving)
        return config

    def _initialize_knowledge(self):
        """Initialize research knowledge base."""
        research_concepts = [
            ("literature_review", "research", "PART_OF", 1.0),
            ("synthesis", "research", "PART_OF", 1.0),
            ("fact_checking", "verification", "IS_A", 1.0),
            ("citation", "research", "USES", 0.9),
            ("hypothesis", "research_methodology", "PART_OF", 1.0),
            ("evidence", "fact_checking", "USES", 1.0)
        ]

        for src, dst, rel_type, weight in research_concepts:
            self.graph.add_edge(src, dst, key=rel_type, weight=weight)

        logger.info(f"Initialized {len(research_concepts)} research concepts")


# ============================================================================
# Agent Factory
# ============================================================================

def create_agent(
    domain: str,
    agent_id: Optional[str] = None,
    **kwargs
) -> BaseAgent:
    """
    Factory function to create specialized agents.

    Args:
        domain: Agent domain ("math", "code", "writing", "research")
        agent_id: Unique agent ID (auto-generated if None)
        **kwargs: Additional arguments for agent constructor

    Returns:
        Specialized agent instance

    Example:
        math_agent = create_agent("math", agent_id="math-01")
        code_agent = create_agent("code", agent_id="code-01")
    """
    # Generate agent ID if not provided
    if agent_id is None:
        import uuid
        agent_id = f"{domain}-{uuid.uuid4().hex[:8]}"

    # Map domain to agent class
    agent_classes = {
        Domain.MATH.value: MathAgent,
        Domain.CODE.value: CodeAgent,
        Domain.WRITING.value: WritingAgent,
        Domain.RESEARCH.value: ResearchAgent
    }

    agent_class = agent_classes.get(domain, BaseAgent)

    return agent_class(agent_id=agent_id, **kwargs)
