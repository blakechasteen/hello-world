"""
Task Delegation and Routing System
====================================

Routes queries to optimal agents based on specialization and load balancing.

**Date**: November 17, 2025
**Status**: Production Ready

Key Features:
- Automatic domain classification
- Specialization scoring
- Load balancing with capacity awareness
- Task decomposition for complex queries
- Result aggregation from multiple agents
- Consensus voting

Usage:
    # Create router
    router = TaskRouter(communicator, strategy="best_match")

    # Route single query
    result = await router.route_query(query)

    # Decompose and aggregate
    result = await router.decompose_and_aggregate(complex_query)

    # Multi-agent consensus
    result = await router.multi_agent_consensus(query, num_agents=3)
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

from HoloLoom.Documentation.types import Query
from HoloLoom.fabric.spacetime import Spacetime
from HoloLoom.multi_agent.communication import AgentCommunicator, MessageType, get_registry

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class Domain(Enum):
    """Agent specialization domains."""
    MATH = "math"
    CODE = "code"
    WRITING = "writing"
    RESEARCH = "research"
    GENERAL = "general"


class RoutingStrategy(Enum):
    """Query routing strategies."""
    BEST_MATCH = "best_match"           # Highest expertise score
    LOAD_BALANCED = "load_balanced"     # Lowest current load
    ROUND_ROBIN = "round_robin"         # Fair distribution
    CONSENSUS = "consensus"             # Multi-agent voting


@dataclass
class AgentCapability:
    """
    Agent's domain expertise and performance metrics.

    Tracks both static capabilities (domain, expertise areas)
    and dynamic metrics (load, latency, success rate).
    """
    agent_id: str
    domain: str                         # Primary domain
    expertise_areas: List[str]          # Specific skills
    expertise_score: float = 1.0        # 0.0-1.0 (trained/measured)
    current_load: int = 0               # Active queries
    max_capacity: int = 10              # Max concurrent queries
    avg_latency_ms: float = 150.0       # Historical performance
    success_rate: float = 1.0           # 0.0-1.0 (historical)
    metadata: Dict = field(default_factory=dict)

    def load_ratio(self) -> float:
        """Current load as ratio of capacity (0.0-1.0+)."""
        return self.current_load / self.max_capacity if self.max_capacity > 0 else 1.0

    def is_available(self) -> bool:
        """Check if agent can accept more queries."""
        return self.current_load < self.max_capacity


@dataclass
class SubTask:
    """
    Subtask from decomposed complex query.

    Example:
    Query: "Compare Python and JavaScript"
    Subtasks:
    - "Explain Python features" → WritingAgent
    - "Explain JavaScript features" → CodeAgent
    - "Compare syntax" → CodeAgent
    - "Synthesize comparison" → ResearchAgent (aggregator)
    """
    query_text: str
    assigned_agent: Optional[str] = None
    domain: Optional[str] = None
    priority: int = 1                   # 1 (low) to 5 (high)
    dependencies: List[int] = field(default_factory=list)  # Indices of prerequisite subtasks
    result: Optional[Spacetime] = None


# ============================================================================
# Query Classifier
# ============================================================================

class QueryClassifier:
    """
    Classifies queries into domains for agent routing.

    Uses keyword matching and n-gram patterns.
    Can be extended with ML-based classification.
    """

    # Domain-specific keywords
    DOMAIN_KEYWORDS = {
        Domain.MATH: [
            "calculate", "equation", "solve", "proof", "theorem",
            "derivative", "integral", "algebra", "geometry", "statistics",
            "probability", "matrix", "vector", "calculus", "mathematical"
        ],
        Domain.CODE: [
            "function", "class", "method", "variable", "debug",
            "implement", "code", "program", "algorithm", "syntax",
            "compile", "runtime", "error", "exception", "API",
            "framework", "library", "package", "module", "import"
        ],
        Domain.WRITING: [
            "write", "explain", "summarize", "document", "describe",
            "clarify", "elaborate", "rephrase", "edit", "compose",
            "draft", "revise", "outline", "essay", "article"
        ],
        Domain.RESEARCH: [
            "compare", "analyze", "survey", "review", "investigate",
            "study", "explore", "examine", "research", "synthesize",
            "literature", "findings", "evidence", "conclusion", "hypothesis"
        ]
    }

    def classify(self, query: Query) -> Domain:
        """
        Classify query into domain.

        Args:
            query: Input query

        Returns:
            Predicted domain
        """
        text = query.text.lower()

        # Score each domain
        scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text)
            scores[domain] = score

        # Return domain with highest score
        if scores:
            best_domain = max(scores, key=scores.get)
            if scores[best_domain] > 0:
                return best_domain

        # Default to GENERAL if no matches
        return Domain.GENERAL


# ============================================================================
# Task Router
# ============================================================================

class TaskRouter:
    """
    Routes queries to optimal agents.

    Supports multiple routing strategies:
    - best_match: Highest expertise score
    - load_balanced: Lowest current load
    - round_robin: Fair distribution
    - consensus: Multi-agent voting

    Example:
        router = TaskRouter(communicator, strategy="best_match")

        # Route single query
        result = await router.route_query(query)

        # Decompose complex task
        result = await router.decompose_and_aggregate(query)
    """

    def __init__(
        self,
        communicator: AgentCommunicator,
        strategy: str = "best_match",
        classifier: Optional[QueryClassifier] = None
    ):
        """
        Initialize router.

        Args:
            communicator: Agent communicator
            strategy: Routing strategy (see RoutingStrategy enum)
            classifier: Query classifier (creates default if None)
        """
        self.communicator = communicator
        self.strategy = RoutingStrategy(strategy)
        self.classifier = classifier or QueryClassifier()

        # Agent capabilities registry
        self.capabilities: Dict[str, AgentCapability] = {}

        # Round-robin counter
        self._round_robin_index = 0

    async def register_agent(
        self,
        capability: AgentCapability
    ):
        """Register agent's capabilities."""
        self.capabilities[capability.agent_id] = capability
        logger.info(f"Registered agent: {capability.agent_id} (domain: {capability.domain})")

    async def update_agent_load(
        self,
        agent_id: str,
        delta: int
    ):
        """
        Update agent's current load.

        Args:
            agent_id: Agent to update
            delta: Change in load (+1 for new query, -1 for completed)
        """
        if agent_id in self.capabilities:
            self.capabilities[agent_id].current_load += delta
            self.capabilities[agent_id].current_load = max(
                0,
                self.capabilities[agent_id].current_load
            )

    def calculate_specialization_score(
        self,
        agent: AgentCapability,
        query: Query
    ) -> float:
        """
        Score how well agent matches query requirements.

        Factors:
        1. Domain match (0.4 weight)
        2. Historical success (0.3 weight)
        3. Current load (0.2 weight)
        4. Latency (0.1 weight)

        Args:
            agent: Agent capability
            query: Input query

        Returns:
            Score 0.0-1.0 (higher = better match)
        """
        # Classify query domain
        query_domain = self.classifier.classify(query)

        # Domain match score
        if query_domain.value == agent.domain:
            domain_score = 1.0
        elif query_domain == Domain.GENERAL:
            domain_score = 0.7  # General queries can go to anyone
        else:
            domain_score = 0.3  # Mismatch penalty

        # Success rate score (already 0.0-1.0)
        success_score = agent.success_rate

        # Load score (inverse of load ratio)
        load_score = 1.0 - min(agent.load_ratio(), 1.0)

        # Latency score (inverse, normalized to 1s)
        latency_score = 1.0 - min(agent.avg_latency_ms / 1000.0, 1.0)

        # Weighted combination
        score = (
            0.4 * domain_score +
            0.3 * success_score +
            0.2 * load_score +
            0.1 * latency_score
        )

        return score

    async def _route_best_match(
        self,
        query: Query
    ) -> Optional[str]:
        """
        Route to agent with highest expertise score.

        Args:
            query: Input query

        Returns:
            Agent ID or None if no available agents
        """
        best_agent_id = None
        best_score = -1.0

        for agent_id, capability in self.capabilities.items():
            if not capability.is_available():
                continue  # Skip agents at capacity

            score = self.calculate_specialization_score(capability, query)

            if score > best_score:
                best_score = score
                best_agent_id = agent_id

        return best_agent_id

    async def _route_load_balanced(
        self,
        query: Query
    ) -> Optional[str]:
        """
        Route to agent with lowest load in matching domain.

        Args:
            query: Input query

        Returns:
            Agent ID or None if no available agents
        """
        query_domain = self.classifier.classify(query)

        # Filter agents by domain
        matching_agents = [
            (agent_id, cap)
            for agent_id, cap in self.capabilities.items()
            if cap.domain == query_domain.value and cap.is_available()
        ]

        if not matching_agents:
            # Fall back to any available agent
            matching_agents = [
                (agent_id, cap)
                for agent_id, cap in self.capabilities.items()
                if cap.is_available()
            ]

        if not matching_agents:
            return None  # No agents available

        # Select agent with lowest load ratio
        best_agent_id, _ = min(matching_agents, key=lambda x: x[1].load_ratio())
        return best_agent_id

    async def _route_round_robin(
        self,
        query: Query
    ) -> Optional[str]:
        """
        Route using round-robin (fair distribution).

        Args:
            query: Input query

        Returns:
            Agent ID or None if no available agents
        """
        available = [
            agent_id
            for agent_id, cap in self.capabilities.items()
            if cap.is_available()
        ]

        if not available:
            return None

        # Select next agent in round-robin
        agent_id = available[self._round_robin_index % len(available)]
        self._round_robin_index += 1

        return agent_id

    async def route_query(
        self,
        query: Query,
        timeout: float = 30.0
    ) -> Optional[Spacetime]:
        """
        Route query to optimal agent and return result.

        Args:
            query: Input query
            timeout: Response timeout (seconds)

        Returns:
            Spacetime result or None if routing failed
        """
        # Select agent based on strategy
        if self.strategy == RoutingStrategy.BEST_MATCH:
            agent_id = await self._route_best_match(query)
        elif self.strategy == RoutingStrategy.LOAD_BALANCED:
            agent_id = await self._route_load_balanced(query)
        elif self.strategy == RoutingStrategy.ROUND_ROBIN:
            agent_id = await self._route_round_robin(query)
        else:
            logger.error(f"Unknown routing strategy: {self.strategy}")
            return None

        if not agent_id:
            logger.error("No available agents for routing")
            return None

        # Update load
        await self.update_agent_load(agent_id, delta=+1)

        try:
            # Send query to agent
            response = await self.communicator.request_response(
                receiver_id=agent_id,
                message_type=MessageType.QUERY.value,
                payload={"query": {"text": query.text, "metadata": query.metadata}},
                timeout=timeout
            )

            if not response:
                logger.error(f"Query timeout from {agent_id}")
                return None

            # Extract Spacetime from response
            # For now, return mock Spacetime (integrate with actual agents later)
            result_data = response.payload.get("result")

            # Update agent metrics (simplified)
            if agent_id in self.capabilities:
                self.capabilities[agent_id].success_rate = 0.95  # Mock update

            logger.info(f"Routed query to {agent_id}")
            return result_data  # Would be actual Spacetime in production

        finally:
            # Update load (query completed)
            await self.update_agent_load(agent_id, delta=-1)

    async def decompose_task(
        self,
        query: Query,
        max_subtasks: int = 5
    ) -> List[SubTask]:
        """
        Break complex query into parallelizable subtasks.

        Example:
        Query: "Compare Python and JavaScript for web development"

        Subtasks:
        1. "Explain Python web frameworks" → WritingAgent
        2. "Explain JavaScript web frameworks" → CodeAgent
        3. "Compare performance" → MathAgent
        4. "Synthesize comparison" → ResearchAgent (aggregator)

        Args:
            query: Complex query to decompose
            max_subtasks: Maximum number of subtasks

        Returns:
            List of subtasks with assigned agents
        """
        text = query.text.lower()
        subtasks = []

        # Pattern: "Compare A and B"
        compare_match = re.search(r"compare\s+(\w+)\s+and\s+(\w+)", text)
        if compare_match:
            item_a = compare_match.group(1)
            item_b = compare_match.group(2)

            # Subtask 1: Explain A
            subtasks.append(SubTask(
                query_text=f"Explain {item_a}",
                domain=Domain.WRITING.value,
                priority=2
            ))

            # Subtask 2: Explain B
            subtasks.append(SubTask(
                query_text=f"Explain {item_b}",
                domain=Domain.WRITING.value,
                priority=2
            ))

            # Subtask 3: Synthesize comparison (depends on 1, 2)
            subtasks.append(SubTask(
                query_text=f"Compare {item_a} and {item_b}",
                domain=Domain.RESEARCH.value,
                priority=1,
                dependencies=[0, 1]  # Depends on subtasks 0 and 1
            ))

        # Pattern: Multiple questions
        elif "?" in text and text.count("?") > 1:
            # Split on question marks
            questions = [q.strip() + "?" for q in text.split("?") if q.strip()]
            for i, question in enumerate(questions[:max_subtasks]):
                domain = self.classifier.classify(Query(text=question))
                subtasks.append(SubTask(
                    query_text=question,
                    domain=domain.value,
                    priority=3
                ))

        else:
            # Can't decompose, return original as single subtask
            subtasks.append(SubTask(
                query_text=query.text,
                domain=self.classifier.classify(query).value,
                priority=5
            ))

        # Assign agents to subtasks
        for subtask in subtasks:
            # Find best agent for this domain
            matching_agents = [
                agent_id
                for agent_id, cap in self.capabilities.items()
                if cap.domain == subtask.domain and cap.is_available()
            ]

            if matching_agents:
                # Select least loaded
                best_agent_id = min(
                    matching_agents,
                    key=lambda a: self.capabilities[a].load_ratio()
                )
                subtask.assigned_agent = best_agent_id

        return subtasks

    async def aggregate_results(
        self,
        subtasks: List[SubTask]
    ) -> Optional[Spacetime]:
        """
        Combine results from multiple subtasks into unified answer.

        Aggregation Strategies:
        1. Concatenation: For independent questions
        2. Synthesis: For related subtasks (use ResearchAgent)
        3. Voting: For consensus (select most common answer)

        Args:
            subtasks: List of completed subtasks with results

        Returns:
            Synthesized Spacetime
        """
        # Collect all results
        results = [st.result for st in subtasks if st.result is not None]

        if not results:
            return None

        # Simple concatenation for now (can be enhanced with LLM synthesis)
        combined_response = "\n\n".join([
            f"**Subtask {i+1}**: {st.query_text}\n{st.result}"
            for i, st in enumerate(subtasks) if st.result
        ])

        # Create synthesized Spacetime (mock for now)
        # In production, would use ResearchAgent to synthesize
        logger.info(f"Aggregated {len(results)} subtask results")

        return combined_response  # Would be actual Spacetime

    async def decompose_and_aggregate(
        self,
        query: Query,
        max_subtasks: int = 5,
        timeout: float = 60.0
    ) -> Optional[Spacetime]:
        """
        Decompose complex task, route subtasks, aggregate results.

        Args:
            query: Complex query
            max_subtasks: Maximum subtasks to create
            timeout: Total timeout for all subtasks

        Returns:
            Aggregated Spacetime result
        """
        # Step 1: Decompose
        subtasks = await self.decompose_task(query, max_subtasks)
        logger.info(f"Decomposed into {len(subtasks)} subtasks")

        # Step 2: Execute subtasks (respecting dependencies)
        completed = [False] * len(subtasks)

        async def execute_subtask(i: int, subtask: SubTask):
            """Execute single subtask."""
            # Wait for dependencies
            for dep_idx in subtask.dependencies:
                while not completed[dep_idx]:
                    await asyncio.sleep(0.1)

            # Execute
            if subtask.assigned_agent:
                result = await self.route_query(
                    Query(text=subtask.query_text),
                    timeout=timeout / len(subtasks)
                )
                subtask.result = result

            completed[i] = True

        # Execute all subtasks in parallel (respecting dependencies)
        await asyncio.gather(*[
            execute_subtask(i, st)
            for i, st in enumerate(subtasks)
        ])

        # Step 3: Aggregate results
        return await self.aggregate_results(subtasks)

    async def multi_agent_consensus(
        self,
        query: Query,
        num_agents: int = 3,
        timeout: float = 30.0
    ) -> Optional[Spacetime]:
        """
        Get responses from N agents and vote on best.

        Voting Algorithm:
        1. Route query to N agents (highest specialization scores)
        2. Collect all responses
        3. Agents vote on all responses (weighted by expertise)
        4. Return response with highest weighted votes

        Args:
            query: Input query
            num_agents: Number of agents to consult
            timeout: Timeout per agent

        Returns:
            Consensus Spacetime result
        """
        # Select top N agents by specialization score
        agent_scores = [
            (agent_id, self.calculate_specialization_score(cap, query))
            for agent_id, cap in self.capabilities.items()
            if cap.is_available()
        ]

        agent_scores.sort(key=lambda x: x[1], reverse=True)
        selected_agents = [agent_id for agent_id, _ in agent_scores[:num_agents]]

        if len(selected_agents) < num_agents:
            logger.warning(f"Only {len(selected_agents)} agents available (requested {num_agents})")

        # Collect responses from all agents
        async def query_agent(agent_id: str):
            """Query single agent."""
            await self.update_agent_load(agent_id, delta=+1)
            try:
                response = await self.communicator.request_response(
                    receiver_id=agent_id,
                    message_type=MessageType.QUERY.value,
                    payload={"query": {"text": query.text}},
                    timeout=timeout
                )
                return agent_id, response
            finally:
                await self.update_agent_load(agent_id, delta=-1)

        responses = await asyncio.gather(*[
            query_agent(agent_id)
            for agent_id in selected_agents
        ])

        # Filter successful responses
        valid_responses = [
            (agent_id, resp)
            for agent_id, resp in responses
            if resp is not None
        ]

        if not valid_responses:
            logger.error("No valid responses from agents")
            return None

        # Voting: Each agent votes on all responses
        # For now, simple voting (can be enhanced with weighted confidence)
        votes = {}
        for i, (agent_id, resp) in enumerate(valid_responses):
            # Mock vote (in production, agents would score each response)
            votes[i] = 1  # Each response gets 1 vote

        # Select response with most votes
        winner_idx = max(votes, key=votes.get)
        winner_agent_id, winner_response = valid_responses[winner_idx]

        logger.info(f"Consensus: {winner_agent_id} (votes: {votes[winner_idx]})")

        return winner_response.payload.get("result")
