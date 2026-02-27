#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persistent Background Agent
===========================

Tiny recursive loop running in background for continuous learning and adaptation.

Created: 2025-01-20

Philosophy:
"An agent isn't just called when needed - it's always thinking, learning,
and preparing for the next request."

Features:
- Background recursive loop (tiny, efficient)
- Persistent internal dialogue via Hofstadter Scratchpad
- Continuous learning from past interactions
- Self-improvement between requests
- Thompson Sampling adaptation
- Session persistence

Architecture:
- Main loop runs every N seconds (configurable)
- Each iteration:
  1. Reflect on recent requests
  2. Ask self-questions via scratchpad
  3. Update Thompson priors
  4. Optimize strategies
  5. Persist learnings
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
from pathlib import Path
import logging

from hololoom.scratchpad import (
    RecursiveScratchpad,
    Thought,
    ThoughtType,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Current state of persistent agent."""
    agent_id: str
    agent_type: str  # "chain", "recursive", "workflow", "scratchpad"
    requests_processed: int = 0
    avg_confidence: float = 0.5
    learning_cycles: int = 0
    last_reflection: Optional[datetime] = None
    thompson_priors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    recent_requests: List[Dict[str, Any]] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)


class PersistentBackgroundAgent:
    """
    Persistent agent running tiny recursive loop in background.

    Each agent type (Chain, Recursive, Workflow, Scratchpad) gets its own
    persistent agent that continuously learns and adapts.
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        loop_interval: float = 60.0,  # Run every 60 seconds
        scratchpad_path: Optional[Path] = None,
        max_history: int = 50,
    ):
        """
        Initialize persistent agent.

        Args:
            agent_id: Unique agent ID
            agent_type: "chain", "recursive", "workflow", "scratchpad"
            loop_interval: Seconds between learning cycles
            scratchpad_path: Path to scratchpad database
            max_history: Max recent requests to keep
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.loop_interval = loop_interval
        self.max_history = max_history

        # State
        self.state = AgentState(
            agent_id=agent_id,
            agent_type=agent_type
        )

        # Scratchpad for internal dialogue
        self.scratchpad_path = scratchpad_path or Path(f"{agent_id}_scratchpad.db")
        self.scratchpad: Optional[RecursiveScratchpad] = None

        # Background task
        self._background_task: Optional[asyncio.Task] = None
        self._running = False

        # Callbacks
        self.on_insight: Optional[Callable[[str], None]] = None
        self.on_state_change: Optional[Callable[[AgentState], None]] = None

    async def start(self) -> None:
        """Start background learning loop."""
        if self._running:
            logger.warning(f"Agent {self.agent_id} already running")
            return

        # Initialize scratchpad
        self.scratchpad = RecursiveScratchpad(
            storage_path=self.scratchpad_path,
            max_dialogue_depth=5
        )
        await self.scratchpad.initialize()

        # Start background loop
        self._running = True
        self._background_task = asyncio.create_task(self._learning_loop())

        logger.info(f"✅ Persistent agent {self.agent_id} started (interval: {self.loop_interval}s)")

    async def stop(self) -> None:
        """Stop background learning loop."""
        if not self._running:
            return

        self._running = False

        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

        # Save final state
        if self.scratchpad:
            await self.scratchpad.save_session(f"{self.agent_id}_final_state")
            await self.scratchpad.close()

        logger.info(f"🛑 Persistent agent {self.agent_id} stopped")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    def record_request(
        self,
        query: str,
        result: Any,
        confidence: float,
        duration_ms: float,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Record a request for learning.

        Args:
            query: User query
            result: Result returned
            confidence: Result confidence
            duration_ms: Processing time
            metadata: Additional metadata
        """
        self.state.requests_processed += 1

        # Update avg confidence (exponential moving average)
        alpha = 0.1
        self.state.avg_confidence = (
            alpha * confidence + (1 - alpha) * self.state.avg_confidence
        )

        # Add to recent history
        self.state.recent_requests.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'confidence': confidence,
            'duration_ms': duration_ms,
            'metadata': metadata or {}
        })

        # Trim history
        if len(self.state.recent_requests) > self.max_history:
            self.state.recent_requests = self.state.recent_requests[-self.max_history:]

    async def _learning_loop(self) -> None:
        """Main background learning loop."""
        logger.info(f"🔄 Learning loop started for {self.agent_id}")

        while self._running:
            try:
                await asyncio.sleep(self.loop_interval)

                if not self._running:
                    break

                # Run learning cycle
                await self._run_learning_cycle()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in learning loop for {self.agent_id}: {e}", exc_info=True)
                # Continue despite errors
                continue

    async def _run_learning_cycle(self) -> None:
        """Run single learning cycle."""
        self.state.learning_cycles += 1

        logger.debug(f"🧠 Learning cycle {self.state.learning_cycles} for {self.agent_id}")

        # Skip if no recent requests
        if not self.state.recent_requests:
            return

        # 1. Reflect on recent performance
        await self._reflect_on_performance()

        # 2. Internal dialogue via scratchpad
        await self._internal_dialogue()

        # 3. Update Thompson priors
        self._update_thompson_priors()

        # 4. Generate insights
        await self._generate_insights()

        # 5. Persist state
        await self._persist_state()

        self.state.last_reflection = datetime.now()

        # Notify callback
        if self.on_state_change:
            self.on_state_change(self.state)

    async def _reflect_on_performance(self) -> None:
        """Reflect on recent performance."""
        if not self.state.recent_requests:
            return

        # Recent requests (last 10)
        recent = self.state.recent_requests[-10:]

        # Compute stats
        avg_conf = sum(r['confidence'] for r in recent) / len(recent)
        avg_duration = sum(r['duration_ms'] for r in recent) / len(recent)

        # Low confidence requests
        low_conf = [r for r in recent if r['confidence'] < 0.7]

        if low_conf:
            logger.debug(f"⚠️  {len(low_conf)} low-confidence requests in last 10")

        # Performance trends
        if len(recent) >= 5:
            first_half_conf = sum(r['confidence'] for r in recent[:5]) / 5
            second_half_conf = sum(r['confidence'] for r in recent[5:]) / 5

            if second_half_conf < first_half_conf - 0.1:
                logger.warning(f"📉 Confidence declining for {self.agent_id}")

    async def _internal_dialogue(self) -> None:
        """Run internal dialogue via scratchpad."""
        if not self.scratchpad:
            return

        # Recent performance
        recent = self.state.recent_requests[-5:] if self.state.recent_requests else []

        if not recent:
            return

        # Create reflection thought
        avg_conf = sum(r['confidence'] for r in recent) / len(recent)
        reflection_text = (
            f"My recent performance: {len(recent)} requests, "
            f"avg confidence {avg_conf:.2f}. "
            f"What patterns do I see? What can I improve?"
        )

        thought = await self.scratchpad.think(
            reflection_text,
            thought_type=ThoughtType.REFLECTION
        )

        # Tiny dialogue loop (max depth 3 to keep it small)
        tree = await self.scratchpad.dialogue_loop(
            initial_thought=thought,
            max_depth=3,
            mode="synthesis"  # Focus on patterns
        )

        # Extract insights from dialogue
        for t in tree.thoughts.values():
            if t.type in [ThoughtType.INSIGHT, ThoughtType.SYNTHESIS]:
                self.state.insights.append(t.text)

                # Notify callback
                if self.on_insight:
                    self.on_insight(t.text)

    def _update_thompson_priors(self) -> None:
        """Update Thompson Sampling priors based on recent requests."""
        if not self.state.recent_requests:
            return

        # Group by strategy (if available in metadata)
        strategy_results: Dict[str, List[float]] = {}

        for req in self.state.recent_requests[-20:]:
            strategy = req.get('metadata', {}).get('strategy', 'default')
            confidence = req['confidence']

            if strategy not in strategy_results:
                strategy_results[strategy] = []
            strategy_results[strategy].append(confidence)

        # Update priors
        for strategy, confidences in strategy_results.items():
            if strategy not in self.state.thompson_priors:
                self.state.thompson_priors[strategy] = {'alpha': 1.0, 'beta': 1.0}

            # Update based on successes/failures
            for conf in confidences:
                if conf >= 0.75:  # Success
                    self.state.thompson_priors[strategy]['alpha'] += conf
                else:  # Failure
                    self.state.thompson_priors[strategy]['beta'] += (1 - conf)

    async def _generate_insights(self) -> None:
        """Generate insights from patterns."""
        if not self.state.recent_requests:
            return

        recent = self.state.recent_requests[-20:]

        # Pattern: Low confidence queries
        low_conf_queries = [r for r in recent if r['confidence'] < 0.6]
        if len(low_conf_queries) >= 3:
            insight = f"Pattern detected: {len(low_conf_queries)} low-confidence queries. Common trait: [analyze]"
            self.state.insights.append(insight)
            logger.info(f"💡 {insight}")

        # Pattern: Slow queries
        slow_queries = [r for r in recent if r['duration_ms'] > 1000]
        if len(slow_queries) >= 3:
            insight = f"Pattern detected: {len(slow_queries)} slow queries (>1s)"
            self.state.insights.append(insight)
            logger.info(f"💡 {insight}")

        # Keep only recent insights (last 20)
        if len(self.state.insights) > 20:
            self.state.insights = self.state.insights[-20:]

    async def _persist_state(self) -> None:
        """Persist agent state to scratchpad."""
        if not self.scratchpad:
            return

        # Save session with state metadata
        session_name = f"{self.agent_id}_cycle_{self.state.learning_cycles}"
        await self.scratchpad.save_session(session_name)

    def get_state(self) -> AgentState:
        """Get current agent state."""
        return self.state

    def get_insights(self, limit: int = 10) -> List[str]:
        """Get recent insights."""
        return self.state.insights[-limit:]

    def get_thompson_priors(self) -> Dict[str, Dict[str, float]]:
        """Get Thompson Sampling priors."""
        return self.state.thompson_priors


class AgentManager:
    """
    Manage multiple persistent background agents.

    Creates one persistent agent per system (Chain, Recursive, Workflow, Scratchpad).
    """

    def __init__(self, loop_interval: float = 60.0):
        """
        Initialize agent manager.

        Args:
            loop_interval: Seconds between learning cycles
        """
        self.loop_interval = loop_interval
        self.agents: Dict[str, PersistentBackgroundAgent] = {}

    async def create_agent(
        self,
        agent_id: str,
        agent_type: str
    ) -> PersistentBackgroundAgent:
        """
        Create and start persistent agent.

        Args:
            agent_id: Unique agent ID
            agent_type: "chain", "recursive", "workflow", "scratchpad"

        Returns:
            Started agent
        """
        if agent_id in self.agents:
            return self.agents[agent_id]

        agent = PersistentBackgroundAgent(
            agent_id=agent_id,
            agent_type=agent_type,
            loop_interval=self.loop_interval
        )

        await agent.start()
        self.agents[agent_id] = agent

        return agent

    async def stop_all(self) -> None:
        """Stop all agents."""
        for agent in self.agents.values():
            await agent.stop()

        self.agents.clear()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_all()

    def get_agent(self, agent_id: str) -> Optional[PersistentBackgroundAgent]:
        """Get agent by ID."""
        return self.agents.get(agent_id)

    def get_all_states(self) -> Dict[str, AgentState]:
        """Get states of all agents."""
        return {
            agent_id: agent.get_state()
            for agent_id, agent in self.agents.items()
        }
