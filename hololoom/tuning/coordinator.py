"""
Master Tuning Coordinator.

Orchestrates all tuning agents using a meta-bandit.
Selects which agent to run based on impact history.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict
import time

from hololoom.tuning.base import TuningAgent, ThompsonBandit
from hololoom.tuning.persistence import TuningStateManager
from hololoom.tuning.timeout_tuner import TimeoutTuner
from hololoom.tuning.cache_tuner import CacheTuner
from hololoom.tuning.threshold_tuner import ThresholdTuner
from hololoom.tuning.memory_tuner import MemoryTuner
from hololoom.tuning.complexity_tuner import ComplexityTuner
from hololoom.tuning.policy_tuner import PolicyTuner
from hololoom.tuning.physics_tuner import PhysicsTuner

logger = logging.getLogger(__name__)


class MasterTuningCoordinator:
    """
    Coordinates all tuning agents.

    Uses Thompson Sampling meta-bandit to select which agent to run.
    Agents with high recent impact get selected more frequently.
    """

    def __init__(self, state_dir: str = "./tuning_state"):
        self.agents: Dict[str, TuningAgent] = {}
        self.meta_bandit: Optional[ThompsonBandit] = None
        self.persistence = TuningStateManager(state_dir)

        # Global metrics
        self.total_queries = 0
        self.total_tuning_cycles = 0
        self.agent_impact_history = defaultdict(list)

        # Performance baseline
        self.baseline_metrics: Dict[str, float] = {}

        # Circuit breaker
        self.circuit_breaker_open = False
        self.consecutive_failures = 0
        self.failure_threshold = 3

        # Initialize agents
        self._initialize_agents()

        # Load saved state
        self._load_state()

    def _initialize_agents(self):
        """Initialize tuning agents."""
        # Agent 1: TimeoutTuner
        self.agents['timeout'] = TimeoutTuner()

        # Agent 2: CacheTuner
        self.agents['cache'] = CacheTuner()

        # Agent 3: ThresholdTuner
        self.agents['threshold'] = ThresholdTuner()

        # Agent 4: MemoryTuner
        self.agents['memory'] = MemoryTuner()

        # Agent 5: ComplexityTuner
        self.agents['complexity'] = ComplexityTuner()

        # Agent 6: PolicyTuner
        self.agents['policy'] = PolicyTuner()

        # Agent 7: PhysicsTuner (THE FINAL AGENT!)
        self.agents['physics'] = PhysicsTuner()

        # Create meta-bandit (one arm per agent)
        self.meta_bandit = ThompsonBandit(n_arms=len(self.agents))

        logger.info(f"Initialized {len(self.agents)} tuning agents")

    async def run_tuning_cycle(self) -> Dict[str, Any]:
        """
        Run one tuning iteration.

        Selects agent via Thompson Sampling, runs tuning, updates meta-bandit.

        Returns:
            Tuning result with agent name, impact, actions
        """
        if self.circuit_breaker_open:
            logger.warning("Circuit breaker OPEN - skipping tuning cycle")
            return {'status': 'circuit_breaker_open'}

        self.total_tuning_cycles += 1

        # Measure baseline performance
        baseline = await self.measure_system_performance()

        # Sample which agent to activate
        agent_idx = self.meta_bandit.sample()
        agent_names = list(self.agents.keys())
        agent_name = agent_names[agent_idx]
        agent = self.agents[agent_name]

        logger.info(f"Tuning cycle {self.total_tuning_cycles}: Activating {agent_name}")

        try:
            # Run agent tuning
            result = await agent.run_tuning_cycle()

            # Measure improvement
            new_metrics = await self.measure_system_performance()

            # Calculate impact
            impact = self._calculate_impact(baseline, new_metrics)

            # Update meta-bandit
            success = impact > 0.0
            self.meta_bandit.update(agent_idx, success=success, confidence=abs(impact))

            # Track impact history
            self.agent_impact_history[agent_name].append({
                'cycle': self.total_tuning_cycles,
                'impact': impact,
                'timestamp': time.time(),
            })

            # Reset failure counter on success
            if success:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.failure_threshold:
                    self.circuit_breaker_open = True
                    logger.error("Circuit breaker OPEN - too many failures")

            # Save state
            self._save_state()

            result['impact'] = impact
            result['baseline'] = baseline
            result['new_metrics'] = new_metrics

            logger.info(f"Tuning impact: {impact:+.3f}")

            return result

        except Exception as e:
            logger.error(f"Tuning cycle failed: {e}")
            self.consecutive_failures += 1
            return {'status': 'error', 'error': str(e)}

    async def measure_system_performance(self) -> Dict[str, float]:
        """
        Measure overall system performance.

        Returns:
            Dict of performance metrics
        """
        # Placeholder - integrate with actual orchestrator metrics
        # In production, this would query Prometheus or MetricsCollector

        metrics = {}

        # Aggregate metrics from all agents
        for agent_name, agent in self.agents.items():
            try:
                agent_metrics = await agent.measure_performance()
                for key, value in agent_metrics.items():
                    metrics[f'{agent_name}_{key}'] = value
            except Exception as e:
                logger.warning(f"Failed to measure {agent_name}: {e}")

        return metrics

    def _calculate_impact(self, baseline: Dict[str, float], new: Dict[str, float]) -> float:
        """
        Calculate tuning impact score.

        Weighted combination of improvements across metrics.
        """
        # Impact calculation weights
        WEIGHTS = {
            'latency': 0.4,        # Latency improvements
            'confidence': 0.3,     # Confidence improvements
            'cache_hit_rate': 0.2, # Cache performance
            'timeout_fps': 0.1,    # Timeout false positives
        }

        impacts = []

        # Latency metrics (lower is better)
        latency_keys = [k for k in baseline.keys() if 'latency' in k or 'duration' in k or 'p95' in k]
        for key in latency_keys:
            if key in new:
                improvement = (baseline[key] - new[key]) / baseline[key] if baseline[key] > 0 else 0
                impacts.append(('latency', improvement))

        # Confidence metrics (higher is better)
        conf_keys = [k for k in baseline.keys() if 'confidence' in k]
        for key in conf_keys:
            if key in new:
                improvement = (new[key] - baseline[key]) / baseline[key] if baseline[key] > 0 else 0
                impacts.append(('confidence', improvement))

        # Cache hit rate (higher is better)
        cache_keys = [k for k in baseline.keys() if 'hit_rate' in k or 'cache' in k]
        for key in cache_keys:
            if key in new:
                improvement = (new[key] - baseline[key]) / baseline[key] if baseline[key] > 0 else 0
                impacts.append(('cache_hit_rate', improvement))

        # Timeout false positives (lower is better)
        timeout_keys = [k for k in baseline.keys() if 'timeout' in k and 'false' in k]
        for key in timeout_keys:
            if key in new:
                improvement = (baseline[key] - new[key]) / baseline[key] if baseline[key] > 0 else 0
                impacts.append(('timeout_fps', improvement))

        # Weighted average
        if not impacts:
            return 0.0

        weighted_impact = 0.0
        total_weight = 0.0

        for metric_type, improvement in impacts:
            weight = WEIGHTS.get(metric_type, 0.1)
            weighted_impact += improvement * weight
            total_weight += weight

        return weighted_impact / total_weight if total_weight > 0 else 0.0

    def get_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all agents."""
        stats = {}

        for agent_name, agent in self.agents.items():
            stats[agent_name] = {
                'total_cycles': agent.total_tuning_cycles,
                'successful_tunings': agent.successful_tunings,
                'success_rate': agent.get_success_rate(),
                'impact_history': self.agent_impact_history[agent_name][-10:],  # Last 10
            }

        return stats

    def get_meta_bandit_stats(self) -> Dict[int, Dict[str, float]]:
        """Get meta-bandit statistics (which agents selected)."""
        return self.meta_bandit.get_stats()

    def _save_state(self):
        """Save tuning state to disk."""
        state = {
            'total_queries': self.total_queries,
            'total_cycles': self.total_tuning_cycles,
            'agent_states': {
                name: agent.get_state()
                for name, agent in self.agents.items()
            },
            'meta_bandit_state': self.meta_bandit.get_state(),
            'global_metrics': self.baseline_metrics,
            'circuit_breaker_open': self.circuit_breaker_open,
            'consecutive_failures': self.consecutive_failures,
        }

        self.persistence.save_state(state)

    def _load_state(self):
        """Load tuning state from disk."""
        state = self.persistence.load_state()

        if state is None:
            logger.info("No saved state, starting fresh")
            return

        # Restore global counters
        self.total_queries = state.get('total_queries', 0)
        self.total_tuning_cycles = state.get('total_cycles', 0)

        # Restore agent states
        for agent_name, agent_state in state.get('agent_states', {}).items():
            if agent_name in self.agents:
                self.agents[agent_name].load_state(agent_state)

        # Restore meta-bandit
        meta_bandit_state = state.get('meta_bandit_state')
        if meta_bandit_state:
            self.meta_bandit = ThompsonBandit.from_state(meta_bandit_state)

        # Restore circuit breaker
        self.circuit_breaker_open = state.get('circuit_breaker_open', False)
        self.consecutive_failures = state.get('consecutive_failures', 0)

        logger.info(f"Loaded state: {self.total_tuning_cycles} cycles completed")

    def reset_circuit_breaker(self):
        """Manually reset circuit breaker (use after fixing issues)."""
        self.circuit_breaker_open = False
        self.consecutive_failures = 0
        logger.info("Circuit breaker RESET")

    async def run_background_tuning(self, interval_seconds: float = 300.0):
        """
        Run tuning cycles in background loop.

        Args:
            interval_seconds: Time between tuning cycles (default 5 minutes)
        """
        logger.info(f"Starting background tuning (interval={interval_seconds}s)")

        while True:
            try:
                await asyncio.sleep(interval_seconds)

                if not self.circuit_breaker_open:
                    result = await self.run_tuning_cycle()
                    logger.info(f"Background tuning: {result.get('agent')} impact={result.get('impact', 0):.3f}")

            except Exception as e:
                logger.error(f"Background tuning error: {e}")

    def record_query(self):
        """Record that a query was processed (for tuning frequency)."""
        self.total_queries += 1

    def get_tuning_summary(self) -> Dict[str, Any]:
        """Get comprehensive tuning summary."""
        return {
            'total_queries': self.total_queries,
            'total_tuning_cycles': self.total_tuning_cycles,
            'circuit_breaker_open': self.circuit_breaker_open,
            'consecutive_failures': self.consecutive_failures,
            'agents': self.get_agent_stats(),
            'meta_bandit': self.get_meta_bandit_stats(),
        }
