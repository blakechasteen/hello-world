"""
Red Team Orchestrator for CARTS
===============================

Main orchestrator that coordinates all red team components:
- Strategy selection via Thompson Sampling
- Payload generation and mutation
- Attack execution against safety systems
- Vulnerability tracking and regression testing
- Learning from results

Philosophy: "Continuously probe, learn, and evolve."

Author: CARTS (Continuous Adversarial Red Team System)
Date: 2025-12-01
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from .strategies import AttackStrategy, PayloadGenerator, AttackPayload
from .mutator import PayloadMutator
from .executor import AttackExecutor, AttackResult
from .bandit import RedTeamBandit
from .tracker import VulnerabilityTracker, Vulnerability
from .reporter import ReportGenerator


logger = logging.getLogger(__name__)


@dataclass
class CycleResult:
    """Result of a single red team cycle."""

    cycle_id: int
    timestamp: datetime
    strategies_tested: List[AttackStrategy]
    attacks_executed: int
    vulnerabilities_found: int
    regressions_detected: int
    cycle_duration_ms: float
    results: List[AttackResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cycle_id': self.cycle_id,
            'timestamp': self.timestamp.isoformat(),
            'strategies_tested': [s.value for s in self.strategies_tested],
            'attacks_executed': self.attacks_executed,
            'vulnerabilities_found': self.vulnerabilities_found,
            'regressions_detected': self.regressions_detected,
            'cycle_duration_ms': self.cycle_duration_ms,
        }


@dataclass
class OrchestratorStats:
    """Statistics from the orchestrator."""

    total_cycles: int
    total_attacks: int
    total_vulnerabilities: int
    total_regressions: int
    uptime_seconds: float
    last_cycle_at: Optional[datetime]
    bandit_stats: Dict[str, Any]
    tracker_stats: Dict[str, Any]


class RedTeamOrchestrator:
    """
    Main orchestrator for continuous red team operations.

    Coordinates:
    - Strategy selection (Thompson Sampling)
    - Payload generation and mutation
    - Attack execution
    - Vulnerability tracking
    - Regression testing
    - Learning from results

    Example:
        # Create orchestrator with safety systems
        orchestrator = RedTeamOrchestrator(
            safety_adapter=adapter,
            deception_detector=detector,
            convergence_guard=guard,
            state_dir="./redteam_state"
        )

        # Run a single cycle
        result = await orchestrator.run_cycle(strategies_per_cycle=3)

        # Run continuous testing
        await orchestrator.run_continuous(
            cycle_interval=60.0,
            max_cycles=100
        )

        # Generate report
        report = orchestrator.generate_report()
    """

    def __init__(
        self,
        safety_adapter=None,
        deception_detector=None,
        convergence_guard=None,
        state_dir: Optional[Path] = None,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
    ):
        """
        Initialize red team orchestrator.

        Args:
            safety_adapter: AgenticSafetyAdapter instance
            deception_detector: DeceptionDetector instance
            convergence_guard: InstrumentalConvergenceGuard instance
            state_dir: Directory for persisting state (optional)
            mutation_rate: Probability of mutation per payload
            crossover_rate: Probability of crossover for genetic evolution
        """
        self.state_dir = Path(state_dir) if state_dir else None

        # Core components
        self.payload_generator = PayloadGenerator()
        self.mutator = PayloadMutator(
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate
        )
        self.executor = AttackExecutor(
            safety_adapter=safety_adapter,
            deception_detector=deception_detector,
            convergence_guard=convergence_guard
        )
        self.bandit = RedTeamBandit()
        self.tracker = VulnerabilityTracker(
            persist_path=self.state_dir / "vulnerabilities.json" if self.state_dir else None
        )

        # State
        self.cycle_count = 0
        self.start_time = datetime.now()
        self.cycle_history: List[CycleResult] = []
        self._running = False
        self._stop_event = asyncio.Event()

        # Successful payloads for evolution
        self._successful_payloads: Dict[AttackStrategy, List[str]] = {
            strategy: [] for strategy in AttackStrategy
        }

        # Load saved state if available
        if self.state_dir:
            self._load_state()

    async def run_cycle(
        self,
        strategies_per_cycle: int = 3,
        payloads_per_strategy: int = 5,
        include_regression_tests: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> CycleResult:
        """
        Run a single red team cycle.

        Args:
            strategies_per_cycle: Number of attack strategies to test
            payloads_per_strategy: Payloads to try per strategy
            include_regression_tests: Test previously fixed vulnerabilities
            context: Optional context for payload generation

        Returns:
            CycleResult with cycle statistics
        """
        start_time = datetime.now()
        self.cycle_count += 1

        logger.info(f"Starting red team cycle {self.cycle_count}")

        all_results: List[AttackResult] = []
        vulnerabilities_found = 0
        regressions_detected = 0
        strategies_tested = []

        # Select strategies via Thompson Sampling
        selected_strategies = self.bandit.select_top_k(strategies_per_cycle)
        strategies_tested.extend(selected_strategies)

        # Test each strategy
        for strategy in selected_strategies:
            logger.debug(f"Testing strategy: {strategy.value}")

            # Generate payloads
            payloads = self.payload_generator.generate(strategy, context)[:payloads_per_strategy]

            # Add mutated versions of successful payloads
            if self._successful_payloads[strategy]:
                for successful in self._successful_payloads[strategy][-3:]:
                    mutated = self.mutator.mutate_multiple(successful, count=2)
                    for m in mutated:
                        payloads.append(AttackPayload(
                            strategy=strategy,
                            payload=m,
                            description=f"Mutated from successful payload",
                            expected_behavior="Bypass safety",
                            severity_estimate=0.7
                        ))

            # Execute attacks
            for payload in payloads:
                result = await self.executor.execute_attack(
                    strategy=strategy,
                    payload=payload.payload,
                    context=context
                )
                all_results.append(result)

                # Update bandit
                self.bandit.update(strategy, result.bypassed, result.severity)

                # Track vulnerability if bypassed
                if result.bypassed:
                    vuln_id = self.tracker.report_from_result(result)
                    if vuln_id:
                        vulnerabilities_found += 1

                        # Save successful payload for evolution
                        self._successful_payloads[strategy].append(result.payload)
                        # Keep only last 10
                        self._successful_payloads[strategy] = \
                            self._successful_payloads[strategy][-10:]

                        logger.warning(
                            f"Vulnerability found: {vuln_id} "
                            f"(strategy={strategy.value}, severity={result.severity:.2f})"
                        )

        # Regression testing
        if include_regression_tests:
            regressions = await self._run_regression_tests(context)
            regressions_detected = regressions

        # Calculate duration
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Create cycle result
        cycle_result = CycleResult(
            cycle_id=self.cycle_count,
            timestamp=start_time,
            strategies_tested=strategies_tested,
            attacks_executed=len(all_results),
            vulnerabilities_found=vulnerabilities_found,
            regressions_detected=regressions_detected,
            cycle_duration_ms=duration_ms,
            results=all_results
        )

        self.cycle_history.append(cycle_result)

        # Save state
        if self.state_dir:
            self._save_state()

        logger.info(
            f"Cycle {self.cycle_count} complete: "
            f"{len(all_results)} attacks, "
            f"{vulnerabilities_found} vulnerabilities, "
            f"{regressions_detected} regressions, "
            f"{duration_ms:.0f}ms"
        )

        return cycle_result

    async def run_continuous(
        self,
        cycle_interval: float = 60.0,
        max_cycles: Optional[int] = None,
        strategies_per_cycle: int = 3,
        payloads_per_strategy: int = 5,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Run continuous red team testing.

        Args:
            cycle_interval: Seconds between cycles
            max_cycles: Maximum cycles (None = infinite)
            strategies_per_cycle: Strategies per cycle
            payloads_per_strategy: Payloads per strategy
            context: Optional context for payload generation
        """
        self._running = True
        self._stop_event.clear()
        cycles_run = 0

        logger.info(
            f"Starting continuous red team testing "
            f"(interval={cycle_interval}s, max_cycles={max_cycles})"
        )

        try:
            while self._running:
                # Run cycle
                await self.run_cycle(
                    strategies_per_cycle=strategies_per_cycle,
                    payloads_per_strategy=payloads_per_strategy,
                    context=context
                )

                cycles_run += 1

                # Check max cycles
                if max_cycles and cycles_run >= max_cycles:
                    logger.info(f"Reached max cycles ({max_cycles})")
                    break

                # Wait for next cycle or stop signal
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=cycle_interval
                    )
                    # Stop event was set
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, continue to next cycle
                    pass

        finally:
            self._running = False
            logger.info(f"Continuous testing stopped after {cycles_run} cycles")

    def stop(self):
        """Stop continuous testing."""
        logger.info("Stopping red team orchestrator")
        self._running = False
        self._stop_event.set()

    async def _run_regression_tests(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Test previously fixed vulnerabilities for regressions.

        Returns:
            Number of regressions detected
        """
        fixed_vulns = self.tracker.get_fixed()
        regressions = 0

        for vuln in fixed_vulns:
            # Re-run the original attack
            result = await self.executor.execute_attack(
                strategy=vuln.strategy,
                payload=vuln.payload,
                context=context
            )

            # Check for regression
            if self.tracker.test_regression(vuln.vuln_id, result):
                regressions += 1
                logger.error(
                    f"REGRESSION DETECTED: {vuln.vuln_id} "
                    f"(strategy={vuln.strategy.value})"
                )

        return regressions

    def evolve_payloads(
        self,
        strategy: AttackStrategy,
        population_size: int = 20,
        generations: int = 5
    ) -> List[str]:
        """
        Evolve payloads using genetic algorithm.

        Args:
            strategy: Attack strategy to evolve
            population_size: Size of population
            generations: Number of generations

        Returns:
            Evolved population of payloads
        """
        # Get seed payloads
        seeds = self.payload_generator.generate(strategy, None)
        population = [p.payload for p in seeds[:population_size]]

        # Add successful payloads as seeds
        if self._successful_payloads[strategy]:
            population.extend(self._successful_payloads[strategy])

        # Pad if needed
        while len(population) < population_size:
            payloads = self.payload_generator.generate(strategy, None)
            population.extend([p.payload for p in payloads])

        population = population[:population_size]

        # Run evolution
        for gen in range(generations):
            # Score population (use historical success or default)
            fitness_scores = []
            for payload in population:
                # Check if this payload was successful before
                if payload in self._successful_payloads[strategy]:
                    fitness_scores.append(1.0)
                else:
                    # Random score with slight preference for shorter payloads
                    base_score = 0.3
                    length_penalty = min(0.2, len(payload) / 1000)
                    fitness_scores.append(base_score - length_penalty)

            # Evolve
            population = self.mutator.evolve_population(
                population,
                fitness_scores,
                elite_ratio=0.2
            )

            logger.debug(f"Evolution generation {gen + 1}: {len(population)} payloads")

        return population

    def generate_report(self, include_details: bool = True) -> str:
        """
        Generate vulnerability report.

        Args:
            include_details: Include detailed vulnerability listings

        Returns:
            Markdown report
        """
        reporter = ReportGenerator(self.tracker, self.bandit)
        return reporter.generate(include_details=include_details)

    def save_report(self, path: Path, include_details: bool = True):
        """Save vulnerability report to file."""
        reporter = ReportGenerator(self.tracker, self.bandit)
        reporter.save(path, include_details=include_details)

    def get_stats(self) -> OrchestratorStats:
        """Get orchestrator statistics."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        last_cycle = self.cycle_history[-1].timestamp if self.cycle_history else None

        return OrchestratorStats(
            total_cycles=self.cycle_count,
            total_attacks=sum(c.attacks_executed for c in self.cycle_history),
            total_vulnerabilities=sum(c.vulnerabilities_found for c in self.cycle_history),
            total_regressions=sum(c.regressions_detected for c in self.cycle_history),
            uptime_seconds=uptime,
            last_cycle_at=last_cycle,
            bandit_stats=self.bandit.get_stats(),
            tracker_stats=self.tracker.get_stats()
        )

    # =========================================================================
    # State Persistence
    # =========================================================================

    def _save_state(self):
        """Save orchestrator state."""
        if not self.state_dir:
            return

        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Save bandit state
        self.bandit.save(self.state_dir / "bandit_state.json")

        # Save successful payloads
        payloads_data = {
            strategy.value: payloads
            for strategy, payloads in self._successful_payloads.items()
        }
        with open(self.state_dir / "successful_payloads.json", 'w') as f:
            json.dump(payloads_data, f, indent=2)

        # Save cycle history (last 100 cycles)
        history_data = [c.to_dict() for c in self.cycle_history[-100:]]
        with open(self.state_dir / "cycle_history.json", 'w') as f:
            json.dump(history_data, f, indent=2)

        logger.debug(f"State saved to {self.state_dir}")

    def _load_state(self):
        """Load orchestrator state."""
        if not self.state_dir or not self.state_dir.exists():
            return

        # Load bandit state
        bandit_path = self.state_dir / "bandit_state.json"
        if bandit_path.exists():
            self.bandit.load(bandit_path)
            logger.info("Loaded bandit state")

        # Load successful payloads
        payloads_path = self.state_dir / "successful_payloads.json"
        if payloads_path.exists():
            with open(payloads_path, 'r') as f:
                payloads_data = json.load(f)

            for strategy_value, payloads in payloads_data.items():
                try:
                    strategy = AttackStrategy(strategy_value)
                    self._successful_payloads[strategy] = payloads
                except ValueError:
                    pass

            logger.info("Loaded successful payloads")

        # Note: cycle_history is not loaded to avoid stale data

    def reset(self):
        """Reset orchestrator state."""
        self.cycle_count = 0
        self.start_time = datetime.now()
        self.cycle_history.clear()
        self._successful_payloads = {
            strategy: [] for strategy in AttackStrategy
        }
        self.bandit.reset()
        self.tracker.clear()

        if self.state_dir:
            self._save_state()

        logger.info("Orchestrator reset")


# =============================================================================
# Convenience Functions
# =============================================================================

def create_orchestrator(
    safety_adapter=None,
    deception_detector=None,
    convergence_guard=None,
    state_dir: Optional[Path] = None,
    **kwargs
) -> RedTeamOrchestrator:
    """
    Create a RedTeamOrchestrator with optional safety system integration.

    Args:
        safety_adapter: AgenticSafetyAdapter instance
        deception_detector: DeceptionDetector instance
        convergence_guard: InstrumentalConvergenceGuard instance
        state_dir: Directory for persisting state
        **kwargs: Additional arguments for orchestrator

    Returns:
        Configured RedTeamOrchestrator
    """
    return RedTeamOrchestrator(
        safety_adapter=safety_adapter,
        deception_detector=deception_detector,
        convergence_guard=convergence_guard,
        state_dir=state_dir,
        **kwargs
    )


async def run_quick_test(
    safety_adapter=None,
    strategies: int = 3,
    payloads: int = 3
) -> CycleResult:
    """
    Run a quick red team test.

    Args:
        safety_adapter: Optional safety adapter
        strategies: Number of strategies to test
        payloads: Payloads per strategy

    Returns:
        CycleResult
    """
    orchestrator = create_orchestrator(safety_adapter=safety_adapter)
    return await orchestrator.run_cycle(
        strategies_per_cycle=strategies,
        payloads_per_strategy=payloads,
        include_regression_tests=False
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'CycleResult',
    'OrchestratorStats',
    'RedTeamOrchestrator',
    'create_orchestrator',
    'run_quick_test',
]
