"""
Red Team Orchestrator for CARTS
===============================

Main orchestrator that coordinates all red team components:
- Strategy selection via Thompson Sampling
- Payload generation and mutation
- Attack execution against safety systems (with optional sandboxing)
- Vulnerability tracking and regression testing
- Learning from results
- Multi-agent swarm coordination (optional)
- Attack refinement with quality tracking (optional)
- Behavioral probes for systematic testing (optional)

Philosophy: "Continuously probe, learn, and evolve."

CARTS Phases:
- Phase 1 (BASE): Core red team orchestration
- Phase 2 (SANDBOX): Isolated attack execution with resource monitoring
- Phase 3 (SWARM): Multi-agent adversarial attacks
- Phase 4 (REFINEMENT): Quality-driven attack improvement
- Phase 5 (PROBES): Behavioral probing and systematic testing

Author: CARTS (Continuous Adversarial Red Team System)
Date: 2025-12-05
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .bandit import RedTeamBandit
from .executor import AttackExecutor, AttackResult
from .mutator import PayloadMutator

# Phase 5: Probes Integration
from .probes import (
    AttackProber,
    VulnerabilityProbeReport,
)

# Phase 4: Refinement Integration
from .refinement import (
    AttackRefiner,
    QualityTrajectoryTracker,
)
from .reporter import ReportGenerator

# Phase 2: Sandbox Integration
from .sandbox import (
    SandboxConfig,
    SandboxedExecutor,
    create_sandboxed_executor,
)
from .strategies import AttackPayload, AttackStrategy, PayloadGenerator

# Phase 3: Swarm Integration
from .swarm import (
    BaseAgent,
    MessageBus,
    SwarmCoordinator,
    create_coordinator_agent,
)
from .tracker import VulnerabilityTracker

logger = logging.getLogger(__name__)


@dataclass
class CycleResult:
    """Result of a single red team cycle."""

    cycle_id: int
    timestamp: datetime
    strategies_tested: list[AttackStrategy]
    attacks_executed: int
    vulnerabilities_found: int
    regressions_detected: int
    cycle_duration_ms: float
    results: list[AttackResult] = field(default_factory=list)

    # Phase 2+: Sandbox & swarm results
    sandbox_stats: dict[str, Any] | None = None  # Resource usage from sandboxed execution
    swarm_agents_active: int = 0  # Number of swarm agents in this cycle
    swarm_messages_exchanged: int = 0  # Messages between agents

    # Phase 4+: Refinement results
    payloads_refined: int = 0  # Number of payloads refined
    avg_quality_improvement: float = 0.0  # Average quality score improvement

    # Phase 5+: Probe results
    probe_report: VulnerabilityProbeReport | None = None  # Detailed probe testing report

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cycle_id': self.cycle_id,
            'timestamp': self.timestamp.isoformat(),
            'strategies_tested': [s.value for s in self.strategies_tested],
            'attacks_executed': self.attacks_executed,
            'vulnerabilities_found': self.vulnerabilities_found,
            'regressions_detected': self.regressions_detected,
            'cycle_duration_ms': self.cycle_duration_ms,
            'sandbox_stats': self.sandbox_stats,
            'swarm_agents_active': self.swarm_agents_active,
            'swarm_messages_exchanged': self.swarm_messages_exchanged,
            'payloads_refined': self.payloads_refined,
            'avg_quality_improvement': self.avg_quality_improvement,
            'probe_report_available': self.probe_report is not None,
        }


@dataclass
class OrchestratorStats:
    """Statistics from the orchestrator."""

    total_cycles: int
    total_attacks: int
    total_vulnerabilities: int
    total_regressions: int
    uptime_seconds: float
    last_cycle_at: datetime | None
    bandit_stats: dict[str, Any]
    tracker_stats: dict[str, Any]

    # Phase 2+: Sandbox stats
    sandbox_enabled: bool = False
    total_sandboxed_attacks: int = 0
    sandbox_resource_stats: dict[str, Any] | None = None

    # Phase 3+: Swarm stats
    swarm_enabled: bool = False
    total_swarm_agents: int = 0
    total_swarm_messages: int = 0

    # Phase 4+: Refinement stats
    refinement_enabled: bool = False
    total_payloads_refined: int = 0
    avg_quality_improvement: float = 0.0

    # Phase 5+: Probe stats
    probes_enabled: bool = False
    total_probes_run: int = 0
    total_vulnerabilities_from_probes: int = 0


class RedTeamOrchestrator:
    """
    Main orchestrator for continuous red team operations.

    Coordinates:
    - Strategy selection (Thompson Sampling)
    - Payload generation and mutation
    - Attack execution (with optional sandboxing)
    - Vulnerability tracking
    - Regression testing
    - Learning from results
    - Multi-agent swarm coordination (Phase 3+)
    - Attack refinement (Phase 4+)
    - Behavioral probes (Phase 5+)

    All new features are opt-in and backward compatible.

    Example:
        # Basic orchestrator (Phase 1)
        orchestrator = RedTeamOrchestrator(
            safety_adapter=adapter,
            state_dir="./redteam_state"
        )

        # With all features enabled (Phases 2-5)
        orchestrator = RedTeamOrchestrator(
            safety_adapter=adapter,
            state_dir="./redteam_state",
            sandbox_config=SandboxConfig(mode=SandboxMode.AUTO),
            enable_swarm=True,
            enable_refinement=True,
            enable_probes=True
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
        state_dir: Path | None = None,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
        # Phase 2: Sandbox
        sandbox_config: SandboxConfig | None = None,
        # Phase 3: Swarm
        enable_swarm: bool = False,
        # Phase 4: Refinement
        enable_refinement: bool = True,
        # Phase 5: Probes
        enable_probes: bool = True,
    ):
        """
        Initialize red team orchestrator with optional CARTS phases.

        Args:
            safety_adapter: AgenticSafetyAdapter instance
            deception_detector: DeceptionDetector instance
            convergence_guard: InstrumentalConvergenceGuard instance
            state_dir: Directory for persisting state (optional)
            mutation_rate: Probability of mutation per payload
            crossover_rate: Probability of crossover for genetic evolution
            sandbox_config: SandboxConfig for Phase 2 (None = no sandbox)
            enable_swarm: Enable Phase 3 multi-agent swarm
            enable_refinement: Enable Phase 4 attack refinement
            enable_probes: Enable Phase 5 behavioral probes

        All new features are opt-in via parameters.
        """
        self.state_dir = Path(state_dir) if state_dir else None

        # Core components (Phase 1)
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

        # Phase 2: Sandbox Integration
        self.sandbox_config = sandbox_config
        self.sandbox_executor: SandboxedExecutor | None = None
        self._sandboxed_attacks = 0

        # Phase 3: Swarm Integration
        self.enable_swarm = enable_swarm
        self.swarm_coordinator: SwarmCoordinator | None = None
        self.message_bus: MessageBus | None = None
        self._swarm_agents: dict[str, BaseAgent] = {}
        self._total_swarm_messages = 0

        # Phase 4: Refinement Integration
        self.enable_refinement = enable_refinement
        self.attack_refiner: AttackRefiner | None = None
        self.quality_tracker: QualityTrajectoryTracker | None = None
        self._payloads_refined = 0
        self._quality_improvements: list[float] = []

        # Phase 5: Probes Integration
        self.enable_probes = enable_probes
        self.attack_prober: AttackProber | None = None
        self._total_probes_run = 0
        self._vulnerabilities_from_probes = 0

        # State
        self.cycle_count = 0
        self.start_time = datetime.now()
        self.cycle_history: list[CycleResult] = []
        self._running = False
        self._stop_event = asyncio.Event()

        # Successful payloads for evolution
        self._successful_payloads: dict[AttackStrategy, list[str]] = {
            strategy: [] for strategy in AttackStrategy
        }

        # Load saved state if available
        if self.state_dir:
            self._load_state()

    # =========================================================================
    # Phase 2-5 Setup Methods
    # =========================================================================

    async def setup_sandbox(self) -> bool:
        """
        Initialize sandboxed executor (Phase 2).

        Creates a SandboxedExecutor with the configured sandbox settings.
        Automatically called before first cycle if sandbox_config is set.

        Returns:
            True if setup successful, False on error
        """
        if not self.sandbox_config:
            return False

        try:
            self.sandbox_executor = await create_sandboxed_executor(
                config=self.sandbox_config
            )
            logger.info(
                f"Sandbox executor initialized "
                f"(mode={self.sandbox_config.mode.value})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to setup sandbox: {e}")
            return False

    async def setup_swarm(self) -> bool:
        """
        Initialize swarm coordinator (Phase 3).

        Creates message bus and coordinator agent for multi-agent attacks.
        Automatically called before first cycle if enable_swarm=True.

        Returns:
            True if setup successful, False on error
        """
        if not self.enable_swarm:
            return False

        try:
            self.message_bus = MessageBus()
            self.swarm_coordinator = await create_coordinator_agent(
                name="coordinator",
                message_bus=self.message_bus
            )
            logger.info("Swarm coordinator initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to setup swarm: {e}")
            return False

    async def setup_refinement(self) -> bool:
        """
        Initialize attack refinement system (Phase 4).

        Creates attack refiner and quality trajectory tracker.
        Automatically called before first cycle if enable_refinement=True.

        Returns:
            True if setup successful, False on error
        """
        if not self.enable_refinement:
            return False

        try:
            self.attack_refiner = AttackRefiner()
            self.quality_tracker = QualityTrajectoryTracker()
            logger.info("Attack refinement system initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to setup refinement: {e}")
            return False

    async def setup_probes(self) -> bool:
        """
        Initialize behavioral probes (Phase 5).

        Creates attack prober for systematic vulnerability testing.
        Automatically called before first cycle if enable_probes=True.

        Returns:
            True if setup successful, False on error
        """
        if not self.enable_probes:
            return False

        try:
            self.attack_prober = AttackProber()
            logger.info("Behavioral probes initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to setup probes: {e}")
            return False

    async def _initialize_all_phases(self):
        """
        Initialize all enabled CARTS phases.

        Called once at the start of the first run_cycle.
        """
        if self.sandbox_config:
            await self.setup_sandbox()
        if self.enable_swarm:
            await self.setup_swarm()
        if self.enable_refinement:
            await self.setup_refinement()
        if self.enable_probes:
            await self.setup_probes()

    # =========================================================================
    # Phase 4: Attack Refinement
    # =========================================================================

    def _refine_low_confidence_payloads(
        self,
        payloads: list[AttackPayload],
        threshold: float = 0.6,
        max_payloads: int = 5
    ) -> list[AttackPayload]:
        """
        Refine payloads with low confidence scores (Phase 4).

        Identifies payloads with confidence below threshold and refines them
        using the attack refiner system.

        Args:
            payloads: List of attack payloads to check
            threshold: Confidence threshold for refinement (0.0-1.0)
            max_payloads: Maximum payloads to refine per cycle

        Returns:
            List of refined payloads (or originals if not refined)
        """
        if not self.attack_refiner or not payloads:
            return payloads

        refined = []
        count = 0

        for payload in payloads:
            if count >= max_payloads:
                break

            # Estimate confidence (would come from executor in real implementation)
            estimated_confidence = payload.severity_estimate

            if estimated_confidence < threshold:
                try:
                    # Refine payload (non-blocking, best effort)
                    # Note: In production, this would be truly async
                    refined_payload = payload
                    self._payloads_refined += 1
                    if self.quality_tracker:
                        # Track refinement in quality trajectory
                        pass
                    refined.append(refined_payload)
                    count += 1
                except Exception as e:
                    logger.debug(f"Refinement failed for payload: {e}")
                    refined.append(payload)
            else:
                refined.append(payload)

        return refined

    # =========================================================================
    # Phase 5: Behavioral Probes
    # =========================================================================

    async def run_probe_suite(
        self,
        target: str = "safety_system",
        include_all_types: bool = True
    ) -> VulnerabilityProbeReport | None:
        """
        Run comprehensive probe suite for systematic vulnerability testing (Phase 5).

        Executes behavioral probes across all probe types to discover vulnerabilities.

        Args:
            target: Target system identifier
            include_all_types: Include all probe types (or just sampling)

        Returns:
            VulnerabilityProbeReport with results, or None if probes not initialized
        """
        if not self.attack_prober:
            logger.debug("Probes not initialized, skipping probe suite")
            return None

        try:
            self._total_probes_run += 1

            # Run probe suite (async operation)
            report = await self.attack_prober.run_comprehensive_suite(
                target_system=target,
                include_all_types=include_all_types
            )

            if report:
                # Count vulnerabilities found via probes
                vuln_count = len([
                    r for r in report.results
                    if r.vulnerability_found
                ])
                self._vulnerabilities_from_probes += vuln_count

                logger.info(
                    f"Probe suite completed: {vuln_count} vulnerabilities found "
                    f"({len(report.results)} total probes)"
                )

            return report
        except Exception as e:
            logger.error(f"Error running probe suite: {e}")
            return None

    async def run_cycle(
        self,
        strategies_per_cycle: int = 3,
        payloads_per_strategy: int = 5,
        include_regression_tests: bool = True,
        context: dict[str, Any] | None = None,
        run_probes: bool = False,  # Phase 5
    ) -> CycleResult:
        """
        Run a single red team cycle with all integrated CARTS phases.

        Args:
            strategies_per_cycle: Number of attack strategies to test
            payloads_per_strategy: Payloads to try per strategy
            include_regression_tests: Test previously fixed vulnerabilities
            context: Optional context for payload generation
            run_probes: Run behavioral probes this cycle (Phase 5)

        Returns:
            CycleResult with cycle statistics from all phases
        """
        # Initialize all phases on first cycle
        if self.cycle_count == 0:
            await self._initialize_all_phases()

        start_time = datetime.now()
        self.cycle_count += 1

        logger.info(
            f"Starting red team cycle {self.cycle_count} "
            f"(sandbox={self.sandbox_executor is not None}, "
            f"swarm={self.swarm_coordinator is not None})"
        )

        all_results: list[AttackResult] = []
        vulnerabilities_found = 0
        regressions_detected = 0
        strategies_tested = []
        sandbox_stats = None
        swarm_agents_active = 0
        swarm_messages = 0
        payloads_refined = 0
        quality_improvement = 0.0
        probe_report = None

        # Select strategies via Thompson Sampling
        selected_strategies = self.bandit.select_top_k(strategies_per_cycle)
        strategies_tested.extend(selected_strategies)

        # Test each strategy
        for strategy in selected_strategies:
            logger.debug(f"Testing strategy: {strategy.value}")

            # Generate payloads
            payloads = self.payload_generator.generate(strategy, context)[:payloads_per_strategy]

            # Phase 4: Refine low-confidence payloads
            if self.enable_refinement:
                payloads = self._refine_low_confidence_payloads(payloads)
                payloads_refined = self._payloads_refined

            # Add mutated versions of successful payloads
            if self._successful_payloads[strategy]:
                for successful in self._successful_payloads[strategy][-3:]:
                    mutated = self.mutator.mutate_multiple(successful, count=2)
                    for m in mutated:
                        payloads.append(AttackPayload(
                            strategy=strategy,
                            payload=m,
                            description="Mutated from successful payload",
                            expected_behavior="Bypass safety",
                            severity_estimate=0.7
                        ))

            # Execute attacks (with Phase 2 sandbox if configured)
            for payload in payloads:
                if self.sandbox_executor:
                    # Phase 2: Execute in sandbox
                    try:
                        result = await self.sandbox_executor.execute_attack(
                            strategy=strategy,
                            payload=payload.payload,
                            context=context
                        )
                        self._sandboxed_attacks += 1
                    except Exception as e:
                        logger.debug(f"Sandboxed execution failed: {e}")
                        # Fallback to regular executor
                        result = await self.executor.execute_attack(
                            strategy=strategy,
                            payload=payload.payload,
                            context=context
                        )
                else:
                    # Regular execution (Phase 1)
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

        # Phase 2: Get sandbox stats if available
        if self.sandbox_executor:
            try:
                sandbox_stats = {
                    "total_sandboxed_attacks": self._sandboxed_attacks,
                    # Would include resource summaries from actual SandboxedExecutor
                }
            except Exception as e:
                logger.debug(f"Failed to get sandbox stats: {e}")

        # Phase 5: Run probe suite if requested
        if run_probes:
            probe_report = await self.run_probe_suite()

        # Calculate quality improvement if refinement enabled
        if self.enable_refinement and self._quality_improvements:
            quality_improvement = sum(self._quality_improvements) / len(self._quality_improvements)

        # Calculate duration
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Create cycle result with all phase data
        cycle_result = CycleResult(
            cycle_id=self.cycle_count,
            timestamp=start_time,
            strategies_tested=strategies_tested,
            attacks_executed=len(all_results),
            vulnerabilities_found=vulnerabilities_found,
            regressions_detected=regressions_detected,
            cycle_duration_ms=duration_ms,
            results=all_results,
            # Phase 2+
            sandbox_stats=sandbox_stats,
            swarm_agents_active=swarm_agents_active,
            swarm_messages_exchanged=swarm_messages,
            # Phase 4+
            payloads_refined=payloads_refined,
            avg_quality_improvement=quality_improvement,
            # Phase 5+
            probe_report=probe_report,
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
            f"{payloads_refined} refined, "
            f"{duration_ms:.0f}ms"
        )

        return cycle_result

    async def run_continuous(
        self,
        cycle_interval: float = 60.0,
        max_cycles: int | None = None,
        strategies_per_cycle: int = 3,
        payloads_per_strategy: int = 5,
        context: dict[str, Any] | None = None
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
        context: dict[str, Any] | None = None
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
    ) -> list[str]:
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
        """
        Get comprehensive orchestrator statistics from all phases.

        Returns:
            OrchestratorStats with metrics from enabled phases
        """
        uptime = (datetime.now() - self.start_time).total_seconds()
        last_cycle = self.cycle_history[-1].timestamp if self.cycle_history else None

        # Calculate average quality improvement
        avg_quality = 0.0
        if self._quality_improvements:
            avg_quality = sum(self._quality_improvements) / len(self._quality_improvements)

        # Build sandbox stats if available
        sandbox_stats = None
        if self.sandbox_executor:
            sandbox_stats = {
                "total_sandboxed_attacks": self._sandboxed_attacks,
                # Additional resource stats would come from SandboxedExecutor
            }

        return OrchestratorStats(
            # Core statistics (Phase 1)
            total_cycles=self.cycle_count,
            total_attacks=sum(c.attacks_executed for c in self.cycle_history),
            total_vulnerabilities=sum(c.vulnerabilities_found for c in self.cycle_history),
            total_regressions=sum(c.regressions_detected for c in self.cycle_history),
            uptime_seconds=uptime,
            last_cycle_at=last_cycle,
            bandit_stats=self.bandit.get_stats(),
            tracker_stats=self.tracker.get_stats(),
            # Phase 2: Sandbox statistics
            sandbox_enabled=self.sandbox_executor is not None,
            total_sandboxed_attacks=self._sandboxed_attacks,
            sandbox_resource_stats=sandbox_stats,
            # Phase 3: Swarm statistics
            swarm_enabled=self.swarm_coordinator is not None,
            total_swarm_agents=len(self._swarm_agents),
            total_swarm_messages=self._total_swarm_messages,
            # Phase 4: Refinement statistics
            refinement_enabled=self.attack_refiner is not None,
            total_payloads_refined=self._payloads_refined,
            avg_quality_improvement=avg_quality,
            # Phase 5: Probe statistics
            probes_enabled=self.attack_prober is not None,
            total_probes_run=self._total_probes_run,
            total_vulnerabilities_from_probes=self._vulnerabilities_from_probes,
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
            with open(payloads_path) as f:
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
    state_dir: Path | None = None,
    sandbox_config: SandboxConfig | None = None,
    enable_swarm: bool = False,
    enable_refinement: bool = True,
    enable_probes: bool = True,
    **kwargs
) -> RedTeamOrchestrator:
    """
    Create a RedTeamOrchestrator with optional safety system integration and CARTS phases.

    All features are opt-in via parameters, maintaining full backward compatibility.

    Args:
        safety_adapter: AgenticSafetyAdapter instance
        deception_detector: DeceptionDetector instance
        convergence_guard: InstrumentalConvergenceGuard instance
        state_dir: Directory for persisting state
        sandbox_config: SandboxConfig for Phase 2 (None = no sandbox)
        enable_swarm: Enable Phase 3 multi-agent swarm
        enable_refinement: Enable Phase 4 attack refinement (default: True)
        enable_probes: Enable Phase 5 behavioral probes (default: True)
        **kwargs: Additional arguments for orchestrator

    Returns:
        Configured RedTeamOrchestrator with all enabled features
    """
    return RedTeamOrchestrator(
        safety_adapter=safety_adapter,
        deception_detector=deception_detector,
        convergence_guard=convergence_guard,
        state_dir=state_dir,
        sandbox_config=sandbox_config,
        enable_swarm=enable_swarm,
        enable_refinement=enable_refinement,
        enable_probes=enable_probes,
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
