"""
Zero-G Launch System Integration for Skills
============================================

**Created**: November 22, 2025
**Purpose**: Connect meta-skills to Zero-G platform with memory-informed decisioning

This module integrates the Skills Meta-System with Zero-G's orbital infrastructure:
- Preflight: skill_tester, skill_security_analyzer
- Orbit: continuous_learning_capture monitors missions
- Mission Control: skill_gap_analyzer identifies gaps
- Optimization: token_budget_adviser for efficiency

Architecture:
```
Zero-G Platform (Orbital Infrastructure)
├── Launch System
│   ├── Preflight ← skill_tester, skill_security_analyzer
│   ├── Countdown
│   ├── Lift-Off
│   └── Orbit ← continuous_learning_capture
│
├── Mission Control ← skill_gap_analyzer, token_budget_adviser
│
└── App Orbit Layer ← Skills dock here
    ├── Loom Core (shared infrastructure)
    │   ├── WarpSpace (semantic retrieval)
    │   ├── YarnGraph (knowledge graph)
    │   └── ResonanceShed (multimodal fusion)
    │
    └── Event Bus (skill-to-skill communication)
```

Key Innovation: **Memory-Informed Decisioning**
All skills receive WarpSpace and YarnGraph in execution context,
enabling intelligent decisions based on past experiences.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import asyncio

from .protocol import (
    DockableSkill,
    SkillManifest,
    SkillExecutionContext,
    SkillExecutionResult,
    SkillLifecycleState,
    SkillEvent,
    EventType,
    EventBus,
    MissionControl,
    SkillDockingManager
)

logger = logging.getLogger(__name__)


# ============================================================================
# Zero-G Integration Configuration
# ============================================================================

@dataclass
class ZeroGConfig:
    """Configuration for Zero-G integration."""

    # Loom Core (shared infrastructure)
    enable_warp_space: bool = True
    enable_yarn_graph: bool = True
    enable_resonance_shed: bool = True
    enable_event_bus: bool = True
    enable_mission_control: bool = True

    # Memory-informed decisioning
    memory_query_limit: int = 10
    memory_confidence_threshold: float = 0.7
    use_hybrid_retrieval: bool = True  # BM25 + semantic

    # Preflight configuration
    enable_skill_tester: bool = True
    enable_security_analyzer: bool = True
    preflight_timeout_seconds: float = 30.0

    # Orbit monitoring
    enable_continuous_learning: bool = True
    learning_capture_interval: float = 60.0  # seconds

    # Mission Control
    enable_gap_analyzer: bool = True
    gap_analysis_interval: float = 3600.0  # 1 hour
    enable_token_adviser: bool = True
    token_budget_limit: int = 100000

    # Lifecycle timeouts
    countdown_timeout: float = 10.0
    liftoff_timeout: float = 120.0
    orbit_timeout: float = 300.0


# ============================================================================
# Memory-Informed Decision Making
# ============================================================================

class MemoryInformedDecisionEngine:
    """
    Enables skills to make intelligent decisions based on past experiences.

    Integrates with HoloLoom's memory systems:
    - WarpSpace: Semantic retrieval of relevant knowledge
    - YarnGraph: Knowledge graph traversal for context
    - ResonanceShed: Multimodal fusion for complex queries
    """

    def __init__(
        self,
        warp_space: Optional[Any] = None,
        yarn_graph: Optional[Any] = None,
        config: Optional[ZeroGConfig] = None
    ):
        self.warp_space = warp_space
        self.yarn_graph = yarn_graph
        self.config = config or ZeroGConfig()

    async def query_memory(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query memory systems for relevant knowledge.

        Args:
            query: The question or context to search for
            context: Additional context (metadata, filters)

        Returns:
            List of relevant memories with confidence scores
        """
        memories = []

        # Try WarpSpace (semantic retrieval)
        if self.warp_space:
            try:
                results = await self._query_warp_space(query, context)
                memories.extend(results)
            except Exception as e:
                logger.warning(f"WarpSpace query failed: {e}")

        # Try YarnGraph (graph traversal)
        if self.yarn_graph:
            try:
                results = await self._query_yarn_graph(query, context)
                memories.extend(results)
            except Exception as e:
                logger.warning(f"YarnGraph query failed: {e}")

        # Filter by confidence threshold
        memories = [
            m for m in memories
            if m.get('confidence', 0.0) >= self.config.memory_confidence_threshold
        ]

        # Sort by confidence and limit
        memories.sort(key=lambda m: m.get('confidence', 0.0), reverse=True)
        return memories[:self.config.memory_query_limit]

    async def _query_warp_space(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Query WarpSpace for semantic similarity."""
        # Placeholder - integrate with actual WarpSpace implementation
        # In production: await self.warp_space.recall(query, k=10)
        return []

    async def _query_yarn_graph(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Query YarnGraph for knowledge graph traversal."""
        # Placeholder - integrate with actual YarnGraph implementation
        # In production: await self.yarn_graph.traverse(query, max_depth=3)
        return []

    async def inform_decision(
        self,
        decision_context: Dict[str, Any],
        skill_name: str
    ) -> Dict[str, Any]:
        """
        Inform a skill's decision based on memory.

        Args:
            decision_context: Context for the decision
            skill_name: Name of the skill making the decision

        Returns:
            Enhanced context with memory-informed insights
        """
        # Extract query from context
        query = decision_context.get('query', '')
        if not query:
            query = f"Previous decisions for {skill_name}"

        # Query memory
        memories = await self.query_memory(query, decision_context)

        # Build enhanced context
        enhanced_context = decision_context.copy()
        enhanced_context['memory_informed'] = True
        enhanced_context['relevant_memories'] = memories
        enhanced_context['memory_count'] = len(memories)

        if memories:
            enhanced_context['memory_confidence'] = sum(
                m.get('confidence', 0.0) for m in memories
            ) / len(memories)
        else:
            enhanced_context['memory_confidence'] = 0.0

        return enhanced_context


# ============================================================================
# Preflight Integration
# ============================================================================

class PreflightCoordinator:
    """
    Coordinates preflight checks for skills before launch.

    Runs:
    - skill_tester: Validates skill functionality
    - skill_security_analyzer: Checks for vulnerabilities
    """

    def __init__(
        self,
        config: Optional[ZeroGConfig] = None,
        event_bus: Optional[EventBus] = None
    ):
        self.config = config or ZeroGConfig()
        self.event_bus = event_bus

    async def run_preflight(
        self,
        skill: DockableSkill,
        context: SkillExecutionContext
    ) -> Dict[str, Any]:
        """
        Run complete preflight sequence.

        Args:
            skill: Skill to validate
            context: Execution context

        Returns:
            Preflight results with pass/fail and details
        """
        results = {
            'passed': True,
            'checks_run': 0,
            'checks_passed': 0,
            'checks_failed': 0,
            'failures': [],
            'warnings': [],
            'details': {}
        }

        # Run skill preflight
        try:
            skill_result = await asyncio.wait_for(
                skill.preflight(context),
                timeout=self.config.preflight_timeout_seconds
            )

            results['checks_run'] += skill_result.checks_run
            results['checks_passed'] += skill_result.checks_passed
            results['checks_failed'] += skill_result.checks_failed
            results['failures'].extend(skill_result.failures)
            results['warnings'].extend(skill_result.warnings)
            results['details']['skill_preflight'] = skill_result

            if not skill_result.passed:
                results['passed'] = False

        except asyncio.TimeoutError:
            results['passed'] = False
            results['failures'].append(f"Preflight timeout ({self.config.preflight_timeout_seconds}s)")
            results['checks_failed'] += 1
        except Exception as e:
            results['passed'] = False
            results['failures'].append(f"Preflight error: {e}")
            results['checks_failed'] += 1

        # Run skill_tester checks (if enabled)
        if self.config.enable_skill_tester:
            tester_result = await self._run_skill_tester(skill, context)
            results['checks_run'] += tester_result['checks_run']
            results['checks_passed'] += tester_result['checks_passed']
            results['checks_failed'] += tester_result['checks_failed']
            results['details']['skill_tester'] = tester_result

            if not tester_result['passed']:
                results['passed'] = False
                results['failures'].extend(tester_result['failures'])

        # Run skill_security_analyzer checks (if enabled)
        if self.config.enable_security_analyzer:
            security_result = await self._run_security_analyzer(skill, context)
            results['checks_run'] += security_result['checks_run']
            results['checks_passed'] += security_result['checks_passed']
            results['checks_failed'] += security_result['checks_failed']
            results['details']['security_analyzer'] = security_result

            if not security_result['passed']:
                results['passed'] = False
                results['failures'].extend(security_result['failures'])

        # Emit event
        if self.event_bus:
            await self.event_bus.emit(SkillEvent(
                event_type=EventType.SKILL_STARTED if results['passed'] else EventType.SKILL_FAILED,
                skill_name=skill.manifest.name,
                timestamp=context.timestamp,
                payload={'preflight_results': results}
            ))

        return results

    async def _run_skill_tester(
        self,
        skill: DockableSkill,
        context: SkillExecutionContext
    ) -> Dict[str, Any]:
        """Run skill_tester meta-skill validation."""
        # Placeholder - integrate with actual skill_tester implementation
        return {
            'passed': True,
            'checks_run': 3,
            'checks_passed': 3,
            'checks_failed': 0,
            'failures': [],
            'tests': ['functionality', 'error_handling', 'timeout_behavior']
        }

    async def _run_security_analyzer(
        self,
        skill: DockableSkill,
        context: SkillExecutionContext
    ) -> Dict[str, Any]:
        """Run skill_security_analyzer meta-skill checks."""
        # Placeholder - integrate with actual skill_security_analyzer
        return {
            'passed': True,
            'checks_run': 5,
            'checks_passed': 5,
            'checks_failed': 0,
            'failures': [],
            'checks': [
                'input_validation',
                'output_sanitization',
                'resource_limits',
                'privilege_escalation',
                'data_leakage'
            ]
        }


# ============================================================================
# Orbit Monitoring
# ============================================================================

class OrbitMonitor:
    """
    Monitors skills in orbit and captures learning.

    Uses continuous_learning_capture to:
    - Monitor mission outcomes
    - Feed learning back to ReflectionBuffer
    - Auto-propose new skills based on patterns
    """

    def __init__(
        self,
        config: Optional[ZeroGConfig] = None,
        event_bus: Optional[EventBus] = None,
        memory_engine: Optional[MemoryInformedDecisionEngine] = None
    ):
        self.config = config or ZeroGConfig()
        self.event_bus = event_bus
        self.memory_engine = memory_engine
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}

    async def start_monitoring(self, skill_name: str):
        """Start monitoring a skill in orbit."""
        if not self.config.enable_continuous_learning:
            return

        if skill_name in self.monitoring_tasks:
            logger.warning(f"Already monitoring {skill_name}")
            return

        task = asyncio.create_task(self._monitor_loop(skill_name))
        self.monitoring_tasks[skill_name] = task
        logger.info(f"Started monitoring {skill_name}")

    async def stop_monitoring(self, skill_name: str):
        """Stop monitoring a skill."""
        if skill_name not in self.monitoring_tasks:
            return

        task = self.monitoring_tasks[skill_name]
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        del self.monitoring_tasks[skill_name]
        logger.info(f"Stopped monitoring {skill_name}")

    async def _monitor_loop(self, skill_name: str):
        """Continuous monitoring loop for a skill."""
        while True:
            try:
                await asyncio.sleep(self.config.learning_capture_interval)

                # Capture learning
                await self._capture_learning(skill_name)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring {skill_name}: {e}")

    async def _capture_learning(self, skill_name: str):
        """Capture learning from skill execution."""
        # Placeholder - integrate with continuous_learning_capture
        # In production:
        # - Query memory for skill outcomes
        # - Detect patterns (≥3 occurrences)
        # - Auto-propose new skills
        # - Feed to ReflectionBuffer

        logger.debug(f"Capturing learning for {skill_name}")

        # Emit learning event
        if self.event_bus:
            await self.event_bus.emit(SkillEvent(
                event_type=EventType.PATTERN_DETECTED,
                skill_name=skill_name,
                timestamp="",  # TODO: Add timestamp
                payload={'learning_captured': True}
            ))


# ============================================================================
# Mission Control
# ============================================================================

class MissionControlHub:
    """
    Mission Control for skill ecosystem.

    Responsibilities:
    - skill_gap_analyzer: Identifies missing capabilities
    - token_budget_adviser: Optimizes token usage
    - Health monitoring
    - Performance metrics
    """

    def __init__(
        self,
        config: Optional[ZeroGConfig] = None,
        event_bus: Optional[EventBus] = None,
        memory_engine: Optional[MemoryInformedDecisionEngine] = None
    ):
        self.config = config or ZeroGConfig()
        self.event_bus = event_bus
        self.memory_engine = memory_engine
        self.registered_skills: Dict[str, SkillManifest] = {}
        self.skill_health: Dict[str, Dict[str, Any]] = {}
        self.background_tasks: List[asyncio.Task] = []

    async def start(self):
        """Start Mission Control background tasks."""
        # Start gap analyzer
        if self.config.enable_gap_analyzer:
            task = asyncio.create_task(self._gap_analysis_loop())
            self.background_tasks.append(task)

        logger.info("Mission Control started")

    async def stop(self):
        """Stop Mission Control."""
        for task in self.background_tasks:
            task.cancel()

        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        logger.info("Mission Control stopped")

    async def register_skill(self, manifest: SkillManifest) -> str:
        """Register a skill with Mission Control."""
        skill_id = f"{manifest.name}:{manifest.version}"
        self.registered_skills[skill_id] = manifest
        self.skill_health[skill_id] = {
            'status': 'registered',
            'executions': 0,
            'successes': 0,
            'failures': 0,
            'avg_latency_ms': 0.0,
            'tokens_used': 0
        }

        logger.info(f"Registered skill: {skill_id}")
        return skill_id

    async def update_state(
        self,
        skill_name: str,
        state: SkillLifecycleState
    ) -> None:
        """Update skill lifecycle state."""
        for skill_id in self.registered_skills:
            if skill_id.startswith(skill_name):
                self.skill_health[skill_id]['status'] = state.value

                # Emit event
                if self.event_bus:
                    await self.event_bus.emit(SkillEvent(
                        event_type=EventType.SKILL_STARTED if state == SkillLifecycleState.LIFT_OFF else EventType.CUSTOM,
                        skill_name=skill_name,
                        timestamp="",
                        payload={'state': state.value}
                    ))
                break

    async def report_metrics(
        self,
        skill_name: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Report performance metrics for a skill."""
        for skill_id in self.registered_skills:
            if skill_id.startswith(skill_name):
                health = self.skill_health[skill_id]
                health['executions'] += 1

                if metrics.get('success'):
                    health['successes'] += 1
                else:
                    health['failures'] += 1

                # Update running average latency
                if 'execution_time_ms' in metrics:
                    n = health['executions']
                    avg = health['avg_latency_ms']
                    new_val = metrics['execution_time_ms']
                    health['avg_latency_ms'] = ((n - 1) * avg + new_val) / n

                # Update token usage
                if 'tokens_used' in metrics:
                    health['tokens_used'] += metrics['tokens_used']

                # Check token budget
                if self.config.enable_token_adviser:
                    if health['tokens_used'] > self.config.token_budget_limit:
                        logger.warning(
                            f"Skill {skill_name} exceeded token budget: "
                            f"{health['tokens_used']} / {self.config.token_budget_limit}"
                        )

                        # Emit warning event
                        if self.event_bus:
                            await self.event_bus.emit(SkillEvent(
                                event_type=EventType.QUALITY_WARNING,
                                skill_name=skill_name,
                                timestamp="",
                                payload={
                                    'warning': 'token_budget_exceeded',
                                    'tokens_used': health['tokens_used'],
                                    'budget': self.config.token_budget_limit
                                }
                            ))
                break

    async def health_check(self, skill_name: str) -> Dict[str, Any]:
        """Get health status for a skill."""
        for skill_id in self.registered_skills:
            if skill_id.startswith(skill_name):
                return self.skill_health[skill_id].copy()

        return {'status': 'unknown'}

    async def _gap_analysis_loop(self):
        """Periodic gap analysis to identify missing skills."""
        while True:
            try:
                await asyncio.sleep(self.config.gap_analysis_interval)
                await self._run_gap_analysis()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in gap analysis: {e}")

    async def _run_gap_analysis(self):
        """Run skill gap analyzer."""
        # Placeholder - integrate with skill_gap_analyzer meta-skill
        # In production:
        # - Query memory for user requests
        # - Identify patterns of missing capabilities
        # - Generate roadmap of needed skills
        # - Prioritize by demand

        logger.debug("Running gap analysis")

        # Emit gap identified event (if gaps found)
        if self.event_bus:
            # Example: Detected need for "data_visualization" skill
            await self.event_bus.emit(SkillEvent(
                event_type=EventType.GAP_IDENTIFIED,
                skill_name="mission_control",
                timestamp="",
                payload={
                    'gap': 'Example: data_visualization skill needed',
                    'priority': 'medium',
                    'demand': 5  # 5 user requests
                }
            ))


# ============================================================================
# Zero-G Orchestrator
# ============================================================================

class ZeroGOrchestrator:
    """
    Main orchestrator integrating skills with Zero-G platform.

    Provides:
    - Unified docking interface
    - Memory-informed decision making
    - Lifecycle management (Preflight → Orbit)
    - Mission Control integration
    - Event bus coordination
    """

    def __init__(
        self,
        config: Optional[ZeroGConfig] = None,
        warp_space: Optional[Any] = None,
        yarn_graph: Optional[Any] = None,
        event_bus: Optional[EventBus] = None
    ):
        self.config = config or ZeroGConfig()

        # Core infrastructure
        self.memory_engine = MemoryInformedDecisionEngine(
            warp_space=warp_space,
            yarn_graph=yarn_graph,
            config=self.config
        )

        # Lifecycle components
        self.preflight_coordinator = PreflightCoordinator(
            config=self.config,
            event_bus=event_bus
        )
        self.orbit_monitor = OrbitMonitor(
            config=self.config,
            event_bus=event_bus,
            memory_engine=self.memory_engine
        )
        self.mission_control = MissionControlHub(
            config=self.config,
            event_bus=event_bus,
            memory_engine=self.memory_engine
        )

        # Docking manager
        self.docking_manager = SkillDockingManager(
            warp_space=warp_space,
            yarn_graph=yarn_graph,
            event_bus=event_bus,
            mission_control=self.mission_control
        )

    async def start(self):
        """Start Zero-G orchestrator."""
        await self.mission_control.start()
        logger.info("Zero-G Orchestrator started")

    async def stop(self):
        """Stop Zero-G orchestrator."""
        await self.mission_control.stop()

        # Stop all orbit monitoring
        for skill_name in list(self.orbit_monitor.monitoring_tasks.keys()):
            await self.orbit_monitor.stop_monitoring(skill_name)

        logger.info("Zero-G Orchestrator stopped")

    async def dock_and_launch(
        self,
        skill: DockableSkill,
        parameters: Dict[str, Any]
    ) -> SkillExecutionResult:
        """
        Complete launch sequence: Dock → Preflight → Launch → Orbit.

        Args:
            skill: Skill to launch
            parameters: Execution parameters

        Returns:
            Execution result with complete provenance
        """
        skill_name = skill.manifest.name

        # 1. Dock skill
        success, message = await self.docking_manager.dock_skill(skill)
        if not success:
            return SkillExecutionResult(
                success=False,
                output=None,
                errors=[f"Docking failed: {message}"]
            )

        logger.info(f"Docked: {skill_name}")

        # 2. Preflight checks
        context = self.docking_manager._create_context(skill_name)
        preflight_result = await self.preflight_coordinator.run_preflight(
            skill,
            context
        )

        if not preflight_result['passed']:
            return SkillExecutionResult(
                success=False,
                output=None,
                errors=preflight_result['failures'],
                warnings=preflight_result['warnings']
            )

        logger.info(f"Preflight passed: {skill_name}")

        # 3. Enhance parameters with memory
        enhanced_params = await self.memory_engine.inform_decision(
            parameters,
            skill_name
        )

        # 4. Execute skill (Lift-Off → Orbit)
        result = await self.docking_manager.execute_skill(
            skill_name,
            enhanced_params
        )

        # 5. Start orbit monitoring (if successful)
        if result.success:
            await self.orbit_monitor.start_monitoring(skill_name)
            logger.info(f"In orbit: {skill_name}")

        return result

    async def get_mission_status(self) -> Dict[str, Any]:
        """Get complete mission status."""
        docked_skills = self.docking_manager.get_docked_skills()

        status = {
            'docked_skills': docked_skills,
            'skills_in_orbit': list(self.orbit_monitor.monitoring_tasks.keys()),
            'health': {}
        }

        # Get health for each docked skill
        for skill_name in docked_skills:
            health = await self.mission_control.health_check(skill_name)
            status['health'][skill_name] = health

        return status


# ============================================================================
# Convenience Functions
# ============================================================================

async def create_zero_g_orchestrator(
    config: Optional[ZeroGConfig] = None,
    warp_space: Optional[Any] = None,
    yarn_graph: Optional[Any] = None,
    event_bus: Optional[EventBus] = None
) -> ZeroGOrchestrator:
    """
    Create and start Zero-G orchestrator.

    Args:
        config: Zero-G configuration
        warp_space: WarpSpace instance for semantic retrieval
        yarn_graph: YarnGraph instance for knowledge graph
        event_bus: Event bus for skill communication

    Returns:
        Started orchestrator
    """
    orchestrator = ZeroGOrchestrator(
        config=config,
        warp_space=warp_space,
        yarn_graph=yarn_graph,
        event_bus=event_bus
    )

    await orchestrator.start()
    return orchestrator


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Configuration
    'ZeroGConfig',

    # Core components
    'MemoryInformedDecisionEngine',
    'PreflightCoordinator',
    'OrbitMonitor',
    'MissionControlHub',
    'ZeroGOrchestrator',

    # Convenience
    'create_zero_g_orchestrator',
]
