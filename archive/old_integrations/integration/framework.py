"""
Integration Framework for HoloLoom

Enables composable department-based workflows with:
- Pipeline composition (declarative workflow definition)
- Parallel execution (independent stages run concurrently)
- Graceful degradation (optional stages don't block critical path)
- Confidence aggregation (weighted confidence from all stages)
- Complete audit trail (full provenance tracking)

Architecture:
    StageDefinition → PipelineDefinition → IntegrationFramework

Usage:
    framework = IntegrationFramework(registry, config)
    result = await framework.execute_pipeline(query, PIPELINE_QA)

Author: HoloLoom Team
Date: 2025-01-21
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from HoloLoom.config import Config
from HoloLoom.departments.protocol import Department, DepartmentRequest, DepartmentResponse, ConfidenceMetadata
from HoloLoom.departments.registry import DepartmentRegistry
from HoloLoom.protocols.types import Query

logger = logging.getLogger(__name__)


# ============================================================================
# Stage Types
# ============================================================================

class StageType(Enum):
    """Types of processing stages in a pipeline."""
    RETRIEVAL = "retrieval"              # Memory retrieval, context gathering
    REASONING = "reasoning"              # Core reasoning, decision making
    QUALITY_ASSURANCE = "quality_assurance"  # Code quality, slop detection
    VERIFICATION = "verification"        # Result verification, fact checking
    GUIDANCE = "guidance"                # User guidance, recommendations
    ANALYSIS = "analysis"                # Data analysis, pattern detection
    PLANNING = "planning"                # Task planning, goal decomposition
    EXECUTION = "execution"              # Tool execution, action taking
    OUTPUT = "output"                    # Response generation, formatting
    COORDINATION = "coordination"        # Multi-agent coordination
    LEARNING = "learning"                # Learning from feedback


# ============================================================================
# Stage Definition
# ============================================================================

@dataclass
class StageDefinition:
    """
    Defines a single processing stage in a pipeline.

    Attributes:
        stage_type: Type of processing (RETRIEVAL, REASONING, etc.)
        department_name: Which department executes this stage
        timeout_ms: Maximum execution time in milliseconds
        required: Whether stage must succeed for pipeline to continue
        confidence_threshold: Minimum confidence to proceed (0.0-1.0)
        parallel_with: List of stage names that can run in parallel with this one
        fallback_action: Optional fallback function if stage fails
        metadata: Additional stage-specific configuration
    """
    stage_type: StageType
    department_name: str
    timeout_ms: int = 30000  # 30 seconds default
    required: bool = True
    confidence_threshold: float = 0.5
    parallel_with: Optional[List[str]] = None
    fallback_action: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate stage definition."""
        if self.timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be positive, got {self.timeout_ms}")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}")


# ============================================================================
# Stage Result
# ============================================================================

@dataclass
class StageResult:
    """Result from executing a single stage."""
    stage_name: str
    stage_type: StageType
    success: bool
    result: Optional[DepartmentResponse] = None
    skipped: bool = False
    error: Optional[Exception] = None
    confidence: float = 0.0
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def is_viable(self, threshold: float = 0.5) -> bool:
        """Check if result meets confidence threshold."""
        return self.success and self.confidence >= threshold


# ============================================================================
# Pipeline Definition
# ============================================================================

@dataclass
class PipelineDefinition:
    """
    Defines a complete processing pipeline.

    Attributes:
        name: Pipeline identifier
        stages: List of stage definitions in execution order
        default_mode: Default execution mode (BARE, FAST, FUSED)
        enable_parallel: Whether to enable parallel stage execution
        enable_graceful_degradation: Whether to continue on optional stage failures
        max_total_timeout_ms: Maximum total pipeline execution time
        metadata: Additional pipeline-specific configuration
    """
    name: str
    stages: List[StageDefinition]
    default_mode: str = "FAST"
    enable_parallel: bool = True
    enable_graceful_degradation: bool = True
    max_total_timeout_ms: int = 60000  # 60 seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_critical_path(self) -> List[StageDefinition]:
        """Get stages that must succeed."""
        return [s for s in self.stages if s.required]

    def get_optional_stages(self) -> List[StageDefinition]:
        """Get stages that can fail without blocking pipeline."""
        return [s for s in self.stages if not s.required]

    def get_stage_by_name(self, name: str) -> Optional[StageDefinition]:
        """Get stage definition by department name."""
        for stage in self.stages:
            if stage.department_name == name:
                return stage
        return None


# ============================================================================
# Pipeline Result
# ============================================================================

@dataclass
class PipelineResult:
    """Result from executing a complete pipeline."""
    pipeline_name: str
    success: bool
    final_response: Any
    stage_results: Dict[str, StageResult]
    overall_confidence: float
    critical_path_viable: bool
    total_duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_successful_stages(self) -> List[str]:
        """Get names of stages that succeeded."""
        return [name for name, result in self.stage_results.items() if result.success]

    def get_failed_stages(self) -> List[str]:
        """Get names of stages that failed."""
        return [name for name, result in self.stage_results.items()
                if not result.success and not result.skipped]

    def get_skipped_stages(self) -> List[str]:
        """Get names of stages that were skipped."""
        return [name for name, result in self.stage_results.items() if result.skipped]


# ============================================================================
# Dependency Graph
# ============================================================================

class DependencyGraph:
    """
    Builds and analyzes stage dependencies for parallel execution.

    Identifies which stages can run in parallel vs which must be sequential.
    """

    def __init__(self, pipeline: PipelineDefinition):
        self.pipeline = pipeline
        self.dependencies: Dict[str, Set[str]] = {}
        self.parallel_groups: List[List[StageDefinition]] = []
        self._build_graph()

    def _build_graph(self) -> None:
        """Build dependency graph from pipeline definition."""
        # Initialize dependencies
        for stage in self.pipeline.stages:
            self.dependencies[stage.department_name] = set()

        # Stage depends on all previous stages EXCEPT those it can run in parallel with
        for i, stage in enumerate(self.pipeline.stages):
            for j in range(i):
                prev_stage = self.pipeline.stages[j]

                # Check if stages can run in parallel
                if stage.parallel_with and prev_stage.department_name in stage.parallel_with:
                    continue  # No dependency

                # Otherwise, stage depends on previous stage
                self.dependencies[stage.department_name].add(prev_stage.department_name)

        # Build parallel groups
        self._identify_parallel_groups()

    def _identify_parallel_groups(self) -> None:
        """Identify groups of stages that can run in parallel."""
        visited = set()

        for stage in self.pipeline.stages:
            if stage.department_name in visited:
                continue

            # Find all stages that can run in parallel with this one
            parallel_group = [stage]
            visited.add(stage.department_name)

            if stage.parallel_with:
                for other_stage in self.pipeline.stages:
                    if (other_stage.department_name in stage.parallel_with and
                        other_stage.department_name not in visited):
                        parallel_group.append(other_stage)
                        visited.add(other_stage.department_name)

            self.parallel_groups.append(parallel_group)

    def get_parallel_groups(self) -> List[List[StageDefinition]]:
        """Get groups of stages that can execute in parallel."""
        return self.parallel_groups

    def can_execute(self, stage_name: str, completed: Set[str]) -> bool:
        """Check if stage can execute given completed stages."""
        deps = self.dependencies.get(stage_name, set())
        return deps.issubset(completed)


# ============================================================================
# Integration Framework
# ============================================================================

class IntegrationFramework:
    """
    Core integration framework for composing department-based workflows.

    Features:
    - Parallel execution of independent stages
    - Graceful degradation for optional stages
    - Confidence aggregation across stages
    - Complete audit trail with timing data
    - Smart fallback strategies

    Usage:
        framework = IntegrationFramework(registry, config)
        result = await framework.execute_pipeline(query, PIPELINE_QA)
    """

    def __init__(
        self,
        registry: DepartmentRegistry,
        config: Config,
        enable_logging: bool = True
    ):
        """
        Initialize integration framework.

        Args:
            registry: Department registry for discovery
            config: HoloLoom configuration
            enable_logging: Whether to enable detailed logging
        """
        self.registry = registry
        self.config = config
        self.enable_logging = enable_logging
        self.logger = logger if enable_logging else logging.getLogger("null")

    async def execute_pipeline(
        self,
        query: Query,
        pipeline: PipelineDefinition,
        context: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Execute a complete pipeline with parallel execution and graceful degradation.

        Args:
            query: Input query
            pipeline: Pipeline definition
            context: Optional context dictionary

        Returns:
            PipelineResult with complete execution trace
        """
        start_time = datetime.now()
        context = context or {}
        stage_results: Dict[str, StageResult] = {}

        self.logger.info(f"🚀 Starting pipeline: {pipeline.name}")
        self.logger.info(f"   Stages: {len(pipeline.stages)} total, "
                        f"{len(pipeline.get_critical_path())} critical, "
                        f"{len(pipeline.get_optional_stages())} optional")

        try:
            # Build dependency graph
            dep_graph = DependencyGraph(pipeline)
            parallel_groups = dep_graph.get_parallel_groups()

            self.logger.info(f"   Parallel groups: {len(parallel_groups)}")

            # Execute parallel groups
            for group_idx, group in enumerate(parallel_groups):
                self.logger.info(f"\n📦 Group {group_idx + 1}/{len(parallel_groups)}: "
                               f"{len(group)} stage(s)")

                # Execute stages in group concurrently
                group_results = await self._execute_stage_group(
                    group, query, context, stage_results, pipeline
                )

                # Update stage results
                for stage_name, result in group_results.items():
                    stage_results[stage_name] = result

                # Check if critical path still viable
                if not self._is_critical_path_viable(pipeline, stage_results):
                    self.logger.error("❌ Critical path failed, aborting pipeline")
                    break

            # Synthesize final result
            result = self._synthesize_result(
                pipeline, query, stage_results, start_time
            )

            self.logger.info(f"\n✅ Pipeline complete: {pipeline.name}")
            self.logger.info(f"   Success: {result.success}")
            self.logger.info(f"   Confidence: {result.overall_confidence:.2f}")
            self.logger.info(f"   Duration: {result.total_duration_ms:.0f}ms")
            self.logger.info(f"   Successful: {len(result.get_successful_stages())}/{len(pipeline.stages)}")

            return result

        except asyncio.TimeoutError:
            self.logger.error(f"⏱️ Pipeline timeout: {pipeline.name}")
            return self._create_timeout_result(pipeline, stage_results, start_time)

        except Exception as e:
            self.logger.error(f"💥 Pipeline error: {pipeline.name} - {e}")
            return self._create_error_result(pipeline, stage_results, start_time, e)

    async def _execute_stage_group(
        self,
        group: List[StageDefinition],
        query: Query,
        context: Dict[str, Any],
        previous_results: Dict[str, StageResult],
        pipeline: PipelineDefinition
    ) -> Dict[str, StageResult]:
        """Execute a group of stages in parallel."""
        tasks = []
        stage_names = []

        for stage in group:
            task = self._execute_stage(stage, query, context, previous_results, pipeline)
            tasks.append(task)
            stage_names.append(stage.department_name)

        # Execute all stages concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Package results
        group_results = {}
        for stage_name, result in zip(stage_names, results):
            if isinstance(result, Exception):
                # Stage raised exception
                stage = next(s for s in group if s.department_name == stage_name)
                group_results[stage_name] = StageResult(
                    stage_name=stage_name,
                    stage_type=stage.stage_type,
                    success=False,
                    error=result,
                    skipped=not stage.required  # Skip if optional
                )
            else:
                group_results[stage_name] = result

        return group_results

    async def _execute_stage(
        self,
        stage: StageDefinition,
        query: Query,
        context: Dict[str, Any],
        previous_results: Dict[str, StageResult],
        pipeline: PipelineDefinition
    ) -> StageResult:
        """Execute a single stage with timeout and fallback."""
        start_time = datetime.now()

        self.logger.info(f"   🔧 {stage.stage_type.value}: {stage.department_name} "
                        f"({'required' if stage.required else 'optional'})")

        try:
            # Get department from registry
            department = self.registry.get_department(stage.department_name)
            if not department:
                raise ValueError(f"Department not found: {stage.department_name}")

            # Build request
            request = DepartmentRequest(
                task_id=f"{pipeline.name}_{stage.department_name}_{datetime.now().timestamp()}",
                task_type=stage.stage_type.value,
                parameters={
                    "query": query.text,
                    "context": context,
                    "previous_results": {
                        name: result.result for name, result in previous_results.items()
                        if result.result is not None
                    },
                    "metadata": stage.metadata
                },
                metadata={
                    "pipeline": pipeline.name,
                    "stage_type": stage.stage_type.value
                }
            )

            # Execute with timeout
            response = await asyncio.wait_for(
                department.execute(request),
                timeout=stage.timeout_ms / 1000.0
            )

            # Extract confidence
            confidence = 0.8  # Default
            if hasattr(response, 'confidence') and response.confidence:
                if isinstance(response.confidence, ConfidenceMetadata):
                    confidence = response.confidence.score
                elif isinstance(response.confidence, (int, float)):
                    confidence = float(response.confidence)

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            self.logger.info(f"      ✓ Success (confidence: {confidence:.2f}, "
                           f"duration: {duration_ms:.0f}ms)")

            return StageResult(
                stage_name=stage.department_name,
                stage_type=stage.stage_type,
                success=True,
                result=response,
                confidence=confidence,
                duration_ms=duration_ms
            )

        except asyncio.TimeoutError:
            self.logger.warning(f"      ⏱️ Timeout ({stage.timeout_ms}ms)")
            return await self._handle_stage_failure(
                stage, "timeout", None, start_time
            )

        except Exception as e:
            self.logger.error(f"      ✗ Error: {e}")
            return await self._handle_stage_failure(
                stage, "error", e, start_time
            )

    async def _handle_stage_failure(
        self,
        stage: StageDefinition,
        failure_type: str,
        error: Optional[Exception],
        start_time: datetime
    ) -> StageResult:
        """Handle stage failure with fallback strategies."""
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Try fallback action if defined
        if stage.fallback_action:
            try:
                self.logger.info(f"      🔄 Attempting fallback...")
                fallback_result = await stage.fallback_action()
                return StageResult(
                    stage_name=stage.department_name,
                    stage_type=stage.stage_type,
                    success=True,
                    result=fallback_result,
                    confidence=0.5,  # Lower confidence for fallback
                    duration_ms=duration_ms
                )
            except Exception as fallback_error:
                self.logger.error(f"      ✗ Fallback failed: {fallback_error}")

        # If optional, skip
        if not stage.required:
            self.logger.info(f"      ⏭️ Skipping optional stage")
            return StageResult(
                stage_name=stage.department_name,
                stage_type=stage.stage_type,
                success=False,
                skipped=True,
                error=error,
                duration_ms=duration_ms
            )

        # Critical stage failed
        return StageResult(
            stage_name=stage.department_name,
            stage_type=stage.stage_type,
            success=False,
            error=error,
            duration_ms=duration_ms
        )

    def _is_critical_path_viable(
        self,
        pipeline: PipelineDefinition,
        stage_results: Dict[str, StageResult]
    ) -> bool:
        """Check if critical path is still viable."""
        for stage in pipeline.get_critical_path():
            result = stage_results.get(stage.department_name)
            if result and not result.success:
                return False
        return True

    def _synthesize_result(
        self,
        pipeline: PipelineDefinition,
        query: Query,
        stage_results: Dict[str, StageResult],
        start_time: datetime
    ) -> PipelineResult:
        """Synthesize final pipeline result."""
        # Determine success
        critical_path_viable = self._is_critical_path_viable(pipeline, stage_results)
        success = critical_path_viable and len(stage_results) > 0

        # Aggregate confidence
        overall_confidence = self._aggregate_confidence(pipeline, stage_results)

        # Extract final response (from last successful stage)
        final_response = None
        for stage in reversed(pipeline.stages):
            result = stage_results.get(stage.department_name)
            if result and result.success and result.result:
                final_response = result.result
                break

        # Calculate duration
        total_duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        return PipelineResult(
            pipeline_name=pipeline.name,
            success=success,
            final_response=final_response,
            stage_results=stage_results,
            overall_confidence=overall_confidence,
            critical_path_viable=critical_path_viable,
            total_duration_ms=total_duration_ms,
            metadata={
                "query": query.text,
                "mode": pipeline.default_mode,
                "parallel_enabled": pipeline.enable_parallel
            }
        )

    def _aggregate_confidence(
        self,
        pipeline: PipelineDefinition,
        stage_results: Dict[str, StageResult]
    ) -> float:
        """
        Aggregate confidence from all stages.

        Strategy:
        1. Critical path determines minimum confidence (weakest link)
        2. Optional stages can boost confidence up to +0.15
        """
        # Get critical path confidences
        critical_confidences = []
        for stage in pipeline.get_critical_path():
            result = stage_results.get(stage.department_name)
            if result and result.success:
                critical_confidences.append(result.confidence)

        if not critical_confidences:
            return 0.0  # Pipeline failed

        # Base confidence is weakest link in critical path
        base_confidence = min(critical_confidences)

        # Optional stages boost confidence
        optional_successes = sum(
            1 for stage in pipeline.get_optional_stages()
            if stage_results.get(stage.department_name)
            and stage_results[stage.department_name].success
        )

        confidence_boost = min(0.15, optional_successes * 0.05)

        return min(1.0, base_confidence + confidence_boost)

    def _create_timeout_result(
        self,
        pipeline: PipelineDefinition,
        stage_results: Dict[str, StageResult],
        start_time: datetime
    ) -> PipelineResult:
        """Create result for timeout case."""
        return PipelineResult(
            pipeline_name=pipeline.name,
            success=False,
            final_response=None,
            stage_results=stage_results,
            overall_confidence=0.0,
            critical_path_viable=False,
            total_duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
            metadata={"error": "timeout"}
        )

    def _create_error_result(
        self,
        pipeline: PipelineDefinition,
        stage_results: Dict[str, StageResult],
        start_time: datetime,
        error: Exception
    ) -> PipelineResult:
        """Create result for error case."""
        return PipelineResult(
            pipeline_name=pipeline.name,
            success=False,
            final_response=None,
            stage_results=stage_results,
            overall_confidence=0.0,
            critical_path_viable=False,
            total_duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
            metadata={"error": str(error)}
        )


# ============================================================================
# Utility Functions
# ============================================================================

def create_integration_framework(
    registry: DepartmentRegistry,
    config: Config
) -> IntegrationFramework:
    """
    Factory function to create integration framework.

    Args:
        registry: Department registry
        config: HoloLoom configuration

    Returns:
        Configured IntegrationFramework instance
    """
    return IntegrationFramework(registry, config)