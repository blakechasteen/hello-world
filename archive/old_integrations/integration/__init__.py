"""
HoloLoom Integration Framework

Provides composable department-based workflows with:
- Declarative pipeline definitions
- Parallel execution engine
- Graceful degradation
- Complete audit trails

Quick Start:
    ```python
    from HoloLoom.integration import (
        IntegrationFramework,
        get_pipeline,
        create_integration_framework
    )
    from HoloLoom.departments.registry import DepartmentRegistry
    from HoloLoom.config import Config

    # Setup
    registry = DepartmentRegistry()
    config = Config.fused()
    framework = create_integration_framework(registry, config)

    # Execute pipeline
    from HoloLoom.protocols.types import Query
    result = await framework.execute_pipeline(
        Query(text="Your question here"),
        get_pipeline("quality_assured")
    )
    ```

Author: HoloLoom Team
Date: 2025-01-21
"""

# Core framework
from HoloLoom.integration.framework import (
    IntegrationFramework,
    PipelineDefinition,
    PipelineResult,
    StageDefinition,
    StageResult,
    StageType,
    DependencyGraph,
    create_integration_framework,
)

# Predefined pipelines
from HoloLoom.integration.pipelines import (
    PIPELINE_SIMPLE,
    PIPELINE_VERIFIED,
    PIPELINE_QUALITY_ASSURED,
    PIPELINE_COLLABORATIVE,
    PIPELINE_COMPREHENSIVE,
    PIPELINE_SHUTTLE_OPTIMIZED,
    PIPELINES,
    get_pipeline,
    list_pipelines,
    get_pipeline_info,
)

__all__ = [
    # Core classes
    "IntegrationFramework",
    "PipelineDefinition",
    "PipelineResult",
    "StageDefinition",
    "StageResult",
    "StageType",
    "DependencyGraph",

    # Factory functions
    "create_integration_framework",

    # Predefined pipelines
    "PIPELINE_SIMPLE",
    "PIPELINE_VERIFIED",
    "PIPELINE_QUALITY_ASSURED",
    "PIPELINE_COLLABORATIVE",
    "PIPELINE_COMPREHENSIVE",
    "PIPELINE_SHUTTLE_OPTIMIZED",
    "PIPELINES",

    # Pipeline utilities
    "get_pipeline",
    "list_pipelines",
    "get_pipeline_info",
]

__version__ = "1.0.0"