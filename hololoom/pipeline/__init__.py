"""
HoloLoom Pipeline Framework
============================

Generic infrastructure for building typed, dependency-resolved pipelines.

The framework provides:
- Stage[C] protocol with reads/writes declarations
- Level-aware resolver (topological sort into parallel groups)
- PipelineRunner with pluggable RunPolicy
- Content-addressed stage caching
- ReadWriteTracker for CI verification

This package is generic — it knows nothing about weaving, queries, or
any specific HoloLoom domain. It works with any typed context dataclass.
"""

from hololoom.pipeline.cache import StageCache
from hololoom.pipeline.resolver import (
    resolve_levels,
    validate_completeness,
    validate_disjoint_writes,
    validate_stage_fields,
)
from hololoom.pipeline.runner import PipelineRunner
from hololoom.pipeline.testing import ReadWriteTracker
from hololoom.pipeline.types import DefaultPolicy, RunPolicy, Stage

__all__ = [
    'Stage',
    'RunPolicy',
    'DefaultPolicy',
    'resolve_levels',
    'validate_completeness',
    'validate_disjoint_writes',
    'validate_stage_fields',
    'PipelineRunner',
    'StageCache',
    'ReadWriteTracker',
]
