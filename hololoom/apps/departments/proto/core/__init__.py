"""Proto core engine module.

Thin waist orchestrator for Proto code agent department.

Exports:
    - ProtoEngine: Main orchestrator
    - ProtoConfig: Configuration
    - ProtoMode: Execution modes
"""

from hololoom.apps.departments.proto.core.engine import ProtoEngine
from hololoom.apps.departments.proto.core.config import ProtoConfig, ProtoMode

__all__ = [
    "ProtoEngine",
    "ProtoConfig",
    "ProtoMode",
]
