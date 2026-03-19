"""Proto core engine module.

Thin waist orchestrator for Proto code agent department.

Exports:
    - ProtoEngine: Main orchestrator
    - ProtoConfig: Configuration
    - ProtoMode: Execution modes
"""

from hololoom.apps.departments.proto.core.config import ProtoConfig, ProtoMode
from hololoom.apps.departments.proto.core.engine import ProtoEngine

__all__ = [
    "ProtoEngine",
    "ProtoConfig",
    "ProtoMode",
]
