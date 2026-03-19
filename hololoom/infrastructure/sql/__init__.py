"""
SQL Backend for Hybrid Query Routing

Part 2: Foundation Infrastructure (Days 1-5)

Public API:
- SQLBackend: Main backend class
- SQLConfig: Configuration dataclass
- create_sql_backend: Factory function
- load_mock_data: Mock data loader
"""

from hololoom.infrastructure.sql.backend import (
    QueryResult,
    SQLBackend,
    SQLConfig,
    create_sql_backend,
)
from hololoom.infrastructure.sql.mock_data import (
    MOCK_AUDIT_TRAILS,
    MOCK_PERMISSIONS,
    MOCK_POLICY_RULES,
    MOCK_TRANSACTIONS,
    load_mock_data,
)

__all__ = [
    # Backend
    "SQLBackend",
    "SQLConfig",
    "QueryResult",
    "create_sql_backend",

    # Mock data
    "load_mock_data",
    "MOCK_POLICY_RULES",
    "MOCK_TRANSACTIONS",
    "MOCK_AUDIT_TRAILS",
    "MOCK_PERMISSIONS",
]
