"""
SQL Backend for Hybrid Query Routing

Part 2: Foundation Infrastructure (Days 1-5)

Public API:
- SQLBackend: Main backend class
- SQLConfig: Configuration dataclass
- create_sql_backend: Factory function
- load_mock_data: Mock data loader
"""

from HoloLoom.infrastructure.sql.backend import (
    SQLBackend,
    SQLConfig,
    QueryResult,
    create_sql_backend
)

from HoloLoom.infrastructure.sql.mock_data import (
    load_mock_data,
    MOCK_POLICY_RULES,
    MOCK_TRANSACTIONS,
    MOCK_AUDIT_TRAILS,
    MOCK_PERMISSIONS
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
