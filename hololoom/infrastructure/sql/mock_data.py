#!/usr/bin/env python3
"""
Mock Data Loader for SQL Backend

Loads validated test data from Demo 3 into SQL backend.
Part 2: Foundation Infrastructure (Days 1-5)
"""

import asyncio
import logging
from typing import List, Tuple

from hololoom.infrastructure.sql.backend import SQLBackend, SQLConfig, create_sql_backend


logger = logging.getLogger(__name__)


# ============================================================================
# Mock Data (from Demo 3 - validated)
# ============================================================================

MOCK_POLICY_RULES = [
    ("bee_001", "Varroa Treatment Schedule", "treatment", 1,
     {"frequency": "monthly", "method": "oxalic_acid", "threshold": 3},
     1.0, "beekeeping", "neo4j_policy_001"),

    ("bee_002", "Honey Harvest Guidelines", "harvest", 1,
     {"min_weight": "60lbs", "season": "late_summer", "leave_reserve": "40lbs"},
     1.0, "beekeeping", "neo4j_policy_002"),

    ("bee_003", "Hive Inspection Protocol", "inspection", 1,
     {"frequency": "weekly", "checklist": ["queen_health", "brood_pattern", "food_stores"]},
     1.0, "beekeeping", "neo4j_policy_003"),

    ("bee_004", "Pest Management Protocol", "pest_control", 1,
     {"methods": ["integrated_pest_management", "screen_bottom_board", "drone_comb"]},
     1.0, "beekeeping", "neo4j_policy_004"),

    ("bee_005", "Winter Preparation Checklist", "seasonal", 1,
     {"insulation": "required", "food_stores": "60lbs", "entrance_reducer": "install"},
     1.0, "beekeeping", "neo4j_policy_005"),
]

MOCK_TRANSACTIONS = [
    ("txn_001", "treatment", "hive", "hive_042", "user_123",
     {"action": "apply_oxalic_acid", "dose": "2.5g", "result": "success"},
     "neo4j_hive_042"),

    ("txn_002", "inspection", "hive", "hive_042", "user_123",
     {"findings": "healthy_queen", "brood_pattern": "good", "mites_count": 2},
     "neo4j_hive_042"),

    ("txn_003", "harvest", "hive", "hive_001", "user_456",
     {"weight": "75lbs", "quality": "excellent", "moisture": "17%"},
     "neo4j_hive_001"),

    ("txn_004", "treatment", "hive", "hive_001", "user_123",
     {"action": "formic_acid_strip", "duration": "14_days", "result": "success"},
     "neo4j_hive_001"),

    ("txn_005", "inspection", "hive", "hive_001", "user_456",
     {"findings": "queen_missing", "brood_pattern": "spotty", "action": "requeen"},
     "neo4j_hive_001"),
]

MOCK_AUDIT_TRAILS = [
    ("audit_001", "access", "hive", "hive_042", "user_123",
     {"location": "North Apiary"},
     {"location": "North Apiary", "last_inspection": "2025-01-01"},
     False),

    ("audit_002", "modification", "policy", "bee_001", "admin_001",
     {"threshold": 2},
     {"threshold": 3},
     True),

    ("audit_003", "access", "hive", "hive_042", "user_789",
     None,
     {"location": "North Apiary"},
     False),

    ("audit_004", "modification", "hive", "hive_001", "user_456",
     {"status": "active"},
     {"status": "needs_requeening"},
     True),
]

MOCK_PERMISSIONS = [
    ("perm_001", "user_123", "hive", "read_write", "neo4j_user_123", None),
    ("perm_002", "user_456", "hive", "read", "neo4j_user_456", None),
    ("perm_003", "admin_001", "policy", "admin", "neo4j_admin_001", None),
    ("perm_004", "user_789", "hive", "read", "neo4j_user_789", None),
]


# ============================================================================
# Mock Data Loader
# ============================================================================

async def load_mock_data(backend: SQLBackend) -> dict:
    """
    Load mock data into SQL backend

    Args:
        backend: Connected SQL backend

    Returns:
        Summary statistics
    """
    logger.info("Loading mock data into SQL backend")

    stats = {
        "policies_inserted": 0,
        "transactions_inserted": 0,
        "audits_inserted": 0,
        "permissions_inserted": 0,
        "errors": []
    }

    # Insert policy rules
    for policy in MOCK_POLICY_RULES:
        rule_id, rule_name, rule_type, version, rule_logic, confidence, domain, neo4j_node_id = policy

        result = await backend.insert_policy_rule(
            rule_id=rule_id,
            rule_name=rule_name,
            rule_type=rule_type,
            rule_logic=rule_logic,
            confidence=confidence,
            domain=domain,
            neo4j_node_id=neo4j_node_id
        )

        if result.success:
            stats["policies_inserted"] += 1
        else:
            stats["errors"].append(f"Policy {rule_id}: {result.error}")

    # Insert transaction logs
    for txn in MOCK_TRANSACTIONS:
        txn_id, txn_type, entity_type, entity_id, user_id, action_data, neo4j_node_id = txn

        result = await backend.insert_transaction_log(
            transaction_id=txn_id,
            transaction_type=txn_type,
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=user_id,
            action_data=action_data,
            neo4j_node_id=neo4j_node_id
        )

        if result.success:
            stats["transactions_inserted"] += 1
        else:
            stats["errors"].append(f"Transaction {txn_id}: {result.error}")

    # Insert audit trails
    for audit in MOCK_AUDIT_TRAILS:
        audit_id, audit_type, resource_type, resource_id, user_id, before_state, after_state, compliance_flag = audit

        result = await backend.insert_audit_trail(
            audit_id=audit_id,
            audit_type=audit_type,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            before_state=before_state,
            after_state=after_state,
            compliance_flag=compliance_flag
        )

        if result.success:
            stats["audits_inserted"] += 1
        else:
            stats["errors"].append(f"Audit {audit_id}: {result.error}")

    # Insert user permissions
    for perm in MOCK_PERMISSIONS:
        perm_id, user_id, resource_type, perm_level, neo4j_user_node, expires_at = perm

        result = await backend.insert_user_permission(
            permission_id=perm_id,
            user_id=user_id,
            resource_type=resource_type,
            permission_level=perm_level,
            neo4j_user_node=neo4j_user_node,
            expires_at=expires_at
        )

        if result.success:
            stats["permissions_inserted"] += 1
        else:
            stats["errors"].append(f"Permission {perm_id}: {result.error}")

    logger.info(f"Mock data loaded: {stats}")
    return stats


# ============================================================================
# Main (for testing)
# ============================================================================

async def main():
    """Test mock data loading"""
    print("=" * 80)
    print("Mock Data Loader Test")
    print("=" * 80)
    print()

    # Create backend (will use SQLite by default)
    config = SQLConfig(sqlite_path="./data/test_mock_data.db")

    async with create_sql_backend(config) as backend:
        print(f"Connected to {backend.backend_type}")
        print()

        # Load mock data
        stats = await load_mock_data(backend)

        print("Mock Data Loading Results:")
        print(f"  Policies:     {stats['policies_inserted']}/{len(MOCK_POLICY_RULES)}")
        print(f"  Transactions: {stats['transactions_inserted']}/{len(MOCK_TRANSACTIONS)}")
        print(f"  Audits:       {stats['audits_inserted']}/{len(MOCK_AUDIT_TRAILS)}")
        print(f"  Permissions:  {stats['permissions_inserted']}/{len(MOCK_PERMISSIONS)}")

        if stats['errors']:
            print(f"\nErrors ({len(stats['errors'])}):")
            for error in stats['errors']:
                print(f"  - {error}")
        else:
            print("\n[PASS] All mock data inserted successfully")

        # Test query
        print("\nTesting query:")
        result = await backend.get_policy_rule("bee_001")

        if result.success and result.row_count > 0:
            row = result.rows[0]
            print(f"  Policy: {row['rule_name']}")
            print(f"  Type:   {row['rule_type']}")
            print(f"  Logic:  {row['rule_logic']}")
            print(f"\n[PASS] Query successful (latency: {result.latency_ms:.2f}ms)")
        else:
            print(f"\n[FAIL] Query failed: {result.error}")

    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
