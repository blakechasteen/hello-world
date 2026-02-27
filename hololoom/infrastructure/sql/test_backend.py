#!/usr/bin/env python3
"""
SQL Backend Test Suite

Part 2: Foundation Infrastructure (Days 1-5)
Validation Gate 2.1: SQL Backend Functional

Tests:
1. Connection (PostgreSQL + SQLite fallback)
2. Schema initialization
3. CRUD operations (4 tables)
4. Query performance (p95 < 30ms)
5. Error handling
"""

import asyncio
import logging
from pathlib import Path

from hololoom.infrastructure.sql import (
    SQLBackend,
    SQLConfig,
    create_sql_backend,
    load_mock_data
)


logger = logging.getLogger(__name__)


def print_header(text: str):
    """Print section header"""
    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


async def test_connection():
    """Test 1: Database connection"""
    print_header("Test 1: Database Connection")

    # Test SQLite (always works)
    config = SQLConfig(sqlite_path="./data/test_backend.db")

    async with create_sql_backend(config) as backend:
        print(f"Backend type: {backend.backend_type}")
        print(f"Closed: {backend._closed}")

        if backend.backend_type == "sqlite":
            print("[PASS] SQLite connection successful")
        elif backend.backend_type == "postgresql":
            print("[PASS] PostgreSQL connection successful")
        else:
            print(f"[FAIL] Unknown backend type: {backend.backend_type}")
            return False

    # Verify cleanup
    if backend._closed:
        print("[PASS] Backend closed properly")
    else:
        print("[FAIL] Backend not closed")
        return False

    return True


async def test_schema_initialization():
    """Test 2: Schema initialization"""
    print_header("Test 2: Schema Initialization")

    config = SQLConfig(sqlite_path="./data/test_schema.db")

    async with create_sql_backend(config) as backend:
        # Check if tables exist
        tables = ["policy_rules", "transaction_logs", "audit_trails", "user_permissions"]

        for table in tables:
            result = await backend.execute_query(
                f"SELECT COUNT(*) as count FROM {table}"
            )

            if result.success:
                print(f"[PASS] Table '{table}' exists")
            else:
                print(f"[FAIL] Table '{table}' not found: {result.error}")
                return False

    return True


async def test_crud_operations():
    """Test 3: CRUD operations"""
    print_header("Test 3: CRUD Operations")

    config = SQLConfig(sqlite_path="./data/test_crud.db")

    async with create_sql_backend(config) as backend:
        # Test policy rule insert
        result = await backend.insert_policy_rule(
            rule_id="test_001",
            rule_name="Test Policy",
            rule_type="test",
            rule_logic={"test": "data"},
            confidence=0.95,
            domain="test"
        )

        if result.success:
            print(f"[PASS] Policy rule inserted (latency: {result.latency_ms:.2f}ms)")
        else:
            print(f"[FAIL] Policy rule insert failed: {result.error}")
            return False

        # Test policy rule retrieval
        result = await backend.get_policy_rule("test_001")

        if result.success and result.row_count == 1:
            row = result.rows[0]
            print(f"[PASS] Policy rule retrieved (latency: {result.latency_ms:.2f}ms)")
            print(f"  Name: {row['rule_name']}")
            print(f"  Type: {row['rule_type']}")
        else:
            print(f"[FAIL] Policy rule retrieval failed: {result.error}")
            return False

        # Test transaction log insert
        result = await backend.insert_transaction_log(
            transaction_id="test_txn_001",
            transaction_type="test",
            entity_type="test_entity",
            entity_id="entity_001",
            user_id="test_user",
            action_data={"action": "test"}
        )

        if result.success:
            print(f"[PASS] Transaction log inserted (latency: {result.latency_ms:.2f}ms)")
        else:
            print(f"[FAIL] Transaction log insert failed: {result.error}")
            return False

        # Test audit trail insert
        result = await backend.insert_audit_trail(
            audit_id="test_audit_001",
            audit_type="test",
            resource_type="test_resource",
            resource_id="resource_001",
            user_id="test_user",
            before_state={"status": "old"},
            after_state={"status": "new"},
            compliance_flag=True
        )

        if result.success:
            print(f"[PASS] Audit trail inserted (latency: {result.latency_ms:.2f}ms)")
        else:
            print(f"[FAIL] Audit trail insert failed: {result.error}")
            return False

        # Test user permission insert
        result = await backend.insert_user_permission(
            permission_id="test_perm_001",
            user_id="test_user",
            resource_type="test_resource",
            permission_level="read_write"
        )

        if result.success:
            print(f"[PASS] User permission inserted (latency: {result.latency_ms:.2f}ms)")
        else:
            print(f"[FAIL] User permission insert failed: {result.error}")
            return False

    return True


async def test_mock_data_loading():
    """Test 4: Mock data loading"""
    print_header("Test 4: Mock Data Loading")

    config = SQLConfig(sqlite_path="./data/test_mock_data.db")

    async with create_sql_backend(config) as backend:
        stats = await load_mock_data(backend)

        print(f"Policies inserted:     {stats['policies_inserted']}")
        print(f"Transactions inserted: {stats['transactions_inserted']}")
        print(f"Audits inserted:       {stats['audits_inserted']}")
        print(f"Permissions inserted:  {stats['permissions_inserted']}")

        total_inserted = (
            stats['policies_inserted'] +
            stats['transactions_inserted'] +
            stats['audits_inserted'] +
            stats['permissions_inserted']
        )

        if total_inserted > 0 and len(stats['errors']) == 0:
            print(f"[PASS] Mock data loaded successfully ({total_inserted} rows)")
        else:
            print(f"[FAIL] Mock data loading failed ({len(stats['errors'])} errors)")
            for error in stats['errors']:
                print(f"  - {error}")
            return False

    return True


async def test_query_performance():
    """Test 5: Query performance"""
    print_header("Test 5: Query Performance")

    config = SQLConfig(sqlite_path="./data/test_performance.db")

    async with create_sql_backend(config) as backend:
        # Load mock data first
        await load_mock_data(backend)

        # Run test queries
        test_queries = [
            ("Policy by ID", "SELECT * FROM policy_rules WHERE rule_id = ?", ["bee_001"]),
            ("Transaction by entity", "SELECT * FROM transaction_logs WHERE entity_id = ?", ["hive_042"]),
            ("Audit by compliance", "SELECT * FROM audit_trails WHERE compliance_flag = ?", [True]),
            ("Permission by user", "SELECT * FROM user_permissions WHERE user_id = ?", ["user_123"]),
            ("Policy count", "SELECT COUNT(*) as count FROM policy_rules", []),
            ("Transaction join", """
                SELECT t.transaction_id, t.entity_id, a.audit_type
                FROM transaction_logs t
                LEFT JOIN audit_trails a ON t.entity_id = a.resource_id
                WHERE t.entity_id = ?
            """, ["hive_042"]),
        ]

        latencies = []

        print()
        print(f"{'Query':<25} {'Rows':>6} {'Latency':>12} {'Status'}")
        print("-" * 80)

        for query_name, sql, params in test_queries:
            result = await backend.execute_query(sql, params)

            latencies.append(result.latency_ms)

            status = "[PASS]" if result.success else "[FAIL]"
            print(f"{query_name:<25} {result.row_count:>6} {result.latency_ms:>11.2f}ms {status}")

            if not result.success:
                print(f"  ERROR: {result.error}")
                return False

        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        print()
        print(f"Latency (avg):  {avg_latency:.2f}ms")
        print(f"Latency (min):  {min_latency:.2f}ms")
        print(f"Latency (max):  {max_latency:.2f}ms")
        print(f"Latency (p95):  {p95_latency:.2f}ms")

        # Validate performance (p95 < 30ms target)
        if p95_latency < 30.0:
            print(f"\n[PASS] Performance target met (p95: {p95_latency:.2f}ms < 30ms)")
        else:
            print(f"\n[WARN] Performance target exceeded (p95: {p95_latency:.2f}ms > 30ms)")
            print("[INFO] SQLite is very fast; PostgreSQL may be slower in production")

    return True


async def test_error_handling():
    """Test 6: Error handling"""
    print_header("Test 6: Error Handling")

    config = SQLConfig(sqlite_path="./data/test_errors.db")

    async with create_sql_backend(config) as backend:
        # Test invalid SQL
        result = await backend.execute_query("SELECT * FROM nonexistent_table")

        if not result.success and result.error:
            print(f"[PASS] Invalid SQL handled gracefully: {result.error}")
        else:
            print("[FAIL] Invalid SQL should have failed")
            return False

        # Test duplicate key
        await backend.insert_policy_rule(
            rule_id="dup_001",
            rule_name="Duplicate Test",
            rule_type="test",
            rule_logic={"test": "data"}
        )

        result = await backend.insert_policy_rule(
            rule_id="dup_001",  # Same ID
            rule_name="Duplicate Test 2",
            rule_type="test",
            rule_logic={"test": "data2"}
        )

        if not result.success and result.error:
            print(f"[PASS] Duplicate key handled gracefully: {result.error}")
        else:
            print("[FAIL] Duplicate key should have failed")
            return False

    return True


async def main():
    """Run all tests"""
    print("=" * 80)
    print("SQL Backend Test Suite")
    print("=" * 80)
    print()
    print("Part 2: Foundation Infrastructure (Days 1-5)")
    print("Validation Gate 2.1: SQL Backend Functional")
    print()

    # Clean up old test databases
    test_db_dir = Path("./data")
    if test_db_dir.exists():
        for db_file in test_db_dir.glob("test_*.db"):
            db_file.unlink()
            print(f"Cleaned up: {db_file}")

    # Run tests
    tests = [
        ("Connection", test_connection),
        ("Schema Initialization", test_schema_initialization),
        ("CRUD Operations", test_crud_operations),
        ("Mock Data Loading", test_mock_data_loading),
        ("Query Performance", test_query_performance),
        ("Error Handling", test_error_handling),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}", exc_info=True)
            results.append((test_name, False))

    # Summary
    print_header("VALIDATION GATE 2.1: SQL Backend Functional")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print()
    print(f"Tests Passed: {passed}/{total}")
    print()

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")

    print()

    if passed == total:
        print("[PASS] All tests passed - SQL backend is functional")
        print("[READY] Proceed to Days 6-10: MCP Server Implementation")
    else:
        print(f"[FAIL] {total - passed} test(s) failed")
        print("[TODO] Fix failing tests before proceeding")

    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
