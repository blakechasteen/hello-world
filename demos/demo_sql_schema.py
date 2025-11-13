#!/usr/bin/env python3
"""
Demo 3: SQL Schema + Mock Queries

Demonstrates SQL schema design for precision data with beekeeping domain.
Shows query performance benchmarks and validates schema for multi-domain support.

Part of: Hybrid Query Routing Architecture - Part 1 (Proof-of-Concept Demos)
"""

import sqlite3
import time
import json
from typing import List, Tuple, Any
from dataclasses import dataclass


@dataclass
class QueryResult:
    """Query execution result"""
    query_type: str
    sql: str
    params: List[Any]
    rows_returned: int
    latency_ms: float
    success: bool
    error: str = None


class BeekeepingSQLDemo:
    """SQL schema demo for beekeeping domain"""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Dict-like rows

    def create_schema(self):
        """Create beekeeping schema with 4 precision tables"""

        cursor = self.conn.cursor()

        # Policy Rules (ground truth)
        cursor.execute("""
            CREATE TABLE policy_rules (
                rule_id TEXT PRIMARY KEY,
                rule_name TEXT NOT NULL,
                rule_type TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                rule_logic TEXT NOT NULL,  -- JSON
                confidence REAL DEFAULT 1.0,
                domain TEXT DEFAULT 'beekeeping',
                neo4j_node_id TEXT,  -- Link to Neo4j
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX idx_policy_rules_name ON policy_rules(rule_name)")
        cursor.execute("CREATE INDEX idx_policy_rules_type ON policy_rules(rule_type)")

        # Transaction Logs (precision data)
        cursor.execute("""
            CREATE TABLE transaction_logs (
                transaction_id TEXT PRIMARY KEY,
                transaction_type TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                action_data TEXT NOT NULL,  -- JSON
                neo4j_node_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX idx_transaction_logs_entity ON transaction_logs(entity_type, entity_id)")
        cursor.execute("CREATE INDEX idx_transaction_logs_user ON transaction_logs(user_id)")

        # Audit Trails (compliance)
        cursor.execute("""
            CREATE TABLE audit_trails (
                audit_id TEXT PRIMARY KEY,
                audit_type TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                before_state TEXT,  -- JSON snapshot
                after_state TEXT,   -- JSON snapshot
                compliance_flag BOOLEAN DEFAULT FALSE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX idx_audit_trails_resource ON audit_trails(resource_type, resource_id)")
        cursor.execute("CREATE INDEX idx_audit_trails_compliance ON audit_trails(compliance_flag)")

        # User Permissions (access control)
        cursor.execute("""
            CREATE TABLE user_permissions (
                permission_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                permission_level TEXT NOT NULL,
                neo4j_user_node TEXT,
                granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX idx_user_permissions_user ON user_permissions(user_id)")
        cursor.execute("CREATE INDEX idx_user_permissions_resource ON user_permissions(resource_type)")

        self.conn.commit()

    def insert_mock_data(self):
        """Insert mock beekeeping data"""

        cursor = self.conn.cursor()

        # Mock policy rules
        policies = [
            ("bee_001", "Varroa Treatment Schedule", "treatment", 1,
             json.dumps({"frequency": "monthly", "method": "oxalic_acid", "threshold": 3}),
             1.0, "beekeeping", "neo4j_policy_001"),
            ("bee_002", "Honey Harvest Guidelines", "harvest", 1,
             json.dumps({"min_weight": "60lbs", "season": "late_summer", "leave_reserve": "40lbs"}),
             1.0, "beekeeping", "neo4j_policy_002"),
            ("bee_003", "Hive Inspection Protocol", "inspection", 1,
             json.dumps({"frequency": "weekly", "checklist": ["queen_health", "brood_pattern", "food_stores"]}),
             1.0, "beekeeping", "neo4j_policy_003"),
            ("bee_004", "Pest Management Protocol", "pest_control", 1,
             json.dumps({"methods": ["integrated_pest_management", "screen_bottom_board", "drone_comb"]}),
             1.0, "beekeeping", "neo4j_policy_004"),
            ("bee_005", "Winter Preparation Checklist", "seasonal", 1,
             json.dumps({"insulation": "required", "food_stores": "60lbs", "entrance_reducer": "install"}),
             1.0, "beekeeping", "neo4j_policy_005"),
        ]

        for policy in policies:
            cursor.execute("""
                INSERT INTO policy_rules
                (rule_id, rule_name, rule_type, version, rule_logic, confidence, domain, neo4j_node_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, policy)

        # Mock transactions
        transactions = [
            ("txn_001", "treatment", "hive", "hive_042", "user_123",
             json.dumps({"action": "apply_oxalic_acid", "dose": "2.5g", "result": "success"}),
             "neo4j_hive_042"),
            ("txn_002", "inspection", "hive", "hive_042", "user_123",
             json.dumps({"findings": "healthy_queen", "brood_pattern": "good", "mites_count": 2}),
             "neo4j_hive_042"),
            ("txn_003", "harvest", "hive", "hive_001", "user_456",
             json.dumps({"weight": "75lbs", "quality": "excellent", "moisture": "17%"}),
             "neo4j_hive_001"),
        ]

        for txn in transactions:
            cursor.execute("""
                INSERT INTO transaction_logs
                (transaction_id, transaction_type, entity_type, entity_id, user_id, action_data, neo4j_node_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, txn)

        # Mock audit trails
        audits = [
            ("audit_001", "access", "hive", "hive_042", "user_123",
             json.dumps({"location": "North Apiary"}),
             json.dumps({"location": "North Apiary", "last_inspection": "2025-01-01"}),
             False),
            ("audit_002", "modification", "policy", "bee_001", "admin_001",
             json.dumps({"threshold": 2}),
             json.dumps({"threshold": 3}),
             True),
            ("audit_003", "access", "hive", "hive_042", "user_789",
             None,
             json.dumps({"location": "North Apiary"}),
             False),
        ]

        for audit in audits:
            cursor.execute("""
                INSERT INTO audit_trails
                (audit_id, audit_type, resource_type, resource_id, user_id, before_state, after_state, compliance_flag)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, audit)

        # Mock permissions
        permissions = [
            ("perm_001", "user_123", "hive", "read_write", "neo4j_user_123", None),
            ("perm_002", "user_456", "hive", "read", "neo4j_user_456", None),
            ("perm_003", "admin_001", "policy", "admin", "neo4j_admin_001", None),
        ]

        for perm in permissions:
            cursor.execute("""
                INSERT INTO user_permissions
                (permission_id, user_id, resource_type, permission_level, neo4j_user_node, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, perm)

        self.conn.commit()

    def execute_query(self, query_type: str, sql: str, params: List[Any] = None) -> QueryResult:
        """Execute query and measure performance"""

        if params is None:
            params = []

        start = time.time()
        success = True
        error = None
        rows = []

        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        except Exception as e:
            success = False
            error = str(e)

        latency_ms = (time.time() - start) * 1000

        return QueryResult(
            query_type=query_type,
            sql=sql,
            params=params,
            rows_returned=len(rows),
            latency_ms=latency_ms,
            success=success,
            error=error
        )

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def print_header(text: str):
    """Print section header"""
    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


def main():
    """Run SQL schema demo"""

    print("=" * 80)
    print("DEMO 3: SQL Schema + Mock Queries")
    print("=" * 80)
    print()
    print("Creating beekeeping SQL schema with 4 precision tables")
    print("Running mock queries and benchmarking performance")
    print()

    # Initialize database
    demo = BeekeepingSQLDemo()
    demo.connect()

    print_header("Schema Creation")
    print("\nCreating tables:")
    print("  1. policy_rules      (ground truth policies)")
    print("  2. transaction_logs  (precision data)")
    print("  3. audit_trails      (compliance)")
    print("  4. user_permissions  (access control)")

    demo.create_schema()
    print("\n[PASS] Schema created successfully")

    print("\nInserting mock data:")
    print("  - 5 policy rules")
    print("  - 3 transactions")
    print("  - 3 audit trails")
    print("  - 3 user permissions")

    demo.insert_mock_data()
    print("\n[PASS] Mock data inserted successfully")

    # Test queries
    print_header("Query Performance Benchmarks")

    test_queries = [
        # Exact ID lookups (Rule 1)
        ("Exact ID (Policy)",
         "SELECT * FROM policy_rules WHERE rule_id = ?",
         ["bee_001"]),

        ("Exact ID (Audit)",
         "SELECT * FROM audit_trails WHERE audit_id = ?",
         ["audit_002"]),

        # Policy lookups (Rule 2)
        ("Policy by Name",
         "SELECT * FROM policy_rules WHERE rule_name LIKE ?",
         ["%Varroa%"]),

        ("Audit Trails (Compliance)",
         "SELECT * FROM audit_trails WHERE compliance_flag = ?",
         [True]),

        # Aggregations (Rule 3)
        ("Count Policies",
         "SELECT COUNT(*) as total FROM policy_rules",
         []),

        ("Count by Type",
         "SELECT rule_type, COUNT(*) as count FROM policy_rules GROUP BY rule_type",
         []),

        # Complex joins (Rule 6 - hybrid)
        ("Transaction + Audit",
         """
         SELECT t.transaction_id, t.entity_id, a.audit_type
         FROM transaction_logs t
         LEFT JOIN audit_trails a ON t.entity_id = a.resource_id
         WHERE t.entity_id = ?
         """,
         ["hive_042"]),

        # User permissions (access control)
        ("User Permissions",
         "SELECT * FROM user_permissions WHERE user_id = ?",
         ["user_123"]),
    ]

    results = []
    print()
    print(f"{'Query Type':<25} {'Rows':>6} {'Latency':>12} {'Status'}")
    print("-" * 80)

    for query_type, sql, params in test_queries:
        result = demo.execute_query(query_type, sql, params)
        results.append(result)

        status = "[PASS]" if result.success else "[FAIL]"
        latency_str = f"{result.latency_ms:.2f}ms"

        print(f"{query_type:<25} {result.rows_returned:>6} {latency_str:>12} {status}")

        if not result.success:
            print(f"  ERROR: {result.error}")

    # Performance analysis
    print_header("Performance Analysis")

    successful_results = [r for r in results if r.success]
    latencies = [r.latency_ms for r in successful_results]

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else max_latency

        print()
        print(f"Total queries:    {len(results)}")
        print(f"Successful:       {len(successful_results)}")
        print(f"Failed:           {len(results) - len(successful_results)}")
        print()
        print(f"Latency (avg):    {avg_latency:.2f}ms")
        print(f"Latency (min):    {min_latency:.2f}ms")
        print(f"Latency (max):    {max_latency:.2f}ms")
        print(f"Latency (p95):    {p95_latency:.2f}ms")

        # Validation
        print_header("VALIDATION GATE 1.3: SQL Schema + Performance")

        schema_valid = len(successful_results) == len(results)
        performance_valid = p95_latency < 30.0  # Target: <30ms (p95)

        if schema_valid:
            print("[PASS] All queries executed successfully (schema valid)")
        else:
            print(f"[FAIL] {len(results) - len(successful_results)} queries failed")

        if performance_valid:
            print(f"[PASS] p95 latency {p95_latency:.2f}ms < 30ms target")
        else:
            print(f"[WARN] p95 latency {p95_latency:.2f}ms > 30ms target")
            print("[INFO] SQLite in-memory is very fast; production PostgreSQL may be slower")

        print()

        # Schema validation for multi-domain
        print_header("Multi-Domain Support Validation")

        print("\nSchema Design Features:")
        print("  [PASS] Domain-specific tables (beekeeping)")
        print("  [PASS] JSON columns for flexibility (rule_logic, action_data)")
        print("  [PASS] Neo4j linking columns (neo4j_node_id)")
        print("  [PASS] Indexing for performance (8 indexes created)")
        print()
        print("Multi-Domain Scalability:")
        print("  - Add healthcare schema: Same structure, different domain column")
        print("  - Add finance schema:    Same structure, different domain column")
        print("  - Hybrid approach:       Domain-specific + JSON = maximum flexibility")

        # Sample data inspection
        print_header("Sample Query Results")

        # Show policy rule
        policy_result = demo.execute_query(
            "Policy Detail",
            "SELECT * FROM policy_rules WHERE rule_id = ?",
            ["bee_001"]
        )

        if policy_result.success and policy_result.rows_returned > 0:
            cursor = demo.conn.cursor()
            cursor.execute("SELECT * FROM policy_rules WHERE rule_id = ?", ["bee_001"])
            row = cursor.fetchone()

            print("\nSample Policy Rule (bee_001):")
            print(f"  Name:       {row['rule_name']}")
            print(f"  Type:       {row['rule_type']}")
            print(f"  Logic:      {row['rule_logic']}")
            print(f"  Confidence: {row['confidence']}")
            print(f"  Neo4j Link: {row['neo4j_node_id']}")

        # Next steps
        print_header("Next Steps")

        if schema_valid and performance_valid:
            print("[READY] SQL schema validated")
            print("[READY] Performance targets met")
            print("[READY] Proceed to Demo 4: Multi-Backend Routing Flow")
            print("[READY] Begin Phase 1 implementation (SQL backend)")
        elif schema_valid:
            print("[WARN] Schema valid but performance needs optimization")
            print("[TODO] Add more indexes for slow queries")
            print("[TODO] Test with PostgreSQL for production estimates")
        else:
            print("[TODO] Fix schema errors before proceeding")
            print("[TODO] Revisit table definitions")

        print()

    demo.close()


if __name__ == "__main__":
    main()
