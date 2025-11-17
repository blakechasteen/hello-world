#!/usr/bin/env python3
"""
Custom Database Backend Template and Tutorial

This file serves as both:
1. A working example of a custom storage backend
2. A step-by-step tutorial for creating your own backends
3. A template you can copy and modify

TUTORIAL: How to Create a Custom Storage Backend
================================================

Step 1: Import Required Modules
-------------------------------
You need to import:
- BaseStorageBackend from plugins.base
- typing for type hints
- Your database driver/client library
- Any other dependencies

Step 2: Inherit from BaseStorageBackend
---------------------------------------
Your class must inherit from BaseStorageBackend and implement
all required methods defined in the protocol.

Step 3: Implement Required Methods
----------------------------------
You MUST implement these methods:
- init_storage(storage_path: str) -> None
- save_prompt(prompt_data: Dict[str, Any]) -> str
- get_prompt(name, branch, version, commit_hash) -> Optional[Dict[str, Any]]
- list_prompts(branch: str) -> List[Dict[str, Any]]
- create_branch(branch_name, from_branch) -> None
- get_current_branch() -> str
- set_current_branch(branch_name: str) -> None
- get_commit_history(name, branch, limit) -> List[Dict[str, Any]]
- save_evaluation(eval_data: Dict[str, Any]) -> None
- save_chain(chain_data: Dict[str, Any]) -> None
- get_chain(name: str) -> Optional[Dict[str, Any]]
- close() -> None

Step 4: Add Custom Features (Optional)
--------------------------------------
You can add backend-specific methods for:
- Performance optimization
- Advanced queries
- Monitoring and statistics
- Backup and recovery
- Migration tools

Step 5: Test Your Backend
-------------------------
Create tests to verify:
- All required methods work correctly
- Error handling is robust
- Performance meets requirements
- Data integrity is maintained

Example Custom Backend: CassandraDB
===================================
Below is a complete example using Apache Cassandra as the storage backend.
This demonstrates all the concepts above.
"""

import hashlib
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

# Step 1: Import your database driver
try:
    from cassandra.cluster import Cluster
    from cassandra.auth import PlainTextAuthProvider
    from cassandra.query import SimpleStatement
    CASSANDRA_AVAILABLE = True
except ImportError:
    CASSANDRA_AVAILABLE = False
    print("Warning: cassandra-driver not available. Install with: pip install cassandra-driver")

# Import base class
from ...base import BaseStorageBackend


# Step 2: Create your backend class
class CustomDBStorage(BaseStorageBackend):
    """
    Custom Database Storage Backend - Apache Cassandra Example

    This backend demonstrates how to create a custom storage implementation
    using Apache Cassandra (a distributed NoSQL database).

    Key Features:
    - Distributed architecture for high availability
    - Horizontal scalability
    - Tunable consistency levels
    - Time-series optimized storage
    - Multi-datacenter replication

    Schema Design:
    --------------
    Cassandra uses a denormalized schema optimized for read patterns.

    Table: prompts
    - partition_key: (branch, name)  # Distributes data across nodes
    - clustering_key: version DESC   # Orders versions within partition
    - columns: content, commit_hash, created_at, metadata

    Table: branches
    - partition_key: name
    - columns: head_commit, created_at

    Table: evaluations
    - partition_key: (prompt_name, commit_hash)
    - clustering_key: created_at DESC
    - columns: test_case, expected, actual, score, metrics

    Table: chains
    - partition_key: name
    - columns: steps, description, created_at
    """

    def __init__(
        self,
        contact_points: List[str] = ['127.0.0.1'],
        port: int = 9042,
        keyspace: str = 'promptly',
        username: Optional[str] = None,
        password: Optional[str] = None,
        consistency_level: str = 'LOCAL_QUORUM',
        replication_factor: int = 3,
    ):
        """
        Initialize Cassandra storage backend

        Args:
            contact_points: List of Cassandra node addresses
            port: Cassandra port (default: 9042)
            keyspace: Keyspace name (like database in SQL)
            username: Optional username for authentication
            password: Optional password for authentication
            consistency_level: Read/write consistency level
            replication_factor: Number of replicas

        Tutorial Note:
        -------------
        In __init__, you should:
        1. Call super().__init__() with name and description
        2. Store configuration parameters
        3. Initialize client variables (but don't connect yet)
        4. Set up any local state (caches, locks, etc.)
        """
        # Always call parent constructor
        super().__init__(
            name="custom_db",
            description="Custom Database Storage (Cassandra Example)"
        )

        # Check if driver is available
        if not CASSANDRA_AVAILABLE:
            raise ImportError(
                "cassandra-driver is required. "
                "Install with: pip install cassandra-driver"
            )

        # Store configuration
        self.contact_points = contact_points
        self.port = port
        self.keyspace = keyspace
        self.username = username
        self.password = password
        self.consistency_level = consistency_level
        self.replication_factor = replication_factor

        # Initialize client variables (will be set in init_storage)
        self.cluster = None
        self.session = None
        self.current_branch = 'main'

    def init_storage(self, storage_path: str) -> None:
        """
        Initialize database connection and schema

        Args:
            storage_path: Connection string or configuration

        Tutorial Note:
        -------------
        In init_storage, you should:
        1. Parse the storage_path parameter
        2. Connect to your database
        3. Create tables/collections if they don't exist
        4. Run any initialization logic
        5. Verify the connection works
        """
        # Step 1: Setup authentication if provided
        auth_provider = None
        if self.username and self.password:
            auth_provider = PlainTextAuthProvider(
                username=self.username,
                password=self.password
            )

        # Step 2: Connect to Cassandra cluster
        try:
            self.cluster = Cluster(
                contact_points=self.contact_points,
                port=self.port,
                auth_provider=auth_provider,
            )
            self.session = self.cluster.connect()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Cassandra: {e}")

        # Step 3: Create keyspace if it doesn't exist
        self._create_keyspace()

        # Step 4: Use the keyspace
        self.session.set_keyspace(self.keyspace)

        # Step 5: Create tables
        self._create_tables()

        # Step 6: Initialize main branch
        self._ensure_branch_exists('main')

    def _create_keyspace(self) -> None:
        """Create keyspace with replication strategy"""
        create_keyspace_query = f"""
            CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
            WITH replication = {{
                'class': 'SimpleStrategy',
                'replication_factor': {self.replication_factor}
            }}
        """
        self.session.execute(create_keyspace_query)

    def _create_tables(self) -> None:
        """
        Create database tables/collections

        Tutorial Note:
        -------------
        Design your schema to match your storage requirements:
        - Primary keys for efficient lookups
        - Indexes for common queries
        - Appropriate data types
        - Consider normalization vs. denormalization
        """
        # Prompts table - partitioned by branch and name, ordered by version
        self.session.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                branch text,
                name text,
                version int,
                content text,
                commit_hash text,
                created_at timestamp,
                metadata text,
                PRIMARY KEY ((branch, name), version)
            ) WITH CLUSTERING ORDER BY (version DESC)
        """)

        # Branches table
        self.session.execute("""
            CREATE TABLE IF NOT EXISTS branches (
                name text PRIMARY KEY,
                head_commit text,
                created_at timestamp
            )
        """)

        # Evaluations table
        self.session.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                prompt_name text,
                commit_hash text,
                eval_id uuid,
                test_case text,
                expected text,
                actual text,
                score double,
                metrics text,
                evaluator_name text,
                created_at timestamp,
                PRIMARY KEY ((prompt_name, commit_hash), created_at)
            ) WITH CLUSTERING ORDER BY (created_at DESC)
        """)

        # Chains table
        self.session.execute("""
            CREATE TABLE IF NOT EXISTS chains (
                name text PRIMARY KEY,
                steps text,
                description text,
                created_at timestamp
            )
        """)

        # Create secondary indexes for common queries
        try:
            self.session.execute(
                "CREATE INDEX IF NOT EXISTS ON prompts (commit_hash)"
            )
        except Exception:
            pass  # Index might already exist

    def _ensure_branch_exists(self, branch_name: str) -> None:
        """Ensure a branch exists in the database"""
        query = "SELECT name FROM branches WHERE name = ?"
        result = self.session.execute(query, [branch_name]).one()

        if not result:
            insert_query = """
                INSERT INTO branches (name, head_commit, created_at)
                VALUES (?, ?, ?)
            """
            self.session.execute(
                insert_query,
                [branch_name, 'init', datetime.now()]
            )

    def _generate_commit_hash(self, name: str, content: str, timestamp: str) -> str:
        """
        Generate a unique commit hash

        Tutorial Note:
        -------------
        Commit hashes should be:
        - Deterministic (same input = same hash)
        - Unique (different content = different hash)
        - Short enough to be user-friendly
        - Long enough to avoid collisions
        """
        data = f"{name}:{content}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    def save_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """
        Save a prompt to the database

        Args:
            prompt_data: Dictionary containing:
                - name: Prompt name (required)
                - content: Prompt content (required)
                - branch: Branch name (default: 'main')
                - metadata: Additional metadata (optional)

        Returns:
            Commit hash for this version

        Tutorial Note:
        -------------
        In save_prompt, you should:
        1. Extract required fields from prompt_data
        2. Generate a unique commit hash
        3. Determine the version number
        4. Insert the new prompt version
        5. Update branch metadata
        6. Return the commit hash
        """
        # Step 1: Extract fields
        name = prompt_data['name']
        content = prompt_data['content']
        branch = prompt_data.get('branch', 'main')
        metadata = prompt_data.get('metadata', {})

        # Step 2: Generate commit hash
        timestamp = datetime.now().isoformat()
        commit_hash = self._generate_commit_hash(name, content, timestamp)

        # Step 3: Get next version number
        version = self._get_next_version(name, branch)

        # Step 4: Insert prompt
        insert_query = """
            INSERT INTO prompts (branch, name, version, content, commit_hash, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self.session.execute(
            insert_query,
            [
                branch,
                name,
                version,
                content,
                commit_hash,
                datetime.now(),
                json.dumps(metadata) if metadata else None
            ]
        )

        # Step 5: Update branch head
        self._update_branch_head(branch, commit_hash)

        # Step 6: Return commit hash
        return commit_hash

    def _get_next_version(self, name: str, branch: str) -> int:
        """Get the next version number for a prompt"""
        query = """
            SELECT MAX(version) as max_version
            FROM prompts
            WHERE branch = ? AND name = ?
        """
        result = self.session.execute(query, [branch, name]).one()

        if result and result.max_version is not None:
            return result.max_version + 1
        return 1

    def _update_branch_head(self, branch_name: str, commit_hash: str) -> None:
        """Update the head commit for a branch"""
        update_query = """
            UPDATE branches
            SET head_commit = ?
            WHERE name = ?
        """
        self.session.execute(update_query, [commit_hash, branch_name])

    def get_prompt(
        self,
        name: str,
        branch: str = "main",
        version: Optional[int] = None,
        commit_hash: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a prompt from the database

        Tutorial Note:
        -------------
        In get_prompt, you should:
        1. Handle different retrieval modes (version, commit_hash, latest)
        2. Query your database efficiently
        3. Convert database row to dictionary
        4. Return None if not found
        """
        if commit_hash:
            # Query by commit hash (slower due to index)
            query = """
                SELECT branch, name, version, content, commit_hash, created_at, metadata
                FROM prompts
                WHERE commit_hash = ?
                ALLOW FILTERING
            """
            result = self.session.execute(query, [commit_hash]).one()
        elif version is not None:
            # Query by specific version
            query = """
                SELECT branch, name, version, content, commit_hash, created_at, metadata
                FROM prompts
                WHERE branch = ? AND name = ? AND version = ?
            """
            result = self.session.execute(query, [branch, name, version]).one()
        else:
            # Get latest version
            query = """
                SELECT branch, name, version, content, commit_hash, created_at, metadata
                FROM prompts
                WHERE branch = ? AND name = ?
                LIMIT 1
            """
            result = self.session.execute(query, [branch, name]).one()

        if not result:
            return None

        return {
            'name': result.name,
            'content': result.content,
            'branch': result.branch,
            'version': result.version,
            'commit_hash': result.commit_hash,
            'created_at': result.created_at.isoformat(),
            'metadata': json.loads(result.metadata) if result.metadata else {},
        }

    def list_prompts(self, branch: str = "main") -> List[Dict[str, Any]]:
        """
        List all prompts on a branch

        Tutorial Note:
        -------------
        Be careful with list operations on large datasets.
        Consider:
        - Pagination for large result sets
        - Efficient queries (avoid full table scans)
        - Returning only necessary fields
        """
        # Get latest version of each prompt on the branch
        # Note: In Cassandra, this requires iterating over partitions
        query = """
            SELECT DISTINCT branch, name
            FROM prompts
            WHERE branch = ?
            ALLOW FILTERING
        """
        unique_prompts = self.session.execute(query, [branch])

        prompts = []
        for row in unique_prompts:
            # Get latest version for each prompt
            latest_query = """
                SELECT name, version, commit_hash, created_at
                FROM prompts
                WHERE branch = ? AND name = ?
                LIMIT 1
            """
            latest = self.session.execute(latest_query, [branch, row.name]).one()
            if latest:
                prompts.append({
                    'name': latest.name,
                    'branch': branch,
                    'version': latest.version,
                    'commit_hash': latest.commit_hash,
                    'created_at': latest.created_at.isoformat(),
                })

        return prompts

    def create_branch(self, branch_name: str, from_branch: str = "main") -> None:
        """Create a new branch"""
        # Get head commit from source branch
        query = "SELECT head_commit FROM branches WHERE name = ?"
        result = self.session.execute(query, [from_branch]).one()

        if not result:
            raise ValueError(f"Source branch '{from_branch}' not found")

        # Create new branch
        insert_query = """
            INSERT INTO branches (name, head_commit, created_at)
            VALUES (?, ?, ?)
        """
        self.session.execute(
            insert_query,
            [branch_name, result.head_commit, datetime.now()]
        )

    def get_current_branch(self) -> str:
        """Get the current branch name"""
        return self.current_branch

    def set_current_branch(self, branch_name: str) -> None:
        """Set the current branch"""
        # Verify branch exists
        query = "SELECT name FROM branches WHERE name = ?"
        result = self.session.execute(query, [branch_name]).one()

        if not result:
            raise ValueError(f"Branch '{branch_name}' not found")

        self.current_branch = branch_name

    def get_commit_history(
        self,
        name: Optional[str] = None,
        branch: str = "main",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get commit history"""
        if name:
            # Get history for specific prompt
            query = """
                SELECT name, version, commit_hash, created_at
                FROM prompts
                WHERE branch = ? AND name = ?
                LIMIT ?
            """
            results = self.session.execute(query, [branch, name, limit])
        else:
            # Get recent commits across all prompts
            # Note: This is less efficient in Cassandra
            query = """
                SELECT name, version, commit_hash, created_at
                FROM prompts
                WHERE branch = ?
                ALLOW FILTERING
            """
            results = self.session.execute(query, [branch])

        commits = []
        for row in results:
            commits.append({
                'commit_hash': row.commit_hash,
                'name': row.name,
                'version': row.version,
                'branch': branch,
                'created_at': row.created_at.isoformat(),
            })

        # Sort by timestamp and limit
        commits.sort(key=lambda x: x['created_at'], reverse=True)
        return commits[:limit]

    def save_evaluation(self, eval_data: Dict[str, Any]) -> None:
        """Save evaluation results"""
        insert_query = """
            INSERT INTO evaluations (
                prompt_name, commit_hash, eval_id, test_case,
                expected, actual, score, metrics, evaluator_name, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.session.execute(
            insert_query,
            [
                eval_data['prompt_name'],
                eval_data['commit_hash'],
                uuid.uuid4(),
                eval_data['test_case'],
                eval_data.get('expected'),
                eval_data.get('actual'),
                eval_data.get('score'),
                json.dumps(eval_data.get('metrics', {})),
                eval_data.get('evaluator_name'),
                datetime.now(),
            ]
        )

    def save_chain(self, chain_data: Dict[str, Any]) -> None:
        """Save a prompt chain"""
        insert_query = """
            INSERT INTO chains (name, steps, description, created_at)
            VALUES (?, ?, ?, ?)
        """
        self.session.execute(
            insert_query,
            [
                chain_data['name'],
                json.dumps(chain_data['steps']),
                chain_data.get('description'),
                datetime.now(),
            ]
        )

    def get_chain(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a prompt chain"""
        query = "SELECT name, steps, description, created_at FROM chains WHERE name = ?"
        result = self.session.execute(query, [name]).one()

        if not result:
            return None

        return {
            'name': result.name,
            'steps': json.loads(result.steps),
            'description': result.description,
            'created_at': result.created_at.isoformat(),
        }

    def close(self) -> None:
        """
        Close database connection

        Tutorial Note:
        -------------
        Always clean up resources:
        - Close database connections
        - Release connection pools
        - Flush any pending writes
        - Clean up temporary resources
        """
        if self.cluster:
            self.cluster.shutdown()

    # Custom methods specific to this backend

    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics

        Tutorial Note:
        -------------
        Custom methods like this are optional but recommended:
        - Monitoring and observability
        - Performance metrics
        - Health checks
        - Administrative operations
        """
        stats = {
            'keyspace': self.keyspace,
            'consistency_level': self.consistency_level,
            'replication_factor': self.replication_factor,
        }

        # Count prompts
        try:
            result = self.session.execute("SELECT COUNT(*) as count FROM prompts").one()
            stats['total_prompts'] = result.count
        except Exception:
            stats['total_prompts'] = 0

        # Count branches
        try:
            result = self.session.execute("SELECT COUNT(*) as count FROM branches").one()
            stats['total_branches'] = result.count
        except Exception:
            stats['total_branches'] = 0

        return stats


# ============================================================================
# TESTING YOUR CUSTOM BACKEND
# ============================================================================

def test_custom_backend():
    """
    Example test function for your custom backend

    Tutorial Note:
    -------------
    Create comprehensive tests to verify:
    1. Connection and initialization
    2. CRUD operations (Create, Read, Update, Delete)
    3. Branching and versioning
    4. Error handling
    5. Performance under load
    6. Data integrity

    Run tests with:
        python -m pytest custom_storage/custom_db.py::test_custom_backend
    """
    # This is a placeholder - real tests would require a running Cassandra instance
    print("Tutorial: How to test your custom backend")
    print("=" * 60)
    print("1. Start your database (e.g., docker run -d cassandra)")
    print("2. Create test fixtures")
    print("3. Test each method individually")
    print("4. Test error conditions")
    print("5. Test concurrent access")
    print("6. Test data integrity")
    print("7. Clean up test data")


# ============================================================================
# QUICK START GUIDE
# ============================================================================

"""
Quick Start: Using Your Custom Backend
======================================

1. Install Dependencies:
   pip install cassandra-driver

2. Start Cassandra:
   docker run -d --name cassandra -p 9042:9042 cassandra:latest

3. Use the Backend:

from promptly.plugins.storage.custom_storage import CustomDBStorage

# Initialize
storage = CustomDBStorage(
    contact_points=['127.0.0.1'],
    keyspace='promptly_demo'
)
storage.init_storage('cassandra://localhost:9042/promptly_demo')

# Save a prompt
commit_hash = storage.save_prompt({
    'name': 'my_prompt',
    'content': 'Summarize this text: {text}',
    'branch': 'main',
    'metadata': {'author': 'john'}
})

# Retrieve the prompt
prompt = storage.get_prompt('my_prompt', branch='main')
print(f"Content: {prompt['content']}")

# List all prompts
prompts = storage.list_prompts(branch='main')
for p in prompts:
    print(f"- {p['name']} (v{p['version']})")

# Get statistics
stats = storage.get_storage_statistics()
print(f"Total prompts: {stats['total_prompts']}")

# Close connection
storage.close()


Migration from Other Backends
=============================

To migrate from SQLite/JSON/PostgreSQL to your custom backend:

from promptly.plugins.storage.migrate import migrate_storage

stats = migrate_storage(
    from_backend='sqlite',
    from_path='./promptly.db',
    to_backend='custom_db',
    to_path='cassandra://localhost:9042/promptly'
)

print(f"Migrated {stats['prompts_migrated']} prompts")


Common Customization Patterns
=============================

1. Add Caching Layer:
   - Use Redis for frequently accessed prompts
   - Implement cache invalidation on writes
   - Add cache warming on startup

2. Add Full-Text Search:
   - Integrate Elasticsearch or Solr
   - Index prompt content and metadata
   - Provide advanced search queries

3. Add Versioning Strategies:
   - Implement copy-on-write
   - Add deduplication for identical content
   - Compress old versions

4. Add Access Control:
   - User authentication
   - Role-based permissions
   - Audit logging

5. Add Monitoring:
   - Prometheus metrics
   - Health check endpoints
   - Performance profiling


Troubleshooting
==============

Common Issues:

1. Connection Errors:
   - Verify database is running
   - Check network connectivity
   - Validate credentials

2. Schema Errors:
   - Ensure tables are created
   - Check data types match
   - Verify indexes exist

3. Performance Issues:
   - Add indexes for common queries
   - Use connection pooling
   - Implement caching
   - Optimize query patterns

4. Data Integrity:
   - Use transactions where possible
   - Add foreign key constraints
   - Implement validation

For more help:
- Read the StorageBackend protocol in plugins/base.py
- Check existing backends for examples
- Review the migration tool for testing
"""


if __name__ == '__main__':
    # Run tutorial test
    test_custom_backend()

    # Print documentation
    print(__doc__)
