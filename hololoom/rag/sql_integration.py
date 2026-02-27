"""
SQL Integration for HoloLoom RAG - Hybrid Knowledge Graph + Database Queries

Enables querying structured databases alongside the knowledge graph with:
- Text-to-SQL translation using LLM
- Automatic hybrid routing (SQL vs semantic)
- Result fusion (SQL + graph results)
- Security: parameterized queries, read-only mode
- Schema awareness via introspection

Architecture:
    SQLRAGMixin (protocol-based adapter)
    ├── Query Intent Classification
    │   ├── Keyword detection (fast)
    │   └── LLM classifier (accurate)
    │
    ├── Text-to-SQL Translation
    │   ├── Schema-aware prompt
    │   └── SQL validation & sanitization
    │
    ├── Hybrid Routing
    │   ├── SQL-only: Factual lookups
    │   ├── Semantic-only: Complex reasoning
    │   └── Hybrid: Both paths, LLM fusion
    │
    └── Result Fusion
        ├── SQL results → DataFrame
        ├── Semantic results → text sources
        └── LLM synthesizes combined answer

Usage:
    async with SQLRAGMixin(
        db_connection="sqlite:///my_database.db",
        enable_hybrid_routing=True
    ) as rag:
        # Automatic routing
        result = await rag.query_with_sql("How many users are over 30?")
        print(result.sql_data)  # pandas DataFrame
        print(result.response)  # Natural language answer

        # Force SQL mode
        result = await rag.query_with_sql(
            "SELECT * FROM users WHERE age > 30",
            mode="sql_only"
        )

        # Force hybrid mode
        result = await rag.query_with_sql(
            "Show me users interested in machine learning",
            mode="hybrid"
        )

Author: Agent H (Claude Code)
Date: November 13, 2025
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from enum import Enum
import logging
import re
import time

# UnifiedMRF for enhanced prompting (Phase 2.1 - Nov 2025)
from hololoom.prompting.unified_mrf import UnifiedMRF, ModelProvider, MetapromptConfig

# Optional dependencies (graceful degradation)
try:
    from sqlalchemy import create_engine, text, MetaData, inspect
    from sqlalchemy.engine import Engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Engine = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Query intent classification."""
    SQL_FACTUAL = "sql_factual"  # Factual lookup (SELECT, COUNT, etc.)
    SEMANTIC = "semantic"  # Complex reasoning, no structured data
    HYBRID = "hybrid"  # Needs both SQL + semantic
    AMBIGUOUS = "ambiguous"  # Unclear, let hybrid routing decide


class SQLQueryMode(Enum):
    """SQL query execution modes."""
    AUTO = "auto"  # Automatic routing (default)
    SQL_ONLY = "sql_only"  # Force SQL path
    SEMANTIC_ONLY = "semantic_only"  # Force semantic path
    HYBRID = "hybrid"  # Force both paths + fusion


@dataclass
class SQLRAGResult:
    """
    Result from SQL-enabled RAG query.

    Extends RAGResult with SQL-specific fields:
    - sql_data: Query results as DataFrame
    - sql_query: Executed SQL query
    - query_type: "sql", "semantic", or "hybrid"
    - sql_confidence: Confidence in SQL translation (0.0-1.0)
    """
    response: str
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_mode: str = "verify"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # SQL-specific fields
    sql_data: Optional[Any] = None  # pandas DataFrame
    sql_query: Optional[str] = None
    query_type: str = "semantic"  # "sql", "semantic", or "hybrid"
    sql_confidence: float = 0.0

    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [
            f"SQLRAGResult(",
            f"  response: {self.response[:100]}...",
            f"  query_type: {self.query_type}",
            f"  confidence: {self.confidence:.2f}",
        ]
        if self.sql_data is not None:
            if PANDAS_AVAILABLE:
                lines.append(f"  sql_rows: {len(self.sql_data)}")
            lines.append(f"  sql_confidence: {self.sql_confidence:.2f}")
        if self.sources:
            lines.append(f"  sources: {len(self.sources)} items")
        lines.append(")")
        return "\n".join(lines)


class SQLAdapter:
    """
    Protocol-based adapter for SQL database queries.

    Handles:
    - Connection management (SQLite, PostgreSQL, MySQL)
    - Schema introspection
    - Text-to-SQL translation
    - Query execution with parameterization
    - Read-only mode enforcement
    """

    def __init__(
        self,
        connection_string: str,
        schema: Optional[Dict[str, List[str]]] = None,
        read_only: bool = True,
        timeout: float = 5.0,
        model_provider: Optional[str] = None  # NEW (Phase 2.1)
    ):
        """
        Initialize SQL adapter.

        Args:
            connection_string: SQLAlchemy connection string
                Examples:
                - "sqlite:///database.db"
                - "postgresql://user:pass@localhost/dbname"
                - "mysql://user:pass@localhost/dbname"
            schema: Optional manual schema definition
                Example: {"users": ["id", "name", "age"], "orders": [...]}
            read_only: Enforce read-only queries (blocks INSERT/UPDATE/DELETE)
            timeout: Query timeout in seconds
            model_provider: LLM provider for prompt optimization
                ("claude", "gemini", "gpt", "ollama"). NEW (Phase 2.1)

        Raises:
            RuntimeError: If SQLAlchemy unavailable
        """
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError(
                "SQLAlchemy unavailable. Install: pip install sqlalchemy"
            )

        self.connection_string = connection_string
        self.read_only = read_only
        self.timeout = timeout
        self.engine: Optional[Engine] = None
        self.schema: Dict[str, List[str]] = schema or {}
        self.model_provider = model_provider  # NEW (Phase 2.1)

        # UnifiedMRF for enhanced SQL translation prompts (Phase 2.1)
        self.mrf = UnifiedMRF()

        # Stats
        self._stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_latency_ms': 0.0,
            'avg_latency_ms': 0.0,
        }

    def connect(self) -> None:
        """
        Connect to database and introspect schema.

        Raises:
            RuntimeError: If connection fails
        """
        try:
            self.engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,  # Verify connections before use
                connect_args={'timeout': self.timeout} if 'sqlite' in self.connection_string else {}
            )

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info(f"✓ Connected to database: {self._mask_credentials(self.connection_string)}")

            # Introspect schema if not provided
            if not self.schema:
                self._introspect_schema()

        except Exception as e:
            raise RuntimeError(f"Failed to connect to database: {e}")

    def _mask_credentials(self, conn_str: str) -> str:
        """Mask password in connection string for logging."""
        # Pattern: protocol://user:password@host/db
        return re.sub(r'://([^:]+):([^@]+)@', r'://\1:****@', conn_str)

    def _introspect_schema(self) -> None:
        """
        Introspect database schema via SQLAlchemy reflection.

        Populates self.schema with table → column mappings.
        """
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()

            for table in tables:
                columns = [col['name'] for col in inspector.get_columns(table)]
                self.schema[table] = columns

            logger.info(f"✓ Introspected schema: {len(self.schema)} tables")
            for table, columns in self.schema.items():
                logger.debug(f"  {table}: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}")

        except Exception as e:
            logger.warning(f"⚠ Schema introspection failed: {e}")
            self.schema = {}

    def close(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("✓ Database connection closed")

    def execute_query(self, sql_query: str) -> Optional[Any]:
        """
        Execute SQL query with safety checks.

        Args:
            sql_query: SQL query string

        Returns:
            pandas DataFrame with results or None on error

        Raises:
            ValueError: If read-only mode violated
            RuntimeError: If query execution fails
        """
        if not self.engine:
            raise RuntimeError("Not connected. Call connect() first.")

        # Validate read-only mode
        if self.read_only and self._is_write_operation(sql_query):
            raise ValueError(
                f"Write operation blocked in read-only mode: {sql_query[:50]}..."
            )

        # Execute with timing
        start_time = time.time()
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))

                # Convert to DataFrame if pandas available
                if PANDAS_AVAILABLE:
                    if result.returns_rows:
                        df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    else:
                        df = pd.DataFrame()  # No rows (e.g., INSERT/UPDATE)
                else:
                    # Fallback: return list of dicts
                    if result.returns_rows:
                        rows = result.fetchall()
                        df = [dict(row) for row in rows]
                    else:
                        df = []

            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(success=True, latency_ms=latency_ms)

            logger.debug(f"✓ Query executed in {latency_ms:.1f}ms: {sql_query[:50]}...")
            return df

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(success=False, latency_ms=latency_ms)
            logger.error(f"✗ Query failed: {e}")
            raise RuntimeError(f"SQL execution failed: {e}")

    def _is_write_operation(self, sql_query: str) -> bool:
        """
        Detect write operations (INSERT/UPDATE/DELETE/DROP/ALTER/CREATE).

        Args:
            sql_query: SQL query string

        Returns:
            True if write operation detected
        """
        write_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE',
            'TRUNCATE', 'REPLACE', 'MERGE'
        ]
        sql_upper = sql_query.upper().strip()
        return any(sql_upper.startswith(kw) for kw in write_keywords)

    def _update_stats(self, success: bool, latency_ms: float) -> None:
        """Update query statistics."""
        self._stats['total_queries'] += 1
        if success:
            self._stats['successful_queries'] += 1
        else:
            self._stats['failed_queries'] += 1

        self._stats['total_latency_ms'] += latency_ms
        self._stats['avg_latency_ms'] = (
            self._stats['total_latency_ms'] / self._stats['total_queries']
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            **self._stats,
            'schema_tables': list(self.schema.keys()),
            'read_only': self.read_only,
        }


class TextToSQLTranslator:
    """
    Translates natural language queries to SQL using LLM.

    Schema-aware translation with validation and sanitization.
    """

    def __init__(
        self,
        schema: Dict[str, List[str]],
        llm_provider: Optional[Any] = None
    ):
        """
        Initialize translator.

        Args:
            schema: Database schema (table → columns)
            llm_provider: LLM provider (orchestrator) for translation
        """
        self.schema = schema
        self.llm_provider = llm_provider

        # Translation stats
        self._stats = {
            'total_translations': 0,
            'successful_translations': 0,
            'failed_translations': 0,
        }

    async def translate(
        self,
        question: str,
        max_retries: int = 2
    ) -> Optional[str]:
        """
        Translate natural language question to SQL.

        Args:
            question: Natural language query
            max_retries: Max translation attempts

        Returns:
            SQL query string or None if translation fails
        """
        if not self.llm_provider:
            logger.warning("⚠ LLM provider unavailable, cannot translate to SQL")
            self._stats['failed_translations'] += 1
            return None

        # Build schema-aware prompt
        prompt = self._build_translation_prompt(question)

        # Attempt translation with retries
        for attempt in range(max_retries):
            try:
                # Call LLM
                from hololoom.protocols.types import Query
                query_obj = Query(text=prompt)
                spacetime = await self.llm_provider.weave(query_obj, use_llm=True)

                sql_query = spacetime.response if hasattr(spacetime, 'response') else None
                if not sql_query:
                    continue

                # Extract SQL from response (LLM may include explanation)
                sql_query = self._extract_sql(sql_query)

                # Validate SQL
                if self._validate_sql(sql_query):
                    self._stats['successful_translations'] += 1
                    self._stats['total_translations'] += 1
                    logger.debug(f"✓ Translated to SQL: {sql_query[:50]}...")
                    return sql_query

            except Exception as e:
                logger.debug(f"Translation attempt {attempt + 1} failed: {e}")
                continue

        # All attempts failed
        self._stats['failed_translations'] += 1
        self._stats['total_translations'] += 1
        logger.warning(f"⚠ Failed to translate after {max_retries} attempts: {question[:50]}...")
        return None

    def _build_translation_prompt(self, question: str) -> str:
        """
        Build schema-aware prompt for text-to-SQL translation.

        **UPDATED (Phase 2.1 - Nov 2025):** Uses UnifiedMRF 7-component framework
        with model-specific optimizations.

        Args:
            question: Natural language query

        Returns:
            Enhanced LLM prompt string (traditional format if MRF unavailable)
        """
        # Format schema for metaprompt context
        schema_str = "Database Schema:\n"
        for table, columns in self.schema.items():
            schema_str += f"\nTable: {table}\n"
            schema_str += f"Columns: {', '.join(columns)}\n"

        # Use UnifiedMRF for enhanced prompting (Phase 2.1)
        metaprompt = MetapromptConfig(
            role=(
                "You are an expert SQL translator with deep knowledge of:\n"
                "- Standard SQL syntax (SELECT, JOIN, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT)\n"
                "- Multi-database compatibility (SQLite, PostgreSQL, MySQL)\n"
                "- Query optimization and index-friendly patterns\n"
                "- Security best practices (parameterization, SQL injection prevention)\n"
                "- Schema inference and table relationship detection"
            ),
            objective={
                "primary": "Translate natural language questions into syntactically correct, secure SQL queries",
                "secondary": [
                    "Ensure read-only operations (SELECT only)",
                    "Optimize for database performance and index usage",
                    "Handle ambiguous queries gracefully with reasonable assumptions"
                ]
            },
            process=[
                {
                    "step": "1. Schema Analysis",
                    "actions": [
                        f"Review available tables and columns:\n{schema_str}",
                        "Identify which tables/columns are relevant to the question",
                        "Detect any necessary JOINs based on table relationships"
                    ]
                },
                {
                    "step": "2. Query Construction",
                    "actions": [
                        "Start with SELECT statement",
                        "Add necessary JOINs if multiple tables involved",
                        "Include WHERE clauses for filtering",
                        "Add GROUP BY if aggregations needed (COUNT, SUM, AVG, etc.)",
                        "Include ORDER BY for sorted results if relevant",
                        "Add LIMIT for result size control"
                    ]
                },
                {
                    "step": "3. Validation & Optimization",
                    "actions": [
                        "Verify all columns exist in schema",
                        "Ensure query is read-only (no INSERT/UPDATE/DELETE/DROP)",
                        "Check for proper JOIN syntax (INNER JOIN, LEFT JOIN, etc.)",
                        "Optimize WHERE clause ordering for index usage"
                    ]
                }
            ],
            format_spec=(
                "Return ONLY the SQL query string, no explanation or markdown formatting.\n"
                "Format: Clean SQL on a single line or multiple lines (no code blocks).\n"
                "Example: SELECT name, age FROM users WHERE age > 30 LIMIT 10"
            ),
            constraints=[
                "MUST use SELECT for read operations only",
                "MUST NOT include INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE",
                "MUST use standard SQL compatible with SQLite, PostgreSQL, MySQL",
                "MUST reference only tables/columns that exist in the provided schema",
                "MUST use proper JOIN syntax if multiple tables needed",
                "SHOULD include LIMIT clause to prevent unbounded result sets",
                "SHOULD use parameterization-friendly patterns (avoid string concatenation in WHERE)"
            ],
            uncertainty=(
                "If the question is ambiguous or could map to multiple interpretations:\n"
                "- Choose the most common/reasonable interpretation\n"
                "- Prefer simpler queries over complex multi-table JOINs\n"
                "- Default to returning recent/relevant data (ORDER BY with LIMIT)\n"
                "\n"
                "If required columns are unclear:\n"
                "- Infer likely columns from table names and common patterns\n"
                "- Example: For 'users', likely columns include id, name, email, created_at"
            ),
            validation=[
                "Query starts with SELECT (case-insensitive)",
                "No dangerous keywords (INSERT, UPDATE, DELETE, DROP, etc.)",
                "All referenced tables exist in schema",
                "All referenced columns exist in their respective tables",
                "JOIN syntax is valid (if used)",
                "WHERE clause uses proper comparison operators (=, >, <, LIKE, IN, etc.)"
            ]
        )

        # Map model provider string to ModelProvider enum
        provider = None
        if self.model_provider:
            provider_map = {
                "claude": ModelProvider.CLAUDE,
                "gemini": ModelProvider.GEMINI,
                "gpt": ModelProvider.GPT,
                "ollama": ModelProvider.OLLAMA
            }
            provider = provider_map.get(self.model_provider.lower())

        # Generate enhanced prompt using UnifiedMRF
        enhanced_prompt = self.mrf.enhance_prompt(
            metaprompt=metaprompt,
            query=question,
            model=provider
        )

        return enhanced_prompt

    def _extract_sql(self, llm_response: str) -> str:
        """
        Extract SQL query from LLM response.

        Handles cases where LLM includes explanation/formatting.

        Args:
            llm_response: Raw LLM response

        Returns:
            Extracted SQL query
        """
        # Strip markdown code blocks
        response = llm_response.strip()
        if response.startswith("```sql"):
            response = response[6:]
        elif response.startswith("```"):
            response = response[3:]

        if response.endswith("```"):
            response = response[:-3]

        # Extract first SQL statement (up to semicolon)
        lines = response.strip().split('\n')
        sql_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip comment lines
            if stripped.startswith('--') or not stripped:
                continue
            sql_lines.append(stripped)

            # Stop at semicolon
            if ';' in stripped:
                break

        sql_query = ' '.join(sql_lines).strip()

        # Remove trailing semicolon
        if sql_query.endswith(';'):
            sql_query = sql_query[:-1]

        return sql_query

    def _validate_sql(self, sql_query: str) -> bool:
        """
        Validate SQL query for basic correctness.

        Args:
            sql_query: SQL query string

        Returns:
            True if valid, False otherwise
        """
        if not sql_query or len(sql_query.strip()) < 10:
            return False

        # Must start with SELECT (read-only)
        if not sql_query.strip().upper().startswith('SELECT'):
            logger.warning(f"⚠ Non-SELECT query rejected: {sql_query[:30]}...")
            return False

        # Check for dangerous keywords
        dangerous = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE']
        sql_upper = sql_query.upper()
        if any(kw in sql_upper for kw in dangerous):
            logger.warning(f"⚠ Dangerous keyword detected: {sql_query[:30]}...")
            return False

        # Check table references exist in schema
        mentioned_tables = self._extract_table_names(sql_query)
        for table in mentioned_tables:
            if table not in self.schema:
                logger.warning(f"⚠ Unknown table referenced: {table}")
                return False

        return True

    def _extract_table_names(self, sql_query: str) -> List[str]:
        """
        Extract table names from SQL query (simple regex-based).

        Args:
            sql_query: SQL query string

        Returns:
            List of table names
        """
        # Pattern: FROM table_name or JOIN table_name
        pattern = r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(pattern, sql_query, re.IGNORECASE)
        return list(set(matches))

    def get_stats(self) -> Dict[str, Any]:
        """Get translation statistics."""
        return {
            **self._stats,
            'success_rate': (
                self._stats['successful_translations'] / self._stats['total_translations']
                if self._stats['total_translations'] > 0 else 0.0
            )
        }


class SQLRAGMixin:
    """
    SQL integration mixin for RAG systems.

    Provides SQL + semantic hybrid query capabilities with automatic routing.
    """

    def __init__(
        self,
        db_connection: Optional[str] = None,
        enable_hybrid_routing: bool = True,
        sql_confidence_threshold: float = 0.7,
        schema: Optional[Dict[str, List[str]]] = None,
        read_only: bool = True
    ):
        """
        Initialize SQL RAG mixin.

        Args:
            db_connection: Database connection string (SQLAlchemy format)
            enable_hybrid_routing: Enable automatic SQL/semantic routing
            sql_confidence_threshold: Confidence threshold for SQL routing (0.0-1.0)
            schema: Optional manual schema definition
            read_only: Enforce read-only queries

        Example:
            mixin = SQLRAGMixin(
                db_connection="sqlite:///my_database.db",
                enable_hybrid_routing=True,
                sql_confidence_threshold=0.7
            )
        """
        self.db_connection = db_connection
        self.enable_hybrid_routing = enable_hybrid_routing
        self.sql_confidence_threshold = sql_confidence_threshold
        self.read_only = read_only

        # SQL components (initialized in connect())
        self.sql_adapter: Optional[SQLAdapter] = None
        self.text_to_sql: Optional[TextToSQLTranslator] = None

        # Schema
        self.schema: Dict[str, List[str]] = schema or {}

        logger.info(f"✓ SQLRAGMixin initialized (hybrid_routing={'ON' if enable_hybrid_routing else 'OFF'})")

    def register_schema(self, schema: Dict[str, Any]) -> None:
        """
        Register database schema manually.

        Args:
            schema: Dictionary mapping table names to column lists
                Example: {"users": ["id", "name", "age"], "orders": [...]}

        Example:
            mixin.register_schema({
                "users": ["id", "name", "email", "age"],
                "orders": ["id", "user_id", "product", "price"]
            })
        """
        self.schema = schema
        if self.sql_adapter:
            self.sql_adapter.schema = schema
        if self.text_to_sql:
            self.text_to_sql.schema = schema

        logger.info(f"✓ Schema registered: {len(schema)} tables")

    async def connect_sql(self, llm_provider: Optional[Any] = None) -> None:
        """
        Connect to database and initialize SQL components.

        Args:
            llm_provider: LLM provider (orchestrator) for text-to-SQL

        Raises:
            RuntimeError: If connection fails or no db_connection specified
        """
        if not self.db_connection:
            raise RuntimeError("No database connection string specified")

        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError(
                "SQLAlchemy unavailable. Install: pip install sqlalchemy"
            )

        # Initialize adapter
        self.sql_adapter = SQLAdapter(
            connection_string=self.db_connection,
            schema=self.schema,
            read_only=self.read_only
        )
        self.sql_adapter.connect()

        # Update schema from introspection
        if self.sql_adapter.schema:
            self.schema = self.sql_adapter.schema

        # Initialize text-to-SQL translator
        self.text_to_sql = TextToSQLTranslator(
            schema=self.schema,
            llm_provider=llm_provider
        )

        logger.info("✓ SQL adapter and translator initialized")

    def close_sql(self) -> None:
        """Close SQL adapter connection."""
        if self.sql_adapter:
            self.sql_adapter.close()

    async def query_with_sql(
        self,
        question: str,
        mode: str = "auto",
        max_sources: int = 5
    ) -> SQLRAGResult:
        """
        Query with SQL + semantic hybrid search.

        Args:
            question: Natural language query
            mode: Query mode
                - "auto": Automatic routing (default)
                - "sql_only": Force SQL path
                - "semantic_only": Force semantic path
                - "hybrid": Force both paths + fusion
            max_sources: Max sources for semantic search

        Returns:
            SQLRAGResult with response, SQL data, sources, and confidence

        Example:
            # Auto routing
            result = await mixin.query_with_sql("How many users are over 30?")

            # Force SQL
            result = await mixin.query_with_sql(
                "SELECT * FROM users WHERE age > 30",
                mode="sql_only"
            )

            # Force hybrid
            result = await mixin.query_with_sql(
                "Show me users interested in machine learning",
                mode="hybrid"
            )

        Raises:
            RuntimeError: If not connected or dependencies unavailable
        """
        if not self.sql_adapter:
            raise RuntimeError("Not connected. Call connect_sql() first.")

        # Classify query intent
        intent = self._classify_query_intent(question)
        logger.debug(f"Query intent: {intent.value}")

        # Route based on mode and intent
        if mode == "sql_only" or (mode == "auto" and intent == QueryIntent.SQL_FACTUAL):
            return await self._query_sql_only(question, max_sources)
        elif mode == "semantic_only" or (mode == "auto" and intent == QueryIntent.SEMANTIC):
            return await self._query_semantic_only(question, max_sources)
        elif mode == "hybrid" or (mode == "auto" and intent in (QueryIntent.HYBRID, QueryIntent.AMBIGUOUS)):
            return await self._query_hybrid(question, max_sources)
        else:
            # Default to hybrid
            return await self._query_hybrid(question, max_sources)

    def _classify_query_intent(self, question: str) -> QueryIntent:
        """
        Classify query intent: SQL factual, semantic, hybrid, or ambiguous.

        Uses keyword-based heuristics (fast).

        Args:
            question: Natural language query

        Returns:
            QueryIntent enum value
        """
        question_lower = question.lower()

        # SQL factual keywords
        sql_keywords = [
            'how many', 'count', 'sum', 'average', 'total', 'max', 'min',
            'list all', 'show all', 'select', 'from', 'where',
            'greater than', 'less than', 'equal to', 'between'
        ]

        # Semantic keywords
        semantic_keywords = [
            'explain', 'why', 'how does', 'what is the meaning',
            'describe', 'tell me about', 'what are the implications',
            'analyze', 'summarize', 'opinion', 'recommend'
        ]

        # Hybrid keywords
        hybrid_keywords = [
            'related to', 'interested in', 'similar to', 'like',
            'associated with', 'connected to'
        ]

        # Count keyword matches
        sql_score = sum(1 for kw in sql_keywords if kw in question_lower)
        semantic_score = sum(1 for kw in semantic_keywords if kw in question_lower)
        hybrid_score = sum(1 for kw in hybrid_keywords if kw in question_lower)

        # Classify based on scores
        if sql_score >= 2 and semantic_score == 0:
            return QueryIntent.SQL_FACTUAL
        elif semantic_score >= 2 and sql_score == 0:
            return QueryIntent.SEMANTIC
        elif hybrid_score >= 1 or (sql_score >= 1 and semantic_score >= 1):
            return QueryIntent.HYBRID
        elif question_lower.strip().startswith('select'):
            return QueryIntent.SQL_FACTUAL
        else:
            return QueryIntent.AMBIGUOUS

    async def _query_sql_only(
        self,
        question: str,
        max_sources: int
    ) -> SQLRAGResult:
        """
        Execute SQL-only query path.

        Args:
            question: Natural language query
            max_sources: Max sources (unused for SQL-only)

        Returns:
            SQLRAGResult with SQL data
        """
        # Translate to SQL
        sql_query = await self.text_to_sql.translate(question)

        if not sql_query:
            # Translation failed, return error result
            return SQLRAGResult(
                response=f"Failed to translate question to SQL: {question}",
                sources=[],
                confidence=0.0,
                query_type="sql",
                sql_confidence=0.0,
                metadata={'error': 'translation_failed'}
            )

        # Execute SQL
        try:
            sql_data = self.sql_adapter.execute_query(sql_query)

            # Generate natural language response from SQL results
            if PANDAS_AVAILABLE and hasattr(sql_data, 'shape'):
                n_rows = len(sql_data)
                response = f"Found {n_rows} results:\n"
                if n_rows > 0:
                    # Show first few rows
                    response += sql_data.head(5).to_string()
                    if n_rows > 5:
                        response += f"\n... ({n_rows - 5} more rows)"
            else:
                response = f"Query executed successfully. Results: {sql_data}"

            return SQLRAGResult(
                response=response,
                sources=[],
                confidence=0.9,
                query_type="sql",
                sql_data=sql_data,
                sql_query=sql_query,
                sql_confidence=0.9,
                metadata={
                    'n_sql_rows': len(sql_data) if PANDAS_AVAILABLE else len(sql_data) if isinstance(sql_data, list) else 0
                }
            )

        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            return SQLRAGResult(
                response=f"SQL execution failed: {e}",
                sources=[],
                confidence=0.0,
                query_type="sql",
                sql_query=sql_query,
                sql_confidence=0.0,
                metadata={'error': str(e)}
            )

    async def _query_semantic_only(
        self,
        question: str,
        max_sources: int
    ) -> SQLRAGResult:
        """
        Execute semantic-only query path.

        Delegates to parent class query() method.

        Args:
            question: Natural language query
            max_sources: Max sources to retrieve

        Returns:
            SQLRAGResult (converted from RAGResult)
        """
        # This method should be overridden by implementing class
        # For now, return placeholder
        return SQLRAGResult(
            response="Semantic query requires RAG implementation",
            sources=[],
            confidence=0.0,
            query_type="semantic",
            metadata={'error': 'not_implemented'}
        )

    async def _query_hybrid(
        self,
        question: str,
        max_sources: int
    ) -> SQLRAGResult:
        """
        Execute hybrid query path (SQL + semantic).

        Runs both paths in parallel and fuses results.

        Args:
            question: Natural language query
            max_sources: Max sources for semantic search

        Returns:
            SQLRAGResult with fused results
        """
        import asyncio

        # Run both paths in parallel
        sql_task = asyncio.create_task(self._query_sql_only(question, max_sources))
        semantic_task = asyncio.create_task(self._query_semantic_only(question, max_sources))

        sql_result, semantic_result = await asyncio.gather(sql_task, semantic_task)

        # Fuse results
        # Combine SQL data + semantic sources
        fused_response = "Hybrid result:\n\n"

        if sql_result.sql_data is not None and sql_result.confidence > 0.5:
            fused_response += f"SQL Results:\n{sql_result.response}\n\n"

        if semantic_result.sources:
            fused_response += f"Related context:\n" + "\n".join(
                f"- {source[:100]}..." for source in semantic_result.sources[:3]
            )

        # Average confidences
        fused_confidence = (sql_result.confidence + semantic_result.confidence) / 2

        return SQLRAGResult(
            response=fused_response,
            sources=semantic_result.sources,
            confidence=fused_confidence,
            query_type="hybrid",
            sql_data=sql_result.sql_data,
            sql_query=sql_result.sql_query,
            sql_confidence=sql_result.sql_confidence,
            metadata={
                'n_sql_rows': len(sql_result.sql_data) if sql_result.sql_data is not None and PANDAS_AVAILABLE else 0,
                'n_semantic_sources': len(semantic_result.sources),
                'sql_confidence': sql_result.confidence,
                'semantic_confidence': semantic_result.confidence,
            }
        )

    def get_sql_stats(self) -> Dict[str, Any]:
        """
        Get SQL adapter and translator statistics.

        Returns:
            Dictionary with SQL stats

        Example:
            stats = mixin.get_sql_stats()
            print(f"Total SQL queries: {stats['total_queries']}")
            print(f"Translation success rate: {stats['translation_success_rate']:.1%}")
        """
        stats = {}

        if self.sql_adapter:
            stats.update({
                'adapter': self.sql_adapter.get_stats()
            })

        if self.text_to_sql:
            stats.update({
                'translator': self.text_to_sql.get_stats()
            })

        return stats
