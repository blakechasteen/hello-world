#!/usr/bin/env python3
"""
SaaS Backend for HoloLoom API-Only SaaS

Extends SQLBackend pattern with SaaS-specific CRUD operations:
- Customer management (signup, profile)
- Subscription management (plans, billing)
- API key management (generation, validation, revocation)
- Usage tracking (daily aggregation, Stripe reporting)

Phase 1: API-only (BYO backend) - 100 hours
"""

import asyncio
import hashlib
import json
import logging
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# Optional PostgreSQL support (graceful degradation)
try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    logging.warning("asyncpg not available - falling back to SQLite")

# Optional bcrypt for password hashing
try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False
    logging.warning("bcrypt not available - using SHA256 for passwords (not recommended for production)")


logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SaaSConfig:
    """
    SaaS backend configuration

    Feature flags allow you to use only what you need:
    - Auth only: enable_billing=False, enable_usage_tracking=False
    - Auth + Usage: enable_billing=False, enable_usage_tracking=True
    - Full billing: enable_billing=True (requires Stripe)

    Use factory methods for common configurations:
    - SaaSConfig.auth_only() - Minimal auth (no billing, no usage tracking)
    - SaaSConfig.with_usage() - Auth + usage analytics (no billing)
    - SaaSConfig.with_billing() - Full Stripe integration
    """
    # PostgreSQL settings
    host: str = "localhost"
    port: int = 5432
    database: str = "hololoom_saas"
    user: str = "hololoom"
    password: str = "hololoom"

    # Connection pool settings
    min_pool_size: int = 2
    max_pool_size: int = 10
    pool_timeout: float = 30.0

    # SQLite fallback
    sqlite_path: str = "./data/hololoom_saas.db"
    fallback_to_sqlite: bool = True

    # Schema
    schema_path: str = "HoloLoom/saas/schema.sql"

    # API Key settings
    key_prefix: str = "holo"
    key_length: int = 32  # Secret key length

    # Feature flags (modular - use only what you need)
    enable_billing: bool = False  # Requires Stripe - disabled by default
    enable_usage_tracking: bool = True  # Track queries (no billing required)
    enable_audit_log: bool = True  # Log events for security

    # Stripe settings (only used if enable_billing=True)
    stripe_api_key: str | None = None
    stripe_webhook_secret: str | None = None

    @classmethod
    def auth_only(
        cls,
        sqlite_path: str = "./data/hololoom_saas.db",
        key_prefix: str = "holo",
        **kwargs
    ) -> "SaaSConfig":
        """
        Minimal configuration - authentication only.

        No usage tracking, no billing. Just customers and API keys.
        Perfect for self-hosted applications that don't need metering.
        """
        return cls(
            sqlite_path=sqlite_path,
            fallback_to_sqlite=True,
            key_prefix=key_prefix,
            enable_billing=False,
            enable_usage_tracking=False,
            enable_audit_log=True,
            **kwargs
        )

    @classmethod
    def with_usage(
        cls,
        sqlite_path: str = "./data/hololoom_saas.db",
        key_prefix: str = "holo",
        **kwargs
    ) -> "SaaSConfig":
        """
        Auth + usage tracking (no billing).

        Tracks queries per customer for analytics, without charging.
        Good for freemium apps or internal usage monitoring.
        """
        return cls(
            sqlite_path=sqlite_path,
            fallback_to_sqlite=True,
            key_prefix=key_prefix,
            enable_billing=False,
            enable_usage_tracking=True,
            enable_audit_log=True,
            **kwargs
        )

    @classmethod
    def with_billing(
        cls,
        stripe_api_key: str,
        stripe_webhook_secret: str | None = None,
        host: str = "localhost",
        port: int = 5432,
        database: str = "hololoom_saas",
        user: str = "hololoom",
        password: str = "hololoom",
        **kwargs
    ) -> "SaaSConfig":
        """
        Full configuration with Stripe billing.

        Includes auth, usage tracking, and payment processing.
        Requires Stripe API key. Recommended for production SaaS.
        """
        return cls(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            fallback_to_sqlite=False,  # Production should use PostgreSQL
            enable_billing=True,
            enable_usage_tracking=True,
            enable_audit_log=True,
            stripe_api_key=stripe_api_key,
            stripe_webhook_secret=stripe_webhook_secret,
            **kwargs
        )


@dataclass
class Customer:
    """Customer data model"""
    customer_id: str
    email: str
    name: str
    company: str | None = None
    stripe_customer_id: str | None = None
    status: str = "active"
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class Subscription:
    """Subscription data model"""
    subscription_id: str
    customer_id: str
    plan: str = "free"
    stripe_subscription_id: str | None = None
    stripe_price_id: str | None = None
    status: str = "active"
    current_period_start: datetime | None = None
    current_period_end: datetime | None = None
    queries_included: int = 0
    price_per_query_cents: float | None = None
    cancel_at_period_end: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class APIKey:
    """API Key data model"""
    key_id: str
    customer_id: str
    key_prefix: str
    name: str = "Default"
    status: str = "active"
    rate_limit_qps: float = 10.0
    created_at: datetime | None = None
    expires_at: datetime | None = None
    last_used: datetime | None = None
    request_count: int = 0
    # Note: key_hash is stored in DB but not exposed in this model


@dataclass
class UsageRecord:
    """Daily usage record"""
    id: str
    customer_id: str
    date: date
    queries_count: int = 0
    tokens_used: int = 0
    reported_to_stripe: bool = False
    stripe_usage_record_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class QueryResult:
    """SQL query result with metadata"""
    rows: list[dict[str, Any]]
    row_count: int
    latency_ms: float
    success: bool
    error: str | None = None
    backend: str = "postgresql"


# ============================================================================
# SaaS Backend
# ============================================================================

class SaaSBackend:
    """
    SaaS backend for customer, subscription, and usage management

    Supports PostgreSQL (production) and SQLite (development/fallback).

    Features are modular based on config:
    - Auth only: Just customers and API keys
    - Auth + Usage: Add usage tracking (no billing)
    - Full billing: Add Stripe integration

    Use SaaSConfig factory methods:
    - SaaSConfig.auth_only() - Minimal setup
    - SaaSConfig.with_usage() - Add analytics
    - SaaSConfig.with_billing(stripe_api_key) - Full SaaS
    """

    def __init__(self, config: SaaSConfig | None = None):
        self.config = config or SaaSConfig()
        self.pool: Any | None = None  # asyncpg pool or None
        self.sqlite_conn: sqlite3.Connection | None = None
        self.backend_type: str = "uninitialized"
        self._closed = False

    @property
    def billing_enabled(self) -> bool:
        """Check if billing features are enabled"""
        return self.config.enable_billing

    @property
    def usage_tracking_enabled(self) -> bool:
        """Check if usage tracking is enabled"""
        return self.config.enable_usage_tracking

    @property
    def audit_log_enabled(self) -> bool:
        """Check if audit logging is enabled"""
        return self.config.enable_audit_log

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def connect(self):
        """Connect to database (PostgreSQL or SQLite fallback)"""
        if HAS_ASYNCPG:
            try:
                await self._connect_postgresql()
                return
            except Exception as e:
                logger.warning(f"PostgreSQL connection failed: {e}")
                if not self.config.fallback_to_sqlite:
                    raise

        # Fallback to SQLite
        await self._connect_sqlite()

    async def _connect_postgresql(self):
        """Connect to PostgreSQL with connection pooling"""
        logger.info(f"Connecting to PostgreSQL at {self.config.host}:{self.config.port}")

        self.pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password,
            min_size=self.config.min_pool_size,
            max_size=self.config.max_pool_size,
            timeout=self.config.pool_timeout
        )

        self.backend_type = "postgresql"
        logger.info("PostgreSQL connection pool established")

        # Initialize schema
        await self._initialize_schema()

    async def _connect_sqlite(self):
        """Connect to SQLite (fallback or development)"""
        logger.info(f"Using SQLite backend at {self.config.sqlite_path}")

        # Create directory if needed
        db_path = Path(self.config.sqlite_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.sqlite_conn = sqlite3.connect(
            self.config.sqlite_path,
            check_same_thread=False
        )
        self.sqlite_conn.row_factory = sqlite3.Row

        self.backend_type = "sqlite"
        logger.info("SQLite connection established")

        # Initialize schema
        await self._initialize_schema()

    async def _initialize_schema(self):
        """Initialize database schema"""
        schema_sql = self._read_schema()

        if self.backend_type == "postgresql":
            async with self.pool.acquire() as conn:
                await conn.execute(schema_sql)
        else:
            cursor = self.sqlite_conn.cursor()
            cursor.executescript(schema_sql)
            self.sqlite_conn.commit()

        logger.info(f"{self.backend_type} schema initialized")

    def _read_schema(self) -> str:
        """Read schema SQL file"""
        schema_path = Path(self.config.schema_path)

        if schema_path.exists():
            return schema_path.read_text()

        module_dir = Path(__file__).parent
        schema_path = module_dir / "schema.sql"
        if schema_path.exists():
            return schema_path.read_text()

        raise FileNotFoundError(f"Schema file not found: {self.config.schema_path}")

    async def execute_query(
        self,
        query: str,
        params: list[Any] | None = None
    ) -> QueryResult:
        """Execute SQL query and return results"""
        if self._closed:
            raise RuntimeError("Backend is closed")

        start_time = asyncio.get_event_loop().time()

        try:
            if self.backend_type == "postgresql":
                result = await self._execute_postgresql(query, params)
            elif self.backend_type == "sqlite":
                result = await self._execute_sqlite(query, params)
            else:
                raise RuntimeError("Backend not initialized")

            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            return QueryResult(
                rows=result,
                row_count=len(result),
                latency_ms=latency_ms,
                success=True,
                backend=self.backend_type
            )

        except Exception as e:
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.error(f"Query failed: {e}")

            return QueryResult(
                rows=[],
                row_count=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
                backend=self.backend_type
            )

    async def _execute_postgresql(
        self,
        query: str,
        params: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute query on PostgreSQL"""
        async with self.pool.acquire() as conn:
            if params:
                rows = await conn.fetch(query, *params)
            else:
                rows = await conn.fetch(query)

            return [dict(row) for row in rows]

    async def _execute_sqlite(
        self,
        query: str,
        params: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute query on SQLite (wrapped in asyncio)"""
        loop = asyncio.get_event_loop()

        def _sync_execute():
            cursor = self.sqlite_conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.sqlite_conn.commit()
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

        return await loop.run_in_executor(None, _sync_execute)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID with prefix"""
        random_part = secrets.token_hex(12)
        return f"{prefix}_{random_part}"

    def _hash_password(self, password: str) -> str:
        """Hash password with bcrypt or SHA256 fallback"""
        if HAS_BCRYPT:
            return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        else:
            # Fallback (not recommended for production)
            return hashlib.sha256(password.encode()).hexdigest()

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        if HAS_BCRYPT:
            try:
                return bcrypt.checkpw(password.encode(), password_hash.encode())
            except Exception:
                return False
        else:
            return hashlib.sha256(password.encode()).hexdigest() == password_hash

    def _hash_api_key(self, key_secret: str) -> str:
        """Hash API key secret with SHA256"""
        return hashlib.sha256(key_secret.encode()).hexdigest()

    def _generate_api_key(self) -> tuple[str, str, str]:
        """
        Generate API key components

        Returns:
            (key_id, key_secret, key_prefix)
            key_secret is shown once to user, key_prefix is for identification
        """
        key_id = self._generate_id(self.config.key_prefix)
        key_secret = secrets.token_urlsafe(self.config.key_length)
        key_prefix = key_secret[:8]
        return key_id, key_secret, key_prefix

    def _sql_param(self, index: int) -> str:
        """Get SQL parameter placeholder"""
        if self.backend_type == "postgresql":
            return f"${index}"
        else:
            return "?"

    # ========================================================================
    # Customer CRUD
    # ========================================================================

    async def insert_customer(
        self,
        email: str,
        name: str,
        password: str,
        company: str | None = None,
        stripe_customer_id: str | None = None
    ) -> tuple[Customer, str]:
        """
        Create new customer with subscription

        Returns:
            (Customer, subscription_id)
        """
        customer_id = self._generate_id("cust")
        subscription_id = self._generate_id("sub")
        password_hash = self._hash_password(password)

        if self.backend_type == "postgresql":
            query = """
                INSERT INTO customers
                (customer_id, email, name, company, password_hash, stripe_customer_id, status)
                VALUES ($1, $2, $3, $4, $5, $6, 'active')
                RETURNING *
            """
            params = [customer_id, email, name, company, password_hash, stripe_customer_id]
        else:
            query = """
                INSERT INTO customers
                (customer_id, email, name, company, password_hash, stripe_customer_id, status)
                VALUES (?, ?, ?, ?, ?, ?, 'active')
            """
            params = [customer_id, email, name, company, password_hash, stripe_customer_id]

        result = await self.execute_query(query, params)
        if not result.success:
            raise Exception(f"Failed to insert customer: {result.error}")

        # Create default subscription (free tier)
        await self.insert_subscription(customer_id, subscription_id, plan="free")

        customer = Customer(
            customer_id=customer_id,
            email=email,
            name=name,
            company=company,
            stripe_customer_id=stripe_customer_id,
            status="active"
        )

        return customer, subscription_id

    async def get_customer(self, customer_id: str) -> Customer | None:
        """Get customer by ID"""
        if self.backend_type == "postgresql":
            query = "SELECT * FROM customers WHERE customer_id = $1"
        else:
            query = "SELECT * FROM customers WHERE customer_id = ?"

        result = await self.execute_query(query, [customer_id])
        if not result.success or result.row_count == 0:
            return None

        row = result.rows[0]
        return Customer(
            customer_id=row["customer_id"],
            email=row["email"],
            name=row["name"],
            company=row.get("company"),
            stripe_customer_id=row.get("stripe_customer_id"),
            status=row["status"],
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at")
        )

    async def get_customer_by_email(self, email: str) -> Customer | None:
        """Get customer by email"""
        if self.backend_type == "postgresql":
            query = "SELECT * FROM customers WHERE email = $1"
        else:
            query = "SELECT * FROM customers WHERE email = ?"

        result = await self.execute_query(query, [email])
        if not result.success or result.row_count == 0:
            return None

        row = result.rows[0]
        return Customer(
            customer_id=row["customer_id"],
            email=row["email"],
            name=row["name"],
            company=row.get("company"),
            stripe_customer_id=row.get("stripe_customer_id"),
            status=row["status"],
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at")
        )

    async def authenticate_customer(self, email: str, password: str) -> Customer | None:
        """Authenticate customer by email and password"""
        if self.backend_type == "postgresql":
            query = "SELECT * FROM customers WHERE email = $1 AND status = 'active'"
        else:
            query = "SELECT * FROM customers WHERE email = ? AND status = 'active'"

        result = await self.execute_query(query, [email])
        if not result.success or result.row_count == 0:
            return None

        row = result.rows[0]
        if not self._verify_password(password, row["password_hash"]):
            return None

        return Customer(
            customer_id=row["customer_id"],
            email=row["email"],
            name=row["name"],
            company=row.get("company"),
            stripe_customer_id=row.get("stripe_customer_id"),
            status=row["status"]
        )

    async def update_customer(
        self,
        customer_id: str,
        name: str | None = None,
        company: str | None = None,
        stripe_customer_id: str | None = None
    ) -> Customer | None:
        """Update customer fields"""
        updates = []
        params = []
        param_idx = 1

        if name is not None:
            if self.backend_type == "postgresql":
                updates.append(f"name = ${param_idx}")
            else:
                updates.append("name = ?")
            params.append(name)
            param_idx += 1

        if company is not None:
            if self.backend_type == "postgresql":
                updates.append(f"company = ${param_idx}")
            else:
                updates.append("company = ?")
            params.append(company)
            param_idx += 1

        if stripe_customer_id is not None:
            if self.backend_type == "postgresql":
                updates.append(f"stripe_customer_id = ${param_idx}")
            else:
                updates.append("stripe_customer_id = ?")
            params.append(stripe_customer_id)
            param_idx += 1

        if not updates:
            return await self.get_customer(customer_id)

        if self.backend_type == "postgresql":
            updates.append("updated_at = CURRENT_TIMESTAMP")
            query = f"""
                UPDATE customers
                SET {', '.join(updates)}
                WHERE customer_id = ${param_idx}
            """
        else:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            query = f"""
                UPDATE customers
                SET {', '.join(updates)}
                WHERE customer_id = ?
            """

        params.append(customer_id)
        await self.execute_query(query, params)

        return await self.get_customer(customer_id)

    # ========================================================================
    # Subscription CRUD
    # ========================================================================

    async def insert_subscription(
        self,
        customer_id: str,
        subscription_id: str | None = None,
        plan: str = "free",
        stripe_subscription_id: str | None = None,
        stripe_price_id: str | None = None
    ) -> Subscription:
        """Create subscription for customer"""
        if subscription_id is None:
            subscription_id = self._generate_id("sub")

        # Get plan defaults
        plan_config = self._get_plan_config(plan)

        if self.backend_type == "postgresql":
            query = """
                INSERT INTO subscriptions
                (subscription_id, customer_id, plan, stripe_subscription_id, stripe_price_id,
                 status, queries_included, price_per_query_cents)
                VALUES ($1, $2, $3, $4, $5, 'active', $6, $7)
            """
            params = [
                subscription_id, customer_id, plan, stripe_subscription_id, stripe_price_id,
                plan_config["queries_included"], plan_config["price_per_query_cents"]
            ]
        else:
            query = """
                INSERT INTO subscriptions
                (subscription_id, customer_id, plan, stripe_subscription_id, stripe_price_id,
                 status, queries_included, price_per_query_cents)
                VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
            """
            params = [
                subscription_id, customer_id, plan, stripe_subscription_id, stripe_price_id,
                plan_config["queries_included"], plan_config["price_per_query_cents"]
            ]

        await self.execute_query(query, params)

        return Subscription(
            subscription_id=subscription_id,
            customer_id=customer_id,
            plan=plan,
            stripe_subscription_id=stripe_subscription_id,
            stripe_price_id=stripe_price_id,
            status="active",
            queries_included=plan_config["queries_included"],
            price_per_query_cents=plan_config["price_per_query_cents"]
        )

    def _get_plan_config(self, plan: str) -> dict[str, Any]:
        """Get default configuration for a plan"""
        plans = {
            "free": {"queries_included": 0, "price_per_query_cents": 0.0, "rate_limit_qps": 1.0},
            "payg": {"queries_included": 0, "price_per_query_cents": 0.1, "rate_limit_qps": 10.0},
            "pro": {"queries_included": 50000, "price_per_query_cents": 0.05, "rate_limit_qps": 50.0},
            "enterprise": {"queries_included": 0, "price_per_query_cents": 0.0, "rate_limit_qps": 100.0}
        }
        return plans.get(plan, plans["free"])

    async def get_subscription(self, customer_id: str) -> Subscription | None:
        """Get active subscription for customer"""
        if self.backend_type == "postgresql":
            query = """
                SELECT * FROM subscriptions
                WHERE customer_id = $1 AND status = 'active'
                ORDER BY created_at DESC LIMIT 1
            """
        else:
            query = """
                SELECT * FROM subscriptions
                WHERE customer_id = ? AND status = 'active'
                ORDER BY created_at DESC LIMIT 1
            """

        result = await self.execute_query(query, [customer_id])
        if not result.success or result.row_count == 0:
            return None

        row = result.rows[0]
        return Subscription(
            subscription_id=row["subscription_id"],
            customer_id=row["customer_id"],
            plan=row["plan"],
            stripe_subscription_id=row.get("stripe_subscription_id"),
            stripe_price_id=row.get("stripe_price_id"),
            status=row["status"],
            current_period_start=row.get("current_period_start"),
            current_period_end=row.get("current_period_end"),
            queries_included=row.get("queries_included", 0),
            price_per_query_cents=row.get("price_per_query_cents"),
            cancel_at_period_end=row.get("cancel_at_period_end", False)
        )

    async def update_subscription(
        self,
        subscription_id: str,
        plan: str | None = None,
        status: str | None = None,
        stripe_subscription_id: str | None = None,
        current_period_start: datetime | None = None,
        current_period_end: datetime | None = None,
        cancel_at_period_end: bool | None = None
    ) -> QueryResult:
        """Update subscription fields"""
        updates = []
        params = []
        param_idx = 1

        if plan is not None:
            plan_config = self._get_plan_config(plan)
            if self.backend_type == "postgresql":
                updates.append(f"plan = ${param_idx}")
                param_idx += 1
                updates.append(f"queries_included = ${param_idx}")
                param_idx += 1
                updates.append(f"price_per_query_cents = ${param_idx}")
                param_idx += 1
            else:
                updates.extend(["plan = ?", "queries_included = ?", "price_per_query_cents = ?"])
            params.extend([plan, plan_config["queries_included"], plan_config["price_per_query_cents"]])

        if status is not None:
            if self.backend_type == "postgresql":
                updates.append(f"status = ${param_idx}")
                param_idx += 1
            else:
                updates.append("status = ?")
            params.append(status)

        if stripe_subscription_id is not None:
            if self.backend_type == "postgresql":
                updates.append(f"stripe_subscription_id = ${param_idx}")
                param_idx += 1
            else:
                updates.append("stripe_subscription_id = ?")
            params.append(stripe_subscription_id)

        if current_period_start is not None:
            if self.backend_type == "postgresql":
                updates.append(f"current_period_start = ${param_idx}")
                param_idx += 1
            else:
                updates.append("current_period_start = ?")
            params.append(current_period_start)

        if current_period_end is not None:
            if self.backend_type == "postgresql":
                updates.append(f"current_period_end = ${param_idx}")
                param_idx += 1
            else:
                updates.append("current_period_end = ?")
            params.append(current_period_end)

        if cancel_at_period_end is not None:
            if self.backend_type == "postgresql":
                updates.append(f"cancel_at_period_end = ${param_idx}")
                param_idx += 1
            else:
                updates.append("cancel_at_period_end = ?")
            params.append(cancel_at_period_end)

        if not updates:
            return QueryResult(rows=[], row_count=0, latency_ms=0, success=True, backend=self.backend_type)

        updates.append("updated_at = CURRENT_TIMESTAMP")

        if self.backend_type == "postgresql":
            query = f"""
                UPDATE subscriptions
                SET {', '.join(updates)}
                WHERE subscription_id = ${param_idx}
            """
        else:
            query = f"""
                UPDATE subscriptions
                SET {', '.join(updates)}
                WHERE subscription_id = ?
            """

        params.append(subscription_id)
        return await self.execute_query(query, params)

    # ========================================================================
    # API Key CRUD
    # ========================================================================

    async def create_api_key(
        self,
        customer_id: str,
        name: str = "Default",
        rate_limit_qps: float | None = None,
        expires_in_days: int | None = None
    ) -> tuple[APIKey, str]:
        """
        Create new API key for customer

        Returns:
            (APIKey, key_secret) - key_secret is shown ONCE to user
        """
        key_id, key_secret, key_prefix = self._generate_api_key()
        key_hash = self._hash_api_key(key_secret)

        # Get rate limit from subscription if not specified
        if rate_limit_qps is None:
            subscription = await self.get_subscription(customer_id)
            if subscription:
                plan_config = self._get_plan_config(subscription.plan)
                rate_limit_qps = plan_config["rate_limit_qps"]
            else:
                rate_limit_qps = 10.0

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        if self.backend_type == "postgresql":
            query = """
                INSERT INTO api_keys
                (key_id, customer_id, key_hash, key_prefix, name, status, rate_limit_qps, expires_at)
                VALUES ($1, $2, $3, $4, $5, 'active', $6, $7)
            """
            params = [key_id, customer_id, key_hash, key_prefix, name, rate_limit_qps, expires_at]
        else:
            query = """
                INSERT INTO api_keys
                (key_id, customer_id, key_hash, key_prefix, name, status, rate_limit_qps, expires_at)
                VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
            """
            params = [key_id, customer_id, key_hash, key_prefix, name, rate_limit_qps, expires_at]

        await self.execute_query(query, params)

        api_key = APIKey(
            key_id=key_id,
            customer_id=customer_id,
            key_prefix=key_prefix,
            name=name,
            status="active",
            rate_limit_qps=rate_limit_qps,
            expires_at=expires_at
        )

        return api_key, key_secret

    async def get_api_key_by_secret(self, key_secret: str) -> APIKey | None:
        """Get API key by validating secret (for authentication)"""
        key_hash = self._hash_api_key(key_secret)
        key_prefix = key_secret[:8]

        if self.backend_type == "postgresql":
            query = """
                SELECT * FROM api_keys
                WHERE key_prefix = $1 AND key_hash = $2 AND status = 'active'
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """
        else:
            query = """
                SELECT * FROM api_keys
                WHERE key_prefix = ? AND key_hash = ? AND status = 'active'
                AND (expires_at IS NULL OR expires_at > datetime('now'))
            """

        result = await self.execute_query(query, [key_prefix, key_hash])
        if not result.success or result.row_count == 0:
            return None

        row = result.rows[0]
        return APIKey(
            key_id=row["key_id"],
            customer_id=row["customer_id"],
            key_prefix=row["key_prefix"],
            name=row["name"],
            status=row["status"],
            rate_limit_qps=row.get("rate_limit_qps", 10.0),
            expires_at=row.get("expires_at"),
            last_used=row.get("last_used"),
            request_count=row.get("request_count", 0)
        )

    async def get_api_keys_for_customer(self, customer_id: str) -> list[APIKey]:
        """Get all API keys for customer"""
        if self.backend_type == "postgresql":
            query = "SELECT * FROM api_keys WHERE customer_id = $1 ORDER BY created_at DESC"
        else:
            query = "SELECT * FROM api_keys WHERE customer_id = ? ORDER BY created_at DESC"

        result = await self.execute_query(query, [customer_id])
        if not result.success:
            return []

        return [
            APIKey(
                key_id=row["key_id"],
                customer_id=row["customer_id"],
                key_prefix=row["key_prefix"],
                name=row["name"],
                status=row["status"],
                rate_limit_qps=row.get("rate_limit_qps", 10.0),
                expires_at=row.get("expires_at"),
                last_used=row.get("last_used"),
                request_count=row.get("request_count", 0)
            )
            for row in result.rows
        ]

    async def update_api_key_usage(self, key_id: str) -> QueryResult:
        """Update API key last_used and increment request_count"""
        if self.backend_type == "postgresql":
            query = """
                UPDATE api_keys
                SET last_used = CURRENT_TIMESTAMP, request_count = request_count + 1
                WHERE key_id = $1
            """
        else:
            query = """
                UPDATE api_keys
                SET last_used = datetime('now'), request_count = request_count + 1
                WHERE key_id = ?
            """

        return await self.execute_query(query, [key_id])

    async def revoke_api_key(self, key_id: str, customer_id: str) -> QueryResult:
        """Revoke API key (must belong to customer)"""
        if self.backend_type == "postgresql":
            query = """
                UPDATE api_keys
                SET status = 'revoked'
                WHERE key_id = $1 AND customer_id = $2
            """
        else:
            query = """
                UPDATE api_keys
                SET status = 'revoked'
                WHERE key_id = ? AND customer_id = ?
            """

        return await self.execute_query(query, [key_id, customer_id])

    # ========================================================================
    # Usage Tracking
    # ========================================================================

    async def record_usage(
        self,
        customer_id: str,
        date_val: date | None = None,
        queries_delta: int = 1,
        tokens_delta: int = 0
    ) -> QueryResult:
        """
        Record or increment daily usage.

        Requires: enable_usage_tracking=True in config.
        Returns no-op QueryResult if usage tracking is disabled.
        """
        # Check feature flag
        if not self.usage_tracking_enabled:
            return QueryResult(
                rows=[], row_count=0, latency_ms=0, success=True,
                backend=self.backend_type
            )

        if date_val is None:
            date_val = date.today()

        usage_id = self._generate_id("usage")

        if self.backend_type == "postgresql":
            # Upsert with ON CONFLICT
            query = """
                INSERT INTO usage_records (id, customer_id, date, queries_count, tokens_used)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (customer_id, date)
                DO UPDATE SET
                    queries_count = usage_records.queries_count + $4,
                    tokens_used = usage_records.tokens_used + $5,
                    updated_at = CURRENT_TIMESTAMP
            """
            params = [usage_id, customer_id, date_val, queries_delta, tokens_delta]
        else:
            # SQLite: Try insert, on failure update
            # First try to insert
            query = """
                INSERT OR REPLACE INTO usage_records
                (id, customer_id, date, queries_count, tokens_used, updated_at)
                VALUES (
                    COALESCE(
                        (SELECT id FROM usage_records WHERE customer_id = ? AND date = ?),
                        ?
                    ),
                    ?,
                    ?,
                    COALESCE(
                        (SELECT queries_count FROM usage_records WHERE customer_id = ? AND date = ?),
                        0
                    ) + ?,
                    COALESCE(
                        (SELECT tokens_used FROM usage_records WHERE customer_id = ? AND date = ?),
                        0
                    ) + ?,
                    CURRENT_TIMESTAMP
                )
            """
            params = [
                customer_id, date_val, usage_id,
                customer_id, date_val,
                customer_id, date_val, queries_delta,
                customer_id, date_val, tokens_delta
            ]

        return await self.execute_query(query, params)

    async def get_usage_for_period(
        self,
        customer_id: str,
        start_date: date,
        end_date: date | None = None
    ) -> list[UsageRecord]:
        """
        Get usage records for a date range.

        Requires: enable_usage_tracking=True in config.
        Returns empty list if usage tracking is disabled.
        """
        # Check feature flag
        if not self.usage_tracking_enabled:
            return []

        if end_date is None:
            end_date = date.today()

        if self.backend_type == "postgresql":
            query = """
                SELECT * FROM usage_records
                WHERE customer_id = $1 AND date >= $2 AND date <= $3
                ORDER BY date DESC
            """
        else:
            query = """
                SELECT * FROM usage_records
                WHERE customer_id = ? AND date >= ? AND date <= ?
                ORDER BY date DESC
            """

        result = await self.execute_query(query, [customer_id, start_date, end_date])
        if not result.success:
            return []

        return [
            UsageRecord(
                id=row["id"],
                customer_id=row["customer_id"],
                date=row["date"] if isinstance(row["date"], date) else date.fromisoformat(row["date"]),
                queries_count=row.get("queries_count", 0),
                tokens_used=row.get("tokens_used", 0),
                reported_to_stripe=row.get("reported_to_stripe", False),
                stripe_usage_record_id=row.get("stripe_usage_record_id")
            )
            for row in result.rows
        ]

    async def get_unreported_usage(self) -> list[UsageRecord]:
        """
        Get all usage records not yet reported to Stripe.

        Requires: enable_billing=True in config.
        Returns empty list if billing is disabled.
        """
        # Check feature flag - billing-specific operation
        if not self.billing_enabled:
            return []

        if self.backend_type == "postgresql":
            query = """
                SELECT * FROM usage_records
                WHERE reported_to_stripe = FALSE AND queries_count > 0
                ORDER BY date ASC
            """
        else:
            query = """
                SELECT * FROM usage_records
                WHERE reported_to_stripe = 0 AND queries_count > 0
                ORDER BY date ASC
            """

        result = await self.execute_query(query, [])
        if not result.success:
            return []

        return [
            UsageRecord(
                id=row["id"],
                customer_id=row["customer_id"],
                date=row["date"] if isinstance(row["date"], date) else date.fromisoformat(row["date"]),
                queries_count=row.get("queries_count", 0),
                tokens_used=row.get("tokens_used", 0),
                reported_to_stripe=False
            )
            for row in result.rows
        ]

    async def mark_usage_reported(
        self,
        usage_id: str,
        stripe_usage_record_id: str | None = None
    ) -> QueryResult:
        """
        Mark usage record as reported to Stripe.

        Requires: enable_billing=True in config.
        Returns no-op QueryResult if billing is disabled.
        """
        # Check feature flag - billing-specific operation
        if not self.billing_enabled:
            return QueryResult(
                rows=[], row_count=0, latency_ms=0, success=True,
                backend=self.backend_type
            )

        if self.backend_type == "postgresql":
            query = """
                UPDATE usage_records
                SET reported_to_stripe = TRUE,
                    stripe_usage_record_id = $2,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = $1
            """
        else:
            query = """
                UPDATE usage_records
                SET reported_to_stripe = 1,
                    stripe_usage_record_id = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """
            # Note: SQLite params order is different
            return await self.execute_query(query, [stripe_usage_record_id, usage_id])

        return await self.execute_query(query, [usage_id, stripe_usage_record_id])

    async def get_usage_for_date(self, customer_id: str, date_val: date) -> UsageRecord | None:
        """
        Get usage record for a specific date.

        Requires: enable_usage_tracking=True in config.
        Returns None if usage tracking is disabled.
        """
        # Check feature flag
        if not self.usage_tracking_enabled:
            return None

        if self.backend_type == "postgresql":
            query = """
                SELECT * FROM usage_records
                WHERE customer_id = $1 AND date = $2
            """
        else:
            query = """
                SELECT * FROM usage_records
                WHERE customer_id = ? AND date = ?
            """

        result = await self.execute_query(query, [customer_id, date_val])
        if not result.success or result.row_count == 0:
            return None

        row = result.rows[0]
        return UsageRecord(
            id=row["id"],
            customer_id=row["customer_id"],
            date=row["date"] if isinstance(row["date"], date) else date.fromisoformat(row["date"]),
            queries_count=row.get("queries_count", 0),
            tokens_used=row.get("tokens_used", 0),
            reported_to_stripe=row.get("reported_to_stripe", False),
            stripe_usage_record_id=row.get("stripe_usage_record_id")
        )

    # ========================================================================
    # Audit Logging
    # ========================================================================

    async def log_event(
        self,
        event_type: str,
        customer_id: str | None = None,
        event_data: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None
    ) -> QueryResult:
        """
        Log an audit event.

        Requires: enable_audit_log=True in config.
        Returns no-op QueryResult if audit logging is disabled.
        """
        # Check feature flag
        if not self.audit_log_enabled:
            return QueryResult(
                rows=[], row_count=0, latency_ms=0, success=True,
                backend=self.backend_type
            )

        audit_id = self._generate_id("audit")

        if self.backend_type == "postgresql":
            query = """
                INSERT INTO audit_log
                (id, customer_id, event_type, event_data, ip_address, user_agent)
                VALUES ($1, $2, $3, $4, $5, $6)
            """
            params = [audit_id, customer_id, event_type, json.dumps(event_data) if event_data else None, ip_address, user_agent]
        else:
            query = """
                INSERT INTO audit_log
                (id, customer_id, event_type, event_data, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            params = [audit_id, customer_id, event_type, json.dumps(event_data) if event_data else None, ip_address, user_agent]

        return await self.execute_query(query, params)

    # ========================================================================
    # Lifecycle
    # ========================================================================

    async def close(self):
        """Close database connections"""
        if self._closed:
            return

        logger.info(f"Closing {self.backend_type} connection")

        if self.pool:
            await self.pool.close()
            self.pool = None

        if self.sqlite_conn:
            self.sqlite_conn.close()
            self.sqlite_conn = None

        self._closed = True
        logger.info("SaaS backend closed")


# ============================================================================
# Factory Function
# ============================================================================

def create_saas_backend(config: SaaSConfig | None = None) -> SaaSBackend:
    """
    Create SaaS backend with default or custom configuration

    Args:
        config: SaaS configuration (uses defaults if None)

    Returns:
        SaaSBackend instance (not yet connected - use async context manager)
    """
    if config is None:
        config = SaaSConfig()

    return SaaSBackend(config)
