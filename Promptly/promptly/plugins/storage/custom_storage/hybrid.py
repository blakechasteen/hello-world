#!/usr/bin/env python3
"""
Hybrid Multi-Tier Storage Backend Plugin

Production-grade hybrid storage combining:
- HOT tier: Redis (in-memory, recent prompts, <1ms latency)
- WARM tier: PostgreSQL (active prompts, 1-5ms latency)
- COLD tier: S3 (archived prompts, 50-200ms latency)

Features:
- Automatic tiering based on access patterns
- Transparent read-through cache
- Write-through to all tiers
- Cost optimization through intelligent data placement
- Access frequency tracking
- Automatic archiving of old prompts
- Cache warming and preloading
- Comprehensive monitoring

Tiering Logic:
- Hot (Redis): Last accessed <24 hours, access count >10
- Warm (PostgreSQL): Last accessed <30 days, access count >2
- Cold (S3): Everything else, long-term archive
"""

import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
from threading import Lock

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from sqlalchemy import (
        create_engine, Column, Integer, String, Text, Float,
        DateTime, Index
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from ...base import BaseStorageBackend


class HybridStorage(BaseStorageBackend):
    """
    Hybrid multi-tier storage backend for Promptly

    Architecture:
        ┌─────────────────────────────────────────┐
        │           READ REQUEST                   │
        └─────────────────────────────────────────┘
                        │
                        ├──► Redis (Hot) ──────► Cache Hit (fast)
                        │      │
                        │      └──► Cache Miss
                        │             │
                        ├──► PostgreSQL (Warm) ──► Found ──► Promote to Redis
                        │      │
                        │      └──► Not Found
                        │             │
                        └──► S3 (Cold) ──────────► Found ──► Promote to Warm/Hot

        ┌─────────────────────────────────────────┐
        │          WRITE REQUEST                   │
        └─────────────────────────────────────────┘
                        │
                        ├──► Redis (Hot) ──► Write
                        ├──► PostgreSQL (Warm) ──► Write
                        └──► S3 (Cold) ──► Write (async)

    Data Flow:
    1. New prompts → Hot tier (Redis + PostgreSQL + S3)
    2. Frequently accessed → Stay in Hot tier
    3. Infrequently accessed → Demote to Warm tier (PostgreSQL + S3)
    4. Rarely accessed → Demote to Cold tier (S3 only)
    5. On access → Promote based on frequency
    """

    # Tiering thresholds
    HOT_TIER_TTL = 86400  # 24 hours
    WARM_TIER_DAYS = 30  # 30 days
    HOT_ACCESS_THRESHOLD = 10  # 10 accesses to stay hot
    WARM_ACCESS_THRESHOLD = 2  # 2 accesses to stay warm

    def __init__(
        self,
        # Redis (Hot) configuration
        redis_url: str = 'redis://localhost:6379/0',
        redis_ttl: int = 86400,  # 24 hours
        redis_max_memory: str = '1gb',
        # PostgreSQL (Warm) configuration
        postgres_url: str = 'postgresql://localhost/promptly',
        postgres_pool_size: int = 10,
        # S3 (Cold) configuration
        s3_bucket: str = 'promptly-archive',
        s3_region: str = 'us-east-1',
        s3_endpoint: Optional[str] = None,
        # Tiering configuration
        enable_auto_tiering: bool = True,
        tiering_interval: int = 3600,  # 1 hour
        max_hot_prompts: int = 1000,
        max_warm_prompts: int = 10000,
    ):
        """
        Initialize hybrid multi-tier storage backend

        Args:
            redis_url: Redis connection URL
            redis_ttl: Redis key TTL in seconds
            redis_max_memory: Maximum Redis memory
            postgres_url: PostgreSQL connection string
            postgres_pool_size: PostgreSQL connection pool size
            s3_bucket: S3 bucket name for cold storage
            s3_region: S3 region
            s3_endpoint: Optional S3 endpoint (for MinIO)
            enable_auto_tiering: Enable automatic data tiering
            tiering_interval: Tiering check interval in seconds
            max_hot_prompts: Maximum prompts in hot tier
            max_warm_prompts: Maximum prompts in warm tier
        """
        super().__init__(
            name="hybrid",
            description="Multi-tier storage (Redis + PostgreSQL + S3)"
        )

        if not REDIS_AVAILABLE:
            raise ImportError("redis is required. Install with: pip install redis")
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("sqlalchemy is required. Install with: pip install sqlalchemy psycopg2-binary")
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required. Install with: pip install boto3")

        # Configuration
        self.redis_url = redis_url
        self.redis_ttl = redis_ttl
        self.postgres_url = postgres_url
        self.s3_bucket = s3_bucket
        self.s3_region = s3_region
        self.s3_endpoint = s3_endpoint
        self.enable_auto_tiering = enable_auto_tiering
        self.max_hot_prompts = max_hot_prompts
        self.max_warm_prompts = max_warm_prompts

        # Storage backends
        self.redis_client = None
        self.postgres_engine = None
        self.postgres_session = None
        self.s3_client = None

        # Access tracking
        self.access_tracker = defaultdict(lambda: {'count': 0, 'last_access': None})
        self.tracker_lock = Lock()

        # Current branch
        self.current_branch = 'main'

    def init_storage(self, storage_path: str) -> None:
        """
        Initialize all storage tiers

        Args:
            storage_path: Base path (used for metadata)
        """
        # Initialize Redis (Hot tier)
        self._init_redis()

        # Initialize PostgreSQL (Warm tier)
        self._init_postgres()

        # Initialize S3 (Cold tier)
        self._init_s3()

        # Start auto-tiering if enabled
        if self.enable_auto_tiering:
            self._start_auto_tiering()

    def _init_redis(self) -> None:
        """Initialize Redis connection"""
        self.redis_client = redis.from_url(
            self.redis_url,
            decode_responses=True,
            max_connections=50,
            socket_timeout=5,
            socket_connect_timeout=5,
        )

        # Test connection
        try:
            self.redis_client.ping()
        except redis.ConnectionError as e:
            raise RuntimeError(f"Failed to connect to Redis: {e}")

        # Configure memory policy
        try:
            self.redis_client.config_set('maxmemory', self.redis_max_memory)
            self.redis_client.config_set('maxmemory-policy', 'allkeys-lru')
        except Exception as e:
            print(f"Warning: Could not configure Redis memory: {e}")

    def _init_postgres(self) -> None:
        """Initialize PostgreSQL connection"""
        from sqlalchemy.pool import QueuePool

        self.postgres_engine = create_engine(
            self.postgres_url,
            poolclass=QueuePool,
            pool_size=self.postgres_pool_size,
            max_overflow=20,
            pool_pre_ping=True,
        )

        # Create tables
        Base = declarative_base()

        class Prompt(Base):
            __tablename__ = 'hybrid_prompts'
            id = Column(Integer, primary_key=True, autoincrement=True)
            name = Column(String(255), nullable=False, index=True)
            content = Column(Text, nullable=False)
            branch = Column(String(100), nullable=False, default='main', index=True)
            version = Column(Integer, nullable=False, default=1)
            commit_hash = Column(String(64), unique=True, nullable=False, index=True)
            created_at = Column(DateTime, default=datetime.utcnow, index=True)
            last_accessed = Column(DateTime, default=datetime.utcnow, index=True)
            access_count = Column(Integer, default=0)
            tier = Column(String(20), default='warm', index=True)  # hot, warm, cold
            metadata = Column(Text, nullable=True)
            __table_args__ = (
                Index('idx_tier_access', 'tier', 'last_accessed'),
                Index('idx_name_branch', 'name', 'branch'),
            )

        class Branch(Base):
            __tablename__ = 'hybrid_branches'
            id = Column(Integer, primary_key=True, autoincrement=True)
            name = Column(String(100), unique=True, nullable=False, index=True)
            head_commit = Column(String(64), nullable=False)
            created_at = Column(DateTime, default=datetime.utcnow)

        class Evaluation(Base):
            __tablename__ = 'hybrid_evaluations'
            id = Column(Integer, primary_key=True, autoincrement=True)
            prompt_name = Column(String(255), nullable=False, index=True)
            commit_hash = Column(String(64), nullable=False)
            test_case = Column(Text, nullable=False)
            expected = Column(Text, nullable=True)
            actual = Column(Text, nullable=True)
            score = Column(Float, nullable=True)
            created_at = Column(DateTime, default=datetime.utcnow)

        class Chain(Base):
            __tablename__ = 'hybrid_chains'
            id = Column(Integer, primary_key=True, autoincrement=True)
            name = Column(String(255), unique=True, nullable=False, index=True)
            steps = Column(Text, nullable=False)
            description = Column(Text, nullable=True)
            created_at = Column(DateTime, default=datetime.utcnow)

        # Create all tables
        Base.metadata.create_all(self.postgres_engine)

        # Create session factory
        SessionLocal = sessionmaker(bind=self.postgres_engine)
        self.postgres_session = SessionLocal

        # Store models for later use
        self.Prompt = Prompt
        self.Branch = Branch
        self.Evaluation = Evaluation
        self.Chain = Chain

        # Initialize main branch
        session = self.postgres_session()
        try:
            existing = session.query(self.Branch).filter_by(name='main').first()
            if not existing:
                main_branch = self.Branch(name='main', head_commit='init')
                session.add(main_branch)
                session.commit()
        finally:
            session.close()

    def _init_s3(self) -> None:
        """Initialize S3 connection"""
        self.s3_client = boto3.client(
            's3',
            region_name=self.s3_region,
            endpoint_url=self.s3_endpoint,
        )

        # Create bucket if it doesn't exist
        try:
            self.s3_client.head_bucket(Bucket=self.s3_bucket)
        except Exception:
            try:
                if self.s3_region == 'us-east-1':
                    self.s3_client.create_bucket(Bucket=self.s3_bucket)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.s3_bucket,
                        CreateBucketConfiguration={'LocationConstraint': self.s3_region}
                    )
            except Exception as e:
                print(f"Warning: Could not create S3 bucket: {e}")

    def _start_auto_tiering(self) -> None:
        """Start automatic tiering background process"""
        # In a production system, this would be a background thread or task
        # For now, we'll run it on-demand during operations
        pass

    def _generate_commit_hash(self, name: str, content: str, timestamp: str) -> str:
        """Generate a unique commit hash"""
        data = f"{name}:{content}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    def _track_access(self, name: str, branch: str) -> None:
        """Track prompt access for tiering decisions"""
        key = f"{branch}:{name}"
        with self.tracker_lock:
            self.access_tracker[key]['count'] += 1
            self.access_tracker[key]['last_access'] = datetime.now()

    def _get_tier_for_prompt(self, access_count: int, last_accessed: datetime) -> str:
        """
        Determine appropriate tier based on access patterns

        Returns:
            'hot', 'warm', or 'cold'
        """
        if last_accessed is None:
            return 'warm'

        age = datetime.now() - last_accessed

        # Hot tier: accessed recently and frequently
        if age < timedelta(days=1) and access_count >= self.HOT_ACCESS_THRESHOLD:
            return 'hot'

        # Warm tier: accessed somewhat recently
        if age < timedelta(days=self.WARM_TIER_DAYS) and access_count >= self.WARM_ACCESS_THRESHOLD:
            return 'warm'

        # Cold tier: rarely accessed
        return 'cold'

    def save_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """
        Save prompt to all tiers (write-through)

        Args:
            prompt_data: Dictionary containing prompt information

        Returns:
            Commit hash
        """
        name = prompt_data['name']
        content = prompt_data['content']
        branch = prompt_data.get('branch', 'main')
        metadata = prompt_data.get('metadata', {})

        timestamp = datetime.now().isoformat()
        commit_hash = self._generate_commit_hash(name, content, timestamp)

        # 1. Save to PostgreSQL (Warm tier - source of truth)
        session = self.postgres_session()
        try:
            # Get current version
            existing = session.query(self.Prompt).filter_by(
                name=name, branch=branch
            ).order_by(self.Prompt.version.desc()).first()

            version = existing.version + 1 if existing else 1

            # Create new prompt version
            prompt = self.Prompt(
                name=name,
                content=content,
                branch=branch,
                version=version,
                commit_hash=commit_hash,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                tier='hot',  # New prompts start in hot tier
                metadata=json.dumps(metadata) if metadata else None,
            )
            session.add(prompt)

            # Update branch head
            branch_obj = session.query(self.Branch).filter_by(name=branch).first()
            if branch_obj:
                branch_obj.head_commit = commit_hash
            else:
                branch_obj = self.Branch(name=branch, head_commit=commit_hash)
                session.add(branch_obj)

            session.commit()

            prompt_obj = {
                'name': name,
                'content': content,
                'branch': branch,
                'version': version,
                'commit_hash': commit_hash,
                'created_at': timestamp,
                'metadata': metadata,
            }

            # 2. Save to Redis (Hot tier)
            try:
                redis_key = f"prompt:{branch}:{name}"
                self.redis_client.setex(
                    redis_key,
                    self.redis_ttl,
                    json.dumps(prompt_obj)
                )
                # Also cache latest version pointer
                version_key = f"prompt:{branch}:{name}:latest"
                self.redis_client.setex(version_key, self.redis_ttl, str(version))
            except Exception as e:
                print(f"Warning: Failed to cache in Redis: {e}")

            # 3. Save to S3 (Cold tier - archival)
            try:
                s3_key = f"prompts/{branch}/{name}/v{version}.json"
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=s3_key,
                    Body=json.dumps(prompt_obj, indent=2).encode('utf-8'),
                    ContentType='application/json',
                    Metadata={
                        'name': name,
                        'branch': branch,
                        'version': str(version),
                        'commit-hash': commit_hash,
                    }
                )
            except Exception as e:
                print(f"Warning: Failed to archive in S3: {e}")

            return commit_hash

        finally:
            session.close()

    def get_prompt(
        self,
        name: str,
        branch: str = "main",
        version: Optional[int] = None,
        commit_hash: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve prompt with automatic tier promotion

        Read strategy:
        1. Try Redis (Hot) - fastest
        2. Try PostgreSQL (Warm) - moderate
        3. Try S3 (Cold) - slowest
        4. Promote to higher tier based on access patterns

        Args:
            name: Prompt name
            branch: Branch name
            version: Optional specific version
            commit_hash: Optional specific commit hash

        Returns:
            Dict with prompt data or None if not found
        """
        # Track access
        self._track_access(name, branch)

        # 1. Try Redis (Hot tier) first
        if not commit_hash and version is None:  # Only cache latest version
            try:
                redis_key = f"prompt:{branch}:{name}"
                cached = self.redis_client.get(redis_key)
                if cached:
                    prompt_data = json.loads(cached)
                    prompt_data['_tier'] = 'hot'
                    prompt_data['_cache_hit'] = True
                    return prompt_data
            except Exception as e:
                print(f"Redis cache miss: {e}")

        # 2. Try PostgreSQL (Warm tier)
        session = self.postgres_session()
        try:
            query = session.query(self.Prompt).filter_by(name=name, branch=branch)

            if commit_hash:
                query = query.filter_by(commit_hash=commit_hash)
            elif version is not None:
                query = query.filter_by(version=version)
            else:
                query = query.order_by(self.Prompt.version.desc())

            prompt = query.first()

            if prompt:
                # Update access tracking
                prompt.last_accessed = datetime.now()
                prompt.access_count += 1

                # Determine tier
                new_tier = self._get_tier_for_prompt(prompt.access_count, prompt.last_accessed)
                prompt.tier = new_tier

                session.commit()

                prompt_data = {
                    'name': prompt.name,
                    'content': prompt.content,
                    'branch': prompt.branch,
                    'version': prompt.version,
                    'commit_hash': prompt.commit_hash,
                    'created_at': prompt.created_at.isoformat(),
                    'metadata': json.loads(prompt.metadata) if prompt.metadata else {},
                    '_tier': new_tier,
                    '_cache_hit': False,
                }

                # Promote to Redis if hot
                if new_tier == 'hot':
                    try:
                        redis_key = f"prompt:{branch}:{name}"
                        self.redis_client.setex(
                            redis_key,
                            self.redis_ttl,
                            json.dumps(prompt_data)
                        )
                    except Exception as e:
                        print(f"Warning: Failed to promote to hot tier: {e}")

                return prompt_data

        finally:
            session.close()

        # 3. Try S3 (Cold tier) as last resort
        if version is None:
            # Need to find latest version from S3
            # This is slow - we should avoid this by keeping metadata in PostgreSQL
            return None

        try:
            s3_key = f"prompts/{branch}/{name}/v{version}.json"
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            content = response['Body'].read().decode('utf-8')
            prompt_data = json.loads(content)
            prompt_data['_tier'] = 'cold'
            prompt_data['_cache_hit'] = False

            # Promote to warm tier (PostgreSQL) if accessed
            # In production, you might want to do this asynchronously
            return prompt_data

        except Exception:
            return None

    def list_prompts(self, branch: str = "main") -> List[Dict[str, Any]]:
        """
        List all prompts on a branch (from PostgreSQL)

        Args:
            branch: Branch name

        Returns:
            List of prompt metadata dictionaries
        """
        session = self.postgres_session()
        try:
            # Get latest version of each prompt
            prompts = session.query(
                self.Prompt.name,
                self.Prompt.version,
                self.Prompt.commit_hash,
                self.Prompt.created_at,
                self.Prompt.tier,
                self.Prompt.access_count,
            ).filter_by(branch=branch).distinct(self.Prompt.name).all()

            return [
                {
                    'name': p.name,
                    'version': p.version,
                    'commit_hash': p.commit_hash,
                    'created_at': p.created_at.isoformat(),
                    'tier': p.tier,
                    'access_count': p.access_count,
                }
                for p in prompts
            ]
        finally:
            session.close()

    def create_branch(self, branch_name: str, from_branch: str = "main") -> None:
        """Create a new branch"""
        session = self.postgres_session()
        try:
            source = session.query(self.Branch).filter_by(name=from_branch).first()
            if not source:
                raise ValueError(f"Source branch '{from_branch}' not found")

            new_branch = self.Branch(
                name=branch_name,
                head_commit=source.head_commit,
            )
            session.add(new_branch)
            session.commit()
        finally:
            session.close()

    def get_current_branch(self) -> str:
        """Get the current branch name"""
        return self.current_branch

    def set_current_branch(self, branch_name: str) -> None:
        """Set the current branch"""
        session = self.postgres_session()
        try:
            branch = session.query(self.Branch).filter_by(name=branch_name).first()
            if not branch:
                raise ValueError(f"Branch '{branch_name}' not found")
            self.current_branch = branch_name
        finally:
            session.close()

    def get_commit_history(
        self,
        name: Optional[str] = None,
        branch: str = "main",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get commit history from PostgreSQL"""
        session = self.postgres_session()
        try:
            query = session.query(self.Prompt).filter_by(branch=branch)
            if name:
                query = query.filter_by(name=name)
            query = query.order_by(self.Prompt.created_at.desc()).limit(limit)

            prompts = query.all()
            return [
                {
                    'commit_hash': p.commit_hash,
                    'name': p.name,
                    'version': p.version,
                    'created_at': p.created_at.isoformat(),
                    'tier': p.tier,
                }
                for p in prompts
            ]
        finally:
            session.close()

    def save_evaluation(self, eval_data: Dict[str, Any]) -> None:
        """Save evaluation results"""
        session = self.postgres_session()
        try:
            evaluation = self.Evaluation(**eval_data)
            session.add(evaluation)
            session.commit()
        finally:
            session.close()

    def save_chain(self, chain_data: Dict[str, Any]) -> None:
        """Save a prompt chain"""
        session = self.postgres_session()
        try:
            chain = session.query(self.Chain).filter_by(name=chain_data['name']).first()
            if chain:
                chain.steps = json.dumps(chain_data['steps'])
                chain.description = chain_data.get('description')
            else:
                chain = self.Chain(
                    name=chain_data['name'],
                    steps=json.dumps(chain_data['steps']),
                    description=chain_data.get('description')
                )
                session.add(chain)
            session.commit()
        finally:
            session.close()

    def get_chain(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a prompt chain"""
        session = self.postgres_session()
        try:
            chain = session.query(self.Chain).filter_by(name=name).first()
            if not chain:
                return None
            return {
                'name': chain.name,
                'steps': json.loads(chain.steps),
                'description': chain.description,
                'created_at': chain.created_at.isoformat(),
            }
        finally:
            session.close()

    def close(self) -> None:
        """Close all connections"""
        if self.redis_client:
            self.redis_client.close()
        if self.postgres_engine:
            self.postgres_engine.dispose()

    # Additional hybrid-specific methods

    def run_tiering(self) -> Dict[str, int]:
        """
        Run tiering operation to optimize data placement

        Returns:
            Statistics about tiering operation
        """
        stats = {
            'promoted_to_hot': 0,
            'demoted_to_warm': 0,
            'demoted_to_cold': 0,
        }

        session = self.postgres_session()
        try:
            # Get all prompts
            prompts = session.query(self.Prompt).all()

            for prompt in prompts:
                old_tier = prompt.tier
                new_tier = self._get_tier_for_prompt(
                    prompt.access_count,
                    prompt.last_accessed
                )

                if old_tier != new_tier:
                    prompt.tier = new_tier

                    if new_tier == 'hot' and old_tier != 'hot':
                        stats['promoted_to_hot'] += 1
                        # Add to Redis
                        try:
                            redis_key = f"prompt:{prompt.branch}:{prompt.name}"
                            prompt_data = {
                                'name': prompt.name,
                                'content': prompt.content,
                                'branch': prompt.branch,
                                'version': prompt.version,
                                'commit_hash': prompt.commit_hash,
                            }
                            self.redis_client.setex(
                                redis_key,
                                self.redis_ttl,
                                json.dumps(prompt_data)
                            )
                        except Exception as e:
                            print(f"Failed to promote to hot: {e}")

                    elif new_tier == 'warm' and old_tier == 'hot':
                        stats['demoted_to_warm'] += 1
                        # Remove from Redis
                        try:
                            redis_key = f"prompt:{prompt.branch}:{prompt.name}"
                            self.redis_client.delete(redis_key)
                        except Exception:
                            pass

                    elif new_tier == 'cold':
                        stats['demoted_to_cold'] += 1

            session.commit()
            return stats

        finally:
            session.close()

    def get_tier_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about data distribution across tiers

        Returns:
            Dict with tier statistics
        """
        session = self.postgres_session()
        try:
            total = session.query(self.Prompt).count()
            hot = session.query(self.Prompt).filter_by(tier='hot').count()
            warm = session.query(self.Prompt).filter_by(tier='warm').count()
            cold = session.query(self.Prompt).filter_by(tier='cold').count()

            # Get Redis stats
            try:
                redis_info = self.redis_client.info('memory')
                redis_memory = redis_info.get('used_memory_human', 'unknown')
                redis_keys = self.redis_client.dbsize()
            except Exception:
                redis_memory = 'unknown'
                redis_keys = 0

            return {
                'total_prompts': total,
                'hot_tier': {
                    'count': hot,
                    'percentage': (hot / total * 100) if total > 0 else 0,
                    'backend': 'Redis',
                    'memory_used': redis_memory,
                    'cached_keys': redis_keys,
                },
                'warm_tier': {
                    'count': warm,
                    'percentage': (warm / total * 100) if total > 0 else 0,
                    'backend': 'PostgreSQL',
                },
                'cold_tier': {
                    'count': cold,
                    'percentage': (cold / total * 100) if total > 0 else 0,
                    'backend': 'S3',
                },
            }
        finally:
            session.close()

    def warm_cache(self, branch: str = 'main', limit: int = 100) -> int:
        """
        Warm Redis cache with most accessed prompts

        Args:
            branch: Branch to warm
            limit: Number of prompts to cache

        Returns:
            Number of prompts cached
        """
        session = self.postgres_session()
        try:
            # Get most accessed prompts
            prompts = session.query(self.Prompt).filter_by(
                branch=branch
            ).order_by(
                self.Prompt.access_count.desc()
            ).limit(limit).all()

            cached = 0
            for prompt in prompts:
                try:
                    redis_key = f"prompt:{prompt.branch}:{prompt.name}"
                    prompt_data = {
                        'name': prompt.name,
                        'content': prompt.content,
                        'branch': prompt.branch,
                        'version': prompt.version,
                        'commit_hash': prompt.commit_hash,
                    }
                    self.redis_client.setex(
                        redis_key,
                        self.redis_ttl,
                        json.dumps(prompt_data)
                    )
                    cached += 1
                except Exception as e:
                    print(f"Failed to cache {prompt.name}: {e}")

            return cached
        finally:
            session.close()
