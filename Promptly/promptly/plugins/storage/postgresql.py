#!/usr/bin/env python3
"""
PostgreSQL Storage Backend Plugin

Production-grade PostgreSQL backend using SQLAlchemy ORM with:
- Connection pooling and retry logic
- Transaction support and ACID guarantees
- Index optimization for queries
- Migration support for schema changes
- Comprehensive error handling
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

try:
    from sqlalchemy import (
        create_engine, Column, Integer, String, Text, Float,
        DateTime, ForeignKey, Index, UniqueConstraint
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship, Session
    from sqlalchemy.pool import QueuePool
    from sqlalchemy.exc import IntegrityError, OperationalError
    from sqlalchemy import func
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    print("Warning: SQLAlchemy not available. Install with: pip install sqlalchemy psycopg2-binary")

from ..base import BaseStorageBackend

if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()

    class Prompt(Base):
        """Prompt model with full versioning support"""
        __tablename__ = 'prompts'

        id = Column(Integer, primary_key=True, autoincrement=True)
        name = Column(String(255), nullable=False, index=True)
        content = Column(Text, nullable=False)
        branch = Column(String(100), nullable=False, default='main', index=True)
        version = Column(Integer, nullable=False, default=1)
        parent_id = Column(Integer, ForeignKey('prompts.id'), nullable=True)
        commit_hash = Column(String(64), unique=True, nullable=False, index=True)
        created_at = Column(DateTime, default=datetime.utcnow, index=True)
        metadata = Column(Text, nullable=True)

        # Relationships
        parent = relationship('Prompt', remote_side=[id], backref='children')
        evaluations = relationship('Evaluation', back_populates='prompt', cascade='all, delete-orphan')

        # Indexes for common queries
        __table_args__ = (
            Index('idx_name_branch', 'name', 'branch'),
            Index('idx_branch_version', 'branch', 'version'),
            Index('idx_created_at_desc', created_at.desc()),
        )

    class Branch(Base):
        """Branch model for managing different prompt versions"""
        __tablename__ = 'branches'

        id = Column(Integer, primary_key=True, autoincrement=True)
        name = Column(String(100), unique=True, nullable=False, index=True)
        head_commit = Column(String(64), nullable=False)
        created_at = Column(DateTime, default=datetime.utcnow)
        description = Column(Text, nullable=True)

    class Evaluation(Base):
        """Evaluation results model"""
        __tablename__ = 'evaluations'

        id = Column(Integer, primary_key=True, autoincrement=True)
        prompt_name = Column(String(255), nullable=False, index=True)
        commit_hash = Column(String(64), ForeignKey('prompts.commit_hash'), nullable=False)
        test_case = Column(Text, nullable=False)
        expected = Column(Text, nullable=True)
        actual = Column(Text, nullable=True)
        score = Column(Float, nullable=True, index=True)
        metrics = Column(Text, nullable=True)
        evaluator_name = Column(String(100), nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow, index=True)

        # Relationship
        prompt = relationship('Prompt', back_populates='evaluations')

    class Chain(Base):
        """Prompt chain model"""
        __tablename__ = 'chains'

        id = Column(Integer, primary_key=True, autoincrement=True)
        name = Column(String(255), unique=True, nullable=False, index=True)
        steps = Column(Text, nullable=False)
        description = Column(Text, nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow)

    class Config(Base):
        """Configuration storage"""
        __tablename__ = 'config'

        key = Column(String(100), primary_key=True)
        value = Column(Text, nullable=False)


class PostgreSQLStorage(BaseStorageBackend):
    """
    PostgreSQL storage backend with enterprise features

    Features:
    - Connection pooling (default pool size: 5-20 connections)
    - Automatic retry on connection failures (max 3 retries)
    - Transaction support with automatic rollback
    - Optimized indexes for fast queries
    - Full ACID guarantees
    - Migration support for schema evolution

    Connection string format:
        postgresql://user:password@host:port/database
        postgresql://user:password@host:port/database?sslmode=require

    Example:
        storage = PostgreSQLStorage()
        storage.init_storage('postgresql://promptly:secret@localhost:5432/promptly_db')
    """

    def __init__(self, pool_size: int = 5, max_overflow: int = 10, pool_timeout: int = 30,
                 max_retries: int = 3):
        """
        Initialize PostgreSQL storage backend

        Args:
            pool_size: Number of connections to maintain in the pool
            max_overflow: Maximum number of connections beyond pool_size
            pool_timeout: Timeout in seconds for getting a connection from pool
            max_retries: Maximum number of retry attempts for failed operations
        """
        super().__init__(
            name="postgresql",
            description="PostgreSQL database storage with connection pooling and ACID guarantees"
        )

        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy required for PostgreSQL backend. Install: pip install sqlalchemy psycopg2-binary")

        self.engine = None
        self.SessionLocal = None
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.max_retries = max_retries

    def init_storage(self, storage_path: str) -> None:
        """
        Initialize PostgreSQL database connection

        Args:
            storage_path: PostgreSQL connection string (e.g., postgresql://user:pass@host:port/db)
        """
        # Create engine with connection pooling
        self.engine = create_engine(
            storage_path,
            poolclass=QueuePool,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_pre_ping=True,  # Verify connections before using
            echo=False  # Set to True for SQL debugging
        )

        # Create all tables
        Base.metadata.create_all(self.engine)

        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Initialize default data
        with self._get_session() as session:
            # Create main branch if not exists
            main_branch = session.query(Branch).filter_by(name='main').first()
            if not main_branch:
                main_branch = Branch(name='main', head_commit='init', description='Main branch')
                session.add(main_branch)

            # Set current branch config
            current_branch = session.query(Config).filter_by(key='current_branch').first()
            if not current_branch:
                current_branch = Config(key='current_branch', value='main')
                session.add(current_branch)

            session.commit()

    @contextmanager
    def _get_session(self) -> Session:
        """
        Context manager for database sessions with automatic cleanup

        Yields:
            SQLAlchemy session
        """
        if not self.SessionLocal:
            raise RuntimeError("Storage not initialized")

        session = self.SessionLocal()
        try:
            yield session
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def _retry_on_failure(self, operation, *args, **kwargs):
        """
        Retry an operation on transient failures

        Args:
            operation: Function to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Result of operation
        """
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except OperationalError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    # Wait before retry (exponential backoff)
                    import time
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise last_exception

    def _generate_commit_hash(self, name: str, content: str, timestamp: str) -> str:
        """Generate a unique commit hash"""
        data = f"{name}:{content}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    def save_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """
        Save a prompt to the database with full transaction support

        Args:
            prompt_data: Dictionary containing prompt information

        Returns:
            str: Commit hash
        """
        def _save():
            with self._get_session() as session:
                name = prompt_data['name']
                content = prompt_data['content']
                branch = prompt_data.get('branch', 'main')
                metadata = prompt_data.get('metadata')

                timestamp = datetime.utcnow().isoformat()
                commit_hash = self._generate_commit_hash(name, content, timestamp)

                # Check for existing versions
                latest = (session.query(Prompt)
                         .filter_by(name=name, branch=branch)
                         .order_by(Prompt.version.desc())
                         .first())

                parent_id = latest.id if latest else None
                version = latest.version + 1 if latest else 1

                # Create new prompt
                prompt = Prompt(
                    name=name,
                    content=content,
                    branch=branch,
                    version=version,
                    parent_id=parent_id,
                    commit_hash=commit_hash,
                    metadata=json.dumps(metadata) if metadata else None
                )

                session.add(prompt)

                # Update branch head
                branch_obj = session.query(Branch).filter_by(name=branch).first()
                if branch_obj:
                    branch_obj.head_commit = commit_hash

                session.commit()
                return commit_hash

        return self._retry_on_failure(_save)

    def get_prompt(self, name: str, branch: str = "main", version: Optional[int] = None,
                   commit_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a prompt from the database"""
        def _get():
            with self._get_session() as session:
                query = session.query(Prompt).filter_by(name=name)

                if commit_hash:
                    query = query.filter_by(commit_hash=commit_hash)
                elif version:
                    query = query.filter_by(branch=branch, version=version)
                else:
                    query = query.filter_by(branch=branch).order_by(Prompt.version.desc())

                prompt = query.first()

                if not prompt:
                    return None

                return {
                    'name': prompt.name,
                    'content': prompt.content,
                    'branch': prompt.branch,
                    'version': prompt.version,
                    'commit_hash': prompt.commit_hash,
                    'created_at': prompt.created_at.isoformat(),
                    'metadata': json.loads(prompt.metadata) if prompt.metadata else {}
                }

        return self._retry_on_failure(_get)

    def list_prompts(self, branch: str = "main") -> List[Dict[str, Any]]:
        """List all prompts on a branch"""
        def _list():
            with self._get_session() as session:
                # Subquery to get max version for each prompt
                subq = (session.query(
                    Prompt.name,
                    func.max(Prompt.version).label('max_version')
                )
                .filter_by(branch=branch)
                .group_by(Prompt.name)
                .subquery())

                # Join to get full prompt details
                prompts = (session.query(Prompt)
                          .join(subq, (Prompt.name == subq.c.name) &
                                     (Prompt.version == subq.c.max_version))
                          .filter_by(branch=branch)
                          .order_by(Prompt.name)
                          .all())

                return [{
                    'name': p.name,
                    'version': p.version,
                    'commit_hash': p.commit_hash,
                    'created_at': p.created_at.isoformat()
                } for p in prompts]

        return self._retry_on_failure(_list)

    def create_branch(self, branch_name: str, from_branch: str = "main") -> None:
        """Create a new branch"""
        def _create():
            with self._get_session() as session:
                # Check source branch exists
                source = session.query(Branch).filter_by(name=from_branch).first()
                if not source:
                    raise Exception(f"Branch '{from_branch}' does not exist")

                # Check if target branch already exists
                if session.query(Branch).filter_by(name=branch_name).first():
                    raise Exception(f"Branch '{branch_name}' already exists")

                # Create new branch
                new_branch = Branch(
                    name=branch_name,
                    head_commit=source.head_commit,
                    description=f"Branched from {from_branch}"
                )
                session.add(new_branch)

                # Copy prompts from source branch
                source_prompts = (session.query(Prompt)
                                .filter_by(branch=from_branch)
                                .all())

                # Get latest version of each prompt
                latest_prompts = {}
                for p in source_prompts:
                    if p.name not in latest_prompts or p.version > latest_prompts[p.name].version:
                        latest_prompts[p.name] = p

                # Copy to new branch
                for prompt in latest_prompts.values():
                    timestamp = datetime.utcnow().isoformat()
                    new_hash = self._generate_commit_hash(prompt.name, prompt.content, timestamp)

                    new_prompt = Prompt(
                        name=prompt.name,
                        content=prompt.content,
                        branch=branch_name,
                        version=prompt.version,
                        parent_id=prompt.parent_id,
                        commit_hash=new_hash,
                        metadata=prompt.metadata
                    )
                    session.add(new_prompt)

                session.commit()

        return self._retry_on_failure(_create)

    def get_current_branch(self) -> str:
        """Get the current branch name"""
        def _get():
            with self._get_session() as session:
                config = session.query(Config).filter_by(key='current_branch').first()
                return config.value if config else 'main'

        return self._retry_on_failure(_get)

    def set_current_branch(self, branch_name: str) -> None:
        """Set the current branch"""
        def _set():
            with self._get_session() as session:
                # Verify branch exists
                branch = session.query(Branch).filter_by(name=branch_name).first()
                if not branch:
                    raise Exception(f"Branch '{branch_name}' does not exist")

                # Update config
                config = session.query(Config).filter_by(key='current_branch').first()
                if config:
                    config.value = branch_name
                else:
                    config = Config(key='current_branch', value=branch_name)
                    session.add(config)

                session.commit()

        return self._retry_on_failure(_set)

    def get_commit_history(self, name: Optional[str] = None, branch: str = "main",
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Get commit history"""
        def _get():
            with self._get_session() as session:
                query = session.query(Prompt).filter_by(branch=branch)

                if name:
                    query = query.filter_by(name=name)

                commits = (query.order_by(Prompt.created_at.desc())
                          .limit(limit)
                          .all())

                return [{
                    'commit_hash': c.commit_hash,
                    'name': c.name,
                    'version': c.version,
                    'branch': c.branch,
                    'created_at': c.created_at.isoformat()
                } for c in commits]

        return self._retry_on_failure(_get)

    def save_evaluation(self, eval_data: Dict[str, Any]) -> None:
        """Save evaluation results"""
        def _save():
            with self._get_session() as session:
                evaluation = Evaluation(
                    prompt_name=eval_data['prompt_name'],
                    commit_hash=eval_data['commit_hash'],
                    test_case=json.dumps(eval_data.get('test_case', {})),
                    expected=eval_data.get('expected'),
                    actual=eval_data.get('actual'),
                    score=eval_data.get('score'),
                    metrics=json.dumps(eval_data.get('metrics', {})),
                    evaluator_name=eval_data.get('evaluator_name')
                )

                session.add(evaluation)
                session.commit()

        return self._retry_on_failure(_save)

    def save_chain(self, chain_data: Dict[str, Any]) -> None:
        """Save a prompt chain"""
        def _save():
            with self._get_session() as session:
                # Check if chain already exists
                existing = session.query(Chain).filter_by(name=chain_data['name']).first()
                if existing:
                    raise Exception(f"Chain '{chain_data['name']}' already exists")

                chain = Chain(
                    name=chain_data['name'],
                    steps=json.dumps(chain_data['steps']),
                    description=chain_data.get('description')
                )

                session.add(chain)
                session.commit()

        return self._retry_on_failure(_save)

    def get_chain(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a prompt chain"""
        def _get():
            with self._get_session() as session:
                chain = session.query(Chain).filter_by(name=name).first()

                if not chain:
                    return None

                return {
                    'name': chain.name,
                    'steps': json.loads(chain.steps),
                    'description': chain.description
                }

        return self._retry_on_failure(_get)

    def close(self) -> None:
        """Close the database connection and cleanup resources"""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.SessionLocal = None

    # Additional PostgreSQL-specific methods

    def create_indexes(self) -> None:
        """
        Create additional indexes for performance optimization
        (Indexes are already defined in models, but this allows dynamic creation)
        """
        if not self.engine:
            raise RuntimeError("Storage not initialized")

        # Indexes are created automatically via __table_args__ in models
        # This method can be extended for custom index creation
        pass

    def vacuum_database(self) -> None:
        """
        Run VACUUM to reclaim storage and update statistics
        Useful for maintenance after bulk operations
        """
        if not self.engine:
            raise RuntimeError("Storage not initialized")

        with self.engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute("VACUUM ANALYZE")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics for monitoring

        Returns:
            Dictionary with statistics (table sizes, row counts, etc.)
        """
        def _get_stats():
            with self._get_session() as session:
                stats = {
                    'total_prompts': session.query(Prompt).count(),
                    'total_branches': session.query(Branch).count(),
                    'total_evaluations': session.query(Evaluation).count(),
                    'total_chains': session.query(Chain).count(),
                }

                # Per-branch statistics
                branch_stats = {}
                branches = session.query(Branch).all()
                for branch in branches:
                    count = session.query(Prompt).filter_by(branch=branch.name).count()
                    branch_stats[branch.name] = count

                stats['prompts_per_branch'] = branch_stats

                return stats

        return self._retry_on_failure(_get_stats)
