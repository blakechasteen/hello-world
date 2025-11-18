"""Database connections and session management"""
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from neo4j import AsyncGraphDatabase
from qdrant_client import QdrantClient
import redis.asyncio as redis

from backend.config import settings

# SQLAlchemy Base
Base = declarative_base()

# PostgreSQL Engine
engine = create_async_engine(
    settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    echo=settings.DEBUG,
    pool_pre_ping=True,
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Neo4j Driver
neo4j_driver = AsyncGraphDatabase.driver(
    settings.NEO4J_URI,
    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
)


async def get_neo4j():
    """Dependency for getting Neo4j session"""
    async with neo4j_driver.session() as session:
        yield session


# Qdrant Client
qdrant_client = QdrantClient(
    host=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT,
)


def get_qdrant():
    """Dependency for getting Qdrant client"""
    return qdrant_client


# Redis Client
redis_client = None


async def get_redis():
    """Dependency for getting Redis client"""
    global redis_client
    if redis_client is None:
        redis_client = await redis.from_url(
            settings.REDIS_URL,
            decode_responses=True
        )
    return redis_client


async def close_connections():
    """Close all database connections gracefully"""
    # Close SQLAlchemy
    await engine.dispose()

    # Close Neo4j
    await neo4j_driver.close()

    # Close Redis
    if redis_client:
        await redis_client.close()
