"""
Promptly Storage Backend Plugins

Available storage backends:
- SQLiteStorage: SQLite database (default, built-in)
- JSONStorage: JSON file storage (git-friendly)
- PostgreSQLStorage: PostgreSQL with connection pooling (requires: sqlalchemy, psycopg2-binary)
- MongoDBStorage: MongoDB document storage (requires: pymongo)
- RedisStorage: Redis in-memory storage (requires: redis)
- GitStorage: Git repository with version control (requires: gitpython)
"""

from .sqlite import SQLiteStorage
from .json_file import JSONStorage

# Import optional backends with graceful degradation
try:
    from .postgresql import PostgreSQLStorage
    POSTGRESQL_AVAILABLE = True
except ImportError:
    PostgreSQLStorage = None
    POSTGRESQL_AVAILABLE = False

try:
    from .mongodb import MongoDBStorage
    MONGODB_AVAILABLE = True
except ImportError:
    MongoDBStorage = None
    MONGODB_AVAILABLE = False

try:
    from .redis import RedisStorage
    REDIS_AVAILABLE = True
except ImportError:
    RedisStorage = None
    REDIS_AVAILABLE = False

try:
    from .git import GitStorage
    GIT_AVAILABLE = True
except ImportError:
    GitStorage = None
    GIT_AVAILABLE = False

# Export all available backends
__all__ = [
    'SQLiteStorage',
    'JSONStorage',
    'PostgreSQLStorage',
    'MongoDBStorage',
    'RedisStorage',
    'GitStorage',
    'POSTGRESQL_AVAILABLE',
    'MONGODB_AVAILABLE',
    'REDIS_AVAILABLE',
    'GIT_AVAILABLE',
    'get_available_backends',
    'create_storage_backend'
]


def get_available_backends() -> dict:
    """
    Get list of available storage backends

    Returns:
        Dict mapping backend names to availability status
    """
    return {
        'sqlite': True,  # Always available
        'json': True,    # Always available
        'postgresql': POSTGRESQL_AVAILABLE,
        'mongodb': MONGODB_AVAILABLE,
        'redis': REDIS_AVAILABLE,
        'git': GIT_AVAILABLE
    }


def create_storage_backend(backend_type: str, **kwargs):
    """
    Factory function to create storage backend instances

    Args:
        backend_type: Type of backend ('sqlite', 'json', 'postgresql', 'mongodb', 'redis', 'git')
        **kwargs: Backend-specific configuration options

    Returns:
        Storage backend instance

    Raises:
        ValueError: If backend type is invalid or not available

    Example:
        # SQLite (default)
        storage = create_storage_backend('sqlite')

        # PostgreSQL with connection pooling
        storage = create_storage_backend(
            'postgresql',
            pool_size=10,
            max_overflow=20
        )

        # MongoDB with replica set
        storage = create_storage_backend(
            'mongodb',
            replica_set='rs0'
        )

        # Redis with TTL
        storage = create_storage_backend(
            'redis',
            ttl_seconds=3600
        )

        # Git with remote
        storage = create_storage_backend(
            'git',
            remote_url='https://github.com/user/prompts.git',
            auto_push=True
        )
    """
    backend_type = backend_type.lower()

    if backend_type == 'sqlite':
        return SQLiteStorage(**kwargs)

    elif backend_type == 'json':
        return JSONStorage(**kwargs)

    elif backend_type == 'postgresql':
        if not POSTGRESQL_AVAILABLE:
            raise ValueError(
                "PostgreSQL backend not available. "
                "Install dependencies: pip install sqlalchemy psycopg2-binary"
            )
        return PostgreSQLStorage(**kwargs)

    elif backend_type == 'mongodb':
        if not MONGODB_AVAILABLE:
            raise ValueError(
                "MongoDB backend not available. "
                "Install dependencies: pip install pymongo"
            )
        return MongoDBStorage(**kwargs)

    elif backend_type == 'redis':
        if not REDIS_AVAILABLE:
            raise ValueError(
                "Redis backend not available. "
                "Install dependencies: pip install redis"
            )
        return RedisStorage(**kwargs)

    elif backend_type == 'git':
        if not GIT_AVAILABLE:
            raise ValueError(
                "Git backend not available. "
                "Install dependencies: pip install gitpython"
            )
        return GitStorage(**kwargs)

    else:
        available = [k for k, v in get_available_backends().items() if v]
        raise ValueError(
            f"Invalid backend type: '{backend_type}'. "
            f"Available backends: {', '.join(available)}"
        )
