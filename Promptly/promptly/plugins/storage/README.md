# Promptly Storage Backends

Production-grade storage backend implementations for the Promptly prompt management system.

## Overview

Promptly supports 6 storage backends, from simple file-based storage to enterprise databases:

| Backend | Type | Dependencies | Use Case |
|---------|------|--------------|----------|
| **SQLite** | SQL Database | None | Development, single-user |
| **JSON File** | File System | None | Git-friendly, simple |
| **PostgreSQL** | SQL Database | sqlalchemy, psycopg2 | Production, multi-user |
| **MongoDB** | Document DB | pymongo | Flexible schema, scaling |
| **Redis** | In-Memory | redis | High-performance, caching |
| **Git** | Version Control | gitpython | Collaboration, versioning |

## Quick Start

### Installation

```bash
# Core (includes SQLite and JSON backends)
# No additional dependencies needed

# PostgreSQL backend
pip install sqlalchemy psycopg2-binary

# MongoDB backend
pip install pymongo

# Redis backend
pip install redis

# Git backend
pip install gitpython

# All backends
pip install -r Promptly/promptly/plugins/storage/requirements-storage.txt
```

### Usage

```python
from Promptly.promptly.plugins.storage import create_storage_backend

# SQLite (default)
storage = create_storage_backend('sqlite')
storage.init_storage('./promptly.db')

# PostgreSQL
storage = create_storage_backend('postgresql', pool_size=10)
storage.init_storage('postgresql://user:pass@localhost/promptly_db')

# MongoDB
storage = create_storage_backend('mongodb')
storage.init_storage('mongodb://localhost:27017/promptly_db')

# Redis
storage = create_storage_backend('redis', ttl_seconds=3600)
storage.init_storage('redis://localhost:6379/0')

# Git
storage = create_storage_backend(
    'git',
    remote_url='https://github.com/user/prompts.git',
    auto_push=True
)
storage.init_storage('./promptly_repo')
```

## Features by Backend

### SQLite
- ✅ ACID transactions
- ✅ Single-file database
- ✅ Zero configuration
- ✅ Built into Python
- ❌ Limited concurrency
- ❌ Single-server only

### JSON File
- ✅ Human-readable
- ✅ Git-friendly
- ✅ Easy backup
- ✅ No dependencies
- ❌ No indexing
- ❌ Slower at scale

### PostgreSQL
- ✅ Connection pooling
- ✅ ACID guarantees
- ✅ Advanced indexing
- ✅ High concurrency
- ✅ Replication support
- ⚠️ Requires server setup

### MongoDB
- ✅ Document-based
- ✅ Flexible schema
- ✅ Horizontal scaling
- ✅ Full-text search
- ✅ Replica sets
- ⚠️ Requires server setup

### Redis
- ✅ In-memory (ultra-fast)
- ✅ Sub-ms latency
- ✅ Pub/sub support
- ✅ TTL expiration
- ⚠️ Limited by RAM
- ⚠️ Data persistence optional

### Git
- ✅ Full version history
- ✅ Native git workflow
- ✅ Branching/merging
- ✅ Remote sync
- ⚠️ Slower operations
- ⚠️ Not for high-frequency updates

## Tools

### Migration Tool

Migrate data between backends:

```bash
python -m Promptly.promptly.plugins.storage.migrate \
    --from-backend sqlite \
    --from-path ./promptly.db \
    --to-backend postgresql \
    --to-path "postgresql://user:pass@localhost/promptly"
```

### Benchmark Tool

Compare backend performance:

```bash
python -m Promptly.promptly.plugins.storage.benchmark \
    --backends sqlite postgresql mongodb redis \
    --operations 1000 \
    --output results.json
```

## Configuration

Edit `integration_config.yaml`:

```yaml
storage:
  # Active backend
  backend: postgresql

  # PostgreSQL settings
  postgresql:
    connection_string: postgresql://promptly:password@localhost/promptly_db
    pool_size: 10
    max_overflow: 20

  # MongoDB settings
  mongodb:
    connection_string: mongodb://localhost:27017/promptly_db
    max_pool_size: 100

  # Redis settings
  redis:
    connection_string: redis://localhost:6379/0
    ttl_seconds: 3600  # Optional expiration

  # Git settings
  git:
    path: ./promptly_repo
    remote_url: https://github.com/user/prompts.git
    auto_push: false
```

## Performance Comparison

Typical performance (your results may vary):

| Backend | Writes/sec | Reads/sec | Latency |
|---------|-----------|-----------|---------|
| Redis | ~5000 | ~8000 | <1ms |
| PostgreSQL | ~500 | ~2000 | 1-5ms |
| MongoDB | ~400 | ~1500 | 2-10ms |
| SQLite | ~300 | ~1000 | 1-10ms |
| JSON File | ~100 | ~200 | 5-20ms |
| Git | ~10 | ~50 | 50-200ms |

Run benchmarks for your environment:
```bash
python -m Promptly.promptly.plugins.storage.benchmark --backends sqlite redis
```

## Backend Selection Guide

### Development
**Recommended: SQLite or JSON File**
- Zero setup
- Easy debugging
- Version control (JSON)

### Small Team (<10 users)
**Recommended: PostgreSQL**
- Good performance
- Multi-user support
- Production-ready

### Production (100+ users)
**Recommended: PostgreSQL + Redis**
- PostgreSQL for persistence
- Redis for caching
- Horizontal scaling ready

### High Performance
**Recommended: Redis**
- In-memory storage
- Sub-ms latency
- Optional persistence

### Collaboration
**Recommended: Git**
- Full version history
- Native git workflow
- Easy sharing

### Microservices
**Recommended: MongoDB**
- Flexible schema
- Easy horizontal scaling
- Document-centric

## File Structure

```
Promptly/promptly/plugins/storage/
├── __init__.py              # Backend factory and exports
├── base.py                  # Storage protocol definitions
├── sqlite.py               # SQLite implementation
├── json_file.py            # JSON file implementation
├── postgresql.py           # PostgreSQL implementation (new)
├── mongodb.py              # MongoDB implementation (new)
├── redis.py                # Redis implementation (new)
├── git.py                  # Git implementation (new)
├── migrate.py              # Migration tool
├── benchmark.py            # Performance benchmark
├── STORAGE_BACKENDS.md     # Complete documentation
├── README.md               # This file
└── requirements-storage.txt # Optional dependencies
```

## Documentation

See **[STORAGE_BACKENDS.md](STORAGE_BACKENDS.md)** for:
- Detailed backend documentation
- Configuration examples
- Deployment guides
- Troubleshooting
- Performance tuning
- Best practices

## Examples

### Example 1: Development Setup

```python
from Promptly.promptly.plugins.storage import create_storage_backend

# Use SQLite for quick start
storage = create_storage_backend('sqlite')
storage.init_storage('./dev.db')

# Save a prompt
storage.save_prompt({
    'name': 'summarizer',
    'content': 'Summarize: {text}',
    'branch': 'main'
})

# Get it back
prompt = storage.get_prompt('summarizer')
print(prompt['content'])
```

### Example 2: Production with PostgreSQL

```python
import os
from Promptly.promptly.plugins.storage import create_storage_backend

# Create PostgreSQL backend with connection pooling
storage = create_storage_backend(
    'postgresql',
    pool_size=20,
    max_overflow=40,
    max_retries=3
)

# Initialize from environment variable
db_url = os.getenv('DATABASE_URL')
storage.init_storage(db_url)

# Use as normal
storage.save_prompt({
    'name': 'prompt1',
    'content': 'Content',
    'branch': 'production'
})

# Get statistics
stats = storage.get_statistics()
print(f"Total prompts: {stats['total_prompts']}")
```

### Example 3: Caching with Redis

```python
from Promptly.promptly.plugins.storage import create_storage_backend

# Primary storage (PostgreSQL)
primary = create_storage_backend('postgresql')
primary.init_storage('postgresql://localhost/promptly')

# Cache layer (Redis with 1-hour TTL)
cache = create_storage_backend('redis', ttl_seconds=3600)
cache.init_storage('redis://localhost:6379/0')

# Write-through cache
def get_prompt_cached(name, branch='main'):
    # Try cache first
    prompt = cache.get_prompt(name, branch)
    if prompt:
        return prompt

    # Cache miss - get from primary
    prompt = primary.get_prompt(name, branch)
    if prompt:
        # Store in cache
        cache.save_prompt(prompt)

    return prompt
```

### Example 4: Git-based Collaboration

```python
from Promptly.promptly.plugins.storage import create_storage_backend

# Create Git backend with auto-sync
storage = create_storage_backend(
    'git',
    remote_url='https://github.com/team/prompts.git',
    author_name='Bot',
    author_email='bot@example.com',
    auto_push=True,  # Auto-push on save
    auto_pull=True   # Auto-pull before operations
)

storage.init_storage('./prompts_repo')

# Create feature branch
storage.create_branch('feature/new-prompts', from_branch='main')
storage.set_current_branch('feature/new-prompts')

# Make changes
storage.save_prompt({
    'name': 'new_prompt',
    'content': 'New prompt content',
    'branch': 'feature/new-prompts'
})

# Merge back to main
storage.set_current_branch('main')
result = storage.merge_branch('feature/new-prompts')
```

## Testing

Run tests for storage backends:

```bash
# Test basic functionality
python -m pytest Promptly/promptly/test_storage.py

# Test specific backend
python -m pytest Promptly/promptly/test_storage.py -k postgresql

# Run benchmarks
python -m Promptly.promptly.plugins.storage.benchmark \
    --backends sqlite postgresql mongodb redis git \
    --operations 100
```

## Migration Between Backends

```bash
# SQLite → PostgreSQL
python -m Promptly.promptly.plugins.storage.migrate \
    --from-backend sqlite --from-path ./promptly.db \
    --to-backend postgresql --to-path "postgresql://localhost/promptly"

# JSON → MongoDB
python -m Promptly.promptly.plugins.storage.migrate \
    --from-backend json --from-path ./data \
    --to-backend mongodb --to-path "mongodb://localhost/promptly"

# With specific branches
python -m Promptly.promptly.plugins.storage.migrate \
    --from-backend sqlite --from-path ./promptly.db \
    --to-backend redis --to-path "redis://localhost:6379/0" \
    --branches main production
```

## Troubleshooting

### Backend Not Available

```python
from Promptly.promptly.plugins.storage import get_available_backends

# Check which backends are available
backends = get_available_backends()
for name, available in backends.items():
    status = "✓" if available else "✗"
    print(f"{name}: {status}")
```

### Connection Issues

```python
try:
    storage = create_storage_backend('postgresql')
    storage.init_storage('postgresql://localhost/promptly')
except Exception as e:
    print(f"Connection failed: {e}")
    # Fall back to SQLite
    storage = create_storage_backend('sqlite')
    storage.init_storage('./promptly.db')
```

### Performance Issues

```bash
# Run benchmark to identify bottleneck
python -m Promptly.promptly.plugins.storage.benchmark \
    --backends sqlite postgresql \
    --operations 1000

# Check statistics
python -c "
from Promptly.promptly.plugins.storage import create_storage_backend
storage = create_storage_backend('postgresql')
storage.init_storage('postgresql://localhost/promptly')
print(storage.get_statistics())
"
```

## Contributing

To add a new storage backend:

1. Implement `BaseStorageBackend` protocol in `base.py`
2. Add implementation file (e.g., `mybackend.py`)
3. Update `__init__.py` to export the backend
4. Add tests
5. Update documentation
6. Add to benchmark suite

## License

See main Promptly LICENSE file.

## Support

- Documentation: [STORAGE_BACKENDS.md](STORAGE_BACKENDS.md)
- Issues: GitHub Issues
- Benchmarks: `python -m ...plugins.storage.benchmark --help`
- Migration: `python -m ...plugins.storage.migrate --help`
