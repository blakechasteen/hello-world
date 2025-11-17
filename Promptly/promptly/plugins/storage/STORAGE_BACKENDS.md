# Promptly Storage Backends

Comprehensive guide to all available storage backends for Promptly.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Storage Backends](#storage-backends)
  - [SQLite](#sqlite)
  - [JSON File](#json-file)
  - [PostgreSQL](#postgresql)
  - [MongoDB](#mongodb)
  - [Redis](#redis)
  - [Git](#git)
- [Backend Comparison](#backend-comparison)
- [Migration](#migration)
- [Performance](#performance)
- [Deployment Guide](#deployment-guide)

---

## Overview

Promptly supports multiple storage backends to accommodate different deployment scenarios, from local development to production clusters. Each backend implements the same `StorageBackend` protocol, ensuring consistent behavior across all storage options.

### Available Backends

| Backend | Dependencies | Use Case | Complexity |
|---------|-------------|----------|------------|
| SQLite | None (built-in) | Development, single-user | Low |
| JSON File | None (built-in) | Git-friendly, simple | Low |
| PostgreSQL | sqlalchemy, psycopg2 | Production, multi-user | Medium |
| MongoDB | pymongo | Document-centric, flexible schema | Medium |
| Redis | redis | High-performance, caching | Medium |
| Git | gitpython | Version control, collaboration | High |

---

## Quick Start

### Using the Configuration File

Edit `integration_config.yaml`:

```yaml
storage:
  backend: postgresql  # Choose backend

  postgresql:
    connection_string: postgresql://user:pass@localhost/promptly_db
    pool_size: 10
```

### Programmatic Usage

```python
from Promptly.promptly.plugins.storage import create_storage_backend

# Create backend
storage = create_storage_backend('postgresql', pool_size=10)

# Initialize
storage.init_storage('postgresql://user:pass@localhost/promptly_db')

# Use the backend
commit_hash = storage.save_prompt({
    'name': 'my_prompt',
    'content': 'Summarize the following text: {text}',
    'branch': 'main',
    'metadata': {'author': 'user@example.com'}
})

# Get prompt
prompt = storage.get_prompt('my_prompt', branch='main')

# Cleanup
storage.close()
```

---

## Storage Backends

### SQLite

**Default backend** - No additional dependencies required.

#### Features
- Single-file database
- ACID transactions
- Zero configuration
- Perfect for development and single-user deployments

#### Installation
No installation required - SQLite is built into Python.

#### Configuration

```yaml
storage:
  backend: sqlite
  sqlite:
    path: ./promptly.db
```

#### Usage

```python
from Promptly.promptly.plugins.storage import SQLiteStorage

storage = SQLiteStorage()
storage.init_storage('./promptly.db')
```

#### Pros
- No setup required
- Fast for small datasets
- Single file for easy backup
- Built-in to Python

#### Cons
- Not suitable for high concurrency
- Limited scalability
- Single-server only

#### Best For
- Development
- Single-user applications
- Small teams (<5 users)
- Embedded applications

---

### JSON File

**Git-friendly** - Store prompts as JSON files.

#### Features
- Human-readable storage
- Git-friendly (easy to track changes)
- No database dependencies
- Simple backup (just copy directory)

#### Installation
No installation required - uses Python's built-in JSON.

#### Configuration

```yaml
storage:
  backend: json
  json:
    path: ./promptly_data
```

#### File Structure

```
promptly_data/
├── prompts/
│   ├── prompt1.json
│   └── prompt2.json
├── branches/
│   ├── main.json
│   └── develop.json
├── chains/
│   └── chain1.json
└── .promptly/
    ├── config.json
    └── index.json
```

#### Usage

```python
from Promptly.promptly.plugins.storage import JSONStorage

storage = JSONStorage()
storage.init_storage('./promptly_data')
```

#### Pros
- Human-readable
- Works well with git
- Easy to inspect/debug
- No database setup

#### Cons
- Slower for large datasets
- No built-in indexing
- File system limitations
- Not suitable for high concurrency

#### Best For
- Version-controlled prompts
- Sharing prompts via git
- Simple deployments
- Documentation-focused workflows

---

### PostgreSQL

**Production-grade** - Enterprise database with full ACID guarantees.

#### Features
- Connection pooling
- Transaction support
- Advanced indexing
- High concurrency
- Replication support
- Full ACID guarantees

#### Installation

```bash
pip install sqlalchemy psycopg2-binary
```

For production, also install PostgreSQL server:

```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql

# Start PostgreSQL
sudo systemctl start postgresql  # Linux
brew services start postgresql   # macOS
```

#### Configuration

```yaml
storage:
  backend: postgresql
  postgresql:
    connection_string: postgresql://user:password@localhost:5432/promptly_db
    pool_size: 10
    max_overflow: 20
    pool_timeout: 30
    max_retries: 3
```

#### Database Setup

```sql
-- Create database
CREATE DATABASE promptly_db;

-- Create user
CREATE USER promptly WITH PASSWORD 'your_secure_password';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE promptly_db TO promptly;
```

#### Usage

```python
from Promptly.promptly.plugins.storage import PostgreSQLStorage

storage = PostgreSQLStorage(
    pool_size=10,
    max_overflow=20,
    max_retries=3
)

storage.init_storage('postgresql://promptly:password@localhost/promptly_db')

# Get statistics
stats = storage.get_statistics()
print(f"Total prompts: {stats['total_prompts']}")

# Vacuum database (maintenance)
storage.vacuum_database()
```

#### Connection String Formats

```
# Local
postgresql://user:password@localhost/database

# Remote
postgresql://user:password@host.example.com:5432/database

# SSL
postgresql://user:password@host.example.com/database?sslmode=require

# Unix socket
postgresql:///database?host=/var/run/postgresql
```

#### Pros
- Production-ready
- Excellent performance at scale
- Advanced features (JSON columns, full-text search)
- Strong data consistency
- Mature ecosystem

#### Cons
- Requires database server
- More complex setup
- Higher resource usage

#### Best For
- Production deployments
- Multi-user applications
- Large teams
- Enterprise environments
- High-traffic applications

---

### MongoDB

**Document-oriented** - Flexible schema for rapid development.

#### Features
- Document-based storage (JSON-native)
- Flexible schema evolution
- Horizontal scaling (sharding)
- Replica sets for high availability
- Full-text search
- Aggregation pipeline

#### Installation

```bash
pip install pymongo
```

Install MongoDB server:

```bash
# Ubuntu/Debian
sudo apt-get install mongodb

# macOS
brew tap mongodb/brew
brew install mongodb-community

# Start MongoDB
sudo systemctl start mongod  # Linux
brew services start mongodb-community  # macOS
```

#### Configuration

```yaml
storage:
  backend: mongodb
  mongodb:
    connection_string: mongodb://localhost:27017/promptly_db
    max_pool_size: 100
    timeout_ms: 5000
    retry_writes: true
    replica_set: null  # Set if using replica sets
```

#### Usage

```python
from Promptly.promptly.plugins.storage import MongoDBStorage

storage = MongoDBStorage(
    max_pool_size=100,
    replica_set='rs0'  # If using replica sets
)

storage.init_storage('mongodb://localhost:27017/promptly_db')

# Full-text search
results = storage.full_text_search('summarization', branch='main', limit=10)

# Export branch
prompts = storage.export_branch_to_json('main')

# Statistics
stats = storage.get_statistics()
print(f"Database size: {stats['database_size_bytes']}")
```

#### Connection String Formats

```
# Local
mongodb://localhost:27017/database

# Authenticated
mongodb://user:password@localhost:27017/database

# Replica set
mongodb://host1:27017,host2:27017,host3:27017/database?replicaSet=rs0

# MongoDB Atlas
mongodb+srv://user:password@cluster.mongodb.net/database
```

#### Pros
- Flexible schema
- Easy horizontal scaling
- Native JSON storage
- Rich query language
- Good for rapidly changing data models

#### Cons
- Eventual consistency (in some configs)
- More memory usage
- Requires MongoDB server

#### Best For
- Rapid development
- Evolving schemas
- Document-centric applications
- Microservices architectures
- Real-time analytics

---

### Redis

**In-memory** - Ultra-fast storage with optional persistence.

#### Features
- In-memory storage (sub-millisecond latency)
- Optional persistence (RDB + AOF)
- Pub/sub for real-time notifications
- TTL support for temporary prompts
- Redis Streams for audit logs
- Sorted sets for version ordering

#### Installation

```bash
pip install redis
```

Install Redis server:

```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Start Redis
sudo systemctl start redis  # Linux
brew services start redis   # macOS
```

#### Configuration

```yaml
storage:
  backend: redis
  redis:
    connection_string: redis://localhost:6379/0
    max_connections: 50
    socket_timeout: 5
    ttl_seconds: null  # Set for auto-expiration
```

#### Usage

```python
from Promptly.promptly.plugins.storage import RedisStorage

storage = RedisStorage(
    max_connections=50,
    ttl_seconds=3600  # Prompts expire after 1 hour
)

storage.init_storage('redis://localhost:6379/0')

# Set TTL for specific prompt
storage.set_prompt_ttl('temp_prompt', 'main', ttl_seconds=300)

# Subscribe to events
def on_prompt_saved(message):
    print(f"Prompt saved: {message}")

storage.subscribe_to_events('prompt_saved', on_prompt_saved)

# Backup to disk
storage.backup_to_rdb()

# Statistics
stats = storage.get_statistics()
print(f"Memory used: {stats['used_memory_human']}")
```

#### Connection String Formats

```
# Local
redis://localhost:6379/0

# Authenticated
redis://:password@localhost:6379/0

# SSL/TLS
rediss://localhost:6380/0

# With parameters
redis://localhost:6379/0?socket_timeout=5&socket_connect_timeout=5
```

#### Redis Persistence

Configure persistence in `redis.conf`:

```
# RDB snapshots
save 900 1      # Save after 900s if 1 key changed
save 300 10     # Save after 300s if 10 keys changed
save 60 10000   # Save after 60s if 10000 keys changed

# AOF (Append-Only File)
appendonly yes
appendfsync everysec  # Sync every second
```

#### Pros
- Extremely fast (in-memory)
- Sub-millisecond latency
- Pub/sub for real-time features
- Great for caching
- TTL support

#### Cons
- Limited by available RAM
- Data loss risk (if not configured for persistence)
- Not a primary database (best as cache)

#### Best For
- Caching layer
- Session storage
- Real-time applications
- Temporary prompts
- High-performance requirements
- Development/testing

---

### Git

**Version control** - Native git integration for full history.

#### Features
- True git repository
- Native branching and merging
- Full git history (log, blame, diff)
- Remote repository support (GitHub, GitLab, etc.)
- Conflict resolution
- Git tags for versioning
- Commit hooks integration

#### Installation

```bash
pip install gitpython
```

Git must be installed on the system:

```bash
# Ubuntu/Debian
sudo apt-get install git

# macOS
brew install git
```

#### Configuration

```yaml
storage:
  backend: git
  git:
    path: ./promptly_repo
    remote_url: https://github.com/user/prompts.git
    author_name: Promptly
    author_email: promptly@example.com
    auto_push: false
    auto_pull: false
```

#### Usage

```python
from Promptly.promptly.plugins.storage import GitStorage

storage = GitStorage(
    remote_url='https://github.com/user/prompts.git',
    author_name='Your Name',
    author_email='you@example.com',
    auto_push=True  # Automatically push to remote
)

# Clone from remote or init local
storage.init_storage('./promptly_repo')

# Merge branches
result = storage.merge_branch('feature-branch', strategy='ours')

# Push to remote
storage.push_to_remote('origin', 'main')

# Pull from remote
storage.pull_from_remote('origin', 'main')

# Create tag
storage.create_tag('v1.0.0', message='Release 1.0.0')

# Get diff
diff = storage.get_diff('abc123', 'def456', prompt_name='my_prompt')
```

#### Remote Repository Setup

```bash
# GitHub
gh repo create prompts --private
git remote add origin https://github.com/user/prompts.git

# GitLab
# Create repo in GitLab UI
git remote add origin https://gitlab.com/user/prompts.git

# SSH keys
git remote add origin git@github.com:user/prompts.git
```

#### Pros
- Full version control history
- Native git tooling
- Easy collaboration
- Built-in conflict resolution
- Works with existing git workflows
- Free hosting (GitHub, GitLab)

#### Cons
- Slower than database backends
- Not suitable for high-frequency updates
- Requires git knowledge
- Merge conflicts

#### Best For
- Collaborative prompt engineering
- Prompts as code
- Audit trail requirements
- Integration with existing git workflows
- Documentation-heavy projects
- Open-source prompt libraries

---

## Backend Comparison

### Performance

Based on typical workloads (run `benchmark.py` for your environment):

| Backend | Write (ops/sec) | Read (ops/sec) | List (ops/sec) | Latency |
|---------|----------------|----------------|----------------|---------|
| Redis | ~5000 | ~8000 | ~1000 | <1ms |
| PostgreSQL | ~500 | ~2000 | ~200 | 1-5ms |
| MongoDB | ~400 | ~1500 | ~150 | 2-10ms |
| SQLite | ~300 | ~1000 | ~100 | 1-10ms |
| JSON File | ~100 | ~200 | ~50 | 5-20ms |
| Git | ~10 | ~50 | ~20 | 50-200ms |

### Scalability

| Backend | Max Prompts | Concurrent Users | Horizontal Scaling |
|---------|------------|------------------|-------------------|
| Redis | 100K-1M | 1000+ | Cluster mode |
| PostgreSQL | 10M+ | 100+ | Replication, partitioning |
| MongoDB | 10M+ | 100+ | Sharding |
| SQLite | 100K | 1-10 | No |
| JSON File | 10K | 1-5 | No |
| Git | 10K | 1-10 | Remote repos |

### Feature Matrix

| Feature | SQLite | JSON | PostgreSQL | MongoDB | Redis | Git |
|---------|--------|------|------------|---------|-------|-----|
| ACID Transactions | ✅ | ❌ | ✅ | ⚠️ | ⚠️ | ✅ |
| Full-text Search | ❌ | ❌ | ✅ | ✅ | ⚠️ | ❌ |
| Horizontal Scaling | ❌ | ❌ | ⚠️ | ✅ | ✅ | ❌ |
| Connection Pooling | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| Real-time Pub/Sub | ❌ | ❌ | ✅ | ⚠️ | ✅ | ❌ |
| TTL/Expiration | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Git Integration | ⚠️ | ✅ | ❌ | ❌ | ❌ | ✅ |
| Version Control | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ✅ |
| Remote Sync | ❌ | ⚠️ | ✅ | ✅ | ✅ | ✅ |

Legend: ✅ Full support | ⚠️ Partial support | ❌ Not supported

---

## Migration

Migrate between backends using the migration tool:

### Command Line

```bash
# Migrate from SQLite to PostgreSQL
python -m Promptly.promptly.plugins.storage.migrate \
    --from-backend sqlite \
    --from-path ./promptly.db \
    --to-backend postgresql \
    --to-path "postgresql://user:pass@localhost/promptly"

# Migrate specific branches
python -m Promptly.promptly.plugins.storage.migrate \
    --from-backend json \
    --from-path ./data \
    --to-backend mongodb \
    --to-path "mongodb://localhost/promptly" \
    --branches main production

# Dry run
python -m Promptly.promptly.plugins.storage.migrate \
    --from-backend sqlite \
    --from-path ./promptly.db \
    --to-backend redis \
    --to-path "redis://localhost:6379/0" \
    --dry-run
```

### Programmatic

```python
from Promptly.promptly.plugins.storage.migrate import migrate_storage

stats = migrate_storage(
    from_backend='sqlite',
    from_path='./promptly.db',
    to_backend='postgresql',
    to_path='postgresql://user:pass@localhost/promptly',
    branches=['main', 'production'],
    include_evaluations=True
)

print(f"Migrated {stats['prompts_migrated']} prompts")
```

---

## Performance

Benchmark storage backends:

```bash
# Benchmark all available backends
python -m Promptly.promptly.plugins.storage.benchmark \
    --backends sqlite postgresql mongodb redis \
    --operations 1000 \
    --output benchmark_results.json

# Quick benchmark
python -m Promptly.promptly.plugins.storage.benchmark \
    --backends sqlite redis \
    --operations 100
```

### Optimization Tips

#### PostgreSQL
- Use connection pooling
- Create appropriate indexes
- Tune `shared_buffers` and `work_mem`
- Regular VACUUM

#### MongoDB
- Create indexes on frequently queried fields
- Use projection to limit returned fields
- Enable replica sets for read scaling
- Shard for write scaling

#### Redis
- Configure appropriate `maxmemory` policy
- Use pipelining for batch operations
- Enable persistence (RDB + AOF)
- Monitor memory usage

#### SQLite
- Use WAL mode for better concurrency
- Increase cache size
- Use transactions for batch writes
- Regular VACUUM

---

## Deployment Guide

### Local Development

**Recommended: SQLite or JSON File**

```yaml
storage:
  backend: sqlite
  sqlite:
    path: ./promptly.db
```

### Small Team (<10 users)

**Recommended: PostgreSQL**

```yaml
storage:
  backend: postgresql
  postgresql:
    connection_string: postgresql://promptly:password@localhost/promptly_db
    pool_size: 5
```

### Production (100+ users)

**Recommended: PostgreSQL with Redis Cache**

Primary storage:
```yaml
storage:
  backend: postgresql
  postgresql:
    connection_string: postgresql://promptly:password@db-server/promptly_db
    pool_size: 20
    max_overflow: 40
```

Cache layer (separate configuration):
```python
# Use Redis for caching frequently accessed prompts
cache = create_storage_backend('redis', ttl_seconds=3600)
cache.init_storage('redis://cache-server:6379/0')
```

### Microservices

**Recommended: MongoDB or PostgreSQL**

```yaml
storage:
  backend: mongodb
  mongodb:
    connection_string: mongodb://mongo1:27017,mongo2:27017,mongo3:27017/promptly?replicaSet=rs0
    max_pool_size: 100
```

### High-Performance Requirements

**Recommended: Redis with PostgreSQL Backup**

```yaml
# Primary (Redis)
storage:
  backend: redis
  redis:
    connection_string: redis://redis-server:6379/0
    max_connections: 100

# Periodic backup to PostgreSQL
# (Implement sync job)
```

### Collaborative Prompt Engineering

**Recommended: Git**

```yaml
storage:
  backend: git
  git:
    path: ./promptly_repo
    remote_url: https://github.com/team/prompts.git
    author_name: CI Bot
    author_email: ci@example.com
    auto_push: true
    auto_pull: true
```

### Docker Deployment

```dockerfile
# PostgreSQL backend
services:
  promptly:
    image: promptly:latest
    environment:
      STORAGE_BACKEND: postgresql
      STORAGE_PATH: postgresql://promptly:password@postgres/promptly_db
    depends_on:
      - postgres

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: promptly_db
      POSTGRES_USER: promptly
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Kubernetes Deployment

```yaml
# PostgreSQL with persistent volume
apiVersion: apps/v1
kind: Deployment
metadata:
  name: promptly
spec:
  template:
    spec:
      containers:
      - name: promptly
        env:
        - name: STORAGE_BACKEND
          value: postgresql
        - name: STORAGE_PATH
          valueFrom:
            secretKeyRef:
              name: promptly-secrets
              key: database-url
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
```

---

## Troubleshooting

### PostgreSQL

**Connection refused:**
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Check connection settings
psql -h localhost -U promptly -d promptly_db
```

**Too many connections:**
```sql
-- Check current connections
SELECT count(*) FROM pg_stat_activity;

-- Increase max_connections in postgresql.conf
max_connections = 200
```

### MongoDB

**Authentication failed:**
```javascript
// Create user in MongoDB
use promptly_db
db.createUser({
  user: "promptly",
  pwd: "password",
  roles: ["readWrite"]
})
```

**Slow queries:**
```javascript
// Enable profiling
db.setProfilingLevel(2)

// Check slow queries
db.system.profile.find().sort({ts: -1}).limit(10)

// Create indexes
db.prompts.createIndex({ name: 1, branch: 1 })
```

### Redis

**Out of memory:**
```bash
# Check memory usage
redis-cli INFO memory

# Set maxmemory policy in redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
```

**Connection timeout:**
```bash
# Increase timeout in redis.conf
timeout 300

# Check network connectivity
redis-cli -h localhost -p 6379 ping
```

---

## Additional Resources

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [Redis Documentation](https://redis.io/documentation)
- [Git Documentation](https://git-scm.com/doc)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [PyMongo Documentation](https://pymongo.readthedocs.io/)

---

## Support

For issues or questions:
- Open an issue on GitHub
- Check the documentation
- Run benchmarks to identify bottlenecks
- Use migration tool to switch backends if needed
