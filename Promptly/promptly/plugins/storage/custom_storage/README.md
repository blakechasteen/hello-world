# Advanced Custom Storage Backends

This directory contains advanced custom storage backend implementations that demonstrate the extensibility and power of Promptly's storage system.

## 📚 Table of Contents

- [Overview](#overview)
- [Available Backends](#available-backends)
- [Architecture Diagrams](#architecture-diagrams)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Migration Guide](#migration-guide)
- [Performance Comparison](#performance-comparison)
- [Security Best Practices](#security-best-practices)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

Promptly's storage backend system is designed for **maximum extensibility**. This directory showcases four advanced implementations that go beyond the built-in backends (SQLite, JSON, PostgreSQL, MongoDB, Redis, Git):

1. **S3/MinIO Storage** - Cloud object storage with CDN integration
2. **Hybrid Storage** - Multi-tier architecture for cost optimization
3. **Blockchain Storage** - Immutable history with cryptographic proof
4. **Custom DB Template** - Step-by-step tutorial for creating your own

Each backend demonstrates different architectural patterns and use cases, serving both as production-ready implementations and educational examples.

## 🗄️ Available Backends

### 1. S3/MinIO Storage Backend

**File:** `s3.py`

Production-grade object storage for cloud-native deployments.

#### Features
- ✅ S3/MinIO object versioning
- ✅ Metadata as S3 tags and object metadata
- ✅ Lifecycle policies for automatic archiving
- ✅ CloudFront CDN integration for global distribution
- ✅ Presigned URLs for secure temporary access
- ✅ Cross-region replication for disaster recovery
- ✅ Server-side encryption (AES256)
- ✅ Connection pooling and automatic retries

#### Use Cases
- Multi-region deployments
- Large-scale prompt repositories
- CDN-backed prompt delivery
- Disaster recovery requirements
- Compliance with data retention policies

#### Architecture
```
┌──────────────────────────────────────────────┐
│         Application (Promptly)                │
└──────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌──────────────┐       ┌──────────────┐
│     S3       │       │  CloudFront  │
│  (Storage)   │◄──────│    (CDN)     │
└──────────────┘       └──────────────┘
        │                       │
        │                       ▼
        │              ┌──────────────┐
        │              │   Global     │
        │              │   Users      │
        │              └──────────────┘
        │
        ├──► Lifecycle Policies
        │      • 90 days → Standard-IA
        │      • 180 days → Glacier
        │
        └──► Cross-Region Replication
               • Disaster Recovery
               • Multi-Region Access
```

---

### 2. Hybrid Multi-Tier Storage Backend

**File:** `hybrid.py`

Intelligent multi-tier storage for optimal cost and performance.

#### Features
- ✅ Hot tier (Redis) - Sub-millisecond latency
- ✅ Warm tier (PostgreSQL) - Moderate latency, ACID guarantees
- ✅ Cold tier (S3) - Long-term archive, lowest cost
- ✅ Automatic tiering based on access patterns
- ✅ Transparent read-through cache
- ✅ Write-through to all tiers
- ✅ Cost optimization algorithms
- ✅ Access frequency tracking

#### Use Cases
- Large prompt repositories with variable access patterns
- Cost-sensitive deployments
- High-performance requirements for popular prompts
- Long-term archival needs

#### Architecture
```
┌──────────────────────────────────────────────┐
│         READ REQUEST (Optimized Path)         │
└──────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    ┌──────┐   ┌──────┐   ┌──────┐
    │ HOT  │   │ WARM │   │ COLD │
    │Redis │   │ PG   │   │  S3  │
    │<1ms  │   │1-5ms │   │50ms+ │
    └──────┘   └──────┘   └──────┘
        │           │           │
        └───────────┴───────────┘
                    │
            Automatic Promotion
            Based on Access Patterns

┌──────────────────────────────────────────────┐
│        WRITE REQUEST (Write-Through)          │
└──────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    Redis       PostgreSQL      S3
    (Hot)         (Warm)      (Cold)
     └─────────────┴────────────┘
          All tiers updated
          simultaneously

Tiering Logic:
─────────────
HOT:  Last access <24h AND access count >10
WARM: Last access <30d AND access count >2
COLD: Everything else
```

#### Performance Characteristics

| Tier | Storage | Latency | Cost | Capacity |
|------|---------|---------|------|----------|
| Hot  | Redis   | <1ms    | $$$  | 1K prompts |
| Warm | PostgreSQL | 1-5ms | $$   | 10K prompts |
| Cold | S3      | 50ms+   | $    | Unlimited |

---

### 3. Blockchain + IPFS Storage Backend

**File:** `blockchain.py`

Immutable, verifiable prompt history with cryptographic proof.

#### Features
- ✅ Blockchain for metadata (Ethereum/Polygon/Hyperledger)
- ✅ IPFS for content storage (decentralized)
- ✅ Smart contract-based versioning
- ✅ Cryptographic proof of authenticity
- ✅ Timestamp anchoring for provenance
- ✅ Content addressing (hash-based)
- ✅ Multi-signature support
- ✅ Immutable audit trail

#### Use Cases
- Regulatory compliance (immutable audit trail)
- Intellectual property protection
- Collaborative prompt engineering with transparency
- Version control with cryptographic proof
- Decentralized prompt repositories

#### Architecture
```
┌──────────────────────────────────────────────┐
│         Promptly Application                  │
└──────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌────────────────┐      ┌────────────────┐
│   Blockchain   │      │      IPFS      │
│   (Metadata)   │      │   (Content)    │
└────────────────┘      └────────────────┘
        │                       │
        ├─► Version Registry    ├─► Content Storage
        ├─► Access Control      ├─► Distributed Nodes
        ├─► Timestamp Proof     ├─► Pin Services
        └─► Event Logs          └─► Gateway Access

Smart Contract Structure:
────────────────────────
contract PromptRegistry {
  struct Prompt {
    string ipfsHash;      // Content on IPFS
    uint256 timestamp;    // Block timestamp
    address author;       // Creator address
    uint256 version;      // Version number
  }

  mapping(string => Prompt) prompts;

  event PromptRegistered(
    string commitHash,
    address author,
    string ipfsHash,
    uint256 timestamp
  );
}

Verification Flow:
─────────────────
1. Retrieve metadata from blockchain
2. Get IPFS hash from smart contract
3. Fetch content from IPFS
4. Verify commit hash matches
5. Return cryptographic proof
```

#### Cryptographic Proof Example
```json
{
  "proof_type": "blockchain_timestamp",
  "blockchain": "ethereum",
  "commit_hash": "a3f2b9c1e4d5",
  "transaction_hash": "0x9b2f...",
  "ipfs_hash": "QmX4t...",
  "timestamp": "2025-11-17T10:30:00Z",
  "author": "0x742d...",
  "verification_url": "https://ipfs.io/ipfs/QmX4t...",
  "blockchain_explorer": "https://etherscan.io/tx/0x9b2f..."
}
```

---

### 4. Custom Database Template

**File:** `custom_db.py`

Comprehensive tutorial and template for building your own backends.

#### Features
- ✅ Step-by-step tutorial
- ✅ Complete working example (Cassandra)
- ✅ Inline documentation
- ✅ Testing guide
- ✅ Best practices
- ✅ Common patterns
- ✅ Troubleshooting tips

#### Covered Topics
- Protocol implementation
- Schema design
- Error handling
- Connection pooling
- Performance optimization
- Testing strategies
- Migration patterns

---

## 🏗️ Architecture Diagrams

### Storage Backend Protocol

All backends implement the `BaseStorageBackend` protocol:

```
┌─────────────────────────────────────────────┐
│       BaseStorageBackend (Protocol)          │
├─────────────────────────────────────────────┤
│ Required Properties:                         │
│  • name: str                                 │
│  • description: str                          │
├─────────────────────────────────────────────┤
│ Required Methods:                            │
│  • init_storage(path)                        │
│  • save_prompt(data) → commit_hash          │
│  • get_prompt(name, ...) → Dict             │
│  • list_prompts(branch) → List              │
│  • create_branch(name, from_branch)         │
│  • get_current_branch() → str               │
│  • set_current_branch(name)                  │
│  • get_commit_history(...) → List           │
│  • save_evaluation(data)                     │
│  • save_chain(data)                          │
│  • get_chain(name) → Dict                   │
│  • close()                                   │
└─────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┬──────────┐
        ▼           ▼           ▼          ▼
┌──────────┐  ┌──────────┐  ┌───────┐  ┌──────┐
│    S3    │  │  Hybrid  │  │ Block │  │Custom│
│ Storage  │  │ Storage  │  │ chain │  │  DB  │
└──────────┘  └──────────┘  └───────┘  └──────┘
```

### Data Flow Architecture

```
┌──────────────────────────────────────────────┐
│          Promptly Application                 │
├──────────────────────────────────────────────┤
│  • Prompt Engineering                         │
│  • Version Control                            │
│  • Evaluation & Testing                       │
│  • Chain Processing                           │
└──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────┐
│       Storage Backend Abstraction             │
├──────────────────────────────────────────────┤
│  • Protocol-based interface                   │
│  • Pluggable implementations                  │
│  • Graceful degradation                       │
└──────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┬──────────┐
        ▼           ▼           ▼          ▼
┌──────────┐  ┌──────────┐  ┌───────┐  ┌──────┐
│ Built-in │  │  Cloud   │  │ Novel │  │Custom│
│ Backends │  │ Backends │  │ Techs │  │  DB  │
├──────────┤  ├──────────┤  ├───────┤  ├──────┤
│ SQLite   │  │    S3    │  │Blockchn│ │Cassndra│
│ JSON     │  │  Hybrid  │  │  IPFS  │ │  Neo4j │
│PostgreSQL│  │          │  │        │ │  Mongo │
│ MongoDB  │  │          │  │        │ │  etc.  │
│  Redis   │  │          │  │        │ │        │
│   Git    │  │          │  │        │ │        │
└──────────┘  └──────────┘  └───────┘  └──────┘
```

---

## 🚀 Quick Start

### S3/MinIO Backend

```python
from Promptly.promptly.plugins.storage.custom_storage import S3Storage

# Initialize
storage = S3Storage(
    region='us-east-1',
    endpoint_url=None,  # Use AWS S3
    # endpoint_url='http://localhost:9000',  # Or use MinIO
    cloudfront_domain='d111111abcdef8.cloudfront.net',  # Optional
    enable_versioning=True,
    enable_encryption=True,
)

storage.init_storage('my-promptly-bucket')

# Save prompt
commit = storage.save_prompt({
    'name': 'summarizer',
    'content': 'Summarize: {text}',
    'branch': 'main',
    'metadata': {'author': 'team'}
})

# Get presigned URL for sharing
url = storage.generate_presigned_url('summarizer', expiration=3600)
print(f"Share URL: {url}")

# Get CDN URL for fast access
cdn_url = storage.get_cloudfront_url('summarizer')
print(f"CDN URL: {cdn_url}")
```

### Hybrid Storage Backend

```python
from Promptly.promptly.plugins.storage.custom_storage import HybridStorage

# Initialize with all three tiers
storage = HybridStorage(
    redis_url='redis://localhost:6379/0',
    postgres_url='postgresql://localhost/promptly',
    s3_bucket='promptly-archive',
    enable_auto_tiering=True,
    max_hot_prompts=1000,
    max_warm_prompts=10000,
)

storage.init_storage('./hybrid_cache')

# Save prompt (writes to all tiers)
commit = storage.save_prompt({
    'name': 'analyzer',
    'content': 'Analyze: {data}',
})

# Read is optimized (checks hot → warm → cold)
prompt = storage.get_prompt('analyzer')
print(f"Retrieved from: {prompt.get('_tier')}")  # 'hot', 'warm', or 'cold'

# Run tiering to optimize data placement
stats = storage.run_tiering()
print(f"Promoted to hot: {stats['promoted_to_hot']}")

# View tier distribution
tier_stats = storage.get_tier_statistics()
print(f"Hot tier: {tier_stats['hot_tier']['count']} prompts")
print(f"Warm tier: {tier_stats['warm_tier']['count']} prompts")
print(f"Cold tier: {tier_stats['cold_tier']['count']} prompts")
```

### Blockchain + IPFS Backend

```python
from Promptly.promptly.plugins.storage.custom_storage import BlockchainStorage

# Initialize
storage = BlockchainStorage(
    blockchain_type='ethereum',
    rpc_url='https://mainnet.infura.io/v3/YOUR_KEY',
    ipfs_host='ipfs.infura.io',
    ipfs_port=5001,
    pin_content=True,
)

storage.init_storage('./blockchain_cache')

# Save prompt (immutable)
commit = storage.save_prompt({
    'name': 'classifier',
    'content': 'Classify: {input}',
})

# Get prompt with blockchain metadata
prompt = storage.get_prompt('classifier')
print(f"Transaction: {prompt['_blockchain']['tx_hash']}")
print(f"IPFS: {prompt['_blockchain']['ipfs_hash']}")
print(f"Author: {prompt['_blockchain']['author']}")

# Verify authenticity
proof = storage.verify_prompt_authenticity(commit)
print(f"Verified: {proof['verified']}")

# Export cryptographic proof
export = storage.export_proof_of_existence(commit)
print(f"Blockchain explorer: {export['blockchain_explorer']}")
print(f"IPFS gateway: {export['verification_url']}")
```

---

## 📦 Installation

### S3/MinIO Backend

```bash
pip install boto3

# Optional: Install AWS CLI for configuration
pip install awscli
aws configure
```

### Hybrid Storage Backend

```bash
pip install redis sqlalchemy psycopg2-binary boto3

# Start required services with Docker
docker run -d --name redis -p 6379:6379 redis:alpine
docker run -d --name postgres -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:15
```

### Blockchain Backend

```bash
pip install web3 ipfshttpclient

# Start IPFS daemon
ipfs daemon

# For Ethereum development (Ganache)
npm install -g ganache-cli
ganache-cli --port 8545
```

### Custom DB Template (Cassandra Example)

```bash
pip install cassandra-driver

# Start Cassandra with Docker
docker run -d --name cassandra -p 9042:9042 cassandra:latest
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# S3 Configuration
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
export S3_BUCKET_NAME=my-promptly-bucket
export CLOUDFRONT_DOMAIN=d111111abcdef8.cloudfront.net

# Hybrid Storage Configuration
export REDIS_URL=redis://localhost:6379/0
export POSTGRES_URL=postgresql://user:pass@localhost/promptly
export S3_ARCHIVE_BUCKET=promptly-archive

# Blockchain Configuration
export ETHEREUM_RPC_URL=https://mainnet.infura.io/v3/YOUR_KEY
export IPFS_HOST=ipfs.infura.io
export IPFS_PORT=5001
export PRIVATE_KEY=your_private_key

# Custom DB Configuration
export CASSANDRA_CONTACT_POINTS=127.0.0.1,127.0.0.2
export CASSANDRA_KEYSPACE=promptly
export CASSANDRA_USERNAME=cassandra
export CASSANDRA_PASSWORD=password
```

### Configuration File (`config.yaml`)

```yaml
storage:
  backend: hybrid  # s3, hybrid, blockchain, custom_db

  s3:
    region: us-east-1
    bucket: my-promptly-bucket
    cloudfront_domain: d111111abcdef8.cloudfront.net
    enable_versioning: true
    enable_encryption: true
    lifecycle_days: 90

  hybrid:
    redis_url: redis://localhost:6379/0
    redis_ttl: 86400
    postgres_url: postgresql://localhost/promptly
    postgres_pool_size: 10
    s3_bucket: promptly-archive
    enable_auto_tiering: true
    max_hot_prompts: 1000
    max_warm_prompts: 10000

  blockchain:
    blockchain_type: ethereum
    rpc_url: https://mainnet.infura.io/v3/YOUR_KEY
    ipfs_host: ipfs.infura.io
    ipfs_port: 5001
    pin_content: true

  custom_db:
    contact_points: [127.0.0.1]
    keyspace: promptly
    replication_factor: 3
```

---

## 📖 Usage Examples

See [USAGE_EXAMPLES.md](./USAGE_EXAMPLES.md) for comprehensive examples including:

- Multi-region deployment with S3
- Cost optimization with hybrid storage
- Immutable audit trail with blockchain
- Building custom backends

---

## 🔄 Migration Guide

See [MIGRATION.md](./MIGRATION.md) for detailed migration instructions.

Quick migration example:

```python
from Promptly.promptly.plugins.storage.custom_storage.migrate import migrate_custom_storage

# Migrate from PostgreSQL to S3
stats = migrate_custom_storage(
    from_backend='postgresql',
    from_config={'connection_string': 'postgresql://localhost/promptly'},
    to_backend='s3',
    to_config={'bucket': 'my-bucket', 'region': 'us-east-1'},
    branches=['main', 'production'],
    dry_run=False,
)

print(f"Migrated {stats['prompts_migrated']} prompts")
print(f"Errors: {stats['errors']}")
```

---

## 📊 Performance Comparison

| Backend | Write (ops/s) | Read (ops/s) | Latency (P99) | Cost/GB/month | Scalability |
|---------|--------------|-------------|---------------|----------------|-------------|
| **S3** | 100 | 500 | 50ms | $0.023 | ★★★★★ |
| **Hybrid (Hot)** | 5000 | 8000 | <1ms | $20 | ★★★★☆ |
| **Hybrid (Warm)** | 500 | 2000 | 5ms | $2 | ★★★★☆ |
| **Hybrid (Cold)** | 100 | 500 | 50ms | $0.023 | ★★★★★ |
| **Blockchain** | 10 | 50 | 200ms | Variable* | ★★★☆☆ |

*Blockchain costs include gas fees (variable) + IPFS storage

### Performance by Use Case

| Use Case | Recommended Backend | Reason |
|----------|-------------------|---------|
| Global CDN delivery | S3 + CloudFront | Low latency worldwide |
| High-frequency reads | Hybrid (Hot tier) | Sub-millisecond latency |
| Cost optimization | Hybrid (Auto-tier) | Intelligent data placement |
| Compliance & audit | Blockchain | Immutable trail |
| Development | Custom DB (Local) | Easy setup |

---

## 🔒 Security Best Practices

### S3/MinIO Backend

```python
# ✅ Enable encryption at rest
storage = S3Storage(
    enable_encryption=True,  # AES256
    storage_class='STANDARD',
)

# ✅ Use IAM roles instead of keys
# Don't: Pass access keys directly
# Do: Use EC2 instance roles or ECS task roles

# ✅ Enable versioning and MFA delete
# AWS CLI:
# aws s3api put-bucket-versioning --bucket my-bucket \
#   --versioning-configuration Status=Enabled,MFADelete=Enabled

# ✅ Use presigned URLs for temporary access
url = storage.generate_presigned_url(
    'sensitive_prompt',
    expiration=900  # 15 minutes
)
```

### Hybrid Storage Backend

```python
# ✅ Encrypt sensitive data in Redis
# Use Redis encryption or application-level encryption

# ✅ Use SSL/TLS for all connections
storage = HybridStorage(
    redis_url='rediss://localhost:6379/0',  # Note: rediss://
    postgres_url='postgresql://user:pass@localhost/db?sslmode=require',
    s3_endpoint='https://...',  # HTTPS only
)

# ✅ Implement access control
# - PostgreSQL: Row-level security
# - Redis: ACL rules
# - S3: Bucket policies
```

### Blockchain Backend

```python
# ✅ Secure private key storage
import os
from cryptography.fernet import Fernet

# Never hardcode private keys
private_key = os.environ.get('PRIVATE_KEY')  # From secure vault

# ✅ Use multi-signature for critical operations
storage = BlockchainStorage(
    require_multi_sig=True,
    required_signatures=3,
)

# ✅ Pin critical content on IPFS
storage = BlockchainStorage(
    pin_content=True,  # Prevent garbage collection
)

# ✅ Verify all blockchain transactions
proof = storage.verify_prompt_authenticity(commit_hash)
assert proof['verified'], "Tampered data detected!"
```

### General Best Practices

1. **Principle of Least Privilege**: Grant minimum required permissions
2. **Encryption**: Encrypt data at rest and in transit
3. **Audit Logging**: Log all access and modifications
4. **Key Rotation**: Regularly rotate credentials
5. **Network Security**: Use VPCs, security groups, firewalls
6. **Backup & Recovery**: Regular backups, tested recovery procedures
7. **Monitoring**: Alert on suspicious activity

---

## 🐛 Troubleshooting

### S3 Connection Issues

```python
# Problem: AccessDenied errors
# Solution: Check IAM permissions

# Minimum required permissions:
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:DeleteObject",
        "s3:PutObjectTagging"
      ],
      "Resource": [
        "arn:aws:s3:::my-bucket/*",
        "arn:aws:s3:::my-bucket"
      ]
    }
  ]
}

# Problem: Slow uploads
# Solution: Enable multipart upload and connection pooling
storage = S3Storage(
    max_retries=3,
    timeout=30,
)
```

### Hybrid Storage Issues

```python
# Problem: Redis out of memory
# Solution: Configure maxmemory and eviction policy

# redis.conf:
# maxmemory 2gb
# maxmemory-policy allkeys-lru

# Or programmatically:
storage.redis_client.config_set('maxmemory', '2gb')
storage.redis_client.config_set('maxmemory-policy', 'allkeys-lru')

# Problem: PostgreSQL connection pool exhausted
# Solution: Increase pool size
storage = HybridStorage(
    postgres_pool_size=20,  # Increase from default 10
    postgres_max_overflow=40,
)
```

### Blockchain Issues

```python
# Problem: IPFS content not found
# Solution: Pin important content and use redundant gateways

# Pin content
storage.ipfs.pin.add(ipfs_hash)

# Use multiple gateways
gateways = [
    'https://ipfs.io',
    'https://cloudflare-ipfs.com',
    'https://gateway.pinata.cloud',
]

# Problem: High gas fees
# Solution: Use L2 solutions or sidechains
storage = BlockchainStorage(
    blockchain_type='polygon',  # Lower fees than Ethereum
    rpc_url='https://polygon-rpc.com',
)
```

---

## 📚 Additional Resources

- [Configuration Guide](./CONFIG_GUIDE.md)
- [Migration Guide](./MIGRATION.md)
- [Usage Examples](./USAGE_EXAMPLES.md)
- [API Reference](./API_REFERENCE.md)
- [Contributing](./CONTRIBUTING.md)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## 📄 License

See main project LICENSE file.
