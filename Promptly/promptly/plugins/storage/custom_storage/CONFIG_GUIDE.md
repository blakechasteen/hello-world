# Configuration Guide - Advanced Custom Storage Backends

This guide provides detailed configuration instructions for all custom storage backends.

## Table of Contents

- [S3/MinIO Configuration](#s3minio-configuration)
- [Hybrid Storage Configuration](#hybrid-storage-configuration)
- [Blockchain Storage Configuration](#blockchain-storage-configuration)
- [Environment Variables](#environment-variables)
- [Configuration Files](#configuration-files)
- [Production Deployment](#production-deployment)

---

## S3/MinIO Configuration

### AWS S3 Setup

#### 1. Create S3 Bucket

```bash
# Using AWS CLI
aws s3 mb s3://my-promptly-bucket --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket my-promptly-bucket \
    --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption \
    --bucket my-promptly-bucket \
    --server-side-encryption-configuration '{
        "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "AES256"
            }
        }]
    }'
```

#### 2. IAM Configuration

Create IAM policy (`promptly-s3-policy.json`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PromptlyS3Access",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:GetObjectVersion",
        "s3:PutObjectTagging",
        "s3:GetObjectTagging",
        "s3:PutBucketVersioning",
        "s3:GetBucketVersioning",
        "s3:PutLifecycleConfiguration"
      ],
      "Resource": [
        "arn:aws:s3:::my-promptly-bucket",
        "arn:aws:s3:::my-promptly-bucket/*"
      ]
    }
  ]
}
```

Apply policy:

```bash
# Create policy
aws iam create-policy \
    --policy-name PromptlyS3Policy \
    --policy-document file://promptly-s3-policy.json

# Attach to user/role
aws iam attach-user-policy \
    --user-name promptly-user \
    --policy-arn arn:aws:iam::ACCOUNT_ID:policy/PromptlyS3Policy
```

#### 3. CloudFront CDN Setup (Optional)

```bash
# Create CloudFront distribution
aws cloudfront create-distribution \
    --distribution-config file://cloudfront-config.json

# cloudfront-config.json
{
  "CallerReference": "promptly-cdn-$(date +%s)",
  "Origins": {
    "Quantity": 1,
    "Items": [{
      "Id": "S3-my-promptly-bucket",
      "DomainName": "my-promptly-bucket.s3.amazonaws.com",
      "S3OriginConfig": {
        "OriginAccessIdentity": ""
      }
    }]
  },
  "DefaultCacheBehavior": {
    "TargetOriginId": "S3-my-promptly-bucket",
    "ViewerProtocolPolicy": "redirect-to-https",
    "MinTTL": 0,
    "DefaultTTL": 86400,
    "MaxTTL": 31536000
  },
  "Enabled": true
}
```

#### 4. Python Configuration

```python
from Promptly.promptly.plugins.storage.custom_storage import S3Storage

storage = S3Storage(
    region='us-east-1',
    aws_access_key_id='AKIA...',  # Or use IAM role
    aws_secret_access_key='...',  # Or use IAM role
    cloudfront_domain='d111111abcdef8.cloudfront.net',
    enable_versioning=True,
    enable_encryption=True,
    storage_class='STANDARD',  # or STANDARD_IA, GLACIER
    lifecycle_days=90,
)

storage.init_storage('s3://my-promptly-bucket')
```

### MinIO Setup

#### 1. Start MinIO Server

```bash
# Using Docker
docker run -d \
    --name minio \
    -p 9000:9000 \
    -p 9001:9001 \
    -e MINIO_ROOT_USER=admin \
    -e MINIO_ROOT_PASSWORD=password123 \
    -v /data/minio:/data \
    minio/minio server /data --console-address ":9001"

# Access MinIO Console: http://localhost:9001
```

#### 2. Create Bucket

```bash
# Using mc (MinIO Client)
mc alias set myminio http://localhost:9000 admin password123
mc mb myminio/promptly-bucket
mc version enable myminio/promptly-bucket
```

#### 3. Python Configuration

```python
storage = S3Storage(
    endpoint_url='http://localhost:9000',
    aws_access_key_id='admin',
    aws_secret_access_key='password123',
    region='us-east-1',  # MinIO ignores region but it's required
)

storage.init_storage('promptly-bucket')
```

---

## Hybrid Storage Configuration

### Component Setup

#### 1. Redis Setup

```bash
# Docker
docker run -d \
    --name redis \
    -p 6379:6379 \
    -v /data/redis:/data \
    redis:alpine redis-server \
    --maxmemory 2gb \
    --maxmemory-policy allkeys-lru \
    --appendonly yes

# Configuration file (redis.conf)
maxmemory 2gb
maxmemory-policy allkeys-lru
appendonly yes
appendfsync everysec
```

#### 2. PostgreSQL Setup

```bash
# Docker
docker run -d \
    --name postgres \
    -p 5432:5432 \
    -e POSTGRES_PASSWORD=password \
    -e POSTGRES_DB=promptly \
    -v /data/postgres:/var/lib/postgresql/data \
    postgres:15

# Initialize database
psql -h localhost -U postgres -d promptly -c "
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
    CREATE EXTENSION IF NOT EXISTS btree_gin;
"
```

#### 3. S3/MinIO Setup

See S3 configuration above.

#### 4. Python Configuration

```python
from Promptly.promptly.plugins.storage.custom_storage import HybridStorage

storage = HybridStorage(
    # Redis (Hot tier)
    redis_url='redis://localhost:6379/0',
    redis_ttl=86400,  # 24 hours
    redis_max_memory='2gb',

    # PostgreSQL (Warm tier)
    postgres_url='postgresql://postgres:password@localhost/promptly',
    postgres_pool_size=10,

    # S3 (Cold tier)
    s3_bucket='promptly-archive',
    s3_region='us-east-1',
    s3_endpoint=None,  # or MinIO endpoint

    # Tiering configuration
    enable_auto_tiering=True,
    tiering_interval=3600,  # 1 hour
    max_hot_prompts=1000,
    max_warm_prompts=10000,
)

storage.init_storage('./hybrid_cache')
```

### Tuning Parameters

```python
# High-performance configuration
storage = HybridStorage(
    redis_ttl=3600,  # Shorter TTL for faster eviction
    postgres_pool_size=20,  # More connections
    max_hot_prompts=5000,  # More prompts in hot tier
)

# Cost-optimized configuration
storage = HybridStorage(
    redis_ttl=86400,  # Longer TTL
    max_hot_prompts=500,  # Fewer hot prompts
    max_warm_prompts=5000,  # Fewer warm prompts
    enable_auto_tiering=True,  # Aggressive archiving
)

# Balanced configuration
storage = HybridStorage(
    redis_ttl=43200,  # 12 hours
    max_hot_prompts=1000,
    max_warm_prompts=10000,
)
```

---

## Blockchain Storage Configuration

### Ethereum Setup

#### 1. Development (Ganache)

```bash
# Install Ganache
npm install -g ganache-cli

# Start Ganache
ganache-cli \
    --port 8545 \
    --accounts 10 \
    --defaultBalanceEther 1000
```

#### 2. Production (Infura)

```python
storage = BlockchainStorage(
    blockchain_type='ethereum',
    rpc_url='https://mainnet.infura.io/v3/YOUR_PROJECT_ID',
    chain_id=1,  # Mainnet
    private_key=os.environ.get('PRIVATE_KEY'),
    gas_price=50_000_000_000,  # 50 Gwei
)
```

#### 3. L2 Solutions (Polygon)

```python
storage = BlockchainStorage(
    blockchain_type='polygon',
    rpc_url='https://polygon-rpc.com',
    chain_id=137,  # Polygon Mainnet
    private_key=os.environ.get('PRIVATE_KEY'),
    gas_price=30_000_000_000,  # 30 Gwei
)
```

### IPFS Setup

#### 1. Local IPFS Node

```bash
# Install IPFS
wget https://dist.ipfs.io/go-ipfs/v0.20.0/go-ipfs_v0.20.0_linux-amd64.tar.gz
tar -xvzf go-ipfs_v0.20.0_linux-amd64.tar.gz
cd go-ipfs
sudo bash install.sh

# Initialize and start
ipfs init
ipfs daemon
```

#### 2. Infura IPFS

```python
storage = BlockchainStorage(
    ipfs_host='ipfs.infura.io',
    ipfs_port=5001,
    ipfs_gateway='https://gateway.ipfs.io',
)
```

#### 3. Pinata (Pin Service)

```python
# Use Pinata API for pinning
import requests

def pin_to_pinata(ipfs_hash):
    url = 'https://api.pinata.cloud/pinning/pinByHash'
    headers = {
        'pinata_api_key': 'YOUR_API_KEY',
        'pinata_secret_api_key': 'YOUR_SECRET_KEY'
    }
    data = {'hashToPin': ipfs_hash}
    response = requests.post(url, json=data, headers=headers)
    return response.json()
```

### Full Configuration

```python
from Promptly.promptly.plugins.storage.custom_storage import BlockchainStorage
import os

storage = BlockchainStorage(
    # Blockchain
    blockchain_type='polygon',  # Lower fees than Ethereum
    rpc_url=os.environ.get('POLYGON_RPC_URL'),
    chain_id=137,
    contract_address='0x...',  # Your deployed contract
    private_key=os.environ.get('PRIVATE_KEY'),
    gas_price=30_000_000_000,

    # IPFS
    ipfs_host='ipfs.infura.io',
    ipfs_port=5001,
    ipfs_gateway='https://gateway.pinata.cloud',
    pin_content=True,

    # Security
    require_multi_sig=False,  # Enable for production
)

storage.init_storage('./blockchain_cache')
```

---

## Environment Variables

Create a `.env` file:

```bash
# S3/MinIO
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=my-promptly-bucket
S3_ENDPOINT_URL=  # Leave empty for AWS, set for MinIO
CLOUDFRONT_DOMAIN=d111111abcdef8.cloudfront.net

# Hybrid Storage
REDIS_URL=redis://localhost:6379/0
POSTGRES_URL=postgresql://postgres:password@localhost/promptly
S3_ARCHIVE_BUCKET=promptly-archive

# Blockchain
BLOCKCHAIN_TYPE=polygon
BLOCKCHAIN_RPC_URL=https://polygon-rpc.com
CHAIN_ID=137
CONTRACT_ADDRESS=0x...
PRIVATE_KEY=0x...
IPFS_HOST=ipfs.infura.io
IPFS_PORT=5001

# Custom DB
CASSANDRA_CONTACT_POINTS=127.0.0.1
CASSANDRA_KEYSPACE=promptly
CASSANDRA_USERNAME=cassandra
CASSANDRA_PASSWORD=password
```

Load environment variables:

```python
from dotenv import load_dotenv
import os

load_dotenv()

storage = S3Storage(
    region=os.environ.get('AWS_DEFAULT_REGION'),
    cloudfront_domain=os.environ.get('CLOUDFRONT_DOMAIN'),
)
```

---

## Configuration Files

### YAML Configuration

`storage_config.yaml`:

```yaml
storage:
  backend: hybrid  # s3, hybrid, blockchain, custom_db

  s3:
    region: us-east-1
    bucket: my-promptly-bucket
    endpoint_url:  # Leave empty for AWS
    cloudfront_domain: d111111abcdef8.cloudfront.net
    enable_versioning: true
    enable_encryption: true
    storage_class: STANDARD
    lifecycle_days: 90
    max_retries: 3
    timeout: 30

  hybrid:
    redis:
      url: redis://localhost:6379/0
      ttl: 86400
      max_memory: 2gb
    postgres:
      url: postgresql://localhost/promptly
      pool_size: 10
      max_overflow: 20
    s3:
      bucket: promptly-archive
      region: us-east-1
      endpoint_url:
    tiering:
      enable_auto_tiering: true
      tiering_interval: 3600
      max_hot_prompts: 1000
      max_warm_prompts: 10000
      hot_tier_ttl: 86400
      warm_tier_days: 30

  blockchain:
    blockchain_type: polygon
    rpc_url: https://polygon-rpc.com
    chain_id: 137
    contract_address: 0x...
    gas_price: 30000000000
    ipfs:
      host: ipfs.infura.io
      port: 5001
      gateway: https://gateway.pinata.cloud
      pin_content: true
    security:
      require_multi_sig: false
      required_signatures: 2

  custom_db:
    contact_points:
      - 127.0.0.1
    port: 9042
    keyspace: promptly
    username: cassandra
    password: password
    consistency_level: LOCAL_QUORUM
    replication_factor: 3
```

Load configuration:

```python
import yaml

with open('storage_config.yaml') as f:
    config = yaml.safe_load(f)

backend_type = config['storage']['backend']
backend_config = config['storage'][backend_type]

if backend_type == 's3':
    storage = S3Storage(**backend_config)
elif backend_type == 'hybrid':
    storage = HybridStorage(
        redis_url=backend_config['redis']['url'],
        postgres_url=backend_config['postgres']['url'],
        s3_bucket=backend_config['s3']['bucket'],
        **backend_config['tiering']
    )
# ... etc
```

---

## Production Deployment

### S3 Production Checklist

- [ ] Enable versioning
- [ ] Enable encryption (AES256 or KMS)
- [ ] Configure lifecycle policies
- [ ] Setup CloudFront CDN
- [ ] Enable access logging
- [ ] Configure bucket policies
- [ ] Setup cross-region replication
- [ ] Enable MFA delete
- [ ] Configure backup retention

### Hybrid Storage Production Checklist

- [ ] Redis persistence (AOF + RDB)
- [ ] PostgreSQL replication
- [ ] S3 backup for cold tier
- [ ] Connection pooling configured
- [ ] Monitoring and alerts
- [ ] Auto-tiering enabled
- [ ] Resource limits set
- [ ] SSL/TLS for all connections

### Blockchain Production Checklist

- [ ] Private key secure storage (HSM/Vault)
- [ ] Multi-signature enabled
- [ ] IPFS pinning service
- [ ] Gas price optimization
- [ ] Contract audit completed
- [ ] Backup private keys
- [ ] Monitor blockchain events
- [ ] Emergency shutdown mechanism

### Monitoring

```python
# Prometheus metrics example
from prometheus_client import Counter, Histogram, Gauge

# S3 metrics
s3_operations = Counter('promptly_s3_operations_total', 'S3 operations', ['operation'])
s3_latency = Histogram('promptly_s3_latency_seconds', 'S3 operation latency')
s3_errors = Counter('promptly_s3_errors_total', 'S3 errors', ['error_type'])

# Hybrid metrics
hybrid_tier_distribution = Gauge('promptly_hybrid_tier_prompts', 'Prompts per tier', ['tier'])
hybrid_cache_hit_rate = Gauge('promptly_hybrid_cache_hit_rate', 'Cache hit rate')

# Blockchain metrics
blockchain_gas_used = Histogram('promptly_blockchain_gas_used', 'Gas used per transaction')
ipfs_pin_success = Counter('promptly_ipfs_pins_total', 'IPFS pins', ['status'])
```

---

## Advanced Topics

### Connection Pooling

```python
# S3 connection pooling (via botocore)
from botocore.config import Config

boto_config = Config(
    max_pool_connections=50,
    retries={'max_attempts': 3, 'mode': 'adaptive'}
)

# PostgreSQL connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    postgres_url,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)
```

### SSL/TLS Configuration

```python
# Redis with TLS
storage = HybridStorage(
    redis_url='rediss://localhost:6379/0',  # Note: rediss://
)

# PostgreSQL with SSL
storage = HybridStorage(
    postgres_url='postgresql://user:pass@host/db?sslmode=require'
)

# S3 always uses HTTPS by default
```

### Backup and Recovery

```python
# Backup configuration
backup_config = {
    's3': {
        'versioning': True,
        'lifecycle_retention_days': 365,
    },
    'hybrid': {
        'redis_snapshot_interval': 3600,
        'postgres_backup_schedule': '0 2 * * *',  # Daily at 2 AM
    }
}
```

For more advanced topics, see the main [README.md](./README.md).
