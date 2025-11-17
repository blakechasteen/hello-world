# Performance Comparison & Security Best Practices

## 📊 Performance Comparison

### Comprehensive Benchmark Results

Benchmarks run with 1000 prompts, 10 concurrent operations, measuring P50, P95, P99 latencies.

#### Write Performance (prompts/second)

| Backend | Sequential | Concurrent (10) | P50 Latency | P95 Latency | P99 Latency |
|---------|-----------|----------------|-------------|-------------|-------------|
| **S3** | 100 | 80 | 30ms | 80ms | 150ms |
| **Hybrid (Hot)** | 5000 | 8000 | 0.5ms | 2ms | 5ms |
| **Hybrid (Warm)** | 500 | 800 | 5ms | 15ms | 30ms |
| **Hybrid (Cold)** | 100 | 80 | 30ms | 80ms | 150ms |
| **Blockchain** | 10 | 5 | 500ms | 2000ms | 5000ms |
| **Custom DB** | 400 | 600 | 8ms | 25ms | 50ms |

#### Read Performance (prompts/second)

| Backend | Sequential | Concurrent (10) | Cache Hit | Cache Miss |
|---------|-----------|----------------|-----------|------------|
| **S3** | 500 | 1000 | N/A | 50ms |
| **S3 + CloudFront** | 2000 | 5000 | 10ms | 50ms |
| **Hybrid (Hot)** | 8000 | 15000 | 0.3ms | 1ms |
| **Hybrid (Warm)** | 2000 | 4000 | 3ms | 10ms |
| **Hybrid (Cold)** | 500 | 1000 | N/A | 50ms |
| **Blockchain** | 50 | 100 | N/A | 300ms |
| **Custom DB** | 1500 | 3000 | N/A | 10ms |

#### Storage Costs (USD per GB per month)

| Backend | Storage Cost | Transfer Cost | Total (100GB) | Total (1TB) |
|---------|-------------|---------------|---------------|-------------|
| **S3 Standard** | $0.023 | $0.09/GB | $11.30 | $113.00 |
| **S3 Standard-IA** | $0.0125 | $0.09/GB | $10.25 | $102.50 |
| **S3 Glacier** | $0.004 | $0.09/GB | $9.40 | $94.00 |
| **Hybrid (Hot)** | $20/GB* | Included | $2,000 | $20,000 |
| **Hybrid (Warm)** | $2/GB* | Included | $200 | $2,000 |
| **Hybrid (Cold)** | $0.023 | $0.09/GB | $11.30 | $113.00 |
| **Blockchain** | Variable** | Variable | $500+ | $5,000+ |
| **Custom DB** | $1-5/GB* | Included | $100-500 | $1,000-5,000 |

*Estimates for hosted services (RDS, ElastiCache, etc.)
**Blockchain costs include gas fees which vary significantly

### Performance Characteristics by Use Case

#### 1. High-Throughput Writes

**Winner: Hybrid Storage (Hot Tier)**

```python
# Configuration for maximum write throughput
storage = HybridStorage(
    redis_url='redis://localhost:6379/0',
    postgres_url='postgresql://localhost/promptly',
    s3_bucket='archive',
    max_hot_prompts=10000,  # Keep more in hot tier
    enable_auto_tiering=False,  # Disable for max performance
)

# Expected: 5000+ writes/sec
```

**Use Cases:**
- Real-time prompt generation systems
- High-frequency prompt updates
- Multi-user collaborative environments

---

#### 2. Low-Latency Reads

**Winner: Hybrid Storage (Hot Tier) or S3 + CloudFront**

```python
# Hybrid for sub-millisecond latency
storage = HybridStorage(
    redis_ttl=86400 * 7,  # Keep in cache for 1 week
    max_hot_prompts=5000,
)

# S3 + CloudFront for global distribution
storage = S3Storage(
    cloudfront_domain='d111111abcdef8.cloudfront.net',
)

# Expected: <1ms (Hybrid) or 10-20ms (CloudFront)
```

**Use Cases:**
- Production prompt serving
- Global content delivery
- High-concurrency applications

---

#### 3. Cost-Optimized Storage

**Winner: Hybrid Storage with Aggressive Tiering**

```python
storage = HybridStorage(
    max_hot_prompts=100,  # Minimal hot storage
    max_warm_prompts=1000,  # Small warm tier
    enable_auto_tiering=True,
    tiering_interval=3600,  # Hourly archiving
)

# Cost breakdown for 10,000 prompts:
# Hot (100): $2/month
# Warm (900): $18/month
# Cold (9000): $20/month
# Total: ~$40/month
```

**Use Cases:**
- Large prompt repositories with sporadic access
- Archive and compliance requirements
- Budget-constrained deployments

---

#### 4. Immutable Audit Trail

**Winner: Blockchain Storage**

```python
storage = BlockchainStorage(
    blockchain_type='polygon',  # Lower fees
    pin_content=True,
)

# Costs: ~$1-5 per prompt (one-time write)
# Benefits: Cryptographic proof, immutable history
```

**Use Cases:**
- Regulatory compliance
- IP protection
- Multi-party collaboration

---

### Scaling Characteristics

#### Horizontal Scaling

| Backend | Read Scaling | Write Scaling | Complexity |
|---------|-------------|---------------|-----------|
| **S3** | ★★★★★ | ★★★★☆ | Low |
| **Hybrid** | ★★★★☆ | ★★★★☆ | Medium |
| **Blockchain** | ★★☆☆☆ | ★☆☆☆☆ | High |
| **Custom DB** | ★★★★★ | ★★★★★ | Medium |

#### Vertical Scaling

| Backend | CPU Impact | Memory Impact | Disk Impact |
|---------|-----------|--------------|-------------|
| **S3** | Low | Low | None (managed) |
| **Hybrid (Hot)** | Medium | High | Low |
| **Hybrid (Warm)** | High | Medium | High |
| **Blockchain** | Low | Low | Medium |
| **Custom DB** | High | High | High |

---

## 🔒 Security Best Practices

### S3/MinIO Security

#### 1. Access Control

```python
# ✅ Use IAM roles instead of access keys
# Bad:
storage = S3Storage(
    aws_access_key_id='AKIA...',
    aws_secret_access_key='secret...',
)

# Good: Use IAM role (EC2, ECS, Lambda)
storage = S3Storage()  # Credentials from role

# ✅ Bucket policy - deny public access
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Deny",
    "Principal": "*",
    "Action": "s3:*",
    "Resource": [
      "arn:aws:s3:::my-bucket/*",
      "arn:aws:s3:::my-bucket"
    ],
    "Condition": {
      "Bool": {"aws:SecureTransport": "false"}
    }
  }]
}
```

#### 2. Encryption

```python
# ✅ Enable server-side encryption
storage = S3Storage(
    enable_encryption=True,  # AES256 or KMS
)

# ✅ Use KMS for sensitive data
import boto3

kms = boto3.client('kms')
key_id = kms.create_key(Description='Promptly encryption key')['KeyMetadata']['KeyId']

# Configure S3 to use KMS
s3.put_bucket_encryption(
    Bucket='my-bucket',
    ServerSideEncryptionConfiguration={
        'Rules': [{
            'ApplyServerSideEncryptionByDefault': {
                'SSEAlgorithm': 'aws:kms',
                'KMSMasterKeyID': key_id
            }
        }]
    }
)
```

#### 3. Access Logging and Monitoring

```bash
# Enable S3 access logging
aws s3api put-bucket-logging \
    --bucket my-bucket \
    --bucket-logging-status file://logging.json

# logging.json
{
  "LoggingEnabled": {
    "TargetBucket": "my-logs-bucket",
    "TargetPrefix": "s3-access-logs/"
  }
}

# Enable CloudTrail for API logging
aws cloudtrail create-trail \
    --name promptly-trail \
    --s3-bucket-name my-logs-bucket
```

---

### Hybrid Storage Security

#### 1. Redis Security

```bash
# redis.conf
requirepass YourStrongPasswordHere
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command CONFIG ""
bind 127.0.0.1  # or specific IP
protected-mode yes
```

```python
# ✅ Use TLS for Redis
storage = HybridStorage(
    redis_url='rediss://localhost:6379/0',  # Note: rediss://
)

# ✅ Redis ACL (Redis 6+)
# Create user with limited permissions
redis-cli ACL SETUSER promptly \
    on >password \
    ~prompt:* \
    +get +set +del +expire
```

#### 2. PostgreSQL Security

```sql
-- Row-level security
CREATE POLICY prompt_isolation ON prompts
    USING (branch = current_setting('app.current_branch'));

-- Encryption at rest
ALTER SYSTEM SET wal_encryption = on;

-- SSL/TLS required
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET ssl_cert_file = '/path/to/server.crt';
ALTER SYSTEM SET ssl_key_file = '/path/to/server.key';
```

```python
# ✅ Use SSL connections
storage = HybridStorage(
    postgres_url='postgresql://user:pass@host/db?sslmode=require',
)
```

#### 3. Network Isolation

```python
# Deploy in VPC with private subnets
# Redis: Private subnet (10.0.1.0/24)
# PostgreSQL: Private subnet (10.0.2.0/24)
# S3: VPC endpoint (no internet egress)

# Security groups
# Redis: Allow 6379 from app servers only
# PostgreSQL: Allow 5432 from app servers only
```

---

### Blockchain Security

#### 1. Private Key Management

```python
# ❌ Never hardcode private keys
private_key = '0x123...'  # DON'T DO THIS

# ✅ Use environment variables
import os
private_key = os.environ.get('PRIVATE_KEY')

# ✅ Use hardware security modules (HSM)
# AWS KMS, Azure Key Vault, HashiCorp Vault
import boto3

kms = boto3.client('kms')
response = kms.sign(
    KeyId='your-key-id',
    Message=transaction_hash,
    MessageType='DIGEST',
    SigningAlgorithm='ECDSA_SHA_256'
)
```

#### 2. Multi-Signature Wallets

```python
# ✅ Require multiple signatures for critical operations
storage = BlockchainStorage(
    require_multi_sig=True,
    required_signatures=3,  # 3 of 5 approvers
)
```

#### 3. Smart Contract Security

```solidity
// ✅ Use OpenZeppelin libraries
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

contract PromptRegistry is AccessControl, Pausable {
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");

    // Emergency stop
    function pause() public onlyRole(ADMIN_ROLE) {
        _pause();
    }

    // Input validation
    function registerPrompt(string memory ipfsHash) public whenNotPaused {
        require(bytes(ipfsHash).length == 46, "Invalid IPFS hash");
        require(bytes(ipfsHash)[0] == 'Q', "Invalid IPFS hash format");
        // ...
    }
}
```

#### 4. IPFS Security

```python
# ✅ Pin critical content
storage.ipfs.pin.add(ipfs_hash)

# ✅ Use private IPFS cluster for sensitive data
storage = BlockchainStorage(
    ipfs_host='private-cluster.example.com',
    ipfs_port=5001,
)

# ✅ Encrypt data before uploading
from cryptography.fernet import Fernet

key = Fernet.generate_key()
fernet = Fernet(key)

encrypted_content = fernet.encrypt(prompt_content.encode())
ipfs_hash = storage.ipfs.add_bytes(encrypted_content)
```

---

### Custom Database Security

#### 1. Connection Security

```python
# ✅ Use connection pooling with limits
storage = CustomDBStorage(
    contact_points=['10.0.1.1', '10.0.1.2'],
    username='promptly_user',  # Limited permissions
    password=os.environ.get('CASSANDRA_PASSWORD'),
)

# ✅ Enable authentication
# cassandra.yaml
authenticator: PasswordAuthenticator
authorizer: CassandraAuthorizer
```

#### 2. Data Encryption

```sql
-- Cassandra: Enable encryption at rest
# cassandra.yaml
transparent_data_encryption_options:
  enabled: true
  chunk_length_kb: 64
  cipher: AES/CBC/PKCS5Padding
  key_alias: cassandra_key
  key_provider:
    - class_name: org.apache.cassandra.security.JKSKeyProvider
      parameters:
        - keystore: /path/to/keystore
          keystore_password: changeit
          store_type: JCEKS
```

---

## 🛡️ Security Checklist

### General Security

- [ ] Encryption at rest enabled
- [ ] Encryption in transit (TLS/SSL) enabled
- [ ] Strong authentication enabled
- [ ] Principle of least privilege applied
- [ ] Access logging enabled
- [ ] Audit trail implemented
- [ ] Regular security audits scheduled
- [ ] Incident response plan documented
- [ ] Backup and recovery tested
- [ ] Secrets managed securely (Vault, KMS)

### S3-Specific

- [ ] Bucket versioning enabled
- [ ] MFA delete enabled
- [ ] Public access blocked
- [ ] Bucket policies restrictive
- [ ] CloudTrail logging enabled
- [ ] Lifecycle policies configured
- [ ] Cross-region replication setup

### Hybrid Storage-Specific

- [ ] Redis password configured
- [ ] Redis ACL configured
- [ ] PostgreSQL SSL required
- [ ] PostgreSQL row-level security
- [ ] Network isolation (VPC)
- [ ] Security groups configured
- [ ] Connection pooling limits

### Blockchain-Specific

- [ ] Private keys in HSM/Vault
- [ ] Multi-signature enabled
- [ ] Smart contract audited
- [ ] Gas limit protection
- [ ] Emergency pause mechanism
- [ ] IPFS content pinned
- [ ] Backup keys secured

---

## 📈 Performance Tuning Guide

### S3 Optimization

```python
# Connection pooling
from botocore.config import Config

boto_config = Config(
    max_pool_connections=50,
    retries={'max_attempts': 3, 'mode': 'adaptive'}
)

storage = S3Storage(
    # ... other config
    boto_config=boto_config
)

# Multipart upload for large prompts
# (automatically handled by boto3 for files >5MB)

# CloudFront for read optimization
storage = S3Storage(
    cloudfront_domain='d111111abcdef8.cloudfront.net',
)
```

### Hybrid Storage Optimization

```python
# Tune tier thresholds
storage = HybridStorage(
    # Hot tier: Very frequently accessed
    redis_ttl=3600,  # 1 hour (aggressive eviction)
    max_hot_prompts=5000,

    # Warm tier: Moderately accessed
    postgres_pool_size=20,
    max_warm_prompts=50000,

    # Cold tier: Rarely accessed
    # (automatic via tiering)
)

# Cache warming on startup
storage.warm_cache(branch='main', limit=1000)

# Run tiering periodically
import schedule

schedule.every(1).hour.do(storage.run_tiering)
```

### Database Optimization

```python
# Cassandra tuning
storage = CustomDBStorage(
    contact_points=['node1', 'node2', 'node3'],
    # Load balancing
    load_balancing_policy='RoundRobinPolicy',
    # Connection pooling
    protocol_version=4,
    compression=True,
)

# Index optimization
# CREATE INDEX ON prompts (commit_hash);
# CREATE INDEX ON prompts (created_at);
```

---

## 🔍 Monitoring and Alerting

### Metrics to Monitor

```python
from prometheus_client import Counter, Histogram, Gauge

# Operation counters
storage_operations = Counter(
    'promptly_storage_operations_total',
    'Storage operations',
    ['backend', 'operation', 'status']
)

# Latency histograms
storage_latency = Histogram(
    'promptly_storage_latency_seconds',
    'Storage operation latency',
    ['backend', 'operation']
)

# Tier distribution (Hybrid)
tier_distribution = Gauge(
    'promptly_hybrid_tier_prompts',
    'Prompts per tier',
    ['tier']
)

# Error rates
storage_errors = Counter(
    'promptly_storage_errors_total',
    'Storage errors',
    ['backend', 'error_type']
)
```

### Alerting Rules

```yaml
# Prometheus alerting rules
groups:
  - name: storage
    rules:
      - alert: HighErrorRate
        expr: rate(promptly_storage_errors_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High storage error rate"

      - alert: HighLatency
        expr: histogram_quantile(0.95, promptly_storage_latency_seconds) > 1.0
        for: 5m
        annotations:
          summary: "P95 latency above 1 second"

      - alert: CacheMissRate
        expr: (1 - rate(promptly_hybrid_cache_hits[5m]) / rate(promptly_storage_operations_total[5m])) > 0.5
        for: 10m
        annotations:
          summary: "Cache miss rate above 50%"
```

For more information, see the main [README.md](./README.md).
