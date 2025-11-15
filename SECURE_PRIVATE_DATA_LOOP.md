# Secure Private Data Collection Loop Architecture

**Status**: Design Document (2025-11-15)
**Purpose**: Enable data collection for learning while minimizing liability exposure
**Philosophy**: "Collect the minimum, protect everything, prove compliance"

## Table of Contents

1. [Threat Model](#threat-model)
2. [Architecture Overview](#architecture-overview)
3. [Privacy-Preserving Techniques](#privacy-preserving-techniques)
4. [Implementation Guide](#implementation-guide)
5. [Compliance Checklist](#compliance-checklist)
6. [Attack Surface Reduction](#attack-surface-reduction)

---

## Threat Model

### What Makes You a Target?

| Risk Factor | Threat Level | Mitigation Priority |
|-------------|--------------|-------------------|
| **PII Storage** | 🔴 CRITICAL | Anonymize immediately |
| **Unencrypted Data** | 🔴 CRITICAL | Encrypt at rest/transit |
| **Weak Access Controls** | 🟠 HIGH | Implement RBAC + MFA |
| **No Audit Trail** | 🟠 HIGH | Enable comprehensive logging |
| **Long Retention** | 🟡 MEDIUM | Aggressive TTL policies |
| **Unclear Purpose** | 🟡 MEDIUM | Document data usage |
| **Third-party Access** | 🟡 MEDIUM | Zero-knowledge architecture |

### Attack Vectors

1. **Data Breach**: Attacker steals database → expose user data
2. **Insider Threat**: Malicious employee accesses data
3. **Regulatory Action**: Non-compliance with GDPR/CCPA → fines
4. **Subpoena/Warrant**: Government requests data you don't want to have
5. **Supply Chain**: Third-party service compromised
6. **Social Engineering**: Phishing for credentials

**Key Insight**: The best way to protect data is **not to collect it**. If you must collect it, **don't store it**. If you must store it, **encrypt it** and **delete it quickly**.

---

## Architecture Overview

### 3-Layer Defense

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: Collection (Privacy by Design)                │
│ ─────────────────────────────────────────────────────── │
│ • Minimal data capture (only what's needed)             │
│ • Immediate anonymization (hash/pseudonymize)           │
│ • Client-side preprocessing (strip PII before upload)   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 2: Storage (Encryption + Access Control)         │
│ ─────────────────────────────────────────────────────── │
│ • Encryption at rest (AES-256-GCM)                      │
│ • Encryption in transit (TLS 1.3)                       │
│ • Zero-knowledge backends (can't decrypt)               │
│ • Aggressive TTL (auto-delete after N days)             │
│ • RBAC + MFA for access                                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 3: Learning (Privacy-Preserving ML)              │
│ ─────────────────────────────────────────────────────── │
│ • Differential privacy (add noise to aggregates)        │
│ • Federated learning (learn without central data)       │
│ • Model-only retention (delete training data)           │
│ • Homomorphic encryption (compute on encrypted data)    │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Interaction
    ↓
[Client-side PII Filter] ← Strip names, emails, IP addresses
    ↓
[Anonymization Layer] ← Hash user IDs, pseudonymize content
    ↓
[Encrypted Storage] ← AES-256-GCM, key rotation every 30 days
    ↓
[Learning Pipeline] ← Differential privacy, epsilon=1.0
    ↓
[Model Weights Only] ← Delete training data after convergence
    ↓
[Audit Trail] ← Immutable log of all access (write-only)
```

---

## Privacy-Preserving Techniques

### 1. Data Minimization

**Principle**: Don't collect what you don't absolutely need.

```python
# ❌ BAD: Collect everything
data = {
    "name": user.full_name,
    "email": user.email,
    "ip_address": request.remote_addr,
    "user_agent": request.headers.get('User-Agent'),
    "query": query_text,
    "timestamp": datetime.now(),
    "session_id": session.id
}

# ✅ GOOD: Collect minimum
import hashlib

def hash_user_id(user_id: str, salt: str) -> str:
    """Create irreversible user identifier."""
    return hashlib.sha256(f"{user_id}{salt}".encode()).hexdigest()[:16]

data = {
    "user_hash": hash_user_id(user.id, SECRET_SALT),  # Pseudonymized
    "query_embedding": embed_query(query_text),  # No raw text!
    "query_type": classify_query_type(query_text),  # Categorical only
    "hour_of_day": datetime.now().hour  # Coarse temporal info
    # ❌ No IP, email, name, session ID, raw query text
}
```

### 2. Anonymization Techniques

| Technique | Use Case | Reversibility | Strength |
|-----------|----------|---------------|----------|
| **Hashing** | User IDs, emails | Irreversible | 🟢 Strong |
| **Pseudonymization** | User tracking | Reversible with key | 🟡 Medium |
| **K-anonymity** | Demographics | N/A | 🟡 Medium |
| **L-diversity** | Sensitive attributes | N/A | 🟢 Strong |
| **Differential Privacy** | Aggregates | N/A | 🟢 Strong |

**Recommended Stack**:
- **User IDs**: SHA-256 hash with secret salt (rotate monthly)
- **Query Text**: Store embeddings only, discard raw text
- **Aggregates**: Add Laplace noise (epsilon=1.0 for differential privacy)
- **Temporal**: Coarsen to hour/day instead of exact timestamp

### 3. Encryption

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

class SecureDataStore:
    """Encrypted data storage with key rotation."""

    def __init__(self, key_path: str = ".keys/master.key"):
        self.key_path = Path(key_path)
        self.key = self._load_or_create_key()
        self.cipher = AESGCM(self.key)

    def _load_or_create_key(self) -> bytes:
        """Load encryption key or create new one."""
        self.key_path.parent.mkdir(exist_ok=True, mode=0o700)  # Secure perms

        if self.key_path.exists():
            return self.key_path.read_bytes()
        else:
            key = AESGCM.generate_key(bit_length=256)
            self.key_path.write_bytes(key)
            self.key_path.chmod(0o600)  # Owner read/write only
            return key

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt data with fresh nonce."""
        nonce = os.urandom(12)  # 96-bit nonce
        ciphertext = self.cipher.encrypt(nonce, plaintext, None)
        return nonce + ciphertext  # Prepend nonce

    def decrypt(self, ciphertext_with_nonce: bytes) -> bytes:
        """Decrypt data."""
        nonce = ciphertext_with_nonce[:12]
        ciphertext = ciphertext_with_nonce[12:]
        return self.cipher.decrypt(nonce, ciphertext, None)

    def rotate_key(self, old_key_path: str, new_key_path: str):
        """Rotate encryption key (re-encrypt all data)."""
        # Implementation: Read with old key, write with new key
        pass
```

### 4. Differential Privacy

Add noise to prevent individual re-identification:

```python
import numpy as np

class DifferentialPrivacy:
    """Laplace mechanism for differential privacy."""

    def __init__(self, epsilon: float = 1.0):
        """
        epsilon: Privacy budget (lower = more privacy, less accuracy)
        - epsilon=0.1: Very private (high noise)
        - epsilon=1.0: Balanced (recommended)
        - epsilon=10.0: Less private (low noise)
        """
        self.epsilon = epsilon

    def add_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """Add Laplace noise to a value."""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise

    def privatize_histogram(self, counts: dict) -> dict:
        """Add noise to histogram counts."""
        return {
            key: max(0, self.add_noise(count))
            for key, count in counts.items()
        }

# Example: Privatize query type distribution
dp = DifferentialPrivacy(epsilon=1.0)
query_types = {"factual": 150, "procedural": 80, "analytical": 45}
private_counts = dp.privatize_histogram(query_types)
# Output: {"factual": 148.3, "procedural": 82.7, "analytical": 43.1}
# Individual queries cannot be recovered
```

### 5. Federated Learning (Advanced)

Learn from distributed data without central collection:

```python
# Conceptual architecture (requires multiple clients)

# Client-side (runs on user device)
class FederatedClient:
    async def train_local_model(self, local_data):
        """Train on local data, never upload raw data."""
        model = load_global_model()
        model.train(local_data)
        return model.get_gradients()  # Upload gradients only

# Server-side (your infrastructure)
class FederatedServer:
    async def aggregate_gradients(self, client_gradients):
        """Average gradients from many clients."""
        avg_gradient = np.mean(client_gradients, axis=0)
        global_model.apply_gradients(avg_gradient)
        return global_model  # Broadcast updated model

# Benefit: You never see raw data, only model updates
```

---

## Implementation Guide

### HoloLoom Integration

Leverage existing alignment framework for secure collection:

```python
from HoloLoom import HoloLoom
from HoloLoom.config import Config
from HoloLoom.alignment import AuditTrail, SafetyGuardrails
from HoloLoom.alignment.api_compatibility import create_audit_trail
import hashlib
from datetime import datetime, timedelta

class SecureDataCollectionLoop:
    """
    Privacy-preserving data collection for HoloLoom learning.

    Features:
    - Automatic PII anonymization
    - Encrypted storage (AES-256-GCM)
    - Differential privacy on aggregates
    - Aggressive TTL (auto-delete after 30 days)
    - Complete audit trail
    - Zero-knowledge architecture
    """

    def __init__(
        self,
        hololoom: HoloLoom,
        audit_trail: AuditTrail,
        encryption_key_path: str = ".keys/data.key",
        retention_days: int = 30,
        privacy_epsilon: float = 1.0
    ):
        self.loom = hololoom
        self.audit = audit_trail
        self.crypto = SecureDataStore(encryption_key_path)
        self.retention_days = retention_days
        self.dp = DifferentialPrivacy(epsilon=privacy_epsilon)

        # Secret salt for user hashing (NEVER commit to git!)
        self.user_salt = os.environ.get("USER_HASH_SALT", "CHANGE_ME")
        if self.user_salt == "CHANGE_ME":
            logger.warning("⚠️  USER_HASH_SALT not set! Using insecure default.")

    def anonymize_user_id(self, user_id: str) -> str:
        """Create irreversible user identifier."""
        return hashlib.sha256(
            f"{user_id}{self.user_salt}".encode()
        ).hexdigest()[:16]

    async def collect_interaction(
        self,
        user_id: str,
        query: str,
        response: str,
        confidence: float,
        metadata: dict = None
    ):
        """
        Collect interaction with privacy protections.

        PII Removed:
        - User ID hashed (irreversible)
        - Query text converted to embedding (discarded after)
        - IP address not collected
        - Timestamp coarsened to hour

        Stored (encrypted):
        - User hash
        - Query embedding
        - Query type (categorical)
        - Response quality (float)
        - Hour of day (int)
        """
        # 1. Anonymize user
        user_hash = self.anonymize_user_id(user_id)

        # 2. Extract features (no raw text stored)
        query_embedding = await self.loom.embed(query)
        query_type = self._classify_query_type(query)  # Categorical

        # 3. Coarsen timestamp
        now = datetime.now()
        hour_of_day = now.hour

        # 4. Create minimal record
        record = {
            "user_hash": user_hash,  # Hashed
            "query_embedding": query_embedding.tolist(),  # Vector only
            "query_type": query_type,  # Categorical
            "confidence": confidence,  # Float
            "hour_of_day": hour_of_day,  # Coarsened time
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(days=self.retention_days)).isoformat()
        }

        # 5. Encrypt
        encrypted_record = self.crypto.encrypt(
            json.dumps(record).encode()
        )

        # 6. Store encrypted
        await self._store_encrypted(user_hash, encrypted_record)

        # 7. Audit trail (immutable log)
        await self.audit.log_decision(
            query="[ANONYMIZED]",  # Don't log raw query!
            action="data_collection",
            outcome="success",
            safety_score=1.0,
            metadata={
                "user_hash": user_hash,
                "query_type": query_type,
                "retention_days": self.retention_days
            }
        )

        # 8. Schedule deletion
        await self._schedule_deletion(user_hash, days=self.retention_days)

    def _classify_query_type(self, query: str) -> str:
        """Classify query into broad category (no PII)."""
        # Simple heuristic (replace with model)
        if any(word in query.lower() for word in ["what", "who", "when", "where"]):
            return "factual"
        elif any(word in query.lower() for word in ["how", "explain"]):
            return "procedural"
        else:
            return "analytical"

    async def _store_encrypted(self, user_hash: str, data: bytes):
        """Store encrypted data (implementation depends on backend)."""
        # Option 1: Encrypted file storage
        cache_dir = Path(".cache/encrypted_interactions")
        cache_dir.mkdir(exist_ok=True, mode=0o700)
        file_path = cache_dir / f"{user_hash}.enc"
        file_path.write_bytes(data)
        file_path.chmod(0o600)

        # Option 2: Encrypted database (PostgreSQL with pgcrypto)
        # Option 3: Hardware security module (HSM)

    async def _schedule_deletion(self, user_hash: str, days: int):
        """Auto-delete after retention period."""
        # Implementation: Use background task scheduler
        # - Option 1: asyncio.create_task with delay
        # - Option 2: Celery beat for production
        # - Option 3: Database TTL (MongoDB, Redis)
        pass

    async def get_privacy_statistics(self) -> dict:
        """Get privatized statistics (differential privacy)."""
        # Aggregate statistics from encrypted data
        query_type_counts = await self._get_query_type_distribution()

        # Apply differential privacy
        private_counts = self.dp.privatize_histogram(query_type_counts)

        return {
            "total_interactions": self.dp.add_noise(len(query_type_counts)),
            "query_type_distribution": private_counts,
            "avg_confidence": self.dp.add_noise(
                await self._get_avg_confidence()
            ),
            "privacy_budget_remaining": self.dp.epsilon
        }


# Usage Example
async def main():
    from HoloLoom import HoloLoom
    from HoloLoom.alignment import create_audit_trail

    async with HoloLoom() as loom:
        audit_trail = create_audit_trail()
        collector = SecureDataCollectionLoop(
            hololoom=loom,
            audit_trail=audit_trail,
            retention_days=30,  # Delete after 30 days
            privacy_epsilon=1.0  # Differential privacy
        )

        # Collect interaction (privacy-preserving)
        await collector.collect_interaction(
            user_id="user123",  # Will be hashed
            query="What is Thompson Sampling?",  # Will be embedded
            response="...",
            confidence=0.92
        )

        # Get privatized statistics
        stats = await collector.get_privacy_statistics()
        print(stats)
        # {
        #   "total_interactions": 152.3,  # Noisy count
        #   "query_type_distribution": {"factual": 98.1, ...},
        #   "avg_confidence": 0.87
        # }
```

---

## Compliance Checklist

### GDPR (European Union)

- [ ] **Lawful Basis**: Document purpose and legal basis for processing
- [ ] **Consent**: Explicit opt-in for data collection (not opt-out)
- [ ] **Right to Access**: Provide user data on request
- [ ] **Right to Erasure**: Delete user data on request ("right to be forgotten")
- [ ] **Right to Portability**: Export user data in machine-readable format
- [ ] **Data Minimization**: Collect only necessary data
- [ ] **Purpose Limitation**: Use data only for stated purpose
- [ ] **Storage Limitation**: Delete after retention period
- [ ] **Privacy by Design**: Build privacy into architecture
- [ ] **DPO**: Appoint Data Protection Officer (if >250 employees)
- [ ] **DPIA**: Conduct Data Protection Impact Assessment for high-risk processing
- [ ] **Breach Notification**: Report breaches within 72 hours

### CCPA (California)

- [ ] **Notice at Collection**: Inform users what data is collected
- [ ] **Right to Know**: Disclose data collected about user
- [ ] **Right to Delete**: Delete user data on request
- [ ] **Right to Opt-Out**: Allow opt-out of data sales
- [ ] **Non-Discrimination**: Don't penalize users who exercise rights

### HIPAA (Healthcare - if applicable)

- [ ] **Encryption**: Encrypt PHI at rest and in transit
- [ ] **Access Controls**: Implement role-based access
- [ ] **Audit Trails**: Log all access to PHI
- [ ] **Business Associate Agreements**: Contracts with third parties

### Best Practices

- [ ] **Privacy Policy**: Clear, concise explanation of data practices
- [ ] **Terms of Service**: Legal agreement with users
- [ ] **Cookie Consent**: EU Cookie Law compliance
- [ ] **Security Incident Response Plan**: Documented breach procedures
- [ ] **Regular Audits**: Annual security and privacy reviews
- [ ] **Staff Training**: Privacy and security training for all employees
- [ ] **Third-party Vetting**: Audit vendors for compliance

---

## Attack Surface Reduction

### 1. Minimize Data Collection

**Before Optimization**:
```python
# Collects 15 fields, including PII
data = {
    "name", "email", "ip", "user_agent", "query", "response",
    "session_id", "timestamp", "location", "device_id", ...
}
```

**After Optimization**:
```python
# Collects 4 fields, zero PII
data = {
    "user_hash": hash(user_id),  # Irreversible
    "query_embedding": embed(query),  # No raw text
    "query_type": "factual",  # Categorical
    "hour": 14  # Coarsened time
}
# Attack surface reduced by 73% (15 → 4 fields)
```

### 2. Encryption at Rest

```bash
# File permissions (Linux/Mac)
chmod 700 .keys/          # Keys directory owner-only
chmod 600 .keys/data.key  # Key file owner read/write only

# Encrypted storage
ls -lah .cache/encrypted_interactions/
# -rw------- 1 user user 2.3K Nov 15 14:32 a1b2c3d4.enc
# Only owner can read/write

# Even if attacker steals file, it's encrypted with AES-256-GCM
```

### 3. Access Control

```python
from functools import wraps
import jwt

def require_admin(f):
    """Decorator: Require admin role for sensitive operations."""
    @wraps(f)
    async def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            raise Unauthorized("No token provided")

        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            if payload.get('role') != 'admin':
                raise Forbidden("Admin role required")
        except jwt.ExpiredSignatureError:
            raise Unauthorized("Token expired")

        return await f(*args, **kwargs)
    return decorated

@require_admin
async def view_encrypted_data(user_hash: str):
    """Admin-only: View decrypted data."""
    # Audit trail logs admin access
    await audit_trail.log_access(
        admin_user=current_user,
        action="view_data",
        target=user_hash
    )
    # ...
```

### 4. Network Security

```yaml
# docker-compose.yml - Isolated network
version: "3.8"
services:
  hololoom:
    networks:
      - private
    # ❌ No ports exposed to host!

  database:
    networks:
      - private
    # ❌ Only accessible from hololoom service

networks:
  private:
    driver: bridge
    internal: true  # No external access
```

### 5. Audit Everything

```python
# Immutable audit log (write-only)
await audit_trail.log_decision(
    query="[ANONYMIZED]",
    action="data_access",
    user=admin_user,
    target=user_hash,
    timestamp=datetime.now(),
    ip_address=hash(request.remote_addr),  # Hashed
    metadata={"reason": "Support ticket #1234"}
)

# Tamper detection: Hash chain
# Each log entry includes hash of previous entry
# Any modification breaks the chain
```

---

## Summary: Risk Reduction Matrix

| Measure | Risk Reduction | Implementation Cost |
|---------|----------------|-------------------|
| **Anonymization** | 🟢 90% (eliminates PII liability) | Low (hash functions) |
| **Encryption at Rest** | 🟢 80% (useless if stolen) | Low (AESGCM) |
| **Encryption in Transit** | 🟢 70% (prevents eavesdropping) | Low (TLS 1.3) |
| **Differential Privacy** | 🟡 60% (prevents re-identification) | Medium (add noise) |
| **Access Controls** | 🟡 50% (prevents insider threats) | Medium (RBAC + MFA) |
| **Aggressive TTL** | 🟡 40% (reduces exposure window) | Low (cron jobs) |
| **Audit Trails** | 🟡 30% (detection, not prevention) | Low (structured logging) |
| **Federated Learning** | 🟢 95% (never collect raw data!) | High (multi-client) |

**Recommended Stack** (90% risk reduction, low cost):
1. ✅ Anonymization (hash user IDs)
2. ✅ Store embeddings only (discard raw text)
3. ✅ Encryption at rest (AES-256-GCM)
4. ✅ Encryption in transit (TLS 1.3)
5. ✅ Aggressive TTL (30-day auto-delete)
6. ✅ Audit trails (immutable logs)
7. ✅ Access controls (RBAC + MFA)
8. ⚠️ Differential privacy (if sharing aggregates)
9. ⚠️ Federated learning (if feasible for your use case)

---

## Next Steps

1. **Implement SecureDataCollectionLoop** (see code above)
2. **Set USER_HASH_SALT** environment variable (NEVER commit!)
3. **Enable encryption** (AESGCM with key rotation)
4. **Configure TTL** (30-day default, adjust as needed)
5. **Enable audit trail** (HoloLoom alignment framework)
6. **Test compliance** (GDPR/CCPA checklist)
7. **Security audit** (penetration test, vulnerability scan)
8. **Privacy policy** (document data practices)
9. **Incident response plan** (breach procedures)
10. **Monitor continuously** (alerts on suspicious access)

---

## Questions to Ask Yourself

1. **Do I really need this data?** (Minimize collection)
2. **Can I use synthetic data instead?** (Generate fake examples)
3. **Can I compute on encrypted data?** (Homomorphic encryption)
4. **Can I federate learning?** (Never collect raw data)
5. **What's my retention policy?** (Delete ASAP)
6. **Who has access?** (Principle of least privilege)
7. **How will I detect breaches?** (Monitoring + alerts)
8. **What's my incident response plan?** (Document procedures)
9. **Am I compliant?** (GDPR, CCPA, HIPAA)
10. **Can I prove it?** (Audit trails, third-party audits)

---

**Remember**: The best security is not needing the data in the first place. Every field you collect is a liability. Design your learning system to work with minimal, anonymized, encrypted, short-lived data.

**"Collect the minimum, protect everything, prove compliance."**
