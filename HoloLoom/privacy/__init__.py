"""
HoloLoom Privacy and Compliance Module
=======================================

**Purpose**: Complete tenant isolation, PII protection, and compliance automation
for multi-tenant HoloLoom deployments.

**Date**: 2025-11-17
**Phase**: Tenant Isolation and PII Flows

## Features

### 1. PII Detection and Classification
- 15+ PII entity types (SSN, email, phone, credit card, IP address, etc.)
- Regex-based detection (fast, zero dependencies)
- Confidence scoring and automatic redaction
- Luhn algorithm validation for credit cards

### 2. Tenant Isolation
- Tenant-scoped memory namespacing
- Row-level security enforcement
- Cross-tenant access prevention
- Quota management and rate limiting
- Complete audit logging

### 3. PII Flow Tracking
- End-to-end PII lifecycle tracking
- Automatic detection at each processing stage
- Complete audit trail with timestamps
- Data lineage visualization
- Right-to-erasure automation

### 4. Data Encryption
- AES-256-GCM encryption for data at rest
- Per-tenant encryption keys
- Automatic key rotation (monthly/quarterly/yearly)
- Field-level encryption for PII
- Integration-ready for AWS KMS, Azure Key Vault

### 5. Compliance Reporting
- GDPR Article 30 records of processing
- GDPR Article 15 data subject access requests (DSAR)
- GDPR Article 17 right to erasure
- HIPAA §164.308/§164.312 safeguards reporting
- SOC 2 Trust Service Criteria monitoring

## Quick Start

```python
from HoloLoom.privacy import (
    # PII Detection
    create_default_detector,
    PIIType,

    # Tenant Isolation
    TenantRegistry,
    TenantIsolationLayer,
    TenantContext,
    TenantTier,

    # PII Flow Tracking
    PIIFlowTracker,
    PIIFlowStage,
    PIIAction,

    # Encryption
    create_encryption_manager,

    # Compliance
    ComplianceManager,
)

# 1. Create PII detector
detector = create_default_detector()

# Analyze text for PII
result = detector.analyze("My email is user@example.com and SSN is 123-45-6789")
print(f"Risk level: {result.risk_level}")  # "CRITICAL" (SSN detected)
print(f"Redacted: {result.redacted_text}")  # "My email is [EMAIL_REDACTED]..."

# 2. Set up tenant isolation
registry = TenantRegistry()

# Create tenant
tenant = await registry.create_tenant(
    tenant_id="acme_corp",
    name="Acme Corporation",
    tier=TenantTier.ENTERPRISE,
    compliance_mode="hipaa",
    encryption_required=True
)

# Create isolation layer
isolation = TenantIsolationLayer(registry)

# Create execution context
context = TenantContext(
    tenant_id="acme_corp",
    user_id="john.doe@acme.com",
    permissions={"read", "write"}
)

# Validate operation before execution
await isolation.validate_operation(context, operation="write")

# Scope memory keys to tenant
scoped_key = isolation.scope_key("memory_123", context.tenant_id)
# Result: "tenant:acme_corp:memory_123"

# 3. Track PII flows
tracker = PIIFlowTracker(detector)

# Track ingestion
ingestion_event = await tracker.track_ingestion(
    text="User email is john@acme.com",
    context=context,
    purpose="User query"
)

# Track storage
storage_event = await tracker.track_storage(
    data={"email": "john@acme.com"},
    context=context,
    parent_event_id=ingestion_event.event_id,
    purpose="Store to knowledge graph"
)

# 4. Encrypt sensitive data
encryption_manager = create_encryption_manager()

# Generate key for tenant
key = encryption_manager.generate_key_for_tenant("acme_corp")

# Encrypt PII
encrypted = encryption_manager.encrypt_string(
    plaintext="user@example.com",
    tenant_id="acme_corp",
    metadata={"field": "email"}
)

# Decrypt PII
plaintext = encryption_manager.decrypt_string(encrypted, tenant_id="acme_corp")

# 5. Generate compliance reports
compliance = ComplianceManager(
    tenant_registry=registry,
    pii_tracker=tracker,
    detector=detector
)

# GDPR Article 30 report
gdpr_report = await compliance.generate_gdpr_article30_report(
    tenant_id="acme_corp",
    start_date=datetime(2025, 10, 1),
    end_date=datetime(2025, 11, 1)
)

# Handle DSAR
dsar = await compliance.handle_dsar(
    tenant_id="acme_corp",
    data_subject_email="john@acme.com"
)

# Handle right to erasure
erasure = await compliance.handle_right_to_erasure(
    tenant_id="acme_corp",
    data_subject_email="john@acme.com"
)

# HIPAA report
hipaa_report = await compliance.generate_hipaa_report(tenant_id="acme_corp")
```

## Integration with HoloLoom Memory Systems

### Knowledge Graph Integration

```python
from HoloLoom.memory.graph import KG
from HoloLoom.privacy import TenantIsolationLayer, TenantContext

# Create tenant-isolated knowledge graph
kg = KG()
isolation = TenantIsolationLayer(registry)

context = TenantContext(
    tenant_id="acme_corp",
    user_id="john@acme.com",
    permissions={"read", "write"}
)

# Add edge with tenant scoping
edge_id = isolation.scope_key("edge_123", context.tenant_id)
kg.add_edge({
    "id": edge_id,
    "src": "entity_1",
    "dst": "entity_2",
    "type": "RELATED_TO"
})

# Retrieve with tenant filtering
edges = kg.get_edges()
tenant_edges = await isolation.filter_cross_tenant_data(
    edges,
    tenant_id=context.tenant_id,
    key_field="id"
)
```

### Vector Store Integration (Qdrant)

```python
from HoloLoom.privacy import EncryptionManager, TenantContext

encryption = EncryptionManager()
context = TenantContext(tenant_id="acme_corp", ...)

# Encrypt embedding before storing
embedding = [0.1, 0.2, 0.3, ...]  # Your embedding vector
encrypted_embedding = encryption.encrypt(
    plaintext=json.dumps(embedding).encode(),
    tenant_id=context.tenant_id,
    metadata={"type": "embedding"}
)

# Store to Qdrant with tenant scoping
point_id = isolation.scope_key("point_123", context.tenant_id)
qdrant_client.upsert(
    collection_name="embeddings",
    points=[{
        "id": point_id,
        "vector": encrypted_embedding.ciphertext,
        "payload": {
            "tenant_id": context.tenant_id,
            "encrypted": True,
            "key_id": encrypted_embedding.key_id
        }
    }]
)
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Application Layer                    │
│         (HoloLoom Orchestrator, Departments)        │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│               Privacy & Compliance Layer             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │PII Detection│  │Tenant Isol.  │  │PII Tracking │ │
│  │  • 15+ types│  │• Namespacing │  │• Audit trail│ │
│  │  • Redaction│  │• Access ctrl │  │• Lineage    │ │
│  └─────────────┘  └──────────────┘  └─────────────┘ │
│  ┌─────────────┐  ┌──────────────┐                  │
│  │ Encryption  │  │  Compliance  │                  │
│  │• AES-256-GCM│  │• GDPR/HIPAA  │                  │
│  │• Per-tenant │  │• SOC 2       │                  │
│  └─────────────┘  └──────────────┘                  │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│                 Storage Layer                        │
│    (Knowledge Graph, Vector Store, Cache)           │
└─────────────────────────────────────────────────────┘
```

## Compliance Coverage

### GDPR
- ✅ Article 30: Records of processing activities
- ✅ Article 15: Data subject access requests (DSAR)
- ✅ Article 17: Right to erasure
- ✅ Article 32: Security of processing (encryption)
- 🟡 Article 33: Breach notification (requires incident response integration)

### HIPAA
- ✅ §164.308: Administrative safeguards
- ✅ §164.312: Technical safeguards (access control, audit, encryption)
- 🟡 §164.530: Administrative requirements (requires policy documentation)

### SOC 2
- ✅ CC6: Logical access controls
- ✅ CC7: System monitoring
- 🟡 CC8: Change management (requires change tracking integration)

## Testing

Run comprehensive privacy and compliance tests:

```bash
# All privacy tests
pytest HoloLoom/privacy/tests/ -v

# Specific test modules
pytest HoloLoom/privacy/tests/test_pii_detection.py -v
pytest HoloLoom/privacy/tests/test_tenant_isolation.py -v
pytest HoloLoom/privacy/tests/test_pii_flow_tracking.py -v
pytest HoloLoom/privacy/tests/test_encryption.py -v
pytest HoloLoom/privacy/tests/test_compliance.py -v
```

## Documentation

- **[PRIVACY_ARCHITECTURE.md](PRIVACY_ARCHITECTURE.md)**: Complete architecture guide
- **[COMPLIANCE_GUIDE.md](COMPLIANCE_GUIDE.md)**: GDPR/HIPAA/SOC2 compliance
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)**: Integration with HoloLoom
- **[API_REFERENCE.md](API_REFERENCE.md)**: Complete API documentation

Author: HoloLoom Security Team
Timestamp: 2025-11-17
"""

# PII Detection
from .pii_detection import (
    PIIType,
    PIIDetection,
    PIIAnalysisResult,
    PIIDetector,
    PIIPatterns,
    hash_pii,
    create_default_detector,
)

# Tenant Isolation
from .tenant_isolation import (
    TenantTier,
    TenantStatus,
    TenantMetadata,
    TenantContext,
    TenantIsolationError,
    CrossTenantAccessError,
    TenantSuspendedError,
    TenantQuotaExceededError,
    InsufficientPermissionsError,
    TenantIsolationLayer,
    TenantRegistry,
)

# PII Flow Tracking
from .pii_flow_tracking import (
    PIIFlowStage,
    PIIAction,
    PIIFlowEvent,
    PIIFlowChain,
    PIIFlowTracker,
)

# Encryption
from .encryption import (
    EncryptionAlgorithm,
    KeyRotationPolicy,
    EncryptionKey,
    EncryptedData,
    EncryptionManager,
    derive_key_from_password,
    create_encryption_manager,
)

# Compliance
from .compliance import (
    ComplianceStandard,
    ComplianceLevel,
    GDPRArticle30Report,
    DataSubjectAccessRequest,
    RightToErasureRequest,
    HIPAAComplianceReport,
    SOC2ComplianceReport,
    ComplianceManager,
)


__all__ = [
    # PII Detection
    'PIIType',
    'PIIDetection',
    'PIIAnalysisResult',
    'PIIDetector',
    'PIIPatterns',
    'hash_pii',
    'create_default_detector',

    # Tenant Isolation
    'TenantTier',
    'TenantStatus',
    'TenantMetadata',
    'TenantContext',
    'TenantIsolationError',
    'CrossTenantAccessError',
    'TenantSuspendedError',
    'TenantQuotaExceededError',
    'InsufficientPermissionsError',
    'TenantIsolationLayer',
    'TenantRegistry',

    # PII Flow Tracking
    'PIIFlowStage',
    'PIIAction',
    'PIIFlowEvent',
    'PIIFlowChain',
    'PIIFlowTracker',

    # Encryption
    'EncryptionAlgorithm',
    'KeyRotationPolicy',
    'EncryptionKey',
    'EncryptedData',
    'EncryptionManager',
    'derive_key_from_password',
    'create_encryption_manager',

    # Compliance
    'ComplianceStandard',
    'ComplianceLevel',
    'GDPRArticle30Report',
    'DataSubjectAccessRequest',
    'RightToErasureRequest',
    'HIPAAComplianceReport',
    'SOC2ComplianceReport',
    'ComplianceManager',
]
