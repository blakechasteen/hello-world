# EdWIN AI Tutor - Security Guide

**Comprehensive security best practices and hardening guide**

**Implementation Date**: November 15, 2025

---

## Table of Contents

1. [Security Overview](#security-overview)
2. [Authentication & Authorization](#authentication--authorization)
3. [Network Security](#network-security)
4. [Data Protection](#data-protection)
5. [Infrastructure Security](#infrastructure-security)
6. [Application Security](#application-security)
7. [Monitoring & Incident Response](#monitoring--incident-response)
8. [Compliance](#compliance)
9. [Security Checklist](#security-checklist)

---

## Security Overview

EdWIN implements defense-in-depth security with multiple layers:

1. **Network Layer**: Firewall, TLS, rate limiting
2. **Application Layer**: Authentication, authorization, input validation
3. **Data Layer**: Encryption at rest and in transit
4. **Infrastructure Layer**: Container security, secret management
5. **Monitoring Layer**: Audit logging, intrusion detection

### Security Principles

- **Least Privilege**: Grant minimum permissions required
- **Defense in Depth**: Multiple security layers
- **Fail Secure**: Default to deny access
- **Separation of Duties**: Isolate critical operations
- **Security by Design**: Built-in from the start

---

## Authentication & Authorization

### JWT Token Security

**Current Implementation**:
- HS256 algorithm
- 24-hour access token expiry
- 30-day refresh token expiry
- Secure secret key generation

**Best Practices**:

```python
# Generate secure JWT secret (32 bytes minimum)
import secrets
JWT_SECRET_KEY = secrets.token_urlsafe(32)

# Store in environment variable, NEVER in code
# Use Kubernetes secrets in production
```

**Token Storage**:
- ✅ Store in httpOnly cookies (web)
- ✅ Store in secure storage (mobile: Keychain/KeyStore)
- ❌ NEVER store in localStorage (XSS vulnerable)

### Password Security

**Requirements** (enforced in `auth.py`):
- Minimum 12 characters (production)
- At least 1 uppercase letter
- At least 1 lowercase letter
- At least 1 digit
- At least 1 special character

**Hashing**:
- Algorithm: bcrypt (via passlib)
- Automatically handles salt generation
- Configurable work factor (default: 12)

**Best Practices**:
```python
# NEVER log passwords
logger.info(f"Login attempt for user: {username}")  # ✅
logger.info(f"Password: {password}")  # ❌ NEVER DO THIS

# Always hash before storage
hashed = hash_password(plain_password)

# Use constant-time comparison
if verify_password(input_password, stored_hash):  # ✅
```

### Role-Based Access Control (RBAC)

**Hierarchy**:
```
ADMIN (highest)
  └── Full platform access
    └── TEACHER
        └── Classroom management
          └── PARENT
              └── Read child's data
                └── STUDENT (lowest)
                    └── Own data only
```

**Implementation**:
```python
from EduVerse.edwin.auth import require_role, UserRole

@app.get("/admin/users")
async def list_users(user: User = Depends(require_role(UserRole.ADMIN))):
    # Only admins can access
    return users

@app.get("/student/progress/{student_id}")
async def get_progress(
    student_id: str,
    user: User = Depends(can_access_student_data(student_id))
):
    # Students see own data, parents see children, teachers see classroom
    return progress
```

### Session Management

**Security Measures**:
- Session IDs: 32-byte random tokens
- Stored in Redis with TTL
- Automatic expiration (24 hours)
- IP address tracking (optional)
- User agent validation (optional)

**Logout**:
```python
# Invalidate single session
invalidate_session(session_id)

# Invalidate all user sessions (on password change)
invalidate_user_sessions(user_id)
```

---

## Network Security

### TLS/SSL

**Production Requirements**:
- ✅ TLS 1.2 minimum (1.3 preferred)
- ✅ Strong cipher suites only
- ✅ HSTS enabled
- ✅ Certificate auto-renewal (cert-manager)

**Configure cert-manager**:
```yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@edwin.edu
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

**Ingress TLS**:
```yaml
spec:
  tls:
  - hosts:
    - api.edwin.edu
    secretName: edwin-tls
```

### Firewall Rules

**Kubernetes Network Policies**:

```yaml
# Restrict API pod egress
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: edwin-api-netpol
  namespace: edwin
spec:
  podSelector:
    matchLabels:
      component: api
  policyTypes:
  - Egress
  egress:
  # Allow database access only
  - to:
    - podSelector:
        matchLabels:
          component: neo4j
    ports:
    - protocol: TCP
      port: 7687
  - to:
    - podSelector:
        matchLabels:
          component: qdrant
    ports:
    - protocol: TCP
      port: 6333
  - to:
    - podSelector:
        matchLabels:
          component: redis
    ports:
    - protocol: TCP
      port: 6379
```

### Rate Limiting

**Nginx Ingress**:
```yaml
annotations:
  nginx.ingress.kubernetes.io/limit-rps: "100"
  nginx.ingress.kubernetes.io/limit-connections: "10"
  nginx.ingress.kubernetes.io/limit-burst-multiplier: "2"
```

**Application Level** (FastAPI):
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/login")
@limiter.limit("5/minute")  # 5 attempts per minute
async def login(request: Request):
    ...
```

### CORS Configuration

**Production Settings**:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://api.edwin.edu",
        "https://dashboard.edwin.edu",
        "https://mobile.edwin.edu"
    ],  # NEVER use "*" in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=3600
)
```

---

## Data Protection

### Encryption at Rest

**Database Encryption**:

**Neo4j**:
```yaml
env:
- name: NEO4J_dbms_ssl_policy_bolt_enabled
  value: "true"
- name: NEO4J_dbms_ssl_policy_bolt_base__directory
  value: "/certificates"
```

**Qdrant**:
- Uses file system encryption (e.g., LUKS on Linux)
- Kubernetes PVC encryption via cloud provider

**Secrets Management**:
```bash
# Use Kubernetes secrets (base64 encoded)
kubectl create secret generic edwin-secrets \
  --from-literal=jwt-secret="$(openssl rand -base64 32)" \
  --from-literal=neo4j-password="$(openssl rand -base64 16)" \
  -n edwin
```

### Encryption in Transit

**All network traffic encrypted**:
- Client ↔ Ingress: TLS 1.2+
- Ingress ↔ Services: mTLS (optional with service mesh)
- Services ↔ Databases: TLS connections

**Neo4j TLS**:
```python
NEO4J_URI = "bolt+s://neo4j:7687"  # TLS enabled
```

### Sensitive Data Handling

**PII (Personally Identifiable Information)**:
- Student names, emails, grades
- Parent contact information
- Usage analytics

**Protection Measures**:
```python
# Audit logging for PII access
@audit_log(event="PII_ACCESS", resource="student_profile")
async def get_student(student_id: str):
    ...

# Data minimization
# Only store necessary fields
class Student:
    name: str  # Required
    email: str  # Required
    phone: Optional[str]  # Optional, not always collected

# Data retention
# Delete inactive accounts after 2 years
# Archive student data upon graduation
```

---

## Infrastructure Security

### Container Security

**Image Scanning**:
```bash
# Scan images for vulnerabilities
docker scan edwin-ai-tutor:latest

# Use Trivy
trivy image edwin-ai-tutor:latest
```

**Best Practices**:
- ✅ Use official base images (`python:3.11-slim`)
- ✅ Multi-stage builds (reduce attack surface)
- ✅ Non-root user (`USER edwin`)
- ✅ Read-only root filesystem (where possible)
- ✅ Drop unnecessary capabilities

**Security Context**:
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop:
    - ALL
```

### Kubernetes Security

**RBAC**:
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: edwin-api-role
  namespace: edwin
rules:
- apiGroups: [""]
  resources: ["secrets", "configmaps"]
  verbs: ["get", "list"]
```

**Pod Security Standards**:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: edwin
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

### Secret Rotation

**Automated Rotation** (recommended every 90 days):

```bash
#!/bin/bash
# scripts/rotate-secrets.sh

# Generate new JWT secret
NEW_JWT_SECRET=$(openssl rand -base64 32)

# Update Kubernetes secret
kubectl patch secret edwin-secrets -n edwin \
  -p="{\"data\":{\"JWT_SECRET_KEY\":\"$(echo -n "$NEW_JWT_SECRET" | base64)\"}}"

# Rolling restart API pods to pick up new secret
kubectl rollout restart deployment edwin-api -n edwin
```

---

## Application Security

### Input Validation

**Always validate user input**:
```python
from pydantic import BaseModel, Field, validator

class StudentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    grade: int = Field(..., ge=4, le=12)
    email: EmailStr  # Built-in email validation

    @validator('name')
    def validate_name(cls, v):
        if not v.replace(' ', '').isalpha():
            raise ValueError('Name must contain only letters')
        return v
```

### SQL/NoSQL Injection Prevention

**Use parameterized queries**:
```python
# Cypher (Neo4j) - ✅ SAFE
query = "MATCH (s:Student {student_id: $student_id}) RETURN s"
result = session.run(query, student_id=user_input)

# ❌ UNSAFE - NEVER DO THIS
query = f"MATCH (s:Student {{student_id: '{user_input}'}}) RETURN s"
```

### XSS Prevention

**Content Security Policy**:
```python
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response
```

### File Upload Security

**Validation**:
```python
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.pdf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

async def validate_upload(file: UploadFile):
    # Check extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Invalid file type")

    # Check size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")

    # Check MIME type
    import magic
    mime = magic.from_buffer(content, mime=True)
    if mime not in ['image/jpeg', 'image/png', 'application/pdf']:
        raise HTTPException(400, "Invalid file content")

    return content
```

---

## Monitoring & Incident Response

### Audit Logging

**Log all security events**:
```python
import logging

security_logger = logging.getLogger("edwin.security")

# Log authentication events
security_logger.info(f"LOGIN_SUCCESS: user={username}, ip={ip_address}")
security_logger.warning(f"LOGIN_FAILED: user={username}, ip={ip_address}")

# Log authorization failures
security_logger.warning(f"UNAUTHORIZED_ACCESS: user={user_id}, resource={resource}")

# Log data access
security_logger.info(f"PII_ACCESS: user={user_id}, student={student_id}")
```

**Centralized Logging**:
- Use ELK Stack (Elasticsearch, Logstash, Kibana)
- Or cloud provider logging (CloudWatch, Stackdriver)

### Intrusion Detection

**Monitor for suspicious activity**:
- Multiple failed login attempts
- Access to sensitive endpoints
- Unusual API usage patterns
- Database query anomalies

**Alerts**:
```python
# Alert on 10 failed logins in 1 minute
if failed_login_count > 10:
    send_alert(
        severity="HIGH",
        message=f"Brute force attack detected: {ip_address}"
    )
```

### Incident Response Plan

**Severity Levels**:
1. **CRITICAL**: Data breach, system compromise
2. **HIGH**: Authentication bypass, privilege escalation
3. **MEDIUM**: Successful brute force, DoS attack
4. **LOW**: Failed login attempts, suspicious activity

**Response Procedure**:
1. **Detect**: Automated alerts + monitoring
2. **Contain**: Block attacker IP, revoke compromised credentials
3. **Investigate**: Review logs, identify scope
4. **Remediate**: Patch vulnerabilities, restore from backup
5. **Post-Mortem**: Document incident, improve defenses

---

## Compliance

### FERPA (Family Educational Rights and Privacy Act)

**Requirements**:
- ✅ Student data encryption
- ✅ Access controls (only teachers/parents/student)
- ✅ Audit logging of data access
- ✅ Data retention policies
- ✅ Parental consent for minors

### COPPA (Children's Online Privacy Protection Act)

**Requirements** (for students under 13):
- ✅ Parental consent required
- ✅ Limited data collection
- ✅ No third-party sharing without consent
- ✅ Right to review/delete child's data

**Implementation**:
```python
class Student(BaseModel):
    age: int

    @validator('age')
    def require_consent(cls, v):
        if v < 13:
            # Check parental consent
            if not has_parental_consent():
                raise ValueError("Parental consent required")
        return v
```

### GDPR (General Data Protection Regulation)

**If serving EU users**:
- ✅ Right to access data
- ✅ Right to be forgotten (data deletion)
- ✅ Data portability
- ✅ Consent management
- ✅ Data breach notification (72 hours)

---

## Security Checklist

### Pre-Deployment

- [ ] Change all default passwords
- [ ] Generate secure JWT secret (32+ bytes)
- [ ] Configure TLS certificates
- [ ] Set up firewall rules
- [ ] Enable rate limiting
- [ ] Configure CORS properly
- [ ] Scan Docker images for vulnerabilities
- [ ] Review Kubernetes RBAC policies
- [ ] Set up audit logging
- [ ] Configure backup strategy

### Post-Deployment

- [ ] Verify TLS is working (https://)
- [ ] Test authentication/authorization
- [ ] Verify database encryption
- [ ] Check audit logs are working
- [ ] Test rate limiting
- [ ] Verify backups are running
- [ ] Set up monitoring alerts
- [ ] Document incident response plan
- [ ] Train team on security procedures

### Ongoing

- [ ] Rotate secrets every 90 days
- [ ] Review audit logs weekly
- [ ] Scan for vulnerabilities monthly
- [ ] Update dependencies quarterly
- [ ] Security audit annually
- [ ] Incident response drills bi-annually

---

## Security Contacts

- **Security Team**: security@edwin.edu
- **Incident Response**: incident@edwin.edu
- **Bug Bounty**: https://edwin.edu/security/bug-bounty

---

**Last Updated**: November 15, 2025
