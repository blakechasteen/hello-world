# HoloLoom Security Onboarding

**Welcome to HoloLoom!**

This document provides essential security training for all new team members. Security is everyone's responsibility, and understanding our security infrastructure is critical to protecting our users and maintaining compliance.

---

## Table of Contents

1. [Security Overview](#security-overview)
2. [Your Security Responsibilities](#your-security-responsibilities)
3. [Secure Coding Practices](#secure-coding-practices)
4. [Incident Response](#incident-response)
5. [Compliance Requirements](#compliance-requirements)
6. [Tools and Resources](#tools-and-resources)

---

## Security Overview

### HoloLoom's Security Posture

**Security Level**: 4.5 / 5.0 (99% Secure)
**Architecture**: 10-layer defense-in-depth
**Compliance**: SOC2 (98%), GDPR (97%), ISO 27001 (100%)

### 10-Layer Defense

1. **Network Security** - WAF, DDoS protection, TLS 1.3
2. **Authentication** - OAuth2, JWT, MFA
3. **Authorization** - RBAC (4 roles, 17 permissions)
4. **Rate Limiting** - Distributed (Redis-backed)
5. **Input Validation** - Pydantic schemas, SQL injection prevention
6. **Privacy** - Differential privacy, PII anonymization
7. **Secrets** - Fernet encryption, rotation
8. **Monitoring** - SIEM, ML anomaly detection, dashboards
9. **Incident Response** - SOAR playbooks, forensic logging
10. **Compliance** - SOC2, GDPR, ISO 27001 automation

---

## Your Security Responsibilities

### All Team Members

✅ **DO**:
- Use strong, unique passwords (minimum 12 characters)
- Enable MFA on all accounts (GitHub, Slack, Email, etc.)
- Keep software up-to-date (OS, browsers, IDEs)
- Report security incidents immediately (Slack #security)
- Complete annual security training
- Follow data handling policies (no PII in logs)
- Lock your workstation when away (Windows+L / Ctrl+Alt+L)

❌ **DON'T**:
- Share passwords or API keys
- Commit secrets to Git (use `.env` files, never commit)
- Click suspicious links (phishing awareness)
- Use public WiFi without VPN
- Store credentials in plaintext
- Bypass security controls

### Developers

Additional responsibilities:
- Follow secure coding standards (OWASP Top 10)
- Run security linters before committing (`safety check`, `bandit`)
- Never disable security features (WAF, RBAC, rate limiting)
- Use parameterized queries (prevent SQL injection)
- Validate all user input (Pydantic schemas)
- Handle secrets securely (environment variables, never hardcode)
- Perform code reviews with security mindset

### DevOps/SRE

Additional responsibilities:
- Maintain infrastructure security (firewalls, patching)
- Monitor security alerts (Grafana, SIEM)
- Respond to incidents (follow SOAR playbooks)
- Manage secrets rotation (quarterly)
- Perform backups (daily, encrypted)
- Apply security updates within 48 hours (critical vulnerabilities)

---

## Secure Coding Practices

### OWASP Top 10 Prevention

#### 1. Broken Access Control

**Problem**: Users accessing data/functions they shouldn't.

**Prevention**:
```python
# ✅ Good: Use RBAC decorator
from HoloLoom.security.rbac import require_permission, Permission

@require_permission(Permission.ADMIN_WRITE)
async def delete_user(user_id: str):
    # Only admins can delete users
    pass

# ❌ Bad: No permission check
async def delete_user(user_id: str):
    await db.delete(user_id)  # Anyone can delete!
```

**Checklist**:
- [ ] Every endpoint has authentication
- [ ] RBAC decorators on sensitive functions
- [ ] Ownership checks (user A can't access user B's data)
- [ ] No direct object references (use UUIDs, not sequential IDs)

---

#### 2. Cryptographic Failures

**Problem**: Weak or missing encryption.

**Prevention**:
```python
# ✅ Good: Use encryption for PII
from HoloLoom.security.encryption import encrypt_pii, decrypt_pii

async def store_user_email(user_id: str, email: str):
    encrypted_email = encrypt_pii(email)  # AES-256-GCM
    await db.store(user_id, encrypted_email)

# ❌ Bad: Plaintext PII
async def store_user_email(user_id: str, email: str):
    await db.store(user_id, email)  # Plaintext!
```

**Checklist**:
- [ ] All PII encrypted at rest (AES-256-GCM)
- [ ] TLS 1.3 for data in transit
- [ ] Strong password hashing (PBKDF2-HMAC-SHA256, 100k iterations)
- [ ] No hardcoded secrets (use environment variables)

---

#### 3. Injection

**Problem**: Untrusted data executed as code (SQL, NoSQL, command injection).

**Prevention**:
```python
# ✅ Good: Use ORM (parameterized queries)
from sqlalchemy.orm import Session

async def get_user(session: Session, username: str):
    return session.query(User).filter(User.username == username).first()

# ❌ Bad: String concatenation (SQL injection!)
async def get_user(username: str):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    return await db.execute(query)  # Vulnerable to: ' OR 1=1--
```

**Checklist**:
- [ ] Use SQLAlchemy ORM (no raw SQL)
- [ ] Validate input with Pydantic schemas
- [ ] Never use `eval()`, `exec()`, `pickle.loads()` on user input
- [ ] Command injection: Use `subprocess.run()` with list args, not shell=True

---

#### 4. Insecure Design

**Problem**: Missing security controls by design.

**Prevention**:
```python
# ✅ Good: Rate limiting on auth endpoints
from HoloLoom.security.rate_limiting import rate_limit

@rate_limit(max_requests=5, window_seconds=60)  # 5 attempts per minute
async def login(username: str, password: str):
    # Prevent brute force attacks
    pass

# ❌ Bad: No rate limiting
async def login(username: str, password: str):
    # Attacker can try 1000 passwords/sec
    pass
```

**Checklist**:
- [ ] Rate limiting on all public endpoints
- [ ] Account lockout after 5 failed logins
- [ ] File upload limits (10MB max)
- [ ] Request timeouts (60s max)

---

#### 5. Security Misconfiguration

**Problem**: Default configs, verbose errors, missing headers.

**Prevention**:
```python
# ✅ Good: Generic error messages
from fastapi import HTTPException

async def get_user(user_id: str):
    user = await db.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# ❌ Bad: Stack trace exposed
async def get_user(user_id: str):
    try:
        return await db.get(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # Exposes DB schema!
```

**Checklist**:
- [ ] No default credentials (change all defaults)
- [ ] Debug mode OFF in production (`DEBUG=False`)
- [ ] Generic error messages (no stack traces)
- [ ] Security headers present (HSTS, CSP, X-Frame-Options)

---

#### 6. Vulnerable Components

**Problem**: Outdated dependencies with known CVEs.

**Prevention**:
```bash
# ✅ Good: Check for vulnerabilities before deploying
safety check  # Python
npm audit     # Node.js

# Update vulnerable packages
pip install --upgrade package-name
```

**Checklist**:
- [ ] Run `safety check` in CI/CD pipeline
- [ ] Update dependencies monthly
- [ ] Subscribe to security advisories (GitHub Dependabot)
- [ ] Pin versions in `requirements.txt` (avoid `package==*`)

---

#### 7. Identification & Authentication Failures

**Problem**: Weak passwords, missing MFA, session hijacking.

**Prevention**:
```python
# ✅ Good: Enforce strong passwords
from pydantic import BaseModel, validator

class UserRegistration(BaseModel):
    password: str

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 12:
            raise ValueError("Password must be at least 12 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain uppercase")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain digit")
        return v

# ❌ Bad: Accept weak passwords
class UserRegistration(BaseModel):
    password: str  # "123456" accepted!
```

**Checklist**:
- [ ] Password minimum 12 characters
- [ ] Complexity requirements (upper, lower, digit, special)
- [ ] MFA enforced for admin accounts
- [ ] Session timeout (30 minutes)
- [ ] Regenerate session ID on login

---

#### 8. Software & Data Integrity

**Problem**: Unsigned updates, insecure deserialization.

**Prevention**:
```python
# ✅ Good: Use JSON (safe)
import json

data = json.loads(user_input)

# ❌ Bad: Use pickle (dangerous!)
import pickle

data = pickle.loads(user_input)  # Remote code execution!
```

**Checklist**:
- [ ] Never use `pickle.loads()` on untrusted data
- [ ] Verify GPG signatures on software updates
- [ ] Hash chain for forensic logs (SHA-256)
- [ ] Code signing for Docker images

---

#### 9. Security Logging & Monitoring Failures

**Problem**: No logs for security events, no alerting.

**Prevention**:
```python
# ✅ Good: Log security events
import logging

logger = logging.getLogger(__name__)

async def admin_action(user_id: str, action: str):
    logger.info(f"Admin action: user={user_id}, action={action}")
    await audit_trail.log(user_id, action)  # Immutable audit log

# ❌ Bad: No logging
async def admin_action(user_id: str, action: str):
    # No record of admin actions!
    pass
```

**Checklist**:
- [ ] Log all authentication events (login, logout, failed attempts)
- [ ] Log all admin actions
- [ ] Log security events (SQL injection attempts, rate limit violations)
- [ ] Retain logs for 365 days (SOC2/ISO 27001 requirement)
- [ ] Never log PII or secrets

---

#### 10. Server-Side Request Forgery (SSRF)

**Problem**: Attacker makes server request internal resources.

**Prevention**:
```python
# ✅ Good: Validate and block internal IPs
from ipaddress import ip_address, ip_network

BLOCKED_NETWORKS = [
    ip_network("10.0.0.0/8"),
    ip_network("172.16.0.0/12"),
    ip_network("192.168.0.0/16"),
    ip_network("169.254.169.254/32"),  # AWS metadata
]

async def fetch_url(url: str):
    parsed = urlparse(url)
    ip = ip_address(socket.gethostbyname(parsed.hostname))

    for network in BLOCKED_NETWORKS:
        if ip in network:
            raise ValueError(f"Blocked IP: {ip}")

    return await httpx.get(url)

# ❌ Bad: No validation
async def fetch_url(url: str):
    return await httpx.get(url)  # Can access http://localhost:6379
```

**Checklist**:
- [ ] Validate URLs before fetching
- [ ] Block internal IP ranges
- [ ] Block cloud metadata IPs (169.254.169.254)
- [ ] Use allow-list (not deny-list) for URL schemes (http, https only)

---

### Secrets Management

**Never commit secrets to Git!**

**✅ Good Practices**:
```python
# Use environment variables
import os

API_KEY = os.getenv("API_KEY")  # From .env file
if not API_KEY:
    raise ValueError("API_KEY not set")
```

**Check before committing**:
```bash
# Scan for secrets in staged files
git diff --cached | grep -i "password\|secret\|api_key"

# Use git-secrets tool
git secrets --scan
```

**If you accidentally commit a secret**:
1. Rotate the secret immediately (generate new key)
2. Remove from Git history: `git filter-branch` or BFG Repo-Cleaner
3. Report to security team (Slack #security)

---

### Code Review Checklist

**Security checklist for reviewers**:
- [ ] No secrets in code
- [ ] Input validation present (Pydantic)
- [ ] Authentication/authorization checks
- [ ] SQL injection prevented (ORM used)
- [ ] Error handling doesn't leak info
- [ ] Rate limiting on public endpoints
- [ ] Logs don't contain PII
- [ ] HTTPS enforced
- [ ] CSRF protection (if forms)
- [ ] XSS prevention (HTML escaping)

---

## Incident Response

### What is a Security Incident?

**Examples**:
- Unauthorized access to systems
- Data breach or exfiltration
- Malware infection
- Phishing email (successful)
- DDoS attack
- Insider threat
- Lost/stolen laptop
- Accidental data exposure (public S3 bucket)

### Reporting Procedure

**If you suspect a security incident**:

1. **DO NOT** investigate further (preserve evidence)
2. **IMMEDIATELY** report to:
   - Slack: #security channel
   - Email: security@hololoom.local
   - PagerDuty: (for after-hours emergencies)

3. **Provide details**:
   - What happened?
   - When did it happen?
   - What systems/data affected?
   - Any evidence (logs, screenshots)?

4. **Security team will**:
   - Acknowledge within 15 minutes
   - Triage severity (Critical, High, Medium, Low)
   - Activate SOAR playbook if needed
   - Communicate status updates

### SOAR Playbooks

HoloLoom has 5 automated incident response playbooks:

1. **SQL Injection Response** (7 steps, <5s execution)
   - Block attacker IP
   - Revoke sessions from that IP
   - Collect forensics (logs, packet captures)
   - Alert security team
   - Create incident ticket

2. **Brute Force Response** (6 steps)
   - Lock targeted account (30 min)
   - Block attacker IP (1 hour)
   - Alert security team

3. **DDoS Mitigation** (8 steps)
   - Enable rate limiting (stricter)
   - Block attacker IPs
   - Scale infrastructure (auto-scaling)

4. **Data Breach Containment** (10 steps)
   - Isolate affected systems
   - Collect forensics
   - Notify GDPR within 72 hours
   - Customer communication

5. **Anomaly Investigation** (5 steps)
   - Collect context (logs, metrics)
   - Correlate with other events
   - Escalate if confirmed threat

**Your role**: Report, don't investigate. Let SOAR + security team handle.

---

## Compliance Requirements

### SOC2 Type II

**What**: Industry-standard security audit
**Why**: Required by enterprise customers
**Your role**:
- Follow security policies
- Complete quarterly access reviews
- Attend annual security training
- Respond to auditor interviews (if selected)

**Evidence you may need to provide**:
- Code review comments (security-focused)
- Training completion certificates
- Access request/approval logs

### GDPR (General Data Protection Regulation)

**What**: EU data privacy law
**Why**: We process EU residents' data
**Your role**:
- Minimize PII collection
- Use differential privacy + anonymization
- Respond to DSRs (Data Subject Requests) within 30 days
- Report data breaches within 72 hours

**Don't log PII**:
```python
# ❌ Bad: Logs PII
logger.info(f"User {user_email} logged in")  # Email is PII!

# ✅ Good: Logs hashed ID
logger.info(f"User {hash_user_id(user_id)} logged in")
```

### ISO 27001

**What**: International security management standard
**Why**: Demonstrates security maturity
**Your role**:
- Follow documented procedures
- Report security incidents
- Complete training

---

## Tools and Resources

### Security Tools

**Pre-commit hooks** (run before every commit):
```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Runs automatically on git commit:
# - safety check (vulnerable dependencies)
# - bandit (security linting)
# - git-secrets (secret detection)
```

**IDE Plugins**:
- **VS Code**: Snyk, SonarLint, GitLens
- **PyCharm**: Security plugin, SQL injection detector

**CLI Tools**:
- `safety check` - Check Python dependencies for CVEs
- `npm audit` - Check Node.js dependencies
- `bandit` - Python security linter
- `git-secrets` - Prevent secret commits

### Documentation

**Essential Reading**:
- [SECURITY_IMPLEMENTATION_COMPLETE.md](../../SECURITY_IMPLEMENTATION_COMPLETE.md) - Full security architecture
- [CLAUDE.md](../../CLAUDE.md) - Developer reference (security section)
- [INCIDENT_RESPONSE_PLAN.md](../INCIDENT_RESPONSE_PLAN.md) - What to do during incidents

**Quick References**:
- [OWASP Top 10](https://owasp.org/www-project-top-ten/) - Top web vulnerabilities
- [OWASP Cheat Sheets](https://cheatsheetseries.owasp.org/) - Security best practices

### Training

**Required Training** (annual):
- [ ] Security Onboarding (this document) - 2 hours
- [ ] Phishing Awareness - 30 minutes
- [ ] Secure Coding (OWASP Top 10) - 3 hours
- [ ] Incident Response - 1 hour

**Optional Training** (recommended):
- SANS Security Awareness
- OWASP Secure Coding Practices
- Cloud Security Fundamentals (AWS/GCP/Azure)

### Contacts

**Security Team**:
- CISO: ciso@hololoom.local
- Security Lead: security@hololoom.local
- Incident Response: Slack #security, PagerDuty

**Emergency**: PagerDuty (critical incidents only)

---

## Assessment

**Complete the security quiz to finish onboarding**: [Security Quiz](https://forms.hololoom.local/security-quiz)

**Topics covered**:
- OWASP Top 10 (10 questions)
- Secrets management (5 questions)
- Incident reporting (5 questions)
- Compliance (5 questions)

**Pass requirement**: 20/25 (80%)

---

## Acknowledgment

By signing below, I acknowledge that I have read and understood HoloLoom's security policies and will follow secure coding practices.

**Name**: ______________________
**Date**: ______________________
**Signature**: ______________________

**Manager Approval**: ______________________
**Date**: ______________________

---

**Welcome to the team! Let's build secure software together.** 🔒

**Document Version**: 1.0
**Last Updated**: November 2025
**Next Review**: November 2026
