# mythRL Safety-First Checklist
**Quick Reference for Safe Development and Deployment**

---

## 🚨 CRITICAL - Never Do This

### ❌ 1. Arbitrary Code Execution
```python
# DANGEROUS - Never use eval() with any user input
eval(user_input)  # 💀 Arbitrary code execution

# DANGEROUS - Never use exec() with user input
exec(user_code)  # 💀 Arbitrary code execution

# SAFE ALTERNATIVE
import ast
ast.literal_eval(safe_string)  # Only parses literals
```

### ❌ 2. Unsafe Deserialization
```python
# DANGEROUS - Never unpickle untrusted data
import pickle
pickle.loads(user_data)  # 💀 Code execution during deserialization

# SAFE ALTERNATIVE
import json
json.loads(user_data)  # Safe, no code execution
```

### ❌ 3. Command Injection
```python
# DANGEROUS - Never use shell=True with user input
subprocess.run(f"command {user_input}", shell=True)  # 💀 Shell injection

# SAFE ALTERNATIVE
subprocess.run(["command", user_input], shell=False)  # ✅ Safe
```

### ❌ 4. SQL Injection
```python
# DANGEROUS - Never concatenate SQL strings
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")  # 💀 SQL injection

# SAFE ALTERNATIVE
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))  # ✅ Parameterized
```

### ❌ 5. Path Traversal
```python
# DANGEROUS - Never directly concatenate paths
open(base_dir + "/" + user_filename)  # 💀 Can access ../../etc/passwd

# SAFE ALTERNATIVE
from pathlib import Path
safe_path = (Path(base_dir) / user_filename).resolve()
if safe_path.is_relative_to(base_dir):
    open(safe_path)  # ✅ Safe
```

---

## ✅ REQUIRED - Always Do This

### 1. Input Validation
```python
# ALWAYS validate before use
def process_hive(hive_data):
    # Validate
    errors = HiveValidator.validate(hive_data)
    if errors:
        raise ValidationError(f"Invalid hive: {errors}")

    # Now safe to use
    return create_hive(hive_data)
```

### 2. Input Sanitization
```python
# ALWAYS sanitize text for display
import html

def sanitize_user_text(text: str) -> str:
    # Remove dangerous characters
    text = text.replace('\x00', '')
    # HTML escape
    text = html.escape(text)
    # Limit length
    text = text[:500]
    return text
```

### 3. Type Safety
```python
# ALWAYS use strong typing
from enum import Enum
from dataclasses import dataclass

class Status(Enum):
    GOOD = "good"
    BAD = "bad"

@dataclass
class Hive:
    status: Status  # ✅ Type-safe, not string
```

### 4. Least Privilege
```python
# ALWAYS run with minimum required permissions

# BAD - Running as root
# sudo python app.py

# GOOD - Run as unprivileged user
# useradd -r -s /bin/false keepapp
# sudo -u keepapp python app.py
```

### 5. Error Handling
```python
# NEVER expose internal details to users
try:
    process_request()
except Exception as e:
    # BAD
    return f"Error: {e}"  # 💀 Leaks stack trace

    # GOOD
    logger.error(f"Processing failed: {e}")
    return "Invalid request"  # ✅ Generic message
```

---

## 🔍 Code Review Checklist

Before committing code, verify:

- [ ] **No dangerous functions**: No `eval()`, `exec()`, `pickle.loads()`
- [ ] **Input validated**: All user input validated before use
- [ ] **Output escaped**: All output sanitized for display context
- [ ] **Parameterized queries**: All SQL uses parameters, not string concat
- [ ] **Safe subprocess**: All subprocess calls use array syntax, no `shell=True`
- [ ] **Path validation**: All file paths checked for traversal
- [ ] **Type safety**: Strong typing used (dataclasses, enums, protocols)
- [ ] **Error handling**: No internal details leaked to users
- [ ] **Secrets management**: No hardcoded credentials
- [ ] **Logging**: No sensitive data in logs

---

## 📋 Pre-Deployment Checklist

Before deploying to production:

### Critical
- [ ] Fix all CRITICAL vulnerabilities (eval, pickle, shell=True)
- [ ] Run security scanner: `bandit -r . -ll`
- [ ] Check dependencies: `safety check`
- [ ] Enable HTTPS only (no HTTP)
- [ ] Set up WAF (Web Application Firewall)

### High Priority
- [ ] Add input validation for all endpoints
- [ ] Add input sanitization for all text fields
- [ ] Implement rate limiting
- [ ] Add CSRF protection
- [ ] Add security headers (CSP, HSTS, X-Frame-Options)
- [ ] Implement audit logging
- [ ] Set up monitoring and alerts

### Medium Priority
- [ ] Add authentication if multi-user
- [ ] Add authorization/RBAC if needed
- [ ] Implement session management
- [ ] Add password policy if applicable
- [ ] Set up backup encryption
- [ ] Document incident response plan

### Low Priority
- [ ] Add security headers (more comprehensive)
- [ ] Implement Content Security Policy
- [ ] Add Subresource Integrity (SRI)
- [ ] Set up security scanning in CI/CD
- [ ] Schedule penetration testing

---

## 🛠️ Security Tools

### Install Security Scanners
```bash
# Python security tools
pip install bandit safety

# Run Bandit (static analysis)
bandit -r . -f json -o bandit_report.json

# Check for known vulnerabilities
safety check --json > safety_report.json
```

### Pre-Commit Hook
```bash
# Add to .git/hooks/pre-commit
#!/bin/bash
echo "Running security checks..."

# Run bandit on changed files
bandit -ll $(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')
if [ $? -ne 0 ]; then
    echo "Security issues found! Fix before committing."
    exit 1
fi

echo "Security checks passed ✅"
```

---

## 📊 Current Security Status

### mythRL Ecosystem

| Component | Status | Critical Issues | Documentation |
|-----------|--------|-----------------|---------------|
| **Keep** | ✅ SAFE | 0 | [SECURITY.md](apps/keep/SECURITY.md) |
| **HoloLoom** | 🔴 UNSAFE | 3 | [SECURITY_AUDIT.md](SECURITY_AUDIT.md) |
| **Promptly** | 🟡 REVIEW | 2 | [SECURITY_AUDIT.md](SECURITY_AUDIT.md) |
| **darkTrace** | 🟡 REVIEW | 1 | [SECURITY_AUDIT.md](SECURITY_AUDIT.md) |

### Priority Fixes Required

1. **CRIT-1**: [smart_operation_selector.py:1081](HoloLoom/warp/math/smart_operation_selector.py#L1081) - Replace `eval()` with `ast.literal_eval()`
2. **CRIT-2**: [trajectory_recorder.py:14](apps/darkTrace/darkTrace/observers/trajectory_recorder.py#L14) - Replace `pickle` with JSON
3. **CRIT-3**: [ultraprompt_ollama.py:138](apps/Promptly/promptly/tools/ultraprompt_ollama.py#L138) - Remove `shell=True`

**Estimated Fix Time**: 8 hours total

---

## 🚀 Safe Deployment Pattern

### Development
```bash
# Local development - use test data only
export KEEP_ENV=development
export KEEP_DEBUG=true
python run_dev_server.py
```

### Staging
```bash
# Staging - test with production-like data
export KEEP_ENV=staging
export KEEP_DEBUG=false
export KEEP_SECRET_KEY=$(openssl rand -hex 32)
export KEEP_DB_URL=postgresql://staging_db
python run_server.py
```

### Production
```bash
# Production - full security
export KEEP_ENV=production
export KEEP_DEBUG=false  # CRITICAL: Never true in prod
export KEEP_SECRET_KEY=$(vault read -field=value secret/keep/secret_key)
export KEEP_DB_URL=$(vault read -field=value secret/keep/db_url)
export KEEP_RATE_LIMIT=100  # Requests per minute
export KEEP_HTTPS_ONLY=true
export KEEP_HSTS_ENABLED=true

# Run as unprivileged user
sudo -u keepapp python run_server.py
```

---

## 📞 Security Incident Response

### If Security Issue Detected

1. **STOP** - Immediately isolate affected system
2. **ASSESS** - Determine scope and impact
3. **CONTAIN** - Stop the attack vector
4. **REMEDIATE** - Deploy fixes
5. **DOCUMENT** - Record timeline
6. **NOTIFY** - Inform affected parties

### Contact
- **Email**: security@mythrl.local (create this)
- **Response Time**: <24 hours for critical issues
- **Private Disclosure**: Create private GitHub issue

---

## 📚 Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

---

## 🎯 Security Goals

### Short Term (1 week)
- [ ] Fix all CRITICAL vulnerabilities
- [ ] Implement input validation
- [ ] Add audit logging

### Medium Term (1 month)
- [ ] Add comprehensive sanitization
- [ ] Implement rate limiting
- [ ] Set up monitoring

### Long Term (3 months)
- [ ] Complete penetration testing
- [ ] Achieve SOC 2 compliance (if applicable)
- [ ] Automated security scanning in CI/CD

---

**Remember**: Security is not a feature, it's a requirement. When in doubt, be more restrictive.

---

**Last Updated**: 2025-10-28
**Version**: 1.0
