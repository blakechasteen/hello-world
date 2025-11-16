# HoloLoom Penetration Testing Checklist

**Quick reference for penetration testers**
**Use alongside**: [PENETRATION_TESTING_PLAN.md](PENETRATION_TESTING_PLAN.md)

---

## Pre-Engagement

- [ ] ROE (Rules of Engagement) signed
- [ ] NDA signed
- [ ] Scope document approved
- [ ] Emergency contacts confirmed
- [ ] Testing windows agreed
- [ ] Test credentials received
- [ ] Network diagrams reviewed
- [ ] Backup communication channel tested

---

## Reconnaissance

### Passive
- [ ] DNS enumeration (nslookup, dig, whois)
- [ ] Subdomain discovery (Sublist3r, Amass)
- [ ] Google dorking
- [ ] GitHub/GitLab code search
- [ ] Shodan/Censys scans
- [ ] Certificate Transparency logs (crt.sh)
- [ ] LinkedIn employee enumeration

### Active
- [ ] Port scan (Nmap TCP SYN)
- [ ] Service version detection
- [ ] OS fingerprinting
- [ ] UDP scan (DNS, SNMP, NTP)
- [ ] WAF detection (wafw00f)
- [ ] SSL/TLS testing (testssl.sh)
- [ ] Technology fingerprinting (Wappalyzer)
- [ ] API endpoint discovery

---

## OWASP Top 10

### A01: Broken Access Control
- [ ] Vertical privilege escalation (low → admin)
- [ ] Horizontal privilege escalation (user A → user B)
- [ ] IDOR (Insecure Direct Object References)
- [ ] Missing function-level access control
- [ ] Force browsing to admin panels
- [ ] Parameter tampering (role=admin)

### A02: Cryptographic Failures
- [ ] Weak SSL/TLS configuration (testssl.sh)
- [ ] Unencrypted data transmission (HTTP)
- [ ] Weak password hashing
- [ ] Hardcoded secrets in code
- [ ] Sensitive data in logs/cache
- [ ] Weak encryption algorithms (DES, MD5)

### A03: Injection
- [ ] SQL Injection (`' OR 1=1--`)
- [ ] NoSQL Injection (`{"$ne": null}`)
- [ ] Command Injection (`; cat /etc/passwd`)
- [ ] LDAP Injection
- [ ] XPath Injection
- [ ] Template Injection (SSTI)
- [ ] XML Injection (XXE)

### A04: Insecure Design
- [ ] Missing rate limiting
- [ ] No account lockout
- [ ] Weak password policy
- [ ] Resource exhaustion (large file upload)
- [ ] Business logic flaws
- [ ] Insecure direct object references

### A05: Security Misconfiguration
- [ ] Default credentials
- [ ] Directory listing enabled
- [ ] Verbose error messages
- [ ] Unnecessary HTTP methods (TRACE, PUT)
- [ ] Missing security headers
- [ ] Exposed admin panels
- [ ] Default ports/services

### A06: Vulnerable Components
- [ ] Outdated software versions
- [ ] Known CVEs (safety check, npm audit)
- [ ] Unpatched dependencies
- [ ] End-of-life software
- [ ] Vulnerable libraries (Log4Shell, etc.)

### A07: Identification & Authentication
- [ ] Weak password requirements
- [ ] Missing MFA
- [ ] Session fixation
- [ ] Insecure password recovery
- [ ] Credential stuffing
- [ ] Brute force attacks
- [ ] Session timeout testing

### A08: Software & Data Integrity
- [ ] Unsigned software updates
- [ ] Insecure deserialization
- [ ] Missing integrity checks
- [ ] CI/CD pipeline vulnerabilities
- [ ] Supply chain attacks

### A09: Logging & Monitoring Failures
- [ ] Missing audit logs
- [ ] Insufficient log retention
- [ ] No tamper detection
- [ ] Missing alerting
- [ ] Log injection
- [ ] Sensitive data in logs

### A10: SSRF (Server-Side Request Forgery)
- [ ] SSRF to internal services (`http://localhost:6379`)
- [ ] SSRF to cloud metadata (`http://169.254.169.254`)
- [ ] SSRF to internal network (`http://172.25.0.1`)
- [ ] DNS rebinding attacks

---

## Authentication & Authorization

### OAuth2/OpenID Connect
- [ ] Authorization code interception
- [ ] Token leakage (logs/cache)
- [ ] Token replay
- [ ] Scope escalation
- [ ] Open redirect
- [ ] PKCE bypass

### RBAC
- [ ] Role hierarchy bypass
- [ ] Permission tampering (JWT)
- [ ] Temporal permission bypass (TTL)
- [ ] Role enumeration
- [ ] Policy injection

### JWT
- [ ] None algorithm attack
- [ ] Weak secret brute force
- [ ] Kid header injection
- [ ] JWK injection
- [ ] Signature bypass
- [ ] Expired token acceptance

---

## API Security

### OWASP API Security Top 10
- [ ] Broken object level authorization
- [ ] Broken user authentication
- [ ] Excessive data exposure
- [ ] Lack of resources & rate limiting
- [ ] Broken function level authorization
- [ ] Mass assignment
- [ ] Security misconfiguration
- [ ] Injection
- [ ] Improper assets management
- [ ] Insufficient logging & monitoring

### REST API Specific
- [ ] HTTP verb tampering
- [ ] API versioning issues
- [ ] CORS misconfiguration
- [ ] GraphQL introspection enabled
- [ ] Batch request abuse
- [ ] API key exposure

---

## Network Security

### Firewall & WAF
- [ ] WAF bypass (obfuscation)
- [ ] IP whitelisting bypass (X-Forwarded-For)
- [ ] Geographic blocking bypass (VPN/proxy)
- [ ] WAF fingerprinting
- [ ] Rule evasion testing

### Network Services
- [ ] Open ports (unnecessary services)
- [ ] Unencrypted protocols (FTP, Telnet)
- [ ] Weak SSH configuration
- [ ] Default SNMP community strings
- [ ] NTP amplification

---

## Database Security

### PostgreSQL
- [ ] Direct database access (port 5432)
- [ ] Weak database credentials
- [ ] SQL injection (application level)
- [ ] Privilege escalation (PostgreSQL user)
- [ ] Data exfiltration

### Redis
- [ ] Unauthenticated access
- [ ] Weak password
- [ ] Command injection
- [ ] Data dump
- [ ] Lua sandbox escape

### Neo4j
- [ ] Default credentials (neo4j/neo4j)
- [ ] Cypher injection
- [ ] Unencrypted Bolt protocol
- [ ] Graph traversal attacks

---

## Web Application

### XSS (Cross-Site Scripting)
- [ ] Reflected XSS (`<script>alert(1)</script>`)
- [ ] Stored XSS (comment field)
- [ ] DOM-based XSS
- [ ] Blind XSS (webhook callback)
- [ ] mXSS (mutation XSS)
- [ ] XSS filter bypass

### CSRF (Cross-Site Request Forgery)
- [ ] Missing CSRF token
- [ ] Weak CSRF token
- [ ] CSRF token in URL
- [ ] SameSite cookie not set
- [ ] Referer header check bypass

### File Upload
- [ ] Executable file upload (.php, .jsp)
- [ ] Path traversal (../../etc/passwd)
- [ ] MIME type bypass
- [ ] File size bypass
- [ ] Double extension (.php.jpg)
- [ ] Malicious file content (web shell)

### SSRF
- [ ] Internal service access
- [ ] Cloud metadata access
- [ ] Port scanning via SSRF
- [ ] Protocol smuggling (file://, gopher://)

---

## Infrastructure

### Docker
- [ ] Docker escape
- [ ] Exposed Docker socket
- [ ] Privileged containers
- [ ] Host path mounts
- [ ] Weak container isolation

### CI/CD
- [ ] Exposed .git directory
- [ ] CI/CD secrets in environment
- [ ] Pipeline injection
- [ ] Artifact poisoning
- [ ] Build script tampering

---

## Post-Exploitation

### Lateral Movement
- [ ] Internal network scanning
- [ ] Credential harvesting
- [ ] Pass-the-hash
- [ ] Kerberos attacks (if AD)
- [ ] SSH key theft

### Persistence
- [ ] Backdoor accounts
- [ ] Cron job modification
- [ ] Web shell upload
- [ ] Rootkit installation
- [ ] SSH authorized_keys

### Data Exfiltration
- [ ] Large data transfer
- [ ] DNS tunneling
- [ ] Steganography
- [ ] Encrypted channel (HTTPS)
- [ ] Slow exfiltration

---

## Incident Response Testing

### SOAR Playbooks
- [ ] SQL injection playbook triggered
- [ ] Brute force playbook triggered
- [ ] DDoS mitigation playbook triggered
- [ ] Data breach playbook triggered
- [ ] Anomaly investigation playbook triggered

### Alerting
- [ ] Slack alerts received
- [ ] Email alerts received
- [ ] PagerDuty escalation
- [ ] SMS alerts (Twilio)

### Forensic Logging
- [ ] Events logged to PostgreSQL
- [ ] Hash chain integrity verified
- [ ] Tamper detection working
- [ ] Log search performance (<100ms)
- [ ] Compliance export generated

---

## Compliance Validation

### SOC2
- [ ] Access controls tested (CC6.1)
- [ ] Encryption verified (CC6.6)
- [ ] Monitoring validated (CC7.2)
- [ ] Alerting confirmed (CC7.3)

### GDPR
- [ ] Data encryption verified (Article 32)
- [ ] Breach notification tested (Article 33)
- [ ] Privacy by design validated (Article 25)
- [ ] DSR handling tested (Article 15)

### ISO 27001
- [ ] Vulnerability management (A.12.6)
- [ ] Secure development (A.14.2)
- [ ] Incident management (A.16.1)

---

## Reporting Checklist

- [ ] Executive summary written (2-3 pages)
- [ ] Technical report drafted (20-50 pages)
- [ ] All findings documented
- [ ] PoC included for each finding
- [ ] Severity assigned (CVSS)
- [ ] Remediation recommendations
- [ ] Screenshots/evidence attached
- [ ] Timeline for fixes proposed
- [ ] Re-testing plan outlined
- [ ] Compliance impact assessed

---

## Remediation Verification

- [ ] Critical findings fixed (24 hours)
- [ ] High findings fixed (7 days)
- [ ] Medium findings fixed (30 days)
- [ ] Low findings fixed (90 days)
- [ ] Re-test confirmed fixes
- [ ] No new vulnerabilities introduced
- [ ] Documentation updated
- [ ] Security baseline updated

---

## Sign-Off

**Tester**: ______________________
**Date**: ______________________
**Signature**: ______________________

**Client Approval**: ______________________
**Date**: ______________________
**Signature**: ______________________

---

**Document Version**: 1.0
**Last Updated**: November 2025
**Next Review**: November 2026
