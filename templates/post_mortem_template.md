# Incident Post-Mortem Report Template

---

## Post-Mortem Report: [Incident Name]

**Incident ID**: INC-2025-BREACH-0001
**Date Created**: November 19, 2025
**Report Status**: FINAL
**Classification**: CONFIDENTIAL - INTERNAL ONLY

---

## Executive Summary

This post-mortem report documents lessons learned from the data breach incident
that affected HoloLoom on November 16, 2025.

**Key Facts**:
- Incident Severity: SEV-1 (CRITICAL)
- Incident Type: Data Breach (SQL Injection)
- Time to Detect: 12 hours
- Time to Contain: 45 minutes after detection
- Time to Resolve: 2 hours
- Customers Affected: 15,000 out of 200,000 (7.5%)
- Data Exposed: Names, emails, phone numbers, hashed passwords
- Root Cause: SQL injection vulnerability + compromised credentials

**Outcome**:
- ✅ Attacker successfully blocked and traced to law enforcement
- ✅ Vulnerability patched and deployed to production
- ✅ No evidence of secondary compromises
- ✅ All affected customers notified
- ✅ Regulatory agencies notified
- ✅ No customer data fraud detected post-incident

---

## Incident Timeline

### Detailed Chronological Event Log

**November 15, 2025**

| Time | Event | Source | Details |
|------|-------|--------|---------|
| 22:30 UTC | Phishing email sent | External attacker | Email spoofing company HR with password reset link to attacker site |
| 22:45 UTC | Employee clicks phishing link | [Employee Name], HR Dept | Employee enters credentials on attacker site, password captured |

**November 16, 2025**

| Time | Event | Source | Details |
|------|-------|--------|---------|
| 02:15 UTC | First attacker login | IP: 203.0.113.45 | Employee account compromised, attacker gains database access |
| 02:30 UTC | Database reconnaissance | Attacker | Attacker queries database schema (information_schema) |
| 02:45 UTC | SQL injection vulnerability tested | Attacker | Attacker tests UNION injection on login form, succeeds |
| 03:00 UTC | Initial data exfiltration | Attacker | Attacker begins downloading customer database |
| 03:00-06:00 UTC | Continued data transfer | Attacker | 4.2 GB of customer data downloaded to ftp.attacker-ip.ru |
| 06:00 UTC | Data transfer completes | Attacker | Exfiltration successful, attacker maintains access |
| 14:30 UTC | **ALERT FIRED** | SIEM System | "Unusual Database Activity" alert triggered by volume anomaly |
| 14:31 UTC | Alert reviewed | SOC Analyst | Analyst confirms alert is real (not false positive) |
| 14:32 UTC | Incident declared | On-call IC | Incident ticket INC-2025-BREACH-0001 created, SEV-1 assigned |
| 14:35 UTC | Team paged | PagerDuty | Incident Commander, Security Lead, DBA, legal team paged |
| 14:40 UTC | War room assembled | Slack | #incident-war-room-2025-BREACH-0001 created, team joins |
| 14:45 UTC | Forensics firm contacted | IC | Mandiant engaged for investigation |
| 14:50 UTC | Attacker IP blocked | Network Team | Firewall rule added to drop traffic from 203.0.113.45 |
| 14:55 UTC | Database access log collection | DBA | Query logs, audit logs captured for forensics |
| 15:00 UTC | Root cause identified | Security Lead | SQL injection in login form identified |
| 15:15 UTC | Executive notification | IC | CEO, CTO, CISO notified of breach |
| 16:00 UTC | Patch developed | Dev Team | SQL injection fix prepared and tested |
| 16:30 UTC | Patch deployed | DevOps | Fix deployed to production, verified working |
| 17:00 UTC | **INCIDENT CONTAINED** | IC | All attack vectors mitigated, system secured |
| 18:00 UTC | Forensics evidence collected | Mandiant | Memory dumps, disk images, network captures collected |
| 20:00 UTC | Initial forensics analysis | Mandiant | Dwell time (12 hours), exfiltration confirmed |
| 22:00 UTC | Day 1 wrap-up | IC | Status: Investigating, contained, no further threat |

### Post-Detection (Days 2-3)

| Date | Activity | Owner |
|------|----------|-------|
| Nov 17 | Customer notification emails sent | Comms |
| Nov 17 | FAQ published on website | Comms |
| Nov 18 | Press release issued | PR |
| Nov 18 | Complete forensic report delivered | Mandiant |
| Nov 19 | Post-mortem meeting conducted | IC |
| Nov 19 | Post-mortem report completed | SecOps |

---

## Root Cause Analysis

### Primary Cause

**SQL Injection Vulnerability in Login Form**

Vulnerability Details:
- Location: `/api/auth/login` endpoint
- Parameter: `email` field
- Type: UNION-based SQL injection
- Exploitability: CRITICAL
- Discoverer: Attacker (during dwell time)

Code Analysis:
```python
# VULNERABLE CODE (Before Fix)
def login(email, password):
    query = f"SELECT * FROM users WHERE email = '{email}' AND password_hash = MD5('{password}')"
    result = db.execute(query)  # ❌ Vulnerable: String concatenation
    return result

# FIXED CODE (After Fix)
def login(email, password):
    query = "SELECT * FROM users WHERE email = %s AND password_hash = MD5(%s)"
    result = db.execute(query, [email, password])  # ✅ Safe: Parameterized query
    return result
```

Why It Was Vulnerable:
1. User input concatenated directly into SQL query (worst practice)
2. No input validation or sanitization
3. No parameterized query (prepared statements)
4. No WAF to detect injection patterns
5. Limited database permissions (user could access all tables)

### Secondary Cause

**Compromised Employee Credentials**

How Credentials Were Obtained:
- Phishing email (spoofing HR department)
- Email linked to attacker-controlled domain
- Employee entered credentials thinking it was legitimate password reset
- No MFA to prevent unauthorized use of stolen credentials

Why Employee Was Vulnerable:
1. No MFA on employee accounts (currently optional)
2. No email gateway filtering to block spoofed emails
3. Limited phishing awareness training
4. No recent security awareness campaign

### Tertiary Contributing Factors

**Detection Delay (12 hours)**
- Root Cause: Alert threshold too high
- Impact: Attacker had 12 hours to exfiltrate data
- Lesson: Need lower threshold or real-time monitoring

**Database Permissions Too Permissive**
- Root Cause: Application user could SELECT from all tables
- Impact: Attacker could access customer data
- Lesson: Implement least-privilege principle

**No Data Exfiltration Monitoring**
- Root Cause: No DLP system in place
- Impact: 4.2 GB downloaded without alert
- Lesson: Deploy DLP to detect mass data transfers

---

## 5 Whys Analysis

```
Q1: Why was customer data breached?
A:  Attacker exploited SQL injection vulnerability in login form

Q2: Why did SQL injection vulnerability exist?
A:  Developer used string concatenation instead of parameterized queries

Q3: Why wasn't this caught in code review?
A:  Code review process didn't have security focus, no security training

Q4: Why no security training?
A:  No formal secure coding training program existed

Q5: Why no training program?
A:  Security was treated as after-thought, not embedded in development

ROOT CAUSE: Lack of security culture in development organization
REMEDY: Implement secure coding training, code review checklist, static analysis
```

---

## What Went Well ✅

### Detection
- ✅ SIEM alert fired successfully (14:30 UTC)
- ✅ Alert was clear and actionable
- ✅ SOC analyst correctly identified as real incident
- ✅ Avoided false positive confusion

### Response
- ✅ Incident declared within 1 minute of alert review
- ✅ Team assembled quickly (all on-call responders responded within 5 minutes)
- ✅ War room established and communication flowing
- ✅ Escalation to executive within 30 minutes

### Technical Actions
- ✅ Attacker blocked within 45 minutes of detection
- ✅ Root cause identified within 30 minutes of assembly
- ✅ Patch developed and deployed within 2 hours
- ✅ Forensic evidence preserved (memory dumps, logs, disk images)
- ✅ No re-infection after attacker blocked

### Investigation
- ✅ Forensics firm engaged quickly
- ✅ Complete timeline reconstructed
- ✅ Dwell time accurately determined (12 hours)
- ✅ Exfiltration confirmed and quantified

### Communication
- ✅ Executive leadership notified within 1 hour
- ✅ Clear, consistent messaging to leadership
- ✅ Customer notifications sent within 24 hours
- ✅ Regulatory agencies notified within 72 hours (GDPR requirement)
- ✅ Transparent and honest communication with customers

### Prevention
- ✅ Vulnerability patched before it could be exploited again
- ✅ Attacker IP blocked organization-wide
- ✅ Employee credentials reset
- ✅ Law enforcement notified for potential prosecution

---

## What Went Poorly ❌

### Detection
- ❌ Detection delay: 12 hours from initial compromise
- ❌ Should have detected within 1-2 hours with better monitoring
- ❌ Alert threshold too high (only fired due to volume anomaly)
- ❌ No user behavior analytics (would have caught earlier)

### Prevention (Pre-Incident)
- ❌ SQL injection vulnerability existed for ~6 months (estimated)
- ❌ No code review security checklist
- ❌ No static code analysis in CI/CD pipeline
- ❌ No penetration testing had been conducted
- ❌ No secure coding training for developers

### Credentials
- ❌ No MFA on employee accounts (made credentials dangerous when stolen)
- ❌ Email gateway didn't catch phishing email (spoofing)
- ❌ No recent phishing awareness training
- ❌ No email gateway advanced threat protection

### Database Security
- ❌ Database user had excessive permissions (could access all tables)
- ❌ No database activity monitoring (only discovered via manual alert)
- ❌ Password hashing algorithm was MD5 (outdated), should be bcrypt/Argon2
- ❌ No encryption at rest for customer data

### Dwell Time
- ❌ 12 hours is too long for detection
- ❌ Indicates need for real-time monitoring
- ❌ No alerts on reconnaissance activity (schema queries)
- ❌ No alerts on mass data access

---

## Impact Assessment

### Financial Impact

| Category | Amount | Notes |
|----------|--------|-------|
| Forensics firm | $50,000 | Mandiant investigation |
| Credit monitoring | $20,000 | 24 months for 15K users |
| Legal/regulatory | $30,000 | Counsel, compliance |
| Notifications | $10,000 | Email, systems |
| PR/communications | $25,000 | PR firm, messaging |
| **Immediate Total** | **$135,000** | |
| | | |
| Lost revenue (est.) | $50-200K | Customer churn estimate |
| Future security invest. | $500K+ | Prevention improvements |
| **Total 12-month** | **$685K+** | |
| | | |
| Insurance coverage | ~$100K | 70-80% of immediate costs |
| **Out of pocket** | ~$585K+** | After insurance |

### Customer Impact

- 15,000 customers notified (7.5% of user base)
- Average customer impact: Low-Medium (names/emails, not financial data)
- Credit monitoring offered: 24 months free
- Expected churn: 2-5% (15,000 × 0.025 = 375 customers)
- Revenue impact: ~$50-200K (depending on churn rate)

### Reputational Impact

- Negative news coverage (expected for large breaches)
- Social media sentiment: Mixed (many appreciative of transparent response)
- Customer trust: Short-term loss, recovery expected in 6-12 months
- Investor confidence: Slightly impacted (depends on market)
- Employee morale: Stable (company is fixing the problem)

### Regulatory Impact

- GDPR: Notification complete, no fines anticipated (compliant process)
- CCPA: Notification complete, no fines anticipated
- Potential lawsuits: Expected (class action likely within 6-12 months)
- Settlement estimate: $5M-20M (typical for this size breach)
- Timeline: 12-24 months to resolution

---

## Action Items

### Immediate Actions (0-30 days)

| # | Action | Owner | Deadline | Priority | Status |
|---|--------|-------|----------|----------|--------|
| 1 | Deploy MFA to all admin accounts | IT Security | Nov 23 | CRITICAL | OPEN |
| 2 | Enhanced email security (ATP) | Email Team | Nov 23 | CRITICAL | OPEN |
| 3 | Implement parameterized queries in all code | Dev Lead | Dec 7 | CRITICAL | OPEN |
| 4 | Database password reset (all accounts) | DBA | Nov 20 | CRITICAL | COMPLETED |
| 5 | Vulnerability scan of all public-facing code | Security | Nov 30 | HIGH | OPEN |
| 6 | Review all database permissions (least privilege) | DBA | Dec 7 | HIGH | OPEN |

### Short-term Actions (1-3 months)

| # | Action | Owner | Deadline | Priority | Status |
|---|--------|-------|----------|----------|--------|
| 7 | Deploy data loss prevention (DLP) system | Security | Jan 31 | CRITICAL | OPEN |
| 8 | Implement database activity monitoring (DAM) | DBA | Dec 31 | CRITICAL | OPEN |
| 9 | Secure coding training for all developers | Security | Dec 31 | HIGH | OPEN |
| 10 | Phishing awareness training for all staff | HR/Security | Dec 15 | HIGH | OPEN |
| 11 | Implement code scanning (SAST) in CI/CD | DevOps | Jan 31 | HIGH | OPEN |
| 12 | Network segmentation review | Network | Jan 31 | MEDIUM | OPEN |
| 13 | Third-party security audit | Security | Feb 28 | HIGH | OPEN |

### Long-term Actions (3-12 months)

| # | Action | Owner | Deadline | Priority | Status |
|---|--------|-------|----------|----------|--------|
| 14 | Pursue ISO 27001 certification | Security | June 30 | MEDIUM | OPEN |
| 15 | Implement zero-trust architecture | Architecture | Sept 30 | MEDIUM | OPEN |
| 16 | Quarterly incident response drills | Security | Mar 31 (start) | MEDIUM | OPEN |
| 17 | Annual penetration testing | Security | June 30 | MEDIUM | OPEN |
| 18 | Privileged access management (PAM) system | IT | Dec 31 | LOW | OPEN |

### Metrics & Success Criteria

Each action item will be tracked with:
- **Owner**: Who is responsible
- **Deadline**: When it must be complete
- **Success Criteria**: How we measure success
- **Verification**: How we confirm it's done
- **Review Date**: When leadership reviews progress

Example:
```
Action #1: Deploy MFA to all admin accounts
Owner: IT Security Lead
Deadline: November 23, 2025
Success Criteria:
  - 100% of admin accounts have MFA enabled
  - MFA enforcement tested and working
  - All admin users trained on MFA usage
Verification:
  - System scan shows 100% MFA enrollment
  - Test login with MFA disabled → BLOCKED
  - User feedback survey shows >90% successful logins
Review Date: November 30, 2025
```

---

## Recommendations

### High Priority (Implement ASAP)

1. **Mandatory MFA for All Admin Accounts**
   - **Impact**: Prevents unauthorized access even with stolen credentials
   - **Cost**: ~$10K for software
   - **Timeline**: 2 weeks
   - **Recommendation**: APPROVE

2. **Deploy Data Loss Prevention (DLP)**
   - **Impact**: Detects and blocks mass data exfiltration
   - **Cost**: ~$50K/year
   - **Timeline**: 8 weeks
   - **Recommendation**: APPROVE

3. **Secure Coding Training Program**
   - **Impact**: Prevents SQL injection and similar vulnerabilities
   - **Cost**: ~$20K/year
   - **Timeline**: 4 weeks to launch
   - **Recommendation**: APPROVE

4. **Code Scanning (SAST) in CI/CD**
   - **Impact**: Catches vulnerabilities before deployment
   - **Cost**: ~$15K/year
   - **Timeline**: 6 weeks
   - **Recommendation**: APPROVE

### Medium Priority (Next 90 days)

5. **Database Activity Monitoring (DAM)**
   - **Impact**: Real-time alerts on suspicious database access
   - **Cost**: ~$30K/year
   - **Timeline**: 12 weeks
   - **Recommendation**: APPROVE

6. **Network Segmentation**
   - **Impact**: Limits lateral movement if one system compromised
   - **Cost**: ~$75K (one-time)
   - **Timeline**: 12 weeks
   - **Recommendation**: APPROVE

7. **Annual Third-Party Security Audit**
   - **Impact**: Identifies vulnerabilities before attackers
   - **Cost**: ~$50K/year
   - **Timeline**: 8 weeks (first audit)
   - **Recommendation**: APPROVE

### Lower Priority (6-12 months)

8. **ISO 27001 Certification**
   - **Impact**: Demonstrates commitment to information security
   - **Cost**: ~$100K
   - **Timeline**: 6-9 months
   - **Recommendation**: PURSUE

9. **Zero-Trust Architecture**
   - **Impact**: Assumes all network traffic is untrusted
   - **Cost**: ~$200K+ (major architectural change)
   - **Timeline**: 6-12 months
   - **Recommendation**: PLAN FOR

---

## Lessons Learned Summary

### Key Takeaways

1. **Security must be embedded in development, not an afterthought**
   - SQL injection was a preventable vulnerability with proper training
   - Code review is not enough without security focus
   - Automated scanning (SAST) in CI/CD is essential

2. **Detection time is critical**
   - 12-hour detection window is too long
   - Need real-time monitoring (not alert thresholds)
   - Database activity monitoring would have caught this in minutes

3. **MFA would have prevented the initial compromise**
   - Phishing email + stolen credentials only works without MFA
   - MFA is now mandatory best practice
   - Cost is minimal compared to breach cost

4. **Data loss prevention is essential**
   - 4.2 GB transfer should have triggered alerts
   - DLP system would have blocked or detected exfiltration
   - Essential for preventing data breaches

5. **Incident response plan works**
   - Team assembled and responded quickly
   - Communication was clear and effective
   - Escalation procedures worked
   - Training and drills paid off

### Areas for Improvement

1. **Pre-incident security posture**
   - Need secure coding training
   - Need code review security checklist
   - Need automated scanning in CI/CD
   - Need penetration testing

2. **Detection capabilities**
   - Need real-time database monitoring
   - Need user behavior analytics
   - Need DLP system
   - Need improved alert rules

3. **Prevention measures**
   - Need MFA enforcement (not optional)
   - Need advanced email security
   - Need network segmentation
   - Need least-privilege database access

4. **Post-incident learning**
   - Need regular security reviews
   - Need quarterly incident response drills
   - Need annual penetration testing
   - Need continuous improvement process

---

## Conclusion

The November 16, 2025 data breach was a significant security incident that affected
15,000 customers and required urgent response. While it represents a failure of our
security controls, our incident response was professional and effective.

**Key Results**:
- ✅ Incident detected and contained within 1 hour of detection
- ✅ Root cause identified within 2 hours
- ✅ Vulnerability patched within 2 hours
- ✅ Customers notified within 24 hours
- ✅ Regulatory compliance maintained
- ✅ No secondary breaches or re-infections
- ✅ Law enforcement engaged

**Going Forward**:
This incident is an opportunity to significantly strengthen our security posture.
We are implementing comprehensive improvements across detection, prevention, and
response. Our goal is to reduce the likelihood and impact of future incidents.

---

## Appendices

### Appendix A: Detailed Timeline (Hour-by-Hour)

[Detailed timeline from section above]

### Appendix B: Technical Details

- SQL injection vulnerability (CWE-89)
- Attacker tools and techniques used
- Network traffic analysis
- Database query analysis
- Forensic findings

### Appendix C: Regulatory Notifications Sent

- GDPR Article 33 notification (DPA of [Country])
- CCPA notification (CA Attorney General)
- State breach notification laws (all affected states)
- Customer notifications (all affected individuals)

### Appendix D: Customer Impact Analysis

- List of affected customers (anonymized)
- Data exposure severity assessment
- Fraud risk assessment by individual
- Credit monitoring enrollment status
- Customer support inquiries and resolution

---

**Report Approved By**:

_________________________                    ___________________
Chief Information Security Officer          Date

_________________________                    ___________________
Chief Executive Officer                     Date

_________________________                    ___________________
General Counsel                             Date

---

**Document Status**: ✅ FINAL
**Classification**: CONFIDENTIAL - INTERNAL ONLY
**Retention**: 7 years (for regulatory compliance)
**Last Updated**: November 19, 2025
