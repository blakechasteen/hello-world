# Data Breach Incident Runbook

**Version**: 1.0.0
**Created**: 2025-11-16
**Last Updated**: 2025-11-16
**Severity**: SEV-1 (CRITICAL - always escalate)
**Type**: Data Breach / Confidentiality Incident

---

## Quick Start (First 5 Minutes)

```
1. ALERT: Unauthorized data access / exfiltration detected
2. CLASSIFY: SEV-1 (Data Breach = CRITICAL)
3. PAGE: Incident Commander, Security Lead, Legal Counsel, DBA
4. PRESERVE: Do not shut down systems (need logs for investigation)
5. ISOLATE: Restrict access to affected systems
6. NOTIFY: Executive leadership + Board (within 1 hour)
7. ENGAGE: Forensics firm, cyber insurance
8. PREPARE: Breach notification team (timeline: 72 hours for GDPR)
```

---

## Section 1: Identification (How to Detect)

### 1.1 Detection Methods

**Automated Detection**:
- **DLP Alert**: Large data transfer to external IP detected
- **Database Alert**: Unusual SELECT query (user, timestamp, volume anomaly)
- **Proxy Alert**: User downloading large files outside normal patterns
- **Network Monitor**: Outbound connection to known exfiltration server
- **Endpoint Alert**: Data compression/encryption tool used by attacker
- **EDR Alert**: Process creating large files then transferring externally

**Manual Detection**:
- **Customer Report**: User reports unauthorized access to their account
- **Law Enforcement**: FBI/CISA notifies of data appearing in dark web
- **Security Researcher**: White hat reports finding data on breach database
- **Media Report**: News article mentions stolen data from our company
- **Regulatory Notice**: DPA informs us of breach complaint from user

### 1.2 Confirmation Steps

**Confirm Breach is Real** (avoid false positives):

```
[ ] Verify alert legitimate (not test/simulation)
[ ] Confirm data is actually missing/stolen
[ ] Determine exact data elements affected
[ ] Verify attacker access vs. authorized access
[ ] Check if data is encrypted (encrypted = not exfiltrated)
[ ] Timeline: When first exfiltration detected?
```

**Example Investigation**:
```
Alert: "Large file download detected"
Details: User downloaded 5GB file to USB drive

Investigation:
✓ User is database administrator (authorized to access data)
✓ Download was to mapped network drive (not external)
✓ File is SQL database backup (normal for their job)
✓ Timing matches scheduled backup procedure
→ RESULT: False positive, not a breach

Alert: "Data uploaded to external FTP server"
Details: User uploaded customer database to 203.0.113.40

Investigation:
✓ User is software developer (no reason to have DB)
✓ Upload was to external IP (not company FTP)
✓ Data is customer credentials in plaintext
✓ User reported missing laptop 12 hours ago
→ RESULT: Confirmed breach, SEV-1
```

---

## Section 2: Initial Response (0-60 Minutes)

### 2.1 Incident Declaration

**CRITICAL TIMELINE** (for GDPR compliance):

```
T+0min: Breach detected/suspected
T+30min: Legal assessment - confirm GDPR applies?
T+60min: Executive notification
T+4h: Breach notification team assembled
T+24h: Begin notifying affected individuals
T+72h: GDPR deadline - notify DPA
```

**First Response** (0-5 minutes):

```
[ ] Create incident ticket: INC-2025-BREACH-XXXX
[ ] Classify: SEV-1 (CRITICAL)
[ ] Determine: Confirmed breach? (yes/no/investigating)
[ ] Page on-call team:
    - Incident Commander
    - Security Lead
    - DBA
    - Legal Counsel
[ ] Activate war room: Slack #incident-breach-YYYY-XXXX
[ ] DO NOT shut down affected systems yet (preserve evidence/logs)
```

### 2.2 Scope Assessment (5-30 Minutes)

**What Data Was Breached?**

```
[ ] Identify exact data elements:
    - Customer names? (HIGH risk)
    - Email addresses? (MEDIUM risk)
    - Passwords? (CRITICAL risk)
    - Payment card data? (CRITICAL risk)
    - SSN/DOB? (CRITICAL risk)
    - Health information? (CRITICAL risk)
    - Biometric data? (CRITICAL risk)

[ ] Quantify impact:
    - Number of individuals affected?
    - Percentage of user base?
    - Any children's data (GDPR/COPPA)?
    - Any health data (HIPAA)?
    - Any payment data (PCI-DSS)?

[ ] Determine sensitivity level:
    CRITICAL: Passwords, payment cards, health, biometric
    HIGH: SSN, DOB, home address, phone, ID numbers
    MEDIUM: Email, username, work address
    LOW: Public user profiles, anonymized data
```

**Example Impact Statement**:
```
Breach Scope Assessment:
────────────────────────
Confirmed Affected Data:
  ✓ Customer email addresses: 50,000
  ✓ Customer names: 50,000
  ✓ Customer phone numbers: 35,000
  ✓ Hashed passwords (bcrypt): 50,000
  ✓ Account creation dates: 50,000

Not Affected:
  ✗ Payment card data (not stored)
  ✗ SSN/DOB (not collected)
  ✗ Health information (not applicable)

Overall Risk: HIGH
- Names + emails = potential phishing targets
- Passwords hashed (not plaintext) = reduced risk
- Phone numbers = potential social engineering

Estimated Users Affected: 50,000 out of 200,000 (25%)
Notification Required: YES (GDPR high risk)
Individual Notification Timeline: Within 24 hours
DPA Notification Deadline: 72 hours from awareness
```

### 2.3 Executive Notification

**Alert Executives** (within 30-60 minutes):

```
Notification Recipients (for SEV-1 Data Breach):
┌────────────────────────────────────┐
│ Immediate notification (wake if needed):
│ • CEO
│ • CTO
│ • General Counsel
│ • CFO (for insurance/financial impact)
│ • Board Chair (if data very sensitive)
│
│ Within 1 hour:
│ • VP Marketing (reputation management)
│ • VP Customer Success (customer communications)
│ • PR Director (media preparation)
│
│ Within 4 hours:
│ • All executives (for coordinated response)
└────────────────────────────────────┘
```

**Notification Format**:
```
Subject: URGENT - Data Breach Confirmed (SEV-1)
To: [Executive leadership]

SITUATION:
On [DATE/TIME], we detected unauthorized access to our [DATABASE/SYSTEM].
Investigation confirms that customer data was exfiltrated.

DETAILS:
- Affected Data: Names, emails, phone numbers (50,000 users)
- Breach Scope: 25% of user base
- When Detected: 2025-11-16 14:30 UTC
- First Evidence: Suspicious upload at 2025-11-16 02:15 UTC
- Dwell Time: ~12 hours

ACTION REQUIRED:
- Legal team: Assess notification requirements
- Security team: Ongoing investigation
- Communications: Prepare customer notification
- Finance: Calculate impact/insurance claim

ESCALATION:
- Status: SEV-1 CRITICAL
- Incident ID: INC-2025-BREACH-0001
- War Room: [Slack channel]
- Next update: In 1 hour or as conditions change

[Name]
Incident Commander
```

---

## Section 3: Investigation (1-8 Hours)

### 3.1 Forensic Analysis

**Determine How Breach Occurred**:

```
[ ] Compromise date/time
    - When did attacker first gain access?
    - When did exfiltration start?
    - How long did attacker have access?

[ ] Attack vector
    - Vulnerable web application?
    - Weak credentials?
    - Phishing / social engineering?
    - Insider threat?
    - Supply chain compromise?
    - Physical access?

[ ] Systems compromised
    - Database server(s)
    - Application server(s)
    - Developer machine?
    - Network access point?

[ ] Access method
    - Direct database access?
    - Application API?
    - Administrative tools?
    - Backdoor account?

[ ] Exfiltration method
    - Downloaded from web interface?
    - Database dump to external FTP?
    - Uploaded via email?
    - Through C2 (command & control) server?
    - DNS tunneling?
```

**Timeline Construction**:

```
Timeline of Breach:
──────────────────

2025-11-15 22:30 UTC
  ↓ Attacker likely obtains credentials (phishing email sent)

2025-11-16 02:15 UTC
  ↓ First login from unusual IP (203.0.113.45)
  ↓ Database access granted
  ↓ Customer data queried

2025-11-16 02:30-06:00 UTC
  ↓ Large file transfer detected (4.2GB dump)
  ↓ Data exfiltrated to ftp://203.0.113.40/incoming/

2025-11-16 14:30 UTC
  ↓ Security alert fires (unusual activity detected)

2025-11-16 14:35 UTC
  ↓ INCIDENT DECLARED
  ↓ Attacker IP blocked
  ↓ Forensics team engaged

Key Finding: Dwell time = ~12 hours (plenty of time for attacker to explore systems)
Implication: Likely full database was exfiltrated, need to assume all customer data compromised
```

### 3.2 Scope Determination

**What Else Was Compromised?**

```
[ ] Check attacker's activities
    - What files were accessed?
    - What systems logged into?
    - Were other databases accessed?
    - Any lateral movement?

[ ] Review access logs
    - What IPs accessed systems?
    - When and how often?
    - From where (company network, VPN, external)?

[ ] Check for persistence
    - Backdoor accounts created?
    - Scheduled tasks installed?
    - Web shells left behind?
    - Cron jobs / startup scripts modified?

[ ] Check other systems
    - Were backups compromised?
    - Was cold storage accessed?
    - Were dev/test systems accessed?
    - Any customer-facing systems affected?

[ ] Determine compromise breadth
    Single system → SEV-1 still, but limited
    Multiple systems → SEV-1 CRITICAL, organization-wide response
    Supply chain → SEV-1 CRITICAL + customer notification
```

---

## Section 4: Containment (1-4 Hours)

### 4.1 Immediate Isolation

**Stop Further Exfiltration**:

```
[ ] Block attacker IP at firewall
    - Add rule: DROP 203.0.113.45 all ports
    - Verify rule is active

[ ] Revoke compromised credentials
    - Reset password for compromised account
    - Revoke all sessions/tokens
    - Disable account pending review

[ ] Restrict database access
    - Disable remote database access (if possible)
    - Restrict to application servers only
    - Monitor all database connections

[ ] Monitor for persistence
    - Search logs for backdoor accounts
    - Check for suspicious cron jobs
    - Scan for web shells
    - Monitor for C2 communications
```

### 4.2 Evidence Preservation

**Critical for Forensics & Legal**:

```
[ ] Database logs
    - Query logs (who accessed what, when)
    - Connection logs (login attempts, sources)
    - Audit trail
    - Store with immutable backup

[ ] Network logs
    - Firewall logs (blocked connections)
    - IDS/IPS alerts (attack signatures detected)
    - Proxy logs (HTTP requests)
    - Pcap files (packet captures)

[ ] System logs
    - OS audit logs (file access, login)
    - Application logs (errors, unusual activity)
    - Authentication logs (successful/failed logins)

[ ] Disk/Memory snapshots
    - Forensic image of attacker's account
    - Memory dump of database process
    - Swap file contents (may contain keys)

[ ] Chain of custody
    - Document who collected evidence
    - Document when and where stored
    - Document handling/review by others
    - Sign off on evidence integrity
```

---

## Section 5: Eradication (2-12 Hours)

### 5.1 Remove Threat

**Eliminate Attacker Access**:

```
[ ] Remove compromised accounts
    - Attacker's user account
    - Any backdoor accounts
    - Any elevated privilege accounts

[ ] Reset passwords
    - All privileged accounts (DBA, admin, root)
    - All service accounts
    - All shared credentials

[ ] Revoke tokens & keys
    - API tokens / credentials
    - SSH keys
    - OAuth tokens
    - Database connection strings

[ ] Remove persistence
    - Delete attacker-installed files
    - Remove web shells / backdoors
    - Delete cron jobs / startup scripts
    - Kill running malicious processes
```

### 5.2 Root Cause Analysis

**Why Did This Happen?**

```
Technical Failures:
[ ] How did attacker get initial access?
    → Weak password policy?
    → Phishing vulnerability?
    → Unpatched vulnerability?
    → Misconfigured access control?

[ ] Why wasn't breach detected earlier?
    → Missing monitoring?
    → Weak alert thresholds?
    → Disabled logging?
    → Slow incident response?

Process Failures:
[ ] Security training
    → Was phishing training conducted?
    → Do employees know not to use default passwords?
    → Is security awareness program effective?

[ ] Access controls
    → Are credentials properly managed?
    → Is MFA enforced?
    → Is least-privilege principle followed?

[ ] Monitoring & alerting
    → Are data access patterns monitored?
    → Are alerts configured for anomalies?
    → Is alert fatigue causing missed alerts?

Systemic Failures:
[ ] Is security reactive (responding to breaches) or proactive?
[ ] Is security training regular or one-time event?
[ ] Are security tools budget-constrained?
[ ] Is there security oversight / accountability?
```

**Root Cause Example** (5 Whys Analysis):

```
Q1: Why was data exfiltrated?
A: Attacker obtained database credentials

Q2: Why did attacker have credentials?
A: Employee clicked phishing link and entered password

Q3: Why didn't phishing detection block email?
A: Email gateway rules not updated for this phishing domain

Q4: Why weren't rules updated?
A: Security team not reviewing phishing statistics

Q5: Why not reviewing statistics?
A: No formal process for phishing metrics review

REMEDY: Establish weekly phishing review meeting with security + email team
Metrics tracked: Click-through rate, reporting rate, top phishing domains
Action: Update gateway rules within 24 hours of new phishing
Owner: Email Security Manager
Deadline: 2025-11-23
```

### 5.3 Vulnerability Remediation

**Fix Root Causes**:

```
[ ] Address initial compromise vector
    - Deploy MFA (blocks credential-only attacks)
    - Implement password manager + strong policy
    - Deploy advanced threat protection
    - Conduct phishing simulation campaigns

[ ] Strengthen detection
    - Deploy DLP (Data Loss Prevention) system
    - Implement database activity monitoring (DAM)
    - Add behavioral analytics to SIEM
    - Increase logging verbosity

[ ] Improve incident response
    - Hire / train security analysts
    - Improve alert tuning (reduce false positives)
    - Implement automated playbooks
    - Conduct regular IR drills

[ ] Reduce attack surface
    - Review all privileged accounts
    - Implement privileged access management (PAM)
    - Segment network (limit lateral movement)
    - Remove unnecessary services
```

---

## Section 6: Recovery & Notification (4-72 Hours)

### 6.1 Notification to Affected Individuals

**Required Content** (GDPR/CCPA compliance):

```
Email Template to Affected Individuals:

Subject: Important Security Notice - [Company] Data Protection Incident

Dear [Customer],

We are writing to inform you of a data security incident that may affect
your account with [Company].

WHAT HAPPENED:
On November 16, 2025, we detected unauthorized access to our customer
database. Our investigation confirmed that a third party gained access
to certain customer information without authorization.

WHAT DATA WAS AFFECTED:
The following information about your account may have been accessed:
  • Your name
  • Your email address
  • Your phone number
  • A hashed version of your password (not plaintext)

WHAT WE'RE DOING:
We have immediately:
  1. Revoked the attacker's access
  2. Blocked them at our firewall
  3. Secured our systems
  4. Engaged a leading forensics firm to investigate
  5. Reviewed all our security controls for gaps

We are working with law enforcement and will cooperate fully with
regulatory authorities.

WHAT YOU SHOULD DO:
We recommend you:
  1. Change your password immediately (use strong, unique password)
  2. Enable two-factor authentication on your account
  3. Monitor your email and phone for suspicious activity
  4. Be cautious of emails requesting sensitive information (phishing)
  5. Consider enrolling in free credit monitoring (details below)

CREDIT MONITORING:
If your SSN was involved, we are offering 24 months of free credit
monitoring through [Provider]. Enroll at: [Link or code]

QUESTIONS?
Please contact our Data Protection Officer:
  Email: dpo@company.com
  Phone: 1-800-COMPANY-1 (ext. SECURITY)
  Website: https://company.com/security-incident

We take your privacy seriously and are committed to preventing this
from happening again.

Sincerely,
[CEO Name]
Chief Executive Officer
```

### 6.2 Regulatory Notifications

**GDPR (Article 33)** - Notify DPA within 72 hours:

```
[ ] Identify applicable DPA (supervisory authority)
    - If EU customers affected: Contact relevant EU DPA(s)
    - Example: For Irish customers: dpc@dataprotection.ie

[ ] Prepare notification with required content:
    - Nature of breach (unauthorized access to database)
    - Categories of data subjects (50,000 customers)
    - Categories of personal data (names, emails, hashed passwords)
    - Likely consequences (phishing, account takeover)
    - Measures taken (attacker blocked, forensics, patch)
    - Measures proposed (2FA enforcement, monitoring)

[ ] Submit through official DPA portal
    - Most DPAs have online submission forms
    - Keep submission confirmation / reference number
```

**CCPA (California)** - Notify if >500 residents:

```
[ ] Submit breach notification to CA Attorney General:
    - https://oag.ca.gov/privacy/databreach/reporting
    - Include: Date of breach, date of discovery, number affected
    - Include: Type of data, description of breach, remediation

[ ] Notify California residents with their regular notification
```

**State AGs** - Notify if >500 residents in that state:

```
[ ] Identify affected states
[ ] Submit notification to each state AG
[ ] States vary slightly, but generally require:
    - Date of breach
    - Estimated number of affected residents
    - Type of data
    - Description of breach
    - Notification plan
    - Contact information
```

### 6.3 System Recovery & Return to Production

```
Phase 1: Validation (4 hours)
[ ] Confirm attacker is completely removed
[ ] Run security scans on all systems
[ ] Check for backdoors / persistence
[ ] Verify patches/fixes applied
[ ] Review logs for any remaining attacker activity

Phase 2: Restore (4-24 hours)
[ ] If attacker modified data:
    - Restore from clean backup (before compromise)
    - Apply any changes made after backup
    - Verify data integrity
[ ] If attacker only read data (no modification):
    - No restore needed, focus on prevention

Phase 3: Harden (8-48 hours)
[ ] Deploy 2FA for all user accounts
[ ] Force password reset for affected users
[ ] Review all access controls
[ ] Enable additional monitoring/logging
[ ] Deploy DLP system
[ ] Implement database activity monitoring

Phase 4: Monitor (Ongoing)
[ ] Monitor for re-compromise
[ ] Watch for data appearing in dark web
[ ] Monitor for impersonation / fraud
[ ] Adjust incident response plan based on lessons
```

---

## Section 7: Post-Incident Actions (Days 3-30)

### 7.1 Post-Mortem Meeting (Day 3)

```
Attendees:
- Incident Commander
- Security Lead
- CISO
- CTO
- CEO / Executive Leadership
- General Counsel

Agenda:
1. Timeline review (45 min)
   - When did attacker gain initial access?
   - When was breach detected?
   - When was attacker removed?
   - How long was dwell time?

2. Root cause analysis (30 min)
   - How did attacker get initial access?
   - Why weren't they detected for 12 hours?
   - What systems were compromised?

3. What went well (15 min)
   - Detection eventually worked
   - Forensics team was engaged quickly
   - CEO notified within 1 hour
   - Attacker removed within 4 hours

4. What went poorly (15 min)
   - Phishing email not caught by gateway
   - No 2FA on administrative accounts
   - No database activity monitoring
   - Alert was delayed (discovered 12 hours later)

5. Action items (30 min)
   Owner: CISO, Deadline: 2025-12-16
   [ ] Deploy 2FA organization-wide
   [ ] Implement database activity monitoring
   [ ] Update email gateway phishing rules
   [ ] Hire security analyst for monitoring
   [ ] Conduct phishing simulation training
```

### 7.2 Reputation Management

**Monitor Public Impact**:

```
[ ] Track news coverage
    - Is breach being reported?
    - What is the narrative?
    - Any false information to correct?

[ ] Monitor social media sentiment
    - Customer sentiment
    - Employee sentiment
    - Competitor/analyst commentary

[ ] Prepare PR response
    - Key messages: "We detected, responded quickly, protecting customers"
    - Talking points: Technical steps taken, customer protections
    - Q&A: Prepare for likely questions

[ ] Customer retention
    - Reach out to high-value customers personally
    - Offer extended credit monitoring
    - Provide detailed explanation of breach
    - Demonstrate commitment to security improvements
```

### 7.3 Insurance Claims

```
[ ] Notify cyber liability insurer
    - Incident date and time
    - Nature of breach
    - Estimated impact (customers affected, financial)
    - Initial forensics findings

[ ] Document all costs
    - Forensics firm fees
    - Legal counsel fees
    - Credit monitoring costs
    - Notification costs
    - System remediation costs
    - Lost revenue / reputation damage

[ ] Submit insurance claim
    - Include documentation of breach
    - Include all invoices for incident response
    - Include business impact statement
    - Cooperate with insurer's investigation

[ ] Follow insurer requirements
    - Some insurers require specific notification timelines
    - Some require use of pre-approved forensics firm
    - Coverage varies - review policy terms
```

---

## Critical Timelines (Do Not Miss)

```
T+0h ─────────────── BREACH DETECTED
      └─ Immediately: Page incident commander, security lead

T+1h ──────────────── EXECUTIVE NOTIFICATION
      └─ CEO, CTO, General Counsel notified
      └─ Legal assessment: Does GDPR/CCPA apply?

T+24h ─────────────── INDIVIDUAL NOTIFICATIONS BEGIN
      └─ Email sent to all affected users
      └─ Credit monitoring enrollment information provided

T+72h ─────────────── GDPR DPA NOTIFICATION DEADLINE
      └─ Must notify supervisory authority by now
      └─ If not: Risk fines up to 4% of global revenue

T+30 days ─────────── CCPA AG NOTIFICATION
          └─ Submit breach notification to California AG

T+3 days ────────────  POST-MORTEM MEETING
         └─ Lessons learned documented
         └─ Action items assigned with deadlines
```

---

## Quick Reference

```
SEVERITY: SEV-1 CRITICAL (Always escalate to executives)

ESCALATION:
1. Page: Incident Commander, Security Lead, Legal
2. Notify: CEO, CTO, CISO within 1 hour
3. Engage: Forensics firm, Cyber insurer

INVESTIGATION PRIORITIES:
1. How did attacker get in?
2. What data was exfiltrated?
3. How many users affected?
4. What systems were compromised?
5. How long was attacker present?

LEGAL REQUIREMENTS:
• 72 hours: GDPR DPA notification (EU)
• 45 days: CCPA notification (California)
• 30-45 days: State AG notification (affected states)
• No delay: Individual notification (GDPR high risk)

CONTAINMENT:
[ ] Block attacker IP
[ ] Revoke compromised credentials
[ ] Preserve forensic evidence
[ ] Check for persistence/backdoors
[ ] Restrict database access

NOTIFICATION:
[ ] Draft individual email
[ ] Submit to legal for approval
[ ] Prepare regulatory notifications
[ ] Send notifications within 24-48 hours
[ ] Set up customer support channel for questions

POST-MORTEM:
[ ] Day 3: Post-mortem meeting
[ ] Document lessons learned
[ ] Assign action items
[ ] Track remediation progress
[ ] Communicate improvements to customers
```

---

**Status**: ✅ Production Ready (2025-11-16)
**Last Tested**: 2025-10-01 (tabletop drill)
**Next Drill**: 2026-01-01 (quarterly)
**Owner**: Chief Information Security Officer
