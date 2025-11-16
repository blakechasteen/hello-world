# Insider Threat Incident Runbook

**Version**: 1.0.0
**Created**: 2025-11-16
**Severity**: SEV-1 to SEV-2 (depends on scope/intent)
**Type**: Insider Threat / Malicious Employee / Negligent Employee

---

## Quick Start (First 10 Minutes)

```
🚨 INSIDER THREAT SUSPECTED - SEV-1/2 ESCALATION

1. DO NOT ALERT EMPLOYEE YET (preserve evidence)
2. DO NOT CONFRONT EMPLOYEE directly (may destroy evidence/flee)
3. PAGE: Incident Commander, Security Lead, HR Lead, Legal Counsel
4. PRESERVE: Collect forensic evidence before termination
5. BLOCK: Prepare for immediate access revocation
6. COORDINATE: Work with HR and Legal (may involve police)
7. INVESTIGATE: Understand scope and intent
```

**Triggers for Insider Threat Investigation**:
```
Behavioral:
[ ] Employee reports missing/accessing unauthorized data
[ ] Employee has taken laptop/mobile device for extended period
[ ] Employee attempting to access data outside their role
[ ] Employee querying employee records, financial data, etc.
[ ] Employee accessing systems before/after normal hours

Technical Indicators:
[ ] Unusual data access patterns
[ ] Mass file downloading/copying
[ ] Accessing customer databases they shouldn't see
[ ] Creating backdoor accounts
[ ] Installing monitoring software
[ ] Accessing VPN from unusual locations

Administrative Indicators:
[ ] Employee announced resignation coming (flights booked?)
[ ] Employee recently passed over for promotion
[ ] Employee complaints/performance issues
[ ] Employee financial distress (known to company)
[ ] Employee has given notice and is in exit period

Concerning Combinations:
[ ] Termination scheduled + mass file access = HIGH RISK
[ ] Access elevated recently + unusual queries = SUSPICIOUS
[ ] Multiple systems accessed at once + multiple IPs = POSSIBLE COMPROMISE
```

---

## Section 1: Identification & Assessment

### Confirm Insider Threat

```
Step 1: Verify Alert is Real
[ ] Is it actually this employee accessing the data?
[ ] Could another employee be using their account?
[ ] Could this be legitimate work activity?
[ ] Is the timing normal for this employee?

Example Investigation:
Alert: "Employee accessed customer database"
├─ Check: Is this employee's normal job?
│  └─ IF YES: Might be legitimate
│  └─ IF NO: Suspicious
├─ Check: What time (business hours or middle of night)?
│  └─ IF 2am: Very suspicious
│  └─ IF 9am: More normal
├─ Check: From what IP?
│  └─ IF company network: More normal
│  └─ IF residential IP: Could be working from home (or breach)
├─ Check: What data accessed?
│  └─ IF customer list: Sensitive
│  └─ IF public data: Less concerning
└─ Decision: Malicious or innocent explanation?
```

### Assess Severity

```
SEV-1 (CRITICAL) - Malicious Intent Indicators:
[ ] Confirmed unauthorized access to sensitive data (customer PII, passwords, source code)
[ ] Evidence of data exfiltration (files copied to personal device, email, cloud)
[ ] Clear intent to steal IP or harm company (sabotage, deletion of data)
[ ] Employee has history of threats or concerning behavior
[ ] Employee notified they're being terminated and immediately access data
[ ] Multiple sensitive systems accessed simultaneously

SEV-2 (HIGH) - Potential Threat:
[ ] Unauthorized access but unclear intent (maybe curious?)
[ ] Data accessed but unclear if exfiltrated
[ ] Employee accessing data outside normal role (maybe escalating)
[ ] Performance issues + access violations (possible revenge)
[ ] Suspicious but isolated incident

SEV-3 (MEDIUM) - Likely Innocent:
[ ] Employee accessed data, explained legitimate reason
[ ] Isolated incident, no pattern of concern
[ ] Data accessed but not sensitive
[ ] Employee cooperates with investigation
```

### Determine Scope

```
What Data Was Accessed?
[ ] Customer personal information (names, emails, phone, address)
  └─ CRITICAL: May trigger GDPR/CCPA breach notification
[ ] Customer financial data (payment cards, bank info)
  └─ CRITICAL: PCI-DSS breach
[ ] Passwords/credentials
  └─ CRITICAL: Must rotate all credentials
[ ] Source code / proprietary algorithms
  └─ CRITICAL: Potential IP theft
[ ] Employee records (SSN, background check info)
  └─ HIGH: Privacy violation
[ ] Strategic plans / business plans
  └─ HIGH: Competitive disadvantage
[ ] Sales/customer information
  └─ MEDIUM: Competitive disadvantage but less critical

Quantify Impact:
[ ] How much data accessed? (KB? MB? GB?)
[ ] How many records accessed? (1? 100? 10,000?)
[ ] How many different systems accessed?
[ ] Over what time period? (minutes? hours? days? weeks?)
```

---

## Section 2: Initial Response (T+0 to T+30 Minutes)

### Assembly & Coordination

```
Incident Declaration:
[ ] Create incident ticket: INC-2025-INSIDER-XXXX
[ ] Severity level assigned (SEV-1, SEV-2, or SEV-3)
[ ] Activate war room: #incident-insider-threat-YYYY-XXXX

Notification (do NOT notify employee yet):
[ ] Incident Commander (coordination)
[ ] Security Lead (investigation)
[ ] HR Lead (employment law, termination procedure)
[ ] General Counsel (legal compliance, potential prosecution)
[ ] CISO (executive visibility)

For SEV-1: Also notify:
[ ] CEO (executive decision-making)
[ ] CFO (financial impact, insurance claim)
[ ] Law Enforcement (may need police/FBI involvement)
```

### Preserve Evidence

**CRITICAL: Collect evidence BEFORE termination**

```
Digital Evidence (within 30 minutes):
[ ] Capture user access logs
    - All logins (dates, times, IP addresses)
    - All file access (what files, when)
    - All database queries (when, what data)
    - All email activity

[ ] Email investigation
    - Export email inbox / sent items
    - Search for suspicious emails:
      - Emails to personal email account?
      - Emails to competitors?
      - Emails with file attachments (especially unusual ones)?
    - Check email forwarding rules (did they set auto-forward?)
    - Check shared account access (did they grant themselves access?)

[ ] USB/Mobile device tracking
    - Did they connect USB drives to their computer?
    - Did they sync data to cloud storage?
    - Did they connect mobile devices?
    - Download logs if mobile management available

[ ] File system analysis
    - Recent files accessed (last 24-48 hours)
    - Recently modified files
    - Files copied to external drives
    - Files compressed/archived (ZIP, TAR)

[ ] Endpoint data
    - Enable EDR forensics collection (if available)
    - Capture browser history
    - Check for file transfer tools (curl, FTP, SFTP)
    - Check installed software (passwords managers, cloud sync?)

[ ] Cloud storage investigation
    - Personal cloud accounts (Google Drive, OneDrive, Dropbox, iCloud)
    - Did employee sync company files to personal cloud?
    - Check access logs of shared drives
    - Check if files shared with personal email accounts

[ ] Network evidence
    - Firewall logs (what external IPs connected to company?)
    - Proxy logs (what URLs accessed?)
    - DLP logs (what files transferred out?)
```

### Physical Evidence

```
Before Termination Meeting:
[ ] Secure employee's laptop
    - Do NOT let them continue using it
    - Do NOT allow them to "wrap up work"
    - Collect laptop for forensics
    - (May need IT to help disable remote access first)

[ ] Collect mobile devices
    - Company-issued phone/tablet
    - Check if personal device has company data
    - (With care to privacy limits)

[ ] Check office/desk
    - Are there USB drives present?
    - Printed documents with sensitive data?
    - Post-it notes with passwords?
    - External hard drives?
    - Note any items that seem relevant

[ ] Secure badge/access
    - Collect physical badge before leaving
    - Prevents re-entry to facility
```

---

## Section 3: Investigation (1-4 Hours)

### Forensic Analysis

```
Timeline Construction:
└─ When did suspicious activity start?
   ├─ Days/weeks/months before?
   ├─ After promotion denial?
   ├─ After performance review?
   ├─ After resignation announcement?
   └─ Correlates with employment event?

Pattern Analysis:
├─ Is this first incident or pattern?
├─ Has employee accessed unauthorized data before?
├─ Are they accessing more data over time?
├─ Any failed access attempts (probing)?
└─ Is access limited to certain systems/data?

Intent Assessment:
├─ Evidence of deliberate theft?
│  └─ Files compressed (ZIP/TAR)?
│  └─ Files copied to external device?
│  └─ Files emailed to external address?
│  └─ Files uploaded to cloud/FTP?
├─ Evidence of sabotage?
│  └─ Deleting data?
│  └─ Modifying data?
│  └─ Installing malware/backdoors?
├─ Evidence of espionage?
│  └─ Accessing competitor data?
│  └─ Accessing strategic/financial data?
│  └─ Accessing source code?
│  └─ Copying to external systems?
└─ Or just careless/negligent?
   └─ Accessed by mistake?
   └─ No evidence of intent to harm?
   └─ Shared password with others?
```

### Determine Motivation

```
Financial Motivation:
[ ] Does employee have financial problems?
    - Known to company (HR records)?
    - Salary review shows dissatisfaction?
    - Offer from competitor (if competitor data accessed)?

Revenge Motivation:
[ ] Recent negative employment events?
    - Passed over for promotion?
    - Negative performance review?
    - Denied raise?
    - Disciplinary action?
    - Termination notice already given?

Espionage Motivation:
[ ] Contact with competitors?
    - Applying for job at competitor?
    - Competitor employee contacting them?
    - Travel to competitor offices?
    - Communication (email/LinkedIn/phone) with competitor?

Negligence Motivation:
[ ] No malicious intent, just careless?
    - Shared password with unauthorized person?
    - Left computer unlocked?
    - Gave access to family member?
    - Didn't understand data sensitivity?
```

---

## Section 4: Containment

### Immediate Access Revocation

**Coordinate with HR for simultaneous execution**

```
Before Termination Meeting:
[ ] Coordinate with IT to have access removal ready
[ ] Prepare termination meeting location (with HR and Security)
[ ] Ensure Physical Security present (if concerned about violence)
[ ] Have backup IT staff ready to execute access removal
[ ] Ensure employee doesn't have advance notice (preserve evidence)

Termination Meeting Execution:
├─ HR Lead: Conducts termination meeting with employee
├─ Security: Present (neutral, for safety)
├─ IT: Standing by to execute access revocation
└─ Timeline: Plan for immediate revocation after employee learns they're terminated

Simultaneous Actions (all at once):
[ ] Disable all IT access:
    - Disable user account in Active Directory
    - Revoke VPN access
    - Revoke API credentials
    - Revoke SSH keys
    - Disable email account (preserve for forensics, don't delete)
    - Revoke cloud storage access (Google Drive, Dropbox, etc)
    - Revoke database access

[ ] Revoke physical access:
    - Disable badge in access control system
    - Notify Physical Security of badge deactivation
    - Remove from facility access list

[ ] Secure data:
    - Change passwords for any shared credentials (if employee had them)
    - Rotate API keys
    - Reset OAuth tokens

[ ] Preserve evidence:
    - All digital evidence preserved (logs, emails, files)
    - All physical evidence collected (laptop, phone, badge)
    - Chain of custody documented
```

### Post-Termination Monitoring

```
Monitor for Retaliation/Persistence:
[ ] Check if they try to access systems (after termination):
    - Will show intent/attempt to continue damage
    - Alert should fire if they try to log in
    - Document all attempts

[ ] Monitor network for compromise:
    - Did they leave backdoor account?
    - Did they install malware/persistence?
    - Scan systems for suspicious accounts/processes

[ ] Monitor dark web for data sale:
    - Check breach marketplaces
    - Search for company/customer data
    - May take weeks to appear

[ ] Monitor for external communication:
    - Are they contacting other employees?
    - Offering company secrets for sale?
    - Threatening to publish data?
```

---

## Section 5: Legal & HR Actions

### HR Process

```
Termination Documentation:
[ ] Document all findings in writing
[ ] HR maintains termination record (separate from security incident)
[ ] Offer severance package (if company policy)
[ ] Explain any legal claims/liability
[ ] Obtain signed release (if negotiated)

Exit Interview:
[ ] May be appropriate to discuss data access findings
[ ] Depends on company counsel advice
[ ] Could affect severance negotiations

References:
[ ] Decide what can be said about termination to future employers
[ ] Typically limited to "employed from X to Y, eligible for rehire: NO"
```

### Legal Actions

```
Law Enforcement Involvement (for SEV-1 breaches):
[ ] Report to local police (criminal investigation)
[ ] May also report to FBI (if interstate/computer fraud)
[ ] Provide all evidence
[ ] Cooperate with investigation
[ ] May lead to criminal prosecution

Civil Litigation:
[ ] Employee may sue for wrongful termination
[ ] Company may counter-sue for damages
[ ] Consult employment counsel
[ ] Document everything defensively
[ ] Keep investigation findings confidential

Non-Compete / NDA Enforcement:
[ ] If employee violated NDA or non-compete:
    - Send cease & desist letter
    - Threaten legal action if they disclose/compete
    - May result in injunction
[ ] Consult counsel before sending

Trade Secret Protection:
[ ] If source code/IP was stolen:
    - File DMCA takedown notices (if posted online)
    - Notify marketplace platforms
    - May pursue criminal prosecution
    - May sue in civil court for damages
```

---

## Section 6: Remediation & Prevention

### Immediate Actions

```
Credentials Rotation:
[ ] Reset all credentials employee had access to:
    - Database passwords
    - API keys
    - Service account credentials
    - SSH keys
    - OAuth tokens
    - VPN shared credentials

Data Review:
[ ] If data exfiltrated, breach notification may be required:
    - GDPR: If customer/employee PII accessed
    - CCPA: If California customer data accessed
    - State laws: If state resident data accessed
    - Follow breach notification procedures

Notification (if data breach occurred):
[ ] Notify affected individuals
[ ] Notify regulators (DPA, state AG)
[ ] See BREACH_NOTIFICATION_PROCEDURES.md for details
```

### Prevent Future Incidents

```
Technical Controls:
[ ] Implement Privileged Access Management (PAM)
    - Limit who can access sensitive systems
    - Log and monitor privileged access
    - Require MFA for admin access

[ ] Deploy DLP (Data Loss Prevention)
    - Alert on mass file transfers
    - Block exfiltration to cloud/USB
    - Monitor email attachments

[ ] Enhanced Monitoring:
    - Database activity monitoring (DAM)
    - User behavior analytics (UBA)
    - File integrity monitoring
    - Endpoint monitoring (EDR)

[ ] Network Segmentation:
    - Limit employee access by role
    - Don't give everyone access to all data
    - Principle of least privilege

Process Controls:
[ ] Access Reviews:
    - Quarterly review of who has access to what
    - Remove access for employees who don't need it
    - Audit privileged accounts

[ ] Off-boarding Procedure:
    - Comprehensive checklist
    - All access removed immediately
    - All devices collected
    - All credentials changed
    - Exit interview

[ ] Whistleblower Program:
    - Anonymous way to report concerns
    - Protects reporters from retaliation
    - May catch insider threats early

Background Checks:
[ ] Conduct background checks on hire
    - Helps identify candidates with history
    - May reveal financial distress
    - May reveal criminal history
```

---

## Section 7: Post-Incident

### Post-Mortem (Day 3)

```
Questions:
[ ] When did suspicious activity begin?
[ ] When did we detect it?
[ ] How much data was accessed/stolen?
[ ] Was data actually exfiltrated or just accessed?
[ ] What was employee's motivation?
[ ] Were any controls bypassed?
[ ] Could we have detected this earlier?

Lessons Learned:
[ ] What system failures allowed this?
[ ] Were access controls sufficient?
[ ] Was monitoring adequate?
[ ] Should off-boarding procedure been faster?
[ ] Were there early warning signs we missed?

Action Items:
1. Implement DLP system
   - Owner: Security Lead
   - Deadline: 2026-03-31
   - Status: OPEN

2. Deploy database activity monitoring (DAM)
   - Owner: Database Team
   - Deadline: 2026-02-28
   - Status: OPEN

3. Review all privileged access
   - Owner: IT Security
   - Deadline: 2025-12-31
   - Status: OPEN

4. Implement access review process (quarterly)
   - Owner: Identity Lead
   - Deadline: 2025-12-15
   - Status: OPEN

5. Update off-boarding procedure
   - Owner: HR + IT
   - Deadline: 2025-12-01
   - Status: OPEN
```

---

## Quick Reference

```
SEVERITY: SEV-1 or SEV-2 (escalate to HR, Legal, CEO)

DO NOT:
✗ Alert employee before evidence collected
✗ Confront employee alone
✗ Let employee use computer after discovery
✗ Ignore potential crime (report to police)
✗ Delete evidence or discuss with others

IMMEDIATE (5 min):
1. Page incident commander, security lead, HR lead, legal
2. Preserve all digital evidence (logs, emails, files)
3. Plan evidence collection
4. Prepare for access revocation

INVESTIGATION (30 min):
- What data was accessed?
- Was it exfiltrated?
- What's the intent (theft, sabotage, espionage)?
- When did activity start?
- Is there a pattern?

TERMINATION (1-2 hours):
- Coordinate with HR on meeting timing
- Prepare access revocation list
- Collect laptop/phone/badge
- Execute termination + access removal simultaneously
- Preserve evidence

POST-TERMINATION:
[ ] Monitor for re-access attempts
[ ] Monitor dark web for data sales
[ ] Coordinate with law enforcement (if crime)
[ ] Notify affected individuals (if breach)
[ ] Implement preventive measures

ESCALATION:
- Data exfiltrated + regulatory data: Activate breach notification
- Potential crime: Notify law enforcement
- IP theft: Prepare legal action
- Financial impact: Notify CEO/Board
```

---

**Status**: ✅ Production Ready (2025-11-16)
**Last Tested**: 2025-09-15 (role-play scenario)
**Next Drill**: 2026-01-15
**Owner**: Chief Information Security Officer + HR Director
