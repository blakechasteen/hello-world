# SQL Injection Incident Runbook

**Version**: 1.0.0
**Created**: 2025-11-16
**Last Updated**: 2025-11-16
**Severity**: SEV-1 (usually - confirmed injection can lead to data breach)
**Type**: Injection Attack / Data Breach

---

## Quick Start (First 5 Minutes)

```
1. ALERT RECEIVED: "SQL Injection Detected" or "Unusual Database Queries"
2. CONFIRM: Check SIEM/WAF logs - is there actual attack activity?
3. CLASSIFY: If confirmed injection + data exfiltration → SEV-1
           If detected + blocked → SEV-2
           If suspicious pattern but no access → SEV-3
4. PAGE: Incident Commander + Security Lead + DBA
5. PRESERVE: Capture database logs, query history, network traffic
6. ISOLATE: Block attacker's IP at firewall (if known)
7. CREATE: JIRA incident ticket: INC-2025-XXXX
8. NOTIFY: Legal team (potential breach)
```

---

## Section 1: Identification (How to Detect)

### 1.1 Detection Signatures

**WAF/IDS Alerts**:
- UNION-based injection: `UNION SELECT`, `UNION ALL SELECT`
- Time-based blind: `WAITFOR`, `BENCHMARK`, `SLEEP`
- Error-based: `extractvalue()`, `updatexml()`, `syntax error`
- Out-of-band: `INTO OUTFILE`, `INTO DUMPFILE`, `dns exfiltration`
- Comment-based: `--`, `#`, `/**/` with unusual characters

**Database Logs**:
- Unusual `SELECT` queries from application user
- Queries selecting from `information_schema` / `mysql` schema
- `UNION` queries not from legitimate application
- Very long query strings
- Queries with unusual timing (`SLEEP`, `WAITFOR`)

**Application Logs**:
- `500 Internal Server Error` with SQL syntax error
- Parameter values containing SQL keywords (`'`, `UNION`, `--`, etc)
- Unusual query parameters in web server logs

**Network IDS**:
- Large data transfer from database server
- Database server connecting outbound (exfiltration)
- Repeated connection attempts with different payloads

### 1.2 Verification Steps

**Confirm It's Real** (eliminate false positives):

```
[ ] Check source IP: Legitimate user or attacker?
[ ] Check parameter: Is parameter normally user-controlled?
[ ] Check payload: Does it actually work (test in sandbox)?
[ ] Check database: Did query actually execute?
[ ] Check timing: Did this start after recent deployment?
```

**Example: False Positive**
- Alert: "SQL Injection detected"
- Source: Internal IP (10.0.0.50)
- Parameter: `search=contractor's services`
- Reason: Single quote in legitimate text, not SQL injection
- Action: Tune WAF rule, close alert

**Example: Real Positive**
- Alert: "SQL Injection detected"
- Source: External IP (203.0.113.45)
- Parameter: `id=1' UNION SELECT user(), version(), database()--`
- Reason: Multiple SQL keywords, valid UNION syntax, not legitimate
- Action: Escalate to SEV-1, begin investigation

---

## Section 2: Initial Response (0-30 Minutes)

### 2.1 Immediate Actions

**First 5 Minutes**:
```
[ ] Acknowledge alert
[ ] Create incident ticket: INC-2025-XXXX
[ ] Page Incident Commander + Security Lead + DBA
[ ] Activate war room (Slack #incident-sql-injection-XXXX)
[ ] Preserve evidence:
    - Capture database logs (query history)
    - Capture network traffic (tcpdump if possible)
    - Take memory snapshot of database process
```

**5-15 Minutes** (while team assembles):
```
[ ] Confirm injection is real (not false positive)
[ ] Identify affected database(s)
[ ] Identify vulnerable parameter(s)
[ ] Assess severity:
    - Has data been exfiltrated? → SEV-1
    - Is attack ongoing? → SEV-1
    - Was attack blocked? → SEV-2
    - Is attack theoretical? → SEV-3
[ ] If SEV-1: Wake executive, notify legal
```

**15-30 Minutes** (team assembled):
```
[ ] Determine attack scope:
    - Which tables were accessed?
    - Which rows were selected?
    - Was data exfiltrated (check network logs)?
    - How long was access active?
[ ] Initial containment:
    - Block attacker's IP at firewall
    - If vulnerable parameter: Restrict access to application
[ ] Investigation planning
```

### 2.2 Escalation Decision

**SEV-1** (Immediate escalation to executive):
- Attacker confirmed accessing sensitive data
- Data exfiltration confirmed or suspected
- Ongoing active attack
- Multiple systems compromised

**SEV-2** (Security team mobilization):
- Successful injection confirmed but no data access (yet)
- Attack in progress, not yet contained
- Possible credential compromise

**SEV-3** (Routine investigation):
- Injection attempt blocked by WAF
- False positive investigation
- Theoretical vulnerability, no actual access

---

## Section 3: Investigation (30 Minutes - 4 Hours)

### 3.1 Determine Attack Vector

**Vulnerable Parameter Analysis**:
```
[ ] Find the vulnerable parameter
    - Which web form field?
    - Which API parameter?
    - Which URL parameter?
    Example: https://example.com/api/users?id=1' OR '1'='1

[ ] Trace back to database query
    - Find source code (query, ORM call, etc)
    - Document the vulnerable code
    - Identify similar patterns (other parameters)

[ ] Determine why it's vulnerable
    - Not using parameterized queries?
    - Not escaping user input?
    - String concatenation instead of prepared statement?
    Example (vulnerable):
    query = "SELECT * FROM users WHERE id = " + user_input
    Example (safe):
    query = "SELECT * FROM users WHERE id = ?"
    cursor.execute(query, [user_input])
```

### 3.2 Determine Attack Scope

**What Data Was Accessed?**

```
Analyze Database Query Logs:
[ ] Filter logs by timestamp (start of attack)
[ ] Search for attacker's IP address
[ ] Identify all SELECT queries from that IP
[ ] For each query, determine tables/rows accessed
[ ] Rate sensitivity of each table:
    - accounts table (passwords)→ CRITICAL
    - user_profile table (names/emails) → HIGH
    - public_posts table → MEDIUM
    - system_logs table → LOW
```

**Exfiltration Detection**:

```
Check Network Traffic Logs:
[ ] Database server outbound connections?
[ ] Large data transfers from database port?
[ ] Connections to unexpected destinations?
[ ] Data sent via HTTP GET (URL-encoded)?
    Example: attacker sends query results via DNS tunneling

Check Database Logs:
[ ] "SELECT ... INTO OUTFILE" commands?
[ ] FTP/HTTP module calls (MySQL)?
[ ] UDF (User Defined Function) calls?
[ ] Out-of-band exfiltration (DNS, HTTP callbacks)?

Timeline of Exfiltration:
- When did exfiltration start?
- When did it end?
- How much data transferred (estimate)?
- To what IP address/domain?
```

**Dwell Time Analysis**:
```
When did attacker first gain access?
├─ Check database audit logs for first injection
├─ Check web server logs for first attack attempt
├─ Check for earlier compromise (backdoors, lateral movement)
└─ Estimated dwell time: Detection time - First access time

Example:
- Attack detected: 2025-11-16 14:30 UTC
- First injection attempt in logs: 2025-11-15 22:45 UTC
- Dwell time: 15 hours 45 minutes
- Risk: 15 hours of potential data access
```

### 3.3 Identify Similar Vulnerabilities

**Code Audit** (critical - attacker likely tested multiple parameters):

```
[ ] Review application codebase for similar patterns
[ ] Search for string concatenation in queries
    - grep -r 'SELECT.*".*+' src/
    - grep -r "SELECT.*'.*+" src/
[ ] Search for missing parameterization
    - grep -r 'query(' src/
    - grep -r '.execute(' src/
[ ] Identify all user-controlled parameters
    - GET parameters
    - POST parameters
    - HTTP headers
    - JSON fields
[ ] Test each parameter with injection payloads:
    - Test query: `UNION SELECT NULL,NULL,NULL`
    - Test query: `' OR '1'='1`
    - Test query: `1; DROP TABLE users;--`

[ ] Create list of vulnerable parameters (order by risk)
```

---

## Section 4: Containment (30 Minutes - 2 Hours)

### 4.1 Short-Term Containment

**Immediate Actions** (blocking the attack):

```
[ ] Block attacker's IP at firewall
    - Add rule: Block [attacker_ip] port 80,443
    - Verify rule is active (test from attacker IP)

[ ] Disable vulnerable parameter (if possible)
    - Disable URL parameter parsing
    - Return 400 Bad Request for queries
    - Roll back recent code changes

[ ] Restrict database access
    - Application user: Revoke SELECT from information_schema
    - Application user: Revoke SELECT from mysql schema
    - Application user: Restrict to necessary tables only

[ ] Enable database query logging (if not already)
    - MySQL: SET GLOBAL log_queries_not_using_indexes = ON
    - PostgreSQL: log_statements = 'all'
    - Monitor query logs in real-time
```

**Example Firewall Rule**:
```
Block-SQLi-Attacker:
  source: 203.0.113.45 (attacker IP)
  action: DROP
  log: YES
  comment: "Blocking SQL injection attacker INC-2025-XXXX"
```

### 4.2 Long-Term Containment

**Patch Vulnerable Code**:

```
[ ] Identify vulnerable query
[ ] Rewrite with parameterized query
    BEFORE: query = "SELECT * FROM users WHERE id = " + user_id
    AFTER:  query = "SELECT * FROM users WHERE id = ?"
             cursor.execute(query, [user_id])

[ ] Test patch in sandbox
    - Does legitimate traffic work?
    - Do injection payloads fail?
    - Are there performance implications?

[ ] Deploy patch
    - Update code repository
    - Deploy to staging
    - Run regression tests
    - Deploy to production
    - Monitor for errors

[ ] Verify patch worked
    - Attempt injection on patched parameter
    - Verify injection fails (parameter treated as value, not SQL)
```

**Example Parameterized Query** (Python):
```python
# VULNERABLE:
user_id = request.args.get('id')
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)

# SAFE:
user_id = request.args.get('id')
query = "SELECT * FROM users WHERE id = %s"
cursor.execute(query, [user_id])
```

### 4.3 Evidence Preservation

```
[ ] Database logs (preserve before rotation)
    - Query history
    - User access logs
    - Audit trail

[ ] Network traffic (preserve PCAP files)
    - tcpdump from database server
    - WAF logs
    - IDS alerts

[ ] Application logs (preserve all relevant logs)
    - Web server access logs
    - Application error logs
    - Database driver logs

[ ] Memory dumps (from database process)
    - Used to recover ephemeral data

[ ] Store securely with chain of custody
    - Who? (person who collected)
    - When? (timestamp)
    - What? (file hash, size)
    - Where? (secure storage location)
    - Why? (incident INC-XXXX)
```

---

## Section 5: Eradication (1-4 Hours)

### 5.1 Root Cause Analysis

**Why Was This Vulnerable?**

```
Technical Reasons:
[ ] Code not using parameterized queries
[ ] Input validation not implemented
[ ] WAF rules not catching pattern
[ ] Database permissions too permissive
[ ] No rate limiting on parameter

Process Reasons:
[ ] Code review didn't catch vulnerability
[ ] Security testing not performed
[ ] Parameter not tested for injection
[ ] Deployment without security review

Systemic Reasons:
[ ] No secure coding training
[ ] No input validation library in place
[ ] No automated vulnerability scanning
[ ] Security not part of development culture
[ ] Incident response procedures outdated
```

**5 Whys Analysis**:
```
Q: Why was SQL injection not detected?
A: WAF rule didn't match UNION injection pattern

Q: Why didn't WAF rule match?
A: Rule was written for older injection patterns

Q: Why was rule not updated?
A: No feedback loop from production attacks to WAF team

Q: Why no feedback loop?
A: Security team and DevOps team not communicating

Q: Why not communicating?
A: No regular security briefing for DevOps

REMEDY: Implement weekly security briefing for DevOps with alert patterns
```

### 5.2 Vulnerability Remediation

**Fix the Code**:
```
Vulnerability: User ID parameter is concatenated into SQL query

Risk: Attacker can inject SQL code, read/modify/delete any data

Fix Option 1 (Recommended): Use parameterized query
[ ] Rewrite query with placeholders
[ ] Pass user input as separate parameter
[ ] Database driver handles escaping automatically
[ ] Guarantees injection protection

Fix Option 2: Input Validation
[ ] Whitelist allowed characters
[ ] Validate input format (if ID, must be numeric)
[ ] Still use parameterized queries!
[ ] Validation is defense-in-depth, not primary protection

Fix Option 3: Stored Procedures
[ ] Move SQL to database stored procedure
[ ] Call procedure with parameterized inputs
[ ] Stored procedure has strict input validation
[ ] Requires careful design (still vulnerable if concatenates!)
```

**Verification**:
```
[ ] Patch deployed to production
[ ] Confirm patch is running:
    - curl https://vulnerable-app/test
    - Verify injection payload is rejected (not executed)

[ ] Regression testing:
    - Legitimate queries still work
    - Performance not degraded
    - No new errors introduced

[ ] Penetration testing:
    - Attempt UNION injection: BLOCKED ✓
    - Attempt time-based blind: BLOCKED ✓
    - Attempt error-based: BLOCKED ✓
    - Attempt out-of-band: BLOCKED ✓
```

### 5.3 Threat Removal

**Clean Up Attacker Traces**:

```
[ ] Remove attacker's backdoors
    - Check for new user accounts
    - Check for web shells (unusual files in webroot)
    - Check for cron jobs from attacker

[ ] Reset credentials
    - Database: Change application user password
    - Database: Reset any compromised service accounts
    - Application: Force password reset for affected users

[ ] Check for lateral movement
    - Did attacker break into other systems?
    - Check logs on other database servers
    - Check logs on application servers
    - Check for persistence mechanisms
```

---

## Section 6: Recovery (2-8 Hours)

### 6.1 System Restoration

**For Data Integrity**:

```
If attacker modified data:
[ ] Restore database from clean backup
[ ] Identify when backup was taken (before attack)
[ ] Verify backup integrity (test restore in sandbox)
[ ] Restore from backup:
    - Stop application server
    - Stop database
    - Restore database from backup
    - Verify data integrity
    - Start database
    - Start application server

If only data was read (no modification):
[ ] No data restoration needed
[ ] Focus on preventing future access
```

### 6.2 Recovery Validation

**Before Returning to Production**:

```
[ ] Patch applied and verified
[ ] Similar vulnerabilities fixed
[ ] Backdoors removed
[ ] Attacker IP blocked at firewall
[ ] Database permissions restricted
[ ] Credentials rotated
[ ] Data integrity verified
[ ] WAF rules updated
[ ] Monitoring/alerting configured
[ ] Backup tested (if restored)
[ ] Security team sign-off
```

### 6.3 Phased Rollout

```
Phase 1: Isolated Test (30 min)
└─ Restore to test environment
  └─ Run security validation
  └─ Verify application still works

Phase 2: Monitor (4-12 hours)
└─ Restore single production server
  └─ Monitor for errors/issues
  └─ Check application logs
  └─ Check database performance

Phase 3: Gradual Return (hours)
└─ Restore remaining servers
  └─ Test critical functionality
  └─ Monitor system metrics
  └─ Verify no re-exploitation
```

---

## Section 7: Post-Mortem (3 Days After)

### 7.1 Post-Mortem Meeting

**Attendees**: Incident Commander, Security Lead, DBA, DevOps, Developers

**Topics**:

```
What Happened?
- Attack timeline (when start, when detected, when patched)
- Attack scope (what data, how accessed)
- Root cause (vulnerable parameter, missing parameterization)

What Went Well?
- Detection was relatively fast (SIEM alerted within X minutes)
- Team assembled quickly
- Evidence was preserved
- Patch was deployed quickly

What Went Poorly?
- WAF rule didn't catch injection pattern
- Code review didn't catch vulnerability
- No security testing in pipeline
- Feedback from production not reaching DevOps

Action Items
1. Update all parameterized query code
   - Owner: Development Lead
   - Deadline: 2025-12-16
   - Status: OPEN

2. Implement input validation library
   - Owner: Security Lead
   - Deadline: 2025-12-31
   - Status: OPEN

3. Add injection testing to CI/CD pipeline
   - Owner: DevOps Lead
   - Deadline: 2026-01-15
   - Status: OPEN

4. Update WAF rules for UNION injection
   - Owner: Network Security
   - Deadline: 2025-11-20
   - Status: OPEN

5. Train developers on secure coding
   - Owner: Security Lead
   - Deadline: 2025-12-15
   - Status: OPEN
```

### 7.2 Lessons Learned Documentation

```
Incident: SQL Injection - Customer Database
Date: 2025-11-16
Impact: 15,000 users potentially exposed
Duration: 15 hours (before detection)

What Worked:
✓ SIEM alert triggered quickly
✓ WAF partially blocked attack
✓ DBA could trace queries in logs
✓ Evidence preservation successful
✓ Executive notification clear

What Didn't Work:
✗ Code review missed vulnerable parameter
✗ No injection testing in pipeline
✗ WAF rules not comprehensive
✗ Development and security not communicating
✗ No automated code scanning

Prevention for Next Time:
→ All new code must use parameterized queries (enforce in PR review)
→ Add SAST (static analysis) to CI/CD pipeline
→ Update WAF rules for all injection patterns
→ Quarterly security review meeting with DevOps
→ Annual penetration testing to find similar issues
```

---

## Quick Reference Card

```
SEVERITY DECISION:
  SEV-1: Injection + exfiltration OR in-progress attack
  SEV-2: Injection confirmed but no access OR blocked after access
  SEV-3: Injection attempt blocked OR false positive

IMMEDIATE ACTIONS (5 min):
  1. Acknowledge alert
  2. Confirm it's real
  3. Page team
  4. Block attacker IP
  5. Preserve evidence

INVESTIGATION (30 min):
  1. Vulnerable parameter?
  2. Data was accessed?
  3. How much data?
  4. Similar vulnerabilities?

CONTAINMENT (1-2 hours):
  1. Block IP at firewall
  2. Disable parameter
  3. Revoke excess DB permissions
  4. Update WAF rules

ERADICATION (1-4 hours):
  1. Patch vulnerable code (parameterized query)
  2. Remove backdoors
  3. Reset credentials
  4. Update WAF rules

RECOVERY (2-8 hours):
  1. Restore from backup (if data modified)
  2. Deploy patch to production
  3. Verify patch works
  4. Monitor for re-exploitation

POST-MORTEM (Day 3):
  1. Timeline review
  2. Root cause analysis (5 whys)
  3. What went well / poorly
  4. Action items assigned + tracked
```

---

**Status**: ✅ Production Ready (2025-11-16)
**Last Tested**: 2025-10-15 (tabletop drill)
**Next Drill**: 2026-01-15 (quarterly)
**Owner**: Security Team Lead
