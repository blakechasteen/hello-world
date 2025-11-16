# Ransomware Incident Runbook

**Version**: 1.0.0
**Created**: 2025-11-16
**Severity**: SEV-1 (CRITICAL - always)
**Type**: Malware / Availability / Confidentiality (double extortion)

---

## Quick Start (First 5 Minutes)

```
🚨 RANSOMWARE DETECTED - SEV-1 IMMEDIATE ESCALATION

1. DO NOT PAY RANSOM (likely illegal, funds criminals)
2. DO NOT NEGOTIATE with attacker
3. DO NOT CONNECT infected systems to backup systems
4. IMMEDIATELY ISOLATE all affected systems
5. PAGE: Incident Commander, Security Lead, DBA, CTO
6. PRESERVE: Do not shut down (need logs for recovery)
7. ENGAGE: Forensics firm, law enforcement, cyber insurance
8. ASSESS: Can we decrypt or restore from backup?
```

**DO NOT**:
```
✗ DO NOT shut down infected systems (preserve evidence/memory)
✗ DO NOT connect to other systems (ransomware spreads)
✗ DO NOT pay ransom (FBI doesn't recommend, funds criminals)
✗ DO NOT call attacker's phone number
✗ DO NOT pay in cryptocurrency without legal advice
✗ DO NOT delete backup systems (restore depends on them)
✗ DO NOT format drives (may destroy recovery options)
```

---

## Section 1: Identification

### Ransomware Indicators

```
File System Changes:
[ ] New file extension on many files (.encrypted, .locked, etc)
[ ] Files encrypted with strong encryption (can't open)
[ ] Ransom note in text file on desktop / file shares
[ ] File timestamps changed to current time

Behavior Indicators:
[ ] System extremely slow (encryption in progress)
[ ] High CPU usage for unknown process
[ ] High disk I/O activity
[ ] Network traffic to unusual IPs

System Indicators:
[ ] Windows Update stopped / disabled
[ ] Antivirus disabled or removed
[ ] Backup process killed / disabled
[ ] Admin accounts created or modified
[ ] Registry changes (disabled UAC, security features)

Log Indicators:
[ ] Process runs with SYSTEM privileges (escalated)
[ ] Massive file access/modification in short time
[ ] Backup deletion attempts in logs
[ ] Network reconnaissance activity

Ransom Note Content:
- Your files have been encrypted
- Contact attacker at [email / chat] to pay
- Bitcoin wallet address for payment
- Threat: "Delete proof" after N days
- Often: "Double extortion" threat (data exfiltrated too)
```

### Confirmation Steps

```
[ ] Identify file extension(s) (.encrypt, .locky, .wannacry, etc)
[ ] Check if files actually encrypted (try opening one)
[ ] Identify ransom note location and content
[ ] Identify affected systems (scan network)
[ ] Determine attack scope (how many systems?)
[ ] Check for data exfiltration (network logs for large transfers)
[ ] Identify entry vector (phishing, RDP, vulnerability)
```

---

## Section 2: Immediate Response (0-30 Minutes)

### Critical First Steps

```
T+0min: Ransomware confirmed
├─ DO NOT PANIC
├─ DO NOT shut down infected system
├─ Page Incident Commander + Security Lead + CTO
├─ Activate war room: #incident-ransomware-YYYY-XXXX
└─ Notify executive leadership

T+5min: Initial containment
├─ Isolate infected system from network
│  └─ Unplug network cable (physical disconnect)
│  └─ Disable WiFi if present
│  └─ Keep system powered on (preserve memory/logs)
├─ Check for infected systems on network
│  └─ Run anti-malware scan on other systems
│  └─ Look for attacker's lateral movement
└─ Preserve forensic evidence
   └─ Capture memory dump (before shutdown)
   └─ Take forensic image of infected disk

T+10min: Assessment
├─ Identify ransomware family
│  └─ File extension: .encrypted, .locked, etc
│  └─ Ransom note content: Look for attacker name/email
│  └─ Check: https://www.nomoreransom.org/ for decryption tools
├─ Assess scope
│  └─ How many systems infected?
│  └─ What percentage of organization?
│  └─ Mission-critical systems affected?
└─ Check for backups
   └─ Do we have clean backups?
   └─ Are backups also encrypted/damaged?
   └─ Can we restore from backups?

T+20min: Executive notification
├─ CEO notified (SEV-1)
├─ CFO informed (financial impact estimate)
├─ General Counsel engaged (legal implications)
├─ Insurance provider contacted (claim eligibility)
├─ Board informed (if publicly traded company)
└─ Prepare communication for employees
```

### System Isolation Procedure

```
CRITICAL: Disconnect before attacker can spread

Infected System:
1. Unplug network cable from back of computer
2. Disable WiFi (if wireless connection active)
3. Leave system powered on (don't shut down)
4. Place physical alert on system ("DO NOT USE - INFECTED")
5. Move system to isolated area

Network Isolation:
1. Remove infected system from network switch
2. Block system IP at firewall
3. If VM: Disconnect from network, power down (don't shut down host)

Check for Additional Infected Systems:
1. Scan all systems for same file extension
2. Scan all systems for same ransom note filename
3. Run antivirus full scan on all systems
4. Check for lateral movement in firewall logs
```

---

## Section 3: Investigation (30 Minutes - 4 Hours)

### Identify Ransomware Family

```
Step 1: File Extension Analysis
[ ] Note all file extensions:
    .wannacry, .encrypt, .locky, .notpetya, etc
[ ] Search filename for clues:
    Often ransomware appends family name

Step 2: Ransom Note Analysis
[ ] Open ransom note (text file)
[ ] Look for ransomware name:
    Example: "This is WannaCry. Your files are encrypted..."
[ ] Check contact information:
    - Email: [attacker@protonmail.com]
    - Chat: [TOR onion URL]
    - Bitcoin wallet: [address]

Step 3: Determine Decryption Availability
[ ] Visit: https://www.nomoreransom.org/
[ ] Search for ransomware family
[ ] Check if free decryption tool available:
    - Some families have decryption tools (keys leaked, etc)
    - If available: Great! Can recover without payment
    - If not available: Must restore from backup or pay

Step 4: Check Ransomware Databases
[ ] https://www.aliendownloader.com/ (ID ransomware)
[ ] https://www.hybrid-analysis.com/ (file analysis)
[ ] Search file hashes in SIEM
[ ] Check if known/tracked ransomware
```

### Determine Attack Vector

```
Entry Point Analysis:

[ ] Phishing Email?
    - Check email logs for suspicious attachments
    - Look for recent credential harvesting campaigns
    - Check if user account compromised

[ ] RDP/SSH Exposed?
    - Check for brute force attempts in logs
    - Did weak password allow compromise?
    - Does external system have RDP exposed to internet?

[ ] Unpatched Vulnerability?
    - Check vulnerability disclosures (recent CVEs)
    - Did we fail to patch known vulnerability?
    - What vulnerability was exploited?

[ ] Insecure VPN / Supply Chain?
    - Did contractor have compromised access?
    - Was VPN password weak or breached?
    - Did third party grant access?

[ ] Lateral Movement?
    - If one system infected first, how did attacker move?
    - What privilege escalation was used?
    - How many "hops" before hitting backups?

Timeline of Attack:
Time: When did infection start?
├─ First file modified timestamp
├─ When did ransomware start encryption?
├─ When was backup last taken?
├─ When was attacker in system before encryption?
└─ Dwell time: Hours? Days? Weeks?

Critical Question: Are backups also compromised?
└─ Can we restore from backup?
└─ Or is backup also encrypted?
└─ When was backup last verified as clean?
```

### Assess Data Exfiltration

```
"Double Extortion" Threat:
Many modern ransomware variants also steal data:
- Attacker exfiltrates database
- Threatens to sell/publish if no ransom
- Increases pressure to pay

Investigation:
[ ] Check network logs for unusual data transfers:
    - Large outbound transfers before encryption?
    - Uploads to attacker FTP/cloud server?
    - Data compressed for exfiltration (ZIP, TAR)?

[ ] Check if attacker mentions data theft:
    - Ransom note says "we also have your files"?
    - Attacker provides sample of stolen data?
    - Threat includes publication timeline?

[ ] Check for data in dark web:
    - Data marketplaces
    - Ransomware-as-a-Service (RaaS) leak sites
    - Later: May need to monitor for months

[ ] Legal assessment:
    - If customer PII/payment data exfiltrated: GDPR breach notification required
    - Even if you don't pay ransom, must notify regulators
```

---

## Section 4: Containment (1-4 Hours)

### Prevent Spread

```
Firewall Rules:
[ ] Block infected systems from reaching others:
    - Deny infected-system.ip to network-segment-A
    - Deny infected-system.ip to database-servers
    - Deny infected-system.ip to backup-systems

[ ] Block C2 (Command & Control) communications:
    - Block known C2 server IPs at firewall
    - Block outbound on unusual ports (8080, 8888, etc)
    - Monitor for DNS queries to attacker domains

Network Segmentation:
[ ] Isolate infection zone:
    - If departmental attack: Isolate that department
    - If company-wide: Harder to contain, focus on backups
    - Preserve network for forensics (don't delete logs)

Backup Protection:
[ ] CRITICAL: Verify backups are not encrypted:
    - Try to restore test file from backup
    - If backup accessible: You can recover!
    - If backup encrypted/deleted: Much harder recovery

[ ] Take backup systems offline (air-gap):
    - Disconnect backup systems from network
    - Prevents attacker from deleting/encrypting backups
    - Limits attacker's damage

[ ] Verify backup integrity:
    - Test restore procedure
    - Verify restored files are not corrupted
    - Verify restore completes successfully

[ ] Document backup status:
    - When was last good backup?
    - How many incremental backups available?
    - Can we restore to point-in-time before attack?
    - Data loss estimate if we restore from backup?
```

### Evidence Preservation

```
Forensic Evidence Collection:

Before Shutdown:
[ ] Memory dump of infected system
    - Use WinPMEM (Windows) or LiME (Linux)
    - Dump to external drive
    - May reveal encryption keys in memory

[ ] Network traffic capture
    - tcpdump of network traffic during encryption
    - Identify attacker C2 connections
    - Identify data exfiltration (if any)

[ ] Logs (before they rotate/delete):
    - Windows Event Logs (Security, Application, System)
    - Authentication logs (who logged in, when)
    - File access logs (what was accessed, when)
    - Network firewall logs
    - Antivirus quarantine logs

Forensic Image:
[ ] Create forensic image of infected disk
    - Use dd, Encase, or forensic tool
    - Preserve entire disk (can take hours)
    - Verify hash matches original (prove integrity)

Chain of Custody:
[ ] Document everything:
    - Who collected evidence (name, date, time)
    - What evidence (type, size, hash)
    - Where stored (secure location)
    - Why collected (incident INC-XXXX)
```

---

## Section 5: Recovery Strategy Decision

### Decision Tree: Pay or Restore?

```
Are backups available AND clean?
├─ YES
│  └─ Can we restore?
│     ├─ YES: Restore from backup (NO RANSOM)
│     └─ NO: Evaluate payment option
├─ NO
│  └─ Data exfiltrated?
│     ├─ YES: Higher pressure to pay
│     └─ NO: Can tolerate data loss?
│        ├─ YES: Restore from older backup / rebuild
│        └─ NO: Consider payment (BUT consult legal first!)
```

### Option 1: Restore from Backup (PREFERRED)

**This is the BEST option if available. Do NOT pay ransom.**

```
Prerequisites:
[ ] Verified clean backup exists
[ ] Backup is accessible and not encrypted
[ ] Can restore all critical systems
[ ] Data loss is acceptable (may lose recent changes)

Recovery Process:
1. Isolate all affected systems (network disconnect)
2. Take forensic image (for investigation/law enforcement)
3. Shutdown infected systems
4. Restore systems from clean backup:
   - Boot from backup
   - Or restore database from backup
   - Or rebuild OS and restore data
5. Verify integrity:
   - File counts match expectations
   - Database consistency checks pass
   - Application starts successfully
6. Hardening before return to production:
   - Patch OS/applications
   - Reset all passwords
   - Enable MFA on admin accounts
   - Deploy EDR/antivirus
   - Update firewall rules
7. Gradually return to production:
   - Single system first, monitor for issues
   - Critical systems next
   - Non-critical systems last

Timeline: 4-24 hours to restore (depending on backup size)
```

### Option 2: Attempt Decryption (If Available)

```
[ ] Check nomoreransom.org for decryption tool
[ ] If available (some ransomware has broken encryption):
    1. Download decryption tool
    2. Test on single encrypted file
    3. If successful: Apply to all files
    4. Verify files are actually decrypted
    5. Monitor for corruption/corruption

Timeline: Minutes to hours (depending on file count)
```

### Option 3: Pay Ransom (ONLY as last resort)

**CONSULT LEGAL COUNSEL FIRST!**

```
Before Paying:
[ ] Legal: Verify payment is legal (may violate sanctions)
[ ] Insurance: Check cyber insurance policy covers payment
[ ] Law Enforcement: Notify FBI/local police (required for some incidents)
[ ] CISO: Understand risks:
    - No guarantee decryption key actually works
    - Attacker may demand more money
    - Funds criminal enterprise
    - May violate export/sanctions regulations
    - Encourages more attacks

IF YOU DECIDE TO PAY (after legal approval):

1. Verify authenticity:
   - Is this really the attacker?
   - Or a scam pretending to be attacker?

2. Obtain quote:
   - Attacker provides price
   - May be negotiable (no fixed price)
   - Usually in Bitcoin or Monero

3. Acquire cryptocurrency:
   - Buy cryptocurrency through exchange
   - Comply with KYC (Know Your Customer) laws
   - Document transaction (for law enforcement)

4. Negotiate:
   - May be able to negotiate lower price
   - Ask for discount for "early payment"
   - Document all communications

5. Transfer payment:
   - Send cryptocurrency to attacker's address
   - Keep confirmation/receipt

6. Receive decryption tool:
   - Attacker sends decryption key/tool
   - Test on single file first
   - Apply to all files

7. Restore data:
   - Decrypt all files
   - Verify decryption successful
   - Check for data corruption

8. STILL INVESTIGATE:
   - Even if you paid, still investigate
   - Still patch vulnerability
   - Still report to law enforcement
   - Still improve defenses
   - Still restore from backups (encrypted version deleted)
```

---

## Section 6: Recovery & Hardening

### Before Returning to Production

```
Security Hardening Checklist:

[ ] Root cause patched
    - Patch unpatched vulnerability?
    - Disable exposed RDP port?
    - Reset compromised credentials?
    - Removed backdoors?

[ ] Credentials reset
    - Force password reset for all users
    - Rotate service account credentials
    - Rotate API keys
    - Reset domain admin passwords

[ ] Hardening measures
    - Enable MFA on all admin accounts
    - Deploy EDR (Endpoint Detection & Response)
    - Enable Windows Defender (if not already)
    - Deploy network segmentation (don't connect all systems)
    - Restrict access to backups (limited accounts)
    - Disable RDP if not needed (or use VPN gateway)

[ ] Backups hardened
    - Air-gap backup (not connected to network)
    - Immutable backups (can't be deleted by attacker)
    - Offline backup copies (in secure vault)
    - Test restore regularly (monthly)

[ ] Detection improved
    - Deploy ransomware detection (behavioral)
    - Alert on mass file creation/encryption
    - Alert on unexpected process execution
    - Monitor for file extension changes

[ ] Architecture reviewed
    - Network segmentation (limit lateral movement)
    - Principle of least privilege (limit who can access what)
    - Remove unnecessary services/accounts
    - Restrict admin access (where can admins log in from?)
```

### Phased Return to Production

```
Phase 1: Testing (4-8 hours)
[ ] Restore single non-critical system
[ ] Verify system functionality
[ ] Check security scans clean
[ ] Verify no re-infection

Phase 2: Validation (12-24 hours)
[ ] Restore critical systems one by one
[ ] Test critical functionality
[ ] Verify data integrity
[ ] Monitor for issues

Phase 3: Return to Operations (24-48 hours)
[ ] Gradually restore all systems
[ ] Monitor continuously
[ ] Be ready to isolate if re-infection
[ ] Communicate status to users
```

---

## Section 7: Post-Incident Actions

### Law Enforcement Notification

```
Contact FBI Cybercrime Division:
[ ] File incident report
[ ] Provide all forensic evidence
[ ] Identify attacker (if known)
[ ] Provide timeline
[ ] Note any payments made

FBI Contact:
- FBI Cybercrime Division: www.fbi.gov/investigate/cyber
- IC3 (Internet Crime Complaint Center): www.ic3.gov
- Local FBI field office: [search by location]
```

### Post-Mortem (Day 3)

```
Questions to Answer:

What Happened?
[ ] How did attacker get initial access?
[ ] Timeline of infection (when to when)
[ ] How much data encrypted?
[ ] Was data exfiltrated?
[ ] How quickly detected?

What Went Well?
[ ] Incident response was quick
[ ] Backups were available
[ ] Forensics preserved
[ ] Communication clear

What Went Poorly?
[ ] Vulnerability not patched
[ ] Backup not tested recently
[ ] No network segmentation
[ ] No MFA on admin accounts
[ ] No EDR deployed

Action Items:
1. Patch root cause vulnerability
   - Owner: DevOps Lead
   - Deadline: 2025-11-23
   - Status: OPEN

2. Deploy MFA organization-wide
   - Owner: Identity Lead
   - Deadline: 2025-12-15
   - Status: OPEN

3. Implement immutable backups
   - Owner: Storage Lead
   - Deadline: 2025-12-31
   - Status: OPEN

4. Deploy EDR to all endpoints
   - Owner: Security Lead
   - Deadline: 2025-12-31
   - Status: OPEN

5. Network segmentation review
   - Owner: Network Lead
   - Deadline: 2026-01-15
   - Status: OPEN
```

---

## Quick Reference

```
SEVERITY: SEV-1 CRITICAL (Always escalate to CEO/Board)

DO NOT:
✗ Pay ransom without legal approval
✗ Shut down infected systems immediately (preserve evidence)
✗ Connect infected to backup systems (malware spreads)
✗ Negotiate with attacker
✗ Delete files (might be needed for recovery)

IMMEDIATE (5 min):
1. Isolate infected system (unplug network)
2. Page incident commander, security lead, CTO
3. Preserve forensic evidence (memory dump)
4. Assess if data exfiltrated (network logs)

INVESTIGATION (30 min):
- Identify ransomware family (file extension, ransom note)
- Check nomoreransom.org for decryption tool
- Assess backup availability and integrity
- Determine entry vector (phishing, RDP, vulnerability)

DECISION (1-2 hours):
- Do we have clean backups? → Restore (no ransom)
- Is decryption tool available? → Use it
- Otherwise: Consult legal about payment

RECOVERY (4-24 hours):
[ ] Restore from backup (preferred)
[ ] Or use decryption tool (if available)
[ ] Or pay ransom (only with legal approval)
[ ] Harden before returning to production

HARDENING:
[ ] Patch root cause vulnerability
[ ] Deploy MFA + EDR
[ ] Test backups regularly
[ ] Segment network (limit lateral movement)
[ ] Air-gap backups (offline copies)

POST-MORTEM (Day 3):
[ ] Timeline + root cause analysis
[ ] What went well/poorly
[ ] Action items assigned
[ ] Verify fixes are being tracked
```

---

**Status**: ✅ Production Ready (2025-11-16)
**Last Tested**: 2025-10-15 (backup restoration drill)
**Next Drill**: 2026-01-15
**Owner**: Chief Information Security Officer
