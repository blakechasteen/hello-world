# HoloLoom Incident Response Plan (IRP)

**Version**: 1.0.0
**Created**: 2025-11-16
**Last Updated**: 2025-11-16
**Status**: ✅ Production Ready
**Compliance**: NIST SP 800-61, GDPR Article 33, CCPA §1798.150

---

## Executive Summary

This Incident Response Plan provides comprehensive procedures for detecting, responding to, and recovering from security incidents affecting HoloLoom's systems, data, and users. The plan follows NIST SP 800-61 Rev.3 framework (6 phases) and integrates GDPR/CCPA breach notification requirements.

**Key Metrics**:
- **MTTD** (Mean Time to Detect): Target <15 minutes for SEV-1
- **MTTR** (Mean Time to Respond): <1 hour for SEV-1
- **MTRC** (Mean Time to Resolve/Contain): <4 hours for SEV-1
- **GDPR Notification**: 72 hours maximum from breach awareness
- **False Positive Rate**: <10% (high-confidence alerts only)

---

## 1. Preparation Phase (Policies, Tools, Training)

### 1.1 Incident Response Team Structure

**Incident Commander (IC)**
- Overall incident coordination and escalation
- Decision authority for containment and eradication
- Status communication to executives
- On-call rotation: 24/7 coverage
- Target response time: 5 minutes

**Security Lead (SL)**
- Technical investigation and analysis
- Threat identification and classification
- Containment strategy recommendation
- Evidence preservation
- On-call rotation: 24/7 coverage
- Target response time: 5 minutes

**Communications Lead (CL)**
- Internal notifications (Slack, email)
- Customer communications (email, in-app messages)
- Regulatory notifications (GDPR, state AGs)
- Press release coordination
- Available during business hours + on-call
- Target response time: 30 minutes

**Legal Counsel**
- Regulatory compliance verification
- Breach notification legal requirements
- Litigation risk assessment
- Contract implications
- Available during business hours + on-call
- Target response time: 1 hour

**PR/Marketing Lead**
- Public communications strategy
- Press release drafting and approval
- Social media messaging
- Reputation management
- Available during business hours
- Target response time: 2 hours

**Technical Responders** (Team of 3-5)
- Containment: Stop threat propagation
- Eradication: Remove threat from systems
- Recovery: Restore affected systems
- Evidence collection: Preserve forensics
- On-call rotation: 24/7 coverage
- Target response time: 15 minutes

### 1.2 Incident Response Tools & Infrastructure

**Required Infrastructure**:
- **Incident Tracking System**: JIRA Service Desk or Opsgenie
- **SIEM**: Splunk, ELK Stack, or cloud provider native
- **EDR**: Endpoint Detection and Response (CrowdStrike, Microsoft Defender)
- **Forensics Lab**: Isolated network for evidence analysis
- **Communication**: Encrypted Slack channel, phone bridge
- **Documentation**: Shared incident workspace (Confluence/Notion)
- **Version Control**: Git with signed commits for IR activities

**Tool Checklist** (before an incident occurs):
- [ ] SIEM dashboards configured and tested
- [ ] Alert rules validated (low false positive rate)
- [ ] Forensics lab network isolated and accessible
- [ ] Communication channels (Slack, phone, email) tested
- [ ] On-call schedule published and acknowledged
- [ ] Escalation phone numbers and paging configured
- [ ] Backup communication methods (satellite phone, secondary carrier)
- [ ] Incident response playbooks accessible offline
- [ ] Legal templates reviewed and signed
- [ ] Insurance policies reviewed (cyber liability)

### 1.3 Training & Drills

**Initial Training** (before assignment):
- All team members: 2-hour incident response basics
- Security team: 4-hour technical deep dive
- Leadership: 1-hour executive briefing
- Legal: 2-hour regulatory compliance training
- Communications: 2-hour messaging and templates

**Recurring Training** (every 6 months):
- Full team: 1-hour updates on new threats
- Security team: 2-hour technical labs
- Leadership: 30-minute threat landscape briefing

**Drills & Tabletop Exercises** (every 3 months):
- Quarterly: 2-hour simulated incident (rotated scenarios)
- Annual: Full-scale 8-hour incident simulation
- Post-exercise: Lessons learned documented and tracked

**Training Records**:
```
Last training: 2025-10-15 (Incident Response 101)
Next training: 2026-04-15 (Incident Response 101)
Last drill: 2025-10-22 (Data Breach Scenario)
Next drill: 2026-01-22 (DDoS Scenario)
```

---

## 2. Detection & Analysis Phase

### 2.1 Incident Detection Sources

**Automated Detection** (SIEM, EDR, Monitoring):
- **Alert Rule**: High failed authentication attempts (>10 in 5 min)
  - Source: Auth logs, SIEM
  - Severity: SEV-2 (potential compromise)
  - Action: Lock account, reset password, notify user

- **Alert Rule**: Unusual data access patterns
  - Source: Database audit logs, SIEM
  - Severity: SEV-2 or SEV-3 (depending on scope)
  - Action: Restrict access, begin investigation

- **Alert Rule**: Malware detected on endpoint
  - Source: EDR, antivirus
  - Severity: SEV-1 or SEV-2 (depending on prevalence)
  - Action: Isolate endpoint, preserve forensics

- **Alert Rule**: Ransomware activity detected
  - Source: EDR, file integrity monitoring
  - Severity: SEV-1 (critical)
  - Action: Isolate all affected systems immediately

- **Alert Rule**: Unauthorized privileged access
  - Source: IAM audit logs, SIEM
  - Severity: SEV-1 (credential compromise)
  - Action: Revoke credentials immediately

- **Alert Rule**: Data exfiltration detected
  - Source: DLP, network monitoring
  - Severity: SEV-1 (potential breach)
  - Action: Begin breach notification protocol

**Manual Detection**:
- User reports suspicious activity
- Security team member notices anomaly during review
- Customer reports unauthorized access to their data
- Law enforcement contacts organization about suspected breach

**Detection Escalation**:
1. **Alert fired** → Automated response rule triggers
2. **Human review** → On-call security engineer within 5 minutes
3. **Classification** → Severity assessment within 15 minutes
4. **Escalation** → Incident commander notified if SEV-1/2

### 2.2 Incident Classification

**Severity Levels**:

| Severity | Criteria | Examples | MTTD | MTTR | Escalation |
|----------|----------|----------|------|------|------------|
| **SEV-1 CRITICAL** | Confirmed breach, active threat, complete outage, mass impact | Data breach, ransomware, DDoS with >95% unavailability | <15 min | <1 hour | Immediate page, wake executives |
| **SEV-2 HIGH** | Confirmed attack, credential compromise, partial outage | Successful exploit, single account breach, 30-50% unavailability | <30 min | <4 hours | Page on-call team |
| **SEV-3 MEDIUM** | Blocked attack, policy violation, anomaly | Failed attack, anomaly detected, unauthorized access blocked | <1 hour | <24 hours | Create ticket, notify team |
| **SEV-4 LOW** | Potential threat, false positive, investigation | Suspicious log entry, potential policy violation | <24 hours | <72 hours | Routine investigation |

**Incident Type Classification**:

| Type | Description | SEV Range | Examples |
|------|-------------|-----------|----------|
| **Data Breach** | Unauthorized access or exfiltration | SEV-1 | Database breach, credential theft, customer PII exposure |
| **Malware** | Virus, worm, trojan, ransomware | SEV-1 to SEV-2 | Ransomware infection, backdoor, cryptominer |
| **DDoS** | Denial of service attack | SEV-2 to SEV-3 | Volumetric attack, application-layer attack, botnet |
| **Access Control** | Unauthorized system access | SEV-1 to SEV-2 | Privilege escalation, broken access control, compromised credentials |
| **Injection Attack** | SQL injection, command injection, code injection | SEV-1 to SEV-3 | SQL injection with data exfiltration, remote code execution |
| **Social Engineering** | Phishing, pretexting, vishing | SEV-2 to SEV-3 | Credential theft via phishing, business email compromise |
| **Insider Threat** | Malicious or negligent employee | SEV-1 to SEV-2 | Data theft by employee, unauthorized access, sabotage |
| **Misconfiguration** | Unintentional security exposure | SEV-2 to SEV-3 | Public S3 bucket, exposed API key, default credentials |
| **Supply Chain** | Third-party compromise | SEV-1 to SEV-2 | Vendor software update compromised, dependency vulnerability |
| **Policy Violation** | Breach of security policy | SEV-3 to SEV-4 | Unauthorized software installation, credential sharing |

**Scope Classification**:

| Scope | Impact | Examples |
|-------|--------|----------|
| **Critical System** | Core HoloLoom services down | Main orchestrator, knowledge graph, embedding service |
| **High-Value Data** | Customer PII, payment info, secrets | User credentials, SSN, credit cards, API keys |
| **Multi-User** | Multiple users/customers affected | 100+ users, whole organization, external customers |
| **Single User** | One user/account affected | Individual developer account, test user |
| **Test Environment** | Non-production systems only | Dev, staging, lab environments |

### 2.3 Initial Assessment Checklist

When an incident is reported:

```
[ ] Time incident was first observed (UTC timestamp)
[ ] Severity level assigned (SEV-1/2/3/4)
[ ] Incident type identified (Breach/Malware/DDoS/etc)
[ ] Scope assessed (critical/high-value/multi-user/single/test)
[ ] Initial impact statement (how many users affected, what data at risk)
[ ] Affected systems identified (list by hostname/service)
[ ] Timeline established (when did it start, current status)
[ ] Initial containment action (if any)
[ ] Incident ticket created with ID
[ ] Incident commander assigned
[ ] Security lead assigned
[ ] Communications lead assigned
[ ] Team members paged (if SEV-1/2)
[ ] Executive notification sent (if SEV-1)
[ ] Forensics team notified (if potential breach)
```

---

## 3. Containment Phase (Stop the Bleeding)

### 3.1 Immediate Actions (First 15 Minutes)

**For All Incidents**:
1. Create incident ticket with classification and impact statement
2. Page incident commander, security lead, technical responders
3. Activate war room (Slack channel + phone bridge)
4. Preserve evidence (don't shut down systems yet)
5. Begin detailed logging of all actions taken

**For SEV-1 Incidents**:
1. Wake up executive on-call (if not business hours)
2. Notify legal team immediately
3. Begin customer notification preparation
4. Stand up forensics team
5. Consider system isolation vs. observation

**For SEV-2 Incidents**:
1. Assemble security team for investigation
2. Begin scope assessment
3. Plan containment strategy
4. Notify relevant team leads

### 3.2 Short-Term Containment (Isolate & Stop Spread)

**Goal**: Stop threat propagation without destroying evidence

**Tactics by Incident Type**:

**Data Breach**:
- Revoke compromised credentials immediately
- Force password reset for affected users
- Disable API keys / access tokens
- Block suspicious IPs at firewall
- Isolate affected database server (if not still under attack)
- Preserve database logs for forensics

**Malware Infection**:
- Isolate infected endpoint from network (but don't shut down)
- Block malware C2 domain at firewall
- Kill malware process (preserve memory dump first)
- Disable remote access tools (RDP, SSH)
- Prevent lateral movement (segment network)

**DDoS Attack**:
- Activate DDoS mitigation service (Cloudflare, AWS Shield, etc)
- Redirect traffic through scrubbing center
- Implement rate limiting at WAF
- Scale infrastructure if volumetric attack
- Monitor upstream providers for new attack vectors

**Ransomware**:
- Immediately isolate all affected systems
- Take offline (don't let them sync with unaffected systems)
- DO NOT disconnect affected systems from network monitoring (need logs for recovery)
- Block C2 servers at firewall
- Disable backups (prevent ransomware from destroying backups)
- Activate clean air-gapped recovery environment

**Insider Threat**:
- Immediately revoke employee's access credentials
- Collect their laptop/mobile devices for forensics
- Isolate their user account (don't delete yet)
- Audit all their recent activities
- Monitor for any lateral movement using their credentials
- Prepare for potential severance conversation

### 3.3 Long-Term Containment (Strengthen Defenses)

**Patch Vulnerable Systems**:
- Identify vulnerable software / configurations
- Apply patches in controlled order (test first)
- Verify patches were applied successfully
- Monitor for post-patch issues

**Segment Network**:
- Isolate affected systems from rest of network
- Restrict traffic between segments
- Monitor inter-segment traffic for lateral movement

**Review & Reset Credentials**:
- Force password reset for all affected users
- Revoke and re-issue API keys
- Rotate service account credentials
- Reset database passwords

**Update Detection Rules**:
- Create new alert rules to detect similar attacks
- Increase logging verbosity in affected areas
- Test alerts to ensure they fire correctly

### 3.4 Evidence Preservation

**Critical for Legal & Forensics**:
1. **Capture Memory Dumps**
   - From affected systems before shutdown
   - Use forensically sound tools (dd, WinPMEM, etc)
   - Store with chain of custody documentation

2. **Preserve Logs**
   - Application logs (error, debug, access logs)
   - System logs (auth, syslog, Windows Event Log)
   - Database logs (query logs, audit logs)
   - Network logs (firewall, IDS, proxy)
   - Store in read-only format (immutable storage)

3. **Capture Disk Image**
   - Use forensically sound tools (dd, Encase, etc)
   - Store full disk image with hash (MD5, SHA-256)
   - Document chain of custody
   - Store on isolated, secure storage

4. **Document Everything**
   - Timestamp every action taken
   - Record decisions and reasoning
   - Maintain detailed timeline
   - Photograph screens/logs if needed

---

## 4. Eradication Phase (Remove the Threat)

### 4.1 Root Cause Analysis

**Goal**: Understand HOW attacker got in and WHY defenses failed

**Analysis Steps**:
1. Review system/network logs around time of initial compromise
2. Identify attack vector (phishing, vulnerability, credential stuffing, etc)
3. Trace attacker's actions from entry to discovery
4. Identify all systems/data accessed by attacker
5. Determine dwell time (how long were they in the system)
6. Find lateral movement paths used
7. Identify what triggers could have detected them earlier

**Documentation Template** (see [Root Cause Analysis Template](#root-cause-analysis-template)):
- Confirmed attack vector
- Timeline of attacker's actions
- Systems compromised (ranked by sensitivity)
- Data accessed or exfiltrated (with data classification)
- Lessons learned
- Preventive measures for future

### 4.2 Vulnerability Remediation

**For Each Identified Vulnerability**:
1. **Verify the vulnerability exists** on affected systems
2. **Develop a fix** (patch, workaround, architectural change)
3. **Test the fix** in isolated environment
4. **Plan deployment** (which systems, in what order, rollback plan)
5. **Deploy the fix** with monitoring
6. **Verify the fix** worked and vulnerability is gone
7. **Document the fix** for future reference

**Patch Management**:
- Critical: Deploy within 48 hours
- High: Deploy within 2 weeks
- Medium: Deploy within 30 days
- Low: Deploy within 90 days

### 4.3 Threat Removal

**For Malware**:
1. Identify all instances of malware (using hash/signature)
2. Scan all systems for malware presence
3. Clean/remove malware from infected systems
4. Verify removal with multiple scanning tools
5. Document what was found on each system

**For Unauthorized Access**:
1. Revoke all credentials (passwords, keys, tokens)
2. Reset all service accounts
3. Review access logs for unauthorized actions
4. Disable any backdoors or persistence mechanisms
5. Audit user creation/modification logs

**For Ransomware**:
1. Identify ransomware family (helps determine if decryption available)
2. Check if decryption tool available (https://www.nomoreransom.org/)
3. If no decryption: Restore from clean backups
4. If decryption available: Apply decryption tool
5. Rebuild affected systems from known-good state

### 4.4 System Hardening

**Post-Eradication Hardening**:
- Disable unnecessary services
- Enable security features (ASLR, DEP, SMEP)
- Configure firewall rules strictly
- Enforce MFA on all accounts
- Update firewall/WAF rules
- Implement application whitelisting
- Update antivirus/EDR signatures
- Enable detailed logging

---

## 5. Recovery Phase (Restore Normal Operations)

### 5.1 System Recovery

**For Confidentiality Breaches**:
1. No system rebuild needed (data was read, not corrupted)
2. Focus on preventing future access
3. Notify affected users of breach (see Breach Notification Procedures)
4. May offer credit monitoring, identity protection services

**For Integrity Breaches** (data modified):
1. Identify last known good backup
2. Restore from backup (losing ~N hours of data)
3. Reapply changes since backup (if documented/safe)
4. Validate integrity of restored data
5. Monitor for signs of malware reactivation

**For Availability Breaches** (ransomware, DDoS):
1. **DDoS**: May return to normal once attack stops
2. **Ransomware**: Restore from clean backups
3. Validate all restored systems
4. Run security scans before returning to production
5. Monitor for re-infection

### 5.2 Recovery Validation

**Before Returning to Production**:

```
[ ] All patches applied and verified
[ ] Malware scans clean
[ ] Vulnerable configurations remediated
[ ] Credentials reset / rotated
[ ] Backups verified as clean
[ ] Firewall rules updated
[ ] Monitoring / alerting configured
[ ] User access verified (no unauthorized)
[ ] Security team sign-off received
[ ] Executive approval received
```

### 5.3 Phased Rollout

**For Critical Systems** (especially after ransomware):
1. **Phase 1**: Restore to isolated test environment
2. **Phase 2**: Run security validation tests
3. **Phase 3**: Restore single production instance
4. **Phase 4**: Monitor for 24 hours
5. **Phase 5**: Gradually restore remaining instances
6. **Phase 6**: Return to normal operations

---

## 6. Post-Incident Activity (Lessons Learned)

### 6.1 Post-Mortem Meeting

**Timing**: Within 3 days of incident containment

**Attendees**: Incident commander, security lead, technical responders, management

**Meeting Agenda** (2 hours):
1. **Incident Summary** (10 min): What happened, timeline, impact
2. **Timeline Review** (20 min): Walk through all events chronologically
3. **Root Cause Analysis** (20 min): Why did defenses fail
4. **What Went Well** (10 min): Aspects of response that worked
5. **What Went Poorly** (10 min): Where we stumbled
6. **Action Items** (30 min): Specific, measurable improvements
7. **Owner Assignment** (10 min): Who owns each action item, deadline

**Output**: Post-mortem document (see templates) with action items assigned and tracked

### 6.2 Continuous Improvement

**Trend Analysis** (monthly):
- How many incidents this month? (track MTTD, MTTR, MTRC)
- Common incident types? (focus training/hardening)
- False positive rate? (tune alert rules)
- Are we getting faster at response? (compare to last month)

**Alert Tuning**:
- Too many false positives? Lower sensitivity or add context rules
- Missing real incidents? Increase sensitivity or add new rules
- Rules not firing? Debug and re-test

**Process Improvements**:
- Is documentation clear? Update if confusing
- Are tools working well? Replace if slow/unreliable
- Are team members trained? Schedule additional training
- Are escalation paths clear? Clarify if confused

**Action Item Tracking**:
- Owner of each action item assigned
- Deadline set (typically 30-90 days)
- Status tracked monthly
- Executive visibility if overdue

---

## 7. GDPR Breach Notification (Article 33)

### 7.1 Notification Timelines

**Article 33 Requirements**:
- **72 hours** from "becoming aware" of a breach to notify supervisory authority (DPA)
- No delay in notifying affected individuals (Article 34)
- Notification must contain (see Section 8.3 for details):
  - Nature of breach
  - Categories and ~number of data subjects affected
  - Likely consequences
  - Measures taken or proposed

### 7.2 Decision Tree: Does This Require Notification?

```
Is personal data affected?
├─ NO → No GDPR notification required
└─ YES ↓
   Is there "high risk" of impact to rights/freedoms?
   ├─ NO → Notify DPA only (no individual notification)
   └─ YES ↓
      Notify affected individuals ASAP (no delay)
      Notify DPA within 72 hours
```

**What is "High Risk"**?
- Sensitive personal data (race, religion, political views, biometric, health, sexual orientation, criminal record)
- Financial data (payment info, bank accounts)
- Identification data (SSN, passport number, driver's license)
- If attacker has access to data for extended period
- If attack targeted specifically at this data
- If number of affected individuals is large

### 7.3 Affected Individuals Notification (Article 34)

**Content Required**:
- Name and contact of Data Protection Officer (if applicable)
- Description of breach (nature and consequences)
- Name and contact of our Data Protection Officer
- Measures taken or proposed to protect further breaches
- Contact for more information

**Method**:
- Written communication (email, letter)
- Clear and plain language
- Without undue delay

**Example Template** (see `/home/user/hello-world/templates/customer_notification_email.md`):
```
Subject: Important Security Notice - [Company] Data Protection Incident

Dear [User],

We are writing to inform you of a data security incident that may affect your account.

**What Happened**:
On [DATE], we detected unauthorized access to our systems. Our investigation
confirmed that the following information was affected: [list specific data categories]

**What We're Doing**:
We have immediately [describe containment actions]. We are conducting a full
investigation and have engaged [forensics firm / law enforcement].

**What You Should Do**:
We recommend you [specific recommendations, e.g., change password, monitor credit].

**Questions**:
Contact our Data Protection Officer at dpo@company.com

Sincerely,
[Company Leadership]
```

---

## 8. Communication Templates

### 8.1 Internal Incident Notification

**Slack Message** (first 5 minutes):
```
🚨 INCIDENT DECLARED

Severity: SEV-1 | Type: Data Breach | Time: 2025-11-16 14:30 UTC

Description: Unauthorized access detected to customer database

Incident ID: INC-2025-1142
Incident Commander: @john.smith
War Room: #incident-war-room-2025-1142

Action: Stand by for updates every 15 minutes

🔗 Full incident details: [JIRA ticket]
```

**Email Notification** (full details):
```
To: incident-response-team@company.com
Subject: [SEV-1] Incident INC-2025-1142 - Unauthorized Database Access

Incident Details:
- ID: INC-2025-1142
- Severity: SEV-1 (CRITICAL)
- Type: Data Breach / Unauthorized Access
- Detected: 2025-11-16 14:30 UTC
- Reported By: Security monitoring system

Impact Assessment:
- Systems Affected: Customer database (prod-db-001)
- Users Affected: ~15,000 (estimate)
- Data Affected: Customer names, email addresses, hashed passwords
- Business Impact: Potential customer trust impact, notification required

Initial Response:
- Incident commander assigned: John Smith
- Security lead assigned: Jane Doe
- Technical team paged
- Forensics team activated
- Legal team notified

Current Status: INVESTIGATING
- Evidence being collected
- Scope being determined
- Forensics analysis underway

Next Update: 15:00 UTC (30 minutes)

For questions: #incident-war-room-2025-1142
```

### 8.2 Customer Notification Email

See `/home/user/hello-world/templates/customer_notification_email.md`

### 8.3 Regulatory Notification (GDPR)

See `/home/user/hello-world/templates/regulatory_notification_template.md`

### 8.4 Press Release Template

See `/home/user/hello-world/templates/press_release_template.md`

### 8.5 Post-Mortem Template

See `/home/user/hello-world/templates/post_mortem_template.md`

---

## 9. Special Scenarios

### 9.1 Ransomware Response

**Immediate Actions**:
1. DO NOT PAY RANSOM (likely illegal, funds criminals, no guarantee of decryption)
2. Isolate all affected systems immediately (disconnect from network)
3. Take clean air-gapped backups for forensics
4. Identify ransomware family (check ransom note, file extension)
5. Check https://www.nomoreransom.org/ for decryption tools

**Investigation**:
- Trace back to initial compromise (vulnerability, credential, phishing)
- Determine dwell time (how long was attacker in system)
- Check for backdoors, persistence mechanisms
- Review logs for lateral movement

**Recovery**:
1. Restore from clean backups
2. Apply patches for identified vulnerabilities
3. Reset all credentials
4. Rebuild systems from scratch if uncertain
5. Verify systems are clean before returning to production

**Notification**:
- If personal data exfiltrated: GDPR/CCPA breach notification required
- If encryption only: May not be breach if no data exfiltration
- Determine customer/regulatory notification requirements

### 9.2 Data Breach Response

**Scope Determination**:
1. Exactly what data was accessed/exfiltrated?
2. How many individuals affected?
3. For how long was access available?
4. What is the sensitivity of data (PII, financial, health)?
5. Has data been posted publicly?

**Forensics**:
- Analyze attacker's activities from logs
- Determine if data was actually accessed or just available for access
- Check backups to see if attacker copied data to external system
- Preserve all forensic evidence

**Notification Requirements**:
- GDPR: Notify if "high risk" (see Section 7.2)
- CCPA: Notify if personal information unencrypted/unredacted
- State Laws: Vary by state, generally if unencrypted PII
- PCI DSS: Notify if payment card data breached

**Customer Communication**:
- Be transparent about what was accessed
- Offer credit monitoring / identity protection if SSN/DOB breached
- Provide clear recommendations (change password, monitor accounts)
- Provide contact for questions
- Be prepared for customer attrition

### 9.3 Law Enforcement Cooperation

**When Law Enforcement Contacts Us**:
1. Do not assume they are legitimate (verify with police department)
2. Do not provide evidence without court order (unless serious crime in progress)
3. Consult legal counsel immediately
4. Provide limited information if suspected criminal activity
5. Preserve all evidence according to law enforcement requirements

**Subpoena or Court Order**:
1. Verify legitimacy with legal team
2. Comply with court order
3. Provide information as specified in order
4. Document what was provided and when
5. Notify customer if legally able to do so

---

## 10. On-Call Rotation & Escalation

### 10.1 On-Call Schedule

**Primary On-Call**:
- Incident Commander (rotates weekly)
- Security Lead (rotates weekly)
- Technical Responder (rotates weekly)
- All on-call 24/7 (weekends, holidays included)

**Secondary On-Call**:
- Backup for each role (on-call weekly, after primary)
- Called if primary unavailable

**Escalation On-Call**:
- CISO (paged for SEV-1 only, during business hours)
- CEO (paged for SEV-1 + data breach, during business hours)
- General Counsel (paged for SEV-1 + data breach)

**On-Call Requirements**:
- Acknowledge page within 3 minutes
- Arrive at war room / conference within 5 minutes
- Maintain backup phone for redundancy
- Have laptop/VPN access immediately available
- Communicate status every 15 minutes

### 10.2 Escalation Procedures

**SEV-1 Incident** (Immediate Escalation):
1. Within 3 minutes: Acknowledge page
2. Within 5 minutes: Arrive at war room
3. Within 5 minutes: Page Incident Commander + Security Lead + Responders
4. Within 15 minutes: Page CISO (if business hours) or wake backup
5. Within 15 minutes: Notify CEO (if business hours) or designated backup
6. Within 30 minutes: General Counsel notified
7. Within 60 minutes: Notify Communications Lead
8. Within 60 minutes: First customer/regulatory notification if required

**SEV-2 Incident** (30-minute Escalation):
1. Create incident ticket immediately
2. Page Security Lead + Responders within 15 minutes
3. Incident Commander on standby
4. Assess whether SEV-1 within 30 minutes
5. If remains SEV-2, proceed with contained response
6. Escalate to SEV-1 if new information indicates higher severity

**SEV-3 Incident** (2-hour Escalation):
1. Create incident ticket
2. Security team investigates (no paging required if business hours)
3. If outside business hours, assess whether requires page
4. Generally handled during business hours
5. Escalate to SEV-2 if needed

**SEV-4 Incident** (24-hour Response):
1. Create incident ticket
2. Investigated routinely (no rush)
3. Update ticket as investigation proceeds
4. Escalate if severity changes

---

## 11. Key Metrics & Reporting

### 11.1 Incident Metrics

**Response Speed Metrics**:
- **MTTD** (Mean Time to Detect): Average time from incident start to detection
- **MTTR** (Mean Time to Respond): Average time from detection to first response action
- **MTRC** (Mean Time to Resolve/Contain): Average time from detection to containment
- **MTCO** (Mean Time to Complete/Recover): Average time from detection to full recovery

**Quality Metrics**:
- **False Positive Rate**: % of alerts that are not real incidents
- **Escalation Accuracy**: % of incidents classified at correct severity
- **Mean Recovery Time by Severity**: Track if we're improving over time

### 11.2 Monthly Incident Report

**Report Contents**:
```
HoloLoom Security Incident Report - November 2025

Executive Summary
- Total Incidents: 3 (0 SEV-1, 1 SEV-2, 2 SEV-3)
- Average MTTD: 12 minutes
- Average MTTR: 8 minutes
- Average MTRC: 180 minutes
- Customer Impact: 0 breaches

Incident Details
1. INC-2025-1142: Data Breach (SEV-1) [Oct 22]
   - Detection: Oct 22, 14:30 UTC
   - Root Cause: SQL injection vulnerability
   - Impact: 15K users, customer names/emails
   - Status: ✅ RESOLVED
   - Action Items: 2 open, 3 complete

2. INC-2025-1156: DDoS Attack (SEV-2) [Nov 05]
   - Detection: Nov 05, 09:15 UTC
   - Root Cause: Botnet attack from residential IPs
   - Impact: Service degradation 30 minutes
   - Status: ✅ RESOLVED
   - Action Items: 1 open (implement DDoS mitigation)

3. INC-2025-1178: Suspicious Login (SEV-3) [Nov 12]
   - Detection: Nov 12, 16:45 UTC
   - Root Cause: False positive (legitimate user from new location)
   - Impact: None
   - Status: ✅ CLOSED
   - Action Items: 0

Trend Analysis
- MTTD improving (15→12 min): Better alert tuning
- MTTR stable: Team training is paying off
- MTRC increasing (120→180 min): Indicates more complex incidents
- False positive rate: 5% (good, down from 12% last month)

Improvements This Month
- Updated DDoS alert rules (reduced false positives from 20% to 5%)
- Added new SQL injection detection (caught INC-2025-1142 early)
- Trained team on ransomware response (drill on Oct 18)
- Deployed EDR to all endpoints (now have visibility on all systems)

Recommendations for Next Month
- Implement backup authentication system (reduce single point of failure)
- Increase DDoS mitigation capacity (current threshold is 100Gbps)
- Schedule ransomware response drill
- Review and update incident runbooks
```

---

## 12. References & Resources

- **NIST SP 800-61 Rev.3**: Computer Security Incident Handling Guide
- **GDPR Article 33**: Notification of a personal data breach to the supervisory authority
- **GDPR Article 34**: Communication of a personal data breach to the data subject
- **CCPA §1798.150**: Notification of security breaches
- **NCSC Incident Response Guidance**: https://www.ncsc.gov.uk/
- **Ransomware No More**: https://www.nomoreransom.org/
- **CISA Incident Response**: https://www.cisa.gov/incident-response

---

## Appendix A: Quick Reference Card

Print this and keep at desk:

```
╔════════════════════════════════════════════════════════════════╗
║          HoloLoom Incident Response Quick Reference             ║
╚════════════════════════════════════════════════════════════════╝

STEP 1: DETECT INCIDENT
└─ Alert fires in SIEM / EDR / user reports issue
   └─ Confirm it's real (check recent changes/deployments)

STEP 2: DECLARE INCIDENT (First 5 minutes)
├─ Create JIRA ticket (ID: INC-YYYY-XXXX)
├─ Page Incident Commander + Security Lead
├─ Slack: #incident-war-room-YYYY-XXXX
├─ Classify severity: SEV-1/2/3/4
└─ Start incident recording (for post-mortem)

STEP 3: ASSESS & CONTAIN (Next 15 minutes)
├─ Gather facts: What? When? Who? How?
├─ Determine impact: How many users? What data?
├─ Immediate containment: Isolate if needed, preserve evidence
├─ Escalate to executive (if SEV-1)
└─ Notify legal team (if potential breach)

STEP 4: INVESTIGATE (Ongoing)
├─ Timeline: When did attacker enter? What did they do?
├─ Root cause: How did defenses fail?
├─ Scope: All affected systems and data
└─ Eradication plan: How to remove threat

STEP 5: REMEDIATE (Ongoing)
├─ Fix root cause: Patch, configuration, redesign
├─ Remove threat: Delete malware, revoke credentials
├─ Harden: Implement preventive measures
└─ Validate: Confirm threat is gone

STEP 6: RECOVER (Hours to Days)
├─ Restore from backups (if needed)
├─ Validate systems are clean
├─ Return to production gradually
└─ Monitor for re-infection

STEP 7: COMMUNICATE (Ongoing)
├─ Internal: Slack updates every 15-30 min
├─ Customers: Email if GDPR/CCPA breach
├─ Regulators: DPA notification within 72h
└─ Media: Press release if public disclosure

STEP 8: IMPROVE (3 Days After)
├─ Post-mortem meeting
├─ Root cause analysis
├─ Action items assigned + tracked
└─ Lessons incorporated into procedures

════════════════════════════════════════════════════════════════

ESCALATION NUMBERS (Save in Phone):
- On-Call IC: [number]
- On-Call Security Lead: [number]
- CISO: [number]
- CEO: [number]
- General Counsel: [number]
- War Room: [Slack channel]

════════════════════════════════════════════════════════════════

KEY CONTACTS:
- DPA (GDPR notifications): [email/phone]
- Forensics Firm: [number]
- Cyber Insurance: [claim number]
- Law Enforcement (cyber): [number]
```

---

**Document Status**: ✅ Complete and ready for production deployment
**Last Review**: 2025-11-16
**Next Review**: 2026-05-16 (annual review + updates)
**Owner**: Chief Information Security Officer
**Approval**: [Signature required before deployment]
