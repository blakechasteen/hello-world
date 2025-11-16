# Internal Incident Notification Templates

---

## Slack: First Alert (T+0 minutes)

```
🚨 INCIDENT DECLARED

Severity: SEV-1 | Type: Data Breach | Detection Time: 2025-11-16 14:30 UTC

**Situation**: Unauthorized access detected to customer database

**Details**:
- Systems Affected: prod-db-001, prod-api-02
- Potential Users Affected: 15,000
- Data at Risk: Customer names, emails, hashed passwords
- Incident ID: INC-2025-BREACH-0001
- Incident Commander: @john.smith
- War Room: #incident-war-room-2025-BREACH-0001

**Action**: Team members stand by for updates every 15 minutes

🔗 Full details: [JIRA ticket link]
```

---

## Email: Detailed Notification (T+5 minutes)

```
To: incident-response-team@company.com
Subject: [SEV-1] INCIDENT INC-2025-BREACH-0001 - Unauthorized Database Access

Body:
────────────────────────────────────────────────────────────────

An incident has been declared requiring immediate response.

INCIDENT SUMMARY
────────────────

Incident ID: INC-2025-BREACH-0001
Severity: SEV-1 (CRITICAL)
Type: Data Breach / Unauthorized Access
Detection Time: 2025-11-16 14:30 UTC (T+0)
Detected By: SIEM Alert (Unusual Database Activity Rule)
Reported By: Security Operations Center

SITUATION
─────────

At 14:30 UTC, our SIEM system detected suspicious database activity from
an unauthorized source. Initial investigation confirms unauthorized access
to our customer database.

PRELIMINARY IMPACT ASSESSMENT
──────────────────────────────

Systems Affected:
  • Database: prod-db-001 (primary customer database)
  • Application Server: prod-api-02
  • Time Window: 02:15-14:30 UTC (approx 12 hours)

Data Potentially Affected:
  • Customer Full Names: ~15,000
  • Customer Email Addresses: ~15,000
  • Customer Phone Numbers: ~7,000
  • Hashed Passwords (bcrypt): ~15,000
  • NOT Affected: Payment card data (not stored), SSN (not collected)

User Impact:
  • Customers Affected: ~15,000 out of 200,000 (7.5%)
  • Data Sensitivity: MEDIUM-HIGH (names + emails = phishing risk)
  • Breach Notification: LIKELY REQUIRED (per GDPR Article 34)

Initial Assessment: HIGH RISK
  → Large number of users
  → Sensitive data (personally identifiable)
  → Extended dwell time (12 hours before detection)

INITIAL RESPONSE ACTIONS (ALREADY UNDERWAY)
────────────────────────────────────────────

✓ Incident declared and escalated
✓ Incident Commander assigned: John Smith
✓ Security Lead assigned: Jane Doe
✓ Technical response team paged
✓ War room established (#incident-war-room-2025-BREACH-0001)
✓ Forensics team activated
✓ Evidence preservation begun
✓ Database access restricted pending investigation

ESCALATION & NOTIFICATIONS (IN PROGRESS)
──────────────────────────────────────────

Within Next 30 Minutes:
[ ] CEO, CTO, CISO notified
[ ] General Counsel engaged
[ ] Cyber Insurance provider contacted
[ ] Breach notification team assembled

Executive Briefing: 15:30 UTC (1 hour from now)

WHAT YOU NEED TO DO
────────────────────

If Your Name Below - YOU'RE NEEDED:

📞 Incident Commander (John Smith): Coordinate overall response
🔐 Security Lead (Jane Doe): Lead technical investigation
🏗 DBA Lead (Mike Johnson): Database forensics + access restriction
⚖ General Counsel (Sarah Lee): Legal assessment + regulatory notifications
📱 Communications Lead (Tom Wilson): Prepare customer communications
🖥 Network Security (Alex Chen): Restrict attacker's IP + firewall rules

Everyone Else: Please stand by for updates. Do not discuss incident outside
this channel. More information coming within 1 hour.

STATUS & NEXT UPDATES
──────────────────────

Current Status: INVESTIGATING
  - Active investigation underway
  - Forensics team analyzing logs
  - Scope still being determined

Next Full Update: 15:30 UTC (1 hour)
  - Initial forensics results
  - Confirmed customer count
  - Preliminary timeline
  - Recommended actions

Communication Channel: #incident-war-room-2025-BREACH-0001
  - Real-time updates every 15-30 minutes
  - Technical Q&A as needed
  - Executive updates in main channel

For Questions: Message Incident Commander (@john.smith) directly

This is a serious incident requiring all-hands focus. Thank you for your
attention and support as we respond.

────────────────────────────────────────────────────────────────
Incident ID: INC-2025-BREACH-0001
War Room: #incident-war-room-2025-BREACH-0001
JIRA Ticket: SECURITY-12345
────────────────────────────────────────────────────────────────
```

---

## Email: Hourly Status Update (T+60 minutes)

```
To: incident-response-team@company.com, leadership@company.com
Subject: UPDATE: INC-2025-BREACH-0001 - Incident Status (15:30 UTC)

Body:
────────────────────────────────────────────────────────────────

INCIDENT STATUS UPDATE - Hour 1

Incident: INC-2025-BREACH-0001 (Data Breach)
Severity: SEV-1 CRITICAL
Time Elapsed: 1 hour since detection
Current Status: INVESTIGATING / CONTAINING

KEY FINDINGS (Updated)
──────────────────────

Confirmed Details:
✓ Confirmed unauthorized database access
✓ Attacker IP identified: 203.0.113.45
✓ Attack method: SQL injection via login form
✓ Data exfiltration: CONFIRMED (large file transfer detected)
  └─ 4.2 GB of customer data downloaded to external FTP server
✓ Dwell time: ~12 hours (02:15-14:30 UTC)
✓ Entry vector: Phishing email with stolen credentials

Customer Impact:
✓ Confirmed affected: 15,000 customers (all with account access)
✓ Data confirmed exfiltrated:
  - Customer full names (PII)
  - Email addresses (PII)
  - Phone numbers (PII)
  - Hashed passwords (not plaintext, reduced risk)
  - Account creation dates

Regulatory Assessment:
→ GDPR: Applies (EU customers affected)
→ CCPA: Applies (California customers affected)
→ Notification required: YES (high risk determination)

ACTIONS COMPLETED
──────────────────

Technical Containment:
✓ Attacker IP (203.0.113.45) blocked at firewall
✓ Database access for compromised account revoked
✓ SQL injection vulnerability identified (login form email field)
✓ Database access log collection begun (forensics)
✓ Evidence preservation: Memory dump, PCAP, query logs captured

Investigation Progress:
✓ Root cause confirmed: SQL injection + credential theft
✓ Phishing email identified (sent 2025-11-15 22:30 UTC)
✓ Employee account compromised (email employee contacted, password reset)
✓ Lateral movement investigated: No evidence of other systems compromised

Forensics Initiated:
✓ Third-party forensics firm engaged (Mandiant contacted)
✓ Complete database image capture underway (large file, ~2 hours)
✓ Network PCAP analysis started
✓ Exfiltration destination identified (ftp.attacker-ip.ru)

NEXT ACTIONS (Next 2 Hours)
─────────────────────────────

Immediate (Next 30 minutes):
[ ] Patch SQL injection vulnerability (code fix deployed to staging)
[ ] Test patched code for regression
[ ] Final forensics collection

Short-term (1-2 hours):
[ ] Deploy patch to production
[ ] Begin customer notification process
[ ] Submit DPA notification (GDPR Article 33)
[ ] Prepare press release

Medium-term (Next 24 hours):
[ ] Send notifications to all affected customers
[ ] Publish FAQ on website
[ ] Deploy credit monitoring service
[ ] Conduct follow-up customer communications

COMMUNICATIONS UPDATE
──────────────────────

Notification Timeline:
→ T+24 hours: Customer emails sent
→ T+48 hours: GDPR DPA notification submitted
→ T+72 hours: GDPR deadline (final deadline for DPA notification)

Key Message for Customers:
"We detected unauthorized access, investigated thoroughly,
patched the vulnerability, and are implementing additional safeguards
to prevent future incidents. We take your security seriously."

Internal Communications:
→ All-hands meeting tomorrow at 09:00 UTC (brief overview, Q&A)
→ Executive team meeting: 18:00 UTC today
→ Updates to customers: Starting tomorrow morning

RESOURCE ALLOCATION
────────────────────

People Assigned:
- Incident Commander: John Smith (full-time)
- Security Lead: Jane Doe (full-time)
- Technical Responders: 4 people (full-time)
- Communications Lead: Tom Wilson (full-time)
- Legal Counsel: Sarah Lee (primary focus)
- DBA: Mike Johnson (dedicated to forensics)

External Resources:
- Forensics Firm: Mandiant (8 analysts assigned)
- Cyber Insurance: Cyber Liability claim filed
- Law Enforcement: FBI contacted (investigation stage)
- Crisis Communications: Hired external PR firm

METRICS SO FAR
────────────────

MTTD (Mean Time to Detect): 12 hours (slower than ideal)
  → Alert fired at 14:30 UTC, attack started ~02:15 UTC
  → Improvement needed: Better anomaly detection

MTTR (Mean Time to Respond): 45 minutes
  → From detection to initial response actions

MTRC (Mean Time to Contain): 2 hours (expected)
  → Complete containment estimated within 2 hours

Financial Impact (Preliminary):
→ Forensics firm: ~$50K
→ Credit monitoring: ~$20K
→ Legal/regulatory: ~$30K
→ Customer notifications: ~$10K
→ Estimated total: $110K+
→ Cyber insurance expected to cover majority

QUESTIONS & CONCERNS
─────────────────────

Concern: "What about our reputation?"
Answer: We're proactive, transparent, and customers will appreciate quick
        response. Reputation usually recovers if handled well.

Concern: "Will customers leave?"
Answer: Possible some churn, but most customers understand incidents happen.
        How we handle matters more than incident itself.

Concern: "Is our data safe now?"
Answer: Yes. Vulnerability patched, attacker IP blocked, database access
        restricted. Additional safeguards being implemented.

Concern: "Will this happen again?"
Answer: No single defense is 100% perfect, but we're implementing:
        - Parameterized queries (prevent SQL injection)
        - MFA on admin accounts (prevent credential theft)
        - DLP system (prevent data exfiltration)

NEXT UPDATE
────────────

Time: 16:30 UTC (1 hour from now)
Expected Content:
  - Patch deployment status
  - Final forensics analysis results
  - Customer notification templates (for approval)
  - Press release draft

Slack Channel for Real-Time Updates:
  #incident-war-room-2025-BREACH-0001

Thank you all for your quick response. This is our time to show what
professional incident response looks like.

────────────────────────────────────────────────────────────────
Incident ID: INC-2025-BREACH-0001
Incident Commander: John Smith (john.smith@company.com)
War Room: #incident-war-room-2025-BREACH-0001
────────────────────────────────────────────────────────────────
```

---

## Email: Executive Briefing (T+90 minutes)

```
To: CEO@company.com, Board@company.com, CFO@company.com
Subject: CONFIDENTIAL: Incident INC-2025-BREACH-0001 Executive Briefing

Body:
────────────────────────────────────────────────────────────────

EXECUTIVE BRIEFING: Data Breach Incident
Prepared for: Board of Directors + Executive Leadership
Classified: CONFIDENTIAL - ATTORNEY-CLIENT PRIVILEGED

THE SITUATION (In Plain English)
──────────────────────────────────

We had a data breach. Here's what you need to know:

WHAT HAPPENED:
- An attacker sent a phishing email to one of our employees
- Employee clicked, entered password, attacker gained access
- Attacker used SQL injection to break into our database
- Attacker downloaded customer data (names, emails, phone numbers)
- We discovered it 12 hours later via our security system

WHO WAS AFFECTED:
- 15,000 customers (out of 200,000)
- Their names, emails, phone numbers were stolen
- We do NOT store credit cards (so no payment card breach)
- We do NOT store SSNs (so no Social Security number breach)

WHAT ARE THE RISKS:
- Phishing: Attackers may use stolen emails to phish customers
- Identity theft: Limited (no SSN, no full personal identity)
- Reputational: Customers will lose some trust
- Financial: Legal/notification costs, credit monitoring, lawsuit risk
- Regulatory: GDPR fines (up to 4% of global revenue) if we mess up notification

WHAT ARE WE DOING NOW:
✓ Fixed the vulnerability
✓ Blocked the attacker
✓ Hired forensics firm
✓ Notifying customers tomorrow
✓ Notifying regulators within 72 hours
✓ Implementing additional safeguards

FINANCIAL IMPACT (PRELIMINARY)
────────────────────────────────

Immediate Costs:
  Forensics firm:        $50,000
  Legal/compliance:      $30,000
  Credit monitoring:     $20,000
  Notifications:         $10,000
  ──────────────────────────────
  Total immediate:       $110,000

Expected Insurance Coverage: 70-80% ($77K-$88K)
Out-of-pocket expected:    $22K-$33K

Ongoing Costs (Next 24 months):
  Additional security:   $500K+
  Potential lawsuits:    $100K-$500K (if any)
  Customer churn:        Estimated $50K-$200K revenue impact
  PR/reputation:         $50K+

REGULATORY & LEGAL STATUS
──────────────────────────

GDPR (European Customers):
  Requirement: Notify supervisory authority (DPA) within 72 hours
  Timeline: Notification due by 2025-11-19 17:30 UTC
  Risk: 4% of annual global revenue fine if noncompliant
  Status: Legal team preparing notification now

CCPA (California Customers):
  Requirement: Notify affected residents within 45 days
  Timeline: Due by 2025-12-31
  Risk: Fines + private right of action lawsuits
  Status: Legal team preparing notification now

State Breach Notification Laws:
  Multiple states affected, varying requirements
  General timeline: 30-60 days
  Status: Legal coordinating with all state AGs

Lawsuit Risk Assessment:
  Probability of class action: HIGH (data breach + customer financial info)
  Expected settlement: $5M-$20M (depending on size/damage)
  Timeline: 12-24 months to settlement

OPERATIONAL IMPACT
────────────────────

Reputation Damage:
  - Customers will lose some trust (temporary)
  - May see 2-5% customer churn
  - Media coverage likely (negative initially)
  - Recovery time: 6-12 months (if handled well)

Customer Service Impact:
  - Expect high call volume (5x normal) for 1 week
  - Staff up customer support team
  - Prepare FAQ and talking points

Stakeholder Impact:
  - Investors: May react negatively (monitor stock)
  - Partners: Some may request security audit
  - Employees: May worry about company stability (internal comms needed)

BOARD DECISION REQUIRED
────────────────────────

Notification Strategy:
  Option A (Proactive): We announce incident before it hits news
    → Shows transparency and leadership
    → Allows us to control narrative
    → Recommended: DO THIS

  Option B (Reactive): Wait for news to break
    → Looks like we're hiding something
    → Customers find out from media (bad optics)
    → NOT RECOMMENDED

Credit Monitoring Offering:
  Option A (24 months free): Costs $20K, shows we care
  Option B (No offering): Saves $20K, but customers upset
  Recommendation: Offer the monitoring (worth the investment)

Legal Defense:
  Recommend: Hire specialized cybersecurity law firm
  Cost: $50K-$100K
  Timeline: Needed immediately

Insurance Claim:
  Status: Claim filed with cyber liability insurance
  Expected coverage: $1M policy limit
  Recommendation: Work closely with insurer

NEXT STEPS & TIMELINE
──────────────────────

Today (T+24 hours):
  [ ] Board approves notification strategy
  [ ] Authorize credit monitoring expense
  [ ] Authorize forensics/legal expenses
  [ ] Approve executive communications

Tomorrow:
  [ ] Customer notifications sent (email + in-app)
  [ ] FAQ published on website
  [ ] Customer support team briefed
  [ ] Press release issued (proactive announcement)

Within 72 hours:
  [ ] DPA (GDPR) notification submitted
  [ ] All state AG notifications sent
  [ ] All-hands meeting with employees
  [ ] Investor communications (if publicly traded)

Next 30 days:
  [ ] Complete forensics investigation
  [ ] Implement security improvements
  [ ] Prepare for lawsuits (with counsel)
  [ ] Monitor for dark web data sales
  [ ] Monitor customer sentiment/churn

QUESTIONS FOR BOARD
─────────────────────

1. Approve notification strategy? (Recommend: YES - proactive)
2. Approve credit monitoring offering? (Recommend: YES)
3. Approve forensics firm spending? (Recommend: YES - essential)
4. Authorize CEO/Communications to handle media? (Recommend: YES)
5. Any concerns or questions?

KEY MESSAGES FOR PUBLIC
────────────────────────

To Customers:
"We take your security seriously. When we discovered unauthorized access to
our database, we immediately investigated, patched the vulnerability, and are
implementing additional safeguards. We're offering free credit monitoring as
a thank-you for your trust."

To Employees:
"This incident is a learning opportunity. We're investing in better security
to prevent this from happening again. Your job is secure - the company is
committed to security excellence."

To Investors:
"Cybersecurity incidents are a fact of doing business online. Our incident
response was fast and effective. We're using this as an opportunity to
strengthen our defenses."

To Regulators:
"We proactively notified affected individuals and regulators. We conducted
a thorough investigation and fixed the root cause. We're implementing
additional protections."

BOTTOM LINE
───────────

✓ Situation: Serious but manageable
✓ Our Response: Professional and swift
✓ Customer Impact: Moderate (names/emails, not payment cards)
✓ Financial Impact: ~$110K immediate, potential lawsuits later
✓ Reputational Impact: Temporary damage, recoverable
✓ Regulatory Risk: High (GDPR fines if we mess up notification), mitigatable
✓ Recommendation: APPROVE PROACTIVE NOTIFICATION STRATEGY

We'll come out of this stronger. Questions?

────────────────────────────────────────────────────────────────
Prepared by: Chief Information Security Officer
Date: 2025-11-16 15:30 UTC
Incident ID: INC-2025-BREACH-0001
Classified: CONFIDENTIAL - ATTORNEY-CLIENT PRIVILEGED
────────────────────────────────────────────────────────────────
```

---

**Status**: ✅ Production Ready (2025-11-16)
**Last Updated**: 2025-11-16
**Owner**: Communications Lead + Incident Commander
