# Breach Notification Procedures (GDPR/CCPA Compliance)

**Version**: 1.0.0
**Created**: 2025-11-16
**Last Updated**: 2025-11-16
**Status**: ✅ Production Ready
**Compliance**: GDPR Articles 33-34, CCPA §1798.150, US State Breach Notification Laws

---

## Executive Summary

HoloLoom's breach notification procedures ensure compliance with GDPR, CCPA, and US state breach notification laws. Key requirement: **72 hours from breach awareness to supervisory authority notification** (GDPR).

**Critical Timeline**:
- **T+0h**: Breach detected/confirmed
- **T+1h**: Legal assessment of notification requirement
- **T+4h**: Communications draft prepared
- **T+24h**: Begin affected individual notifications (GDPR: no undue delay)
- **T+72h**: Notify DPA/regulatory authorities (GDPR deadline)

---

## Part 1: GDPR Notification Requirements (Article 33-34)

### 1.1 Determining Breach Notification Requirement

**Step 1: Is Personal Data Affected?**

Personal data includes any information that can identify an individual:
- Name + email address
- Name + phone number
- Name + username
- IP address (if traceable to individual)
- Cookie identifiers (if traceable to user)
- Device identifiers (if traceable to user)
- **Excluding**: Fully anonymized data (cannot be linked to individual), encrypted data with keys not compromised

```
Is personal data affected?
├─ YES → Continue to Step 2
└─ NO → No GDPR notification required (document why)
```

**Step 2: Is There "High Risk" to Rights/Freedoms?**

**High Risk Factors**:
- Sensitive personal data affected (categories below)
- Large number of individuals affected
- Data remained accessible for extended period
- Attacker appeared to specifically target this data
- Reputational/financial harm to individuals

**Sensitive Personal Data** (especially high risk):
- Racial or ethnic origin
- Political opinions
- Religious beliefs
- Trade union membership
- Genetic data
- Biometric data
- Health data
- Sex life or sexual orientation
- Criminal record or allegations

**Financial Data** (high risk):
- Payment card details
- Bank account information
- Social Security number (US)
- Tax identification
- Credit history

**Identification Data** (high risk):
- Passport number
- Driver's license number
- Identity card number
- Visa/travel documents

```
High risk to data subject rights/freedoms?
├─ YES → Notify affected individuals (Article 34) + DPA (Article 33)
└─ NO → Notify DPA only (Article 33), individuals may not need notification
```

**Step 3: Make Notification Decision**

**Decision Matrix**:

| Data Type | Volume | Duration | Risk | Action |
|-----------|--------|----------|------|--------|
| SSN, DOB, Address | Any | Any | High | **NOTIFY** individuals + DPA |
| Email, Username | <100 | <1 hour | Low | Notify DPA only |
| Email, Username | >1000 | >24 hours | Medium | **NOTIFY** individuals + DPA |
| Payment cards | Any | Any | High | **NOTIFY** individuals + DPA |
| Hashed passwords | Any | <24 hours | Low | Notify DPA only |
| Plaintext passwords | Any | Any | High | **NOTIFY** individuals + DPA |

### 1.2 GDPR Article 33: Notification to Supervisory Authority (DPA)

**Timeline**: No later than 72 hours from becoming aware of breach

**Trigger**: "Becoming aware" = first indication that breach occurred
- NOT when fully understood
- NOT when contained
- NOT when root cause determined
- Example: 14:30 UTC when alert fires, countdown starts 14:30 UTC

**Required Content** (Article 33(3)):

1. **Nature of the Breach**
   - What type of attack (SQL injection, malware, credential theft, etc)
   - How data was accessed (unauthorized system access, exfiltration, etc)
   - Technical description (not marketing speak)

2. **Name and Contact of DPO** (if applicable)
   - Data Protection Officer name
   - Email and phone

3. **Categories of Data Subjects** (affected individuals)
   - Examples: Customers, employees, users
   - List each category separately (may affect differently)
   - Approximate number in each category

4. **Categories of Personal Data**
   - What data was affected (names, emails, etc)
   - Be specific about each category
   - Separate "potentially accessed" from "confirmed accessed"

5. **Likely Consequences**
   - Risk of discrimination
   - Risk of identity theft
   - Financial harm (credit card frauds)
   - Reputational harm
   - Physical harm (if location data compromised)

6. **Measures Taken or Proposed**
   - Immediate containment actions
   - Investigation underway
   - Remediation planned
   - Customer notifications to be sent
   - Hardening measures planned

7. **Communication**
   - Single point of contact for DPA questions
   - Technical details attachment (if requested)
   - Willingness to provide additional information

**Notification Process**:

```
1. Detect breach (T+0h)
2. Legal assessment: GDPR applies? (T+30min)
3. Prepare DPA notification (T+4h)
4. Review with Legal Counsel (T+6h)
5. Submit to DPA via official portal (T+12h)
   - Different DPA for each EU member state
   - If multi-country: Notify DPA of each country
6. Acknowledge receipt from DPA (T+48h)
7. Provide additional info if requested (within 10 days typically)
```

**DPA Contact Information by Country**:

```
Germany (Bundesdatenschutzamt):
  Email: poststelle@bfdi.bund.de
  Phone: +49 30 6677 0
  Portal: https://www.bfdi.bund.de/

France (CNIL):
  Email: plainte@cnil.fr
  Web: https://www.cnil.fr/

Ireland (DPC - EU base for many tech companies):
  Email: dataprotection@dataprotection.ie
  Phone: +353 1 6390500
  Portal: https://form.dataprotection.ie/

UK (ICO):
  Email: casework@ico.org.uk
  Phone: +44 303 123 1113
  Portal: https://ico.org.uk/

Spain (AEPD):
  Email: proteccion@aepd.es
  Portal: https://www.aepd.es/

Italy (Garante):
  Email: reclami@gpdp.it
  Phone: +39 06 696 77 500
  Portal: https://www.garanteprivacy.it/
```

### 1.3 GDPR Article 34: Notification to Data Subjects

**Timeline**: Without undue delay (typically same day as breach confirmed)

**Trigger**: If "high risk" to rights/freedoms determined (see Section 1.1)

**When NOT Required**: If breach risk is low (but DPA notification still required)
- Example: Attacker unable to access data (blocked at encryption layer)
- Example: Data was encrypted and attacker didn't obtain keys

**Required Content** (Article 34(3)):

1. **Name and Contact of DPO** (if applicable)
2. **Description of Likely Consequences**
   - Risk of identity theft
   - Risk of financial fraud
   - Risk of discrimination
   - Reputational consequences
3. **Measures Taken or Proposed**
   - Remediation already implemented
   - Ongoing investigation
   - Preventive measures planned
4. **Contact for Questions**
   - Phone number
   - Email address
   - Preferably: DPO direct contact
5. **Recommended Actions**
   - Change password immediately
   - Enable MFA
   - Monitor credit for fraud
   - Dispute fraudulent transactions
   - Enroll in credit freeze (if SSN compromised)
6. **Transparency About Breach**
   - Clear language (no legal jargon)
   - Don't minimize impact
   - Don't blame users
   - Acknowledge failure of controls

**Notification Methods** (Article 34(4)):
- Direct communication (email to registered address)
- In-app notification (if email fails)
- Website notification (if many individuals affected)
- Media announcement (only if impractical to reach individuals)

**Method Selection**:
```
Individual users affected:
├─ <100 → Direct email to each user
├─ 100-10,000 → Email to most + website notice
└─ >10,000 → Email to sample + website notice + media

Staff/employees:
└─ Direct email + phone call from HR

Customers (business accounts):
└─ Direct email from account manager + phone call
```

### 1.4 GDPR Notification Exemption (Article 33(3))

**Notification NOT Required If**:
- Technical and organizational measures (TOM) applied made data unintelligible
  - Example: Data encrypted with strong encryption and attacker didn't get keys
  - Example: Data anonymized such that cannot be linked back to individual
  - Must be able to demonstrate TOM were effective

**Important**: If you claim encryption exemption, you MUST be able to prove:
1. Encryption algorithm used (AES-256, etc)
2. Key management procedure
3. Attacker could NOT obtain keys
4. Data confirmed encrypted in backups / logs
5. No other means to access unencrypted data

**Documentation Required**:
```
Encryption Exemption Claim:
┌─────────────────────────────────────────────┐
│ Data: Customer email addresses              │
│ Volume: 15,000                              │
│ Encryption: AES-256-GCM                     │
│ Key Location: AWS KMS (different account)   │
│ Key Compromised: NO (verified)              │
│ Backups Encrypted: YES (AES-256)            │
│ Unencrypted Copy: NO                        │
│ Conclusion: Notification NOT required       │
└─────────────────────────────────────────────┘
```

---

## Part 2: CCPA Breach Notification (California)

### 2.1 CCPA Applicability

**Does CCPA Apply?**
- Business collects personal information from California residents
- Business revenue >$25M OR processes data of 100K+ residents/households
- If YES → CCPA applies (and similar state laws)

**CCPA Definition of Personal Information**:
- Information that identifies, relates to, describes, or could reasonably be linked with a particular consumer or household
- Includes: Name, SSN, address, email, phone, employment info, browsing history, IP address, etc
- Excludes: Publicly available information (unless compiled in a way that reveals SSN)

### 2.2 CCPA Notification Requirement

**Trigger**: Breach of unencrypted or unredacted personal information

**Exemptions** (like GDPR):
- Information encrypted
- Information reasonably secured
- Attacker lacked ability to read/use information

**Timeline**: "Without unreasonable delay" (California typically interprets as 45 days)

**Who to Notify**:
- Affected California residents
- California Attorney General (if >500 residents affected)

### 2.3 CCPA Notification Content

Must include:
1. Description of the breach
2. Categories of personal information affected
3. General description of what occurred
4. Toll-free number or email for questions
5. Recommendation to monitor for fraud
6. Statement about freezing credit (if SSN affected)

**Example**:
```
Subject: Notice of Security Breach - ABC Company

[Company Name] wants to inform you that a security incident has
affected your personal information.

What Happened:
On [DATE], [Company Name] detected unauthorized access to its
customer database. Our investigation determined that the following
information may have been accessed: [list specific information]

What We're Doing:
We have immediately [describe containment and remediation actions].
We are working with [law enforcement / forensic firm] to investigate.

What You Should Do:
1. Monitor your credit for unauthorized activity
2. Change your password on our website
3. Enroll in credit monitoring (free for 24 months via [provider])
4. File a fraud report with FTC if you notice suspicious activity

Questions?
Contact us at [phone] or [email]

[Company Name]
```

---

## Part 3: US State Breach Notification Laws

### 3.1 State-by-State Requirements

Most US states have breach notification laws. Key variations:

**Timeline**:
- Most states: "Without unreasonable delay"
- Massachusetts, New Hampshire, New York: "Most expedient time possible"
- Nevada: "As quickly as possible"
- Generally interpreted as: Within 30-60 days

**Definition of "Personal Information"**:
- Core: Name + SSN, driver's license, financial account, credit card
- Extended: Medical record, health insurance number, biometric, genetic data
- Some states broader than others

**Exemption for Encryption**:
- Most states: No notification if data encrypted AND key not compromised
- New York: Encryption exemption specifically allowed
- Some states: May still require notification even if encrypted (vary)

**Who to Notify**:
- Affected residents
- Credit reporting agencies (if payment card data and >500 people)
- Media (if >1,000 residents affected)
- State Attorney General (varies by state)

**Key State Contacts**:

```
California (CCPA):
- Attorney General: https://oag.ca.gov/
- Deadline: 45 days

New York (SHIELD Act):
- Attorney General: https://ag.ny.gov/
- Deadline: "Most expedient time possible"
- Notify if name + financial/biometric data

Massachusetts:
- Attorney General: https://www.mass.gov/ago
- Deadline: "Most expedient time possible"
- Notify if name + SSN/financial account

Florida:
- Attorney General: https://www.myfloridalegal.com/
- Deadline: 30 days
- Notify media if >500 people

Texas:
- Attorney General: https://www.texasattorneygeneral.gov/
- Deadline: "Most expedient time possible"

Illinois (Biometric Privacy Act):
- Attorney General: https://www.cyberdriveillinois.com/
- Deadline: 30 days (for biometric data)
```

### 3.2 Multi-State Breach Notification Procedure

**If Breach Affects Residents in Multiple States**:

```
State Compliance Matrix:

State         | Definition    | Timeline  | Who to Notify
─────────────────────────────────────────────────────────
California    | Broad         | 45 days   | CA AG if >500
New York      | Broad         | ASAP      | NY AG always
Massachusetts | Name + SSN    | ASAP      | MA AG if >500
Florida       | Name + Data   | 30 days   | Media if >500
Texas         | Name + Data   | ASAP      | TX AG if >500
```

**Compliance Strategy**:
1. Use MOST STRINGENT requirement for all states
2. If California applies: 45-day deadline (most stringent)
3. If New York applies: Use CCPA standard (broad definition)
4. Notify all applicable state AGs
5. Document compliance with each state's requirements

---

## Part 4: Breach Notification Process Flowchart

```
                    ┌─────────────────────┐
                    │ Breach Detected     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Legal Assessment    │
                    │ (Is PII affected?)  │
                    └──────────┬──────────┘
                               │
                      ┌────────┴────────┐
                      │                 │
                   NO │                 │ YES
                      │                 │
              ┌───────▼────────┐   ┌────▼────────────┐
              │ No Notification│   │ Risk Assessment │
              │ (Document)     │   │ (High Risk?)    │
              └────────────────┘   └────┬────────────┘
                                        │
                                ┌───────┴────────┐
                                │                │
                            LOW │                │ HIGH
                                │                │
                        ┌───────▼──────┐  ┌──────▼──────────┐
                        │ Notify DPA   │  │ Notify DPA +    │
                        │ Only (72h)   │  │ Individuals     │
                        │              │  │ (No delay)      │
                        │ No individual│  │                 │
                        │ notification │  │ Individuals 24h │
                        │ required     │  │                 │
                        └──────────────┘  └────┬────────────┘
                                               │
                                        ┌──────▼────────┐
                                        │ Multi-State?  │
                                        └──────┬────────┘
                                               │
                                    ┌──────────┴──────────┐
                                    │                     │
                                  NO│                     │YES
                                    │                     │
                           ┌────────▼────┐     ┌─────────▼────────┐
                           │ Notify Only │     │ Notify All State │
                           │ Affected    │     │ AGs + Most       │
                           │ State AGs   │     │ Stringent Rules  │
                           └────────┬────┘     └─────────┬────────┘
                                    │                    │
                                    └────────┬───────────┘
                                             │
                                    ┌────────▼──────────┐
                                    │ Media Notification │
                                    │ (if >500 in state) │
                                    └────────────────────┘
```

---

## Part 5: Breach Notification Checklist

### 5.1 Immediate Actions (First 4 Hours)

```
WITHIN 30 MINUTES:
[ ] Legal counsel engaged
[ ] Confirm breach involves personal data
[ ] Confirm affected individuals in US/EU

WITHIN 1 HOUR:
[ ] Assess "high risk" to rights/freedoms
[ ] Determine GDPR applies (if EU residents)
[ ] Determine CCPA/state laws apply (if US residents)
[ ] Assign notification responsibility
[ ] Begin drafting notifications

WITHIN 2 HOURS:
[ ] Get list of affected individuals (count and categories)
[ ] Determine notification method (email, mail, website)
[ ] Draft notification emails/letters
[ ] Obtain quotes for credit monitoring (if needed)
[ ] Research state AG contact information

WITHIN 4 HOURS:
[ ] Legal review of all notifications
[ ] Finalize notification templates
[ ] Prepare DPA notification (if GDPR)
[ ] Prepare state AG notifications
```

### 5.2 Notification Execution (4-24 Hours)

```
WITHIN 12 HOURS:
[ ] Submit DPA notification (if GDPR) - start countdown for 72h
[ ] Schedule notification emails to individuals
[ ] Send press release (if media notification required)
[ ] Brief crisis team on messaging

WITHIN 24 HOURS:
[ ] Send notifications to all affected individuals
[ ] Send notifications to state AGs
[ ] Set up response team for incoming questions
[ ] Monitor social media / news for reaction
[ ] Prepare executive briefing
[ ] Document all actions taken + timestamps

WITHIN 48 HOURS:
[ ] Confirm all notifications were sent successfully
[ ] Begin monitoring for follow-up questions
[ ] Post FAQ on website
[ ] Update status in incident tracking system
```

### 5.3 Follow-Up Actions (Days 3-72)

```
DAY 3:
[ ] Conduct first media monitoring
[ ] Brief executive team on response
[ ] Review social media sentiment
[ ] Prepare for customer call-in volume

DAYS 3-7:
[ ] Respond to inquiries from individuals
[ ] Provide credit monitoring enrollment details
[ ] Monitor for any secondary issues
[ ] Prepare for regulatory inquiries

DAY 30:
[ ] Provide update to regulatory bodies (if requested)
[ ] Summarize lessons learned
[ ] Update incident response procedures (if needed)

DAY 72:
[ ] Ensure all GDPR DPA notifications completed
[ ] Verify all CCPA notifications sent
[ ] Confirm all state AG notifications delivered
[ ] Archive all notification records
[ ] Begin post-mortem process
```

---

## Part 6: Communication Templates

See separate template files:
- `/home/user/hello-world/templates/internal_incident_notification.md`
- `/home/user/hello-world/templates/customer_notification_email.md`
- `/home/user/hello-world/templates/regulatory_notification_template.md`
- `/home/user/hello-world/templates/press_release_template.md`

---

## Part 7: Compliance Documentation

### 7.1 Notification Record Keeping

**What to Document and Keep for 3+ Years**:

```
For Each Breach:
┌──────────────────────────────────────┐
│ Breach Documentation                 │
├──────────────────────────────────────┤
│ • Date/time breach detected          │
│ • Date/time breach confirmed         │
│ • Initial assessment (data types)    │
│ • Risk assessment (high/low)         │
│ • Individuals affected (count)       │
│ • Notification method used           │
│ • Date notifications sent            │
│ • Copies of notifications sent       │
│ • DPA notification (if GDPR)         │
│ • State AG notifications             │
│ • Media notifications                │
│ • Customer feedback received         │
│ • Regulatory inquiries               │
│ • Resolution/remediation steps       │
│ • Legal opinions/advice              │
│ • Insurance claim (if filed)         │
└──────────────────────────────────────┘
```

### 7.2 Regulatory Inquiry Response

**If Regulatory Body Requests Information**:

1. **Do Not Ignore** - Must respond within specified deadline
2. **Consult Legal** - Before providing any information
3. **Provide Truthfully** - Lying to regulators is a crime
4. **Document Everything** - Record what was requested and what was provided
5. **Keep Copies** - Retain all correspondence

**Response Timeline**:
- GDPR DPA inquiry: Typically 30 days
- CCPA AG inquiry: Typically 45 days
- State AG inquiry: Varies, typically 30-60 days

---

## Part 8: Crisis Communication Plan

### 8.1 Message Hierarchy

**Primary Message** (same for all communications):
- "We take security seriously"
- "We notified affected individuals"
- "We are investigating thoroughly"
- "We are implementing safeguards"

**For Affected Individuals**:
- What happened (clear, honest)
- What data was affected (specific)
- What we're doing about it
- What they should do
- How to contact us

**For Regulators**:
- Timeline of events
- Technical details
- Scope of breach
- Remediation actions
- Root cause analysis
- Preventive measures

**For Media**:
- Incident was contained quickly
- No ongoing threat
- Customers have been notified
- Preventive measures implemented
- Commitment to security

### 8.2 Crisis Communication Team

**Roles & Responsibilities**:

| Role | Responsibility | On-Call | Timeline |
|------|-----------------|---------|----------|
| Incident Commander | Overall coordination | 24/7 | 5 min response |
| Legal Counsel | Regulatory compliance | 24/7 | 1 hour response |
| Communications Lead | External messaging | Business hours | 2 hour response |
| PR Director | Media relations | Business hours | 2 hour response |
| Customer Success | Customer support | 24/7 | Immediate |
| Security Lead | Technical details | 24/7 | 30 min response |

---

## Part 9: Recurring Notifications

### 9.1 Annual Notice to Individuals

For breaches where individuals were notified:
- Send reminder email one year later
- Remind about credit monitoring if still available
- Provide update on security improvements
- Request feedback on response

### 9.2 Regulatory Reporting

**GDPR Article 34(4)**: DPA must make breach public list available (some DPAs publish)
- Monitor GDPR DPA websites for published breaches
- Understand public perception
- Prepare response if media picks up story

---

## Compliance Checklist

```
GDPR Compliance:
[ ] Determine personal data affected
[ ] Assess risk to rights/freedoms
[ ] Notify DPA within 72 hours (Article 33)
[ ] Notify individuals without delay (Article 34)
[ ] Provide specific information (categories, consequences, measures)
[ ] Document decision process
[ ] Retain records for 3+ years

CCPA Compliance (California):
[ ] Determine unencrypted personal information affected
[ ] Assess if >500 people affected (determines AG notification)
[ ] Notify individuals within 45 days
[ ] Notify CA AG if >500 (free service: https://oag.ca.gov/)
[ ] Provide specific information (categories, actions)
[ ] Retain records for 3 years

US State Laws:
[ ] Identify all states where residents affected
[ ] Apply most stringent state requirements
[ ] Notify all applicable state AGs
[ ] Use state-specific notification templates
[ ] Retain records by state requirements (typically 3-7 years)

Credit Reporting (if >500 + payment card):
[ ] Notify credit reporting agencies (Equifax, Experian, TransUnion)
[ ] Provide breach information for fraud alerts
[ ] Recommend credit freeze for affected individuals

Insurance:
[ ] Notify cyber liability insurer immediately
[ ] File claim if applicable
[ ] Cooperate with insurer's investigation
[ ] Keep insurer updated on remediation
```

---

**Document Status**: ✅ Complete and ready for production
**Last Review**: 2025-11-16
**Next Review**: 2026-05-16 (annual update for legal/regulatory changes)
**Owner**: General Counsel + Chief Information Security Officer
**Approval**: [Signature required before deployment]
