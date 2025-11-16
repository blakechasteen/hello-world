# Regulatory Notification Templates

---

## GDPR Article 33: Supervisory Authority Notification (DPA)

**Recipient**: Data Protection Authority / Supervisory Authority (DPA)
**Deadline**: 72 hours from breach awareness
**Method**: Official DPA submission portal (varies by country)
**Audience**: Data Protection Officer, GDPR supervisory authorities

---

### Email Transmittal

```
To: [DPA Contact Email]
Subject: GDPR Article 33 Notification - Data Breach INC-2025-BREACH-0001

Body:
─────────────────────────────────────────────────────────────────

Dear [DPA Name] Supervisory Authority,

We are submitting a mandatory breach notification pursuant to Article 33
of the General Data Protection Regulation (GDPR).

BREACH NOTIFICATION DETAILS
────────────────────────────

Reporting Organization:
  Name: HoloLoom, Inc.
  Address: [Company Address]
  Country: [Country]
  Data Protection Officer: [DPO Name] (dpo@hololoom.ai)

Supervisory Authority:
  [DPA Name] of [Country]
  Contact: [DPA Email/Phone]

BREACH DETAILS
─────────────

Date of Breach Awareness: November 16, 2025, 14:30 UTC
Incident ID: INC-2025-BREACH-0001
Nature of Breach: Unauthorized access to customer database via SQL injection

Date of First Evidence: November 16, 2025, 02:15 UTC (estimated)
Discovery Method: Security monitoring system alert (anomalous database activity)

Attack Vector:
  - Phishing email to employee (credential theft)
  - SQL injection vulnerability in web application
  - Unauthorized database access and data exfiltration

Duration of Unauthorized Access: Approximately 12 hours (02:15 to 14:30 UTC)

AFFECTED DATA SUBJECTS
──────────────────────

Total Number Affected: Approximately 15,000 EU residents (out of 200,000 total users)

Categories of Data Subjects:
  ✓ Individual customers (15,000 personal accounts)
  ✗ Employees (not affected)
  ✗ Business customers (separate database, not affected)

CATEGORIES OF PERSONAL DATA AFFECTED
──────────────────────────────────────

The following categories of personal data were affected:

1. Name (Full Name) - SENSITIVE IDENTIFIER
   └─ Number affected: 15,000
   └─ Risk: Identity theft, phishing, social engineering

2. Email Address - CONTACT INFORMATION
   └─ Number affected: 15,000
   └─ Risk: Phishing, unauthorized access, spam

3. Telephone Number (Mobile/Landline) - CONTACT INFORMATION
   └─ Number affected: 7,000
   └─ Risk: Social engineering, harassment, SIM swap attacks

4. Password Hash (Bcrypt) - SECURITY CREDENTIAL
   └─ Number affected: 15,000
   └─ Risk: Low (hashed with strong algorithm, not plaintext)
   └─ Note: Bcrypt hashes require enormous computing power to crack

5. Account Creation Date - METADATA
   └─ Number affected: 15,000
   └─ Risk: Low (not sensitive on its own)

NOT AFFECTED (Data We Do Not Store):
  ✗ Social Security Numbers / Tax IDs
  ✗ Government ID Numbers (Passport, Driver License)
  ✗ Payment Card Information (Not stored per PCI-DSS)
  ✗ Health / Medical Information
  ✗ Biometric Data
  ✗ Financial Account Information (Bank accounts, etc)

LIKELY CONSEQUENCES FOR DATA SUBJECTS
──────────────────────────────────────

The following consequences are likely to result from this breach:

1. PHISHING RISK (HIGH PROBABILITY)
   └─ Attacker has names + emails
   └─ Can craft targeted phishing attacks
   └─ Could lead to account takeover, credential theft

2. IDENTITY THEFT RISK (MEDIUM PROBABILITY)
   └─ Attacker has name + email + phone
   └─ Could potentially open fraudulent accounts
   └─ Risk reduced because no SSN/financial data compromised

3. SOCIAL ENGINEERING RISK (MEDIUM PROBABILITY)
   └─ Attacker has sufficient information to call/text pretending to be support
   └─ Could trick users into revealing additional information
   └─ Could lead to account compromise

4. FINANCIAL FRAUD (LOW PROBABILITY)
   └─ No credit card or bank account information compromised
   └─ Risk is primarily phishing → credential theft → financial fraud
   └─ Not direct financial theft

5. REPUTATIONAL HARM (HIGH PROBABILITY)
   └─ Users will lose trust in our organization
   └─ May switch to competitors
   └─ Media coverage of breach

MEASURES TAKEN
────────────────

We have taken the following immediate measures:

Technical Containment:
  ✓ Attacker IP (203.0.113.45) blocked at firewall (within 45 minutes)
  ✓ Compromised employee account revoked
  ✓ SQL injection vulnerability patched
  ✓ Database activity logging enabled
  ✓ Network traffic to attacker FTP server blocked

Investigation:
  ✓ Third-party forensics firm engaged (Mandiant)
  ✓ Complete forensic image captured
  ✓ Exfiltration destination identified and blocked
  ✓ Dwell time determined (12 hours)
  ✓ Root cause analysis completed

Prevention:
  ✓ Parameterized queries deployed (SQL injection prevention)
  ✓ Multi-factor authentication (MFA) mandated for admin accounts
  ✓ Vulnerability scanning enhanced
  ✓ Database monitoring and alerting implemented

Data Subject Notification:
  ✓ Customer notification emails prepared (compliant with Article 34)
  ✓ Notifications to be sent November 17, 2025 (within 24 hours)
  ✓ Credit monitoring service offered (24 months, free)

Compliance & Reporting:
  ✓ This notification submitted within 72-hour requirement
  ✓ Affected data subjects notified
  ✓ All documentation preserved for investigation

MEASURES PROPOSED
───────────────────

To prevent similar breaches in future, we are implementing:

Short-term (Next 30 days):
  - MFA enforcement for all users (currently optional)
  - Enhanced email security (advanced threat protection)
  - Database activity monitoring (real-time alerts)
  - Increased staff security training (phishing simulations monthly)

Medium-term (Next 90 days):
  - Data loss prevention (DLP) system deployment
  - Network segmentation (limit lateral movement)
  - Privileged access management (PAM) system
  - Third-party security audit (annual)

Long-term (Next 6-12 months):
  - ISO 27001 certification
  - Zero-trust architecture implementation
  - Incident response plan improvements
  - Quarterly tabletop exercises

CONTACT INFORMATION
────────────────────

For questions regarding this notification, please contact:

Data Protection Officer:
  Name: [DPO Name]
  Email: dpo@hololoom.ai
  Phone: [Phone]
  Available: 24/7 for breach-related inquiries

Incident Commander:
  Name: [IC Name]
  Email: [IC Email]
  Phone: [IC Phone]

Legal Counsel:
  Name: [Counsel Name]
  Email: [Counsel Email]
  Phone: [Counsel Phone]

We are available to provide additional information as requested.

ATTACHMENTS
────────────

Attached to this notification:

1. Complete forensic analysis report (Mandiant)
2. Timeline of incident (detailed)
3. Database dump of affected records (encrypted, password protected)
4. Customer notification email (for transparency)
5. Breach impact assessment (detailed)
6. System architecture diagram (showing compromised systems)
7. Chain of custody documentation (evidence handling)

All attachments are marked CONFIDENTIAL - ATTORNEY-CLIENT PRIVILEGED
and are provided under protective order for investigation purposes only.

LEGAL DECLARATION
───────────────────

I declare under penalty of law that the information provided in this
notification is true, accurate, and complete to the best of my knowledge.

I am authorized to sign this notification on behalf of HoloLoom, Inc.

_________________________                    ___________________
[DPO Name]                                  [Date]
Data Protection Officer
HoloLoom, Inc.

────────────────────────────────────────────────────────────────

Incident ID: INC-2025-BREACH-0001
Notification Date: November 16, 2025, 16:00 UTC
Deadline: November 19, 2025, 14:30 UTC (72 hours)
Status: Submitted on time
```

---

## CCPA: California Attorney General Notification

**Recipient**: California Attorney General (if >500 CA residents affected)
**Deadline**: Same time as consumer notification (45 days)
**Method**: https://oag.ca.gov/privacy/databreach/reporting
**Audience**: State Attorney General

---

### Submission Portal Information

```
CALIFORNIA ATTORNEY GENERAL
Data Breach Notification Submission

Website: https://oag.ca.gov/privacy/databreach/reporting
Email (alternative): databreach@doj.ca.gov
Phone: [Phone number if available]

REQUIRED INFORMATION
────────────────────

Your Information:
  Organization Name: HoloLoom, Inc.
  Contact Person: [DPO Name]
  Contact Email: dpo@hololoom.ai
  Contact Phone: [Phone]
  Address: [Company Address], California

Breach Details:
  Date of Breach: November 16, 2025
  Date Discovered: November 16, 2025
  Incident ID: INC-2025-BREACH-0001

Number of California Residents Affected: [Number of CA residents from 15,000 total]

Type(s) of Personal Information Compromised:
  ✓ Name
  ✓ Email address
  ✓ Phone number
  ✓ Encrypted/hashed password

Description of Breach:
  On November 16, 2025, HoloLoom, Inc. detected unauthorized access to its
  customer database. An attacker used a phishing-obtained credential and SQL
  injection vulnerability to access customer information. Approximately [X]
  California residents were affected. The breach was contained within 12 hours,
  and the vulnerability has been patched.

Corrective Actions Taken:
  - Attacker blocked and traced to law enforcement
  - SQL injection vulnerability patched
  - Multi-factor authentication mandated for admin accounts
  - Enhanced monitoring deployed
  - Forensics investigation underway
  - Consumer notification in progress

Notification Method:
  Email notification sent to affected consumers on November 17, 2025

Encryption/Redaction Details:
  Passwords were stored using Bcrypt hashing (not plaintext), reducing but not
  eliminating risk. Other information (name, email, phone) was not encrypted
  but was in a secured database.

---

### Supporting Documentation

Attach or reference:
- Customer notification email (proof of notification)
- Timeline of breach
- Forensic investigation summary
- List of affected CA residents (encrypted/hashed)
- Proof of security improvements
```

---

## US State Attorney General Notification (Multi-State Template)

**Recipient**: State Attorneys General (all affected states)
**Deadline**: Varies by state (typically 30-60 days, recommend 45 days)
**Method**: Official state AG submission process
**Audience**: State law enforcement

---

### Multi-State Submission Template

```
BREACH NOTIFICATION TO STATE ATTORNEYS GENERAL

Submitted to:
  - California Attorney General
  - New York Attorney General
  - Massachusetts Attorney General
  - [Additional affected states]

ORGANIZATION INFORMATION
─────────────────────────

Company Name: HoloLoom, Inc.
HQ Address: [Company Address]
Industry: Software / AI Services
Number of Employees: [Number]
Annual Revenue: [Revenue]

Contact Person: [DPO Name]
Contact Email: dpo@hololoom.ai
Contact Phone: [Phone]

BREACH INFORMATION
───────────────────

Date of Breach: November 16, 2025, 02:15 UTC (estimated)
Date Discovered: November 16, 2025, 14:30 UTC
Incident ID: INC-2025-BREACH-0001

Cause of Breach:
  1. Phishing attack targeting employee (credential theft)
  2. SQL injection vulnerability in web application
  3. Unauthorized database access and data exfiltration

Scope of Breach:
  Total Affected Nationally: 15,000 customers
  [State] Residents Affected: [Number]
  Percentage of Total: [Percentage]%

AFFECTED DATA SUBJECTS
──────────────────────

Types of Residents Affected:
  - Individual account holders age 18+
  - No children's data affected (COPPA N/A)
  - No employee data affected

Categories of Personal Information:
  ✓ Full Name
  ✓ Email Address
  ✓ Phone Number
  ✓ Hashed Password (Bcrypt)
  ✗ NOT AFFECTED: SSN, Government ID, Financial Info, Health Info

NOTIFICATION PLAN
──────────────────

Timeline:
  - November 16, 2025: Breach detected and contained
  - November 17, 2025: Consumer notifications sent via email
  - November 18, 2025: FAQ published on website
  - November 19, 2025: Press release issued (proactive disclosure)
  - By December 17, 2025: Attorney General notifications completed

Method of Notification:
  - Email to registered account email address
  - In-app notification within customer dashboard
  - Website security incident page with FAQ
  - Phone support for customers with questions
  - Credit monitoring service offered (24 months, free)

Consumer Support:
  - 24/7 incident helpline: 1-800-HOLOLOOM ext. SECURITY
  - Email support: security@hololoom.ai
  - Website FAQ: https://www.hololoom.ai/security-incident-faq
  - Credit monitoring enrollment: [Link]

SECURITY IMPROVEMENTS
──────────────────────

Immediate (Within 30 days):
  ✓ SQL injection vulnerability patched
  ✓ Multi-factor authentication mandated for admin
  ✓ Enhanced email security deployed
  ✓ Database monitoring enabled

Short-term (Within 90 days):
  ✓ Data loss prevention (DLP) system implemented
  ✓ Database activity monitoring deployed
  ✓ Security training program enhanced
  ✓ Network segmentation implemented

Long-term (Within 6-12 months):
  ✓ ISO 27001 certification pursued
  ✓ Zero-trust architecture implementation
  ✓ Annual third-party security audit
  ✓ Incident response drills (quarterly)

REGULATORY COMPLIANCE
──────────────────────

This breach notification complies with:
  ✓ [State] Breach Notification Law (within required timeline)
  ✓ GDPR Article 33-34 (EU residents notified)
  ✓ CCPA Article 1798.150 (CA residents notified)
  ✓ Federal standards for notification

No encryption exemption claimed (data was not encrypted).

CONTACT FOR INQUIRIES
───────────────────────

Data Protection Officer:
  Name: [DPO Name]
  Email: dpo@hololoom.ai
  Phone: [Phone]
  Available: 24/7 for Attorney General inquiries

Attorney General liaison:
  Name: [Outside Counsel Name]
  Firm: [Law Firm]
  Email: [Email]
  Phone: [Phone]

We are available to provide additional information or documentation as
requested by your office.

────────────────────────────────────────────────────────────────

Incident ID: INC-2025-BREACH-0001
Notification Status: Submitted
Submission Date: [Date]
Compliance Status: ✅ ON TIME
```

---

**Status**: ✅ Production Ready (2025-11-16)
**Compliance**: GDPR Article 33, CCPA §1798.150, US State Laws
**Last Updated**: 2025-11-16
**Owner**: General Counsel + Data Protection Officer
