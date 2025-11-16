# Customer Notification Email Templates

---

## Standard Breach Notification Email (GDPR/CCPA Compliant)

**Subject**: Important Security Notice - HoloLoom Data Protection Incident

```
Dear [Customer Name],

We are writing to inform you of a data security incident that may affect your account
with HoloLoom.

WHAT HAPPENED
──────────────

On November 16, 2025, at approximately 14:30 UTC, we detected unauthorized access to
our customer database. Our investigation confirmed that a third party gained access to
certain customer information without authorization.

We have no evidence that you experienced any fraudulent activity as a result of this
incident, but we want to ensure you are informed and aware of steps you can take to
protect yourself.

WHAT DATA WAS AFFECTED
──────────────────────

Our investigation determined that the following information about your account may
have been affected by this incident:

✓ Your name
✓ Your email address
✓ Your phone number
✓ A hashed version of your password (NOT in plaintext)
✓ Your account creation date

The following information was NOT affected:
✗ Payment card information (we do not store this)
✗ Social Security number (we do not store this)
✗ Date of birth (we do not store this)
✗ Government-issued ID numbers (we do not store this)

OUR IMMEDIATE ACTIONS
──────────────────────

Upon discovering this incident, we immediately took the following steps:

1. Revoked the attacker's access to our systems
2. Blocked the attacker's IP address at our firewall
3. Secured our systems and halted the attack
4. Engaged a leading cybersecurity forensics firm to investigate
5. Identified and patched the vulnerability that was exploited
6. Notified law enforcement and regulatory authorities

WHAT YOU SHOULD DO
────────────────────

To protect your account and information, we recommend you take the following actions:

1. CHANGE YOUR PASSWORD IMMEDIATELY
   - Visit: https://www.hololoom.ai/account/security
   - Use a strong, unique password you haven't used elsewhere
   - Use a mix of uppercase, lowercase, numbers, and symbols
   - Minimum 12 characters recommended

2. ENABLE TWO-FACTOR AUTHENTICATION
   - We now offer 2FA on all accounts (strongly recommended)
   - Login to your account and enable 2FA in Security Settings
   - This significantly reduces account takeover risk

3. MONITOR YOUR ACCOUNTS
   - Watch for suspicious email activity
   - Monitor your email for phishing attempts
   - Be cautious of unsolicited emails claiming to be from HoloLoom
   - Report suspicious emails to security@hololoom.ai

4. ENROLL IN FREE CREDIT MONITORING
   - We are providing 24 months of free credit monitoring
   - Enrollment details: See below

5. CONSIDER ADDITIONAL PROTECTIONS
   - Place a credit freeze with the three credit bureaus
   - Monitor your credit report for suspicious accounts
   - File a police report if you experience identity theft
   - File a report with the FTC: https://reportidentitytheft.ftc.gov/

FREE CREDIT MONITORING
──────────────────────

To help protect you, we are offering 24 months of complimentary credit monitoring
and identity protection services through [Provider Name].

Enrollment Details:
  - Website: [enrollment URL]
  - Enrollment Code: [unique code per customer]
  - Phone: 1-800-[PROVIDER]-1 (available 24/7)
  - Support Email: support@[provider].com

How to Enroll (Takes 5 minutes):
  1. Visit [enrollment URL]
  2. Enter your enrollment code: [unique code]
  3. Create your account
  4. Enroll in monitoring services
  5. You're protected!

Your enrollment code is unique and available for 90 days. After 90 days, we will
enroll you automatically if you have not already done so.

What You'll Get:
  ✓ Credit monitoring (alerts if new accounts opened in your name)
  ✓ Identity theft insurance ($1M coverage)
  ✓ Fraud resolution services (help if you're a victim)
  ✓ Dark web monitoring (alerts if your data appears in criminal databases)

ABOUT OUR INVESTIGATION
───────────────────────

We have thoroughly investigated this incident and determined:

How the Attacker Got In:
  The attacker used a phishing email to trick one of our employees into
  providing their password. The attacker then used that access to exploit
  a SQL injection vulnerability in our website. We have since:

  ✓ Patched the vulnerability
  ✓ Deployed MFA to prevent similar attacks
  ✓ Enhanced our email security
  ✓ Provided additional security training to staff

Complete Timeline:
  - November 15, 22:30 UTC: Phishing email sent
  - November 16, 02:15 UTC: Attacker gains access
  - November 16, 02:30-06:00 UTC: Attacker downloads customer data
  - November 16, 14:30 UTC: We detect the attack via our security monitoring
  - November 16, 14:35 UTC: We declare incident and begin response
  - November 16, 15:30 UTC: Attacker completely blocked from all systems
  - November 16, 16:30 UTC: Vulnerability patched in production

Dwell Time: Approximately 12 hours from initial compromise to detection. We are
investing in additional monitoring to reduce this in the future.

Data Exfiltration: The attacker did download a copy of customer data. However:
  ✓ The data was encrypted during transfer
  ✓ Payment information was NOT included
  ✓ Passwords were hashed (not plaintext)
  ✓ We are monitoring dark web for data sales
  ✓ We will notify you if your data appears publicly

FREQUENTLY ASKED QUESTIONS
────────────────────────────

Q: Will my credit card be charged for the credit monitoring?
A: No. The monitoring is completely free for 24 months. After 24 months,
   you can choose to continue or discontinue the service.

Q: What if I don't enroll in credit monitoring?
A: Credit monitoring is optional, but highly recommended. Even if you don't
   enroll, please change your password and enable 2FA.

Q: Will there be legal action against the attacker?
A: We have reported this to law enforcement (FBI). Investigation is underway.
   We cannot comment on active law enforcement investigations.

Q: Is my account still secure?
A: Yes. The vulnerability has been patched, the attacker is completely blocked,
   and we have implemented additional safeguards. Your account is now more
   secure than before.

Q: What is your plan to prevent this from happening again?
A: We are implementing:
   ✓ Multi-factor authentication (MFA) on all admin accounts
   ✓ Advanced email security to prevent phishing
   ✓ Database activity monitoring (detect suspicious queries)
   ✓ Endpoint detection and response (EDR) on all systems
   ✓ Regular security audits by external firms

Q: Should I delete my account?
A: No, this is not necessary. Your account is secure, and we have significantly
   improved our security. We hope to continue serving you.

Q: How can I report additional concerns?
A: Please email security@hololoom.ai or call our security hotline at
   1-800-HOLOLOOM ext. SECURITY (available 24/7).

OUR COMMITMENT TO YOU
──────────────────────

We take your privacy and security seriously. This incident represents a failure
of our defenses, and we take full responsibility. We are:

✓ Investing heavily in security improvements
✓ Being transparent about what happened
✓ Offering free protections (credit monitoring)
✓ Cooperating fully with law enforcement
✓ Notifying regulators as required by law
✓ Committing to prevent future incidents

We understand that this incident may have affected your trust in us. We are
committed to rebuilding that trust through actions, not just words.

CONTACT INFORMATION
─────────────────────

For questions about this incident:
  Email: security@hololoom.ai
  Phone: 1-800-HOLOLOOM ext. SECURITY (24/7)
  Website: https://www.hololoom.ai/security-incident
  FAQ: https://www.hololoom.ai/security-incident-faq

For credit monitoring enrollment support:
  Email: support@[provider].com
  Phone: 1-800-[PROVIDER]-1 (24/7)

For identity theft concerns:
  File a report: https://reportidentitytheft.ftc.gov/
  Police report: Contact your local police department
  FTC Guidance: https://www.ftc.gov/news-events/news/2022/06/how-recognize-and-report-identity-theft

REGULATORY NOTIFICATIONS
──────────────────────────

We are notifying applicable regulatory authorities as required by law, including:
  • GDPR Supervisory Authorities (EU)
  • State Attorneys General (affected states)
  • California Attorney General (if applicable)

You can expect to receive official notifications from these authorities if you are
a resident in their jurisdiction.

CLOSING
────────

We sincerely apologize for this incident and the concern it may cause. We are
committed to the highest standards of security and privacy. Thank you for your
continued trust in HoloLoom.

If you have any questions, please do not hesitate to contact us.

Sincerely,

[CEO Name]
Chief Executive Officer
HoloLoom

[Date]
November 16, 2025

────────────────────────────────────────────────────────────────

Incident ID: INC-2025-BREACH-0001
Notice Type: Data Security Incident Notification
Affected Jurisdiction(s): [List of states/countries]
```

---

## Alternative: High-Risk Breach Email (with SSN/Credit Card Compromised)

**Subject**: URGENT: Critical Security Alert - Immediate Action Required

```
Dear [Customer Name],

We must inform you of a serious data security breach affecting your account
and recommend you take immediate action to protect yourself.

⚠️ CRITICAL INFORMATION YOU NEED TO KNOW
──────────────────────────────────────────

This breach is more serious than typical data incidents because:

✗ Your Social Security Number (SSN) was compromised
✗ Your credit card information was compromised
✗ Your identity information was stolen

IMMEDIATE ACTIONS (DO THIS TODAY)
──────────────────────────────────

1. PLACE A CREDIT FREEZE (URGENT - Call today):
   - Equifax: 1-800-349-9960
   - Experian: 1-888-739-0349
   - TransUnion: 1-888-909-8872

   A credit freeze prevents anyone (including you) from opening new credit
   accounts using your SSN. This is the best protection against identity theft.

2. PLACE A FRAUD ALERT (within 1 hour):
   - Call any one of the three bureaus above
   - They will notify the other two
   - Fraud alert requires creditors to verify your identity before opening accounts
   - Fraud alert lasts 1 year

3. CHECK YOUR CREDIT REPORTS (within 1 week):
   - Visit: https://www.annualcreditreport.com (free, official)
   - Review for unauthorized accounts
   - Dispute any accounts you don't recognize
   - Keep copies of disputes

4. MONITOR YOUR FINANCIAL ACCOUNTS (daily for next 30 days):
   - Check your bank accounts for unauthorized transactions
   - Check your credit card statements for unauthorized charges
   - Report any unauthorized activity to your bank immediately

5. FILE A POLICE REPORT (recommended):
   - File a report with your local police department
   - Get a copy of the report (proof of identity theft)
   - Report to FTC: https://reportidentitytheft.ftc.gov/

[Rest of email same as standard template above]
```

---

## Low-Risk Breach Email (Name/Email Only)

**Subject**: Security Notice - Hashed Password Reset Recommended

```
Dear [Customer Name],

We are writing to inform you of a security incident. Out of an abundance of caution,
we recommend some steps to further protect your account.

WHAT HAPPENED
──────────────

On November 16, 2025, we detected unauthorized access to a portion of our database.
Our investigation determined that the following information may have been exposed:

✓ Your name
✓ Your email address
✓ Your hashed password (encrypted one-way, cannot be read by attackers)

The following was NOT exposed:
✗ Payment information (we don't store this)
✗ Social Security number (we don't store this)
✗ Other personal information (we don't store this)

WHAT YOU SHOULD DO
────────────────────

We recommend taking two simple steps:

1. CHANGE YOUR PASSWORD
   - Visit: https://www.hololoom.ai/account/security
   - Create a new strong password

2. ENABLE TWO-FACTOR AUTHENTICATION (2FA)
   - Visit your account security settings
   - Enable 2FA for additional protection

DETAILS
─────────

✓ Vulnerability has been patched
✓ Attacker has been completely blocked
✓ No evidence of fraudulent activity
✓ Your account security is now stronger than before

We are not offering credit monitoring for this incident (SSN not exposed), but
we encourage you to monitor your account for any suspicious activity.

If you have questions: security@hololoom.ai

Thank you for your continued trust.

[CEO Name]
HoloLoom
```

---

**Status**: ✅ Production Ready (2025-11-16)
**Compliance**: GDPR Article 34, CCPA, US State Breach Notification Laws
**Last Updated**: 2025-11-16
**Owner**: Communications Lead + Legal Counsel
