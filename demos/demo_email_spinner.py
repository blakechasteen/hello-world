"""
Email Spinner Demo
==================

Demonstrates the EmailSpinner for ingesting email messages.

Features shown:
- IMAP mailbox connection
- Email header extraction
- Thread/reply detection
- Attachment tracking
- Importance scoring

Author: Claude Code
Date: November 2025
"""

import asyncio
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.spinningWheel.email_spinner import EmailSpinner, EmailMessage


async def demo_email_spinner():
    """Run Email spinner demo."""
    print("=" * 70)
    print("Email Spinner Demo")
    print("=" * 70)

    # Configuration
    imap_server = "imap.gmail.com"  # Gmail example
    email_address = "your-email@gmail.com"
    password = "your-app-password"  # Use app-specific password for Gmail

    spinner = EmailSpinner(
        imap_server=imap_server,
        email_address=email_address,
        password=password,
        days_back=30,
        max_messages=50
    )

    # Check availability
    if not spinner.is_available():
        print("Error: imaplib not available")
        return

    print("\nSpinner capabilities:")
    caps = spinner.get_capabilities()
    print(f"  - Basic processing: {caps.basic_processing}")
    print(f"  - Importance scoring: {caps.importance_scoring}")
    print(f"  - Entity extraction: {caps.entity_extraction}")
    print(f"  - Motif extraction: {caps.motif_extraction}")

    print("\nConfiguration:")
    print(f"  - IMAP Server: {imap_server}")
    print(f"  - Email: {email_address}")
    print(f"  - Days back: {spinner.days_back}")
    print(f"  - Max messages: {spinner.max_messages}")

    print("\nNote: To use with real Gmail account:")
    print("  1. Enable 2FA on your Google account")
    print("  2. Generate app-specific password: myaccount.google.com/apppasswords")
    print("  3. Use app password (not regular password)")
    print("  4. Call: await spinner.spin('INBOX')")

    print("\nSupported mailboxes:")
    print("  - INBOX")
    print("  - [Gmail]/Sent Mail")
    print("  - [Gmail]/Drafts")
    print("  - [Gmail]/All Mail")
    print("  - [Gmail]/Trash")
    print("  - Custom folders")

    print("\nExample usage:")
    print("  result = await spinner.spin(mailbox='INBOX')")
    print("  # or specific mailbox")
    print("  result = await spinner.spin(mailbox='[Gmail]/All Mail')")


async def demo_importance_scoring():
    """Demonstrate email importance scoring."""
    print("\n" + "=" * 70)
    print("Email Importance Scoring Demo")
    print("=" * 70)

    spinner = EmailSpinner(
        imap_server="imap.gmail.com",
        email_address="test@example.com",
        password="password"
    )

    # Example emails
    emails = [
        EmailMessage(
            message_id="<1@example.com>",
            subject="URGENT: Critical Production Issue",
            sender="manager@company.com",
            sender_name="Manager",
            recipients=["you@company.com"],
            body="Action required immediately! " * 50,
            attachments=["incident_report.pdf", "logs.zip"]
        ),
        EmailMessage(
            message_id="<2@example.com>",
            subject="Meeting notes",
            sender="colleague@company.com",
            sender_name="Colleague",
            recipients=[
                "person1@company.com",
                "person2@company.com",
                "person3@company.com",
                "person4@company.com"
            ],
            body="Here are the notes from today's meeting."
        ),
        EmailMessage(
            message_id="<3@example.com>",
            subject="Re: Project discussion",
            sender="supervisor@company.com",
            sender_name="Supervisor",
            recipients=["you@company.com"],
            body="Thanks for the update. Looking forward to the next phase. " * 10,
            in_reply_to="<2@example.com>"
        ),
    ]

    print("\nScoring example emails:\n")

    for i, email in enumerate(emails):
        score = spinner._score_message_importance(email)
        print(f"Email {i+1}: {email.subject}")
        print(f"  From: {email.sender_name}")
        print(f"  Recipients: {len(email.recipients)}")
        print(f"  Attachments: {len(email.attachments)}")
        print(f"  Body length: {len(email.body)} chars")
        print(f"  Is reply: {email.in_reply_to is not None}")
        print(f"  Importance Score: {score.score:.2f}")
        print(f"  Reason: {score.reason}")
        print()


async def demo_email_parsing():
    """Demonstrate email parsing."""
    print("\n" + "=" * 70)
    print("Email Parsing Demo")
    print("=" * 70)

    from HoloLoom.spinningWheel.email_spinner import IMAPClient

    print("\nEmail list parsing examples:")
    print()

    test_cases = [
        "user@example.com",
        "User Name <user@example.com>",
        "user1@example.com, user2@example.com",
        'John Doe <john@example.com>, "Jane Smith" <jane@example.com>',
    ]

    for test in test_cases:
        result = IMAPClient._parse_email_list(test)
        print(f"Input:  {test}")
        print(f"Output: {result}")
        print()


async def main():
    """Run all demos."""
    await demo_email_spinner()
    await demo_importance_scoring()
    await demo_email_parsing()

    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
