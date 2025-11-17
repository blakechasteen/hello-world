#!/usr/bin/env python3
"""
Demo: Email Workflow Templates
================================

Demonstrates all 5 email workflow templates with sample data:

1. Inbox Triage - Automatically classify and respond to emails
2. Meeting Summarization - Transcribe and extract action items
3. Email Newsletter Digest - Aggregate newsletters into weekly digest
4. Calendar Optimization - Optimize calendar and suggest time blocks
5. Customer Support Automation - Auto-triage and respond to support tickets

**Usage**:
    PYTHONPATH=. python demos/demo_email_workflows.py

**Created**: November 2025
"""

import asyncio
import json
from typing import Dict, Any

# Import all workflow templates
from HoloLoom.workflows.templates.email.inbox_triage import InboxTriageTemplate
from HoloLoom.workflows.templates.email.meeting_summary import MeetingSummaryTemplate
from HoloLoom.workflows.templates.email.newsletter_digest import NewsletterDigestTemplate
from HoloLoom.workflows.templates.email.calendar_optimization import CalendarOptimizationTemplate
from HoloLoom.workflows.templates.email.customer_support import CustomerSupportTemplate


def print_banner(text: str):
    """Print a formatted banner"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_section(text: str):
    """Print a section header"""
    print(f"\n{'─' * 80}")
    print(f"  {text}")
    print(f"{'─' * 80}")


async def demo_inbox_triage():
    """Demo 1: Inbox Triage Workflow"""
    print_banner("DEMO 1: Inbox Triage Workflow")

    template = InboxTriageTemplate()

    print(f"\n📧 Workflow: {template.name}")
    print(f"   Category: {template.category}")
    print(f"   Difficulty: {template.difficulty}")
    print(f"   Impact: {template.estimated_time_saved}")
    print(f"   Success Rate: {template.success_rate * 100}%")

    print_section("Workflow Definition")
    workflow = template.get_workflow_definition()
    print(f"   Nodes: {len(workflow['nodes'])}")
    print(f"   Connections: {len(workflow['connections'])}")

    print("\n   Workflow Flow:")
    for i, node in enumerate(workflow['nodes'], 1):
        print(f"     {i}. {node['agentType']} (id: {node['id']})")

    print_section("Sample Test Data")
    test_data = template.get_test_data()
    print(f"\n   Sample Emails: {len(test_data['sample_emails'])}\n")

    for i, email in enumerate(test_data['sample_emails'], 1):
        print(f"   Email {i}:")
        print(f"     From: {email['from']}")
        print(f"     Subject: {email['subject']}")
        print(f"     Expected Category: {email['expected_category']}")

    print_section("Impact Estimate (200 emails/day)")
    impact = template.estimate_impact(200)
    print(f"\n   Time Saved: {impact['time_saved_per_day']['hours']} hours/day")
    print(f"   Cost Savings: {impact['cost_savings']['yearly']}/year")
    print(f"   ROI: {impact['roi']['message']}")

    print_section("Required Credentials")
    for cred in template.get_required_credentials():
        print(f"   • {cred}")


async def demo_meeting_summary():
    """Demo 2: Meeting Summarization Workflow"""
    print_banner("DEMO 2: Meeting Summarization Workflow")

    template = MeetingSummaryTemplate()

    print(f"\n🎤 Workflow: {template.name}")
    print(f"   Impact: {template.estimated_time_saved}")
    print(f"   Success Rate: {template.success_rate * 100}%")

    print_section("Sample Meeting")
    test_data = template.get_test_data()
    meeting = test_data['sample_meeting']

    print(f"\n   Duration: {meeting['duration_minutes']} minutes")
    print(f"   Attendees: {', '.join(meeting['attendees'])}")

    print("\n   Transcript (excerpt):")
    transcript_lines = meeting['transcript'].strip().split('\n')
    for line in transcript_lines[:3]:
        print(f"     {line.strip()}")
    print("     ...")

    print("\n   Expected Action Items:")
    for item in meeting['expected_action_items']:
        print(f"     • {item['owner']}: {item['task']} ({item['deadline']})")

    print_section("Impact Estimate (10 meetings/week)")
    impact = template.estimate_impact(10)
    print(f"\n   Time Saved: {impact['time_saved_per_meeting']}/meeting")
    print(f"   Weekly Savings: {impact['weekly_time_saved']}")
    print(f"   Yearly Savings: {impact['yearly_cost_savings']}")
    print(f"   ROI: {impact['roi']}")


async def demo_newsletter_digest():
    """Demo 3: Email Newsletter Digest Workflow"""
    print_banner("DEMO 3: Email Newsletter Digest Workflow")

    template = NewsletterDigestTemplate()

    print(f"\n📰 Workflow: {template.name}")
    print(f"   Impact: {template.estimated_time_saved}")
    print(f"   Setup Time: {template.setup_time_minutes} minutes")

    print_section("Sample Newsletters")
    test_data = template.get_test_data()

    for i, newsletter in enumerate(test_data['sample_newsletters'], 1):
        print(f"\n   Newsletter {i}:")
        print(f"     From: {newsletter['from']}")
        print(f"     Subject: {newsletter['subject']}")
        print(f"     Key Insights: {', '.join(newsletter['key_insights'])}")

    print_section("Impact Estimate (20 newsletters/week)")
    impact = template.estimate_impact(20)
    print(f"\n   Time Saved: {impact['time_saved_per_week']}")
    print(f"   Yearly Savings: {impact['yearly_savings']}")


async def demo_calendar_optimization():
    """Demo 4: Calendar Optimization Workflow"""
    print_banner("DEMO 4: Calendar Optimization Workflow")

    template = CalendarOptimizationTemplate()

    print(f"\n📅 Workflow: {template.name}")
    print(f"   Difficulty: {template.difficulty}")
    print(f"   Impact: {template.estimated_time_saved}")

    print_section("Sample Calendar Analysis")
    test_data = template.get_test_data()
    calendar = test_data['sample_calendar']

    print(f"\n   Current State:")
    print(f"     Meetings/Week: {calendar['meetings_per_week']}")
    print(f"     Avg Duration: {calendar['average_meeting_duration']} min")
    print(f"     Back-to-back: {calendar['back_to_back_meetings']}")

    print(f"\n   Meeting Types:")
    for meeting_type, count in calendar['meeting_types'].items():
        print(f"     {meeting_type.replace('_', ' ').title()}: {count}")

    print(f"\n   Suggested Optimizations:")
    for suggestion in calendar['expected_suggestions']:
        print(f"     • {suggestion}")

    print_section("Impact Estimate (25 meetings/week)")
    impact = template.estimate_impact(25)
    print(f"\n   Meetings Declined: {impact['meetings_declined']} low-value meetings")
    print(f"   Weekly Time Saved: {impact['weekly_time_saved']}")
    print(f"   Yearly Savings: {impact['yearly_savings']}")


async def demo_customer_support():
    """Demo 5: Customer Support Automation Workflow"""
    print_banner("DEMO 5: Customer Support Automation Workflow")

    template = CustomerSupportTemplate()

    print(f"\n🎫 Workflow: {template.name}")
    print(f"   Impact: {template.estimated_time_saved}")
    print(f"   Success Rate: {template.success_rate * 100}%")

    print_section("Sample Support Tickets")
    test_data = template.get_test_data()

    for i, ticket in enumerate(test_data['sample_tickets'], 1):
        print(f"\n   Ticket {i} ({ticket['id']}):")
        print(f"     Subject: {ticket['subject']}")
        print(f"     Category: {ticket['category']}")
        print(f"     Severity: {ticket['severity']}")
        print(f"     Action: {ticket['expected_action']}")

    print_section("Impact Estimate (100 tickets/day)")
    impact = template.estimate_impact(100)

    print(f"\n   Automation Rate: {impact['automation_rate']}")
    print(f"     Automated: {impact['automated_tickets']} tickets")
    print(f"     Escalated: {impact['escalated_tickets']} tickets")

    print(f"\n   Time Savings:")
    print(f"     Daily: {impact['time_saved_per_day']}")
    print(f"     Weekly: {impact['weekly_time_saved']}")

    print(f"\n   Response Time:")
    print(f"     Manual: {impact['response_time']['manual']}")
    print(f"     Automated: {impact['response_time']['automated']}")
    print(f"     Improvement: {impact['response_time']['improvement']}")

    print(f"\n   Cost Savings:")
    print(f"     Daily: {impact['cost_savings']['daily']}")
    print(f"     Yearly: {impact['cost_savings']['yearly']}")


async def demo_comparison():
    """Compare all 5 workflows"""
    print_banner("WORKFLOW COMPARISON")

    templates = {
        'Inbox Triage': InboxTriageTemplate(),
        'Meeting Summary': MeetingSummaryTemplate(),
        'Newsletter Digest': NewsletterDigestTemplate(),
        'Calendar Optimization': CalendarOptimizationTemplate(),
        'Customer Support': CustomerSupportTemplate()
    }

    print("\n{:<25} {:<15} {:<12} {:<18} {:<12}".format(
        "Workflow", "Time Saved", "Success %", "Difficulty", "Setup (min)"
    ))
    print("─" * 85)

    for name, template in templates.items():
        print("{:<25} {:<15} {:<12} {:<18} {:<12}".format(
            name,
            template.estimated_time_saved,
            f"{template.success_rate * 100:.0f}%",
            template.difficulty,
            template.setup_time_minutes
        ))

    print_section("Total Impact Across All Workflows")

    # Calculate combined impact
    inbox_impact = templates['Inbox Triage'].estimate_impact(200)
    meeting_impact = templates['Meeting Summary'].estimate_impact(10)
    newsletter_impact = templates['Newsletter Digest'].estimate_impact(20)
    calendar_impact = templates['Calendar Optimization'].estimate_impact(25)
    support_impact = templates['Customer Support'].estimate_impact(100)

    total_daily_hours = (
        inbox_impact['time_saved_per_day']['hours'] +
        (float(meeting_impact['time_saved_per_meeting'].split()[0]) * 10 / 5) +  # per week to per day
        (float(newsletter_impact['time_saved_per_week'].split()[0]) / 5) +
        (float(calendar_impact['weekly_time_saved'].split()[0]) / 5) +
        float(support_impact['time_saved_per_day'].split()[0])
    )

    print(f"\n   Combined Time Saved: {total_daily_hours:.1f} hours/day")
    print(f"   Weekly: {total_daily_hours * 5:.1f} hours/week")
    print(f"   Yearly: {total_daily_hours * 250:.0f} hours/year")
    print(f"   Value (at $100/hr): ${total_daily_hours * 250 * 100:,.2f}/year")
    print(f"   ROI (vs $120/year): {(total_daily_hours * 250 * 100) / 120:.0f}x")


async def main():
    """Run all demos"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                                                                           ║
    ║         HoloLoom Email Workflow Templates - Interactive Demo             ║
    ║                                                                           ║
    ║  Showcasing 5 high-impact workflows that save 12+ hours/week            ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    # Run all demos
    await demo_inbox_triage()
    await asyncio.sleep(1)

    await demo_meeting_summary()
    await asyncio.sleep(1)

    await demo_newsletter_digest()
    await asyncio.sleep(1)

    await demo_calendar_optimization()
    await asyncio.sleep(1)

    await demo_customer_support()
    await asyncio.sleep(1)

    await demo_comparison()

    print_banner("Demo Complete!")
    print("""
    🎉 All 5 workflow templates demonstrated!

    Next Steps:
    1. Choose a workflow to deploy
    2. Configure your API credentials
    3. Run with real data
    4. Measure your time savings!

    Documentation: HoloLoom/workflows/templates/email/
    Tests: pytest HoloLoom/workflows/templates/email/tests/ -v

    Happy automating! 🚀
    """)


if __name__ == "__main__":
    asyncio.run(main())
