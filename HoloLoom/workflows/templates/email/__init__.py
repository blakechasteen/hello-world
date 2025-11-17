"""
Email & Communication Workflow Templates
=========================================

High-impact workflow templates for email automation and communication:

1. **Inbox Triage** - Automatically classify and respond to emails (2 hrs/day saved)
2. **Meeting Summarization** - Transcribe and extract action items (30 min/meeting saved)
3. **Email Newsletter Digest** - Aggregate newsletters into weekly digest (1 hr/week saved)
4. **Calendar Optimization** - Optimize calendar and suggest time blocks (5 hrs/week saved)
5. **Customer Support** - Auto-triage and respond to support tickets (20 hrs/week saved)

**Created**: November 2025
**Total Impact**: 12+ hours/week saved

Usage:
    from HoloLoom.workflows.templates.email import InboxTriageTemplate

    template = InboxTriageTemplate()
    workflow = template.get_workflow_definition()
"""

# Export all templates
from HoloLoom.workflows.templates.email.inbox_triage import InboxTriageTemplate
from HoloLoom.workflows.templates.email.meeting_summary import MeetingSummaryTemplate
from HoloLoom.workflows.templates.email.newsletter_digest import NewsletterDigestTemplate
from HoloLoom.workflows.templates.email.calendar_optimization import CalendarOptimizationTemplate
from HoloLoom.workflows.templates.email.customer_support import CustomerSupportTemplate

__all__ = [
    'InboxTriageTemplate',
    'MeetingSummaryTemplate',
    'NewsletterDigestTemplate',
    'CalendarOptimizationTemplate',
    'CustomerSupportTemplate'
]

__version__ = '1.0.0'
__author__ = 'HoloLoom'
