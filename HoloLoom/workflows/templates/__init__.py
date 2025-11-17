"""
HoloLoom Workflow Templates
============================

Pre-built workflow templates organized by category.

Categories:
- Email & Communication (5 templates)
- Code & Development (coming soon)
- Data & Analytics (coming soon)
- Content Creation (coming soon)

**Created**: November 2025
"""

# Email templates
from HoloLoom.workflows.templates.email import (
    InboxTriageTemplate,
    MeetingSummaryTemplate,
    NewsletterDigestTemplate,
    CalendarOptimizationTemplate,
    CustomerSupportTemplate
)

__all__ = [
    # Email templates
    'InboxTriageTemplate',
    'MeetingSummaryTemplate',
    'NewsletterDigestTemplate',
    'CalendarOptimizationTemplate',
    'CustomerSupportTemplate'
]

__version__ = '1.0.0'
