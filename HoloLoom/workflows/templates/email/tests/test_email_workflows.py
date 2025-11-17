#!/usr/bin/env python3
"""
Test Suite for Email Workflow Templates
========================================

Comprehensive tests for all 5 email workflow templates:
1. Inbox Triage
2. Meeting Summarization
3. Email Newsletter Digest
4. Calendar Optimization
5. Customer Support Automation

**Created**: November 2025
"""

import pytest
import json
from typing import Dict, Any


# Import all workflow templates
from HoloLoom.workflows.templates.email.inbox_triage import InboxTriageTemplate
from HoloLoom.workflows.templates.email.meeting_summary import MeetingSummaryTemplate
from HoloLoom.workflows.templates.email.newsletter_digest import NewsletterDigestTemplate
from HoloLoom.workflows.templates.email.calendar_optimization import CalendarOptimizationTemplate
from HoloLoom.workflows.templates.email.customer_support import CustomerSupportTemplate


# ============================================================================
# Test Inbox Triage Workflow
# ============================================================================

class TestInboxTriageWorkflow:
    """Test Inbox Triage workflow template"""

    def test_template_creation(self):
        """Test that template can be created"""
        template = InboxTriageTemplate()
        assert template.id == "inbox-triage"
        assert template.name == "Inbox Triage"
        assert template.category == "Email & Communication"
        assert template.difficulty == "Beginner"

    def test_workflow_definition(self):
        """Test workflow definition structure"""
        template = InboxTriageTemplate()
        workflow = template.get_workflow_definition()

        # Check structure
        assert 'version' in workflow
        assert 'name' in workflow
        assert 'nodes' in workflow
        assert 'connections' in workflow

        # Check nodes
        assert len(workflow['nodes']) == 6  # 6 nodes in workflow
        node_types = [node['agentType'] for node in workflow['nodes']]
        assert 'email_fetcher' in node_types
        assert 'email_classifier' in node_types
        assert 'response_drafter' in node_types
        assert 'slack_notifier' in node_types

        # Check connections
        assert len(workflow['connections']) == 6

    def test_required_credentials(self):
        """Test required credentials"""
        template = InboxTriageTemplate()
        creds = template.get_required_credentials()

        assert 'GMAIL_API_KEY' in creds
        assert 'SLACK_WEBHOOK_URL' in creds
        assert len(creds) == 2

    def test_test_data(self):
        """Test sample data generation"""
        template = InboxTriageTemplate()
        test_data = template.get_test_data()

        assert 'sample_emails' in test_data
        assert len(test_data['sample_emails']) == 5

        # Check first email
        email = test_data['sample_emails'][0]
        assert 'from' in email
        assert 'subject' in email
        assert 'body' in email
        assert 'expected_category' in email

    def test_impact_estimate(self):
        """Test impact calculation"""
        template = InboxTriageTemplate()
        impact = template.estimate_impact(emails_per_day=200)

        assert impact['emails_per_day'] == 200
        assert 'time_saved_per_day' in impact
        assert 'cost_savings' in impact
        assert 'roi' in impact

        # Should save significant time
        assert impact['time_saved_per_day']['hours'] > 1.0

    def test_customization_options(self):
        """Test customization options"""
        template = InboxTriageTemplate()
        options = template.get_customization_options()

        assert 'email_provider' in options
        assert 'classification_model' in options
        assert 'response_tone' in options

        # Check email provider options
        assert options['email_provider']['type'] == 'select'
        assert 'gmail' in options['email_provider']['options']


# ============================================================================
# Test Meeting Summarization Workflow
# ============================================================================

class TestMeetingSummaryWorkflow:
    """Test Meeting Summarization workflow template"""

    def test_template_creation(self):
        """Test template creation"""
        template = MeetingSummaryTemplate()
        assert template.id == "meeting-summary"
        assert template.name == "Meeting Summarization"
        assert template.estimated_time_saved == "30 min/meeting"

    def test_workflow_definition(self):
        """Test workflow definition"""
        template = MeetingSummaryTemplate()
        workflow = template.get_workflow_definition()

        # Check nodes
        assert len(workflow['nodes']) == 6
        node_types = [node['agentType'] for node in workflow['nodes']]
        assert 'audio_transcriber' in node_types
        assert 'action_item_extractor' in node_types
        assert 'synthesizer' in node_types

    def test_test_data(self):
        """Test sample meeting data"""
        template = MeetingSummaryTemplate()
        test_data = template.get_test_data()

        assert 'sample_meeting' in test_data
        meeting = test_data['sample_meeting']

        assert 'transcript' in meeting
        assert 'expected_action_items' in meeting
        assert len(meeting['attendees']) == 3

        # Check action items
        action_items = meeting['expected_action_items']
        assert len(action_items) == 3
        assert 'owner' in action_items[0]
        assert 'task' in action_items[0]
        assert 'deadline' in action_items[0]

    def test_impact_estimate(self):
        """Test impact calculation"""
        template = MeetingSummaryTemplate()
        impact = template.estimate_impact(meetings_per_week=10)

        assert impact['meetings_per_week'] == 10
        assert 'weekly_time_saved' in impact
        assert 'yearly_time_saved' in impact

        # Should save 28 minutes per meeting
        assert 'hours' in impact['weekly_time_saved']


# ============================================================================
# Test Newsletter Digest Workflow
# ============================================================================

class TestNewsletterDigestWorkflow:
    """Test Newsletter Digest workflow template"""

    def test_template_creation(self):
        """Test template creation"""
        template = NewsletterDigestTemplate()
        assert template.id == "newsletter-digest"
        assert template.estimated_time_saved == "1 hour/week"

    def test_workflow_definition(self):
        """Test workflow definition"""
        template = NewsletterDigestTemplate()
        workflow = template.get_workflow_definition()

        # Check nodes
        assert len(workflow['nodes']) == 4
        node_types = [node['agentType'] for node in workflow['nodes']]
        assert 'newsletter_fetcher' in node_types
        assert 'content_extractor' in node_types

    def test_test_data(self):
        """Test sample newsletters"""
        template = NewsletterDigestTemplate()
        test_data = template.get_test_data()

        assert 'sample_newsletters' in test_data
        assert len(test_data['sample_newsletters']) == 2

        newsletter = test_data['sample_newsletters'][0]
        assert 'from' in newsletter
        assert 'key_insights' in newsletter

    def test_impact_estimate(self):
        """Test impact calculation"""
        template = NewsletterDigestTemplate()
        impact = template.estimate_impact(newsletters_per_week=20)

        assert impact['newsletters_per_week'] == 20
        assert 'time_saved_per_week' in impact


# ============================================================================
# Test Calendar Optimization Workflow
# ============================================================================

class TestCalendarOptimizationWorkflow:
    """Test Calendar Optimization workflow template"""

    def test_template_creation(self):
        """Test template creation"""
        template = CalendarOptimizationTemplate()
        assert template.id == "calendar-optimization"
        assert template.difficulty == "Intermediate"

    def test_workflow_definition(self):
        """Test workflow definition"""
        template = CalendarOptimizationTemplate()
        workflow = template.get_workflow_definition()

        # Check nodes
        assert len(workflow['nodes']) == 5
        node_types = [node['agentType'] for node in workflow['nodes']]
        assert 'calendar_analyzer' in node_types

    def test_test_data(self):
        """Test sample calendar data"""
        template = CalendarOptimizationTemplate()
        test_data = template.get_test_data()

        assert 'sample_calendar' in test_data
        calendar = test_data['sample_calendar']

        assert 'meetings_per_week' in calendar
        assert 'expected_suggestions' in calendar
        assert len(calendar['expected_suggestions']) == 4

    def test_impact_estimate(self):
        """Test impact calculation"""
        template = CalendarOptimizationTemplate()
        impact = template.estimate_impact(meetings_per_week=25)

        assert impact['meetings_per_week'] == 25
        assert 'meetings_declined' in impact
        assert impact['meetings_declined'] > 0


# ============================================================================
# Test Customer Support Automation Workflow
# ============================================================================

class TestCustomerSupportWorkflow:
    """Test Customer Support Automation workflow template"""

    def test_template_creation(self):
        """Test template creation"""
        template = CustomerSupportTemplate()
        assert template.id == "customer-support-automation"
        assert template.success_rate == 0.87

    def test_workflow_definition(self):
        """Test workflow definition"""
        template = CustomerSupportTemplate()
        workflow = template.get_workflow_definition()

        # Check nodes
        assert len(workflow['nodes']) == 7
        node_types = [node['agentType'] for node in workflow['nodes']]
        assert 'ticket_fetcher' in node_types
        assert 'safety' in node_types

    def test_test_data(self):
        """Test sample tickets"""
        template = CustomerSupportTemplate()
        test_data = template.get_test_data()

        assert 'sample_tickets' in test_data
        assert len(test_data['sample_tickets']) == 4

        # Check first ticket
        ticket = test_data['sample_tickets'][0]
        assert 'id' in ticket
        assert 'subject' in ticket
        assert 'category' in ticket
        assert 'expected_action' in ticket

    def test_impact_estimate(self):
        """Test impact calculation"""
        template = CustomerSupportTemplate()
        impact = template.estimate_impact(tickets_per_day=100)

        assert impact['tickets_per_day'] == 100
        assert impact['automated_tickets'] == 70  # 70% automation
        assert impact['escalated_tickets'] == 30
        assert 'response_time' in impact
        assert 'cost_savings' in impact


# ============================================================================
# Integration Tests
# ============================================================================

class TestWorkflowIntegration:
    """Integration tests across all workflows"""

    def test_all_templates_export_json(self):
        """Test that all templates can export to JSON"""
        templates = [
            InboxTriageTemplate(),
            MeetingSummaryTemplate(),
            NewsletterDigestTemplate(),
            CalendarOptimizationTemplate(),
            CustomerSupportTemplate()
        ]

        for template in templates:
            workflow = template.get_workflow_definition()
            json_str = json.dumps(workflow)
            assert len(json_str) > 100  # Should have substantial content

    def test_all_templates_have_required_fields(self):
        """Test that all templates have required metadata fields"""
        templates = [
            InboxTriageTemplate(),
            MeetingSummaryTemplate(),
            NewsletterDigestTemplate(),
            CalendarOptimizationTemplate(),
            CustomerSupportTemplate()
        ]

        for template in templates:
            assert hasattr(template, 'id')
            assert hasattr(template, 'name')
            assert hasattr(template, 'category')
            assert hasattr(template, 'estimated_time_saved')
            assert hasattr(template, 'success_rate')
            assert template.success_rate > 0.8  # All should be >80% accuracy

    def test_workflow_compatibility(self):
        """Test that all workflows use compatible agent types"""
        # All workflows should use agent types that exist in the registry
        from HoloLoom.workflows.agent_registry import get_registry

        registry = get_registry()
        available_agents = set(registry.agents.keys())

        templates = [
            InboxTriageTemplate(),
            MeetingSummaryTemplate(),
            NewsletterDigestTemplate(),
            CalendarOptimizationTemplate(),
            CustomerSupportTemplate()
        ]

        for template in templates:
            workflow = template.get_workflow_definition()
            for node in workflow['nodes']:
                agent_type = node['agentType']
                assert agent_type in available_agents, \
                    f"Agent type '{agent_type}' not found in registry for workflow {template.name}"


# ============================================================================
# Performance Tests
# ============================================================================

class TestWorkflowPerformance:
    """Performance and scaling tests"""

    def test_large_email_volume_estimate(self):
        """Test impact estimate with large email volume"""
        template = InboxTriageTemplate()
        impact = template.estimate_impact(emails_per_day=1000)

        # Should still produce reasonable results
        assert impact['time_saved_per_day']['hours'] > 5
        assert 'yearly' in impact['cost_savings']

    def test_many_meetings_estimate(self):
        """Test impact estimate with many meetings"""
        template = MeetingSummaryTemplate()
        impact = template.estimate_impact(meetings_per_week=50)

        # Should scale linearly
        assert 'yearly_time_saved' in impact

    def test_high_ticket_volume(self):
        """Test customer support with high ticket volume"""
        template = CustomerSupportTemplate()
        impact = template.estimate_impact(tickets_per_day=500)

        assert impact['automated_tickets'] == 350  # 70% of 500
        assert '10x faster' in impact['response_time']['improvement']


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
