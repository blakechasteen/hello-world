#!/usr/bin/env python3
"""
Test Suite for Developer Tools Workflows
=========================================

Tests all 5 Developer Tools workflow templates for:
- Workflow definition structure
- Required credentials
- Test data validity
- Impact estimation
- Node and connection integrity

**Created**: November 2025
"""

import pytest
from HoloLoom.workflows.templates.developer import (
    BugTriageTemplate,
    CodeReviewTemplate,
    DependencyMonitorTemplate,
    TestGeneratorTemplate,
    DocGeneratorTemplate
)


class TestBugTriage:
    """Test Bug Triage Automation workflow template"""

    def test_workflow_definition(self):
        template = BugTriageTemplate()
        workflow = template.get_workflow_definition()

        assert workflow['name'] == 'Bug Triage Automation'
        assert len(workflow['nodes']) == 5
        assert any(node['agentType'] == 'bug_parser' for node in workflow['nodes'])
        assert any(node['agentType'] == 'severity_classifier' for node in workflow['nodes'])

    def test_required_credentials(self):
        template = BugTriageTemplate()
        creds = template.get_required_credentials()

        assert 'GITHUB_TOKEN' in creds
        assert 'SLACK_WEBHOOK_URL' in creds

    def test_impact_estimation(self):
        template = BugTriageTemplate()
        impact = template.estimate_impact(bugs_per_week=50)

        assert impact['time_saved_per_week']['hours'] > 20
        assert impact['roi']['ratio'] > 100


class TestCodeReview:
    """Test Code Review Automation workflow template"""

    def test_workflow_definition(self):
        template = CodeReviewTemplate()
        workflow = template.get_workflow_definition()

        assert workflow['name'] == 'Code Review Automation'
        assert len(workflow['nodes']) == 5
        assert any(node['agentType'] == 'code_analyzer' for node in workflow['nodes'])
        assert any(node['agentType'] == 'security_scanner' for node in workflow['nodes'])

    def test_required_credentials(self):
        template = CodeReviewTemplate()
        creds = template.get_required_credentials()

        assert 'GITHUB_TOKEN' in creds


class TestDependencyMonitor:
    """Test Dependency Update Monitor workflow template"""

    def test_workflow_definition(self):
        template = DependencyMonitorTemplate()
        workflow = template.get_workflow_definition()

        assert workflow['name'] == 'Dependency Update Monitor'
        assert len(workflow['nodes']) == 5
        assert any(node['agentType'] == 'dependency_scanner' for node in workflow['nodes'])
        assert any(node['agentType'] == 'version_checker' for node in workflow['nodes'])

    def test_required_credentials(self):
        template = DependencyMonitorTemplate()
        creds = template.get_required_credentials()

        assert 'GITHUB_TOKEN' in creds
        assert 'SLACK_WEBHOOK_URL' in creds


class TestTestGenerator:
    """Test Test Case Generator workflow template"""

    def test_workflow_definition(self):
        template = TestGeneratorTemplate()
        workflow = template.get_workflow_definition()

        assert workflow['name'] == 'Test Case Generator'
        assert len(workflow['nodes']) == 5
        assert any(node['agentType'] == 'test_template_selector' for node in workflow['nodes'])
        assert any(node['agentType'] == 'test_generator' for node in workflow['nodes'])


class TestDocGenerator:
    """Test Documentation Generator workflow template"""

    def test_workflow_definition(self):
        template = DocGeneratorTemplate()
        workflow = template.get_workflow_definition()

        assert workflow['name'] == 'Documentation Generator'
        assert len(workflow['nodes']) == 5
        assert any(node['agentType'] == 'api_extractor' for node in workflow['nodes'])
        assert any(node['agentType'] == 'markdown_generator' for node in workflow['nodes'])


# Integration tests
class TestDeveloperWorkflowsIntegration:
    """Integration tests for all Developer Tools workflows"""

    def test_all_templates_have_metadata(self):
        templates = [
            BugTriageTemplate(),
            CodeReviewTemplate(),
            DependencyMonitorTemplate(),
            TestGeneratorTemplate(),
            DocGeneratorTemplate()
        ]

        for template in templates:
            assert template.id
            assert template.name
            assert template.category == 'Developer Tools'
            assert template.version
            assert 0.0 <= template.success_rate <= 1.0

    def test_all_workflows_valid_structure(self):
        templates = [
            BugTriageTemplate(),
            CodeReviewTemplate(),
            DependencyMonitorTemplate(),
            TestGeneratorTemplate(),
            DocGeneratorTemplate()
        ]

        for template in templates:
            workflow = template.get_workflow_definition()

            assert 'version' in workflow
            assert 'name' in workflow
            assert 'nodes' in workflow
            assert 'connections' in workflow

            node_ids = {node['id'] for node in workflow['nodes']}
            for conn in workflow['connections']:
                assert conn['from'] in node_ids
                assert conn['to'] in node_ids


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
