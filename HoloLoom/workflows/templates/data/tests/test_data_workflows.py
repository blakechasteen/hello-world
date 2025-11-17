#!/usr/bin/env python3
"""
Test Suite for Data & Analytics Workflows
==========================================

Tests all 5 Data & Analytics workflow templates for:
- Workflow definition structure
- Required credentials
- Test data validity
- Impact estimation
- Node and connection integrity

**Created**: November 2025
"""

import pytest
from datetime import datetime
from HoloLoom.workflows.templates.data import (
    ReportGenerationTemplate,
    DataCleaningTemplate,
    CompetitiveIntelligenceTemplate,
    SQLQueryGeneratorTemplate,
    DashboardRefreshTemplate
)


class TestReportGeneration:
    """Test Report Generation workflow template"""

    def test_workflow_definition(self):
        template = ReportGenerationTemplate()
        workflow = template.get_workflow_definition()

        assert workflow['version'] == '1.0.0'
        assert workflow['name'] == 'Report Generation'
        assert len(workflow['nodes']) == 5
        assert len(workflow['connections']) == 4

    def test_required_credentials(self):
        template = ReportGenerationTemplate()
        creds = template.get_required_credentials()

        assert 'DATABASE_URL' in creds
        assert 'SLACK_WEBHOOK_URL' in creds

    def test_impact_estimation(self):
        template = ReportGenerationTemplate()
        impact = template.estimate_impact(reports_per_week=1)

        assert impact['time_saved_per_week']['hours'] >= 3.5
        assert impact['cost_savings']['yearly'] > 0
        assert impact['roi']['ratio'] > 100


class TestDataCleaning:
    """Test Data Cleaning Pipeline workflow template"""

    def test_workflow_definition(self):
        template = DataCleaningTemplate()
        workflow = template.get_workflow_definition()

        assert workflow['name'] == 'Data Cleaning Pipeline'
        assert len(workflow['nodes']) == 6
        assert len(workflow['connections']) == 5

    def test_test_data(self):
        template = DataCleaningTemplate()
        test_data = template.get_test_data()

        assert 'sample_data' in test_data
        assert len(test_data['sample_data']) >= 3

    def test_impact_estimation(self):
        template = DataCleaningTemplate()
        impact = template.estimate_impact(projects_per_month=2)

        assert impact['time_saved_per_month']['hours'] > 0
        assert impact['cost_savings']['yearly'] > 0


class TestCompetitiveIntelligence:
    """Test Competitive Intelligence Monitor workflow template"""

    def test_workflow_definition(self):
        template = CompetitiveIntelligenceTemplate()
        workflow = template.get_workflow_definition()

        assert workflow['name'] == 'Competitive Intelligence Monitor'
        assert len(workflow['nodes']) == 5
        assert any(node['agentType'] == 'web_scraper' for node in workflow['nodes'])
        assert any(node['agentType'] == 'change_detector' for node in workflow['nodes'])

    def test_required_credentials(self):
        template = CompetitiveIntelligenceTemplate()
        creds = template.get_required_credentials()

        assert 'SLACK_WEBHOOK_URL' in creds

    def test_test_data(self):
        template = CompetitiveIntelligenceTemplate()
        test_data = template.get_test_data()

        assert 'sample_urls' in test_data

    def test_impact_estimation(self):
        template = CompetitiveIntelligenceTemplate()
        impact = template.estimate_impact()

        assert impact['time_saved_per_week']['hours'] > 0
        assert impact['roi']['ratio'] > 100


class TestSQLQueryGenerator:
    """Test SQL Query Generator workflow template"""

    def test_workflow_definition(self):
        template = SQLQueryGeneratorTemplate()
        workflow = template.get_workflow_definition()

        assert workflow['name'] == 'SQL Query Generator'
        assert len(workflow['nodes']) == 5
        assert any(node['agentType'] == 'natural_language_parser' for node in workflow['nodes'])
        assert any(node['agentType'] == 'sql_generator' for node in workflow['nodes'])
        assert any(node['agentType'] == 'sql_validator' for node in workflow['nodes'])

    def test_required_credentials(self):
        template = SQLQueryGeneratorTemplate()
        creds = template.get_required_credentials()

        assert 'DATABASE_URL' in creds

    def test_test_data(self):
        template = SQLQueryGeneratorTemplate()
        test_data = template.get_test_data()

        assert 'sample_queries' in test_data
        assert len(test_data['sample_queries']) >= 3

    def test_impact_estimation(self):
        template = SQLQueryGeneratorTemplate()
        impact = template.estimate_impact(queries_per_week=10)

        assert impact['time_saved_per_week']['hours'] > 4


class TestDashboardRefresh:
    """Test Dashboard Auto-Refresh workflow template"""

    def test_workflow_definition(self):
        template = DashboardRefreshTemplate()
        workflow = template.get_workflow_definition()

        assert workflow['name'] == 'Dashboard Auto-Refresh'
        assert len(workflow['nodes']) == 5
        assert any(node['agentType'] == 'metric_calculator' for node in workflow['nodes'])
        assert any(node['agentType'] == 'chart_updater' for node in workflow['nodes'])

    def test_required_credentials(self):
        template = DashboardRefreshTemplate()
        creds = template.get_required_credentials()

        assert 'DATABASE_URL' in creds
        assert 'SLACK_WEBHOOK_URL' in creds
        assert 'GRAFANA_API_KEY' in creds

    def test_test_data(self):
        template = DashboardRefreshTemplate()
        test_data = template.get_test_data()

        assert 'sample_metrics' in test_data

    def test_impact_estimation(self):
        template = DashboardRefreshTemplate()
        impact = template.estimate_impact()

        assert impact['time_saved_per_week']['hours'] >= 14
        assert impact['roi']['ratio'] > 500


# Integration tests
class TestDataWorkflowsIntegration:
    """Integration tests for all Data & Analytics workflows"""

    def test_all_templates_have_metadata(self):
        templates = [
            ReportGenerationTemplate(),
            DataCleaningTemplate(),
            CompetitiveIntelligenceTemplate(),
            SQLQueryGeneratorTemplate(),
            DashboardRefreshTemplate()
        ]

        for template in templates:
            assert template.id
            assert template.name
            assert template.category == 'Data & Analytics'
            assert template.version
            assert template.estimated_time_saved
            assert 0.0 <= template.success_rate <= 1.0
            assert template.setup_time_minutes > 0
            assert len(template.tags) > 0

    def test_all_workflows_valid_structure(self):
        templates = [
            ReportGenerationTemplate(),
            DataCleaningTemplate(),
            CompetitiveIntelligenceTemplate(),
            SQLQueryGeneratorTemplate(),
            DashboardRefreshTemplate()
        ]

        for template in templates:
            workflow = template.get_workflow_definition()

            assert 'version' in workflow
            assert 'name' in workflow
            assert 'description' in workflow
            assert 'nodes' in workflow
            assert 'connections' in workflow
            assert 'metadata' in workflow

            assert len(workflow['nodes']) > 0
            assert len(workflow['connections']) > 0

            # Verify all connections reference valid nodes
            node_ids = {node['id'] for node in workflow['nodes']}
            for conn in workflow['connections']:
                assert conn['from'] in node_ids
                assert conn['to'] in node_ids


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
