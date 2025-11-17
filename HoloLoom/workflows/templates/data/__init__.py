"""
Data & Analytics Workflow Templates
====================================

5 production-ready workflow templates for data automation:
1. Report Generation - Save 4 hours/week
2. Data Cleaning Pipeline - Save 8 hours/project
3. Competitive Intelligence Monitor - 24/7 monitoring
4. SQL Query Generator - 10x faster for non-technical users
5. Dashboard Auto-Refresh - Real-time insights

**Created**: November 2025
"""

from .report_generation import ReportGenerationTemplate, create_report_generation_workflow
from .data_cleaning import DataCleaningTemplate, create_data_cleaning_workflow
from .competitive_intelligence import CompetitiveIntelligenceTemplate, create_competitive_intelligence_workflow
from .sql_query_generator import SQLQueryGeneratorTemplate, create_sql_query_generator_workflow
from .dashboard_refresh import DashboardRefreshTemplate, create_dashboard_refresh_workflow

__all__ = [
    'ReportGenerationTemplate',
    'DataCleaningTemplate',
    'CompetitiveIntelligenceTemplate',
    'SQLQueryGeneratorTemplate',
    'DashboardRefreshTemplate',
    'create_report_generation_workflow',
    'create_data_cleaning_workflow',
    'create_competitive_intelligence_workflow',
    'create_sql_query_generator_workflow',
    'create_dashboard_refresh_workflow',
]
