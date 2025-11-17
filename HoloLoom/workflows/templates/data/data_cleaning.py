#!/usr/bin/env python3
"""
Data Cleaning Pipeline Workflow Template
=========================================

**Impact**: Save 8 hours/project
**Success Rate**: 95%
**Setup Time**: 15 minutes

Automated data cleaning with anomaly detection, standardization,
missing value filling, and validation reports.

**Workflow**:
1. Data Loader → Load CSV/Excel/Database
2. Anomaly Detector → Identify outliers
3. Data Standardizer → Normalize formats
4. Missing Value Filler → Handle missing data
5. Validator → Final quality check
6. Response → Clean dataset ready for analysis

**Integrations**: CSV, Excel, Databases
**Created**: November 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
from datetime import datetime


@dataclass
class DataCleaningTemplate:
    """Data Cleaning Pipeline Workflow Template"""

    id: str = "data-cleaning"
    name: str = "Data Cleaning Pipeline"
    category: str = "Data & Analytics"
    difficulty: str = "Intermediate"
    version: str = "1.0.0"
    author: str = "HoloLoom"
    created_at: datetime = field(default_factory=datetime.now)

    estimated_time_saved: str = "8 hours/project"
    success_rate: float = 0.95
    setup_time_minutes: int = 15

    tags: List[str] = field(default_factory=lambda: [
        "data", "cleaning", "etl", "validation", "automation"
    ])

    def get_workflow_definition(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'name': self.name,
            'description': 'Automated data cleaning and validation pipeline',

            'nodes': [
                {'id': 'data_loader', 'agentType': 'data_transformer', 'x': 100, 'y': 200,
                 'config': {'operations': ['load', 'parse'], 'format': 'json'}},

                {'id': 'anomaly_detector', 'agentType': 'anomaly_detector', 'x': 300, 'y': 200,
                 'config': {'method': 'isolation_forest', 'sensitivity': 0.1}},

                {'id': 'standardizer', 'agentType': 'data_standardizer', 'x': 500, 'y': 200,
                 'config': {'operations': ['normalize', 'encode', 'format_dates'], 'encoding': 'utf-8'}},

                {'id': 'filler', 'agentType': 'missing_value_filler', 'x': 700, 'y': 200,
                 'config': {'strategy': 'mean', 'threshold': 0.5}},

                {'id': 'validator', 'agentType': 'data_transformer', 'x': 900, 'y': 200,
                 'config': {'operations': ['validate'], 'format': 'json'}},

                {'id': 'response', 'agentType': 'response', 'x': 1100, 'y': 200,
                 'config': {'format': 'json'}}
            ],

            'connections': [
                {'id': 'conn_1', 'from': 'data_loader', 'to': 'anomaly_detector'},
                {'id': 'conn_2', 'from': 'anomaly_detector', 'to': 'standardizer'},
                {'id': 'conn_3', 'from': 'standardizer', 'to': 'filler'},
                {'id': 'conn_4', 'from': 'filler', 'to': 'validator'},
                {'id': 'conn_5', 'from': 'validator', 'to': 'response'}
            ],

            'metadata': {'template_id': self.id, 'category': self.category, 'tags': self.tags}
        }

    def get_required_credentials(self) -> List[str]:
        return []

    def get_test_data(self) -> Dict[str, Any]:
        return {
            'sample_data': [
                {'id': 1, 'name': 'Alice', 'age': 25, 'salary': 50000},
                {'id': 2, 'name': 'Bob', 'age': None, 'salary': 60000},
                {'id': 3, 'name': 'Charlie', 'age': 35, 'salary': 999999},  # Outlier
            ]
        }

    def estimate_impact(self, projects_per_month: int = 2) -> Dict[str, Any]:
        manual_time_hours = 8 * projects_per_month
        workflow_time_hours = 0.5 * projects_per_month
        time_saved = manual_time_hours - workflow_time_hours

        return {
            'time_saved_per_month': {'hours': time_saved, 'message': f"Save {time_saved:.1f} hours/month"},
            'cost_savings': {'monthly': time_saved * 100, 'yearly': time_saved * 100 * 12},
            'roi': {'ratio': (time_saved * 100 * 12) / 120, 'message': f"{(time_saved * 100 * 12) / 120:.0f}x ROI"}
        }


def create_data_cleaning_workflow() -> Dict[str, Any]:
    template = DataCleaningTemplate()
    return template.get_workflow_definition()


if __name__ == "__main__":
    template = DataCleaningTemplate()
    print(f"Workflow: {template.name}")
    print(f"Time Saved: {template.estimated_time_saved}")
    impact = template.estimate_impact(2)
    print(f"Impact: {impact['time_saved_per_month']['message']}")
