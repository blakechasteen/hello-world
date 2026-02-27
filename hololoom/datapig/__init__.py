"""
DATAPIG: Data Analysis & Tactical Assessment Program for Integrity Governance

Star Trek-themed data quality validation system.

"Computer, analyze data integrity." - Every Star Trek captain ever

Usage:
    from hololoom.datapig import DataPigDetector

    detector = DataPigDetector()
    issues = detector.analyze_dataset(df)

    for issue in issues:
        print(f"{issue.severity}: {issue.message}")
"""

from .detector import (
    DataPigDetector,
    DataQualityIssue,
    IssueType,
    Severity,
    analyze_dataset,
    engage_warp_validation,
)

__all__ = [
    "DataPigDetector",
    "DataQualityIssue",
    "IssueType",
    "Severity",
    "analyze_dataset",
    "engage_warp_validation",
]

__version__ = "1.0.0-NCC-1701"  # USS Enterprise registry
