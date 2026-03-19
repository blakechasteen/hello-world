#!/usr/bin/env python3
"""
Beekeeping Department - Domain-specific departments for beekeeping operations.

This package contains specialized departments for beekeeping:
- MasterWeaverDepartment: Entity extraction and knowledge management
- (Future) InspectionDepartment: Hive inspection analysis
- (Future) ForecastingDepartment: Colony health prediction
"""

from .masterweaver import MasterWeaverDepartment

__all__ = [
    'MasterWeaverDepartment'
]
