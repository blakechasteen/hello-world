"""
HoloLoom Compliance and Certification Framework

Automated compliance monitoring for SOC2, GDPR, ISO 27001, and other frameworks.

**Created: 2025-11-16**

## Overview

This module provides comprehensive compliance automation including:
- SOC2 Type II preparation and monitoring
- GDPR compliance verification
- ISO 27001 control implementation
- Automated evidence collection
- Compliance dashboards and reporting

## Quick Start

```python
from HoloLoom.security.compliance import (
    ComplianceMonitor,
    SOC2Monitor,
    GDPRMonitor,
    ISO27001Monitor,
    generate_compliance_report
)

# Create comprehensive compliance monitor
monitor = ComplianceMonitor()

# Check overall compliance
status = await monitor.check_compliance()
print(f"Compliance Score: {status.overall_score:.1%}")

# Generate audit-ready report
report = await generate_compliance_report(
    frameworks=['soc2', 'gdpr', 'iso27001'],
    output_format='html'
)
```

## Modules

- `core` - Main compliance monitoring engine
- `soc2` - SOC2 Type II automation
- `gdpr` - GDPR compliance verification
- `iso27001` - ISO 27001 control implementation
- `evidence` - Automated evidence collection
- `reporting` - Compliance reports and dashboards
"""

from .core import (
    ComplianceMonitor,
    ComplianceStatus,
    ComplianceFramework,
    ControlStatus,
    RiskLevel,
)

from .soc2 import (
    SOC2Monitor,
    TrustServiceCriteria,
    SOC2Control,
    SOC2Report,
)

from .gdpr import (
    GDPRMonitor,
    GDPRArticle,
    DataSubjectRight,
    DPIAAssessment,
)

from .iso27001 import (
    ISO27001Monitor,
    ISO27001Control,
    ControlCategory,
    ISMSStatus,
)

from .evidence import (
    EvidenceCollector,
    EvidenceType,
    AuditEvidence,
    EvidenceRepository,
)

from .reporting import (
    ComplianceReporter,
    generate_compliance_report,
    generate_audit_package,
    ComplianceScore,
)

__all__ = [
    # Core
    'ComplianceMonitor',
    'ComplianceStatus',
    'ComplianceFramework',
    'ControlStatus',
    'RiskLevel',

    # SOC2
    'SOC2Monitor',
    'TrustServiceCriteria',
    'SOC2Control',
    'SOC2Report',

    # GDPR
    'GDPRMonitor',
    'GDPRArticle',
    'DataSubjectRight',
    'DPIAAssessment',

    # ISO 27001
    'ISO27001Monitor',
    'ISO27001Control',
    'ControlCategory',
    'ISMSStatus',

    # Evidence
    'EvidenceCollector',
    'EvidenceType',
    'AuditEvidence',
    'EvidenceRepository',

    # Reporting
    'ComplianceReporter',
    'generate_compliance_report',
    'generate_audit_package',
    'ComplianceScore',
]

__version__ = '1.0.0'
__author__ = 'HoloLoom Security Team'
__created__ = '2025-11-16'
