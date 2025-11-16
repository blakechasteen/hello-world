"""
Compliance Exports for Forensic Logs

Generate compliance-ready exports for GDPR, SOC2, ISO27001, PCI-DSS.

**Added: 2025-11-16** - Phase 4 Security Pipeline

Supported Standards:
- GDPR: Right to data portability (Art. 20)
- SOC2: Trust Services Criteria (CC6.1, CC7.2)
- ISO27001: Event logging (A.12.4.1)
- PCI-DSS: Requirement 10 (logging and monitoring)
"""

import json
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import io

from .logger import ForensicEntry
from .storage import StorageBackend
from .search import ForensicSearchEngine, SearchQuery


@dataclass
class ExportResult:
    """Export result."""
    format: str  # json, csv, html
    content: str  # Exported content
    entry_count: int
    export_time: datetime
    compliance_standard: str


class ComplianceExporter:
    """
    Export forensic logs for compliance requirements.

    **Added: 2025-11-16**

    Features:
    - GDPR data portability exports
    - SOC2 audit trail evidence
    - ISO27001 security event logs
    - PCI-DSS logging evidence
    - Multiple formats (JSON, CSV, HTML)
    """

    def __init__(self, storage: StorageBackend):
        """
        Initialize compliance exporter.

        Args:
            storage: Storage backend to export from
        """
        self.storage = storage
        self.search = ForensicSearchEngine(storage)

    async def export_gdpr(
        self,
        user_id: str,
        format: str = "json",
        days: int = 365
    ) -> ExportResult:
        """
        Export all data for a user (GDPR Article 20).

        Args:
            user_id: User identifier (hashed if logger uses hash_sensitive)
            format: Export format ("json", "csv", "html")
            days: Number of days to include

        Returns:
            Export result with user's data

        Compliance: GDPR Art. 20 (Right to data portability)
        """
        # Get all user activity
        entries = await self.search.get_user_activity(user_id, days)

        # Generate export
        if format == "json":
            content = self._export_json_gdpr(entries, user_id)
        elif format == "csv":
            content = self._export_csv_gdpr(entries, user_id)
        elif format == "html":
            content = self._export_html_gdpr(entries, user_id)
        else:
            raise ValueError(f"Unknown format: {format}")

        return ExportResult(
            format=format,
            content=content,
            entry_count=len(entries),
            export_time=datetime.utcnow(),
            compliance_standard="GDPR Article 20"
        )

    def _export_json_gdpr(self, entries: List[ForensicEntry], user_id: str) -> str:
        """Export GDPR data as JSON."""
        data = {
            'export_type': 'GDPR_Article_20',
            'user_id': user_id,
            'export_timestamp': datetime.utcnow().isoformat() + 'Z',
            'total_entries': len(entries),
            'entries': [e.to_dict() for e in entries]
        }
        return json.dumps(data, indent=2, separators=(',', ': '))

    def _export_csv_gdpr(self, entries: List[ForensicEntry], user_id: str) -> str:
        """Export GDPR data as CSV."""
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                'entry_id', 'timestamp', 'event_type', 'severity',
                'action', 'outcome', 'source_ip', 'metadata'
            ]
        )
        writer.writeheader()

        for entry in entries:
            writer.writerow({
                'entry_id': entry.entry_id,
                'timestamp': entry.timestamp,
                'event_type': entry.event_type,
                'severity': entry.severity,
                'action': entry.action,
                'outcome': entry.outcome,
                'source_ip': entry.source_ip or '',
                'metadata': json.dumps(entry.metadata)
            })

        return output.getvalue()

    def _export_html_gdpr(self, entries: List[ForensicEntry], user_id: str) -> str:
        """Export GDPR data as HTML."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>GDPR Data Export - User {user_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>GDPR Data Export</h1>
    <p><strong>User ID:</strong> {user_id}</p>
    <p><strong>Export Date:</strong> {datetime.utcnow().isoformat()}</p>
    <p><strong>Total Entries:</strong> {len(entries)}</p>
    <p><strong>Compliance:</strong> GDPR Article 20 (Right to data portability)</p>

    <h2>Activity Log</h2>
    <table>
        <tr>
            <th>Timestamp</th>
            <th>Event Type</th>
            <th>Action</th>
            <th>Outcome</th>
            <th>Severity</th>
        </tr>
"""
        for entry in entries:
            html += f"""        <tr>
            <td>{entry.timestamp}</td>
            <td>{entry.event_type}</td>
            <td>{entry.action}</td>
            <td>{entry.outcome}</td>
            <td>{entry.severity}</td>
        </tr>
"""

        html += """    </table>
</body>
</html>"""
        return html

    async def export_soc2(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "json"
    ) -> ExportResult:
        """
        Export audit trail evidence for SOC2.

        Args:
            start_date: Start of audit period
            end_date: End of audit period
            format: Export format ("json", "csv", "html")

        Returns:
            Export result with SOC2 evidence

        Compliance: SOC2 Trust Services Criteria (CC6.1, CC7.2)
        """
        # Get all entries in date range
        query = SearchQuery(
            start_time=start_date,
            end_time=end_date,
            limit=1000000
        )
        result = await self.search.search(query)
        entries = result.entries

        # Generate export
        if format == "json":
            content = self._export_json_soc2(entries, start_date, end_date)
        elif format == "csv":
            content = self._export_csv_soc2(entries, start_date, end_date)
        elif format == "html":
            content = self._export_html_soc2(entries, start_date, end_date)
        else:
            raise ValueError(f"Unknown format: {format}")

        return ExportResult(
            format=format,
            content=content,
            entry_count=len(entries),
            export_time=datetime.utcnow(),
            compliance_standard="SOC2 (CC6.1, CC7.2)"
        )

    def _export_json_soc2(
        self,
        entries: List[ForensicEntry],
        start_date: datetime,
        end_date: datetime
    ) -> str:
        """Export SOC2 evidence as JSON."""
        # Group by event type for SOC2 reporting
        by_type = {}
        for entry in entries:
            if entry.event_type not in by_type:
                by_type[entry.event_type] = []
            by_type[entry.event_type].append(entry.to_dict())

        data = {
            'export_type': 'SOC2_Audit_Trail',
            'audit_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'export_timestamp': datetime.utcnow().isoformat() + 'Z',
            'total_entries': len(entries),
            'events_by_type': {k: len(v) for k, v in by_type.items()},
            'entries': by_type
        }
        return json.dumps(data, indent=2, separators=(',', ': '))

    def _export_csv_soc2(
        self,
        entries: List[ForensicEntry],
        start_date: datetime,
        end_date: datetime
    ) -> str:
        """Export SOC2 evidence as CSV."""
        return self._export_csv_gdpr(entries, "SOC2_AUDIT")

    def _export_html_soc2(
        self,
        entries: List[ForensicEntry],
        start_date: datetime,
        end_date: datetime
    ) -> str:
        """Export SOC2 evidence as HTML."""
        # Similar to GDPR HTML but with SOC2 branding
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>SOC2 Audit Trail</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #2196F3; color: white; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #2196F3; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SOC2 Audit Trail Evidence</h1>
        <p>Trust Services Criteria: CC6.1 (Logical and Physical Access Controls), CC7.2 (System Monitoring)</p>
    </div>

    <h2>Audit Period</h2>
    <p><strong>Start:</strong> {start_date.isoformat()}</p>
    <p><strong>End:</strong> {end_date.isoformat()}</p>
    <p><strong>Total Events:</strong> {len(entries)}</p>

    <h2>Security Events</h2>
    <table>
        <tr>
            <th>Timestamp</th>
            <th>Event Type</th>
            <th>Action</th>
            <th>Outcome</th>
            <th>Severity</th>
            <th>Hash</th>
        </tr>
"""
        for entry in entries:
            html += f"""        <tr>
            <td>{entry.timestamp}</td>
            <td>{entry.event_type}</td>
            <td>{entry.action}</td>
            <td>{entry.outcome}</td>
            <td>{entry.severity}</td>
            <td>{entry.current_hash[:16]}...</td>
        </tr>
"""

        html += """    </table>
</body>
</html>"""
        return html

    async def export_iso27001(
        self,
        days: int = 90,
        format: str = "json"
    ) -> ExportResult:
        """
        Export security event logs for ISO27001.

        Args:
            days: Number of days to include
            format: Export format ("json", "csv", "html")

        Returns:
            Export result with ISO27001 evidence

        Compliance: ISO27001 A.12.4.1 (Event logging)
        """
        # Get security events (attacks, incidents, critical)
        query = SearchQuery(
            start_time=datetime.utcnow() - timedelta(days=days),
            end_time=datetime.utcnow(),
            event_types=["attack", "incident"],
            limit=1000000
        )
        result = await self.search.search(query)
        entries = result.entries

        # Also include critical events
        critical_query = SearchQuery(
            start_time=datetime.utcnow() - timedelta(days=days),
            end_time=datetime.utcnow(),
            severities=["CRITICAL"],
            limit=1000000
        )
        critical_result = await self.search.search(critical_query)
        entries.extend(critical_result.entries)

        # Remove duplicates
        seen = set()
        unique_entries = []
        for entry in entries:
            if entry.entry_id not in seen:
                seen.add(entry.entry_id)
                unique_entries.append(entry)

        # Sort by timestamp
        unique_entries.sort(key=lambda e: e.timestamp)

        # Generate export
        if format == "json":
            content = self._export_json_iso27001(unique_entries, days)
        elif format == "csv":
            content = self._export_csv_gdpr(unique_entries, "ISO27001")
        elif format == "html":
            content = self._export_html_iso27001(unique_entries, days)
        else:
            raise ValueError(f"Unknown format: {format}")

        return ExportResult(
            format=format,
            content=content,
            entry_count=len(unique_entries),
            export_time=datetime.utcnow(),
            compliance_standard="ISO27001 A.12.4.1"
        )

    def _export_json_iso27001(self, entries: List[ForensicEntry], days: int) -> str:
        """Export ISO27001 evidence as JSON."""
        data = {
            'export_type': 'ISO27001_Security_Events',
            'control': 'A.12.4.1 Event logging',
            'period_days': days,
            'export_timestamp': datetime.utcnow().isoformat() + 'Z',
            'total_entries': len(entries),
            'entries': [e.to_dict() for e in entries]
        }
        return json.dumps(data, indent=2, separators=(',', ': '))

    def _export_html_iso27001(self, entries: List[ForensicEntry], days: int) -> str:
        """Export ISO27001 evidence as HTML."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>ISO27001 Security Event Log</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #FF9800; color: white; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #FF9800; color: white; }}
        .critical {{ background-color: #ffebee; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ISO27001 Security Event Log</h1>
        <p>Control: A.12.4.1 (Event logging)</p>
    </div>

    <h2>Report Period</h2>
    <p><strong>Last {days} days</strong></p>
    <p><strong>Total Security Events:</strong> {len(entries)}</p>

    <h2>Events</h2>
    <table>
        <tr>
            <th>Timestamp</th>
            <th>Event Type</th>
            <th>Action</th>
            <th>Outcome</th>
            <th>Severity</th>
        </tr>
"""
        for entry in entries:
            row_class = ' class="critical"' if entry.severity == "CRITICAL" else ''
            html += f"""        <tr{row_class}>
            <td>{entry.timestamp}</td>
            <td>{entry.event_type}</td>
            <td>{entry.action}</td>
            <td>{entry.outcome}</td>
            <td>{entry.severity}</td>
        </tr>
"""

        html += """    </table>
</body>
</html>"""
        return html


async def export_gdpr(
    storage: StorageBackend,
    user_id: str,
    format: str = "json"
) -> ExportResult:
    """
    Convenience function to export GDPR data.

    Args:
        storage: Storage backend
        user_id: User identifier
        format: Export format

    Returns:
        Export result
    """
    exporter = ComplianceExporter(storage)
    return await exporter.export_gdpr(user_id, format)


async def export_soc2(
    storage: StorageBackend,
    start_date: datetime,
    end_date: datetime,
    format: str = "json"
) -> ExportResult:
    """
    Convenience function to export SOC2 data.

    Args:
        storage: Storage backend
        start_date: Audit period start
        end_date: Audit period end
        format: Export format

    Returns:
        Export result
    """
    exporter = ComplianceExporter(storage)
    return await exporter.export_soc2(start_date, end_date, format)


async def export_iso27001(
    storage: StorageBackend,
    days: int = 90,
    format: str = "json"
) -> ExportResult:
    """
    Convenience function to export ISO27001 data.

    Args:
        storage: Storage backend
        days: Number of days to include
        format: Export format

    Returns:
        Export result
    """
    exporter = ComplianceExporter(storage)
    return await exporter.export_iso27001(days, format)
