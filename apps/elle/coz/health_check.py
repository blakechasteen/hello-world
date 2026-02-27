#!/usr/bin/env python3
"""
COZ Daily Brief Health Check

Monitors automation system health and exports metrics.

Features:
- System health checks (parsers, disk space, permissions)
- Prometheus metrics export (optional)
- JSON health status API
- Alert generation for failures

Usage:
    # Check health
    python health_check.py

    # Export Prometheus metrics
    python health_check.py --prometheus

    # JSON output
    python health_check.py --json

Created: 2025-11-22
Part of: Option 2 - Automation (Week 1)
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from elle.coz.sync_manager import SyncManager


class HealthStatus:
    """Health check status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck:
    """Health monitoring for COZ daily brief automation"""

    def __init__(self, output_dir: str = "./daily_briefs"):
        """
        Initialize health checker

        Args:
            output_dir: Output directory for briefs
        """
        self.output_dir = Path(output_dir)
        self.checks = []
        self.metrics = {}

    def check_output_directory(self) -> Tuple[bool, str]:
        """
        Check if output directory is writable

        Returns:
            (success, message) tuple
        """
        try:
            # Check if directory exists
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True, exist_ok=True)

            # Test write permissions
            test_file = self.output_dir / ".health_check"
            test_file.write_text("test")
            test_file.unlink()

            return True, f"Output directory writable: {self.output_dir.absolute()}"

        except Exception as e:
            return False, f"Output directory error: {e}"

    def check_disk_space(self, min_mb: int = 100) -> Tuple[bool, str]:
        """
        Check available disk space

        Args:
            min_mb: Minimum required space in MB

        Returns:
            (success, message) tuple
        """
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.output_dir)

            free_mb = free // (1024 * 1024)

            if free_mb < min_mb:
                return False, f"Low disk space: {free_mb}MB (minimum: {min_mb}MB)"

            return True, f"Disk space OK: {free_mb}MB free"

        except Exception as e:
            return False, f"Disk space check error: {e}"

    def check_parsers(self) -> Tuple[bool, str]:
        """
        Check if COZ parsers can initialize

        Returns:
            (success, message) tuple
        """
        try:
            sync = SyncManager()
            sync.parse_all()

            # Count available parsers
            parser_count = sum([
                1 for attr in ['time_tracking', 'cost_tracking', 'production_log',
                              'customer_orders', 'financials']
                if hasattr(sync, attr) and getattr(sync, attr) is not None
            ])

            if parser_count == 0:
                return False, "No parsers available"

            return True, f"{parser_count}/5 parsers available"

        except Exception as e:
            return False, f"Parser initialization error: {e}"

    def check_recent_briefs(self, max_age_hours: int = 48) -> Tuple[bool, str]:
        """
        Check if recent briefs were generated

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            (success, message) tuple
        """
        try:
            if not self.output_dir.exists():
                return False, "Output directory doesn't exist"

            # Find most recent brief
            brief_files = list(self.output_dir.glob("daily_brief_*.md"))

            if not brief_files:
                return False, "No briefs found"

            # Get most recent
            most_recent = max(brief_files, key=lambda p: p.stat().st_mtime)
            age_seconds = time.time() - most_recent.stat().st_mtime
            age_hours = age_seconds / 3600

            if age_hours > max_age_hours:
                return False, f"No recent briefs (last: {age_hours:.1f}h ago)"

            return True, f"Recent brief found (age: {age_hours:.1f}h)"

        except Exception as e:
            return False, f"Recent brief check error: {e}"

    def check_hololoom_available(self) -> Tuple[bool, str]:
        """
        Check if HoloLoom prompt refinement is available

        Returns:
            (success, message) tuple
        """
        try:
            from hololoom.prompting import refine_prompt
            return True, "HoloLoom refinement available"
        except ImportError:
            return False, "HoloLoom not available (refinement disabled)"

    def run_all_checks(self) -> Dict:
        """
        Run all health checks

        Returns:
            Health status dictionary
        """
        checks = {
            "output_directory": self.check_output_directory(),
            "disk_space": self.check_disk_space(),
            "parsers": self.check_parsers(),
            "recent_briefs": self.check_recent_briefs(),
            "hololoom": self.check_hololoom_available(),
        }

        # Calculate overall status
        failures = sum(1 for success, _ in checks.values() if not success)

        if failures == 0:
            overall_status = HealthStatus.HEALTHY
        elif failures <= 2:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNHEALTHY

        # Build result
        result = {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                name: {
                    "success": success,
                    "message": message
                }
                for name, (success, message) in checks.items()
            },
            "summary": {
                "total": len(checks),
                "passed": len(checks) - failures,
                "failed": failures
            }
        }

        return result

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format

        Returns:
            Prometheus-formatted metrics
        """
        health = self.run_all_checks()

        metrics = []

        # Overall health status (0=unhealthy, 1=degraded, 2=healthy)
        status_value = {
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.DEGRADED: 1,
            HealthStatus.HEALTHY: 2
        }[health['status']]

        metrics.append(f"# HELP coz_automation_health_status Overall health status (0=unhealthy, 1=degraded, 2=healthy)")
        metrics.append(f"# TYPE coz_automation_health_status gauge")
        metrics.append(f"coz_automation_health_status {status_value}")
        metrics.append("")

        # Individual check statuses
        metrics.append(f"# HELP coz_automation_check_status Individual check status (0=fail, 1=pass)")
        metrics.append(f"# TYPE coz_automation_check_status gauge")
        for check_name, check_result in health['checks'].items():
            success_value = 1 if check_result['success'] else 0
            metrics.append(f'coz_automation_check_status{{check="{check_name}"}} {success_value}')
        metrics.append("")

        # Total checks
        metrics.append(f"# HELP coz_automation_checks_total Total number of health checks")
        metrics.append(f"# TYPE coz_automation_checks_total gauge")
        metrics.append(f"coz_automation_checks_total {health['summary']['total']}")
        metrics.append("")

        # Failed checks
        metrics.append(f"# HELP coz_automation_checks_failed Number of failed health checks")
        metrics.append(f"# TYPE coz_automation_checks_failed gauge")
        metrics.append(f"coz_automation_checks_failed {health['summary']['failed']}")
        metrics.append("")

        # Timestamp
        timestamp_ms = int(time.time() * 1000)
        metrics.append(f"# HELP coz_automation_last_check_timestamp_ms Last health check timestamp (milliseconds)")
        metrics.append(f"# TYPE coz_automation_last_check_timestamp_ms gauge")
        metrics.append(f"coz_automation_last_check_timestamp_ms {timestamp_ms}")

        return "\n".join(metrics)


def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="COZ Daily Brief Health Check",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./daily_briefs',
        help='Output directory for briefs (default: ./daily_briefs)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output JSON format'
    )
    parser.add_argument(
        '--prometheus',
        action='store_true',
        help='Output Prometheus metrics format'
    )

    args = parser.parse_args()

    # Run health check
    health_check = HealthCheck(output_dir=args.output_dir)

    if args.prometheus:
        # Prometheus format
        print(health_check.export_prometheus())
        return 0

    # Run all checks
    health = health_check.run_all_checks()

    if args.json:
        # JSON format
        print(json.dumps(health, indent=2))
    else:
        # Human-readable format
        print("=" * 70)
        print("COZ Daily Brief Automation - Health Check")
        print("=" * 70)
        print()
        print(f"Overall Status: {health['status'].upper()}")
        print(f"Timestamp: {health['timestamp']}")
        print()
        print(f"Checks: {health['summary']['passed']}/{health['summary']['total']} passing")
        print()

        # Show individual checks
        for check_name, check_result in health['checks'].items():
            status = "[OK]" if check_result['success'] else "[FAIL]"
            print(f"  {status} {check_name}: {check_result['message']}")

        print()

    # Exit code based on health status
    if health['status'] == HealthStatus.HEALTHY:
        return 0
    elif health['status'] == HealthStatus.DEGRADED:
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
