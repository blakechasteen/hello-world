#!/usr/bin/env python3
"""
COZ Daily Brief Deployment Verification

Comprehensive pre-deployment checklist to verify all components are ready.

Usage:
    python verify_deployment.py [--verbose]

Returns:
    0: All checks passed (ready to deploy)
    1: Some checks failed (fix before deploying)
    2: Critical failures (cannot deploy)

Created: 2025-11-22
Part of: Production Deployment
"""

import sys
import subprocess
from pathlib import Path
from typing import Tuple, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class DeploymentVerifier:
    """Verifies deployment readiness"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.checks_passed = 0
        self.checks_failed = 0
        self.critical_failures = 0

    def check(self, name: str, check_func, critical: bool = False) -> bool:
        """
        Run a single check

        Args:
            name: Check name
            check_func: Function that returns (success, message)
            critical: Whether failure is critical

        Returns:
            True if check passed
        """
        try:
            success, message = check_func()

            if success:
                self.checks_passed += 1
                print(f"[OK] {name}")
                if self.verbose and message:
                    print(f"     {message}")
                return True
            else:
                self.checks_failed += 1
                if critical:
                    self.critical_failures += 1
                    print(f"[CRITICAL] {name}")
                else:
                    print(f"[FAIL] {name}")
                if message:
                    print(f"     {message}")
                return False

        except Exception as e:
            self.checks_failed += 1
            if critical:
                self.critical_failures += 1
                print(f"[CRITICAL] {name}")
            else:
                print(f"[FAIL] {name}")
            print(f"     Exception: {e}")
            return False

    def check_file_exists(self, filepath: str) -> Tuple[bool, str]:
        """Check if file exists"""
        path = Path(filepath)
        if path.exists():
            return True, f"Found: {path.absolute()}"
        return False, f"Missing: {path.absolute()}"

    def check_python_import(self, module: str) -> Tuple[bool, str]:
        """Check if Python module can be imported"""
        try:
            __import__(module)
            return True, f"Module '{module}' available"
        except ImportError:
            return False, f"Module '{module}' not found (install with: pip install {module})"

    def check_validation_tests(self) -> Tuple[bool, str]:
        """Check if validation tests pass"""
        try:
            result = subprocess.run(
                [sys.executable, "elle/coz/validate_automation.py"],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                return True, "5/5 validation tests passing"
            else:
                return False, f"Validation tests failed (exit code: {result.returncode})"

        except subprocess.TimeoutExpired:
            return False, "Validation tests timed out (>60 seconds)"
        except Exception as e:
            return False, f"Error running validation tests: {e}"

    def check_health_check(self) -> Tuple[bool, str]:
        """Check if health check runs"""
        try:
            # Import health check module
            from elle.coz.health_check import HealthCheck

            health_check = HealthCheck()
            health = health_check.run_all_checks()

            if health['status'] == 'healthy':
                return True, f"{health['summary']['passed']}/{health['summary']['total']} health checks passing"
            elif health['status'] == 'degraded':
                return True, f"Degraded: {health['summary']['failed']} checks failing (system functional)"
            else:
                return False, f"Unhealthy: {health['summary']['failed']} checks failing"

        except Exception as e:
            return False, f"Health check error: {e}"

    def check_ab_testing_results(self) -> Tuple[bool, str]:
        """Check if A/B testing results exist"""
        csv_path = Path("elle/coz/ab_test_results.csv")
        if csv_path.exists():
            lines = csv_path.read_text().count('\n')
            if lines >= 13:  # Header + 12 iterations
                return True, f"A/B test results: {lines - 1} iterations"
            else:
                return False, f"Incomplete A/B test results: {lines - 1}/12 iterations"
        return False, "A/B test results not found (run: python elle/coz/ab_testing.py --iterations 12)"

    def check_monitoring_configs(self) -> Tuple[bool, str]:
        """Check if monitoring configs exist"""
        configs = [
            "elle/coz/prometheus.yml",
            "elle/coz/prometheus_rules.yml",
            "elle/coz/grafana_dashboard.json"
        ]

        missing = [c for c in configs if not Path(c).exists()]

        if not missing:
            return True, f"All {len(configs)} monitoring configs present"
        return False, f"Missing configs: {', '.join(missing)}"

    def check_deployment_guides(self) -> Tuple[bool, str]:
        """Check if deployment guides exist"""
        guides = [
            "elle/coz/PRODUCTION_DEPLOYMENT_QUICK_START.md",
            "elle/coz/PRODUCTION_RECOMMENDATIONS.md"
        ]

        missing = [g for g in guides if not Path(g).exists()]

        if not missing:
            return True, f"All {len(guides)} deployment guides present"
        return False, f"Missing guides: {', '.join(missing)}"

    def run_all_checks(self) -> int:
        """
        Run all deployment verification checks

        Returns:
            0: All passed
            1: Some failed
            2: Critical failures
        """
        print("=" * 70)
        print("COZ Daily Brief - Deployment Verification")
        print("=" * 70)
        print()

        # Critical checks (must pass to deploy)
        print("[CRITICAL CHECKS]")
        self.check("Core automation script", lambda: self.check_file_exists("elle/coz/daily_brief_automation.py"), critical=True)
        self.check("Intelligence engine", lambda: self.check_file_exists("elle/coz/intelligence.py"), critical=True)
        self.check("Sync manager", lambda: self.check_file_exists("elle/coz/sync_manager.py"), critical=True)
        self.check("Health check system", lambda: self.check_file_exists("elle/coz/health_check.py"), critical=True)
        print()

        # Required dependencies
        print("[REQUIRED DEPENDENCIES]")
        self.check("schedule module", lambda: self.check_python_import("schedule"), critical=True)
        print()

        # Validation tests
        print("[VALIDATION TESTS]")
        self.check("5/5 validation tests", self.check_validation_tests, critical=True)
        print()

        # Health monitoring
        print("[HEALTH MONITORING]")
        self.check("Health check runs", self.check_health_check)
        self.check("Metrics server", lambda: self.check_file_exists("elle/coz/metrics_server.py"))
        print()

        # Monitoring configuration
        print("[MONITORING CONFIGURATION]")
        self.check("Monitoring configs", self.check_monitoring_configs)
        print()

        # Quality testing
        print("[QUALITY TESTING]")
        self.check("A/B testing framework", lambda: self.check_file_exists("elle/coz/ab_testing.py"))
        self.check("A/B testing results", self.check_ab_testing_results)
        print()

        # Documentation
        print("[DOCUMENTATION]")
        self.check("Deployment guides", self.check_deployment_guides)
        print()

        # Optional dependencies (nice-to-have)
        print("[OPTIONAL DEPENDENCIES]")
        self.check("HoloLoom (refinement)", lambda: self.check_python_import("hololoom.prompting"), critical=False)
        self.check("prometheus_client (metrics)", lambda: self.check_python_import("prometheus_client"), critical=False)
        print()

        # Summary
        print("=" * 70)
        print("Verification Summary")
        print("=" * 70)
        total = self.checks_passed + self.checks_failed
        print(f"Passed: {self.checks_passed}/{total}")
        print(f"Failed: {self.checks_failed}/{total}")
        print(f"Critical Failures: {self.critical_failures}")
        print()

        # Recommendation
        if self.critical_failures > 0:
            print("[CRITICAL] CANNOT DEPLOY - Fix critical failures first")
            print()
            print("Critical checks must pass before deployment:")
            print("  - Core automation scripts must exist")
            print("  - Required dependencies must be installed")
            print("  - Validation tests must pass (5/5)")
            return 2

        elif self.checks_failed > 0:
            print("[WARNING] CAN DEPLOY (with warnings)")
            print()
            print("Some non-critical checks failed:")
            print("  - System will work but may have reduced functionality")
            print("  - Fix failed checks for optimal experience")
            print()
            print("Ready to deploy: YES (with warnings)")
            return 1

        else:
            print("[OK] READY TO DEPLOY!")
            print()
            print("All checks passed. System is production-ready:")
            print("  - systemd: PRODUCTION_DEPLOYMENT_QUICK_START.md")
            print("  - Docker: PRODUCTION_DEPLOYMENT_QUICK_START.md")
            print("  - Monitoring: Prometheus + Grafana configs ready")
            print("  - Quality: A/B testing complete (12 iterations)")
            print()
            print("Next steps:")
            print("  1. Deploy to production (10-15 minutes)")
            print("  2. Configure monitoring (5 minutes)")
            print("  3. Run 24-hour soak test")
            print("  4. Gather user feedback")
            return 0


def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="COZ Daily Brief Deployment Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed check information'
    )

    args = parser.parse_args()

    verifier = DeploymentVerifier(verbose=args.verbose)
    return verifier.run_all_checks()


if __name__ == "__main__":
    sys.exit(main())
