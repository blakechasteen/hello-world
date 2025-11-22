#!/usr/bin/env python3
"""
Validate HoloLoom Phase 3 production deployment.

This script performs comprehensive validation checks to ensure Phase 3
adaptive learning is correctly deployed and functioning.

Usage:
    python validate_production.py [--data-dir /path/to/data]

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
"""

import asyncio
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# Add to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from HoloLoom.routing.query_classifier_adaptive import AdaptiveMoonshotClassifier
    from HoloLoom.routing.learning import (
        PatternMiner,
        ContinuousValidator,
        AdaptiveUpdater,
        PerformanceReporter
    )
except ImportError as e:
    print(f"❌ Error importing HoloLoom modules: {e}")
    print("Make sure PYTHONPATH is set correctly")
    sys.exit(1)

class ProductionValidator:
    """Validates Phase 3 production deployment."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.classifier = None
        self.checks_passed = 0
        self.checks_total = 0

    def print_header(self):
        """Print validation header."""
        print("=" * 70)
        print(" HoloLoom Phase 3 - Production Deployment Validation")
        print("=" * 70)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data Directory: {self.data_dir}")
        print("=" * 70)
        print()

    async def check_directory_structure(self) -> bool:
        """Check that required directories exist."""
        self.checks_total += 1
        print("[1/10] Checking directory structure...")

        required_dirs = [
            self.data_dir,
            self.data_dir / "logs",
            self.data_dir / "patterns",
            self.data_dir / "validation",
            self.data_dir / "reports"
        ]

        missing = [d for d in required_dirs if not d.exists()]
        if missing:
            print(f"  ❌ Missing directories: {', '.join(str(d) for d in missing)}")
            return False

        print(f"  ✅ All required directories exist")
        return True

    async def check_classifier_initialization(self) -> bool:
        """Check classifier can be initialized."""
        self.checks_total += 1
        print("[2/10] Testing classifier initialization...")

        try:
            self.classifier = AdaptiveMoonshotClassifier(
                enable_semantic_tier=True,
                enable_adaptive_learning=True,
                data_dir=self.data_dir,
                background_learning=False  # Don't start background loop yet
            )
            print(f"  ✅ Classifier initialized successfully")
            return True
        except Exception as e:
            print(f"  ❌ Failed to initialize classifier: {e}")
            return False

    async def check_classification(self) -> bool:
        """Check basic classification works."""
        self.checks_total += 1
        print("[3/10] Testing query classification...")

        if not self.classifier:
            print(f"  ❌ Classifier not initialized (skipping)")
            return False

        test_queries = [
            ("hi", "trivial"),
            ("what is Python?", "simple"),
            ("explain neural networks", "complex"),
            ("analyze tradeoffs of RL algorithms", "research")
        ]

        try:
            for query, expected in test_queries:
                result = self.classifier.classify(query)
                if result.complexity is None:
                    print(f"  ❌ Classification returned None for: {query}")
                    return False
                if not (0 <= result.confidence <= 1):
                    print(f"  ❌ Invalid confidence {result.confidence} for: {query}")
                    return False

            print(f"  ✅ Classification working ({len(test_queries)} queries tested)")
            return True
        except Exception as e:
            print(f"  ❌ Classification failed: {e}")
            return False

    async def check_jsonl_logging(self) -> bool:
        """Check JSONL logging is working."""
        self.checks_total += 1
        print("[4/10] Testing JSONL logging...")

        if not self.classifier:
            print(f"  ❌ Classifier not initialized (skipping)")
            return False

        try:
            log_file = self.classifier.classification_log_path
            if not log_file.exists():
                print(f"  ❌ JSONL log file does not exist: {log_file}")
                return False

            # Check file has content (from previous classification test)
            if log_file.stat().st_size == 0:
                print(f"  ❌ JSONL log file is empty: {log_file}")
                return False

            # Count lines
            with open(log_file, 'r') as f:
                line_count = sum(1 for _ in f)

            print(f"  ✅ JSONL logging working ({line_count} entries, {log_file.stat().st_size} bytes)")
            return True
        except Exception as e:
            print(f"  ❌ JSONL logging check failed: {e}")
            return False

    async def check_pattern_miner(self) -> bool:
        """Check pattern miner is working."""
        self.checks_total += 1
        print("[5/10] Testing pattern miner...")

        if not self.classifier:
            print(f"  ❌ Classifier not initialized (skipping)")
            return False

        try:
            patterns = self.classifier.pattern_miner.mine_patterns(days_lookback=7)
            print(f"  ✅ Pattern miner working ({len(patterns)} patterns discovered)")
            return True
        except Exception as e:
            print(f"  ❌ Pattern miner failed: {e}")
            return False

    async def check_validator(self) -> bool:
        """Check continuous validator is working."""
        self.checks_total += 1
        print("[6/10] Testing continuous validator...")

        if not self.classifier:
            print(f"  ❌ Classifier not initialized (skipping)")
            return False

        try:
            # Run validation with small sample
            result = await self.classifier.validator.validate_hourly(sample_size=4)

            if not (0 <= result.overall_accuracy <= 1):
                print(f"  ❌ Invalid accuracy: {result.overall_accuracy}")
                return False

            print(f"  ✅ Validator working (accuracy={result.overall_accuracy:.1%})")
            return True
        except Exception as e:
            print(f"  ❌ Validator failed: {e}")
            return False

    async def check_updater(self) -> bool:
        """Check adaptive updater is working."""
        self.checks_total += 1
        print("[7/10] Testing adaptive updater...")

        if not self.classifier:
            print(f"  ❌ Classifier not initialized (skipping)")
            return False

        try:
            status = self.classifier.updater.get_deployment_status()

            print(f"  ✅ Updater working (deployments={status['total_deployments']}, " +
                  f"rollbacks={status['rollbacks']})")
            return True
        except Exception as e:
            print(f"  ❌ Updater failed: {e}")
            return False

    async def check_reporter(self) -> bool:
        """Check performance reporter is working."""
        self.checks_total += 1
        print("[8/10] Testing performance reporter...")

        if not self.classifier:
            print(f"  ❌ Classifier not initialized (skipping)")
            return False

        try:
            # Export Prometheus metrics
            metrics = self.classifier.reporter.export_prometheus_metrics()

            if "moonshot_accuracy" not in metrics:
                print(f"  ❌ Required metric 'moonshot_accuracy' not in export")
                return False

            print(f"  ✅ Reporter working (metrics exported, {len(metrics)} bytes)")
            return True
        except Exception as e:
            print(f"  ❌ Reporter failed: {e}")
            return False

    async def check_background_learning(self) -> bool:
        """Check background learning can start/stop."""
        self.checks_total += 1
        print("[9/10] Testing background learning...")

        if not self.classifier:
            print(f"  ❌ Classifier not initialized (skipping)")
            return False

        try:
            # Start background learning
            await self.classifier.start_background_learning()
            await asyncio.sleep(2)  # Let it start

            # Stop background learning
            await self.classifier.stop_background_learning()

            print(f"  ✅ Background learning working (started and stopped)")
            return True
        except Exception as e:
            print(f"  ❌ Background learning failed: {e}")
            return False

    async def check_learning_statistics(self) -> bool:
        """Check learning statistics can be retrieved."""
        self.checks_total += 1
        print("[10/10] Testing learning statistics...")

        if not self.classifier:
            print(f"  ❌ Classifier not initialized (skipping)")
            return False

        try:
            stats = self.classifier.get_learning_statistics()

            required_keys = [
                'total_queries_logged',
                'patterns_discovered',
                'patterns_deployed',
                'validation_accuracy',
                'regression_alerts'
            ]

            missing = [k for k in required_keys if k not in stats]
            if missing:
                print(f"  ❌ Missing statistics keys: {', '.join(missing)}")
                return False

            print(f"  ✅ Statistics working:")
            print(f"      - Queries logged: {stats['total_queries_logged']}")
            print(f"      - Patterns discovered: {stats['patterns_discovered']}")
            print(f"      - Patterns deployed: {stats['patterns_deployed']}")
            print(f"      - Validation accuracy: {stats['validation_accuracy']:.1%}")
            print(f"      - Regression alerts: {stats['regression_alerts']}")
            return True
        except Exception as e:
            print(f"  ❌ Statistics check failed: {e}")
            return False

    def print_summary(self):
        """Print validation summary."""
        print()
        print("=" * 70)
        print(f" Validation Results: {self.checks_passed}/{self.checks_total} checks passed")
        print("=" * 70)

        if self.checks_passed == self.checks_total:
            print()
            print("✅ All checks passed! Phase 3 deployment validated successfully.")
            print()
            print("Next steps:")
            print("  1. Start production service: systemctl start hololoom")
            print("  2. View Grafana dashboard: http://localhost:3000/d/hololoom-phase3")
            print("  3. Check Prometheus metrics: http://localhost:9090")
            print("  4. Monitor logs: tail -f /var/log/hololoom/production.log")
        else:
            print()
            print(f"❌ {self.checks_total - self.checks_passed} checks failed. Please investigate and fix.")
            print()
            print("Troubleshooting:")
            print("  1. Check data directory permissions")
            print("  2. Verify Python environment and dependencies")
            print("  3. Review logs for errors")
            print("  4. Consult PHASE_3_PRODUCTION_DEPLOYMENT.md")

    async def run(self) -> int:
        """Run all validation checks."""
        self.print_header()

        # Run all checks
        checks = [
            self.check_directory_structure(),
            self.check_classifier_initialization(),
            self.check_classification(),
            self.check_jsonl_logging(),
            self.check_pattern_miner(),
            self.check_validator(),
            self.check_updater(),
            self.check_reporter(),
            self.check_background_learning(),
            self.check_learning_statistics()
        ]

        for check in checks:
            if await check:
                self.checks_passed += 1
            print()  # Blank line between checks

        self.print_summary()

        return 0 if self.checks_passed == self.checks_total else 1

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate HoloLoom Phase 3 production deployment"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('./data'),
        help='Data directory path (default: ./data)'
    )

    args = parser.parse_args()

    # Create validator and run
    validator = ProductionValidator(args.data_dir)
    exit_code = asyncio.run(validator.run())
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
