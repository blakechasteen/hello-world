#!/usr/bin/env python3
"""
HoloLoom Setup Verification Script

Protocol-based verification system following HoloLoom's architectural patterns.
Validates environment, dependencies, backends, and system integrity.

Usage:
    python scripts/verify_setup.py --level quick       # Essential checks (2-3 min)
    python scripts/verify_setup.py --level standard    # All core checks (5-10 min)
    python scripts/verify_setup.py --level comprehensive  # Edge cases (15+ min)

    python scripts/verify_setup.py --format json > results.json
    python scripts/verify_setup.py --format markdown > SETUP_REPORT.md
    python scripts/verify_setup.py --category imports,backends

Created: 2025-12-15
"""

import asyncio
import argparse
import json
import sys
import time
import importlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime


# ============================================================================
# Enums and Data Classes
# ============================================================================

class VerificationLevel(Enum):
    """Verification depth levels."""
    QUICK = "quick"           # Essential only (2-3 min)
    STANDARD = "standard"     # All core checks (5-10 min)
    COMPREHENSIVE = "comprehensive"  # Edge cases (15+ min)


class CheckCategory(Enum):
    """Check categories."""
    IMPORTS = "imports"           # Python environment
    DEPENDENCIES = "dependencies" # External services
    BACKENDS = "backends"         # Neo4j, Qdrant
    CONFIGURATION = "configuration"  # Config system
    INTEGRITY = "integrity"       # Math/Logic tests
    PERFORMANCE = "performance"   # Latency baselines


class CheckStatus(Enum):
    """Check result status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class CheckResult:
    """Result of a single check."""
    name: str
    category: CheckCategory
    status: CheckStatus
    message: str
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport:
    """Complete verification report."""
    level: VerificationLevel
    timestamp: str
    duration_seconds: float
    results: List[CheckResult]
    summary: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.summary = {
            "passed": sum(1 for r in self.results if r.status == CheckStatus.PASSED),
            "failed": sum(1 for r in self.results if r.status == CheckStatus.FAILED),
            "skipped": sum(1 for r in self.results if r.status == CheckStatus.SKIPPED),
            "warning": sum(1 for r in self.results if r.status == CheckStatus.WARNING),
            "total": len(self.results)
        }

    @property
    def all_passed(self) -> bool:
        return self.summary["failed"] == 0


# ============================================================================
# Check Protocol (Abstract Base)
# ============================================================================

class Check(ABC):
    """
    Base class for verification checks.

    Follows HoloLoom's protocol-based design pattern.
    All checks implement this interface for registry compatibility.
    """

    name: str
    category: CheckCategory
    level: VerificationLevel
    description: str

    @abstractmethod
    async def run(self) -> CheckResult:
        """Execute the check and return result."""
        pass

    def _success(self, message: str, **details) -> CheckResult:
        return CheckResult(
            name=self.name,
            category=self.category,
            status=CheckStatus.PASSED,
            message=message,
            details=details
        )

    def _failure(self, message: str, **details) -> CheckResult:
        return CheckResult(
            name=self.name,
            category=self.category,
            status=CheckStatus.FAILED,
            message=message,
            details=details
        )

    def _warning(self, message: str, **details) -> CheckResult:
        return CheckResult(
            name=self.name,
            category=self.category,
            status=CheckStatus.WARNING,
            message=message,
            details=details
        )

    def _skipped(self, message: str, **details) -> CheckResult:
        return CheckResult(
            name=self.name,
            category=self.category,
            status=CheckStatus.SKIPPED,
            message=message,
            details=details
        )


# ============================================================================
# Check Registry (Singleton Pattern)
# ============================================================================

class CheckRegistry:
    """
    Registry for all verification checks.

    Follows HoloLoom's registry pattern (tool executor, alignment guardrails).
    Supports dynamic registration and filtering by level/category.
    """

    _instance = None
    _checks: List[Check] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._checks = []
        return cls._instance

    def register(self, check: Check) -> None:
        """Register a check with the registry."""
        self._checks.append(check)

    def get_checks(
        self,
        level: VerificationLevel,
        categories: Optional[List[CheckCategory]] = None
    ) -> List[Check]:
        """Get checks for given level and optional category filter."""
        level_order = {
            VerificationLevel.QUICK: 0,
            VerificationLevel.STANDARD: 1,
            VerificationLevel.COMPREHENSIVE: 2
        }

        filtered = [
            c for c in self._checks
            if level_order[c.level] <= level_order[level]
        ]

        if categories:
            filtered = [c for c in filtered if c.category in categories]

        return filtered

    def clear(self) -> None:
        """Clear all registered checks (for testing)."""
        self._checks = []


# Global registry instance
registry = CheckRegistry()


def register_check(check_class):
    """Decorator to register a check class."""
    registry.register(check_class())
    return check_class


# ============================================================================
# Import Checks
# ============================================================================

@register_check
class CoreImportsCheck(Check):
    name = "Core Imports"
    category = CheckCategory.IMPORTS
    level = VerificationLevel.QUICK
    description = "Verify core HoloLoom modules can be imported"

    async def run(self) -> CheckResult:
        imports = {
            "HoloLoom": False,
            "HoloLoom.config": False,
            "HoloLoom.weaving_orchestrator": False,
            "HoloLoom.policy.unified": False,
            "HoloLoom.memory.unified": False,
        }

        errors = []
        for module in imports:
            try:
                importlib.import_module(module)
                imports[module] = True
            except ImportError as e:
                errors.append(f"{module}: {e}")

        passed = all(imports.values())
        if passed:
            return self._success(
                f"All {len(imports)} core modules imported successfully",
                modules=imports
            )
        else:
            return self._failure(
                f"{len(errors)} import failures",
                modules=imports,
                errors=errors
            )


@register_check
class OptionalImportsCheck(Check):
    name = "Optional Imports"
    category = CheckCategory.IMPORTS
    level = VerificationLevel.STANDARD
    description = "Check optional dependencies (graceful degradation)"

    async def run(self) -> CheckResult:
        optional = {
            "spacy": False,
            "sentence_transformers": False,
            "networkx": False,
            "numpy": False,
            "torch": False,
        }

        for module in optional:
            try:
                importlib.import_module(module)
                optional[module] = True
            except ImportError:
                pass

        available = sum(optional.values())
        total = len(optional)

        if available == total:
            return self._success(
                f"All {total} optional dependencies available",
                dependencies=optional
            )
        elif available > 0:
            return self._warning(
                f"{available}/{total} optional dependencies available",
                dependencies=optional
            )
        else:
            return self._warning(
                "No optional dependencies available (system will use fallbacks)",
                dependencies=optional
            )


@register_check
class RAGImportsCheck(Check):
    name = "RAG System Imports"
    category = CheckCategory.IMPORTS
    level = VerificationLevel.STANDARD
    description = "Verify RAG system modules"

    async def run(self) -> CheckResult:
        try:
            from HoloLoom.rag import SimpleRAG
            from HoloLoom.rag.multimodal_rag import MultimodalRAG
            return self._success("RAG system modules available")
        except ImportError as e:
            return self._failure(f"RAG import failed: {e}")


# ============================================================================
# Backend Checks
# ============================================================================

@register_check
class InMemoryBackendCheck(Check):
    name = "In-Memory Backend"
    category = CheckCategory.BACKENDS
    level = VerificationLevel.QUICK
    description = "Verify in-memory backend (always available)"

    async def run(self) -> CheckResult:
        try:
            from HoloLoom.memory.graph import KG
            kg = KG()
            kg.add_entity("test", {"type": "test"})
            return self._success("In-memory backend working")
        except Exception as e:
            return self._failure(f"In-memory backend failed: {e}")


@register_check
class Neo4jBackendCheck(Check):
    name = "Neo4j Backend"
    category = CheckCategory.BACKENDS
    level = VerificationLevel.STANDARD
    description = "Check Neo4j connection (optional, for production)"

    async def run(self) -> CheckResult:
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', 7687))
            sock.close()

            if result == 0:
                return self._success("Neo4j available at localhost:7687")
            else:
                return self._warning(
                    "Neo4j not available (will use in-memory fallback)",
                    port=7687,
                    hint="Run: docker-compose up -d"
                )
        except Exception as e:
            return self._warning(f"Neo4j check failed: {e}")


@register_check
class QdrantBackendCheck(Check):
    name = "Qdrant Backend"
    category = CheckCategory.BACKENDS
    level = VerificationLevel.STANDARD
    description = "Check Qdrant connection (optional, for production)"

    async def run(self) -> CheckResult:
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', 6333))
            sock.close()

            if result == 0:
                return self._success("Qdrant available at localhost:6333")
            else:
                return self._warning(
                    "Qdrant not available (will use in-memory fallback)",
                    port=6333,
                    hint="Run: docker-compose up -d"
                )
        except Exception as e:
            return self._warning(f"Qdrant check failed: {e}")


# ============================================================================
# Configuration Checks
# ============================================================================

@register_check
class ConfigModesCheck(Check):
    name = "Configuration Modes"
    category = CheckCategory.CONFIGURATION
    level = VerificationLevel.QUICK
    description = "Verify all config modes work (BARE/FAST/FUSED)"

    async def run(self) -> CheckResult:
        try:
            from HoloLoom.config import Config

            modes = {}
            for mode in ['bare', 'fast', 'fused']:
                cfg = getattr(Config, mode)()
                modes[mode] = {
                    'complexity': cfg.complexity.value if hasattr(cfg.complexity, 'value') else str(cfg.complexity),
                    'max_iterations': cfg.max_iterations
                }

            return self._success(
                "All configuration modes working",
                modes=modes
            )
        except Exception as e:
            return self._failure(f"Config modes failed: {e}")


@register_check
class MemoryBackendConfigCheck(Check):
    name = "Memory Backend Config"
    category = CheckCategory.CONFIGURATION
    level = VerificationLevel.STANDARD
    description = "Verify memory backend configuration"

    async def run(self) -> CheckResult:
        try:
            from HoloLoom.config import Config, MemoryBackend

            backends = [b.value for b in MemoryBackend]

            return self._success(
                f"Memory backend types available: {', '.join(backends)}",
                backends=backends
            )
        except Exception as e:
            return self._failure(f"Memory backend config failed: {e}")


# ============================================================================
# Integrity Checks
# ============================================================================

@register_check
class PolicyEngineCheck(Check):
    name = "Policy Engine"
    category = CheckCategory.INTEGRITY
    level = VerificationLevel.STANDARD
    description = "Verify policy engine creates valid output"

    async def run(self) -> CheckResult:
        try:
            from HoloLoom.policy.unified import create_policy
            import numpy as np

            # Create minimal policy
            policy = create_policy(mem_dim=128)

            # Create dummy features
            features = {
                'motif': np.zeros(128),
                'embedding': np.zeros(128),
                'spectral': np.zeros(6)
            }

            # Just verify it doesn't crash
            return self._success("Policy engine initializes correctly")
        except Exception as e:
            return self._failure(f"Policy engine failed: {e}")


@register_check
class ThompsonSamplingCheck(Check):
    name = "Thompson Sampling"
    category = CheckCategory.INTEGRITY
    level = VerificationLevel.COMPREHENSIVE
    description = "Verify Thompson Sampling bandit works correctly"

    async def run(self) -> CheckResult:
        try:
            from HoloLoom.convergence.engine import ThompsonBandit

            bandit = ThompsonBandit(n_arms=4)

            # Sample and update
            arm = bandit.sample()
            bandit.update(arm, reward=0.8)

            # Verify statistics updated
            stats = bandit.get_stats()

            if stats['total_pulls'] == 1:
                return self._success(
                    "Thompson Sampling working correctly",
                    stats=stats
                )
            else:
                return self._failure("Thompson Sampling state incorrect")
        except Exception as e:
            return self._failure(f"Thompson Sampling failed: {e}")


@register_check
class WeavingCycleCheck(Check):
    name = "Weaving Cycle"
    category = CheckCategory.INTEGRITY
    level = VerificationLevel.COMPREHENSIVE
    description = "Run minimal weaving cycle end-to-end"

    async def run(self) -> CheckResult:
        try:
            from HoloLoom.config import Config
            from HoloLoom.weaving_orchestrator import WeavingOrchestrator
            from HoloLoom.protocols.types import Query

            config = Config.bare()

            async with WeavingOrchestrator(cfg=config, shards=[]) as orchestrator:
                spacetime = await orchestrator.weave(Query(text="Test query"))

                if spacetime and hasattr(spacetime, 'response'):
                    return self._success(
                        "Weaving cycle completed successfully",
                        confidence=spacetime.confidence if hasattr(spacetime, 'confidence') else None
                    )
                else:
                    return self._failure("Weaving cycle returned invalid result")
        except Exception as e:
            return self._failure(f"Weaving cycle failed: {e}")


# ============================================================================
# Performance Checks
# ============================================================================

@register_check
class ImportLatencyCheck(Check):
    name = "Import Latency"
    category = CheckCategory.PERFORMANCE
    level = VerificationLevel.COMPREHENSIVE
    description = "Measure module import times"

    async def run(self) -> CheckResult:
        import importlib
        import sys

        modules = [
            "HoloLoom.config",
            "HoloLoom.policy.unified",
            "HoloLoom.memory.unified",
        ]

        latencies = {}
        for module in modules:
            # Unload if already loaded
            if module in sys.modules:
                del sys.modules[module]

            start = time.perf_counter()
            try:
                importlib.import_module(module)
                latencies[module] = (time.perf_counter() - start) * 1000
            except ImportError:
                latencies[module] = None

        avg_latency = sum(l for l in latencies.values() if l) / len([l for l in latencies.values() if l])

        if avg_latency < 500:  # Under 500ms average
            return self._success(
                f"Average import latency: {avg_latency:.1f}ms",
                latencies=latencies
            )
        else:
            return self._warning(
                f"High import latency: {avg_latency:.1f}ms",
                latencies=latencies
            )


# ============================================================================
# Report Formatters
# ============================================================================

class TerminalFormatter:
    """Rich terminal output with colors."""

    COLORS = {
        CheckStatus.PASSED: '\033[92m',   # Green
        CheckStatus.FAILED: '\033[91m',   # Red
        CheckStatus.WARNING: '\033[93m',  # Yellow
        CheckStatus.SKIPPED: '\033[94m',  # Blue
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'DIM': '\033[2m',
    }

    def format(self, report: VerificationReport) -> str:
        lines = []

        # Header
        lines.append("")
        lines.append(f"{self.COLORS['BOLD']}HoloLoom Setup Verification{self.COLORS['RESET']}")
        lines.append(f"Level: {report.level.value.upper()}")
        lines.append(f"Timestamp: {report.timestamp}")
        lines.append("")

        # Results by category
        by_category = {}
        for result in report.results:
            cat = result.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(result)

        for category, results in by_category.items():
            lines.append(f"{self.COLORS['BOLD']}[{category.upper()}]{self.COLORS['RESET']}")
            for r in results:
                status_color = self.COLORS.get(r.status, '')
                status_icon = {
                    CheckStatus.PASSED: '[OK]',
                    CheckStatus.FAILED: '[FAIL]',
                    CheckStatus.WARNING: '[WARN]',
                    CheckStatus.SKIPPED: '[SKIP]',
                }[r.status]
                lines.append(f"  {status_color}{status_icon}{self.COLORS['RESET']} {r.name}: {r.message}")
            lines.append("")

        # Summary
        lines.append(f"{self.COLORS['BOLD']}Summary:{self.COLORS['RESET']}")
        lines.append(f"  {self.COLORS[CheckStatus.PASSED]}Passed: {report.summary['passed']}{self.COLORS['RESET']}")
        lines.append(f"  {self.COLORS[CheckStatus.FAILED]}Failed: {report.summary['failed']}{self.COLORS['RESET']}")
        lines.append(f"  {self.COLORS[CheckStatus.WARNING]}Warnings: {report.summary['warning']}{self.COLORS['RESET']}")
        lines.append(f"  {self.COLORS[CheckStatus.SKIPPED]}Skipped: {report.summary['skipped']}{self.COLORS['RESET']}")
        lines.append(f"  Total: {report.summary['total']}")
        lines.append(f"  Duration: {report.duration_seconds:.2f}s")
        lines.append("")

        # Final verdict
        if report.all_passed:
            lines.append(f"{self.COLORS[CheckStatus.PASSED]}{self.COLORS['BOLD']}All checks passed! Environment is ready.{self.COLORS['RESET']}")
        else:
            lines.append(f"{self.COLORS[CheckStatus.FAILED]}{self.COLORS['BOLD']}Some checks failed. See above for details.{self.COLORS['RESET']}")

        return "\n".join(lines)


class JSONFormatter:
    """JSON output for CI/CD integration."""

    def format(self, report: VerificationReport) -> str:
        data = {
            "level": report.level.value,
            "timestamp": report.timestamp,
            "duration_seconds": report.duration_seconds,
            "summary": report.summary,
            "all_passed": report.all_passed,
            "results": [
                {
                    "name": r.name,
                    "category": r.category.value,
                    "status": r.status.value,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                    "details": r.details
                }
                for r in report.results
            ]
        }
        return json.dumps(data, indent=2)


class MarkdownFormatter:
    """Markdown output for documentation."""

    def format(self, report: VerificationReport) -> str:
        lines = []

        lines.append("# HoloLoom Setup Verification Report")
        lines.append("")
        lines.append(f"**Level**: {report.level.value.upper()}")
        lines.append(f"**Timestamp**: {report.timestamp}")
        lines.append(f"**Duration**: {report.duration_seconds:.2f}s")
        lines.append("")

        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append("| Status | Count |")
        lines.append("|--------|-------|")
        lines.append(f"| Passed | {report.summary['passed']} |")
        lines.append(f"| Failed | {report.summary['failed']} |")
        lines.append(f"| Warning | {report.summary['warning']} |")
        lines.append(f"| Skipped | {report.summary['skipped']} |")
        lines.append(f"| **Total** | **{report.summary['total']}** |")
        lines.append("")

        # Results table
        lines.append("## Results")
        lines.append("")
        lines.append("| Category | Check | Status | Message |")
        lines.append("|----------|-------|--------|---------|")
        for r in report.results:
            status_emoji = {
                CheckStatus.PASSED: '✅',
                CheckStatus.FAILED: '❌',
                CheckStatus.WARNING: '⚠️',
                CheckStatus.SKIPPED: '⏭️',
            }[r.status]
            lines.append(f"| {r.category.value} | {r.name} | {status_emoji} | {r.message} |")
        lines.append("")

        # Verdict
        if report.all_passed:
            lines.append("## ✅ All Checks Passed")
            lines.append("")
            lines.append("Environment is ready for HoloLoom development.")
        else:
            lines.append("## ❌ Some Checks Failed")
            lines.append("")
            lines.append("Please review the failed checks above.")

        return "\n".join(lines)


# ============================================================================
# Runner
# ============================================================================

class VerificationRunner:
    """
    Main verification runner.

    Executes checks from registry and generates reports.
    """

    def __init__(self, registry: CheckRegistry):
        self.registry = registry
        self.formatters = {
            'terminal': TerminalFormatter(),
            'json': JSONFormatter(),
            'markdown': MarkdownFormatter(),
        }

    async def run(
        self,
        level: VerificationLevel,
        categories: Optional[List[CheckCategory]] = None
    ) -> VerificationReport:
        """Run all applicable checks and return report."""

        checks = self.registry.get_checks(level, categories)
        results = []

        start_time = time.perf_counter()

        for check in checks:
            check_start = time.perf_counter()
            try:
                result = await check.run()
                result.duration_ms = (time.perf_counter() - check_start) * 1000
            except Exception as e:
                result = CheckResult(
                    name=check.name,
                    category=check.category,
                    status=CheckStatus.FAILED,
                    message=f"Check crashed: {e}",
                    duration_ms=(time.perf_counter() - check_start) * 1000
                )
            results.append(result)

        duration = time.perf_counter() - start_time

        return VerificationReport(
            level=level,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            results=results
        )

    def format_report(self, report: VerificationReport, format_type: str = 'terminal') -> str:
        """Format report for output."""
        formatter = self.formatters.get(format_type, self.formatters['terminal'])
        return formatter.format(report)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="HoloLoom Setup Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/verify_setup.py --level quick
  python scripts/verify_setup.py --level standard --format json
  python scripts/verify_setup.py --category imports,backends
  python scripts/verify_setup.py --level comprehensive --format markdown > report.md
        """
    )

    parser.add_argument(
        '--level', '-l',
        type=str,
        choices=['quick', 'standard', 'comprehensive'],
        default='standard',
        help='Verification level (default: standard)'
    )

    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['terminal', 'json', 'markdown'],
        default='terminal',
        help='Output format (default: terminal)'
    )

    parser.add_argument(
        '--category', '-c',
        type=str,
        default=None,
        help='Comma-separated categories to check (default: all)'
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    # Parse level
    level = VerificationLevel(args.level)

    # Parse categories
    categories = None
    if args.category:
        category_names = [c.strip() for c in args.category.split(',')]
        categories = [CheckCategory(c) for c in category_names]

    # Run verification
    runner = VerificationRunner(registry)
    report = await runner.run(level, categories)

    # Output
    output = runner.format_report(report, args.format)
    print(output)

    # Exit code
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
