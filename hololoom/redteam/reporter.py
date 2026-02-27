"""
Report Generator for CARTS
==========================

Generates comprehensive vulnerability reports in Markdown format.

Features:
- Summary statistics (total, open, fixed, critical)
- Attack strategy effectiveness rankings
- Vulnerability details by severity
- Regression tracking
- Trend analysis

Author: CARTS (Continuous Adversarial Red Team System)
Date: 2025-12-01
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from .strategies import AttackStrategy
from .tracker import Vulnerability, VulnerabilityTracker, VulnStatus
from .bandit import RedTeamBandit


@dataclass
class ReportSection:
    """A section of the report."""

    title: str
    content: str
    priority: int = 0  # Higher = more important


class ReportGenerator:
    """
    Generates comprehensive vulnerability reports.

    Creates Markdown reports including:
    - Executive summary with critical findings
    - Attack strategy effectiveness rankings
    - Open vulnerabilities by severity
    - Fixed vulnerabilities (recent)
    - Regression tracking
    - Recommendations

    Example:
        tracker = VulnerabilityTracker()
        bandit = RedTeamBandit()
        reporter = ReportGenerator(tracker, bandit)

        # Generate full report
        report = reporter.generate()
        print(report)

        # Save to file
        reporter.save("vulnerability_report.md")
    """

    def __init__(
        self,
        tracker: VulnerabilityTracker,
        bandit: Optional[RedTeamBandit] = None
    ):
        """
        Initialize reporter.

        Args:
            tracker: VulnerabilityTracker with discovered vulnerabilities
            bandit: Optional RedTeamBandit for strategy statistics
        """
        self.tracker = tracker
        self.bandit = bandit

    def generate(self, include_details: bool = True) -> str:
        """
        Generate full vulnerability report.

        Args:
            include_details: Include detailed vulnerability listings

        Returns:
            Markdown-formatted report
        """
        sections = []

        # Header
        sections.append(self._generate_header())

        # Executive summary
        sections.append(self._generate_executive_summary())

        # Strategy effectiveness (if bandit available)
        if self.bandit:
            sections.append(self._generate_strategy_effectiveness())

        # Critical vulnerabilities
        critical = self.tracker.get_critical()
        if critical:
            sections.append(self._generate_critical_section(critical))

        # Open vulnerabilities
        if include_details:
            open_vulns = self.tracker.get_open()
            if open_vulns:
                sections.append(self._generate_open_section(open_vulns))

        # Needs attention
        attention = self.tracker.get_needing_attention()
        if attention:
            sections.append(self._generate_attention_section(attention))

        # Fixed vulnerabilities (recent)
        fixed = self.tracker.get_fixed()
        if fixed and include_details:
            sections.append(self._generate_fixed_section(fixed))

        # Regressions
        regressions = self.tracker.get_regressions()
        if regressions:
            sections.append(self._generate_regression_section(regressions))

        # Recommendations
        sections.append(self._generate_recommendations())

        # Footer
        sections.append(self._generate_footer())

        return "\n\n".join(s.content for s in sections)

    def generate_summary(self) -> str:
        """Generate a brief summary report (no details)."""
        return self.generate(include_details=False)

    def save(self, path: Path, include_details: bool = True):
        """
        Save report to file.

        Args:
            path: Output file path
            include_details: Include detailed vulnerability listings
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        report = self.generate(include_details=include_details)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)

    # =========================================================================
    # Section Generators
    # =========================================================================

    def _generate_header(self) -> ReportSection:
        """Generate report header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = f"""# CARTS Vulnerability Report

**Generated:** {timestamp}
**System:** HoloLoom Safety Framework
**Report Type:** Adversarial Red Team Assessment

---"""
        return ReportSection("Header", content, priority=100)

    def _generate_executive_summary(self) -> ReportSection:
        """Generate executive summary."""
        stats = self.tracker.get_stats()

        # Determine overall risk level
        critical_count = stats['by_severity'].get('critical', 0)
        high_count = stats['by_severity'].get('high', 0)
        open_count = stats.get('open', 0)

        if critical_count > 0:
            risk_level = "🔴 CRITICAL"
            risk_color = "red"
        elif high_count > 0:
            risk_level = "🟠 HIGH"
            risk_color = "orange"
        elif open_count > 0:
            risk_level = "🟡 MEDIUM"
            risk_color = "yellow"
        else:
            risk_level = "🟢 LOW"
            risk_color = "green"

        content = f"""## Executive Summary

**Overall Risk Level:** {risk_level}

### Key Metrics

| Metric | Value |
|--------|-------|
| Total Vulnerabilities | {stats['total']} |
| Open | {stats.get('open', 0)} |
| Fixed | {stats.get('fixed', 0)} |
| Critical Severity | {critical_count} |
| High Severity | {high_count} |
| Medium Severity | {stats['by_severity'].get('medium', 0)} |
| Low Severity | {stats['by_severity'].get('low', 0)} |
| Total Regressions | {stats.get('regression_count', 0)} |"""

        if stats['total'] > 0:
            content += f"""
| Average Severity | {stats.get('avg_severity', 0):.2f} |
| Oldest Open (days) | {stats.get('oldest_open_days', 0):.1f} |"""

        return ReportSection("Executive Summary", content, priority=90)

    def _generate_strategy_effectiveness(self) -> ReportSection:
        """Generate attack strategy effectiveness section."""
        if not self.bandit:
            return ReportSection("Strategy Effectiveness", "", priority=0)

        stats = self.bandit.get_stats()
        arm_summary = self.bandit.get_arm_summary()

        content = """## Attack Strategy Effectiveness

### Overall Statistics

| Metric | Value |
|--------|-------|
| Total Attacks | {total_pulls} |
| Total Vulnerabilities Found | {total_rewards:.1f} |
| Overall Success Rate | {success_rate:.1%} |
| Best Strategy | {best_strategy} |

### Strategy Rankings

| Rank | Strategy | Expected Reward | Success Rate | Pulls | Uncertainty |
|------|----------|-----------------|--------------|-------|-------------|""".format(
            total_pulls=stats['total_pulls'],
            total_rewards=stats['total_rewards'],
            success_rate=stats['overall_success_rate'],
            best_strategy=stats['best_strategy'] or 'N/A'
        )

        for rank, arm in enumerate(arm_summary, 1):
            content += f"""
| {rank} | {arm['strategy']} | {arm['expected_reward']:.3f} | {arm['success_rate']:.1%} | {arm['pulls']} | {arm['uncertainty']:.3f} |"""

        return ReportSection("Strategy Effectiveness", content, priority=80)

    def _generate_critical_section(
        self,
        vulnerabilities: List[Vulnerability]
    ) -> ReportSection:
        """Generate critical vulnerabilities section."""
        content = """## 🔴 Critical Vulnerabilities

**Action Required:** These vulnerabilities require immediate attention.

"""
        for vuln in vulnerabilities:
            content += self._format_vulnerability(vuln, detailed=True)
            content += "\n---\n"

        return ReportSection("Critical", content, priority=95)

    def _generate_open_section(
        self,
        vulnerabilities: List[Vulnerability]
    ) -> ReportSection:
        """Generate open vulnerabilities section."""
        # Sort by severity (descending)
        sorted_vulns = sorted(vulnerabilities, key=lambda v: v.severity, reverse=True)

        content = f"""## Open Vulnerabilities

**Total Open:** {len(sorted_vulns)}

"""
        for vuln in sorted_vulns[:20]:  # Limit to top 20
            content += self._format_vulnerability(vuln, detailed=False)
            content += "\n"

        if len(sorted_vulns) > 20:
            content += f"\n*... and {len(sorted_vulns) - 20} more*\n"

        return ReportSection("Open", content, priority=70)

    def _generate_attention_section(
        self,
        vulnerabilities: List[Vulnerability]
    ) -> ReportSection:
        """Generate needs-attention section."""
        content = """## ⚠️ Needs Immediate Attention

These vulnerabilities are either critical or have been open for >7 days.

"""
        for vuln in vulnerabilities:
            reason = []
            if vuln.is_critical:
                reason.append("Critical severity")
            if vuln.age_days > 7:
                reason.append(f"Open {vuln.age_days:.0f} days")

            content += f"### {vuln.vuln_id}\n"
            content += f"**Reason:** {', '.join(reason)}\n"
            content += self._format_vulnerability(vuln, detailed=True)
            content += "\n---\n"

        return ReportSection("Attention", content, priority=85)

    def _generate_fixed_section(
        self,
        vulnerabilities: List[Vulnerability]
    ) -> ReportSection:
        """Generate fixed vulnerabilities section."""
        # Sort by fix date (most recent first)
        sorted_vulns = sorted(
            vulnerabilities,
            key=lambda v: v.fixed_at or datetime.min,
            reverse=True
        )

        content = f"""## Fixed Vulnerabilities

**Total Fixed:** {len(sorted_vulns)}

| ID | Strategy | Severity | Fixed At | Verified |
|----|----------|----------|----------|----------|"""

        for vuln in sorted_vulns[:10]:  # Last 10 fixes
            fixed_at = vuln.fixed_at.strftime("%Y-%m-%d") if vuln.fixed_at else "N/A"
            verified = "✅" if vuln.fix_verified else "⏳"
            content += f"""
| {vuln.vuln_id} | {vuln.strategy.value} | {vuln.severity:.2f} | {fixed_at} | {verified} |"""

        return ReportSection("Fixed", content, priority=50)

    def _generate_regression_section(
        self,
        vulnerabilities: List[Vulnerability]
    ) -> ReportSection:
        """Generate regression tracking section."""
        content = """## 🔄 Regressions Detected

These vulnerabilities were fixed but have reappeared.

| ID | Strategy | Regression Count | Last Regression | Current Status |
|----|----------|------------------|-----------------|----------------|"""

        for vuln in vulnerabilities:
            last_regression = (
                vuln.last_regression_at.strftime("%Y-%m-%d")
                if vuln.last_regression_at else "N/A"
            )
            content += f"""
| {vuln.vuln_id} | {vuln.strategy.value} | {vuln.regression_count} | {last_regression} | {vuln.status.value} |"""

        return ReportSection("Regressions", content, priority=75)

    def _generate_recommendations(self) -> ReportSection:
        """Generate recommendations based on findings."""
        stats = self.tracker.get_stats()
        recommendations = []

        # Critical vulnerabilities
        critical_count = stats['by_severity'].get('critical', 0)
        if critical_count > 0:
            recommendations.append(
                f"🔴 **Address {critical_count} critical vulnerabilities immediately.** "
                "These represent the highest risk to system safety."
            )

        # Old vulnerabilities
        oldest = stats.get('oldest_open_days', 0)
        if oldest > 14:
            recommendations.append(
                f"⏰ **Oldest open vulnerability is {oldest:.0f} days old.** "
                "Consider prioritizing or marking as wontfix if not actionable."
            )

        # Regressions
        regression_count = stats.get('regression_count', 0)
        if regression_count > 0:
            recommendations.append(
                f"🔄 **{regression_count} regressions detected.** "
                "Review fix implementation and add regression tests."
            )

        # Strategy insights (if bandit available)
        if self.bandit:
            bandit_stats = self.bandit.get_stats()
            if bandit_stats['best_strategy']:
                recommendations.append(
                    f"🎯 **'{bandit_stats['best_strategy']}' is most effective.** "
                    "Consider strengthening defenses against this attack type."
                )

            # High uncertainty strategies
            exploration = self.bandit.get_exploration_candidates(uncertainty_threshold=0.1)
            if exploration:
                strategies = ', '.join(s.value for s in exploration[:3])
                recommendations.append(
                    f"🔍 **Strategies with high uncertainty: {strategies}.** "
                    "More testing needed to understand effectiveness."
                )

        # No issues
        if not recommendations:
            recommendations.append(
                "✅ **No critical issues found.** "
                "Continue regular red team testing to maintain security posture."
            )

        content = "## Recommendations\n\n"
        for i, rec in enumerate(recommendations, 1):
            content += f"{i}. {rec}\n\n"

        return ReportSection("Recommendations", content, priority=60)

    def _generate_footer(self) -> ReportSection:
        """Generate report footer."""
        content = """---

## About This Report

This report was generated by **CARTS** (Continuous Adversarial Red Team System),
an automated security testing system that continuously probes HoloLoom's safety
guardrails for vulnerabilities.

**Key Components:**
- Thompson Sampling bandit for strategy selection
- Genetic algorithm payload evolution
- Integration with SafetyGuardrails, DeceptionDetector, and InstrumentalConvergenceGuard
- Automated vulnerability tracking and regression detection

**Report Coverage:**
- All discovered vulnerabilities
- Attack strategy effectiveness analysis
- Regression tracking
- Actionable recommendations

*For questions or to report false positives, contact the HoloLoom security team.*"""

        return ReportSection("Footer", content, priority=0)

    # =========================================================================
    # Formatting Helpers
    # =========================================================================

    def _format_vulnerability(
        self,
        vuln: Vulnerability,
        detailed: bool = False
    ) -> str:
        """Format a single vulnerability."""
        severity_emoji = self._severity_emoji(vuln.severity)

        if detailed:
            return f"""### {vuln.vuln_id} {severity_emoji}

**Strategy:** {vuln.strategy.value}
**Severity:** {vuln.severity:.2f} ({vuln.severity_level.value})
**Status:** {vuln.status.value}
**Discovered:** {vuln.discovered_at.strftime("%Y-%m-%d %H:%M")}
**Age:** {vuln.age_days:.1f} days

**Description:**
{vuln.description}

**Payload:**
```
{vuln.payload[:500]}{'...' if len(vuln.payload) > 500 else ''}
```
"""
        else:
            return f"- **{vuln.vuln_id}** {severity_emoji} {vuln.strategy.value} - {vuln.severity:.2f} ({vuln.status.value})"

    def _severity_emoji(self, severity: float) -> str:
        """Get emoji for severity level."""
        if severity >= 0.9:
            return "🔴"
        elif severity >= 0.7:
            return "🟠"
        elif severity >= 0.4:
            return "🟡"
        elif severity >= 0.2:
            return "🔵"
        else:
            return "⚪"


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_report(
    tracker: VulnerabilityTracker,
    bandit: Optional[RedTeamBandit] = None,
    include_details: bool = True
) -> str:
    """Generate a vulnerability report."""
    reporter = ReportGenerator(tracker, bandit)
    return reporter.generate(include_details=include_details)


def save_report(
    tracker: VulnerabilityTracker,
    path: Path,
    bandit: Optional[RedTeamBandit] = None,
    include_details: bool = True
):
    """Generate and save a vulnerability report."""
    reporter = ReportGenerator(tracker, bandit)
    reporter.save(path, include_details=include_details)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ReportSection',
    'ReportGenerator',
    'generate_report',
    'save_report',
]
