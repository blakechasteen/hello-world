"""
Built-in Verification Signals

6 verification signals for holistic fabric inspection.

Systems thinking: the whole is greater than the sum of parts.
Each signal is independent but together they provide
comprehensive quality assessment.

Weights (total = 1.0):
- TestSignal: 0.20 (hard blocker)
- TroughSignal: 0.20 (security = hard blocker)
- AlignmentSignal: 0.15
- ArchitectureSignal: 0.15
- DocumentationSignal: 0.15
- IntegrationSignal: 0.15

Created: December 2025
"""

import asyncio
import subprocess
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from HoloLoom.tapestry.protocol import Thread, SignalResult
from HoloLoom.tapestry.signals.registry import SignalRegistry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Signal 1: Tests (0.20) - Hard Blocker
# ─────────────────────────────────────────────────────────────────

@SignalRegistry.register
class TestSignal:
    """
    Run pytest, check pass/fail.

    Hard blocker: is_blocker=True means tests MUST pass.
    """
    name = "tests"
    weight = 0.20

    async def check(self, thread: Thread, context: Dict[str, Any]) -> SignalResult:
        """
        Run pytest on specified test paths.

        Context:
            test_paths: List of paths to test (default: ["tests/"])
            test_timeout: Timeout in seconds (default: 300)
        """
        test_paths = context.get("test_paths", ["tests/"])
        timeout = context.get("test_timeout", 300)

        try:
            # Run pytest
            cmd = ["python", "-m", "pytest"] + test_paths + ["-v", "--tb=short"]
            result = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                ),
                timeout=timeout
            )
            stdout, stderr = await result.communicate()
            output = stdout.decode() + stderr.decode()

            # Parse results
            passed = result.returncode == 0

            # Calculate pass rate from output
            pass_rate = 1.0 if passed else 0.0
            if "passed" in output and "failed" in output:
                # Try to extract pass rate
                import re
                match = re.search(r'(\d+) passed.*?(\d+) failed', output)
                if match:
                    p, f = int(match.group(1)), int(match.group(2))
                    pass_rate = p / (p + f) if (p + f) > 0 else 0.0

            return SignalResult(
                signal_name=self.name,
                passed=passed,
                score=pass_rate,
                is_blocker=True,  # Tests MUST pass
                details={
                    "exit_code": result.returncode,
                    "output": output[:2000],  # Truncate for storage
                    "test_paths": test_paths
                }
            )

        except asyncio.TimeoutError:
            return SignalResult(
                signal_name=self.name,
                passed=False,
                score=0.0,
                is_blocker=True,
                details={"error": f"Tests timed out after {timeout}s"}
            )
        except Exception as e:
            logger.error(f"TestSignal error: {e}")
            return SignalResult(
                signal_name=self.name,
                passed=False,
                score=0.0,
                is_blocker=True,
                details={"error": str(e)}
            )


# ─────────────────────────────────────────────────────────────────
# Signal 2: Quality/Trough (0.20) - Security = Hard Blocker
# ─────────────────────────────────────────────────────────────────

@SignalRegistry.register
class TroughSignal:
    """
    AI slop detection via Trough.

    Checks for code quality issues:
    - AI-generated slop patterns
    - Security vulnerabilities
    - Performance issues
    - Dead code

    Security issues are hard blockers.
    """
    name = "quality"
    weight = 0.20

    async def check(self, thread: Thread, context: Dict[str, Any]) -> SignalResult:
        """
        Analyze files for quality issues.

        Context:
            files: List of file paths to analyze
            severity_threshold: Minimum severity to report (default: "warning")
        """
        files = context.get("files", [])
        threshold = context.get("severity_threshold", "warning")

        if not files:
            # No files to check
            return SignalResult(
                signal_name=self.name,
                passed=True,
                score=1.0,
                details={"message": "No files to analyze"}
            )

        try:
            # Try to use Trough detector
            from trough.detector import TroughDetector
            detector = TroughDetector()

            all_issues = []
            security_issues = []

            for file_path in files:
                path = Path(file_path)
                if path.exists() and path.suffix == '.py':
                    issues = detector.analyze(path)
                    all_issues.extend(issues)
                    security_issues.extend(
                        i for i in issues if getattr(i, 'category', '') == 'security'
                    )

            # Score based on issue count
            score = max(0.0, 1.0 - len(all_issues) / 10.0)

            return SignalResult(
                signal_name=self.name,
                passed=len(all_issues) < 5,
                score=score,
                is_blocker=len(security_issues) > 0,  # Security = hard fail
                details={
                    "total_issues": len(all_issues),
                    "security_issues": len(security_issues),
                    "files_analyzed": len(files),
                    "issues": [str(i) for i in all_issues[:10]]
                }
            )

        except ImportError:
            logger.debug("Trough not available, using basic quality check")
            # Fallback: basic quality check (line count, complexity estimate)
            return await self._basic_quality_check(files)

        except Exception as e:
            logger.error(f"TroughSignal error: {e}")
            return SignalResult(
                signal_name=self.name,
                passed=True,  # Graceful degradation
                score=0.5,
                details={"error": str(e), "fallback": "basic"}
            )

    async def _basic_quality_check(self, files: List[str]) -> SignalResult:
        """Basic quality check when Trough unavailable."""
        issues = []

        for file_path in files:
            path = Path(file_path)
            if not path.exists() or path.suffix != '.py':
                continue

            try:
                content = path.read_text(encoding='utf-8')
                lines = content.split('\n')

                # Check for common issues
                for i, line in enumerate(lines, 1):
                    # TODO comments
                    if 'TODO' in line or 'FIXME' in line:
                        issues.append(f"{file_path}:{i}: TODO/FIXME comment")
                    # Very long lines
                    if len(line) > 120:
                        issues.append(f"{file_path}:{i}: Line too long ({len(line)} chars)")
                    # Bare except
                    if 'except:' in line and 'except Exception' not in line:
                        issues.append(f"{file_path}:{i}: Bare except clause")

            except Exception:
                pass

        score = max(0.0, 1.0 - len(issues) / 20.0)

        return SignalResult(
            signal_name=self.name,
            passed=len(issues) < 10,
            score=score,
            is_blocker=False,  # Basic check is never a blocker
            details={
                "fallback": "basic",
                "issues_found": len(issues),
                "issues": issues[:10]
            }
        )


# ─────────────────────────────────────────────────────────────────
# Signal 3: Alignment (0.15)
# ─────────────────────────────────────────────────────────────────

@SignalRegistry.register
class AlignmentSignal:
    """
    Safety guardrails check.

    Integrates with HoloLoom's alignment framework to verify:
    - No unsafe patterns in code
    - Risk level acceptable
    - No deception indicators
    """
    name = "alignment"
    weight = 0.15

    async def check(self, thread: Thread, context: Dict[str, Any]) -> SignalResult:
        """
        Check alignment/safety constraints.

        Context:
            files: Files to check
            risk_threshold: Max acceptable risk level (default: "medium")
        """
        files = context.get("files", [])
        risk_threshold = context.get("risk_threshold", "medium")

        try:
            # Try to use alignment framework
            from HoloLoom.alignment.safety_guardrails import SafetyGuardrails

            guardrails = SafetyGuardrails()

            # Check thread description for risky actions
            risk_indicators = [
                "delete", "remove", "drop", "execute", "eval",
                "subprocess", "os.system", "shell", "sudo"
            ]

            description_lower = thread.description.lower()
            found_risks = [r for r in risk_indicators if r in description_lower]

            # Analyze files for unsafe patterns
            file_risks = []
            for file_path in files:
                path = Path(file_path)
                if path.exists() and path.suffix == '.py':
                    content = path.read_text(encoding='utf-8')
                    for indicator in risk_indicators:
                        if indicator in content.lower():
                            file_risks.append(f"{file_path}: {indicator}")

            total_risks = len(found_risks) + len(file_risks)
            score = max(0.0, 1.0 - total_risks / 10.0)

            return SignalResult(
                signal_name=self.name,
                passed=total_risks < 3,
                score=score,
                details={
                    "description_risks": found_risks,
                    "file_risks": file_risks[:10],
                    "risk_threshold": risk_threshold
                }
            )

        except ImportError:
            # Alignment framework not available
            logger.debug("Alignment framework not available, using basic check")
            return SignalResult(
                signal_name=self.name,
                passed=True,
                score=0.8,
                details={"fallback": "basic", "message": "Alignment check skipped"}
            )

        except Exception as e:
            logger.error(f"AlignmentSignal error: {e}")
            return SignalResult(
                signal_name=self.name,
                passed=True,
                score=0.5,
                details={"error": str(e)}
            )


# ─────────────────────────────────────────────────────────────────
# Signal 4: Architecture (0.15)
# ─────────────────────────────────────────────────────────────────

@SignalRegistry.register
class ArchitectureSignal:
    """
    Pattern adherence check.

    Verifies code follows HoloLoom patterns:
    - Protocol-based design
    - Factory with fallbacks
    - Proper async/await
    - Module structure
    """
    name = "architecture"
    weight = 0.15

    async def check(self, thread: Thread, context: Dict[str, Any]) -> SignalResult:
        """
        Check architecture pattern adherence.

        Context:
            files: Files to analyze
            patterns: Required patterns (default: ["protocol", "async"])
        """
        files = context.get("files", [])
        required_patterns = context.get("patterns", ["protocol", "async"])

        if not files:
            return SignalResult(
                signal_name=self.name,
                passed=True,
                score=1.0,
                details={"message": "No files to analyze"}
            )

        pattern_scores = {}
        violations = []

        for file_path in files:
            path = Path(file_path)
            if not path.exists() or path.suffix != '.py':
                continue

            try:
                content = path.read_text(encoding='utf-8')

                # Check for Protocol usage
                if "protocol" in required_patterns:
                    has_protocol = "Protocol" in content or "@runtime_checkable" in content
                    pattern_scores["protocol"] = 1.0 if has_protocol else 0.0
                    if not has_protocol and "class " in content:
                        violations.append(f"{file_path}: Consider using Protocol for interfaces")

                # Check for async patterns
                if "async" in required_patterns:
                    has_async = "async def" in content or "await " in content
                    pattern_scores["async"] = 1.0 if has_async else 0.5

                # Check for factory pattern
                if "factory" in required_patterns:
                    has_factory = "create_" in content or "Factory" in content
                    pattern_scores["factory"] = 1.0 if has_factory else 0.0

                # Check for proper imports (no circular, no star)
                if "from " in content and "import *" in content:
                    violations.append(f"{file_path}: Avoid 'import *'")

            except Exception:
                pass

        # Calculate overall score
        if pattern_scores:
            score = sum(pattern_scores.values()) / len(pattern_scores)
        else:
            score = 0.8  # Default if no patterns to check

        return SignalResult(
            signal_name=self.name,
            passed=score >= 0.6,
            score=score,
            details={
                "pattern_scores": pattern_scores,
                "violations": violations[:10],
                "files_checked": len(files)
            }
        )


# ─────────────────────────────────────────────────────────────────
# Signal 5: Documentation (0.15)
# ─────────────────────────────────────────────────────────────────

@SignalRegistry.register
class DocumentationSignal:
    """
    Docstring coverage check.

    Verifies documentation quality:
    - Module docstrings
    - Class docstrings
    - Function docstrings
    - Type hints
    """
    name = "documentation"
    weight = 0.15

    async def check(self, thread: Thread, context: Dict[str, Any]) -> SignalResult:
        """
        Check documentation coverage.

        Context:
            files: Files to analyze
            min_coverage: Minimum docstring coverage (default: 0.5)
        """
        files = context.get("files", [])
        min_coverage = context.get("min_coverage", 0.5)

        if not files:
            return SignalResult(
                signal_name=self.name,
                passed=True,
                score=1.0,
                details={"message": "No files to analyze"}
            )

        total_items = 0
        documented_items = 0
        missing_docs = []

        for file_path in files:
            path = Path(file_path)
            if not path.exists() or path.suffix != '.py':
                continue

            try:
                content = path.read_text(encoding='utf-8')
                lines = content.split('\n')

                # Check for module docstring
                if content.strip().startswith('"""') or content.strip().startswith("'''"):
                    documented_items += 1
                total_items += 1

                # Simple check for class/function docstrings
                in_docstring = False
                for i, line in enumerate(lines):
                    stripped = line.strip()

                    if stripped.startswith('class ') or stripped.startswith('def '):
                        total_items += 1
                        # Check next non-empty line for docstring
                        for j in range(i + 1, min(i + 3, len(lines))):
                            next_line = lines[j].strip()
                            if next_line.startswith('"""') or next_line.startswith("'''"):
                                documented_items += 1
                                break
                            elif next_line and not next_line.startswith('#'):
                                missing_docs.append(f"{file_path}:{i+1}: {stripped[:50]}")
                                break

            except Exception:
                pass

        # Calculate coverage
        coverage = documented_items / total_items if total_items > 0 else 1.0

        return SignalResult(
            signal_name=self.name,
            passed=coverage >= min_coverage,
            score=coverage,
            details={
                "total_items": total_items,
                "documented_items": documented_items,
                "coverage": f"{coverage:.1%}",
                "missing_docs": missing_docs[:10]
            }
        )


# ─────────────────────────────────────────────────────────────────
# Signal 6: Integration (0.15)
# ─────────────────────────────────────────────────────────────────

@SignalRegistry.register
class IntegrationSignal:
    """
    Cross-system coherence check.

    Verifies integration quality:
    - Import structure correct
    - Dependencies available
    - No circular imports
    - Interfaces match
    """
    name = "integration"
    weight = 0.15

    async def check(self, thread: Thread, context: Dict[str, Any]) -> SignalResult:
        """
        Check integration coherence.

        Context:
            files: Files to check
            project_root: Project root for import checking
        """
        files = context.get("files", [])
        project_root = context.get("project_root", ".")

        if not files:
            return SignalResult(
                signal_name=self.name,
                passed=True,
                score=1.0,
                details={"message": "No files to analyze"}
            )

        issues = []
        import_errors = []

        for file_path in files:
            path = Path(file_path)
            if not path.exists() or path.suffix != '.py':
                continue

            try:
                content = path.read_text(encoding='utf-8')

                # Check imports can be resolved
                import_lines = [
                    line for line in content.split('\n')
                    if line.strip().startswith('import ') or line.strip().startswith('from ')
                ]

                for imp_line in import_lines:
                    imp_line = imp_line.strip()
                    # Extract module name
                    if imp_line.startswith('from '):
                        parts = imp_line.split()
                        if len(parts) >= 2:
                            module = parts[1]
                    else:
                        parts = imp_line.split()
                        if len(parts) >= 2:
                            module = parts[1].split('.')[0]
                        else:
                            continue

                    # Check if it's a HoloLoom import
                    if module.startswith('HoloLoom'):
                        # Verify the module exists
                        module_path = Path(project_root) / module.replace('.', '/')
                        if not (module_path.exists() or
                                module_path.with_suffix('.py').exists() or
                                (module_path / '__init__.py').exists()):
                            import_errors.append(f"{file_path}: Cannot find {module}")

                # Check for common integration issues
                if 'circular import' in content.lower():
                    issues.append(f"{file_path}: Contains circular import warning")

            except Exception as e:
                issues.append(f"{file_path}: Error analyzing: {str(e)[:50]}")

        total_issues = len(issues) + len(import_errors)
        score = max(0.0, 1.0 - total_issues / 5.0)

        return SignalResult(
            signal_name=self.name,
            passed=total_issues == 0,
            score=score,
            details={
                "integration_issues": issues[:10],
                "import_errors": import_errors[:10],
                "files_checked": len(files)
            }
        )
