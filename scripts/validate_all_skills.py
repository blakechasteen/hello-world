#!/usr/bin/env python3
"""
CI/CD Pipeline: Validate all skills

Runs all quality gates:
1. Schema validation
2. Security analysis
3. Testing
4. Token budget analysis

Usage:
    python scripts/validate_all_skills.py
    python scripts/validate_all_skills.py --category domain
    python scripts/validate_all_skills.py --fail-fast
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional


@dataclass
class ValidationResult:
    """Result of validating a single skill"""
    skill_name: str
    category: str
    passed: bool
    gates_passed: Dict[str, bool]
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, any]


class SkillValidator:
    """Validates all skills through quality gates"""

    def __init__(self, skills_root: Path = Path("skills")):
        self.skills_root = skills_root
        self.results: List[ValidationResult] = []

    def validate_all(
        self,
        category: Optional[str] = None,
        fail_fast: bool = False
    ) -> bool:
        """Validate all skills in given category"""

        # Discover skills
        skills = self._discover_skills(category)
        if not skills:
            print("[!] No skills found to validate")
            return True

        print(f"[*] Found {len(skills)} skill(s) to validate")
        print("=" * 60)

        all_passed = True
        for skill_path in skills:
            result = self._validate_skill(skill_path)
            self.results.append(result)

            if not result.passed:
                all_passed = False
                if fail_fast:
                    print(f"\n[FAIL] Validation failed (fail-fast mode)")
                    break

        # Print summary
        self._print_summary()

        return all_passed

    def _discover_skills(self, category: Optional[str] = None) -> List[Path]:
        """Discover all skill directories"""
        skills = []

        categories = [category] if category else ["meta", "domain", "utility"]

        for cat in categories:
            cat_dir = self.skills_root / cat
            if not cat_dir.exists():
                continue

            for skill_dir in cat_dir.iterdir():
                if skill_dir.is_dir() and (skill_dir / "skill.markdown").exists():
                    skills.append(skill_dir)

        return sorted(skills)

    def _validate_skill(self, skill_path: Path) -> ValidationResult:
        """Validate a single skill through all gates"""

        skill_name = skill_path.name
        category = skill_path.parent.name

        print(f"\n[>] Validating: {category}/{skill_name}")
        print("-" * 60)

        gates_passed = {}
        errors = []
        warnings = []
        metrics = {}

        # Gate 1: Schema Validation
        print("  [1/4] Schema validation...", end=" ")
        schema_result = self._validate_schema(skill_path)
        gates_passed["schema"] = schema_result["passed"]
        if schema_result["passed"]:
            print("[OK]")
        else:
            print("[FAIL]")
            errors.extend(schema_result["errors"])
        warnings.extend(schema_result.get("warnings", []))

        # Gate 2: Security Analysis (meta-skill)
        print("  [2/4] Security analysis...", end=" ")
        security_result = self._run_security_analysis(skill_path)
        gates_passed["security"] = security_result["passed"]
        if security_result["passed"]:
            print("[OK]")
        else:
            print("[FAIL]")
            errors.extend(security_result["errors"])
        warnings.extend(security_result.get("warnings", []))

        # Gate 3: Testing (meta-skill)
        print("  [3/4] Testing...", end=" ")
        test_result = self._run_tests(skill_path)
        gates_passed["testing"] = test_result["passed"]
        if test_result["passed"]:
            print("[OK]")
        else:
            print("[FAIL]")
            errors.extend(test_result["errors"])
        warnings.extend(test_result.get("warnings", []))

        # Gate 4: Token Budget Analysis (meta-skill)
        print("  [4/4] Token budget analysis...", end=" ")
        token_result = self._analyze_token_budget(skill_path)
        gates_passed["token_budget"] = token_result["passed"]
        metrics["token_count"] = token_result.get("token_count", 0)
        if token_result["passed"]:
            print(f"[OK] ({metrics['token_count']} tokens)")
        else:
            print(f"[FAIL] ({metrics['token_count']} tokens)")
            errors.extend(token_result["errors"])
        warnings.extend(token_result.get("warnings", []))

        # Overall result
        passed = all(gates_passed.values())

        if passed:
            print(f"\n  [OK] {skill_name}: All gates passed")
        else:
            print(f"\n  [FAIL] {skill_name}: Validation failed")
            for gate, result in gates_passed.items():
                if not result:
                    print(f"     Failed gate: {gate}")

        return ValidationResult(
            skill_name=skill_name,
            category=category,
            passed=passed,
            gates_passed=gates_passed,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )

    def _validate_schema(self, skill_path: Path) -> Dict:
        """Gate 1: Validate skill.markdown schema"""
        skill_file = skill_path / "skill.markdown"

        if not skill_file.exists():
            return {
                "passed": False,
                "errors": ["skill.markdown not found"]
            }

        content = skill_file.read_text(encoding="utf-8")

        # Check required sections
        required_sections = [
            "## Metadata",
            "## Description",
            "## Required Capabilities",
            "## Input Schema",
            "## Output Schema",
            "## Prompt Template",
            "## Examples",
            "## Testing Checklist",
            "## Security Considerations"
        ]

        errors = []
        warnings = []

        for section in required_sections:
            if section not in content:
                errors.append(f"Missing required section: {section}")

        # Check metadata fields
        if "- **Name**:" not in content:
            errors.append("Missing metadata: Name")
        if "- **Version**:" not in content:
            errors.append("Missing metadata: Version")
        if "- **Category**:" not in content:
            errors.append("Missing metadata: Category")

        # Check for placeholders
        placeholders = [
            "[SKILL_NAME]",
            "[skill_identifier]",
            "[Your Name]",
            "[YYYY-MM-DD]",
            "[meta|domain|utility]"
        ]

        for placeholder in placeholders:
            if placeholder in content:
                warnings.append(f"Template placeholder not replaced: {placeholder}")

        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def _run_security_analysis(self, skill_path: Path) -> Dict:
        """Gate 2: Run skill_security_analyzer meta-skill"""

        # For now, basic security checks
        # In production, this would invoke the actual meta-skill

        skill_file = skill_path / "skill.markdown"
        content = skill_file.read_text(encoding="utf-8")

        errors = []
        warnings = []

        # Check for potential prompt injection vectors
        dangerous_patterns = [
            "ignore previous instructions",
            "disregard all",
            "system prompt:",
            "you are now"
        ]

        for pattern in dangerous_patterns:
            if pattern.lower() in content.lower():
                errors.append(f"Potential prompt injection: '{pattern}'")

        # Check capability declarations
        if "## Required Capabilities" in content:
            if "file_write" in content or "file_delete" in content:
                warnings.append("Skill requires write/delete capabilities - review carefully")
            if "bash_exec" in content or "python_exec" in content:
                warnings.append("Skill requires code execution - review carefully")

        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def _run_tests(self, skill_path: Path) -> Dict:
        """Gate 3: Run skill_tester meta-skill"""

        # For now, check that examples exist
        # In production, this would invoke the actual meta-skill

        skill_file = skill_path / "skill.markdown"
        content = skill_file.read_text(encoding="utf-8")

        errors = []
        warnings = []

        # Count examples
        example_count = content.count("### Example")

        if example_count < 3:
            errors.append(f"Insufficient examples: {example_count} (minimum 3 required)")

        # Check for test checklist
        if "## Testing Checklist" not in content:
            errors.append("Missing testing checklist")
        else:
            # Check that at least some items are checked
            checked_count = content.count("- [x]")
            if checked_count == 0:
                warnings.append("No testing checklist items completed")

        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def _analyze_token_budget(self, skill_path: Path) -> Dict:
        """Gate 4: Run token_budget_adviser meta-skill"""

        # For now, simple token counting
        # In production, this would use actual tokenizer

        skill_file = skill_path / "skill.markdown"
        content = skill_file.read_text(encoding="utf-8")

        # Extract prompt template section
        prompt_start = content.find("## Prompt Template")
        if prompt_start == -1:
            return {
                "passed": False,
                "errors": ["Missing Prompt Template section"]
            }

        prompt_end = content.find("## Examples", prompt_start)
        if prompt_end == -1:
            prompt_end = len(content)

        prompt_template = content[prompt_start:prompt_end]

        # Simple token estimate (words * 1.3)
        words = len(prompt_template.split())
        token_estimate = int(words * 1.3)

        errors = []
        warnings = []

        if token_estimate > 1000:
            errors.append(f"Token budget exceeded: {token_estimate} > 1000")
        elif token_estimate > 700:
            warnings.append(f"Token budget high: {token_estimate} (target: 500-700)")

        return {
            "passed": token_estimate <= 1000,
            "token_count": token_estimate,
            "errors": errors,
            "warnings": warnings
        }

    def _print_summary(self):
        """Print validation summary"""

        print("\n" + "=" * 60)
        print("[STATS] VALIDATION SUMMARY")
        print("=" * 60)

        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        print(f"\nTotal skills validated: {total}")
        print(f"[OK] Passed: {passed}")
        print(f"[FAIL] Failed: {failed}")

        if failed > 0:
            print(f"\n[FAIL] Failed skills:")
            for result in self.results:
                if not result.passed:
                    print(f"   • {result.category}/{result.skill_name}")
                    for gate, status in result.gates_passed.items():
                        if not status:
                            print(f"     - {gate}: FAILED")

        # Token budget stats
        token_counts = [r.metrics.get("token_count", 0) for r in self.results]
        if token_counts:
            avg_tokens = sum(token_counts) / len(token_counts)
            max_tokens = max(token_counts)
            print(f"\n[METRICS] Token Budget:")
            print(f"   Average: {avg_tokens:.0f} tokens")
            print(f"   Maximum: {max_tokens} tokens")
            print(f"   Target: 500-700 tokens")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate all skills through quality gates",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--category",
        "-c",
        choices=["meta", "domain", "utility"],
        help="Only validate skills in this category"
    )
    parser.add_argument(
        "--fail-fast",
        "-f",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    validator = SkillValidator()
    success = validator.validate_all(
        category=args.category,
        fail_fast=args.fail_fast
    )

    if args.json:
        # Output JSON for CI/CD integration
        results_json = {
            "total": len(validator.results),
            "passed": sum(1 for r in validator.results if r.passed),
            "failed": sum(1 for r in validator.results if not r.passed),
            "results": [
                {
                    "skill": f"{r.category}/{r.skill_name}",
                    "passed": r.passed,
                    "gates": r.gates_passed,
                    "errors": r.errors,
                    "warnings": r.warnings,
                    "metrics": r.metrics
                }
                for r in validator.results
            ]
        }
        print(json.dumps(results_json, indent=2))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
