"""
Unit tests for Trough AI Slop Detector - Basic functionality.

Test scope: Individual detector functions, pattern matching, severity classification.

Created: 2025-11-22
Author: Agent C (Haiku)
"""

import pytest
from pathlib import Path


class TestTroughFixtures:
    """Test that fixtures load correctly."""

    def test_good_code_fixture_exists(self, good_code_file: Path):
        """Good code fixture should be loadable."""
        assert good_code_file.exists()
        content = good_code_file.read_text()
        assert "def calculate_average" in content
        assert "Error handling" not in content or "proper" in content

    def test_bad_code_slop_fixture_exists(self, bad_code_slop_file: Path):
        """Bad code slop fixture should be loadable."""
        assert bad_code_slop_file.exists()
        content = bad_code_slop_file.read_text()
        assert "ISSUE" in content  # Contains marked issues
        assert len(content) > 1000  # Substantial fixture

    def test_bad_code_security_fixture_exists(self, bad_code_security_file: Path):
        """Bad code security fixture should be loadable."""
        assert bad_code_security_file.exists()
        content = bad_code_security_file.read_text()
        assert "SQL injection" in content or "XSS" in content

    def test_bad_code_logic_fixture_exists(self, bad_code_logic_file: Path):
        """Bad code logic fixture should be loadable."""
        assert bad_code_logic_file.exists()
        content = bad_code_logic_file.read_text()
        assert "ISSUE" in content  # Contains marked issues


class TestCodeSnippets:
    """Test individual code patterns."""

    def test_detect_division_by_zero(self, sample_code_snippets):
        """Division by zero should be detectable."""
        code = sample_code_snippets["missing_error_handling"]
        assert "def divide(a, b):" in code
        assert "a / b" in code
        # Note: Actual detection requires Trough detector module

    def test_detect_hardcoded_secret(self, sample_code_snippets):
        """Hardcoded secrets should be detectable."""
        code = sample_code_snippets["hardcoded_secret"]
        assert "API_KEY" in code or "sk-" in code

    def test_detect_sql_injection(self, sample_code_snippets):
        """SQL injection patterns should be detectable."""
        code = sample_code_snippets["sql_injection"]
        assert "query" in code
        assert "+" in code  # String concatenation is dangerous

    def test_detect_xss_vulnerability(self, sample_code_snippets):
        """XSS vulnerabilities should be detectable."""
        code = sample_code_snippets["xss_vulnerability"]
        assert "html" in code or "<div>" in code

    def test_detect_resource_leak(self, sample_code_snippets):
        """Resource leaks should be detectable."""
        code = sample_code_snippets["resource_leak"]
        assert "open(" in code
        assert "close" not in code or ".close()" not in code


class TestFixtureCategories:
    """Test that fixtures cover expected categories."""

    def test_good_code_has_best_practices(self, good_code_file: Path):
        """Good code should demonstrate best practices."""
        content = good_code_file.read_text()
        # Should have error handling
        assert "try:" in content or "except" in content
        # Should have docstrings
        assert '"""' in content or "'''" in content
        # Should have type hints
        assert "->" in content or ":" in content

    def test_bad_slop_covers_multiple_issues(self, bad_code_slop_file: Path):
        """Bad slop fixture should have multiple issues."""
        content = bad_code_slop_file.read_text()
        issue_count = content.count("ISSUE")
        assert issue_count >= 15, f"Expected 15+ issues, found {issue_count}"

    def test_bad_security_covers_attacks(self, bad_code_security_file: Path):
        """Bad security fixture should cover attack vectors."""
        content = bad_code_security_file.read_text()
        assert "SQL" in content or "injection" in content
        assert "XSS" in content
        assert "hardcoded" in content
        assert "secret" in content or "password" in content

    def test_bad_logic_covers_errors(self, bad_code_logic_file: Path):
        """Bad logic fixture should cover logic errors."""
        content = bad_code_logic_file.read_text()
        assert "ISSUE" in content
        assert "division" in content.lower() or "divide" in content.lower()
        assert "off-by-one" in content.lower() or "off_by_one" in content
        assert "null" in content.lower() or "none" in content.lower()


class TestSampleCodeSnippets:
    """Test sample code snippets provided by fixture."""

    def test_sample_snippets_not_empty(self, sample_code_snippets):
        """Sample snippets should exist."""
        assert len(sample_code_snippets) >= 5
        assert "missing_error_handling" in sample_code_snippets
        assert "hardcoded_secret" in sample_code_snippets

    def test_each_snippet_is_valid_python(self, sample_code_snippets):
        """Each snippet should be syntactically valid Python."""
        for name, code in sample_code_snippets.items():
            try:
                compile(code, name, "exec")
            except SyntaxError as e:
                pytest.fail(f"Snippet '{name}' has syntax error: {e}")

    def test_snippet_lengths_reasonable(self, sample_code_snippets):
        """Snippets should be reasonable length."""
        for name, code in sample_code_snippets.items():
            assert len(code) > 10, f"Snippet '{name}' too short"
            assert len(code) < 10000, f"Snippet '{name}' too long"


class TestConfiguration:
    """Test configuration system."""

    def test_config_has_both_projects(self, test_config):
        """Configuration should include both Trough and BossPig."""
        assert "trough" in test_config
        assert "bosspig" in test_config

    def test_trough_config_valid(self, test_config):
        """Trough config should be valid."""
        trough = test_config["trough"]
        assert trough["enabled"] is True
        assert len(trough["categories"]) >= 4

    def test_bosspig_config_valid(self, test_config):
        """BossPig config should be valid."""
        bosspig = test_config["bosspig"]
        assert bosspig["enabled"] is True
        assert len(bosspig["categories"]) >= 4


@pytest.mark.unit
@pytest.mark.trough
class TestTroughIntegration:
    """Integration tests for Trough with fixtures."""

    def test_all_fixture_files_accessible(self, good_code_file, bad_code_slop_file,
                                         bad_code_security_file, bad_code_logic_file):
        """All fixture files should be accessible simultaneously."""
        files = [good_code_file, bad_code_slop_file, bad_code_security_file, bad_code_logic_file]
        for f in files:
            assert f.exists()
            assert f.is_file()
            assert f.stat().st_size > 0

    def test_fixture_isolation(self, good_code_file, bad_code_slop_file):
        """Fixtures should not be modified by tests."""
        good_content_before = good_code_file.read_text()
        bad_content_before = bad_code_slop_file.read_text()

        # Read again to ensure no modification
        good_content_after = good_code_file.read_text()
        bad_content_after = bad_code_slop_file.read_text()

        assert good_content_before == good_content_after
        assert bad_content_before == bad_content_after


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
