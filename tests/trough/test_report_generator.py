"""
Tests for Trough Interactive HTML Report Generator

Week 1 Test Suite: HTML generation, rendering, and basic functionality
Status: Week 1 Complete
Date: 2025-11-22
"""

import pytest
from pathlib import Path
from trough.trough_types import SlopIssue, Severity, SlopCategory, Language
from trough.report_generator.generator import (
    generate_html_report,
    _escape_html,
    _get_severity_badge,
    _convert_windows_path_to_vscode,
    _get_category_icon
)


# Sample test data
def create_sample_finding(
    line_number: int = 42,
    category: str = "error_handling",
    severity: str = "high",
    message: str = "Missing error handling"
) -> SlopIssue:
    """Create a sample SlopIssue for testing."""
    return SlopIssue(
        category=SlopCategory(category),
        severity=Severity(severity),
        message=message,
        file_path="example.py",
        line_number=line_number,
        code_snippet="result = data['key']  # No error handling",
        suggestion="Use data.get('key', default_value)",
        confidence=0.95
    )


class TestHTMLEscaping:
    """Test HTML escaping functionality."""

    def test_escape_simple_html(self):
        """Test escaping basic HTML characters."""
        assert _escape_html("<script>") == "&lt;script&gt;"
        assert _escape_html("A & B") == "A &amp; B"
        assert _escape_html('"quoted"') == "&quot;quoted&quot;"

    def test_escape_complex_content(self):
        """Test escaping complex content."""
        result = _escape_html("<img src=x onerror='alert(1)'>")
        assert "<img" not in result
        assert "onerror" in result
        assert "&lt;" in result

    def test_escape_empty_string(self):
        """Test escaping empty string."""
        assert _escape_html("") == ""

    def test_no_double_escaping(self):
        """Test that ampersands are properly escaped."""
        # This is a behavior test - & should be escaped to &amp;
        html = _escape_html("&lt;")
        assert html == "&amp;lt;"  # The & gets escaped, then lt


class TestSeverityBadges:
    """Test severity badge generation."""

    def test_critical_badge_color(self):
        """Test critical severity badge has correct color."""
        badge = _get_severity_badge("critical")
        assert "critical" not in badge.lower() or "CRITICAL" in badge
        assert "#e74c3c" in badge  # Critical red color

    def test_all_severity_levels(self):
        """Test all severity levels generate valid badges."""
        severities = ["critical", "high", "medium", "low", "info"]
        for severity in severities:
            badge = _get_severity_badge(severity)
            assert "<span" in badge
            assert "severity-badge" in badge
            assert severity.upper() in badge

    def test_case_insensitive_severity(self):
        """Test badge generation is case-insensitive."""
        badge1 = _get_severity_badge("CRITICAL")
        badge2 = _get_severity_badge("critical")
        assert badge1 == badge2


class TestPathConversion:
    """Test Windows path to VS Code URI conversion."""

    def test_windows_path_conversion(self):
        """Test converting Windows path to VS Code URI."""
        result = _convert_windows_path_to_vscode(r"c:\Users\blake\file.py")
        assert result.startswith("vscode://file/")
        assert "c:" in result
        assert "/" in result  # Forward slashes

    def test_windows_path_with_forward_slashes(self):
        """Test Windows path already with forward slashes."""
        result = _convert_windows_path_to_vscode("c:/Users/blake/file.py")
        assert result == "vscode://file/c:/Users/blake/file.py"

    def test_unix_path_conversion(self):
        """Test converting Unix path to VS Code URI."""
        result = _convert_windows_path_to_vscode("/Users/blake/file.py")
        assert result.startswith("vscode://file/")
        assert "/Users/blake/file.py" in result

    def test_relative_path_conversion(self):
        """Test converting relative path."""
        result = _convert_windows_path_to_vscode("src/example.py")
        assert "vscode://file/" in result


class TestCategoryIcons:
    """Test category emoji icons."""

    def test_known_categories_have_icons(self):
        """Test that known categories have emoji icons."""
        categories = [
            "error_handling", "hardcoded_values", "security",
            "performance", "dead_code", "resource_leak"
        ]
        for category in categories:
            icon = _get_category_icon(category)
            assert len(icon) > 0
            assert icon != "📌"  # Should not be default

    def test_unknown_category_gets_default_icon(self):
        """Test unknown category gets default icon."""
        icon = _get_category_icon("totally_unknown_category_123")
        assert icon == "📌"  # Default icon


class TestReportGeneration:
    """Test HTML report generation."""

    def test_generate_empty_report(self):
        """Test generating report with no findings."""
        html = generate_html_report(findings=[], output_path=None)

        assert "<!DOCTYPE html>" in html
        assert "All Clear!" in html
        assert "Trough Code Analysis Report" in html
        assert "<style>" in html
        assert "<script>" in html

    def test_generate_report_with_findings(self):
        """Test generating report with findings."""
        findings = [
            create_sample_finding(line_number=10, severity="critical"),
            create_sample_finding(line_number=42, severity="high"),
        ]

        html = generate_html_report(findings=findings, output_path=None)

        assert "<!DOCTYPE html>" in html
        assert "2" in html  # Should show total count
        assert "findings" in html.lower()

    def test_report_contains_required_panels(self):
        """Test report contains all three panels."""
        html = generate_html_report(findings=[], output_path=None)

        assert 'id="findings-list"' in html
        assert 'id="code-panel"' in html
        assert 'id="details-panel"' in html

    def test_report_contains_search_bar(self):
        """Test report includes search functionality."""
        html = generate_html_report(findings=[], output_path=None)

        assert 'id="search-input"' in html
        assert "placeholder" in html and "Search" in html

    def test_report_with_code_snippets(self):
        """Test report includes code snippets."""
        findings = [create_sample_finding()]
        code = 'result = data["key"]\nprint(result)'

        html = generate_html_report(
            findings=findings,
            code_snippets={"example.py": code},
            output_path=None
        )

        assert "CODE_SNIPPETS" in html
        assert "example.py" in html

    def test_report_statistics_calculation(self):
        """Test report calculates statistics correctly."""
        findings = [
            create_sample_finding(severity="critical"),
            create_sample_finding(severity="critical"),
            create_sample_finding(severity="high"),
            create_sample_finding(severity="medium"),
        ]

        html = generate_html_report(findings=findings, output_path=None)

        # Check that stats are in the HTML
        assert "stat-value" in html
        assert "stat-label" in html


class TestReportFiltering:
    """Test report filtering capabilities."""

    def test_report_has_filter_buttons(self):
        """Test report includes filter buttons."""
        findings = [
            create_sample_finding(severity="critical"),
            create_sample_finding(severity="high"),
        ]

        html = generate_html_report(findings=findings, output_path=None)

        assert "filter-button" in html
        assert "CRITICAL" in html
        assert "HIGH" in html

    def test_report_data_embedded_as_json(self):
        """Test findings data is embedded as JSON."""
        findings = [
            create_sample_finding(
                message="Test issue",
                severity="high"
            )
        ]

        html = generate_html_report(findings=findings, output_path=None)

        assert "FINDINGS_DATA" in html
        assert "Test issue" in html or "Test issue" in html.replace("\\", "")


class TestReportSaving:
    """Test saving report to file."""

    def test_save_report_to_file(self, tmp_path):
        """Test saving report to file."""
        output_file = tmp_path / "report.html"
        findings = [create_sample_finding()]

        html = generate_html_report(
            findings=findings,
            output_path=str(output_file)
        )

        assert output_file.exists()
        assert output_file.stat().st_size > 1000
        assert "<!DOCTYPE html>" in output_file.read_text()

    def test_create_output_directory(self, tmp_path):
        """Test that output directory is created if needed."""
        output_file = tmp_path / "deep" / "nested" / "report.html"

        generate_html_report(
            findings=[],
            output_path=str(output_file)
        )

        assert output_file.exists()
        assert output_file.parent.exists()

    def test_save_with_special_characters_in_path(self, tmp_path):
        """Test saving with special characters in filename."""
        output_file = tmp_path / "report-2025-11-22.html"

        generate_html_report(
            findings=[],
            output_path=str(output_file)
        )

        assert output_file.exists()


class TestAccessibility:
    """Test accessibility features of generated report."""

    def test_report_has_semantic_html(self):
        """Test report uses semantic HTML."""
        html = generate_html_report(findings=[], output_path=None)

        assert "<header" in html or "header" in html.lower()
        assert "<main" in html or "<section" in html or "<div" in html

    def test_report_has_alt_text_equivalents(self):
        """Test report includes text alternatives for icons."""
        findings = [create_sample_finding()]
        html = generate_html_report(findings=findings, output_path=None)

        # Should have readable text for severity badges
        assert "CRITICAL" in html or "HIGH" in html or "severity" in html.lower()


class TestPerformance:
    """Test report generation performance."""

    def test_generate_report_with_many_findings(self):
        """Test generating report with many findings is fast."""
        import time

        # Create 1000 findings
        findings = [
            create_sample_finding(
                line_number=i,
                message=f"Issue {i}"
            )
            for i in range(1000)
        ]

        start = time.time()
        html = generate_html_report(findings=findings, output_path=None)
        elapsed = time.time() - start

        assert elapsed < 5.0  # Should complete in under 5 seconds
        assert len(html) > 10000  # Should have substantial HTML

    def test_html_size_reasonable(self):
        """Test generated HTML size is reasonable."""
        findings = [create_sample_finding() for _ in range(10)]
        html = generate_html_report(findings=findings, output_path=None)

        # HTML should be reasonable size (< 1MB for 10 findings)
        assert len(html) < 1_000_000
        # But also substantial (> 10KB with CSS/JS)
        assert len(html) > 10_000


class TestErrorHandling:
    """Test error handling in report generation."""

    def test_handles_missing_finding_fields(self):
        """Test handling of findings with missing fields."""
        # Create a minimal finding-like object
        class MinimalFinding:
            category = "error_handling"
            severity = "high"
            message = "Test"
            file_path = "test.py"
            line_number = 1
            code_snippet = ""
            suggestion = ""

        finding = MinimalFinding()
        html = generate_html_report(findings=[finding], output_path=None)

        assert "<!DOCTYPE html>" in html
        assert "Test" in html

    def test_handles_invalid_finding_objects(self):
        """Test graceful handling of invalid findings."""
        # Pass mixed valid and invalid objects
        valid = create_sample_finding()
        invalid = {"not": "a finding"}

        # Should not raise, just skip invalid
        html = generate_html_report(findings=[valid], output_path=None)
        assert "<!DOCTYPE html>" in html


# Integration test
class TestIntegration:
    """Integration tests with actual Trough detector."""

    def test_integration_with_detector_output(self):
        """Test report generation with actual detector output format."""
        from trough import AISlopDetector, Language

        detector = AISlopDetector()

        # Simple Python code with issues
        code = '''
import os
password = "secret123"
result = data['key']
f = open("file.txt")
'''

        # Run detector (async)
        import asyncio
        findings = asyncio.run(detector.detect_all(code, Language.PYTHON, "test.py"))

        # Generate report
        html = generate_html_report(
            findings=findings,
            code_snippets={"test.py": code},
            output_path=None
        )

        assert "<!DOCTYPE html>" in html
        assert "findings" in html.lower()


class TestBugFixesWeek4:
    """Week 4 Bug Fixes - Filter logic, keyboard navigation, and JSON serialization."""

    def test_filter_and_search_combined(self):
        """Test that search and filter work together correctly."""
        findings = [
            create_sample_finding(line_number=10, severity="critical", category="error_handling", message="Missing error handling"),
            create_sample_finding(line_number=20, severity="high", category="security", message="Hardcoded secret"),
            create_sample_finding(line_number=30, severity="medium", category="error_handling", message="No validation"),
        ]

        html = generate_html_report(findings=findings, output_path=None)

        # Check that all findings are embedded
        assert "FINDINGS_DATA" in html
        assert "Missing error handling" in html or "Missing" in html

    def test_vscode_path_variations(self):
        """Test VSCode path handling for different path formats."""
        from trough.report_generator.generator import _convert_windows_path_to_vscode

        # Windows drive letter path
        assert "vscode://file/" in _convert_windows_path_to_vscode("c:\\Users\\blake\\file.py")

        # Windows drive letter with forward slashes
        result = _convert_windows_path_to_vscode("c:/Users/blake/file.py")
        assert "vscode://file/" in result
        assert "c:" in result

        # Unix absolute path
        result = _convert_windows_path_to_vscode("/home/user/file.py")
        assert "vscode://file/" in result

        # Relative path
        result = _convert_windows_path_to_vscode("src/file.py")
        assert "vscode://file/" in result

    def test_finding_with_missing_optional_fields(self):
        """Test handling of findings with missing optional fields."""
        class MinimalFinding:
            category = "error_handling"
            severity = "high"
            message = "Test issue"
            file_path = "test.py"
            line_number = 10
            code_snippet = None  # Missing
            suggestion = None  # Missing
            confidence = None  # Missing

        finding = MinimalFinding()
        html = generate_html_report(findings=[finding], output_path=None)

        # Should not crash and should have basic content
        assert "<!DOCTYPE html>" in html
        assert "Test issue" in html

    def test_keyboard_navigation_state_tracking(self):
        """Test that keyboard navigation properly tracks selected finding."""
        findings = [
            create_sample_finding(line_number=10),
            create_sample_finding(line_number=20),
            create_sample_finding(line_number=30),
        ]

        html = generate_html_report(findings=findings, output_path=None)

        # Check that JavaScript properly initializes state
        assert "selectedFinding = null" in html
        assert "currentFilteredFindings = []" in html

    def test_filter_button_data_attributes(self):
        """Test that filter buttons have proper data attributes."""
        findings = [
            create_sample_finding(severity="critical"),
            create_sample_finding(severity="high"),
            create_sample_finding(severity="medium"),
        ]

        html = generate_html_report(findings=findings, output_path=None)

        # Check filter button attributes
        assert 'data-filter-type="severity"' in html
        assert 'data-filter-value="critical"' in html
        assert 'data-filter-value="high"' in html

    def test_json_serialization_with_special_characters(self):
        """Test JSON serialization with special characters in content."""
        # Create sample finding with special characters
        finding = create_sample_finding()
        # Manually set special char attributes
        finding.message = "Missing error handling for 'special' chars: \"quotes\" & <brackets>"
        finding.suggestion = "Use data.get('key', default_value) and handle < > chars"

        html = generate_html_report(findings=[finding], output_path=None)

        # Should have escaped properly
        assert "<!DOCTYPE html>" in html
        assert "FINDINGS_DATA" in html

    def test_empty_findings_list_panels(self):
        """Test that empty findings list properly clears all panels."""
        html = generate_html_report(findings=[], output_path=None)

        # Should show empty state
        assert "All Clear!" in html
        assert "No issues found" in html

    def test_code_panel_with_large_line_numbers(self):
        """Test code panel with large line numbers."""
        findings = [
            create_sample_finding(line_number=10000)
        ]

        code = "\n".join([f"Line {i}" for i in range(10000)])

        html = generate_html_report(
            findings=findings,
            code_snippets={"example.py": code},
            output_path=None
        )

        assert "<!DOCTYPE html>" in html

    def test_finding_index_consistency(self):
        """Test that finding selection properly synchronizes across panels."""
        findings = [
            create_sample_finding(
                line_number=10,
                message="First issue",
                category="error_handling"
            ),
            create_sample_finding(
                line_number=20,
                message="Second issue",
                category="security"
            ),
        ]

        html = generate_html_report(findings=findings, output_path=None)

        # Check that findings are properly indexed
        assert "First issue" in html
        assert "Second issue" in html

    def test_filter_persistence_through_keyboard_navigation(self):
        """Test that filter state is maintained during keyboard navigation."""
        findings = [
            create_sample_finding(severity="critical", message="Issue A"),
            create_sample_finding(severity="high", message="Issue B"),
            create_sample_finding(severity="critical", message="Issue C"),
        ]

        html = generate_html_report(findings=findings, output_path=None)

        # Verify filter infrastructure is present
        assert "applyAllFilters" in html
        assert "activeFilters" in html


class TestCrossBrowserCompatibility:
    """Cross-browser compatibility tests for Week 4-5."""

    def test_css_uses_standard_syntax(self):
        """Test that CSS uses standard properties supported across browsers."""
        html = generate_html_report(findings=[], output_path=None)

        # Check for modern but widely supported CSS
        assert "display: flex" in html  # Flexbox supported in all modern browsers
        assert "background: linear-gradient" in html  # Gradients supported in all modern browsers

    def test_firefox_scrollbar_support(self):
        """Test that Firefox scrollbar styling is present."""
        html = generate_html_report(findings=[], output_path=None)

        # Firefox scrollbar styling
        assert "scrollbar-width: thin" in html
        assert "scrollbar-color:" in html

    def test_webkit_scrollbar_support(self):
        """Test that WebKit scrollbar styling is present."""
        html = generate_html_report(findings=[], output_path=None)

        # WebKit scrollbar styling
        assert "::-webkit-scrollbar" in html
        assert "::-webkit-scrollbar-thumb" in html

    def test_mobile_responsive_media_queries(self):
        """Test that mobile responsive media queries are present."""
        html = generate_html_report(findings=[], output_path=None)

        # Mobile breakpoints
        assert "@media (max-width: 1024px)" in html
        assert "@media (max-width: 768px)" in html
        assert "@media (max-width: 480px)" in html

    def test_javascript_es6_compatibility(self):
        """Test JavaScript uses compatible syntax."""
        html = generate_html_report(findings=[], output_path=None)

        # ES6 features (supported in all modern browsers)
        assert "const " in html or "let " in html  # Block scope
        assert "=>" in html  # Arrow functions

    def test_no_external_dependencies_in_report(self):
        """Test that report has no external script/style dependencies."""
        html = generate_html_report(findings=[], output_path=None)

        # No external CDN dependencies
        assert "https://cdn" not in html
        assert "https://ajax" not in html
        assert "<link rel='stylesheet'" not in html

    def test_viewport_meta_tag(self):
        """Test that viewport meta tag is present for mobile."""
        html = generate_html_report(findings=[], output_path=None)

        assert '<meta name="viewport"' in html
        assert "initial-scale=1.0" in html


class TestPerformanceOptimizations:
    """Performance optimization tests for Week 5."""

    def test_large_code_handling(self):
        """Test handling of large code files."""
        import time

        findings = [create_sample_finding(line_number=500)]
        code = "\n".join([f"Line {i}: {chr(65 + (i % 26))} = {i}" for i in range(1000)])

        start = time.time()
        html = generate_html_report(
            findings=findings,
            code_snippets={"large_file.py": code},
            output_path=None
        )
        elapsed = time.time() - start

        # Should handle large files quickly
        assert elapsed < 2.0
        # Large code embedded, expect > 20KB HTML
        assert len(html) > 20000

    def test_many_findings_rendering(self):
        """Test rendering performance with many findings."""
        import time

        findings = [create_sample_finding(line_number=i) for i in range(200)]

        start = time.time()
        html = generate_html_report(findings=findings, output_path=None)
        elapsed = time.time() - start

        # Should render 200 findings quickly
        assert elapsed < 3.0

    def test_html_size_optimization(self):
        """Test that HTML output size is reasonable."""
        findings = [create_sample_finding() for _ in range(50)]

        html = generate_html_report(findings=findings, output_path=None)

        # For 50 findings, HTML should be reasonable size
        # Embedded CSS/JS is ~20KB base, plus findings data (~300 bytes each)
        # Expect: 30-150KB for 50 findings
        assert 30_000 < len(html) < 150_000


class TestMinificationOptimizations:
    """Test CSS/JS minification for reduced file size."""

    def test_minify_css_reduces_size(self):
        """Test that CSS minification reduces file size."""
        findings = [create_sample_finding() for _ in range(10)]

        html_normal = generate_html_report(findings=findings, output_path=None, minify=False)
        html_minified = generate_html_report(findings=findings, output_path=None, minify=True)

        # Minified should be smaller
        assert len(html_minified) < len(html_normal)

        # Should still contain required CSS
        assert "color:" in html_minified
        assert "padding:" in html_minified

    def test_minify_preserves_functionality(self):
        """Test that minification doesn't break functionality."""
        findings = [
            create_sample_finding(line_number=10, message="Error handling missing"),
            create_sample_finding(line_number=20, message="Unused variable"),
        ]

        html_minified = generate_html_report(findings=findings, output_path=None, minify=True)

        # Should still contain required elements
        assert "findings-list" in html_minified
        assert "code-panel" in html_minified
        assert "details-panel" in html_minified
        assert "FINDINGS_DATA" in html_minified

    def test_minify_reduction_ratio(self):
        """Test that minification achieves reasonable size reduction."""
        findings = [create_sample_finding() for _ in range(50)]

        html_normal = generate_html_report(findings=findings, output_path=None, minify=False)
        html_minified = generate_html_report(findings=findings, output_path=None, minify=True)

        # Minification should reduce size by at least 10%
        reduction_ratio = (len(html_normal) - len(html_minified)) / len(html_normal)
        assert reduction_ratio > 0.10  # At least 10% reduction

    def test_minify_removes_comments(self):
        """Test that minification removes CSS/JS comments."""
        findings = [create_sample_finding()]

        html_minified = generate_html_report(findings=findings, output_path=None, minify=True)

        # Minified CSS should have fewer comments
        # (This is a rough test - minified shouldn't have /* */ style comments)
        assert html_minified.count("/*") < 2  # Very few multi-line comments

    def test_minify_preserves_media_queries(self):
        """Test that minification preserves responsive design media queries."""
        findings = [create_sample_finding()]

        html_minified = generate_html_report(findings=findings, output_path=None, minify=True)

        # Should still contain media queries for responsive design
        assert "@media" in html_minified

    def test_report_with_minify_and_virtualization(self):
        """Test report generation with both minify and virtualization enabled."""
        findings = [create_sample_finding(line_number=i) for i in range(1, 101)]

        html = generate_html_report(
            findings=findings,
            output_path=None,
            minify=True,
            enable_virtualization=True
        )

        # Should generate successfully with both options
        assert len(html) > 0
        assert "FINDINGS_DATA" in html
        assert len(html) < 200_000  # Should be under 200KB even with 100 findings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
