"""
DATAPIG MCP Tools for Claude Desktop

"Computer, analyze data quality and engage warp drive if safe."

MCP server tools exposing DATAPIG data quality validation to Claude Desktop.
"""

import json
from typing import List, Dict, Any, Optional

# MCP imports
try:
    from mcp.server import MCPServer
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    MCPServer = None

# DATAPIG imports
try:
    from HoloLoom.datapig import (
        DataPigDetector,
        DataQualityIssue,
        Severity,
        IssueType,
        engage_warp_validation,
        analyze_dataset
    )
    from HoloLoom.datapig.config import create_config, DetectorConfig
    DATAPIG_AVAILABLE = True
except ImportError:
    DATAPIG_AVAILABLE = False

# xTerminator imports
try:
    from xterminator.datapig_fixer import fix_dataset
    XTERMINATOR_AVAILABLE = True
except ImportError:
    XTERMINATOR_AVAILABLE = False


# ============================================================================
# MCP Server Setup
# ============================================================================

if MCP_AVAILABLE:
    server = MCPServer(name="datapig-mcp")
else:
    server = None


# ============================================================================
# Tool 1: Validate Data Quality
# ============================================================================

def _validate_data_quality_impl(
    dataset: List[Dict],
    severity_threshold: str = "COMMANDER",
    preset: str = "default",
    **config_overrides
) -> Dict[str, Any]:
    """Implementation of validate_data_quality tool"""

    if not DATAPIG_AVAILABLE:
        return {
            "error": "DATAPIG not available. Install HoloLoom package.",
            "safe_for_warp": False
        }

    # Create detector with config
    config = create_config(preset, **config_overrides)
    detector = DataPigDetector(enable_verbose=False)

    # Analyze dataset
    all_issues = detector.analyze_dataset(dataset)

    # Filter by severity
    threshold_map = {"CAPTAIN": 4, "COMMANDER": 3, "LIEUTENANT": 2, "ENSIGN": 1}
    threshold_val = threshold_map.get(severity_threshold, 2)

    filtered_issues = [
        i for i in all_issues
        if threshold_map.get(i.severity.value, 0) >= threshold_val
    ]

    # Count by type
    by_type = {}
    for issue_type in IssueType:
        count = len([i for i in filtered_issues if i.issue_type == issue_type])
        if count > 0:
            by_type[issue_type.value] = count

    # Count by severity
    by_severity = {}
    for severity in Severity:
        count = len([i for i in filtered_issues if i.severity == severity])
        if count > 0:
            by_severity[severity.value] = count

    # Check if safe for warp
    captain_issues = [i for i in all_issues if i.severity == Severity.CAPTAIN]
    safe_for_warp = len(captain_issues) == 0

    # Generate recommendations
    recommendations = _generate_recommendations(filtered_issues)

    return {
        "safe_for_warp": safe_for_warp,
        "total_issues": len(filtered_issues),
        "by_type": by_type,
        "by_severity": by_severity,
        "issues": [_format_issue(i) for i in filtered_issues[:20]],  # Top 20
        "recommendations": recommendations,
        "stardate": detector._get_stardate() if hasattr(detector, '_get_stardate') else None
    }


if MCP_AVAILABLE and DATAPIG_AVAILABLE:
    @server.tool("validate_data_quality")
    async def validate_data_quality(
        dataset: List[Dict],
        severity_threshold: str = "COMMANDER",
        preset: str = "default"
    ) -> Dict[str, Any]:
        """
        Validate data quality with DATAPIG

        Detects 10 categories of data quality issues:
        1. Schema Drift - Type mismatches, missing fields
        2. Data Leaks - PII/secrets (emails, SSNs, API keys)
        3. Stale Data - Outdated timestamps
        4. Duplicates - Exact duplicate detection
        5. Outliers - Statistical anomalies
        6. Inconsistent Format - Mixed date formats, casing
        7. Missing Relations - Broken foreign keys
        8. Distribution Shift - Dataset drift
        9. Sampling Bias - Class imbalance
        10. Label Noise - Contradictory labels

        Args:
            dataset: List of dictionaries to validate
            severity_threshold: Only report issues >= this severity
                               (CAPTAIN/COMMANDER/LIEUTENANT/ENSIGN)
            preset: Configuration preset
                   (default/strict/lenient/fast/pii_focused/ml_validation)

        Returns:
            Validation report with issues and recommendations

        Example:
            result = await validate_data_quality(
                dataset=[{"id": 1, "email": "test@example.com"}],
                severity_threshold="CAPTAIN",
                preset="pii_focused"
            )

            if not result["safe_for_warp"]:
                print("RED ALERT: Critical data quality issues detected!")
        """
        return _validate_data_quality_impl(dataset, severity_threshold, preset)


# ============================================================================
# Tool 2: Fix Data Quality Issues
# ============================================================================

def _fix_data_quality_impl(
    dataset: List[Dict],
    issues: Optional[List[Dict]] = None,
    validate: bool = True
) -> Dict[str, Any]:
    """Implementation of fix_data_quality tool"""

    if not DATAPIG_AVAILABLE:
        return {"error": "DATAPIG not available"}

    if not XTERMINATOR_AVAILABLE:
        return {"error": "xTerminator not available"}

    # If issues not provided, detect them first
    if issues is None:
        detector = DataPigDetector()
        detected_issues = detector.analyze_dataset(dataset)
    else:
        # Convert from dict format
        detected_issues = [_dict_to_issue(i) for i in issues]

    # Fix issues
    result = fix_dataset(dataset, detected_issues, validate=validate)

    return {
        "success": result["success"],
        "fixes_applied": result["fixes_applied"],
        "validation_passed": result["validation_passed"],
        "validation_errors": result["validation_errors"],
        "fixed_data": result["fixed_data"] if result["success"] else None
    }


if MCP_AVAILABLE and DATAPIG_AVAILABLE and XTERMINATOR_AVAILABLE:
    @server.tool("fix_data_quality")
    async def fix_data_quality(
        dataset: List[Dict],
        issues: Optional[List[Dict]] = None,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Auto-fix data quality issues with xTerminator

        Automatically fixes common data quality issues:
        - Schema drift (add missing fields, cast types)
        - Duplicates (remove duplicates)
        - Inconsistent format (normalize dates, phones)
        - Outliers (cap to bounds)
        - Data leaks (REDACT PII/secrets)
        - Missing relations (set FK to NULL)

        Args:
            dataset: Dataset to fix
            issues: Optional pre-detected issues (if None, will detect first)
            validate: Enable 5-stage validation after fixes

        Returns:
            Fixed dataset with validation status

        Example:
            # Detect + fix in one call
            result = await fix_data_quality(my_dataset)

            if result["success"]:
                clean_data = result["fixed_data"]
                print(f"Applied {len(result['fixes_applied'])} fixes")
        """
        return _fix_data_quality_impl(dataset, issues, validate)


# ============================================================================
# Tool 3: Warp Validation (Quick Check)
# ============================================================================

def _warp_validation_impl(dataset: List[Dict]) -> Dict[str, Any]:
    """Implementation of warp_validation tool"""

    if not DATAPIG_AVAILABLE:
        return {"safe_for_warp": False, "error": "DATAPIG not available"}

    # Quick check: only CAPTAIN-level issues
    detector = DataPigDetector()
    issues = detector.analyze_dataset(dataset)

    captain_issues = [i for i in issues if i.severity == Severity.CAPTAIN]
    safe_for_warp = len(captain_issues) == 0

    return {
        "safe_for_warp": safe_for_warp,
        "critical_issues": len(captain_issues),
        "total_issues": len(issues),
        "message": "*** All systems nominal. Engaging warp drive! ***" if safe_for_warp else f"*** RED ALERT! {len(captain_issues)} critical issues detected! ***"
    }


if MCP_AVAILABLE and DATAPIG_AVAILABLE:
    @server.tool("warp_validation")
    async def warp_validation(dataset: List[Dict]) -> Dict[str, Any]:
        """
        Quick warp drive validation check

        "Engage!" - Captain Picard

        Fast validation check for critical (CAPTAIN-level) issues only.
        Use before production deployment, ML training, or API publishing.

        Args:
            dataset: Dataset to validate

        Returns:
            {safe_for_warp: bool, message: str}

        Example:
            check = await warp_validation(production_data)

            if check["safe_for_warp"]:
                deploy_to_production(production_data)
            else:
                print(check["message"])
        """
        return _warp_validation_impl(dataset)


# ============================================================================
# Helper Functions
# ============================================================================

def _format_issue(issue: DataQualityIssue) -> Dict[str, Any]:
    """Format DATAPIG issue for JSON serialization"""
    return {
        "type": issue.issue_type.value,
        "severity": issue.severity.value,
        "message": issue.message,
        "location": issue.location,
        "details": issue.details,
        "stardate": issue.stardate
    }


def _dict_to_issue(issue_dict: Dict) -> DataQualityIssue:
    """Convert dict to DataQualityIssue (for fix_data_quality)"""
    from HoloLoom.datapig.detector import DataQualityIssue, IssueType, Severity

    return DataQualityIssue(
        issue_type=IssueType[issue_dict["type"]],
        severity=Severity[issue_dict["severity"]],
        message=issue_dict["message"],
        location=issue_dict["location"],
        details=issue_dict.get("details", {}),
        stardate=issue_dict.get("stardate", 0.0)
    )


def _generate_recommendations(issues: List[DataQualityIssue]) -> List[str]:
    """Generate actionable recommendations based on issues"""
    recommendations = []

    # Count by type
    by_type = {}
    for issue in issues:
        by_type[issue.issue_type] = by_type.get(issue.issue_type, 0) + 1

    # Generate recommendations
    if IssueType.DATA_LEAK in by_type:
        recommendations.append(
            f"🚨 CRITICAL: {by_type[IssueType.DATA_LEAK]} data leak(s) detected! "
            "Remove PII/secrets immediately before deployment."
        )

    if IssueType.SAMPLING_BIAS in by_type:
        recommendations.append(
            f"⚠️  {by_type[IssueType.SAMPLING_BIAS]} sampling bias issue(s) detected. "
            "Balance class distribution using SMOTE, oversampling, or stratified sampling."
        )

    if IssueType.LABEL_NOISE in by_type:
        recommendations.append(
            f"⚠️  {by_type[IssueType.LABEL_NOISE]} label noise issue(s) detected. "
            "Review contradictory labels manually."
        )

    if IssueType.SCHEMA_DRIFT in by_type:
        recommendations.append(
            f"ℹ️  {by_type[IssueType.SCHEMA_DRIFT]} schema drift issue(s) detected. "
            "Use fix_data_quality tool to auto-fix missing fields and type mismatches."
        )

    if IssueType.DUPLICATES in by_type:
        recommendations.append(
            f"ℹ️  {by_type[IssueType.DUPLICATES]} duplicate(s) detected. "
            "Use fix_data_quality tool to auto-remove duplicates."
        )

    return recommendations


# ============================================================================
# Export MCP Server
# ============================================================================

def get_mcp_server():
    """Get DATAPIG MCP server for Claude Desktop integration"""
    if not MCP_AVAILABLE:
        raise ImportError("mcp package not available. Install with: pip install mcp")

    if not DATAPIG_AVAILABLE:
        raise ImportError("DATAPIG not available. Install HoloLoom package.")

    return server


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    # Test tools without MCP server
    print("DATAPIG MCP Tools - Standalone Test\n")

    test_data = [
        {"id": 1, "name": "Picard", "email": "picard@enterprise.com"},
        {"id": "2", "name": "Riker"},  # Schema drift + missing email
        {"id": 1, "name": "Picard", "email": "picard@enterprise.com"},  # Duplicate
    ]

    print("1. Validate Data Quality")
    result = _validate_data_quality_impl(test_data, severity_threshold="LIEUTENANT")
    print(json.dumps(result, indent=2))

    print("\n2. Warp Validation")
    warp_result = _warp_validation_impl(test_data)
    print(json.dumps(warp_result, indent=2))

    print("\n3. Fix Data Quality")
    if XTERMINATOR_AVAILABLE:
        fix_result = _fix_data_quality_impl(test_data)
        print(json.dumps({k: v for k, v in fix_result.items() if k != "fixed_data"}, indent=2))
    else:
        print("xTerminator not available - skipping fix test")
