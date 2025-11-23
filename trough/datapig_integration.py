"""
Trough + DATAPIG Integration

"Computer, run comprehensive diagnostic - code AND data quality."

Unifies code quality detection (Trough) with data quality detection (DATAPIG).
Automatically detects data files and runs appropriate validators.
"""

import json
import csv
from pathlib import Path
from typing import List, Optional, Dict, Any

from .trough_types import SlopIssue, Severity, SlopCategory, Language


# Try to import DATAPIG (optional dependency)
try:
    from HoloLoom.datapig import DataPigDetector, DataQualityIssue, IssueType
    from HoloLoom.datapig.config import DetectorConfig
    DATAPIG_AVAILABLE = True
except ImportError:
    DATAPIG_AVAILABLE = False
    DataPigDetector = None
    DataQualityIssue = None


# ============================================================================
# Data File Detection
# ============================================================================

DATA_FILE_EXTENSIONS = {
    '.csv': 'csv',
    '.json': 'json',
    '.jsonl': 'jsonl',
    '.tsv': 'tsv',
    '.parquet': 'parquet',
    '.feather': 'feather',
    '.pkl': 'pickle',
    '.pickle': 'pickle',
}


def is_data_file(file_path: str) -> bool:
    """Check if file is a data file (CSV, JSON, etc.)"""
    suffix = Path(file_path).suffix.lower()
    return suffix in DATA_FILE_EXTENSIONS


def detect_data_format(file_path: str) -> Optional[str]:
    """Detect data file format from extension"""
    suffix = Path(file_path).suffix.lower()
    return DATA_FILE_EXTENSIONS.get(suffix)


# ============================================================================
# Data Loading
# ============================================================================

def load_csv(file_path: str) -> List[Dict]:
    """Load CSV file as list of dicts"""
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_json(file_path: str) -> List[Dict]:
    """Load JSON file (object or array of objects)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Normalize to list of dicts
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        return []


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file (one JSON object per line)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_data_file(file_path: str) -> Optional[List[Dict]]:
    """
    Load data file into list of dicts

    Supports: CSV, JSON, JSONL, TSV
    Returns None if unsupported format or error
    """
    fmt = detect_data_format(file_path)

    try:
        if fmt == 'csv' or fmt == 'tsv':
            return load_csv(file_path)
        elif fmt == 'json':
            return load_json(file_path)
        elif fmt == 'jsonl':
            return load_jsonl(file_path)
        else:
            return None
    except Exception as e:
        # Silently fail on malformed data
        return None


# ============================================================================
# Issue Conversion (DATAPIG → Trough)
# ============================================================================

DATAPIG_TO_SEVERITY = {
    "CAPTAIN": Severity.CRITICAL,     # Red Alert → Critical
    "COMMANDER": Severity.HIGH,       # Yellow Alert → High
    "LIEUTENANT": Severity.MEDIUM,    # Shields up → Medium
    "ENSIGN": Severity.LOW,           # Sensors → Low
}

DATAPIG_TO_CATEGORY = {
    "SCHEMA_DRIFT": SlopCategory.TYPE_MISMATCH,
    "DATA_LEAK": SlopCategory.HARDCODED_VALUES,  # Closest match
    "STALE_DATA": SlopCategory.INCOMPLETE,
    "DUPLICATES": SlopCategory.COPY_PASTE,
    "OUTLIERS": SlopCategory.TYPE_MISMATCH,
    "INCONSISTENT_FORMAT": SlopCategory.NAMING,
    "MISSING_RELATIONS": SlopCategory.TYPE_MISMATCH,
    "DISTRIBUTION_SHIFT": SlopCategory.COPY_PASTE,
    "SAMPLING_BIAS": SlopCategory.INCOMPLETE,
    "LABEL_NOISE": SlopCategory.TYPE_MISMATCH,
}


def convert_datapig_issue(issue: DataQualityIssue, file_path: str) -> SlopIssue:
    """
    Convert DATAPIG DataQualityIssue → Trough SlopIssue

    Args:
        issue: DATAPIG issue
        file_path: Path to data file

    Returns:
        Trough SlopIssue compatible with rest of pipeline
    """
    # Map severity
    severity = DATAPIG_TO_SEVERITY.get(issue.severity.value, Severity.MEDIUM)

    # Map category
    category = DATAPIG_TO_CATEGORY.get(issue.issue_type.value, SlopCategory.TYPE_MISMATCH)

    # Extract line number from location (e.g., "row_5" → line 5)
    line_number = 1  # Default to line 1 for data files
    if "row_" in issue.location:
        try:
            line_number = int(issue.location.split("row_")[1].split(".")[0]) + 1  # +1 for header
        except (IndexError, ValueError):
            line_number = 1

    # Format message with Star Trek flair preserved
    message = f"[DATAPIG] {issue.message}"

    # Generate suggestion based on issue type
    suggestion = generate_fix_suggestion(issue)

    # Create code snippet from location and details
    code_snippet = f"{issue.location}"
    if issue.details:
        # Add key details to snippet for context
        detail_str = ", ".join(f"{k}={v}" for k, v in list(issue.details.items())[:3])
        code_snippet = f"{issue.location} ({detail_str})"

    # Map severity to confidence (inverse relationship)
    severity_to_confidence = {
        Severity.CRITICAL: 0.95,
        Severity.HIGH: 0.85,
        Severity.MEDIUM: 0.70,
        Severity.LOW: 0.50,
    }
    confidence = severity_to_confidence.get(severity, 0.70)

    return SlopIssue(
        category=category,
        severity=severity,
        message=message,
        file_path=file_path,
        line_number=line_number,
        code_snippet=code_snippet,
        suggestion=suggestion,
        confidence=confidence
    )


def generate_fix_suggestion(issue: DataQualityIssue) -> Optional[str]:
    """Generate fix suggestion based on issue type"""

    suggestions = {
        "SCHEMA_DRIFT": "Ensure consistent schema across all rows. Add missing fields or validate types.",
        "DATA_LEAK": "*** CRITICAL: Remove PII/secrets from dataset immediately! Use environment variables for secrets.",
        "STALE_DATA": "Update outdated records or archive historical data.",
        "DUPLICATES": "Remove duplicate rows using data cleaning tools.",
        "OUTLIERS": "Investigate outliers - may be data errors or legitimate edge cases.",
        "INCONSISTENT_FORMAT": "Normalize formats to consistent standard (e.g., ISO 8601 for dates).",
        "MISSING_RELATIONS": "Verify foreign key references or remove orphaned records.",
        "DISTRIBUTION_SHIFT": "Check for dataset drift - may need retraining or validation.",
        "SAMPLING_BIAS": "Balance class distribution using oversampling, undersampling, or SMOTE.",
        "LABEL_NOISE": "Review contradictory labels and resolve conflicts.",
    }

    return suggestions.get(issue.issue_type.value)


# ============================================================================
# Unified Detection Pipeline
# ============================================================================

class UnifiedDetector:
    """
    Unified Code + Data Quality Detector

    Combines Trough (code quality) with DATAPIG (data quality) for
    comprehensive validation.
    """

    def __init__(self, enable_datapig: bool = True, datapig_config: Optional[DetectorConfig] = None):
        """
        Initialize unified detector

        Args:
            enable_datapig: Enable data quality detection (requires DATAPIG)
            datapig_config: DATAPIG configuration (default: balanced preset)
        """
        self.enable_datapig = enable_datapig and DATAPIG_AVAILABLE

        if self.enable_datapig:
            self.datapig = DataPigDetector(enable_verbose=False)
            # TODO: Use datapig_config when detector supports it
        else:
            self.datapig = None

    def detect_data_quality(self, file_path: str) -> List[SlopIssue]:
        """
        Detect data quality issues in data files

        Args:
            file_path: Path to data file (CSV, JSON, etc.)

        Returns:
            List of SlopIssue objects (Trough-compatible)
        """
        if not self.enable_datapig:
            return []

        if not is_data_file(file_path):
            return []

        # Load data
        data = load_data_file(file_path)
        if data is None:
            return []

        # Run DATAPIG detection
        datapig_issues = self.datapig.analyze_dataset(data)

        # Convert to Trough format
        trough_issues = [
            convert_datapig_issue(issue, file_path)
            for issue in datapig_issues
        ]

        return trough_issues

    def detect_all(self, file_path: str, code: Optional[str] = None) -> List[SlopIssue]:
        """
        Detect both code AND data quality issues

        Args:
            file_path: Path to file
            code: Optional code content (for code files)

        Returns:
            Combined list of issues (code + data quality)
        """
        all_issues = []

        # Code quality detection (if code provided)
        if code is not None:
            # Delegate to AISlopDetector
            # (caller should use AISlopDetector directly for code)
            pass

        # Data quality detection
        if is_data_file(file_path):
            data_issues = self.detect_data_quality(file_path)
            all_issues.extend(data_issues)

        return all_issues


# ============================================================================
# Integration Helpers
# ============================================================================

def analyze_file_unified(file_path: str) -> Dict[str, Any]:
    """
    Unified file analysis (code + data quality)

    Returns:
        {
            "file_path": str,
            "file_type": "code" | "data",
            "issues": List[SlopIssue],
            "summary": {
                "total_issues": int,
                "by_severity": {...},
                "by_category": {...}
            }
        }
    """
    detector = UnifiedDetector()

    # Check file type
    if is_data_file(file_path):
        file_type = "data"
        issues = detector.detect_data_quality(file_path)
    else:
        file_type = "code"
        # For code files, use AISlopDetector separately
        issues = []

    # Summarize
    summary = {
        "total_issues": len(issues),
        "by_severity": {
            severity.value: len([i for i in issues if i.severity == severity])
            for severity in Severity
        },
        "by_category": {
            cat.value: len([i for i in issues if i.category == cat])
            for cat in SlopCategory
        }
    }

    return {
        "file_path": file_path,
        "file_type": file_type,
        "issues": issues,
        "summary": summary,
        "datapig_enabled": detector.enable_datapig
    }


def scan_directory_unified(directory: str, include_data: bool = True) -> Dict[str, Any]:
    """
    Scan directory for code + data quality issues

    Args:
        directory: Directory to scan
        include_data: Include data files in scan

    Returns:
        {
            "total_files": int,
            "files_scanned": List[str],
            "total_issues": int,
            "issues_by_file": {file_path: List[SlopIssue]},
            "summary": {...}
        }
    """
    from pathlib import Path

    results = {
        "total_files": 0,
        "files_scanned": [],
        "total_issues": 0,
        "issues_by_file": {},
        "summary": {}
    }

    # Scan all files
    for file_path in Path(directory).rglob('*'):
        if file_path.is_file():
            # Check if should scan
            if not include_data and is_data_file(str(file_path)):
                continue

            # Analyze file
            file_results = analyze_file_unified(str(file_path))

            if file_results["issues"]:
                results["files_scanned"].append(str(file_path))
                results["issues_by_file"][str(file_path)] = file_results["issues"]
                results["total_issues"] += len(file_results["issues"])
                results["total_files"] += 1

    return results
