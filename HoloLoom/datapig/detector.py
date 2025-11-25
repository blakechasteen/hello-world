"""
DATAPIG Data Quality Detector - Star Trek Edition

"I'm a data validator, not a miracle worker!" - Dr. McCoy (adapted)

This module provides comprehensive data quality validation with
Star Trek-themed detection categories.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
import re
from datetime import datetime, timedelta


class Severity(Enum):
    """Data quality issue severity levels"""
    CAPTAIN = "CAPTAIN"      # Critical - needs immediate attention (Red Alert!)
    COMMANDER = "COMMANDER"  # High - significant issue (Yellow Alert)
    LIEUTENANT = "LIEUTENANT"  # Medium - should be addressed (Shields up)
    ENSIGN = "ENSIGN"       # Low - minor issue (Sensors detect anomaly)


class IssueType(Enum):
    """Star Trek-themed data quality issue categories"""
    SCHEMA_DRIFT = "SCHEMA_DRIFT"              # "The laws of physics are different here!"
    DATA_LEAK = "DATA_LEAK"                    # "Hull breach on deck 7!"
    STALE_DATA = "STALE_DATA"                  # "These readings are from last century!"
    DUPLICATES = "DUPLICATES"                  # "We're seeing double, Captain!"
    FUZZY_DUPLICATES = "FUZZY_DUPLICATES"      # "Similar life forms detected nearby!"
    HIGH_ENTROPY_PII = "HIGH_ENTROPY_PII"      # "Encrypted Romulan transmission detected!"
    WEAK_PASSWORD = "WEAK_PASSWORD"            # "Security protocols insufficient, Captain!"
    OUTLIERS = "OUTLIERS"                      # "Captain, these readings are... impossible!"
    INCONSISTENT_FORMAT = "INCONSISTENT_FORMAT"  # "Universal translator malfunction!"
    MISSING_RELATIONS = "MISSING_RELATIONS"    # "Transporter lost the signal!"
    DISTRIBUTION_SHIFT = "DISTRIBUTION_SHIFT"  # "We've entered an alternate reality!"
    SAMPLING_BIAS = "SAMPLING_BIAS"            # "Prime Directive violation detected!"
    LABEL_NOISE = "LABEL_NOISE"               # "Temporal anomaly - contradictory data!"


@dataclass
class DataQualityIssue:
    """A single data quality issue detected by DATAPIG"""
    issue_type: IssueType
    severity: Severity
    message: str
    location: str  # Column, row, or dataset location
    details: Dict[str, Any]
    stardate: float  # When issue was detected

    def __str__(self) -> str:
        """Format issue with Star Trek flair"""
        return (
            f"[{self.severity.value}] {self.issue_type.value}\n"
            f"  Location: {self.location}\n"
            f"  Message: {self.message}\n"
            f"  Stardate: {self.stardate:.2f}\n"
            f"  Details: {self.details}"
        )


class DataPigDetector:
    """
    DATAPIG: Data Analysis & Tactical Assessment Program for Integrity Governance

    "Computer, run level 1 diagnostic on all data systems."

    This detector identifies 10 categories of data quality issues using
    Star Trek-themed detection algorithms.
    """

    def __init__(
        self,
        enable_verbose: bool = False,
        enable_fuzzy_duplicates: bool = True,
        fuzzy_similarity_threshold: float = 0.85,
        fuzzy_use_phonetic: bool = True,
        fuzzy_fields: Optional[List[str]] = None,
        enable_entropy_detection: bool = True,
        high_entropy_threshold: float = 3.0,
        low_entropy_threshold: float = 1.5,
        entropy_min_samples: int = 5
    ):
        self.enable_verbose = enable_verbose
        self.enable_fuzzy_duplicates = enable_fuzzy_duplicates
        self.fuzzy_similarity_threshold = fuzzy_similarity_threshold
        self.fuzzy_use_phonetic = fuzzy_use_phonetic
        self.fuzzy_fields = fuzzy_fields  # None = auto-detect all string fields
        self.enable_entropy_detection = enable_entropy_detection
        self.high_entropy_threshold = high_entropy_threshold
        self.low_entropy_threshold = low_entropy_threshold
        self.entropy_min_samples = entropy_min_samples
        self.issues: List[DataQualityIssue] = []

    def analyze_dataset(self, data: Union[Dict, List[Dict], Any]) -> List[DataQualityIssue]:
        """
        Run full data quality analysis (all detection categories)

        Args:
            data: Dataset to analyze (dict, list of dicts, or pandas DataFrame)

        Returns:
            List of detected issues

        Example:
            detector = DataPigDetector()
            issues = detector.analyze_dataset(my_dataframe)

            for issue in issues:
                if issue.severity == Severity.CAPTAIN:
                    print(f"RED ALERT: {issue.message}")
        """
        self.issues = []
        stardate = self._get_stardate()

        # Convert to standard format
        rows = self._normalize_data(data)

        if not rows:
            return self.issues

        # Run all detection systems
        if self.enable_verbose:
            print("DATAPIG Engaging... All systems online.")

        self._detect_schema_drift(rows, stardate)
        self._detect_data_leaks(rows, stardate)
        self._detect_stale_data(rows, stardate)
        self._detect_duplicates(rows, stardate)

        # Fuzzy duplicate detection (optional, enabled by default)
        if self.enable_fuzzy_duplicates:
            self._detect_fuzzy_duplicates(rows, stardate)

        # Entropy-based PII detection (optional, enabled by default)
        if self.enable_entropy_detection:
            self._detect_entropy_pii(rows, stardate)

        self._detect_outliers(rows, stardate)
        self._detect_inconsistent_formatting(rows, stardate)
        self._detect_missing_relations(rows, stardate)
        self._detect_distribution_shift(rows, stardate)
        self._detect_sampling_bias(rows, stardate)
        self._detect_label_noise(rows, stardate)

        if self.enable_verbose:
            print(f"DATAPIG Analysis Complete. {len(self.issues)} issues detected.")

        return self.issues

    # ============================================================================
    # Detection Systems (Star Trek Themed)
    # ============================================================================

    def _detect_schema_drift(self, rows: List[Dict], stardate: float):
        """
        "Captain, the laws of physics have changed!" - Spock

        Detects schema drift: type mismatches, missing required fields
        """
        if not rows:
            return

        # Get expected schema from first row
        expected_keys = set(rows[0].keys())
        expected_types = {k: type(v) for k, v in rows[0].items()}

        for idx, row in enumerate(rows[1:], start=1):
            # Missing keys
            missing = expected_keys - set(row.keys())
            if missing:
                self.issues.append(DataQualityIssue(
                    issue_type=IssueType.SCHEMA_DRIFT,
                    severity=Severity.COMMANDER,
                    message=f"The laws of physics are different here! Missing fields: {missing}",
                    location=f"row_{idx}",
                    details={"missing_fields": list(missing), "row_index": idx},
                    stardate=stardate
                ))

            # Type mismatches
            for key, expected_type in expected_types.items():
                if key in row and row[key] is not None:
                    actual_type = type(row[key])
                    if actual_type != expected_type:
                        self.issues.append(DataQualityIssue(
                            issue_type=IssueType.SCHEMA_DRIFT,
                            severity=Severity.LIEUTENANT,
                            message=f"Type mismatch in field '{key}': expected {expected_type.__name__}, got {actual_type.__name__}",
                            location=f"row_{idx}.{key}",
                            details={
                                "field": key,
                                "expected_type": expected_type.__name__,
                                "actual_type": actual_type.__name__,
                                "row_index": idx
                            },
                            stardate=stardate
                        ))

    def _detect_data_leaks(self, rows: List[Dict], stardate: float):
        """
        "Hull breach on deck 7!" - Security Officer

        Detects PII/secrets in datasets
        """
        # Patterns for sensitive data
        patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            "api_key": r'\b[A-Za-z0-9_-]{32,}\b',
            "password": r'password["\s:=]+[A-Za-z0-9!@#$%^&*]+',
        }

        for idx, row in enumerate(rows):
            for key, value in row.items():
                if not isinstance(value, str):
                    continue

                for pattern_name, pattern in patterns.items():
                    if re.search(pattern, value, re.IGNORECASE):
                        self.issues.append(DataQualityIssue(
                            issue_type=IssueType.DATA_LEAK,
                            severity=Severity.CAPTAIN,  # RED ALERT!
                            message=f"*** Hull breach detected! Possible {pattern_name} exposure in field '{key}' ***",
                            location=f"row_{idx}.{key}",
                            details={
                                "field": key,
                                "leak_type": pattern_name,
                                "row_index": idx,
                                "value_preview": value[:20] + "..." if len(value) > 20 else value
                            },
                            stardate=stardate
                        ))

    def _detect_stale_data(self, rows: List[Dict], stardate: float):
        """
        "These readings are from last century!" - Chief O'Brien

        Detects outdated timestamps and zombie records
        """
        now = datetime.now()
        stale_threshold = timedelta(days=365)  # 1 year

        timestamp_fields = ["timestamp", "created_at", "updated_at", "last_modified", "date"]

        for idx, row in enumerate(rows):
            for key in timestamp_fields:
                if key not in row:
                    continue

                value = row[key]

                # Try to parse timestamp
                try:
                    if isinstance(value, str):
                        # Try common formats
                        for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                            try:
                                ts = datetime.strptime(value, fmt)
                                break
                            except ValueError:
                                continue
                        else:
                            continue
                    elif isinstance(value, (int, float)):
                        ts = datetime.fromtimestamp(value)
                    else:
                        continue

                    age = now - ts
                    if age > stale_threshold:
                        self.issues.append(DataQualityIssue(
                            issue_type=IssueType.STALE_DATA,
                            severity=Severity.LIEUTENANT,
                            message=f"These readings are from {age.days} days ago! Field '{key}' is stale.",
                            location=f"row_{idx}.{key}",
                            details={
                                "field": key,
                                "age_days": age.days,
                                "timestamp": str(ts),
                                "row_index": idx
                            },
                            stardate=stardate
                        ))

                except (ValueError, OSError):
                    pass

    def _detect_duplicates(self, rows: List[Dict], stardate: float):
        """
        "We're seeing double, Captain!" - Sulu

        Detects exact and fuzzy duplicates
        """
        seen_hashes: Dict[int, int] = {}  # hash -> first occurrence index

        for idx, row in enumerate(rows):
            # Create hash of row (excluding None values for stability)
            row_tuple = tuple(sorted((k, v) for k, v in row.items() if v is not None))
            row_hash = hash(row_tuple)

            if row_hash in seen_hashes:
                self.issues.append(DataQualityIssue(
                    issue_type=IssueType.DUPLICATES,
                    severity=Severity.LIEUTENANT,
                    message=f"We're seeing double! Exact duplicate of row {seen_hashes[row_hash]}",
                    location=f"row_{idx}",
                    details={
                        "duplicate_of": seen_hashes[row_hash],
                        "current_index": idx,
                        "row_data": dict(row)
                    },
                    stardate=stardate
                ))
            else:
                seen_hashes[row_hash] = idx

    def _detect_fuzzy_duplicates(self, rows: List[Dict], stardate: float):
        """
        "Similar life forms detected nearby!" - Sensors

        Detects fuzzy duplicates using Levenshtein distance.
        Complements exact duplicate detection with approximate matching.
        """
        from HoloLoom.datapig.fuzzy_detection import find_fuzzy_duplicates_advanced

        # Auto-detect string fields if not specified
        if self.fuzzy_fields is None:
            # Get all string fields from first row
            if not rows:
                return
            string_fields = [
                key for key, value in rows[0].items()
                if isinstance(value, str) and value  # Non-empty strings only
            ]
        else:
            string_fields = self.fuzzy_fields

        if not string_fields:
            return  # No string fields to check

        # Run fuzzy detection
        matches = find_fuzzy_duplicates_advanced(
            rows,
            fields=string_fields,
            similarity_threshold=self.fuzzy_similarity_threshold,
            use_phonetic=self.fuzzy_use_phonetic
        )

        # Create issues for each fuzzy match
        for match in matches:
            self.issues.append(DataQualityIssue(
                issue_type=IssueType.FUZZY_DUPLICATES,
                severity=Severity.ENSIGN,  # Lower severity than exact duplicates
                message=(
                    f"Similar life forms detected! "
                    f"Rows {match.row1_index} and {match.row2_index} are {match.similarity:.0%} similar "
                    f"in field '{match.field}'"
                ),
                location=f"row_{match.row2_index}",
                details={
                    "duplicate_of": match.row1_index,
                    "current_index": match.row2_index,
                    "field": match.field,
                    "similarity": match.similarity,
                    "edit_distance": match.edit_distance,
                    "value1": match.value1,
                    "value2": match.value2
                },
                stardate=stardate
            ))

    def _detect_entropy_pii(self, rows: List[Dict], stardate: float):
        """
        "Encrypted Romulan transmission detected!" - Communications Officer

        Detects potential PII fields using entropy analysis.
        Identifies high-entropy patterns (SSNs, API keys, UUIDs, hashes) and
        low-entropy weak passwords.
        """
        from HoloLoom.datapig.entropy_detection import detect_pii_by_entropy

        # Run entropy detection
        results = detect_pii_by_entropy(
            rows,
            high_entropy_threshold=self.high_entropy_threshold,
            low_entropy_threshold=self.low_entropy_threshold,
            min_samples=self.entropy_min_samples
        )

        # Create issues for high-entropy PII fields
        for analysis in results:
            if analysis.high_entropy_count > 0 or analysis.suspicious_patterns:
                # High entropy = potential PII (SSN, API keys, etc.)
                self.issues.append(DataQualityIssue(
                    issue_type=IssueType.HIGH_ENTROPY_PII,
                    severity=Severity.COMMANDER,  # High severity - potential PII exposure
                    message=(
                        f"Encrypted transmission detected in field '{analysis.field_name}'! "
                        f"{analysis.high_entropy_count} values with high entropy "
                        f"(avg: {analysis.avg_entropy:.2f}). "
                        f"Patterns: {', '.join(analysis.suspicious_patterns) if analysis.suspicious_patterns else 'Unknown'}"
                    ),
                    location=f"field_{analysis.field_name}",
                    details={
                        "field": analysis.field_name,
                        "avg_entropy": analysis.avg_entropy,
                        "min_entropy": analysis.min_entropy,
                        "max_entropy": analysis.max_entropy,
                        "high_count": analysis.high_entropy_count,
                        "patterns": analysis.suspicious_patterns,
                        "total_values": analysis.high_entropy_count + analysis.low_entropy_count
                    },
                    stardate=stardate
                ))

            # Low entropy = weak passwords
            if analysis.low_entropy_count > len(rows) * 0.7:  # >70% low entropy
                self.issues.append(DataQualityIssue(
                    issue_type=IssueType.WEAK_PASSWORD,
                    severity=Severity.LIEUTENANT,  # Medium severity
                    message=(
                        f"Security protocols insufficient in '{analysis.field_name}'! "
                        f"{analysis.low_entropy_count} values with weak entropy "
                        f"(avg: {analysis.avg_entropy:.2f}). Recommend stronger randomness."
                    ),
                    location=f"field_{analysis.field_name}",
                    details={
                        "field": analysis.field_name,
                        "avg_entropy": analysis.avg_entropy,
                        "low_count": analysis.low_entropy_count,
                        "total_values": analysis.high_entropy_count + analysis.low_entropy_count,
                        "recommendation": "Use stronger password requirements or encryption"
                    },
                    stardate=stardate
                ))

    def _detect_outliers(self, rows: List[Dict], stardate: float):
        """
        "Captain, these readings are... impossible!" - Spock

        Detects statistical anomalies and impossible values
        """
        # Collect numeric columns
        numeric_cols: Dict[str, List[float]] = {}

        for row in rows:
            for key, value in row.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if key not in numeric_cols:
                        numeric_cols[key] = []
                    numeric_cols[key].append(float(value))

        # Check for outliers using IQR method
        for col, values in numeric_cols.items():
            if len(values) < 4:
                continue

            sorted_vals = sorted(values)
            n = len(sorted_vals)
            q1 = sorted_vals[n // 4]
            q3 = sorted_vals[3 * n // 4]
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            for idx, row in enumerate(rows):
                if col in row:
                    value = row[col]
                    if isinstance(value, (int, float)) and (value < lower_bound or value > upper_bound):
                        self.issues.append(DataQualityIssue(
                            issue_type=IssueType.OUTLIERS,
                            severity=Severity.LIEUTENANT,
                            message=f"Impossible readings in '{col}': {value} (expected {lower_bound:.2f} to {upper_bound:.2f})",
                            location=f"row_{idx}.{col}",
                            details={
                                "field": col,
                                "value": value,
                                "lower_bound": lower_bound,
                                "upper_bound": upper_bound,
                                "q1": q1,
                                "q3": q3,
                                "row_index": idx
                            },
                            stardate=stardate
                        ))

    def _detect_inconsistent_formatting(self, rows: List[Dict], stardate: float):
        """
        "Universal translator malfunction!" - Uhura

        Detects inconsistent formatting (dates, phone numbers, etc.)
        """
        # Track format patterns per column
        column_formats: Dict[str, Set[str]] = {}

        for idx, row in enumerate(rows):
            for key, value in row.items():
                if not isinstance(value, str):
                    continue

                # Detect format pattern
                pattern = self._classify_format(value)

                if key not in column_formats:
                    column_formats[key] = set()
                column_formats[key].add(pattern)

        # Report columns with multiple formats
        for col, formats in column_formats.items():
            if len(formats) > 1:
                self.issues.append(DataQualityIssue(
                    issue_type=IssueType.INCONSISTENT_FORMAT,
                    severity=Severity.ENSIGN,
                    message=f"Universal translator malfunction in '{col}'! Multiple formats detected: {formats}",
                    location=f"column_{col}",
                    details={
                        "field": col,
                        "detected_formats": list(formats),
                        "format_count": len(formats)
                    },
                    stardate=stardate
                ))

    def _detect_missing_relations(self, rows: List[Dict], stardate: float):
        """
        "Transporter lost the signal!" - Chief O'Brien

        Detects broken foreign keys and orphaned records
        """
        # Look for foreign key patterns
        fk_pattern = r'(.+)_(id|ref|fk)$'

        # Collect all IDs
        id_fields: Dict[str, Set[Any]] = {}
        fk_fields: Dict[str, List[tuple]] = {}  # fk -> [(row_idx, value)]

        for idx, row in enumerate(rows):
            for key, value in row.items():
                if key.endswith('_id') or key == 'id':
                    if key not in id_fields:
                        id_fields[key] = set()
                    id_fields[key].add(value)

                # Check if this looks like a foreign key
                match = re.match(fk_pattern, key)
                if match and key != 'id':
                    if key not in fk_fields:
                        fk_fields[key] = []
                    fk_fields[key].append((idx, value))

        # Check for orphaned references
        for fk_col, references in fk_fields.items():
            # Try to find corresponding ID field
            base_name = fk_col.replace('_id', '').replace('_ref', '').replace('_fk', '')
            possible_id_fields = [f"{base_name}_id", "id"]

            for id_field in possible_id_fields:
                if id_field in id_fields:
                    valid_ids = id_fields[id_field]

                    for row_idx, fk_value in references:
                        if fk_value not in valid_ids and fk_value is not None:
                            self.issues.append(DataQualityIssue(
                                issue_type=IssueType.MISSING_RELATIONS,
                                severity=Severity.COMMANDER,
                                message=f"Transporter lost the signal! '{fk_col}' references non-existent {id_field}: {fk_value}",
                                location=f"row_{row_idx}.{fk_col}",
                                details={
                                    "foreign_key": fk_col,
                                    "referenced_id": fk_value,
                                    "expected_id_field": id_field,
                                    "row_index": row_idx
                                },
                                stardate=stardate
                            ))
                    break

    def _detect_distribution_shift(self, rows: List[Dict], stardate: float):
        """
        "We've entered an alternate reality!" - Captain Janeway

        Detects dataset drift and distribution changes
        """
        # Simple check: look for categorical columns with unexpected values
        # Track value distributions

        categorical_cols: Dict[str, Dict[Any, int]] = {}

        for row in rows:
            for key, value in row.items():
                if isinstance(value, (str, bool)) or value is None:
                    if key not in categorical_cols:
                        categorical_cols[key] = {}
                    categorical_cols[key][value] = categorical_cols[key].get(value, 0) + 1

        # Check for rare values (< 1% of total)
        for col, value_counts in categorical_cols.items():
            total = sum(value_counts.values())
            rare_threshold = 0.01 * total

            rare_values = [v for v, count in value_counts.items() if count < rare_threshold and count > 0]

            if rare_values and len(value_counts) > 10:  # Only report if significant diversity
                self.issues.append(DataQualityIssue(
                    issue_type=IssueType.DISTRIBUTION_SHIFT,
                    severity=Severity.ENSIGN,
                    message=f"Alternate reality detected in '{col}'! {len(rare_values)} rare values (<1% frequency)",
                    location=f"column_{col}",
                    details={
                        "field": col,
                        "rare_values": rare_values[:10],  # Limit output
                        "total_unique": len(value_counts),
                        "rare_count": len(rare_values)
                    },
                    stardate=stardate
                ))

    def _detect_sampling_bias(self, rows: List[Dict], stardate: float):
        """
        "Prime Directive violation detected!" - Captain Picard

        Detects unrepresentative data splits and sampling bias
        """
        # Check for class imbalance in label columns
        label_candidates = ["label", "class", "category", "target", "y"]

        for label_col in label_candidates:
            if not any(label_col in row for row in rows):
                continue

            # Count class distribution
            class_counts: Dict[Any, int] = {}
            for row in rows:
                if label_col in row:
                    value = row[label_col]
                    class_counts[value] = class_counts.get(value, 0) + 1

            if len(class_counts) < 2:
                continue

            total = sum(class_counts.values())
            min_count = min(class_counts.values())
            max_count = max(class_counts.values())

            # Imbalance ratio > 10:1 is suspicious
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

            if imbalance_ratio > 10:
                self.issues.append(DataQualityIssue(
                    issue_type=IssueType.SAMPLING_BIAS,
                    severity=Severity.COMMANDER,
                    message=f"Prime Directive violation! Severe class imbalance in '{label_col}': {imbalance_ratio:.1f}:1 ratio",
                    location=f"column_{label_col}",
                    details={
                        "field": label_col,
                        "class_distribution": class_counts,
                        "imbalance_ratio": imbalance_ratio,
                        "minority_class_pct": (min_count / total) * 100
                    },
                    stardate=stardate
                ))

    def _detect_label_noise(self, rows: List[Dict], stardate: float):
        """
        "Temporal anomaly detected - contradictory data!" - Seven of Nine

        Detects contradictory labels for same input
        """
        # Group by all features except label columns
        label_candidates = ["label", "class", "category", "target", "y"]

        for label_col in label_candidates:
            if not any(label_col in row for row in rows):
                continue

            # Create feature -> labels mapping
            feature_labels: Dict[str, Set[Any]] = {}

            for idx, row in enumerate(rows):
                if label_col not in row:
                    continue

                # Create feature hash (excluding label)
                features = tuple(sorted((k, v) for k, v in row.items() if k != label_col and v is not None))
                feature_hash = str(hash(features))

                if feature_hash not in feature_labels:
                    feature_labels[feature_hash] = set()
                feature_labels[feature_hash].add(row[label_col])

            # Find contradictions
            contradictions = {k: v for k, v in feature_labels.items() if len(v) > 1}

            if contradictions:
                self.issues.append(DataQualityIssue(
                    issue_type=IssueType.LABEL_NOISE,
                    severity=Severity.COMMANDER,
                    message=f"Temporal anomaly! {len(contradictions)} feature groups have contradictory labels in '{label_col}'",
                    location=f"column_{label_col}",
                    details={
                        "field": label_col,
                        "contradiction_count": len(contradictions),
                        "example_contradictions": list(contradictions.values())[:5]
                    },
                    stardate=stardate
                ))

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _normalize_data(self, data: Any) -> List[Dict]:
        """Convert various data formats to list of dicts"""
        # Try pandas DataFrame
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                return data.to_dict('records')
        except ImportError:
            pass

        # Already list of dicts
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            return data

        # Single dict -> wrap in list
        if isinstance(data, dict):
            return [data]

        return []

    def _get_stardate(self) -> float:
        """Calculate current stardate (TNG formula)"""
        # Stardate 0 = 2323-01-01 (approximately)
        # Formula: [year - 2323] * 1000 + [day of year]
        now = datetime.now()
        year_offset = now.year - 2323
        day_of_year = now.timetuple().tm_yday
        return year_offset * 1000 + day_of_year

    def _classify_format(self, value: str) -> str:
        """Classify string format pattern"""
        if re.match(r'^\d{4}-\d{2}-\d{2}$', value):
            return 'ISO_DATE'
        elif re.match(r'^\d{2}/\d{2}/\d{4}$', value):
            return 'US_DATE'
        elif re.match(r'^\d{3}-\d{3}-\d{4}$', value):
            return 'US_PHONE'
        elif re.match(r'^\+\d{1,3}', value):
            return 'INTL_PHONE'
        elif re.match(r'^[A-Z]{2,3}-\d+$', value):
            return 'CODE_FORMAT'
        elif value.isupper():
            return 'UPPERCASE'
        elif value.islower():
            return 'lowercase'
        elif value.istitle():
            return 'TitleCase'
        else:
            return 'mixed'


# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_dataset(data: Any, verbose: bool = False) -> List[DataQualityIssue]:
    """
    Quick analysis function - one-liner for data validation

    Example:
        issues = analyze_dataset(my_df)
        red_alerts = [i for i in issues if i.severity == Severity.CAPTAIN]
    """
    detector = DataPigDetector(enable_verbose=verbose)
    return detector.analyze_dataset(data)


def engage_warp_validation(data: Any) -> bool:
    """
    "Engage!" - Captain Picard

    Quick validation check - returns True if data is safe for warp speed processing.

    Returns:
        True if no CAPTAIN-level issues, False otherwise
    """
    issues = analyze_dataset(data, verbose=False)
    captain_issues = [i for i in issues if i.severity == Severity.CAPTAIN]

    if captain_issues:
        print(f"*** RED ALERT! {len(captain_issues)} critical issues detected! ***")
        print("Warp engines offline until issues resolved.")
        for issue in captain_issues:
            print(f"  - {issue.message}")
        return False
    else:
        print("*** All systems nominal. Engaging warp drive! ***")
        return True