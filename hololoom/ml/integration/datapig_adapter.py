"""
HoloLoom ML Training System - DataPig Integration

Adapter for data quality validation using HoloLoom's DataPig system.
Validates training data for common issues before model training.

Created: 2025-12-31
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from hololoom.ml.config import ValidationConfig, ValidationStrictness
from hololoom.ml.protocol import (
    DataSplit,
    DataValidationIssue,
    DataValidationResult,
    FeatureNames,
)

logger = logging.getLogger(__name__)


class DataPigMLAdapter:
    """
    Adapter for validating ML training data.

    Checks for:
    - Missing values
    - Outliers (z-score based)
    - Infinite values
    - Constant features (zero variance)
    - Constant target
    - Multicollinearity (high correlation)
    - Sample count sufficiency
    - Target distribution issues

    Usage:
        adapter = DataPigMLAdapter()
        result = adapter.validate_training_data(X, y, feature_names)
        if not result.is_valid:
            print(result.summary())
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize adapter.

        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()

    def validate_training_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[FeatureNames] = None,
    ) -> DataValidationResult:
        """
        Validate training data for common issues.

        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Optional feature names

        Returns:
            DataValidationResult with issues and recommendations
        """
        issues: List[DataValidationIssue] = []
        stats: Dict[str, Any] = {}
        recommendations: List[str] = []

        # Ensure numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Basic stats
        stats["n_samples"] = n_samples
        stats["n_features"] = n_features

        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Run validation checks
        issues.extend(self._check_sample_count(n_samples, n_features))
        issues.extend(self._check_missing_values(X, feature_names))
        issues.extend(self._check_infinite_values(X, feature_names))
        issues.extend(self._check_constant_features(X, feature_names))
        issues.extend(self._check_target(y))
        issues.extend(self._check_outliers(X, feature_names))

        if self.config.check_multicollinearity:
            issues.extend(self._check_multicollinearity(X, feature_names))

        if self.config.check_target_distribution:
            issues.extend(self._check_target_distribution(y))

        # Generate recommendations
        recommendations = self._generate_recommendations(issues)

        # Determine if valid based on blocking issues
        blocking_issues = [i for i in issues if i.blocking]
        is_valid = len(blocking_issues) == 0

        return DataValidationResult(
            is_valid=is_valid,
            issues=issues,
            stats=stats,
            recommendations=recommendations,
        )

    def validate_data_split(self, data: DataSplit) -> DataValidationResult:
        """
        Validate a DataSplit object.

        Args:
            data: DataSplit to validate

        Returns:
            DataValidationResult
        """
        return self.validate_training_data(
            data.X_train,
            data.y_train,
            data.feature_names,
        )

    def _check_sample_count(
        self,
        n_samples: int,
        n_features: int,
    ) -> List[DataValidationIssue]:
        """Check if sample count is sufficient."""
        issues = []

        if n_samples < self.config.min_samples:
            issues.append(
                DataValidationIssue(
                    issue_type="insufficient_samples",
                    severity="error",
                    message=f"Only {n_samples} samples (minimum: {self.config.min_samples})",
                    details={"n_samples": n_samples, "min_required": self.config.min_samples},
                    blocking=self.config.strictness != ValidationStrictness.LOW,
                )
            )

        # Check ratio of samples to features
        ratio = n_samples / max(n_features, 1)
        if ratio < 10:
            issues.append(
                DataValidationIssue(
                    issue_type="low_sample_feature_ratio",
                    severity="warning",
                    message=f"Low sample/feature ratio: {ratio:.1f} (recommend ≥10)",
                    details={"ratio": ratio, "n_samples": n_samples, "n_features": n_features},
                    blocking=self.config.strictness == ValidationStrictness.HIGH,
                )
            )

        return issues

    def _check_missing_values(
        self,
        X: np.ndarray,
        feature_names: FeatureNames,
    ) -> List[DataValidationIssue]:
        """Check for missing values (NaN/None)."""
        issues = []

        for i, name in enumerate(feature_names):
            col = X[:, i]
            n_missing = np.isnan(col).sum() if np.issubdtype(col.dtype, np.floating) else 0
            if n_missing > 0:
                ratio = n_missing / len(col)
                issues.append(
                    DataValidationIssue(
                        issue_type="missing_values",
                        severity="warning" if ratio < self.config.max_missing_ratio else "error",
                        column=name,
                        message=f"{n_missing} missing values ({ratio:.1%})",
                        details={"n_missing": int(n_missing), "ratio": ratio},
                        blocking=ratio > self.config.max_missing_ratio,
                    )
                )

        return issues

    def _check_infinite_values(
        self,
        X: np.ndarray,
        feature_names: FeatureNames,
    ) -> List[DataValidationIssue]:
        """Check for infinite values."""
        issues = []

        for i, name in enumerate(feature_names):
            col = X[:, i]
            if np.issubdtype(col.dtype, np.floating):
                n_inf = np.isinf(col).sum()
                if n_inf > 0:
                    issues.append(
                        DataValidationIssue(
                            issue_type="infinite_values",
                            severity="error",
                            column=name,
                            message=f"{n_inf} infinite values found",
                            details={"n_infinite": int(n_inf)},
                            blocking=True,
                        )
                    )

        return issues

    def _check_constant_features(
        self,
        X: np.ndarray,
        feature_names: FeatureNames,
    ) -> List[DataValidationIssue]:
        """Check for constant (zero variance) features."""
        issues = []

        if not self.config.check_constant_features:
            return issues

        for i, name in enumerate(feature_names):
            col = X[:, i]
            variance = np.nanvar(col)
            if variance == 0 or np.isnan(variance):
                issues.append(
                    DataValidationIssue(
                        issue_type="constant_feature",
                        severity="warning",
                        column=name,
                        message="Feature has zero variance (constant)",
                        details={"variance": float(variance) if not np.isnan(variance) else 0},
                        blocking=self.config.strictness == ValidationStrictness.HIGH,
                    )
                )

        return issues

    def _check_target(self, y: np.ndarray) -> List[DataValidationIssue]:
        """Check target variable for issues."""
        issues = []

        # Check for missing values in target
        if np.issubdtype(y.dtype, np.floating):
            n_missing = np.isnan(y).sum()
            if n_missing > 0:
                issues.append(
                    DataValidationIssue(
                        issue_type="target_missing_values",
                        severity="error",
                        message=f"Target has {n_missing} missing values",
                        details={"n_missing": int(n_missing)},
                        blocking=True,
                    )
                )

            # Check for infinite values in target
            n_inf = np.isinf(y).sum()
            if n_inf > 0:
                issues.append(
                    DataValidationIssue(
                        issue_type="target_infinite_values",
                        severity="error",
                        message=f"Target has {n_inf} infinite values",
                        details={"n_infinite": int(n_inf)},
                        blocking=True,
                    )
                )

        # Check for constant target
        variance = np.nanvar(y)
        if variance == 0:
            issues.append(
                DataValidationIssue(
                    issue_type="constant_target",
                    severity="error",
                    message="Target has zero variance (constant)",
                    details={"variance": 0.0},
                    blocking=True,
                )
            )

        return issues

    def _check_outliers(
        self,
        X: np.ndarray,
        feature_names: FeatureNames,
    ) -> List[DataValidationIssue]:
        """Check for outliers using z-score."""
        issues = []

        for i, name in enumerate(feature_names):
            col = X[:, i]
            if not np.issubdtype(col.dtype, np.floating):
                continue

            # Calculate z-scores
            mean = np.nanmean(col)
            std = np.nanstd(col)
            if std == 0:
                continue

            z_scores = np.abs((col - mean) / std)
            n_outliers = (z_scores > self.config.outlier_threshold).sum()

            if n_outliers > 0:
                ratio = n_outliers / len(col)
                issues.append(
                    DataValidationIssue(
                        issue_type="outliers",
                        severity="warning",
                        column=name,
                        message=f"{n_outliers} outliers (z > {self.config.outlier_threshold})",
                        details={
                            "n_outliers": int(n_outliers),
                            "ratio": ratio,
                            "threshold": self.config.outlier_threshold,
                        },
                        blocking=False,
                    )
                )

        return issues

    def _check_multicollinearity(
        self,
        X: np.ndarray,
        feature_names: FeatureNames,
    ) -> List[DataValidationIssue]:
        """Check for high correlation between features."""
        issues = []

        n_features = X.shape[1]
        if n_features < 2:
            return issues

        # Calculate correlation matrix
        try:
            # Handle NaN values
            X_clean = np.nan_to_num(X, nan=0.0)
            corr_matrix = np.corrcoef(X_clean.T)
        except Exception:
            return issues

        # Find high correlations
        threshold = self.config.multicollinearity_threshold
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = abs(corr_matrix[i, j])
                if corr > threshold:
                    issues.append(
                        DataValidationIssue(
                            issue_type="multicollinearity",
                            severity="warning",
                            message=f"High correlation between {feature_names[i]} and {feature_names[j]}",
                            details={
                                "feature1": feature_names[i],
                                "feature2": feature_names[j],
                                "correlation": float(corr),
                            },
                            blocking=self.config.strictness == ValidationStrictness.HIGH,
                        )
                    )

        return issues

    def _check_target_distribution(self, y: np.ndarray) -> List[DataValidationIssue]:
        """Check target distribution for potential issues."""
        issues = []

        y_clean = y[~np.isnan(y)] if np.issubdtype(y.dtype, np.floating) else y

        # Check for high skewness
        try:
            mean = np.mean(y_clean)
            std = np.std(y_clean)
            if std > 0:
                skewness = np.mean(((y_clean - mean) / std) ** 3)
                if abs(skewness) > 2:
                    issues.append(
                        DataValidationIssue(
                            issue_type="target_skewed",
                            severity="warning",
                            message=f"Target is highly skewed (skewness: {skewness:.2f})",
                            details={"skewness": float(skewness)},
                            blocking=False,
                        )
                    )
        except Exception:
            pass

        return issues

    def _generate_recommendations(
        self,
        issues: List[DataValidationIssue],
    ) -> List[str]:
        """Generate recommendations based on issues found."""
        recommendations = []

        issue_types = {i.issue_type for i in issues}

        if "missing_values" in issue_types:
            recommendations.append(
                "Consider imputation (mean, median, or mode) for missing values"
            )

        if "infinite_values" in issue_types:
            recommendations.append(
                "Replace infinite values with NaN and apply imputation"
            )

        if "constant_feature" in issue_types:
            recommendations.append(
                "Remove constant features as they provide no information"
            )

        if "outliers" in issue_types:
            recommendations.append(
                "Consider outlier removal or robust scaling for features with outliers"
            )

        if "multicollinearity" in issue_types:
            recommendations.append(
                "Consider feature selection or PCA to reduce multicollinearity"
            )

        if "target_skewed" in issue_types:
            recommendations.append(
                "Consider log transformation for skewed target variable"
            )

        if "low_sample_feature_ratio" in issue_types:
            recommendations.append(
                "Consider feature selection or regularization with low sample/feature ratio"
            )

        return recommendations
