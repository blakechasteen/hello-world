"""
HoloLoom ML Training System - SpinningWheel Integration

Adapter for ingesting tabular data through HoloLoom's SpinningWheel system.
Converts CSV, Excel, JSON, Parquet files to ML-ready numpy arrays.

Created: 2025-12-31
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from HoloLoom.ml.config import DataConfig, MissingValueStrategy, NormalizationMethod
from HoloLoom.ml.protocol import DataSplit, FeatureNames

logger = logging.getLogger(__name__)

# Check pandas availability
PANDAS_AVAILABLE = False
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("pandas not available. Install with: pip install pandas>=1.3")


class SpinningWheelMLAdapter:
    """
    Adapter for ingesting tabular data for ML training.

    Supports:
    - CSV, Excel, JSON, Parquet files
    - Automatic categorical encoding (one-hot)
    - Missing value handling
    - Feature normalization
    - Train/test splitting

    Usage:
        adapter = SpinningWheelMLAdapter()
        X, y, features = await adapter.ingest_tabular("data.csv", "target")

        # Or get a full DataSplit
        data_split = await adapter.ingest_and_split("data.csv", "target")
    """

    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize adapter.

        Args:
            config: Data configuration (uses defaults if None)
        """
        self.config = config or DataConfig()

    def _check_pandas(self) -> None:
        """Check if pandas is available."""
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for data ingestion. "
                "Install with: pip install pandas>=1.3"
            )

    async def ingest_tabular(
        self,
        source: Union[str, Path],
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        **read_kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, FeatureNames]:
        """
        Ingest tabular data from file.

        Args:
            source: File path (CSV, Excel, JSON, Parquet)
            target_column: Name of target column
            feature_columns: Optional list of feature columns (uses all except target if None)
            **read_kwargs: Additional arguments for pandas read function

        Returns:
            Tuple of (X, y, feature_names)
        """
        self._check_pandas()

        source = Path(source)

        # Determine file type and read
        df = self._read_file(source, **read_kwargs)

        # Extract target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        y = df[target_column].values

        # Extract features
        if feature_columns:
            missing = set(feature_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Feature columns not found: {missing}")
            X_df = df[feature_columns]
        else:
            X_df = df.drop(columns=[target_column])

        # Handle missing values
        X_df, y = self._handle_missing_values(X_df, y)

        # Encode categoricals
        X_df, feature_names = self._encode_categoricals(X_df)

        # Convert to numpy
        X = X_df.values.astype(np.float64)
        y = y.astype(np.float64) if np.issubdtype(y.dtype, np.number) else y

        # Normalize if configured
        X = self._normalize(X)

        logger.info(
            f"Ingested {len(X)} samples, {X.shape[1]} features from {source.name}"
        )

        return X, y, feature_names

    def _read_file(self, path: Path, **kwargs) -> "pd.DataFrame":
        """Read file based on extension."""
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(path, **kwargs)
        elif suffix in [".xlsx", ".xls"]:
            return pd.read_excel(path, **kwargs)
        elif suffix == ".json":
            return pd.read_json(path, **kwargs)
        elif suffix == ".parquet":
            return pd.read_parquet(path, **kwargs)
        else:
            # Try CSV as fallback
            logger.warning(f"Unknown extension {suffix}, trying CSV")
            return pd.read_csv(path, **kwargs)

    def _handle_missing_values(
        self,
        X_df: "pd.DataFrame",
        y: np.ndarray,
    ) -> Tuple["pd.DataFrame", np.ndarray]:
        """Handle missing values according to config."""
        strategy = self.config.missing_value_strategy

        if strategy == MissingValueStrategy.DROP:
            # Drop rows with any missing values
            mask = ~X_df.isna().any(axis=1)
            X_df = X_df[mask]
            y = y[mask.values]

        elif strategy == MissingValueStrategy.MEAN:
            # Fill numeric with mean
            for col in X_df.select_dtypes(include=[np.number]).columns:
                X_df[col] = X_df[col].fillna(X_df[col].mean())

        elif strategy == MissingValueStrategy.MEDIAN:
            # Fill numeric with median
            for col in X_df.select_dtypes(include=[np.number]).columns:
                X_df[col] = X_df[col].fillna(X_df[col].median())

        elif strategy == MissingValueStrategy.MODE:
            # Fill with mode
            for col in X_df.columns:
                mode_val = X_df[col].mode()
                if len(mode_val) > 0:
                    X_df[col] = X_df[col].fillna(mode_val[0])

        elif strategy == MissingValueStrategy.CONSTANT:
            fill_value = self.config.missing_fill_value or 0
            X_df = X_df.fillna(fill_value)

        elif strategy == MissingValueStrategy.INTERPOLATE:
            X_df = X_df.interpolate(method="linear", limit_direction="both")
            X_df = X_df.fillna(method="ffill").fillna(method="bfill")

        return X_df, y

    def _encode_categoricals(
        self,
        X_df: "pd.DataFrame",
    ) -> Tuple["pd.DataFrame", FeatureNames]:
        """Encode categorical columns."""
        categorical_cols = X_df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if not categorical_cols:
            return X_df, X_df.columns.tolist()

        encoding = self.config.categorical_encoding

        if encoding == "onehot":
            # One-hot encoding
            for col in categorical_cols:
                # Limit categories
                if X_df[col].nunique() > self.config.max_categories:
                    # Keep top N-1 categories, rest become 'other'
                    top_cats = X_df[col].value_counts().head(
                        self.config.max_categories - 1
                    ).index
                    X_df[col] = X_df[col].where(X_df[col].isin(top_cats), "other")

            X_df = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True)

        elif encoding == "label":
            # Label encoding
            for col in categorical_cols:
                X_df[col] = X_df[col].astype("category").cat.codes

        # Get final feature names
        feature_names = X_df.columns.tolist()

        return X_df, feature_names

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize features according to config."""
        method = self.config.normalization

        if method == NormalizationMethod.NONE:
            return X

        elif method == NormalizationMethod.STANDARD:
            # Z-score normalization
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            std[std == 0] = 1  # Avoid division by zero
            return (X - mean) / std

        elif method == NormalizationMethod.MINMAX:
            # Scale to [0, 1]
            min_val = X.min(axis=0)
            max_val = X.max(axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1
            return (X - min_val) / range_val

        elif method == NormalizationMethod.ROBUST:
            # Robust scaling using median and IQR
            median = np.median(X, axis=0)
            q1 = np.percentile(X, 25, axis=0)
            q3 = np.percentile(X, 75, axis=0)
            iqr = q3 - q1
            iqr[iqr == 0] = 1
            return (X - median) / iqr

        return X

    async def ingest_and_split(
        self,
        source: Union[str, Path],
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        **read_kwargs: Any,
    ) -> DataSplit:
        """
        Ingest data and create train/test split.

        Args:
            source: File path
            target_column: Target column name
            feature_columns: Optional feature columns
            **read_kwargs: Additional read arguments

        Returns:
            DataSplit with train/test data
        """
        X, y, feature_names = await self.ingest_tabular(
            source, target_column, feature_columns, **read_kwargs
        )

        # Split data
        n_samples = len(X)
        n_test = int(n_samples * self.config.test_size)
        n_val = int(n_samples * self.config.val_size) if self.config.val_size > 0 else 0

        if self.config.shuffle:
            rng = np.random.default_rng(self.config.random_state)
            indices = rng.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        # Split indices
        test_indices = indices[:n_test]
        val_indices = indices[n_test : n_test + n_val] if n_val > 0 else None
        train_indices = indices[n_test + n_val :]

        # Create split arrays
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        X_val, y_val = (
            (X[val_indices], y[val_indices]) if val_indices is not None else (None, None)
        )

        return DataSplit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            target_name=target_column,
        )

    async def ingest_from_memory_shards(
        self,
        shards: List[Any],
        target_column: str,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, FeatureNames]:
        """
        Ingest tabular data from HoloLoom MemoryShards.

        Args:
            shards: List of MemoryShard objects with tabular content
            target_column: Target column name
            feature_columns: Optional feature columns

        Returns:
            Tuple of (X, y, feature_names)
        """
        self._check_pandas()

        # Collect data from shards
        records = []
        for shard in shards:
            if hasattr(shard, "content") and isinstance(shard.content, dict):
                records.append(shard.content)
            elif hasattr(shard, "to_dict"):
                records.append(shard.to_dict())

        if not records:
            raise ValueError("No tabular data found in memory shards")

        df = pd.DataFrame(records)

        # Use existing processing pipeline
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        y = df[target_column].values

        if feature_columns:
            X_df = df[feature_columns]
        else:
            X_df = df.drop(columns=[target_column])

        X_df, y = self._handle_missing_values(X_df, y)
        X_df, feature_names = self._encode_categoricals(X_df)
        X = X_df.values.astype(np.float64)
        X = self._normalize(X)

        return X, y, feature_names
