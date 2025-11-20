#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DS-STAR Data Analyzer
=====================

Extracts "anchors" (key information) from data files.
Maps to HoloLoom's SpinningWheel adapters.

Created: 2025-01-20
Based on: Google Research DS-STAR paper
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np


@dataclass
class DataAnchor:
    """Key information extracted from data."""
    name: str                    # Column/field name
    type: str                    # Data type (numeric, categorical, text, etc.)
    sample_values: List[Any]     # Representative samples
    statistics: Dict[str, Any]   # Min, max, mean, unique count, etc.
    missing_count: int = 0
    unique_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataProfile:
    """Complete profile of a dataset."""
    file_path: str
    format: str                  # csv, json, parquet, etc.
    row_count: int
    column_count: int
    anchors: List[DataAnchor]
    schema: Dict[str, str]       # Column name → type
    relationships: List[Dict[str, Any]] = field(default_factory=list)  # Detected relationships
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataAnalyzer:
    """
    Analyzes data files and extracts anchors.

    Integrates with HoloLoom's SpinningWheel for various formats.
    """

    def __init__(self, max_sample_size: int = 5, max_unique_show: int = 10):
        self.max_sample_size = max_sample_size
        self.max_unique_show = max_unique_show

    def analyze_file(self, file_path: str) -> DataProfile:
        """
        Analyze a data file and extract profile.

        Args:
            file_path: Path to data file

        Returns:
            DataProfile with anchors and statistics
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Detect format
        format_type = path.suffix.lower().lstrip('.')

        # Load data based on format
        if format_type == 'csv':
            df = pd.read_csv(file_path)
        elif format_type in ['xlsx', 'xls']:
            df = pd.read_excel(file_path)
        elif format_type == 'json':
            df = pd.read_json(file_path)
        elif format_type == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        # Extract anchors for each column
        anchors = []
        for col in df.columns:
            anchor = self._analyze_column(df, col)
            anchors.append(anchor)

        # Build schema
        schema = {col: str(df[col].dtype) for col in df.columns}

        # Detect relationships (correlations for numeric columns)
        relationships = self._detect_relationships(df)

        return DataProfile(
            file_path=file_path,
            format=format_type,
            row_count=len(df),
            column_count=len(df.columns),
            anchors=anchors,
            schema=schema,
            relationships=relationships,
            metadata={
                'memory_usage': df.memory_usage(deep=True).sum(),
                'has_nulls': df.isnull().any().any(),
            }
        )

    def _analyze_column(self, df: pd.DataFrame, col: str) -> DataAnchor:
        """Analyze a single column."""
        series = df[col]
        dtype = series.dtype

        # Determine type
        if pd.api.types.is_numeric_dtype(dtype):
            col_type = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            col_type = 'datetime'
        elif pd.api.types.is_bool_dtype(dtype):
            col_type = 'boolean'
        elif pd.api.types.is_categorical_dtype(dtype) or series.nunique() < 20:
            col_type = 'categorical'
        else:
            col_type = 'text'

        # Sample values (non-null)
        valid_values = series.dropna()
        sample_size = min(self.max_sample_size, len(valid_values))
        sample_values = valid_values.sample(n=sample_size).tolist() if len(valid_values) > 0 else []

        # Statistics
        statistics = {}
        if col_type == 'numeric':
            statistics = {
                'min': float(series.min()) if not series.empty else None,
                'max': float(series.max()) if not series.empty else None,
                'mean': float(series.mean()) if not series.empty else None,
                'median': float(series.median()) if not series.empty else None,
                'std': float(series.std()) if not series.empty else None,
            }
        elif col_type == 'categorical':
            value_counts = series.value_counts()
            statistics = {
                'mode': str(series.mode().iloc[0]) if not series.mode().empty else None,
                'top_values': value_counts.head(self.max_unique_show).to_dict(),
            }
        elif col_type == 'text':
            statistics = {
                'avg_length': series.str.len().mean() if not series.empty else 0,
                'max_length': series.str.len().max() if not series.empty else 0,
            }

        return DataAnchor(
            name=col,
            type=col_type,
            sample_values=sample_values,
            statistics=statistics,
            missing_count=int(series.isnull().sum()),
            unique_count=int(series.nunique()),
            metadata={'dtype': str(dtype)}
        )

    def _detect_relationships(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect relationships between columns."""
        relationships = []

        # Numeric correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()

            # Find strong correlations (> 0.7 or < -0.7)
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j:  # Avoid duplicates
                        corr = corr_matrix.loc[col1, col2]
                        if abs(corr) > 0.7:
                            relationships.append({
                                'type': 'correlation',
                                'column1': col1,
                                'column2': col2,
                                'strength': float(corr),
                            })

        return relationships

    def summarize_profile(self, profile: DataProfile) -> str:
        """
        Generate human-readable summary of data profile.

        Args:
            profile: DataProfile to summarize

        Returns:
            Markdown-formatted summary
        """
        lines = [
            f"# Data Profile: {Path(profile.file_path).name}",
            f"",
            f"**Format**: {profile.format}",
            f"**Dimensions**: {profile.row_count:,} rows × {profile.column_count} columns",
            f"",
            f"## Columns",
            f""
        ]

        for anchor in profile.anchors:
            lines.append(f"### {anchor.name} ({anchor.type})")
            lines.append(f"- **Unique values**: {anchor.unique_count:,}")
            lines.append(f"- **Missing**: {anchor.missing_count:,} ({anchor.missing_count/profile.row_count*100:.1f}%)")

            if anchor.statistics:
                lines.append(f"- **Statistics**: {anchor.statistics}")

            if anchor.sample_values:
                lines.append(f"- **Samples**: {anchor.sample_values}")

            lines.append("")

        if profile.relationships:
            lines.append("## Detected Relationships")
            for rel in profile.relationships:
                if rel['type'] == 'correlation':
                    lines.append(f"- **{rel['column1']}** ↔ **{rel['column2']}**: {rel['strength']:.2f} correlation")
            lines.append("")

        return "\n".join(lines)


# Convenience function
def analyze_data(file_path: str) -> DataProfile:
    """
    Quick analysis of a data file.

    Args:
        file_path: Path to data file

    Returns:
        DataProfile with anchors
    """
    analyzer = DataAnalyzer()
    return analyzer.analyze_file(file_path)
