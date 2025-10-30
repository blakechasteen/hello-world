"""
Structured Data Processor

Processes structured data (JSON, CSV, databases).
"""

import time
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import numpy as np
import json

from .protocol import (
    InputProcessorProtocol,
    ProcessedInput,
    ModalityType,
    StructuredFeatures,
    InputMetadata
)


class StructuredDataProcessor:
    """
    Structured data processor for JSON, CSV, and databases.
    
    Features:
    - JSON/CSV parsing
    - Schema detection
    - Column type inference
    - Relationship extraction
    - Summary statistics
    """
    
    def __init__(self, embedder=None):
        """Initialize structured data processor."""
        # If no embedder provided, create simple fallback
        if embedder is None:
            from .simple_embedder import StructuredEmbedder
            self.embedder = StructuredEmbedder()
        else:
            self.embedder = embedder
        
        # Try to load pandas
        self.pd = None
        try:
            import pandas as pd
            self.pd = pd
        except ImportError:
            print("Warning: pandas not available. Limited functionality.")
    
    async def process(
        self,
        input_data: Union[str, Path, Dict, bytes],
        data_format: Optional[str] = None,  # 'json', 'csv', 'auto'
        **kwargs
    ) -> ProcessedInput:
        """
        Process structured data input.
        
        Args:
            input_data: File path, dict, or raw data
            data_format: Data format ('json', 'csv', 'auto')
        
        Returns:
            ProcessedInput with structured data features
        """
        start_time = time.time()
        
        # Detect format and load data
        if isinstance(input_data, dict):
            data = input_data
            source = input_data.get('source')
            format_type = 'dict'
        elif isinstance(input_data, (str, Path)):
            path = Path(input_data)
            source = str(path)
            
            if data_format == 'auto' or data_format is None:
                data_format = self._detect_format(path)
            
            if data_format == 'json':
                data = self._load_json(path)
                format_type = 'json'
            elif data_format == 'csv':
                data = self._load_csv(path)
                format_type = 'csv'
            else:
                raise ValueError(f"Unsupported format: {data_format}")
        else:
            # Try to parse as JSON
            try:
                data = json.loads(input_data)
                format_type = 'json'
                source = None
            except:
                raise ValueError("Cannot parse input data")
        
        # Create features
        features = StructuredFeatures()
        
        # Analyze structure
        if isinstance(data, dict):
            features.schema = self._infer_schema_dict(data)
            features.column_count = len(data)
            features.row_count = 1
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                features.schema = self._infer_schema_list_of_dicts(data)
                features.column_count = len(features.schema)
                features.row_count = len(data)
        elif self.pd and hasattr(data, 'columns'):  # DataFrame
            features.schema = self._infer_schema_dataframe(data)
            features.column_count = len(data.columns)
            features.row_count = len(data)
            features.summary_stats = self._get_summary_stats(data)
        
        # Extract relationships (simple foreign key detection)
        features.relationships = self._extract_relationships(data, features.schema)
        
        # Generate embedding using embedder
        embedding = self.embedder.encode(data)
        
        # Create content description
        content = f"{format_type.upper()} data: {features.row_count} rows x {features.column_count} columns"
        if features.schema:
            sample_cols = list(features.schema.keys())[:3]
            content += f" | Columns: {', '.join(sample_cols)}"
        
        processing_time = (time.time() - start_time) * 1000
        
        return ProcessedInput(
            modality=ModalityType.STRUCTURED,
            content=content,
            embedding=embedding,
            confidence=0.95,
            source=source,
            features={'structured': features}
        )
    
    def _detect_format(self, path: Path) -> str:
        """Detect file format from extension."""
        suffix = path.suffix.lower()
        if suffix == '.json':
            return 'json'
        elif suffix in ['.csv', '.tsv']:
            return 'csv'
        else:
            raise ValueError(f"Cannot detect format for {path}")
    
    def _load_json(self, path: Path) -> Any:
        """Load JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_csv(self, path: Path) -> Any:
        """Load CSV file."""
        if self.pd:
            return self.pd.read_csv(path)
        else:
            # Fallback: load as list of dicts
            import csv
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)
    
    def _infer_schema_dict(self, data: Dict) -> Dict[str, str]:
        """Infer schema from dictionary."""
        schema = {}
        for key, value in data.items():
            schema[key] = self._infer_type(value)
        return schema
    
    def _infer_schema_list_of_dicts(self, data: List[Dict]) -> Dict[str, str]:
        """Infer schema from list of dictionaries."""
        if not data:
            return {}
        
        # Use first item to get keys
        schema = {}
        for key in data[0].keys():
            # Sample values from multiple rows
            values = [row.get(key) for row in data[:10] if key in row]
            schema[key] = self._infer_type_from_list(values)
        
        return schema
    
    def _infer_schema_dataframe(self, df) -> Dict[str, str]:
        """Infer schema from pandas DataFrame."""
        schema = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            if 'int' in dtype:
                schema[col] = 'integer'
            elif 'float' in dtype:
                schema[col] = 'float'
            elif 'bool' in dtype:
                schema[col] = 'boolean'
            elif 'datetime' in dtype:
                schema[col] = 'datetime'
            else:
                schema[col] = 'string'
        return schema
    
    def _infer_type(self, value: Any) -> str:
        """Infer type of single value."""
        if isinstance(value, bool):
            return 'boolean'
        elif isinstance(value, int):
            return 'integer'
        elif isinstance(value, float):
            return 'float'
        elif isinstance(value, str):
            return 'string'
        elif isinstance(value, list):
            return 'array'
        elif isinstance(value, dict):
            return 'object'
        elif value is None:
            return 'null'
        else:
            return 'unknown'
    
    def _infer_type_from_list(self, values: List[Any]) -> str:
        """Infer type from list of values."""
        if not values:
            return 'unknown'
        
        types = [self._infer_type(v) for v in values if v is not None]
        if not types:
            return 'null'
        
        # Return most common type
        from collections import Counter
        return Counter(types).most_common(1)[0][0]
    
    def _get_summary_stats(self, df) -> Dict[str, Any]:
        """Get summary statistics for DataFrame."""
        stats = {}
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric'] = {}
            for col in numeric_cols:
                stats['numeric'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        # Categorical columns
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            stats['categorical'] = {}
            for col in object_cols[:5]:  # Limit to 5 columns
                value_counts = df[col].value_counts().head(5)
                stats['categorical'][col] = {
                    'unique_count': int(df[col].nunique()),
                    'top_values': value_counts.to_dict()
                }
        
        return stats
    
    def _extract_relationships(self, data: Any, schema: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract potential relationships (foreign keys)."""
        relationships = []
        
        # Simple heuristic: look for columns ending with '_id'
        for col_name, col_type in schema.items():
            if col_name.endswith('_id') and col_type == 'integer':
                # Potential foreign key
                target_table = col_name[:-3]  # Remove '_id'
                relationships.append({
                    'from_column': col_name,
                    'to_table': target_table,
                    'type': 'foreign_key',
                    'confidence': 0.7
                })
        
        return relationships
    
    def _create_structure_embedding(self, features: StructuredFeatures, dim: int = 128) -> np.ndarray:
        """
        Create embedding from structure.
        
        Simple hash-based embedding of schema.
        """
        embedding = np.zeros(dim)
        
        # Hash column names and types
        for col_name, col_type in features.schema.items():
            hash_val = hash(f"{col_name}:{col_type}") % dim
            embedding[hash_val] += 1
        
        # Add row/column counts
        embedding[0] = features.row_count / 1000.0  # Normalized
        embedding[1] = features.column_count / 100.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def get_modality(self) -> ModalityType:
        """Return modality type."""
        return ModalityType.STRUCTURED
    
    def is_available(self) -> bool:
        """Check if processor is available."""
        return True  # Basic JSON parsing always available
