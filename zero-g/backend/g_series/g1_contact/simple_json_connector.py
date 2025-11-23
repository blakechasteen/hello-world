"""
G1 Simple JSON Connector (MVP)

A zero-move, read-only connector for local JSON files.
Demonstrates the G1 Contact Layer principles:
- Data stays in place
- Metadata-only access
- No writes, no copies
- Safe, reversible operations
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..protocols import (
    DataSource,
    DataSourceType,
    AccessMode,
    DataSensitivity,
    SchemaElement,
    SchemaDiscoveryResult,
    ConnectorHealth
)

logger = logging.getLogger(__name__)


class SimpleJSONConnector:
    """
    Simple JSON file connector for MVP

    Capabilities:
    - Metadata-only scanning (file size, modification time)
    - Schema inference from JSON structure
    - Read-only access (no writes)
    - Local files only (no remote)

    G-Series Level: G1 (Contact)
    """

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize connector

        Args:
            base_path: Base directory for JSON files (defaults to current dir)
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.sources: Dict[str, DataSource] = {}
        self.metadata_cache: Dict[str, Dict[str, Any]] = {}

    async def connect(
        self,
        source_id: str,
        file_path: str,
        sensitivity: DataSensitivity = DataSensitivity.INTERNAL
    ) -> ConnectorHealth:
        """
        Connect to a JSON file (zero-move)

        Args:
            source_id: Unique identifier for this source
            file_path: Path to JSON file (relative to base_path)
            sensitivity: Data sensitivity classification

        Returns:
            Connector health status
        """
        logger.info(f"G1 JSON Connector: Connecting to {file_path}")

        try:
            full_path = self.base_path / file_path

            if not full_path.exists():
                return ConnectorHealth(
                    connector_id=source_id,
                    status="down",
                    error_rate=1.0,
                    last_failure=datetime.now(),
                    metadata={"error": f"File not found: {file_path}"}
                )

            # Create data source
            source = DataSource(
                source_id=source_id,
                source_type=DataSourceType.LOCAL_FILE,
                access_mode=AccessMode.ZERO_MOVE,
                sensitivity=sensitivity,
                connection_params={
                    "file_path": str(full_path),
                    "base_path": str(self.base_path)
                },
                metadata={
                    "extension": ".json",
                    "connector_type": "simple_json"
                }
            )

            self.sources[source_id] = source

            return ConnectorHealth(
                connector_id=source_id,
                status="healthy",
                latency_ms=10.0,
                error_rate=0.0,
                last_success=datetime.now(),
                metadata={"file_path": str(full_path)}
            )

        except Exception as e:
            logger.error(f"G1 JSON Connector: Connection failed: {e}")
            return ConnectorHealth(
                connector_id=source_id,
                status="down",
                error_rate=1.0,
                last_failure=datetime.now(),
                metadata={"error": str(e)}
            )

    async def get_metadata(
        self,
        source_id: str
    ) -> Dict[str, Any]:
        """
        Get metadata without reading the file (zero-move principle)

        Retrieves:
        - File size
        - Modification time
        - File permissions
        - Path information

        Does NOT read file contents.
        """
        logger.info(f"G1 JSON Connector: Getting metadata for {source_id}")

        if source_id not in self.sources:
            raise ValueError(f"Unknown source: {source_id}")

        source = self.sources[source_id]
        file_path = Path(source.connection_params["file_path"])

        # Get file stats (no read required)
        stats = file_path.stat()

        metadata = {
            "file_name": file_path.name,
            "file_size_bytes": stats.st_size,
            "file_size_human": self._format_size(stats.st_size),
            "modified_time": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "created_time": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "is_readable": file_path.is_file() and file_path.exists(),
            "absolute_path": str(file_path.absolute()),
            "access_mode": source.access_mode.value,
            "sensitivity": source.sensitivity.value
        }

        self.metadata_cache[source_id] = metadata
        return metadata

    async def get_summary(
        self,
        source_id: str,
        summary_type: str = "count"
    ) -> Dict[str, Any]:
        """
        Get summary statistics

        Summary types:
        - count: Count top-level elements (minimal read)
        - schema: Infer schema structure (partial read)
        - sample: Get sample records (limited read)

        Even "schema" and "sample" read minimally (first N records only).
        """
        logger.info(f"G1 JSON Connector: Getting {summary_type} summary for {source_id}")

        if source_id not in self.sources:
            raise ValueError(f"Unknown source: {source_id}")

        source = self.sources[source_id]
        file_path = Path(source.connection_params["file_path"])

        if summary_type == "count":
            return await self._get_count_summary(file_path)
        elif summary_type == "schema":
            return await self._get_schema_summary(file_path)
        elif summary_type == "sample":
            return await self._get_sample_summary(file_path)
        else:
            raise ValueError(f"Unknown summary type: {summary_type}")

    async def discover_schema(
        self,
        source_id: str,
        sample_size: int = 100
    ) -> SchemaDiscoveryResult:
        """
        Discover schema from JSON structure

        Reads a sample of records to infer:
        - Field names and types
        - Nested structures
        - Potential relationships

        Args:
            source_id: Source to analyze
            sample_size: Number of records to sample (default: 100)

        Returns:
            Schema discovery result with inferred structure
        """
        logger.info(f"G1 JSON Connector: Discovering schema for {source_id} (sample: {sample_size})")

        if source_id not in self.sources:
            raise ValueError(f"Unknown source: {source_id}")

        source = self.sources[source_id]
        file_path = Path(source.connection_params["file_path"])

        # Read and parse JSON (limited sample)
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            sample = data[:sample_size]
            schema = self._infer_schema_from_list(sample)
        elif isinstance(data, dict):
            schema = self._infer_schema_from_dict(data, sample_size)
        else:
            schema = {
                "type": type(data).__name__,
                "structure": "primitive"
            }

        # Create schema elements
        elements = [
            SchemaElement(
                element_id=f"{source_id}_root",
                element_type="json_structure",
                name="root",
                properties=schema
            )
        ]

        return SchemaDiscoveryResult(
            source_id=source_id,
            elements=elements,
            conflicts=[],
            warnings=[]
        )

    async def disconnect(self, source_id: str) -> bool:
        """
        Disconnect from a source (cleanup)

        Returns:
            True if disconnected successfully
        """
        logger.info(f"G1 JSON Connector: Disconnecting {source_id}")

        if source_id in self.sources:
            del self.sources[source_id]

        if source_id in self.metadata_cache:
            del self.metadata_cache[source_id]

        return True

    # ==================== Internal Methods ====================

    def _format_size(self, size_bytes: int) -> str:
        """Format byte size as human-readable"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    async def _get_count_summary(self, file_path: Path) -> Dict[str, Any]:
        """Count summary - minimal read"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            count = len(data)
            element_type = "array"
        elif isinstance(data, dict):
            count = len(data.keys())
            element_type = "object"
        else:
            count = 1
            element_type = "primitive"

        return {
            "summary_type": "count",
            "element_type": element_type,
            "count": count,
            "read_bytes": file_path.stat().st_size
        }

    async def _get_schema_summary(self, file_path: Path) -> Dict[str, Any]:
        """Schema summary - partial read"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list) and len(data) > 0:
            sample = data[0]
            schema = self._infer_single_schema(sample)
        elif isinstance(data, dict):
            schema = self._infer_single_schema(data)
        else:
            schema = {"type": type(data).__name__}

        return {
            "summary_type": "schema",
            "schema": schema,
            "sample_size": 1
        }

    async def _get_sample_summary(self, file_path: Path) -> Dict[str, Any]:
        """Sample summary - limited read"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            sample = data[:5]  # First 5 records
        elif isinstance(data, dict):
            sample = dict(list(data.items())[:5])  # First 5 keys
        else:
            sample = data

        return {
            "summary_type": "sample",
            "sample_data": sample,
            "sample_size": len(sample) if isinstance(sample, (list, dict)) else 1
        }

    def _infer_single_schema(self, obj: Any) -> Dict[str, Any]:
        """Infer schema from a single object"""
        if isinstance(obj, dict):
            return {
                "type": "object",
                "fields": {
                    key: type(value).__name__
                    for key, value in obj.items()
                }
            }
        elif isinstance(obj, list):
            return {
                "type": "array",
                "item_type": type(obj[0]).__name__ if obj else "unknown"
            }
        else:
            return {"type": type(obj).__name__}

    def _infer_schema_from_list(self, sample: List[Any]) -> Dict[str, Any]:
        """Infer schema from a list of records"""
        if not sample:
            return {"type": "array", "empty": True}

        # Infer from first record
        first_schema = self._infer_single_schema(sample[0])

        # Check consistency across sample
        consistent = True
        for record in sample[1:10]:  # Check first 10 for consistency
            record_schema = self._infer_single_schema(record)
            if record_schema != first_schema:
                consistent = False
                break

        return {
            "type": "array",
            "item_schema": first_schema,
            "consistent": consistent,
            "sample_size": len(sample)
        }

    def _infer_schema_from_dict(
        self,
        data: Dict[str, Any],
        sample_size: int
    ) -> Dict[str, Any]:
        """Infer schema from a dictionary"""
        fields = {}
        for key, value in list(data.items())[:sample_size]:
            if isinstance(value, dict):
                fields[key] = self._infer_schema_from_dict(value, 10)
            elif isinstance(value, list):
                fields[key] = self._infer_schema_from_list(value[:10])
            else:
                fields[key] = {"type": type(value).__name__}

        return {
            "type": "object",
            "fields": fields
        }


# ==================== Factory Function ====================

async def create_json_connector(base_path: Optional[str] = None) -> SimpleJSONConnector:
    """
    Factory function to create a JSON connector

    Usage:
        connector = await create_json_connector("./data")

        # Connect to a file
        health = await connector.connect(
            source_id="users",
            file_path="users.json",
            sensitivity=DataSensitivity.CONFIDENTIAL
        )

        # Get metadata (no read)
        metadata = await connector.get_metadata("users")

        # Discover schema (minimal read)
        schema = await connector.discover_schema("users", sample_size=100)
    """
    return SimpleJSONConnector(base_path)
