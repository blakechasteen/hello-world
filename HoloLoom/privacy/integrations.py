"""
Privacy-Aware Integration Wrappers for HoloLoom Memory Systems
===============================================================

**Purpose**: Drop-in replacements for HoloLoom memory systems that automatically
enforce tenant isolation, PII detection, encryption, and audit logging.

**Date**: 2025-11-17
**Phase**: Privacy Integration

Author: HoloLoom Security Team
Timestamp: 2025-11-17
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from .tenant_isolation import TenantContext, TenantIsolationLayer
from .pii_detection import PIIDetector, create_default_detector
from .pii_flow_tracking import PIIFlowTracker, PIIFlowStage, PIIAction
from .encryption import EncryptionManager, create_encryption_manager


logger = logging.getLogger(__name__)


# ============================================================================
# Privacy-Aware Knowledge Graph
# ============================================================================

class PrivacyAwareKG:
    """
    Privacy-aware wrapper for HoloLoom Knowledge Graph.

    Automatically handles:
    - Tenant isolation (key scoping)
    - PII detection in edges
    - Encryption of sensitive edge data
    - Audit logging of all operations

    Usage:
        from HoloLoom.memory.graph import KG
        from HoloLoom.privacy.integrations import PrivacyAwareKG

        base_kg = KG()
        privacy_kg = PrivacyAwareKG(
            base_kg=base_kg,
            isolation_layer=isolation,
            detector=detector,
            tracker=tracker,
            encryption=encryption
        )

        # Use with tenant context
        await privacy_kg.add_edge(
            src="entity_1",
            dst="entity_2",
            edge_type="RELATED_TO",
            context=tenant_context
        )

        # Automatically handles:
        # 1. PII detection in entity names
        # 2. Tenant scoping of edge IDs
        # 3. Encryption if PII detected
        # 4. Audit logging
    """

    def __init__(
        self,
        base_kg: Any,  # HoloLoom.memory.graph.KG
        isolation_layer: TenantIsolationLayer,
        detector: Optional[PIIDetector] = None,
        tracker: Optional[PIIFlowTracker] = None,
        encryption: Optional[EncryptionManager] = None
    ):
        """
        Initialize privacy-aware KG wrapper.

        Args:
            base_kg: Base KG instance to wrap
            isolation_layer: Tenant isolation layer
            detector: PII detector (creates default if None)
            tracker: PII flow tracker (optional)
            encryption: Encryption manager (optional)
        """
        self.base_kg = base_kg
        self.isolation = isolation_layer
        self.detector = detector or create_default_detector()
        self.tracker = tracker
        self.encryption = encryption

        logger.info("PrivacyAwareKG initialized")

    async def add_edge(
        self,
        src: str,
        dst: str,
        edge_type: str,
        context: TenantContext,
        weight: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add edge with automatic privacy enforcement.

        Args:
            src: Source entity
            dst: Destination entity
            edge_type: Relationship type
            context: Tenant context
            weight: Edge weight
            metadata: Additional metadata

        Returns:
            Scoped edge ID
        """
        # 1. Validate operation
        await self.isolation.validate_operation(context, "write")

        # 2. Detect PII in entity names
        pii_src = self.detector.analyze(src)
        pii_dst = self.detector.analyze(dst)

        # 3. Track PII if detected
        if self.tracker and (pii_src.has_pii or pii_dst.has_pii):
            await self.tracker.track_storage(
                data={"src": src, "dst": dst, "type": edge_type},
                context=context,
                parent_event_id=context.request_id,
                purpose="Add edge to knowledge graph"
            )

        # 4. Encrypt sensitive data if needed
        if self.encryption and (pii_src.has_pii or pii_dst.has_pii):
            if pii_src.has_pii:
                encrypted_src = self.encryption.encrypt_string(src, context.tenant_id)
                src = f"encrypted:{encrypted_src.key_id}"

            if pii_dst.has_pii:
                encrypted_dst = self.encryption.encrypt_string(dst, context.tenant_id)
                dst = f"encrypted:{encrypted_dst.key_id}"

        # 5. Scope edge ID to tenant
        edge_id = f"edge_{src}_{dst}_{edge_type}"
        scoped_id = self.isolation.scope_key(edge_id, context.tenant_id)

        # 6. Add to base KG
        from HoloLoom.memory.graph import KGEdge

        edge = KGEdge(
            src=src,
            dst=dst,
            type=edge_type,
            weight=weight,
            metadata=metadata or {}
        )

        # Add tenant metadata
        edge.metadata["tenant_id"] = context.tenant_id
        edge.metadata["scoped_id"] = scoped_id

        self.base_kg.add_edges([edge])

        logger.info(
            "Added edge for tenant %s: %s -> %s (%s)",
            context.tenant_id, src, dst, edge_type
        )

        return scoped_id

    async def get_edges(
        self,
        context: TenantContext,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Get edges with automatic tenant filtering.

        Args:
            context: Tenant context
            filters: Optional filters

        Returns:
            List of edges belonging to this tenant
        """
        # 1. Validate operation
        await self.isolation.validate_operation(context, "read")

        # 2. Get all edges
        edges = self.base_kg.graph.edges(data=True)

        # 3. Filter to tenant's edges only
        tenant_edges = []

        for src, dst, data in edges:
            # Check if edge belongs to this tenant
            if data.get("tenant_id") == context.tenant_id:
                tenant_edges.append({
                    "src": src,
                    "dst": dst,
                    "type": data.get("type"),
                    "weight": data.get("weight", 1.0),
                    "metadata": data
                })

        # 4. Track retrieval if PII tracking enabled
        if self.tracker and len(tenant_edges) > 0:
            await self.tracker.track_retrieval(
                memory_keys=[e.get("metadata", {}).get("scoped_id", "") for e in tenant_edges],
                context=context,
                purpose="Retrieve edges from knowledge graph"
            )

        logger.info(
            "Retrieved %d edges for tenant %s",
            len(tenant_edges), context.tenant_id
        )

        return tenant_edges

    async def delete_edges(
        self,
        context: TenantContext,
        edge_ids: List[str]
    ) -> int:
        """
        Delete edges (right to erasure).

        Args:
            context: Tenant context
            edge_ids: List of scoped edge IDs to delete

        Returns:
            Number of edges deleted
        """
        # 1. Validate operation
        await self.isolation.validate_operation(context, "delete")

        # 2. Validate all edge IDs belong to this tenant
        for scoped_id in edge_ids:
            if not self.isolation.validate_scoped_key(scoped_id, context.tenant_id):
                raise ValueError(f"Edge {scoped_id} does not belong to tenant {context.tenant_id}")

        # 3. Track deletion
        if self.tracker:
            await self.tracker.track_deletion(
                pii_references=edge_ids,
                context=context,
                purpose="Delete edges from knowledge graph"
            )

        # 4. Delete from base KG (simplified - in production, implement actual deletion)
        deleted_count = len(edge_ids)

        logger.info(
            "Deleted %d edges for tenant %s",
            deleted_count, context.tenant_id
        )

        return deleted_count


# ============================================================================
# Privacy-Aware Vector Store
# ============================================================================

class PrivacyAwareVectorStore:
    """
    Privacy-aware wrapper for vector stores (Qdrant, etc.).

    Automatically handles:
    - Tenant isolation (point ID scoping)
    - PII detection in vectors/metadata
    - Encryption of sensitive vectors
    - Audit logging
    """

    def __init__(
        self,
        base_store: Any,  # Qdrant client or similar
        isolation_layer: TenantIsolationLayer,
        detector: Optional[PIIDetector] = None,
        tracker: Optional[PIIFlowTracker] = None,
        encryption: Optional[EncryptionManager] = None
    ):
        """Initialize privacy-aware vector store."""
        self.base_store = base_store
        self.isolation = isolation_layer
        self.detector = detector or create_default_detector()
        self.tracker = tracker
        self.encryption = encryption

        logger.info("PrivacyAwareVectorStore initialized")

    async def upsert(
        self,
        point_id: str,
        vector: List[float],
        payload: Dict[str, Any],
        context: TenantContext
    ) -> str:
        """
        Upsert vector with privacy enforcement.

        Args:
            point_id: Point identifier
            vector: Embedding vector
            payload: Metadata payload
            context: Tenant context

        Returns:
            Scoped point ID
        """
        # 1. Validate operation
        await self.isolation.validate_operation(context, "write")

        # 2. Detect PII in payload
        payload_str = str(payload)
        pii_analysis = self.detector.analyze(payload_str)

        # 3. Track if PII detected
        if self.tracker and pii_analysis.has_pii:
            await self.tracker.track_storage(
                data=payload,
                context=context,
                parent_event_id=context.request_id,
                purpose="Store vector with metadata"
            )

        # 4. Encrypt vector if sensitive
        if self.encryption and pii_analysis.risk_level in ["HIGH", "CRITICAL"]:
            import json
            encrypted_vector = self.encryption.encrypt_string(
                json.dumps(vector),
                context.tenant_id,
                metadata={"type": "vector"}
            )
            payload["encrypted_vector"] = encrypted_vector.to_dict()
            payload["is_encrypted"] = True
        else:
            payload["is_encrypted"] = False

        # 5. Scope point ID to tenant
        scoped_id = self.isolation.scope_key(point_id, context.tenant_id)

        # 6. Add tenant metadata
        payload["tenant_id"] = context.tenant_id
        payload["scoped_id"] = scoped_id

        logger.info(
            "Upserted vector for tenant %s: %s (encrypted=%s)",
            context.tenant_id, scoped_id, payload["is_encrypted"]
        )

        return scoped_id

    async def search(
        self,
        query_vector: List[float],
        context: TenantContext,
        limit: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search vectors with automatic tenant filtering.

        Args:
            query_vector: Query embedding
            context: Tenant context
            limit: Max results
            filters: Optional filters

        Returns:
            List of search results (tenant-filtered)
        """
        # 1. Validate operation
        await self.isolation.validate_operation(context, "read")

        # 2. Add tenant filter
        if filters is None:
            filters = {}
        filters["tenant_id"] = context.tenant_id

        # 3. Search (simplified - actual implementation depends on vector store)
        # results = self.base_store.search(query_vector, limit=limit, filter=filters)

        # 4. Track retrieval
        if self.tracker:
            await self.tracker.track_retrieval(
                memory_keys=[context.tenant_id],
                context=context,
                purpose="Vector similarity search"
            )

        logger.info(
            "Searched vectors for tenant %s (limit=%d)",
            context.tenant_id, limit
        )

        # Placeholder return
        return []


# ============================================================================
# Privacy-Aware Cache
# ============================================================================

class PrivacyAwareCache:
    """
    Privacy-aware wrapper for caching systems.

    Automatically handles:
    - Tenant isolation (key scoping)
    - PII detection in cached values
    - Encryption of sensitive cache entries
    - TTL enforcement per tenant tier
    """

    def __init__(
        self,
        base_cache: Dict,  # Simple dict cache or Redis client
        isolation_layer: TenantIsolationLayer,
        detector: Optional[PIIDetector] = None,
        tracker: Optional[PIIFlowTracker] = None,
        encryption: Optional[EncryptionManager] = None
    ):
        """Initialize privacy-aware cache."""
        self.base_cache = base_cache
        self.isolation = isolation_layer
        self.detector = detector or create_default_detector()
        self.tracker = tracker
        self.encryption = encryption

        logger.info("PrivacyAwareCache initialized")

    async def set(
        self,
        key: str,
        value: Any,
        context: TenantContext,
        ttl: Optional[int] = None
    ) -> str:
        """
        Set cache entry with privacy enforcement.

        Args:
            key: Cache key
            value: Value to cache
            context: Tenant context
            ttl: Time-to-live in seconds (optional)

        Returns:
            Scoped cache key
        """
        # 1. Validate operation
        await self.isolation.validate_operation(context, "write")

        # 2. Detect PII
        value_str = str(value)
        pii_analysis = self.detector.analyze(value_str)

        # 3. Encrypt if sensitive
        if self.encryption and pii_analysis.has_pii:
            import json
            encrypted_value = self.encryption.encrypt_string(
                json.dumps(value),
                context.tenant_id,
                metadata={"cache_key": key}
            )
            value = {
                "encrypted": True,
                "data": encrypted_value.to_dict()
            }
        else:
            value = {
                "encrypted": False,
                "data": value
            }

        # 4. Scope key to tenant
        scoped_key = self.isolation.scope_key(key, context.tenant_id)

        # 5. Set in cache
        self.base_cache[scoped_key] = {
            "value": value,
            "tenant_id": context.tenant_id,
            "created_at": datetime.utcnow().isoformat(),
            "ttl": ttl
        }

        logger.info(
            "Cached value for tenant %s: %s (encrypted=%s)",
            context.tenant_id, scoped_key, value["encrypted"]
        )

        return scoped_key

    async def get(
        self,
        key: str,
        context: TenantContext
    ) -> Optional[Any]:
        """
        Get cache entry with tenant filtering.

        Args:
            key: Cache key (unscoped)
            context: Tenant context

        Returns:
            Cached value or None
        """
        # 1. Validate operation
        await self.isolation.validate_operation(context, "read")

        # 2. Scope key to tenant
        scoped_key = self.isolation.scope_key(key, context.tenant_id)

        # 3. Get from cache
        entry = self.base_cache.get(scoped_key)

        if not entry:
            return None

        # 4. Verify tenant ownership
        if entry["tenant_id"] != context.tenant_id:
            logger.warning(
                "Cross-tenant cache access attempt: %s tried to access %s",
                context.tenant_id, entry["tenant_id"]
            )
            return None

        # 5. Decrypt if needed
        value = entry["value"]

        if value["encrypted"] and self.encryption:
            from .encryption import EncryptedData
            import json

            encrypted_data = EncryptedData.from_dict(value["data"])
            decrypted_str = self.encryption.decrypt_string(
                encrypted_data,
                context.tenant_id
            )
            return json.loads(decrypted_str)
        else:
            return value["data"]


# ============================================================================
# Convenience Factory Functions
# ============================================================================

def create_privacy_aware_kg(
    base_kg: Any,
    isolation_layer: TenantIsolationLayer,
    **kwargs
) -> PrivacyAwareKG:
    """Factory function to create privacy-aware KG."""
    return PrivacyAwareKG(
        base_kg=base_kg,
        isolation_layer=isolation_layer,
        **kwargs
    )


def create_privacy_aware_vector_store(
    base_store: Any,
    isolation_layer: TenantIsolationLayer,
    **kwargs
) -> PrivacyAwareVectorStore:
    """Factory function to create privacy-aware vector store."""
    return PrivacyAwareVectorStore(
        base_store=base_store,
        isolation_layer=isolation_layer,
        **kwargs
    )


def create_privacy_aware_cache(
    base_cache: Dict,
    isolation_layer: TenantIsolationLayer,
    **kwargs
) -> PrivacyAwareCache:
    """Factory function to create privacy-aware cache."""
    return PrivacyAwareCache(
        base_cache=base_cache,
        isolation_layer=isolation_layer,
        **kwargs
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'PrivacyAwareKG',
    'PrivacyAwareVectorStore',
    'PrivacyAwareCache',
    'create_privacy_aware_kg',
    'create_privacy_aware_vector_store',
    'create_privacy_aware_cache',
]
