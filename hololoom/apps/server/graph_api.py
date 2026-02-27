#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG API
============

Level 3 Graph RAG endpoints for knowledge graph operations.

Features:
- Multi-hop graph traversal
- Entity extraction from code
- Relationship discovery
- Interactive graph visualization
- Path finding between entities

Created: 2025-01-20
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import re

router = APIRouter(prefix="/graph", tags=["graph"])
logger = logging.getLogger(__name__)

# Import HoloLoom components
try:
    from HoloLoom.memory.graph import KG, KGEdge
    from HoloLoom.visualization.knowledge_graph import render_knowledge_graph_from_kg, render_knowledge_graph_from_networkx
    import networkx as nx
    GRAPH_AVAILABLE = True
    logger.info("✓ GraphRAG components loaded successfully")
except ImportError as e:
    GRAPH_AVAILABLE = False
    logger.warning(f"⚠ Graph components not available: {e}")
    KG = None
    nx = None

# Global knowledge graph instance
kg_instance: Optional[Any] = None


def get_kg():
    """Get or create knowledge graph instance."""
    global kg_instance
    if kg_instance is None and GRAPH_AVAILABLE:
        kg_instance = KG()
    return kg_instance


# ============== REQUEST/RESPONSE MODELS ==============

class GraphQueryRequest(BaseModel):
    start_entity: str
    target_entity: Optional[str] = None
    max_hops: int = 3
    relationship_types: Optional[List[str]] = None


class GraphQueryResponse(BaseModel):
    paths: List[List[str]]
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    visualization_html: Optional[str] = None


class ExtractEntitiesRequest(BaseModel):
    code: str
    language: str


class ExtractEntitiesResponse(BaseModel):
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]


class VisualizeGraphRequest(BaseModel):
    entities: List[str]
    include_related: bool = True
    max_depth: int = 2


# ============== 1. MULTI-HOP GRAPH QUERY ==============

@router.post("/query", response_model=GraphQueryResponse)
async def query_graph(request: GraphQueryRequest):
    """
    Multi-hop graph traversal (Level 3 GraphRAG).

    Find paths between entities using BFS/DFS.
    Supports directed and undirected traversal.
    """

    if not GRAPH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="GraphRAG not available. Install networkx: pip install networkx"
        )

    kg = get_kg()
    if not kg:
        raise HTTPException(status_code=500, detail="Failed to initialize knowledge graph")

    try:
        paths = []
        entities = []
        relationships = []

        # Find paths
        if request.target_entity:
            # Find all paths from start to target
            try:
                paths = list(nx.all_simple_paths(
                    kg.graph,
                    source=request.start_entity,
                    target=request.target_entity,
                    cutoff=request.max_hops
                ))
            except nx.NodeNotFound:
                # One or both nodes don't exist
                return GraphQueryResponse(
                    paths=[],
                    entities=[],
                    relationships=[],
                    visualization_html=None
                )
        else:
            # BFS from start entity
            visited = set()
            queue = [(request.start_entity, [request.start_entity])]

            while queue and len(paths) < 10:  # Limit to 10 paths
                node, path = queue.pop(0)

                if len(path) > request.max_hops:
                    continue

                if node in visited:
                    continue

                visited.add(node)
                paths.append(path)

                # Add neighbors
                if node in kg.graph:
                    for neighbor in kg.graph.neighbors(node):
                        if neighbor not in visited:
                            queue.append((neighbor, path + [neighbor]))

        # Extract entities and relationships from paths
        for path in paths:
            for i, entity in enumerate(path):
                if entity not in [e['id'] for e in entities]:
                    entities.append({
                        'id': entity,
                        'type': 'entity',
                        'properties': {}
                    })

                if i < len(path) - 1:
                    # Get edge data
                    edge_data = kg.graph.get_edge_data(entity, path[i + 1])
                    if edge_data:
                        for key, data in edge_data.items():
                            relationships.append({
                                'from': entity,
                                'to': path[i + 1],
                                'type': data.get('type', 'RELATED'),
                                'weight': data.get('weight', 1.0)
                            })

        # Generate visualization
        html = None
        try:
            html = render_knowledge_graph_from_kg(
                kg,
                title=f"Graph Query: {request.start_entity}",
                highlighted_path=paths[0] if paths else None
            )
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")

        return GraphQueryResponse(
            paths=paths,
            entities=entities,
            relationships=relationships,
            visualization_html=html
        )

    except Exception as e:
        logger.error(f"Graph query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Graph query failed: {str(e)}")


# ============== 2. EXTRACT ENTITIES FROM CODE ==============

@router.post("/entities", response_model=ExtractEntitiesResponse)
async def extract_entities(request: ExtractEntitiesRequest):
    """
    Extract code entities (functions, classes, variables).

    Supports: Python, TypeScript, JavaScript
    Uses regex-based extraction (TODO: upgrade to tree-sitter)
    """

    try:
        entities = []
        relationships = []

        if request.language == 'python':
            # Extract classes
            class_pattern = r'class\s+(\w+)'
            for match in re.finditer(class_pattern, request.code):
                entities.append({
                    'id': match.group(1),
                    'type': 'class',
                    'language': 'python',
                    'line': request.code[:match.start()].count('\n') + 1
                })

            # Extract functions
            func_pattern = r'def\s+(\w+)'
            for match in re.finditer(func_pattern, request.code):
                entities.append({
                    'id': match.group(1),
                    'type': 'function',
                    'language': 'python',
                    'line': request.code[:match.start()].count('\n') + 1
                })

            # Extract imports
            import_pattern = r'from\s+(\w+)\s+import|import\s+(\w+)'
            for match in re.finditer(import_pattern, request.code):
                module = match.group(1) or match.group(2)
                entities.append({
                    'id': module,
                    'type': 'module',
                    'language': 'python',
                    'line': request.code[:match.start()].count('\n') + 1
                })

        elif request.language in ['typescript', 'javascript']:
            # Extract classes
            class_pattern = r'class\s+(\w+)'
            for match in re.finditer(class_pattern, request.code):
                entities.append({
                    'id': match.group(1),
                    'type': 'class',
                    'language': request.language,
                    'line': request.code[:match.start()].count('\n') + 1
                })

            # Extract functions
            func_pattern = r'function\s+(\w+)|const\s+(\w+)\s*=\s*\(|export\s+function\s+(\w+)'
            for match in re.finditer(func_pattern, request.code):
                name = match.group(1) or match.group(2) or match.group(3)
                if name:
                    entities.append({
                        'id': name,
                        'type': 'function',
                        'language': request.language,
                        'line': request.code[:match.start()].count('\n') + 1
                    })

            # Extract interfaces
            interface_pattern = r'interface\s+(\w+)'
            for match in re.finditer(interface_pattern, request.code):
                entities.append({
                    'id': match.group(1),
                    'type': 'interface',
                    'language': request.language,
                    'line': request.code[:match.start()].count('\n') + 1
                })

        return ExtractEntitiesResponse(
            entities=entities,
            relationships=relationships
        )

    except Exception as e:
        logger.error(f"Entity extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")


# ============== 3. VISUALIZE GRAPH ==============

@router.post("/visualize")
async def visualize_graph(request: VisualizeGraphRequest):
    """
    Generate interactive graph visualization.

    Creates force-directed graph layout.
    Supports highlighting and path tracing.
    """

    if not GRAPH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Graph visualization not available. Install networkx"
        )

    kg = get_kg()
    if not kg:
        raise HTTPException(status_code=500, detail="Failed to initialize knowledge graph")

    try:
        # Create subgraph containing requested entities
        subgraph_nodes = set(request.entities)

        if request.include_related:
            # Add neighbors up to max_depth
            for entity in request.entities:
                if entity in kg.graph:
                    # BFS to find neighbors
                    visited = {entity}
                    queue = [(entity, 0)]

                    while queue:
                        node, depth = queue.pop(0)

                        if depth >= request.max_depth:
                            continue

                        for neighbor in kg.graph.neighbors(node):
                            if neighbor not in visited:
                                visited.add(neighbor)
                                subgraph_nodes.add(neighbor)
                                queue.append((neighbor, depth + 1))

        # Create subgraph
        subgraph = kg.graph.subgraph(subgraph_nodes)

        # Render visualization
        html = render_knowledge_graph_from_networkx(
            subgraph,
            title="Code Knowledge Graph"
        )

        return {"html": html, "node_count": len(subgraph_nodes)}

    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


# ============== 4. FIND RELATIONSHIPS ==============

@router.get("/relationships/{entity}")
async def get_entity_relationships(entity: str, max_depth: int = 1):
    """
    Get all relationships for an entity.

    Returns:
        - Direct relationships (depth=1)
        - Extended relationships (depth>1)
    """

    if not GRAPH_AVAILABLE:
        raise HTTPException(status_code=503, detail="GraphRAG not available")

    kg = get_kg()
    if not kg or entity not in kg.graph:
        return {
            "entity": entity,
            "relationships": [],
            "message": "Entity not found in graph"
        }

    try:
        relationships = []

        # Get all edges for this entity
        for neighbor in kg.graph.neighbors(entity):
            edge_data = kg.graph.get_edge_data(entity, neighbor)
            if edge_data:
                for key, data in edge_data.items():
                    relationships.append({
                        'from': entity,
                        'to': neighbor,
                        'type': data.get('type', 'RELATED'),
                        'weight': data.get('weight', 1.0)
                    })

        return {
            "entity": entity,
            "relationships": relationships,
            "total": len(relationships)
        }

    except Exception as e:
        logger.error(f"Relationship query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ============== 5. ADD ENTITIES/RELATIONSHIPS ==============

class AddEntityRequest(BaseModel):
    entity_id: str
    entity_type: str = "generic"
    properties: Optional[Dict[str, Any]] = None


class AddRelationshipRequest(BaseModel):
    from_entity: str
    to_entity: str
    relationship_type: str = "RELATED"
    weight: float = 1.0


@router.post("/entities/add")
async def add_entity(request: AddEntityRequest):
    """Add entity to knowledge graph."""

    if not GRAPH_AVAILABLE:
        raise HTTPException(status_code=503, detail="GraphRAG not available")

    kg = get_kg()
    if not kg:
        raise HTTPException(status_code=500, detail="Failed to initialize knowledge graph")

    try:
        # Add node to graph
        kg.graph.add_node(request.entity_id, **{
            'type': request.entity_type,
            **(request.properties or {})
        })

        return {
            "success": True,
            "entity_id": request.entity_id,
            "message": "Entity added successfully"
        }

    except Exception as e:
        logger.error(f"Failed to add entity: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add entity: {str(e)}")


@router.post("/relationships/add")
async def add_relationship(request: AddRelationshipRequest):
    """Add relationship between entities."""

    if not GRAPH_AVAILABLE:
        raise HTTPException(status_code=503, detail="GraphRAG not available")

    kg = get_kg()
    if not kg:
        raise HTTPException(status_code=500, detail="Failed to initialize knowledge graph")

    try:
        # Add edge to graph
        kg.add_edges([
            KGEdge(
                source=request.from_entity,
                target=request.to_entity,
                edge_type=request.relationship_type,
                weight=request.weight
            )
        ])

        return {
            "success": True,
            "from": request.from_entity,
            "to": request.to_entity,
            "type": request.relationship_type,
            "message": "Relationship added successfully"
        }

    except Exception as e:
        logger.error(f"Failed to add relationship: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add relationship: {str(e)}")
