"""
Graph Visualization Router
===========================
Endpoints for knowledge graph visualization (HTML + JSON).

Extracted from agentic_api.py (March 2026 Refactor).
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from hololoom.apps.server.server_state import state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/html")
async def get_graph_html(
    max_nodes: int = 50,
    title: str = "HoloLoom Knowledge Graph",
    highlight_recent: bool = True
):
    """Get knowledge graph as interactive D3.js HTML visualization."""
    from hololoom import HoloLoom
    from hololoom.visualization.knowledge_graph import render_knowledge_graph_from_kg

    try:
        async with HoloLoom(config=state.config) as loom:
            kg = loom.memory_manager.knowledge_graph

            highlighted_path = None
            if highlight_recent and hasattr(loom, 'awareness_graph'):
                recent_nodes = loom.awareness_graph.get_top_activated(limit=5)
                if recent_nodes:
                    highlighted_path = [node for node, _ in recent_nodes]

            subtitle = f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            html = render_knowledge_graph_from_kg(
                kg,
                title=title,
                subtitle=subtitle,
                max_nodes=max_nodes,
                highlighted_path=highlighted_path
            )
            return HTMLResponse(content=html)

    except Exception as e:
        logger.error(f"Graph visualization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data")
async def get_graph_data(
    max_nodes: int = 50,
    include_metadata: bool = False
):
    """Get knowledge graph as JSON (nodes, edges, metadata)."""
    from hololoom import HoloLoom

    try:
        async with HoloLoom(config=state.config) as loom:
            kg = loom.memory_manager.knowledge_graph

            all_nodes = list(kg.G.nodes())[:max_nodes]
            node_id_set = set(all_nodes)

            nodes = []
            for node_id in all_nodes:
                node_data = kg.G.nodes.get(node_id, {})
                degree = kg.G.degree(node_id)
                node_obj = {
                    "id": node_id,
                    "label": node_id,
                    "degree": degree,
                    "type": node_data.get('node_type', 'default')
                }
                if include_metadata:
                    node_obj["metadata"] = node_data
                nodes.append(node_obj)

            edges = []
            for src, dst, key, data in kg.G.edges(keys=True, data=True):
                if src in node_id_set and dst in node_id_set:
                    edge_obj = {
                        "src": src,
                        "dst": dst,
                        "type": data.get('type', 'unknown'),
                        "weight": data.get('weight', 1.0)
                    }
                    if include_metadata:
                        edge_obj["metadata"] = data
                    edges.append(edge_obj)

            return {
                "nodes": nodes,
                "edges": edges,
                "metadata": {
                    "total_nodes": kg.G.number_of_nodes(),
                    "total_edges": kg.G.number_of_edges(),
                    "rendered_nodes": len(nodes),
                    "rendered_edges": len(edges)
                }
            }

    except Exception as e:
        logger.error(f"Graph data export failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
