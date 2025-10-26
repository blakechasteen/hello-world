# Visual Drag-and-Drop Orchestrator for HoloLoom
## Architecture Design for Interactive Memory & Analysis Composition

**Date**: October 22, 2025  
**Purpose**: Design a visual, drag-and-drop interface for composing complex analysis pipelines with flexible memory backends

---

## Executive Summary

### What We're Building

A **visual orchestration system** that allows users to:

1. **Drag and drop** analysis components (motif detectors, embedders, memory stores, policies)
2. **Connect** them visually to create custom pipelines
3. **Query multiple memory backends** simultaneously (Neo4j + Qdrant + SQLite + in-memory)
4. **See results** from each backend with smart fusion
5. **Save/load** pipeline configurations
6. **Monitor** execution in real-time

### Why This Matters

**Current State**: Code-based configuration. To change retrieval strategy, you edit Python files.

**Future State**: Visual canvas. Drag a "Neo4j Store" node, connect it to a "Temporal Navigator", wire to a "Pattern Detector", and run.

**Value Proposition**:
- **No-code pipeline creation** for non-technical users
- **Rapid experimentation** with different memory backends
- **Visual debugging** - see exactly what each component does
- **Reusable templates** - save successful pipelines

---

## Architecture Overview

### Three-Layer System

```
┌─────────────────────────────────────────────────────────┐
│                  PRESENTATION LAYER                      │
│         (React/Vue Frontend + WebSocket)                 │
│                                                          │
│  ┌────────────┐  ┌──────────┐  ┌──────────────┐       │
│  │  Canvas    │  │  Library │  │  Inspector   │       │
│  │  (Nodes +  │  │  (Drag   │  │  (Results +  │       │
│  │   Edges)   │  │   Drop)  │  │   Metrics)   │       │
│  └────────────┘  └──────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              ORCHESTRATION ENGINE                        │
│            (Python FastAPI + AsyncIO)                    │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Pipeline Builder                                │   │
│  │  - Node Registry (all available components)     │   │
│  │  - Dependency Resolution                         │   │
│  │  - Execution Graph Builder                       │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Execution Coordinator                           │   │
│  │  - Parallel execution across backends           │   │
│  │  - Result fusion (weighted combination)         │   │
│  │  - Streaming updates (WebSocket)                │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              COMPONENT LAYER                             │
│       (Protocol-Based HoloLoom Components)               │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Memory   │  │Navigator │  │ Pattern  │             │
│  │ Stores   │  │          │  │ Detector │             │
│  └──────────┘  └──────────┘  └──────────┘             │
│                                                          │
│  Neo4j | Qdrant | SQLite | InMemory | Hofstadter       │
└─────────────────────────────────────────────────────────┘
```

---

## Component Design

### Node Types (Visual Building Blocks)

#### 1. **Source Nodes** (Data Input)
- **Query Input**: Text query from user
- **File Upload**: Drag-and-drop files (PDF, JSON, CSV, audio)
- **Memory Snapshot**: Load existing memory state

#### 2. **Memory Store Nodes** (Storage Backends)
- **Neo4j Store**: Thread-based graph storage
- **Qdrant Store**: Multi-scale vector search
- **SQLite Store**: Embedded SQL database
- **InMemory Store**: Fast ephemeral cache
- **Mem0 Store**: Intelligent extraction backend

#### 3. **Feature Extraction Nodes** (Analysis)
- **Motif Detector**: Regex/hybrid/LLM pattern detection
- **Embedder**: Matryoshka multi-scale embeddings
- **Spectral Analyzer**: Graph Laplacian features
- **Temporal Analyzer**: Time-based features

#### 4. **Navigator Nodes** (Memory Traversal)
- **Hofstadter Navigator**: Self-referential sequences
- **Graph Navigator**: Neo4j Cypher traversal
- **Temporal Navigator**: Time-based navigation
- **Semantic Navigator**: Similarity-based navigation

#### 5. **Pattern Detector Nodes** (Discovery)
- **Strange Loop Detector**: Cycle detection
- **Cluster Detector**: Spectral clustering
- **Resonance Detector**: Hofstadter resonances
- **Thread Detector**: Narrative threads

#### 6. **Fusion Nodes** (Combining Results)
- **Weighted Fusion**: Configurable weights per backend
- **Rank Fusion**: Reciprocal rank fusion
- **Threshold Fusion**: Only high-confidence results
- **Ensemble Fusion**: Multiple strategies combined

#### 7. **Policy Nodes** (Decision Making)
- **Neural Policy**: Transformer-based decisions
- **Thompson Sampling**: Bayesian exploration
- **Epsilon-Greedy**: Fixed exploration rate
- **Argmax Policy**: Pure exploitation

#### 8. **Output Nodes** (Results)
- **Memory Results**: Retrieved memories
- **Pattern Results**: Discovered patterns
- **Visualization**: Graph/timeline/heatmap
- **Export**: JSON/CSV/Markdown

---

## Visual Canvas Interface

### Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  HoloLoom Visual Orchestrator                    [Save] [Run] [?] │
├─────────────┬────────────────────────────────────────────────────┤
│             │                                                     │
│  LIBRARY    │              CANVAS                                │
│             │                                                     │
│ 📥 Sources  │    ┌─────────┐                                    │
│  • Query    │    │ Query   │                                    │
│  • File     │    │ Input   │                                    │
│             │    └────┬────┘                                    │
│ 💾 Stores   │         │                                         │
│  • Neo4j    │    ┌────▼────┐    ┌──────────┐                  │
│  • Qdrant   │    │ Motif   │───▶│ Embedder │                  │
│  • SQLite   │    │Detector │    └────┬─────┘                  │
│  • InMemory │    └────┬────┘         │                         │
│             │         │              │                         │
│ 🔍 Analysis │    ┌────▼──────────────▼────┐                   │
│  • Motif    │    │                         │                   │
│  • Embedder │    │   Multi-Store Query     │                   │
│  • Spectral │    │   (Neo4j + Qdrant +     │                   │
│             │    │    SQLite + InMemory)    │                   │
│ 🧭 Navigate │    └────┬────────────────────┘                   │
│  • Hofstad  │         │                                         │
│  • Graph    │    ┌────▼─────┐      ┌──────────┐               │
│  • Temporal │    │ Weighted │─────▶│ Results  │               │
│             │    │  Fusion  │      │ Inspector│               │
│ 🎨 Patterns │    └──────────┘      └──────────┘               │
│  • Loops    │                                                   │
│  • Clusters │                                                   │
│             │                                                   │
│ 🎯 Policy   │                                                   │
│  • Neural   │                                                   │
│  • Thompson │                                                   │
│             │                                                   │
│ 📤 Output   │                                                   │
│  • Results  │                                                   │
│  • Viz      │                                                   │
│  • Export   │                                                   │
└─────────────┴────────────────────────────────────────────────────┘
```

### Node Visual Design

Each node is a draggable card:

```
┌─────────────────────────┐
│ 🗄️ Neo4j Store          │  ← Icon + Title
├─────────────────────────┤
│ Status: ✅ Connected     │  ← Status indicator
│ Threads: 1,247          │  ← Key metrics
│ Latest: 2m ago          │  ← Timestamp
├─────────────────────────┤
│ ●  Query Input          │  ← Input ports (left)
│                         │
│          Results  ●     │  ← Output ports (right)
└─────────────────────────┘
```

**Color Coding**:
- 📥 **Blue**: Input/Source nodes
- 💾 **Green**: Storage backends
- 🔍 **Purple**: Analysis/Feature extraction
- 🧭 **Orange**: Navigation
- 🎨 **Pink**: Pattern detection
- 🎯 **Red**: Policy/Decision
- 📤 **Gray**: Output/Export

---

## Backend Architecture

### Orchestration Engine (`orchestrator/engine.py`)

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import asyncio
from enum import Enum

class NodeType(Enum):
    SOURCE = "source"
    STORE = "store"
    FEATURE = "feature"
    NAVIGATOR = "navigator"
    PATTERN = "pattern"
    FUSION = "fusion"
    POLICY = "policy"
    OUTPUT = "output"

@dataclass
class NodeDefinition:
    """Visual node definition"""
    id: str
    type: NodeType
    component: str  # e.g., "Neo4jStore", "HofstadterNavigator"
    config: Dict[str, Any]
    inputs: List[str]  # Input port IDs
    outputs: List[str]  # Output port IDs

@dataclass
class EdgeDefinition:
    """Connection between nodes"""
    id: str
    source: str  # Source node ID
    source_port: str
    target: str  # Target node ID
    target_port: str

@dataclass
class PipelineDefinition:
    """Complete pipeline specification"""
    id: str
    name: str
    nodes: List[NodeDefinition]
    edges: List[EdgeDefinition]
    metadata: Dict[str, Any]

class VisualOrchestrator:
    """
    Main orchestration engine for visual pipelines.
    
    Responsibilities:
    1. Build execution graph from visual pipeline
    2. Resolve dependencies between nodes
    3. Execute nodes in parallel where possible
    4. Stream results via WebSocket
    5. Handle errors gracefully
    """
    
    def __init__(self):
        self.node_registry = self._build_registry()
        self.active_pipelines: Dict[str, PipelineDefinition] = {}
        
    def _build_registry(self) -> Dict[str, type]:
        """Register all available component types"""
        from HoloLoom.memory.stores import (
            Neo4jStore, QdrantStore, SQLiteStore, InMemoryStore
        )
        from HoloLoom.memory.navigators import HofstadterNavigator
        from HoloLoom.memory.detectors import MultiPatternDetector
        
        return {
            # Stores
            "Neo4jStore": Neo4jStore,
            "QdrantStore": QdrantStore,
            "SQLiteStore": SQLiteStore,
            "InMemoryStore": InMemoryStore,
            
            # Navigators
            "HofstadterNavigator": HofstadterNavigator,
            
            # Detectors
            "PatternDetector": MultiPatternDetector,
            
            # More components...
        }
    
    async def execute_pipeline(
        self, 
        pipeline: PipelineDefinition,
        inputs: Dict[str, Any],
        websocket = None
    ) -> Dict[str, Any]:
        """
        Execute a visual pipeline.
        
        Args:
            pipeline: Pipeline definition from frontend
            inputs: Initial inputs (e.g., query text)
            websocket: Optional WebSocket for streaming updates
            
        Returns:
            Final results from output nodes
        """
        # 1. Build execution graph
        graph = self._build_execution_graph(pipeline)
        
        # 2. Topological sort for execution order
        execution_order = self._topological_sort(graph)
        
        # 3. Initialize node instances
        node_instances = {}
        for node in pipeline.nodes:
            component_class = self.node_registry[node.component]
            node_instances[node.id] = component_class(**node.config)
        
        # 4. Execute nodes in order
        node_outputs = {}
        
        for node_id in execution_order:
            node = next(n for n in pipeline.nodes if n.id == node_id)
            instance = node_instances[node_id]
            
            # Gather inputs from upstream nodes
            node_inputs = self._gather_inputs(node, node_outputs, inputs)
            
            # Execute node
            if websocket:
                await websocket.send_json({
                    "type": "node_start",
                    "node_id": node_id,
                    "component": node.component
                })
            
            try:
                result = await self._execute_node(instance, node_inputs)
                node_outputs[node_id] = result
                
                if websocket:
                    await websocket.send_json({
                        "type": "node_complete",
                        "node_id": node_id,
                        "result": result
                    })
                    
            except Exception as e:
                if websocket:
                    await websocket.send_json({
                        "type": "node_error",
                        "node_id": node_id,
                        "error": str(e)
                    })
                raise
        
        # 5. Return outputs from output nodes
        output_nodes = [n for n in pipeline.nodes if n.type == NodeType.OUTPUT]
        return {n.id: node_outputs[n.id] for n in output_nodes}
    
    def _build_execution_graph(self, pipeline: PipelineDefinition) -> Dict:
        """Build directed acyclic graph for execution"""
        graph = {node.id: [] for node in pipeline.nodes}
        
        for edge in pipeline.edges:
            graph[edge.source].append(edge.target)
        
        return graph
    
    def _topological_sort(self, graph: Dict) -> List[str]:
        """Topological sort for execution order"""
        # Kahn's algorithm
        in_degree = {node: 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] += 1
        
        queue = [node for node in graph if in_degree[node] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(graph):
            raise ValueError("Pipeline contains cycles!")
        
        return result
    
    def _gather_inputs(
        self, 
        node: NodeDefinition, 
        node_outputs: Dict, 
        initial_inputs: Dict
    ) -> Dict[str, Any]:
        """Gather inputs for a node from upstream outputs"""
        inputs = {}
        
        # Get edges that target this node
        for input_port in node.inputs:
            # Find upstream edge
            # ... implementation
            pass
        
        return inputs
    
    async def _execute_node(self, instance, inputs: Dict) -> Any:
        """Execute a single node with its inputs"""
        # Dispatch based on node type
        if hasattr(instance, 'retrieve'):
            # It's a store
            return await instance.retrieve(**inputs)
        elif hasattr(instance, 'navigate'):
            # It's a navigator
            return await instance.navigate(**inputs)
        elif hasattr(instance, 'detect_patterns'):
            # It's a pattern detector
            return await instance.detect_patterns(**inputs)
        else:
            raise ValueError(f"Unknown node type: {type(instance)}")
```

---

## Multi-Backend Query System

### Parallel Execution Across All Stores

```python
class MultiStoreQueryNode:
    """
    Special node that queries ALL memory backends in parallel.
    
    This is a killer feature - query Neo4j, Qdrant, SQLite, 
    and in-memory simultaneously, then fuse results.
    """
    
    def __init__(self, stores: List[MemoryStore]):
        self.stores = stores
        self.store_names = ["neo4j", "qdrant", "sqlite", "memory"]
    
    async def query_all(
        self, 
        query: str, 
        strategy: RecallStrategy,
        limit: int = 10
    ) -> Dict[str, List[Memory]]:
        """
        Query all stores in parallel.
        
        Returns:
            Dict mapping store name → results
        """
        # Query all stores concurrently
        tasks = [
            self._query_store(store, name, query, strategy, limit)
            for store, name in zip(self.stores, self.store_names)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Organize by store
        store_results = {}
        for name, result in zip(self.store_names, results):
            if isinstance(result, Exception):
                store_results[name] = {
                    "error": str(result),
                    "memories": []
                }
            else:
                store_results[name] = {
                    "memories": result,
                    "count": len(result)
                }
        
        return store_results
    
    async def _query_store(
        self, 
        store: MemoryStore, 
        name: str,
        query: str, 
        strategy: RecallStrategy,
        limit: int
    ) -> List[Memory]:
        """Query a single store"""
        try:
            memory_query = MemoryQuery(
                text=query,
                strategy=strategy,
                limit=limit
            )
            
            result = await store.retrieve(memory_query, strategy)
            return result.memories
            
        except Exception as e:
            logger.error(f"Error querying {name}: {e}")
            raise
```

---

## Fusion Strategies

### Weighted Fusion Node

```python
class WeightedFusionNode:
    """
    Fuse results from multiple stores with configurable weights.
    
    Example:
        Neo4j: 40%    (strong for graph relationships)
        Qdrant: 30%   (strong for semantic similarity)
        SQLite: 20%   (strong for structured queries)
        InMemory: 10% (recent context)
    """
    
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Ensure weights sum to 1.0"""
        total = sum(self.weights.values())
        for store in self.weights:
            self.weights[store] /= total
    
    async def fuse(
        self, 
        store_results: Dict[str, List[Memory]]
    ) -> List[Memory]:
        """
        Fuse results from multiple stores.
        
        Algorithm:
        1. Pool all unique memories
        2. Score each memory based on which stores returned it
        3. Weight scores by store weights
        4. Sort by final score
        """
        # Pool all memories with their source stores
        memory_sources = {}  # memory_id → set of stores that returned it
        memory_objects = {}  # memory_id → Memory object
        
        for store_name, result in store_results.items():
            if "error" in result:
                continue
                
            for memory in result["memories"]:
                if memory.id not in memory_sources:
                    memory_sources[memory.id] = set()
                    memory_objects[memory.id] = memory
                
                memory_sources[memory.id].add(store_name)
        
        # Score each memory
        memory_scores = {}
        for mem_id, stores in memory_sources.items():
            score = sum(
                self.weights.get(store, 0.0) 
                for store in stores
            )
            memory_scores[mem_id] = score
        
        # Sort by score
        ranked = sorted(
            memory_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return memories in ranked order
        return [memory_objects[mem_id] for mem_id, _ in ranked]
```

---

## Frontend Architecture

### React Component Structure

```
src/
├── components/
│   ├── Canvas/
│   │   ├── Canvas.tsx              # Main canvas component
│   │   ├── Node.tsx                # Visual node component
│   │   ├── Edge.tsx                # Connection line
│   │   └── MiniMap.tsx             # Overview navigator
│   │
│   ├── Library/
│   │   ├── Library.tsx             # Sidebar with draggable components
│   │   ├── NodeCategory.tsx        # Collapsible category
│   │   └── NodePreview.tsx         # Component preview card
│   │
│   ├── Inspector/
│   │   ├── Inspector.tsx           # Right panel for results
│   │   ├── ResultsView.tsx         # Display query results
│   │   ├── MetricsView.tsx         # Show performance metrics
│   │   └── ErrorView.tsx           # Display errors
│   │
│   └── Toolbar/
│       ├── Toolbar.tsx             # Top toolbar
│       ├── SaveButton.tsx          # Save pipeline
│       └── RunButton.tsx           # Execute pipeline
│
├── hooks/
│   ├── usePipeline.ts              # Pipeline state management
│   ├── useWebSocket.ts             # Real-time updates
│   └── useNodeRegistry.ts          # Available components
│
├── api/
│   ├── orchestrator.ts             # API client for backend
│   └── websocket.ts                # WebSocket connection
│
└── types/
    ├── nodes.ts                    # Node type definitions
    ├── edges.ts                    # Edge type definitions
    └── pipeline.ts                 # Pipeline type definitions
```

### Example: Canvas Component

```tsx
import React, { useState, useCallback } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { NodeLibrary } from './Library/NodeLibrary';
import { Inspector } from './Inspector/Inspector';
import { Toolbar } from './Toolbar/Toolbar';
import { CustomNode } from './Canvas/CustomNode';

const nodeTypes = {
  custom: CustomNode,
};

export const VisualOrchestrator: React.FC = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  
  const onConnect = useCallback(
    (connection: Connection) => setEdges((eds) => addEdge(connection, eds)),
    [setEdges]
  );
  
  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      
      const nodeData = JSON.parse(
        event.dataTransfer.getData('application/reactflow')
      );
      
      const position = {
        x: event.clientX,
        y: event.clientY,
      };
      
      const newNode = {
        id: `${nodeData.type}-${Date.now()}`,
        type: 'custom',
        position,
        data: nodeData,
      };
      
      setNodes((nds) => nds.concat(newNode));
    },
    [setNodes]
  );
  
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);
  
  const handleRun = async () => {
    // Build pipeline from nodes and edges
    const pipeline = {
      id: `pipeline-${Date.now()}`,
      name: 'My Pipeline',
      nodes: nodes.map(n => ({
        id: n.id,
        type: n.data.nodeType,
        component: n.data.component,
        config: n.data.config || {},
        inputs: [], // Derived from edges
        outputs: [],
      })),
      edges: edges.map(e => ({
        id: e.id,
        source: e.source,
        source_port: e.sourceHandle,
        target: e.target,
        target_port: e.targetHandle,
      })),
      metadata: {},
    };
    
    // Execute via API
    const response = await fetch('/api/pipeline/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        pipeline,
        inputs: { query: 'winter beekeeping' },
      }),
    });
    
    const results = await response.json();
    console.log('Pipeline results:', results);
  };
  
  return (
    <div className="visual-orchestrator">
      <Toolbar onRun={handleRun} />
      
      <div className="orchestrator-layout">
        <NodeLibrary />
        
        <div className="canvas-container" onDrop={onDrop} onDragOver={onDragOver}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            onNodeClick={(_, node) => setSelectedNode(node)}
          >
            <Background />
            <Controls />
            <MiniMap />
          </ReactFlow>
        </div>
        
        <Inspector selectedNode={selectedNode} />
      </div>
    </div>
  );
};
```

---

## Real-Time Execution Monitoring

### WebSocket Protocol

```python
# Backend: WebSocket handler
from fastapi import WebSocket
import json

class PipelineWebSocket:
    """Real-time pipeline execution monitoring"""
    
    async def handle_execution(
        self, 
        websocket: WebSocket,
        pipeline: PipelineDefinition,
        inputs: Dict[str, Any]
    ):
        """Stream execution updates to frontend"""
        await websocket.accept()
        
        try:
            # Send initial status
            await websocket.send_json({
                "type": "execution_start",
                "pipeline_id": pipeline.id,
                "total_nodes": len(pipeline.nodes)
            })
            
            # Execute with streaming updates
            orchestrator = VisualOrchestrator()
            results = await orchestrator.execute_pipeline(
                pipeline, 
                inputs, 
                websocket=websocket
            )
            
            # Send final results
            await websocket.send_json({
                "type": "execution_complete",
                "results": results
            })
            
        except Exception as e:
            await websocket.send_json({
                "type": "execution_error",
                "error": str(e)
            })
        
        finally:
            await websocket.close()
```

**Message Types**:

1. **execution_start**: Pipeline begins
2. **node_start**: Node begins execution
3. **node_progress**: Node reports progress (optional)
4. **node_complete**: Node finishes successfully
5. **node_error**: Node encounters error
6. **execution_complete**: Pipeline finishes
7. **execution_error**: Pipeline-level error

---

## Example Pipelines (Templates)

### Template 1: Multi-Backend Retrieval

```json
{
  "name": "Multi-Backend Query",
  "description": "Query all memory backends and fuse results",
  "nodes": [
    {
      "id": "query-input",
      "type": "source",
      "component": "QueryInput",
      "config": {}
    },
    {
      "id": "multi-store",
      "type": "store",
      "component": "MultiStoreQuery",
      "config": {
        "stores": ["neo4j", "qdrant", "sqlite", "memory"]
      }
    },
    {
      "id": "fusion",
      "type": "fusion",
      "component": "WeightedFusion",
      "config": {
        "weights": {
          "neo4j": 0.4,
          "qdrant": 0.3,
          "sqlite": 0.2,
          "memory": 0.1
        }
      }
    },
    {
      "id": "results",
      "type": "output",
      "component": "ResultsView",
      "config": {}
    }
  ],
  "edges": [
    {
      "source": "query-input",
      "target": "multi-store"
    },
    {
      "source": "multi-store",
      "target": "fusion"
    },
    {
      "source": "fusion",
      "target": "results"
    }
  ]
}
```

### Template 2: Pattern Discovery Pipeline

```json
{
  "name": "Pattern Discovery",
  "description": "Find strange loops and resonances",
  "nodes": [
    {
      "id": "neo4j",
      "type": "store",
      "component": "Neo4jStore"
    },
    {
      "id": "hofstadter",
      "type": "navigator",
      "component": "HofstadterNavigator"
    },
    {
      "id": "loop-detector",
      "type": "pattern",
      "component": "StrangeLoopDetector"
    },
    {
      "id": "resonance-detector",
      "type": "pattern",
      "component": "ResonanceDetector"
    },
    {
      "id": "visualization",
      "type": "output",
      "component": "PatternVisualization"
    }
  ],
  "edges": [
    {
      "source": "neo4j",
      "target": "hofstadter"
    },
    {
      "source": "hofstadter",
      "target": "loop-detector"
    },
    {
      "source": "hofstadter",
      "target": "resonance-detector"
    },
    {
      "source": "loop-detector",
      "target": "visualization"
    },
    {
      "source": "resonance-detector",
      "target": "visualization"
    }
  ]
}
```

---

## Implementation Roadmap

### Phase 1: Core Backend (Week 1-2)

**Goal**: Get orchestration engine working

- [ ] Implement `VisualOrchestrator` class
- [ ] Build node registry
- [ ] Implement execution graph builder
- [ ] Topological sort for execution order
- [ ] WebSocket streaming
- [ ] Multi-store query node
- [ ] Weighted fusion node

**Deliverable**: Backend API that can execute pipelines defined in JSON

---

### Phase 2: Basic Frontend (Week 3-4)

**Goal**: Get visual canvas working

- [ ] Set up React + ReactFlow
- [ ] Implement canvas with drag-and-drop
- [ ] Create node components (visual cards)
- [ ] Implement node library sidebar
- [ ] Basic toolbar (save/load/run)
- [ ] WebSocket connection to backend
- [ ] Results inspector panel

**Deliverable**: Working visual interface for building pipelines

---

### Phase 3: Memory Backend Implementations (Week 5-6)

**Goal**: Implement all protocol backends

- [ ] Neo4j store implementation
- [ ] Qdrant store implementation
- [ ] SQLite store implementation
- [ ] InMemory store implementation
- [ ] Hofstadter navigator
- [ ] Pattern detectors

**Deliverable**: All memory backends functional in visual orchestrator

---

### Phase 4: Advanced Features (Week 7-8)

**Goal**: Polish and enhancement

- [ ] Pipeline templates (save/load)
- [ ] Real-time execution visualization
- [ ] Error handling and retry logic
- [ ] Configuration panels for nodes
- [ ] Export results (JSON/CSV/Markdown)
- [ ] Comparison view (side-by-side results)

**Deliverable**: Production-ready visual orchestrator

---

## Technical Stack

### Backend
- **FastAPI**: REST API + WebSocket
- **AsyncIO**: Parallel execution
- **Pydantic**: Data validation
- **Existing HoloLoom**: All protocol implementations

### Frontend
- **React**: UI framework
- **ReactFlow**: Visual node editor
- **TailwindCSS**: Styling
- **Zustand**: State management
- **SWR**: Data fetching

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Multi-service orchestration
- **Traefik**: Reverse proxy (if deploying)

---

## Key Benefits

### For Users

1. **No-Code Pipeline Creation**: Build complex analyses without writing code
2. **Visual Debugging**: See exactly where queries go and what they return
3. **Rapid Experimentation**: Try different memory backends instantly
4. **Reusable Templates**: Save successful pipelines and share them
5. **Real-Time Monitoring**: Watch execution in real-time

### For Developers

1. **Protocol-Based**: Easy to add new components (just implement protocol)
2. **Parallel Execution**: Automatically parallelizes independent nodes
3. **WebSocket Streaming**: Real-time updates without polling
4. **Clean Separation**: Frontend/backend completely decoupled
5. **Testable**: Each component can be tested independently

---

## Next Steps

### Option A: Backend First (Recommended)

**Week 1**: Implement core orchestration engine
**Week 2**: Add multi-store query and fusion
**Week 3**: Build basic REST API
**Week 4**: Add WebSocket streaming

**Deliverable**: Working backend that can execute pipelines via API

### Option B: Prototype Frontend

**Week 1**: Set up React + ReactFlow
**Week 2**: Build visual canvas with mock nodes
**Week 3**: Implement drag-and-drop
**Week 4**: Connect to mock backend

**Deliverable**: Visual prototype to validate UX

### Option C: Parallel Development

**Backend Team**: Orchestration engine + API
**Frontend Team**: Visual canvas + components

**Timeline**: 6-8 weeks to production

---

## Success Metrics

### Technical
- ✅ Execute pipelines with 10+ nodes
- ✅ Query 4+ memory backends in parallel
- ✅ Sub-second latency for simple queries
- ✅ Real-time WebSocket updates (<100ms)

### User Experience
- ✅ Create pipeline in <5 minutes
- ✅ Understand results immediately
- ✅ Save/load pipelines easily
- ✅ Share pipelines with team

---

## Conclusion

This visual orchestrator transforms HoloLoom from a **code-first** system to a **user-first** system. 

**Key Innovation**: Query multiple memory backends simultaneously and visually see how each contributes to the final result.

**Implementation Path**: Start with backend orchestration engine, add WebSocket streaming, build React frontend, wire them together.

**Timeline**: 6-8 weeks to production-ready system.

**Next Action**: Choose Option A, B, or C and start building!

