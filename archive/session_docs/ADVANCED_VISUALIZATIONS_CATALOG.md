# Advanced Visualizations Catalog: Meaning First

**Date**: October 29, 2025
**Philosophy**: Every visualization must reveal actionable insight

---

## Already Built ✅

1. **Tufte Sparklines** - Trends inline with metrics
2. **Small Multiples** - Query comparison with consistent scales
3. **Data Density Tables** - Maximum info per inch
4. **Semantic Heatmaps** - 244D dimension activations
5. **Bottleneck Detection** - Performance warnings

---

## Category 1: Network & Graph Visualizations

### 1.1 Knowledge Graph Network Map 🌐

**What it reveals**: How entities relate and cluster in memory

**Visual**:
```
     Weaving ──uses──> Thompson Sampling
        │                   │
     contains            related to
        │                   │
    Orchestrator ─────> Policy Engine
        │                   │
     manages            optimizes
        │                   │
    Memory <──────────> Decision
```

**Implementation**:
- **Library**: D3.js force-directed graph or Plotly network
- **Data source**: YarnGraph (Neo4j or NetworkX)
- **Features**:
  - Node size = frequency of access
  - Edge thickness = relationship strength
  - Color = entity type
  - Clusters = semantic communities
  - Interactive: Click node → show details

**Use cases**:
- Debug entity extraction
- Find knowledge gaps (isolated nodes)
- Discover unexpected connections

**File**: `HoloLoom/visualization/knowledge_graph_map.py`
**Complexity**: Medium (3-4 hours)
**Impact**: HIGH - reveals memory structure

---

### 1.2 Query Similarity Network 🕸️

**What it reveals**: Which queries are semantically similar

**Visual**:
```
    "What is X?" ──0.92──> "Explain X"
          │                    │
        0.85                 0.78
          │                    │
    "How does X work?" ──0.88──> "X architecture"
```

**Implementation**:
- **Algorithm**: Cosine similarity on embeddings
- **Layout**: Force-directed (similar queries cluster)
- **Threshold**: Only show edges > 0.7 similarity

**Features**:
- Node color = query success (green) vs failure (red)
- Node size = latency (smaller = faster)
- Hover = full query text + metrics
- Clusters = topic areas

**Use cases**:
- Find redundant queries
- Identify query patterns
- Guide cache strategy

**File**: `HoloLoom/visualization/query_network.py`
**Complexity**: Medium (2-3 hours)
**Impact**: MEDIUM - pattern discovery

---

## Category 2: Spatial & Projection Visualizations

### 2.1 Semantic Space Projection (244D → 2D) 🎨

**What it reveals**: How queries and memories distribute in semantic space

**Visual** (2D t-SNE or UMAP):
```
     Factual Queries (cluster)
          ●●●●
         ●    ●

                    Exploratory Queries
                         ●●
                        ●  ●
                         ●●

    Optimization Queries
      ●●●
```

**Implementation**:
- **Projection**: t-SNE or UMAP (244D → 2D)
- **Points**: Each query = point
- **Colors**: By query intent or success
- **Size**: By confidence

**Features**:
- Interactive zoom/pan
- Selection → show query details
- Convex hulls around clusters
- New query → animate position

**Use cases**:
- Visualize semantic coverage
- Find query outliers
- Identify gaps in training

**File**: `HoloLoom/visualization/semantic_projection.py`
**Complexity**: Medium (3-4 hours)
**Impact**: HIGH - understand semantic space

---

### 2.2 Embedding Space Clusters 🔵

**What it reveals**: Natural groupings in memory

**Visual** (3D scatter with Plotly):
```
3D space with clusters:
- Cluster 1 (blue): Architecture queries
- Cluster 2 (green): Performance queries
- Cluster 3 (red): Debugging queries
```

**Implementation**:
- **Algorithm**: K-means or DBSCAN clustering
- **Visualization**: Plotly 3D scatter
- **Axes**: First 3 PCA components

**Features**:
- Rotate 3D view
- Color = cluster ID
- Size = cluster confidence
- Labels = cluster centroid summary

**Use cases**:
- Discover query categories
- Optimize retrieval strategy
- Balance training data

**File**: `HoloLoom/visualization/embedding_clusters.py`
**Complexity**: Medium (2-3 hours)
**Impact**: MEDIUM - memory organization

---

## Category 3: Temporal & Evolution Visualizations

### 3.1 Semantic Dimension Evolution 📈

**What it reveals**: How semantic dimensions change over time

**Visual** (Horizon chart):
```
Dimension: "Technical"
Hour 1  ▁▂▃▄▅▆▅▄▃▂▁
Hour 2  ▂▃▄▅▄▃▂▁▁▂▃
Hour 3  ▃▄▅▄▃▂▁▁▂▃▄
```

**Implementation**:
- **Chart type**: Horizon chart (Tufte-inspired)
- **X-axis**: Time
- **Y-axis**: Dimension activation strength
- **Layers**: Multiple dimensions stacked

**Features**:
- Show top 5-10 dimensions
- Color = positive (blue) vs negative (red)
- Zoom to time range
- Highlight significant changes

**Use cases**:
- Detect semantic drift
- Monitor system behavior
- Find temporal patterns

**File**: `HoloLoom/visualization/dimension_evolution.py`
**Complexity**: High (4-5 hours)
**Impact**: HIGH - temporal analysis

---

### 3.2 Confidence Trajectory 🎯

**What it reveals**: How confidence evolves during decision-making

**Visual** (Line chart with milestones):
```
Confidence
1.0  │                         ●  (final: 0.92)
     │                     ●●
     │                 ●●
     │             ●●
0.5  │         ●●
     │     ●●
0.0  └─────────────────────────────> Stage
     Extract Retrieve Converge Decide
```

**Implementation**:
- **Chart**: Line with stage markers
- **Data**: Confidence at each stage
- **Features**:
  - Gradient fill under line
  - Stage labels at bottom
  - Threshold line at 0.9
  - Highlight if confidence drops

**Use cases**:
- Debug low-confidence decisions
- Find stages that reduce confidence
- Optimize decision pipeline

**File**: `HoloLoom/visualization/confidence_trajectory.py`
**Complexity**: Low (1-2 hours)
**Impact**: MEDIUM - decision quality

---

### 3.3 Memory Access Heatmap ⏱️

**What it reveals**: Which memories are frequently accessed

**Visual** (Calendar-style heatmap):
```
Memory Shard Access Frequency

Shard ID    Mon  Tue  Wed  Thu  Fri
Memory A    ███  ███  ██   █    ███
Memory B    █    ██   ███  ███  ██
Memory C    ███  █    █    ██   ███
```

**Implementation**:
- **Chart**: Calendar heatmap
- **Cells**: Access count (darker = more)
- **Axes**: Memory ID × Time

**Features**:
- Hover → show memory content
- Click → drill into access details
- Filter by time range
- Highlight cold memories (never accessed)

**Use cases**:
- Identify hot memories (optimize caching)
- Find stale memories (cleanup)
- Balance memory distribution

**File**: `HoloLoom/visualization/memory_heatmap.py`
**Complexity**: Medium (2-3 hours)
**Impact**: MEDIUM - memory optimization

---

## Category 4: Multi-Modal Visualizations

### 4.1 Cross-Modal Attention Map 🎭

**What it reveals**: Which modalities influenced decision most

**Visual** (Sankey diagram):
```
Text Input ════════════╗
                       ║ 60% weight
Image Input ══════╗    ║
               ║  ╚════╝
Audio Input    ║  ║
             ║  ║  ║
             ╚══╩══╩═══> Final Decision
                30%  10%
```

**Implementation**:
- **Chart**: Sankey flow diagram
- **Flows**: Modality → decision weight
- **Width**: Contribution strength

**Features**:
- Color = modality type
- Hover → show features extracted
- Animate flow on load
- Show attention scores

**Use cases**:
- Debug multi-modal fusion
- Understand modality importance
- Optimize fusion strategy

**File**: `HoloLoom/visualization/attention_flow.py`
**Complexity**: High (4-5 hours)
**Impact**: HIGH - multi-modal insights

---

### 4.2 Modality Contribution Treemap 🌳

**What it reveals**: Hierarchical breakdown of modality contributions

**Visual** (Treemap):
```
┌─────────────────────────────────┐
│ Text (60%)                      │
│  ┌────────────┬────────────┐    │
│  │ Entities   │ Topics     │    │
│  │  (35%)     │  (25%)     │    │
│  └────────────┴────────────┘    │
├─────────────────────────────────┤
│ Images (30%)  │ Audio (10%)    │
│  ┌──────────┐ │                │
│  │ Objects  │ │ Transcription  │
│  │ (20%)    │ │ (10%)          │
│  └──────────┘ │                │
└─────────────────────────────────┘
```

**Implementation**:
- **Chart**: Treemap (rectangles)
- **Size**: Contribution percentage
- **Color**: Modality type
- **Nesting**: Modality → feature type

**Features**:
- Interactive drill-down
- Hover → show exact values
- Click → show raw features
- Animate transitions

**Use cases**:
- Understand feature importance
- Balance multi-modal fusion
- Debug feature extraction

**File**: `HoloLoom/visualization/modality_treemap.py`
**Complexity**: Medium (3-4 hours)
**Impact**: HIGH - multi-modal analysis

---

## Category 5: Decision & Exploration Visualizations

### 5.1 Thompson Sampling Exploration Map 🎲

**What it reveals**: Exploration vs exploitation balance

**Visual** (Scatter plot):
```
Exploitation
    │     ●●●●●  (cached, high confidence)
    │    ●●●●
    │   ●●
    │  ●
    │ ●
Exploration
    └───────────────────> Time
```

**Implementation**:
- **Chart**: Scatter with trend line
- **Y-axis**: Exploration score (0-1)
- **X-axis**: Query sequence
- **Color**: Success (green) vs failure (red)

**Features**:
- Trend line shows drift
- Threshold bands (0-0.3 exploit, 0.7-1.0 explore)
- Hover → show query + stats
- Click → show bandit stats

**Use cases**:
- Monitor exploration strategy
- Detect exploration bias
- Optimize epsilon parameter

**File**: `HoloLoom/visualization/exploration_map.py`
**Complexity**: Medium (2-3 hours)
**Impact**: MEDIUM - RL optimization

---

### 5.2 Tool Selection Sunburst 🌅

**What it reveals**: Tool usage hierarchy and patterns

**Visual** (Sunburst diagram):
```
       answer (60%)
      /     |      \
  simple  detailed  technical
  (20%)   (25%)     (15%)

    compare (25%)
      /        \
  side-by-side  diff
   (15%)        (10%)

    search (15%)
```

**Implementation**:
- **Chart**: Sunburst (circular hierarchy)
- **Inner ring**: Tool type
- **Outer rings**: Sub-categories
- **Size**: Usage frequency

**Features**:
- Click to zoom
- Hover → show count + examples
- Color = success rate
- Animate on data update

**Use cases**:
- Understand tool usage
- Find underused tools
- Optimize tool selection

**File**: `HoloLoom/visualization/tool_sunburst.py`
**Complexity**: Medium (3-4 hours)
**Impact**: MEDIUM - tool analysis

---

## Category 6: Performance & Optimization Visualizations

### 6.1 Latency Distribution Violin Plot 🎻

**What it reveals**: Performance distribution (not just averages)

**Visual** (Violin plot):
```
                 ╭─╮
                ╭┤ ├╮
              ╭─┤ │ ├─╮
            ╭─┤ │ │ │ ├─╮
          ╭─┤ │ │ │ │ │ ├─╮
         ┤ │ │ │ │ │ │ │ │
         └─┴─┴─┴─┴─┴─┴─┴─┘
         20 40 60 80 100 120 140ms

Median: 85ms | P95: 130ms | P99: 145ms
```

**Implementation**:
- **Chart**: Violin or ridgeline plot
- **Data**: All query latencies
- **Overlays**: Median, P95, P99 lines

**Features**:
- Show full distribution (not just mean)
- Overlay box plot
- Multiple violins for comparison
- Interactive tooltips

**Use cases**:
- Find outliers (long tail)
- Detect bimodal distributions
- Set realistic SLAs

**File**: `HoloLoom/visualization/latency_violin.py`
**Complexity**: Low (1-2 hours)
**Impact**: MEDIUM - performance analysis

---

### 6.2 Stage Waterfall Chart 💧

**What it reveals**: Sequential stage timing with dependencies

**Visual** (Waterfall):
```
Pattern Selection     ████ (5ms)
Retrieval                 ████████████████ (50ms)
Convergence                                ████████ (30ms)
Tool Execution                                     ████████████████ (60ms)
─────────────────────────────────────────────────────────────>
0ms                                                         145ms
```

**Implementation**:
- **Chart**: Horizontal stacked bars with offsets
- **Bars**: Stages with start time offset
- **Width**: Duration

**Features**:
- Color = stage type
- Hover → show stage details
- Click → drill into sub-stages
- Highlight bottlenecks (>40%)

**Use cases**:
- Visualize pipeline flow
- Find sequential bottlenecks
- Identify parallelization opportunities

**File**: `HoloLoom/visualization/stage_waterfall.py`
**Complexity**: Low (1-2 hours)
**Impact**: HIGH - pipeline optimization

---

### 6.3 Cache Effectiveness Gauge 🎯

**What it reveals**: Cache performance at a glance

**Visual** (Gauge/speedometer):
```
        ┌───────┐
      ╱   75%   ╲
     │  HIT RATE │
     │    ▲      │
     │   ╱ ╲     │
     └──╱───╲────┘
       ╱     ╲
    BAD  OK  GOOD
    0%   50%  100%
```

**Implementation**:
- **Chart**: Radial gauge
- **Needle**: Current hit rate
- **Zones**: Red (<50%), yellow (50-75%), green (>75%)

**Features**:
- Animate needle
- Show trend (↑↓)
- Display savings (ms and %)
- Click → show cache details

**Use cases**:
- Monitor cache health
- Quick status check
- Alert on degradation

**File**: `HoloLoom/visualization/cache_gauge.py`
**Complexity**: Low (1-2 hours)
**Impact**: MEDIUM - cache monitoring

---

## Category 7: Spatial & Geographic Visualizations

### 7.1 Memory Embedding Space Map 🗺️

**What it reveals**: Spatial distribution of memories

**Visual** (Interactive map metaphor):
```
     Topic A Region
    ████████████
    ████████████
                 Topic B Region
                 ████████████

  Query Path: A → B → A
     (shows traversal)
```

**Implementation**:
- **Layout**: 2D projection of embedding space
- **Regions**: Colored by topic/cluster
- **Paths**: Query retrieval trajectory
- **Markers**: Memory shards

**Features**:
- Pan/zoom like a map
- Paths show retrieval order
- Heat overlay for density
- Search highlights matching memories

**Use cases**:
- Visualize memory landscape
- Find memory clusters
- Understand retrieval patterns

**File**: `HoloLoom/visualization/memory_map.py`
**Complexity**: High (4-5 hours)
**Impact**: HIGH - spatial understanding

---

## Priority Matrix

| Visualization | Complexity | Impact | Priority | Effort |
|---------------|------------|--------|----------|--------|
| **Knowledge Graph Map** | Medium | HIGH | ★★★★★ | 3-4h |
| **Semantic Space Projection** | Medium | HIGH | ★★★★★ | 3-4h |
| **Confidence Trajectory** | Low | MEDIUM | ★★★★☆ | 1-2h |
| **Stage Waterfall** | Low | HIGH | ★★★★★ | 1-2h |
| **Cross-Modal Attention** | High | HIGH | ★★★★★ | 4-5h |
| **Modality Treemap** | Medium | HIGH | ★★★★☆ | 3-4h |
| **Memory Access Heatmap** | Medium | MEDIUM | ★★★☆☆ | 2-3h |
| **Dimension Evolution** | High | HIGH | ★★★★☆ | 4-5h |
| **Query Similarity Network** | Medium | MEDIUM | ★★★☆☆ | 2-3h |
| **Latency Violin** | Low | MEDIUM | ★★★☆☆ | 1-2h |
| **Cache Gauge** | Low | MEDIUM | ★★★☆☆ | 1-2h |
| **Thompson Sampling Map** | Medium | MEDIUM | ★★★☆☆ | 2-3h |
| **Tool Sunburst** | Medium | MEDIUM | ★★★☆☆ | 3-4h |
| **Memory Map** | High | HIGH | ★★★★☆ | 4-5h |
| **Embedding Clusters** | Medium | MEDIUM | ★★★☆☆ | 2-3h |

---

## Recommended Next Sprints

### Sprint 4: Network & Knowledge (Week 4, 6-8 hours)
1. **Knowledge Graph Map** (3-4h) - Reveals memory structure
2. **Stage Waterfall** (1-2h) - Pipeline optimization
3. **Confidence Trajectory** (1-2h) - Decision quality

**Impact**: Understand how memories connect and how decisions form

---

### Sprint 5: Spatial & Semantic (Week 5, 6-8 hours)
1. **Semantic Space Projection** (3-4h) - Visualize 244D space
2. **Embedding Clusters** (2-3h) - Find natural groupings
3. **Cache Gauge** (1-2h) - Quick health check

**Impact**: See the semantic landscape

---

### Sprint 6: Multi-Modal Insights (Week 6, 7-9 hours)
1. **Cross-Modal Attention** (4-5h) - Fusion transparency
2. **Modality Treemap** (3-4h) - Feature importance

**Impact**: Understand multi-modal decisions

---

## Implementation Strategy

**For each visualization**:
1. Create dedicated module in `HoloLoom/visualization/`
2. Define data structure (what data needed)
3. Implement rendering (HTML/JS or Plotly)
4. Add to dashboard strategy selector
5. Write tests (visual validation)
6. Document usage in CLAUDE.md

**Guiding Principles**:
- **Meaning first**: Every pixel must convey information
- **Actionable**: Users should know what to do after seeing it
- **Interactive**: Allow drill-down and exploration
- **Consistent**: Follow Tufte principles
- **Fast**: Render in <500ms

---

## Technologies

**Recommended Stack**:
- **2D Charts**: Plotly (interactive, Python-native)
- **3D Visualizations**: Plotly 3D or Three.js
- **Networks**: D3.js force-directed or Cytoscape.js
- **Maps**: Leaflet.js or Mapbox
- **Projections**: scikit-learn (t-SNE, UMAP)
- **Rendering**: Same HTML renderer infrastructure

**Zero-dependency goal**: Prefer pure HTML/CSS/SVG where possible

---

## Summary

**Categories**:
1. Network & Graph (2 visualizations)
2. Spatial & Projection (2)
3. Temporal & Evolution (3)
4. Multi-Modal (2)
5. Decision & Exploration (2)
6. Performance & Optimization (3)
7. Spatial & Geographic (1)

**Total**: 15 new visualizations
**Effort**: 35-45 hours across 3 sprints
**Impact**: Complete visualization suite for all HoloLoom insights

**Quick Wins** (low effort, high impact):
- Stage Waterfall (1-2h, HIGH impact)
- Confidence Trajectory (1-2h, MEDIUM impact)
- Cache Gauge (1-2h, MEDIUM impact)

**Choose your adventure**: Which category interests you most?
