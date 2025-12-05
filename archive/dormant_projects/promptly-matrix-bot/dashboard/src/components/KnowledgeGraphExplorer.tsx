/**
 * Knowledge Graph Explorer Component
 *
 * Interactive D3.js force-directed graph visualization showing:
 * - Entity nodes (concepts, entities, motifs)
 * - Relationship edges (IS_A, USES, MENTIONS, LEADS_TO, PART_OF)
 * - Click-to-explore functionality
 * - Relationship filtering
 * - Path highlighting for reasoning chains
 */

import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { GraphNode, GraphEdge, KnowledgeGraph } from '../types';
import { Filter, ZoomIn, ZoomOut, RotateCcw, Maximize2 } from 'lucide-react';

interface KnowledgeGraphExplorerProps {
  graph?: KnowledgeGraph;
  onNodeClick?: (node: GraphNode) => void;
  highlightedPath?: string[];
}

const EDGE_COLORS = {
  IS_A: '#3B82F6',      // Blue
  USES: '#10B981',      // Green
  MENTIONS: '#6B7280',  // Gray
  LEADS_TO: '#F59E0B',  // Orange
  PART_OF: '#8B5CF6',   // Purple
};

const NODE_COLORS = {
  entity: '#3B82F6',   // Blue
  motif: '#F59E0B',    // Orange
  concept: '#8B5CF6',  // Purple
};

export const KnowledgeGraphExplorer: React.FC<KnowledgeGraphExplorerProps> = ({
  graph,
  onNodeClick,
  highlightedPath = [],
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [filters, setFilters] = useState<Set<string>>(new Set());
  const [zoomLevel, setZoomLevel] = useState(1);

  useEffect(() => {
    if (!graph || !svgRef.current || !containerRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = 600;

    // Clear previous graph
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    // Create zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
        setZoomLevel(event.transform.k);
      });

    svg.call(zoom);

    const g = svg.append('g');

    // Filter edges based on selected filters
    const filteredEdges = filters.size > 0
      ? graph.edges.filter(e => filters.has(e.type))
      : graph.edges;

    // Create force simulation
    const simulation = d3.forceSimulation(graph.nodes as any)
      .force('link', d3.forceLink(filteredEdges as any)
        .id((d: any) => d.id)
        .distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    // Create arrow markers for directed edges
    const defs = svg.append('defs');
    Object.entries(EDGE_COLORS).forEach(([type, color]) => {
      defs.append('marker')
        .attr('id', `arrow-${type}`)
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 20)
        .attr('refY', 0)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', color);
    });

    // Create edges
    const link = g.append('g')
      .selectAll('line')
      .data(filteredEdges)
      .enter()
      .append('line')
      .attr('stroke', d => EDGE_COLORS[d.type] || '#6B7280')
      .attr('stroke-width', d => Math.sqrt(d.weight) * 2)
      .attr('stroke-opacity', 0.6)
      .attr('marker-end', d => `url(#arrow-${d.type})`);

    // Create edge labels
    const linkLabel = g.append('g')
      .selectAll('text')
      .data(filteredEdges)
      .enter()
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('fill', '#6B7280')
      .attr('opacity', 0.7)
      .text(d => d.type);

    // Create nodes
    const node = g.append('g')
      .selectAll('circle')
      .data(graph.nodes)
      .enter()
      .append('circle')
      .attr('r', d => Math.sqrt(d.connections) * 5 + 8)
      .attr('fill', d => NODE_COLORS[d.type])
      .attr('stroke', '#fff')
      .attr('stroke-width', d => highlightedPath.includes(d.id) ? 4 : 2)
      .attr('cursor', 'pointer')
      .on('click', (event, d) => {
        event.stopPropagation();
        setSelectedNode(d);
        if (onNodeClick) {
          onNodeClick(d);
        }
      })
      .on('mouseover', function(event, d) {
        d3.select(this)
          .attr('stroke-width', 4)
          .attr('stroke', '#F59E0B');

        // Show tooltip
        tooltip
          .style('opacity', 1)
          .html(`
            <strong>${d.label}</strong><br/>
            Type: ${d.type}<br/>
            Connections: ${d.connections}
          `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px');
      })
      .on('mouseout', function(event, d) {
        d3.select(this)
          .attr('stroke-width', highlightedPath.includes(d.id) ? 4 : 2)
          .attr('stroke', '#fff');

        tooltip.style('opacity', 0);
      })
      .call(d3.drag<any, any>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended) as any);

    // Create node labels
    const nodeLabel = g.append('g')
      .selectAll('text')
      .data(graph.nodes)
      .enter()
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.3em')
      .attr('font-size', '12px')
      .attr('font-weight', d => highlightedPath.includes(d.id) ? 'bold' : 'normal')
      .attr('fill', '#fff')
      .attr('pointer-events', 'none')
      .text(d => d.label.length > 15 ? d.label.substring(0, 12) + '...' : d.label);

    // Tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'graph-tooltip')
      .style('position', 'absolute')
      .style('background', 'rgba(0, 0, 0, 0.8)')
      .style('color', '#fff')
      .style('padding', '8px')
      .style('border-radius', '4px')
      .style('font-size', '12px')
      .style('pointer-events', 'none')
      .style('opacity', 0);

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      linkLabel
        .attr('x', (d: any) => (d.source.x + d.target.x) / 2)
        .attr('y', (d: any) => (d.source.y + d.target.y) / 2);

      node
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y);

      nodeLabel
        .attr('x', (d: any) => d.x)
        .attr('y', (d: any) => d.y);
    });

    // Drag functions
    function dragstarted(event: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: any) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: any) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }

    // Cleanup
    return () => {
      simulation.stop();
      tooltip.remove();
    };
  }, [graph, filters, highlightedPath, onNodeClick]);

  const handleFilterToggle = (edgeType: string) => {
    const newFilters = new Set(filters);
    if (newFilters.has(edgeType)) {
      newFilters.delete(edgeType);
    } else {
      newFilters.add(edgeType);
    }
    setFilters(newFilters);
  };

  const handleZoomIn = () => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.transition().call(
      d3.zoom<SVGSVGElement, unknown>().scaleBy as any,
      1.3
    );
  };

  const handleZoomOut = () => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.transition().call(
      d3.zoom<SVGSVGElement, unknown>().scaleBy as any,
      0.7
    );
  };

  const handleReset = () => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.transition().call(
      d3.zoom<SVGSVGElement, unknown>().transform as any,
      d3.zoomIdentity
    );
    setFilters(new Set());
    setSelectedNode(null);
  };

  const handleFitToScreen = () => {
    if (!svgRef.current || !containerRef.current || !graph) return;

    const svg = d3.select(svgRef.current);
    const width = containerRef.current.clientWidth;
    const height = 600;

    // Calculate bounding box
    const bounds = {
      minX: d3.min(graph.nodes, d => (d as any).x || 0) || 0,
      maxX: d3.max(graph.nodes, d => (d as any).x || 0) || width,
      minY: d3.min(graph.nodes, d => (d as any).y || 0) || 0,
      maxY: d3.max(graph.nodes, d => (d as any).y || 0) || height,
    };

    const graphWidth = bounds.maxX - bounds.minX;
    const graphHeight = bounds.maxY - bounds.minY;

    const scale = Math.min(
      width / graphWidth,
      height / graphHeight
    ) * 0.9;

    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerY = (bounds.minY + bounds.maxY) / 2;

    const transform = d3.zoomIdentity
      .translate(width / 2, height / 2)
      .scale(scale)
      .translate(-centerX, -centerY);

    svg.transition().call(
      d3.zoom<SVGSVGElement, unknown>().transform as any,
      transform
    );
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="mb-4">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Knowledge Graph Explorer
        </h2>
        <p className="text-gray-600">
          Interactive visualization of entity relationships and connections
        </p>
      </div>

      {/* Controls */}
      <div className="mb-4 flex items-center justify-between">
        {/* Zoom Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleZoomIn}
            className="p-2 bg-gray-100 hover:bg-gray-200 rounded transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-5 h-5 text-gray-700" />
          </button>
          <button
            onClick={handleZoomOut}
            className="p-2 bg-gray-100 hover:bg-gray-200 rounded transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-5 h-5 text-gray-700" />
          </button>
          <button
            onClick={handleFitToScreen}
            className="p-2 bg-gray-100 hover:bg-gray-200 rounded transition-colors"
            title="Fit to Screen"
          >
            <Maximize2 className="w-5 h-5 text-gray-700" />
          </button>
          <button
            onClick={handleReset}
            className="p-2 bg-gray-100 hover:bg-gray-200 rounded transition-colors"
            title="Reset View"
          >
            <RotateCcw className="w-5 h-5 text-gray-700" />
          </button>
          <span className="text-sm text-gray-600 ml-2">
            Zoom: {(zoomLevel * 100).toFixed(0)}%
          </span>
        </div>

        {/* Relationship Filters */}
        <div className="flex items-center gap-2">
          <Filter className="w-5 h-5 text-gray-600" />
          <span className="text-sm text-gray-600">Filter:</span>
          {Object.entries(EDGE_COLORS).map(([type, color]) => (
            <button
              key={type}
              onClick={() => handleFilterToggle(type)}
              className={`
                px-3 py-1 rounded text-xs font-medium transition-all
                ${filters.has(type) || filters.size === 0
                  ? 'opacity-100'
                  : 'opacity-30'
                }
              `}
              style={{
                backgroundColor: `${color}20`,
                color: color,
                border: `2px solid ${color}`,
              }}
            >
              {type}
            </button>
          ))}
        </div>
      </div>

      {/* Graph Canvas */}
      <div
        ref={containerRef}
        className="border-2 border-gray-200 rounded-lg overflow-hidden bg-gray-50"
        style={{ height: '600px' }}
      >
        <svg ref={svgRef} />
      </div>

      {/* Selected Node Details */}
      {selectedNode && (
        <div className="mt-4 p-4 bg-indigo-50 border-2 border-indigo-500 rounded-lg">
          <h3 className="font-medium text-indigo-900 mb-2">Selected Node</h3>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="text-indigo-600">Label:</span>{' '}
              <span className="font-medium">{selectedNode.label}</span>
            </div>
            <div>
              <span className="text-indigo-600">Type:</span>{' '}
              <span className="font-medium">{selectedNode.type}</span>
            </div>
            <div>
              <span className="text-indigo-600">Connections:</span>{' '}
              <span className="font-medium">{selectedNode.connections}</span>
            </div>
            <div>
              <span className="text-indigo-600">ID:</span>{' '}
              <span className="font-mono text-xs">{selectedNode.id}</span>
            </div>
          </div>
          {selectedNode.metadata && Object.keys(selectedNode.metadata).length > 0 && (
            <div className="mt-2 pt-2 border-t border-indigo-300">
              <span className="text-indigo-600 text-sm">Metadata:</span>
              <pre className="mt-1 text-xs bg-indigo-100 p-2 rounded overflow-auto">
                {JSON.stringify(selectedNode.metadata, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}

      {/* Legend */}
      <div className="mt-4 p-4 bg-gray-50 border border-gray-200 rounded-lg">
        <h3 className="font-medium text-gray-900 mb-3">Legend</h3>

        <div className="grid grid-cols-2 gap-4">
          {/* Node Types */}
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">Node Types:</p>
            <div className="space-y-1">
              {Object.entries(NODE_COLORS).map(([type, color]) => (
                <div key={type} className="flex items-center gap-2 text-sm">
                  <div
                    className="w-4 h-4 rounded-full border-2 border-white"
                    style={{ backgroundColor: color }}
                  />
                  <span className="capitalize">{type}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Edge Types */}
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">Relationship Types:</p>
            <div className="space-y-1">
              {Object.entries(EDGE_COLORS).map(([type, color]) => (
                <div key={type} className="flex items-center gap-2 text-sm">
                  <div
                    className="w-8 h-0.5"
                    style={{ backgroundColor: color }}
                  />
                  <span>{type}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <p className="mt-3 text-xs text-gray-600">
          <strong>Tip:</strong> Drag nodes to rearrange, click to select, use mouse wheel to zoom
        </p>
      </div>
    </div>
  );
};
