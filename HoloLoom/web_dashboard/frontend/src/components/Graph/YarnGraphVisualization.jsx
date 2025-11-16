import React, { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ForceDirectedLayout, getEdgeColor, getNodeSize } from '../../utils/graphLayout'
import { apiClient } from '../../utils/api'

function YarnGraphVisualization({ filters }) {
  const svgRef = useRef(null)
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] })
  const [loading, setLoading] = useState(true)
  const [selectedNode, setSelectedNode] = useState(null)

  // Fetch graph data
  useEffect(() => {
    fetchGraphData()
  }, [])

  const fetchGraphData = async () => {
    try {
      // Try to fetch from API, fallback to demo data
      try {
        const response = await apiClient.getGraphData()
        setGraphData(response.data)
      } catch (error) {
        // Use demo data if API unavailable
        console.log('Using demo graph data')
        setGraphData(createDemoGraphData())
      }
      setLoading(false)
    } catch (error) {
      console.error('Failed to fetch graph data:', error)
      setLoading(false)
    }
  }

  // Render graph with D3
  useEffect(() => {
    if (!svgRef.current || graphData.nodes.length === 0) return

    renderGraph()
  }, [graphData, filters])

  const renderGraph = () => {
    const svg = d3.select(svgRef.current)
    const width = svgRef.current.clientWidth
    const height = svgRef.current.clientHeight

    // Clear previous content
    svg.selectAll('*').remove()

    // Apply filters
    let filteredNodes = graphData.nodes.slice(0, filters.maxNodes)
    let filteredEdges = graphData.edges.filter(edge =>
      filteredNodes.find(n => n.id === edge.src) &&
      filteredNodes.find(n => n.id === edge.dst)
    )

    if (filters.edgeTypes && filters.edgeTypes.length > 0) {
      filteredEdges = filteredEdges.filter(edge =>
        filters.edgeTypes.includes(edge.type)
      )
    }

    // Compute layout
    const layout = new ForceDirectedLayout(width, height, {
      iterations: filters.layoutSpeed === 'fast' ? 150 : filters.layoutSpeed === 'slow' ? 500 : 300
    })
    const positionedNodes = layout.layout(filteredNodes, filteredEdges)

    // Calculate node degrees
    const degrees = new Map()
    filteredEdges.forEach(edge => {
      degrees.set(edge.src, (degrees.get(edge.src) || 0) + 1)
      degrees.set(edge.dst, (degrees.get(edge.dst) || 0) + 1)
    })

    positionedNodes.forEach(node => {
      node.degree = degrees.get(node.id) || 0
    })

    const maxDegree = Math.max(...positionedNodes.map(n => n.degree), 1)
    const minDegree = Math.min(...positionedNodes.map(n => n.degree), 0)

    // Create arrow markers for edges
    const defs = svg.append('defs')
    const edgeTypes = [...new Set(filteredEdges.map(e => e.type))]
    edgeTypes.forEach(type => {
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
        .attr('fill', getEdgeColor(type))
    })

    // Draw edges
    const edges = svg.append('g')
      .selectAll('line')
      .data(filteredEdges)
      .enter()
      .append('line')
      .attr('x1', d => positionedNodes.find(n => n.id === d.src)?.x || 0)
      .attr('y1', d => positionedNodes.find(n => n.id === d.src)?.y || 0)
      .attr('x2', d => positionedNodes.find(n => n.id === d.dst)?.x || 0)
      .attr('y2', d => positionedNodes.find(n => n.id === d.dst)?.y || 0)
      .attr('stroke', d => getEdgeColor(d.type))
      .attr('stroke-width', 2)
      .attr('opacity', 0.6)
      .attr('marker-end', d => `url(#arrow-${d.type})`)

    // Draw nodes
    const nodes = svg.append('g')
      .selectAll('circle')
      .data(positionedNodes)
      .enter()
      .append('circle')
      .attr('cx', d => d.x)
      .attr('cy', d => d.y)
      .attr('r', d => getNodeSize(d.degree, 8, 24, minDegree, maxDegree))
      .attr('fill', '#6366f1')
      .attr('stroke', '#4f46e5')
      .attr('stroke-width', 2)
      .attr('cursor', 'pointer')
      .on('click', (event, d) => {
        setSelectedNode(d)
      })
      .on('mouseover', function() {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('stroke-width', 4)
          .attr('r', function() {
            return +d3.select(this).attr('r') * 1.2
          })
      })
      .on('mouseout', function() {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('stroke-width', 2)
          .attr('r', d => getNodeSize(d.degree, 8, 24, minDegree, maxDegree))
      })

    // Draw labels (if enabled)
    if (filters.showLabels) {
      svg.append('g')
        .selectAll('text')
        .data(positionedNodes)
        .enter()
        .append('text')
        .attr('x', d => d.x)
        .attr('y', d => d.y + getNodeSize(d.degree, 8, 24, minDegree, maxDegree) + 12)
        .attr('text-anchor', 'middle')
        .attr('font-size', '11px')
        .attr('font-weight', '500')
        .attr('fill', '#374151')
        .text(d => d.label.substring(0, 15))
    }
  }

  const createDemoGraphData = () => {
    return {
      nodes: [
        { id: 'attention', label: 'Attention', degree: 4 },
        { id: 'transformer', label: 'Transformer', degree: 5 },
        { id: 'neural_network', label: 'Neural Network', degree: 3 },
        { id: 'bert', label: 'BERT', degree: 2 },
        { id: 'gpt', label: 'GPT', degree: 2 },
        { id: 'multihead', label: 'Multi-Head', degree: 2 },
        { id: 'selfattention', label: 'Self-Attention', degree: 2 },
        { id: 'embedding', label: 'Embedding', degree: 3 },
        { id: 'encoder', label: 'Encoder', degree: 2 },
        { id: 'decoder', label: 'Decoder', degree: 2 },
      ],
      edges: [
        { src: 'attention', dst: 'transformer', type: 'uses', weight: 1.0 },
        { src: 'transformer', dst: 'neural_network', type: 'is_a', weight: 1.0 },
        { src: 'attention', dst: 'neural_network', type: 'part_of', weight: 0.8 },
        { src: 'bert', dst: 'transformer', type: 'is_a', weight: 1.0 },
        { src: 'gpt', dst: 'transformer', type: 'is_a', weight: 1.0 },
        { src: 'multihead', dst: 'attention', type: 'is_a', weight: 1.0 },
        { src: 'selfattention', dst: 'attention', type: 'is_a', weight: 1.0 },
        { src: 'embedding', dst: 'transformer', type: 'uses', weight: 1.0 },
        { src: 'encoder', dst: 'transformer', type: 'part_of', weight: 1.0 },
        { src: 'decoder', dst: 'transformer', type: 'part_of', weight: 1.0 },
      ]
    }
  }

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
        <div className="flex items-center justify-center h-[600px]">
          <div className="spinner" />
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200">
      <div className="grid grid-cols-1 lg:grid-cols-4 divide-y lg:divide-y-0 lg:divide-x divide-gray-200">
        {/* Graph Canvas */}
        <div className="lg:col-span-3 p-6">
          <svg
            ref={svgRef}
            className="w-full border border-gray-200 rounded-lg bg-gray-50"
            style={{ height: '600px' }}
          />
        </div>

        {/* Node Details Panel */}
        <div className="p-6">
          <h3 className="text-sm font-semibold text-gray-900 mb-4">
            {selectedNode ? 'Node Details' : 'Graph Info'}
          </h3>

          {selectedNode ? (
            <div className="space-y-4">
              <div>
                <p className="text-xs text-gray-500 mb-1">Node ID</p>
                <p className="text-sm font-medium text-gray-900">{selectedNode.id}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500 mb-1">Label</p>
                <p className="text-sm font-medium text-gray-900">{selectedNode.label}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500 mb-1">Degree</p>
                <p className="text-sm font-medium text-gray-900">{selectedNode.degree || 0}</p>
              </div>
              <button
                onClick={() => setSelectedNode(null)}
                className="w-full px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
              >
                Clear Selection
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <p className="text-xs text-gray-500 mb-1">Total Nodes</p>
                <p className="text-2xl font-bold text-gray-900">{graphData.nodes.length}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500 mb-1">Total Edges</p>
                <p className="text-2xl font-bold text-gray-900">{graphData.edges.length}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500 mb-1">Visible</p>
                <p className="text-sm text-gray-700">
                  {Math.min(filters.maxNodes, graphData.nodes.length)} nodes
                </p>
              </div>
            </div>
          )}

          {/* Legend */}
          <div className="mt-8 pt-6 border-t border-gray-200">
            <p className="text-xs font-semibold text-gray-500 uppercase mb-3">Edge Types</p>
            <div className="space-y-2">
              {['is_a', 'uses', 'part_of', 'mentions'].map(type => (
                <div key={type} className="flex items-center gap-2">
                  <div
                    className="w-4 h-0.5"
                    style={{ backgroundColor: getEdgeColor(type) }}
                  />
                  <span className="text-xs text-gray-600">{type.replace('_', ' ')}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default YarnGraphVisualization
