import React from 'react'
import YarnGraphVisualization from '../components/Graph/YarnGraphVisualization'
import GraphControls from '../components/Graph/GraphControls'

function GraphView() {
  const [graphFilters, setGraphFilters] = React.useState({
    maxNodes: 50,
    edgeTypes: [],
    showLabels: true,
    layoutSpeed: 'normal',
  })

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Yarn Graph</h1>
        <p className="text-gray-600 mt-1">Interactive knowledge graph visualization</p>
      </div>

      {/* Controls */}
      <GraphControls
        filters={graphFilters}
        onFiltersChange={setGraphFilters}
      />

      {/* Graph Visualization */}
      <YarnGraphVisualization filters={graphFilters} />
    </div>
  )
}

export default GraphView
