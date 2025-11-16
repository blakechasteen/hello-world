import React from 'react'
import { Filter, Layout, Tag, Eye, EyeOff } from 'lucide-react'

function GraphControls({ filters, onFiltersChange }) {
  const updateFilter = (key, value) => {
    onFiltersChange({ ...filters, [key]: value })
  }

  const edgeTypes = [
    { value: 'is_a', label: 'Is A', color: '#3b82f6' },
    { value: 'uses', label: 'Uses', color: '#10b981' },
    { value: 'part_of', label: 'Part Of', color: '#8b5cf6' },
    { value: 'mentions', label: 'Mentions', color: '#6b7280' },
    { value: 'leads_to', label: 'Leads To', color: '#f59e0b' },
  ]

  const layoutSpeeds = [
    { value: 'fast', label: 'Fast', description: '150 iterations' },
    { value: 'normal', label: 'Normal', description: '300 iterations' },
    { value: 'slow', label: 'Slow', description: '500 iterations' },
  ]

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center gap-2 mb-6">
        <Filter className="w-5 h-5 text-gray-600" />
        <h2 className="text-lg font-semibold text-gray-900">Graph Controls</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Max Nodes */}
        <div>
          <label className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-3">
            <Tag className="w-4 h-4" />
            Max Nodes
          </label>
          <input
            type="range"
            min="10"
            max="100"
            value={filters.maxNodes}
            onChange={(e) => updateFilter('maxNodes', parseInt(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>10</span>
            <span className="font-medium text-gray-900">{filters.maxNodes}</span>
            <span>100</span>
          </div>
        </div>

        {/* Layout Speed */}
        <div>
          <label className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-3">
            <Layout className="w-4 h-4" />
            Layout Speed
          </label>
          <div className="grid grid-cols-3 gap-2">
            {layoutSpeeds.map(speed => (
              <button
                key={speed.value}
                onClick={() => updateFilter('layoutSpeed', speed.value)}
                className={`px-3 py-2 text-xs font-medium rounded-lg border-2 transition-all ${
                  filters.layoutSpeed === speed.value
                    ? 'border-hololoom-primary bg-hololoom-primary/5 text-hololoom-primary'
                    : 'border-gray-200 text-gray-700 hover:border-gray-300'
                }`}
              >
                {speed.label}
              </button>
            ))}
          </div>
        </div>

        {/* Show Labels Toggle */}
        <div>
          <label className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-3">
            {filters.showLabels ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
            Node Labels
          </label>
          <button
            onClick={() => updateFilter('showLabels', !filters.showLabels)}
            className={`w-full px-4 py-2.5 rounded-lg font-medium transition-all ${
              filters.showLabels
                ? 'bg-hololoom-primary text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            {filters.showLabels ? 'Visible' : 'Hidden'}
          </button>
        </div>
      </div>

      {/* Edge Type Filters */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <label className="text-sm font-medium text-gray-700 mb-3 block">
          Edge Type Filters
        </label>
        <div className="flex flex-wrap gap-2">
          {edgeTypes.map(type => {
            const isSelected = filters.edgeTypes.includes(type.value)
            return (
              <button
                key={type.value}
                onClick={() => {
                  const newTypes = isSelected
                    ? filters.edgeTypes.filter(t => t !== type.value)
                    : [...filters.edgeTypes, type.value]
                  updateFilter('edgeTypes', newTypes)
                }}
                className={`px-3 py-1.5 rounded-full text-xs font-medium border-2 transition-all ${
                  isSelected
                    ? 'border-transparent text-white'
                    : 'border-gray-200 text-gray-700 hover:border-gray-300'
                }`}
                style={isSelected ? { backgroundColor: type.color } : {}}
              >
                {type.label}
              </button>
            )
          })}
        </div>
        <p className="text-xs text-gray-500 mt-2">
          {filters.edgeTypes.length === 0
            ? 'All edge types visible'
            : `Showing ${filters.edgeTypes.length} edge type(s)`}
        </p>
      </div>
    </div>
  )
}

export default GraphControls
