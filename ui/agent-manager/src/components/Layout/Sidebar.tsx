import { useState } from 'react'
import { useAgents, useSidebarOpen, useAppStore } from '@stores/appStore'
import { AgentState } from '../../types'

/**
 * Sidebar Component
 *
 * Displays:
 * - List of active agents with state badges
 * - Agent filtering by type/state
 * - Collapse/expand toggle
 * - Agent quick actions
 */
export default function Sidebar() {
  const agents = useAgents()
  const sidebarOpen = useSidebarOpen()
  const { setSidebarOpen, setSelectedAgent } = useAppStore()
  const [filterState, setFilterState] = useState<AgentState | 'all'>('all')

  const filteredAgents =
    filterState === 'all'
      ? agents
      : agents.filter((agent) => agent.state === filterState)

  const _stateColors: Record<AgentState, string> = {
    idle: 'bg-gray-900 text-gray-300',
    ready: 'bg-emerald-950 text-emerald-200',
    running: 'bg-cyan-950 text-cyan-200',
    success: 'bg-emerald-900 text-emerald-100',
    warning: 'bg-amber-950 text-amber-200',
    error: 'bg-red-950 text-red-200',
    paused: 'bg-orange-950 text-orange-200',
  }

  const stateLabels: Record<AgentState, string> = {
    idle: 'Idle',
    ready: 'Ready',
    running: 'Running',
    success: 'Success',
    warning: 'Warning',
    error: 'Error',
    paused: 'Paused',
  }

  return (
    <aside
      className={`bg-surface-secondary border-r border-dark-light transition-all duration-300 overflow-y-auto flex flex-col ${
        sidebarOpen ? 'w-64' : 'w-16'
      }`}
    >
      {/* Toggle Button */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="p-4 hover:bg-surface-tertiary transition-colors"
        title={sidebarOpen ? 'Collapse' : 'Expand'}
      >
        <span className="text-xl">{sidebarOpen ? '◀️' : '▶️'}</span>
      </button>

      {/* Sidebar Content */}
      {sidebarOpen && (
        <div className="flex-1 flex flex-col p-4 gap-4 min-h-0">
          {/* Filter Section */}
          <div className="space-y-2">
            <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
              Filter
            </h3>
            <select
              value={filterState}
              onChange={(e) => setFilterState(e.target.value as AgentState | 'all')}
              className="w-full bg-surface-tertiary border border-dark-light rounded px-2 py-1 text-xs text-text-primary focus:outline-none focus:ring-2 focus:ring-holo-primary"
            >
              <option value="all">All Agents</option>
              <option value="idle">Idle</option>
              <option value="ready">Ready</option>
              <option value="running">Running</option>
              <option value="success">Success</option>
              <option value="error">Error</option>
            </select>
          </div>

          {/* Agent List */}
          <div className="space-y-2 flex-1 overflow-y-auto">
            <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
              Agents ({filteredAgents.length})
            </h3>

            {filteredAgents.length === 0 ? (
              <div className="text-xs text-text-tertiary text-center py-4">
                No agents found
              </div>
            ) : (
              <div className="space-y-2">
                {filteredAgents.map((agent) => (
                  <button
                    key={agent.id}
                    onClick={() => setSelectedAgent(agent.id)}
                    className="card-interactive w-full p-3 text-left group"
                  >
                    <div className="flex items-start gap-2 min-w-0">
                      {/* Icon */}
                      <span className="text-lg flex-shrink-0 mt-0.5">
                        {agent.type === 'query'
                          ? '❓'
                          : agent.type === 'process'
                            ? '⚙️'
                            : agent.type === 'memory'
                              ? '🧠'
                              : agent.type === 'decision'
                                ? '🎯'
                                : agent.type === 'output'
                                  ? '📤'
                                  : '🔄'}
                      </span>

                      {/* Info */}
                      <div className="min-w-0 flex-1">
                        <div className="text-sm font-medium text-text-primary truncate">
                          {agent.name}
                        </div>
                        <div className="text-xs text-text-tertiary">
                          {agent.type.charAt(0).toUpperCase() +
                            agent.type.slice(1)}
                        </div>
                      </div>
                    </div>

                    {/* Badge */}
                    <div className="mt-2 flex items-center gap-2">
                      <span className={`badge-agent-${agent.state}`}>
                        {stateLabels[agent.state]}
                      </span>
                      <span className="text-xs text-text-tertiary">
                        {(agent.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Stats Footer */}
          <div className="border-t border-dark-light pt-4 space-y-2 text-xs">
            <div className="flex justify-between text-text-secondary">
              <span>Total Agents:</span>
              <span className="font-semibold">{agents.length}</span>
            </div>
            <div className="flex justify-between text-text-secondary">
              <span>Running:</span>
              <span className="font-semibold text-cyan-400">
                {agents.filter((a) => a.state === 'running').length}
              </span>
            </div>
            <div className="flex justify-between text-text-secondary">
              <span>Errors:</span>
              <span className="font-semibold text-red-400">
                {agents.filter((a) => a.state === 'error').length}
              </span>
            </div>
          </div>
        </div>
      )}
    </aside>
  )
}
