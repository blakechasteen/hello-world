import { useAgents } from '@stores/appStore'

/**
 * Agents View
 *
 * Displays:
 * - Detailed agent list with type/state/metrics
 * - Agent filtering and search
 * - Individual agent controls
 * - Agent performance graphs
 */
export default function AgentsView() {
  const agents = useAgents()

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-text-primary">Agents</h1>
        <p className="text-text-secondary mt-1">
          Manage and monitor individual agents in the swarm
        </p>
      </div>

      <div className="card p-6">
        <h2 className="text-xl font-semibold text-text-primary mb-4">
          Agent List ({agents.length})
        </h2>

        {agents.length === 0 ? (
          <div className="text-center py-12 text-text-secondary">
            <p className="text-lg mb-2">No agents connected</p>
            <p className="text-sm">Agents will appear here when they connect to the backend</p>
          </div>
        ) : (
          <div className="space-y-3">
            {agents.map((agent) => (
              <div
                key={agent.id}
                className="card-interactive p-4 flex items-center justify-between"
              >
                <div className="flex items-center gap-4 flex-1 min-w-0">
                  <span className="text-2xl flex-shrink-0">
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
                  <div className="min-w-0 flex-1">
                    <div className="font-semibold text-text-primary truncate">
                      {agent.name}
                    </div>
                    <div className="text-xs text-text-tertiary">
                      {agent.type} • {agent.reasoningMode}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-4 flex-shrink-0">
                  <div className="text-right text-sm">
                    <div className="text-text-primary font-semibold">
                      {(agent.confidence * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-text-tertiary">confidence</div>
                  </div>
                  <span className={`badge-agent-${agent.state}`}>
                    {agent.state.charAt(0).toUpperCase() + agent.state.slice(1)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="card p-6">
        <h2 className="text-xl font-semibold text-text-primary mb-4">
          Detailed View
        </h2>
        <div className="text-center py-12 text-text-secondary">
          <p>📊 Agent details and graphs coming soon</p>
        </div>
      </div>
    </div>
  )
}
