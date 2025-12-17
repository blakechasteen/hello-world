import { useCurrentView, useAppStore } from '@stores/appStore'
import { DashboardView } from '../../types'

interface HeaderProps {
  isConnected: boolean
}

/**
 * Header Component
 *
 * Top navigation bar with:
 * - HoloLoom branding
 * - View switcher (Overview, Agents, Tasks, Metrics, Logs)
 * - Connection status indicator
 * - Quick actions
 */
export default function Header({ isConnected }: HeaderProps) {
  const currentView = useCurrentView()
  const { setCurrentView } = useAppStore()

  const views: { id: DashboardView; label: string; icon: string }[] = [
    { id: DashboardView.OVERVIEW, label: 'Overview', icon: '📊' },
    { id: DashboardView.AGENTS, label: 'Agents', icon: '🤖' },
    { id: DashboardView.TASKS, label: 'Tasks', icon: '📋' },
    { id: DashboardView.METRICS, label: 'Metrics', icon: '📈' },
    { id: DashboardView.LOGS, label: 'Logs', icon: '📝' },
    { id: DashboardView.SETTINGS, label: 'Settings', icon: '⚙️' },
  ]

  return (
    <header className="bg-surface-secondary border-b border-dark-light px-6 py-4 shadow-dark-md">
      <div className="flex items-center justify-between mb-4">
        {/* Logo & Title */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-holo-primary to-holo-secondary rounded-lg flex items-center justify-center">
            <span className="text-white font-bold">HL</span>
          </div>
          <h1 className="text-xl font-bold text-text-primary">Agent Manager</h1>
        </div>

        {/* Connection Status */}
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              isConnected ? 'bg-agent-success glow-success' : 'bg-agent-error'
            }`}
          />
          <span className="text-xs text-text-secondary">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* View Navigation */}
      <nav className="flex gap-1 overflow-x-auto">
        {views.map((view) => (
          <button
            key={view.id}
            onClick={() => setCurrentView(view.id)}
            className={`px-3 py-2 rounded-md text-sm font-medium whitespace-nowrap transition-all ${
              currentView === view.id
                ? 'bg-holo-primary text-white shadow-glow-primary'
                : 'text-text-secondary hover:bg-surface-tertiary hover:text-text-primary'
            }`}
          >
            {view.icon} {view.label}
          </button>
        ))}
      </nav>
    </header>
  )
}
