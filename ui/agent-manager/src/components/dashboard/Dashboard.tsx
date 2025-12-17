import { Suspense, lazy } from 'react'
import { useCurrentView } from '@stores/appStore'

/**
 * Dashboard Component
 *
 * Routes between different views:
 * - Overview: System-wide metrics and quick stats
 * - Agents: Detailed agent management
 * - Tasks: Task queue and execution tracking
 * - Metrics: Performance and historical data
 * - Logs: System logs and diagnostics
 * - Settings: Configuration options
 *
 * Uses code splitting for better performance
 */

// Lazy load view components
const OverviewView = lazy(() => import('./views/OverviewView'))
const AgentsView = lazy(() => import('./views/AgentsView'))
const TasksView = lazy(() => import('./views/TasksView'))
const MetricsView = lazy(() => import('./views/MetricsView'))
const LogsView = lazy(() => import('./views/LogsView'))
const SettingsView = lazy(() => import('./views/SettingsView'))

const LoadingSpinner = () => (
  <div className="flex items-center justify-center h-full">
    <div className="text-center space-y-4">
      <div className="inline-block">
        <div className="w-12 h-12 border-4 border-surface-tertiary border-t-holo-primary rounded-full animate-spin" />
      </div>
      <p className="text-text-secondary">Loading...</p>
    </div>
  </div>
)

export default function Dashboard() {
  const currentView = useCurrentView()

  const renderView = () => {
    switch (currentView) {
      case 'overview':
        return <OverviewView />
      case 'agents':
        return <AgentsView />
      case 'tasks':
        return <TasksView />
      case 'metrics':
        return <MetricsView />
      case 'logs':
        return <LogsView />
      case 'settings':
        return <SettingsView />
      default:
        return <OverviewView />
    }
  }

  return (
    <main className="flex-1 overflow-y-auto bg-surface-primary">
      <Suspense fallback={<LoadingSpinner />}>
        <div className="p-6 min-h-full">
          {renderView()}
        </div>
      </Suspense>
    </main>
  )
}
