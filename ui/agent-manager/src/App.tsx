import { useEffect, useState } from 'react'
import Header from '@components/Layout/Header'
import Sidebar from '@components/Layout/Sidebar'
import Dashboard from '@components/dashboard/Dashboard'
import { useAppStore } from '@stores/appStore'

/**
 * Root App Component
 *
 * Manages the main layout and integration between HoloLoom backend.
 * Implements:
 * - Dark mode by default (enforced at HTML root)
 * - Responsive sidebar layout
 * - Real-time connection to Agent Manager backend (port 8002)
 * - Global state management via Zustand
 *
 * Architecture:
 * Header (navigation, status)
 *   ├─ Sidebar (agent list, controls)
 *   └─ Main (dashboard, metrics, logs)
 */
function App() {
  const [isConnected, setIsConnected] = useState(false)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const { setAgents, addLog } = useAppStore()

  useEffect(() => {
    // Establish connection to Agent Manager backend
    const connectToBackend = async () => {
      try {
        // Test Agent Manager API connectivity first (port 8002)
        const response = await fetch('/agent-health')
        if (!response.ok) {
          throw new Error('Agent Manager health check failed')
        }

        setIsConnected(true)
        setConnectionError(null)
        addLog('Backend', 'Connected to Agent Manager backend', 'success')

        // Initialize threads list
        await fetchThreads()
      } catch (error) {
        const message =
          error instanceof Error ? error.message : 'Failed to connect to backend'
        setConnectionError(message)
        addLog('Backend', `Connection error: ${message}`, 'error')
        setIsConnected(false)

        // Retry connection after 3 seconds
        const timeout = setTimeout(connectToBackend, 3000)
        return () => clearTimeout(timeout)
      }
      return undefined
    }

    const fetchThreads = async () => {
      try {
        // Fetch threads from Agent Manager API (proxied to port 8002/api/threads)
        const response = await fetch('/agent-api/threads')
        if (!response.ok) throw new Error('Failed to fetch threads')

        const data = await response.json()
        // data is { threads: [...], total, active, completed, failed }
        setAgents(data.threads || [])
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown error'
        addLog('Threads', `Failed to load threads: ${message}`, 'warning')
      }
    }

    connectToBackend()
  }, [setAgents, addLog])

  return (
    <div className="h-screen w-screen flex flex-col bg-surface-primary overflow-hidden dark">
      {/* Connection Status Banner */}
      {!isConnected && connectionError && (
        <div className="bg-red-950 border-b border-red-700 px-4 py-3 text-red-200 text-sm">
          <div className="flex items-center justify-between">
            <span>⚠️ {connectionError}</span>
            <span className="text-xs">Retrying...</span>
          </div>
        </div>
      )}

      {/* Main Layout */}
      <Header isConnected={isConnected} />

      <div className="flex flex-1 overflow-hidden gap-0">
        <Sidebar />
        <Dashboard />
      </div>
    </div>
  )
}

export default App
