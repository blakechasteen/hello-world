/**
 * StatusBar Component - System status display
 *
 * Shows:
 * - Connection status
 * - Session ID
 * - AR session state
 * - Performance metrics (future)
 *
 * Created: 2025-11-22
 */

interface StatusBarProps {
  connected: boolean
  sessionId: string | null
  arActive: boolean
}

export default function StatusBar({ connected, sessionId, arActive }: StatusBarProps) {
  return (
    <div className="status-bar">
      {/* Connection status */}
      <div className="status-indicator">
        <div
          className="status-dot"
          style={{
            background: connected ? 'var(--color-success)' : 'var(--color-error)',
            animation: connected ? 'blink 2s ease-in-out infinite' : 'none',
          }}
        />
        <span>{connected ? 'Connected' : 'Disconnected'}</span>
      </div>

      {/* Session info */}
      {sessionId && (
        <div>
          Session: {sessionId.slice(0, 8)}...
        </div>
      )}

      {/* AR status */}
      <div className="status-indicator">
        <span>{arActive ? 'AR Active' : 'AR Standby'}</span>
      </div>
    </div>
  )
}
