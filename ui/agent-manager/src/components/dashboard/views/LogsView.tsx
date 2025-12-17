import { useState } from 'react'
import { useRecentLogs, useErrorLogs } from '@stores/appStore'

/**
 * Logs View
 *
 * Displays:
 * - System logs with filtering
 * - Error tracking and diagnostics
 * - Log search and export
 */
export default function LogsView() {
  const logs = useRecentLogs(100)
  const errorLogs = useErrorLogs()
  const [logLevel, setLogLevel] = useState<string | 'all'>('all')

  const filteredLogs =
    logLevel === 'all'
      ? logs
      : logs.filter((log) => log.level === logLevel)

  const levelColors: Record<string, string> = {
    info: 'text-blue-400',
    success: 'text-emerald-400',
    warning: 'text-amber-400',
    error: 'text-red-400',
  }

  const levelIcons: Record<string, string> = {
    info: 'ℹ️',
    success: '✅',
    warning: '⚠️',
    error: '❌',
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-text-primary">System Logs</h1>
        <p className="text-text-secondary mt-1">
          Real-time logging and diagnostics
        </p>
      </div>

      {/* Log Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card p-4 text-center">
          <div className="text-2xl font-bold text-blue-400">
            {logs.filter((l) => l.level === 'info').length}
          </div>
          <div className="text-xs text-text-secondary mt-1">Info</div>
        </div>
        <div className="card p-4 text-center">
          <div className="text-2xl font-bold text-emerald-400">
            {logs.filter((l) => l.level === 'success').length}
          </div>
          <div className="text-xs text-text-secondary mt-1">Success</div>
        </div>
        <div className="card p-4 text-center">
          <div className="text-2xl font-bold text-amber-400">
            {logs.filter((l) => l.level === 'warning').length}
          </div>
          <div className="text-xs text-text-secondary mt-1">Warning</div>
        </div>
        <div className="card p-4 text-center">
          <div className="text-2xl font-bold text-red-400">{errorLogs.length}</div>
          <div className="text-xs text-text-secondary mt-1">Error</div>
        </div>
      </div>

      {/* Log Filter */}
      <div className="card p-6">
        <div className="flex items-center gap-4 mb-4">
          <label className="text-sm font-medium text-text-secondary">
            Filter by level:
          </label>
          <select
            value={logLevel}
            onChange={(e) => setLogLevel(e.target.value)}
            className="bg-surface-tertiary border border-dark-light rounded px-3 py-1.5 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-holo-primary"
          >
            <option value="all">All Levels</option>
            <option value="info">Info</option>
            <option value="success">Success</option>
            <option value="warning">Warning</option>
            <option value="error">Error</option>
          </select>
        </div>

        {/* Logs List */}
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {filteredLogs.length === 0 ? (
            <div className="text-center py-8 text-text-tertiary">
              No logs found
            </div>
          ) : (
            filteredLogs.map((log) => (
              <div
                key={log.id}
                className="bg-surface-tertiary rounded px-4 py-3 text-sm border-l-4 border-surface-tertiary hover:border-holo-primary transition-colors"
              >
                <div className="flex items-start gap-3">
                  <span className="text-lg flex-shrink-0">
                    {levelIcons[log.level] || '📝'}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className={`font-semibold ${levelColors[log.level]}`}>
                        {log.level.toUpperCase()}
                      </span>
                      <span className="text-text-secondary">{log.source}</span>
                      <span className="text-text-tertiary text-xs">
                        {new Date(log.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="text-text-primary mt-1 break-words">
                      {log.message}
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
