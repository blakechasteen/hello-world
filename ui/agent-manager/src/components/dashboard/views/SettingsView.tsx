import { useState } from 'react'
import { useAutoRefresh, useRefreshInterval, useAppStore } from '@stores/appStore'

/**
 * Settings View
 *
 * Configurable options for:
 * - Auto-refresh interval
 * - Display preferences
 * - Data retention
 * - Backend connection settings
 */
export default function SettingsView() {
  const autoRefresh = useAutoRefresh()
  const refreshInterval = useRefreshInterval()
  const { setAutoRefresh, setRefreshInterval } = useAppStore()

  const [settings, setSettings] = useState({
    autoRefresh,
    refreshInterval,
    theme: 'dark' as const,
    compactMode: false,
    showNotifications: true,
    logRetention: 7, // days
  })

  const handleSave = () => {
    setAutoRefresh(settings.autoRefresh)
    setRefreshInterval(settings.refreshInterval)
    // Additional settings would be saved here
  }

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h1 className="text-3xl font-bold text-text-primary">Settings</h1>
        <p className="text-text-secondary mt-1">
          Configure dashboard behavior and preferences
        </p>
      </div>

      {/* General Settings */}
      <div className="card p-6 space-y-6">
        <h2 className="text-xl font-semibold text-text-primary">General</h2>

        {/* Auto Refresh */}
        <div className="space-y-2">
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.autoRefresh}
              onChange={(e) =>
                setSettings({ ...settings, autoRefresh: e.target.checked })
              }
              className="w-4 h-4 accent-holo-primary"
            />
            <span className="text-text-primary font-medium">
              Auto-Refresh Data
            </span>
          </label>
          <p className="text-xs text-text-tertiary ml-7">
            Automatically update metrics and logs
          </p>
        </div>

        {/* Refresh Interval */}
        {settings.autoRefresh && (
          <div className="space-y-2 pl-7 border-l border-dark-light">
            <label className="text-sm font-medium text-text-secondary">
              Refresh Interval
            </label>
            <select
              value={settings.refreshInterval}
              onChange={(e) =>
                setSettings({
                  ...settings,
                  refreshInterval: parseInt(e.target.value),
                })
              }
              className="w-full bg-surface-tertiary border border-dark-light rounded px-3 py-2 text-text-primary focus:outline-none focus:ring-2 focus:ring-holo-primary"
            >
              <option value={1000}>1 second</option>
              <option value={5000}>5 seconds</option>
              <option value={10000}>10 seconds</option>
              <option value={30000}>30 seconds</option>
              <option value={60000}>1 minute</option>
            </select>
          </div>
        )}

        {/* Compact Mode */}
        <div className="space-y-2">
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.compactMode}
              onChange={(e) =>
                setSettings({ ...settings, compactMode: e.target.checked })
              }
              className="w-4 h-4 accent-holo-primary"
            />
            <span className="text-text-primary font-medium">
              Compact Mode
            </span>
          </label>
          <p className="text-xs text-text-tertiary ml-7">
            Reduce padding and spacing in tables and lists
          </p>
        </div>

        {/* Notifications */}
        <div className="space-y-2">
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.showNotifications}
              onChange={(e) =>
                setSettings({
                  ...settings,
                  showNotifications: e.target.checked,
                })
              }
              className="w-4 h-4 accent-holo-primary"
            />
            <span className="text-text-primary font-medium">
              Show Notifications
            </span>
          </label>
          <p className="text-xs text-text-tertiary ml-7">
            Display alerts for errors and important events
          </p>
        </div>

        {/* Log Retention */}
        <div className="space-y-2 border-t border-dark-light pt-6">
          <label className="text-sm font-medium text-text-secondary">
            Log Retention (days)
          </label>
          <input
            type="number"
            min="1"
            max="90"
            value={settings.logRetention}
            onChange={(e) =>
              setSettings({
                ...settings,
                logRetention: parseInt(e.target.value),
              })
            }
            className="w-full bg-surface-tertiary border border-dark-light rounded px-3 py-2 text-text-primary focus:outline-none focus:ring-2 focus:ring-holo-primary"
          />
          <p className="text-xs text-text-tertiary">
            Logs older than this will be archived
          </p>
        </div>
      </div>

      {/* Backend Settings */}
      <div className="card p-6 space-y-6">
        <h2 className="text-xl font-semibold text-text-primary">Backend</h2>

        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium text-text-secondary block mb-1">
              Backend URL
            </label>
            <input
              type="text"
              value="http://localhost:8000"
              disabled
              className="w-full bg-surface-tertiary border border-dark-light rounded px-3 py-2 text-text-secondary cursor-not-allowed opacity-75"
            />
            <p className="text-xs text-text-tertiary mt-1">
              Currently configured in vite.config.ts
            </p>
          </div>

          <div>
            <label className="text-sm font-medium text-text-secondary block mb-1">
              Agent Manager URL
            </label>
            <input
              type="text"
              value="http://localhost:8002"
              disabled
              className="w-full bg-surface-tertiary border border-dark-light rounded px-3 py-2 text-text-secondary cursor-not-allowed opacity-75"
            />
            <p className="text-xs text-text-tertiary mt-1">
              Currently configured in vite.config.ts
            </p>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-4">
        <button
          onClick={handleSave}
          className="px-6 py-2 bg-holo-primary text-white rounded-lg font-medium hover:bg-holo-primary-light transition-colors shadow-glow-primary"
        >
          Save Settings
        </button>
        <button
          onClick={() => {
            /* Reset to defaults */
          }}
          className="px-6 py-2 bg-surface-tertiary text-text-primary rounded-lg font-medium hover:bg-dark-light transition-colors"
        >
          Reset to Defaults
        </button>
      </div>

      {/* Info */}
      <div className="card p-4 bg-surface-tertiary text-sm text-text-secondary">
        <p>
          💡 Settings are stored in browser localStorage and will persist across sessions.
        </p>
      </div>
    </div>
  )
}
