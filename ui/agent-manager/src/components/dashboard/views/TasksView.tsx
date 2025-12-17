import { useTasks } from '@stores/appStore'

/**
 * Tasks View
 *
 * Displays:
 * - Task queue (pending, running, completed)
 * - Task execution timeline
 * - Task filtering and search
 * - Per-task details and logs
 */
export default function TasksView() {
  const tasks = useTasks()

  const pending = tasks.filter((t) => t.status === 'pending').length
  const running = tasks.filter((t) => t.status === 'running').length
  const completed = tasks.filter((t) => t.status === 'completed').length
  const failed = tasks.filter((t) => t.status === 'failed').length

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-text-primary">Tasks</h1>
        <p className="text-text-secondary mt-1">
          Monitor task execution and queue status
        </p>
      </div>

      {/* Task Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card p-4 text-center">
          <div className="text-2xl font-bold text-amber-400">{pending}</div>
          <div className="text-xs text-text-secondary mt-1">Pending</div>
        </div>
        <div className="card p-4 text-center">
          <div className="text-2xl font-bold text-cyan-400">{running}</div>
          <div className="text-xs text-text-secondary mt-1">Running</div>
        </div>
        <div className="card p-4 text-center">
          <div className="text-2xl font-bold text-emerald-400">{completed}</div>
          <div className="text-xs text-text-secondary mt-1">Completed</div>
        </div>
        <div className="card p-4 text-center">
          <div className="text-2xl font-bold text-red-400">{failed}</div>
          <div className="text-xs text-text-secondary mt-1">Failed</div>
        </div>
      </div>

      <div className="card p-6">
        <h2 className="text-xl font-semibold text-text-primary mb-4">
          Task Queue ({tasks.length})
        </h2>

        {tasks.length === 0 ? (
          <div className="text-center py-12 text-text-secondary">
            <p className="text-lg mb-2">No tasks in queue</p>
            <p className="text-sm">Tasks will appear here when agents process queries</p>
          </div>
        ) : (
          <div className="space-y-3">
            {tasks.map((task) => (
              <div
                key={task.id}
                className="card-interactive p-4 flex items-center justify-between"
              >
                <div className="flex items-center gap-4 flex-1 min-w-0">
                  <span className="text-xl flex-shrink-0">
                    {task.status === 'pending'
                      ? '⏳'
                      : task.status === 'running'
                        ? '⚡'
                        : task.status === 'completed'
                          ? '✅'
                          : '❌'}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="font-semibold text-text-primary truncate">
                      {task.query}
                    </div>
                    <div className="text-xs text-text-tertiary">
                      {task.agentId}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-4 flex-shrink-0">
                  <div className="text-right text-sm">
                    <div className="text-text-primary font-semibold">
                      {task.endTime
                        ? ((task.endTime - task.startTime) / 1000).toFixed(2)
                        : '-'}
                    </div>
                    <div className="text-xs text-text-tertiary">seconds</div>
                  </div>
                  <span
                    className={`px-2.5 py-0.5 rounded text-xs font-medium ${
                      task.status === 'pending'
                        ? 'bg-amber-950 text-amber-200'
                        : task.status === 'running'
                          ? 'bg-cyan-950 text-cyan-200'
                          : task.status === 'completed'
                            ? 'bg-emerald-950 text-emerald-200'
                            : 'bg-red-950 text-red-200'
                    }`}
                  >
                    {task.status.charAt(0).toUpperCase() + task.status.slice(1)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
