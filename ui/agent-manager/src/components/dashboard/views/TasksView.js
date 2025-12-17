import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useTasks } from '@stores/appStore';
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
    const tasks = useTasks();
    const pending = tasks.filter((t) => t.status === 'pending').length;
    const running = tasks.filter((t) => t.status === 'running').length;
    const completed = tasks.filter((t) => t.status === 'completed').length;
    const failed = tasks.filter((t) => t.status === 'failed').length;
    return (_jsxs("div", { className: "space-y-6", children: [_jsxs("div", { children: [_jsx("h1", { className: "text-3xl font-bold text-text-primary", children: "Tasks" }), _jsx("p", { className: "text-text-secondary mt-1", children: "Monitor task execution and queue status" })] }), _jsxs("div", { className: "grid grid-cols-2 md:grid-cols-4 gap-4", children: [_jsxs("div", { className: "card p-4 text-center", children: [_jsx("div", { className: "text-2xl font-bold text-amber-400", children: pending }), _jsx("div", { className: "text-xs text-text-secondary mt-1", children: "Pending" })] }), _jsxs("div", { className: "card p-4 text-center", children: [_jsx("div", { className: "text-2xl font-bold text-cyan-400", children: running }), _jsx("div", { className: "text-xs text-text-secondary mt-1", children: "Running" })] }), _jsxs("div", { className: "card p-4 text-center", children: [_jsx("div", { className: "text-2xl font-bold text-emerald-400", children: completed }), _jsx("div", { className: "text-xs text-text-secondary mt-1", children: "Completed" })] }), _jsxs("div", { className: "card p-4 text-center", children: [_jsx("div", { className: "text-2xl font-bold text-red-400", children: failed }), _jsx("div", { className: "text-xs text-text-secondary mt-1", children: "Failed" })] })] }), _jsxs("div", { className: "card p-6", children: [_jsxs("h2", { className: "text-xl font-semibold text-text-primary mb-4", children: ["Task Queue (", tasks.length, ")"] }), tasks.length === 0 ? (_jsxs("div", { className: "text-center py-12 text-text-secondary", children: [_jsx("p", { className: "text-lg mb-2", children: "No tasks in queue" }), _jsx("p", { className: "text-sm", children: "Tasks will appear here when agents process queries" })] })) : (_jsx("div", { className: "space-y-3", children: tasks.map((task) => (_jsxs("div", { className: "card-interactive p-4 flex items-center justify-between", children: [_jsxs("div", { className: "flex items-center gap-4 flex-1 min-w-0", children: [_jsx("span", { className: "text-xl flex-shrink-0", children: task.status === 'pending'
                                                ? '⏳'
                                                : task.status === 'running'
                                                    ? '⚡'
                                                    : task.status === 'completed'
                                                        ? '✅'
                                                        : '❌' }), _jsxs("div", { className: "min-w-0 flex-1", children: [_jsx("div", { className: "font-semibold text-text-primary truncate", children: task.query }), _jsx("div", { className: "text-xs text-text-tertiary", children: task.agentId })] })] }), _jsxs("div", { className: "flex items-center gap-4 flex-shrink-0", children: [_jsxs("div", { className: "text-right text-sm", children: [_jsx("div", { className: "text-text-primary font-semibold", children: task.endTime
                                                        ? ((task.endTime - task.startTime) / 1000).toFixed(2)
                                                        : '-' }), _jsx("div", { className: "text-xs text-text-tertiary", children: "seconds" })] }), _jsx("span", { className: `px-2.5 py-0.5 rounded text-xs font-medium ${task.status === 'pending'
                                                ? 'bg-amber-950 text-amber-200'
                                                : task.status === 'running'
                                                    ? 'bg-cyan-950 text-cyan-200'
                                                    : task.status === 'completed'
                                                        ? 'bg-emerald-950 text-emerald-200'
                                                        : 'bg-red-950 text-red-200'}`, children: task.status.charAt(0).toUpperCase() + task.status.slice(1) })] })] }, task.id))) }))] })] }));
}
//# sourceMappingURL=TasksView.js.map