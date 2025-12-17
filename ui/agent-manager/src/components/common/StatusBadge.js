import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export const StatusBadge = ({ status, size = 'md', showLabel = true, className = '', }) => {
    // Size classes
    const sizeClasses = {
        sm: 'px-2 py-1 text-xs',
        md: 'px-2.5 py-1.5 text-sm',
        lg: 'px-3 py-2 text-base',
    };
    // Status configuration: color, icon, label, animation
    const statusConfig = {
        idle: {
            bgColor: 'bg-slate-700',
            textColor: 'text-slate-200',
            icon: '○',
            label: 'Idle',
            animate: false,
        },
        running: {
            bgColor: 'bg-blue-600',
            textColor: 'text-white',
            icon: '▶',
            label: 'Running',
            animate: true,
        },
        paused: {
            bgColor: 'bg-amber-600',
            textColor: 'text-white',
            icon: '⏸',
            label: 'Paused',
            animate: false,
        },
        completed: {
            bgColor: 'bg-emerald-600',
            textColor: 'text-white',
            icon: '✓',
            label: 'Completed',
            animate: false,
        },
        failed: {
            bgColor: 'bg-red-600',
            textColor: 'text-white',
            icon: '✕',
            label: 'Failed',
            animate: false,
        },
        cancelled: {
            bgColor: 'bg-slate-700',
            textColor: 'text-slate-300 line-through',
            icon: '×',
            label: 'Cancelled',
            animate: false,
        },
    };
    const config = statusConfig[status];
    return (_jsxs("span", { className: `
        inline-flex items-center gap-1.5 rounded-full font-semibold
        ${sizeClasses[size]}
        ${config.bgColor}
        ${config.textColor}
        ${config.animate ? 'animate-pulse' : ''}
        ${className}
      `, children: [_jsx("span", { className: `inline-block ${config.animate ? 'animate-bounce' : ''}`, children: config.icon }), showLabel && _jsx("span", { children: config.label })] }));
};
export const StatusIndicator = ({ status, size = 'md', className = '', }) => {
    const sizeClasses = {
        sm: 'w-2 h-2',
        md: 'w-2.5 h-2.5',
        lg: 'w-3 h-3',
    };
    const colorClasses = {
        idle: 'bg-slate-500',
        running: 'bg-blue-500 animate-pulse',
        paused: 'bg-amber-500',
        completed: 'bg-emerald-500',
        failed: 'bg-red-500',
        cancelled: 'bg-slate-400',
    };
    return (_jsx("div", { className: `
        inline-flex rounded-full
        ${sizeClasses[size]}
        ${colorClasses[status]}
        ${className}
      `, title: status }));
};
export const StatusGrid = ({ idle = 0, running = 0, paused = 0, completed = 0, failed = 0, cancelled = 0, className = '', }) => {
    const statuses = [
        { key: 'idle', value: idle, config: { bgColor: 'bg-slate-700', label: 'Idle' } },
        { key: 'running', value: running, config: { bgColor: 'bg-blue-600', label: 'Running' } },
        { key: 'paused', value: paused, config: { bgColor: 'bg-amber-600', label: 'Paused' } },
        { key: 'completed', value: completed, config: { bgColor: 'bg-emerald-600', label: 'Completed' } },
        { key: 'failed', value: failed, config: { bgColor: 'bg-red-600', label: 'Failed' } },
        { key: 'cancelled', value: cancelled, config: { bgColor: 'bg-slate-700', label: 'Cancelled' } },
    ];
    return (_jsx("div", { className: `grid grid-cols-3 gap-2 ${className}`, children: statuses.map((status) => status.value > 0 && (_jsxs("div", { className: `${status.config.bgColor} rounded-lg px-3 py-2 text-center text-white`, children: [_jsx("div", { className: "text-lg font-bold", children: status.value }), _jsx("div", { className: "text-xs opacity-90", children: status.config.label })] }, status.key))) }));
};
export default StatusBadge;
//# sourceMappingURL=StatusBadge.js.map