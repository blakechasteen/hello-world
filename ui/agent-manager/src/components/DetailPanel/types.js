/**
 * DetailPanel Type Definitions
 * HoloLoom Agent Manager UI - Phase 4
 */
/**
 * Source type display configuration
 */
export const MEMORY_SOURCE_CONFIG = {
    graph: { icon: '🔗', label: 'Graph', color: '#3B82F6' },
    vector: { icon: '📊', label: 'Vector', color: '#10B981' },
    cache: { icon: '⚡', label: 'Cache', color: '#F59E0B' },
    hot_pattern: { icon: '🔥', label: 'Hot', color: '#EF4444' },
    awareness: { icon: '👁', label: 'Awareness', color: '#8B5CF6' },
    spring: { icon: '🌊', label: 'Spring', color: '#06B6D4' },
    wave: { icon: '〰', label: 'Wave', color: '#EC4899' },
};
/**
 * File status display configuration
 */
export const FILE_STATUS_CONFIG = {
    modified: { icon: '●', label: 'Modified', color: '#F59E0B' },
    read: { icon: '○', label: 'Read', color: '#64748B' },
    created: { icon: '+', label: 'Created', color: '#10B981' },
    deleted: { icon: '-', label: 'Deleted', color: '#EF4444' },
};
/**
 * Agent color palette (up to 8 distinct agents)
 */
export const AGENT_COLORS = [
    '#3B82F6', // blue
    '#10B981', // emerald
    '#F59E0B', // amber
    '#EF4444', // red
    '#8B5CF6', // violet
    '#EC4899', // pink
    '#06B6D4', // cyan
    '#84CC16', // lime
];
/**
 * Get agent color by index (wraps around)
 */
export function getAgentColor(index) {
    return AGENT_COLORS[index % AGENT_COLORS.length];
}
/**
 * Detail panel tab configuration
 */
export const DETAIL_TABS = [
    { id: 'history', label: 'History', icon: '📜' },
    { id: 'memory', label: 'Memory', icon: '🧠' },
    { id: 'files', label: 'Files', icon: '📁' },
];
//# sourceMappingURL=types.js.map