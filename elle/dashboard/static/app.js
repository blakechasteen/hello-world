/**
 * Elle Core Dashboard - Frontend JavaScript
 *
 * Features:
 * - API client for all endpoints
 * - Real-time data updates
 * - Chart rendering utilities
 * - Sparkline generation
 * - Notification system
 *
 * Created: 2025-11-15
 * Author: Blake Chasteen (Agent C)
 */

// ============================================================================
// API CLIENT
// ============================================================================

const API = {
    baseUrl: window.location.origin,

    // Helper for GET requests
    async get(endpoint) {
        const response = await fetch(`${this.baseUrl}${endpoint}`);
        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }
        return await response.json();
    },

    // Helper for POST requests
    async post(endpoint, data = {}) {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }
        return await response.json();
    },

    // Budget endpoints
    budgets: {
        list: () => API.get('/api/budgets'),
        get: (id) => API.get(`/api/budgets/${id}`),
        variance: (id) => API.get(`/api/budgets/${id}/variance`),
        forecast: (id, weeks = 12) => API.get(`/api/budgets/${id}/forecast?weeks=${weeks}`),
    },

    // SOP endpoints
    sops: {
        list: () => API.get('/api/sops'),
        get: (id) => API.get(`/api/sops/${id}`),
    },

    // Task endpoints
    tasks: {
        list: (limit = 50) => API.get(`/api/tasks?limit=${limit}`),
        active: () => API.get('/api/tasks/active'),
        start: (taskName, sopId, category) =>
            API.post('/api/tasks/start', { task_name: taskName, sop_id: sopId, category }),
        end: (unitsProduced, qualityScore, notes) =>
            API.post('/api/tasks/end', {
                units_produced: unitsProduced,
                quality_score: qualityScore,
                notes,
            }),
        pause: (id) => API.post(`/api/tasks/${id}/pause`),
        resume: (id) => API.post(`/api/tasks/${id}/resume`),
    },

    // Analytics endpoints
    analytics: () => API.get('/api/analytics'),
    recommendations: () => API.get('/api/recommendations'),
    weeklyPlan: () => API.get('/api/recommendations/weekly'),
};

// ============================================================================
// FORMATTING UTILITIES
// ============================================================================

const Format = {
    // Currency formatting
    currency(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
        }).format(value);
    },

    // Percentage formatting
    percent(value) {
        return `${value.toFixed(1)}%`;
    },

    // Number formatting
    number(value, decimals = 0) {
        return value.toFixed(decimals);
    },

    // Duration formatting (hours)
    duration(hours) {
        if (hours < 1) {
            return `${Math.round(hours * 60)}m`;
        }
        return `${hours.toFixed(1)}h`;
    },

    // Date formatting
    date(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric',
        });
    },

    // Time formatting
    time(dateString) {
        const date = new Date(dateString);
        return date.toLocaleTimeString('en-US', {
            hour: 'numeric',
            minute: '2-digit',
        });
    },

    // Relative time
    relative(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diff = now - date;
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);

        if (minutes < 1) return 'just now';
        if (minutes < 60) return `${minutes}m ago`;
        if (hours < 24) return `${hours}h ago`;
        return `${days}d ago`;
    },
};

// ============================================================================
// SPARKLINE GENERATION
// ============================================================================

function createSparkline(values, width = 60, height = 20) {
    if (!values || values.length === 0) return '';

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    const points = values
        .map((v, i) => {
            const x = (i / (values.length - 1)) * width;
            const y = height - ((v - min) / range) * height;
            return `${x},${y}`;
        })
        .join(' ');

    return `
        <svg class="sparkline" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
            <polyline
                points="${points}"
                fill="none"
                stroke="#4682b4"
                stroke-width="1.5"
            />
        </svg>
    `;
}

// ============================================================================
// NOTIFICATION SYSTEM
// ============================================================================

const Notifications = {
    show(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 16px 24px;
            background: ${type === 'error' ? '#d32f2f' : '#2d882d'};
            color: white;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    },

    error(message) {
        this.show(message, 'error');
    },

    success(message) {
        this.show(message, 'success');
    },
};

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// ============================================================================
// ACTIVE NAVIGATION
// ============================================================================

function setActiveNav() {
    const path = window.location.pathname;
    const navLinks = document.querySelectorAll('nav a');

    navLinks.forEach((link) => {
        const href = link.getAttribute('href');
        if (path === href || (path === '/' && href === '/')) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

// ============================================================================
// AUTO-REFRESH
// ============================================================================

class AutoRefresh {
    constructor(callback, interval = 5000) {
        this.callback = callback;
        this.interval = interval;
        this.timerId = null;
    }

    start() {
        this.stop(); // Clear any existing timer
        this.callback(); // Initial call
        this.timerId = setInterval(this.callback, this.interval);
    }

    stop() {
        if (this.timerId) {
            clearInterval(this.timerId);
            this.timerId = null;
        }
    }
}

// ============================================================================
// CHART HELPERS (for custom Tufte charts)
// ============================================================================

const Chart = {
    // Create a simple bar chart (Tufte style)
    bar(data, options = {}) {
        const {
            width = 300,
            height = 200,
            color = '#4682b4',
            labelKey = 'label',
            valueKey = 'value',
        } = options;

        const max = Math.max(...data.map((d) => d[valueKey]));
        const barWidth = width / data.length - 4;

        const bars = data
            .map((d, i) => {
                const barHeight = (d[valueKey] / max) * (height - 40);
                const x = i * (barWidth + 4);
                const y = height - barHeight - 20;
                return `
                    <rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" fill="${color}" />
                    <text x="${x + barWidth / 2}" y="${height - 5}" text-anchor="middle" font-size="10">${d[labelKey]}</text>
                `;
            })
            .join('');

        return `
            <svg width="${width}" height="${height}">
                ${bars}
            </svg>
        `;
    },

    // Create a line chart (Tufte style)
    line(data, options = {}) {
        const { width = 300, height = 200, color = '#4682b4' } = options;

        if (!data || data.length === 0) return '';

        const min = Math.min(...data);
        const max = Math.max(...data);
        const range = max - min || 1;

        const points = data
            .map((v, i) => {
                const x = (i / (data.length - 1)) * width;
                const y = height - ((v - min) / range) * (height - 40) - 20;
                return `${x},${y}`;
            })
            .join(' ');

        return `
            <svg width="${width}" height="${height}">
                <polyline points="${points}" fill="none" stroke="${color}" stroke-width="2" />
            </svg>
        `;
    },
};

// ============================================================================
// ERROR HANDLING
// ============================================================================

async function handleApiCall(apiCall, errorMessage = 'An error occurred') {
    try {
        return await apiCall();
    } catch (error) {
        console.error(error);
        Notifications.error(errorMessage);
        throw error;
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    setActiveNav();
    console.log('Elle Core Dashboard initialized');
});

// ============================================================================
// EXPORTS
// ============================================================================

window.API = API;
window.Format = Format;
window.createSparkline = createSparkline;
window.Notifications = Notifications;
window.AutoRefresh = AutoRefresh;
window.Chart = Chart;
window.handleApiCall = handleApiCall;
