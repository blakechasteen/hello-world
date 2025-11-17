// Tool Builder Editor - Main JavaScript

// Global utilities and helpers

function formatDate(isoString) {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    return date.toLocaleString();
}

function validateToolName(name) {
    const pattern = /^[a-zA-Z][a-zA-Z0-9_-]*$/;
    return pattern.test(name);
}

function generateToolId(name) {
    return `tool_${name.toLowerCase().replace(/[^a-z0-9]/g, '_')}_${Date.now()}`;
}

// API client helpers
const api = {
    async get(url) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.json();
    },

    async post(url, data) {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Request failed');
        }
        return response.json();
    },

    async put(url, data) {
        const response = await fetch(url, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Request failed');
        }
        return response.json();
    },

    async delete(url) {
        const response = await fetch(url, { method: 'DELETE' });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Request failed');
        }
        return response.json();
    }
};

// Export for use in components
window.toolBuilderUtils = {
    formatDate,
    validateToolName,
    generateToolId,
    api
};

console.log('Tool Builder Editor initialized');
