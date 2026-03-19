"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ExecutionClient = void 0;
const axios_1 = __importDefault(require("axios"));
class ExecutionClient {
    constructor() {
        this.baseURL = 'http://localhost:8765';
        this.websocket = null;
        this.eventHandlers = new Map();
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.client = axios_1.default.create({
            baseURL: this.baseURL,
            timeout: 120000, // 2 minute timeout for long-running operations
            validateStatus: (status) => status < 500
        });
        // Add request interceptor
        this.client.interceptors.request.use((config) => {
            console.log(`Execution API Request: ${config.method?.toUpperCase()} ${config.url}`);
            return config;
        });
        // Add response interceptor
        this.client.interceptors.response.use((response) => response, (error) => {
            console.error(`Execution API Error:`, error.message);
            throw error;
        });
    }
    /**
     * Execute a single skill
     */
    async executeSkill(request) {
        try {
            const response = await this.client.post('/execute/skill', {
                ...request,
                backend: request.backend || 'ollama',
                model: request.model || 'llama3.2:3b'
            });
            return response.data;
        }
        catch (error) {
            throw new Error(`Failed to execute skill: ${error.message}`);
        }
    }
    /**
     * Execute a chain of skills
     */
    async executeChain(request) {
        try {
            const response = await this.client.post('/execute/chain', {
                ...request,
                backend: request.backend || 'ollama',
                model: request.model || 'llama3.2:3b'
            });
            return response.data;
        }
        catch (error) {
            throw new Error(`Failed to execute chain: ${error.message}`);
        }
    }
    /**
     * Execute a recursive loop
     */
    async executeLoop(request) {
        try {
            const response = await this.client.post('/execute/loop', {
                ...request,
                loop_type: request.loop_type || 'refine',
                max_iterations: request.max_iterations || 5,
                quality_threshold: request.quality_threshold || 0.9,
                backend: request.backend || 'ollama',
                model: request.model || 'llama3.2:3b'
            });
            return response.data;
        }
        catch (error) {
            throw new Error(`Failed to execute loop: ${error.message}`);
        }
    }
    /**
     * Get execution status
     */
    async getStatus(executionId) {
        try {
            const response = await this.client.get(`/execute/status/${executionId}`);
            return response.data;
        }
        catch (error) {
            if (error.response?.status === 404) {
                throw new Error('Execution not found');
            }
            throw new Error(`Failed to get status: ${error.message}`);
        }
    }
    /**
     * Poll for execution status until complete
     */
    async pollUntilComplete(executionId, onProgress, intervalMs = 1000) {
        while (true) {
            const status = await this.getStatus(executionId);
            if (onProgress) {
                onProgress(status);
            }
            if (status.status === 'completed' || status.status === 'failed') {
                return status;
            }
            await new Promise(resolve => setTimeout(resolve, intervalMs));
        }
    }
    /**
     * Connect to WebSocket for real-time updates
     */
    connectWebSocket() {
        if (this.websocket?.readyState === WebSocket.OPEN) {
            console.log('WebSocket already connected');
            return;
        }
        const wsUrl = this.baseURL.replace('http://', 'ws://') + '/ws/execution';
        console.log(`Connecting to WebSocket: ${wsUrl}`);
        this.websocket = new WebSocket(wsUrl);
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;
            // Start ping interval
            setInterval(() => {
                if (this.websocket?.readyState === WebSocket.OPEN) {
                    this.websocket.send('ping');
                }
            }, 30000);
        };
        this.websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'pong') {
                    return; // Ignore pong responses
                }
                const executionEvent = data;
                this.handleEvent(executionEvent);
            }
            catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
            this.websocket = null;
            // Attempt to reconnect
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
                console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                setTimeout(() => this.connectWebSocket(), delay);
            }
        };
    }
    /**
     * Disconnect WebSocket
     */
    disconnectWebSocket() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
    }
    /**
     * Register event handler for specific execution
     */
    onExecutionEvent(executionId, handler) {
        if (!this.eventHandlers.has(executionId)) {
            this.eventHandlers.set(executionId, []);
        }
        this.eventHandlers.get(executionId).push(handler);
    }
    /**
     * Remove event handler
     */
    removeExecutionHandler(executionId, handler) {
        const handlers = this.eventHandlers.get(executionId);
        if (handlers) {
            const index = handlers.indexOf(handler);
            if (index > -1) {
                handlers.splice(index, 1);
            }
        }
    }
    /**
     * Clear all handlers for an execution
     */
    clearExecutionHandlers(executionId) {
        this.eventHandlers.delete(executionId);
    }
    /**
     * Handle incoming event from WebSocket
     */
    handleEvent(event) {
        const handlers = this.eventHandlers.get(event.execution_id);
        if (handlers) {
            handlers.forEach(handler => {
                try {
                    handler(event);
                }
                catch (error) {
                    console.error('Error in execution event handler:', error);
                }
            });
        }
    }
    /**
     * Get WebSocket connection status
     */
    isWebSocketConnected() {
        return this.websocket?.readyState === WebSocket.OPEN;
    }
}
exports.ExecutionClient = ExecutionClient;
//# sourceMappingURL=ExecutionClient.js.map