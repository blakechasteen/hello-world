import axios, { AxiosInstance } from 'axios';

export interface SkillExecutionRequest {
    skill_name: string;
    user_input: string;
    backend?: string;
    model?: string;
}

export interface ChainExecutionRequest {
    skill_names: string[];
    initial_input: string;
    backend?: string;
    model?: string;
}

export interface LoopExecutionRequest {
    skill_name: string;
    user_input: string;
    loop_type?: 'refine' | 'critique' | 'decompose' | 'verify' | 'explore' | 'hofstadter';
    max_iterations?: number;
    quality_threshold?: number;
    backend?: string;
    model?: string;
}

export interface ExecutionResponse {
    execution_id: string;
    status: string;
    message: string;
}

export interface ExecutionStatus {
    execution_id: string;
    status: 'queued' | 'running' | 'completed' | 'failed';
    progress: number;  // 0.0 to 1.0
    current_step: string;
    output?: string;
    error?: string;
    metadata?: {
        iterations?: number;
        improvement_history?: number[];
        results?: Array<{ skill: string; output: string }>;
    };
}

export interface ExecutionEvent {
    execution_id: string;
    event: 'status_update' | 'iteration_update' | 'completed' | 'failed';
    progress?: number;
    step?: string;
    iteration?: number;
    quality?: number;
    output?: string;
    error?: string;
    results?: any[];
    improvement_history?: number[];
    stop_reason?: string;
}

export type ExecutionEventHandler = (event: ExecutionEvent) => void;

export class ExecutionClient {
    private client: AxiosInstance;
    private readonly baseURL = 'http://localhost:8765';
    private websocket: WebSocket | null = null;
    private eventHandlers: Map<string, ExecutionEventHandler[]> = new Map();
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;

    constructor() {
        this.client = axios.create({
            baseURL: this.baseURL,
            timeout: 120000,  // 2 minute timeout for long-running operations
            validateStatus: (status) => status < 500
        });

        // Add request interceptor
        this.client.interceptors.request.use((config) => {
            console.log(`Execution API Request: ${config.method?.toUpperCase()} ${config.url}`);
            return config;
        });

        // Add response interceptor
        this.client.interceptors.response.use(
            (response) => response,
            (error) => {
                console.error(`Execution API Error:`, error.message);
                throw error;
            }
        );
    }

    /**
     * Execute a single skill
     */
    async executeSkill(request: SkillExecutionRequest): Promise<ExecutionResponse> {
        try {
            const response = await this.client.post('/execute/skill', {
                ...request,
                backend: request.backend || 'ollama',
                model: request.model || 'llama3.2:3b'
            });
            return response.data;
        } catch (error: any) {
            throw new Error(`Failed to execute skill: ${error.message}`);
        }
    }

    /**
     * Execute a chain of skills
     */
    async executeChain(request: ChainExecutionRequest): Promise<ExecutionResponse> {
        try {
            const response = await this.client.post('/execute/chain', {
                ...request,
                backend: request.backend || 'ollama',
                model: request.model || 'llama3.2:3b'
            });
            return response.data;
        } catch (error: any) {
            throw new Error(`Failed to execute chain: ${error.message}`);
        }
    }

    /**
     * Execute a recursive loop
     */
    async executeLoop(request: LoopExecutionRequest): Promise<ExecutionResponse> {
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
        } catch (error: any) {
            throw new Error(`Failed to execute loop: ${error.message}`);
        }
    }

    /**
     * Get execution status
     */
    async getStatus(executionId: string): Promise<ExecutionStatus> {
        try {
            const response = await this.client.get(`/execute/status/${executionId}`);
            return response.data;
        } catch (error: any) {
            if (error.response?.status === 404) {
                throw new Error('Execution not found');
            }
            throw new Error(`Failed to get status: ${error.message}`);
        }
    }

    /**
     * Poll for execution status until complete
     */
    async pollUntilComplete(
        executionId: string,
        onProgress?: (status: ExecutionStatus) => void,
        intervalMs: number = 1000
    ): Promise<ExecutionStatus> {
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
                    return;  // Ignore pong responses
                }

                const executionEvent: ExecutionEvent = data;
                this.handleEvent(executionEvent);
            } catch (error) {
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
    onExecutionEvent(executionId: string, handler: ExecutionEventHandler) {
        if (!this.eventHandlers.has(executionId)) {
            this.eventHandlers.set(executionId, []);
        }
        this.eventHandlers.get(executionId)!.push(handler);
    }

    /**
     * Remove event handler
     */
    removeExecutionHandler(executionId: string, handler: ExecutionEventHandler) {
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
    clearExecutionHandlers(executionId: string) {
        this.eventHandlers.delete(executionId);
    }

    /**
     * Handle incoming event from WebSocket
     */
    private handleEvent(event: ExecutionEvent) {
        const handlers = this.eventHandlers.get(event.execution_id);
        if (handlers) {
            handlers.forEach(handler => {
                try {
                    handler(event);
                } catch (error) {
                    console.error('Error in execution event handler:', error);
                }
            });
        }
    }

    /**
     * Get WebSocket connection status
     */
    isWebSocketConnected(): boolean {
        return this.websocket?.readyState === WebSocket.OPEN;
    }
}
