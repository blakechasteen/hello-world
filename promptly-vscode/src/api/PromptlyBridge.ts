import axios, { AxiosInstance } from 'axios';
import * as child_process from 'child_process';

interface PromptMetadata {
    name: string;
    branch: string;
    tags: string[];
    created: string;
}

interface PromptData {
    content: string;
    metadata: PromptMetadata;
}

export class PromptlyBridge {
    private client: AxiosInstance;
    private serverProcess: child_process.ChildProcess | null = null;
    private readonly baseURL = 'http://localhost:8765';
    private isHealthy: boolean = false;
    private healthCheckInterval: NodeJS.Timeout | null = null;

    constructor() {
        this.client = axios.create({
            baseURL: this.baseURL,
            timeout: 5000,
            maxRedirects: 5,
            validateStatus: (status) => status < 500 // Don't throw on 4xx errors
        });

        // Add request interceptor for logging
        this.client.interceptors.request.use((config) => {
            console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
            return config;
        });

        // Add response interceptor for error handling
        this.client.interceptors.response.use(
            (response) => response,
            (error) => {
                if (error.response) {
                    console.error(`API Error ${error.response.status}: ${error.response.data?.detail || error.message}`);
                } else if (error.request) {
                    console.error('API Error: No response received');
                } else {
                    console.error(`API Error: ${error.message}`);
                }
                throw error;
            }
        );
    }

    async start(): Promise<void> {
        // Check if server is already running
        try {
            await this.client.get('/health');
            console.log('Promptly bridge server already running');
            return;
        } catch (error) {
            // Server not running, start it
            console.log('Starting Promptly bridge server...');
        }

        // Start Python FastAPI server
        // Note: In production, we'd find the Python interpreter from venv
        // For prototype, assuming 'python' is in PATH
        const pythonScript = 'Promptly/promptly/vscode_bridge.py';

        this.serverProcess = child_process.spawn('python', [pythonScript], {
            cwd: process.cwd(),
            stdio: 'pipe'
        });

        if (this.serverProcess.stdout) {
            this.serverProcess.stdout.on('data', (data) => {
                console.log(`Bridge: ${data}`);
            });
        }

        if (this.serverProcess.stderr) {
            this.serverProcess.stderr.on('data', (data) => {
                console.error(`Bridge error: ${data}`);
            });
        }

        // Wait for server to be ready
        await this.waitForServer();
    }

    async stop(): Promise<void> {
        if (this.serverProcess) {
            this.serverProcess.kill();
            this.serverProcess = null;
        }
    }

    private async waitForServer(maxAttempts: number = 10): Promise<void> {
        for (let i = 0; i < maxAttempts; i++) {
            try {
                await this.client.get('/health');
                console.log('Promptly bridge server ready');
                return;
            } catch (error) {
                await new Promise(resolve => setTimeout(resolve, 500));
            }
        }
        throw new Error('Failed to start Promptly bridge server');
    }

    async listPrompts(): Promise<PromptMetadata[]> {
        try {
            const response = await this.client.get('/prompts');
            return response.data.prompts || [];
        } catch (error) {
            console.error('Failed to list prompts:', error);
            return [];
        }
    }

    async getPrompt(name: string): Promise<PromptData | null> {
        try {
            const response = await this.client.get(`/prompts/${encodeURIComponent(name)}`);
            return response.data;
        } catch (error) {
            console.error(`Failed to get prompt ${name}:`, error);
            return null;
        }
    }
}