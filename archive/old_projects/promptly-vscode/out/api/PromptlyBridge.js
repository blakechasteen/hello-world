"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.PromptlyBridge = void 0;
const axios_1 = __importDefault(require("axios"));
const child_process = __importStar(require("child_process"));
class PromptlyBridge {
    constructor() {
        this.serverProcess = null;
        this.baseURL = 'http://localhost:8765';
        this.isHealthy = false;
        this.healthCheckInterval = null;
        this.client = axios_1.default.create({
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
        this.client.interceptors.response.use((response) => response, (error) => {
            if (error.response) {
                console.error(`API Error ${error.response.status}: ${error.response.data?.detail || error.message}`);
            }
            else if (error.request) {
                console.error('API Error: No response received');
            }
            else {
                console.error(`API Error: ${error.message}`);
            }
            throw error;
        });
    }
    async start() {
        // Check if server is already running
        try {
            const response = await this.client.get('/health');
            if (response.data.promptly_available) {
                console.log('Promptly bridge server already running');
                this.isHealthy = true;
                this.startHealthCheck();
                return;
            }
        }
        catch (error) {
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
                console.error(`Bridge: ${data}`);
            });
        }
        this.serverProcess.on('exit', (code) => {
            console.log(`Bridge process exited with code ${code}`);
            this.isHealthy = false;
        });
        // Wait for server to be ready
        await this.waitForServer();
        this.startHealthCheck();
    }
    async stop() {
        // Stop health check
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
            this.healthCheckInterval = null;
        }
        // Kill server process
        if (this.serverProcess) {
            console.log('Stopping Promptly bridge server...');
            this.serverProcess.kill();
            this.serverProcess = null;
        }
        this.isHealthy = false;
    }
    startHealthCheck() {
        // Check server health every 30 seconds
        this.healthCheckInterval = setInterval(async () => {
            try {
                const response = await this.client.get('/health');
                this.isHealthy = response.data.promptly_available;
                if (!this.isHealthy) {
                    console.warn('Promptly bridge is unhealthy');
                }
            }
            catch (error) {
                console.error('Health check failed:', error);
                this.isHealthy = false;
            }
        }, 30000);
    }
    getHealthStatus() {
        return this.isHealthy;
    }
    async waitForServer(maxAttempts = 20) {
        for (let i = 0; i < maxAttempts; i++) {
            try {
                const response = await this.client.get('/health');
                if (response.data.promptly_available) {
                    console.log('Promptly bridge server ready');
                    this.isHealthy = true;
                    return;
                }
            }
            catch (error) {
                // Server not ready yet
            }
            await new Promise(resolve => setTimeout(resolve, 500));
        }
        throw new Error('Failed to start Promptly bridge server after 10 seconds');
    }
    async listPrompts() {
        if (!this.isHealthy) {
            console.warn('Bridge is not healthy, returning empty list');
            return [];
        }
        try {
            const response = await this.client.get('/prompts');
            if (response.status === 200 && response.data.prompts) {
                return response.data.prompts;
            }
            return [];
        }
        catch (error) {
            if (error.response?.status === 503) {
                console.error('Promptly core not available');
            }
            else {
                console.error('Failed to list prompts:', error.message);
            }
            return [];
        }
    }
    async getPrompt(name) {
        if (!this.isHealthy) {
            console.warn('Bridge is not healthy, cannot get prompt');
            return null;
        }
        try {
            const response = await this.client.get(`/prompts/${encodeURIComponent(name)}`);
            if (response.status === 200) {
                return response.data;
            }
            if (response.status === 404) {
                console.warn(`Prompt not found: ${name}`);
            }
            return null;
        }
        catch (error) {
            if (error.response?.status === 404) {
                console.warn(`Prompt not found: ${name}`);
            }
            else if (error.response?.status === 503) {
                console.error('Promptly core not available');
            }
            else {
                console.error(`Failed to get prompt ${name}:`, error.message);
            }
            return null;
        }
    }
}
exports.PromptlyBridge = PromptlyBridge;
//# sourceMappingURL=PromptlyBridge.js.map