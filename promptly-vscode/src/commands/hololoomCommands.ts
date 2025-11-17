import * as vscode from 'vscode';

interface IngestResponse {
    success: boolean;
    files_indexed?: number;
    code_elements?: number;
    comments?: number;
    todos?: number;
    entities_created?: number;
    error?: string;
    stats?: {
        files: number;
        entities: number;
        [key: string]: any;
    };
}

export class HoloLoomCommands {
    private baseUrl: string;

    constructor() {
        const config = vscode.workspace.getConfiguration('promptly');
        this.baseUrl = config.get<string>('hololoomUrl') || 'http://localhost:8000';
    }

    async remember(content: string): Promise<string> {
        try {
            const axios = require('axios');
            const response = await axios.post(`${this.baseUrl}/api/remember`, {
                content,
                context: {
                    workspace: vscode.workspace.name,
                    file: vscode.window.activeTextEditor?.document.fileName,
                    timestamp: new Date().toISOString()
                }
            });

            if (response.status === 200) {
                return `✅ **Saved to HoloLoom memory**\n\n_"${content}"_`;
            } else {
                throw new Error(`HTTP ${response.status}`);
            }

        } catch (error: any) {
            if (error.code === 'ECONNREFUSED') {
                return `❌ HoloLoom server not running at ${this.baseUrl}\n\nStart it with:\n\`\`\`\ncd HoloLoom/server\npython agentic_api.py\n\`\`\``;
            }
            return `❌ Failed to save: ${error.message}`;
        }
    }

    /**
     * Ingest multiple files incrementally (batch update from file watcher)
     * This is used for auto-updating the knowledge graph when files change.
     */
    async ingestFilesIncremental(
        filePaths: string[],
        workspacePath: string
    ): Promise<IngestResponse> {
        try {
            const axios = require('axios');

            const response = await axios.post(
                `${this.baseUrl}/ingest/workspace/incremental`,
                {
                    files: filePaths,
                    workspace_path: workspacePath
                },
                {
                    timeout: 30000  // 30 second timeout for batch operation
                }
            );

            return response.data;
        } catch (error: any) {
            if (error.code === 'ECONNREFUSED' || error.message.includes('ECONNREFUSED')) {
                return {
                    success: false,
                    error: `HoloLoom server not running at ${this.baseUrl}`
                };
            }
            return {
                success: false,
                error: error.message || 'Unknown error'
            };
        }
    }

    /**
     * Ingest entire workspace (full index operation)
     * Used for initial workspace indexing or manual re-indexing.
     */
    async ingestWorkspaceFull(workspacePath: string): Promise<IngestResponse> {
        try {
            const axios = require('axios');

            const response = await axios.post(
                `${this.baseUrl}/ingest/workspace`,
                {
                    workspace_path: workspacePath,
                    languages: ['python', 'typescript', 'javascript'],
                    exclude_patterns: [
                        '**/node_modules/**',
                        '**/.git/**',
                        '**/.venv/**',
                        '**/dist/**',
                        '**/build/**',
                        '**/__pycache__/**'
                    ]
                },
                {
                    timeout: 120000  // 2 minute timeout for full workspace
                }
            );

            return response.data;
        } catch (error: any) {
            if (error.code === 'ECONNREFUSED' || error.message.includes('ECONNREFUSED')) {
                return {
                    success: false,
                    error: `HoloLoom server not running at ${this.baseUrl}`
                };
            }
            return {
                success: false,
                error: error.message || 'Unknown error'
            };
        }
    }

    async recall(query: string): Promise<string> {
        try {
            const axios = require('axios');
            const response = await axios.post(`${this.baseUrl}/api/recall`, {
                query,
                k: 5
            });

            const memories = response.data.memories || [];

            if (memories.length === 0) {
                return `🔍 **No memories found for:** "${query}"`;
            }

            let result = `**HoloLoom Recall Results:**\n\n`;

            memories.forEach((m: any, i: number) => {
                const confidence = (m.confidence * 100).toFixed(0);
                result += `**${i + 1}.** ${m.content}\n`;
                result += `   _Confidence: ${confidence}% | ${m.timestamp || 'unknown time'}_\n\n`;
            });

            return result;

        } catch (error: any) {
            if (error.code === 'ECONNREFUSED') {
                return `❌ HoloLoom server not running at ${this.baseUrl}`;
            }
            return `❌ Recall failed: ${error.message}`;
        }
    }

    async query(text: string): Promise<{ response: string; confidence?: number }> {
        try {
            const axios = require('axios');
            const response = await axios.post(`${this.baseUrl}/query`, {
                text,
                mode: 'verify',
                max_steps: 3
            });

            return {
                response: response.data.response || response.data.answer || 'No response',
                confidence: response.data.confidence
            };

        } catch (error: any) {
            throw error;
        }
    }
}
