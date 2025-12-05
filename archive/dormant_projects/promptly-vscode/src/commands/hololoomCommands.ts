import * as vscode from 'vscode';

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
