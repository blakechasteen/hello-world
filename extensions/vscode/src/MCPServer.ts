/**
 * MCP Server for Claude Code
 * Exposes VS Code extension capabilities via Model Context Protocol over WebSocket
 *
 * Architecture:
 * - WebSocket server on port 9001
 * - Implements MCP protocol for tool discovery and invocation
 * - Bridges Matrix ChatOps → VS Code via standardized protocol
 * - Supports bidirectional communication for push notifications
 */

import * as vscode from 'vscode';
import WebSocket from 'ws';
import { CodeContextProvider } from './CodeContextProvider';
import { HoloLoomBridge, CodeContext, AgenticResult } from './HoloLoomBridge';

// ============================================================================
// MCP Protocol Types
// ============================================================================

interface MCPRequest {
    jsonrpc: '2.0';
    id: string | number;
    method: string;
    params?: any;
}

interface MCPResponse {
    jsonrpc: '2.0';
    id: string | number;
    result?: any;
    error?: {
        code: number;
        message: string;
        data?: any;
    };
}

interface MCPNotification {
    jsonrpc: '2.0';
    method: string;
    params?: any;
}

interface ToolDefinition {
    name: string;
    description: string;
    inputSchema: {
        type: 'object';
        properties: Record<string, any>;
        required?: string[];
    };
}

// ============================================================================
// MCP Server Implementation
// ============================================================================

export class MCPServer {
    private wss: WebSocket.Server | null = null;
    private clients: Set<WebSocket> = new Set();
    private contextProvider: CodeContextProvider;
    private bridge: HoloLoomBridge;
    private port: number;
    private outputChannel: vscode.OutputChannel;

    constructor(
        contextProvider: CodeContextProvider,
        bridge: HoloLoomBridge,
        port: number = 9001
    ) {
        this.contextProvider = contextProvider;
        this.bridge = bridge;
        this.port = port;
        this.outputChannel = vscode.window.createOutputChannel('Claude Code MCP');
    }

    /**
     * Start the MCP WebSocket server
     */
    async start(): Promise<void> {
        if (this.wss) {
            this.log('MCP Server already running');
            return;
        }

        try {
            this.wss = new WebSocket.Server({ port: this.port });

            this.wss.on('connection', (ws: WebSocket) => {
                this.handleConnection(ws);
            });

            this.wss.on('error', (error: Error) => {
                this.log(`MCP Server error: ${error.message}`);
            });

            this.log(`MCP Server started on ws://localhost:${this.port}`);

            // Show status bar notification
            vscode.window.showInformationMessage(
                `Claude Code MCP Server ready on port ${this.port}`
            );
        } catch (error: any) {
            this.log(`Failed to start MCP Server: ${error.message}`);
            throw error;
        }
    }

    /**
     * Stop the MCP WebSocket server
     */
    async stop(): Promise<void> {
        if (!this.wss) {
            return;
        }

        // Close all client connections
        for (const client of this.clients) {
            client.close();
        }
        this.clients.clear();

        // Close server
        await new Promise<void>((resolve) => {
            this.wss!.close(() => {
                this.log('MCP Server stopped');
                resolve();
            });
        });

        this.wss = null;
    }

    /**
     * Handle new WebSocket connection
     */
    private handleConnection(ws: WebSocket): void {
        this.clients.add(ws);
        this.log(`New MCP client connected (total: ${this.clients.size})`);

        ws.on('message', async (data: WebSocket.Data) => {
            try {
                const message = JSON.parse(data.toString());
                await this.handleMessage(ws, message);
            } catch (error: any) {
                this.sendError(ws, null, -32700, 'Parse error');
            }
        });

        ws.on('close', () => {
            this.clients.delete(ws);
            this.log(`MCP client disconnected (remaining: ${this.clients.size})`);
        });

        ws.on('error', (error: Error) => {
            this.log(`WebSocket error: ${error.message}`);
        });

        // Send welcome notification
        this.sendNotification(ws, 'mcp/connected', {
            server: 'Claude Code MCP Server',
            version: '1.0.0',
            capabilities: ['tools', 'notifications']
        });
    }

    /**
     * Handle MCP protocol message
     */
    private async handleMessage(ws: WebSocket, message: MCPRequest): Promise<void> {
        const { method, params, id } = message;

        switch (method) {
            case 'tools/list':
                this.sendResponse(ws, id, { tools: this.getTools() });
                break;

            case 'tools/call':
                await this.handleToolCall(ws, id, params);
                break;

            case 'initialize':
                this.sendResponse(ws, id, {
                    protocolVersion: '2024-11-05',
                    serverInfo: {
                        name: 'claude-code-mcp',
                        version: '1.0.0'
                    },
                    capabilities: {
                        tools: {}
                    }
                });
                break;

            case 'ping':
                this.sendResponse(ws, id, { pong: true });
                break;

            default:
                this.sendError(ws, id, -32601, `Method not found: ${method}`);
        }
    }

    /**
     * Get list of available tools
     */
    private getTools(): ToolDefinition[] {
        return [
            {
                name: 'code/query',
                description: 'Ask a question about code using HoloLoom agentic reasoning',
                inputSchema: {
                    type: 'object',
                    properties: {
                        question: {
                            type: 'string',
                            description: 'The question to ask'
                        },
                        mode: {
                            type: 'string',
                            enum: ['direct', 'verify', 'research', 'plan_execute'],
                            description: 'Reasoning mode',
                            default: 'verify'
                        },
                        includeContext: {
                            type: 'boolean',
                            description: 'Include current editor context',
                            default: true
                        }
                    },
                    required: ['question']
                }
            },
            {
                name: 'code/refactor',
                description: 'Refactor code with a specific instruction',
                inputSchema: {
                    type: 'object',
                    properties: {
                        instruction: {
                            type: 'string',
                            description: 'What refactoring to perform'
                        },
                        code: {
                            type: 'string',
                            description: 'Code to refactor (defaults to current selection)'
                        }
                    },
                    required: ['instruction']
                }
            },
            {
                name: 'code/explain',
                description: 'Explain selected code or concept',
                inputSchema: {
                    type: 'object',
                    properties: {
                        target: {
                            type: 'string',
                            description: 'What to explain (defaults to current selection)'
                        },
                        depth: {
                            type: 'string',
                            enum: ['brief', 'detailed', 'comprehensive'],
                            description: 'Explanation depth',
                            default: 'detailed'
                        }
                    }
                }
            },
            {
                name: 'code/test',
                description: 'Generate tests for code',
                inputSchema: {
                    type: 'object',
                    properties: {
                        code: {
                            type: 'string',
                            description: 'Code to test (defaults to current selection)'
                        },
                        testType: {
                            type: 'string',
                            enum: ['unit', 'integration', 'edge', 'all'],
                            description: 'Type of tests to generate',
                            default: 'unit'
                        }
                    }
                }
            },
            {
                name: 'code/fix',
                description: 'Suggest fixes for code issues',
                inputSchema: {
                    type: 'object',
                    properties: {
                        code: {
                            type: 'string',
                            description: 'Code with issues (defaults to current file)'
                        },
                        includeDiagnostics: {
                            type: 'boolean',
                            description: 'Include VS Code diagnostics',
                            default: true
                        }
                    }
                }
            },
            {
                name: 'code/context',
                description: 'Get current editor context',
                inputSchema: {
                    type: 'object',
                    properties: {
                        includeSelection: {
                            type: 'boolean',
                            default: true
                        },
                        includeDiagnostics: {
                            type: 'boolean',
                            default: true
                        }
                    }
                }
            }
        ];
    }

    /**
     * Handle tool invocation
     */
    private async handleToolCall(ws: WebSocket, id: string | number, params: any): Promise<void> {
        const { name, arguments: args } = params;

        try {
            let result: any;

            switch (name) {
                case 'code/query':
                    result = await this.toolQuery(args);
                    break;

                case 'code/refactor':
                    result = await this.toolRefactor(args);
                    break;

                case 'code/explain':
                    result = await this.toolExplain(args);
                    break;

                case 'code/test':
                    result = await this.toolTest(args);
                    break;

                case 'code/fix':
                    result = await this.toolFix(args);
                    break;

                case 'code/context':
                    result = await this.toolContext(args);
                    break;

                default:
                    throw new Error(`Unknown tool: ${name}`);
            }

            this.sendResponse(ws, id, { content: [{ type: 'text', text: result }] });
        } catch (error: any) {
            this.sendError(ws, id, -32603, error.message);
        }
    }

    // ========================================================================
    // Tool Implementations
    // ========================================================================

    private async toolQuery(args: any): Promise<string> {
        const { question, mode = 'verify', includeContext = true } = args;

        const context = includeContext ? this.contextProvider.getCurrentContext() : undefined;

        const result: AgenticResult = await this.bridge.query(question, context, mode);

        // Format response
        let response = `**Answer** (confidence: ${(result.confidence * 100).toFixed(1)}%):\n${result.response}\n\n`;

        if (result.reasoning_mode) {
            response += `**Reasoning Mode**: ${result.reasoning_mode}\n`;
        }

        if (result.steps_taken && result.steps_taken.length > 0) {
            response += `\n**Steps Taken** (${result.steps_taken.length}):\n`;
            result.steps_taken.forEach((step, i) => {
                response += `${i + 1}. ${step.type}: ${step.query || step.finding || 'completed'}\n`;
            });
        }

        if (result.verification) {
            response += `\n**Verification**:\n`;
            response += `- Verified: ${result.verification.verified}\n`;
            if (result.verification.contradictions.length > 0) {
                response += `- Contradictions: ${result.verification.contradictions.join(', ')}\n`;
            }
        }

        return response;
    }

    private async toolRefactor(args: any): Promise<string> {
        const { instruction, code } = args;

        const editor = vscode.window.activeTextEditor;
        const targetCode = code || (editor?.selection && !editor.selection.isEmpty
            ? editor.document.getText(editor.selection)
            : null);

        if (!targetCode) {
            throw new Error('No code to refactor. Please select code or provide code parameter.');
        }

        const context = this.contextProvider.getCurrentContext();
        const language = context.languageId || 'code';
        const question = `Refactor this ${language} code: ${instruction}\n\nCode:\n${targetCode}`;

        const result: AgenticResult = await this.bridge.query(question, context, 'plan_execute');

        return `**Refactoring Result**:\n\n${result.response}\n\nConfidence: ${(result.confidence * 100).toFixed(1)}%`;
    }

    private async toolExplain(args: any): Promise<string> {
        const { target, depth = 'detailed' } = args;

        const editor = vscode.window.activeTextEditor;
        const toExplain = target || (editor?.selection && !editor.selection.isEmpty
            ? editor.document.getText(editor.selection)
            : null);

        if (!toExplain) {
            throw new Error('Nothing to explain. Please select code or provide target parameter.');
        }

        const context = this.contextProvider.getCurrentContext();
        const depthInstruction = {
            'brief': 'briefly',
            'detailed': 'in detail',
            'comprehensive': 'comprehensively with examples'
        }[depth];

        const question = `Explain ${depthInstruction} the following:\n\n${toExplain}`;

        const result: AgenticResult = await this.bridge.query(question, context, 'verify');

        return result.response;
    }

    private async toolTest(args: any): Promise<string> {
        const { code, testType = 'unit' } = args;

        const editor = vscode.window.activeTextEditor;
        const targetCode = code || (editor?.selection && !editor.selection.isEmpty
            ? editor.document.getText(editor.selection)
            : editor?.document.getText());

        if (!targetCode) {
            throw new Error('No code to generate tests for.');
        }

        const context = this.contextProvider.getCurrentContext();
        const language = context.languageId || 'code';

        const testTypeMap: Record<string, string> = {
            'unit': 'Unit tests',
            'integration': 'Integration tests',
            'edge': 'Edge case tests',
            'all': 'Comprehensive tests (unit, integration, edge cases)'
        };

        const question = `Generate ${testTypeMap[testType]} for this ${language} code:\n\n${targetCode}`;

        const result: AgenticResult = await this.bridge.query(question, context, 'plan_execute');

        return result.response;
    }

    private async toolFix(args: any): Promise<string> {
        const { code, includeDiagnostics = true } = args;

        const editor = vscode.window.activeTextEditor;
        const targetCode = code || editor?.document.getText();

        if (!targetCode) {
            throw new Error('No code to fix.');
        }

        const context = this.contextProvider.getCurrentContext();

        let question = 'Fix the issues in this code:\n\n';

        if (includeDiagnostics && editor) {
            const diagnostics = vscode.languages.getDiagnostics(editor.document.uri);
            if (diagnostics.length > 0) {
                question += 'Issues detected:\n';
                diagnostics.forEach(d => {
                    question += `- ${d.message}\n`;
                });
                question += '\n';
                context.diagnostics = diagnostics;
            }
        }

        question += `Code:\n${targetCode}`;

        const result: AgenticResult = await this.bridge.query(question, context, 'plan_execute');

        return result.response;
    }

    private async toolContext(args: any): Promise<string> {
        const { includeSelection = true, includeDiagnostics = true } = args;

        const context = this.contextProvider.getCurrentContext();
        const editor = vscode.window.activeTextEditor;

        let response = '**Current Editor Context**:\n\n';
        response += `- File: ${context.fileName || 'N/A'}\n`;
        response += `- Language: ${context.languageId || 'N/A'}\n`;
        response += `- Workspace: ${context.workspace || 'N/A'}\n\n`;

        if (includeSelection && context.selection) {
            response += `**Selected Code**:\n\`\`\`${context.languageId || ''}\n${context.selection}\n\`\`\`\n\n`;
        }

        if (includeDiagnostics && editor) {
            const diagnostics = vscode.languages.getDiagnostics(editor.document.uri);
            if (diagnostics.length > 0) {
                response += `**Diagnostics** (${diagnostics.length}):\n`;
                diagnostics.forEach((d, i) => {
                    response += `${i + 1}. [${d.severity}] ${d.message}\n`;
                });
            }
        }

        return response;
    }

    // ========================================================================
    // Push Notifications (VS Code → Matrix)
    // ========================================================================

    /**
     * Send notification to all connected clients
     */
    public broadcastNotification(method: string, params: any): void {
        const notification: MCPNotification = {
            jsonrpc: '2.0',
            method,
            params
        };

        for (const client of this.clients) {
            if (client.readyState === WebSocket.OPEN) {
                client.send(JSON.stringify(notification));
            }
        }
    }

    /**
     * Notify about file changes
     */
    public notifyFileChange(uri: vscode.Uri, changeType: string): void {
        this.broadcastNotification('file/changed', {
            uri: uri.toString(),
            changeType,
            timestamp: Date.now()
        });
    }

    /**
     * Notify about diagnostics
     */
    public notifyDiagnostics(uri: vscode.Uri, diagnostics: vscode.Diagnostic[]): void {
        if (diagnostics.length === 0) return;

        this.broadcastNotification('diagnostics/updated', {
            uri: uri.toString(),
            diagnostics: diagnostics.map(d => ({
                message: d.message,
                severity: d.severity,
                range: {
                    start: { line: d.range.start.line, character: d.range.start.character },
                    end: { line: d.range.end.line, character: d.range.end.character }
                }
            })),
            timestamp: Date.now()
        });
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    private sendResponse(ws: WebSocket, id: string | number, result: any): void {
        const response: MCPResponse = {
            jsonrpc: '2.0',
            id,
            result
        };
        ws.send(JSON.stringify(response));
    }

    private sendError(ws: WebSocket, id: string | number | null, code: number, message: string): void {
        const response: MCPResponse = {
            jsonrpc: '2.0',
            id: id || 0,
            error: {
                code,
                message
            }
        };
        ws.send(JSON.stringify(response));
    }

    private sendNotification(ws: WebSocket, method: string, params: any): void {
        const notification: MCPNotification = {
            jsonrpc: '2.0',
            method,
            params
        };
        ws.send(JSON.stringify(notification));
    }

    private log(message: string): void {
        const timestamp = new Date().toISOString();
        this.outputChannel.appendLine(`[${timestamp}] ${message}`);
        console.log(`[MCP Server] ${message}`);
    }

    /**
     * Get server status
     */
    public getStatus(): { running: boolean; port: number; clients: number } {
        return {
            running: this.wss !== null,
            port: this.port,
            clients: this.clients.size
        };
    }
}
