/**
 * HoloLoomBridge - Enhanced API client for HoloLoom backend
 *
 * Features:
 * - Type-safe API client matching backend endpoints
 * - Request/response serialization
 * - Error handling and retries
 * - Connection status monitoring
 * - Batch operations
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import * as vscode from 'vscode';
import {
    QueryRequest,
    AgenticResponse,
    CodeContext,
    ReasoningMode,
    SimilarCodeResult,
    TestGenerationResult,
    DocumentationResult,
    RefactoringResult,
    WorkspaceIndexStats
} from '../types';

export class HoloLoomBridge {
    private client: AxiosInstance;
    private baseUrl: string;
    private connected: boolean = false;
    private statusBar: vscode.StatusBarItem;

    constructor(baseUrl: string = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.client = axios.create({
            baseURL: baseUrl,
            timeout: 60000, // 60 second timeout for complex queries
            headers: {
                'Content-Type': 'application/json'
            }
        });

        // Create status bar item
        this.statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
        this.statusBar.text = '$(sync~spin) Squad: Connecting...';
        this.statusBar.show();

        this.checkConnection();
    }

    /**
     * Check if HoloLoom backend is available
     */
    async checkConnection(): Promise<boolean> {
        try {
            const response = await this.client.get('/health', { timeout: 5000 });
            this.connected = response.status === 200;

            if (this.connected) {
                this.statusBar.text = '$(check) Squad: Connected';
                this.statusBar.backgroundColor = undefined;
            } else {
                this.statusBar.text = '$(error) Squad: Disconnected';
                this.statusBar.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
            }

            return this.connected;
        } catch (error) {
            this.connected = false;
            this.statusBar.text = '$(error) Squad: Server Offline';
            this.statusBar.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
            return false;
        }
    }

    /**
     * Main query endpoint - agentic reasoning
     */
    async query(
        text: string,
        context?: CodeContext,
        mode: ReasoningMode = 'verify',
        maxSteps: number = 5
    ): Promise<AgenticResponse> {
        if (!this.connected) {
            await this.checkConnection();
            if (!this.connected) {
                throw new Error('HoloLoom server is not available. Please start the server and try again.');
            }
        }

        try {
            const request: QueryRequest = {
                text,
                context,
                mode,
                max_steps: maxSteps
            };

            const response = await this.client.post<AgenticResponse>('/query', request);
            return response.data;

        } catch (error) {
            throw this.handleError(error);
        }
    }

    /**
     * Explain code - detailed explanation of code selection
     */
    async explainCode(
        code: string,
        context: CodeContext
    ): Promise<AgenticResponse> {
        const prompt = `Explain this code in detail:\n\n${code}\n\nProvide:
1. What the code does
2. Key concepts and patterns used
3. Potential improvements or issues
4. Related concepts to learn`;

        return this.query(prompt, context, 'verify', 3);
    }

    /**
     * Find similar code in workspace
     */
    async findSimilar(
        code: string,
        context: CodeContext,
        limit: number = 10
    ): Promise<AgenticResponse> {
        const prompt = `Find similar code patterns to:\n\n${code}\n\nSearch for:
1. Similar algorithms or logic
2. Similar data structures
3. Similar patterns or idioms`;

        return this.query(prompt, context, 'research', 5);
    }

    /**
     * Generate unit tests
     */
    async generateTests(
        code: string,
        context: CodeContext,
        framework?: string
    ): Promise<AgenticResponse> {
        const frameworkHint = framework ? ` using ${framework}` : '';
        const prompt = `Generate comprehensive unit tests${frameworkHint} for:\n\n${code}\n\nInclude:
1. Happy path tests
2. Edge cases
3. Error handling tests
4. Mock data if needed`;

        return this.query(prompt, context, 'plan_execute', 7);
    }

    /**
     * Add documentation
     */
    async addDocumentation(
        code: string,
        context: CodeContext
    ): Promise<AgenticResponse> {
        const prompt = `Generate appropriate documentation for:\n\n${code}\n\nInclude:
1. Description of functionality
2. Parameter descriptions
3. Return value description
4. Usage examples
5. Any important notes or warnings`;

        return this.query(prompt, context, 'verify', 3);
    }

    /**
     * Suggest refactorings
     */
    async suggestRefactorings(
        code: string,
        context: CodeContext
    ): Promise<AgenticResponse> {
        const prompt = `Analyze this code and suggest refactorings:\n\n${code}\n\nLook for:
1. Code smells
2. Performance issues
3. Readability improvements
4. Best practice violations
5. Security concerns`;

        return this.query(prompt, context, 'verify', 5);
    }

    /**
     * Review code changes (git diff)
     */
    async reviewChanges(
        diff: string,
        context: CodeContext
    ): Promise<AgenticResponse> {
        const prompt = `Review these code changes:\n\n${diff}\n\nProvide:
1. Summary of changes
2. Potential bugs or issues
3. Suggestions for improvement
4. Test coverage recommendations`;

        return this.query(prompt, context, 'verify', 5);
    }

    /**
     * Index entire workspace
     */
    async indexWorkspace(
        workspacePath: string,
        languages?: string[],
        excludePatterns?: string[]
    ): Promise<WorkspaceIndexStats> {
        try {
            const response = await this.client.post('/ingest/workspace', {
                workspace_path: workspacePath,
                languages,
                exclude_patterns: excludePatterns || ['**/node_modules/**', '**/.git/**', '**/dist/**', '**/out/**']
            });

            return {
                filesIndexed: response.data.ingestion_stats?.files_processed || 0,
                entitiesFound: response.data.ingestion_stats?.entities_found || 0,
                duration: response.data.ingestion_stats?.duration || 0,
                languages: response.data.ingestion_stats?.languages || []
            };

        } catch (error) {
            throw this.handleError(error);
        }
    }

    /**
     * Add memory to HoloLoom
     */
    async remember(
        content: string,
        metadata?: {
            workspace?: string;
            file?: string;
            language?: string;
        }
    ): Promise<boolean> {
        try {
            const response = await this.client.post('/memories/add', {
                text: content,
                episode: metadata?.workspace || 'vscode',
                entities: [],
                motifs: [],
                metadata: {
                    ...metadata,
                    source: 'vscode-squad',
                    timestamp: new Date().toISOString()
                }
            });

            return response.data.success === true;

        } catch (error) {
            throw this.handleError(error);
        }
    }

    /**
     * Get server statistics
     */
    async getStats(): Promise<any> {
        try {
            const response = await this.client.get('/stats');
            return response.data;
        } catch (error) {
            throw this.handleError(error);
        }
    }

    /**
     * Get audit trail
     */
    async getAuditTrail(limit: number = 100): Promise<any> {
        try {
            const response = await this.client.get('/audit-trail', {
                params: { limit }
            });
            return response.data;
        } catch (error) {
            throw this.handleError(error);
        }
    }

    /**
     * Detect logic errors in code
     */
    async detectLogicErrors(
        code: string,
        language: string,
        filePath: string = 'temp'
    ): Promise<any> {
        try {
            const response = await this.client.post('/detect/logic', {
                code,
                language,
                file_path: filePath
            });

            return response.data;

        } catch (error) {
            throw this.handleError(error);
        }
    }

    /**
     * Handle API errors
     */
    private handleError(error: any): Error {
        if (axios.isAxiosError(error)) {
            const axiosError = error as AxiosError;

            if (axiosError.code === 'ECONNREFUSED') {
                this.connected = false;
                this.statusBar.text = '$(error) Squad: Server Offline';
                return new Error(
                    `HoloLoom server is not running at ${this.baseUrl}\n\n` +
                    `Start it with:\n` +
                    `cd HoloLoom/server\n` +
                    `python agentic_api.py`
                );
            }

            if (axiosError.response) {
                const status = axiosError.response.status;
                const message = axiosError.response.data?.detail || axiosError.message;

                if (status === 429) {
                    return new Error(`Rate limit exceeded. Please try again in a moment.`);
                }

                if (status === 400) {
                    return new Error(`Invalid request: ${message}`);
                }

                if (status >= 500) {
                    return new Error(`Server error: ${message}`);
                }

                return new Error(`Request failed (${status}): ${message}`);
            }

            return new Error(`Network error: ${axiosError.message}`);
        }

        return error instanceof Error ? error : new Error(String(error));
    }

    /**
     * Update configuration
     */
    updateConfig() {
        const config = vscode.workspace.getConfiguration('squad');
        const newBaseUrl = config.get<string>('hololoomUrl') || 'http://localhost:8000';

        if (newBaseUrl !== this.baseUrl) {
            this.baseUrl = newBaseUrl;
            this.client = axios.create({
                baseURL: newBaseUrl,
                timeout: 60000,
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            this.checkConnection();
        }
    }

    /**
     * Cleanup
     */
    dispose() {
        this.statusBar.dispose();
    }
}
