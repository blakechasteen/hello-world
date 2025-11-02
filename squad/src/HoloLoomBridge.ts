/**
 * HoloLoom Bridge
 * Communication layer between VS Code and HoloLoom Python server
 */

import axios, { AxiosInstance } from 'axios';
import * as vscode from 'vscode';

export interface QueryRequest {
    text: string;
    context?: CodeContext;
    mode?: 'direct' | 'verify' | 'research' | 'plan_execute';
    max_steps?: number;
}

export interface CodeContext {
    currentFile?: string;
    fileName?: string;
    languageId?: string;
    selection?: string;
    diagnostics?: vscode.Diagnostic[];
    workspace?: string;
}

export interface AgenticResult {
    response: string;
    confidence: number;
    reasoning_mode: string;
    steps_taken: ReasoningStep[];
    total_queries: number;
    total_duration_ms: number;
    verification?: VerificationResult;
}

export interface ReasoningStep {
    type: string;
    query?: string;
    confidence?: number;
    finding?: string;
    completed?: boolean;
}

export interface VerificationResult {
    verified: boolean;
    confidence: number;
    contradictions: string[];
    supporting_evidence: string[];
    suggested_refinements: string[];
}

export class HoloLoomBridge {
    private client: AxiosInstance;
    private serverUrl: string;

    constructor(serverUrl: string = 'http://localhost:8000') {
        this.serverUrl = serverUrl;
        this.client = axios.create({
            baseURL: serverUrl,
            timeout: 60000,
            headers: {
                'Content-Type': 'application/json'
            }
        });
    }

    async healthCheck(): Promise<boolean> {
        try {
            const response = await this.client.get('/health');
            return response.status === 200;
        } catch {
            return false;
        }
    }

    async query(
        text: string,
        context?: CodeContext,
        mode: string = 'verify',
        maxSteps: number = 5
    ): Promise<AgenticResult> {
        const request: QueryRequest = {
            text,
            context,
            mode: mode as any,
            max_steps: maxSteps
        };

        const response = await this.client.post('/query', request);
        return response.data;
    }

    async queryWithVerification(
        text: string,
        context?: CodeContext,
        verificationCriteria?: string[]
    ): Promise<AgenticResult> {
        // Force VERIFY mode and add verification criteria
        const enhancedContext = {
            ...context,
            verificationCriteria: verificationCriteria || [
                'correctness',
                'type_safety',
                'best_practices'
            ]
        };

        return this.query(text, enhancedContext, 'verify', 10);
    }

    async evaluateCodeQuality(
        code: string,
        criteria: string[] = ['correctness', 'maintainability', 'performance']
    ): Promise<any> {
        const response = await this.client.post('/evaluate', {
            code,
            criteria
        });
        return response.data;
    }

    async chat(message: string, context?: CodeContext): Promise<string> {
        const response = await this.client.post('/chat', {
            message,
            context
        });
        return response.data.response;
    }

    async getStats(): Promise<any> {
        const response = await this.client.get('/stats');
        return response.data;
    }

    async ingestWorkspace(workspacePath: string): Promise<any> {
        const response = await this.client.post('/ingest/workspace', {
            workspace_path: workspacePath
        });
        return response.data;
    }

    async stop(): Promise<void> {
        // Graceful shutdown if needed
    }
}