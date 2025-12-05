/**
 * Verification Service
 * Handles code verification using HoloLoom backend
 */

import * as vscode from 'vscode';
import { HoloLoomBridge, CodeContext } from './HoloLoomBridge';

export interface VerificationError {
    line: number;
    column: number | null;
    message: string;
    severity: 'error' | 'warning' | 'info';
    code: string | null;
    source: string;
}

export interface VerificationResult {
    success: boolean;
    language: string;
    checks_run: string[];
    error_count: number;
    warning_count: number;
    errors: VerificationError[];
    warnings: VerificationError[];
    summary: string;
}

export interface HallucinationDetection {
    reference: string;
    type: string;
    line: number;
    confidence: number;
    reason: string;
    suggestions: string[];
    fix: string | null;
    context: string;
}

export interface HallucinationResult {
    success: boolean;
    language: string;
    hallucinations: HallucinationDetection[];
    summary: {
        total_hallucinations: number;
        by_type: Record<string, number>;
        high_confidence: any[];
        suggested_fixes: any[];
    };
}

export class VerificationService {
    private bridge: HoloLoomBridge;
    private diagnosticsCollection: vscode.DiagnosticCollection;

    constructor(bridge: HoloLoomBridge) {
        this.bridge = bridge;
        this.diagnosticsCollection = vscode.languages.createDiagnosticCollection('squad');
    }

    /**
     * Verify code using HoloLoom backend
     */
    async verifyCode(
        code: string,
        language: string,
        filePath: string = 'temp'
    ): Promise<VerificationResult> {
        const response = await this.bridge.client.post('/verify/code', {
            code,
            language,
            file_path: filePath,
            check_syntax: true,
            check_types: true,
            check_lint: false
        });

        return response.data.verification;
    }

    /**
     * Detect hallucinations in code
     */
    async detectHallucinations(
        code: string,
        language: string,
        filePath: string = 'temp'
    ): Promise<HallucinationResult> {
        const response = await this.bridge.client.post('/detect/hallucinations', {
            code,
            language,
            file_path: filePath,
            strict: false
        });

        return response.data;
    }

    /**
     * Verify code and show diagnostics in editor
     */
    async verifyAndShowDiagnostics(
        editor: vscode.TextEditor
    ): Promise<VerificationResult> {
        const code = editor.document.getText();
        const language = this.mapLanguageId(editor.document.languageId);
        const filePath = editor.document.fileName;

        const result = await this.verifyCode(code, language, filePath);

        // Convert errors to VS Code diagnostics
        const diagnostics: vscode.Diagnostic[] = [];

        for (const error of result.errors) {
            const range = new vscode.Range(
                error.line - 1,
                error.column || 0,
                error.line - 1,
                error.column ? error.column + 1 : 1000
            );

            const severity = error.severity === 'error'
                ? vscode.DiagnosticSeverity.Error
                : vscode.DiagnosticSeverity.Warning;

            const diagnostic = new vscode.Diagnostic(
                range,
                error.message,
                severity
            );

            diagnostic.source = `Squad (${error.source})`;
            if (error.code) {
                diagnostic.code = error.code;
            }

            diagnostics.push(diagnostic);
        }

        // Show diagnostics
        this.diagnosticsCollection.set(editor.document.uri, diagnostics);

        return result;
    }

    /**
     * Detect hallucinations and show as code actions
     */
    async detectAndShowHallucinations(
        editor: vscode.TextEditor
    ): Promise<HallucinationResult> {
        const code = editor.document.getText();
        const language = this.mapLanguageId(editor.document.languageId);
        const filePath = editor.document.fileName;

        const result = await this.detectHallucinations(code, language, filePath);

        // Convert hallucinations to diagnostics with suggestions
        const diagnostics: vscode.Diagnostic[] = [];

        for (const halluc of result.hallucinations) {
            const range = new vscode.Range(
                halluc.line - 1,
                0,
                halluc.line - 1,
                1000
            );

            const severity = halluc.confidence > 0.8
                ? vscode.DiagnosticSeverity.Error
                : vscode.DiagnosticSeverity.Warning;

            let message = `${halluc.reason}`;
            if (halluc.suggestions.length > 0) {
                message += `\n\nDid you mean: ${halluc.suggestions.slice(0, 3).join(', ')}`;
            }

            const diagnostic = new vscode.Diagnostic(
                range,
                message,
                severity
            );

            diagnostic.source = `Squad (hallucination detector)`;
            diagnostic.code = halluc.type;

            diagnostics.push(diagnostic);
        }

        // Add to existing diagnostics
        const existing = this.diagnosticsCollection.get(editor.document.uri) || [];
        this.diagnosticsCollection.set(
            editor.document.uri,
            [...existing, ...diagnostics]
        );

        return result;
    }

    /**
     * Clear diagnostics for a document
     */
    clearDiagnostics(uri: vscode.Uri) {
        this.diagnosticsCollection.delete(uri);
    }

    /**
     * Clear all diagnostics
     */
    clearAllDiagnostics() {
        this.diagnosticsCollection.clear();
    }

    /**
     * Map VS Code language ID to HoloLoom language
     */
    private mapLanguageId(languageId: string): string {
        const map: Record<string, string> = {
            'python': 'python',
            'typescript': 'typescript',
            'typescriptreact': 'typescript',
            'javascript': 'javascript',
            'javascriptreact': 'javascript'
        };

        return map[languageId] || languageId;
    }

    /**
     * Get verification summary for status bar
     */
    getSummary(result: VerificationResult): string {
        if (result.error_count === 0) {
            return '✅ No errors';
        }

        return `❌ ${result.error_count} error(s)`;
    }

    /**
     * Dispose resources
     */
    dispose() {
        this.diagnosticsCollection.dispose();
    }
}
