/**
 * Fix AI Slop Command
 * ====================
 * The killer feature: Takes AI-generated broken code and makes it work.
 *
 * Flow:
 * 1. Detect hallucinations (non-existent functions/classes)
 * 2. Run verification (compiler/linter)
 * 3. Ask HoloLoom to fix issues (with codebase context)
 * 4. Verify fixed code
 * 5. Repeat until working or max iterations
 */

import * as vscode from 'vscode';
import { HoloLoomBridge } from './HoloLoomBridge';
import { VerificationService, VerificationResult, HallucinationResult } from './VerificationService';

export interface FixSlopOptions {
    maxIterations?: number;
    verifyTypes?: boolean;
    detectHallucinations?: boolean;
    applyAutomatically?: boolean;
}

export interface FixIteration {
    iteration: number;
    originalCode: string;
    fixedCode: string;
    verificationResult: VerificationResult | null;
    hallucinationResult: HallucinationResult | null;
    changes: string;
    success: boolean;
}

export interface FixSlopResult {
    success: boolean;
    finalCode: string;
    iterations: FixIteration[];
    totalIterations: number;
    message: string;
}

export class FixSlopCommand {
    private bridge: HoloLoomBridge;
    private verificationService: VerificationService;

    constructor(bridge: HoloLoomBridge, verificationService: VerificationService) {
        this.bridge = bridge;
        this.verificationService = verificationService;
    }

    /**
     * Main entry point: Fix AI slop code
     */
    async fix(
        editor: vscode.TextEditor,
        options: FixSlopOptions = {}
    ): Promise<FixSlopResult> {
        const {
            maxIterations = 5,
            verifyTypes = true,
            detectHallucinations = true,
            applyAutomatically = false
        } = options;

        const originalCode = editor.selection.isEmpty
            ? editor.document.getText()
            : editor.document.getText(editor.selection);

        const language = editor.document.languageId;
        const fileName = editor.document.fileName;

        let currentCode = originalCode;
        const iterations: FixIteration[] = [];
        let success = false;

        // Show progress
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: 'Squad: Fixing AI Slop',
            cancellable: true
        }, async (progress, token) => {
            for (let i = 0; i < maxIterations; i++) {
                if (token.isCancellationRequested) {
                    break;
                }

                progress.report({
                    message: `Iteration ${i + 1}/${maxIterations}...`,
                    increment: (100 / maxIterations)
                });

                // Step 1: Detect hallucinations
                let hallucinationResult: HallucinationResult | null = null;
                if (detectHallucinations) {
                    hallucinationResult = await this.verificationService.detectHallucinations(
                        currentCode,
                        this.mapLanguage(language),
                        fileName
                    );

                    if (hallucinationResult.hallucinations.length > 0) {
                        console.log(`Found ${hallucinationResult.hallucinations.length} hallucinations`);
                    }
                }

                // Step 2: Run verification
                const verificationResult = await this.verificationService.verifyCode(
                    currentCode,
                    this.mapLanguage(language),
                    fileName
                );

                // Check if we're done
                const hasHallucinations = hallucinationResult && hallucinationResult.hallucinations.length > 0;
                const hasErrors = verificationResult.error_count > 0;

                if (!hasHallucinations && !hasErrors) {
                    success = true;
                    iterations.push({
                        iteration: i + 1,
                        originalCode: currentCode,
                        fixedCode: currentCode,
                        verificationResult,
                        hallucinationResult,
                        changes: 'No changes needed',
                        success: true
                    });
                    break;
                }

                // Step 3: Build fix prompt
                const fixPrompt = this.buildFixPrompt(
                    currentCode,
                    language,
                    verificationResult,
                    hallucinationResult,
                    i
                );

                // Step 4: Ask HoloLoom to fix (with retry logic)
                let fixSuccess = false;
                let retryCount = 0;
                const maxRetries = 3;

                while (!fixSuccess && retryCount < maxRetries) {
                    try {
                        const result = await this.bridge.query(
                            fixPrompt,
                            {
                                currentFile: currentCode,
                                fileName: fileName,
                                languageId: language,
                                workspace: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
                            },
                            'verify',  // Use VERIFY mode for careful checking
                            10  // Allow more steps for complex fixes
                        );

                        // Extract code from response
                        const fixedCode = this.extractCode(result.response);

                        if (!fixedCode || fixedCode === currentCode) {
                            // No improvement
                            iterations.push({
                                iteration: i + 1,
                                originalCode: currentCode,
                                fixedCode: currentCode,
                                verificationResult,
                                hallucinationResult,
                                changes: 'No improvements made',
                                success: false
                            });
                            fixSuccess = true;  // Exit retry loop
                            break;
                        }

                        iterations.push({
                            iteration: i + 1,
                            originalCode: currentCode,
                            fixedCode: fixedCode,
                            verificationResult,
                            hallucinationResult,
                            changes: this.summarizeChanges(currentCode, fixedCode),
                            success: false
                        });

                        currentCode = fixedCode;
                        fixSuccess = true;  // Successfully fixed

                    } catch (error: any) {
                        retryCount++;
                        console.error(`Fix iteration failed (attempt ${retryCount}/${maxRetries}):`, error);

                        // Check if it's a rate limit error
                        if (error.response?.status === 429) {
                            const retryAfter = error.response?.data?.retry_after || 60;
                            vscode.window.showWarningMessage(
                                `Rate limit exceeded. Waiting ${retryAfter}s before retry...`
                            );
                            await this.sleep(retryAfter * 1000);
                            continue;  // Retry immediately after waiting
                        }

                        // For other errors, use exponential backoff
                        if (retryCount < maxRetries) {
                            const backoffMs = Math.min(1000 * Math.pow(2, retryCount), 10000);
                            console.log(`Retrying in ${backoffMs}ms...`);
                            await this.sleep(backoffMs);
                        } else {
                            // Max retries exceeded
                            vscode.window.showErrorMessage(
                                `Failed to fix code after ${maxRetries} attempts. Error: ${error.message}`
                            );
                            break;  // Exit iteration loop
                        }
                    }
                }

                // If all retries failed, exit iteration loop
                if (!fixSuccess) {
                    break;
                }
            }
        });

        // Final verification
        let finalMessage = '';
        if (success) {
            finalMessage = `✅ Code fixed in ${iterations.length} iteration(s)!`;
        } else {
            const lastIteration = iterations[iterations.length - 1];
            const errorCount = lastIteration?.verificationResult?.error_count || 0;
            finalMessage = `⚠️ Could not fix all issues after ${maxIterations} iterations. ${errorCount} error(s) remain.`;
        }

        return {
            success,
            finalCode: currentCode,
            iterations,
            totalIterations: iterations.length,
            message: finalMessage
        };
    }

    /**
     * Build fix prompt with all available context
     */
    private buildFixPrompt(
        code: string,
        language: string,
        verificationResult: VerificationResult,
        hallucinationResult: HallucinationResult | null,
        iteration: number
    ): string {
        let prompt = `Fix the following ${language} code to make it work correctly.\n\n`;

        // Add hallucination info
        if (hallucinationResult && hallucinationResult.hallucinations.length > 0) {
            prompt += `**Hallucinations Detected (references to non-existent entities):**\n`;
            for (const h of hallucinationResult.hallucinations.slice(0, 5)) {
                prompt += `- Line ${h.line}: '${h.reference}' ${h.reason}\n`;
                if (h.suggestions.length > 0) {
                    prompt += `  Suggestions: ${h.suggestions.slice(0, 3).join(', ')}\n`;
                }
                if (h.fix) {
                    prompt += `  Fix: ${h.fix}\n`;
                }
            }
            prompt += '\n';
        }

        // Add verification errors
        if (verificationResult.error_count > 0) {
            prompt += `**Compilation/Type Errors:**\n`;
            for (const error of verificationResult.errors.slice(0, 10)) {
                prompt += `- Line ${error.line}: ${error.message}`;
                if (error.code) {
                    prompt += ` [${error.code}]`;
                }
                prompt += '\n';
            }
            prompt += '\n';
        }

        prompt += `**Current Code:**\n\`\`\`${language}\n${code}\n\`\`\`\n\n`;

        prompt += `**Instructions:**\n`;
        prompt += `1. Only use functions/classes that exist in the codebase (check hallucinations list)\n`;
        prompt += `2. Fix all compilation and type errors\n`;
        prompt += `3. Preserve the original intent and logic\n`;
        prompt += `4. Return ONLY the fixed code in a code block\n`;
        prompt += `5. Do not add explanations outside the code block\n`;

        if (iteration > 0) {
            prompt += `\n(Iteration ${iteration + 1} - previous attempts did not fully resolve the issues)\n`;
        }

        return prompt;
    }

    /**
     * Extract code from markdown response
     */
    private extractCode(response: string): string {
        // Try to extract code block
        const codeBlockRegex = /```(?:\w+)?\n([\s\S]*?)```/g;
        const matches = Array.from(response.matchAll(codeBlockRegex));

        if (matches.length > 0) {
            // Return the last code block (most likely the fixed code)
            return matches[matches.length - 1][1].trim();
        }

        // If no code blocks, return original
        return response.trim();
    }

    /**
     * Summarize changes between old and new code
     */
    private summarizeChanges(oldCode: string, newCode: string): string {
        const oldLines = oldCode.split('\n');
        const newLines = newCode.split('\n');

        let additions = 0;
        let deletions = 0;
        let modifications = 0;

        // Simple diff
        const maxLen = Math.max(oldLines.length, newLines.length);
        for (let i = 0; i < maxLen; i++) {
            const oldLine = oldLines[i];
            const newLine = newLines[i];

            if (!oldLine && newLine) {
                additions++;
            } else if (oldLine && !newLine) {
                deletions++;
            } else if (oldLine !== newLine) {
                modifications++;
            }
        }

        return `+${additions} -${deletions} ~${modifications}`;
    }

    /**
     * Map VS Code language ID to HoloLoom language
     */
    private mapLanguage(languageId: string): string {
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
     * Sleep for specified milliseconds (for retry backoff)
     */
    private sleep(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Show result in side panel with diff
     */
    async showResult(result: FixSlopResult, originalCode: string): Promise<void> {
        if (result.success) {
            vscode.window.showInformationMessage(result.message);

            // Show diff if code changed
            if (result.finalCode !== originalCode) {
                await this.showDiff(originalCode, result.finalCode);
            }
        } else {
            const action = await vscode.window.showWarningMessage(
                result.message,
                'Show Details',
                'Apply Anyway'
            );

            if (action === 'Show Details') {
                await this.showIterationDetails(result);
            } else if (action === 'Apply Anyway') {
                // User wants to apply even if not fully fixed
                return;
            }
        }
    }

    /**
     * Show diff between original and fixed code
     */
    private async showDiff(original: string, fixed: string): Promise<void> {
        const originalUri = vscode.Uri.parse('squad:original.ts');
        const fixedUri = vscode.Uri.parse('squad:fixed.ts');

        // Register text document content provider
        const provider = new class implements vscode.TextDocumentContentProvider {
            provideTextDocumentContent(uri: vscode.Uri): string {
                if (uri.toString() === originalUri.toString()) {
                    return original;
                }
                return fixed;
            }
        };

        const registration = vscode.workspace.registerTextDocumentContentProvider('squad', provider);

        await vscode.commands.executeCommand(
            'vscode.diff',
            originalUri,
            fixedUri,
            'Squad: Original ↔ Fixed'
        );

        // Clean up after showing diff
        setTimeout(() => registration.dispose(), 60000);  // 1 minute
    }

    /**
     * Show iteration details in output channel
     */
    private async showIterationDetails(result: FixSlopResult): Promise<void> {
        const channel = vscode.window.createOutputChannel('Squad: Fix Slop Details');
        channel.clear();
        channel.show();

        channel.appendLine('='.repeat(80));
        channel.appendLine('Squad: Fix AI Slop - Detailed Report');
        channel.appendLine('='.repeat(80));
        channel.appendLine('');

        channel.appendLine(`Total Iterations: ${result.totalIterations}`);
        channel.appendLine(`Final Status: ${result.success ? 'SUCCESS' : 'INCOMPLETE'}`);
        channel.appendLine('');

        for (const iteration of result.iterations) {
            channel.appendLine(`${'='.repeat(40)} Iteration ${iteration.iteration} ${'='.repeat(40)}`);
            channel.appendLine('');

            if (iteration.hallucinationResult) {
                channel.appendLine(`Hallucinations: ${iteration.hallucinationResult.hallucinations.length}`);
                for (const h of iteration.hallucinationResult.hallucinations.slice(0, 5)) {
                    channel.appendLine(`  - Line ${h.line}: ${h.reference} (${h.type})`);
                }
                channel.appendLine('');
            }

            if (iteration.verificationResult) {
                channel.appendLine(`Errors: ${iteration.verificationResult.error_count}`);
                channel.appendLine(`Warnings: ${iteration.verificationResult.warning_count}`);
                for (const error of iteration.verificationResult.errors.slice(0, 5)) {
                    channel.appendLine(`  - Line ${error.line}: ${error.message}`);
                }
                channel.appendLine('');
            }

            channel.appendLine(`Changes: ${iteration.changes}`);
            channel.appendLine('');
        }

        channel.appendLine('='.repeat(80));
        channel.appendLine('Final Code:');
        channel.appendLine('='.repeat(80));
        channel.appendLine(result.finalCode);
    }
}
