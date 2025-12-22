/**
 * PromptlyBridge - Bridge between VS Code extension and Promptly Python CLI
 *
 * Executes Python CLI commands and parses JSON output for VS Code integration.
 */

import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';

export interface Prompt {
    id: number;
    name: string;
    content: string;
    version: number;
    branch: string;
    commit_hash: string;
    created_at: string;
    updated_at: string;
    metadata: Record<string, any>;
}

export interface PromptVersion {
    version: number;
    content: string;
    commit_hash: string;
    message: string;
    created_at: string;
    metadata: Record<string, any>;
}

export interface Chain {
    id: number;
    name: string;
    steps: ChainStep[];
    created_at: string;
    metadata: Record<string, any>;
}

export interface ChainStep {
    prompt_name: string;
    output_var?: string;
    condition?: string;
}

export interface Skill {
    id: number;
    name: string;
    description: string;
    prompt_template: string;
    input_schema: Record<string, any>;
    output_schema: Record<string, any>;
    created_at: string;
    metadata: Record<string, any>;
}

export interface EvalResult {
    prompt_name: string;
    score: number;
    criteria: Record<string, number>;
    feedback: string;
    timestamp: string;
}

export interface BridgeResult<T> {
    success: boolean;
    data?: T;
    error?: string;
}

export class PromptlyBridge {
    private pythonPath: string;
    private outputChannel: vscode.OutputChannel;
    private scope: 'local' | 'global';

    constructor() {
        this.outputChannel = vscode.window.createOutputChannel('Promptly');
        this.pythonPath = this.getPythonPath();
        this.scope = this.getScope();
    }

    private getPythonPath(): string {
        const config = vscode.workspace.getConfiguration('promptly');
        return config.get<string>('pythonPath') || 'python';
    }

    private getScope(): 'local' | 'global' {
        const config = vscode.workspace.getConfiguration('promptly');
        return config.get<'local' | 'global'>('scope') || 'local';
    }

    /**
     * Refresh configuration from VS Code settings
     */
    public refreshConfig(): void {
        this.pythonPath = this.getPythonPath();
        this.scope = this.getScope();
    }

    /**
     * Execute a Promptly CLI command
     */
    private async execute(args: string[]): Promise<BridgeResult<any>> {
        return new Promise((resolve) => {
            const fullArgs = ['-m', 'promptly.cli.main', ...args, '--json'];

            this.outputChannel.appendLine(`> ${this.pythonPath} ${fullArgs.join(' ')}`);

            const process = spawn(this.pythonPath, fullArgs, {
                cwd: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
                env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
            });

            let stdout = '';
            let stderr = '';

            process.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            process.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            process.on('close', (code) => {
                this.outputChannel.appendLine(stdout);
                if (stderr) {
                    this.outputChannel.appendLine(`STDERR: ${stderr}`);
                }

                if (code === 0) {
                    try {
                        // Try to parse JSON output
                        const jsonMatch = stdout.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
                        if (jsonMatch) {
                            const data = JSON.parse(jsonMatch[0]);
                            resolve({ success: true, data });
                        } else {
                            // No JSON, return raw output
                            resolve({ success: true, data: stdout.trim() });
                        }
                    } catch (e) {
                        // Return raw output if JSON parsing fails
                        resolve({ success: true, data: stdout.trim() });
                    }
                } else {
                    resolve({ success: false, error: stderr || stdout || `Exit code: ${code}` });
                }
            });

            process.on('error', (err) => {
                resolve({ success: false, error: err.message });
            });
        });
    }

    // ==================== Prompt Operations ====================

    /**
     * List all prompts
     */
    async listPrompts(): Promise<BridgeResult<Prompt[]>> {
        const result = await this.execute(['list', '--scope', this.scope]);
        if (result.success && Array.isArray(result.data)) {
            return { success: true, data: result.data };
        }
        return { success: true, data: [] };
    }

    /**
     * Get a specific prompt
     */
    async getPrompt(name: string, version?: number): Promise<BridgeResult<Prompt>> {
        const args = ['get', name, '--scope', this.scope];
        if (version !== undefined) {
            args.push('--version', version.toString());
        }
        return this.execute(args);
    }

    /**
     * Add or update a prompt
     */
    async addPrompt(name: string, content: string, message?: string): Promise<BridgeResult<Prompt>> {
        const args = ['add', name, content, '--scope', this.scope];
        if (message) {
            args.push('--message', message);
        }
        return this.execute(args);
    }

    /**
     * Delete a prompt
     */
    async deletePrompt(name: string): Promise<BridgeResult<boolean>> {
        return this.execute(['delete', name, '--scope', this.scope, '--force']);
    }

    /**
     * Get prompt history
     */
    async getPromptHistory(name: string, limit?: number): Promise<BridgeResult<PromptVersion[]>> {
        const args = ['log', name, '--scope', this.scope];
        if (limit !== undefined) {
            args.push('--limit', limit.toString());
        }
        return this.execute(args);
    }

    /**
     * Render a prompt with variables
     */
    async renderPrompt(name: string, variables: Record<string, string>): Promise<BridgeResult<string>> {
        const args = ['get', name, '--scope', this.scope, '--render'];
        for (const [key, value] of Object.entries(variables)) {
            args.push('--var', `${key}=${value}`);
        }
        return this.execute(args);
    }

    // ==================== Branch Operations ====================

    /**
     * List all branches
     */
    async listBranches(): Promise<BridgeResult<string[]>> {
        return this.execute(['branches', '--scope', this.scope]);
    }

    /**
     * Create a new branch
     */
    async createBranch(name: string, fromBranch?: string): Promise<BridgeResult<boolean>> {
        const args = ['branch', name, '--scope', this.scope];
        if (fromBranch) {
            args.push('--from', fromBranch);
        }
        return this.execute(args);
    }

    /**
     * Checkout a branch
     */
    async checkoutBranch(name: string): Promise<BridgeResult<boolean>> {
        return this.execute(['checkout', name, '--scope', this.scope]);
    }

    /**
     * Get current branch
     */
    async getCurrentBranch(): Promise<BridgeResult<string>> {
        const result = await this.execute(['info', '--scope', this.scope]);
        if (result.success && result.data?.current_branch) {
            return { success: true, data: result.data.current_branch };
        }
        return { success: true, data: 'main' };
    }

    // ==================== Chain Operations ====================

    /**
     * List all chains
     */
    async listChains(): Promise<BridgeResult<Chain[]>> {
        const result = await this.execute(['chain', 'list', '--scope', this.scope]);
        if (result.success && Array.isArray(result.data)) {
            return { success: true, data: result.data };
        }
        return { success: true, data: [] };
    }

    /**
     * Create a new chain
     */
    async createChain(name: string, steps: ChainStep[]): Promise<BridgeResult<Chain>> {
        // Convert steps to CLI format
        const stepStrs = steps.map(s => {
            let str = s.prompt_name;
            if (s.output_var) str += `:${s.output_var}`;
            return str;
        });

        return this.execute(['chain', 'create', name, '--steps', stepStrs.join(','), '--scope', this.scope]);
    }

    /**
     * Run a chain
     */
    async runChain(name: string, input: Record<string, string>): Promise<BridgeResult<any>> {
        const args = ['chain', 'run', name, '--scope', this.scope];
        for (const [key, value] of Object.entries(input)) {
            args.push('--input', `${key}=${value}`);
        }
        return this.execute(args);
    }

    // ==================== Skill Operations ====================

    /**
     * List all skills
     */
    async listSkills(): Promise<BridgeResult<Skill[]>> {
        const result = await this.execute(['skill', 'list', '--scope', this.scope]);
        if (result.success && Array.isArray(result.data)) {
            return { success: true, data: result.data };
        }
        return { success: true, data: [] };
    }

    /**
     * Get a skill
     */
    async getSkill(name: string): Promise<BridgeResult<Skill>> {
        return this.execute(['skill', 'get', name, '--scope', this.scope]);
    }

    /**
     * Create a skill
     */
    async createSkill(
        name: string,
        description: string,
        promptTemplate: string,
        inputSchema?: Record<string, any>,
        outputSchema?: Record<string, any>
    ): Promise<BridgeResult<Skill>> {
        const args = [
            'skill', 'add', name,
            '--description', description,
            '--template', promptTemplate,
            '--scope', this.scope
        ];

        if (inputSchema) {
            args.push('--input-schema', JSON.stringify(inputSchema));
        }
        if (outputSchema) {
            args.push('--output-schema', JSON.stringify(outputSchema));
        }

        return this.execute(args);
    }

    /**
     * Run a skill
     */
    async runSkill(name: string, input: Record<string, any>): Promise<BridgeResult<any>> {
        const args = ['skill', 'run', name, '--scope', this.scope];
        for (const [key, value] of Object.entries(input)) {
            args.push('--input', `${key}=${JSON.stringify(value)}`);
        }
        return this.execute(args);
    }

    // ==================== Evaluation Operations ====================

    /**
     * Evaluate a prompt with LLM Judge
     */
    async evalPrompt(name: string, testInput?: string): Promise<BridgeResult<EvalResult>> {
        const args = ['eval', name, '--scope', this.scope];
        if (testInput) {
            args.push('--input', testInput);
        }
        return this.execute(args);
    }

    // ==================== Export/Import Operations ====================

    /**
     * Export a prompt to YAML
     */
    async exportPrompt(name: string): Promise<BridgeResult<string>> {
        return this.execute(['export', name, '--scope', this.scope]);
    }

    /**
     * Import a prompt from YAML
     */
    async importPrompt(yamlContent: string): Promise<BridgeResult<Prompt>> {
        // Write to temp file and import
        const fs = require('fs');
        const os = require('os');
        const tempFile = path.join(os.tmpdir(), `promptly_import_${Date.now()}.yaml`);

        try {
            fs.writeFileSync(tempFile, yamlContent);
            const result = await this.execute(['import', tempFile, '--scope', this.scope]);
            fs.unlinkSync(tempFile);
            return result;
        } catch (e: any) {
            return { success: false, error: e.message };
        }
    }

    // ==================== Info Operations ====================

    /**
     * Get Promptly system info
     */
    async getInfo(): Promise<BridgeResult<any>> {
        return this.execute(['info', '--scope', this.scope]);
    }

    /**
     * Check if Promptly CLI is available
     */
    async checkAvailability(): Promise<boolean> {
        const result = await this.execute(['--version']);
        return result.success;
    }

    /**
     * Show output channel
     */
    showOutput(): void {
        this.outputChannel.show();
    }

    /**
     * Dispose resources
     */
    dispose(): void {
        this.outputChannel.dispose();
    }
}
