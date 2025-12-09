/**
 * SkillsManager - Handles skill discovery, installation, and invocation
 *
 * Connects to the Skills Marketplace API and manages local skill installations.
 */

import * as vscode from 'vscode';
import axios, { AxiosInstance } from 'axios';

// Types matching the marketplace API
export interface Skill {
    skill_id: string;
    name: string;
    description: string;
    version: string;
    category: 'domain' | 'utility' | 'integration' | 'workflow';
    author: string;
    rating: number;
    downloads: number;
    requirements?: string[];
    tags?: string[];
    created_at?: string;
    updated_at?: string;
}

export interface SkillVersion {
    version: string;
    changelog: string;
    published_at: string;
    min_hololoom_version?: string;
}

export interface SkillDetail extends Skill {
    documentation?: string;
    changelog?: SkillVersion[];
    dependencies?: string[];
    examples?: string[];
}

export interface InstalledSkill {
    skill_id: string;
    version: string;
    installed_at: string;
    enabled: boolean;
    config?: Record<string, any>;
}

export interface SkillInvocationResult {
    success: boolean;
    output?: any;
    error?: string;
    duration_ms: number;
}

export type SkillEventType =
    | 'skill_installed'
    | 'skill_removed'
    | 'skill_updated'
    | 'skill_invoked'
    | 'skill_error';

export interface SkillEvent {
    type: SkillEventType;
    skill_id: string;
    timestamp: Date;
    data?: any;
}

export class SkillsManager implements vscode.Disposable {
    private client: AxiosInstance;
    private installedSkills: Map<string, InstalledSkill> = new Map();
    private skillCache: Map<string, Skill> = new Map();
    private cacheExpiry: number = 5 * 60 * 1000; // 5 minutes
    private lastCacheUpdate: number = 0;

    private _onSkillEvent = new vscode.EventEmitter<SkillEvent>();
    readonly onSkillEvent = this._onSkillEvent.event;

    private statusBarItem: vscode.StatusBarItem;

    constructor(
        private marketplaceUrl: string = 'http://localhost:8080',
        private context?: vscode.ExtensionContext
    ) {
        this.client = axios.create({
            baseURL: marketplaceUrl,
            timeout: 10000,
            headers: {
                'Content-Type': 'application/json'
            }
        });

        // Create status bar item for skills
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            99
        );
        this.statusBarItem.command = 'skills.openExplorer';
        this.statusBarItem.tooltip = 'HoloLoom Skills';

        // Load installed skills from storage
        this.loadInstalledSkills();
        this.updateStatusBar();
    }

    // ========== Skill Discovery ==========

    /**
     * Get all available skills from marketplace
     */
    async getAvailableSkills(options?: {
        category?: string;
        search?: string;
        page?: number;
        limit?: number;
    }): Promise<Skill[]> {
        try {
            const response = await this.client.get('/api/skills', {
                params: options
            });

            // Update cache
            for (const skill of response.data) {
                this.skillCache.set(skill.skill_id, skill);
            }
            this.lastCacheUpdate = Date.now();

            return response.data;
        } catch (error: any) {
            console.error('Failed to fetch skills:', error.message);

            // Return cached skills if available
            if (this.skillCache.size > 0) {
                vscode.window.showWarningMessage(
                    'Using cached skill list. Marketplace may be offline.'
                );
                return Array.from(this.skillCache.values());
            }

            throw error;
        }
    }

    /**
     * Get detailed information about a specific skill
     */
    async getSkillDetail(skillId: string): Promise<SkillDetail> {
        try {
            const response = await this.client.get(`/api/skills/${skillId}`);
            return response.data;
        } catch (error: any) {
            console.error(`Failed to fetch skill ${skillId}:`, error.message);
            throw error;
        }
    }

    /**
     * Search skills by query
     */
    async searchSkills(query: string): Promise<Skill[]> {
        return this.getAvailableSkills({ search: query });
    }

    /**
     * Get skills by category
     */
    async getSkillsByCategory(category: string): Promise<Skill[]> {
        return this.getAvailableSkills({ category });
    }

    // ========== Skill Installation ==========

    /**
     * Install a skill from the marketplace
     */
    async installSkill(skillId: string, version?: string): Promise<void> {
        try {
            // Check if already installed
            if (this.installedSkills.has(skillId)) {
                const installed = this.installedSkills.get(skillId)!;
                if (!version || installed.version === version) {
                    vscode.window.showInformationMessage(
                        `Skill "${skillId}" is already installed (v${installed.version})`
                    );
                    return;
                }
            }

            // Show progress
            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: `Installing skill: ${skillId}`,
                cancellable: false
            }, async (progress) => {
                progress.report({ increment: 0, message: 'Fetching skill...' });

                // Get skill details
                const skill = await this.getSkillDetail(skillId);
                const targetVersion = version || skill.version;

                progress.report({ increment: 30, message: 'Checking dependencies...' });

                // Check dependencies
                if (skill.dependencies && skill.dependencies.length > 0) {
                    for (const dep of skill.dependencies) {
                        if (!this.installedSkills.has(dep)) {
                            const install = await vscode.window.showWarningMessage(
                                `Skill "${skillId}" requires "${dep}". Install it?`,
                                'Yes', 'No'
                            );
                            if (install === 'Yes') {
                                await this.installSkill(dep);
                            } else {
                                throw new Error(`Missing dependency: ${dep}`);
                            }
                        }
                    }
                }

                progress.report({ increment: 30, message: 'Installing...' });

                // Register with marketplace
                await this.client.post(`/api/skills/${skillId}/install`, {
                    version: targetVersion
                });

                progress.report({ increment: 30, message: 'Configuring...' });

                // Save to installed skills
                const installedSkill: InstalledSkill = {
                    skill_id: skillId,
                    version: targetVersion,
                    installed_at: new Date().toISOString(),
                    enabled: true
                };

                this.installedSkills.set(skillId, installedSkill);
                await this.saveInstalledSkills();

                progress.report({ increment: 10, message: 'Complete!' });

                // Emit event
                this._onSkillEvent.fire({
                    type: 'skill_installed',
                    skill_id: skillId,
                    timestamp: new Date(),
                    data: { version: targetVersion }
                });

                this.updateStatusBar();
            });

            vscode.window.showInformationMessage(
                `Successfully installed skill: ${skillId}`
            );

        } catch (error: any) {
            vscode.window.showErrorMessage(
                `Failed to install skill "${skillId}": ${error.message}`
            );
            throw error;
        }
    }

    /**
     * Remove an installed skill
     */
    async removeSkill(skillId: string): Promise<void> {
        if (!this.installedSkills.has(skillId)) {
            vscode.window.showWarningMessage(`Skill "${skillId}" is not installed`);
            return;
        }

        const confirm = await vscode.window.showWarningMessage(
            `Remove skill "${skillId}"? This cannot be undone.`,
            'Remove', 'Cancel'
        );

        if (confirm !== 'Remove') {
            return;
        }

        try {
            // Unregister from marketplace
            await this.client.post(`/api/skills/${skillId}/uninstall`);

            // Remove from local storage
            this.installedSkills.delete(skillId);
            await this.saveInstalledSkills();

            // Emit event
            this._onSkillEvent.fire({
                type: 'skill_removed',
                skill_id: skillId,
                timestamp: new Date()
            });

            this.updateStatusBar();

            vscode.window.showInformationMessage(
                `Successfully removed skill: ${skillId}`
            );

        } catch (error: any) {
            vscode.window.showErrorMessage(
                `Failed to remove skill "${skillId}": ${error.message}`
            );
        }
    }

    /**
     * Update an installed skill to a new version
     */
    async updateSkill(skillId: string, version?: string): Promise<void> {
        if (!this.installedSkills.has(skillId)) {
            throw new Error(`Skill "${skillId}" is not installed`);
        }

        const installed = this.installedSkills.get(skillId)!;
        const skill = await this.getSkillDetail(skillId);
        const targetVersion = version || skill.version;

        if (installed.version === targetVersion) {
            vscode.window.showInformationMessage(
                `Skill "${skillId}" is already at version ${targetVersion}`
            );
            return;
        }

        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: `Updating skill: ${skillId}`,
            cancellable: false
        }, async (progress) => {
            progress.report({ increment: 50, message: 'Updating...' });

            // Update registration
            await this.client.post(`/api/skills/${skillId}/update`, {
                version: targetVersion
            });

            // Update local record
            installed.version = targetVersion;
            await this.saveInstalledSkills();

            progress.report({ increment: 50, message: 'Complete!' });

            // Emit event
            this._onSkillEvent.fire({
                type: 'skill_updated',
                skill_id: skillId,
                timestamp: new Date(),
                data: {
                    from_version: installed.version,
                    to_version: targetVersion
                }
            });
        });

        vscode.window.showInformationMessage(
            `Successfully updated skill "${skillId}" to v${targetVersion}`
        );
    }

    // ========== Skill Invocation ==========

    /**
     * Invoke a skill with given input
     */
    async invokeSkill(
        skillId: string,
        input: Record<string, any>
    ): Promise<SkillInvocationResult> {
        if (!this.installedSkills.has(skillId)) {
            return {
                success: false,
                error: `Skill "${skillId}" is not installed`,
                duration_ms: 0
            };
        }

        const installed = this.installedSkills.get(skillId)!;
        if (!installed.enabled) {
            return {
                success: false,
                error: `Skill "${skillId}" is disabled`,
                duration_ms: 0
            };
        }

        const startTime = Date.now();

        try {
            const response = await this.client.post(`/api/skills/${skillId}/invoke`, {
                input,
                config: installed.config
            });

            const duration = Date.now() - startTime;

            // Emit event
            this._onSkillEvent.fire({
                type: 'skill_invoked',
                skill_id: skillId,
                timestamp: new Date(),
                data: { duration_ms: duration, success: true }
            });

            return {
                success: true,
                output: response.data,
                duration_ms: duration
            };

        } catch (error: any) {
            const duration = Date.now() - startTime;

            // Emit error event
            this._onSkillEvent.fire({
                type: 'skill_error',
                skill_id: skillId,
                timestamp: new Date(),
                data: { error: error.message, duration_ms: duration }
            });

            return {
                success: false,
                error: error.message,
                duration_ms: duration
            };
        }
    }

    /**
     * Get a quick pick for skill selection
     */
    async selectSkill(filter?: {
        installed?: boolean;
        category?: string;
    }): Promise<Skill | undefined> {
        let skills: Skill[];

        if (filter?.installed) {
            skills = await this.getInstalledSkillDetails();
        } else {
            skills = await this.getAvailableSkills({
                category: filter?.category
            });
        }

        const items = skills.map(skill => ({
            label: skill.name,
            description: `v${skill.version} - ${skill.category}`,
            detail: skill.description,
            skill
        }));

        const selected = await vscode.window.showQuickPick(items, {
            placeHolder: 'Select a skill',
            matchOnDescription: true,
            matchOnDetail: true
        });

        return selected?.skill;
    }

    // ========== Skill Management ==========

    /**
     * Get all installed skills
     */
    getInstalledSkills(): InstalledSkill[] {
        return Array.from(this.installedSkills.values());
    }

    /**
     * Get installed skill details (with full skill info)
     */
    async getInstalledSkillDetails(): Promise<Skill[]> {
        const skills: Skill[] = [];

        for (const installed of this.installedSkills.values()) {
            try {
                // Check cache first
                if (this.skillCache.has(installed.skill_id)) {
                    skills.push(this.skillCache.get(installed.skill_id)!);
                } else {
                    const skill = await this.getSkillDetail(installed.skill_id);
                    skills.push(skill);
                }
            } catch (error) {
                console.error(`Failed to get details for ${installed.skill_id}`);
            }
        }

        return skills;
    }

    /**
     * Check if a skill is installed
     */
    isInstalled(skillId: string): boolean {
        return this.installedSkills.has(skillId);
    }

    /**
     * Enable or disable a skill
     */
    async setSkillEnabled(skillId: string, enabled: boolean): Promise<void> {
        const installed = this.installedSkills.get(skillId);
        if (!installed) {
            throw new Error(`Skill "${skillId}" is not installed`);
        }

        installed.enabled = enabled;
        await this.saveInstalledSkills();

        vscode.window.showInformationMessage(
            `Skill "${skillId}" ${enabled ? 'enabled' : 'disabled'}`
        );
    }

    /**
     * Update skill configuration
     */
    async configureSkill(
        skillId: string,
        config: Record<string, any>
    ): Promise<void> {
        const installed = this.installedSkills.get(skillId);
        if (!installed) {
            throw new Error(`Skill "${skillId}" is not installed`);
        }

        installed.config = { ...installed.config, ...config };
        await this.saveInstalledSkills();
    }

    // ========== Health Check ==========

    /**
     * Check marketplace connectivity
     */
    async healthCheck(): Promise<boolean> {
        try {
            await this.client.get('/api/health');
            return true;
        } catch {
            return false;
        }
    }

    // ========== Private Helpers ==========

    private async loadInstalledSkills(): Promise<void> {
        if (this.context) {
            const stored = this.context.globalState.get<InstalledSkill[]>(
                'hololoom.installedSkills',
                []
            );

            for (const skill of stored) {
                this.installedSkills.set(skill.skill_id, skill);
            }
        }
    }

    private async saveInstalledSkills(): Promise<void> {
        if (this.context) {
            await this.context.globalState.update(
                'hololoom.installedSkills',
                Array.from(this.installedSkills.values())
            );
        }
    }

    private updateStatusBar(): void {
        const count = this.installedSkills.size;
        this.statusBarItem.text = `$(extensions) ${count} Skills`;
        this.statusBarItem.show();
    }

    // ========== Dispose ==========

    dispose(): void {
        this.statusBarItem.dispose();
        this._onSkillEvent.dispose();
    }
}

/**
 * Factory function to create SkillsManager
 */
export function createSkillsManager(
    context: vscode.ExtensionContext
): SkillsManager {
    const config = vscode.workspace.getConfiguration('skills');
    const marketplaceUrl = config.get<string>(
        'marketplaceUrl',
        'http://localhost:8080'
    );

    return new SkillsManager(marketplaceUrl, context);
}
