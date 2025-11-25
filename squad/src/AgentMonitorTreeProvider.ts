/**
 * Agent Monitor Tree Provider
 *
 * Real-time visualization of HoloLoom agent reasoning in VS Code sidebar.
 *
 * Features:
 * - Project-based tree structure
 * - Live status updates via WebSocket
 * - Two-line feed display
 * - Click to view reasoning tree
 * - Auto-reconnect on disconnect
 *
 * Status: Production Ready (November 2025)
 */

import * as vscode from 'vscode';
import WebSocket from 'ws';

// ============================================================================
// Types
// ============================================================================

interface AgentSession {
    agent_id: string;
    project: string;
    query: string;
    mode: string;
    status: string;
    feed_line1: string;
    feed_line2: string;
    current_step: number;
    total_steps: number;
    files: string[];
    start_time: string;
    total_duration_ms?: number;
}

interface ProjectGroup {
    name: string;
    agents: AgentSession[];
}

interface WebSocketMessage {
    type: string;
    agent_id?: string;
    project?: string;
    query?: string;
    mode?: string;
    status?: string;
    line1?: string;
    line2?: string;
    step?: number;
    total_steps?: number;
    [key: string]: any;
}

// ============================================================================
// Tree Items
// ============================================================================

class AgentTreeItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly session?: AgentSession,
        public readonly isProject: boolean = false,
        public readonly isFeed: boolean = false
    ) {
        super(label, collapsibleState);

        if (isProject) {
            this.contextValue = 'project';
            this.iconPath = new vscode.ThemeIcon('folder');
        } else if (isFeed) {
            this.contextValue = 'feed';
            this.iconPath = new vscode.ThemeIcon('comment');
        } else if (session) {
            this.contextValue = 'agent';
            this.iconPath = this.getStatusIcon(session.status);
            this.tooltip = this.getTooltip(session);
            this.description = `${session.mode} (${session.current_step}/${session.total_steps})`;

            // Make agent clickable
            this.command = {
                command: 'hololoom.showAgentDetails',
                title: 'Show Agent Details',
                arguments: [session]
            };
        }
    }

    private getStatusIcon(status: string): vscode.ThemeIcon {
        switch (status) {
            case 'completed':
                return new vscode.ThemeIcon('check', new vscode.ThemeColor('testing.iconPassed'));
            case 'running':
                return new vscode.ThemeIcon('sync~spin', new vscode.ThemeColor('testing.iconQueued'));
            case 'waiting':
            case 'verify':
            case 'research':
                return new vscode.ThemeIcon('clock', new vscode.ThemeColor('testing.iconQueued'));
            case 'failed':
                return new vscode.ThemeIcon('error', new vscode.ThemeColor('testing.iconFailed'));
            default:
                return new vscode.ThemeIcon('circle-outline');
        }
    }

    private getTooltip(session: AgentSession): string {
        const duration = session.total_duration_ms
            ? `${session.total_duration_ms.toFixed(0)}ms`
            : 'In progress';

        return `${session.query}\n\nMode: ${session.mode}\nStatus: ${session.status}\nDuration: ${duration}`;
    }
}

// ============================================================================
// Tree Data Provider
// ============================================================================

export class AgentMonitorTreeProvider implements vscode.TreeDataProvider<AgentTreeItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<AgentTreeItem | undefined | null | void>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    private sessions: Map<string, AgentSession> = new Map();
    private ws: WebSocket | null = null;
    private reconnectTimeout: NodeJS.Timeout | null = null;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 10;

    constructor(
        private serverUrl: string = 'http://localhost:8000',
        private wsUrl: string = 'ws://localhost:8000/ws/monitor'
    ) {
        this.initialize();
    }

    // ========================================================================
    // Initialization
    // ========================================================================

    private async initialize() {
        // Load initial state from REST API
        await this.loadSessions();

        // Connect WebSocket for real-time updates
        this.connectWebSocket();
    }

    private async loadSessions() {
        try {
            const response = await fetch(`${this.serverUrl}/api/monitor/sessions`);
            if (response.ok) {
                const data = await response.json();

                // Clear and rebuild sessions map
                this.sessions.clear();
                for (const session of data.sessions) {
                    this.sessions.set(session.agent_id, session);
                }

                this.refresh();
            }
        } catch (error) {
            console.error('Failed to load sessions:', error);
            vscode.window.showErrorMessage('Failed to connect to HoloLoom server. Is it running?');
        }
    }

    // ========================================================================
    // WebSocket Connection
    // ========================================================================

    private connectWebSocket() {
        if (this.ws) {
            this.ws.close();
        }

        try {
            this.ws = new WebSocket(this.wsUrl);

            this.ws.on('open', () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                vscode.window.setStatusBarMessage('$(pulse) HoloLoom connected', 3000);
            });

            this.ws.on('message', (data: WebSocket.Data) => {
                try {
                    const message: WebSocketMessage = JSON.parse(data.toString());
                    this.handleWebSocketMessage(message);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            });

            this.ws.on('close', () => {
                console.log('WebSocket disconnected');
                this.scheduleReconnect();
            });

            this.ws.on('error', (error) => {
                console.error('WebSocket error:', error);
            });

        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            this.scheduleReconnect();
        }
    }

    private scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            vscode.window.showWarningMessage('Lost connection to HoloLoom server. Click refresh to retry.');
            return;
        }

        // Exponential backoff: 1s, 2s, 4s, 8s, ...
        const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
        this.reconnectAttempts++;

        this.reconnectTimeout = setTimeout(() => {
            console.log(`Reconnecting (attempt ${this.reconnectAttempts})...`);
            this.connectWebSocket();
        }, delay);
    }

    // ========================================================================
    // WebSocket Message Handling
    // ========================================================================

    private handleWebSocketMessage(message: WebSocketMessage) {
        switch (message.type) {
            case 'agent_started':
                this.handleAgentStarted(message);
                break;

            case 'agent_step':
                this.handleAgentStep(message);
                break;

            case 'agent_status':
                this.handleAgentStatus(message);
                break;

            case 'agent_feed':
                this.handleAgentFeed(message);
                break;

            case 'agent_completed':
                this.handleAgentCompleted(message);
                break;

            case 'agent_failed':
                this.handleAgentFailed(message);
                break;

            default:
                console.log('Unknown message type:', message.type);
        }
    }

    private handleAgentStarted(message: WebSocketMessage) {
        const session: AgentSession = {
            agent_id: message.agent_id!,
            project: message.project!,
            query: message.query!,
            mode: message.mode!,
            status: 'running',
            feed_line1: '',
            feed_line2: '',
            current_step: 0,
            total_steps: 0,
            files: message.files || [],
            start_time: message.timestamp || new Date().toISOString()
        };

        this.sessions.set(session.agent_id, session);
        this.refresh();

        vscode.window.setStatusBarMessage(`$(pulse) Agent started: ${session.query.substring(0, 30)}...`, 3000);
    }

    private handleAgentStep(message: WebSocketMessage) {
        const session = this.sessions.get(message.agent_id!);
        if (session) {
            session.current_step = message.step!;
            session.total_steps = message.total_steps!;
            this.refresh();
        }
    }

    private handleAgentStatus(message: WebSocketMessage) {
        const session = this.sessions.get(message.agent_id!);
        if (session) {
            session.status = message.status!;
            if (message.files) {
                session.files = message.files;
            }
            this.refresh();
        }
    }

    private handleAgentFeed(message: WebSocketMessage) {
        const session = this.sessions.get(message.agent_id!);
        if (session) {
            session.feed_line1 = message.line1!;
            session.feed_line2 = message.line2!;
            this.refresh();
        }
    }

    private handleAgentCompleted(message: WebSocketMessage) {
        const session = this.sessions.get(message.agent_id!);
        if (session) {
            session.status = 'completed';
            session.total_duration_ms = message.total_duration_ms;
            this.refresh();

            vscode.window.setStatusBarMessage(`$(check) Agent completed: ${session.query.substring(0, 30)}...`, 3000);
        }
    }

    private handleAgentFailed(message: WebSocketMessage) {
        const session = this.sessions.get(message.agent_id!);
        if (session) {
            session.status = 'failed';
            this.refresh();

            vscode.window.showErrorMessage(`Agent failed: ${message.error?.substring(0, 100)}`);
        }
    }

    // ========================================================================
    // Tree Data Provider Implementation
    // ========================================================================

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: AgentTreeItem): vscode.TreeItem {
        return element;
    }

    async getChildren(element?: AgentTreeItem): Promise<AgentTreeItem[]> {
        if (!element) {
            // Root level: show projects
            return this.getProjectGroups();
        }

        if (element.isProject) {
            // Project level: show agents in this project
            return this.getAgentsForProject(element.label);
        }

        if (element.session && !element.isFeed) {
            // Agent level: show two-line feed
            return this.getFeedForAgent(element.session);
        }

        return [];
    }

    private getProjectGroups(): AgentTreeItem[] {
        // Group sessions by project
        const projects = new Map<string, AgentSession[]>();

        for (const session of this.sessions.values()) {
            if (!projects.has(session.project)) {
                projects.set(session.project, []);
            }
            projects.get(session.project)!.push(session);
        }

        // Create tree items
        const items: AgentTreeItem[] = [];

        for (const [project, agents] of projects.entries()) {
            const label = `${project} (${agents.length})`;
            items.push(new AgentTreeItem(
                label,
                vscode.TreeItemCollapsibleState.Expanded,
                undefined,
                true // isProject
            ));
        }

        return items.sort((a, b) => a.label.localeCompare(b.label));
    }

    private getAgentsForProject(projectLabel: string): AgentTreeItem[] {
        // Extract project name from label (e.g., "mythRL (2)" -> "mythRL")
        const projectName = projectLabel.split(' ')[0];

        const agents = Array.from(this.sessions.values())
            .filter(s => s.project === projectName);

        return agents.map(session => {
            const queryPreview = session.query.length > 50
                ? session.query.substring(0, 50) + '...'
                : session.query;

            return new AgentTreeItem(
                queryPreview,
                vscode.TreeItemCollapsibleState.Collapsed,
                session,
                false // isProject
            );
        });
    }

    private getFeedForAgent(session: AgentSession): AgentTreeItem[] {
        const items: AgentTreeItem[] = [];

        if (session.feed_line1) {
            items.push(new AgentTreeItem(
                session.feed_line1,
                vscode.TreeItemCollapsibleState.None,
                session,
                false,
                true // isFeed
            ));
        }

        if (session.feed_line2) {
            items.push(new AgentTreeItem(
                session.feed_line2,
                vscode.TreeItemCollapsibleState.None,
                session,
                false,
                true // isFeed
            ));
        }

        return items;
    }

    // ========================================================================
    // Public Methods
    // ========================================================================

    async forceRefresh() {
        await this.loadSessions();
    }

    dispose() {
        if (this.ws) {
            this.ws.close();
        }
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
        }
    }
}