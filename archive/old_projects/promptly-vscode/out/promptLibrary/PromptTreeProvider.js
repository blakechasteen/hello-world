"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.PromptTreeItem = exports.PromptTreeProvider = void 0;
const vscode = __importStar(require("vscode"));
class PromptTreeProvider {
    constructor(bridge) {
        this.bridge = bridge;
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
    }
    refresh() {
        this._onDidChangeTreeData.fire();
    }
    getTreeItem(element) {
        return element;
    }
    async getChildren(element) {
        if (!element) {
            // Root level: show branches
            const prompts = await this.bridge.listPrompts();
            // Group by branch
            const branches = new Map();
            for (const prompt of prompts) {
                const branch = prompt.branch || 'main';
                if (!branches.has(branch)) {
                    branches.set(branch, []);
                }
                branches.get(branch).push(prompt);
            }
            // Create branch items
            const items = [];
            for (const [branch, branchPrompts] of branches) {
                items.push(new PromptTreeItem(branch, vscode.TreeItemCollapsibleState.Expanded, 'branch', branchPrompts));
            }
            return items;
        }
        else if (element.contextValue === 'branch') {
            // Branch level: show prompts
            return element.children.map(prompt => new PromptTreeItem(prompt.name, vscode.TreeItemCollapsibleState.None, 'prompt', [], prompt));
        }
        return [];
    }
}
exports.PromptTreeProvider = PromptTreeProvider;
class PromptTreeItem extends vscode.TreeItem {
    constructor(label, collapsibleState, contextValue, children = [], promptData) {
        super(label, collapsibleState);
        this.label = label;
        this.collapsibleState = collapsibleState;
        this.contextValue = contextValue;
        this.children = children;
        this.promptData = promptData;
        if (contextValue === 'branch') {
            this.iconPath = new vscode.ThemeIcon('git-branch');
            this.description = `${children.length} prompts`;
            this.tooltip = this.label;
        }
        else if (contextValue === 'prompt') {
            this.iconPath = new vscode.ThemeIcon('file-text');
            this.command = {
                command: 'promptly.viewPrompt',
                title: 'View Prompt',
                arguments: [{ promptName: promptData.name }]
            };
            if (promptData.tags && promptData.tags.length > 0) {
                this.description = promptData.tags.join(', ');
            }
            // Set tooltip with full details
            this.tooltip = `${promptData.name}\nBranch: ${promptData.branch}\nTags: ${promptData.tags?.join(', ') || 'none'}`;
        }
    }
}
exports.PromptTreeItem = PromptTreeItem;
//# sourceMappingURL=PromptTreeProvider.js.map