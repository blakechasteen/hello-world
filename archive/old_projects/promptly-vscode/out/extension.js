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
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const PromptTreeProvider_1 = require("./promptLibrary/PromptTreeProvider");
const PromptlyBridge_1 = require("./api/PromptlyBridge");
const ExecutionClient_1 = require("./api/ExecutionClient");
const ExecutionPanel_1 = require("./webviews/ExecutionPanel");
let bridge;
let executionClient;
async function activate(context) {
    console.log('Promptly extension activating...');
    // Initialize Python bridge
    bridge = new PromptlyBridge_1.PromptlyBridge();
    await bridge.start();
    // Initialize execution client
    executionClient = new ExecutionClient_1.ExecutionClient();
    // Create tree data provider
    const promptTreeProvider = new PromptTreeProvider_1.PromptTreeProvider(bridge);
    // Register tree view
    const treeView = vscode.window.createTreeView('promptLibrary', {
        treeDataProvider: promptTreeProvider,
        showCollapseAll: true
    });
    // Register refresh command
    const refreshCommand = vscode.commands.registerCommand('promptly.refresh', async () => {
        promptTreeProvider.refresh();
        vscode.window.showInformationMessage('Prompt library refreshed');
    });
    // Register view prompt command
    const viewPromptCommand = vscode.commands.registerCommand('promptly.viewPrompt', async (item) => {
        if (item && item.promptName) {
            const promptData = await bridge.getPrompt(item.promptName);
            if (promptData) {
                const doc = await vscode.workspace.openTextDocument({
                    content: promptData.content,
                    language: 'markdown'
                });
                await vscode.window.showTextDocument(doc, { preview: true });
            }
        }
    });
    // Register execution panel command
    const openExecutionCommand = vscode.commands.registerCommand('promptly.openExecution', () => {
        ExecutionPanel_1.ExecutionPanel.createOrShow(context.extensionUri, executionClient);
    });
    context.subscriptions.push(treeView, refreshCommand, viewPromptCommand, openExecutionCommand);
    vscode.window.showInformationMessage('Promptly is ready!');
    console.log('Promptly extension activated');
}
async function deactivate() {
    if (bridge) {
        await bridge.stop();
    }
}
//# sourceMappingURL=extension.js.map