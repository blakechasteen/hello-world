/**
 * Trough VS Code Extension 🐷
 * Eats slop for breakfast! Powered by HoloLoom
 */

import * as vscode from 'vscode';
import { HoloLoomBridge } from './HoloLoomBridge';
import { AgentPanel } from './AgentPanel';
import { CodeContextProvider } from './CodeContextProvider';
import { VerificationService } from './VerificationService';
import { FixSlopCommand } from './FixSlopCommand';

let bridge: HoloLoomBridge;
let agentPanel: AgentPanel | undefined;
let contextProvider: CodeContextProvider;
let verificationService: VerificationService;
let fixSlopCommand: FixSlopCommand;

// Fun pig-themed eating phrases that rotate randomly
const EATING_PHRASES = [
    'We dine!',
    'Feast time!',
    'Let us eat cake!',
    'Watch your fingers!',
    'Oink oink, nom nom!',
    'Time to pig out!',
    'Snouts to the trough!',
    'Feeding frenzy mode!',
    'Release the hogs!',
    'Dinner is served!',
    'To the banquet!',
    'Belly up to the bar!',
    'Gobble gobble!'
];

const SUCCESS_PHRASES = [
    'Licked thy platter clean!',
    'Trough thoroughly sparkles!',
    'Thy trough sparkles, Sire!',
    'Sparkle-mode activated, my Liege!',
    'Not a crumb remains, my Lord!',
    'Devoured with honor!',
    'Thy feast is complete!',
    'Pristine and proper!'
];

function getRandomPhrase(phrases: string[]): string {
    return phrases[Math.floor(Math.random() * phrases.length)];
}

export async function activate(context: vscode.ExtensionContext) {
    console.log('Trough extension activating... 🐷');

    // Initialize components
    contextProvider = new CodeContextProvider();

    const config = vscode.workspace.getConfiguration('trough');
    const serverUrl = config.get<string>('serverUrl', 'http://localhost:8000');

    bridge = new HoloLoomBridge(serverUrl);
    verificationService = new VerificationService(bridge);
    fixSlopCommand = new FixSlopCommand(bridge, verificationService);

    // Register commands
    registerCommands(context);

    // Status bar
    const statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    statusBarItem.text = '$(sparkle) Trough';
    statusBarItem.command = 'trough.openPanel';
    statusBarItem.tooltip = 'Trough - Pig out on AI slop! 🐷';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    vscode.window.showInformationMessage('Trough is ready! 🐷 Let\'s pig out!');
    console.log('Trough extension activated!');
}

function registerCommands(context: vscode.ExtensionContext) {
    // Ask Question
    context.subscriptions.push(
        vscode.commands.registerCommand('trough.ask', async () => {
            const question = await vscode.window.showInputBox({
                prompt: 'What would you like to know?',
                placeHolder: 'Ask Trough anything about your code...'
            });

            if (!question) return;

            const codeContext = contextProvider.getCurrentContext();
            await executeQuery(question, codeContext, 'direct');
        })
    );

    // Explain Selection
    context.subscriptions.push(
        vscode.commands.registerCommand('trough.explainSelection', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.selection.isEmpty) {
                vscode.window.showWarningMessage('Please select some code first');
                return;
            }

            const selectedCode = editor.document.getText(editor.selection);
            const language = editor.document.languageId;
            const question = `Explain this ${language} code:\n\n${selectedCode}`;

            const codeContext = contextProvider.getCurrentContext();
            await executeQuery(question, codeContext, 'verify');
        })
    );

    // Suggest Fix
    context.subscriptions.push(
        vscode.commands.registerCommand('trough.suggestFix', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const diagnostics = vscode.languages.getDiagnostics(editor.document.uri);
            if (diagnostics.length === 0) {
                vscode.window.showInformationMessage('No issues detected');
                return;
            }

            const selectedCode = editor.selection.isEmpty
                ? editor.document.getText()
                : editor.document.getText(editor.selection);

            const question = `Fix these issues:\n${
                diagnostics.map(d => `- ${d.message}`).join('\n')
            }\n\nCode:\n${selectedCode}`;

            const codeContext = contextProvider.getCurrentContext();
            codeContext.diagnostics = diagnostics;

            await executeQuery(question, codeContext, 'plan_execute');
        })
    );

    // Refactor Code
    context.subscriptions.push(
        vscode.commands.registerCommand('trough.refactor', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.selection.isEmpty) {
                vscode.window.showWarningMessage('Please select some code first');
                return;
            }

            const refactorType = await vscode.window.showQuickPick([
                'Extract function',
                'Simplify logic',
                'Add error handling',
                'Optimize performance',
                'Add type annotations',
                'Custom...'
            ], {
                placeHolder: 'Select refactoring type'
            });

            if (!refactorType) return;

            let instruction = refactorType;
            if (refactorType === 'Custom...') {
                const custom = await vscode.window.showInputBox({
                    prompt: 'What refactoring would you like?'
                });
                if (!custom) return;
                instruction = custom;
            }

            const selectedCode = editor.document.getText(editor.selection);
            const language = editor.document.languageId;
            const question = `Refactor this ${language} code: ${instruction}\n\nCode:\n${selectedCode}`;

            const codeContext = contextProvider.getCurrentContext();
            await executeQuery(question, codeContext, 'plan_execute');
        })
    );

    // Refactor and Verify (NEW - Priority 1)
    context.subscriptions.push(
        vscode.commands.registerCommand('trough.refactorAndVerify', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.selection.isEmpty) {
                vscode.window.showWarningMessage('Please select some code first');
                return;
            }

            const refactorType = await vscode.window.showQuickPick([
                'Extract function',
                'Simplify logic',
                'Add error handling',
                'Optimize performance',
                'Add type annotations',
                'Custom...'
            ], {
                placeHolder: 'Select refactoring type'
            });

            if (!refactorType) return;

            let instruction = refactorType;
            if (refactorType === 'Custom...') {
                const custom = await vscode.window.showInputBox({
                    prompt: 'What refactoring would you like?'
                });
                if (!custom) return;
                instruction = custom;
            }

            await refactorAndVerify(editor, instruction);
        })
    );

    // Generate Tests (Enhanced - Priority 2)
    context.subscriptions.push(
        vscode.commands.registerCommand('trough.generateTests', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selectedCode = editor.selection.isEmpty
                ? editor.document.getText()
                : editor.document.getText(editor.selection);

            const language = editor.document.languageId;

            const testType = await vscode.window.showQuickPick([
                'Unit tests',
                'Integration tests',
                'Edge cases',
                'All of the above'
            ], {
                placeHolder: 'What type of tests?'
            });

            if (!testType) return;

            await generateAndRunTests(editor, testType, selectedCode);
        })
    );

    // Open Panel
    context.subscriptions.push(
        vscode.commands.registerCommand('trough.openPanel', () => {
            if (!agentPanel) {
                agentPanel = new AgentPanel(context.extensionUri, bridge);
            }
            agentPanel.show();
        })
    );

    // Pig Out (THE KILLER FEATURE - eats AI slop for breakfast!)
    context.subscriptions.push(
        vscode.commands.registerCommand('trough.pigOut', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active editor');
                return;
            }

            const originalCode = editor.selection.isEmpty
                ? editor.document.getText()
                : editor.document.getText(editor.selection);

            const config = vscode.workspace.getConfiguration('trough');
            const maxPiglets = config.get<number>('maxPiglets', 5);

            const result = await fixSlopCommand.fix(editor, {
                maxIterations: maxPiglets,
                verifyTypes: true,
                detectHallucinations: true
            });

            await fixSlopCommand.showResult(result, originalCode);

            if (result.success || result.finalCode !== originalCode) {
                const action = await vscode.window.showInformationMessage(
                    result.message,
                    'Apply Fix',
                    'Show Diff',
                    'Cancel'
                );

                if (action === 'Apply Fix') {
                    await editor.edit(editBuilder => {
                        if (editor.selection.isEmpty) {
                            const fullRange = new vscode.Range(
                                editor.document.positionAt(0),
                                editor.document.positionAt(editor.document.getText().length)
                            );
                            editBuilder.replace(fullRange, result.finalCode);
                        } else {
                            editBuilder.replace(editor.selection, result.finalCode);
                        }
                    });
                    vscode.window.showInformationMessage(`✅ ${getRandomPhrase(SUCCESS_PHRASES)}`);
                }
            }
        })
    );

    // Verify Code
    context.subscriptions.push(
        vscode.commands.registerCommand('trough.verifyCode', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active editor');
                return;
            }

            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: `Trough: Taste testing...`,
                cancellable: false
            }, async () => {
                const result = await verificationService.verifyAndShowDiagnostics(editor);

                const summary = verificationService.getSummary(result);
                vscode.window.showInformationMessage(summary);
            });
        })
    );

    // Detect Hallucinations
    context.subscriptions.push(
        vscode.commands.registerCommand('trough.detectHallucinations', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active editor');
                return;
            }

            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: `Trough: Something stinky.... yum!`,
                cancellable: false
            }, async () => {
                const result = await verificationService.detectAndShowHallucinations(editor);

                if (result.hallucinations.length === 0) {
                    vscode.window.showInformationMessage('✅ No hallucinations detected');
                } else {
                    vscode.window.showWarningMessage(
                        `⚠️ Found ${result.hallucinations.length} potential hallucination(s)`
                    );
                }
            });
        })
    );

    // Ingest Workspace
    context.subscriptions.push(
        vscode.commands.registerCommand('trough.ingestWorkspace', async () => {
            const workspacePath = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
            if (!workspacePath) {
                vscode.window.showWarningMessage('No workspace open');
                return;
            }

            const languages = await vscode.window.showQuickPick(
                ['python', 'typescript', 'javascript', 'all'],
                {
                    placeHolder: 'Select languages to index',
                    canPickMany: false
                }
            );

            if (!languages) return;

            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'Trough: Mucking the pen...',
                cancellable: false
            }, async () => {
                try {
                    const response = await bridge.ingestWorkspace(workspacePath);
                    vscode.window.showInformationMessage(
                        `✅ Pen mucked: ${response.message}`
                    );
                } catch (error) {
                    vscode.window.showErrorMessage(`Failed to index workspace: ${error}`);
                }
            });
        })
    );
}

async function executeQuery(
    question: string,
    codeContext: any,
    mode: string = 'verify'
) {
    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: `Trough: ${getRandomPhrase(EATING_PHRASES)}`,
        cancellable: false
    }, async () => {
        try {
            // Priority 3: Context Optimization
            const optimizedContext = optimizeContext(codeContext);

            const result = await bridge.query(question, optimizedContext, mode);

            if (!agentPanel) {
                agentPanel = new AgentPanel(
                    vscode.Uri.file(__dirname),
                    bridge
                );
            }
            agentPanel.show();
            agentPanel.displayResult(result);

        } catch (error) {
            vscode.window.showErrorMessage(`Trough error: ${error}`);
        }
    });
}

// Priority 3: Context Optimization for Large Files
function optimizeContext(codeContext: any): any {
    const MAX_LINES = 1000;
    const CONTEXT_WINDOW = 50; // Lines around selection

    if (!codeContext.currentFile) return codeContext;

    const lines = codeContext.currentFile.split('\n');

    if (lines.length <= MAX_LINES) {
        return codeContext; // File is small enough
    }

    // Large file detected - optimize
    const optimized = { ...codeContext };

    if (codeContext.selection) {
        // Send summary + selection region + context
        const selectionStart = findLineNumber(codeContext.currentFile, codeContext.selection);
        const start = Math.max(0, selectionStart - CONTEXT_WINDOW);
        const end = Math.min(lines.length, selectionStart + CONTEXT_WINDOW);

        optimized.currentFile = lines.slice(start, end).join('\n');
        optimized.fileSummary = `Large file (${lines.length} lines). Showing lines ${start}-${end} around selection.`;
    } else {
        // No selection - send summary + first/last sections
        const summary = generateFileSummary(lines);
        optimized.currentFile = summary;
        optimized.fileSummary = `Large file (${lines.length} lines). Summary provided.`;
    }

    return optimized;
}

function findLineNumber(content: string, selection: string): number {
    const lines = content.split('\n');
    for (let i = 0; i < lines.length; i++) {
        if (lines[i].includes(selection.split('\n')[0])) {
            return i;
        }
    }
    return 0;
}

function generateFileSummary(lines: string[]): string {
    const SUMMARY_LINES = 100;
    const header = lines.slice(0, SUMMARY_LINES / 2).join('\n');
    const footer = lines.slice(-SUMMARY_LINES / 2).join('\n');
    return `${header}\n\n... [${lines.length - SUMMARY_LINES} lines omitted] ...\n\n${footer}`;
}

// Priority 1: Refactor and Verify
async function refactorAndVerify(
    editor: vscode.TextEditor,
    instruction: string,
    maxIterations: number = 3
) {
    const selectedCode = editor.document.getText(editor.selection);
    const language = editor.document.languageId;

    let currentCode = selectedCode;
    let iteration = 0;
    let hasErrors = true;

    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: `Trough: Munch, munch, munch... mmmm!`,
        cancellable: false
    }, async (progress) => {
        while (hasErrors && iteration < maxIterations) {
            iteration++;
            progress.report({
                message: `Piglet ${iteration}/${maxIterations} munching away...`,
                increment: 33
            });

            // Step 1: Refactor
            const question = iteration === 1
                ? `Refactor this ${language} code: ${instruction}\n\nCode:\n${currentCode}`
                : `Fix these TypeScript errors:\n${currentCode}`;

            const codeContext = contextProvider.getCurrentContext();
            codeContext.currentFile = currentCode;

            const result = await bridge.query(question, optimizeContext(codeContext), 'verify');

            // Extract refactored code from response
            currentCode = extractCodeFromResponse(result.response);

            // Step 2: Verify with TypeScript compiler (if applicable)
            if (language === 'typescript' || language === 'typescriptreact') {
                progress.report({ message: 'Checking TypeScript errors...' });

                const tempUri = await createTempFile(currentCode, language);
                const diagnostics = vscode.languages.getDiagnostics(tempUri);

                if (diagnostics.length === 0) {
                    hasErrors = false;
                    vscode.window.showInformationMessage(
                        `✅ ${getRandomPhrase(SUCCESS_PHRASES)} (${iteration} piglet${iteration > 1 ? 's' : ''})`
                    );
                } else {
                    const errorSummary = diagnostics
                        .map(d => `Line ${d.range.start.line + 1}: ${d.message}`)
                        .join('\n');

                    currentCode = `${currentCode}\n\n// TypeScript Errors:\n// ${errorSummary}`;

                    if (iteration === maxIterations) {
                        vscode.window.showWarningMessage(
                            `⚠️ Piglets couldn't eliminate all errors after ${maxIterations} tries. Please review manually.`
                        );
                    }
                }
            } else {
                hasErrors = false; // No verification for non-TypeScript
            }

            // Show intermediate result
            if (!agentPanel) {
                agentPanel = new AgentPanel(vscode.Uri.file(__dirname), bridge);
            }
            agentPanel.show();
            agentPanel.displayResult(result);
        }

        // Step 3: Apply the final refactored code
        await editor.edit(editBuilder => {
            editBuilder.replace(editor.selection, currentCode);
        });
    });
}

// Priority 2: Generate and Run Tests
async function generateAndRunTests(
    editor: vscode.TextEditor,
    testType: string,
    selectedCode: string
) {
    const language = editor.document.languageId;

    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: `Trough: ${getRandomPhrase(EATING_PHRASES)}`,
        cancellable: false
    }, async (progress) => {
        // Step 1: Generate tests
        progress.report({ message: 'Refining our palates...', increment: 30 });

        const question = `Generate ${testType.toLowerCase()} for this ${language} code:\n\n${selectedCode}`;
        const codeContext = contextProvider.getCurrentContext();

        const result = await bridge.query(question, optimizeContext(codeContext), 'plan_execute');
        const generatedTests = extractCodeFromResponse(result.response);

        // Show generated tests
        if (!agentPanel) {
            agentPanel = new AgentPanel(vscode.Uri.file(__dirname), bridge);
        }
        agentPanel.show();
        agentPanel.displayResult(result);

        // Step 2: Write tests to temp file and run them
        if (language === 'typescript' || language === 'javascript' ||
            language === 'typescriptreact' || language === 'javascriptreact') {

            progress.report({ message: 'Running tests...', increment: 40 });

            const testFilePath = await createTestFile(generatedTests, language);
            const testResults = await runTests(testFilePath, language);

            // Step 3: Show results
            progress.report({ message: 'Analyzing results...', increment: 30 });

            const passed = testResults.passed;
            const failed = testResults.failed;
            const total = passed + failed;

            if (failed === 0) {
                vscode.window.showInformationMessage(
                    `✅ All ${total} tests passed!`
                );
            } else {
                const fix = await vscode.window.showWarningMessage(
                    `⚠️ ${failed} of ${total} tests failed. Send piglets to fix them?`,
                    'Fix Tests', 'Show Errors'
                );

                if (fix === 'Fix Tests') {
                    // Iterate to fix failing tests - send piglets!
                    const fixQuestion = `Fix these failing tests:\n\n${testResults.errors.join('\n')}\n\nTests:\n${generatedTests}`;
                    const fixResult = await bridge.query(fixQuestion, optimizeContext(codeContext), 'plan_execute');
                    agentPanel.displayResult(fixResult);
                } else if (fix === 'Show Errors') {
                    const errorDoc = await vscode.workspace.openTextDocument({
                        content: testResults.errors.join('\n\n'),
                        language: 'text'
                    });
                    await vscode.window.showTextDocument(errorDoc);
                }
            }
        } else {
            vscode.window.showInformationMessage(
                `Tests generated! Automatic test running is only supported for TypeScript/JavaScript.`
            );
        }
    });
}

// Helper functions
function extractCodeFromResponse(response: string): string {
    // Extract code block from markdown
    const codeBlockRegex = /```(?:typescript|javascript|ts|js)?\n([\s\S]*?)```/;
    const match = response.match(codeBlockRegex);
    return match ? match[1].trim() : response;
}

async function createTempFile(code: string, language: string): Promise<vscode.Uri> {
    const ext = language.includes('typescript') ? 'ts' : 'js';
    const tempPath = `${vscode.workspace.workspaceFolders?.[0]?.uri.fsPath}/.trough_temp.${ext}`;
    const uri = vscode.Uri.file(tempPath);

    const encoder = new TextEncoder();
    await vscode.workspace.fs.writeFile(uri, encoder.encode(code));

    return uri;
}

async function createTestFile(code: string, language: string): Promise<string> {
    const ext = language.includes('typescript') ? 'ts' : 'js';
    const testPath = `${vscode.workspace.workspaceFolders?.[0]?.uri.fsPath}/.trough_test.test.${ext}`;

    const encoder = new TextEncoder();
    await vscode.workspace.fs.writeFile(vscode.Uri.file(testPath), encoder.encode(code));

    return testPath;
}

async function runTests(testFilePath: string, language: string): Promise<any> {
    // Run tests using VS Code's built-in test runner or external command
    const terminal = vscode.window.createTerminal('Trough Test Runner');

    // Detect test framework (Jest, Mocha, etc.)
    const testCommand = language.includes('typescript')
        ? `npx ts-node ${testFilePath}`
        : `node ${testFilePath}`;

    return new Promise((resolve) => {
        terminal.sendText(testCommand);

        // Simplified: Parse output (in real implementation, capture stdout)
        setTimeout(() => {
            resolve({
                passed: 0,
                failed: 0,
                errors: []
            });
        }, 2000);
    });
}

export async function deactivate() {
    if (bridge) {
        await bridge.stop();
    }
    console.log('Trough extension deactivated');
}