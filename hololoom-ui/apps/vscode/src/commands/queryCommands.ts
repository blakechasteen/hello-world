/**
 * Query-related commands
 *
 * These commands handle code analysis operations using HoloLoom.
 */

import * as vscode from 'vscode';
import { HoloLoomClient } from '../api/holoLoomClient';
import { ChatViewProvider } from '../providers/chatViewProvider';
import { getSelectedText, getEditorContext, formatContextForPrompt } from '../utils/context';

/**
 * Register query-related commands
 */
export function registerQueryCommands(
  context: vscode.ExtensionContext,
  client: HoloLoomClient,
  chatViewProvider: ChatViewProvider
): void {
  // Query Selection
  context.subscriptions.push(
    vscode.commands.registerCommand('hololoom.query', async () => {
      const selectedText = getSelectedText();

      const query = selectedText || await vscode.window.showInputBox({
        prompt: 'Enter your query',
        placeHolder: 'What would you like to know?',
      });

      if (!query) {
        return;
      }

      await vscode.commands.executeCommand('hololoom.chatView.focus');
      await chatViewProvider.sendQueryWithContext(query);
    })
  );

  // Explain Selection
  context.subscriptions.push(
    vscode.commands.registerCommand('hololoom.explain', async () => {
      const selectedText = getSelectedText();

      if (!selectedText) {
        vscode.window.showWarningMessage('Please select some code to explain');
        return;
      }

      const editorContext = getEditorContext();
      const contextStr = editorContext ? formatContextForPrompt(editorContext) : '';

      const query = `Please explain this code:\n\n${contextStr}`;

      await vscode.commands.executeCommand('hololoom.chatView.focus');
      await chatViewProvider.sendQuery(query, 'verify');
    })
  );

  // Quick Query from Command Palette
  context.subscriptions.push(
    vscode.commands.registerCommand('hololoom.quickQuery', async () => {
      const query = await vscode.window.showInputBox({
        prompt: 'Enter your query',
        placeHolder: 'Ask HoloLoom anything...',
      });

      if (!query) {
        return;
      }

      const mode = await vscode.window.showQuickPick(
        [
          { label: 'Direct', description: 'Single-pass answer (~150ms)', value: 'direct' },
          { label: 'Verify', description: 'Answer with verification (~600ms)', value: 'verify' },
          { label: 'Research', description: 'Multi-query exploration (~900ms)', value: 'research' },
          { label: 'Plan & Execute', description: 'Goal decomposition (~750ms)', value: 'plan_execute' },
        ],
        {
          placeHolder: 'Select reasoning mode',
        }
      );

      const reasoningMode = mode?.value as 'direct' | 'verify' | 'research' | 'plan_execute' || 'verify';

      await vscode.commands.executeCommand('hololoom.chatView.focus');
      await chatViewProvider.sendQueryWithContext(query, reasoningMode);
    })
  );
}
