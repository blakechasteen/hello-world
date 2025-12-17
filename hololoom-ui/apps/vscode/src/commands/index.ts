import * as vscode from 'vscode';
import { HoloLoomClient } from '../api/holoLoomClient';
import { ChatViewProvider } from '../providers/chatViewProvider';
import { getSelectedText, getEditorContext, formatContextForPrompt } from '../utils/context';
import { getConfig } from '../utils/config';

/**
 * Register all extension commands
 */
export function registerCommands(
  context: vscode.ExtensionContext,
  client: HoloLoomClient,
  chatViewProvider: ChatViewProvider
): void {
  // Open Chat Panel
  context.subscriptions.push(
    vscode.commands.registerCommand('hololoom.openChat', () => {
      // Focus the chat view in the sidebar
      vscode.commands.executeCommand('hololoom.chatView.focus');
    })
  );

  // Query Selection
  context.subscriptions.push(
    vscode.commands.registerCommand('hololoom.query', async () => {
      const selectedText = getSelectedText();

      // If no selection, prompt for query
      const query = selectedText || await vscode.window.showInputBox({
        prompt: 'Enter your query',
        placeHolder: 'What would you like to know?',
      });

      if (!query) {
        return;
      }

      // Focus chat panel and send query
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

      // Focus chat panel and send query in verify mode for better accuracy
      await vscode.commands.executeCommand('hololoom.chatView.focus');
      await chatViewProvider.sendQuery(query, 'verify');
    })
  );

  // Remember Selection
  context.subscriptions.push(
    vscode.commands.registerCommand('hololoom.remember', async () => {
      const selectedText = getSelectedText();

      if (!selectedText) {
        vscode.window.showWarningMessage('Please select some content to remember');
        return;
      }

      const editorContext = getEditorContext();

      // Ask for optional note
      const note = await vscode.window.showInputBox({
        prompt: 'Add a note (optional)',
        placeHolder: 'What should I remember about this?',
      });

      const content = note
        ? `${note}\n\n${selectedText}`
        : selectedText;

      try {
        await client.storeMemory({
          content,
          type: editorContext?.language || 'code',
          metadata: {
            file: editorContext?.file,
            language: editorContext?.language,
            line: editorContext?.lineNumber,
          },
        });

        vscode.window.showInformationMessage('Content saved to memory');
      } catch (error) {
        vscode.window.showErrorMessage(
          `Failed to save memory: ${error instanceof Error ? error.message : 'Unknown error'}`
        );
      }
    })
  );

  // Open Memory Browser
  context.subscriptions.push(
    vscode.commands.registerCommand('hololoom.openMemory', () => {
      vscode.commands.executeCommand('hololoom.memoryView.focus');
    })
  );

  // Reconnect to Server
  context.subscriptions.push(
    vscode.commands.registerCommand('hololoom.reconnect', async () => {
      const config = getConfig();
      client.setBaseUrl(config.apiEndpoint);

      vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: 'Connecting to HoloLoom...',
          cancellable: false,
        },
        async () => {
          try {
            const health = await client.checkHealth();
            if (health.status === 'healthy') {
              vscode.window.showInformationMessage('Connected to HoloLoom');
            } else if (health.status === 'degraded') {
              vscode.window.showWarningMessage('HoloLoom is running in degraded mode');
            } else {
              vscode.window.showErrorMessage('HoloLoom server is unhealthy');
            }
          } catch (error) {
            vscode.window.showErrorMessage(
              `Failed to connect: ${error instanceof Error ? error.message : 'Unknown error'}`
            );
          }
        }
      );
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

      // Show quick pick for reasoning mode
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

  // Search Memories
  context.subscriptions.push(
    vscode.commands.registerCommand('hololoom.searchMemory', async () => {
      const query = await vscode.window.showInputBox({
        prompt: 'Search your memories',
        placeHolder: 'What are you looking for?',
      });

      if (!query) {
        return;
      }

      try {
        const memories = await client.searchMemory({ query, limit: 10 });

        if (memories.length === 0) {
          vscode.window.showInformationMessage('No memories found');
          return;
        }

        // Show quick pick with results
        const items = memories.map((m) => ({
          label: m.content.substring(0, 60) + (m.content.length > 60 ? '...' : ''),
          description: m.type,
          detail: m.content,
          memory: m,
        }));

        const selected = await vscode.window.showQuickPick(items, {
          placeHolder: 'Select a memory to insert',
          matchOnDescription: true,
          matchOnDetail: true,
        });

        if (selected) {
          const editor = vscode.window.activeTextEditor;
          if (editor) {
            editor.edit((editBuilder) => {
              editBuilder.insert(editor.selection.active, selected.memory.content);
            });
          } else {
            // Copy to clipboard if no editor
            await vscode.env.clipboard.writeText(selected.memory.content);
            vscode.window.showInformationMessage('Memory copied to clipboard');
          }
        }
      } catch (error) {
        vscode.window.showErrorMessage(
          `Search failed: ${error instanceof Error ? error.message : 'Unknown error'}`
        );
      }
    })
  );
}
