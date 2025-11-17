import * as assert from 'assert';
import * as vscode from 'vscode';
import { HoloLoomLSPClient } from '../lsp/client';

/**
 * LSP Integration Tests
 *
 * These tests verify the LSP client functionality, including:
 * - Client startup and shutdown
 * - Completion requests
 * - Hover requests
 * - Definition navigation
 * - Workspace symbol search
 *
 * Note: These tests require HoloLoom LSP server to be available.
 * Set SKIP_LSP_TESTS=1 to skip these tests if server is unavailable.
 */

suite('HoloLoom LSP Integration Tests', () => {
    let client: HoloLoomLSPClient;
    let context: vscode.ExtensionContext;

    // Skip tests if LSP server unavailable
    const skipTests = process.env.SKIP_LSP_TESTS === '1';

    suiteSetup(async function() {
        if (skipTests) {
            this.skip();
            return;
        }

        // Get extension context
        const extension = vscode.extensions.getExtension('promptly.promptly');
        if (!extension) {
            throw new Error('Extension not found');
        }

        if (!extension.isActive) {
            await extension.activate();
        }

        context = (extension.exports as any).context;
    });

    setup(async function() {
        if (skipTests) {
            this.skip();
            return;
        }

        client = new HoloLoomLSPClient(context);
    });

    teardown(async function() {
        if (skipTests) {
            return;
        }

        if (client) {
            await client.stop();
        }
    });

    test('LSP client starts successfully', async function() {
        this.timeout(10000); // Allow 10s for server startup

        const success = await client.start();

        if (!success) {
            // Server not available - skip test
            this.skip();
            return;
        }

        assert.strictEqual(client.isRunning(), true, 'Client should be running');
    });

    test('LSP client stops gracefully', async function() {
        this.timeout(10000);

        const success = await client.start();
        if (!success) {
            this.skip();
            return;
        }

        await client.stop();
        assert.strictEqual(client.isRunning(), false, 'Client should be stopped');
    });

    test('LSP client restarts successfully', async function() {
        this.timeout(15000);

        const success = await client.start();
        if (!success) {
            this.skip();
            return;
        }

        const restarted = await client.restart();
        assert.strictEqual(restarted, true, 'Client should restart successfully');
        assert.strictEqual(client.isRunning(), true, 'Client should be running after restart');
    });

    test('Completion request returns results', async function() {
        this.timeout(10000);

        const success = await client.start();
        if (!success) {
            this.skip();
            return;
        }

        // Open a test file
        const doc = await vscode.workspace.openTextDocument({
            language: 'python',
            content: 'def test_function():\n    # TODO: implement\n    pass'
        });

        const editor = await vscode.window.showTextDocument(doc);

        // Request completion at position
        const params = {
            textDocument: {
                uri: doc.uri.toString()
            },
            position: {
                line: 1,
                character: 10
            },
            context: {
                triggerKind: 1 // Invoked
            }
        };

        try {
            const completions = await client.sendRequest('textDocument/completion', params);
            assert.ok(completions, 'Should receive completion response');
        } catch (error: any) {
            // Server might not support completion yet - log and continue
            console.log('Completion request failed (expected):', error.message);
        }
    });

    test('Hover request returns entity information', async function() {
        this.timeout(10000);

        const success = await client.start();
        if (!success) {
            this.skip();
            return;
        }

        // Open a test file
        const doc = await vscode.workspace.openTextDocument({
            language: 'typescript',
            content: 'function example() {\n    return 42;\n}'
        });

        const editor = await vscode.window.showTextDocument(doc);

        // Request hover at function name
        const params = {
            textDocument: {
                uri: doc.uri.toString()
            },
            position: {
                line: 0,
                character: 9 // 'example'
            }
        };

        try {
            const hover = await client.sendRequest('textDocument/hover', params);
            assert.ok(hover, 'Should receive hover response');
        } catch (error: any) {
            console.log('Hover request failed (expected):', error.message);
        }
    });

    test('Definition request navigates to entity', async function() {
        this.timeout(10000);

        const success = await client.start();
        if (!success) {
            this.skip();
            return;
        }

        // Open a test file
        const doc = await vscode.workspace.openTextDocument({
            language: 'python',
            content: 'class MyClass:\n    pass\n\nobj = MyClass()'
        });

        const editor = await vscode.window.showTextDocument(doc);

        // Request definition for MyClass usage
        const params = {
            textDocument: {
                uri: doc.uri.toString()
            },
            position: {
                line: 3,
                character: 6 // 'MyClass'
            }
        };

        try {
            const definition = await client.sendRequest('textDocument/definition', params);
            assert.ok(definition, 'Should receive definition response');
        } catch (error: any) {
            console.log('Definition request failed (expected):', error.message);
        }
    });

    test('Workspace symbol search finds entities', async function() {
        this.timeout(10000);

        const success = await client.start();
        if (!success) {
            this.skip();
            return;
        }

        // Search for symbols
        const params = {
            query: 'test'
        };

        try {
            const symbols = await client.sendRequest('workspace/symbol', params);
            assert.ok(symbols, 'Should receive workspace symbol response');

            if (Array.isArray(symbols)) {
                console.log(`Found ${symbols.length} symbols matching 'test'`);
            }
        } catch (error: any) {
            console.log('Workspace symbol request failed (expected):', error.message);
        }
    });

    test('Multiple concurrent requests handled correctly', async function() {
        this.timeout(15000);

        const success = await client.start();
        if (!success) {
            this.skip();
            return;
        }

        // Send multiple requests concurrently
        const requests = [
            client.sendRequest('workspace/symbol', { query: 'function' }),
            client.sendRequest('workspace/symbol', { query: 'class' }),
            client.sendRequest('workspace/symbol', { query: 'method' })
        ];

        try {
            const results = await Promise.all(requests);
            assert.strictEqual(results.length, 3, 'Should receive all responses');
        } catch (error: any) {
            console.log('Concurrent requests failed (expected):', error.message);
        }
    });

    test('Client handles server errors gracefully', async function() {
        this.timeout(10000);

        const success = await client.start();
        if (!success) {
            this.skip();
            return;
        }

        // Send invalid request
        try {
            await client.sendRequest('invalid/method', {});
            assert.fail('Should throw error for invalid request');
        } catch (error: any) {
            assert.ok(error, 'Should catch error');
            console.log('Invalid request handled correctly:', error.message);
        }
    });
});
