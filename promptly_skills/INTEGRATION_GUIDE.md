# Promptly Strategy Framework - Integration Guide

**Simple, elegant integrations for popular platforms**

---

## 📋 Table of Contents

1. [VS Code Extension](#vs-code-extension)
2. [Matrix Bot](#matrix-bot)
3. [Slack Bot](#slack-bot)
4. [Discord Bot](#discord-bot)
5. [API Server](#api-server)

---

## VS Code Extension

### Overview

Integrate Promptly strategies directly into VS Code for enhanced code assistance.

### Quick Setup

1. **Create Extension Structure**:
```
vscode-promptly/
├── package.json
├── src/
│   ├── extension.ts
│   └── promptly.ts
└── README.md
```

2. **package.json**:
```json
{
  "name": "promptly-strategies",
  "displayName": "Promptly Strategy Framework",
  "description": "Enhanced prompting with auto-detected strategies",
  "version": "1.0.0",
  "engines": { "vscode": "^1.60.0" },
  "activationEvents": ["onCommand:promptly.enhance"],
  "main": "./out/extension.js",
  "contributes": {
    "commands": [
      {
        "command": "promptly.enhance",
        "title": "Promptly: Enhance with Strategy"
      },
      {
        "command": "promptly.auto",
        "title": "Promptly: Auto-Detect Strategy"
      }
    ],
    "keybindings": [
      {
        "command": "promptly.auto",
        "key": "ctrl+shift+p",
        "mac": "cmd+shift+p"
      }
    ]
  }
}
```

3. **src/extension.ts**:
```typescript
import * as vscode from 'vscode';
import { PromptlyClient } from './promptly';

export function activate(context: vscode.ExtensionContext) {
    const client = new PromptlyClient('http://localhost:5000');

    // Enhanced prompt command
    let enhance = vscode.commands.registerCommand('promptly.enhance', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        // Get selection or show input
        let query = editor.document.getText(editor.selection);
        if (!query) {
            query = await vscode.window.showInputBox({
                prompt: 'Enter your query',
                placeHolder: 'e.g., explain this code'
            }) || '';
        }

        if (!query) return;

        // Pick strategy
        const strategies = await client.getStrategies();
        const items = strategies.map((s: any) => ({
            label: s.name,
            description: s.description
        }));
        items.unshift({ label: 'auto', description: 'Auto-detect best strategy' });

        const selected = await vscode.window.quickPick(items, {
            placeHolder: 'Select a strategy'
        });

        if (!selected) return;

        // Enhance
        const result = await client.enhance(query, selected.label);

        // Show result in new editor
        const doc = await vscode.workspace.openTextDocument({
            content: result.content,
            language: 'markdown'
        });
        await vscode.window.showTextDocument(doc);
    });

    // Auto-detect command
    let auto = vscode.commands.registerCommand('promptly.auto', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const query = editor.document.getText(editor.selection);
        if (!query) return;

        const result = await client.enhance(query, 'auto');

        const doc = await vscode.workspace.openTextDocument({
            content: result.content,
            language: 'markdown'
        });
        await vscode.window.showTextDocument(doc);
    });

    context.subscriptions.push(enhance, auto);
}
```

4. **src/promptly.ts**:
```typescript
import axios from 'axios';

export class PromptlyClient {
    constructor(private baseUrl: string) {}

    async getStrategies() {
        const res = await axios.get(`${this.baseUrl}/api/strategies`);
        return res.data.strategies;
    }

    async enhance(query: string, strategy: string) {
        const res = await axios.post(`${this.baseUrl}/api/enhance`, {
            query,
            strategy
        });
        return res.data;
    }
}
```

5. **Build and Install**:
```bash
npm install
npm run compile
vsce package
code --install-extension promptly-strategies-1.0.0.vsix
```

---

## Matrix Bot

### Overview

Integrate Promptly into Matrix chat for collaborative prompting.

### Quick Setup

1. **Install Dependencies**:
```bash
pip install matrix-nio
```

2. **Create Bot** (`matrix_bot.py`):
```python
import asyncio
from nio import AsyncClient, MatrixRoom, RoomMessageText
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from HoloLoom.prompting.registry import get_registry
from HoloLoom.prompting.auto_detect import AutoDetector
from HoloLoom.prompting.strategy import StrategyContext

class PromptlyBot:
    def __init__(self, homeserver, username, password):
        self.client = AsyncClient(homeserver, username)
        self.password = password
        self.registry = get_registry()
        self.detector = AutoDetector(registry=self.registry)

    async def message_callback(self, room: MatrixRoom, event: RoomMessageText):
        # Ignore own messages
        if event.sender == self.client.user:
            return

        message = event.body.strip()

        # Command: !promptly <strategy> <query>
        if message.startswith('!promptly'):
            parts = message.split(maxsplit=2)
            if len(parts) < 3:
                await self.client.room_send(
                    room.room_id,
                    "m.room.message",
                    {"msgtype": "m.text", "body": "Usage: !promptly <strategy|auto> <query>"}
                )
                return

            strategy_name = parts[1]
            query = parts[2]

            # Auto-detect or use specific strategy
            context = StrategyContext(query=query)

            if strategy_name == 'auto':
                suggestions = await self.detector.detect(context, top_k=1)
                if suggestions:
                    strategy_name = suggestions[0][0]
                else:
                    await self.client.room_send(
                        room.room_id,
                        "m.room.message",
                        {"msgtype": "m.text", "body": "❌ No strategies matched"}
                    )
                    return

            strategy = self.registry.get(strategy_name)
            if not strategy:
                await self.client.room_send(
                    room.room_id,
                    "m.room.message",
                    {"msgtype": "m.text", "body": f"❌ Strategy '{strategy_name}' not found"}
                )
                return

            # Enhance
            result = await strategy.enhance(context)

            # Send result (truncate if too long)
            content = result.enhanced_query
            if len(content) > 4000:
                content = content[:4000] + "\n\n... (truncated)"

            response = f"**Strategy**: {strategy_name}\\n**Confidence**: {result.confidence:.2f}\\n\\n{content}"

            await self.client.room_send(
                room.room_id,
                "m.room.message",
                {
                    "msgtype": "m.text",
                    "body": response,
                    "format": "org.matrix.custom.html",
                    "formatted_body": f"<strong>Strategy</strong>: {strategy_name}<br/><strong>Confidence</strong>: {result.confidence:.2f}<br/><br/><pre>{content}</pre>"
                }
            )

    async def start(self):
        print("Logging in...")
        await self.client.login(self.password)

        print("Setting up callbacks...")
        self.client.add_event_callback(self.message_callback, RoomMessageText)

        print("Syncing...")
        await self.client.sync_forever(timeout=30000)

async def main():
    bot = PromptlyBot(
        homeserver="https://matrix.org",
        username="@promptly:matrix.org",
        password="your_password_here"
    )
    await bot.start()

if __name__ == '__main__':
    asyncio.run(main())
```

3. **Run Bot**:
```bash
python matrix_bot.py
```

4. **Usage in Matrix**:
```
!promptly auto explain neural networks
!promptly deep how do transformers work?
!promptly scaffold solve this problem step by step
```

---

## Slack Bot

### Overview

Integrate as a Slack slash command for team collaboration.

### Quick Setup

1. **Create Slack App** at https://api.slack.com/apps

2. **Add Slash Command**: `/promptly <strategy> <query>`

3. **Create Handler** (`slack_bot.py`):
```python
from flask import Flask, request, jsonify
import asyncio
from HoloLoom.prompting.registry import get_registry
from HoloLoom.prompting.strategy import StrategyContext

app = Flask(__name__)
registry = get_registry()

@app.route('/promptly', methods=['POST'])
def promptly_command():
    # Parse Slack command
    text = request.form.get('text', '')
    parts = text.split(maxsplit=1)

    if len(parts) < 2:
        return jsonify({
            "response_type": "ephemeral",
            "text": "Usage: /promptly <strategy|auto> <query>"
        })

    strategy_name, query = parts

    # Enhance query
    context = StrategyContext(query=query)

    if strategy_name == 'auto':
        # Auto-detect (simplified)
        strategy_name = 'deep'  # Default

    strategy = registry.get(strategy_name)
    if not strategy:
        return jsonify({
            "response_type": "ephemeral",
            "text": f"❌ Strategy '{strategy_name}' not found"
        })

    result = asyncio.run(strategy.enhance(context))

    # Return to Slack
    return jsonify({
        "response_type": "in_channel",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Strategy*: {strategy_name}\\n*Confidence*: {result.confidence:.2f}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"```{result.enhanced_query[:3000]}```"
                }
            }
        ]
    })

if __name__ == '__main__':
    app.run(port=3000)
```

4. **Configure Slack** to point to your server endpoint

---

## Discord Bot

### Overview

Discord bot integration for gaming/dev communities.

### Quick Setup

1. **Install**:
```bash
pip install discord.py
```

2. **Create Bot** (`discord_bot.py`):
```python
import discord
from discord.ext import commands
import asyncio
from HoloLoom.prompting.registry import get_registry
from HoloLoom.prompting.strategy import StrategyContext

bot = commands.Bot(command_prefix='!')
registry = get_registry()

@bot.command()
async def promptly(ctx, strategy: str, *, query: str):
    """Enhance query with strategy."""
    if strategy == 'auto':
        # Auto-detect (simplified)
        strategy = 'deep'

    strat = registry.get(strategy)
    if not strat:
        await ctx.send(f"❌ Strategy '{strategy}' not found")
        return

    context = StrategyContext(query=query)
    result = await strat.enhance(context)

    # Send result (truncate if needed)
    content = result.enhanced_query
    if len(content) > 2000:
        content = content[:2000] + "\\n\\n... (truncated)"

    await ctx.send(f"**Strategy**: {strategy}\\n**Confidence**: {result.confidence:.2f}\\n\\n```{content}```")

bot.run('YOUR_BOT_TOKEN')
```

3. **Usage**:
```
!promptly auto explain neural networks
!promptly deep how do transformers work?
```

---

## API Server

### Overview

RESTful API for any client to use Promptly strategies.

### Endpoints

#### GET /api/strategies
List all available strategies.

**Response**:
```json
{
  "strategies": [
    {
      "name": "deep",
      "description": "Deliberate Over-Instruction",
      "category": "meta-prompting"
    }
  ]
}
```

#### POST /api/enhance
Enhance a query with a strategy.

**Request**:
```json
{
  "query": "explain neural networks",
  "strategy": "deep"
}
```

**Response**:
```json
{
  "strategy": "deep",
  "confidence": 0.95,
  "improvement": 0.55,
  "content": "# Exhaustive Deep Analysis...",
  "metadata": {
    "strategy": "deep",
    "sections": 7
  }
}
```

### Running

```bash
python web_server.py
```

Then access at http://localhost:5000

---

## General Integration Pattern

All integrations follow this simple pattern:

```python
from HoloLoom.prompting.registry import get_registry
from HoloLoom.prompting.strategy import StrategyContext

# 1. Get registry
registry = get_registry()

# 2. Get user query
query = "user's question"

# 3. Select strategy (manual or auto)
strategy = registry.get('deep')  # or auto-detect

# 4. Enhance
context = StrategyContext(query=query)
result = await strategy.enhance(context)

# 5. Return result
print(result.enhanced_query)
print(f"Confidence: {result.confidence}")
```

**That's it!** Simple, elegant, composable.

---

## Tips for Integration

1. **Keep it simple** - The framework is already elegant, don't over-complicate
2. **Auto-detect by default** - Let the framework choose the best strategy
3. **Show confidence** - Display confidence scores to users
4. **Truncate long results** - Some strategies produce long outputs
5. **Cache frequently used** - Cache common query+strategy combinations
6. **Learn from feedback** - Use AutoDetector's feedback mechanism

---

## License

MIT - Integrate freely into your projects!
