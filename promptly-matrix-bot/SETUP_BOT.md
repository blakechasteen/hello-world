# Promptly Matrix Bot - Setup Guide

## Step 1: Configure Your Bot Account

Edit `config/matrix_config.json` with your bot's credentials:

```json
{
  "homeserver": "https://matrix.org",
  "user": "@your-bot-username:matrix.org",
  "password": "your_bot_password",
  "auto_join_rooms": true
}
```

**Replace**:
- `@your-bot-username:matrix.org` - Your bot's Matrix user ID
- `your_bot_password` - Your bot's password

## Step 2: Install Dependencies

```bash
# Install Matrix SDK
pip install matrix-nio[e2e] aiohttp

# Install other bot dependencies (if not already installed)
pip install -r requirements.txt
```

## Step 3: Run the Bot

```bash
# From the promptly-matrix-bot directory
python run_bot.py
```

You should see:
```
🤖 Starting Promptly bot: @your-bot:matrix.org
🌐 Homeserver: https://matrix.org
🔑 Logging in...
✅ Login successful!
🚀 Starting bot event loop...
💬 Bot is now listening for messages
   (Press Ctrl+C to stop)
```

## Step 4: Test the Bot

In your Matrix room with the bot:

1. Send: `!help`
2. Bot should respond with available commands

## Available Commands

- `!help` - Show available commands
- `!optimize <command>` - Get optimization suggestions
- `!run <command>` - Execute command with reliability checks
- `!code-review <code>` - Review code for issues
- `!explain <topic>` - Get AI explanation

## Troubleshooting

**Login failed**:
- Check your bot's username and password in `config/matrix_config.json`
- Ensure username includes homeserver: `@bot:matrix.org`

**Bot not responding**:
- Make sure the bot is in the room
- Try inviting the bot again
- Check bot logs for errors

**Dependencies missing**:
```bash
pip install matrix-nio[e2e] aiohttp
```

## Configuration Options

### Auto-Join Rooms
```json
"auto_join_rooms": true  // Bot auto-accepts room invites
```

### Custom Homeserver
```json
"homeserver": "https://your-server.com"
```

## Next Steps

Once the bot is running:

1. **Test commands** in your Matrix room
2. **Configure GitHub integration** (Phase 5)
3. **Set up monitoring** (Phase 6B)
4. **Enable authentication** (Phase 6A)

## Production Deployment

For production, see:
- `PHASE_6_COMPLETE_SUMMARY.md` - Full Phase 6 deployment guide
- `docker-compose.prod.yml` - Docker deployment
- `config/prometheus.yml` - Monitoring setup
