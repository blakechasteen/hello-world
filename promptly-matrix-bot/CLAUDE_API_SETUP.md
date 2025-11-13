# Claude API Integration - Setup Guide

The Promptly Matrix Bot now uses the **Anthropic API** directly instead of requiring a Claude CLI installation. This provides reliable, always-available Claude capabilities right from Matrix chat!

## What Changed

- ✅ **Before**: Required Claude CLI installation (which doesn't actually exist)
- ✅ **Now**: Uses Anthropic API directly via Python SDK
- ✅ **Result**: Claude commands work reliably from Matrix chat

## Quick Setup

### 1. Get an Anthropic API Key

1. Go to [https://console.anthropic.com/](https://console.anthropic.com/)
2. Sign in or create an account
3. Navigate to "API Keys" section
4. Click "Create Key"
5. Copy your API key (starts with `sk-ant-...`)

### 2. Add API Key to Environment

Edit `.env` file in the `promptly-matrix-bot` directory:

```bash
# Claude API Integration (Phase 2 ChatOps)
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Security Note**: Never commit your API key to git. The `.env` file is already in `.gitignore`.

### 3. Restart the Bot

```bash
python run_bot.py
```

You should see:
```
INFO - Claude API available
```

## Using Claude Commands from Matrix

Once configured, you can use these commands from any Matrix room with the bot:

### Code Review

```
@promptly review bot/git_handler.py
```

Claude will analyze the code and provide:
- Code quality assessment
- Potential bugs
- Performance considerations
- Security concerns
- Improvement suggestions

### Code Explanation

```
@promptly explain bot/claude_bridge.py
```

Claude will explain:
- What the code does
- How it works
- Key design decisions
- Important implementation details

### Code Refactoring

```
@promptly refactor bot/git_handler.py "use async/await"
```

Claude will provide:
- Refactoring approach explanation
- Complete refactored code
- Rationale for changes

## API Pricing

The Anthropic API is a paid service with generous free tier and pay-as-you-go pricing:

- **Model**: Claude 3.5 Sonnet (latest)
- **Input**: ~$3 per million tokens
- **Output**: ~$15 per million tokens

**Typical Costs:**
- Code review (500-line file): ~$0.01 - $0.02
- Code explanation: ~$0.005 - $0.015
- Refactoring: ~$0.015 - $0.03

For most individual use, this is extremely affordable!

## Architecture

### Old CLI-Based Approach (Didn't Work)

```
Matrix Bot → CLI subprocess → Claude Code (doesn't exist)
```

### New API-Based Approach (Works!)

```
Matrix Bot → Anthropic SDK → Claude API → Response
```

## Files Modified

1. **bot/claude_bridge.py** - Rewritten to use Anthropic API
   - Uses `anthropic` Python SDK
   - Direct API calls with proper error handling
   - Secure file reading within repo boundaries

2. **bot/promptly_bot.py** - Updated initialization
   - Uses `ANTHROPIC_API_KEY` instead of `CLAUDE_PATH`
   - Better error messages

3. **bot/claude_methods.py** - Updated error messages
   - Directs users to API console instead of download page

4. **.env** - Added API key configuration
   - Placeholder for `ANTHROPIC_API_KEY`

## Features

### ✅ Code Review
- Comprehensive analysis
- Best practices checking
- Security review
- Performance considerations

### ✅ Code Explanation
- Clear explanations
- Design rationale
- Implementation details

### ✅ Code Refactoring
- Refactoring suggestions
- Complete refactored code
- Change explanations

### ✅ Security
- Files must be within repo boundaries
- No arbitrary file access
- Safe error handling

## Testing

Test the API integration:

```bash
cd promptly-matrix-bot
python bot/claude_bridge.py
```

This will:
1. Check if Anthropic SDK is installed
2. Check if API key is set
3. Test a simple chat query

## Troubleshooting

### "Claude API not available"

**Cause**: API key not set or invalid

**Fix**:
1. Check `.env` has `ANTHROPIC_API_KEY=sk-ant-...`
2. Verify key is valid at [https://console.anthropic.com/](https://console.anthropic.com/)
3. Restart bot

### "Review failed: File not found"

**Cause**: File path is relative to `GIT_REPO_PATH`

**Fix**: Use paths relative to the repo root:
```
@promptly review bot/git_handler.py  # Correct
@promptly review /absolute/path/...  # Won't work
```

### "Review failed: File outside repository"

**Cause**: Security check preventing access outside repo

**Fix**: This is intentional! Only files within `GIT_REPO_PATH` can be reviewed.

### API Rate Limits

**Cause**: Too many requests in short time

**Fix**:
- Wait a few seconds between requests
- Anthropic has generous rate limits for most use cases

## Benefits Over CLI Approach

1. **Actually Works**: No phantom CLI dependency
2. **Reliable**: Direct API calls, no subprocess issues
3. **Better Errors**: Clear messages about what went wrong
4. **Secure**: Proper file access controls
5. **Up-to-Date**: Uses latest Claude 3.5 Sonnet model
6. **Maintainable**: Pure Python, no external dependencies

## Next Steps

1. ✅ Get API key from console.anthropic.com
2. ✅ Add to `.env`
3. ✅ Restart bot
4. ✅ Test with `@promptly review bot/git_handler.py` in Matrix

Enjoy your working Claude Code integration! 🎉
