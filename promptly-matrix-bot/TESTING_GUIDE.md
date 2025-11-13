# Testing Guide - Your New ChatOps Platform!

## Quick Start

1. Start the bot (if not running):
   ```
   python run_bot.py
   ```

2. Open Element (or your Matrix client)
3. Find your room with @promptlybot
4. Start testing!

## Test Commands

### Test 1: Check Bot is Alive
```
@promptly help
```

### Test 2: Git Status (Safe)
```
@promptly git status
```

### Test 3: Git Log (Safe)
```
@promptly git log
```

### Test 4: Git Diff (Safe)
```
@promptly git diff
```

### Test 5: Git Branches (Safe)
```
@promptly git branch
```

### Test 6: Conversational AI
```
@promptly what is Thompson Sampling?
```

### Test 7: Claude Code Review (if Claude installed)
```
@promptly review bot/git_handler.py
```

### Test 8: Alternative Format
```
!git status
!help
```

## Advanced Tests (Careful - These Write!)

### Git Commit (Creates real commit!)
```
@promptly git commit "Test from chat"
```

### Git Push (Pushes to remote!)
```
@promptly git push
```

## Troubleshooting

**Bot not responding?**
- Check bot is running
- Restart: Ctrl+C then python run_bot.py

**"Git not configured"?**
- Check .env has GIT_REPO_PATH set

**"Claude Code not available"?**
- Install from https://claude.ai/download

## Success Checklist

- [ ] Git status works
- [ ] Git log shows commits
- [ ] Conversational AI works
- [ ] Bot responds to @ and ! commands

## You're Done!

You now have a full ChatOps development environment!

Next: Test it live in Matrix! Start with "@promptly git status"
