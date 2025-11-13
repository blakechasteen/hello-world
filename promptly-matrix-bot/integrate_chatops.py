#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatOps Integration Script
Integrates Git and Claude Code handlers into promptly_bot.py
"""

import sys
import io

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def integrate_chatops():
    """Integrate Git and Claude handlers into promptly_bot.py"""

    # Read the current file
    with open('bot/promptly_bot.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Add Git and Claude initialization to __init__
    init_marker = "        logger.info(f\"Initialized Promptly bot: {user_id}\")"

    if "# Initialize Git handler (ChatOps Phase 1)" not in content:
        git_claude_init = """
        # Initialize Git handler (ChatOps Phase 1)
        git_repo_path = os.getenv("GIT_REPO_PATH")
        try:
            from .git_handler import GitHandler
            self.git_handler = GitHandler(git_repo_path) if git_repo_path else None
            if self.git_handler:
                logger.info(f"Git handler initialized for repo: {git_repo_path}")
        except ValueError as e:
            logger.warning(f"Git handler init failed: {e}")
            self.git_handler = None
        except Exception as e:
            logger.warning(f"Git handler init error: {e}")
            self.git_handler = None

        # Initialize Claude Code bridge (ChatOps Phase 2)
        claude_path = os.getenv("CLAUDE_PATH", "claude")
        try:
            from .claude_bridge import ClaudeBridge
            self.claude_bridge = ClaudeBridge(
                claude_path=claude_path,
                repo_path=git_repo_path
            )
            if self.claude_bridge.is_available():
                logger.info("Claude Code CLI available")
            else:
                logger.warning("Claude Code CLI not found")
                self.claude_bridge = None
        except Exception as e:
            logger.warning(f"Claude Bridge init failed: {e}")
            self.claude_bridge = None

"""
        content = content.replace(init_marker, git_claude_init + "        " + init_marker)
        print("✅ Added Git and Claude initialization to __init__")
    else:
        print("⏭️  Git/Claude initialization already present")

    # 2. Add Git command routing
    routing_marker = "        elif cmd_type == 'list':\n            return await self.cmd_list(command, room)"

    if "elif cmd_type == 'git-status':" not in content:
        git_routing = """
        # Git commands (ChatOps Phase 1)
        elif cmd_type == 'git-status':
            return await self.cmd_git_status(command, room)
        elif cmd_type == 'git-log':
            return await self.cmd_git_log(command, room)
        elif cmd_type == 'git-diff':
            return await self.cmd_git_diff(command, room)
        elif cmd_type == 'git-branch':
            return await self.cmd_git_branch(command, room)
        elif cmd_type == 'git-commit':
            return await self.cmd_git_commit(command, room)
        elif cmd_type == 'git-push':
            return await self.cmd_git_push(command, room)
        elif cmd_type == 'git-pull':
            return await self.cmd_git_pull(command, room)
        # Claude Code commands (ChatOps Phase 2)
        elif cmd_type == 'claude-review':
            return await self.cmd_claude_review(command, room)
        elif cmd_type == 'claude-explain':
            return await self.cmd_claude_explain(command, room)
        elif cmd_type == 'claude-refactor':
            return await self.cmd_claude_refactor(command, room)"""

        content = content.replace(routing_marker, routing_marker + git_routing)
        print("✅ Added Git and Claude command routing")
    else:
        print("⏭️  Git/Claude routing already present")

    # 3. Add Git command methods (if not present)
    if "async def cmd_git_status" not in content:
        # Read git methods
        with open('bot/git_methods.py', 'r', encoding='utf-8') as f:
            git_methods = f.read()

        # Extract just the methods (skip the docstring at top)
        git_methods = git_methods.split('# ========== Git Commands')[1]

        # Find insertion point (after cmd_chat method)
        chat_method_end = content.find('    async def send_message(')
        if chat_method_end == -1:
            print("❌ Could not find insertion point for Git methods")
            return False

        # Insert Git methods with proper indentation
        indented_git_methods = "    # ========== Git Commands" + git_methods
        content = content[:chat_method_end] + indented_git_methods + "\n\n" + content[chat_method_end:]
        print("✅ Added 7 Git command methods")
    else:
        print("⏭️  Git methods already present")

    # Write the updated content
    with open('bot/promptly_bot.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("\n✅ Integration complete!")
    return True

if __name__ == '__main__':
    try:
        success = integrate_chatops()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Integration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
