#!/usr/bin/env python3
"""
Claude Code Integration Script

Adds Claude Code commands to Promptly Matrix Bot.
"""

import re


def apply_claude_integration():
    """Apply Claude Code integration"""

    print("=== Claude Code Integration ===\n")

    # Step 1: Update command_parser.py
    print("Step 1: Adding Claude commands to parser...")

    with open('bot/command_parser.py', 'r', encoding='utf-8') as f:
        parser_content = f.read()

    if 'claude-review' not in parser_content:
        # Add command patterns
        git_pull_line = "        'git-pull': r'(?:@promptly(?:bot)?\s+git\s+pull|!git\s+pull)',\n    }"
        claude_commands = """        'git-pull': r'(?:@promptly(?:bot)?\s+git\s+pull|!git\s+pull)',
        # Claude Code commands
        'claude-review': r'(?:@promptly(?:bot)?\s+(?:claude\s+)?review\s+(.+)|!review\s+(.+))',
        'claude-explain': r'(?:@promptly(?:bot)?\s+(?:claude\s+)?explain\s+(.+)|!explain\s+(.+))',
        'claude-refactor': r'(?:@promptly(?:bot)?\s+(?:claude\s+)?refactor\s+(\S+)\s+"([^"]+)"|!refactor\s+(\S+)\s+"([^"]+)")',
    }"""

        parser_content = parser_content.replace(git_pull_line, claude_commands, 1)

        # Add command extraction
        git_pull_extract = "        elif cmd_type == 'git-pull':\n            return {'type': 'git-pull'}\n\n        else:"

        claude_extract = """        elif cmd_type == 'git-pull':
            return {'type': 'git-pull'}

        # Claude Code commands
        elif cmd_type == 'claude-review':
            file_path = groups[0] if groups[0] else groups[1]
            return {
                'type': 'claude-review',
                'file_path': file_path.strip()
            }

        elif cmd_type == 'claude-explain':
            file_path = groups[0] if groups[0] else groups[1]
            return {
                'type': 'claude-explain',
                'file_path': file_path.strip()
            }

        elif cmd_type == 'claude-refactor':
            file_path = groups[0] if groups[0] else groups[2]
            instruction = groups[1] if groups[1] else groups[3]
            return {
                'type': 'claude-refactor',
                'file_path': file_path.strip(),
                'instruction': instruction
            }

        else:"""

        parser_content = parser_content.replace(git_pull_extract, claude_extract, 1)

        with open('bot/command_parser.py', 'w', encoding='utf-8') as f:
            f.write(parser_content)

        print("[OK] Added Claude commands to parser")
    else:
        print("[SKIP] Claude commands already in parser")

    # Step 2: Update promptly_bot.py
    print("\nStep 2: Adding Claude bridge to bot...")

    with open('bot/promptly_bot.py', 'r', encoding='utf-8') as f:
        bot_content = f.read()

    # Add import
    if 'from .claude_bridge import ClaudeBridge' not in bot_content:
        git_handler_import = '        from .git_handler import GitHandler\n'
        claude_import = '        from .git_handler import GitHandler\n        from .claude_bridge import ClaudeBridge\n'
        bot_content = bot_content.replace(git_handler_import, claude_import, 1)
        print("[OK] Added ClaudeBridge import")

    # Add initialization
    if 'self.claude_bridge' not in bot_content:
        init_marker = '''            self.git_handler = None

        logger.info(f"Initialized Promptly bot: {user_id}")'''

        init_code = '''            self.git_handler = None

        # Initialize Claude Code bridge (ChatOps Phase 2)
        claude_path = os.getenv("CLAUDE_PATH", "claude")
        try:
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

        logger.info(f"Initialized Promptly bot: {user_id}")'''

        bot_content = bot_content.replace(init_marker, init_code, 1)
        print("[OK] Added Claude bridge initialization")

    # Add routing
    if 'cmd_claude_review' not in bot_content:
        routing_marker = "        elif cmd_type == 'git-pull':\n            return await self.cmd_git_pull(command, room)\n        else:"

        routing_code = """        elif cmd_type == 'git-pull':
            return await self.cmd_git_pull(command, room)
        elif cmd_type == 'claude-review':
            return await self.cmd_claude_review(command, room)
        elif cmd_type == 'claude-explain':
            return await self.cmd_claude_explain(command, room)
        elif cmd_type == 'claude-refactor':
            return await self.cmd_claude_refactor(command, room)
        else:"""

        bot_content = bot_content.replace(routing_marker, routing_code, 1)
        print("[OK] Added Claude command routing")

    # Add methods
    if 'async def cmd_claude_review' not in bot_content:
        # Read claude methods from file
        with open('bot/claude_methods.py', 'r', encoding='utf-8') as f:
            methods = f.read()

        # Find insertion point (after git methods, before send_message)
        insert_marker = '\n    async def send_message('
        bot_content = bot_content.replace(
            insert_marker,
            '\n\n' + methods + insert_marker,
            1
        )
        print("[OK] Added Claude command methods")

    # Write back
    with open('bot/promptly_bot.py', 'w', encoding='utf-8') as f:
        f.write(bot_content)

    print("\n[OK] Claude Code integration complete!\n")
    print("You can now use:")
    print("  - @promptly review <file>")
    print("  - @promptly explain <file>")
    print("  - @promptly refactor <file> \"instruction\"")


if __name__ == "__main__":
    apply_claude_integration()
