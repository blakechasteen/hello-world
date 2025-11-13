#!/usr/bin/env python3
"""
Final Git Integration Script

Applies git integration with proper string handling.
"""

import re


def apply_integration():
    """Apply git integration safely"""

    filepath = 'bot/promptly_bot.py'

    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Convert to string for processing
    content = ''.join(lines)

    # Step 1: Add import
    if 'from .git_handler import GitHandler' not in content:
        import_line = '        from .code_reviewer import get_code_reviewer\n'
        new_import = '        from .code_reviewer import get_code_reviewer\n        from .git_handler import GitHandler\n'
        content = content.replace(import_line, new_import, 1)
        print("[OK] Added GitHandler import")

    # Step 2: Add initialization
    if 'self.git_handler' not in content:
        init_marker = '        self.code_reviewer = get_code_reviewer()\n\n        logger.info(f"Initialized Promptly bot: {user_id}")'
        init_code = '''        self.code_reviewer = get_code_reviewer()

        # Initialize Git handler (ChatOps Phase 1)
        git_repo_path = os.getenv("GIT_REPO_PATH")
        try:
            self.git_handler = GitHandler(git_repo_path) if git_repo_path else None
            if self.git_handler:
                logger.info(f"Git handler initialized for repo: {git_repo_path}")
        except ValueError as e:
            logger.warning(f"Git handler init failed: {e}")
            self.git_handler = None

        logger.info(f"Initialized Promptly bot: {user_id}")'''

        content = content.replace(init_marker, init_code, 1)
        print("[OK] Added git handler initialization")

    # Step 3: Add routing
    if 'cmd_git_status' not in content:
        routing_marker = "        elif cmd_type == 'list':\n            return await self.cmd_list(command, room)\n        else:"

        routing_code = """        elif cmd_type == 'list':
            return await self.cmd_list(command, room)
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
        else:"""

        content = content.replace(routing_marker, routing_code, 1)
        print("[OK] Added git command routing")

    # Step 4: Add methods - Read from the clean git_methods.py and add proper indentation
    if 'async def cmd_git_status' not in content:
        # Read git_methods.py
        with open('bot/git_methods.py', 'r', encoding='utf-8') as f:
            methods_content = f.read()

        # Find the methods section
        start_marker = '# ========== Git Commands (ChatOps Phase 1) =========='
        methods_start = methods_content.find(start_marker)

        if methods_start != -1:
            # Get everything from the marker onwards
            methods = methods_content[methods_start:]

            # Find where to insert (before send_message)
            insert_marker = '    async def send_message('
            if insert_marker in content:
                # Add proper indentation and insert
                content = content.replace(
                    insert_marker,
                    '\n    ' + methods + '\n\n    async def send_message(',
                    1
                )
                print("[OK] Added git command methods")

    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print("\n[OK] Git integration complete!")
    return True


if __name__ == "__main__":
    apply_integration()
