#!/usr/bin/env python3
"""
Safe Git Integration Script

This script carefully applies git integration to promptly_bot.py with:
- Backup before changes
- Line-by-line validation
- Syntax checking after each change
- Rollback on error
"""

import os
import shutil
import subprocess
import sys


def backup_file(filepath):
    """Create backup of file"""
    backup_path = filepath + '.backup'
    shutil.copy2(filepath, backup_path)
    print(f"[BACKUP] Created backup: {backup_path}")
    return backup_path


def validate_syntax(filepath):
    """Check Python syntax"""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'py_compile', filepath],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"[ERROR] Syntax validation failed: {e}")
        return False


def restore_backup(filepath, backup_path):
    """Restore file from backup"""
    shutil.copy2(backup_path, filepath)
    print(f"[RESTORE] Restored from backup")


def apply_integration():
    """Apply git integration safely"""

    filepath = 'bot/promptly_bot.py'

    # Step 1: Backup
    print("\n=== Step 1: Creating Backup ===")
    backup_path = backup_file(filepath)

    # Step 2: Read file
    print("\n=== Step 2: Reading File ===")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    print(f"[OK] File read: {len(content)} characters")

    # Step 3: Add import
    print("\n=== Step 3: Adding GitHandler Import ===")
    if 'from .git_handler import GitHandler' not in content:
        # Find the right place (after code_reviewer import)
        import_line = '        from .code_reviewer import get_code_reviewer\n'
        if import_line in content:
            content = content.replace(
                import_line,
                import_line + '        from .git_handler import GitHandler\n'
            )
            print("[OK] Import added")
        else:
            print("[ERROR] Could not find import location")
            restore_backup(filepath, backup_path)
            return False
    else:
        print("[SKIP] Import already exists")

    # Step 4: Initialize git handler
    print("\n=== Step 4: Initializing Git Handler ===")
    if 'self.git_handler' not in content:
        init_code = '''
        # Initialize Git handler (ChatOps Phase 1)
        git_repo_path = os.getenv("GIT_REPO_PATH")
        try:
            self.git_handler = GitHandler(git_repo_path) if git_repo_path else None
            if self.git_handler:
                logger.info(f"Git handler initialized for repo: {git_repo_path}")
        except ValueError as e:
            logger.warning(f"Git handler init failed: {e}")
            self.git_handler = None
'''

        # Find insertion point
        marker = '        self.code_reviewer = get_code_reviewer()\n\n        logger.info(f"Initialized Promptly bot: {user_id}")'
        if marker in content:
            content = content.replace(
                marker,
                '        self.code_reviewer = get_code_reviewer()\n' + init_code + '\n        logger.info(f"Initialized Promptly bot: {user_id}")'
            )
            print("[OK] Git handler initialization added")
        else:
            print("[ERROR] Could not find initialization location")
            restore_backup(filepath, backup_path)
            return False
    else:
        print("[SKIP] Git handler already initialized")

    # Step 5: Add routing
    print("\n=== Step 5: Adding Git Command Routing ===")
    if 'cmd_git_status' not in content:
        routing_code = '''        elif cmd_type == 'git-status':
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
        '''

        # Find insertion point
        marker = "        elif cmd_type == 'list':\n            return await self.cmd_list(command, room)\n        else:"
        if marker in content:
            content = content.replace(
                marker,
                "        elif cmd_type == 'list':\n            return await self.cmd_list(command, room)\n" + routing_code + "else:"
            )
            print("[OK] Routing added")
        else:
            print("[ERROR] Could not find routing location")
            restore_backup(filepath, backup_path)
            return False
    else:
        print("[SKIP] Routing already exists")

    # Step 6: Add methods
    print("\n=== Step 6: Adding Git Command Methods ===")
    if 'async def cmd_git_status' not in content:
        # Read methods from file
        with open('bot/git_methods.py', 'r', encoding='utf-8') as f:
            methods_content = f.read()

        # Extract just the methods (skip the docstring at top)
        methods_start = methods_content.find('# ========== Git Commands')
        if methods_start == -1:
            print("[ERROR] Could not find methods in git_methods.py")
            restore_backup(filepath, backup_path)
            return False

        methods = methods_content[methods_start:]

        # Find insertion point (before send_message)
        marker = '    async def send_message('
        if marker in content:
            # Insert with proper indentation
            insertion = '\n' + methods + '\n    '
            content = content.replace(
                marker,
                insertion + marker
            )
            print("[OK] Methods added")
        else:
            print("[ERROR] Could not find methods insertion location")
            restore_backup(filepath, backup_path)
            return False
    else:
        print("[SKIP] Methods already exist")

    # Step 7: Write file
    print("\n=== Step 7: Writing File ===")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[OK] File written: {len(content)} characters")

    # Step 8: Validate syntax
    print("\n=== Step 8: Validating Syntax ===")
    if not validate_syntax(filepath):
        print("[ERROR] Syntax validation failed! Rolling back...")
        restore_backup(filepath, backup_path)
        return False

    print("[OK] Syntax valid!")

    # Step 9: Test import
    print("\n=== Step 9: Testing Import ===")
    try:
        result = subprocess.run(
            [sys.executable, '-c', 'from bot.promptly_bot import PromptlyBot; print("OK")'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        if result.returncode == 0:
            print("[OK] Import successful!")
        else:
            print(f"[ERROR] Import failed: {result.stderr}")
            restore_backup(filepath, backup_path)
            return False
    except Exception as e:
        print(f"[ERROR] Import test failed: {e}")
        restore_backup(filepath, backup_path)
        return False

    # Success!
    print("\n" + "="*50)
    print("SUCCESS! Git integration applied!")
    print("="*50)
    print("\nYou can now use:")
    print("  @promptly git status")
    print("  @promptly git log")
    print("  @promptly git diff")
    print("  @promptly git branch")
    print('  @promptly git commit "message"')
    print("  @promptly git push")
    print("  @promptly git pull")
    print("\nBackup saved at:", backup_path)
    print("\nTo test: python run_bot.py")

    return True


if __name__ == "__main__":
    success = apply_integration()
    sys.exit(0 if success else 1)
