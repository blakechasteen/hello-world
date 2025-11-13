#!/usr/bin/env python3
"""
Apply Git Integration to promptly_bot.py

This script adds git command handling to the Promptly Matrix Bot.
"""

import re


def apply_git_integration():
    """Apply git integration patches to promptly_bot.py"""

    file_path = "bot/promptly_bot.py"

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Add GitHandler import
    import_pattern = r'(from \.code_reviewer import get_code_reviewer)'
    import_replacement = r'\1\n        from .git_handler import GitHandler'

    if 'from .git_handler import GitHandler' not in content:
        content = re.sub(import_pattern, import_replacement, content)
        print("[OK] Added GitHandler import")
    else:
        print("[SKIP]  GitHandler import already exists")

    # 2. Add git handler initialization
    init_pattern = r'(self\.code_reviewer = get_code_reviewer\(\))\n\n(        logger\.info\(f"Initialized Promptly bot: {user_id}"\))'
    init_replacement = r'''\1

        # Initialize Git handler (ChatOps Phase 1)
        git_repo_path = os.getenv("GIT_REPO_PATH")
        try:
            self.git_handler = GitHandler(git_repo_path) if git_repo_path else None
            if self.git_handler:
                logger.info(f"[OK] Git handler initialized for repo: {git_repo_path}")
            else:
                logger.info("Git handler not initialized (set GIT_REPO_PATH to enable)")
        except ValueError as e:
            logger.warning(f"Git handler initialization failed: {e}")
            self.git_handler = None

\2'''

    if 'self.git_handler' not in content:
        content = re.sub(init_pattern, init_replacement, content)
        print("[OK] Added git handler initialization")
    else:
        print("[SKIP]  Git handler initialization already exists")

    # 3. Add git command routing
    routing_pattern = r"(elif cmd_type == 'list':\n            return await self\.cmd_list\(command, room\))\n        (else:)"
    routing_replacement = r'''\1
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
        \2'''

    if 'cmd_git_status' not in content:
        content = re.sub(routing_pattern, routing_replacement, content)
        print("[OK] Added git command routing")
    else:
        print("[SKIP]  Git command routing already exists")

    # 4. Add git command methods (insert before send_message method)
    git_methods = '''
    # ========== Git Commands (ChatOps Phase 1) ==========

    async def cmd_git_status(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle git status command"""
        if not self.git_handler:
            return {
                "body": "❌ Git not configured. Set GIT_REPO_PATH in .env",
                "html": "<p>❌ Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
            }

        try:
            status = self.git_handler.status(short=True)
            branch = self.git_handler.get_current_branch()

            body = f"📊 **Git Status**\\n\\nBranch: `{branch}`\\n\\n```\\n{status}\\n```"
            html = f"<p>📊 <strong>Git Status</strong></p><p>Branch: <code>{branch}</code></p><pre>{status}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Git status error: {e}")
            return {
                "body": f"❌ Git status failed: {e}",
                "html": f"<p>❌ Git status failed: <code>{e}</code></p>"
            }

    async def cmd_git_log(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle git log command"""
        if not self.git_handler:
            return {
                "body": "❌ Git not configured. Set GIT_REPO_PATH in .env",
                "html": "<p>❌ Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
            }

        try:
            log = self.git_handler.log(max_count=5, oneline=True)
            branch = self.git_handler.get_current_branch()

            body = f"📜 **Recent Commits** ({branch})\\n\\n```\\n{log}\\n```"
            html = f"<p>📜 <strong>Recent Commits</strong> ({branch})</p><pre>{log}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Git log error: {e}")
            return {
                "body": f"❌ Git log failed: {e}",
                "html": f"<p>❌ Git log failed: <code>{e}</code></p>"
            }

    async def cmd_git_diff(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle git diff command"""
        if not self.git_handler:
            return {
                "body": "❌ Git not configured. Set GIT_REPO_PATH in .env",
                "html": "<p>❌ Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
            }

        try:
            diff = self.git_handler.diff()

            if not diff:
                return {
                    "body": "No changes to show (working tree clean)",
                    "html": "<p>No changes to show (working tree clean)</p>"
                }

            if len(diff) > 2000:
                diff = diff[:2000] + "\\n\\n... (truncated, too long for chat)"

            body = f"📝 **Git Diff**\\n\\n```\\n{diff}\\n```"
            html = f"<p>📝 <strong>Git Diff</strong></p><pre>{diff}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Git diff error: {e}")
            return {
                "body": f"❌ Git diff failed: {e}",
                "html": f"<p>❌ Git diff failed: <code>{e}</code></p>"
            }

    async def cmd_git_branch(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle git branch command"""
        if not self.git_handler:
            return {
                "body": "❌ Git not configured. Set GIT_REPO_PATH in .env",
                "html": "<p>❌ Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
            }

        try:
            branches = self.git_handler.branch()

            body = f"🌿 **Git Branches**\\n\\n```\\n{branches}\\n```"
            html = f"<p>🌿 <strong>Git Branches</strong></p><pre>{branches}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Git branch error: {e}")
            return {
                "body": f"❌ Git branch failed: {e}",
                "html": f"<p>❌ Git branch failed: <code>{e}</code></p>"
            }

    async def cmd_git_commit(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle git commit command"""
        if not self.git_handler:
            return {
                "body": "❌ Git not configured. Set GIT_REPO_PATH in .env",
                "html": "<p>❌ Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
            }

        try:
            message = command.get('message', '')
            if not message:
                return {
                    "body": "❌ Commit message required. Usage: `@promptly git commit \\"your message\\"`",
                    "html": "<p>❌ Commit message required. Usage: <code>@promptly git commit \\"your message\\"</code></p>"
                }

            result = self.git_handler.commit(message, add_all=True)

            body = f"[OK] **Commit Created**\\n\\nMessage: `{message}`\\n\\n```\\n{result}\\n```"
            html = f"<p>[OK] <strong>Commit Created</strong></p><p>Message: <code>{message}</code></p><pre>{result}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Git commit error: {e}")
            return {
                "body": f"❌ Git commit failed: {e}",
                "html": f"<p>❌ Git commit failed: <code>{e}</code></p>"
            }

    async def cmd_git_push(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle git push command"""
        if not self.git_handler:
            return {
                "body": "❌ Git not configured. Set GIT_REPO_PATH in .env",
                "html": "<p>❌ Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
            }

        try:
            branch = self.git_handler.get_current_branch()
            result = self.git_handler.push(branch=branch)

            body = f"[OK] **Pushed to Remote**\\n\\nBranch: `{branch}`\\n\\n```\\n{result}\\n```"
            html = f"<p>[OK] <strong>Pushed to Remote</strong></p><p>Branch: <code>{branch}</code></p><pre>{result}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Git push error: {e}")
            return {
                "body": f"❌ Git push failed: {e}\\n\\nNote: Push requires authentication",
                "html": f"<p>❌ Git push failed: <code>{e}</code></p><p>Note: Push requires authentication</p>"
            }

    async def cmd_git_pull(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle git pull command"""
        if not self.git_handler:
            return {
                "body": "❌ Git not configured. Set GIT_REPO_PATH in .env",
                "html": "<p>❌ Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
            }

        try:
            result = self.git_handler.pull()

            body = f"[OK] **Pulled from Remote**\\n\\n```\\n{result}\\n```"
            html = f"<p>[OK] <strong>Pulled from Remote</strong></p><pre>{result}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Git pull error: {e}")
            return {
                "body": f"❌ Git pull failed: {e}",
                "html": f"<p>❌ Git pull failed: <code>{e}</code></p>"
            }

'''

    methods_pattern = r'(            }\n\n)(    async def send_message\()'
    methods_replacement = r'\1' + git_methods + r'\2'

    if 'async def cmd_git_status' not in content:
        content = re.sub(methods_pattern, methods_replacement, content, flags=re.DOTALL)
        print("[OK] Added git command methods")
    else:
        print("[SKIP]  Git command methods already exist")

    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("\n[OK] Git integration applied successfully!")
    print("\nYou can now use:")
    print("  - @promptly git status")
    print("  - @promptly git log")
    print("  - @promptly git diff")
    print("  - @promptly git branch")
    print("  - @promptly git commit \"message\"")
    print("  - @promptly git push")
    print("  - @promptly git pull")


if __name__ == "__main__":
    apply_git_integration()
