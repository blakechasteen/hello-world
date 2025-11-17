    # ========== Claude Code Commands (ChatOps Phase 2) ==========

    async def cmd_claude_review(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle Claude Code review command"""
        if not self.claude_bridge:
            return {
                "body": "Claude API not available. Set ANTHROPIC_API_KEY in .env",
                "html": "<p>Claude API not available. Get API key from <a href='https://console.anthropic.com'>console.anthropic.com</a></p>"
            }

        file_path = command.get('file_path', '')
        if not file_path:
            return {
                "body": 'Usage: @promptly review <file_path>',
                "html": '<p>Usage: <code>@promptly review &lt;file_path&gt;</code></p>'
            }

        # Show processing message
        await self.send_message(room.room_id, f"Reviewing {file_path}...")

        try:
            result = self.claude_bridge.review(file_path)

            if len(result) > 3000:
                result = result[:3000] + "\n\n... (truncated)"

            body = f"Code Review: {file_path}\n\n{result}"
            html = f"<p><strong>Code Review:</strong> <code>{file_path}</code></p><pre>{result}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Claude review error: {e}")
            return {
                "body": f"Review failed: {e}",
                "html": f"<p>Review failed: <code>{e}</code></p>"
            }

    async def cmd_claude_explain(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle Claude Code explain command"""
        if not self.claude_bridge:
            return {
                "body": "Claude API not available. Set ANTHROPIC_API_KEY in .env",
                "html": "<p>Claude API not available. Get API key from <a href='https://console.anthropic.com'>console.anthropic.com</a></p>"
            }

        file_path = command.get('file_path', '')
        if not file_path:
            return {
                "body": 'Usage: @promptly explain <file_path>',
                "html": '<p>Usage: <code>@promptly explain &lt;file_path&gt;</code></p>'
            }

        # Show processing message
        await self.send_message(room.room_id, f"Explaining {file_path}...")

        try:
            result = self.claude_bridge.explain(file_path)

            if len(result) > 3000:
                result = result[:3000] + "\n\n... (truncated)"

            body = f"Explanation: {file_path}\n\n{result}"
            html = f"<p><strong>Explanation:</strong> <code>{file_path}</code></p><pre>{result}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Claude explain error: {e}")
            return {
                "body": f"Explanation failed: {e}",
                "html": f"<p>Explanation failed: <code>{e}</code></p>"
            }

    async def cmd_claude_refactor(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle Claude Code refactor command"""
        if not self.claude_bridge:
            return {
                "body": "Claude API not available. Set ANTHROPIC_API_KEY in .env",
                "html": "<p>Claude API not available. Get API key from <a href='https://console.anthropic.com'>console.anthropic.com</a></p>"
            }

        file_path = command.get('file_path', '')
        instruction = command.get('instruction', '')

        if not file_path or not instruction:
            return {
                "body": 'Usage: @promptly refactor <file_path> "instruction"',
                "html": '<p>Usage: <code>@promptly refactor &lt;file_path&gt; "instruction"</code></p>'
            }

        # Show processing message
        await self.send_message(room.room_id, f"Refactoring {file_path}...")

        try:
            result = self.claude_bridge.refactor(file_path, instruction)

            if len(result) > 3000:
                result = result[:3000] + "\n\n... (truncated)"

            body = f"Refactoring: {file_path}\n\nInstruction: {instruction}\n\n{result}"
            html = f"<p><strong>Refactoring:</strong> <code>{file_path}</code></p><p>Instruction: <em>{instruction}</em></p><pre>{result}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Claude refactor error: {e}")
            return {
                "body": f"Refactoring failed: {e}",
                "html": f"<p>Refactoring failed: <code>{e}</code></p>"
            }
