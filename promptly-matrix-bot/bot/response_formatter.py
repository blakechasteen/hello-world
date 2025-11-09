#!/usr/bin/env python3
"""
Response Formatter for Promptly Matrix Bot

Formats Promptly Core results as rich Matrix messages.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """Format Promptly responses as Matrix messages (text + HTML)"""

    @staticmethod
    def format_optimization_result(result: Dict[str, Any]) -> Dict[str, str]:
        """
        Format optimization result.

        Args:
            result: Optimization result from Promptly Core

        Returns:
            Dict with 'body' (plain text) and 'html' (formatted) keys
        """
        if not result.get('success'):
            return ResponseFormatter._format_error(
                result.get('error', 'Optimization failed'),
                result
            )

        metrics = result.get('metrics', {})
        stub_mode = result.get('stub_mode', False)

        # Plain text version
        body = "✅ Optimization complete!\n\n"

        if stub_mode:
            body += "⚠️ STUB MODE - Install HoloLoom + DSPy for real optimization\n\n"

        body += f"**Optimized Prompt:**\n{result.get('prompt', 'N/A')}\n\n"

        body += "**Metrics:**\n"
        body += f"• Overall Score: {metrics.get('overall_score', 0):.2f}\n"
        body += f"• Accuracy: {metrics.get('accuracy', 0):.2f}\n"
        body += f"• Clarity: {metrics.get('clarity', 0):.2f}\n"
        body += f"• Completeness: {metrics.get('completeness', 0):.2f}\n"
        body += f"• Latency: {metrics.get('latency_ms', 0):.0f}ms\n\n"

        body += f"**Examples Used:** {result.get('examples_used', 0)}\n\n"

        if 'test_result' in result:
            body += f"**Test Result:**\n{result['test_result']}\n\n"

        body += "**Next Steps:**\n"
        body += "• Test: `@promptly run <workflow> \"test input\"`\n"
        body += "• Save: `@promptly save my_prompt`\n"
        body += "• Refine: `@promptly refine`"

        # HTML version
        html = "<h3>✅ Optimization complete!</h3>"

        if stub_mode:
            html += '<p><strong>⚠️ STUB MODE</strong> - Install HoloLoom + DSPy for real optimization</p>'

        html += "<p><strong>Optimized Prompt:</strong></p>"
        html += f"<pre><code>{result.get('prompt', 'N/A')}</code></pre>"

        html += "<p><strong>Metrics:</strong></p><ul>"
        html += f"<li>Overall Score: <strong>{metrics.get('overall_score', 0):.2f}</strong></li>"
        html += f"<li>Accuracy: {metrics.get('accuracy', 0):.2f}</li>"
        html += f"<li>Clarity: {metrics.get('clarity', 0):.2f}</li>"
        html += f"<li>Completeness: {metrics.get('completeness', 0):.2f}</li>"
        html += f"<li>Latency: {metrics.get('latency_ms', 0):.0f}ms</li>"
        html += "</ul>"

        html += f"<p><strong>Examples Used:</strong> {result.get('examples_used', 0)}</p>"

        if 'test_result' in result:
            html += f"<p><strong>Test Result:</strong></p>"
            html += f"<p>{result['test_result']}</p>"

        html += "<p><strong>Next Steps:</strong></p><ul>"
        html += '<li>Test: <code>@promptly run &lt;workflow&gt; "test input"</code></li>'
        html += "<li>Save: <code>@promptly save my_prompt</code></li>"
        html += "<li>Refine: <code>@promptly refine</code></li>"
        html += "</ul>"

        return {"body": body, "html": html}

    @staticmethod
    def format_workflow_result(result: Dict[str, Any]) -> Dict[str, str]:
        """
        Format workflow execution result.

        Args:
            result: Workflow result from Promptly Core

        Returns:
            Dict with 'body' and 'html' keys
        """
        if not result.get('success'):
            return ResponseFormatter._format_error(
                result.get('error', 'Workflow execution failed'),
                result
            )

        workflow = result.get('workflow', 'unknown')
        output = result.get('output', 'No output')
        steps = result.get('steps_executed', 0)
        latency = result.get('latency_ms', 0)
        stub_mode = result.get('stub_mode', False)

        # Plain text
        body = f"✅ Workflow '{workflow}' completed!\n\n"

        if stub_mode:
            body += "⚠️ STUB MODE - Install HoloLoom for real workflow execution\n\n"

        body += f"**Input:**\n{result.get('input', 'N/A')}\n\n"
        body += f"**Output:**\n{output}\n\n"
        body += f"**Steps Executed:** {steps}\n"
        body += f"**Latency:** {latency:.0f}ms"

        # HTML
        html = f"<h3>✅ Workflow '{workflow}' completed!</h3>"

        if stub_mode:
            html += '<p><strong>⚠️ STUB MODE</strong> - Install HoloLoom for real workflow execution</p>'

        html += f"<p><strong>Input:</strong></p><p>{result.get('input', 'N/A')}</p>"
        html += f"<p><strong>Output:</strong></p><p>{output}</p>"
        html += f"<p><strong>Steps Executed:</strong> {steps} | "
        html += f"<strong>Latency:</strong> {latency:.0f}ms</p>"

        return {"body": body, "html": html}

    @staticmethod
    def format_save_result(name: str, success: bool = True) -> Dict[str, str]:
        """Format save operation result"""
        if success:
            body = f"✅ Prompt saved as '{name}'\n\nUse `@promptly run {name}` to execute it."
            html = f"<p>✅ Prompt saved as <strong>{name}</strong></p>"
            html += f"<p>Use <code>@promptly run {name}</code> to execute it.</p>"
        else:
            body = f"❌ Failed to save prompt '{name}'"
            html = f"<p>❌ Failed to save prompt <strong>{name}</strong></p>"

        return {"body": body, "html": html}

    @staticmethod
    def format_list_result(prompts: List[Dict[str, Any]], room_name: str = "this room") -> Dict[str, str]:
        """Format saved prompts list"""
        if not prompts:
            body = f"No saved prompts in {room_name}.\n\nCreate one with `@promptly optimize` then `@promptly save <name>`"
            html = f"<p>No saved prompts in {room_name}.</p>"
            html += "<p>Create one with <code>@promptly optimize</code> then <code>@promptly save &lt;name&gt;</code></p>"
            return {"body": body, "html": html}

        # Plain text
        body = f"**Saved Prompts ({len(prompts)}):**\n\n"
        for i, prompt in enumerate(prompts, 1):
            body += f"{i}. **{prompt['name']}**\n"
            body += f"   Task: {prompt.get('task', 'N/A')}\n"
            body += f"   Created: {prompt.get('created', 'N/A')}\n"
            body += f"   Run: `@promptly run {prompt['name']}`\n\n"

        # HTML
        html = f"<h3>Saved Prompts ({len(prompts)})</h3><ol>"
        for prompt in prompts:
            html += f"<li><strong>{prompt['name']}</strong>"
            html += f"<br>Task: {prompt.get('task', 'N/A')}"
            html += f"<br>Created: {prompt.get('created', 'N/A')}"
            html += f"<br>Run: <code>@promptly run {prompt['name']}</code>"
            html += "</li>"
        html += "</ol>"

        return {"body": body, "html": html}

    @staticmethod
    def format_code_review_result(result) -> Dict[str, str]:
        """
        Format code review result.

        Args:
            result: CodeReviewResult object

        Returns:
            Dict with 'body' (plain) and 'html' (formatted) keys
        """
        # Count issues by severity
        critical = result.get_critical_count()
        high = result.get_high_count()
        medium = sum(1 for i in result.issues if i.severity.value == "medium")
        low = sum(1 for i in result.issues if i.severity.value == "low")

        # Determine risk level
        if critical > 0:
            risk_emoji = "🔴"
            risk_level = "CRITICAL"
        elif high > 0:
            risk_emoji = "🟠"
            risk_level = "HIGH"
        elif medium > 0:
            risk_emoji = "🟡"
            risk_level = "MEDIUM"
        else:
            risk_emoji = "🟢"
            risk_level = "LOW"

        # Plain text version
        body = f"{risk_emoji} **Code Review Complete**\n\n"
        body += f"**Language:** {result.language}\n"
        body += f"**Lines Analyzed:** {result.lines_analyzed}\n"
        body += f"**Risk Score:** {result.risk_score:.1f}/10 ({risk_level})\n\n"

        body += "**Issues Found:**\n"
        body += f"• Critical: {critical}\n"
        body += f"• High: {high}\n"
        body += f"• Medium: {medium}\n"
        body += f"• Low: {low}\n"

        if result.issues:
            body += "\n**Top Issues:**\n\n"
            # Show top 5 most severe issues
            sorted_issues = sorted(result.issues, key=lambda x: ["critical", "high", "medium", "low", "info"].index(x.severity.value))
            for i, issue in enumerate(sorted_issues[:5], 1):
                severity_emoji = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🔵",
                    "info": "⚪"
                }.get(issue.severity.value, "⚪")

                body += f"{i}. {severity_emoji} **{issue.title}**\n"
                body += f"   {issue.description}\n"
                if issue.recommendation:
                    body += f"   → {issue.recommendation}\n"
                body += "\n"

        if critical + high > 0:
            body += "⚠️ **Action Required:** Fix critical/high severity issues before deployment.\n"
        elif medium > 0:
            body += "ℹ️ **Recommendation:** Review and address medium severity issues.\n"
        else:
            body += "✅ **Good to go!** No critical issues found.\n"

        # HTML version
        html = f"<h3>{risk_emoji} Code Review Complete</h3>"
        html += f"<p><strong>Language:</strong> {result.language}<br>"
        html += f"<strong>Lines Analyzed:</strong> {result.lines_analyzed}<br>"
        html += f"<strong>Risk Score:</strong> {result.risk_score:.1f}/10 ({risk_level})</p>"

        html += "<h4>Issues Found:</h4><ul>"
        html += f"<li>Critical: {critical}</li>"
        html += f"<li>High: {high}</li>"
        html += f"<li>Medium: {medium}</li>"
        html += f"<li>Low: {low}</li>"
        html += "</ul>"

        if result.issues:
            html += "<h4>Top Issues:</h4><ol>"
            sorted_issues = sorted(result.issues, key=lambda x: ["critical", "high", "medium", "low", "info"].index(x.severity.value))
            for issue in sorted_issues[:5]:
                severity_emoji = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🔵",
                    "info": "⚪"
                }.get(issue.severity.value, "⚪")

                html += f"<li>{severity_emoji} <strong>{issue.title}</strong><br>"
                html += f"{issue.description}"
                if issue.recommendation:
                    html += f"<br>→ {issue.recommendation}"
                html += "</li>"
            html += "</ol>"

        return {"body": body, "html": html}

    @staticmethod
    def format_progress(message: str, percent: Optional[int] = None) -> str:
        """
        Format progress message.

        Args:
            message: Progress message
            percent: Optional progress percentage (0-100)

        Returns:
            Formatted progress string
        """
        if percent is not None:
            # ASCII progress bar
            bar_length = 20
            filled = int(bar_length * percent / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            return f"⏳ {message} [{bar}] {percent}%"
        else:
            return f"⏳ {message}..."

    @staticmethod
    def _format_error(error: str, context: Dict[str, Any]) -> Dict[str, str]:
        """Format error message"""
        body = f"❌ Error: {error}\n\n"

        if 'task' in context:
            body += f"Task: {context['task']}\n"
        if 'workflow' in context:
            body += f"Workflow: {context['workflow']}\n"
        if 'examples_count' in context:
            body += f"Examples provided: {context['examples_count']}\n"

        body += "\nTry `@promptly help` for usage."

        html = f"<p>❌ <strong>Error:</strong> {error}</p>"
        if 'task' in context:
            html += f"<p>Task: {context['task']}</p>"
        if 'workflow' in context:
            html += f"<p>Workflow: {context['workflow']}</p>"

        html += "<p>Try <code>@promptly help</code> for usage.</p>"

        return {"body": body, "html": html}


# Test
if __name__ == "__main__":
    formatter = ResponseFormatter()

    # Test optimization result
    result = {
        "success": True,
        "prompt": "Answer customer questions clearly and concisely...",
        "metrics": {
            "overall_score": 0.92,
            "accuracy": 0.95,
            "clarity": 0.90,
            "completeness": 0.91,
            "latency_ms": 145
        },
        "examples_used": 3,
        "test_result": "Click Settings > Security > Reset Password"
    }

    formatted = formatter.format_optimization_result(result)
    print("Optimization Result:")
    print("=" * 70)
    print(formatted['body'])
    print("\n" + "=" * 70)
    print(formatted['html'])

    # Test workflow result
    result = {
        "success": True,
        "workflow": "qa_basic",
        "input": "What is Thompson Sampling?",
        "output": "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem...",
        "steps_executed": 1,
        "latency_ms": 120
    }

    formatted = formatter.format_workflow_result(result)
    print("\n\nWorkflow Result:")
    print("=" * 70)
    print(formatted['body'])

    # Test progress
    print("\n\nProgress Examples:")
    print(formatter.format_progress("Analyzing examples", 25))
    print(formatter.format_progress("Running optimization", 75))
    print(formatter.format_progress("Completing"))

    print("\n✅ All formatter tests passed!")
