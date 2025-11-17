# -*- coding: utf-8 -*-
"""
Response Formatter
==================

Formats query results for human consumption.

Supports multiple output formats:
- text (human-readable)
- json (machine-readable)
- org (org-mode format)
"""

from typing import Dict, Any, List
from datetime import datetime
import json


class ResponseFormatter:
    """
    Formats query results into human-readable output.
    """

    @staticmethod
    def format(result: Dict[str, Any], format: str = 'text') -> str:
        """
        Format query result.

        Args:
            result: Query result dict
            format: Output format ('text', 'json', 'org')

        Returns:
            Formatted string
        """
        if format == 'json':
            return ResponseFormatter._format_json(result)
        elif format == 'org':
            return ResponseFormatter._format_org(result)
        else:
            return ResponseFormatter._format_text(result)

    @staticmethod
    def _format_json(result: Dict[str, Any]) -> str:
        """Format as JSON."""
        return json.dumps(result, indent=2, default=str)

    @staticmethod
    def _format_text(result: Dict[str, Any]) -> str:
        """Format as human-readable text."""
        query_type = result.get('query_type', 'unknown')
        results = result.get('results', [])
        count = result.get('count', 0)

        if result.get('error'):
            return f"Error: {result['error']}"

        output = []

        # Add query info
        output.append(f"Query: {result.get('query', 'N/A')}")
        output.append(f"Type: {query_type}")
        output.append("")

        # Format based on query type
        if query_type == 'current_tasks':
            output.append(f"Current Tasks ({count}):")
            output.append("")
            if results:
                for i, task in enumerate(results, 1):
                    output.append(f"  {i}. {task.get('title', task.get('id'))}")
                    output.append(f"     State: {task.get('state', 'unknown')}")
            else:
                output.append("  No tasks currently in progress")

        elif query_type == 'deadlines':
            output.append(f"Upcoming Deadlines ({count}):")
            output.append("")
            if results:
                for i, task in enumerate(results, 1):
                    days = task.get('days_until', '?')
                    deadline = task.get('deadline', 'unknown')

                    # Parse deadline for display
                    try:
                        dt = datetime.fromisoformat(deadline.split('T')[0])
                        deadline_str = dt.strftime('%a, %b %d')
                    except:
                        deadline_str = deadline

                    urgency = ""
                    if isinstance(days, int):
                        if days < 0:
                            urgency = "⚠️  OVERDUE"
                        elif days == 0:
                            urgency = "🔴 TODAY"
                        elif days == 1:
                            urgency = "🟡 TOMORROW"
                        elif days <= 3:
                            urgency = "🟢 THIS WEEK"

                    output.append(f"  {i}. {task.get('title', task.get('id'))}")
                    output.append(f"     Due: {deadline_str} ({days} days) {urgency}")
            else:
                output.append("  No upcoming deadlines")

        elif query_type == 'temporal':
            output.append(f"Timeline ({count} events):")
            output.append("")
            if results:
                for i, event in enumerate(results, 1):
                    try:
                        ts = datetime.fromisoformat(event['timestamp'])
                        time_str = ts.strftime('%Y-%m-%d %H:%M')
                    except:
                        time_str = event.get('timestamp', 'unknown')

                    output.append(f"  {i}. [{time_str}] {event.get('change_type', 'unknown')}")
                    output.append(f"     Task: {event.get('title', event.get('heading_id'))}")

                    if event.get('old_value') or event.get('new_value'):
                        output.append(f"     Change: {event.get('old_value', 'N/A')} → {event.get('new_value', 'N/A')}")
            else:
                output.append("  No historical events found")

        elif query_type == 'stats':
            output.append("Knowledge Graph Statistics:")
            output.append("")
            stats = results if isinstance(results, dict) else {}

            for key, value in stats.items():
                # Format key nicely
                formatted_key = key.replace('_', ' ').title()
                output.append(f"  {formatted_key}: {value}")

        elif query_type == 'search':
            output.append(f"Search Results ({count}):")
            output.append("")
            if results:
                for i, item in enumerate(results, 1):
                    output.append(f"  {i}. {item.get('title', item.get('id'))}")
                    if 'match_score' in item:
                        output.append(f"     Relevance: {item['match_score']:.2f}")
            else:
                output.append("  No matching items found")

        else:
            # Generic formatting
            output.append(f"Results ({count}):")
            output.append("")
            if results:
                for i, item in enumerate(results, 1):
                    output.append(f"  {i}. {item}")
            else:
                output.append("  No results")

        return "\n".join(output)

    @staticmethod
    def _format_org(result: Dict[str, Any]) -> str:
        """Format as org-mode text."""
        query_type = result.get('query_type', 'unknown')
        results = result.get('results', [])
        count = result.get('count', 0)

        output = []

        # Org header
        output.append(f"* Query Results: {result.get('query', 'N/A')}")
        output.append(f"  :PROPERTIES:")
        output.append(f"  :QUERY-TYPE: {query_type}")
        output.append(f"  :RESULT-COUNT: {count}")
        output.append(f"  :TIMESTAMP: {datetime.now().isoformat()}")
        output.append(f"  :END:")
        output.append("")

        if query_type == 'current_tasks':
            output.append("** Current Tasks")
            output.append("")
            for task in results:
                output.append(f"*** {task.get('state', 'TODO')} {task.get('title', task.get('id'))}")
                output.append(f"    :PROPERTIES:")
                output.append(f"    :ID: {task.get('id')}")
                output.append(f"    :END:")
                output.append("")

        elif query_type == 'deadlines':
            output.append("** Upcoming Deadlines")
            output.append("")
            for task in results:
                output.append(f"*** TODO {task.get('title', task.get('id'))}")
                deadline = task.get('deadline', '')
                if deadline:
                    try:
                        dt = datetime.fromisoformat(deadline.split('T')[0])
                        deadline_str = f"<{dt.strftime('%Y-%m-%d')}>"
                    except:
                        deadline_str = deadline
                    output.append(f"    DEADLINE: {deadline_str}")
                output.append(f"    :PROPERTIES:")
                output.append(f"    :ID: {task.get('id')}")
                output.append(f"    :DAYS-UNTIL: {task.get('days_until', '?')}")
                output.append(f"    :END:")
                output.append("")

        elif query_type == 'temporal':
            output.append("** Timeline")
            output.append("")
            for event in results:
                try:
                    ts = datetime.fromisoformat(event['timestamp'])
                    time_str = ts.strftime('%Y-%m-%d %H:%M')
                except:
                    time_str = event.get('timestamp', 'unknown')

                output.append(f"*** [{time_str}] {event.get('change_type', 'unknown')}")
                output.append(f"    {event.get('title', event.get('heading_id'))}")
                if event.get('old_value') or event.get('new_value'):
                    output.append(f"    Change: {event.get('old_value')} → {event.get('new_value')}")
                output.append("")

        elif query_type == 'stats':
            output.append("** Statistics")
            output.append("")
            stats = results if isinstance(results, dict) else {}
            output.append("| Metric | Value |")
            output.append("|--------+-------|")
            for key, value in stats.items():
                formatted_key = key.replace('_', ' ').title()
                output.append(f"| {formatted_key} | {value} |")

        elif query_type == 'search':
            output.append("** Search Results")
            output.append("")
            for item in results:
                output.append(f"*** [[id:{item.get('id')}][{item.get('title', item.get('id'))}]]")
                if 'match_score' in item:
                    output.append(f"    Relevance: {item['match_score']:.2f}")
                output.append("")

        return "\n".join(output)


# ============================================================================
# Convenience Functions
# ============================================================================

def format_result(result: Dict[str, Any], format: str = 'text') -> str:
    """
    Format query result.

    Args:
        result: Query result from QueryEngine
        format: Output format ('text', 'json', 'org')

    Returns:
        Formatted string
    """
    return ResponseFormatter.format(result, format)
