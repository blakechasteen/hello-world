"""
Voice Command Parser - Convert speech to Chronos commands

Simple pattern matching for the 5 verbs
"""

import re
from typing import Optional, Dict, Any, List


class VoiceCommandParser:
    """Parse voice transcriptions into Chronos commands"""

    # Command patterns (case-insensitive)
    PATTERNS = {
        "start": [
            r"^start\s+(.+)$",
            r"^begin\s+(.+)$",
            r"^starting\s+(.+)$",
        ],
        "stop": [
            r"^stop$",
            r"^done$",
            r"^finished$",
            r"^complete$",
            r"^end$",
        ],
        "log": [
            r"^log\s+(.+?)\s+(?:for\s+)?(.+)$",  # "log email_review for 20 minutes"
        ],
        "note": [
            r"^note[\s:]?\s*(.+)$",
            r"^add\s+note[\s:]?\s*(.+)$",
            r"^remember[\s:]?\s*(.+)$",
        ],
        "status": [
            r"^status$",
            r"^what(?:'s|\s+is)\s+(?:the\s+)?(?:current\s+)?(?:active\s+)?(?:task)?$",
            r"^active\s+task$",
        ]
    }

    def parse(self, transcript: str) -> Optional[Dict[str, Any]]:
        """
        Parse voice command into structured command

        Returns:
            {
                "verb": "start" | "stop" | "log" | "note" | "status",
                "args": {...}  # Verb-specific arguments
            }

        Returns None if command not recognized
        """
        text = transcript.strip().lower()

        # Try each verb pattern
        for verb, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.match(pattern, text, re.IGNORECASE)
                if match:
                    return self._extract_command(verb, match, text)

        return None

    def _extract_command(
        self,
        verb: str,
        match: re.Match,
        original_text: str
    ) -> Dict[str, Any]:
        """Extract command details based on verb"""

        if verb == "start":
            task = match.group(1).strip()
            task_normalized = self._normalize_task_name(task)

            # Extract tags if present (e.g., "compost sifting #farm #manual")
            tags = self._extract_tags(task)
            if tags:
                # Remove tags from task name
                task_normalized = re.sub(r'\s*#\w+', '', task_normalized)

            return {
                "verb": "start",
                "args": {
                    "task": task_normalized,
                    "tags": tags
                }
            }

        elif verb == "stop":
            return {
                "verb": "stop",
                "args": {}
            }

        elif verb == "log":
            task = match.group(1).strip()
            duration_str = match.group(2).strip()

            task_normalized = self._normalize_task_name(task)
            duration_sec = self._parse_duration_natural(duration_str)

            if duration_sec is None:
                # Failed to parse duration
                return None

            return {
                "verb": "log",
                "args": {
                    "task": task_normalized,
                    "duration_sec": duration_sec
                }
            }

        elif verb == "note":
            text = match.group(1).strip()
            return {
                "verb": "note",
                "args": {
                    "text": text
                }
            }

        elif verb == "status":
            return {
                "verb": "status",
                "args": {}
            }

        return None

    def _normalize_task_name(self, raw: str) -> str:
        """
        Convert natural language to task name

        Examples:
            "compost sifting" → "compost_sifting"
            "email review" → "email_review"
        """
        # Remove tags first
        raw = re.sub(r'\s*#\w+', '', raw)

        # Convert spaces to underscores, lowercase
        return raw.strip().replace(" ", "_").lower()

    def _extract_tags(self, text: str) -> List[str]:
        """
        Extract hashtags from text

        Example: "compost sifting #farm #manual" → ["farm", "manual"]
        """
        tags = re.findall(r'#(\w+)', text)
        return tags

    def _parse_duration_natural(self, duration_str: str) -> Optional[float]:
        """
        Parse natural language duration to seconds

        Examples:
            "20 minutes" → 1200
            "1 hour" → 3600
            "30 seconds" → 30
            "1.5 hours" → 5400
        """
        duration_str = duration_str.strip().lower()

        # Pattern: number + unit
        patterns = [
            (r'(\d+(?:\.\d+)?)\s*(?:hour|hours|hr|hrs|h)', 3600),
            (r'(\d+(?:\.\d+)?)\s*(?:minute|minutes|min|mins|m)', 60),
            (r'(\d+(?:\.\d+)?)\s*(?:second|seconds|sec|secs|s)', 1),
        ]

        for pattern, multiplier in patterns:
            match = re.search(pattern, duration_str)
            if match:
                value = float(match.group(1))
                return value * multiplier

        return None


def format_confirmation(command: Dict[str, Any]) -> str:
    """
    Format command as confirmation message (for voice feedback)

    Example:
        {"verb": "start", "args": {"task": "compost_sifting"}}
        → "Starting compost sifting. Say 'done' when finished."
    """
    verb = command["verb"]
    args = command["args"]

    if verb == "start":
        task = args["task"].replace("_", " ")
        return f"Starting {task}. Say 'done' when finished."

    elif verb == "stop":
        return "Stopping current task."

    elif verb == "log":
        task = args["task"].replace("_", " ")
        duration_sec = args["duration_sec"]
        hours = int(duration_sec // 3600)
        minutes = int((duration_sec % 3600) // 60)

        if hours > 0:
            duration_str = f"{hours} hour{'s' if hours > 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"
        else:
            duration_str = f"{minutes} minute{'s' if minutes != 1 else ''}"

        return f"Logging {task} for {duration_str}."

    elif verb == "note":
        return "Note added."

    elif verb == "status":
        return "Checking status."

    return "Command understood."
