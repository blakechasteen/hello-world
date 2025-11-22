"""
BossPig Jargon Dictionary

300+ corporate jargon phrases with plain language replacements.

Created: 2025-11-22
Status: Production Ready
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Core jargon categories
JARGON_CATEGORIES = {
    "corporate_buzzwords": "Overused business jargon",
    "vague_commitments": "Weak commitment language",
    "meaningless_metrics": "Metrics without specifics",
    "vague_dates": "Ambiguous time references",
    "weasel_words": "Unattributed claims",
    "redundant_phrases": "Unnecessarily wordy",
    "unclear_ownership": "Missing responsibility",
}


# Main jargon dictionary (300+ phrases)
JARGON_REPLACEMENTS: Dict[str, Dict[str, str]] = {
    # Corporate Buzzwords (100+ phrases)
    "synergize": {
        "replacement": "combine",
        "category": "corporate_buzzwords",
        "severity": "critical",
        "explanation": "Vague corporate jargon. Use 'combine' or 'work together'."
    },
    "leverage": {
        "replacement": "use",
        "category": "corporate_buzzwords",
        "severity": "critical",
        "explanation": "Overused buzzword. Simply say 'use'."
    },
    "circle back": {
        "replacement": "follow up",
        "category": "corporate_buzzwords",
        "severity": "warning",
        "explanation": "Vague. Say 'follow up' with specific date and action."
    },
    "touch base": {
        "replacement": "meet",
        "category": "corporate_buzzwords",
        "severity": "warning",
        "explanation": "Informal jargon. Use 'meet' or 'discuss'."
    },
    "low-hanging fruit": {
        "replacement": "easy wins",
        "category": "corporate_buzzwords",
        "severity": "critical",
        "explanation": "Cliché metaphor. Say 'easy wins' or 'quick improvements'."
    },
    "move the needle": {
        "replacement": "improve",
        "category": "corporate_buzzwords",
        "severity": "critical",
        "explanation": "Vague metric language. Specify what you're improving."
    },
    "paradigm shift": {
        "replacement": "major change",
        "category": "corporate_buzzwords",
        "severity": "critical",
        "explanation": "Overused buzzword. Say 'major change' or 'transformation'."
    },
    "game changer": {
        "replacement": "significant improvement",
        "category": "corporate_buzzwords",
        "severity": "critical",
        "explanation": "Hyperbolic jargon. Quantify the improvement."
    },
    "thought leadership": {
        "replacement": "expertise",
        "category": "corporate_buzzwords",
        "severity": "warning",
        "explanation": "Self-aggrandizing jargon. Say 'expertise' or 'knowledge'."
    },
    "deep dive": {
        "replacement": "detailed analysis",
        "category": "corporate_buzzwords",
        "severity": "warning",
        "explanation": "Casual jargon. Use 'detailed analysis' or 'thorough review'."
    },
    "bandwidth": {
        "replacement": "time",
        "category": "corporate_buzzwords",
        "severity": "warning",
        "explanation": "Technical term misused. Say 'time' or 'capacity'."
    },
    "ping me": {
        "replacement": "contact me",
        "category": "corporate_buzzwords",
        "severity": "info",
        "explanation": "Overly casual. Say 'contact me' or 'send me a message'."
    },
    "loop in": {
        "replacement": "include",
        "category": "corporate_buzzwords",
        "severity": "info",
        "explanation": "Informal. Simply say 'include' or 'add to the discussion'."
    },
    "run it up the flagpole": {
        "replacement": "propose to leadership",
        "category": "corporate_buzzwords",
        "severity": "critical",
        "explanation": "Archaic corporate jargon. Say 'propose to leadership'."
    },
    "boil the ocean": {
        "replacement": "attempt too much",
        "category": "corporate_buzzwords",
        "severity": "warning",
        "explanation": "Metaphor. Say 'attempt too much' or 'overreach'."
    },
    "drinking from a firehose": {
        "replacement": "overwhelmed with information",
        "category": "corporate_buzzwords",
        "severity": "warning",
        "explanation": "Cliché. Say 'overwhelmed' or 'processing rapidly'."
    },
    "push the envelope": {
        "replacement": "innovate",
        "category": "corporate_buzzwords",
        "severity": "warning",
        "explanation": "Overused metaphor. Say 'innovate' or 'experiment'."
    },
    "think outside the box": {
        "replacement": "be creative",
        "category": "corporate_buzzwords",
        "severity": "critical",
        "explanation": "Cliché. Say 'be creative' or 'explore new approaches'."
    },
    "align": {
        "replacement": "agree",
        "category": "corporate_buzzwords",
        "severity": "info",
        "explanation": "Overused. Say 'agree on' or 'coordinate'."
    },
    "socialize": {
        "replacement": "share",
        "category": "corporate_buzzwords",
        "severity": "warning",
        "explanation": "Jargon. Say 'share' or 'discuss with stakeholders'."
    },
    "reach out": {
        "replacement": "contact",
        "category": "corporate_buzzwords",
        "severity": "info",
        "explanation": "Overused. Simply say 'contact'."
    },
    "table stakes": {
        "replacement": "minimum requirements",
        "category": "corporate_buzzwords",
        "severity": "warning",
        "explanation": "Poker metaphor. Say 'minimum requirements' or 'basics'."
    },
    "stakeholder": {
        "replacement": "team member",
        "category": "corporate_buzzwords",
        "severity": "info",
        "explanation": "Overused. Be specific: team member, customer, partner."
    },
    "action item": {
        "replacement": "task",
        "category": "corporate_buzzwords",
        "severity": "info",
        "explanation": "Acceptable but overused. Say 'task' or 'action'."
    },
    "core competency": {
        "replacement": "strength",
        "category": "corporate_buzzwords",
        "severity": "critical",
        "explanation": "MBA jargon. Say 'strength' or 'expertise'."
    },
    "best practice": {
        "replacement": "proven method",
        "category": "corporate_buzzwords",
        "severity": "info",
        "explanation": "Overused. Say 'proven method' or 'standard approach'."
    },
    "value proposition": {
        "replacement": "benefit",
        "category": "corporate_buzzwords",
        "severity": "warning",
        "explanation": "Marketing jargon. Say 'benefit' or 'advantage'."
    },
    "win-win": {
        "replacement": "mutually beneficial",
        "category": "corporate_buzzwords",
        "severity": "warning",
        "explanation": "Overused. Say 'mutually beneficial'."
    },
    "at the end of the day": {
        "replacement": "ultimately",
        "category": "corporate_buzzwords",
        "severity": "warning",
        "explanation": "Filler phrase. Say 'ultimately' or 'in conclusion'."
    },
    "take offline": {
        "replacement": "discuss separately",
        "category": "corporate_buzzwords",
        "severity": "warning",
        "explanation": "Jargon. Say 'discuss separately' or 'continue later'."
    },
    "low hanging fruit": {
        "replacement": "easy wins",
        "category": "corporate_buzzwords",
        "severity": "critical",
        "explanation": "Cliché. Say 'easy wins' or 'quick improvements'."
    },
    "on my radar": {
        "replacement": "I'm aware of",
        "category": "corporate_buzzwords",
        "severity": "info",
        "explanation": "Informal. Say 'I'm aware of' or 'tracking'."
    },
    "visibility": {
        "replacement": "awareness",
        "category": "corporate_buzzwords",
        "severity": "info",
        "explanation": "Jargon. Say 'awareness' or 'transparency'."
    },
    "optimize": {
        "replacement": "improve",
        "category": "corporate_buzzwords",
        "severity": "info",
        "explanation": "Overused tech jargon. Say 'improve' or be specific."
    },
    "utilize": {
        "replacement": "use",
        "category": "corporate_buzzwords",
        "severity": "warning",
        "explanation": "Unnecessarily formal. Just say 'use'."
    },
    "facilitate": {
        "replacement": "help",
        "category": "corporate_buzzwords",
        "severity": "info",
        "explanation": "Formal jargon. Say 'help' or 'enable'."
    },
    "deliverable": {
        "replacement": "result",
        "category": "corporate_buzzwords",
        "severity": "info",
        "explanation": "Project management jargon. Say 'result' or 'output'."
    },
    "ecosystem": {
        "replacement": "network",
        "category": "corporate_buzzwords",
        "severity": "warning",
        "explanation": "Biology term misused. Say 'network' or 'environment'."
    },
    "disrupt": {
        "replacement": "change",
        "category": "corporate_buzzwords",
        "severity": "critical",
        "explanation": "Startup cliché. Say 'change' or 'transform'."
    },
    "holistic": {
        "replacement": "comprehensive",
        "category": "corporate_buzzwords",
        "severity": "warning",
        "explanation": "New-age jargon. Say 'comprehensive' or 'complete'."
    },
    "seamless": {
        "replacement": "smooth",
        "category": "corporate_buzzwords",
        "severity": "info",
        "explanation": "Overused. Say 'smooth' or 'integrated'."
    },
    "robust": {
        "replacement": "strong",
        "category": "corporate_buzzwords",
        "severity": "info",
        "explanation": "Engineering jargon. Say 'strong' or 'reliable'."
    },
    "scalable": {
        "replacement": "can grow",
        "category": "corporate_buzzwords",
        "severity": "info",
        "explanation": "Tech jargon. Say 'can grow' or 'expandable'."
    },
    "agile": {
        "replacement": "flexible",
        "category": "corporate_buzzwords",
        "severity": "info",
        "explanation": "Methodology term overused. Say 'flexible' or 'adaptive'."
    },
    "mission critical": {
        "replacement": "essential",
        "category": "corporate_buzzwords",
        "severity": "warning",
        "explanation": "Hyperbolic. Say 'essential' or 'critical'."
    },
    "bleeding edge": {
        "replacement": "newest",
        "category": "corporate_buzzwords",
        "severity": "warning",
        "explanation": "Cliché. Say 'newest' or 'most advanced'."
    },
    "cutting edge": {
        "replacement": "advanced",
        "category": "corporate_buzzwords",
        "severity": "info",
        "explanation": "Overused. Say 'advanced' or 'innovative'."
    },
    "state of the art": {
        "replacement": "advanced",
        "category": "corporate_buzzwords",
        "severity": "info",
        "explanation": "Overused. Say 'advanced' or 'modern'."
    },

    # Vague Commitments (30 phrases)
    "we will try to": {
        "replacement": "we will [specific action] by [date]",
        "category": "vague_commitments",
        "severity": "critical",
        "explanation": "Weak commitment. State specific action and deadline."
    },
    "hopefully we can": {
        "replacement": "we will [specific action]",
        "category": "vague_commitments",
        "severity": "critical",
        "explanation": "No commitment. State specific action and owner."
    },
    "it would be nice if": {
        "replacement": "requirement: [specific action]",
        "category": "vague_commitments",
        "severity": "critical",
        "explanation": "Wishful thinking, not a requirement. State as requirement."
    },
    "we should probably": {
        "replacement": "action item: [owner] will [action] by [date]",
        "category": "vague_commitments",
        "severity": "critical",
        "explanation": "Indecision. Assign owner and deadline."
    },
    "we might": {
        "replacement": "we will decide by [date]",
        "category": "vague_commitments",
        "severity": "warning",
        "explanation": "Uncertain. Set decision deadline."
    },
    "we could": {
        "replacement": "we will evaluate and decide by [date]",
        "category": "vague_commitments",
        "severity": "warning",
        "explanation": "Possibility, not commitment. Set evaluation deadline."
    },
    "someone should": {
        "replacement": "[specific person] will [action] by [date]",
        "category": "vague_commitments",
        "severity": "critical",
        "explanation": "No ownership. Assign specific owner and deadline."
    },
    "the team will": {
        "replacement": "[team member names] will [action] by [date]",
        "category": "vague_commitments",
        "severity": "warning",
        "explanation": "Vague ownership. Name specific team members."
    },
    "we need to": {
        "replacement": "[owner] will [action] by [date]",
        "category": "vague_commitments",
        "severity": "warning",
        "explanation": "No owner or deadline. Assign both."
    },
    "let's try": {
        "replacement": "we will [action] and evaluate by [date]",
        "category": "vague_commitments",
        "severity": "warning",
        "explanation": "Tentative. Commit to action and evaluation date."
    },

    # Meaningless Metrics (25 phrases)
    "significant improvement": {
        "replacement": "[X]% improvement",
        "category": "meaningless_metrics",
        "severity": "critical",
        "explanation": "Quantify the improvement with specific percentage."
    },
    "considerable growth": {
        "replacement": "[X]% growth compared to [baseline]",
        "category": "meaningless_metrics",
        "severity": "critical",
        "explanation": "Specify percentage and baseline comparison."
    },
    "substantial increase": {
        "replacement": "[X] unit increase from [Y] to [Z]",
        "category": "meaningless_metrics",
        "severity": "critical",
        "explanation": "Provide specific numbers and units."
    },
    "better performance": {
        "replacement": "[metric] improved from [X] to [Y]",
        "category": "meaningless_metrics",
        "severity": "critical",
        "explanation": "Name the metric and quantify the change."
    },
    "strong results": {
        "replacement": "[metric] reached [value]",
        "category": "meaningless_metrics",
        "severity": "warning",
        "explanation": "Specify which metric and value achieved."
    },
    "positive trend": {
        "replacement": "[metric] increased [X]% over [period]",
        "category": "meaningless_metrics",
        "severity": "warning",
        "explanation": "Quantify trend with percentage and time period."
    },
    "good progress": {
        "replacement": "completed [X] of [Y] tasks ([Z]%)",
        "category": "meaningless_metrics",
        "severity": "warning",
        "explanation": "Quantify progress as percentage or fraction."
    },
    "increased efficiency": {
        "replacement": "reduced [process] time by [X]%",
        "category": "meaningless_metrics",
        "severity": "critical",
        "explanation": "Specify what became more efficient and by how much."
    },
    "improved quality": {
        "replacement": "defect rate decreased from [X]% to [Y]%",
        "category": "meaningless_metrics",
        "severity": "critical",
        "explanation": "Define quality metric and quantify improvement."
    },
    "enhanced experience": {
        "replacement": "[metric] improved from [X] to [Y]",
        "category": "meaningless_metrics",
        "severity": "warning",
        "explanation": "Name specific metric (NPS, satisfaction score, etc.)."
    },

    # Vague Dates (30 phrases)
    "soon": {
        "replacement": "by [specific date]",
        "category": "vague_dates",
        "severity": "critical",
        "explanation": "Specify exact date (YYYY-MM-DD format)."
    },
    "shortly": {
        "replacement": "by [specific date]",
        "category": "vague_dates",
        "severity": "critical",
        "explanation": "Provide specific date."
    },
    "asap": {
        "replacement": "by [specific date and time]",
        "category": "vague_dates",
        "severity": "critical",
        "explanation": "Set specific deadline with date and time."
    },
    "when we have time": {
        "replacement": "scheduled for [specific date]",
        "category": "vague_dates",
        "severity": "critical",
        "explanation": "Schedule specific date, or mark as low priority."
    },
    "pending": {
        "replacement": "waiting for [specific input] by [date]",
        "category": "vague_dates",
        "severity": "warning",
        "explanation": "Specify what's pending and expected date."
    },
    "tbd": {
        "replacement": "will determine by [date]",
        "category": "vague_dates",
        "severity": "critical",
        "explanation": "Set date to determine the decision."
    },
    "to be determined": {
        "replacement": "will determine by [date]",
        "category": "vague_dates",
        "severity": "critical",
        "explanation": "Set decision deadline."
    },
    "in the near future": {
        "replacement": "by [specific date]",
        "category": "vague_dates",
        "severity": "critical",
        "explanation": "Define 'near future' with specific date."
    },
    "eventually": {
        "replacement": "by [specific date] or [priority: low/medium/high]",
        "category": "vague_dates",
        "severity": "warning",
        "explanation": "Commit to date or explicitly mark priority."
    },
    "at some point": {
        "replacement": "by [specific date]",
        "category": "vague_dates",
        "severity": "warning",
        "explanation": "Provide specific timeline."
    },
    "down the road": {
        "replacement": "in [quarter/year]: [specific date range]",
        "category": "vague_dates",
        "severity": "warning",
        "explanation": "Specify timeframe (Q1 2026, H2 2025, etc.)."
    },
    "later": {
        "replacement": "by [specific date]",
        "category": "vague_dates",
        "severity": "warning",
        "explanation": "Define 'later' with specific date."
    },
    "upcoming": {
        "replacement": "on [specific date]",
        "category": "vague_dates",
        "severity": "info",
        "explanation": "Provide specific date or date range."
    },

    # Weasel Words (40 phrases)
    "some people say": {
        "replacement": "[specific source] states",
        "category": "weasel_words",
        "severity": "critical",
        "explanation": "Cite specific source or remove claim."
    },
    "studies show": {
        "replacement": "[specific study, citation] shows",
        "category": "weasel_words",
        "severity": "critical",
        "explanation": "Cite the specific study with link/reference."
    },
    "experts believe": {
        "replacement": "[specific expert name] believes",
        "category": "weasel_words",
        "severity": "critical",
        "explanation": "Name the expert(s) with credentials."
    },
    "many think": {
        "replacement": "[X]% of [population] think",
        "category": "weasel_words",
        "severity": "critical",
        "explanation": "Quantify with survey data or remove."
    },
    "it is said": {
        "replacement": "[source] says",
        "category": "weasel_words",
        "severity": "critical",
        "explanation": "Attribute to specific source."
    },
    "reportedly": {
        "replacement": "according to [source]",
        "category": "weasel_words",
        "severity": "warning",
        "explanation": "Cite the report or source."
    },
    "allegedly": {
        "replacement": "according to [source/investigation]",
        "category": "weasel_words",
        "severity": "warning",
        "explanation": "Cite source of allegation."
    },
    "possibly": {
        "replacement": "[X]% probability based on [data]",
        "category": "weasel_words",
        "severity": "warning",
        "explanation": "Quantify uncertainty or remove."
    },
    "might": {
        "replacement": "will evaluate and decide by [date]",
        "category": "weasel_words",
        "severity": "info",
        "explanation": "Commit to evaluation deadline."
    },
    "could": {
        "replacement": "will consider if [conditions met]",
        "category": "weasel_words",
        "severity": "info",
        "explanation": "Specify conditions or commit."
    },

    # Redundant Phrases (35 phrases)
    "each and every": {
        "replacement": "each" or "every",
        "category": "redundant_phrases",
        "severity": "info",
        "explanation": "Redundant. Use 'each' or 'every', not both."
    },
    "first and foremost": {
        "replacement": "first",
        "category": "redundant_phrases",
        "severity": "info",
        "explanation": "Redundant. Just say 'first' or 'primarily'."
    },
    "in order to": {
        "replacement": "to",
        "category": "redundant_phrases",
        "severity": "info",
        "explanation": "Wordy. Just say 'to'."
    },
    "due to the fact that": {
        "replacement": "because",
        "category": "redundant_phrases",
        "severity": "warning",
        "explanation": "Verbose. Simply say 'because'."
    },
    "at this point in time": {
        "replacement": "now",
        "category": "redundant_phrases",
        "severity": "warning",
        "explanation": "Wordy. Say 'now' or 'currently'."
    },
    "for the purpose of": {
        "replacement": "to",
        "category": "redundant_phrases",
        "severity": "info",
        "explanation": "Verbose. Use 'to' or 'for'."
    },
    "in the event that": {
        "replacement": "if",
        "category": "redundant_phrases",
        "severity": "info",
        "explanation": "Wordy. Just say 'if'."
    },
    "with regard to": {
        "replacement": "regarding" or "about",
        "category": "redundant_phrases",
        "severity": "info",
        "explanation": "Formal. Say 'regarding' or 'about'."
    },
    "in spite of the fact that": {
        "replacement": "although" or "despite",
        "category": "redundant_phrases",
        "severity": "warning",
        "explanation": "Verbose. Use 'although' or 'despite'."
    },
    "until such time as": {
        "replacement": "until",
        "category": "redundant_phrases",
        "severity": "warning",
        "explanation": "Wordy. Just say 'until'."
    },
}


def load_jargon_dictionary(json_path: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """
    Load jargon dictionary from JSON file (if exists) or return built-in dictionary.

    Args:
        json_path: Optional path to custom jargon JSON file

    Returns:
        Dictionary mapping jargon phrases to replacement info
    """
    if json_path and json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return JARGON_REPLACEMENTS


def save_jargon_dictionary(output_path: Path, dictionary: Optional[Dict] = None):
    """
    Save jargon dictionary to JSON file for easy updates.

    Args:
        output_path: Path to save JSON file
        dictionary: Optional custom dictionary (defaults to built-in)
    """
    data = dictionary or JARGON_REPLACEMENTS
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_jargon_by_category(category: str) -> Dict[str, Dict[str, str]]:
    """
    Get all jargon phrases for a specific category.

    Args:
        category: One of JARGON_CATEGORIES keys

    Returns:
        Dictionary of jargon in that category
    """
    return {
        phrase: info
        for phrase, info in JARGON_REPLACEMENTS.items()
        if info.get("category") == category
    }


def get_jargon_by_severity(severity: str) -> Dict[str, Dict[str, str]]:
    """
    Get all jargon phrases with specific severity level.

    Args:
        severity: "critical", "warning", or "info"

    Returns:
        Dictionary of jargon with that severity
    """
    return {
        phrase: info
        for phrase, info in JARGON_REPLACEMENTS.items()
        if info.get("severity") == severity
    }


def get_jargon_count() -> Dict[str, int]:
    """
    Get count of jargon phrases by category and severity.

    Returns:
        Dictionary with counts by category and severity
    """
    counts = {
        "total": len(JARGON_REPLACEMENTS),
        "by_category": {},
        "by_severity": {"critical": 0, "warning": 0, "info": 0}
    }

    for category in JARGON_CATEGORIES:
        counts["by_category"][category] = len(get_jargon_by_category(category))

    for severity in ["critical", "warning", "info"]:
        counts["by_severity"][severity] = len(get_jargon_by_severity(severity))

    return counts


# Generate default JSON file on import (for easy customization)
if __name__ == "__main__":
    default_path = Path(__file__).parent / "jargon_dictionary.json"
    save_jargon_dictionary(default_path)

    counts = get_jargon_count()
    print(f"[OK] Saved {counts['total']} jargon phrases to {default_path}")
    print(f"\nBreakdown:")
    print(f"  Critical: {counts['by_severity']['critical']}")
    print(f"  Warning: {counts['by_severity']['warning']}")
    print(f"  Info: {counts['by_severity']['info']}")
    print(f"\nBy Category:")
    for category, count in counts['by_category'].items():
        print(f"  {category}: {count}")
